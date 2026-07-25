"""TunnelBook backend: SPAM superpixels -> mark region boundaries (split/delete) -> solve.

Flow:
  POST /api/sessions        upload image -> resize -> datasets/ cache lookup by content hash
                            (frozen segmentation + depth from tunnelbook.data.dataset; on miss
                            compute (cached) SPAM superpixels + depth, then save a new dataset)
                            -> optional Canny edge-conditioning (tunnelbook.data.edges) so
                            boundaries follow real object edges, not just the superpixel
                            tessellation -> ProblemInstance -> per-boundary segments + baseline
                            layer solve -> SAM image embedding warmed up here too (best-effort)
                            so the first "select object" click isn't the one paying that cost.
  POST /api/sessions/{id}/solve   mark boundaries split_soft/split_hard/delete -> build the
                            MIP with those terms -> solve -> return a layer-assignment preview.
  POST /api/sessions/{id}/select-object   point-prompted SAM (tunnelbook.data.sam_point): click a
                            point -> mark every boundary tracing that object's outline.
  /api/sessions/{id}/edge-configs   save/load named edge-mark sets in the session's dataset dir
                            (marks stored as original region-label pairs).

Runs in tunnelbook's uv env (torch/cv2/skimage/gurobi + fastapi via the `serve` extra):
  cd generator/src/checkpoint1
  uv run --project ../../.. --extra serve --extra depth python -m uvicorn server:app --port 8000
Needs an active Gurobi license.  Set SEG_BACKEND=slic for the CPU-only path (no CUDA/SPAM);
see generator/README.md.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import time
import uuid
import zipfile
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel
from skimage.segmentation import find_boundaries

from pathlib import Path

from tunnelbook.config import Config
from tunnelbook.data.dataset import (
    datasets_root, find_dataset, indices_to_pairs, label_map_md5, list_edge_configs,
    load_dataset, load_edge_config, resolve_marks, save_dataset, save_edge_config,
)
from tunnelbook.serialize import load_result
from tunnelbook.data.edges import (
    condition_label_map, detect_canny_raster, edge_canny_alignment, edge_gradient_strength,
    edge_params_for, image_gradient_strength,
)
from tunnelbook.data.labelmap import build_instance, edge_depth_scores, region_boundary_segments
from tunnelbook.data.realimage import compute_depth, segment_image
from tunnelbook.data.sam_point import get_sam_predictor, predict_point_mask
from tunnelbook.data.spam_segment import SAM_CHECKPOINT, segment_image_spam
from tunnelbook.experiment import PipelineParams, Prepared, write_run_artifacts
from tunnelbook.export import build_layer_ai_docs
from tunnelbook.export.ai import _detect_texture_edges, _trace_skeleton
from tunnelbook.model import build_model
from tunnelbook.solve import InfeasibleError, solve
from tunnelbook.viz.layermap import render_layer_map
from tunnelbook.viz.wood import render_front_view, wood_layer_stack

app = FastAPI()
_sessions: dict[str, dict] = {}

MAX_DIM = 1024
NSPIX = 200
POINTS_PER_SIDE = 16
USE_SAM = os.environ.get("SPAM_SAM", "1") != "0"          # SPAM_SAM=0 -> faster, no SAM
# Segmentation backend: "spam" (learned superpixels, needs CUDA + SAM download) is the default;
# "slic" is the CPU-only skimage path (tunnelbook.data.realimage.segment_image) -- no CUDA, no SAM,
# so a teammate can run the whole UI on a laptop. Set SEG_BACKEND=slic to use it.
SEG_BACKEND = os.environ.get("SEG_BACKEND", "spam").lower()
# Depth checkpoint: "large" (default, 335M params, best quality) or "small"/"base" for faster
# CPU-only local testing (see tunnelbook.data.depthmodel.MODELS). Set DEPTH_MODEL=small.
DEPTH_MODEL = os.environ.get("DEPTH_MODEL", "large").lower()
# "select object" (point-prompted SAM): min fraction of a boundary's pixels that must lie on the
# clicked object's outline to mark it. A fixed cutoff, not a percentile -- this is a direct "is
# this boundary part of the traced object" decision, not a ranking over all boundaries.
SELECT_OBJECT_ALIGN_THRESHOLD = 0.3
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".spam_cache")
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


# ── request models ──────────────────────────────────────────────────────────────
class Marking(BaseModel):
    index: int
    type: str  # "split_soft" | "split_hard" | "delete"


class SolveRequest(BaseModel):
    markings: List[Marking] = []
    n_layers: int = 5
    lambda_split: float = 1.0
    time_limit: Optional[float] = 180.0
    objective: str = "depth"          # "depth" (fixed-bin fidelity) | "cut" (depth-aware cut only)
    min_layer_area: float = 0.04      # cut objective (DEFAULT floor): force each layer to own >= this fraction of image area
    min_layer_regions: int = 0        # cut objective: optional count floor (>= k superpixels/layer); 0 = off (area floor is default)
    lambda_cut: float = 1.0           # cut objective: weight on the boundary-cut term
    lambda_depth: float = 0.5         # cut objective: k-median depth-anchor strength (0 = pure cut;
                                      # marks stop mattering above ~0.5, so the useful range is 0..0.5)
    connectivity: bool = True         # flow (fabrication) constraints; off = pieces may float
    connectivity_method: str = "lazy"  # "flow" | "lazy" (cut-set rows added lazily at MIPSOL)
    y_monotone: bool = True           # y == cumsum(x): "full backing" (fewer free binaries); the
                                      # depth-preview branch pins this back to False (free-y support)
    norel_time: float = 60.0          # cut objective: seconds of Gurobi NoRel heuristic (0 = off);
                                      # rescues N>=4 incumbents (japan N=7: none -> comps 11, rho .90)
    mip_focus: int = 1                # cut objective: Gurobi MIPFocus (1 = feasibility focus)
    cut_score: str = "meandiff"       # cut objective: "meandiff" (region-level depth gap) | "laplacian"
    cut_log_sigma: float = 2.0        # cut objective: LoG spatial scale (px) for the crease score
    cut_sigma: float = 0.15           # cut objective: kappa drop-off width exp(-s^2/2sigma^2)
    lambda_coherence: float = 0.0     # cut objective: depth-plateau coherence tie-breaker (0 = off).
                                      # ON (0.05) fixes free plateau splits (robert-keane N=5:
                                      # mountain+sky unified, z spread, penalty paid ~0), but on
                                      # hard mark-heavy N=7 solves the extra rows degraded the
                                      # 180s incumbent (redcharlie: z collapse, rho -0.57 vs 0.05)
                                      # -- hence default OFF; enable via the UI toggle per solve
    coherence_eps: float = 0.02       # plateau depth-gap threshold for the coherence pairs


class ScoresRequest(BaseModel):
    cut_score: str = "meandiff"
    cut_log_sigma: float = 2.0


class SelectObjectRequest(BaseModel):
    x: float
    y: float


class EdgeConfigSaveRequest(BaseModel):
    name: str
    markings: List[Marking] = []
    n_layers: Optional[int] = None


class StandRequest(BaseModel):
    n_layers: int
    spoke_h_in: float = 1.4
    base_h_in: float = 0.65


class ExportLayersRequest(SolveRequest):
    """Solve payload (inherited) + per-layer .ai export params."""
    mode: str = "engraving"           # "engraving" (cut + engrave outlines) | "outline" (cut only)
    content_width_in: float = 12.0    # physical artwork width; height follows image aspect
    border_in: float = 0.5            # red frame band around each sheet
    # optional per-layer engrave overrides (length must == n_layers); None = use the global mode/floor
    engrave_layers: Optional[List[bool]] = None          # per-plane engrave on/off
    min_engrave_in_per_layer: Optional[List[float]] = None  # per-plane detail floor (inches)


# ── helpers ─────────────────────────────────────────────────────────────────────
def _load_rgb(img_bytes: bytes) -> np.ndarray:
    """Decode + downsample to MAX_DIM (same convention as tunnelbook.load_image)."""
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = pil.size
    scale = MAX_DIM / float(max(w, h))
    if scale < 1.0:
        pil = pil.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return np.asarray(pil)


def _cached_spam(rgb: np.ndarray) -> np.ndarray:
    """SPAM label map, cached on disk by image-hash + params (SPAM is slow)."""
    key = hashlib.md5(np.ascontiguousarray(rgb)).hexdigest()[:16]
    fp = os.path.join(CACHE_DIR, f"spam_{key}_md{MAX_DIM}_n{NSPIX}_pps{POINTS_PER_SIDE}_sam{int(USE_SAM)}.npy")
    if os.path.exists(fp):
        print(f"[spam] cache hit {os.path.basename(fp)}")
        return np.load(fp)
    print(f"[spam] segmenting (SAM={'on' if USE_SAM else 'off'})… slow, will cache")
    labels = segment_image_spam(
        rgb, nspix=NSPIX, points_per_side=POINTS_PER_SIDE,
        sam_checkpoint=SAM_CHECKPOINT if USE_SAM else "",
    )
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(fp, labels)
    return labels


def _segment(rgb: np.ndarray) -> np.ndarray:
    """Superpixel label map for the requested backend (``SEG_BACKEND``).

    ``"slic"`` -> CPU-only skimage SLIC (no CUDA, no SAM download); ``"spam"`` (default) ->
    the learned SPAM segmenter (cached on disk, needs CUDA).
    """
    if SEG_BACKEND == "slic":
        print("[seg] SLIC (CPU) backend")
        return segment_image(rgb)
    return _cached_spam(rgb)


def _png_bytes(rgb_uint8: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def _png_bytes_rgba(rgba_uint8: np.ndarray) -> bytes:
    """PNG-encode an (H, W, 4) RGBA array, preserving the alpha channel (cv2 wants BGRA)."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba_uint8, cv2.COLOR_RGBA2BGRA))
    return buf.tobytes()


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _overlay_img(inst, layers, rgb: np.ndarray) -> np.ndarray:
    """Regions coloured by (1-based) layer (viridis), blended over the photo."""
    N = inst.n_layers
    lay = inst.paint(np.asarray(layers, dtype=float))       # (H,W), 1..N per region
    norm = (lay - 1.0) / max(1, N - 1)
    color = (cm.viridis(np.nan_to_num(norm))[..., :3] * 255).astype(np.uint8)
    return (0.6 * color + 0.4 * rgb).astype(np.uint8)


def _layer_sheets(inst, sol, rgb: np.ndarray) -> List[np.ndarray]:
    """Per-layer *retained material* (the real laser sheet = y): visible regions in full colour,
    hidden **support** tinted red, cut-away dimmed.  This is the fabrication view -- a layer that
    looks like floating pieces in the semantic preview shows here connected via its support.
    """
    dim = (rgb * 0.16).astype(np.uint8)
    red = np.array([220, 60, 60], dtype=np.float32)
    out = []
    for l in range(inst.n_layers):
        vis = inst.paint(sol.x[:, l].astype(float), fill=0.0) > 0.5
        sup = inst.paint(sol.support[:, l].astype(float), fill=0.0) > 0.5
        img = dim.copy()
        img[sup] = (0.45 * rgb[sup] + 0.55 * red).astype(np.uint8)  # support = red-tinted
        img[vis] = rgb[vis]                                          # visible = full colour
        out.append(img)
    return out


def _layer_masks(inst, sol, rgb: np.ndarray, max_dim: int = 512, *,
                 min_engrave_in_per_layer=None, content_width_in: float = 12.0,
                 engrave: bool = True, engrave_layers=None):
    """Per-layer RGBA textures for the 3D book-stack preview, TWO styles per layer.

    ``min_engrave_in_per_layer`` / ``content_width_in`` set the *per-layer* engrave detail floor
    (same mapping as the .ai export), so the 3D stack can mirror the density the user is tuning;
    ``None`` keeps the exporter default 0.03".  ``engrave=False`` (outline mode) or a per-plane
    ``engrave_layers[l]==False`` draws that sheet with cut lines only, no burns.

    Material on each plane is the *retained* sheet (``sol.x`` visible + ``sol.support``), so the
    stack shows the physical pages, not just the visible silhouettes:
      photo -- the photo where visible; support material as a paper-toned wash of the photo
               (readable as "backing", still hints at what sits in front of it);
      sheet -- the fabrication look mirroring the .ai export: paper stock, dark Canny engrave
               (30/100, the export's parity edges) on visible regions only, red cut lines along
               every material boundary, and the outer frame band with its cut rectangle.
    Both styles share one canvas padded by the frame band so toggling never shifts geometry.
    Front layer = layer_01 (index 0).  Downscaled to ``max_dim`` (INTER_AREA antialiases the
    silhouette edges) so the N*2 masks don't bloat the solve response.
    Returns ``(photo_masks, sheet_masks)``."""
    H, W = rgb.shape[:2]
    band = max(8, round(0.04 * max(H, W)))          # preview frame margin (export: border_in)
    paper = np.array([243, 240, 232], dtype=np.uint8)
    wash = np.array([203, 198, 188], dtype=np.float32)
    ink, red = (64, 60, 54), (198, 40, 40)
    # centerline engrave preview: same machinery as the .ai exporter's "centerline" style
    # (visible-masked Canny -> skeleton -> single-burn polylines with a physical length floor),
    # so the sheet texture shows what actually gets burned -- not the doubled Canny loops.
    from skimage.morphology import skeletonize
    s = max_dim / (max(H, W) + 2 * band)
    photo_out, sheet_out = [], []
    for l in range(inst.n_layers):
        vis = inst.paint(sol.x[:, l].astype(float), fill=0.0) > 0.5
        sup = inst.paint(sol.support[:, l].astype(float), fill=0.0) > 0.5
        mat = vis | sup
        ph = np.zeros((H, W, 4), dtype=np.uint8)
        ph[..., :3][vis] = rgb[vis]
        ph[..., :3][sup] = (0.35 * rgb[sup] + 0.65 * wash).astype(np.uint8)
        ph[..., 3] = np.where(mat, 255, 0).astype(np.uint8)
        sh = np.zeros((H, W, 4), dtype=np.uint8)
        sh[..., :3][mat] = paper
        floor_l = (min_engrave_in_per_layer[l]
                   if min_engrave_in_per_layer and l < len(min_engrave_in_per_layer) else 0.03)
        min_len_px = max(2.0, float(floor_l) * W / max(1e-6, float(content_width_in)))
        on_l = engrave and (engrave_layers[l]
                            if engrave_layers and l < len(engrave_layers) else True)
        if on_l:
            emap = _detect_texture_edges(rgb, vis, 30, 100)
            for chain in _trace_skeleton(skeletonize(emap > 0)):
                if len(chain) < 2:
                    continue
                seg = np.diff(chain, axis=0)
                if np.hypot(seg[:, 0], seg[:, 1]).sum() < min_len_px:
                    continue
                cv2.polylines(sh, [chain.round().astype(np.int32)], False, (*ink, 255), 2)
        cut = cv2.morphologyEx(mat.astype(np.uint8), cv2.MORPH_GRADIENT,
                               np.ones((3, 3), np.uint8)) > 0
        sh[..., :3][cut] = red
        sh[..., 3] = np.where(mat | cut, 255, 0).astype(np.uint8)
        ph = cv2.copyMakeBorder(ph, band, band, band, band,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        sh = cv2.copyMakeBorder(sh, band, band, band, band, cv2.BORDER_CONSTANT,
                                value=(int(paper[0]), int(paper[1]), int(paper[2]), 255))
        cv2.rectangle(sh, (1, 1), (sh.shape[1] - 2, sh.shape[0] - 2), (*red, 255), 2)
        if s < 1:
            dst = (max(1, round(sh.shape[1] * s)), max(1, round(sh.shape[0] * s)))
            ph = cv2.resize(ph, dst, interpolation=cv2.INTER_AREA)
            sh = cv2.resize(sh, dst, interpolation=cv2.INTER_AREA)
        photo_out.append(ph)
        sheet_out.append(sh)
    return photo_out, sheet_out


def _engrave_preview_img(sol, rgb: np.ndarray, layer: int, min_engrave_in: float,
                         content_width_in: float, engrave: bool, max_dim: int = 640):
    """Render ONE layer's fabrication sheet (paper + centerline engrave at the given detail floor
    + red cut lines) as an RGBA preview -- the same machinery as :func:`_layer_masks` / the .ai
    exporter, but a single plane with a *parameterized* engrave floor, so the UI can show what a
    given per-layer ``min_engrave_in`` actually burns.  Reads the cached solution -- no solving.
    Returns ``(rgba_uint8, n_strokes)``."""
    from skimage.morphology import skeletonize
    inst = sol.inst
    H, W = rgb.shape[:2]
    band = max(8, round(0.04 * max(H, W)))
    paper = np.array([243, 240, 232], dtype=np.uint8)
    ink, red = (64, 60, 54), (198, 40, 40)
    l = max(0, min(int(layer), inst.n_layers - 1))
    # exporter mapping: a stroke shorter than min_engrave_in inches -- at print scale where W px
    # spans content_width_in inches -- is dropped, so a higher floor burns fewer, longer strokes.
    min_len_px = max(2.0, float(min_engrave_in) * W / max(1e-6, float(content_width_in)))
    vis = inst.paint(sol.x[:, l].astype(float), fill=0.0) > 0.5
    sup = inst.paint(sol.support[:, l].astype(float), fill=0.0) > 0.5
    mat = vis | sup
    sh = np.zeros((H, W, 4), dtype=np.uint8)
    sh[..., :3][mat] = paper
    n_strokes = 0
    if engrave:
        emap = _detect_texture_edges(rgb, vis, 30, 100)
        for chain in _trace_skeleton(skeletonize(emap > 0)):
            if len(chain) < 2:
                continue
            seg = np.diff(chain, axis=0)
            if np.hypot(seg[:, 0], seg[:, 1]).sum() < min_len_px:
                continue
            cv2.polylines(sh, [chain.round().astype(np.int32)], False, (*ink, 255), 2)
            n_strokes += 1
    cut = cv2.morphologyEx(mat.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    sh[..., :3][cut] = red
    sh[..., 3] = np.where(mat | cut, 255, 0).astype(np.uint8)
    sh = cv2.copyMakeBorder(sh, band, band, band, band, cv2.BORDER_CONSTANT,
                            value=(int(paper[0]), int(paper[1]), int(paper[2]), 255))
    cv2.rectangle(sh, (1, 1), (sh.shape[1] - 2, sh.shape[0] - 2), (*red, 255), 2)
    s = max_dim / max(sh.shape[0], sh.shape[1])
    if s < 1:
        sh = cv2.resize(sh, (max(1, round(sh.shape[1] * s)), max(1, round(sh.shape[0] * s))),
                        interpolation=cv2.INTER_AREA)
    return sh, n_strokes


def _edges_payload(inst, scores=None, canny_align=None) -> List[dict]:
    """One clickable item per region-adjacency boundary, aligned to inst.edges order.

    ``scores`` (aligned to ``inst.edges``) is the normalized depth-edge strength s_e in [0,1]
    used for the frontend crease-map heatmap. ``canny_align`` (keyed by positional (i,j), from
    ``edge_canny_alignment``) is how much of the boundary coincides with a detected Canny edge --
    a depth-independent signal the frontend's "auto select edges" uses alongside the depth score,
    since a visually sharp silhouette can sit on a depth-flat/noisy stretch of the estimate.
    """
    segs = region_boundary_segments(inst.label_map, inst.region_ids)
    out = []
    for e, (i, j) in enumerate(inst.edges):
        i, j = int(i), int(j)
        s = segs.get((i, j))
        score = float(scores[e]) if scores is not None and e < len(scores) else 0.0
        align = float(canny_align.get((i, j), 0.0)) if canny_align is not None else 0.0
        out.append({"index": e, "i": i, "j": j, "score": score, "cannyAlign": align,
                    "segments": [] if s is None else s.tolist()})
    return out


def _display_scores(inst, depth, method: str, log_sigma: float) -> np.ndarray:
    """Per-edge crease scores max-normalized for the UI heat ramp ONLY.  meandiff returns raw
    |d_i - d_j| (max ~0.3 on real photos), which renders nearly dead against the [0,1] ramp;
    dividing by the max lets the strongest edge hit 1.0 (a no-op for laplacian, which is already
    min-max normalized).  Solver pricing is untouched: cut_costs re-derives the raw scores."""
    scores = edge_depth_scores(inst, depth_map=depth, method=method, log_sigma=log_sigma)
    smax = float(scores.max()) if scores.size else 0.0
    return scores / smax if smax > 0 else scores


def _build_cfg(req: "SolveRequest") -> Config:
    """Config for the requested objective.  ``"cut"`` drops the depth-fidelity term and lets the
    depth-aware boundary cut (plus the user's edge marks) drive the layering, with a per-layer
    *area* floor (default) so it can't collapse onto one sheet -- each layer must own at least
    ``min_layer_area`` of the image; ``"depth"`` is the original fixed-bin fidelity.  The count
    floor ``min_layer_regions`` remains available (0 = off) but the area floor is the default."""
    if req.objective == "cut":
        # soft splits are folded into the cut cost as zero-cost edges (build_model), so the
        # separate soft-split reward is off (lambda_split=0) to avoid double counting.
        # lambda_depth > 0 blends in a k-median depth anchor (at 0 the term is off and the
        # model is identical to the original pure-cut solve).
        return Config(n_layers=req.n_layers, fix_x=False,
                      depth_model="kmedian", lambda_depth=req.lambda_depth, lambda_support=0.0,
                      lambda_cut=req.lambda_cut, cut_cost="aware", cut_score=req.cut_score,
                      cut_log_sigma=req.cut_log_sigma, cut_sigma=req.cut_sigma,
                      use_contact_weight=True,  # kappa_e scales with shared-boundary length
                      min_layer_area=req.min_layer_area,
                      min_layer_regions=req.min_layer_regions,
                      connectivity=req.connectivity,
                      connectivity_method=req.connectivity_method,
                      y_monotone=req.y_monotone,
                      norel_time=req.norel_time, mip_focus=req.mip_focus,
                      lambda_coherence=req.lambda_coherence,
                      coherence_eps=req.coherence_eps,
                      lambda_split=0.0, mip_gap=0.02,
                      time_limit=req.time_limit, verbose=False,
                      log_progress=True)  # stream incumbents (+ z medians) to the console
    # depth objective solves in seconds -- the NoRel/MIPFocus rescue is left off here.
    # y_monotone is pinned False: this branch prices hidden support (lambda_support=1.0), which
    # is only meaningful with free y -- full backing would make it the degenerate A_r(N-1-L_r).
    return Config(n_layers=req.n_layers, fix_x=False, lambda_support=1.0,
                  lambda_split=req.lambda_split, mip_gap=0.01,
                  connectivity=req.connectivity,
                  connectivity_method=req.connectivity_method,
                  y_monotone=False,
                  time_limit=req.time_limit, verbose=False, log_progress=True)


def _req_to_pipeline_params(req: "SolveRequest") -> PipelineParams:
    """Faithful SolveRequest -> PipelineParams for the run record's config.json.  Branches on the
    objective to mirror exactly what ``_build_cfg`` passes to ``Config`` -- copying ``req.*`` blindly
    would misreport the depth branch, which leaves the cut-only knobs (norel/mip_focus/min_layer_*/
    coherence/contact-weight) at Config defaults.  Segmentation/depth fields stay at defaults;
    ``write_run_artifacts`` nulls them and records the dataset's frozen provenance instead.  Keep in
    sync with ``_build_cfg`` above (and ``examples/cluster_sweep.py`` ``_SERVER_CUT_DEFAULTS``)."""
    common = dict(
        layers=req.n_layers,
        time_limit=req.time_limit if req.time_limit is not None else 180.0,
        bins="linear",              # server's build_instance uses the default depth_method="linear"
        connectivity=req.connectivity,
        connectivity_method=req.connectivity_method,
        log_progress=True,          # _build_cfg sets log_progress=True in both branches
        export_ai=True,             # archive the cut+engrave .ai next to the run, like the sweep
    )
    if req.objective == "cut":
        return PipelineParams(
            **common,
            y_monotone=req.y_monotone,   # full backing on the cut objective
            depth_model="kmedian", lambda_depth=req.lambda_depth,
            lambda_support=0.0, lambda_smooth=0.0,
            lambda_cut=req.lambda_cut, lambda_split=0.0,
            cut_cost="aware", cut_score=req.cut_score, cut_log_sigma=req.cut_log_sigma,
            cut_sigma=req.cut_sigma,
            use_contact_weight=True,
            min_layer_area=req.min_layer_area, min_layer_regions=req.min_layer_regions,
            norel_time=req.norel_time, mip_focus=req.mip_focus,
            lambda_coherence=req.lambda_coherence, coherence_eps=req.coherence_eps,
            mip_gap=0.02,
        )
    # objective == "depth": _build_cfg omits the cut-only knobs -> record Config defaults, not req.*
    # (and pins y_monotone False so hidden support stays free-y, matching _build_cfg)
    return PipelineParams(
        **common,
        y_monotone=False,
        depth_model="bins", lambda_depth=1.0, lambda_support=1.0, lambda_smooth=0.0,
        lambda_cut=0.0, lambda_split=req.lambda_split,
        cut_cost="uniform", use_contact_weight=False,
        min_layer_area=0.0, min_layer_regions=0,
        norel_time=0.0, mip_focus=0, lambda_coherence=0.0,
        mip_gap=0.01,
    )


def _resolve_markings(inst, markings) -> Dict[str, list]:
    """Positional edge-index marks -> ``{split_soft|split_hard|delete: [(i, j), ...]}`` region-id
    pairs against this instance's edge list (out-of-range indices skipped).  Shared by the solve
    (feeds build_model) and the run record (config.json marks), so both see the same pairs."""
    out: Dict[str, list] = {"split_soft": [], "split_hard": [], "delete": []}
    n_edges = len(inst.edges)
    for mk in markings:
        if not (0 <= mk.index < n_edges):
            continue
        i, j = inst.edges[mk.index]
        if mk.type in out:
            out[mk.type].append((int(i), int(j)))
    return out


def _solve_signature(req: "SolveRequest") -> tuple:
    """A hashable fingerprint of everything that affects the *solve*, used to decide whether a
    cached ``sess["sol"]`` can be reused for export instead of re-solving (see ``export_ai``).

    Deliberately over exactly the solve-affecting fields the ``/export-ai`` payload also sends, so
    a solve and the export that follows it produce the *same* fingerprint.  Solve-only knobs the
    export payload never carries (``lambda_split``, ``cut_log_sigma``, ``min_layer_regions``) are
    excluded: on export they fall back to ``SolveRequest`` defaults, so including them would
    spuriously miss the cache and trigger a needless re-solve.  ``time_limit`` is sent by both
    payloads but stays excluded on purpose: it changes how long the search runs, not which
    solution is sought, so an export should reuse the solve the user just watched."""
    marks = tuple(sorted((int(m.index), str(m.type)) for m in (req.markings or [])))
    def _r(x):
        return round(float(x), 6)
    return (
        marks, int(req.n_layers), str(req.objective), _r(req.lambda_depth),
        bool(req.connectivity), str(req.connectivity_method), bool(req.y_monotone),
        _r(req.norel_time), int(req.mip_focus), _r(req.lambda_coherence),
        _r(req.min_layer_area), str(req.cut_score), _r(req.lambda_cut),
        _r(req.cut_sigma),
    )


def _solve_layers(sess: dict, req: "SolveRequest"):
    """Build the instance + model for this session's marks/config and solve.  Returns
    ``(inst, sol, timing, marks)`` -- the per-stage timing and the resolved marks are threaded into
    the run record so a UI solve is persisted with the same schema as a cluster-sweep run."""
    cfg = _build_cfg(req)
    timing: Dict[str, float] = {}
    t = time.perf_counter()
    inst = build_instance(sess["label_map"], sess["depth"], cfg)
    timing["build_instance"] = time.perf_counter() - t
    marks = _resolve_markings(inst, req.markings)
    t = time.perf_counter()
    # depth_map is needed by the laplacian cut score; harmless for the depth objective.
    gm = build_model(inst, cfg, depth_map=sess["depth"],
                     split_soft=marks["split_soft"] or None,
                     split_hard=marks["split_hard"] or None,
                     delete=marks["delete"] or None)
    timing["build_model"] = time.perf_counter() - t
    t = time.perf_counter()
    sol = solve(gm)  # InfeasibleError propagates to the endpoint (never saved on infeasible)
    timing["solve"] = time.perf_counter() - t
    return inst, sol, timing, marks


def _save_run(sess: dict, session_id: str, req: "SolveRequest", inst, sol,
              timing: Dict[str, float], marks: Dict[str, list]) -> str:
    """Persist one UI solve through the *same* artifact schema the cluster sweep writes --
    ``config.json`` / ``metrics.json`` / ``timing.json`` / ``result.npz`` / per-layer ``.ai`` /
    figures -- via :func:`tunnelbook.experiment.write_run_artifacts`, reusing the already-solved
    ``(inst, sol)`` (no re-solve).  This makes a UI run reproducible and directly comparable to a
    sweep row: it records dataset provenance (slug / rgb_md5 / label_map_md5 / seg / depth), the full
    resolved config, the anytime trajectory, and the solution arrays."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RUNS_DIR, f"{session_id[:8]}_{ts}")
    p = _req_to_pipeline_params(req)
    meta = sess.get("dataset_meta")
    if sess.get("edge_condition"):  # session re-split the frozen label map; record it truthfully
        p.edge_condition = True
        p.canny_low = sess.get("canny_low", p.canny_low)
        p.canny_high = sess.get("canny_high", p.canny_high)
    prepared = Prepared(
        slug=sess["dataset_slug"], rgb=sess["rgb"], label_map=sess["label_map"],
        n_regions=inst.n_regions,  # conditioned count (meta keeps the original md5/n_regions)
        thresh=str((meta or {}).get("seg", {}).get("backend", "dataset")),
        depth=sess["depth"], timing=timing, dataset_meta=meta,
    )
    # pass the real cfg the solve used so the recorded tau_px is truthful (not remapped from p);
    # collapse the always-3-key marks dict to {} when empty so config.json matches a no-marks sweep
    write_run_artifacts(
        out_dir, p, prepared, inst, sol, timing,
        image_name=sess.get("filename", "image"), status=sol.status,
        cfg=_build_cfg(req), marks=(marks if any(marks.values()) else {}),
        edge_config=None, save_npz=True)
    return out_dir


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/sessions")
async def create_session(
    image: UploadFile = File(...), n_layers: int = Form(5),
    edge_condition: bool = Form(False), canny_low: int = Form(50), canny_high: int = Form(150),
):
    try:
        rgb = _load_rgb(await image.read())
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    # datasets/ cache: frozen segmentation + depth keyed by RGB content hash (no GPU on a hit)
    d = find_dataset(rgb)
    heal = False  # a corrupt dataset dir must be force-rewritten, not skipped as "exists"
    if d is not None:
        try:
            b = load_dataset(d)
            label_map, depth = b.label_map, b.depth
            print(f"[dataset] hit {b.slug} (skipping segmentation+depth)")
        except Exception as e:  # corrupt/torn dataset dir: recompute + rewrite, don't 500 forever
            print(f"[dataset] {d.name} unreadable ({type(e).__name__}: {e}); recomputing")
            d, heal = None, True
    if d is None:
        label_map = _segment(rgb)
        depth = compute_depth(rgb, "model", model_name=DEPTH_MODEL)  # Depth Anything V2 (the laplacian cut needs real depth)
        seg_params = (
            {"backend": "slic", "max_dim": MAX_DIM} if SEG_BACKEND == "slic"
            else {"backend": SEG_BACKEND, "nspix": NSPIX, "points_per_side": POINTS_PER_SIDE,
                  "sam": USE_SAM, "max_dim": MAX_DIM}
        )
        d = save_dataset(rgb, label_map, depth,
                         source_name=image.filename or "image",
                         seg_params=seg_params,
                         depth_params={"source": "model", "model_name": DEPTH_MODEL, "infer_size": None},
                         force=heal)  # heal: overwrite the corrupt copy (edge configs survive)
        print(f"[dataset] saved {d.name}")

    # Edge conditioning: re-split superpixels so their boundaries follow real Canny object edges
    # instead of only the superpixel tessellation. Recomputed per session (not cached in the
    # dataset) -- it's CPU-cheap and deterministic from (rgb, base label_map, canny thresholds).
    if edge_condition:
        label_map, n_conditioned = condition_label_map(
            rgb, label_map, canny_low=canny_low, canny_high=canny_high)
        print(f"[edge_condition] regions -> {n_conditioned}")

    cfg = Config(n_layers=n_layers, fix_x=False, lambda_support=1.0, verbose=False)
    inst = build_instance(label_map, depth, cfg)

    # frozen dataset provenance for the run record (meta.json describes the *base*, pre-conditioning
    # segmentation/depth -- the original label_map_md5/n_regions, matching prepare_from_dataset).
    meta = json.loads((d / "meta.json").read_text())

    h, w = rgb.shape[:2]
    session_id = str(uuid.uuid4())
    sess = {
        "rgb": rgb, "label_map": label_map, "depth": depth, "inst": inst,
        "width": w, "height": h, "filename": image.filename or "image",
        "n_regions": inst.n_regions,
        "dataset_dir": str(d), "dataset_slug": d.name,
        # provenance threaded into _save_run's config.json (dataset + edge-conditioning knobs)
        "dataset_meta": meta,
        "edge_condition": edge_condition, "canny_low": canny_low, "canny_high": canny_high,
    }
    _sessions[session_id] = sess

    # Warm up SAM's image embedding here (same step as segmentation/depth) rather than lazily on
    # the first "select object" click, so that click is fast instead of paying the CPU
    # image-encoder cost right when the user is trying to use the feature. Best-effort: if it
    # fails (e.g. no network for the first-ever checkpoint download), leave sam_image_set unset
    # so /select-object's own lazy set_image still runs as a fallback.
    try:
        get_sam_predictor().set_image(rgb)
        sess["sam_image_set"] = True
    except Exception as e:
        print(f"[sam] warm-up failed, will retry lazily on first click: {e}")

    # depth-edge strength per boundary for the crease heatmap; meandiff matches the UI's default
    # cut-score toggle, so the first render shows the same field the solver would price
    scores = _display_scores(inst, depth, method="meandiff", log_sigma=2.0)
    # Visual edge strength per boundary: independent of depth, so a visually sharp silhouette
    # still reads as a real edge even where the depth estimate is too smooth/noisy to show a jump.
    # Uses continuous gradient magnitude (image_gradient_strength), not the binary Canny raster --
    # after edge-conditioning, most boundaries were literally created by wherever Canny fired, so
    # a binary "coincides with that raster" check saturates near 1.0 for the majority of them and
    # can't tell a strong real edge from a faint one that barely cleared Canny's threshold.
    _ep = edge_params_for(rgb.shape)
    gradient_map = image_gradient_strength(rgb, blur_ksize=int(_ep["blur_ksize"]),
                                           blur_sigma=_ep["blur_sigma"])
    canny_align = edge_gradient_strength(inst.label_map, inst.region_ids, gradient_map)
    edges = _edges_payload(inst, scores, canny_align)
    # initial view = the plain photo; the frontend draws the superpixel tessellation as SVG on
    # top, so upload shows the segmentation only -- no solved-looking depth bins before a solve
    baseline = _data_url(_png_bytes(rgb))

    # lossless per-pixel positional-region-index map (idx+1: R = low byte, G = high byte,
    # 0 = unmodeled pixel) so the frontend can hit-test regions under the cursor
    lut = np.full(int(label_map.max()) + 1, -1, dtype=np.int64)
    lut[np.asarray(inst.region_ids)] = np.arange(inst.n_regions)
    idx1 = (lut[label_map] + 1).astype(np.uint16)
    enc = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    enc[..., 0] = (idx1 & 0xFF).astype(np.uint8)
    enc[..., 1] = (idx1 >> 8).astype(np.uint8)
    region_map = _data_url(_png_bytes(enc))

    print(f"[create_session] {w}x{h} | regions={inst.n_regions} | edges={len(edges)}")
    return {
        "sessionId": session_id, "width": w, "height": h,
        "nLayers": n_layers, "nRegions": inst.n_regions,
        "edges": edges, "baselineOverlay": baseline,
        "regionMap": region_map, "regionDepth": [float(v) for v in inst.mean_depth],
        "datasetSlug": d.name, "edgeConfigs": list_edge_configs(d),
    }


@app.post("/api/sessions/{session_id}/solve")
async def solve_session(session_id: str, req: SolveRequest):
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    try:
        inst, sol, timing, marks = _solve_layers(sess, req)
    except InfeasibleError as e:
        raise HTTPException(status_code=409, detail=f"No feasible layering: {e}")
    except ValueError as e:  # e.g. min_layer_regions * n_layers > n_regions
        raise HTTPException(status_code=400, detail=str(e))

    # Cache the solved (inst, sol) so a following /export-ai with the same solve params reuses it
    # instead of re-solving (a time-limited MIP would otherwise burn ~time_limit again AND could
    # return a *different* incumbent than the one previewed here).
    sess.update(inst=inst, sol=sol, solve_sig=_solve_signature(req), marks=marks)

    overlay_img = _overlay_img(inst, sol.layer_of, sess["rgb"])
    layer_imgs = _layer_sheets(inst, sol, sess["rgb"])  # retained material (support marked red)
    run_dir = _save_run(sess, session_id, req, inst, sol, timing, marks)

    print(f"[solve] {len(req.markings)} marks | N={req.n_layers} | {sol.status} "
          f"obj={sol.obj_total:.3f} in {sol.runtime:.1f}s -> runs/{os.path.basename(run_dir)}")
    photo_masks, sheet_masks = _layer_masks(inst, sol, sess["rgb"])
    return {
        "overlay": _data_url(_png_bytes(overlay_img)),
        "layers": [_data_url(_png_bytes(im)) for im in layer_imgs],
        # per-layer material textures for the 3D stacked book preview (front = layer_01):
        # photo texture + .ai-style fabrication sheet, both incl. support material
        "masks": [_data_url(_png_bytes_rgba(m)) for m in photo_masks],
        "sheetMasks": [_data_url(_png_bytes_rgba(m)) for m in sheet_masks],
        "nLayers": req.n_layers, "objective": sol.obj_breakdown,
        "status": sol.status, "runtime": sol.runtime,
        "z": sol.extra.get("z"), "layerOf": [int(v) for v in sol.layer_of],
    }


@app.post("/api/sessions/{session_id}/scores")
async def recompute_scores(session_id: str, req: ScoresRequest):
    """Recompute per-boundary depth-edge strength for the crease map. No solve -> fast; used to
    redraw the heatmap live as the user drags the LoG-scale slider."""
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    scores = _display_scores(sess["inst"], sess["depth"],
                             method=req.cut_score, log_sigma=req.cut_log_sigma)
    return {"scores": [float(s) for s in scores]}


@app.post("/api/sessions/{session_id}/select-object")
async def select_object(session_id: str, req: SelectObjectRequest):
    """Point-prompted SAM: click a point -> mark every boundary tracing that object's outline.

    The predictor's image embedding is set once per session (the slow CPU encoder pass) and
    reused for every subsequent click, whether on this object or another one in the same photo.
    """
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")

    predictor = get_sam_predictor()
    if not sess.get("sam_image_set"):
        predictor.set_image(sess["rgb"])
        sess["sam_image_set"] = True

    mask, score = predict_point_mask(predictor, req.x, req.y)
    outline = find_boundaries(mask, mode="outer")
    inst = sess["inst"]
    align = edge_canny_alignment(inst.label_map, inst.region_ids, outline, dilate=2)

    edge_indices = [
        e for e, (i, j) in enumerate(inst.edges)
        if align.get((int(i), int(j)), 0.0) >= SELECT_OBJECT_ALIGN_THRESHOLD
    ]
    return {"edgeIndices": edge_indices, "score": score}


@app.get("/api/sessions/{session_id}/edge-configs")
async def get_edge_configs(session_id: str):
    """Named edge configs stored in this session's dataset dir."""
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    return {"configs": list_edge_configs(sess["dataset_dir"])}


@app.post("/api/sessions/{session_id}/edge-configs")
async def save_session_edge_config(session_id: str, req: EdgeConfigSaveRequest):
    """Persist the current UI marks as ``edges/<name>.json`` (original region-label pairs)."""
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    pairs = indices_to_pairs(sess["inst"], [{"index": m.index, "type": m.type}
                                            for m in req.markings])
    try:
        save_edge_config(sess["dataset_dir"], req.name, pairs,
                         lm_md5=label_map_md5(sess["inst"].label_map), n_layers=req.n_layers)
    except ValueError as e:  # bad name or unknown mark type
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "name": req.name, "nMarks": len(pairs),
            "configs": list_edge_configs(sess["dataset_dir"])}


@app.get("/api/sessions/{session_id}/edge-configs/{name}")
async def load_session_edge_config(session_id: str, name: str):
    """Resolve a stored config back to UI edge indices for this session's instance."""
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    try:
        cfg = load_edge_config(sess["dataset_dir"], name)
        resolved = resolve_marks(sess["inst"], cfg)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No edge config named {name!r}")
    except ValueError as e:  # bad name or label_map_md5 mismatch (segmentation changed)
        raise HTTPException(status_code=409, detail=str(e))
    return {"name": name, "markings": resolved["markings"], "nLayers": cfg.get("n_layers")}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"ok": True}


@app.post("/api/sessions/{session_id}/export-stand")
async def export_stand(session_id: str, req: StandRequest):
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    if req.n_layers < 1:
        raise HTTPException(status_code=400, detail="n_layers must be at least 1")

    from points import build_stand_ai
    try:
        ai_content = build_stand_ai(n_layers=req.n_layers, spoke_h_in=req.spoke_h_in,
                                    base_h_in=req.base_h_in)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stand generation failed: {e}")

    base = re.sub(r"[^\x00-\x7F]+", "_", sess["filename"].rsplit(".", 1)[0])
    return Response(content=ai_content.encode("latin-1"), media_type="application/postscript",
                    headers={"Content-Disposition": f'attachment; filename="{base}_stand.ai"'})


@app.post("/api/sessions/{session_id}/export-ai")
async def export_ai(session_id: str, req: ExportLayersRequest):
    """Return a zip of per-layer ``.ai`` sheets (cut silhouette + engrave outlines), built by the
    shared ``export.build_layer_ai_docs``.

    Export settings (mode, per-layer engrave density) do **not** affect the solve, so if the
    session already holds a solution for these exact solve params -- from the interactive solve, a
    prior export, or a loaded run -- it is reused and nothing re-solves.  Only a genuine change to
    the solve params (or exporting before any solve) triggers a solve; that result is cached too.
    """
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    sig = _solve_signature(req)
    sol = sess.get("sol")
    if sol is None or sess.get("solve_sig") != sig:
        try:
            _inst, sol, _timing, _marks = _solve_layers(sess, req)
        except InfeasibleError as e:
            raise HTTPException(status_code=409, detail=f"No feasible layering: {e}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        sess.update(inst=_inst, sol=sol, solve_sig=sig, marks=_marks)
    else:
        print(f"[export] reusing cached solve for {session_id[:8]} (no re-solve)")

    n_layers = sol.inst.n_layers
    for name, arr in (("engrave_layers", req.engrave_layers),
                      ("min_engrave_in_per_layer", req.min_engrave_in_per_layer)):
        if arr is not None and len(arr) != n_layers:
            raise HTTPException(status_code=400,
                                detail=f"{name} must have length {n_layers}, got {len(arr)}")

    try:
        # centerline engrave (single-burn skeleton strokes) -- matches run_one's sweep exports
        # and the 3D preview's sheet texture; "canny" loops burn every edge twice.  Per-layer
        # engrave_layers / min_engrave_in_per_layer override the global mode/floor for that plane.
        docs = build_layer_ai_docs(sol, content_width_in=req.content_width_in,
                                   engrave=(req.mode == "engraving"), border_in=req.border_in,
                                   rgb=sess["rgb"], engrave_style="centerline",
                                   engrave_layers=req.engrave_layers,
                                   min_engrave_in_per_layer=req.min_engrave_in_per_layer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Layer export failed: {e}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, doc in docs.items():
            zf.writestr(name, doc.encode("latin-1"))
    base = re.sub(r"[^\x00-\x7F]+", "_", sess["filename"].rsplit(".", 1)[0])
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{base}_layers.zip"'})


class EngravePreviewRequest(BaseModel):
    layer: int = 0
    min_engrave_in: float = 0.03
    content_width_in: float = 12.0
    mode: str = "engraving"          # "engraving" | "outline" (outline = cut only, no burns)


@app.post("/api/sessions/{session_id}/engrave-preview")
def engrave_preview(session_id: str, req: EngravePreviewRequest):
    """Render one layer's fabrication sheet at a given engrave detail floor, from the session's
    cached solution -- no solving.  Pairs the per-layer density slider with a live view of exactly
    what that floor burns (same centerlines the .ai export writes)."""
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    sol = sess.get("sol")
    if sol is None:
        raise HTTPException(status_code=409, detail="No solution yet -- solve or load a run first")
    img, n_strokes = _engrave_preview_img(
        sol, sess["rgb"], req.layer, req.min_engrave_in, req.content_width_in,
        engrave=(req.mode == "engraving"))
    return {"image": _data_url(_png_bytes_rgba(img)), "layer": int(req.layer), "nStrokes": n_strokes}


class LayerStackRequest(BaseModel):
    min_engrave_in_per_layer: Optional[List[float]] = None
    content_width_in: float = 12.0
    mode: str = "engraving"                 # "engraving" | "outline"
    engrave_layers: Optional[List[bool]] = None


@app.post("/api/sessions/{session_id}/layer-stack")
def layer_stack(session_id: str, req: LayerStackRequest):
    """All layers' 3D-stack textures (photo + fabrication sheet) from the cached solve, rendered at
    the given per-layer engrave densities -- no solving.  Feeds the ``BookVisualizer`` so the 3D
    tunnel-book preview mirrors exactly the density the user is tuning."""
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    sol = sess.get("sol")
    if sol is None:
        raise HTTPException(status_code=409, detail="No solution yet -- solve or load a run first")
    photo, sheet = _layer_masks(
        sol.inst, sol, sess["rgb"],
        min_engrave_in_per_layer=req.min_engrave_in_per_layer,
        content_width_in=req.content_width_in, engrave=(req.mode == "engraving"),
        engrave_layers=req.engrave_layers)
    return {"masks": [_data_url(_png_bytes_rgba(m)) for m in photo],
            "sheetMasks": [_data_url(_png_bytes_rgba(m)) for m in sheet]}


class ExportFrontViewRequest(BaseModel):
    min_engrave_in_per_layer: Optional[List[float]] = None  # per-plane detail floor (in); None=0.03
    engrave_layers: Optional[List[bool]] = None             # per-plane engrave on/off; None = all on
    content_width_in: float = 12.0    # physical artwork width -> same in/px mapping as the .ai export
    mode: str = "engraving"           # "engraving" | "outline" (outline = cut silhouettes, no burns)
    backdrop: str = "wood"            # "wood" | "plain"
    perspective: float = 0.037        # per-layer foreshortening step; 0 = flat front view
    out_long_px: int = 1500           # supersample target for the long edge


@app.post("/api/sessions/{session_id}/export-front-view")
def export_front_view(session_id: str, req: ExportFrontViewRequest):
    """Zip of the stacked plywood front view (front_view.png) + each layer's wood sheet
    (wood_layer_NN.png, RGBA, front first), rendered from the cached solution at the given
    per-layer engrave densities -- no solving.  Same sheets examples/render_front_view.py writes."""
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    sol = sess.get("sol")
    if sol is None:
        raise HTTPException(status_code=409, detail="No solution yet -- solve or load a run first")
    n_layers = sol.inst.n_layers
    for name, arr in (("engrave_layers", req.engrave_layers),
                      ("min_engrave_in_per_layer", req.min_engrave_in_per_layer)):
        if arr is not None and len(arr) != n_layers:
            raise HTTPException(status_code=400,
                                detail=f"{name} must have length {n_layers}, got {len(arr)}")
    try:
        layers = wood_layer_stack(sol.inst, sol, sess["rgb"],
                                  content_width_in=req.content_width_in,
                                  min_engrave_in_per_layer=req.min_engrave_in_per_layer,
                                  engrave=(req.mode == "engraving"),
                                  engrave_layers=req.engrave_layers,
                                  out_long_px=req.out_long_px)
        fv = render_front_view(layers, backdrop=req.backdrop, perspective=req.perspective)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Front view render failed: {e}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("front_view.png", _png_bytes(fv))
        for i, layer in enumerate(layers, 1):     # 1-indexed, as examples/render_front_view.py
            zf.writestr(f"wood_layer_{i:02d}.png", _png_bytes_rgba(layer))
    base = re.sub(r"[^\x00-\x7F]+", "_", sess["filename"].rsplit(".", 1)[0])
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{base}_front_view.zip"'})


class ExportLayerMapRequest(BaseModel):
    out_long_px: int = 2000    # supersample target for the long edge


@app.post("/api/sessions/{session_id}/export-layer-map")
def export_layer_map(session_id: str, req: ExportLayerMapRequest):
    """High-res layer-assignment map PNG from the cached solution -- no solving.  Stretched
    viridis (front = dark purple, back = yellow, like the gallery fig) over the grayscale photo,
    with a numbered-squares legend (tunnelbook.viz.layermap)."""
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    sol = sess.get("sol")
    if sol is None:
        raise HTTPException(status_code=409, detail="No solution yet -- solve or load a run first")
    try:
        img = render_layer_map(sol.inst, sol, sess["rgb"], out_long_px=req.out_long_px)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Layer map render failed: {e}")
    base = re.sub(r"[^\x00-\x7F]+", "_", sess["filename"].rsplit(".", 1)[0])
    return Response(content=_png_bytes(img), media_type="image/png",
                    headers={"Content-Disposition": f'attachment; filename="{base}_layer_map.png"'})


# ── load a previous solve (re-export at new settings without re-solving) ─────────
#
# Every solve -- a UI run (RUNS_DIR) or a cluster sweep (image_outputs/) -- persists a full
# result.npz (tunnelbook.serialize) that load_result() round-trips back into a Solution.  The
# export path (build_layer_ai_docs) needs only that Solution + the source photo, so a saved solve
# can be loaded into a fresh session and re-exported at any per-layer engrave density with zero
# solving.  The photo isn't in the npz; it comes from the run's dataset (datasets/<slug>).
def _run_roots() -> List[tuple]:
    """(label, abs_root) dirs scanned for saved runs: the UI's own ``runs/`` plus the repo's
    ``image_outputs/`` (cluster sweeps) and any ``TUNNELBOOK_RUN_ROOTS`` (os.pathsep-separated).
    Order is stable, so a run id's root index means the same thing between list and load."""
    roots = [("ui", RUNS_DIR)]
    try:
        img = os.path.join(os.path.dirname(str(datasets_root())), "image_outputs")
        roots.append(("sweeps", img))
    except Exception:
        pass
    for extra in filter(None, os.environ.get("TUNNELBOOK_RUN_ROOTS", "").split(os.pathsep)):
        roots.append((os.path.basename(extra.rstrip("/\\")) or "runs", extra))
    seen, out = set(), []
    for label, r in roots:
        rp = os.path.realpath(r)
        if os.path.isdir(rp) and rp not in seen:
            seen.add(rp)
            out.append((label, rp))
    return out


def _encode_run_id(root_idx: int, run_dir: str, root: str) -> str:
    raw = f"{root_idx}:{os.path.relpath(run_dir, root)}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def _decode_run_id(run_id: str) -> str:
    """run_id -> validated absolute run dir (inside one of the roots, containing result.npz).
    Guards against path traversal: the resolved dir must stay within its declared root."""
    try:
        raw = base64.urlsafe_b64decode(run_id.encode()).decode()
        idx_s, rel = raw.split(":", 1)
        root_idx = int(idx_s)
    except Exception:
        raise HTTPException(status_code=400, detail="malformed run id")
    roots = _run_roots()
    if not 0 <= root_idx < len(roots):
        raise HTTPException(status_code=404, detail="unknown run root")
    root = roots[root_idx][1]
    cand = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath([cand, root]) != root:
        raise HTTPException(status_code=400, detail="run id escapes its root")
    if not os.path.isfile(os.path.join(cand, "result.npz")):
        raise HTTPException(status_code=404, detail="run has no result.npz")
    return cand


def _resolve_run_path(path: str) -> str:
    """User-typed run path -> validated absolute run dir (must contain result.npz).

    Accepts the run dir itself or its result.npz; absolute, ``~``-relative, or relative to the
    repo root (the dir holding datasets/ -- same anchor as ``_run_roots``) or the CWD.  Unlike
    run ids this deliberately allows arbitrary absolute paths: it is the escape hatch for runs
    beyond the listing cap on a local dev tool; result.npz presence is the gate."""
    p = os.path.expanduser(path.strip())
    cands = [p] if os.path.isabs(p) else [
        os.path.join(os.path.dirname(str(datasets_root())), p),   # repo root, as _run_roots
        os.path.abspath(p),                                       # CWD fallback
    ]
    for cand in cands:
        cand = os.path.realpath(cand)
        if os.path.basename(cand) == "result.npz":
            cand = os.path.dirname(cand)
        if os.path.isfile(os.path.join(cand, "result.npz")):
            return cand
    raise HTTPException(status_code=404,
                        detail=f"no result.npz found for {path!r} (tried: {', '.join(cands)})")


def _run_display_config(cfg: dict) -> dict:
    """config.json (PipelineParams space) -> the SolveRequest/frontend knobs, inverting
    ``_req_to_pipeline_params``.  ``depth_model=="kmedian"`` is the server's cut branch."""
    objective = "cut" if cfg.get("depth_model") == "kmedian" else "depth"
    return {
        "objective": objective,
        "nLayers": int(cfg.get("layers", 5)),
        "lambdaDepth": float(cfg.get("lambda_depth", 0.0)) if objective == "cut" else 0.0,
        "minLayerArea": float(cfg.get("min_layer_area", 0.10)),
        "cutScore": cfg.get("cut_score", "laplacian"),
        "cutLogSigma": float(cfg.get("cut_log_sigma", 2.0)),
        "connectivity": bool(cfg.get("connectivity", True)),
        "connectivityMethod": cfg.get("connectivity_method", "flow"),
        "yMonotone": bool(cfg.get("y_monotone", False)),
        "norelTime": float(cfg.get("norel_time", 60.0)),
        "mipFocus": int(cfg.get("mip_focus", 1)),
        "lambdaCoherence": float(cfg.get("lambda_coherence", 0.0)),
        "lambdaSplit": float(cfg.get("lambda_split", 1.0)),
    }


def _run_thumb(run_dir: str, width: int = 220) -> Optional[str]:
    """Small JPEG data URL from the run's overview.png (or None); lets the picker show the run."""
    fp = os.path.join(run_dir, "overview.png")
    if not os.path.isfile(fp):
        return None
    try:
        img = cv2.imread(fp)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, max(1, round(h * width / w))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None
    except Exception:
        return None


def _run_meta(run_dir: str, root_idx: int, root: str, label: str, with_thumb: bool) -> dict:
    try:
        cfg = json.loads((Path(run_dir) / "config.json").read_text())
    except Exception:
        cfg = {}
    status = obj = None
    rowfp = Path(run_dir) / "row.json"
    if rowfp.is_file():
        try:
            row = json.loads(rowfp.read_text())
            status, obj = row.get("status"), row.get("obj_total")
        except Exception:
            pass
    disp = _run_display_config(cfg)
    meta = {
        "id": _encode_run_id(root_idx, run_dir, root),
        "name": os.path.basename(run_dir),
        "source": label,
        "slug": (cfg.get("dataset") or {}).get("slug"),
        "nLayers": disp["nLayers"], "objective": disp["objective"],
        "cutScore": disp["cutScore"], "lambdaDepth": disp["lambdaDepth"],
        "minLayerArea": disp["minLayerArea"], "status": status, "objTotal": obj,
        "mtime": os.path.getmtime(run_dir),
    }
    if with_thumb:
        meta["thumb"] = _run_thumb(run_dir)
    return meta


@app.get("/api/runs")
def list_runs(limit: int = 40, thumbs: bool = True):
    """Saved solves across the run roots, most-recent first, each re-exportable via /api/load-run."""
    found = []
    for root_idx, (label, root) in enumerate(_run_roots()):
        for dirpath, dirnames, filenames in os.walk(root):
            if "result.npz" in filenames:
                found.append((os.path.getmtime(dirpath), root_idx, label, root, dirpath))
                dirnames[:] = []  # a run dir holds no nested runs -- don't descend
            if len(found) >= 4000:
                break
    found.sort(key=lambda t: t[0], reverse=True)
    cap = 200 if thumbs else 1000     # thumbs (per-run JPEG encode) are the expensive part
    found = found[:max(1, min(int(limit), cap))]
    return {"runs": [_run_meta(d, ri, root, label, thumbs) for _mt, ri, label, root, d in found]}


class LoadRunRequest(BaseModel):
    runId: Optional[str] = None   # id from /api/runs
    path: Optional[str] = None    # direct run-dir (or result.npz) path; absolute or repo-relative


@app.post("/api/load-run")
def load_run(req: LoadRunRequest):
    """Hydrate a fresh session from a saved run's result.npz (+ its dataset photo) and return the
    solved layering + its config.  Export then reuses this solution -- change the per-layer engrave
    density and re-export with no solve.  (Changing a *solve* param would re-solve, as usual.)"""
    if bool(req.runId) == bool(req.path):
        raise HTTPException(status_code=400, detail="provide exactly one of runId or path")
    run_dir = _decode_run_id(req.runId) if req.runId else _resolve_run_path(req.path)
    try:
        cfg = json.loads((Path(run_dir) / "config.json").read_text())
    except Exception:
        cfg = {}
    slug = (cfg.get("dataset") or {}).get("slug")
    if not slug:
        raise HTTPException(status_code=400,
                            detail="run has no dataset provenance; cannot recover its source image")
    dset_dir = Path(datasets_root()) / slug
    if not (dset_dir / "meta.json").is_file():
        raise HTTPException(status_code=404,
                            detail=f"dataset {slug!r} not found under {datasets_root()}; "
                                   "cannot recover the source image for this run")
    try:
        sol = load_result(str(Path(run_dir) / "result.npz"))  # the solve, straight from disk
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not load solution: {e}")
    inst = sol.inst
    b = load_dataset(dset_dir)                                 # source photo + depth (npz has neither)
    rgb, depth = b.rgb, b.depth
    disp = _run_display_config(cfg)
    n = inst.n_layers

    # A SolveRequest carrying this run's solve params (no marks): its signature is what the frontend
    # reproduces on export, so the loaded sol is reused instead of re-solved.
    req0 = SolveRequest(
        markings=[], n_layers=n, objective=disp["objective"], lambda_depth=disp["lambdaDepth"],
        min_layer_area=disp["minLayerArea"], connectivity=disp["connectivity"],
        connectivity_method=disp["connectivityMethod"], y_monotone=disp["yMonotone"],
        norel_time=disp["norelTime"], mip_focus=disp["mipFocus"],
        lambda_coherence=disp["lambdaCoherence"], cut_score=disp["cutScore"],
        cut_log_sigma=disp["cutLogSigma"], lambda_split=disp["lambdaSplit"],
    )
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "rgb": rgb, "label_map": inst.label_map, "depth": depth, "inst": inst,
        "sol": sol, "solve_sig": _solve_signature(req0), "marks": None,
        "width": rgb.shape[1], "height": rgb.shape[0], "filename": os.path.basename(run_dir),
        "n_regions": inst.n_regions, "dataset_dir": str(dset_dir), "dataset_slug": slug,
        "dataset_meta": b.meta, "edge_condition": bool(cfg.get("edge_condition", False)),
        "canny_low": int(cfg.get("canny_low", 50)), "canny_high": int(cfg.get("canny_high", 150)),
        "loaded_from": run_dir,
    }
    print(f"[load-run] {os.path.basename(run_dir)} -> session {session_id[:8]} "
          f"| {slug} | N={n} | {sol.status}")
    return {
        "sessionId": session_id, "nLayers": n, "status": sol.status, "runtime": sol.runtime,
        "objective": sol.obj_breakdown, "layerOf": [int(v) for v in sol.layer_of],
        "z": sol.extra.get("z"),
        "overlay": _data_url(_png_bytes(_overlay_img(inst, sol.layer_of, rgb))),
        "datasetSlug": slug, "runName": os.path.basename(run_dir), "config": disp,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
