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
import uuid
import zipfile
from datetime import datetime
from typing import List, Optional

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

from tunnelbook.config import Config
from tunnelbook.data.dataset import (
    find_dataset, indices_to_pairs, label_map_md5, list_edge_configs,
    load_dataset, load_edge_config, resolve_marks, save_dataset, save_edge_config,
)
from tunnelbook.data.edges import (
    condition_label_map, detect_canny_raster, edge_canny_alignment, edge_gradient_strength,
    edge_params_for, image_gradient_strength,
)
from tunnelbook.data.labelmap import build_instance, edge_depth_scores, region_boundary_segments
from tunnelbook.data.realimage import compute_depth, segment_image
from tunnelbook.data.sam_point import get_sam_predictor, predict_point_mask
from tunnelbook.data.spam_segment import SAM_CHECKPOINT, segment_image_spam
from tunnelbook.export import build_layer_ai_docs
from tunnelbook.model import build_model
from tunnelbook.solve import InfeasibleError, solve

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
    min_layer_area: float = 0.10      # cut objective (DEFAULT floor): force each layer to own >= this fraction of image area
    min_layer_regions: int = 0        # cut objective: optional count floor (>= k superpixels/layer); 0 = off (area floor is default)
    lambda_cut: float = 1.0           # cut objective: weight on the boundary-cut term
    lambda_depth: float = 0.0         # cut objective: k-median depth-anchor strength (0 = pure cut;
                                      # marks stop mattering above ~0.5, so the useful range is 0..0.5)
    connectivity: bool = True         # flow (fabrication) constraints; off = pieces may float
    connectivity_method: str = "flow"  # "flow" | "lazy" (cut-set rows added lazily at MIPSOL)
    y_monotone: bool = False          # y == cumsum(x): "full backing" (fewer free binaries)
    norel_time: float = 60.0          # cut objective: seconds of Gurobi NoRel heuristic (0 = off);
                                      # rescues N>=4 incumbents (japan N=7: none -> comps 11, rho .90)
    mip_focus: int = 1                # cut objective: Gurobi MIPFocus (1 = feasibility focus)
    cut_score: str = "laplacian"      # cut objective: "laplacian" | "meandiff"
    cut_log_sigma: float = 2.0        # cut objective: LoG spatial scale (px) for the crease score
    lambda_coherence: float = 0.0     # cut objective: depth-plateau coherence tie-breaker (0 = off).
                                      # ON (0.05) fixes free plateau splits (robert-keane N=5:
                                      # mountain+sky unified, z spread, penalty paid ~0), but on
                                      # hard mark-heavy N=7 solves the extra rows degraded the
                                      # 180s incumbent (redcharlie: z collapse, rho -0.57 vs 0.05)
                                      # -- hence default OFF; enable via the UI toggle per solve
    coherence_eps: float = 0.02       # plateau depth-gap threshold for the coherence pairs


class ScoresRequest(BaseModel):
    cut_score: str = "laplacian"
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


def _layer_imgs(inst, layer_of, rgb: np.ndarray) -> List[np.ndarray]:
    """One image per layer (semantic view): that layer's regions in full colour, rest dimmed.
    Used for the baseline preview, which has no solve -> no support to show."""
    lay = inst.paint(np.asarray(layer_of, dtype=float))     # (H,W), 1..N per region
    dim = (rgb * 0.18).astype(np.uint8)
    out = []
    for l in range(1, inst.n_layers + 1):
        img = dim.copy()
        m = lay == l
        img[m] = rgb[m]
        out.append(img)
    return out


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


def _layer_masks(inst, sol, rgb: np.ndarray, max_dim: int = 512):
    """Per-layer RGBA textures for the 3D book-stack preview, TWO styles per layer.

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
    engrave = cv2.dilate(cv2.Canny(rgb, 30, 100), np.ones((2, 2), np.uint8)) > 0
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
        sh[..., :3][vis & engrave] = ink
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
                      cut_log_sigma=req.cut_log_sigma,
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
    # depth objective solves in seconds -- the NoRel/MIPFocus rescue is left off here
    return Config(n_layers=req.n_layers, fix_x=False, lambda_support=1.0,
                  lambda_split=req.lambda_split, mip_gap=0.01,
                  connectivity=req.connectivity,
                  connectivity_method=req.connectivity_method,
                  y_monotone=req.y_monotone,
                  time_limit=req.time_limit, verbose=False, log_progress=True)


def _solve_layers(sess: dict, req: "SolveRequest"):
    cfg = _build_cfg(req)
    inst = build_instance(sess["label_map"], sess["depth"], cfg)
    n_edges = len(inst.edges)
    split_soft, split_hard, delete = [], [], []
    for mk in req.markings:
        if not (0 <= mk.index < n_edges):
            continue
        i, j = inst.edges[mk.index]
        pair = (int(i), int(j))
        if mk.type == "split_soft":
            split_soft.append(pair)
        elif mk.type == "split_hard":
            split_hard.append(pair)
        elif mk.type == "delete":
            delete.append(pair)
    # depth_map is needed by the laplacian cut score; harmless for the depth objective.
    sol = solve(build_model(inst, cfg, depth_map=sess["depth"],
                            split_soft=split_soft or None,
                            split_hard=split_hard or None,
                            delete=delete or None))
    return inst, sol


def _save_run(session_id: str, req: "SolveRequest", inst, sol,
              overlay_img: np.ndarray, layer_imgs: List[np.ndarray]) -> str:
    """Persist one solve: inputs (marks + config) and outputs (overlay + per-layer PNGs)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(RUNS_DIR, f"{session_id[:8]}_{ts}")
    os.makedirs(d, exist_ok=True)
    marks = []
    for mk in req.markings:
        if 0 <= mk.index < len(inst.edges):
            i, j = inst.edges[mk.index]
            marks.append({"index": mk.index, "i": int(i), "j": int(j), "type": mk.type})
    meta = {
        "session": session_id, "timestamp": ts,
        "n_layers": req.n_layers, "lambda_split": req.lambda_split,
        "objective_mode": req.objective, "min_layer_regions": req.min_layer_regions,
        "min_layer_area": req.min_layer_area,
        "lambda_cut": req.lambda_cut, "cut_score": req.cut_score,
        # full solver config, so a run dir is never ambiguous about what actually ran
        "lambda_depth": req.lambda_depth, "cut_log_sigma": req.cut_log_sigma,
        "time_limit": req.time_limit,
        "connectivity": req.connectivity, "connectivity_method": req.connectivity_method,
        "y_monotone": req.y_monotone,
        "norel_time": req.norel_time, "mip_focus": req.mip_focus,
        "lambda_coherence": req.lambda_coherence, "coherence_eps": req.coherence_eps,
        "status": sol.status, "runtime": sol.runtime, "objective": sol.obj_breakdown,
        # gurobi diagnostics: gap/bound/nodes (+ lazy cut count and k-median z when present)
        "solver": sol.extra.get("solver"),
        "lazy_cuts": sol.extra.get("lazy_cuts"),
        "z": sol.extra.get("z"),
        "markings": marks, "layer_of": [int(v) for v in sol.layer_of],
    }
    with open(os.path.join(d, "run.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(d, "overlay.png"), "wb") as f:
        f.write(_png_bytes(overlay_img))
    for k, im in enumerate(layer_imgs, 1):
        with open(os.path.join(d, f"layer_{k}.png"), "wb") as f:
            f.write(_png_bytes(im))
    return d


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

    h, w = rgb.shape[:2]
    session_id = str(uuid.uuid4())
    sess = {
        "rgb": rgb, "label_map": label_map, "depth": depth, "inst": inst,
        "width": w, "height": h, "filename": image.filename or "image",
        "n_regions": inst.n_regions,
        "dataset_dir": str(d), "dataset_slug": d.name,
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

    # depth-edge strength per boundary (LoG laplacian, normalized [0,1]) for the crease heatmap
    scores = edge_depth_scores(inst, depth_map=depth, method="laplacian", log_sigma=2.0)
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
    # initial view = depth-binned layers, no solve -> instant upload; user solves on demand
    lhat1 = inst.lhat + 1
    baseline = _data_url(_png_bytes(_overlay_img(inst, lhat1, rgb)))
    baseline_layers = [_data_url(_png_bytes(im)) for im in _layer_imgs(inst, lhat1, rgb)]

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
        "edges": edges, "baselineOverlay": baseline, "baselineLayers": baseline_layers,
        "regionMap": region_map, "regionDepth": [float(v) for v in inst.mean_depth],
        "datasetSlug": d.name, "edgeConfigs": list_edge_configs(d),
    }


@app.post("/api/sessions/{session_id}/solve")
async def solve_session(session_id: str, req: SolveRequest):
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    try:
        inst, sol = _solve_layers(sess, req)
    except InfeasibleError as e:
        raise HTTPException(status_code=409, detail=f"No feasible layering: {e}")
    except ValueError as e:  # e.g. min_layer_regions * n_layers > n_regions
        raise HTTPException(status_code=400, detail=str(e))

    overlay_img = _overlay_img(inst, sol.layer_of, sess["rgb"])
    layer_imgs = _layer_sheets(inst, sol, sess["rgb"])  # retained material (support marked red)
    run_dir = _save_run(session_id, req, inst, sol, overlay_img, layer_imgs)

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
    scores = edge_depth_scores(sess["inst"], depth_map=sess["depth"],
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
    """Re-solve with the current marks/config and return a zip of per-layer ``.ai`` sheets
    (cut silhouette + engrave outlines), built by the shared ``export.build_layer_ai_docs``."""
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Unknown session")
    try:
        _inst, sol = _solve_layers(sess, req)
    except InfeasibleError as e:
        raise HTTPException(status_code=409, detail=f"No feasible layering: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        docs = build_layer_ai_docs(sol, content_width_in=req.content_width_in,
                                   engrave=(req.mode == "engraving"), border_in=req.border_in,
                                   rgb=sess["rgb"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Layer export failed: {e}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, doc in docs.items():
            zf.writestr(name, doc.encode("latin-1"))
    base = re.sub(r"[^\x00-\x7F]+", "_", sess["filename"].rsplit(".", 1)[0])
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{base}_layers.zip"'})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
