"""TunnelBook backend: SPAM superpixels -> mark region boundaries (split/delete) -> solve.

Flow:
  POST /api/sessions        upload image -> resize -> (cached) SPAM superpixels -> depth ->
                            ProblemInstance -> per-boundary segments + baseline layer solve.
  POST /api/sessions/{id}/solve   mark boundaries split_soft/split_hard/delete -> build the
                            MIP with those terms -> solve -> return a layer-assignment preview.

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

from tunnelbook.config import Config
from tunnelbook.data.labelmap import build_instance, edge_depth_scores, region_boundary_segments
from tunnelbook.data.realimage import compute_depth, segment_image
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
    cut_score: str = "laplacian"      # cut objective: "laplacian" | "meandiff"
    cut_log_sigma: float = 2.0        # cut objective: LoG spatial scale (px) for the crease score


class ScoresRequest(BaseModel):
    cut_score: str = "laplacian"
    cut_log_sigma: float = 2.0


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


def _edges_payload(inst, scores=None) -> List[dict]:
    """One clickable item per region-adjacency boundary, aligned to inst.edges order.

    ``scores`` (aligned to ``inst.edges``) is the normalized depth-edge strength s_e in [0,1]
    used for the frontend crease-map heatmap.
    """
    segs = region_boundary_segments(inst.label_map, inst.region_ids)
    out = []
    for e, (i, j) in enumerate(inst.edges):
        i, j = int(i), int(j)
        s = segs.get((i, j))
        score = float(scores[e]) if scores is not None and e < len(scores) else 0.0
        out.append({"index": e, "i": i, "j": j, "score": score,
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
        return Config(n_layers=req.n_layers, fix_x=False,
                      depth_model="bins", lambda_depth=0.0, lambda_support=0.0,
                      lambda_cut=req.lambda_cut, cut_cost="aware", cut_score=req.cut_score,
                      cut_log_sigma=req.cut_log_sigma,
                      use_contact_weight=True,  # kappa_e scales with shared-boundary length
                      min_layer_area=req.min_layer_area,
                      min_layer_regions=req.min_layer_regions,
                      lambda_split=0.0, mip_gap=0.02,
                      time_limit=req.time_limit, verbose=False)
    return Config(n_layers=req.n_layers, fix_x=False, lambda_support=1.0,
                  lambda_split=req.lambda_split, mip_gap=0.01,
                  time_limit=req.time_limit, verbose=False)


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
        "status": sol.status, "runtime": sol.runtime, "objective": sol.obj_breakdown,
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
async def create_session(image: UploadFile = File(...), n_layers: int = Form(5)):
    try:
        rgb = _load_rgb(await image.read())
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")

    label_map = _segment(rgb)
    depth = compute_depth(rgb, "model")  # Depth Anything V2 (the laplacian cut needs real depth)
    cfg = Config(n_layers=n_layers, fix_x=False, lambda_support=1.0, verbose=False)
    inst = build_instance(label_map, depth, cfg)

    h, w = rgb.shape[:2]
    session_id = str(uuid.uuid4())
    sess = {
        "rgb": rgb, "label_map": label_map, "depth": depth, "inst": inst,
        "width": w, "height": h, "filename": image.filename or "image",
        "n_regions": inst.n_regions,
    }
    _sessions[session_id] = sess

    # depth-edge strength per boundary (LoG laplacian, normalized [0,1]) for the crease heatmap
    scores = edge_depth_scores(inst, depth_map=depth, method="laplacian", log_sigma=2.0)
    edges = _edges_payload(inst, scores)
    # initial view = depth-binned layers, no solve -> instant upload; user solves on demand
    lhat1 = inst.lhat + 1
    baseline = _data_url(_png_bytes(_overlay_img(inst, lhat1, rgb)))
    baseline_layers = [_data_url(_png_bytes(im)) for im in _layer_imgs(inst, lhat1, rgb)]

    print(f"[create_session] {w}x{h} | regions={inst.n_regions} | edges={len(edges)}")
    return {
        "sessionId": session_id, "width": w, "height": h,
        "nLayers": n_layers, "nRegions": inst.n_regions,
        "edges": edges, "baselineOverlay": baseline, "baselineLayers": baseline_layers,
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
    return {
        "overlay": _data_url(_png_bytes(overlay_img)),
        "layers": [_data_url(_png_bytes(im)) for im in layer_imgs],
        "nLayers": req.n_layers, "objective": sol.obj_breakdown,
        "status": sol.status, "runtime": sol.runtime,
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
                                   engrave=(req.mode == "engraving"), border_in=req.border_in)
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
