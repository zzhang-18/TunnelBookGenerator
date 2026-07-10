# TunnelBook UI

Interactive front-end for the tunnelbook solver: **upload a photo → mark region boundaries
(split / delete) → solve the layering → export laser-cut sheets**. A React + Vite app talks to a
FastAPI backend that wraps the `tunnelbook` solver package in the repo root.

```
generator/
  src/App.tsx                 the whole React UI (home → edge-marking → output/export)
  src/checkpoint1/server.py   FastAPI backend: segment → mark boundaries → solve → export
  src/checkpoint1/points.py   PostScript writer for the laser-cut *stand* (.ai)
```

## Prerequisites

- **Node** ≥ 18 and npm (for the Vite frontend).
- **[`uv`](https://docs.astral.sh/uv/)** (for the Python backend; it resolves the repo's env).
- **A Gurobi license** at `~/gurobi.lic` — the solve step needs it. A free academic license
  removes the size cap (see the root `README.md`).

## Running it

Two processes: the backend on `:8000`, the frontend dev server on `:5173`. The Vite dev server
**proxies `/api` → `http://localhost:8000`**, so no API-base configuration is needed in dev.

### 1. Backend (FastAPI)

From `generator/src/checkpoint1`, run uvicorn in the repo's uv env with the `serve` (FastAPI) and
`depth` (torch + Depth Anything V2) extras:

```bash
cd generator/src/checkpoint1
uv run --project ../../.. --extra serve --extra depth \
    python -m uvicorn server:app --port 8000
```

`--project ../../..` points uv at the repo root (where `pyproject.toml` lives).

### 2. Frontend (Vite)

```bash
cd generator
npm install
npm run dev          # http://localhost:5173
```

Open the printed URL, upload an image, mark boundaries, **solve / preview**, then **export →** to
the output screen.

## Environment variables

| var | default | meaning |
|-----|---------|---------|
| `SEG_BACKEND` | `spam` | Superpixel backend. `spam` = learned superpixels (**needs CUDA** + a SAM download). `slic` = CPU-only skimage SLIC (`tunnelbook.data.realimage.segment_image`) — no CUDA, no downloads. |
| `SPAM_SAM` | `1` | `0` skips the SAM refinement in the SPAM backend (faster, coarser). Ignored when `SEG_BACKEND=slic`. |

### CPU-only dev path (no GPU)

If you don't have CUDA, run with the SLIC backend:

```bash
cd generator/src/checkpoint1
SEG_BACKEND=slic uv run --project ../../.. --extra serve --extra depth \
    python -m uvicorn server:app --port 8000
```

SLIC segmentation is pure CPU. The depth step (Depth Anything V2) also runs on CPU (slower, but
works — the SPAM-specific CUDA code is never imported in this path). This is enough to exercise the
full upload → mark → solve → export flow on a laptop.

## Exports (output screen)

- **`export_layers.zip`** — re-solves with your current marks and returns one `layer_0k.ai`
  (Adobe Illustrator / PostScript) per layer: red = laser cut silhouette (retained material),
  blue = engrave outlines. The **outline / engraving** toggle (home screen) and frame width /
  border feed straight in. Built by `tunnelbook.export.build_layer_ai_docs`; the server endpoint
  is `POST /api/sessions/{id}/export-ai`.
- **`export_stand.ai`** — the laser-cut display stand (`points.build_stand_ai`), independent of the
  layering.

## Extending: Canny-edge conditioning (handoff)

The goal of the in-progress feature is: **detect Canny edges in the UI and split superpixels along
them**, so the region boundaries the user marks follow real object edges instead of only the
superpixel tessellation.

**How boundaries work today.** After segmentation, the backend enumerates every region-adjacency
boundary and sends it to the UI as a clickable edge:

- `server.py:_edges_payload` builds the list from `region_boundary_segments(inst.label_map,
  inst.region_ids)` (in `src/tunnelbook/data/labelmap.py`). Each item is `{index, i, j, score,
  segments}`, aligned to `inst.edges`.
- The UI (`App.tsx`, `EdgeSelectionScreen`) draws those `segments` as SVG polylines; clicking one
  marks it `split_soft` / `split_hard` / `delete`. Marks post to `/solve` (and now `/export-ai`)
  by their `index` into `inst.edges`.

So today's editable boundaries are exactly the **superpixel adjacencies** — nothing splits a
superpixel that a Canny edge cuts through.

**The building blocks to wire in** (all in `src/tunnelbook/data/edges.py`, already exercised by
`src/tunnelbook/experiment.py:_edge_condition`):

- `edge_params_for(shape)` → per-image blur / min-perimeter / dilation params.
- `canny_edges(rgb, low, high, ...)` → edge mask.
- `extract_edge_contours` / `rasterize_edges` → clean edge polylines / a rasterized edge mask.
- `split_superpixels_at_edges(label_map, edge_mask, dilate, min_size)` → a **new label map** whose
  superpixels are cut where the edges cross them.

`_edge_condition(rgb, lab, params)` in `experiment.py` already chains these end to end and returns a
refined label map. The rough server-side plan: run it (behind a request flag / new params) in
`create_session` before `build_instance`, expose the Canny edges to the UI for preview, and let the
new finer boundaries flow through the existing `_edges_payload` → mark → solve path unchanged.

## Build / typecheck

```bash
cd generator
npm run build        # tsc -b && vite build
```
