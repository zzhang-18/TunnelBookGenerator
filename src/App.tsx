import {
  useState,
  useRef,
  useEffect,
  useMemo,
  type ChangeEvent,
  type MouseEvent as ReactMouseEvent,
} from "react";
import "./App.css";

const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? "";
const apiUrl = (path: string) => `${API_BASE}${path}`;

// Inferno-ish heatmap for the crease map: t in [0,1], dim/cool = smooth, bright/hot = strong edge.
function heatColor(t: number): string {
  const c = Math.max(0, Math.min(1, isFinite(t) ? t : 0));
  const stops: [number, [number, number, number]][] = [
    [0.0, [12, 14, 40]],
    [0.35, [84, 24, 120]],
    [0.6, [201, 44, 92]],
    [0.8, [246, 130, 32]],
    [1.0, [255, 240, 130]],
  ];
  let a = stops[0];
  let b = stops[stops.length - 1];
  for (let k = 0; k < stops.length - 1; k++) {
    if (c >= stops[k][0] && c <= stops[k + 1][0]) {
      a = stops[k];
      b = stops[k + 1];
      break;
    }
  }
  const f = (c - a[0]) / (b[0] - a[0] || 1);
  const mix = (u: number, v: number) => Math.round(u + (v - u) * f);
  return `rgb(${mix(a[1][0], b[1][0])},${mix(a[1][1], b[1][1])},${mix(a[1][2], b[1][2])})`;
}

type ExportMode = "outline" | "engraving";
type Screen = "home" | "edges" | "output";
// Fraction of edges "auto select edges" marks, ranked by depth-edge score (see EdgeItem.score).
// Scores are min-max normalized per image (1.0 = that image's single strongest depth
// discontinuity), so an absolute cutoff is unreliable: one dominant edge can leave every other
// real boundary far below it. A percentile-of-that-image cutoff self-calibrates instead.
const AUTO_SELECT_TOP_FRACTION = 0.2;
type MarkType = "split_soft" | "split_hard" | "delete";
// "brush"/"erase" are canvas-drag modes, not mark types: brush paints split_soft over every
// boundary the stroke passes near, erase clears whatever mark (if any) is there. Distinguished
// from MarkType so `tool` can drive both the click-to-toggle tools and the two drag tools.
type ToolMode = MarkType | "brush" | "erase";
type ConnMethod = "flow" | "lazy";
type CutScore = "laplacian" | "meandiff";
type EdgeItem = {
  index: number; i: number; j: number; segments: number[][]; score?: number; cannyAlign?: number;
};
type EdgeConfigItem = {
  name: string;
  created: string | null;
  n_marks: number;
  counts: Record<MarkType, number>;
};
type SessionData = {
  sessionId: string;
  width: number;
  height: number;
  nLayers: number;
  nRegions: number;
  edges: EdgeItem[];
  baselineOverlay: string | null;
  regionMap?: string | null;
  regionDepth?: number[];
  datasetSlug?: string;
  edgeConfigs?: EdgeConfigItem[];
};
type SolveResult = {
  overlay: string;
  layers?: string[];        // per-layer semantic PNGs (front = layer_01)
  masks?: string[];         // per-layer material textures (photo style) for the 3D stack
  sheetMasks?: string[];    // same planes, .ai-export fabrication look (paper/engrave/cut)
  nLayers: number;
  status: string;
  runtime: number;
  objective: Record<string, number>;
  z?: number[] | null;
  layerOf?: number[];
};

// ─── Icons ────────────────────────────────────────────────────────────────────

const I = {
  Upload: ({ size = 15 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="16 16 12 12 8 16" /><line x1="12" y1="12" x2="12" y2="21" />
      <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
    </svg>
  ),
  ChevronLeft: ({ size = 14 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="15 18 9 12 15 6" />
    </svg>
  ),
  ChevronRight: ({ size = 14 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9 18 15 12 9 6" />
    </svg>
  ),
  ChevronDown: ({ size = 12 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="6 9 12 15 18 9" />
    </svg>
  ),
  ChevronUp: ({ size = 12 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="18 15 12 9 6 15" />
    </svg>
  ),
  Check: ({ size = 11 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  ),
  CheckCircle: ({ size = 17 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" /><polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  ),
  ImagePlus: ({ size = 38 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <polyline points="21 15 16 10 5 21" />
      <line x1="16" y1="5" x2="22" y2="5" /><line x1="19" y1="2" x2="19" y2="8" />
    </svg>
  ),
  Brush: ({ size = 13 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18.37 2.63 14 7l-1.59-1.59a2 2 0 0 0-2.82 0L8 7l9 9 1.59-1.59a2 2 0 0 0 0-2.82L17 10l4.37-4.37a2.12 2.12 0 1 0-3-3Z" />
      <path d="M9 8c-2 3-4 3.5-7 4l8 8c1-.5 3.5-2 4-7" />
    </svg>
  ),
  Download: ({ size = 14 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  ),
  DownloadCloud: ({ size = 16 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="8 17 12 21 16 17" /><line x1="12" y1="12" x2="12" y2="21" />
      <path d="M20.88 18.09A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.29" />
    </svg>
  ),
  Sparkles: ({ size = 16 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z" />
    </svg>
  ),
  Layers: ({ size = 14 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="12 2 2 7 12 12 22 7 12 2" />
      <polyline points="2 17 12 22 22 17" /><polyline points="2 12 12 17 22 12" />
    </svg>
  ),
  ArrowRight: ({ size = 15 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" />
    </svg>
  ),
  BookOpen: ({ size = 14 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
    </svg>
  ),
  Settings: ({ size = 13 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  ),
  X: ({ size = 12 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  ),
  RotateCcw: ({ size = 12 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 .49-4.15" />
    </svg>
  ),
  Scissors: ({ size = 12 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="6" cy="6" r="3" /><circle cx="6" cy="18" r="3" />
      <line x1="20" y1="4" x2="8.12" y2="15.88" />
      <line x1="14.47" y1="14.48" x2="20" y2="20" />
      <line x1="8.12" y1="8.12" x2="12" y2="12" />
    </svg>
  ),
  Eye: ({ size = 13 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" /><circle cx="12" cy="12" r="3" />
    </svg>
  ),
  SquareStack: ({ size = 18 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 10c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h4c1.1 0 2 .9 2 2" />
      <path d="M10 16c-1.1 0-2-.9-2-2v-4c0-1.1.9-2 2-2h4c1.1 0 2 .9 2 2" />
      <rect x="14" y="14" width="8" height="8" rx="2" />
    </svg>
  ),
  Home: ({ size = 13 }: { size?: number }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  ),
};

// ─── API functions ─────────────────────────────────────────────────────────────

async function createSession(
  imageFile: File,
  nLayers: number,
  edgeCondition: boolean,
): Promise<SessionData> {
  const fd = new FormData();
  fd.append("image", imageFile);
  fd.append("n_layers", String(nLayers));
  fd.append("edge_condition", String(edgeCondition));
  const res = await fetch(apiUrl("/api/sessions"), { method: "POST", body: fd });
  if (!res.ok)
    throw new Error(
      (await res.text().catch(() => "")) || `Failed to start session (${res.status})`,
    );
  return res.json();
}

async function solveSession(
  sessionId: string,
  markings: { index: number; type: MarkType }[],
  nLayers: number,
  lambdaSplit: number,
  objective: "depth" | "cut",
  cutLogSigma: number,
  lambdaDepth: number,
  connectivity: boolean,
  yMonotone: boolean,
  connectivityMethod: ConnMethod,
  norelTime: number,
  mipFocus: number,
  lambdaCoherence: number,
  minLayerArea: number,
  cutScore: CutScore,
): Promise<SolveResult> {
  const res = await fetch(apiUrl(`/api/sessions/${sessionId}/solve`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      markings, n_layers: nLayers, lambda_split: lambdaSplit, objective,
      cut_log_sigma: cutLogSigma, lambda_depth: lambdaDepth,
      connectivity, y_monotone: yMonotone,
      connectivity_method: connectivityMethod, norel_time: norelTime, mip_focus: mipFocus,
      lambda_coherence: lambdaCoherence, min_layer_area: minLayerArea,
      cut_score: cutScore,
    }),
  });
  if (!res.ok)
    throw new Error(
      (await res.text().catch(() => "")) || `Solve failed (${res.status})`,
    );
  return res.json();
}

async function fetchScores(
  sessionId: string,
  cutLogSigma: number,
  cutScore: CutScore,
): Promise<number[]> {
  const res = await fetch(apiUrl(`/api/sessions/${sessionId}/scores`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cut_log_sigma: cutLogSigma, cut_score: cutScore }),
  });
  if (!res.ok) throw new Error(`scores failed (${res.status})`);
  return (await res.json()).scores as number[];
}

async function selectObject(
  sessionId: string,
  x: number,
  y: number,
): Promise<{ edgeIndices: number[]; score: number }> {
  const res = await fetch(apiUrl(`/api/sessions/${sessionId}/select-object`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ x, y }),
  });
  if (!res.ok)
    throw new Error((await res.text().catch(() => "")) || `Select object failed (${res.status})`);
  return res.json();
}

async function saveEdgeConfig(
  sessionId: string,
  name: string,
  markings: { index: number; type: MarkType }[],
  nLayers: number | null = null,
): Promise<{ configs: EdgeConfigItem[] }> {
  const res = await fetch(apiUrl(`/api/sessions/${sessionId}/edge-configs`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, markings, n_layers: nLayers }),
  });
  if (!res.ok)
    throw new Error(
      (await res.text().catch(() => "")) || `Config save failed (${res.status})`,
    );
  return res.json();
}

async function loadEdgeConfig(
  sessionId: string,
  name: string,
): Promise<{ markings: { index: number; type: MarkType }[] }> {
  const res = await fetch(
    apiUrl(`/api/sessions/${sessionId}/edge-configs/${encodeURIComponent(name)}`),
  );
  if (!res.ok)
    throw new Error(
      (await res.text().catch(() => "")) || `Config load failed (${res.status})`,
    );
  return res.json();
}

async function deleteSession(sessionId: string) {
  await fetch(apiUrl(`/api/sessions/${sessionId}`), { method: "DELETE" }).catch(() => {});
}

async function exportStand(
  sessionId: string,
  nLayers: number,
  spokeH = 1.4,
  baseH = 0.65,
) {
  const res = await fetch(apiUrl(`/api/sessions/${sessionId}/export-stand`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ n_layers: nLayers, spoke_h_in: spokeH, base_h_in: baseH }),
  });
  if (!res.ok)
    throw new Error(
      (await res.text().catch(() => "")) || `Stand export failed (${res.status})`,
    );
  return res.blob();
}

async function exportLayers(
  sessionId: string,
  markings: { index: number; type: MarkType }[],
  nLayers: number,
  objective: "depth" | "cut",
  lambdaDepth: number,
  connectivity: boolean,
  yMonotone: boolean,
  connectivityMethod: ConnMethod,
  norelTime: number,
  mipFocus: number,
  lambdaCoherence: number,
  minLayerArea: number,
  cutScore: CutScore,
  mode: ExportMode,
  contentWidthIn: number,
  borderIn: number,
  engraveLayers: boolean[],
  minEngraveInPerLayer: number[],
): Promise<Blob> {
  const res = await fetch(apiUrl(`/api/sessions/${sessionId}/export-ai`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      markings, n_layers: nLayers, objective, lambda_depth: lambdaDepth,
      connectivity, y_monotone: yMonotone,
      connectivity_method: connectivityMethod, norel_time: norelTime, mip_focus: mipFocus,
      lambda_coherence: lambdaCoherence, min_layer_area: minLayerArea,
      cut_score: cutScore,
      mode, content_width_in: contentWidthIn, border_in: borderIn,
      // per-layer engrave overrides (omit when engraving is off / arrays not sized yet)
      engrave_layers: mode === "engraving" && engraveLayers.length === nLayers
        ? engraveLayers : null,
      min_engrave_in_per_layer: mode === "engraving" && minEngraveInPerLayer.length === nLayers
        ? minEngraveInPerLayer : null,
    }),
  });
  if (!res.ok)
    throw new Error(
      (await res.text().catch(() => "")) || `Layer export failed (${res.status})`,
    );
  return res.blob();
}

// ─── Shared UI primitives ─────────────────────────────────────────────────────

function HelpModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>×</button>
        <div className="modal-title"><I.Sparkles /> How It Works</div>
        <div className="modal-step">
          <div className="modal-step-label"><I.Upload /> step 1 — Upload</div>
          <p>Drop or click to upload a photo. Enter how many layers (1–10) and press <strong>run</strong>. Edge detection runs automatically.</p>
        </div>
        <hr className="modal-divider" />
        <div className="modal-step">
          <div className="modal-step-label"><I.Brush /> step 2 — Select Edges</div>
          <p>
            All detected boundaries are shown as white lines. Click any edge to mark it as a <strong>cut line</strong> (turns red). Click again to deselect.
            Use <strong>select all</strong> / <strong>clear all</strong> for bulk actions.
          </p>
        </div>
        <hr className="modal-divider" />
        <div className="modal-step">
          <div className="modal-step-label"><I.CheckCircle /> step 3 — Confirm</div>
          <p>Press <strong>confirm cuts</strong> to store your selection. The backend will use these edges together with depth estimation to assign pixels to layers.</p>
        </div>
        <hr className="modal-divider" />
        <div className="modal-step">
          <div className="modal-step-label"><I.Download /> step 4 — Export</div>
          <p><strong>Export Stand</strong> gives an Adobe Illustrator file for the laser-cut tunnel book stand.</p>
        </div>
      </div>
    </div>
  );
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────

function Sidebar({
  screen,
  onGoHome,
}: {
  screen: Screen;
  onGoHome: () => void;
}) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <I.SquareStack />
        <span>tunnel<span className="sidebar-logo-accent">book</span></span>
      </div>
      <nav className="sidebar-nav">
        <div
          className={`sidebar-tab ${screen === "home" ? "sidebar-tab--active" : "sidebar-tab--idle"}`}
          onClick={onGoHome}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && onGoHome()}
        >
          <I.Home />
          <span className="sidebar-tab-label">Upload</span>
          {screen !== "home" && <div className="sidebar-tab-dot" style={{ background: "var(--text-lo)" }} />}
        </div>

        {screen !== "home" && (
          <div
            className={`sidebar-tab ${screen === "edges" ? "sidebar-tab--active" : ""} ${screen === "output" ? "sidebar-tab--done" : ""}`}
          >
            <div
              className="sidebar-tab-dot"
              style={{ background: screen === "output" ? "var(--ink)" : "var(--text-lo)" }}
            />
            <span className="sidebar-tab-label">Edges</span>
            {screen === "output" && (
              <span className="sidebar-tab-check"><I.Check /></span>
            )}
            {screen === "edges" && <div className="sidebar-tab-pulse" />}
          </div>
        )}

        {screen === "output" && (
          <div className="sidebar-tab sidebar-tab--active sidebar-tab--output">
            <I.CheckCircle size={13} />
            <span className="sidebar-tab-label">Output</span>
          </div>
        )}
      </nav>
      <div className="sidebar-footer">
        <span className="sidebar-status">
          {screen === "home" ? "Ready" : screen === "edges" ? "Selecting" : "Complete"}
        </span>
      </div>
    </aside>
  );
}

// ─── Home Screen ──────────────────────────────────────────────────────────────

function HomeScreen({
  onGo,
  isStarting,
  error,
  exportMode,
  onExportModeChange,
}: {
  onGo: (
    f: File, url: string, n: number,
    frameWidthIn: number, frameHeightIn: number, frameBorderIn: number,
    edgeCondition: boolean,
  ) => void;
  isStarting: boolean;
  error: string | null;
  exportMode: ExportMode;
  onExportModeChange: (m: ExportMode) => void;
}) {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [layerCount, setLayerCount] = useState("");
  const [frameWidthIn, setFrameWidthIn] = useState("12");
  const [frameHeightIn, setFrameHeightIn] = useState("9");
  const [frameBorderIn, setFrameBorderIn] = useState("0.5");
  const [edgeCondition, setEdgeCondition] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const parsed = parseInt(layerCount, 10);
  const parsedW = parseFloat(frameWidthIn);
  const parsedH = parseFloat(frameHeightIn);
  const parsedB = parseFloat(frameBorderIn);
  const frameValid =
    !isNaN(parsedW) && parsedW >= 1 && parsedW <= 30 &&
    !isNaN(parsedH) && parsedH >= 1 && parsedH <= 30 &&
    !isNaN(parsedB) && parsedB >= 0 && parsedB <= 4;
  const canGo = !!imageUrl && parsed >= 1 && parsed <= 10 && !isStarting && frameValid;

  const clampNum = (v: string, _min: number, max: number) => {
    const n = parseFloat(v);
    if (isNaN(n)) return v;
    if (n > max) return String(max);
    return v;
  };

  const applyFile = (f: File) => {
    setImageFile(f);
    setImageUrl(URL.createObjectURL(f));
  };
  const handleFile = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) applyFile(f);
  };
  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = (e: React.DragEvent) => {
    if (!e.currentTarget.contains(e.relatedTarget as Node)) setIsDragging(false);
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith("image/")) applyFile(f);
  };

  const outerW = frameValid ? (parsedW + 2 * parsedB).toFixed(2) : "—";
  const outerH = frameValid ? (parsedH + 2 * parsedB).toFixed(2) : "—";

  return (
    <div className="home-screen">
      <div className="home-hero">
        <div className="home-tag">Photo to laser-cut tunnel book</div>
        <h1 className="home-title">
          Tunnel<span className="home-title-accent">Book</span>
          <span className="home-title-small"> Generator</span>
        </h1>
        <div className="home-steps">
          <span className="home-step"><I.Upload size={11} /> upload image</span>
          <span className="home-step-arrow">→</span>
          <span className="home-step"><I.Brush size={11} /> select edges</span>
          <span className="home-step-arrow">→</span>
          <span className="home-step"><I.BookOpen size={11} /> assign layers</span>
          <span className="home-step-arrow">→</span>
          <span className="home-step"><I.Download size={11} /> export cut files</span>
        </div>
      </div>

      <div
        className={`drop-zone ${isDragging ? "drop-zone--dragging" : ""} ${imageUrl ? "drop-zone--filled" : ""}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => !imageUrl && fileRef.current?.click()}
      >
        <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={handleFile} />
        {imageUrl ? (
          <div className="drop-zone-preview">
            <img src={imageUrl} alt="Uploaded" />
            <div className="drop-zone-overlay">
              <button
                className="drop-zone-change"
                onClick={(e) => { e.stopPropagation(); fileRef.current?.click(); }}
              >
                <I.Upload /> change image
              </button>
            </div>
            <div className="drop-zone-filename">{imageFile?.name}</div>
          </div>
        ) : (
          <div className="drop-zone-empty">
            <div className="drop-zone-icon"><I.ImagePlus /></div>
            <div className="drop-zone-text">
              {isDragging ? "drop it here!" : "drag & drop or click to upload"}
            </div>
            <div className="drop-zone-sub">PNG · JPG · WEBP</div>
          </div>
        )}
      </div>

      {/* ── Row 1: layers / default mode / run ── */}
      <div className="config-row">
        <div className="config-group">
          <label className="config-label"><I.Layers /> layers</label>
          <div className="config-input-wrap">
            <input
              className="config-input"
              type="text"
              inputMode="numeric"
              placeholder="?"
              value={layerCount}
              onChange={(e) => {
                const v = e.target.value.replace(/[^0-9]/g, "");
                setLayerCount(!v || parseInt(v) <= 10 ? v : "10");
              }}
              maxLength={2}
            />
            <span className="config-max">/ 10</span>
          </div>
        </div>

        <div className="config-group">
          <label className="config-label"><I.Settings /> default mode</label>
          <div className="mode-toggle">
            <button
              className={`mode-toggle-btn ${exportMode === "outline" ? "mode-toggle-btn--active" : ""}`}
              onClick={() => onExportModeChange("outline")}
            >
              <I.Scissors /> outline
            </button>
            <button
              className={`mode-toggle-btn ${exportMode === "engraving" ? "mode-toggle-btn--active" : ""}`}
              onClick={() => onExportModeChange("engraving")}
            >
              <I.Eye /> engrave
            </button>
          </div>
        </div>

        <button
          className={`go-btn ${canGo ? "go-btn--active" : "go-btn--disabled"}`}
          onClick={() => {
            if (canGo && imageFile && imageUrl)
              onGo(imageFile, imageUrl, parsed, parsedW, parsedH, parsedB, edgeCondition);
          }}
          disabled={!canGo}
        >
          {isStarting ? (
            <><span className="go-btn-spinner" /> detecting edges & preparing SAM…</>
          ) : (
            <>run <I.ArrowRight /></>
          )}
        </button>
      </div>

      {/* ── Row 1b: edge conditioning ── */}
      <div className="config-row" style={{ flexDirection: "column", alignItems: "flex-start", gap: 4 }}>
        <label
          className="config-label"
          style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}
        >
          <input
            type="checkbox"
            checked={edgeCondition}
            onChange={(e) => setEdgeCondition(e.target.checked)}
          />
          <I.Brush size={12} /> condition boundaries on detected edges
        </label>
        <span style={{ fontSize: 11, color: "var(--text-dim)", marginLeft: 22 }}>
          splits superpixels along real Canny object edges, not just the segmentation tessellation
        </span>
      </div>

      {/* ── Row 2: frame dimensions ── */}
      <div className="config-row config-row--frame">
        <div className="frame-row-label"><I.Scissors size={11} /> frame dimensions</div>
        <div className="config-group">
          <label className="config-label">inner width</label>
          <div className="config-input-wrap config-input-wrap--sm">
            <input
              className="config-input config-input--sm"
              type="text"
              inputMode="decimal"
              placeholder="12"
              value={frameWidthIn}
              onChange={(e) => {
                const v = e.target.value.replace(/[^0-9.]/g, "");
                setFrameWidthIn(clampNum(v, 1, 30));
              }}
            />
            <span className="config-max">/ 30 in</span>
          </div>
        </div>
        <div className="config-group">
          <label className="config-label">inner height</label>
          <div className="config-input-wrap config-input-wrap--sm">
            <input
              className="config-input config-input--sm"
              type="text"
              inputMode="decimal"
              placeholder="9"
              value={frameHeightIn}
              onChange={(e) => {
                const v = e.target.value.replace(/[^0-9.]/g, "");
                setFrameHeightIn(clampNum(v, 1, 30));
              }}
            />
            <span className="config-max">/ 30 in</span>
          </div>
        </div>
        <div className="config-group">
          <label className="config-label">border gap</label>
          <div className="config-input-wrap config-input-wrap--sm">
            <input
              className="config-input config-input--sm"
              type="text"
              inputMode="decimal"
              placeholder="0.5"
              value={frameBorderIn}
              onChange={(e) => {
                const v = e.target.value.replace(/[^0-9.]/g, "");
                setFrameBorderIn(clampNum(v, 0, 4));
              }}
            />
            <span className="config-max">/ 4 in</span>
          </div>
        </div>
        <div className={`frame-summary ${!frameValid ? "frame-summary--warn" : ""}`}>
          {frameValid ? (
            <>
              <span className="frame-summary-item">
                <span className="frame-summary-key">inner</span>{parsedW}" × {parsedH}"
              </span>
              <span className="frame-summary-sep">·</span>
              <span className="frame-summary-item">
                <span className="frame-summary-key">border</span>{parsedB}"
              </span>
              <span className="frame-summary-sep">·</span>
              <span className="frame-summary-item">
                <span className="frame-summary-key">outer</span>{outerW}" × {outerH}"
              </span>
              <span className="frame-summary-sep">·</span>
              <span className="frame-summary-note">identical across all layers</span>
            </>
          ) : (
            <span className="frame-summary-warn-text">⚠ enter valid dimensions to continue</span>
          )}
        </div>
      </div>

      {error && (
        <div className="error-banner">
          <span className="error-banner-tag">Error</span> {error}
          <div className="error-banner-sub">
            Ensure the Python server is running and Vite proxies <code>/api</code>.
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Book Visualizer ──────────────────────────────────────────────────────────
// Interactive CSS-3D preview of the solved layer stack (reintroduced from 0e47c58): each
// layer is a translateZ-offset plane, fed the solver's per-layer transparent silhouettes
// (SolveResult.masks) so the sheets show through in depth like a real tunnel book.

function BookVisualizer({ masks, sheetMasks }: { masks: string[]; sheetMasks?: string[] }) {
  const [rotX, setRotX] = useState(16);
  const [rotY, setRotY] = useState(-26);
  const [scale, setScale] = useState(1);
  const [dragging, setDragging] = useState(false);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
  // texture source: photo pixels vs the .ai-style fabrication sheet (paper/engrave/cut)
  const [tex, setTex] = useState<"photo" | "sheet">("photo");
  const hasSheet = !!sheetMasks && sheetMasks.length === masks.length;
  const planes = tex === "sheet" && hasSheet ? sheetMasks! : masks;
  const drag = useRef<{ x: number; y: number; rx: number; ry: number } | null>(null);
  const sceneRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!dragging) return;
    const move = (e: globalThis.MouseEvent) => {
      if (!drag.current) return;
      setRotY(drag.current.ry + (e.clientX - drag.current.x) * 0.5);
      setRotX(drag.current.rx - (e.clientY - drag.current.y) * 0.3);
    };
    const up = () => setDragging(false);
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
  }, [dragging]);

  useEffect(() => {
    const el = sceneRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      setScale((s) => Math.min(3, Math.max(0.35, s - e.deltaY * 0.0012)));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const onDown = (e: ReactMouseEvent) => {
    drag.current = { x: e.clientX, y: e.clientY, rx: rotX, ry: rotY };
    setDragging(true);
    e.preventDefault();
  };
  const toggle = (i: number) =>
    setHidden((prev) => {
      const s = new Set(prev);
      s.has(i) ? s.delete(i) : s.add(i);
      return s;
    });

  const N = masks.length;
  const GAP = 28; // px of depth between adjacent sheets
  return (
    <div className="viz-book-wrap">
      <div
        ref={sceneRef}
        className="viz-scene"
        onMouseDown={onDown}
        style={{ cursor: dragging ? "grabbing" : "grab" }}
      >
        <div
          className="viz-book"
          style={{
            transform: `translateZ(-${(N * GAP) / 2}px) scale(${scale}) rotateX(${rotX}deg) rotateY(${rotY}deg)`,
          }}
        >
          {planes.map((src, i) =>
            hidden.has(i) ? null : (
              <img
                key={`${tex}-${i}`}
                src={src}
                alt={`layer ${i + 1}`}
                draggable={false}
                style={{
                  position: "absolute",
                  inset: 0,
                  width: "100%",
                  height: "100%",
                  objectFit: "contain",
                  pointerEvents: "none",
                  transform: `translateZ(${(N - 1 - i) * GAP}px)`,
                  zIndex: N - i,
                  filter: "drop-shadow(0 3px 10px rgba(0,0,0,0.25))",
                }}
              />
            ),
          )}
        </div>
        <div className="viz-scene-hint">scroll to zoom · drag to rotate</div>
      </div>

      <div className="viz-rotate-row">
        <button className="viz-ctrl-btn" onClick={() => setRotY((r) => r - 30)} title="Rotate left">
          <I.ChevronLeft />
        </button>
        <div className="viz-zoom-controls">
          <button className="viz-ctrl-btn" onClick={() => setScale((s) => Math.max(0.35, s - 0.15))} title="Zoom out">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="5" y1="12" x2="19" y2="12" /></svg>
          </button>
          <span className="viz-ctrl-label">{Math.round(scale * 100)}%</span>
          <button className="viz-ctrl-btn" onClick={() => setScale((s) => Math.min(3, s + 0.15))} title="Zoom in">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
          </button>
        </div>
        <button className="viz-ctrl-btn" onClick={() => { setRotX(16); setRotY(-26); setScale(1); }} title="Reset view">
          <I.RotateCcw />
        </button>
        {hasSheet && (
          <div className="viz-zoom-controls" title="Plane texture: photo pixels vs the .ai fabrication sheet (paper + engrave + red cut lines); both include the support material">
            {(["photo", "sheet"] as const).map((t) => (
              <button
                key={t}
                className="viz-ctrl-btn"
                onClick={() => setTex(t)}
                style={{
                  width: "auto", padding: "0 8px", fontSize: 9,
                  fontFamily: "var(--font-mono)",
                  color: tex === t ? "var(--ink)" : undefined,
                  fontWeight: tex === t ? 700 : 400,
                }}
              >
                {t}
              </button>
            ))}
          </div>
        )}
        <span className="viz-ctrl-label-layers">{N - hidden.size}/{N} layers</span>
        <button className="viz-ctrl-btn" onClick={() => setRotY((r) => r + 30)} title="Rotate right">
          <I.ChevronRight />
        </button>
      </div>

      <div className="viz-layer-list">
        {masks.map((_, i) => {
          const on = !hidden.has(i);
          return (
            <button
              key={i}
              className={`viz-layer-row-header ${on ? "" : "viz-layer-row-header--locked"}`}
              onClick={() => toggle(i)}
              title={on ? "Hide layer" : "Show layer"}
              style={{ opacity: on ? 1 : 0.5 }}
            >
              <div className="viz-layer-swatch" style={{ background: heatColor(N > 1 ? i / (N - 1) : 0) }} />
              <span className="viz-layer-name">Layer {i + 1}</span>
              <I.Eye />
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ─── Edge Selection Screen ────────────────────────────────────────────────────

function EdgeSelectionScreen({
  sessionId,
  sessionWidth,
  sessionHeight,
  numLayers,
  edges,
  baselineOverlay,
  regionMap,
  regionDepth,
  edgeConfigs,
  onSubmit,
  onBack,
}: {
  sessionId: string;
  sessionWidth: number;
  sessionHeight: number;
  numLayers: number;
  edges: EdgeItem[];
  baselineOverlay: string | null;
  regionMap: string | null;
  regionDepth: number[];
  edgeConfigs: EdgeConfigItem[];
  onSubmit: (
    markings: { index: number; type: MarkType }[],
    objective: "depth" | "cut",
    markCount: number,
    lambdaDepth: number,
    connectivity: boolean,
    yMonotone: boolean,
    connectivityMethod: ConnMethod,
    norelTime: number,
    mipFocus: number,
    lambdaCoherence: number,
    minLayerArea: number,
    cutScore: CutScore,
  ) => void;
  onBack: () => void;
}) {
  const [marks, setMarks] = useState<Record<number, MarkType>>({});
  const [tool, setTool] = useState<ToolMode>("split_soft");
  // brush/erase drag state: brushSize is an image-pixel radius; brushCursor tracks the pointer
  // in image-pixel coords for the visual radius circle; isPainting is true between pointerdown
  // and pointerup so pointermove knows whether to keep painting
  const [brushSize, setBrushSize] = useState(20);
  const [isPainting, setIsPainting] = useState(false);
  const [brushCursor, setBrushCursor] = useState<{ x: number; y: number } | null>(null);
  const [objective, setObjective] = useState<"depth" | "cut">("depth");
  const [lambdaDepth, setLambdaDepth] = useState(0);
  const [connectivity, setConnectivity] = useState(true);
  const [yMonotone, setYMonotone] = useState(false);
  // solver knobs — defaults mirror the server's SolveRequest (norel_time=60, mip_focus=1, flow)
  const [lazyConn, setLazyConn] = useState(false);
  const [norelOn, setNorelOn] = useState(true);
  const [mipFocusOn, setMipFocusOn] = useState(true);
  // depth-plateau coherence tie-breaker — default OFF (mirrors SolveRequest); 0.05 is the
  // validated weight when toggled on
  const [cohOn, setCohOn] = useState(false);
  const [lambdaCoh, setLambdaCoh] = useState(0.05);
  // per-layer visible-area floor — mirrors SolveRequest.min_layer_area (cut objective)
  const [minLayerArea, setMinLayerArea] = useState(0.10);
  // crease/cut score: laplacian (LoG across the boundary) vs meandiff (region-mean depth gap)
  const [cutScore, setCutScore] = useState<CutScore>("laplacian");
  const [logSigma, setLogSigma] = useState(2.0);
  // per-pixel positional region index decoded from regionMap (idx+1 in R + G<<8; 0 = none)
  const regionIdxRef = useRef<{ data: Uint8ClampedArray; w: number; h: number } | null>(null);
  const [hoverRegion, setHoverRegion] = useState<{ idx: number; px: number; py: number; w: number; h: number } | null>(null);
  const [scoreOverride, setScoreOverride] = useState<number[] | null>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [overlay, setOverlay] = useState<string | null>(baselineOverlay);
  const [isSolving, setIsSolving] = useState(false);
  const [solveError, setSolveError] = useState<string | null>(null);
  const [result, setResult] = useState<SolveResult | null>(null);
  // named edge configs stored in the session's dataset dir
  const [configs, setConfigs] = useState<EdgeConfigItem[]>(edgeConfigs);
  const [configName, setConfigName] = useState("");
  const [selectedConfig, setSelectedConfig] = useState("");
  const [configBusy, setConfigBusy] = useState(false);
  const [configError, setConfigError] = useState<string | null>(null);
  // point-prompted SAM "select object": click the photo (not an edge) to mark its outline
  const [objectSelectMode, setObjectSelectMode] = useState(false);
  // "advanced" accordion: objective + connectivity + cut-tuning/solver knobs, collapsed by
  // default -- most sessions never need to leave depth-fit + default solver settings
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [isSelectingObject, setIsSelectingObject] = useState(false);
  const [selectObjectError, setSelectObjectError] = useState<string | null>(null);

  const TOOLS: { key: ToolMode; label: string; color: string }[] = [
    { key: "split_soft", label: "Soft split", color: "#f59e0b" },
    { key: "split_hard", label: "Hard split", color: "#ef4444" },
    { key: "delete", label: "Delete", color: "#3b82f6" },
    { key: "brush", label: "Brush", color: "#10b981" },
    { key: "erase", label: "Erase", color: "#9ca3af" },
  ];
  const colorOf = (m: MarkType) => TOOLS.find((t) => t.key === m)!.color;

  // click-to-toggle marking (Soft split / Hard split / Delete only -- brush/erase paint via drag)
  const applyMark = (i: number) => {
    if (tool !== "split_soft" && tool !== "split_hard" && tool !== "delete") return;
    setMarks((prev) => {
      const next = { ...prev };
      if (next[i] === tool) delete next[i];
      else next[i] = tool;
      return next;
    });
  };

  // Bounding box per edge (image-pixel space), memoized so brush painting can cheaply reject
  // edges nowhere near the pointer before checking individual segments.
  const edgeBBoxes = useMemo(() => edges.map((e) => {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const [x1, y1, x2, y2] of e.segments) {
      minX = Math.min(minX, x1, x2); maxX = Math.max(maxX, x1, x2);
      minY = Math.min(minY, y1, y2); maxY = Math.max(maxY, y1, y2);
    }
    return { minX, minY, maxX, maxY };
  }), [edges]);

  const edgeNearPoint = (arrIdx: number, x: number, y: number, radius: number) => {
    const b = edgeBBoxes[arrIdx];
    if (!b || x < b.minX - radius || x > b.maxX + radius || y < b.minY - radius || y > b.maxY + radius) {
      return false;
    }
    const r2 = radius * radius;
    for (const [x1, y1, x2, y2] of edges[arrIdx].segments) {
      const dx1 = x1 - x, dy1 = y1 - y;
      if (dx1 * dx1 + dy1 * dy1 <= r2) return true;
      const dx2 = x2 - x, dy2 = y2 - y;
      if (dx2 * dx2 + dy2 * dy2 <= r2) return true;
    }
    return false;
  };

  // brush: mark every boundary within brushSize of (x, y) as split_soft. erase: clear whatever
  // mark (if any) is there. Both are drag tools -- see the pointer handlers below.
  const paintAt = (x: number, y: number) => {
    if (tool !== "brush" && tool !== "erase") return;
    setMarks((prev) => {
      let changed = false;
      const next = { ...prev };
      edges.forEach((e, arrIdx) => {
        if (!edgeNearPoint(arrIdx, x, y, brushSize)) return;
        if (tool === "brush") {
          if (next[e.index] !== "split_soft") { next[e.index] = "split_soft"; changed = true; }
        } else if (next[e.index] !== undefined) {
          delete next[e.index];
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  };

  const svgImagePoint = (ev: { clientX: number; clientY: number; currentTarget: SVGSVGElement }) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    return {
      x: ((ev.clientX - rect.left) / rect.width) * sessionWidth,
      y: ((ev.clientY - rect.top) / rect.height) * sessionHeight,
    };
  };

  const handleBrushPointerDown = (ev: React.PointerEvent<SVGSVGElement>) => {
    if (objectSelectMode || (tool !== "brush" && tool !== "erase")) return;
    ev.currentTarget.setPointerCapture(ev.pointerId);
    setIsPainting(true);
    const { x, y } = svgImagePoint(ev);
    setBrushCursor({ x, y });
    paintAt(x, y);
  };

  const handleBrushPointerMove = (ev: React.PointerEvent<SVGSVGElement>) => {
    if (objectSelectMode || (tool !== "brush" && tool !== "erase")) return;
    const { x, y } = svgImagePoint(ev);
    setBrushCursor({ x, y });
    if (isPainting) paintAt(x, y);
  };

  const handleBrushPointerUp = () => setIsPainting(false);
  const handleBrushPointerLeave = () => { setIsPainting(false); setBrushCursor(null); };

  const markingsArray = () =>
    Object.entries(marks).map(([index, type]) => ({ index: Number(index), type }));

  const counts: Record<MarkType, number> = { split_soft: 0, split_hard: 0, delete: 0 };
  Object.values(marks).forEach((m) => (counts[m] += 1));
  const markCount = Object.keys(marks).length;

  const edgePath = (segs: number[][]) =>
    segs.map(([x1, y1, x2, y2]) => `M${x1} ${y1}L${x2} ${y2}`).join("");

  // crease strength per edge: live-recomputed override (slider) or the session baseline
  const scoreOf = (e: EdgeItem) =>
    scoreOverride ? scoreOverride[e.index] ?? 0 : e.score ?? 0;

  // Boundary "realness" = whichever is stronger: the depth-edge score, or how much the boundary
  // coincides with a detected Canny edge (cannyAlign, independent of depth -- see server.py). A
  // silhouette can be visually sharp but sit on a smooth/noisy stretch of the depth estimate, so
  // depth score alone misses it; taking the max lets either signal carry a real edge.
  const boundaryStrength = (e: EdgeItem) => Math.max(scoreOf(e), e.cannyAlign ?? 0);

  // Ranks edges by strength * sqrt(boundary length) rather than strength alone, then marks the
  // top AUTO_SELECT_TOP_FRACTION as split_soft -- a suggestion, not a forced cut, so the solver
  // still decides. Both signals are medians/fractions over the boundary's pixels, so a short,
  // spiky (often noisy) sliver can outscore a long, genuinely significant boundary (e.g. a
  // mountain ridge) purely because there's less to average over; weighting by length keeps long
  // real boundaries competitive. Rank-based (not an absolute cutoff) so it adapts to each image's
  // own distribution. Existing marks are left untouched: this only fills in unmarked edges.
  const weightOf = (e: EdgeItem) => boundaryStrength(e) * Math.sqrt(Math.max(1, e.segments.length));

  const handleAutoSelect = () => {
    if (edges.length === 0) return;
    const k = Math.max(1, Math.ceil(edges.length * AUTO_SELECT_TOP_FRACTION));
    const top = new Set(
      [...edges].sort((a, b) => weightOf(b) - weightOf(a)).slice(0, k).map((e) => e.index),
    );
    setMarks((prev) => {
      const next = { ...prev };
      edges.forEach((e) => {
        if (next[e.index] === undefined && top.has(e.index)) {
          next[e.index] = "split_soft";
        }
      });
      return next;
    });
  };

  // "select object" (point-prompted SAM): active only when objectSelectMode is on, and only for
  // clicks that reach the svg background (edge <g>s stopPropagation on their own clicks). Maps
  // the click to image-pixel coords via the svg's rendered size vs. its viewBox, matching the
  // convention every edge segment is already in.
  const handleObjectClick = async (ev: React.MouseEvent<SVGSVGElement>) => {
    if (!objectSelectMode || isSelectingObject) return;
    const rect = ev.currentTarget.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * sessionWidth;
    const y = ((ev.clientY - rect.top) / rect.height) * sessionHeight;
    setIsSelectingObject(true);
    setSelectObjectError(null);
    try {
      const { edgeIndices } = await selectObject(sessionId, x, y);
      setMarks((prev) => {
        const next = { ...prev };
        edgeIndices.forEach((i) => {
          if (next[i] === undefined) next[i] = "split_soft";
        });
        return next;
      });
    } catch (err: any) {
      setSelectObjectError(err?.message ?? "Select object failed");
    } finally {
      setIsSelectingObject(false);
    }
  };

  // value the crease map colours by = "cuttability" (low cut cost). A soft mark sets the cut
  // cost to 0, so it reads as maximally cuttable (brightest); depth creases fall out naturally.
  const creaseCuttability = (e: EdgeItem) =>
    marks[e.index] === "split_soft" ? 1 : scoreOf(e);

  // debounced live recompute of the crease scores when the LoG scale slider moves
  const didMountSigma = useRef(false);
  useEffect(() => {
    if (!didMountSigma.current) {
      didMountSigma.current = true;
      return; // initial render uses the baseline scores already in `edges`
    }
    const t = setTimeout(() => {
      fetchScores(sessionId, logSigma, cutScore)
        .then(setScoreOverride)
        .catch(() => {});
    }, 150);
    return () => clearTimeout(t);
  }, [logSigma, cutScore, sessionId]);

  // decode the region-index map once per session for hover hit-testing
  useEffect(() => {
    regionIdxRef.current = null;
    if (!regionMap) return;
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = img.width;
      c.height = img.height;
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      const { data } = ctx.getImageData(0, 0, img.width, img.height);
      regionIdxRef.current = { data, w: img.width, h: img.height };
    };
    img.src = regionMap;
  }, [regionMap]);

  const handleRegionHover = (ev: ReactMouseEvent<HTMLDivElement>) => {
    const m = regionIdxRef.current;
    if (!m) return;
    const rect = ev.currentTarget.getBoundingClientRect();
    const relX = (ev.clientX - rect.left) / rect.width;
    const relY = (ev.clientY - rect.top) / rect.height;
    const ix = Math.min(m.w - 1, Math.max(0, Math.floor(relX * m.w)));
    const iy = Math.min(m.h - 1, Math.max(0, Math.floor(relY * m.h)));
    const o = (iy * m.w + ix) * 4;
    const idx = m.data[o] + (m.data[o + 1] << 8) - 1;
    if (idx < 0) {
      setHoverRegion(null);
      return;
    }
    setHoverRegion({
      idx, px: ev.clientX - rect.left, py: ev.clientY - rect.top,
      w: rect.width, h: rect.height,
    });
  };

  const strokeFor = (i: number, m: MarkType | undefined) => {
    const hov = hoveredIdx === i;
    if (!m) return hov ? "rgba(255,255,255,0.85)" : "rgba(255,255,255,0.22)";
    return colorOf(m);
  };

  // wire values for the solver-knob checkboxes (server treats 0 as "off")
  const connMethod: ConnMethod = lazyConn ? "lazy" : "flow";
  const norelTime = norelOn ? 60 : 0;
  const mipFocus = mipFocusOn ? 1 : 0;
  const lambdaCoherence = cohOn ? lambdaCoh : 0;

  const handleSolve = async () => {
    setIsSolving(true);
    setSolveError(null);
    try {
      const r = await solveSession(sessionId, markingsArray(), numLayers, 1.0, objective, logSigma, lambdaDepth, connectivity, yMonotone, connMethod, norelTime, mipFocus, lambdaCoherence, minLayerArea, cutScore);
      setOverlay(r.overlay);
      setResult(r);
    } catch (err: any) {
      setSolveError(err?.message ?? "Solve failed");
    } finally {
      setIsSolving(false);
    }
  };

  const handleConfigSave = async () => {
    setConfigBusy(true);
    setConfigError(null);
    try {
      const r = await saveEdgeConfig(sessionId, configName, markingsArray(), numLayers);
      setConfigs(r.configs);
      setSelectedConfig(configName);
    } catch (err: any) {
      setConfigError(err?.message ?? "Config save failed");
    } finally {
      setConfigBusy(false);
    }
  };

  const handleConfigLoad = async () => {
    setConfigBusy(true);
    setConfigError(null);
    try {
      const r = await loadEdgeConfig(sessionId, selectedConfig);
      setMarks(Object.fromEntries(r.markings.map((m) => [m.index, m.type])));
    } catch (err: any) {
      setConfigError(err?.message ?? "Config load failed");
    } finally {
      setConfigBusy(false);
    }
  };

  return (
    <div className="layer-screen">
      <div className="layer-topbar">
        <button className="back-btn" onClick={onBack}>
          <I.ChevronLeft /> back
        </button>
        <div className="layer-topbar-center">
          <div className="layer-badge" style={{ background: "var(--ink)" }}>
            Mark boundaries
          </div>
          <span className="layer-of">{numLayers} layers · {edges.length} boundaries</span>
        </div>
        <div className="layer-progress-wrap" style={{ alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 11, color: "var(--text-dim)" }}>{markCount} marked</span>
        </div>
      </div>

      <div className="canvas-card">
        <div className="canvas-tools canvas-tools--stacked">
          <div className="canvas-tools-row">
            <span className="canvas-tools-hint" style={{ marginRight: 4 }}>tool:</span>
            {TOOLS.filter((t) => t.key !== "brush" && t.key !== "erase").map((t) => (
              <button
                key={t.key}
                className="ctrl-btn"
                onClick={() => { setTool(t.key); setObjectSelectMode(false); }}
                style={{
                  borderColor: tool === t.key ? t.color : "transparent",
                  color: tool === t.key ? t.color : "var(--text-dim)",
                  fontWeight: tool === t.key ? 700 : 400,
                }}
              >
                <span style={{
                  display: "inline-block", width: 9, height: 9, borderRadius: 2,
                  background: t.color, marginRight: 6, verticalAlign: "middle",
                }} />
                {t.label}
              </button>
            ))}
            <button
              className="ctrl-btn ctrl-btn--ghost"
              onClick={() => setMarks({})}
              disabled={markCount === 0}
              style={{ marginLeft: "auto" }}
            >
              clear all
            </button>
          </div>
          <div className="canvas-tools-row">
            <span className="canvas-tools-hint" style={{ marginRight: 4 }}>draw:</span>
            {TOOLS.filter((t) => t.key === "brush" || t.key === "erase").map((t) => (
              <button
                key={t.key}
                className="ctrl-btn"
                onClick={() => { setTool(t.key); setObjectSelectMode(false); }}
                title={
                  t.key === "brush" ? "Drag over the photo to mark every boundary the stroke passes near as a soft split"
                  : "Drag over the photo to clear whatever mark (of any type) is on boundaries the stroke passes near"
                }
                style={{
                  borderColor: tool === t.key ? t.color : "transparent",
                  color: tool === t.key ? t.color : "var(--text-dim)",
                  fontWeight: tool === t.key ? 700 : 400,
                }}
              >
                <span style={{
                  display: "inline-block", width: 9, height: 9, borderRadius: 2,
                  background: t.color, marginRight: 6, verticalAlign: "middle",
                }} />
                {t.label}
              </button>
            ))}
            {(tool === "brush" || tool === "erase") && (
              <label title="Radius (image pixels) within which a boundary is painted/erased" style={{ fontSize: 11, color: "var(--text-dim)" }}>
                size {brushSize}px
                <input
                  type="range" min={5} max={80} step={1} value={brushSize}
                  onChange={(ev) => setBrushSize(Number(ev.target.value))}
                  style={{ width: 90 }}
                />
              </label>
            )}
          </div>
          {/* auto: AI-assisted batch marking (SAM object outline / heuristic ranking) --
              everything below "advanced" is manual solver/objective tuning */}
          <div className="canvas-tools-row">
            <span className="canvas-tools-hint" style={{ marginRight: 4 }}>auto:</span>
            <button
              className="ctrl-btn ctrl-btn--ghost"
              onClick={() => setObjectSelectMode((v) => !v)}
              title="Click a point on the photo to run SAM and mark every boundary tracing that object's outline as a soft split -- leaves your existing marks untouched. SAM warms up during upload, so clicks are normally fast; occasionally the first one on a fresh server still needs a moment."
              style={{
                borderColor: objectSelectMode ? "var(--ink)" : "transparent",
                color: objectSelectMode ? "var(--ink)" : "var(--text-dim)",
                fontWeight: objectSelectMode ? 700 : 400,
              }}
            >
              {isSelectingObject ? <span className="go-btn-spinner" /> : <I.Sparkles size={11} />}
              {isSelectingObject
                ? "segmenting…"
                : objectSelectMode ? "click an object…" : "select object"}
            </button>
            <button
              className="ctrl-btn ctrl-btn--ghost"
              onClick={handleAutoSelect}
              title="Mark this image's strongest ~20% of boundaries (by depth discontinuity or Canny edge alignment) as soft splits (a suggestion, not a forced cut) -- leaves your existing marks untouched"
            >
              <I.Sparkles size={11} /> auto select edges
            </button>
            <button
              className="ctrl-btn ctrl-btn--ghost"
              onClick={() => setAdvancedOpen((v) => !v)}
              style={{ marginLeft: "auto" }}
            >
              {advancedOpen ? <I.ChevronUp size={11} /> : <I.ChevronDown size={11} />} advanced
            </button>
          </div>
          {advancedOpen && (
          <>
          <div className="canvas-tools-row">
            <span className="canvas-tools-hint" style={{ marginRight: 4 }}>objective:</span>
            {([
              { key: "depth", label: "depth fit" },
              { key: "cut", label: "cut" },
            ] as const).map((o) => (
              <button
                key={o.key}
                className="ctrl-btn"
                onClick={() => setObjective(o.key)}
                title={o.key === "cut"
                  ? "depth-aware boundary cut only (>=5 spx/layer); your edge marks drive the splits"
                  : "fixed depth-bin fidelity (baseline)"}
                style={{
                  borderColor: objective === o.key ? "var(--ink)" : "transparent",
                  color: objective === o.key ? "var(--ink)" : "var(--text-dim)",
                  fontWeight: objective === o.key ? 700 : 400,
                }}
              >
                {o.label}
              </button>
            ))}
            {objective === "cut" && (
              <div style={{ display: "flex", alignItems: "center", gap: 4, marginLeft: 8 }}>
                <span className="canvas-tools-hint" style={{ marginRight: 2 }}>score:</span>
                {([
                  { key: "laplacian", label: "laplacian" },
                  { key: "meandiff", label: "meandiff" },
                ] as const).map((s) => (
                  <button
                    key={s.key}
                    className="ctrl-btn"
                    onClick={() => setCutScore(s.key)}
                    title={s.key === "laplacian"
                      ? "Cut cost from the Laplacian-of-Gaussian sampled across each boundary (crisp depth edges; uses the LoG σ slider below)"
                      : "Cut cost from the region-mean depth gap |d_i − d_j| (simpler, σ-independent; cheaper to cut where mean depths differ)"}
                    style={{
                      borderColor: cutScore === s.key ? "var(--ink)" : "transparent",
                      color: cutScore === s.key ? "var(--ink)" : "var(--text-dim)",
                      fontWeight: cutScore === s.key ? 700 : 400,
                    }}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            )}
          </div>
          {/* advanced: connectivity/fabrication model -- applies to either objective */}
          <div className="canvas-tools-row">
            <span className="canvas-tools-hint" style={{ marginRight: 4 }}>advanced · connectivity:</span>
            <label
              title="Flow connectivity: every retained piece must reach the frame. Off = explore unfabricable layerings (pieces may float)"
              style={{ fontSize: 11, color: "var(--text-dim)" }}
            >
              <input
                type="checkbox" checked={connectivity}
                onChange={(ev) => setConnectivity(ev.target.checked)}
              />
              fabrication
            </label>
            <label
              title="Full backing: once a region shows on layer l, every sheet behind l keeps its material (y = cumsum x). Same front view, simpler MIP, heavier build"
              style={{ fontSize: 11, color: "var(--text-dim)" }}
            >
              <input
                type="checkbox" checked={yMonotone}
                onChange={(ev) => setYMonotone(ev.target.checked)}
              />
              full backing
            </label>
            <label
              title="Lazy cut-set connectivity: no flow variables; frame-connectivity cuts are added only when a candidate violates them. Same optimum as flow, usually better incumbents on N>=4 cut solves. Needs fabrication ON"
              style={{ fontSize: 11, color: "var(--text-dim)", opacity: connectivity ? 1 : 0.4 }}
            >
              <input
                type="checkbox" checked={lazyConn} disabled={!connectivity}
                onChange={(ev) => setLazyConn(ev.target.checked)}
              />
              lazy conn
            </label>
          </div>
          {/* advanced: cut-objective-only tuning -- hidden entirely under depth fit, where none
              of these knobs do anything (norel/mip focus/coherence are cut-only by their own
              tooltips; keeping them visible under depth fit was dead UI). Split from the solver
              row below it so neither row is long enough to wrap unpredictably. */}
          {objective === "cut" && (
            <div className="canvas-tools-row">
              <span className="canvas-tools-hint" style={{ marginRight: 4 }}>advanced · cut tuning:</span>
              <label
                title="k-median depth anchor: 0 = pure cut (your marks drive everything); above ~0.5 depth dominates and marks stop mattering"
                style={{ fontSize: 11, color: "var(--text-dim)" }}
              >
                depth λ {lambdaDepth.toFixed(2)}
                <input
                  type="range" min={0} max={1} step={0.05} value={lambdaDepth}
                  onChange={(ev) => setLambdaDepth(Number(ev.target.value))}
                  style={{ width: 110 }}
                />
              </label>
              <label
                title="Area floor: every layer must own at least this fraction of the image (0 = off). If the background dominates the photo, a high floor forces far content to be carved across layers — lower it (5%) to let sky/mountains consolidate"
                style={{ fontSize: 11, color: "var(--text-dim)" }}
              >
                area floor {(minLayerArea * 100).toFixed(0)}%
                <input
                  type="range" min={0} max={0.2} step={0.01} value={minLayerArea}
                  onChange={(ev) => setMinLayerArea(Number(ev.target.value))}
                  style={{ width: 90 }}
                />
              </label>
              <label
                title="Depth-plateau coherence: regions the depth map can't tell apart (gap < 0.02) resist being split across sheets. Fixes arbitrary sky seams / bisected far objects (validated N<=5); on hard 7-layer solves with many marks it can degrade the 180s incumbent — toggle off if layering worsens"
                style={{ fontSize: 11, color: "var(--text-dim)" }}
              >
                <input
                  type="checkbox" checked={cohOn}
                  onChange={(ev) => setCohOn(ev.target.checked)}
                />
                coherence
              </label>
              {cohOn && (
                <label
                  title="Coherence weight: tie-breaker scale (0.05 default). If it visibly fights the layering, it's too high"
                  style={{ fontSize: 11, color: "var(--text-dim)" }}
                >
                  coh λ {lambdaCoh.toFixed(2)}
                  <input
                    type="range" min={0.01} max={0.2} step={0.01} value={lambdaCoh}
                    onChange={(ev) => setLambdaCoh(Number(ev.target.value))}
                    style={{ width: 80 }}
                  />
                </label>
              )}
            </div>
          )}
          {/* advanced: Gurobi search-strategy hints (cut objective only) -- split from the cut-
              tuning row above so neither is long enough to wrap unpredictably */}
          {objective === "cut" && (
            <div className="canvas-tools-row">
              <span className="canvas-tools-hint" style={{ marginRight: 4 }}>advanced · solver:</span>
              <label
                title="Gurobi NoRelHeurTime=60: spend the first 60s in the no-relaxation heuristic. Rescues incumbents on N>=4 cut solves where the LP bound is useless (cut objective only)"
                style={{ fontSize: 11, color: "var(--text-dim)" }}
              >
                <input
                  type="checkbox" checked={norelOn}
                  onChange={(ev) => setNorelOn(ev.target.checked)}
                />
                norel 60s
              </label>
              <label
                title="Gurobi MIPFocus=1: bias the search toward finding feasible solutions over proving bounds (cut objective only)"
                style={{ fontSize: 11, color: "var(--text-dim)" }}
              >
                <input
                  type="checkbox" checked={mipFocusOn}
                  onChange={(ev) => setMipFocusOn(ev.target.checked)}
                />
                mip focus
              </label>
            </div>
          )}
          </>
          )}
        </div>

        {/* named edge configs: save/load the current marks into the session's dataset dir */}
        <div className="canvas-tools">
          <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
            <span className="canvas-tools-hint" style={{ marginRight: 4 }}>configs:</span>
            <input
              type="text"
              placeholder="config name"
              value={configName}
              onChange={(e) => setConfigName(e.target.value.replace(/[^A-Za-z0-9._-]/g, ""))}
              style={{
                width: 130, padding: "4px 8px", borderRadius: 6,
                border: "1px solid var(--border-dim)", background: "var(--bg-card)",
                color: "var(--text-mid)", fontFamily: "var(--font-mono)", fontSize: 11,
              }}
            />
            <button
              className="ctrl-btn ctrl-btn--ghost"
              onClick={handleConfigSave}
              disabled={!configName || markCount === 0 || configBusy}
              title="Save the current marks as a named config in this image's dataset"
            >
              save
            </button>
            <select
              value={selectedConfig}
              onChange={(e) => setSelectedConfig(e.target.value)}
              style={{
                maxWidth: 180, padding: "4px 6px", borderRadius: 6,
                border: "1px solid var(--border-dim)", background: "var(--bg-card)",
                color: "var(--text-mid)", fontFamily: "var(--font-mono)", fontSize: 11,
              }}
            >
              <option value="">— saved configs —</option>
              {configs.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({c.n_marks})
                </option>
              ))}
            </select>
            <button
              className="ctrl-btn ctrl-btn--ghost"
              onClick={handleConfigLoad}
              disabled={!selectedConfig || configBusy}
              title="Replace the current marks with this saved config"
            >
              load
            </button>
            {configError && <span className="seg-error-inline">// {configError}</span>}
          </div>
        </div>

        {/* canvas-wrap is sized to the image; SVG overlays it 1:1 */}
        <div
          className="canvas-wrap"
          style={{ position: "relative", background: "#141311" }}
          onMouseMove={handleRegionHover}
          onMouseLeave={() => setHoverRegion(null)}
        >
          {overlay ? (
            <img
              src={overlay}
              alt="Layer preview"
              draggable={false}
              style={{ display: "block", width: "100%", height: "auto", pointerEvents: "none" }}
            />
          ) : (
            <div style={{ width: "100%", aspectRatio: `${sessionWidth} / ${sessionHeight}` }} />
          )}
          {hoverRegion && regionDepth[hoverRegion.idx] !== undefined && (
            <div style={{
              position: "absolute",
              // anchor on whichever side of the cursor has more room, so the tooltip
              // stays inside the (overflow:hidden) canvas card near the edges
              ...(hoverRegion.px > hoverRegion.w / 2
                ? { right: hoverRegion.w - hoverRegion.px + 14 }
                : { left: hoverRegion.px + 14 }),
              ...(hoverRegion.py > hoverRegion.h / 2
                ? { bottom: hoverRegion.h - hoverRegion.py + 14 }
                : { top: hoverRegion.py + 14 }),
              pointerEvents: "none", zIndex: 5, background: "rgba(253,252,249,0.96)",
              border: "1px solid var(--border-bright)", borderRadius: 4,
              padding: "6px 8px", fontSize: 11, fontFamily: "var(--font-mono)",
              color: "var(--text-mid)", whiteSpace: "pre", lineHeight: 1.5,
              boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
            }}>
              {(() => {
                const d = regionDepth[hoverRegion.idx];
                const lines = [`region ${hoverRegion.idx} · depth ${d.toFixed(3)}`];
                if (result?.layerOf?.[hoverRegion.idx] !== undefined)
                  lines.push(`assigned layer ${result.layerOf[hoverRegion.idx]}`);
                if (result?.z?.length) {
                  const dist = result.z.map((zi) => Math.abs(d - zi));
                  const best = dist.indexOf(Math.min(...dist));
                  result.z.forEach((zi, i) => {
                    const star = i === best ? " ◂" : "";
                    lines.push(`z${i + 1}=${zi.toFixed(2)}  |d−z|=${dist[i].toFixed(3)}${star}`);
                  });
                }
                return lines.join("\n");
              })()}
            </div>
          )}
          <svg
            style={{
              position: "absolute", top: 0, left: 0, width: "100%", height: "100%",
              cursor: objectSelectMode ? (isSelectingObject ? "wait" : "crosshair")
                     : (tool === "brush" || tool === "erase") ? "none" : undefined,
              touchAction: (tool === "brush" || tool === "erase") ? "none" : undefined,
            }}
            viewBox={`0 0 ${sessionWidth} ${sessionHeight}`}
            preserveAspectRatio="none"
            onClick={handleObjectClick}
            onPointerDown={handleBrushPointerDown}
            onPointerMove={handleBrushPointerMove}
            onPointerUp={handleBrushPointerUp}
            onPointerLeave={handleBrushPointerLeave}
          >
            {edges.map((e) => {
              const m = marks[e.index];
              const d = edgePath(e.segments);
              return (
                <g
                  key={e.index}
                  onClick={(ev) => { ev.stopPropagation(); applyMark(e.index); }}
                  onMouseEnter={() => setHoveredIdx(e.index)}
                  onMouseLeave={() => setHoveredIdx(null)}
                  style={{ cursor: "pointer" }}
                >
                  <path d={d} stroke="transparent" strokeWidth={10} fill="none" />
                  <path
                    d={d}
                    stroke={strokeFor(e.index, m)}
                    strokeWidth={m ? 2.5 : 1}
                    fill="none"
                    strokeLinecap="round"
                    pointerEvents="none"
                  />
                </g>
              );
            })}
            {/* brush/erase radius cursor -- shows exactly what the next paint stroke will reach */}
            {(tool === "brush" || tool === "erase") && brushCursor && (
              <circle
                cx={brushCursor.x} cy={brushCursor.y} r={brushSize}
                fill={tool === "brush" ? "rgba(16,185,129,0.15)" : "rgba(156,163,175,0.15)"}
                stroke={tool === "brush" ? "#10b981" : "#9ca3af"}
                strokeWidth={1.5}
                pointerEvents="none"
              />
            )}
          </svg>
        </div>
      </div>

      {/* 3D book preview: solver's per-layer silhouettes stacked in depth (drag to rotate) */}
      {result?.masks && result.masks.length > 0 && (
        <div className="canvas-card">
          <div className="canvas-tools">
            <span className="canvas-tools-hint" style={{ marginRight: 4 }}>
              <I.BookOpen /> book preview — solved layer stack · drag to rotate, scroll to zoom
            </span>
            <span className="viz-ctrl-label-layers">{result.masks.length} layers · {result.status}</span>
          </div>
          <BookVisualizer masks={result.masks} sheetMasks={result.sheetMasks} />
        </div>
      )}

      {/* crease map: live heatmap of per-boundary depth-edge strength (the cut score) */}
      <div className="canvas-card">
        <div className="canvas-tools canvas-tools--stacked">
          <div className="canvas-tools-row">
            <span className="canvas-tools-hint" style={{ fontSize: 12 }}>
              <I.Brush /> crease map — cut-cost field · soft marks read as free · click to mark
            </span>
          </div>
          <div className="canvas-tools-row">
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--text-dim)" }}>
              LoG σ {logSigma.toFixed(1)}px
              <input
                type="range" min={0.5} max={8} step={0.5} value={logSigma}
                onChange={(ev) => setLogSigma(Number(ev.target.value))}
                style={{ width: 110 }}
              />
            </label>
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "var(--text-dim)", marginLeft: "auto" }}>
              <span>costly to cut</span>
              <span style={{
                width: 120, height: 8, borderRadius: 4, display: "inline-block",
                background: "linear-gradient(90deg, rgb(12,14,40), rgb(84,24,120), rgb(201,44,92), rgb(246,130,32), rgb(255,240,130))",
              }} />
              <span>free · crease</span>
            </div>
          </div>
        </div>
        <div className="canvas-wrap" style={{ position: "relative", background: "#101010" }}>
          <div style={{ width: "100%", aspectRatio: `${sessionWidth} / ${sessionHeight}` }} />
          <svg
            style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}
            viewBox={`0 0 ${sessionWidth} ${sessionHeight}`}
            preserveAspectRatio="none"
          >
            {[...edges]
              .sort((a, b) => creaseCuttability(a) - creaseCuttability(b))
              .map((e) => {
                const s = creaseCuttability(e);
                const d = edgePath(e.segments);
                const hov = hoveredIdx === e.index;
                return (
                  <g
                    key={e.index}
                    onClick={(ev) => { ev.stopPropagation(); applyMark(e.index); }}
                    onMouseEnter={() => setHoveredIdx(e.index)}
                    onMouseLeave={() => setHoveredIdx(null)}
                    style={{ cursor: "pointer" }}
                  >
                    {/* wide invisible hit target so thin creases are still easy to click */}
                    <path d={d} stroke="transparent" strokeWidth={10} fill="none" />
                    {/* colour = cut-cost field (soft marks fold in as free); no mark highlight */}
                    <path
                      d={d}
                      stroke={heatColor(s)}
                      strokeWidth={0.5 + s * 3}
                      opacity={0.2 + 0.8 * s}
                      fill="none"
                      strokeLinecap="round"
                      pointerEvents="none"
                    />
                    {/* transient hover cue only (not a selection highlight) */}
                    {hov && (
                      <path
                        d={d}
                        stroke="rgba(255,255,255,0.5)"
                        strokeWidth={1.2}
                        fill="none"
                        strokeLinecap="round"
                        pointerEvents="none"
                      />
                    )}
                  </g>
                );
              })}
          </svg>
        </div>
      </div>

      <div className="layer-controls">
        <div className="layer-controls-left">
          <div className="points-pill" style={{ color: "#f59e0b", borderColor: "#f59e0b44" }}>
            {counts.split_soft} soft splits
          </div>
          <div className="points-pill" style={{ color: "#ef4444", borderColor: "#ef444444" }}>
            {counts.split_hard} hard splits
          </div>
          <div className="points-pill" style={{ color: "#3b82f6", borderColor: "#3b82f644" }}>
            {counts.delete} deletes
          </div>
          {result && (
            <div className="points-pill">
              {result.status} · {result.runtime.toFixed(1)}s · obj {result.objective.total.toFixed(3)}
            </div>
          )}
        </div>
        <div className="layer-controls-right">
          {selectObjectError && <span className="seg-error-inline">// {selectObjectError}</span>}
          {solveError && <span className="seg-error-inline">// {solveError}</span>}
          <button
            className="ctrl-btn ctrl-btn--ghost"
            onClick={() => onSubmit(markingsArray(), objective, markCount, lambdaDepth, connectivity, yMonotone, connMethod, norelTime, mipFocus, lambdaCoherence, minLayerArea, cutScore)}
          >
            export →
          </button>
          <button className="next-btn" onClick={handleSolve} disabled={isSolving}>
            {isSolving ? (
              <><span className="go-btn-spinner" /> solving…</>
            ) : (
              <><I.Layers size={14} /> solve / preview</>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Output Screen ────────────────────────────────────────────────────────────

function OutputScreen({
  imageFile,
  sessionId,
  numLayers,
  selectedEdgeCount,
  markings,
  objective,
  lambdaDepth,
  connectivity,
  yMonotone,
  connectivityMethod,
  norelTime,
  mipFocus,
  lambdaCoherence,
  minLayerArea,
  cutScore,
  exportMode,
  frameWidthIn,
  frameBorderIn,
  engraveLayers,
  minEngraveInPerLayer,
  setEngraveLayers,
  setMinEngraveInPerLayer,
  onBack,
}: {
  imageFile: File | null;
  sessionId: string | null;
  numLayers: number;
  selectedEdgeCount: number;
  markings: { index: number; type: MarkType }[];
  objective: "depth" | "cut";
  lambdaDepth: number;
  connectivity: boolean;
  yMonotone: boolean;
  connectivityMethod: ConnMethod;
  norelTime: number;
  mipFocus: number;
  lambdaCoherence: number;
  minLayerArea: number;
  cutScore: CutScore;
  exportMode: ExportMode;
  frameWidthIn: number;
  frameBorderIn: number;
  engraveLayers: boolean[];
  minEngraveInPerLayer: number[];
  setEngraveLayers: (v: boolean[]) => void;
  setMinEngraveInPerLayer: (v: number[]) => void;
  onBack: () => void;
}) {
  const [isExportingStand, setIsExportingStand] = useState(false);
  const [standError, setStandError] = useState<string | null>(null);
  const [isExportingLayers, setIsExportingLayers] = useState(false);
  const [layersError, setLayersError] = useState<string | null>(null);

  const baseName = imageFile ? imageFile.name.replace(/\.[^/.]+$/, "") : "output";
  const safeBase = baseName.replace(/\s+/g, "_");
  const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "");

  const dlBlob = (name: string, blob: Blob) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadStand = async () => {
    if (!sessionId) return;
    setStandError(null);
    setIsExportingStand(true);
    try {
      dlBlob(
        `TunnelBook_${safeBase}_stand_${stamp}.ai`,
        await exportStand(sessionId, numLayers),
      );
    } catch (err: any) {
      setStandError(err?.message ?? "Stand export failed");
    } finally {
      setIsExportingStand(false);
    }
  };

  const handleDownloadLayers = async () => {
    if (!sessionId) return;
    setLayersError(null);
    setIsExportingLayers(true);
    try {
      dlBlob(
        `TunnelBook_${safeBase}_layers_${stamp}.zip`,
        await exportLayers(
          sessionId, markings, numLayers, objective, lambdaDepth,
          connectivity, yMonotone, connectivityMethod, norelTime, mipFocus,
          lambdaCoherence, minLayerArea, cutScore,
          exportMode, frameWidthIn, frameBorderIn,
          engraveLayers, minEngraveInPerLayer,
        ),
      );
    } catch (err: any) {
      setLayersError(err?.message ?? "Layer export failed");
    } finally {
      setIsExportingLayers(false);
    }
  };

  return (
    <div className="output-screen">
      <div className="output-topbar">
        <button className="back-btn" onClick={onBack}>
          <I.ChevronLeft /> back
        </button>
        <div className="output-title">
          <I.CheckCircle /> <span>Edges saved</span>
        </div>
        <div className="output-badge">{numLayers} layers</div>
      </div>

      <div className="output-card">
        <div className="output-card-header">// {imageFile?.name}</div>
        <div className="output-row">
          <div className="output-row-left">
            <div className="output-dot" style={{ background: "#ef4444" }} />
            <div className="output-info">
              <span className="output-filename">{selectedEdgeCount} cut edge{selectedEdgeCount !== 1 ? "s" : ""} selected</span>
              <div className="output-meta">
                <span className="output-layer-tag">Step 1 complete</span>
                <span className="output-mode-tag"><I.Scissors /> outline</span>
              </div>
            </div>
          </div>
        </div>
        <div style={{ padding: "10px 16px 14px", color: "var(--text-dim)", fontSize: 11, lineHeight: 1.6 }}>
          // edge selections stored · superpixels computed<br />
          // depth binning + layer assignment → step 2
        </div>
      </div>

      {exportMode === "engraving" && numLayers > 0 && (
        <div className="output-card" style={{ marginTop: 12 }}>
          <div className="output-card-header">// per-layer engraving</div>
          <div style={{ padding: "8px 16px 14px", display: "flex", flexDirection: "column", gap: 8 }}>
            {Array.from({ length: numLayers }).map((_, i) => {
              const on = engraveLayers[i] ?? true;
              const floor = minEngraveInPerLayer[i] ?? 0.03;
              return (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12 }}>
                  <label style={{ display: "flex", alignItems: "center", gap: 6, minWidth: 90, cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={on}
                      onChange={(e) => {
                        const next = Array.from({ length: numLayers }, (_, k) => engraveLayers[k] ?? true);
                        next[i] = e.target.checked;
                        setEngraveLayers(next);
                      }}
                    />
                    Layer {i + 1}
                  </label>
                  <input
                    type="range"
                    min={0.01}
                    max={0.2}
                    step={0.01}
                    value={floor}
                    disabled={!on}
                    onChange={(e) => {
                      const next = Array.from({ length: numLayers }, (_, k) => minEngraveInPerLayer[k] ?? 0.03);
                      next[i] = parseFloat(e.target.value);
                      setMinEngraveInPerLayer(next);
                    }}
                    style={{ flex: 1, opacity: on ? 1 : 0.4 }}
                    title="Engrave detail floor — lower keeps finer texture, higher de-speckles"
                  />
                  <span style={{ minWidth: 68, textAlign: "right", color: "var(--text-dim)" }}>
                    {on ? `${floor.toFixed(2)}″ detail` : "off"}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="output-actions">
        <button
          className="action-btn action-btn--layers"
          onClick={handleDownloadLayers}
          disabled={isExportingLayers || !sessionId}
          title={`Solve and export all ${numLayers} layer sheets as .ai (${exportMode})`}
        >
          <I.Layers /> {isExportingLayers ? "Solving…" : "Export layers (.zip)"}
        </button>
        <button
          className="action-btn action-btn--stand"
          onClick={handleDownloadStand}
          disabled={isExportingStand || !sessionId}
          title={`Generate a ${numLayers}-slot laser-cut stand`}
        >
          <I.DownloadCloud /> {isExportingStand ? "Generating…" : "Export stand (.ai)"}
        </button>
      </div>

      {layersError && (
        <div className="error-banner">
          <span className="error-banner-tag">Error</span> {layersError}
        </div>
      )}
      {standError && (
        <div className="error-banner">
          <span className="error-banner-tag">Error</span> {standError}
        </div>
      )}
    </div>
  );
}

// ─── App Root ─────────────────────────────────────────────────────────────────

function App() {
  const [screen, setScreen] = useState<Screen>("home");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [, setImageUrl] = useState<string | null>(null);  // url set on upload; only imageFile is read here
  const [totalLayers, setTotalLayers] = useState(0);
  const [showHelp, setShowHelp] = useState(false);
  const [exportMode, setExportMode] = useState<ExportMode>("outline");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionWidth, setSessionWidth] = useState(0);
  const [sessionHeight, setSessionHeight] = useState(0);
  const [isStarting, setIsStarting] = useState(false);
  const [backendError, setBackendError] = useState<string | null>(null);
  const [frameWidthIn, setFrameWidthIn] = useState(12);
  const [, setFrameHeightIn] = useState(9);  // height follows image aspect in export; only width/border are read
  const [frameBorderIn, setFrameBorderIn] = useState(0.5);
  const [edges, setEdges] = useState<EdgeItem[]>([]);
  const [baselineOverlay, setBaselineOverlay] = useState<string | null>(null);
  const [edgeConfigs, setEdgeConfigs] = useState<EdgeConfigItem[]>([]);
  const [selectedEdgeCount, setSelectedEdgeCount] = useState(0);
  const [markings, setMarkings] = useState<{ index: number; type: MarkType }[]>([]);
  const [objective, setObjective] = useState<"depth" | "cut">("depth");
  const [lambdaDepth, setLambdaDepth] = useState(0);
  const [connectivity, setConnectivity] = useState(true);
  const [yMonotone, setYMonotone] = useState(false);
  const [connectivityMethod, setConnectivityMethod] = useState<ConnMethod>("flow");
  const [norelTime, setNorelTime] = useState(60);
  const [mipFocus, setMipFocus] = useState(1);
  const [lambdaCoherence, setLambdaCoherence] = useState(0);
  const [minLayerArea, setMinLayerArea] = useState(0.10);
  const [cutScore, setCutScore] = useState<CutScore>("laplacian");
  const [regionMap, setRegionMap] = useState<string | null>(null);
  const [regionDepth, setRegionDepth] = useState<number[]>([]);
  // per-layer engrave overrides (sized to the layer count on solve); [] = use the global mode/floor
  const [engraveLayers, setEngraveLayers] = useState<boolean[]>([]);
  const [minEngraveInPerLayer, setMinEngraveInPerLayer] = useState<number[]>([]);

  const reset = async () => {
    if (sessionId) await deleteSession(sessionId);
    setSessionId(null);
    setSessionWidth(0);
    setSessionHeight(0);
    setTotalLayers(0);
    setEdges([]);
    setBaselineOverlay(null);
    setEdgeConfigs([]);
    setSelectedEdgeCount(0);
    setMarkings([]);
    setObjective("depth");
    setBackendError(null);
    setFrameWidthIn(12);
    setFrameHeightIn(9);
    setFrameBorderIn(0.5);
    setEngraveLayers([]);
    setMinEngraveInPerLayer([]);
  };

  const handleGo = async (
    file: File,
    url: string,
    count: number,
    fwIn: number,
    fhIn: number,
    fbIn: number,
    edgeCondition: boolean,
  ) => {
    setFrameWidthIn(fwIn);
    setFrameHeightIn(fhIn);
    setFrameBorderIn(fbIn);
    setBackendError(null);
    setIsStarting(true);
    try {
      const data = await createSession(file, count, edgeCondition);
      setImageFile(file);
      setImageUrl(url);
      setTotalLayers(count);
      setEngraveLayers(Array(count).fill(true));         // all layers engrave by default
      setMinEngraveInPerLayer(Array(count).fill(0.03));  // default detail floor (exporter default)
      setSessionId(data.sessionId);
      setSessionWidth(data.width);
      setSessionHeight(data.height);
      setEdges(data.edges);
      setBaselineOverlay(data.baselineOverlay);
      setRegionMap(data.regionMap ?? null);
      setRegionDepth(data.regionDepth ?? []);
      setEdgeConfigs(data.edgeConfigs ?? []);
      setScreen("edges");
    } catch (err: any) {
      setBackendError(err?.message ?? "Failed to start backend");
    } finally {
      setIsStarting(false);
    }
  };

  const handleEdgeSubmit = (
    marks: { index: number; type: MarkType }[],
    obj: "depth" | "cut",
    markCount: number,
    lambdaD: number,
    conn: boolean,
    yMono: boolean,
    connMethod: ConnMethod,
    norel: number,
    focus: number,
    lambdaCoh: number,
    minArea: number,
    cScore: CutScore,
  ) => {
    setMarkings(marks);
    setObjective(obj);
    setLambdaDepth(lambdaD);
    setConnectivity(conn);
    setYMonotone(yMono);
    setConnectivityMethod(connMethod);
    setLambdaCoherence(lambdaCoh);
    setMinLayerArea(minArea);
    setCutScore(cScore);
    setNorelTime(norel);
    setMipFocus(focus);
    setSelectedEdgeCount(markCount);
    setScreen("output");
  };

  const handleBack = () => {
    setScreen("home");
    reset();
  };

  return (
    <div className="app-root">
      <Sidebar screen={screen} onGoHome={handleBack} />
      <main className="app-main">
        <div className="main-inner">
          {screen === "home" && (
            <HomeScreen
              onGo={handleGo}
              isStarting={isStarting}
              error={backendError}
              exportMode={exportMode}
              onExportModeChange={setExportMode}
            />
          )}
          {screen === "edges" && sessionId && (
            <EdgeSelectionScreen
              sessionId={sessionId}
              sessionWidth={sessionWidth}
              sessionHeight={sessionHeight}
              numLayers={totalLayers}
              edges={edges}
              baselineOverlay={baselineOverlay}
              regionMap={regionMap}
              regionDepth={regionDepth}
              edgeConfigs={edgeConfigs}
              onSubmit={handleEdgeSubmit}
              onBack={handleBack}
            />
          )}
          {screen === "output" && (
            <OutputScreen
              imageFile={imageFile}
              sessionId={sessionId}
              numLayers={totalLayers}
              selectedEdgeCount={selectedEdgeCount}
              markings={markings}
              objective={objective}
              lambdaDepth={lambdaDepth}
              connectivity={connectivity}
              yMonotone={yMonotone}
              connectivityMethod={connectivityMethod}
              norelTime={norelTime}
              mipFocus={mipFocus}
              lambdaCoherence={lambdaCoherence}
              minLayerArea={minLayerArea}
              cutScore={cutScore}
              exportMode={exportMode}
              frameWidthIn={frameWidthIn}
              frameBorderIn={frameBorderIn}
              engraveLayers={engraveLayers}
              minEngraveInPerLayer={minEngraveInPerLayer}
              setEngraveLayers={setEngraveLayers}
              setMinEngraveInPerLayer={setMinEngraveInPerLayer}
              onBack={handleBack}
            />
          )}
        </div>
        <button className="help-btn" onClick={() => setShowHelp(true)} title="How it works">
          ?
        </button>
      </main>
      {showHelp && <HelpModal onClose={() => setShowHelp(false)} />}
    </div>
  );
}

export default App;
