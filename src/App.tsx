import {
  useState,
  useRef,
  useEffect,
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
type MarkType = "split_soft" | "split_hard" | "delete";
type EdgeItem = { index: number; i: number; j: number; segments: number[][]; score?: number };
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

async function createSession(imageFile: File, nLayers: number): Promise<SessionData> {
  const fd = new FormData();
  fd.append("image", imageFile);
  fd.append("n_layers", String(nLayers));
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
): Promise<SolveResult> {
  const res = await fetch(apiUrl(`/api/sessions/${sessionId}/solve`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      markings, n_layers: nLayers, lambda_split: lambdaSplit, objective,
      cut_log_sigma: cutLogSigma, lambda_depth: lambdaDepth,
      connectivity, y_monotone: yMonotone,
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
): Promise<number[]> {
  const res = await fetch(apiUrl(`/api/sessions/${sessionId}/scores`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cut_log_sigma: cutLogSigma, cut_score: "laplacian" }),
  });
  if (!res.ok) throw new Error(`scores failed (${res.status})`);
  return (await res.json()).scores as number[];
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
  mode: ExportMode,
  contentWidthIn: number,
  borderIn: number,
): Promise<Blob> {
  const res = await fetch(apiUrl(`/api/sessions/${sessionId}/export-ai`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      markings, n_layers: nLayers, objective, lambda_depth: lambdaDepth,
      connectivity, y_monotone: yMonotone,
      mode, content_width_in: contentWidthIn, border_in: borderIn,
    }),
  });
  if (!res.ok)
    throw new Error(
      (await res.text().catch(() => "")) || `Layer export failed (${res.status})`,
    );
  return res.blob();
}

// ─── Shared UI primitives ─────────────────────────────────────────────────────

function Stars() {
  return (
    <div className="stars" aria-hidden="true">
      {[...Array(20)].map((_, i) => (
        <div key={i} className={`star star-${i + 1}`} />
      ))}
    </div>
  );
}

function HelpModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>x</button>
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
          <span className="sidebar-tab-label">upload</span>
          {screen !== "home" && <div className="sidebar-tab-dot" style={{ background: "#6366f1" }} />}
        </div>

        {screen !== "home" && (
          <div
            className={`sidebar-tab ${screen === "edges" ? "sidebar-tab--active" : ""} ${screen === "output" ? "sidebar-tab--done" : ""}`}
          >
            <div
              className="sidebar-tab-dot"
              style={{ background: screen === "output" ? "#22c55e" : "#6366f1" }}
            />
            <span className="sidebar-tab-label">edges</span>
            {screen === "output" && (
              <span className="sidebar-tab-check"><I.Check /></span>
            )}
            {screen === "edges" && <div className="sidebar-tab-pulse" />}
          </div>
        )}

        {screen === "output" && (
          <div className="sidebar-tab sidebar-tab--active sidebar-tab--output">
            <I.CheckCircle size={13} />
            <span className="sidebar-tab-label">output</span>
          </div>
        )}
      </nav>
      <div className="sidebar-footer">
        <span className="sidebar-status">
          {screen === "home" ? "ready" : screen === "edges" ? "selecting" : "complete"}
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
  onGo: (f: File, url: string, n: number, frameWidthIn: number, frameHeightIn: number, frameBorderIn: number) => void;
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
        <div className="home-tag">// AI-powered layer segmentation</div>
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
              onGo(imageFile, imageUrl, parsed, parsedW, parsedH, parsedB);
          }}
          disabled={!canGo}
        >
          {isStarting ? (
            <><span className="go-btn-spinner" /> detecting edges…</>
          ) : (
            <>run <I.ArrowRight /></>
          )}
        </button>
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
          <span className="error-banner-tag">// error</span> {error}
          <div className="error-banner-sub">
            Ensure the Python server is running and Vite proxies <code>/api</code>.
          </div>
        </div>
      )}
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
  ) => void;
  onBack: () => void;
}) {
  const [marks, setMarks] = useState<Record<number, MarkType>>({});
  const [tool, setTool] = useState<MarkType>("split_soft");
  const [objective, setObjective] = useState<"depth" | "cut">("depth");
  const [lambdaDepth, setLambdaDepth] = useState(0);
  const [connectivity, setConnectivity] = useState(true);
  const [yMonotone, setYMonotone] = useState(false);
  const [logSigma, setLogSigma] = useState(2.0);
  // per-pixel positional region index decoded from regionMap (idx+1 in R + G<<8; 0 = none)
  const regionIdxRef = useRef<{ data: Uint8ClampedArray; w: number; h: number } | null>(null);
  const [hoverRegion, setHoverRegion] = useState<{ idx: number; px: number; py: number } | null>(null);
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

  const TOOLS: { key: MarkType; label: string; color: string }[] = [
    { key: "split_soft", label: "split · soft", color: "#f59e0b" },
    { key: "split_hard", label: "split · hard", color: "#ef4444" },
    { key: "delete", label: "delete", color: "#3b82f6" },
  ];
  const colorOf = (m: MarkType) => TOOLS.find((t) => t.key === m)!.color;

  const applyMark = (i: number) =>
    setMarks((prev) => {
      const next = { ...prev };
      if (next[i] === tool) delete next[i];
      else next[i] = tool;
      return next;
    });

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
      fetchScores(sessionId, logSigma)
        .then(setScoreOverride)
        .catch(() => {});
    }, 150);
    return () => clearTimeout(t);
  }, [logSigma, sessionId]);

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
    setHoverRegion({ idx, px: ev.clientX - rect.left, py: ev.clientY - rect.top });
  };

  const strokeFor = (i: number, m: MarkType | undefined) => {
    const hov = hoveredIdx === i;
    if (!m) return hov ? "rgba(255,255,255,0.85)" : "rgba(255,255,255,0.22)";
    return colorOf(m);
  };

  const handleSolve = async () => {
    setIsSolving(true);
    setSolveError(null);
    try {
      const r = await solveSession(sessionId, markingsArray(), numLayers, 1.0, objective, logSigma, lambdaDepth, connectivity, yMonotone);
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
          <div className="layer-badge" style={{ background: "#6366f1" }}>
            mark_boundaries
          </div>
          <span className="layer-of">{numLayers} layers · {edges.length} boundaries</span>
        </div>
        <div className="layer-progress-wrap" style={{ alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 11, color: "var(--text-dim)" }}>{markCount} marked</span>
        </div>
      </div>

      <div className="canvas-card">
        <div className="canvas-tools">
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <span className="canvas-tools-hint" style={{ marginRight: 4 }}>
              <I.Brush /> tool:
            </span>
            {TOOLS.map((t) => (
              <button
                key={t.key}
                className="ctrl-btn"
                onClick={() => setTool(t.key)}
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
          </div>
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <span className="canvas-tools-hint" style={{ marginRight: 4 }}>objective:</span>
            {([
              { key: "depth", label: "depth fit" },
              { key: "cut", label: "cut · laplacian" },
            ] as const).map((o) => (
              <button
                key={o.key}
                className="ctrl-btn"
                onClick={() => setObjective(o.key)}
                title={o.key === "cut"
                  ? "depth-aware boundary cut only (>=5 spx/layer); your edge marks drive the splits"
                  : "fixed depth-bin fidelity (baseline)"}
                style={{
                  borderColor: objective === o.key ? "#22d3ee" : "transparent",
                  color: objective === o.key ? "#22d3ee" : "var(--text-dim)",
                  fontWeight: objective === o.key ? 700 : 400,
                }}
              >
                {o.label}
              </button>
            ))}
            {objective === "cut" && (
              <label
                title="k-median depth anchor: 0 = pure cut (your marks drive everything); above ~0.5 depth dominates and marks stop mattering"
                style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--text-dim)", marginLeft: 8 }}
              >
                depth λ {lambdaDepth.toFixed(2)}
                <input
                  type="range" min={0} max={1} step={0.05} value={lambdaDepth}
                  onChange={(ev) => setLambdaDepth(Number(ev.target.value))}
                  style={{ width: 110 }}
                />
              </label>
            )}
            <label
              title="Flow connectivity: every retained piece must reach the frame. Off = explore unfabricable layerings (pieces may float)"
              style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--text-dim)", marginLeft: 8 }}
            >
              <input
                type="checkbox" checked={connectivity}
                onChange={(ev) => setConnectivity(ev.target.checked)}
              />
              fabrication
            </label>
            <label
              title="Full backing: once a region shows on layer l, every sheet behind l keeps its material (y = cumsum x). Same front view, simpler MIP, heavier build"
              style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--text-dim)" }}
            >
              <input
                type="checkbox" checked={yMonotone}
                onChange={(ev) => setYMonotone(ev.target.checked)}
              />
              full backing
            </label>
          </div>
          <button
            className="ctrl-btn ctrl-btn--ghost"
            onClick={() => setMarks({})}
            disabled={markCount === 0}
          >
            clear all
          </button>
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
                border: "1px solid var(--border-dim)", background: "rgba(0,0,0,0.3)",
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
                border: "1px solid var(--border-dim)", background: "rgba(0,0,0,0.3)",
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
          style={{ position: "relative", background: "#0b0b12" }}
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
              position: "absolute", left: hoverRegion.px + 14, top: hoverRegion.py + 14,
              pointerEvents: "none", zIndex: 5, background: "rgba(10,10,18,0.92)",
              border: "1px solid rgba(255,255,255,0.18)", borderRadius: 6,
              padding: "6px 8px", fontSize: 11, fontFamily: "monospace",
              color: "var(--text-dim)", whiteSpace: "pre", lineHeight: 1.5,
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
            style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}
            viewBox={`0 0 ${sessionWidth} ${sessionHeight}`}
            preserveAspectRatio="none"
          >
            {edges.map((e) => {
              const m = marks[e.index];
              const d = edgePath(e.segments);
              return (
                <g
                  key={e.index}
                  onClick={() => applyMark(e.index)}
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
          </svg>
        </div>
      </div>

      {/* crease map: live heatmap of per-boundary depth-edge strength (the cut score) */}
      <div className="canvas-card">
        <div className="canvas-tools">
          <span className="canvas-tools-hint" style={{ marginRight: 4 }}>
            <I.Brush /> crease map — cut-cost field · soft marks read as free · click to mark
          </span>
          <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--text-dim)" }}>
            LoG σ {logSigma.toFixed(1)}px
            <input
              type="range" min={0.5} max={8} step={0.5} value={logSigma}
              onChange={(ev) => setLogSigma(Number(ev.target.value))}
              style={{ width: 110 }}
            />
          </label>
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "var(--text-dim)" }}>
            <span>costly to cut</span>
            <span style={{
              width: 120, height: 8, borderRadius: 4, display: "inline-block",
              background: "linear-gradient(90deg, rgb(12,14,40), rgb(84,24,120), rgb(201,44,92), rgb(246,130,32), rgb(255,240,130))",
            }} />
            <span>free · crease</span>
          </div>
        </div>
        <div className="canvas-wrap" style={{ position: "relative", background: "#07070c" }}>
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
                    onClick={() => applyMark(e.index)}
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
            {counts.split_soft} split·soft
          </div>
          <div className="points-pill" style={{ color: "#ef4444", borderColor: "#ef444444" }}>
            {counts.split_hard} split·hard
          </div>
          <div className="points-pill" style={{ color: "#3b82f6", borderColor: "#3b82f644" }}>
            {counts.delete} delete
          </div>
          {result && (
            <div className="points-pill">
              {result.status} · {result.runtime.toFixed(1)}s · obj {result.objective.total.toFixed(3)}
            </div>
          )}
        </div>
        <div className="layer-controls-right">
          {solveError && <span className="seg-error-inline">// {solveError}</span>}
          <button
            className="ctrl-btn ctrl-btn--ghost"
            onClick={() => onSubmit(markingsArray(), objective, markCount, lambdaDepth, connectivity, yMonotone)}
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
  exportMode,
  frameWidthIn,
  frameBorderIn,
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
  exportMode: ExportMode;
  frameWidthIn: number;
  frameBorderIn: number;
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
          connectivity, yMonotone,
          exportMode, frameWidthIn, frameBorderIn,
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
          <I.CheckCircle /> <span>edges_saved</span>
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
                <span className="output-layer-tag">step_1 complete</span>
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

      <div className="output-actions">
        <button
          className="action-btn action-btn--layers"
          onClick={handleDownloadLayers}
          disabled={isExportingLayers || !sessionId}
          title={`Solve and export all ${numLayers} layer sheets as .ai (${exportMode})`}
        >
          <I.Layers /> {isExportingLayers ? "solving…" : "export_layers.zip"}
        </button>
        <button
          className="action-btn action-btn--stand"
          onClick={handleDownloadStand}
          disabled={isExportingStand || !sessionId}
          title={`Generate a ${numLayers}-slot laser-cut stand`}
        >
          <I.DownloadCloud /> {isExportingStand ? "generating…" : "export_stand.ai"}
        </button>
      </div>

      {layersError && (
        <div className="error-banner">
          <span className="error-banner-tag">// error</span> {layersError}
        </div>
      )}
      {standError && (
        <div className="error-banner">
          <span className="error-banner-tag">// error</span> {standError}
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
  const [regionMap, setRegionMap] = useState<string | null>(null);
  const [regionDepth, setRegionDepth] = useState<number[]>([]);

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
  };

  const handleGo = async (
    file: File,
    url: string,
    count: number,
    fwIn: number,
    fhIn: number,
    fbIn: number,
  ) => {
    setFrameWidthIn(fwIn);
    setFrameHeightIn(fhIn);
    setFrameBorderIn(fbIn);
    setBackendError(null);
    setIsStarting(true);
    try {
      const data = await createSession(file, count);
      setImageFile(file);
      setImageUrl(url);
      setTotalLayers(count);
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
  ) => {
    setMarkings(marks);
    setObjective(obj);
    setLambdaDepth(lambdaD);
    setConnectivity(conn);
    setYMonotone(yMono);
    setSelectedEdgeCount(markCount);
    setScreen("output");
  };

  const handleBack = () => {
    setScreen("home");
    reset();
  };

  return (
    <div className="app-root">
      <Stars />
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
              exportMode={exportMode}
              frameWidthIn={frameWidthIn}
              frameBorderIn={frameBorderIn}
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
