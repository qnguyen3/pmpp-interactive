import type { ReactNode } from 'react';
import type { Tier } from '../../components/notes';
import type { Graph, Role } from '../../sim/bfs';
import { tr } from '../../lib/i18n';

// The drawing of a small directed graph that chapter 18's interactives share: vertices as
// circles, edges as arrows (a pair of opposite edges drawn side by side), optional level
// labels, and optional thread glyphs on vertices or edges. Everything it shows comes from
// its props; it computes nothing about BFS itself.
//
// Colour follows the course's rule: global memory (level[], the frontier) in the global
// tier colour, failure in the failure colour, everything else ink.
export type VertexLook = 'plain' | 'root' | 'visited' | 'new' | 'unvisited' | 'dim';
export type EdgeLook = 'plain' | 'dim' | 'tree' | 'out' | 'in' | 'pick' | 'path';
export interface Glyph {
  /** A thread on a vertex or on an edge (index into g.edges). */
  vertex?: number;
  edge?: number;
  role: Role;
  /** Global thread index, shown on hover and used by checks. */
  t: number;
  selected?: boolean;
}

const W = 420;
const H = 270;
const PAD = 26;
const R = 12;

export function place(g: Graph, w = W, h = H) {
  return g.pos.map((p) => ({ x: PAD + p.x * (w - 2 * PAD), y: PAD + p.y * (h - 2 * PAD) }));
}

export function BfsGraph({
  g,
  label,
  look = () => 'plain',
  edgeLook = () => 'plain',
  tag,
  ring,
  glyphs = [],
  onVertex,
  onEdge,
  onGlyph,
  height = H,
  children,
}: {
  g: Graph;
  label: string;
  look?: (v: number) => VertexLook;
  edgeLook?: (i: number, s: number, d: number) => EdgeLook;
  /** Small text beside a vertex, such as its level. */
  tag?: (v: number) => string | null;
  /** Vertices drawn with an extra ring: the selected one, or one that may be written twice. */
  ring?: (v: number) => 'pick' | 'fail' | null;
  glyphs?: Glyph[];
  onVertex?: (v: number) => void;
  onEdge?: (i: number) => void;
  onGlyph?: (t: number) => void;
  height?: number;
  children?: ReactNode;
}) {
  const pt = place(g, W, height);
  const has = new Set(g.edges.map(([s, d]) => s + ',' + d));
  // The geometry of edge i: from the rim of its source to the rim of its destination,
  // shifted sideways when the opposite edge exists, so both arrows show.
  const geo = (i: number) => {
    const [s, d] = g.edges[i];
    const a = pt[s];
    const b = pt[d];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len;
    const uy = dy / len;
    const off = has.has(d + ',' + s) ? 3.6 : 0;
    const nx = -uy * off;
    const ny = ux * off;
    const x1 = a.x + ux * R + nx;
    const y1 = a.y + uy * R + ny;
    const x2 = b.x - ux * (R + 2) + nx;
    const y2 = b.y - uy * (R + 2) + ny;
    return { x1, y1, x2, y2, ux, uy, mx: (x1 + x2) / 2 - uy * (off ? 6 : 7), my: (y1 + y2) / 2 + ux * (off ? 6 : 7) };
  };
  const stroke: Record<EdgeLook, { s: string; w: number; dash?: string; o?: number }> = {
    plain: { s: 'var(--ink-2)', w: 1.3 },
    dim: { s: 'var(--ink-3)', w: 1, o: 0.45 },
    tree: { s: 'var(--ink)', w: 2.6 },
    out: { s: 'var(--ink)', w: 2.8 },
    in: { s: 'var(--ink)', w: 2.4, dash: '5 3' },
    pick: { s: 'var(--global)', w: 3.2 },
    path: { s: 'var(--global)', w: 3.4 },
  };
  const fill: Record<VertexLook, { f: string; t: string; s: string; o?: number }> = {
    plain: { f: 'var(--cell)', t: 'var(--ink)', s: 'var(--ink)' },
    root: { f: 'var(--ink)', t: 'var(--page)', s: 'var(--ink)' },
    visited: { f: 'var(--global-wash)', t: 'var(--ink)', s: 'var(--global)' },
    new: { f: 'var(--global)', t: 'var(--page)', s: 'var(--global)' },
    unvisited: { f: 'var(--cell)', t: 'var(--ink)', s: 'var(--ink-3)' },
    dim: { f: 'var(--cell)', t: 'var(--ink-3)', s: 'var(--ink-3)', o: 0.6 },
  };
  const glyphAt = (gl: Glyph) => {
    if (gl.vertex !== undefined) return { x: pt[gl.vertex].x - 15, y: pt[gl.vertex].y - 14 };
    const e = geo(gl.edge!);
    return { x: e.mx, y: e.my };
  };
  return (
    <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={label} className="bfs-graph" style={{ width: '100%', maxWidth: 560, display: 'block', touchAction: 'manipulation' }}>
      {g.edges.map(([s, d], i) => {
        const e = geo(i);
        const k = edgeLook(i, s, d);
        const st = stroke[k];
        const hx = e.x2;
        const hy = e.y2;
        const back = 8;
        const side = 4;
        const head = `${hx},${hy} ${hx - e.ux * back - e.uy * side},${hy - e.uy * back + e.ux * side} ${hx - e.ux * back + e.uy * side},${hy - e.uy * back - e.ux * side}`;
        return (
          <g key={i} data-edge={`${s}-${d}`} data-look={k} opacity={st.o ?? 1} onClick={onEdge ? () => onEdge(i) : undefined} style={{ cursor: onEdge ? 'pointer' : undefined }}>
            {onEdge ? <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke="transparent" strokeWidth={12} /> : null}
            <line x1={e.x1} y1={e.y1} x2={hx - e.ux * 6} y2={hy - e.uy * 6} stroke={st.s} strokeWidth={st.w} strokeDasharray={st.dash} />
            <polygon points={head} fill={st.s} />
          </g>
        );
      })}
      {Array.from({ length: g.n }, (_, v) => {
        const k = look(v);
        const f = fill[k];
        const r = ring?.(v);
        const t = tag?.(v);
        return (
          <g key={v} data-vertex={v} data-look={k} onClick={onVertex ? () => onVertex(v) : undefined} style={{ cursor: onVertex ? 'pointer' : undefined }} opacity={f.o ?? 1}>
            {r ? <circle cx={pt[v].x} cy={pt[v].y} r={R + 4.5} fill="none" stroke={r === 'fail' ? 'var(--fail)' : 'var(--ink)'} strokeWidth={2.2} strokeDasharray={r === 'fail' ? '3 2' : undefined} data-ring={r} /> : null}
            <circle cx={pt[v].x} cy={pt[v].y} r={R} fill={f.f} stroke={f.s} strokeWidth={1.4} />
            <text x={pt[v].x} y={pt[v].y + 4} fontSize="11.5" fontWeight="600" textAnchor="middle" style={{ fill: f.t, pointerEvents: 'none' }}>{v}</text>
            {t != null ? (
              <text x={pt[v].x + R + 3} y={pt[v].y - R + 3} fontSize="10.5" data-tag={t} style={{ fill: 'var(--ink-2)', pointerEvents: 'none' }}>{t}</text>
            ) : null}
          </g>
        );
      })}
      {glyphs.map((gl) => {
        const p = glyphAt(gl);
        const st =
          gl.role === 'labels' ? { s: 'var(--ink)', w: 2.2, dash: undefined, o: 1 }
          : gl.role === 'iterates' ? { s: 'var(--ink)', w: 1.6, dash: '2 2', o: 1 }
          : { s: 'var(--ink-3)', w: 1.1, dash: undefined, o: 0.55 };
        // A short zigzag: one thread.
        const d = `M${p.x},${p.y - 8} l3,3 l-6,3 l6,3 l-3,3 l0,3`;
        return (
          <g key={'t' + gl.t} data-thread={gl.t} data-role={gl.role} onClick={onGlyph ? () => onGlyph(gl.t) : undefined} style={{ cursor: onGlyph ? 'pointer' : undefined }}>
            {onGlyph ? <rect x={p.x - 7} y={p.y - 10} width={14} height={20} fill="transparent" /> : null}
            {gl.selected ? <rect x={p.x - 6} y={p.y - 10} width={12} height={19} rx={2} fill="none" stroke="var(--ink)" strokeWidth={1} /> : null}
            <path d={d} fill="none" stroke={st.s} strokeWidth={st.w} strokeDasharray={st.dash} opacity={st.o} />
          </g>
        );
      })}
      {children}
    </svg>
  );
}

/** The legend for thread glyphs. */
export function GlyphLegend({
  idle = tr('launched, stops at the first check', 'được khởi chạy, dừng ở phép kiểm tra đầu tiên'),
  iterates = tr('loops, labels nothing', 'lặp, không gán nhãn gì'),
  labels = tr('loops and labels', 'lặp và gán nhãn'),
}: { idle?: string; iterates?: string; labels?: string }) {
  const text: Record<Role, string> = { idle, iterates, labels, surplus: '' };
  const one = (role: Role) => {
    const st =
      role === 'labels' ? { s: 'var(--ink)', w: 2.2, dash: undefined, o: 1 }
      : role === 'iterates' ? { s: 'var(--ink)', w: 1.6, dash: '2 2', o: 1 }
      : { s: 'var(--ink-3)', w: 1.1, dash: undefined, o: 0.55 };
    return (
      <span key={role} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
        <svg width="12" height="20" viewBox="0 0 12 20" aria-hidden="true">
          <path d="M6,2 l3,3 l-6,3 l6,3 l-3,3 l0,3" fill="none" stroke={st.s} strokeWidth={st.w} strokeDasharray={st.dash} opacity={st.o} />
        </svg>
        {text[role]}
      </span>
    );
  };
  return (
    <span className="legend">
      {one('idle')}
      {one('iterates')}
      {one('labels')}
    </span>
  );
}

/** Identifiers coloured by where they live: arrays in global memory, per-thread values in registers. */
export const BFS_TIERS: Record<string, Tier> = {
  level: 'global', newVertexVisited: 'global', prevFrontier: 'global', currFrontier: 'global', numCurrFrontier: 'global',
  vertex: 'reg', edge: 'reg', neighbor: 'reg', currLevel: 'reg', i: 'reg', unvisited: 'reg', visitSuccess: 'reg',
  currFrontierIdx: 'reg', numPrevFrontier: 'reg', e: 'reg', won: 'reg', expected: 'reg',
};
