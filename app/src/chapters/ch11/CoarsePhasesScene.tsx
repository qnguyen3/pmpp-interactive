import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { koggeStoneWork, sceneData } from '../../sim/coarsescan';
import { warpScanTrace } from '../../sim/warpscan';

// "Three phases of a coarsened scan" (section 11.6): one block of 4 threads, factor 4, on
// Fig. 11.6's sixteen numbers, in the register-tiled form of Fig. 11.13. Every value drawn
// comes from src/sim/coarsescan.ts (sceneData) and src/sim/warpscan.ts; coarsescan.test.ts
// ties the numbers the narration speaks to the same functions. Six beats, one clip each
// (content/ch11/narration/coarse-phases-N.txt).
const d = sceneData();
const P = d.P;
const F = d.F;
const N = d.N;
const ks = warpScanTrace(d.phases.totals); // phase 2's block scan, step by step
const ksRows = [d.phases.totals, ...ks.steps.map((s) => s.lanes.map((l) => l.after))];

const CW = 34;
const GAP = 12;
const X0 = 128;
const ex = (i: number) => X0 + i * CW + Math.floor(i / F) * GAP;
const GY = 30; // global memory, input
const SY = 100; // shared memory
const RY = 178; // registers
const BY = 252; // phase 2 strip
const OY = 318; // global memory, output
const VW = 720;
const VH = 372;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const range = (n: number, from = 0) => Array.from({ length: n }, (_, i) => from + i);

type Tier = 'global' | 'shared' | 'reg';
interface Frame {
  shared?: (number | null)[];
  regs?: (number | null)[][];
  hotGlobal?: Set<number>;
  badGlobal?: Set<number>;
  hotShared?: Set<number>;
  hotReg?: Set<string>;
  out?: (number | null)[];
  hotOut?: Set<number>;
  /** Small labels under a row, one per element: who touches it. */
  tags?: { row: 'global' | 'shared' | 'out'; text: string[]; bad?: boolean };
  /** Phase 2 strip: values under each thread's last element. */
  strip?: { values: (number | null)[]; label: string };
  adds?: (string | null)[];
  arrows?: { x1: number; y1: number; x2: number; y2: number; bad?: boolean }[];
  barrier?: string | null;
  headline?: ReactNode;
  note?: ReactNode;
}

function Cell({ x, y, v, tier, hot, bad }: { x: number; y: number; v: number | null; tier: Tier; hot?: boolean; bad?: boolean }) {
  const fill = bad ? 'var(--fail-wash)' : hot ? `var(--${tier})` : v == null ? 'var(--cell)' : `var(--${tier}-wash)`;
  const stroke = bad ? 'var(--fail)' : `var(--${tier})`;
  return (
    <g>
      <rect x={x} y={y} width={CW - 3} height={26} rx="2" fill={fill} stroke={v == null && !hot && !bad ? 'var(--rule)' : stroke} />
      {v != null ? (
        <text x={x + (CW - 3) / 2} y={y + 17} fontSize="12" textAnchor="middle" style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>{v}</text>
      ) : null}
    </g>
  );
}

function Stage(f: Frame) {
  const label = (y: number, text: string, tier: Tier) => (
    <text x={8} y={y + 17} fontSize="11.5" fontWeight="600" style={{ fill: `var(--${tier})` }}>{text}</text>
  );
  const tagY = (row: 'global' | 'shared' | 'out') => (row === 'global' ? GY : row === 'shared' ? SY : OY) + 38;
  return (
    <svg viewBox={`0 0 ${VW} ${VH}`} role="img" aria-label={tr('One block of four threads scanning sixteen elements', 'Một khối bốn luồng scan mười sáu phần tử')} style={{ width: '100%', maxWidth: VW, display: 'block' }}>
      {label(GY, tr('global (input)', 'toàn cục (đầu vào)'), 'global')}
      {d.values.map((v, i) => (
        <Cell key={'g' + i} x={ex(i)} y={GY} v={v} tier="global" hot={f.hotGlobal?.has(i)} bad={f.badGlobal?.has(i)} />
      ))}
      {label(SY, tr('shared: buffer_s', 'chia sẻ: buffer_s'), 'shared')}
      {(f.shared ?? new Array(N).fill(null)).map((v, i) => (
        <Cell key={'s' + i} x={ex(i)} y={SY} v={v} tier="shared" hot={f.hotShared?.has(i)} />
      ))}
      {f.barrier ? (
        <g>
          <rect x={ex(0)} y={SY + 32} width={ex(N - 1) + CW - 3 - ex(0)} height={18} rx="3" fill="var(--ink)" />
          <text x={(ex(0) + ex(N - 1) + CW) / 2} y={SY + 45} fontSize="11" textAnchor="middle" style={{ fill: 'var(--page)' }}>{f.barrier}</text>
        </g>
      ) : null}
      {label(RY + 12, tr('registers', 'thanh ghi'), 'reg')}
      {range(P).map((t) => (
        <g key={'r' + t}>
          <rect x={ex(t * F) - 4} y={RY} width={F * CW + 5} height={46} rx="4" fill="none" stroke="var(--reg)" strokeDasharray="3 2" />
          <text x={ex(t * F)} y={RY + 10} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>luồng {t}</>}>thread {t}</Translated></text>
          {range(F).map((k) => (
            <Cell key={k} x={ex(t * F + k)} y={RY + 15} v={f.regs?.[t]?.[k] ?? null} tier="reg" hot={f.hotReg?.has(`${t}-${k}`)} />
          ))}
          {f.adds?.[t] ? (
            <text x={ex(t * F) + F * CW - 2} y={RY - 4} fontSize="11.5" fontWeight="700" textAnchor="end" style={{ fill: 'var(--reg)' }}>{f.adds[t]}</text>
          ) : null}
        </g>
      ))}
      {f.strip ? (
        <g>
          <text x={8} y={BY + 17} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--shared)' }}><Translated vi="scan khối">block scan</Translated></text>
          <text x={8} y={BY + 32} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{f.strip.label}</text>
          {f.strip.values.map((v, t) => (
            <Cell key={'b' + t} x={ex(t * F + F - 1)} y={BY} v={v} tier="shared" />
          ))}
        </g>
      ) : null}
      {label(OY, tr('global (output)', 'toàn cục (đầu ra)'), 'global')}
      {(f.out ?? new Array(N).fill(null)).map((v, i) => (
        <Cell key={'o' + i} x={ex(i)} y={OY} v={v} tier="global" hot={f.hotOut?.has(i)} />
      ))}
      {f.tags
        ? f.tags.text.map((t, i) => (
            <text key={'t' + i} x={ex(i) + (CW - 3) / 2} y={tagY(f.tags!.row)} fontSize="9.5" textAnchor="middle"
              style={{ fill: f.tags!.bad ? 'var(--fail)' : 'var(--ink-2)' }}>{t}</text>
          ))
        : null}
      {f.arrows?.map((a, k) => (
        <line key={'a' + k} x1={a.x1} y1={a.y1} x2={a.x2} y2={a.y2} stroke={a.bad ? 'var(--fail)' : 'var(--ink-2)'} strokeWidth="1.4" strokeDasharray={a.bad ? '3 2' : undefined} />
      ))}
      {f.headline ? <text x={8} y={VH - 6} fontSize="13" fontWeight="600" style={{ fill: 'var(--ink)' }}>{f.headline}</text> : null}
      {f.note ? <text x={VW - 8} y={VH - 6} fontSize="11.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{f.note}</text> : null}
    </svg>
  );
}

const centre = (i: number) => ex(i) + (CW - 3) / 2;
const loaded = (upto: number) => d.values.map((v, i) => (Math.floor(i / P) < upto ? v : null));
const allShared = d.values.slice();
const finalRuns = d.phases.runs.map((r, t) => r.map((v) => v + d.phases.added[t]));

const beats: Beat[] = [
  {
    clip: 'coarse-phases-1',
    get label() {
      return tr('The block', 'Khối');
    },
    draw: (p) => (
      <Stage
        hotGlobal={new Set(range(F, F * Math.min(P - 1, Math.floor(p * P * 1.2))))}
        tags={{ row: 'global', text: d.values.map((_, i) => `t${Math.floor(i / F)}`) }}
        headline={tr(`${P} threads, ${N} elements, ${F} each`, `${P} luồng, ${N} phần tử, mỗi luồng ${F}`)}
        note={tr('each thread owns a contiguous run', 'mỗi luồng sở hữu một đoạn liên tiếp')}
      />
    ),
  },
  {
    clip: 'coarse-phases-2',
    get label() {
      return tr('Load', 'Nạp');
    },
    draw: (p) => {
      if (p > 0.78) {
        const c = 0;
        const direct = d.direct[c].elements;
        return (
          <Stage shared={allShared} badGlobal={new Set(direct)}
            tags={{ row: 'global', text: d.values.map((_, i) => (direct.includes(i) ? `t${direct.indexOf(i)}` : '')), bad: true }}
            arrows={direct.map((e) => ({ x1: centre(e), y1: GY + 42, x2: centre(e), y2: SY - 2, bad: true }))}
            headline={tr(`own runs instead: one instruction, ${d.direct[c].runs} separate pieces`, `nếu nạp thẳng đoạn của mình: một lệnh, ${d.direct[c].runs} mảnh rời nhau`)} note={tr('the strided alternative', 'phương án theo bước nhảy')} />
        );
      }
      const c = Math.min(F - 1, Math.floor(ramp(p, 0.05, 0.75) * F));
      const els = d.staged[c].elements;
      return (
        <Stage shared={loaded(c + 1)} hotGlobal={new Set(els)} hotShared={new Set(els)}
          tags={{ row: 'global', text: d.values.map((_, i) => `t${i % P}`) }}
          arrows={els.map((e) => ({ x1: centre(e), y1: GY + 42, x2: centre(e), y2: SY - 2 }))}
          headline={tr(`load instruction ${c + 1} of ${F}: thread t takes element t of chunk ${c}`, `lệnh nạp ${c + 1} trên ${F}: luồng t lấy phần tử t của khúc ${c}`)}
          note={tr(`${d.staged[c].runs} contiguous piece`, `${d.staged[c].runs} mảnh liên tiếp`)} />
      );
    },
  },
  {
    clip: 'coarse-phases-3',
    get label() {
      return tr('Phase 1', 'Pha 1');
    },
    draw: (p) => {
      // Fig. 11.13, lines 13 to 16: element k of each run is read from shared memory and added
      // to the running sum in buffer_r[k - 1], all threads at once.
      const k = p < 0.1 ? 0 : Math.min(F, Math.floor(ramp(p, 0.1, 0.9) * F) + 1);
      const regs = d.phases.runs.map((r) => r.map((v, j) => (j < k ? v : null)));
      return (
        <Stage shared={allShared} regs={regs} barrier={p < 0.1 ? tr('barrier: the whole segment is in shared memory', 'rào chắn: cả đoạn đã nằm trong bộ nhớ chia sẻ') : null}
          hotShared={k > 0 && k <= F ? new Set(range(P).map((t) => t * F + k - 1)) : undefined}
          hotReg={k > 0 ? new Set(range(P).map((t) => `${t}-${k - 1}`)) : undefined}
          arrows={k > 0 ? range(P).map((t) => ({ x1: centre(t * F + k - 1), y1: SY + 50, x2: centre(t * F + k - 1), y2: RY + 13 })) : undefined}
          tags={{ row: 'shared', text: d.values.map((_, i) => `t${Math.floor(i / F)}`) }}
          headline={tr('phase 1: each thread scans its own run, in registers', 'pha 1: mỗi luồng scan đoạn của riêng nó, trong thanh ghi')} note={tr('no barrier, nothing shared', 'không rào chắn, không chia sẻ gì')} />
      );
    },
  },
  {
    clip: 'coarse-phases-4',
    get label() {
      return tr('Phase 2', 'Pha 2');
    },
    draw: (p) => {
      const step = Math.min(ksRows.length - 1, Math.floor(ramp(p, 0.3, 0.9) * ksRows.length));
      return (
        <Stage shared={allShared} regs={d.phases.runs} hotReg={new Set(range(P).map((t) => `${t}-${F - 1}`))}
          strip={{ values: ksRows[step], label: step === 0 ? tr('run totals', 'tổng các đoạn') : step === ksRows.length - 1 ? tr('scanned totals', 'tổng đã scan') : tr(`Kogge-Stone, stride ${ks.steps[step - 1].stride}`, `Kogge-Stone, bước nhảy ${ks.steps[step - 1].stride}`) }}
          arrows={p < 0.3 ? range(P).map((t) => ({ x1: centre(t * F + F - 1), y1: RY + 41, x2: centre(t * F + F - 1), y2: BY - 2 })) : undefined}
          headline={tr('phase 2: one block scan of the 4 run totals', 'pha 2: một lần scan khối trên 4 tổng đoạn')} note={step === ksRows.length - 1 ? `${d.phases.scannedTotals.join(', ')}` : ''} />
      );
    },
  },
  {
    clip: 'coarse-phases-5',
    get label() {
      return tr('Phase 3', 'Pha 3');
    },
    draw: (p) => {
      const added = ramp(p, 0.15, 0.6) >= 1;
      const back = ramp(p, 0.65, 0.95);
      return (
        <Stage shared={back > 0 ? d.phases.final.map((v, i) => (i % F < Math.ceil(back * F) ? v : allShared[i])) : allShared}
          regs={added ? finalRuns : d.phases.runs}
          hotReg={added && back === 0 ? new Set(range(P).flatMap((t) => range(F).map((k) => `${t}-${k}`)).filter((s) => !s.startsWith('0-'))) : undefined}
          hotShared={back > 0 && back < 1 ? new Set(range(N).filter((i) => i % F === Math.ceil(back * F) - 1)) : undefined}
          strip={{ values: d.phases.scannedTotals, label: tr('scanned totals', 'tổng đã scan') }}
          adds={d.phases.added.map((a, t) => (t === 0 ? '+0' : `+${a}`))}
          arrows={p > 0.1 && p < 0.6 ? range(P - 1).map((t) => ({ x1: centre(t * F + F - 1), y1: BY, x2: centre((t + 1) * F + 1), y2: RY + 41 })) : undefined}
          headline={tr('phase 3: add the scanned total of the thread before', 'pha 3: cộng tổng đã scan của luồng đứng trước')} note={back > 0 ? tr('and write the run back', 'và ghi đoạn trở lại') : ''} />
      );
    },
  },
  {
    clip: 'coarse-phases-6',
    get label() {
      return tr('Store', 'Ghi');
    },
    draw: (p) => {
      const c = Math.min(F - 1, Math.floor(ramp(p, 0.05, 0.55) * F));
      const els = d.staged[c].elements;
      const done = p > 0.6;
      return (
        <Stage shared={d.phases.final} regs={finalRuns}
          out={d.phases.final.map((v, i) => (done || Math.floor(i / P) <= c ? v : null))}
          hotShared={done ? undefined : new Set(els)} hotOut={done ? undefined : new Set(els)}
          tags={done ? undefined : { row: 'out', text: d.phases.final.map((_, i) => (Math.floor(i / P) === c ? `t${i % P}` : '')) }}
          arrows={done ? undefined : els.map((e) => ({ x1: centre(e), y1: SY + 26, x2: centre(e), y2: OY - 2 }))}
          headline={done ? tr(`${d.perElement} shared accesses per element; ${d.work} additions, Kogge-Stone ${koggeStoneWork(N)}`, `${d.perElement} lần truy cập bộ nhớ chia sẻ mỗi phần tử; ${d.work} phép cộng, Kogge-Stone ${koggeStoneWork(N)}`) : tr(`store instruction ${c + 1} of ${F}, coalesced`, `lệnh ghi ${c + 1} trên ${F}, được hợp nhất`)}
          note={done ? tr('the scan of all 16', 'scan của cả 16') : ''} />
      );
    },
  },
];

export function CoarsePhasesScene() {
  return <Scene chapter="ch11" title={tr('Three phases of a coarsened scan', 'Ba pha của một scan đã gộp việc')} beats={beats} />;
}
