import { Scene, type Beat } from '../../components/Scene';
import { system, trace, type Phase, type ThreadState, type Trace } from '../../sim/gauss';
import { Translated, tr } from '../../lib/i18n';

// "One thread per row": the book's parallel elimination as a timeline of thread states,
// every frame taken from src/sim/gauss.ts (trace) with exact fractions and IEEE specials.
// Scripts: content/appA/narration/gauss-threads-N.txt.

// ---------------------------------------------------------------- shared drawing (also used by the stepper in SA7)
export const STATE_LABEL: Record<ThreadState, string> = {
  get ready() { return tr('ready', 'sẵn sàng'); },
  get working() { return tr('working', 'đang làm'); },
  get skipping() { return tr('skips: no such variable', 'bỏ qua: không có biến này'); },
  get idle() { return tr('idle', 'rảnh'); },
  get waiting() { return tr('at the barrier', 'ở rào chắn'); },
  get searching() { return tr('pivot search', 'tìm phần tử chốt'); },
  get finished() { return tr('finished', 'đã xong'); },
};

const BAR = 6;
const CELL = 14;
export const timelineWidth = (phases: Phase[]) => phases.reduce((w, p) => w + (p.kind === 'barrier' ? BAR : CELL), 0);

/** Thread lanes over time: one cell per phase, a dark bar per barrier. */
export function Timeline({ phases, n, x, y, lane = 34 }: { phases: Phase[]; n: number; x: number; y: number; lane?: number }) {
  let cx = x;
  const cols = phases.map((p, i) => {
    const at = cx;
    cx += p.kind === 'barrier' ? BAR : CELL;
    if (p.kind === 'barrier')
      return <rect key={i} x={at + 1.5} y={y - 4} width={BAR - 3} height={n * lane + 2} rx="1" fill="var(--ink)" data-kind="barrier" />;
    return (
      <g key={i} data-kind={p.kind}>
        {p.threads.map((s, t) => (
          <rect key={t} x={at + 1} y={y + t * lane} width={CELL - 3} height={lane - 10} rx="2" data-state={s}
            fill={s === 'working' ? 'var(--ink)' : s === 'finished' ? 'var(--ok-wash)' : s === 'searching' ? 'var(--shared-wash)' : s === 'idle' ? 'var(--rule-soft)' : 'none'}
            stroke={s === 'finished' ? 'var(--ok)' : s === 'searching' ? 'var(--shared)' : s === 'skipping' ? 'var(--ink)' : 'var(--rule)'}
            strokeDasharray={s === 'skipping' ? '3 2' : undefined} />
        ))}
      </g>
    );
  });
  return <g>{cols}</g>;
}

const isBad = (v: number) => !Number.isFinite(v);
const MW = 58; // matrix cell pitch
const cellText = (t: string) => (t === '0' ? '·' : t.replace(/-/g, '−'));

/** The augmented matrix after a phase, one row per thread. */
function Matrix({ phase, x, y, vars, lane = 34 }: { phase: Phase; x: number; y: number; vars: string[]; lane?: number }) {
  const n = phase.m.length;
  const w = MW;
  return (
    <g>
      {[...vars, '='].map((v, c) => (
        <text key={c} x={x + c * w + w / 2} y={y - 8} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{v}</text>
      ))}
      {phase.m.map((row, t) =>
        row.map((cell, c) => {
          const bad = isBad(cell.v);
          return (
            <g key={`${t}-${c}`}>
              <rect x={x + c * w} y={y + t * lane} width={w - 4} height={lane - 10} rx="2"
                fill={bad ? 'var(--fail-wash)' : 'var(--cell)'} stroke={bad ? 'var(--fail)' : c === n ? 'var(--ink-3)' : 'var(--rule)'} />
              <text x={x + c * w + (w - 4) / 2} y={y + t * lane + (lane - 10) / 2 + 4.5} fontSize="12.5" textAnchor="middle"
                style={{ fill: bad ? 'var(--fail)' : 'var(--ink)' }}>{cellText(cell.text)}</text>
            </g>
          );
        }),
      )}
    </g>
  );
}

function opText(p: Phase, t: number): string {
  const s = p.threads[t];
  if (p.kind === 'divide' && p.divisors?.[t]) return '÷ ' + cellText(p.divisors[t]!.text).replace('·', '0');
  if (p.kind === 'subtract' && s === 'working') return tr(`− row ${p.col + 1}`, `− hàng ${p.col + 1}`);
  if (s === 'skipping') return tr('skips', 'bỏ qua');
  if (p.kind === 'pivot' && s === 'searching') {
    const j = p.pivot!.rows.indexOf(t);
    return `|${cellText(p.pivot!.abs[j].text).replace('·', '0')}|${p.pivot!.winner === t ? ' max' : ''}`;
  }
  if (p.kind === 'swap' && s === 'working') return tr('swap', 'hoán đổi');
  if (p.kind === 'backsub' && s === 'working') return tr('substitute', 'thế ngược');
  return '';
}

export function phaseTitle(p: Phase, vars: string[]): string {
  switch (p.kind) {
    case 'start':
      return tr('one thread per row, all rows in one block\'s shared memory', 'một luồng mỗi hàng, mọi hàng nằm trong bộ nhớ chia sẻ (shared memory) của một khối');
    case 'pivot':
      return tr(`pivot search for ${vars[p.col]}: max |coefficient| is in row ${p.pivot!.winner + 1}`, `tìm phần tử chốt cho ${vars[p.col]}: |hệ số| lớn nhất ở hàng ${p.pivot!.winner + 1}`);
    case 'swap':
      return tr(`swap: row ${p.swap![1] + 1} moves to the top`, `hoán đổi: hàng ${p.swap![1] + 1} lên trên cùng`);
    case 'divide':
      return tr(`divide: every remaining row by its own coefficient of ${vars[p.col]}`, `chia: mỗi hàng còn lại cho hệ số ${vars[p.col]} của chính nó`);
    case 'subtract':
      return tr(`subtract row ${p.col + 1}: ${vars[p.col]} leaves the rows below`, `trừ hàng ${p.col + 1}: ${vars[p.col]} rời khỏi các hàng bên dưới`);
    case 'barrier':
      return tr('barrier: every thread waits', 'rào chắn: mọi luồng chờ');
    case 'backsub':
      return tr(`back substitution: row ${p.col + 1} gives ${vars[p.col]} = ${cellText(p.m[p.col][p.m.length].text)}`, `thế ngược: hàng ${p.col + 1} cho ${vars[p.col]} = ${cellText(p.m[p.col][p.m.length].text)}`);
    case 'done':
      return tr('solution: ', 'nghiệm: ') + vars.map((v, i) => `${v} = ${cellText(p.m[i][p.m.length].text)}`).join(', ');
  }
}

// ---------------------------------------------------------------- the scene
const T1 = trace(system('book1'), { pivot: false, arith: 'exact' });
const T2 = trace(system('book2'), { pivot: false, arith: 'exact' });
const T3 = trace(system('book2'), { pivot: true, arith: 'exact' });
const VARS = ['X', 'Y', 'Z'];
const W = 720;
const H = 320;
const LANE = 40;
const TY = 74; // top of the lanes and the matrix rows
const TX = 96; // timeline start
const MX = 430; // matrix start
const at = (from: number, to: number, p: number) => Math.min(to, from + Math.floor(p * (to - from + 1)));

function Stage({ t, idx, beat, intro = 1 }: { t: Trace; idx: number; beat: number; intro?: number }) {
  const ph = t.phases[idx];
  const n = t.n;
  const flat = ph.m.flat();
  const nan = flat.filter((c) => Number.isNaN(c.v)).length;
  const inf = flat.filter((c) => Math.abs(c.v) === Infinity).length;
  const nanRows = ph.m.filter((r) => r.some((c) => Number.isNaN(c.v))).length;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Threads eliminating one row each', 'Các luồng, mỗi luồng khử một hàng')} style={{ width: '100%', maxWidth: W, display: 'block' }}
      data-beat={beat} data-system={t.sys.id} data-pivot={t.opts.pivot ? 1 : 0} data-index={idx} data-phase={ph.kind} data-nan={nan} data-inf={inf}
      data-rhs={ph.m.map((r) => r[n].text).join(',')}>
      <text x={20} y={24} fontSize="12" style={{ fill: 'var(--ink-2)' }}>
        {t.sys.id === 'book1' ? tr("the book's first system", 'hệ thứ nhất của sách') : tr("the book's second system", 'hệ thứ hai của sách')}{t.opts.pivot ? tr(', with pivoting', ', có chọn phần tử chốt') : tr(', no pivoting', ', không chọn phần tử chốt')}
      </text>
      <text x={TX} y={TY - 12} fontSize="11" style={{ fill: 'var(--ink-3)' }}>{tr('time →', 'thời gian →')}</text>
      {Array.from({ length: n }, (_, k) => (
        <g key={k} opacity={intro}>
          <text x={20} y={TY + k * LANE + 18} fontSize="12.5" fontWeight="600"><Translated vi={<>luồng {k}</>}>thread {k}</Translated></text>
          <text x={MX - 8} y={TY + k * LANE + 19} fontSize="11.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{opText(ph, k)}</text>
        </g>
      ))}
      <Timeline phases={t.phases.slice(0, idx + 1)} n={n} x={TX} y={TY} lane={LANE} />
      <Matrix phase={ph} x={MX} y={TY} vars={VARS} lane={LANE} />
      {ph.kind === 'barrier' ? (
        <g>
          <rect x={MX - 4} y={TY + n * LANE - 2} width={4 * MW} height={22} rx="3" fill="var(--ink)" />
          <text x={MX + 2 * MW - 4} y={TY + n * LANE + 13} fontSize="12" textAnchor="middle" style={{ fill: 'var(--page)' }}><Translated vi={<>rào chắn: cả {n} luồng chờ</>}>barrier: all {n} threads wait</Translated></text>
        </g>
      ) : null}
      <g transform={`translate(20 ${H - 70})`}>
        {(
          [
            [tr('working', 'đang làm'), 'var(--ink)', 'var(--rule)'],
            [tr('skips', 'bỏ qua'), 'none', 'var(--ink)'],
            [tr('idle', 'rảnh'), 'var(--rule-soft)', 'var(--rule)'],
            [tr('pivot search', 'tìm phần tử chốt'), 'var(--shared-wash)', 'var(--shared)'],
            [tr('finished', 'đã xong'), 'var(--ok-wash)', 'var(--ok)'],
          ] as const
        ).map(([l, f, s], i) => (
          <g key={l} transform={`translate(${i * 112} 0)`}>
            <rect width={12} height={12} rx="2" fill={f} stroke={s} strokeDasharray={l === tr('skips', 'bỏ qua') ? '3 2' : undefined} />
            <text x={17} y={10} fontSize="11">{l}</text>
          </g>
        ))}
        <rect x={5 * 112} width={4} height={14} fill="var(--ink)" />
        <text x={5 * 112 + 9} y={10} fontSize="11">{tr('barrier', 'rào chắn')}</text>
      </g>
      <text x={20} y={H - 18} fontSize="13.5" fontWeight="600" style={{ fill: nan || inf ? 'var(--fail)' : 'var(--ink)' }}>
        {nanRows === n
          ? tr(`NaN in all ${n} rows: the answer is lost, though the system has one`, `NaN ở cả ${n} hàng: đáp án bị mất, dù hệ có nghiệm`)
          : nan || inf
            ? tr('row 1 divided by 0: infinity, and 0 / 0 is NaN', 'hàng 1 chia cho 0: vô cực, và 0 / 0 là NaN')
            : phaseTitle(ph, VARS)}
      </text>
    </svg>
  );
}

const end1 = T1.phases.length - 1;
const end3 = T3.phases.length - 1;
const beats: Beat[] = [
  { clip: 'gauss-threads-1', get label() { return tr('One row each', 'Mỗi luồng một hàng'); }, draw: (p) => <Stage t={T1} idx={0} beat={1} intro={Math.min(1, 0.2 + p * 2)} /> },
  { clip: 'gauss-threads-2', get label() { return tr('Divide, barrier', 'Chia, rào chắn'); }, draw: (p) => <Stage t={T1} idx={at(1, 2, p)} beat={2} /> },
  { clip: 'gauss-threads-3', get label() { return tr('Subtract, barrier', 'Trừ, rào chắn'); }, draw: (p) => <Stage t={T1} idx={at(3, 4, p)} beat={3} /> },
  { clip: 'gauss-threads-4', get label() { return tr('The rest', 'Phần còn lại'); }, draw: (p) => <Stage t={T1} idx={at(5, end1, p)} beat={4} /> },
  { clip: 'gauss-threads-5', get label() { return tr('A zero pivot', 'Phần tử chốt bằng không'); }, draw: (p) => <Stage t={T2} idx={at(0, 4, p)} beat={5} /> },
  { clip: 'gauss-threads-6', get label() { return tr('Pivoting', 'Chọn phần tử chốt'); }, draw: (p) => <Stage t={T3} idx={at(1, end3, p)} beat={6} /> },
];

export function GaussScene() {
  return <Scene chapter="appA" title={tr('One thread per row', 'Một luồng mỗi hàng')} beats={beats} />;
}
