import { useMemo, type ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { SCENE, sceneRuns, type ChainFrame, type ChainPhase, type ChainRun } from '../../sim/inplace';
import { Translated, tr, useLocale } from '../../lib/i18n';

// "The scan chain orders the blocks" (section 12.7). Twelve keys of Fig. 12.10's pattern,
// three blocks of four, two resident slots, the hardware starting blockIdx 2, 1, 0. Six
// beats, one narration clip each (content/ch12/narration/scan-chain-N.txt). Every arrow,
// number, phase and outcome drawn comes from sim/inplace.ts (sceneRuns), pinned in
// inplace.test.ts ("chainTimeline: the scene's runs").
const RUNS = sceneRuns();
const OV = RUNS.overlap;
const N = SCENE.keep.length; // 12
const K = SCENE.perBlock; // 4
const NB = OV.blocks; // 3

const W = 720;
const H = 372;
const AX = 40; // array origin
const CW = 34; // array cell width
const ROW_BEFORE = 46;
const ROW_AFTER = 136;
const GX = 150; // Gantt origin
const GY = 196;
const GC = 30; // Gantt column width
const GR = 34; // Gantt row height

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const cx = (slot: number) => AX + slot * CW + CW / 2;

const PHASE_SHORT: Record<ChainPhase, string> = {
  queued: '',
  get index() { return tr('take', 'lấy'); },
  get read() { return tr('read', 'đọc'); },
  get scan() { return tr('count', 'đếm'); },
  get wait() { return tr('wait', 'chờ'); },
  get publish() { return tr('flag', 'cờ'); },
  get write() { return tr('write', 'ghi'); },
  done: '',
};

/** A row of the array: labels in each slot, with block ranges above. */
function ArrayRow({ y, values, label, keepMarks, read, brackets, numbered, labelBelow, dimFrom }: {
  y: number; values: (number | null)[]; label: string; keepMarks?: boolean; read?: boolean[]; brackets?: boolean; numbered?: boolean;
  labelBelow?: boolean; dimFrom?: number;
}) {
  return (
    <g>
      <text x={AX} y={labelBelow ? y + 44 : y - (brackets ? 24 : 6)} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>{label}</text>
      {brackets
        ? Array.from({ length: NB }, (_, b) => (
            <g key={b}>
              <line x1={AX + b * K * CW + 2} x2={AX + (b + 1) * K * CW - 4} y1={y - 6} y2={y - 6} stroke="var(--ink-2)" />
              <text x={AX + b * K * CW + (K * CW) / 2} y={y - 10} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                {numbered ? tr(`number ${b}'s range`, `vùng của số ${b}`) : tr(`block ${b}`, `khối ${b}`)}
              </text>
            </g>
          ))
        : null}
      {values.map((v, s) => {
        const kept = keepMarks && v != null && SCENE.keep[v];
        return (
          <g key={s} data-slot={s} data-value={v == null ? '' : v} opacity={dimFrom != null && s >= dimFrom ? 0.4 : 1}>
            <rect x={AX + s * CW} y={y} width={CW - 3} height={26} rx="2"
              fill={kept ? 'var(--global-wash)' : 'var(--cell)'} stroke={kept ? 'var(--global)' : 'var(--rule)'} />
            <text x={cx(s) - 1} y={y + 17} fontSize="11" textAnchor="middle" style={{ fill: v == null ? 'var(--ink-3)' : 'var(--ink)' }}>
              {v == null ? '·' : `k${v}`}
            </text>
            {read?.[s] ? <circle cx={cx(s) - 1} cy={y + 32} r="2.6" style={{ fill: 'var(--ink-2)' }} /> : null}
          </g>
        );
      })}
    </g>
  );
}

/** Beat 1: every write of the in-place filter, red where it lands in an earlier block's range. */
function Arrows({ show }: { show: number }) {
  return (
    <g>
      <defs>
        <marker id="sc-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute */}
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink-2)' }} />
        </marker>
        <marker id="sc-arrow-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute */}
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--fail)' }} />
        </marker>
      </defs>
      {OV.writes.map((w, k) => {
        const on = clamp(show * OV.writes.length - k);
        if (on <= 0) return null;
        const bad = w.kind === 'earlier';
        return (
          <line key={w.key} data-arrow={w.key} data-kind={w.kind} opacity={on}
            x1={cx(w.from) - 1} y1={ROW_BEFORE + 28} x2={cx(w.to) - 1} y2={ROW_AFTER - 4}
            stroke={bad ? 'var(--fail)' : 'var(--ink-2)'} strokeWidth={bad ? 2.2 : 1.4}
            markerEnd={bad ? 'url(#sc-arrow-bad)' : 'url(#sc-arrow)'} />
        );
      })}
    </g>
  );
}

/** The blocks over time: one row per blockIdx, one column per step, up to frame `upto`. */
function Gantt({ run, upto, x0 = GX, y0 = GY, col = GC, title, showFlags = true, stuckNote = true, labels = true, compact = false }: {
  run: ChainRun; upto: number; x0?: number; y0?: number; col?: number; title: string; showFlags?: boolean; stuckNote?: boolean; labels?: boolean; compact?: boolean;
}) {
  const frames = run.frames.slice(1, upto + 1);
  const cur: ChainFrame = run.frames[upto];
  const last = run.frames[run.frames.length - 1];
  const stuck = run.outcome === 'starved' && upto === run.frames.length - 1;
  return (
    <g data-gantt={title}>
      <text x={x0 - 110} y={y0 - 12} fontSize="11.5" fontWeight="600">{title}</text>
      {Array.from({ length: NB }, (_, hw) => {
        const b = cur.blocks[hw];
        const y = y0 + hw * GR;
        return (
          <g key={hw} data-hw={hw} data-num={b.num ?? ''} data-phase={b.phase}>
            <text x={x0 - 110} y={y + 17} fontSize="11">blockIdx {hw}</text>
            <text x={x0 - 44} y={y + 17} fontSize="11" fontWeight="600" style={{ fill: b.num == null ? 'var(--ink-3)' : 'var(--reg)' }}>
              {b.num == null ? '·' : `#${b.num}`}
            </text>
            {frames.map((f, k) => {
              const ph = f.blocks[hw].phase;
              if (ph === 'queued' || ph === 'done') return null;
              const isWait = ph === 'wait';
              const isWrite = ph === 'write';
              return (
                <g key={k}>
                  <rect x={x0 + k * col} y={y + 2} width={col - 3} height={GR - 8} rx="2"
                    fill={isWait ? 'var(--fail-wash)' : isWrite ? 'var(--global-wash)' : 'var(--reg-wash)'}
                    stroke={isWait ? 'var(--fail)' : isWrite ? 'var(--global)' : 'var(--reg)'} strokeWidth="0.8" />
                  {labels ? <text x={x0 + k * col + (col - 3) / 2} y={y + 18} fontSize="9.5" textAnchor="middle">{PHASE_SHORT[ph]}</text> : null}
                </g>
              );
            })}
            {stuck && stuckNote && b.phase === 'wait' ? (
              <text x={x0 + frames.length * col + 4} y={y + 18} fontSize="10.5" style={{ fill: 'var(--fail)' }}>{tr('spins on #', 'quay chờ #')}{b.waitsOn}</text>
            ) : null}
            {stuck && stuckNote && b.phase === 'queued' ? (
              <text x={x0 + frames.length * col + 4} y={y + 18} fontSize="10.5" style={{ fill: 'var(--fail)' }}>{tr('no slot: never starts', 'không có chỗ: không bao giờ bắt đầu')}</text>
            ) : null}
          </g>
        );
      })}
      {showFlags
        ? cur.flagAt.map((t, num) => {
            if (t == null || num + 1 >= NB) return null;
            const t2 = cur.flagAt[num + 1];
            const from = last.blocks.findIndex((b) => b.num === num);
            const to = last.blocks.findIndex((b) => b.num === num + 1);
            if (t2 == null || from < 0 || to < 0) return null;
            return (
              <line key={num} x1={x0 + (t - 1) * col + col / 2} y1={y0 + from * GR + (from < to ? GR - 6 : 2)}
                x2={x0 + (t2 - 1) * col + col / 2 - 2} y2={y0 + to * GR + (from < to ? 4 : GR - 6)}
                stroke="var(--ink)" strokeWidth="1.6" markerEnd="url(#sc-flag)" />
            );
          })
        : null}
      <text x={x0} y={y0 + NB * GR + 12} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
        {tr('step', 'bước')} {cur.step}
        {run === RUNS.dynamic ? tr(` · counter ${cur.counter}`, ` · bộ đếm ${cur.counter}`) : ''}
        {upto === run.frames.length - 1 ? (run.outcome === 'finished' ? tr(' · finished', ' · hoàn tất') : compact ? tr(' · starved', ' · bị bỏ đói') : tr(' · starved: nothing can change any more', ' · bị bỏ đói: không gì có thể thay đổi nữa')) : ''}
      </text>
    </g>
  );
}

function Slots({ frame, x = 520, y = 196 }: { frame: ChainFrame; x?: number; y?: number }) {
  return (
    <g data-part="slots">
      <text x={x} y={y - 12} fontSize="11.5" fontWeight="600">{tr('resident slots', 'chỗ thường trú')} ({SCENE.slots})</text>
      {Array.from({ length: SCENE.slots }, (_, s) => {
        const b = frame.blocks.find((x) => x.slot === s);
        return (
          <g key={s} data-slot-holder={b ? b.hw : ''}>
            <rect x={x} y={y + s * 34} width={170} height={26} rx="3" fill="var(--cell)" stroke="var(--ink-2)" />
            <text x={x + 8} y={y + s * 34 + 17} fontSize="11">
              {b ? tr(`blockIdx ${b.hw} as #${b.num}`, `blockIdx ${b.hw} là #${b.num}`) : tr('free', 'trống')}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function Stage({ children, headline }: { children: ReactNode; headline?: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Animation of three blocks filtering one array in place', 'Hoạt cảnh ba khối lọc tại chỗ một mảng')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <defs>
        <marker id="sc-flag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute */}
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink)' }} />
        </marker>
      </defs>
      {children}
      {headline ? <text x={AX} y={H - 8} fontSize="13" fontWeight="600" data-headline="">{headline}</text> : null}
    </svg>
  );
}

const frameAt = (run: ChainRun, a: number, b: number, p: number) => Math.min(b, a + Math.floor(p * (b - a + 1)));
const readBy = (f: ChainFrame) => {
  const out = Array(N).fill(false);
  for (const b of f.blocks)
    if (b.num != null && b.phase !== 'index' && b.phase !== 'queued')
      for (let s = b.num * K; s < (b.num + 1) * K; s++) out[s] = true;
  return out;
};

// RUNS is built once at import, so its event text is in the language of that moment; Running rebuilds
// the runs (cheap: twelve keys) when the language changes and reads the events from those.
function Running({ run, a, b, p, headline }: { run: ChainRun; a: number; b: number; p: number; headline: (f: ChainFrame, k: number) => ReactNode }) {
  const k = frameAt(run, a, b, p);
  const f = run.frames[k];
  const locale = useLocale();
  const live = useMemo(() => sceneRuns(), [locale]);
  return (
    <Stage headline={headline(f, k)}>
      <ArrayRow y={ROW_BEFORE} values={f.array} label={tr('the array, as it is now (a dot: its key has been read)', 'mảng, như hiện tại (dấu chấm: khóa của ô đó đã được đọc)')} keepMarks brackets numbered read={readBy(f)}
        dimFrom={k === run.frames.length - 1 && run.outcome === 'finished' ? OV.count : undefined} />
      {k === run.frames.length - 1 && run.outcome === 'finished' ? (
        <text x={AX + OV.count * CW + 2} y={ROW_BEFORE + 46} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{tr('left over: past the output', 'phần còn sót: nằm sau đầu ra')}</text>
      ) : null}
      <Gantt run={run} upto={k} title={run === RUNS.dynamic ? tr('numbers from a counter', 'số lấy từ một bộ đếm') : tr('numbers from blockIdx', 'số lấy từ blockIdx')} />
      <Slots frame={f} />
      <text x={AX} y={118} fontSize="11" data-events="" style={{ fill: 'var(--ink-2)' }}>
        {(run === RUNS.dynamic ? live.dynamic : live.blockIdx).frames[k].events.join('; ')}
      </text>
    </Stage>
  );
}

const D = RUNS.dynamic;
const B = RUNS.blockIdx;
const lastD = D.frames.length - 1;
const lastB = B.frames.length - 1;
const badWrites = OV.writes.filter((w) => w.kind === 'earlier').length;

const beats: Beat[] = [
  {
    clip: 'scan-chain-1',
    get label() { return tr('Who overwrites whom', 'Ai ghi đè lên ai'); },
    draw: (p) => (
      <Stage headline={p > 0.75 ? tr(`${badWrites} of ${OV.writes.length} writes land where an earlier block still has to read`, `${badWrites} trong ${OV.writes.length} lần ghi rơi vào chỗ một khối trước đó vẫn còn phải đọc`) : tr('one array, three blocks, filtered in place', 'một mảng, ba khối, lọc tại chỗ')}>
        <ArrayRow y={ROW_BEFORE} values={OV.before} label={tr('the array before', 'mảng trước khi lọc')} keepMarks brackets />
        <ArrayRow y={ROW_AFTER} values={OV.after.map((v, s) => (p > 0.6 && s < OV.count ? v : null))} label={tr('the same array after', 'cùng mảng đó sau khi lọc')} labelBelow />
        <Arrows show={ramp(p, 0.08, 0.6)} />
        <g opacity={ramp(p, 0.7, 0.9)} data-part="red-list">
          {OV.writes.filter((w) => w.kind === 'earlier').map((w, k) => (
            <text key={w.key} x={AX} y={212 + k * 20} fontSize="11.5">
              <Translated vi={<>k{w.key} (khối {w.writer}) → ô {w.to}, đè lên k{w.victim}: khối {w.reader} phải đọc k{w.victim} trước</>}>
                k{w.key} (block {w.writer}) → slot {w.to}, over k{w.victim}: block {w.reader} must read k{w.victim} first
              </Translated>
            </text>
          ))}
        </g>
        <g opacity={ramp(p, 0.62, 0.8)}>
          <text x={AX + N * CW + 14} y={ROW_BEFORE + 60} fontSize="11.5" style={{ fill: 'var(--fail)' }} data-bad={badWrites}>{tr('red: lands in an', 'đỏ: rơi vào vùng')}</text>
          <text x={AX + N * CW + 14} y={ROW_BEFORE + 75} fontSize="11.5" style={{ fill: 'var(--fail)' }}>{tr("earlier block's range", 'của một khối trước đó')}</text>
        </g>
      </Stage>
    ),
  },
  {
    clip: 'scan-chain-2',
    get label() { return tr('Numbers from a counter', 'Số lấy từ một bộ đếm'); },
    draw: (p) => (
      <Running run={D} a={0} b={1} p={p}
        headline={(f) => (f.step === 0 ? tr('nothing has started', 'chưa có gì bắt đầu') : tr('the first block to start gets number 0, whatever its blockIdx', 'khối bắt đầu đầu tiên nhận số 0, bất kể blockIdx của nó'))} />
    ),
  },
  {
    clip: 'scan-chain-3',
    get label() { return tr('Publish and wait', 'Công bố và chờ'); },
    draw: (p) => (
      <Running run={D} a={2} b={4} p={p}
        headline={(f) => (f.step < 4 ? tr('both read their keys and count the kept ones', 'cả hai đọc khóa của mình và đếm những khóa được giữ') : tr('#0 publishes with a release; #1 waits with an acquire', '#0 công bố bằng một release; #1 chờ bằng một acquire'))} />
    ),
  },
  {
    clip: 'scan-chain-4',
    get label() { return tr('Write after the flag', 'Ghi sau cờ'); },
    draw: (p) => (
      <Running run={D} a={5} b={lastD} p={p}
        headline={(f, k) => (k < lastD ? tr('each block writes only after the flag of the block before it', 'mỗi khối chỉ ghi sau cờ của khối đứng trước nó') : D.correct && D.hazards === 0 ? tr('finished: the stable filter, and no key overwritten before it was read', 'hoàn tất: bộ lọc ổn định, và không khóa nào bị ghi đè trước khi được đọc') : tr('finished', 'hoàn tất'))} />
    ),
  },
  {
    clip: 'scan-chain-5',
    get label() { return tr('Numbers from blockIdx', 'Số lấy từ blockIdx'); },
    draw: (p) => (
      <Running run={B} a={0} b={lastB} p={p}
        headline={(f, k) => (k < lastB ? tr('same start order, numbers from blockIdx', 'cùng thứ tự bắt đầu, số lấy từ blockIdx') : tr('starved: every resident block waits on one that cannot start', 'bị bỏ đói: mọi khối thường trú đều chờ một khối không thể bắt đầu'))} />
    ),
  },
  {
    clip: 'scan-chain-6',
    get label() { return tr('Each backend', 'Từng backend'); },
    draw: (p) => (
      <Stage headline={tr('the wait is safe only where a started block is promised to finish', 'việc chờ chỉ an toàn ở nơi một khối đã bắt đầu được hứa sẽ chạy xong')}>
        <Gantt run={D} upto={lastD} x0={150} y0={40} col={20} title={tr('numbers from a counter', 'số lấy từ một bộ đếm')} labels={false} compact />
        <Gantt run={B} upto={lastB} x0={500} y0={40} col={20} title={tr('numbers from blockIdx', 'số lấy từ blockIdx')} labels={false} showFlags={false} compact />
        <g opacity={ramp(p, 0.1, 0.3)} data-backend="cuda">
          <text x={AX} y={196} fontSize="12" fontWeight="600">CUDA</text>
          <text x={AX + 70} y={196} fontSize="11.5">{tr('the counter makes the wait legal: started blocks are promised progress', 'bộ đếm làm cho việc chờ hợp lệ: các khối đã bắt đầu được hứa sẽ tiến triển')}</text>
        </g>
        <g opacity={ramp(p, 0.35, 0.55)} data-backend="hip">
          <text x={AX} y={236} fontSize="12" fontWeight="600">HIP</text>
          <text x={AX + 70} y={236} fontSize="11.5">{tr('the same code; unverified, because AMD documents no such promise', 'cùng một mã; unverified, vì AMD không ghi nhận lời hứa nào như vậy trong tài liệu')}</text>
        </g>
        <g opacity={ramp(p, 0.6, 0.8)} data-backend="metal">
          <text x={AX} y={276} fontSize="12" fontWeight="600">Metal</text>
          <text x={AX + 70} y={276} fontSize="11.5">{tr('one threadgroup walks its range in order, with threadgroup barriers only', 'một threadgroup đi qua vùng của nó theo thứ tự, chỉ dùng rào chắn threadgroup')}</text>
        </g>
      </Stage>
    ),
  },
];

export function ScanChainScene() {
  return <Scene chapter="ch12" title={tr('The scan chain orders the blocks', 'Chuỗi scan sắp thứ tự các khối')} beats={beats} />;
}
