import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { DEMO_SEED, amdScalarEligible, distinctCount, filterTrace, replaySchedule, warpAddresses } from '../../sim/constcache';

// "One address, one warp" (section 7.4): six beats, one narration clip each
// (content/ch07/narration/constant-broadcast-N.txt). Every drawing is computed from
// sim/constcache.ts: the filter taps from filterTrace, the codebook's addresses and
// rounds from warpAddresses and replaySchedule with the fixed DEMO_SEED, so the
// count the narration says is the count drawn (pinned in constcache.test.ts).

// ---------------------------------------------------------------- the shared drawing
export type LaneState = 'idle' | 'asking' | 'served' | 'waiting';
export type EntryState = 'idle' | 'asked' | 'hot' | 'done';

/**
 * One warp above one table: a box per lane, a box per entry, a line from each lane to
 * the entry it reads. Used by the scene and by the section 7.4 explorer.
 */
export function WarpTable({
  addresses, tableSize, x = 20, y = 28, width = 680, gap = 120, lane = () => 'asking', entry = () => 'idle',
  lines = 1, hotLines, laneLabel, tableLabel, entryLabel, rounds,
}: {
  addresses: readonly number[];
  tableSize: number;
  x?: number;
  y?: number;
  width?: number;
  gap?: number;
  lane?: (i: number) => LaneState;
  entry?: (e: number) => EntryState;
  /** Opacity of the lane-to-entry lines. */
  lines?: number;
  /** Entries whose lines are drawn heavy (the one being answered). */
  hotLines?: Set<number>;
  laneLabel?: ReactNode;
  tableLabel?: ReactNode;
  entryLabel?: (e: number) => string | null;
  /** Round number shown over each asked entry. */
  rounds?: Map<number, number>;
}) {
  const lw = width / addresses.length;
  const ew = width / tableSize;
  const laneH = 18;
  const ty = y + laneH + gap;
  const laneFill: Record<LaneState, string> = { idle: 'var(--cell)', asking: 'var(--cell)', served: 'var(--const)', waiting: 'var(--cell)' };
  const entryFill: Record<EntryState, string> = { idle: 'var(--cell)', asked: 'var(--cell)', hot: 'var(--const)', done: 'var(--const-wash)' };
  return (
    <g>
      {laneLabel ? <text x={x} y={y - 7} fontSize="12" fontWeight="600">{laneLabel}</text> : null}
      {tableLabel ? <text x={x} y={ty + 24 + 16} fontSize="12" fontWeight="600" style={{ fill: 'var(--const)' }}>{tableLabel}</text> : null}
      {lines > 0
        ? addresses.map((a, i) => {
            const heavy = hotLines?.has(a);
            return (
              <line key={'l' + i} x1={x + i * lw + lw / 2} y1={y + laneH} x2={x + a * ew + ew / 2} y2={ty}
                stroke={heavy ? 'var(--const)' : 'var(--ink-3)'} strokeWidth={heavy ? 1.6 : 0.7}
                opacity={lines * (heavy || !hotLines ? 1 : 0.35)} />
            );
          })
        : null}
      {addresses.map((_, i) => {
        const s = lane(i);
        return (
          <rect key={'n' + i} data-lane={i} data-state={s} x={x + i * lw + Math.min(1, lw * 0.1)} y={y} width={Math.max(1, lw - Math.min(2, lw * 0.2))} height={laneH} rx="1.5"
            fill={laneFill[s]} stroke={s === 'idle' ? 'var(--rule)' : s === 'served' ? 'var(--const)' : 'var(--ink-2)'} strokeWidth={s === 'asking' ? 1.3 : 0.8}
            strokeDasharray={s === 'waiting' ? '2 2' : undefined} opacity={s === 'waiting' ? 0.55 : 1} />
        );
      })}
      {Array.from({ length: tableSize }, (_, e) => {
        const s = entry(e);
        const label = entryLabel?.(e);
        return (
          <g key={'e' + e} data-entry={e} data-state={s}>
            <rect x={x + e * ew + Math.min(0.5, ew * 0.1)} y={ty} width={Math.max(0.8, ew - Math.min(1, ew * 0.2))} height={24} rx="1.5"
              fill={entryFill[s]} stroke={s === 'idle' ? 'var(--rule)' : 'var(--const)'} strokeWidth={s === 'asked' ? 1.4 : 0.8} />
            {label && ew >= 16 ? (
              <text x={x + e * ew + ew / 2} y={ty + 16} fontSize={ew >= 30 ? 11 : 9.5} textAnchor="middle"
                style={{ fill: s === 'hot' ? 'var(--page)' : 'var(--ink-2)' }}>{label}</text>
            ) : null}
            {rounds?.has(e) && ew >= 12 ? (
              <text x={x + e * ew + ew / 2} y={ty - 4} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{rounds.get(e)}</text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

// ---------------------------------------------------------------- the scene's data, all from the model
const WARP = 32; // the book's warp; AMD's 64-wide wavefronts are in the explorer
const R = 2; // the 5x5 filter of the book's examples
const TAPS = filterTrace(R);
const M = 2 * R + 1;
const CODEBOOK = 16;
const CODES = warpAddresses('codebook', { warpSize: WARP, tableSize: CODEBOOK, seed: DEMO_SEED });
const ROUNDS = replaySchedule(CODES);
const DISTINCT = distinctCount(CODES);
const UNIFORM = warpAddresses('uniform', { warpSize: WARP, entry: 0 });

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const tapName = (t: number) => `F[${TAPS[t].fRow}][${TAPS[t].fCol}]`;

function Stage({ children, data, headline }: { children: ReactNode; data: Record<string, string | number>; headline?: ReactNode }) {
  const attrs = Object.fromEntries(Object.entries(data).map(([k, v]) => ['data-' + k, String(v)]));
  return (
    <svg viewBox="0 0 720 380" role="img" aria-label={tr('One warp reading a table in constant memory', 'Một warp đọc một bảng trong bộ nhớ hằng (constant memory)')} style={{ width: '100%', maxWidth: 900, minWidth: 520, display: 'block' }} {...attrs}>
      {children}
      {headline ? <text x={20} y={368} fontSize="13" fontWeight="600" data-headline>{headline}</text> : null}
    </svg>
  );
}

function Counter({ x, y, label, value, of }: { x: number; y: number; label: string; value: number; of?: number }) {
  return (
    <g data-counter={label}>
      <text x={x} y={y} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{label}</text>
      <text x={x} y={y + 24} fontSize="20" fontWeight="650" data-value>{value}</text>
      {of != null ? <text x={x + 14 + String(value).length * 11} y={y + 24} fontSize="12" style={{ fill: 'var(--ink-2)' }}>{tr('of', 'trên')} {of}</text> : null}
    </g>
  );
}

/** The rounds as a row of boxes, filled as they are answered. */
function RoundStrip({ x, y, total, done, width = 420 }: { x: number; y: number; total: number; done: number; width?: number }) {
  const w = Math.min(28, width / total);
  return (
    <g data-strip={total}>
      {Array.from({ length: total }, (_, i) => (
        <rect key={i} x={x + i * w} y={y} width={w - 3} height={16} rx="2"
          fill={i < done ? 'var(--const)' : 'var(--cell)'} stroke="var(--const)" strokeWidth="0.8" />
      ))}
    </g>
  );
}

// A small warp-and-table picture for the vendor panels.
function Mini({ x, y, addresses, tableSize, title, verdict, ok }: {
  x: number; y: number; addresses: readonly number[]; tableSize: number; title: string; verdict: string; ok: boolean | null;
}) {
  return (
    <g>
      <text x={x} y={y} fontSize="12" fontWeight="600">{title}</text>
      <WarpTable addresses={addresses} tableSize={tableSize} x={x} y={y + 22} width={300} gap={70}
        lane={() => 'asking'} entry={(e) => (addresses.includes(e) ? 'asked' : 'idle')} lines={0.9} />
      <text x={x} y={y + 160} fontSize="12" data-verdict={ok == null ? 'none' : ok ? 'yes' : 'no'}
        style={{ fill: ok == null ? 'var(--ink-2)' : ok ? 'var(--ok)' : 'var(--fail)' }}>{verdict}</text>
    </g>
  );
}

const beats: Beat[] = [
  {
    clip: 'constant-broadcast-1',
    get label() {
      return tr('One address', 'Một địa chỉ');
    },
    draw: (p) => {
      const lit = p >= 0.3;
      const served = p >= 0.55;
      return (
        <Stage data={{ beat: 1, distinct: distinctCount(UNIFORM), answers: served ? 1 : 0 }}
          headline={served ? tr('one address, one answer, all 32 lanes served', 'một địa chỉ, một câu trả lời, cả 32 lane được phục vụ') : tr('every lane asks for F[0][0]', 'mọi lane đều hỏi F[0][0]')}>
          <WarpTable addresses={UNIFORM} tableSize={M * M} lines={ramp(p, 0.05, 0.25)} hotLines={served ? new Set([0]) : undefined}
            lane={() => (served ? 'served' : 'asking')} entry={(e) => (e === 0 && lit ? 'hot' : 'idle')}
            laneLabel={tr('one warp: 32 lanes', 'một warp: 32 lane')} tableLabel={tr('F: the 5 × 5 filter in constant memory, row by row', 'F: bộ lọc 5 × 5 trong bộ nhớ hằng, từng hàng một')} entryLabel={(e) => String(e)} />
          <Counter x={20} y={250} label={tr('answers fetched', 'câu trả lời đã nạp')} value={served ? 1 : 0} />
          <Counter x={200} y={250} label={tr('distinct addresses', 'địa chỉ khác nhau')} value={lit ? 1 : 0} />
        </Stage>
      );
    },
  },
  {
    clip: 'constant-broadcast-2',
    get label() {
      return tr('Every tap', 'Mọi tap');
    },
    draw: (p) => {
      const k = Math.min(TAPS.length - 1, Math.floor(ramp(p, 0.04, 0.9) * TAPS.length));
      const addr = UNIFORM.map(() => TAPS[k].address);
      return (
        <Stage data={{ beat: 2, tap: k, distinct: distinctCount(addr), answers: k + 1 }}
          headline={k < TAPS.length - 1 ? tr(`tap ${k + 1} of ${TAPS.length}: ${tapName(k)}, still one address`, `tap ${k + 1} trên ${TAPS.length}: ${tapName(k)}, vẫn một địa chỉ`) : tr(`${TAPS.length} taps, ${TAPS.length} answers, whatever the warp's width`, `${TAPS.length} tap, ${TAPS.length} câu trả lời, dù warp rộng bao nhiêu`)}>
          <WarpTable addresses={addr} tableSize={M * M} hotLines={new Set([TAPS[k].address])}
            lane={() => 'served'} entry={(e) => (e === TAPS[k].address ? 'hot' : e < TAPS[k].address ? 'done' : 'idle')}
            laneLabel={tr('one warp: 32 lanes', 'một warp: 32 lane')} tableLabel={tr('F: the 5 × 5 filter in constant memory, row by row', 'F: bộ lọc 5 × 5 trong bộ nhớ hằng, từng hàng một')} entryLabel={(e) => String(e)} />
          <Counter x={20} y={250} label={tr('answers fetched', 'câu trả lời đã nạp')} value={k + 1} of={TAPS.length} />
          <Counter x={200} y={250} label={tr('distinct addresses this tap', 'địa chỉ khác nhau ở tap này')} value={1} />
          <RoundStrip x={20} y={310} total={TAPS.length} done={k + 1} width={640} />
        </Stage>
      );
    },
  },
  {
    clip: 'constant-broadcast-3',
    get label() {
      return tr('A codebook', 'Một codebook');
    },
    draw: (p) => {
      const asked = true; // the codes are there from the start; the rounds are what unfolds
      const done = Math.floor(ramp(p, 0.25, 0.95) * ROUNDS.length + 1e-9);
      const current = done < ROUNDS.length && p >= 0.25 ? ROUNDS[done] : null;
      const servedLanes = new Set(ROUNDS.slice(0, done).flatMap((r) => r.lanes));
      const doneEntries = new Set(ROUNDS.slice(0, done).map((r) => r.address));
      const order = new Map(ROUNDS.map((r, i) => [r.address, i + 1]));
      return (
        <Stage data={{ beat: 3, distinct: DISTINCT, rounds: done }}
          headline={
            done >= ROUNDS.length
              ? tr(`${DISTINCT} different entries: ${DISTINCT} rounds, one after another`, `${DISTINCT} mục khác nhau: ${DISTINCT} lượt, lần lượt từng lượt`)
              : current
                ? tr(`round ${done + 1}: entry ${current.address} for lane${current.lanes.length > 1 ? 's' : ''} ${current.lanes.join(', ')}`, `lượt ${done + 1}: mục ${current.address} cho lane ${current.lanes.join(', ')}`)
                : tr('each lane asks for the entry its own code names', 'mỗi lane hỏi mục mà mã của chính nó chỉ tới')
          }>
          <WarpTable addresses={CODES} tableSize={CODEBOOK} lines={1} hotLines={current ? new Set([current.address]) : undefined}
            lane={(i) => (servedLanes.has(i) ? 'served' : current?.lanes.includes(i) ? 'asking' : p >= 0.25 ? 'waiting' : 'asking')}
            entry={(e) => (current?.address === e ? 'hot' : doneEntries.has(e) ? 'done' : order.has(e) && asked ? 'asked' : 'idle')}
            rounds={asked ? order : undefined}
            laneLabel={tr('one warp: 32 lanes, each with its own 4-bit weight code', 'một warp: 32 lane, mỗi lane có mã trọng số 4-bit riêng')} tableLabel={tr('a 16-entry codebook in constant memory', 'một codebook 16 mục trong bộ nhớ hằng')} entryLabel={(e) => String(e)} />
          <Counter x={20} y={250} label={tr('rounds answered', 'lượt đã trả lời')} value={done} of={DISTINCT} />
          <Counter x={200} y={250} label={tr('distinct addresses', 'địa chỉ khác nhau')} value={asked ? DISTINCT : 0} />
          <RoundStrip x={20} y={310} total={DISTINCT} done={done} />
        </Stage>
      );
    },
  },
  {
    clip: 'constant-broadcast-4',
    get label() {
      return tr('AMD', 'AMD');
    },
    draw: (p) => (
      <Stage data={{ beat: 4, 'amd-filter': amdScalarEligible(UNIFORM) ? 'yes' : 'no', 'amd-codebook': amdScalarEligible(CODES) ? 'yes' : 'no' }}
        headline={tr('AMD: the scalar cache serves uniform reads only, and AMD gives no size', 'AMD: bộ nhớ đệm vô hướng chỉ phục vụ các lần đọc đồng nhất, và AMD không cho biết kích thước')}>
        <g>
          <Mini x={20} y={30} addresses={UNIFORM} tableSize={M * M} title={tr('filter tap: 1 address', 'tap của bộ lọc: 1 địa chỉ')}
            verdict={amdScalarEligible(UNIFORM) ? tr('uniform: served by the scalar cache (sL1D)', 'đồng nhất: được bộ nhớ đệm vô hướng (sL1D) phục vụ') : tr('not uniform', 'không đồng nhất')} ok={amdScalarEligible(UNIFORM)} />
        </g>
        <g opacity={ramp(p, 0.5, 0.62)}>
          <Mini x={380} y={30} addresses={CODES} tableSize={CODEBOOK} title={tr(`codebook lookup: ${DISTINCT} addresses`, `tra codebook: ${DISTINCT} địa chỉ`)}
            verdict={amdScalarEligible(CODES) ? tr('uniform', 'đồng nhất') : tr('not uniform: the accesses are serialised', 'không đồng nhất: các truy cập bị tuần tự hóa')} ok={amdScalarEligible(CODES)} />
        </g>
        <g opacity={ramp(p, 0.8, 0.9)}>
          <text x={20} y={250} fontSize="12"><Translated vi="Kích thước bộ nhớ __constant__ trên AMD: không được ghi trong tài liệu.">Size of __constant__ memory on AMD: not documented.</Translated></text>
          <text x={20} y={270} fontSize="12" style={{ fill: 'var(--ink-2)' }}><Translated vi="totalConstMem thay vào đó báo cáo lần cấp phát lớn nhất (mã nguồn CLR).">totalConstMem reports the largest allocation instead (CLR source).</Translated></text>
        </g>
      </Stage>
    ),
  },
  {
    clip: 'constant-broadcast-5',
    get label() {
      return tr('Apple', 'Apple');
    },
    draw: (p) => (
      <Stage data={{ beat: 5 }} headline={tr('Apple: no documented rule to model, so the lab measures', 'Apple: không có quy tắc nào được ghi trong tài liệu để mô hình hóa, nên lab sẽ đo')}>
        <g>
          <rect x={20} y={24} width={680} height={196} rx="4" fill="var(--page)" stroke="var(--rule)" />
          <text x={36} y={50} fontSize="12.5" fontWeight="600"><Translated vi="Những gì Apple ghi trong tài liệu về không gian địa chỉ constant">What Apple documents about the constant address space</Translated></text>
          <text x={36} y={78} fontSize="12"><Translated vi="Đặc tả MSL: bộ nhớ buffer chỉ-đọc lấy từ vùng bộ nhớ của thiết bị.">MSL specification: read-only buffer memory from the device memory pool.</Translated></text>
          <text x={36} y={102} fontSize="12" opacity={ramp(p, 0.15, 0.3)}><Translated vi={<>Bài nói WWDC 2016: &ldquo;tối ưu cho các trường hợp có nhiều tái sử dụng dữ liệu&rdquo;;</>}>WWDC 2016 talk: &ldquo;optimized for cases with a lot of data reuse&rdquo;;</Translated></text>
          <text x={36} y={120} fontSize="12" opacity={ramp(p, 0.15, 0.3)}><Translated vi="dữ liệu có giới hạn tĩnh được truyền theo tham chiếu có thể được nạp sẵn vào thanh ghi.">statically bounded data passed by reference can be preloaded into registers.</Translated></text>
          <text x={36} y={150} fontSize="12" opacity={ramp(p, 0.4, 0.55)} style={{ fill: 'var(--ink-2)' }}><Translated vi="Không có trong tài liệu: một bộ nhớ đệm hằng, kích thước của nó, một quy tắc về các lane cùng đọc một địa chỉ.">Not documented: a constant cache, its size, a rule about lanes reading one address.</Translated></text>
          <text x={36} y={190} fontSize="12" opacity={ramp(p, 0.7, 0.85)}><Translated vi="Lab 7.1 đo thời gian constant float* so với một struct setBytes trên máy Mac của bạn.">Lab 7.1 times constant float* against a setBytes struct on your Mac.</Translated></text>
        </g>
      </Stage>
    ),
  },
  {
    clip: 'constant-broadcast-6',
    get label() {
      return tr('Side by side', 'Đặt cạnh nhau');
    },
    draw: (p) => {
      const g = ramp(p, 0.1, 0.5);
      return (
        <Stage data={{ beat: 6, 'filter-rounds': distinctCount(UNIFORM), 'codebook-rounds': DISTINCT }}
          headline={tr('best case and near-worst case for the same cache', 'trường hợp tốt nhất và gần tệ nhất cho cùng một bộ nhớ đệm')}>
          <Mini x={20} y={30} addresses={UNIFORM} tableSize={M * M} title={tr('filter tap (every lane, one address)', 'tap của bộ lọc (mọi lane, một địa chỉ)')} verdict={tr('1 round per read', '1 lượt mỗi lần đọc')} ok={null} />
          <Mini x={380} y={30} addresses={CODES} tableSize={CODEBOOK} title={tr('codebook lookup (one code per lane)', 'tra codebook (mỗi lane một mã)')} verdict={tr(`${DISTINCT} rounds per read`, `${DISTINCT} lượt mỗi lần đọc`)} ok={null} />
          <text x={20} y={232} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="số lần đọc bị tuần tự hóa, theo quy tắc NVIDIA ghi trong tài liệu">serialised reads, NVIDIA's documented rule</Translated></text>
          <rect x={20} y={242} width={Math.max(2, 20 * g)} height={16} fill="var(--const)" />
          <text x={28 + 20 * g} y={255} fontSize="11.5"><Translated vi="bộ lọc: 1">filter: 1</Translated></text>
          <rect x={20} y={266} width={Math.max(2, 20 * DISTINCT * g)} height={16} fill="var(--const)" />
          <text x={28 + 20 * DISTINCT * g} y={279} fontSize="11.5"><Translated vi={<>codebook: {DISTINCT}</>}>codebook: {DISTINCT}</Translated></text>
          <text x={20} y={320} fontSize="12" opacity={ramp(p, 0.6, 0.8)}><Translated vi="llama.cpp trên CUDA và HIP: các bảng lượng tử hóa là mảng static const __device__.">llama.cpp on CUDA and HIP: quantisation tables as static const __device__ arrays.</Translated></text>
        </Stage>
      );
    },
  },
];

export function ConstantBroadcastScene() {
  return <Scene chapter="ch07" title={tr('One address, one warp', 'Một địa chỉ, một warp')} beats={beats} />;
}
