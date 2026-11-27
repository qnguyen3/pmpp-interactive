import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { KIND_LABEL, firstFailure, isDouble, run, stateAt, tileBytes, type Trace, type Variant } from '../../sim/doublebuf';
import { Translated, getLocale, tr as trText } from '../../lib/i18n';

// "Two buffers, one barrier": four threads running the book's exchange loop
// (section 6.7) under the in-order schedule. Every frame of beats 1 to 5 is a state
// of src/sim/doublebuf.ts; beat 6 draws chapter 5's tiled loop with the shared
// memory figures from the same file. Six beats, one narration clip each
// (content/ch06/narration/double-buffer-N.txt).
const P = 4;
const N = 2;
const traces: Record<string, Trace> = {};
// Cached per language: a trace carries its failure messages in the language active when it was run.
const traceOf = (v: Variant) => (traces[getLocale() + v] ??= run({ threads: P, iterations: N, variant: v }, 'in-order'));

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

const SX = 96; // first slot
const SW = 104; // slot pitch
const BW = 88; // slot and thread box width
const AY = 50; // buffer A row
const BY = 126; // buffer B row
const TY = 228; // thread row

/** Splits a sentence into lines of at most `n` characters. */
function wrap(text: string, n: number): string[] {
  const out: string[] = [];
  let line = '';
  for (const w of text.split(' ')) {
    if (line && (line + ' ' + w).length > n) {
      out.push(line);
      line = w;
    } else line = line ? line + ' ' + w : w;
  }
  if (line) out.push(line);
  return out;
}

const VERB: Record<string, string> = {
  get read() {
    return trText('read', 'đã đọc');
  },
  get write() {
    return trText('wrote', 'đã ghi');
  },
  get swap() {
    return trText('swapped', 'đã hoán đổi');
  },
  get barrierRead() {
    return trText('past barrier', 'qua rào chắn');
  },
  get barrierWrite() {
    return trText('past barrier', 'qua rào chắn');
  },
};

interface FrameProps {
  variant: Variant;
  n: number;
  showB?: number;
  headline?: ReactNode;
  sub?: ReactNode;
}

function Frame({ variant, n, showB, headline, sub }: FrameProps) {
  const tr = traceOf(variant);
  const st = stateAt(tr, n);
  const cur = n > 0 ? tr.steps[n - 1] : undefined;
  const dbl = isDouble(variant);
  const b = showB ?? (dbl ? 1 : 0);
  const waiting = st.threads.filter((t) => t.waiting).length;
  const bad = cur?.violation;
  const slotX = (s: number) => SX + s * SW;
  const rowY = (buf: number) => (buf === 0 ? AY : BY);

  const slot = (buf: 0 | 1, s: number) => {
    const sl = st.buffers[buf][s];
    const reading = cur?.read && cur.read.buf === buf && cur.read.slot === s;
    const writing = cur?.write && cur.write.buf === buf && cur.write.slot === s;
    const isBad = reading && bad;
    return (
      <g key={`${buf}-${s}`} data-slot={`${buf}-${s}`} data-value={isNaN(sl.value) ? '' : sl.value} data-version={sl.version}>
        <rect x={slotX(s)} y={rowY(buf)} width={BW} height={46} rx="3"
          fill={writing || (reading && !isBad) ? 'var(--shared)' : 'var(--shared-wash)'}
          stroke={isBad ? 'var(--fail)' : 'var(--shared)'} strokeWidth={isBad ? 3 : 1.2} />
        <text x={slotX(s) + BW / 2} y={rowY(buf) + 24} fontSize="17" textAnchor="middle" fontWeight="600"
          style={{ fill: writing || (reading && !isBad) ? 'var(--page)' : 'var(--ink)' }}>
          {isNaN(sl.value) ? '·' : sl.value}
        </text>
        <text x={slotX(s) + BW / 2} y={rowY(buf) + 39} fontSize="9.5" textAnchor="middle"
          style={{ fill: writing || (reading && !isBad) ? 'var(--page)' : 'var(--ink-2)' }}>
          {sl.version < 0 ? trText('empty', 'trống') : sl.version === 0 ? trText('start', 'ban đầu') : trText(`iteration ${sl.version - 1}`, `vòng lặp ${sl.version - 1}`)}
        </text>
      </g>
    );
  };

  // The current step's arrow, from the thread to the slot it touches.
  let arrow: ReactNode = null;
  if (cur && cur.thread >= 0 && (cur.read || cur.write)) {
    const tgt = cur.read ?? cur.write!;
    const x1 = slotX(cur.thread) + BW / 2;
    const x2 = slotX(tgt.slot) + BW / 2;
    const y2 = rowY(tgt.buf) + 46;
    arrow = (
      <g>
        <line x1={x1} y1={TY} x2={x2} y2={y2 + 4} stroke={bad ? 'var(--fail)' : 'var(--ink)'} strokeWidth="2" />
        <circle cx={x2} cy={y2 + 4} r="3.5" fill={bad ? 'var(--fail)' : 'var(--ink)'} />
      </g>
    );
  }

  return (
    <svg data-variant={variant} data-step={n} viewBox="0 0 600 350" role="img" aria-label={trText('Four threads exchanging values through shared memory', 'Bốn luồng trao đổi giá trị qua bộ nhớ chia sẻ')} style={{ width: '100%', maxWidth: 600, display: 'block' }}>
      <text x={24} y={26} fontSize="13" fontWeight="600" style={{ fill: 'var(--ink)' }}>{headline}</text>
      <text x={24} y={AY + 28} fontSize="12" fontWeight="600" style={{ fill: 'var(--shared)' }}>{dbl || b > 0 ? trText('buffer A', 'buffer A') : trText('buffer', 'buffer')}</text>
      {[0, 1, 2, 3].map((s) => slot(0, s))}
      {b > 0 ? (
        <g opacity={b}>
          <text x={24} y={BY + 28} fontSize="12" fontWeight="600" style={{ fill: 'var(--shared)' }}><Translated vi="buffer B">buffer B</Translated></text>
          {[0, 1, 2, 3].map((s) => slot(1, s))}
        </g>
      ) : null}
      {waiting > 0 ? (
        <g>
          <line x1={SX - 10} x2={SX + 3 * SW + BW + 10} y1={TY - 20} y2={TY - 20} stroke="var(--ink)" strokeWidth="2.5" strokeDasharray="7 4" />
          <text x={24} y={TY - 22} fontSize="12" fontWeight="600" style={{ fill: 'var(--ink)' }}><Translated vi="rào chắn">barrier</Translated></text>
          <text x={24} y={TY - 8} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>{waiting} trên {P} đã tới đây</>}>{waiting} of {P} here</Translated></text>
        </g>
      ) : null}
      {arrow}
      {st.threads.map((th, t) => {
        const isCur = cur?.thread === t;
        const doing = th.done ? trText('finished', 'đã xong') : th.waiting ? trText('waits', 'đang chờ') : th.last == null ? trText('not started', 'chưa bắt đầu') : VERB[th.last];
        const ink = { fill: isCur ? 'var(--page)' : 'var(--ink-2)' };
        return (
          <g key={t} data-thread={t} data-doing={doing} data-pointers={dbl ? `${th.inBuf}${th.outBuf}` : ''}>
            <rect x={slotX(t)} y={TY} width={BW} height={52} rx="3" fill={isCur ? 'var(--ink)' : 'var(--cell)'} stroke="var(--ink)" />
            <text x={slotX(t) + BW / 2} y={TY + 16} fontSize="12" textAnchor="middle" fontWeight="600" style={{ fill: isCur ? 'var(--page)' : 'var(--ink)' }}><Translated vi={<>luồng {t}</>}>thread {t}</Translated></text>
            <text x={slotX(t) + BW / 2} y={TY + 30} fontSize="10" textAnchor="middle" style={ink}>{th.last == null ? '' : trText(`iteration ${th.iter}`, `vòng lặp ${th.iter}`)}</text>
            <text x={slotX(t) + BW / 2} y={TY + 44} fontSize="10" textAnchor="middle" style={ink}>{doing}</text>
            {dbl ? (
              <text x={slotX(t) + BW / 2} y={TY + 66} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                <Translated vi={<>đọc {th.inBuf ? 'B' : 'A'}, ghi {th.outBuf ? 'B' : 'A'}</>}>
                  reads {th.inBuf ? 'B' : 'A'}, writes {th.outBuf ? 'B' : 'A'}
                </Translated>
              </text>
            ) : null}
          </g>
        );
      })}
      {bad ? (
        <text x={24} y={314} fontSize="12" style={{ fill: 'var(--fail)' }}>
          {wrap(`${KIND_LABEL[bad.kind]}: ${bad.message}.`, 86).map((l, k) => (
            <tspan key={k} x={24} dy={k ? 16 : 0} fontWeight={k ? 400 : 600}>{l}</tspan>
          ))}
        </text>
      ) : sub ? (
        <text x={24} y={320} fontSize="12.5" style={{ fill: 'var(--ink-2)' }}>{sub}</text>
      ) : null}
    </svg>
  );
}

/** Step index for position p, running through `upTo` steps by p = `end`, then holding. */
const stepAt = (p: number, upTo: number, end = 0.85) => Math.round(ramp(p, 0.04, end) * upTo);

const STAGES: { label: string; detail: string; tag: 'true' | 'false' | null }[] = [
  {
    get label() {
      return trText('load this phase\'s tiles', 'nạp các ô của giai đoạn này');
    },
    get detail() {
      return trText('write Mds, Nds', 'ghi Mds, Nds');
    },
    tag: null,
  },
  {
    get label() {
      return trText('__syncthreads()  barrier 1', '__syncthreads()  rào chắn 1');
    },
    get detail() {
      return trText('read-after-write: true', 'đọc sau ghi: phụ thuộc thật');
    },
    tag: 'true',
  },
  {
    get label() {
      return trText('dot product from the tiles', 'tích vô hướng từ các ô');
    },
    get detail() {
      return trText('read Mds, Nds', 'đọc Mds, Nds');
    },
    tag: null,
  },
  {
    get label() {
      return trText('__syncthreads()  barrier 2', '__syncthreads()  rào chắn 2');
    },
    get detail() {
      return trText('write-after-read: false', 'ghi sau đọc: phụ thuộc giả');
    },
    tag: 'false',
  },
];

function Tiled({ p }: { p: number }) {
  const gone = ramp(p, 0.3, 0.5);
  const bars = ramp(p, 0.55, 0.8);
  const one = tileBytes(32, 1);
  const two = tileBytes(32, 2);
  return (
    <svg viewBox="0 0 600 340" role="img" aria-label={trText("The tiled kernel's loop with its two barriers", 'Vòng lặp của kernel phân ô với hai rào chắn')} style={{ width: '100%', maxWidth: 600, display: 'block' }}>
      <text x={24} y={26} fontSize="13" fontWeight="600" style={{ fill: 'var(--ink)' }}><Translated vi="kernel phân ô của chương 5, một giai đoạn trong vòng lặp">chapter 5's tiled kernel, one phase of its loop</Translated></text>
      {STAGES.map((s, k) => {
        const fade = s.tag === 'false' ? 1 - gone * 0.8 : 1;
        return (
          <g key={k} opacity={fade}>
            <rect x={24} y={46 + k * 50} width={300} height={38} rx="3" fill={s.tag ? 'var(--ink)' : 'var(--shared-wash)'} stroke={s.tag ? 'var(--ink)' : 'var(--shared)'} />
            <text x={36} y={70 + k * 50} fontSize="12.5" style={{ fill: s.tag ? 'var(--page)' : 'var(--ink)' }}>{s.label}</text>
            <text x={336} y={70 + k * 50} fontSize="12" style={{ fill: 'var(--ink-2)' }}>{s.detail}</text>
            {s.tag === 'false' && gone > 0.5 ? (
              <line x1={20} x2={328} y1={65 + k * 50} y2={65 + k * 50} stroke="var(--fail)" strokeWidth="2.5" />
            ) : null}
          </g>
        );
      })}
      <g opacity={bars}>
        <text x={24} y={268} fontSize="12" style={{ fill: 'var(--ink)' }}><Translated vi="bộ nhớ cho ô ở T = 32: một cặp">tile memory at T = 32: one pair</Translated></text>
        <rect x={250} y={256} width={120 * bars} height={16} fill="var(--shared)" />
        <text x={256 + 120 * bars} y={268} fontSize="12" style={{ fill: 'var(--ink)' }}><Translated vi={<>{one.toLocaleString('en-US')} byte</>}>{one.toLocaleString('en-US')} bytes</Translated></text>
        <text x={24} y={298} fontSize="12" style={{ fill: 'var(--ink)' }}><Translated vi="hai cặp, một rào chắn">two pairs, one barrier</Translated></text>
        <rect x={250} y={286} width={240 * bars} height={16} fill="var(--shared)" />
        <text x={256 + 240 * bars} y={298} fontSize="12" style={{ fill: 'var(--ink)' }}><Translated vi={<>{two.toLocaleString('en-US')} byte</>}>{two.toLocaleString('en-US')} bytes</Translated></text>
      </g>
      {p > 0.82 ? (
        <text x={24} y={330} fontSize="12.5" fontWeight="600" style={{ fill: 'var(--ink)' }}><Translated vi="lab 6.5: bỏ rào chắn 2, giữ cho đúng, và để trình phát hiện race theo dõi">lab 6.5: remove barrier 2, keep it correct, and let the race detector watch</Translated></text>
      ) : null}
    </svg>
  );
}

const fail2 = firstFailure(traceOf('single-no-read-barrier'))!;
const fail3 = firstFailure(traceOf('single-no-write-barrier'))!;

const beats: Beat[] = [
  {
    clip: 'double-buffer-1',
    get label() {
      return trText('One buffer', 'Một buffer');
    },
    draw: (p) => (
      <Frame variant="single-both" n={stepAt(p, traceOf('single-both').steps.length, 0.95)}
        headline={trText("read the neighbour's slot, barrier, write your own slot, barrier", 'đọc ô nhớ của luồng bên cạnh, rào chắn, ghi ô nhớ của mình, rào chắn')}
        sub={trText("each thread passes its right-hand neighbour's value on, plus 10", 'mỗi luồng chuyển tiếp giá trị của luồng bên phải, cộng thêm 10')} />
    ),
  },
  {
    clip: 'double-buffer-2',
    get label() {
      return trText('No read barrier', 'Không có rào chắn sau đọc');
    },
    draw: (p) => (
      <Frame variant="single-no-read-barrier" n={stepAt(p, fail2, 0.7)} headline={trText('the barrier after the read removed', 'đã bỏ rào chắn sau lần đọc')} />
    ),
  },
  {
    clip: 'double-buffer-3',
    get label() {
      return trText('No write barrier', 'Không có rào chắn sau ghi');
    },
    draw: (p) => (
      <Frame variant="single-no-write-barrier" n={stepAt(p, fail3, 0.7)} headline={trText('the barrier after the read restored, the one after the write removed', 'đã khôi phục rào chắn sau lần đọc, bỏ rào chắn sau lần ghi')} />
    ),
  },
  {
    clip: 'double-buffer-4',
    get label() {
      return trText('Same slot', 'Cùng một ô nhớ');
    },
    draw: (p) => (
      <Frame variant="single-no-read-barrier" n={fail2} showB={ramp(p, 0.45, 0.75)}
        headline={p < 0.45 ? trText('iteration 0 reads slot 0, iteration 0 writes slot 0: the same place', 'vòng lặp 0 đọc ô nhớ 0, vòng lặp 0 ghi ô nhớ 0: cùng một chỗ') : trText('a second buffer for the writes', 'một buffer thứ hai cho các lần ghi')} />
    ),
  },
  {
    clip: 'double-buffer-5',
    get label() {
      return trText('Two buffers', 'Hai buffer');
    },
    draw: (p) => (
      <Frame variant="double" n={stepAt(p, traceOf('double').steps.length, 0.95)}
        headline={trText('two buffers: read one, write the other, one barrier, swap', 'hai buffer: đọc một, ghi cái kia, một rào chắn, hoán đổi')}
        sub={trText('a thread that runs ahead writes into the other buffer', 'một luồng chạy trước sẽ ghi vào buffer kia')} />
    ),
  },
  {
    clip: 'double-buffer-6',
    get label() {
      return trText('Back to tiling', 'Quay lại phân ô');
    },
    draw: (p) => <Tiled p={p} />,
  },
];

export function DoubleBufferScene() {
  return <Scene chapter="ch06" title={trText('Two buffers, one barrier', 'Hai buffer, một rào chắn')} beats={beats} />;
}
