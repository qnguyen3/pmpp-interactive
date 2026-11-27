import { useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { Code } from '../../components/Code';
import { Translated, tr } from '../../lib/i18n';
import { realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';

// ---------------------------------------------------------------- memory map
// Our own drawing of the device memory model (the book's is Fig. 5.1, p. 98):
// one grid, two blocks, two threads each. Laid out by where things physically
// are: on the chip, off the chip on the device, and on the host.
type Kind = 'thread' | 'host' | 'reg' | 'local' | 'shared' | 'global' | 'const';
interface Box {
  id: string;
  kind: Kind;
  b: number;
  t: number;
  x: number;
  y: number;
  w: number;
  h: number;
}
const VW = 560;
const VH = 512;
const BLOCK_X = [16, 288];
const BLOCK_W = 256;
const COL_W = 112;
const BAR_X = 16;
const BAR_W = 528;
const colX = (b: number, t: number) => BLOCK_X[b] + 12 + t * 120;
const PAIRS: [number, number][] = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
// Order matters twice: it is the tab order, and later boxes sit on top of earlier
// ones (the local slices are drawn inside global memory).
const BOXES: Box[] = [
  ...PAIRS.map(([b, t]): Box => ({ id: `t${b}${t}`, kind: 'thread', b, t, x: colX(b, t), y: 62, w: COL_W, h: 42 })),
  ...PAIRS.map(([b, t]): Box => ({ id: `r${b}${t}`, kind: 'reg', b, t, x: colX(b, t), y: 112, w: COL_W, h: 42 })),
  ...[0, 1].map((b): Box => ({ id: `s${b}`, kind: 'shared', b, t: 0, x: BLOCK_X[b] + 12, y: 170, w: BLOCK_W - 24, h: 44 })),
  { id: 'g', kind: 'global', b: 0, t: 0, x: BAR_X, y: 280, w: BAR_W, h: 98 },
  ...PAIRS.map(([b, t]): Box => ({ id: `l${b}${t}`, kind: 'local', b, t, x: colX(b, t), y: 318, w: COL_W, h: 46 })),
  { id: 'c', kind: 'const', b: 0, t: 0, x: BAR_X, y: 390, w: BAR_W, h: 42 },
  { id: 'h', kind: 'host', b: 0, t: 0, x: BAR_X, y: 462, w: BAR_W, h: 42 },
];
const isActor = (x: Box) => x.kind === 'thread' || x.kind === 'host';
const TIER_OF: Record<Exclude<Kind, 'thread' | 'host'>, Tier> = {
  reg: 'reg',
  local: 'global', // local memory is physically part of global memory
  shared: 'shared',
  global: 'global',
  const: 'const',
};

type Access = 'rw' | 'r' | 'none';
function access(actor: Box, mem: Box): Access {
  if (actor.kind === 'host') return mem.kind === 'global' || mem.kind === 'const' ? 'rw' : 'none';
  switch (mem.kind) {
    case 'reg':
    case 'local':
      return actor.b === mem.b && actor.t === mem.t ? 'rw' : 'none';
    case 'shared':
      return actor.b === mem.b ? 'rw' : 'none';
    case 'global':
      return 'rw';
    case 'const':
      return 'r';
    default:
      return 'none';
  }
}

type Look = 'plain' | 'selected' | 'idle' | Access;
function lookOf(x: Box, sel: Box | null): Look {
  if (!sel) return 'plain';
  if (x.id === sel.id) return 'selected';
  if (isActor(x) === isActor(sel)) return 'idle';
  return isActor(x) ? access(x, sel) : access(sel, x);
}

function boxName(x: Box): string {
  switch (x.kind) {
    case 'thread':
      return tr(`Thread ${x.t} of block ${x.b}`, `Luồng ${x.t} của khối ${x.b}`);
    case 'host':
      return tr('The host', 'Host');
    case 'reg':
      return tr(`Registers of thread ${x.t}, block ${x.b}`, `Thanh ghi của luồng ${x.t}, khối ${x.b}`);
    case 'local':
      return tr(`Local memory of thread ${x.t}, block ${x.b}`, `Bộ nhớ cục bộ của luồng ${x.t}, khối ${x.b}`);
    case 'shared':
      return tr(`Shared memory of block ${x.b}`, `Bộ nhớ chia sẻ của khối ${x.b}`);
    case 'global':
      return tr('Global memory', 'Bộ nhớ toàn cục');
    case 'const':
      return tr('Constant memory', 'Bộ nhớ hằng');
  }
}
const LABEL: Record<Kind, (x: Box) => string> = {
  thread: (x) => tr(`thread ${x.t}`, `luồng ${x.t}`),
  host: () => tr('host', 'host'),
  reg: () => tr('registers', 'thanh ghi'),
  local: () => tr('local memory', 'bộ nhớ cục bộ'),
  shared: (x) => tr(`shared memory of block ${x.b}`, `bộ nhớ chia sẻ của khối ${x.b}`),
  global: () => tr('global memory', 'bộ nhớ toàn cục'),
  const: () => tr('constant memory', 'bộ nhớ hằng'),
};
// Shown in a box while nothing is selected.
const RESTING: Record<Kind, string> = {
  thread: '',
  get host() {
    return tr('the CPU side of the program', 'phía CPU của chương trình');
  },
  get reg() {
    return tr('private', 'riêng');
  },
  get local() {
    return tr('private slice', 'lát riêng');
  },
  get shared() {
    return tr('one per block', 'mỗi khối một bản');
  },
  get global() {
    return tr('off-chip DRAM, one for everybody', 'DRAM ngoài chip, một bản cho tất cả');
  },
  get const() {
    return tr('kernels can only read it', 'kernel chỉ đọc được');
  },
};
function statusText(x: Box, look: Look): string {
  if (look === 'plain' || look === 'idle') return RESTING[x.kind];
  if (look === 'selected') return isActor(x) ? '' : tr('selected', 'đang chọn');
  if (look === 'none') return tr('no access', 'không truy cập được');
  if (isActor(x)) return look === 'rw' ? tr('reads, writes', 'đọc, ghi') : tr('reads only', 'chỉ đọc');
  return look === 'rw' ? tr('read and write', 'đọc và ghi') : tr('read only', 'chỉ đọc');
}

function MapDetail({ sel }: { sel: Box | null }) {
  if (!sel)
    return (
      <>
        <Translated
          vi={
            <>
              <p>
                <b>Chưa chọn gì.</b>
              </p>
              <p style={{ margin: 0 }}>
                Mỗi ô là một nút bấm. Chọn một luồng hoặc host để xem nó với tới được gì. Chọn một bộ nhớ để xem ai với tới được
                nó. Chọn lại cùng ô đó để bỏ chọn.
              </p>
            </>
          }
        >
          <p>
            <b>Nothing selected.</b>
          </p>
          <p style={{ margin: 0 }}>
            Every box is a button. Select a thread or the host to see what it can reach. Select a memory to see who can
            reach it. Select the same box again to clear.
          </p>
        </Translated>
      </>
    );
  const other = 1 - sel.t;
  const otherB = 1 - sel.b;
  const title = (
    <p>
      <b>{boxName(sel)}</b>
    </p>
  );
  switch (sel.kind) {
    case 'thread':
      return (
        <>
          {title}
          <Translated
            vi={
              <>
                <p style={{ marginBottom: '0.3em' }}>Đọc và ghi:</p>
                <ul>
                  <li>
                    <span className="chip reg">thanh ghi của chính nó</span>
                  </li>
                  <li>
                    <span className="chip global">bộ nhớ cục bộ của chính nó</span>, nằm bên trong bộ nhớ toàn cục
                  </li>
                  <li>
                    <span className="chip shared">bộ nhớ chia sẻ của khối {sel.b}</span>, cùng với luồng {other}
                  </li>
                  <li>
                    <span className="chip global">bộ nhớ toàn cục</span>, cùng với mọi luồng khác và host
                  </li>
                </ul>
                <p>
                  Đọc, không ghi được: <span className="chip const">bộ nhớ hằng</span>
                </p>
                <p style={{ margin: 0 }}>
                  Không với tới được: thanh ghi và bộ nhớ cục bộ của bất kỳ luồng nào khác, kể cả luồng {other} trong cùng khối;
                  bộ nhớ chia sẻ của khối {otherB}.
                </p>
              </>
            }
          >
            <p style={{ marginBottom: '0.3em' }}>Reads and writes:</p>
            <ul>
              <li>
                <span className="chip reg">its own registers</span>
              </li>
              <li>
                <span className="chip global">its own local memory</span>, which sits inside global memory
              </li>
              <li>
                <span className="chip shared">the shared memory of block {sel.b}</span>, together with thread {other}
              </li>
              <li>
                <span className="chip global">global memory</span>, together with every other thread and the host
              </li>
            </ul>
            <p>
              Reads, cannot write: <span className="chip const">constant memory</span>
            </p>
            <p style={{ margin: 0 }}>
              Cannot reach: the registers and local memory of any other thread, including thread {other} of its own
              block; the shared memory of block {otherB}.
            </p>
          </Translated>
        </>
      );
    case 'host':
      return (
        <>
          {title}
          <Translated
            vi={
              <>
                <p>
                  Đọc và ghi <span className="chip global">bộ nhớ toàn cục</span> và{' '}
                  <span className="chip const">bộ nhớ hằng</span>.
                </p>
                <p style={{ margin: 0 }}>
                  Không với tới được bất cứ thứ gì kernel giữ cho riêng mình: không thanh ghi, không bộ nhớ cục bộ, không bộ nhớ chia
                  sẻ. Kernel cũng không ghi được bộ nhớ hằng, nên mọi kết quả kernel tạo ra đều trở về host qua bộ nhớ toàn cục.
                </p>
              </>
            }
          >
            <p>
              Reads and writes <span className="chip global">global memory</span> and{' '}
              <span className="chip const">constant memory</span>.
            </p>
            <p style={{ margin: 0 }}>
              Cannot reach anything a kernel keeps to itself: no registers, no local memory, no shared memory. Kernels
              cannot write constant memory either, so every result a kernel produces comes back to the host through
              global memory.
            </p>
          </Translated>
        </>
      );
    case 'reg':
      return (
        <>
          {title}
          <Translated
            vi={
              <>
                <p>
                  Được đọc và ghi bởi đúng một luồng đó. Luồng {other} của cùng khối không thấy chúng, host cũng vậy.
                </p>
                <p style={{ margin: 0 }}>
                  Nằm trên chip, bên trong bộ xử lý, nên dùng chúng rất rẻ. Mỗi luồng có một bộ: bốn bộ trong hình này.
                </p>
              </>
            }
          >
            <p>
              Read and written by that one thread. Thread {other} of the same block cannot see them, and neither can the
              host.
            </p>
            <p style={{ margin: 0 }}>
              On the chip, inside the processor, which is why using them is cheap. There is one set per thread: four in
              this picture.
            </p>
          </Translated>
        </>
      );
    case 'local':
      return (
        <>
          {title}
          <Translated
            vi={
              <>
                <p>Riêng của đúng một luồng đó, y như thanh ghi của nó.</p>
                <p style={{ margin: 0 }}>
                  Nhưng hãy nhìn chỗ nó được vẽ: bên trong bộ nhớ toàn cục, ngoài chip, và chậm gần như vậy khi truy cập. Nó chứa
                  những gì luồng không giữ được trong thanh ghi: các mảng khai báo trong thân kernel, các thanh ghi bị tràn và phần
                  còn lại của ngăn xếp lời gọi.
                </p>
              </>
            }
          >
            <p>Private to that one thread, exactly like its registers.</p>
            <p style={{ margin: 0 }}>
              Look at where it is drawn, though: inside global memory, off the chip, and about as slow to reach. It holds
              what the thread could not keep in registers: arrays declared in the kernel body, spilled registers and the
              rest of the call stack.
            </p>
          </Translated>
        </>
      );
    case 'shared':
      return (
        <>
          {title}
          <Translated
            vi={
              <>
                <p>
                  Được đọc và ghi bởi cả hai luồng của khối {sel.b}, và không ai khác. Khối {otherB} có bản riêng; host không thấy
                  bản nào.
                </p>
                <p style={{ margin: 0 }}>
                  Nằm trên chip. Mỗi khối một bản: hai bản trong hình này. Đây là nơi duy nhất trên chip mà một luồng có thể để lại
                  một giá trị cho luồng khác.
                </p>
              </>
            }
          >
            <p>
              Read and written by both threads of block {sel.b}, and by nobody else. Block {otherB} has its own; the host
              sees neither.
            </p>
            <p style={{ margin: 0 }}>
              On the chip. One copy per block: two in this picture. It is the only on-chip place where one thread can
              leave a value for another.
            </p>
          </Translated>
        </>
      );
    case 'global':
      return (
        <>
          {title}
          <Translated
            vi={
              <>
                <p>Được đọc và ghi bởi mọi luồng của mọi khối, và bởi host.</p>
                <p style={{ margin: 0 }}>
                  DRAM ngoài chip: bộ nhớ có băng thông là chủ đề của phần 5.1. Một bản duy nhất, dù có bao nhiêu luồng. Bốn lát nét
                  đứt bên trong là bộ nhớ cục bộ của các luồng, mỗi lát đóng kín với tất cả trừ chủ của nó.
                </p>
              </>
            }
          >
            <p>Read and written by every thread of every block, and by the host.</p>
            <p style={{ margin: 0 }}>
              Off-chip DRAM: the memory whose bandwidth section 5.1 was about. One copy, however many threads there are.
              The four dashed slices inside it are the threads' local memories, each closed to everyone but its owner.
            </p>
          </Translated>
        </>
      );
    case 'const':
      return (
        <>
          {title}
          <Translated
            vi={
              <>
                <p>Mọi luồng đều đọc được. Không luồng nào ghi được. Host làm được cả hai.</p>
                <p style={{ margin: 0 }}>
                  Thiết bị phục vụ nó với độ trễ thấp và băng thông cao; sách giải thích cách làm ở chương 7.
                </p>
              </>
            }
          >
            <p>Every thread can read it. No thread can write it. The host can do both.</p>
            <p style={{ margin: 0 }}>
              The device serves it with short latency and high bandwidth; the book explains how in chapter 7.
            </p>
          </Translated>
        </>
      );
  }
}

function MemoryMap() {
  const [selId, setSelId] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const sel = BOXES.find((x) => x.id === selId) ?? null;

  const drawBox = (x: Box) => {
    const look = lookOf(x, sel);
    const status = statusText(x, look);
    const actor = isActor(x);
    const hot = hover === x.id && look !== 'selected';
    const dim = look === 'idle' ? 0.42 : 1;
    const off = look === 'none';
    const wide = x.w > 300;
    let fill: string;
    let stroke: string;
    let ink = 'var(--ink)';
    if (actor) {
      fill = look === 'selected' ? 'var(--ink)' : 'var(--page)';
      stroke = off ? 'var(--ink-3)' : 'var(--ink)';
      if (look === 'selected') ink = 'var(--page)';
    } else {
      const tier = TIER_OF[x.kind as keyof typeof TIER_OF];
      fill = off ? 'none' : `var(--${tier}-wash)`;
      stroke = `var(--${tier})`;
    }
    if (off) ink = 'var(--ink-3)';
    const sw = (look === 'selected' ? 3.5 : look === 'rw' || look === 'r' ? 2.5 : 1.5) + (hot ? 1 : 0);
    const dash = off ? '4 3' : x.kind === 'local' && look !== 'selected' ? '7 3' : undefined;
    const strong = look === 'rw' || look === 'r' || look === 'selected';
    return (
      <g key={x.id} data-id={x.id} data-look={look} opacity={dim}>
        {/* a page-coloured base so the wash reads the same on any background */}
        {!actor && x.kind !== 'local' ? <rect x={x.x} y={x.y} width={x.w} height={x.h} rx="4" fill="var(--page)" /> : null}
        <rect
          x={x.x}
          y={x.y}
          width={x.w}
          height={x.h}
          rx="4"
          fill={fill}
          stroke={stroke}
          strokeWidth={sw}
          strokeDasharray={dash}
          strokeOpacity={off ? 0.55 : 1}
        />
        {wide ? (
          <>
            <text x={x.x + 12} y={x.kind === 'global' ? x.y + 22 : x.y + x.h / 2 + 4.5} fontSize="13" fontWeight="600" style={{ fill: ink }}>
              {LABEL[x.kind](x)}
            </text>
            <text
              x={x.x + x.w - 12}
              y={x.kind === 'global' ? x.y + 22 : x.y + x.h / 2 + 4.5}
              fontSize="11.5"
              textAnchor="end"
              fontWeight={strong ? 600 : 400}
              style={{ fill: strong ? ink : 'var(--ink-2)' }}
            >
              {status}
            </text>
          </>
        ) : (
          <>
            <text x={x.x + 10} y={status ? x.y + 18 : x.y + x.h / 2 + 4.5} fontSize="12.5" fontWeight="600" style={{ fill: ink }}>
              {LABEL[x.kind](x)}
            </text>
            <text
              x={x.x + 10}
              y={x.y + 33}
              fontSize="11"
              fontWeight={strong ? 600 : 400}
              style={{ fill: strong ? ink : off ? 'var(--ink-3)' : 'var(--ink-2)' }}
            >
              {status}
            </text>
          </>
        )}
      </g>
    );
  };

  return (
    <Bench
      title={tr('Memory map', 'Bản đồ bộ nhớ')}
      task={tr(
        'Select a thread to see what it can reach, or a memory to see who can reach it.',
        'Chọn một luồng để xem nó với tới được gì, hoặc một bộ nhớ để xem ai với tới được nó.',
      )}
      foot={tr(
        'Boxes show who may read and write what. They are not to scale, and they say nothing about speed. Texture memory and caches are not drawn.',
        'Các ô cho thấy ai được đọc và ghi cái gì. Chúng không theo tỉ lệ, và không nói gì về tốc độ. Bộ nhớ texture và bộ nhớ đệm không được vẽ.',
      )}
    >
      <div className="cols plot">
        <div>
          <div style={{ position: 'relative', width: '100%', maxWidth: VW }}>
            <svg
              viewBox={`0 0 ${VW} ${VH}`}
              role="img"
              aria-label={tr(
                "Memory map of one grid: two blocks of two threads, each thread with registers, each block with shared memory; below them global memory containing each thread's local memory, constant memory, and the host.",
                'Bản đồ bộ nhớ của một lưới: hai khối, mỗi khối hai luồng, mỗi luồng có thanh ghi, mỗi khối có bộ nhớ chia sẻ; bên dưới là bộ nhớ toàn cục chứa bộ nhớ cục bộ của từng luồng, bộ nhớ hằng, và host.',
              )}
              style={{ width: '100%', display: 'block' }}
            >
              {/* the grid and its two blocks */}
              <rect x="6" y="8" width="548" height="232" rx="6" fill="none" stroke="var(--ink-3)" strokeDasharray="2 4" />
              <text x="18" y="26" fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>
                <Translated vi="lưới (grid): một lần khởi chạy kernel">grid: one launch of a kernel</Translated>
              </text>
              {[0, 1].map((b) => (
                <g key={b}>
                  <rect x={BLOCK_X[b]} y="34" width={BLOCK_W} height="196" rx="5" fill="var(--page)" stroke="var(--rule)" />
                  <text x={BLOCK_X[b] + 12} y="53" fontSize="12" fontWeight="600" style={{ fill: 'var(--ink-2)' }}>
                    <Translated vi={<>khối {b}</>}>block {b}</Translated>
                  </text>
                </g>
              ))}
              {/* each thread sits on its own registers */}
              {PAIRS.map(([b, t]) => (
                <line key={`${b}${t}`} x1={colX(b, t) + COL_W / 2} x2={colX(b, t) + COL_W / 2} y1="104" y2="112" stroke="var(--ink-3)" strokeWidth="1.5" />
              ))}
              {/* chip boundary */}
              <line x1="6" x2="554" y1="258" y2="258" stroke="var(--ink-2)" strokeDasharray="6 4" />
              <text x="554" y="252" fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
                <Translated vi="phía trên đường này: trên chip GPU">above this line: on the GPU chip</Translated>
              </text>
              <text x="554" y="272" fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
                <Translated vi="phía dưới: ngoài chip, trong bộ nhớ của thiết bị">below: off the chip, in the device's memory</Translated>
              </text>
              {/* device / host boundary */}
              <line x1="6" x2="554" y1="442" y2="442" stroke="var(--ink-2)" strokeDasharray="6 4" />
              <text x="554" y="455" fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
                <Translated vi="phía dưới đường này: host">below this line: the host</Translated>
              </text>
              {BOXES.map(drawBox)}
            </svg>
            {BOXES.map((x) => (
              <button
                key={x.id}
                type="button"
                data-pick={x.id}
                aria-label={boxName(x)}
                aria-pressed={selId === x.id}
                onClick={() => setSelId(selId === x.id ? null : x.id)}
                onMouseEnter={() => setHover(x.id)}
                onMouseLeave={() => setHover((h) => (h === x.id ? null : h))}
                style={{
                  position: 'absolute',
                  left: `${(x.x / VW) * 100}%`,
                  top: `${(x.y / VH) * 100}%`,
                  width: `${(x.w / VW) * 100}%`,
                  height: `${(x.h / VH) * 100}%`,
                  background: 'transparent',
                  border: 0,
                  padding: 0,
                  borderRadius: 4,
                }}
              />
            ))}
          </div>
          <div style={{ marginTop: '0.7rem' }}>
            <TierLegend tiers={['reg', 'shared', 'global', 'const']} />
          </div>
        </div>
        <div className="panel" data-map-detail aria-live="polite">
          <MapDetail sel={sel} />
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- where does it live?
// Our own kernel, in the manner of the book's exercise 11 (pp. 120-121).
// Front-end-checked with clang's CUDA mode and run on the CPU emulator against a
// CPU reference, in all three emulator schedules (2026-11-20).
const G = 48;
const T = 256;
const TAPS = 4;
const SMOOTH = `#define TAPS 4

__constant__ float taps[TAPS];      // filter weights, filled in by the host before the launch

__global__ void smooth(const float* in, const float* gain, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float window[TAPS];
    __shared__ float blockGain;
    __shared__ float stage[256];

    stage[threadIdx.x] = (i < n) ? in[i] : 0.0f;
    if (threadIdx.x == 0) blockGain = gain[blockIdx.x];
    __syncthreads();

    for (int t = 0; t < TAPS; ++t) {
        unsigned j = threadIdx.x + t;
        window[t] = (j < blockDim.x) ? stage[j] : 0.0f;
    }
    float acc = 0.0f;
    for (int t = 0; t < TAPS; ++t) {
        acc += taps[t] * window[t];
    }
    if (i < n) out[i] = blockGain * acc;
}

void launchSmooth(const float* in, const float* gain, float* out, int n) {
    smooth<<<48, 256>>>(in, gain, out, n);      // 48 blocks of 256 threads
}`;

type Mem = 'register' | 'local' | 'shared' | 'global' | 'constant';
type Scope = 'thread' | 'block' | 'grid';
// Functions, not constants, so the labels are read in the language selected when they render.
const MEMS: () => [Mem, string][] = () => [
  ['register', tr('a register', 'một thanh ghi')],
  ['local', tr('local memory', 'bộ nhớ cục bộ')],
  ['shared', tr('shared memory', 'bộ nhớ chia sẻ')],
  ['global', tr('global memory', 'bộ nhớ toàn cục')],
  ['constant', tr('constant memory', 'bộ nhớ hằng')],
];
const SCOPES: () => [Scope, string][] = () => [
  ['thread', tr('one thread', 'một luồng')],
  ['block', tr('the threads of one block', 'các luồng của một khối')],
  ['grid', tr('every thread of every block', 'mọi luồng của mọi khối')],
];

interface Decl {
  /** The identifier, as spelled in the code. */
  id: string;
  /** What the row asks about. */
  what: ReactNode;
  decl: string;
  mem: Mem;
  scope: Scope;
  copies: number;
  /** Colour the identifier takes once placed. */
  tier: Tier;
  memWhy: Partial<Record<Mem, ReactNode>>;
  scopeWhy: Partial<Record<Scope, ReactNode>>;
  done: ReactNode;
}

const NOT_SHARED = (
  <>
    <Translated
      vi={
        <>
          Không có gì được chia sẻ trừ khi nó ghi <code>__shared__</code>. Không có qualifier, một biến khai báo trong thân
          kernel là riêng của luồng.
        </>
      }
    >
      Nothing is shared unless it says <code>__shared__</code>. Without the qualifier, a variable declared in the kernel
      body is private to the thread.
    </Translated>
  </>
);
const DECLS: Decl[] = [
  {
    id: 'taps',
    what: (
      <>
        <Translated vi={<>mảng <code>taps</code></>}>
          the array <code>taps</code>
        </Translated>
      </>
    ),
    decl: '__constant__ float taps[TAPS];', // i18n-ignore: code
    mem: 'constant',
    scope: 'grid',
    copies: 1,
    tier: 'const',
    memWhy: {
      global: (
        <>
          <Translated
            vi={
              <>
                Gần đúng. Sách nói biến hằng được lưu trong bộ nhớ toàn cục nhưng được đệm (cache). Loại bộ nhớ mà khai báo này
                chọn vẫn là bộ nhớ hằng, và chính điều đó khiến kernel chỉ đọc được nó.
              </>
            }
          >
            Close. The book says constant variables are stored in global memory but cached. The memory type this
            declaration selects is still constant memory, and that is what makes it read-only for kernels.
          </Translated>
        </>
      ),
      register: <Translated vi="Thanh ghi chứa các biến vô hướng khai báo bên trong kernel. Biến này được khai báo bên ngoài mọi hàm, kèm một qualifier.">Registers hold scalars declared inside a kernel. This is declared outside any function, with a qualifier.</Translated>,
      local: <Translated vi="Bộ nhớ cục bộ chứa các mảng khai báo bên trong kernel. Mảng này được khai báo bên ngoài mọi hàm, kèm một qualifier.">Local memory holds arrays declared inside a kernel. This one is declared outside any function, with a qualifier.</Translated>,
      shared: (
        <>
          <Translated
            vi={
              <>
                Bộ nhớ chia sẻ là theo khối và cần <code>__shared__</code>. Qualifier ở đây là một cái khác.
              </>
            }
          >
            Shared memory is per block and needs <code>__shared__</code>. The qualifier here is a different one.
          </Translated>
        </>
      ),
    },
    scopeWhy: {
      thread: <Translated vi="Không luồng nào sở hữu nó. Mọi luồng đều đọc cùng bốn trọng số.">No thread owns it. Every thread reads the same four weights.</Translated>,
      block: <Translated vi="Không có gì gắn nó với một khối. Nó được khai báo ở phạm vi tệp, và mọi luồng của mọi lưới đều đọc được nó.">Nothing ties it to a block. It is declared at file scope, and every thread of every grid can read it.</Translated>,
    },
    done: <Translated vi="Một mảng cho cả ứng dụng, mọi luồng đọc được, chỉ host ghi được.">One array for the whole application, readable by every thread, writable only by the host.</Translated>,
  },
  {
    id: 'in',
    what: (
      <>
        <Translated vi={<>mảng mà <code>in</code> trỏ tới</>}>
          the array <code>in</code> points to
        </Translated>
      </>
    ),
    decl: 'const float* in', // i18n-ignore: code
    mem: 'global',
    scope: 'grid',
    copies: 1,
    tier: 'global',
    memWhy: {
      register: (
        <>
          <Translated
            vi={
              <>
                Đúng là bản thân con trỏ là một biến vô hướng của từng luồng. Hàng này hỏi về mảng mà nó trỏ tới, mảng do host cấp
                phát trong bộ nhớ toàn cục.
              </>
            }
          >
            The pointer itself is a per-thread scalar, true. The row asks about the array it points to, which the host
            allocated in global memory.
          </Translated>
        </>
      ),
      constant: (
        <>
          <Translated
            vi={
              <>
                <code>const float*</code> là một lời hứa của C++ rằng kernel này sẽ không ghi qua con trỏ. Bộ nhớ hằng của CUDA là
                thứ khác và cần <code>__constant__</code>.
              </>
            }
          >
            <code>const float*</code> is a C++ promise that this kernel will not write through the pointer. CUDA's
            constant memory is something else and needs <code>__constant__</code>.
          </Translated>
        </>
      ),
      local: <Translated vi="Bộ nhớ cục bộ chứa các mảng riêng của một luồng. Mảng này đến từ host và giống nhau với mọi luồng.">Local memory holds a thread's private arrays. This array came from the host and is the same for every thread.</Translated>,
      shared: <Translated vi="Bộ nhớ chia sẻ thuộc về một khối và do chính kernel điền vào. Mảng này đến từ bên ngoài.">Shared memory belongs to one block and is filled by the kernel itself. This array arrives from outside.</Translated>,
    },
    scopeWhy: {
      thread: <Translated vi="Mọi luồng nhận cùng một địa chỉ, nên mọi luồng với tới cùng một mảng.">Every thread receives the same address, so every thread reaches the same array.</Translated>,
      block: (
        <>
          <Translated
            vi={
              <>
                Mọi khối nhận cùng một địa chỉ. Ở đây mỗi khối tình cờ đọc lát của riêng nó, nhưng không có gì ngăn một luồng đọc
                bất kỳ phần tử nào.
              </>
            }
          >
            Every block receives the same address. Each block happens to read its own slice here, but nothing stops a
            thread reading any element.
          </Translated>
        </>
      ),
    },
    done: <Translated vi="Một mảng trong bộ nhớ toàn cục. Cả 12,288 luồng nhận địa chỉ của nó, và mỗi lần nạp từ nó là lưu lượng bộ nhớ toàn cục.">One array in global memory. All 12,288 threads receive its address, and every load from it is global-memory traffic.</Translated>,
  },
  {
    id: 'i',
    what: <code>i</code>,
    decl: 'int i = ...;',
    mem: 'register',
    scope: 'thread',
    copies: G * T,
    tier: 'reg',
    memWhy: {
      local: (
        <>
          <Translated
            vi={
              <>
                Bộ nhớ cục bộ dành cho những gì không vào được thanh ghi, trước hết là mảng. <code>i</code> là một biến vô hướng
                khai báo trong thân kernel, nên theo quy tắc của sách nó được cấp một thanh ghi.
              </>
            }
          >
            Local memory is for what cannot go in a register, arrays above all. <code>i</code> is a scalar declared in
            the kernel body, so by the book's rule it gets a register.
          </Translated>
        </>
      ),
      shared: NOT_SHARED,
      global: (
        <>
          <Translated
            vi={
              <>
                <code>i</code> không phải dữ liệu truyền vào từ bên ngoài. Mỗi luồng tự tính nó từ các chỉ số của mình và giữ nó
                trong một thanh ghi.
              </>
            }
          >
            <code>i</code> is not data passed in from outside. Each thread computes it from its own indices and keeps it
            in a register.
          </Translated>
        </>
      ),
      constant: (
        <>
          <Translated
            vi={
              <>
                Bộ nhớ hằng cần <code>__constant__</code> bên ngoài mọi hàm, và kernel không ghi được nó. <code>i</code> được gán
                bên trong kernel.
              </>
            }
          >
            Constant memory needs <code>__constant__</code> outside any function, and kernels cannot write it.{' '}
            <code>i</code> is assigned inside the kernel.
          </Translated>
        </>
      ),
    },
    scopeWhy: {
      block: <Translated vi="Ở cùng một khối tự nó không chia sẻ gì. Mỗi luồng có giá trị i riêng: bắt buộc phải vậy, vì các giá trị khác nhau.">Being in the same block shares nothing by itself. Each thread has its own value of i: it has to, the values differ.</Translated>,
      grid: <Translated vi="Chỉ dữ liệu trong bộ nhớ toàn cục hoặc bộ nhớ hằng mới thấy được trên toàn lưới. Biến này được khai báo trong thân kernel.">Only data in global or constant memory is visible grid-wide. This is declared in the kernel body.</Translated>,
    },
    done: <Translated vi="Một biến vô hướng khai báo trong thân kernel: mỗi luồng một bản, mỗi bản trong một thanh ghi.">A scalar declared in the kernel body: one copy per thread, each in a register.</Translated>,
  },
  {
    id: 'window',
    what: (
      <>
        <Translated vi={<>mảng <code>window</code></>}>
          the array <code>window</code>
        </Translated>
      </>
    ),
    decl: 'float window[TAPS];', // i18n-ignore: code
    mem: 'local',
    scope: 'thread',
    copies: G * T,
    tier: 'global',
    memWhy: {
      register: (
        <>
          <Translated
            vi={
              <>
                Đó là quy tắc cho biến vô hướng. Một mảng khai báo trong thân kernel mặc định vào bộ nhớ cục bộ. Ngoại lệ dành cho
                các mảng chỉ được truy cập bằng chỉ số hằng, còn mảng này được đánh chỉ số bằng biến vòng lặp.
              </>
            }
          >
            That is the rule for scalars. An array declared in the kernel body goes to local memory by default. The
            exception is for arrays accessed only with constant indices, and this one is indexed by the loop variable.
          </Translated>
        </>
      ),
      shared: NOT_SHARED,
      global: (
        <>
          <Translated
            vi={
              <>
                Gần đúng: về vật lý nó nằm ở đó. Nhưng loại bộ nhớ là bộ nhớ cục bộ, và khác biệt này quan trọng: nó là riêng của
                luồng, điều mà bộ nhớ toàn cục không có.
              </>
            }
          >
            Close: that is where it physically is. But the memory type is local memory, and the difference matters: it is
            private to the thread, which global memory is not.
          </Translated>
        </>
      ),
      constant: <Translated vi="Mỗi luồng đều ghi nó, mà kernel không ghi được bộ nhớ hằng.">Each thread writes it, and kernels cannot write constant memory.</Translated>,
    },
    scopeWhy: {
      block: <Translated vi="Không có qualifier, nên không chia sẻ. Mỗi luồng điền window của riêng nó bằng bốn mẫu của riêng nó.">No qualifier, so no sharing. Each thread fills its own window with its own four samples.</Translated>,
      grid: <Translated vi="Về vật lý nó nằm trong bộ nhớ toàn cục, nhưng chỉ chủ của nó với tới được.">It sits in global memory physically, but only its owner can reach it.</Translated>,
    },
    done: (
      <>
        <Translated
          vi={
            <>
              Một mảng khai báo trong thân kernel: mỗi luồng một mảng, trong bộ nhớ cục bộ. Nó mang màu toàn cục vì bộ nhớ cục bộ
              nằm vật lý trong bộ nhớ toàn cục.
            </>
          }
        >
          An array declared in the kernel body: one per thread, in local memory. It takes the global colour because
          global memory is where local memory physically is.
        </Translated>
      </>
    ),
  },
  {
    id: 'blockGain',
    what: <code>blockGain</code>,
    decl: '__shared__ float blockGain;', // i18n-ignore: code
    mem: 'shared',
    scope: 'block',
    copies: G,
    tier: 'shared',
    memWhy: {
      register: (
        <>
          <Translated
            vi={
              <>
                Nó là biến vô hướng, nhưng qualifier <code>__shared__</code> ghi đè mặc định cho biến vô hướng.
              </>
            }
          >
            It is a scalar, but the <code>__shared__</code> qualifier overrides the default for scalars.
          </Translated>
        </>
      ),
      local: <Translated vi="Bộ nhớ cục bộ là riêng của một luồng. Luồng 0 ghi biến này và 255 luồng còn lại đọc nó.">Local memory is private to a thread. Thread 0 writes this variable and the other 255 threads read it.</Translated>,
      global: (
        <>
          <Translated
            vi={
              <>
                <code>gain</code>, tham số, trỏ vào bộ nhớ toàn cục. <code>blockGain</code> là bản sao riêng của khối cho một phần
                tử của nó.
              </>
            }
          >
            <code>gain</code>, the parameter, points into global memory. <code>blockGain</code> is the block's own copy
            of one element of it.
          </Translated>
        </>
      ),
      constant: <Translated vi="Luồng 0 ghi nó, mà kernel không ghi được bộ nhớ hằng.">Thread 0 writes it, and kernels cannot write constant memory.</Translated>,
    },
    scopeWhy: {
      thread: (
        <>
          <Translated
            vi={
              <>
                <code>__shared__</code> tồn tại để phá vỡ tính riêng tư của luồng. Luồng 0 lưu giá trị và mọi luồng của khối đọc
                nó.
              </>
            }
          >
            <code>__shared__</code> exists to break thread privacy. Thread 0 stores the value and every thread of the
            block reads it.
          </Translated>
        </>
      ),
      grid: <Translated vi="Bộ nhớ chia sẻ dừng ở ranh giới khối. Khối khác có một bản khác, giữ một hệ số gain khác.">Shared memory stops at the block boundary. Another block has a different copy, holding a different gain.</Translated>,
    },
    done: <Translated vi="Một số float mỗi khối. Luồng 0 nạp nó từ bộ nhớ toàn cục một lần, và 256 luồng dùng nó.">One float per block. Thread 0 fetches it from global memory once, and 256 threads use it.</Translated>,
  },
  {
    id: 'stage',
    what: (
      <>
        <Translated vi={<>mảng <code>stage</code></>}>
          the array <code>stage</code>
        </Translated>
      </>
    ),
    decl: '__shared__ float stage[256];', // i18n-ignore: code
    mem: 'shared',
    scope: 'block',
    copies: G,
    tier: 'shared',
    memWhy: {
      local: (
        <>
          <Translated
            vi={
              <>
                Mảng chỉ vào bộ nhớ cục bộ khi chúng là biến tự động thông thường. <code>__shared__</code> đặt mảng này vào bộ nhớ
                chia sẻ.
              </>
            }
          >
            Arrays go to local memory only when they are plain automatic variables. <code>__shared__</code> puts this one
            in shared memory.
          </Translated>
        </>
      ),
      register: <Translated vi="Nó là một mảng, và nó mang một qualifier chỉ rõ bộ nhớ của nó.">It is an array, and it carries a qualifier that names its memory.</Translated>,
      global: (
        <>
          <Translated
            vi={
              <>
                Nội dung của nó đến từ bộ nhớ toàn cục, nhưng bản thân mảng là bản sao trên chip của khối. Đó là toàn bộ lý do nó
                tồn tại.
              </>
            }
          >
            Its contents come from global memory, but the array itself is the block's on-chip copy. That is the whole
            reason it exists.
          </Translated>
        </>
      ),
      constant: <Translated vi="Mỗi luồng ghi một phần tử của nó, mà kernel không ghi được bộ nhớ hằng.">Every thread writes one element of it, and kernels cannot write constant memory.</Translated>,
    },
    scopeWhy: {
      thread: <Translated vi="Mỗi luồng ghi một phần tử rồi đọc các phần tử của luồng bên cạnh. Nó không làm được vậy nếu mảng là riêng.">Each thread writes one element and then reads its neighbours' elements. It could not if the array were private.</Translated>,
      grid: <Translated vi="Bộ nhớ chia sẻ dừng ở ranh giới khối. Đó là lý do một luồng ở cuối khối không thấy được các mẫu của khối kế tiếp.">Shared memory stops at the block boundary. That is why a thread at the end of a block cannot see the next block's samples.</Translated>,
    },
    done: <Translated vi="Một mảng 256 phần tử mỗi khối, trên chip, chỉ các luồng của khối đó thấy được.">One 256-element array per block, on the chip, visible to that block's threads and nobody else.</Translated>,
  },
];

function copiesHint(d: Decl, v: number): ReactNode {
  const perThread = G * T;
  if (!Number.isFinite(v) || v < 0 || !Number.isInteger(v)) return <Translated vi="Số bản sao là một số nguyên.">A count of copies is a whole number.</Translated>;
  if (d.copies === perThread) {
    if (v === G) return <Translated vi="48 là số khối. Biến này là riêng của một luồng, nên hãy đếm luồng.">48 is the number of blocks. This one is private to a thread, so count threads.</Translated>;
    if (v === T) return <Translated vi="256 là số luồng trong một khối, và mỗi khối trong 48 khối có 256 luồng của riêng nó.">256 is the number of threads in one block, and each of the 48 blocks has its own 256.</Translated>;
    if (v === 1) return <Translated vi="Trong mã nguồn có một khai báo, nhưng mỗi luồng chạy kernel đều có bản riêng.">There is one declaration in the source, but every thread that runs the kernel gets its own copy.</Translated>;
    if (d.id === 'window' && v === perThread * TAPS)
      return <Translated vi="Đó là đếm số float. Hàng này hỏi có bao nhiêu mảng: mỗi luồng một mảng.">That counts floats. The row asks how many arrays exist: one per thread.</Translated>;
  } else if (d.copies === G) {
    if (v === perThread)
      return d.id === 'stage' ? (
        <Translated vi="12,288 nghĩa là mỗi luồng một bản. (Đó cũng là tổng số float trong mọi bản sao cộng lại.) Mỗi khối có một mảng.">12,288 would be one per thread. (It is also the number of floats in all the copies together.) There is one array per block.</Translated>
      ) : (
        <>
          <Translated
            vi={
              <>
                Như vậy là mỗi luồng một bản. <code>__shared__</code> làm nó thành mỗi khối một bản.
              </>
            }
          >
            That would be one per thread. <code>__shared__</code> makes it one per block.
          </Translated>
        </>
      );
    if (v === T) return <Translated vi="256 là số luồng dùng mỗi bản. Hãy đếm số bản: mỗi khối một bản.">256 is how many threads use each copy. Count the copies: one per block.</Translated>;
    if (v === 1) return <Translated vi="Một bản duy nhất cho cả lưới sẽ cần khai báo bên ngoài kernel. Biến này là theo khối.">A single copy for the whole grid would need a declaration outside the kernel. This one is per block.</Translated>;
  } else {
    if (d.id === 'taps' && v === TAPS) return <Translated vi="Bốn là số phần tử. Hàng này hỏi có bao nhiêu mảng.">Four is the number of elements. The row asks how many arrays exist.</Translated>;
    if (v === G) return <Translated vi="Các khối không có bản riêng. Cả 48 khối đọc cùng một mảng.">Blocks do not get their own. All 48 read the same array.</Translated>;
    if (v === perThread)
      return d.id === 'in' ? (
        <Translated vi="Mỗi luồng giữ địa chỉ, tổng cộng 12,288 con trỏ. Mảng đằng sau chúng chỉ có một.">Each thread holds the address, 12,288 pointers in all. The array behind them exists once.</Translated>
      ) : (
        <Translated vi="Các luồng không có bản riêng. Một bản phục vụ cả 12,288 luồng.">Threads do not get their own. One copy serves all 12,288 of them.</Translated>
      );
  }
  return <Translated vi="Hãy bắt đầu từ ai truy cập được nó: bao nhiêu luồng, khối hoặc lưới mỗi cái giữ một bản riêng?">Start from who can access it: how many threads, or blocks, or grids each hold their own?</Translated>;
}

interface RowState {
  mem: Mem | '';
  scope: Scope | '';
  copies: string;
  /** The answers as they were when Check was pressed. */
  checked: { mem: Mem; scope: Scope; copies: number } | null;
}
const blankRows = (): Record<string, RowState> =>
  Object.fromEntries(DECLS.map((d) => [d.id, { mem: '', scope: '', copies: '', checked: null }]));
const isSolved = (d: Decl, s: RowState) =>
  !!s.checked && s.checked.mem === d.mem && s.checked.scope === d.scope && s.checked.copies === d.copies;

function WhereItLives() {
  const [rows, setRows] = useState<Record<string, RowState>>(blankRows);
  const set = (id: string, patch: Partial<RowState>) =>
    setRows((r) => ({ ...r, [id]: { ...r[id], checked: null, ...patch } }));
  const solved = DECLS.filter((d) => isSolved(d, rows[d.id]));
  const tiers: Record<string, Tier> = Object.fromEntries(solved.map((d) => [d.id, d.tier]));

  return (
    <Bench
      title={tr('Where does it live?', 'Nó nằm ở đâu?')}
      task={tr(
        `One launch: ${G} blocks of ${T} threads. Place all six names.`,
        `Một lần khởi chạy: ${G} khối, mỗi khối ${T} luồng. Hãy xếp chỗ cho cả sáu cái tên.`,
      )}
      foot={tr(
        "Answers follow the book's declaration rules. A name takes its memory's colour in the code once all three answers for it are right.",
        'Đáp án theo quy tắc khai báo của sách. Một cái tên mang màu bộ nhớ của nó trong mã khi cả ba câu trả lời cho nó đều đúng.',
      )}
    >
      <div data-where>
        <Code
          tabs={[{ label: 'CUDA', code: SMOOTH, tiers, status: 'front-end-checked (labs/ch05/snippets/where_does_it_live.cu)' }]}
          caption={tr(
            'A smoothing filter of our own, written for this exercise. Each thread combines its sample and the three after it inside its block, then scales by a per-block gain.',
            'Một bộ lọc làm mượt của chúng tôi, viết cho bài tập này. Mỗi luồng kết hợp mẫu của nó và ba mẫu kế tiếp trong khối, rồi nhân với một hệ số gain theo khối.',
          )}
        />
        <div className="panel">
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: '0.2rem' }}>
            <b>
              <Translated vi="Sáu cái tên">Six names</Translated>
            </b>
            <span className="muted" data-where-score>
              <Translated vi={<>đã xếp {solved.length} trên {DECLS.length}</>}>
                {solved.length} of {DECLS.length} placed
              </Translated>
            </span>
          </div>
          {DECLS.map((d, n) => {
            const s = rows[d.id];
            const done = isSolved(d, s);
            const ready = s.mem !== '' && s.scope !== '' && s.copies.trim() !== '';
            const c = s.checked;
            const check = () =>
              set(d.id, { checked: { mem: s.mem as Mem, scope: s.scope as Scope, copies: Number(s.copies) } });
            return (
              <div
                key={d.id}
                data-row={d.id}
                data-solved={done ? 'true' : 'false'}
                style={{ padding: '0.7rem 0', borderTop: n ? '1px solid var(--rule-soft)' : undefined }}
              >
                <div className="row" style={{ alignItems: 'flex-end' }}>
                  <div style={{ flex: '1 1 11rem', minWidth: '11rem' }}>
                    <p style={{ margin: 0 }}>{d.what}</p>
                    <div className="muted mono nowrap" style={{ fontSize: '0.74rem' }}>
                      {d.decl}
                    </div>
                  </div>
                  <label className="field">
                    <Translated vi="Bộ nhớ">Memory</Translated>
                    <select
                      aria-label={tr(`Memory for ${d.id}`, `Bộ nhớ cho ${d.id}`)}
                      value={s.mem}
                      disabled={done}
                      onChange={(e) => set(d.id, { mem: e.target.value as Mem | '' })}
                    >
                      <option value="">{tr('choose', 'chọn')}</option>
                      {MEMS().map(([v, label]) => (
                        <option key={v} value={v}>
                          {label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <Translated vi="Ai truy cập được">Who can access it</Translated>
                    <select
                      aria-label={tr(`Who can access ${d.id}`, `Ai truy cập được ${d.id}`)}
                      value={s.scope}
                      disabled={done}
                      onChange={(e) => set(d.id, { scope: e.target.value as Scope | '' })}
                    >
                      <option value="">{tr('choose', 'chọn')}</option>
                      {SCOPES().map(([v, label]) => (
                        <option key={v} value={v}>
                          {label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <Translated vi="Số bản trong lần khởi chạy này">Copies in this launch</Translated>
                    <input
                      type="number"
                      min="0"
                      step="1"
                      aria-label={tr(`Copies of ${d.id}`, `Số bản của ${d.id}`)}
                      value={s.copies}
                      disabled={done}
                      onChange={(e) => set(d.id, { copies: e.target.value })}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && ready) check();
                      }}
                    />
                  </label>
                  <div style={{ minWidth: '4.6rem' }}>
                    {done ? (
                      <span className="chip ok">
                        <Translated vi="đã xếp">placed</Translated>
                      </span>
                    ) : (
                      <button className="primary" aria-label={tr(`Check ${d.id}`, `Kiểm tra ${d.id}`)} disabled={!ready} onClick={check}>
                        <Translated vi="Kiểm tra">Check</Translated>
                      </button>
                    )}
                  </div>
                </div>
                {c ? (
                  <div data-feedback={d.id} role="status" style={{ marginTop: '0.5rem' }}>
                    {done ? (
                      <p style={{ margin: 0 }}>{d.done}</p>
                    ) : (
                      <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none' }}>
                        <li>
                          {c.mem === d.mem ? (
                            <span className="chip ok">
                              <Translated vi="bộ nhớ đúng">memory is right</Translated>
                            </span>
                          ) : (
                            <>
                              <span className="chip fail">
                                <Translated vi="bộ nhớ">memory</Translated>
                              </span>{' '}
                              {d.memWhy[c.mem] ?? <Translated vi="Hãy nhìn khai báo nằm ở đâu và mang qualifier nào.">Look at where the declaration sits and which qualifier it carries.</Translated>}
                            </>
                          )}
                        </li>
                        <li>
                          {c.scope === d.scope ? (
                            <span className="chip ok">
                              <Translated vi="quyền truy cập đúng">access is right</Translated>
                            </span>
                          ) : (
                            <>
                              <span className="chip fail">
                                <Translated vi="truy cập">access</Translated>
                              </span>{' '}
                              {d.scopeWhy[c.scope] ?? <Translated vi="Hãy hỏi những luồng nào có thể gọi tên biến này và nhận cùng một vùng lưu trữ.">Ask which threads could name this variable and get the same storage.</Translated>}
                            </>
                          )}
                        </li>
                        <li style={{ marginBottom: 0 }}>
                          {c.copies === d.copies ? (
                            <span className="chip ok">
                              <Translated vi="số lượng đúng">count is right</Translated>
                            </span>
                          ) : (
                            <>
                              <span className="chip fail">
                                <Translated vi="số lượng">count</Translated>
                              </span> {copiesHint(d, c.copies)}
                            </>
                          )}
                        </li>
                      </ul>
                    )}
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
        <div className="row" style={{ marginTop: '0.8rem', justifyContent: 'space-between' }}>
          <TierLegend tiers={['reg', 'shared', 'global', 'const']} />
          <button className="quiet" onClick={() => setRows(blankRows())}>
            <Translated vi="Làm lại từ đầu">Start again</Translated>
          </button>
        </div>
        {solved.length === DECLS.length ? (
          <div className="panel" style={{ marginTop: '0.8rem' }} data-where-done>
            <p style={{ margin: 0 }}>
              <Translated
                vi={
                  <>
                    <b>Đã xếp đủ sáu.</b> Giờ hãy đọc kernel theo màu. Mỗi luồng xin bộ nhớ toàn cục một phần tử của <code>in</code> và
                    ghi một phần tử của <code>out</code>; mỗi khối còn có một luồng nạp một hệ số gain. Nhưng <code>window</code> cũng
                    mang màu toàn cục: nếu trình biên dịch để mảng đó trong bộ nhớ cục bộ, mỗi lần trong tám lần truy cập của mỗi luồng
                    đều trả độ trễ của bộ nhớ toàn cục mà mã nguồn không hề cho thấy. Bốn biến vô hướng thì đã vào thanh ghi.
                  </>
                }
              >
                <b>All six placed.</b> Now read the kernel by colour. Per thread it asks global memory for one element of{' '}
                <code>in</code> and stores one element of <code>out</code>; one thread per block also fetches a gain. But{' '}
                <code>window</code> has the global colour too: if the compiler leaves that array in local memory, each of
                its eight accesses per thread pays global-memory latency that the source never shows. Four scalars would
                have gone to registers.
              </Translated>
            </p>
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- your devices
// What ./verify.sh recorded from the reader's own devices, for the query row of
// the comparison table. Nothing is shown that a device did not report.
const DEVICE_FIELDS: () => [string, string][] = () => [
  ['hasUnifiedMemory', tr('CPU and GPU share all memory', 'CPU và GPU dùng chung toàn bộ bộ nhớ')],
  ['maxThreadgroupMemoryLength', tr('threadgroup memory available to a kernel, bytes', 'bộ nhớ threadgroup sẵn có cho một kernel, byte')],
  ['sharedMemPerBlock', tr('shared memory per block, bytes', 'bộ nhớ chia sẻ mỗi khối, byte')],
  ['sharedMemPerMultiprocessor', tr('shared memory per multiprocessor, bytes', 'bộ nhớ chia sẻ mỗi multiprocessor, byte')],
  ['totalConstMem', tr('constant memory, bytes', 'bộ nhớ hằng, byte')],
  ['totalGlobalMem', tr('global memory, bytes', 'bộ nhớ toàn cục, byte')],
];
function YourDevices() {
  const { reports, loading } = useReports();
  const rows = realBackends(reports).flatMap((b) =>
    DEVICE_FIELDS().filter(([k]) => reports[b].device[k] !== undefined).map(([k, meaning]) => {
      const raw = String(reports[b].device[k]);
      const value = k === 'hasUnifiedMemory' ? (raw === '1' || raw === 'true' ? tr('yes', 'có') : tr('no', 'không')) : Number(raw).toLocaleString('en-US');
      return { b, name: String(reports[b].device.name ?? b), k, meaning, value };
    }),
  );
  return (
    <div className="panel" data-your-devices style={{ marginBottom: '1.4rem' }}>
      <p style={{ margin: '0 0 0.4rem' }} className="ui">
        <b>
          <Translated vi="Thiết bị của bạn đã trả lời gì">What your own devices answered</Translated>
        </b>
      </p>
      {rows.length ? (
        <table className="data" style={{ margin: 0 }}>
          <thead>
            <tr>
              <th>
                <Translated vi="Backend">Backend</Translated>
              </th>
              <th>
                <Translated vi="Thiết bị">Device</Translated>
              </th>
              <th>
                <Translated vi="Thuộc tính">Property</Translated>
              </th>
              <th>
                <Translated vi="Ý nghĩa">Meaning</Translated>
              </th>
              <th className="num">
                <Translated vi="Giá trị báo cáo">Reported</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.b + r.k}>
                <td>{r.b}</td>
                <td>{r.name}</td>
                <td>
                  <code>{r.k}</code>
                </td>
                <td>{r.meaning}</td>
                <td className="num">{r.value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p style={{ margin: 0 }} className="ui muted">
          {loading
            ? tr('Reading reports.', 'Đang đọc báo cáo.')
            : tr(
                'No device report yet. ./verify.sh records what each of your devices reports, and this table fills in.',
                'Chưa có báo cáo thiết bị nào. ./verify.sh ghi lại những gì từng thiết bị của bạn báo cáo, và bảng này sẽ được điền.',
              )}
        </p>
      )}
    </div>
  );
}

// Row headers of a data table: keep the heavier weight, not the heavier rule.
const ROW_HEAD = { borderBottom: '1px solid var(--rule-soft)' }; // i18n-ignore: CSS value

// ---------------------------------------------------------------- questions
export const questionsS52: Record<string, Question> = {
  'ch05/5.2/q-copies-thread': {
    kind: 'numeric',
    id: 'ch05/5.2/q-copies-thread',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Thân một kernel khai báo <code>float acc = 0.0f;</code>. Kernel được khởi chạy một lần, gồm 1000 khối, mỗi khối 512
              luồng. Trong lần khởi chạy đó có bao nhiêu phiên bản riêng biệt của <code>acc</code>?
            </>
          }
        >
          A kernel body declares <code>float acc = 0.0f;</code>. The kernel is launched once, as 1000 blocks of 512
          threads. How many separate versions of <code>acc</code> exist during that launch?
        </Translated>
      </>
    ),
    answer: 512000,
    unit: 'versions',
    explain: (
      <Translated
        vi={
          <p>
            Một biến tự động có phạm vi luồng: mỗi luồng có một phiên bản riêng, nên 1000 × 512 = 512,000. Mỗi phiên bản chiếm
            một thanh ghi (register) của luồng đó. Nếu nó là một mảng tự động, số lượng vẫn vậy nhưng nơi chứa sẽ là bộ nhớ cục
            bộ (local memory): 512,000 vùng riêng trong bộ nhớ toàn cục.
          </p>
        }
      >
        <p>
          An automatic variable has thread scope: every thread gets a private version, so 1000 × 512 = 512,000. Each one
          occupies a register of its thread. Had it been an automatic array, the count would be the same and the place
          would be local memory: 512,000 private regions of global memory.
        </p>
      </Translated>
    ),
    source: 'ch. 5 exercise 6, p. 119; §5.2, p. 102',
    vi: { unit: 'phiên bản', source: 'bài tập 6 chương 5, tr. 119; §5.2, tr. 102' },
  },
  'ch05/5.2/q-copies-block': {
    kind: 'numeric',
    id: 'ch05/5.2/q-copies-block',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Cùng kernel đó khai báo <code>__shared__ float stage[512];</code>. Cùng lần khởi chạy: 1000 khối, mỗi khối 512
              luồng. Có bao nhiêu phiên bản của mảng <code>stage</code>? Hãy đếm mảng, không đếm phần tử.
            </>
          }
        >
          The same kernel declares <code>__shared__ float stage[512];</code>. Same launch: 1000 blocks of 512 threads.
          How many versions of the array <code>stage</code> exist? Count arrays, not elements.
        </Translated>
      </>
    ),
    answer: 1000,
    unit: 'versions',
    explain: (
      <Translated
        vi={
          <p>
            Phạm vi khối: mỗi khối một bản, nên là 1000, mỗi bản chỉ 512 luồng của khối đó thấy được. Chia đáp án trước cho đáp
            án này, bạn được 512: chừng ấy luồng đọc được một giá trị khi nó nằm trong bộ nhớ chia sẻ (shared memory), trong
            khi một thanh ghi chỉ phục vụ đúng một luồng. Phần 5.3 biến điều đó thành ít lần nạp từ bộ nhớ toàn cục hơn.
          </p>
        }
      >
        <p>
          Block scope: one per block, so 1000, each visible to the 512 threads of its block and to no one else. Divide
          the previous answer by this one and you get 512: that many threads can read a value once it sits in shared
          memory, where a register serves exactly one. Section 5.3 turns that into fewer fetches from global memory.
        </p>
      </Translated>
    ),
    source: 'ch. 5 exercise 7, p. 119; §5.2, pp. 102–103',
    vi: { unit: 'phiên bản', source: 'bài tập 7 chương 5, tr. 119; §5.2, tr. 102–103' },
  },
  'ch05/5.2/q-why-shared': {
    kind: 'choice',
    id: 'ch05/5.2/q-why-shared',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Giả sử một thiết bị dư thanh ghi, nên dung lượng không phải lý lẽ. Các luồng của một khối liên tục cần cùng những
              giá trị từ bộ nhớ toàn cục. Vì sao giữ các giá trị đó trong bộ nhớ chia sẻ mà không phải trong thanh ghi?
            </>
          }
        >
          Suppose a device had registers to spare, so capacity is no argument. A block's threads keep needing the same
          values from global memory. Why hold those values in shared memory and not in registers?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <Translated vi="Đọc bộ nhớ chia sẻ nhanh hơn đọc thanh ghi, nên phép tính chờ toán hạng ít hơn.">
            Reading shared memory is faster than reading a register, so the arithmetic waits less for operands.
          </Translated>
        ),
        why: (
          <Translated vi="Ngược lại mới đúng: bộ nhớ chia sẻ cần một lệnh nạp, toán hạng thanh ghi thì không.">
            It is the other way round: shared memory needs a load instruction, a register operand does not.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một thanh ghi chỉ phục vụ luồng của chính nó. Một giá trị được nạp một lần vào bộ nhớ chia sẻ có thể được mọi luồng của khối đọc.">
            A register serves only its own thread. A value fetched once into shared memory can be read by every thread of the block.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một giá trị giữ trong thanh ghi vẫn tốn băng thông bộ nhớ toàn cục mỗi lần được đọc, vì thanh ghi phải được nạp lại.">
            A value held in a register still uses global-memory bandwidth each time it is read, as the register is refilled.
          </Translated>
        ),
        why: (
          <Translated vi="Một giá trị giữ trong thanh ghi không tốn chút nào. Đó chính là lý do thanh ghi nâng tỉ số FLOP/B.">
            A value kept in a register uses none. That is exactly why registers raise the FLOP/B ratio.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Thanh ghi mất nội dung mỗi khi SM chuyển sang warp khác, nên giá trị phải được nạp lại.">
            Registers lose their contents whenever the SM switches to another warp, so the value is fetched again.
          </Translated>
        ),
        why: (
          <Translated vi="Tệp thanh ghi giữ thanh ghi của mọi luồng đã được lập lịch. Đó là điều khiến việc chuyển đổi không tốn gì.">
            The register file keeps the registers of every scheduled thread. That is what makes the switch free.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Thanh ghi là riêng. Nếu 512 luồng mỗi luồng đều cần cùng một phần tử, 512 thanh ghi nghĩa là 512 lần nạp từ bộ nhớ
            toàn cục. Bộ nhớ chia sẻ thì cả khối thấy được, nên một luồng có thể nạp phần tử đó và các luồng khác đọc bản sao
            trên chip. Khoản tiết kiệm đến từ việc chia sẻ, không phải từ tốc độ của bộ nhớ: thanh ghi nhanh hơn bộ nhớ chia sẻ
            mà vẫn không làm được việc này.
          </p>
        }
      >
        <p>
          Registers are private. If 512 threads each need the same element, 512 registers mean 512 fetches from global
          memory. Shared memory is visible to the whole block, so one thread can fetch the element and the others read
          the on-chip copy. The saving comes from the sharing, not from the speed of the memory: a register is faster
          than shared memory and still could not do this job.
        </p>
      </Translated>
    ),
    source: 'ch. 5 exercise 4, p. 119; §5.2, pp. 99–101',
    vi: { source: 'bài tập 4 chương 5, tr. 119; §5.2, tr. 99–101' },
  },
  'ch05/5.2/q-unified': {
    kind: 'choice',
    id: 'ch05/5.2/q-unified',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Bạn chuyển một chương trình CUDA sang Metal trên Apple silicon. Bản CUDA chép đầu vào sang device, khởi chạy kernel
              rồi chép kết quả về. Trong bản Metal, đầu vào và đầu ra là các buffer ở chế độ lưu trữ shared. Chuyện gì xảy ra với
              hai lần sao chép đó?
            </>
          }
        >
          You port a CUDA program to Metal on Apple silicon. The CUDA version copies the input to the device, launches
          the kernel and copies the result back. In the Metal version the input and the output are buffers in shared
          storage mode. What happens to the two copies?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <Translated vi="Chúng vẫn cần thiết. Chế độ lưu trữ shared chỉ giúp bạn khỏi cấp phát buffer hai lần, mỗi bộ xử lý một lần.">
            They are still needed. Shared storage mode only saves you from allocating the buffer twice, once per processor.
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ có một vùng cấp phát và cả hai bộ xử lý đều truy cập nó. Không có gì để chép qua lại.">
            There is one allocation and both processors address it. There is nothing to copy between.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chúng biến mất, và CPU có thể đọc đầu ra ngay khi đã mã hóa và commit công việc.">
            They disappear, and the CPU may read the output as soon as it has encoded and committed the work.
          </Translated>
        ),
        why: (
          <Translated vi="Commit chỉ lập lịch công việc. Đầu ra chưa có ở đó cho đến khi GPU ghi xong.">
            Committing schedules the work. The output is not there until the GPU has finished writing it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chúng biến mất: CPU và GPU truy cập cùng một vùng cấp phát. Điều còn lại là thứ tự: CPU phải chờ đến khi công việc GPU ghi đầu ra đã hoàn tất.">
            They disappear: CPU and GPU address the same allocation. What remains is ordering: the CPU must wait until the GPU work that writes the output has completed.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chúng được thay bằng một lời gọi báo cho Metal biết dải byte nào của buffer đã bị CPU sửa.">
            They are replaced by a call that tells Metal which byte range of the buffer the CPU modified.
          </Translated>
        ),
        why: (
          <Translated vi="Lời gọi đó thuộc chế độ lưu trữ managed, chế độ Apple đã ngừng khuyến nghị và không có tác dụng trên Apple silicon.">
            That call belongs to managed storage mode, which Apple has deprecated and which has no effect on Apple silicon.
          </Translated>
        ),
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <>
              <p>
                Với unified memory, một buffer chế độ shared là bộ nhớ hệ thống mà cả hai bộ xử lý đều truy cập được, nên một lần
                truyền không có gì để di chuyển và không lời gọi API nào thay thế nó. Con trỏ CPU giữ hợp lệ ở mọi thời điểm và
                không chờ gì cả, nên bảo đảm thứ tự là việc của bạn. Tài liệu của Apple diễn đạt điều đó là bảo đảm các thay đổi
                được lập lịch trên một bộ xử lý hoàn tất trước khi bộ xử lý kia truy cập tài nguyên.
              </p>
              <p>
                Bên trong kernel không có gì thay đổi. Nó vẫn có các giá trị riêng của một luồng, bộ nhớ threadgroup cho nhóm, và
                buffer cho mọi thứ khác, và việc dùng chung buffer với CPU không cho bạn biết kernel đọc nó nhanh đến đâu. Lab 5.1
                đo điều đó.
              </p>
            </>
          }
        >
          <p>
            With unified memory a shared-mode buffer is system memory that both processors can address, so a transfer
            has nothing to move and no API call stands in for it. The pointer the CPU holds is valid at any moment and
            waits for nothing, so the ordering is yours to enforce. Apple's documentation puts it as making sure the
            changes scheduled on one processor complete before the other accesses the resource.
          </p>
          <p>
            Nothing changes inside the kernel. It still has values private to a thread, threadgroup memory for the
            group, and buffers for everything else, and sharing a buffer with the CPU does not tell you how fast a kernel
            can read it. Lab 5.1 measures that.
          </p>
        </Translated>
      </>
    ),
    source:
      'not in the book. Apple Metal documentation on storage modes, checked 2026-11-20 (notes/today/ch05-facts.md, C6)',
    vi: {
      source:
        'không có trong sách. Tài liệu Metal của Apple về các chế độ lưu trữ, kiểm tra ngày 2026-11-20 (notes/today/ch05-facts.md, C6)',
    },
  },
};

const OPERANDS = `// both operands already in registers: one instruction
fadd r1, r2, r3

// one operand still in memory: load it into r2 first, then the same add
load r2, r4, offset
fadd r1, r2, r3`;

export function S52() {
  return (
    <>
      <Translated
        vi={
          <p>
            Từ PyTorch bạn đã biết một câu hỏi về chỗ dữ liệu nằm: host hay device, được giải quyết bằng <code>.to(device)</code>.
            Bên trong kernel câu hỏi tinh hơn. Một biến có thể nằm ở một trong năm loại bộ nhớ, và loại bạn khai báo nó vào quyết
            định hai điều cùng lúc: những luồng nào thấy được nó, và một luồng phải chờ bao lâu để dùng nó.
          </p>
        }
      >
        <p>
          From PyTorch you know one question about where data is: host or device, settled with <code>.to(device)</code>.
          Inside a kernel the question is finer. A variable can be in one of five memories, and the one you declare it
          into decides two things at once: which threads can see it, and how long a thread waits to use it.
        </p>
      </Translated>
      <Source section="5.2" pages="98" figs={tr('Fig. 5.1', 'Hình 5.1')} />

      <h2>
        <Translated vi="Năm loại bộ nhớ">Five memories</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bắt đầu với hai loại mà host cũng với tới được. <b>Bộ nhớ toàn cục (global memory)</b> là nơi chứa các mảng bạn trao
            cho kernel. Mọi luồng của mọi khối đều đọc và ghi được nó, host cũng vậy. Nó cũng là loại chậm: DRAM ngoài chip, bộ
            nhớ có băng thông đã giới hạn mọi thứ ở phần 5.1. <b>Bộ nhớ hằng (constant memory)</b> cũng hiển thị với mọi luồng,
            và host đọc và ghi được nó, nhưng mã kernel chỉ đọc được. Đổi lại, thiết bị phục vụ nó với độ trễ thấp và băng thông
            cao. Sách để phần "bằng cách nào" cho chương 7, ở{' '}
            <a href={href('ch07', '7.4')}>phần 7.4, Bộ nhớ hằng và bộ nhớ đệm</a>.
          </p>
        }
      >
        <p>
          Start with the two that the host can also reach. <b>Global memory</b> is where the arrays you hand to a kernel
          are. Every thread of every block can read and write it, and so can the host. It is also the slow one: off-chip
          DRAM, the memory whose bandwidth limited everything in section 5.1. <b>Constant memory</b> is visible to every
          thread too, and the host can read and write it, but kernel code can only read it. In return the device serves
          it with short latency and high bandwidth. The book leaves the how to chapter 7, in{' '}
          <a href={href('ch07', '7.4')}>section 7.4, Constant memory and caching</a>.
        </p>
      </Translated>
      <Source section="5.2" pages="93, 98" />
      <Translated
        vi={
          <p>
            <b>Thanh ghi (register)</b> nằm trên chip, bên trong bộ xử lý. Chúng rất nhanh và mỗi luồng có bộ riêng: không luồng
            nào đọc được thanh ghi của luồng khác. <b>Bộ nhớ chia sẻ (shared memory)</b> cũng nằm trên chip, chậm hơn thanh ghi
            và nhanh hơn nhiều so với bộ nhớ toàn cục. Nó được cấp phát theo khối. Mọi luồng của một khối đều đọc và ghi được bộ
            nhớ chia sẻ của khối mình, và không luồng nào với tới được bộ nhớ chia sẻ của khối khác. Đó là cách các luồng của
            một khối cùng làm việc trên một đầu vào và chuyển kết quả trung gian cho nhau.
          </p>
        }
      >
        <p>
          <b>Registers</b> are on the chip, inside the processor. They are very fast and each thread has its own: no
          thread can read another thread's registers. <b>Shared memory</b> is on the chip as well, slower than a register
          and much faster than global memory. It is allocated per block. Every thread of a block can read and write its
          block's shared memory, and no thread can reach another block's. It is how the threads of a block work on the
          same inputs and pass intermediate results to each other.
        </p>
      </Translated>
      <Source section="5.2" pages="98–99" />
      <Translated
        vi={
          <p>
            Loại thứ năm, <b>bộ nhớ cục bộ (local memory)</b>, là loại có cái tên gây hiểu lầm. Nó là riêng của một luồng, như
            thanh ghi, nhưng không nằm trên chip: nó được đặt trong bộ nhớ toàn cục và tốn ngang bộ nhớ toàn cục. Nó chứa bất
            cứ thứ gì luồng không giữ được trong thanh ghi: các mảng khai báo trong thân kernel, các thanh ghi mà trình biên dịch
            buộc phải tràn ra, và phần còn lại của ngăn xếp lời gọi. "Cục bộ" nói ai thấy được nó, không phải nó nằm ở đâu.
          </p>
        }
      >
        <p>
          The fifth, <b>local memory</b>, is the one whose name misleads. It is private to a thread, like registers, but
          it is not on the chip: it is placed in global memory and costs about what global memory costs. It holds
          whatever a thread cannot keep in registers: arrays declared in the kernel body, registers the compiler had to
          spill, and the rest of the call stack. "Local" says who can see it, not where it is.
        </p>
      </Translated>
      <Source section="5.2" pages="98" />
      <Translated
        vi={
          <p>
            Bản đồ bên dưới có một lưới (grid) gồm hai khối, mỗi khối hai luồng. Trước khi chọn gì, hãy quyết định: luồng 0 và
            luồng 1 của khối 0 cần chuyển một giá trị cho nhau. Những ô nào có thể mang giá trị đó?
          </p>
        }
      >
        <p>
          The map below has one grid of two blocks with two threads each. Before you select anything, decide: thread 0
          and thread 1 of block 0 need to pass a value between them. Which boxes could carry it?
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Bộ nhớ texture bị bỏ qua, như trong chính hình của sách. Bản đồ chỉ cho thấy quyền truy cập. Nó không có bộ nhớ đệm,
              không có kích thước và không có tốc độ, và một khối thật có nhiều hơn hai luồng rất nhiều.
            </p>
          }
        >
          <p>
            Texture memory is left out, as it is in the book's own figure. The map shows permissions only. It has no
            caches, no sizes and no speeds, and a real block has far more than two threads.
          </p>
        </Translated>
      </Simplification>
      <MemoryMap />
      <Translated
        vi={
          <p>
            Hai ô làm được: bộ nhớ chia sẻ của khối 0, và bộ nhớ toàn cục. Thanh ghi thì không, dù cả hai luồng đều có và chúng
            là thứ nhanh nhất trong hình, vì thanh ghi của một luồng đóng kín ngay cả với luồng bên cạnh trong cùng khối. Và lựa
            chọn trên chip dừng ở ranh giới khối: chọn bộ nhớ chia sẻ của khối 1 thì các luồng của khối 0 tối đi.
          </p>
        }
      >
        <p>
          Two boxes could: the shared memory of block 0, and global memory. Registers could not, although both threads
          have them and they are the fastest thing in the picture, because a thread's registers are closed even to its
          neighbour in the same block. And the on-chip option ends at the block boundary: select the shared memory of
          block 1 and the threads of block 0 go dark.
        </p>
      </Translated>

      <h2>
        <Translated vi="Vì sao toán hạng thanh ghi rẻ">Why a register operand is cheap</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các lệnh số học của bộ xử lý lấy toán hạng từ thanh ghi. Nếu cả hai toán hạng đã ở đó, một phép cộng là một lệnh.
            Nếu một trong hai vẫn còn trong bộ nhớ, bộ xử lý trước hết phải thực thi một lệnh nạp đưa nó vào thanh ghi, rồi mới
            cộng được. Hai trường hợp của sách:
          </p>
        }
      >
        <p>
          A processor's arithmetic instructions take their operands from registers. If both operands are already there,
          an addition is one instruction. If one of them is still in memory, the processor first has to execute a load
          that brings it into a register, and only then can it add. The book's two cases:
        </p>
      </Translated>
      <Source section="5.2" pages="99–100" figs={tr('Fig. 5.2', 'Hình 5.2')} />
      <Code
        tabs={[{ label: tr('pseudo-assembly', 'giả hợp ngữ'), code: OPERANDS }]}
        caption={tr(
          "The book's illustration (p. 100), as schematic assembly. The register file is inside the processor; memory is outside it.",
          'Minh họa của sách (tr. 100), dưới dạng hợp ngữ sơ lược. Tệp thanh ghi nằm trong bộ xử lý; bộ nhớ nằm ngoài nó.',
        )}
      />
      <Translated
        vi={
          <p>
            Vậy một giá trị ở lại trong thanh ghi tiết kiệm ba thứ. Nó tiết kiệm một lệnh. Nó tiết kiệm băng thông: truy cập
            thanh ghi không phải lưu lượng bộ nhớ toàn cục, điều này nâng tỉ số FLOP/B của phần 5.1, và sách ước băng thông gộp
            của các tệp thanh ghi trên mọi SM cao hơn của bộ nhớ toàn cục từ hai bậc độ lớn trở lên. Và nó tiết kiệm năng lượng:
            theo con số của sách, đọc một thanh ghi tốn ít hơn đọc bộ nhớ toàn cục ít nhất mười lần. Hãy coi cả hai độ lớn là cận
            dưới thô của sách, không phải số đo của bất kỳ thiết bị nào.
          </p>
        }
      >
        <p>
          So a value that stays in a register saves three things. It saves an instruction. It saves bandwidth: a register
          access is not global-memory traffic, which raises the FLOP/B ratio of section 5.1, and the book puts the
          combined bandwidth of the register files across all SMs at two orders of magnitude or more above that of global
          memory. And it saves energy: by the book's figure, reading a register costs at least ten times less than
          reading global memory. Take both magnitudes as the book's rough lower bounds, not as measurements of any
          device.
        </p>
      </Translated>
      <Source section="5.2" pages="100–101" />
      <Translated
        vi={
          <p>
            Điểm vướng là dung lượng. Một luồng có số thanh ghi giới hạn, và một kernel cần nhiều hơn mức thiết bị có thể cấp
            cho mỗi luồng ở occupancy (mức chiếm dụng) đầy đủ sẽ chạy với ít luồng thường trú hơn. Bạn đã gặp ngân sách đó ở
            chương 4.
          </p>
        }
      >
        <p>
          The catch is capacity. A thread has a limited number of registers, and a kernel that needs more per thread
          than the device can give at full occupancy runs with fewer threads resident. You met that budget in chapter 4.
        </p>
      </Translated>
      <Source section="5.2" pages="101" />

      <h3>
        <Translated vi="Vì sao tệp thanh ghi của GPU lớn đến vậy">Why GPU register files are so large</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Một lõi CPU lưu thanh ghi của một luồng ra bộ nhớ khi chuyển khỏi luồng đó và khôi phục chúng khi chuyển lại. GPU
            không làm cả hai việc đó. Tệp thanh ghi của một khối xử lý giữ thanh ghi của mọi luồng được lập lịch trên nó, mọi lúc,
            nên chuyển từ warp này sang warp khác không phải chép gì. Đây là cơ chế lập lịch không tốn chi phí của chương 4, và
            là lý do tệp thanh ghi của GPU lớn hơn của CPU nhiều. Tệp này cũng có thể chia theo cả hai cách: ít thanh ghi mỗi
            luồng và nhiều luồng hơn, hoặc ngược lại.
          </p>
        }
      >
        <p>
          A CPU core saves a thread's registers to memory when it switches away from the thread and restores them when
          it switches back. A GPU does neither. The register file of a processing block holds the registers of every
          thread scheduled on it, all the time, so moving from one warp to another copies nothing. This is the
          zero-overhead scheduling of chapter 4, and it is the reason a GPU's register file is much larger than a CPU's.
          The file can also be divided either way: fewer registers per thread and more threads, or the reverse.
        </p>
      </Translated>
      <Source section="5.2" pages="99" figs={tr('sidebar on CPU and GPU register architecture', 'khung bên về kiến trúc thanh ghi CPU và GPU')} />

      <h2>
        <Translated vi="Bộ nhớ chia sẻ là một scratchpad">Shared memory is a scratchpad</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bộ nhớ chia sẻ nằm trên chip, nhưng nó là bộ nhớ chứ không phải tệp thanh ghi. Để dùng một giá trị từ đó, bộ xử lý
            vẫn thực thi một lệnh nạp. Lệnh nạp đó có độ trễ thấp hơn và thông lượng cao hơn nhiều so với lệnh nạp từ bộ nhớ
            toàn cục, nhưng độ trễ dài hơn và băng thông ít hơn so với truy cập thanh ghi. Tên kiến trúc cho loại bộ nhớ này là
            scratchpad. Một SM có nhiều đơn vị xử lý, các luồng của một khối được rải trên chúng, và phần cứng cho phép các đơn
            vị đó truy cập bộ nhớ chia sẻ của khối cùng lúc.
          </p>
        }
      >
        <p>
          Shared memory is on the chip, but it is memory and not a register file. To use a value from it the processor
          still executes a load. That load has lower latency and far higher throughput than a load from global memory,
          and longer latency and less bandwidth than a register access. The architectural name for a memory like this is
          a scratchpad. An SM has several processing units, the threads of a block are spread over them, and the hardware
          lets those units access the block's shared memory at the same time.
        </p>
      </Translated>
      <Source section="5.2" pages="101" figs={tr('Fig. 5.3', 'Hình 5.3')} />
      <Translated
        vi={
          <p>
            Một mở rộng, trong một câu: trên các kiến trúc NVIDIA gần đây, các khối có thể được nhóm thành một cluster, và khi đó
            một luồng có thể với tới bộ nhớ chia sẻ của mọi khối trong cluster của nó, điều mà sách gọi là bộ nhớ chia sẻ phân
            tán (distributed shared memory).
          </p>
        }
      >
        <p>
          One extension, in one sentence: on recent NVIDIA architectures blocks can be grouped into a cluster, and a
          thread can then reach the shared memory of every block in its cluster, which the book calls distributed shared
          memory.
        </p>
      </Translated>
      <Source section="5.2" pages="102" />
      <Translated
        vi={
          <p>
            Không có gì trong chương này hay các lab của nó dùng đến nó. Mọi kernel ở đây giữ đúng quy tắc bản đồ đã cho thấy:
            bộ nhớ chia sẻ kết thúc ở ranh giới khối. Nếu sau này gặp cluster, hãy coi chúng là một tính năng cần kiểm tra trên
            chính thiết bị trước mặt bạn. Các header của HIP khai báo cluster group và thuộc tính khởi chạy, nhưng những thiết bị
            AMD nào hỗ trợ chúng thì chúng tôi chưa kiểm tra; Metal không có tài liệu nào về thứ tương tự
            (notes/today/infra-facts.md, "Where the backends really differ").
          </p>
        }
      >
        <p>
          Nothing in this chapter or its labs uses it. Every kernel here keeps to the rule the map showed: shared memory
          ends at the block boundary. If you meet clusters later, treat them as a feature to test for on the device in
          front of you. HIP's headers declare cluster groups and launch attributes, but which AMD devices support them
          we have not checked; Metal documents nothing like them (notes/today/infra-facts.md, "Where the backends really
          differ").
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the CUDA 13.5 guide (notes/today/ch05-facts.md, A6)', '2026-11-20, đối chiếu với hướng dẫn CUDA 13.5 (notes/today/ch05-facts.md, A6)')}>
        <Translated
          vi={
            <p>
              Hướng dẫn của NVIDIA khớp với mốc thời gian của sách: compute capability 9.0 (Hopper) trở lên.{' '}
              <code>cudaDeviceProp.clusterLaunch</code> cho biết có hỗ trợ hay không.
            </p>
          }
        >
          <p>
            NVIDIA's guide agrees with the book's dating: compute capability 9.0 (Hopper) or later.{' '}
            <code>cudaDeviceProp.clusterLaunch</code> reports support.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Phạm vi, thời gian sống và số bản sao">Scope, lifetime and number of copies</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai từ nữa mô tả một biến. <b>Phạm vi</b> của nó là tập các luồng truy cập được nó: một luồng, các luồng của một
            khối, hoặc mọi luồng của mọi lưới. <b>Thời gian sống</b> của nó là giá trị của nó tồn tại bao lâu. Khai báo bên trong
            thân kernel, một biến sống trong một lần khởi chạy: không gì được mang sang lần khởi chạy sau, và mỗi lần khởi chạy
            phải khởi tạo nó lại. Khai báo bên ngoài mọi hàm, nó sống bằng thời gian sống của ứng dụng và mọi kernel đều thấy nó.
          </p>
        }
      >
        <p>
          Two more words describe a variable. Its <b>scope</b> is the set of threads that can access it: one thread, the
          threads of one block, or all threads of all grids. Its <b>lifetime</b> is how long its value survives. Declared
          inside the kernel body, a variable lives for one launch: nothing carries over to the next launch, and every
          launch has to initialise it again. Declared outside any function, it lives as long as the application and
          every kernel can see it.
        </p>
      </Translated>
      <Source section="5.2" pages="102" />
      <Translated
        vi={
          <p>
            Phạm vi cũng cho bạn biết có bao nhiêu bản sao, phần mà người ta hay nhầm. Phạm vi luồng nghĩa là mỗi luồng một bản
            riêng: khởi chạy một triệu luồng thì có một triệu bản. Phạm vi khối nghĩa là mỗi khối một bản. Khai báo quyết định cả
            ba thuộc tính:
          </p>
        }
      >
        <p>
          Scope also tells you how many copies exist, which is the part people get wrong. Thread scope means one private
          copy per thread: launch a million threads and there are a million of them. Block scope means one per block.
          The declaration decides all three properties:
        </p>
      </Translated>
      <Source section="5.2" pages="102" figs={tr('Fig. 5.4', 'Hình 5.4')} />
      <table className="data">
        <thead>
          <tr>
            <th>
              <Translated vi="Thứ bạn khai báo">What you declare</Translated>
            </th>
            <th>
              <Translated vi="Bộ nhớ">Memory</Translated>
            </th>
            <th>
              <Translated vi="Phạm vi">Scope</Translated>
            </th>
            <th>
              <Translated vi="Thời gian sống">Lifetime</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <Translated vi="một biến vô hướng trong thân kernel">a scalar in the kernel body</Translated>
            </td>
            <td>
              <span className="chip reg">
                <Translated vi="thanh ghi">register</Translated>
              </span>
            </td>
            <td>
              <Translated vi="luồng">thread</Translated>
            </td>
            <td>
              <Translated vi="một lần khởi chạy">one launch</Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="một mảng trong thân kernel">an array in the kernel body</Translated>
            </td>
            <td>
              <span className="chip global">
                <Translated vi="cục bộ">local</Translated>
              </span>
            </td>
            <td>
              <Translated vi="luồng">thread</Translated>
            </td>
            <td>
              <Translated vi="một lần khởi chạy">one launch</Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi={<><code>__shared__</code> trong thân kernel</>}>
                <code>__shared__</code> in the kernel body
              </Translated>
            </td>
            <td>
              <span className="chip shared">
                <Translated vi="chia sẻ">shared</Translated>
              </span>
            </td>
            <td>
              <Translated vi="khối">block</Translated>
            </td>
            <td>
              <Translated vi="một lần khởi chạy">one launch</Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi={<><code>__device__</code> bên ngoài mọi hàm</>}>
                <code>__device__</code> outside any function
              </Translated>
            </td>
            <td>
              <span className="chip global">
                <Translated vi="toàn cục">global</Translated>
              </span>
            </td>
            <td>
              <Translated vi="mọi lưới">all grids</Translated>
            </td>
            <td>
              <Translated vi="ứng dụng">application</Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi={<><code>__constant__</code> bên ngoài mọi hàm</>}>
                <code>__constant__</code> outside any function
              </Translated>
            </td>
            <td>
              <span className="chip const">
                <Translated vi="hằng">constant</Translated>
              </span>
            </td>
            <td>
              <Translated vi="mọi lưới">all grids</Translated>
            </td>
            <td>
              <Translated vi="ứng dụng">application</Translated>
            </td>
          </tr>
        </tbody>
      </table>
      <Translated vi={<p>Mỗi hàng có một chú thích đáng biết.</p>}>
        <p>Each row has a footnote worth knowing.</p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              <b>Biến vô hướng.</b> Thanh ghi là trường hợp rẻ, nhưng mỗi biến vô hướng là thêm một thanh ghi mỗi luồng, và quá
              nhiều sẽ tốn occupancy.
            </li>
            <li>
              <b>Mảng.</b> Một mảng trong thân kernel trông như biến trên stack nhưng hành xử như bộ nhớ toàn cục, kể cả độ trễ
              dài. Trình biên dịch có thể giữ một mảng nhỏ trong thanh ghi khi mọi truy cập đều dùng chỉ số hằng. Chương 6 quay lại
              chuyện này ở <a href={href('ch06', '6.6')}>phần 6.6, Trải vòng lặp</a>.
            </li>
            <li>
              <b>
                <code>__shared__</code>.
              </b>{' '}
              Mỗi khối một bản, mọi luồng của khối thấy được, mất đi khi kernel kết thúc. Việc thường gặp của nó là giữ mảnh dữ
              liệu toàn cục mà khối dùng nhiều trong một pha của phép tính. Thường thì thuật toán phải được tổ chức lại mới có được
              các pha như vậy, đó là điều phần 5.3 làm.
            </li>
            <li>
              <b>
                <code>__device__</code> đứng một mình.
              </b>{' '}
              Một biến toàn cục thông thường trong bộ nhớ toàn cục: chậm, mọi kernel thấy được, và tồn tại lâu dài. Nó có thể
              mang dữ liệu từ lần khởi chạy này sang lần sau mà không cần tham số, và sách gọi đó là phong cách tồi.
            </li>
            <li>
              <b>
                <code>__constant__</code>.
              </b>{' '}
              Phải được khai báo bên ngoài mọi hàm, và mã kernel không thay đổi được nó. Nó được lưu trong bộ nhớ toàn cục nhưng
              được đệm (cache), đó là nguồn gốc tốc độ của nó. Sách cho tổng kích thước mọi biến hằng là 65,536 byte "currently".
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            <b>Scalars.</b> Registers are the cheap case, but every scalar is one more register per thread, and too many
            cost occupancy.
          </li>
          <li>
            <b>Arrays.</b> An array in the kernel body looks like a stack variable and behaves like global memory, long
            delays included. The compiler may keep a small one in registers when every access uses a constant index.
            Chapter 6 comes back to this in <a href={href('ch06', '6.6')}>section 6.6, Loop unrolling</a>.
          </li>
          <li>
            <b>
              <code>__shared__</code>.
            </b>{' '}
            One copy per block, visible to all its threads, gone when the kernel finishes. Its usual job is to hold the
            piece of global data that the block uses heavily during one phase of the computation. Often the algorithm
            has to be reorganised to have such phases at all, which is what section 5.3 does.
          </li>
          <li>
            <b>
              <code>__device__</code> alone.
            </b>{' '}
            An ordinary global variable in global memory: slow, visible to every kernel, and persistent. It can carry
            data from one launch to the next without a parameter, and the book calls that bad style.
          </li>
          <li>
            <b>
              <code>__constant__</code>.
            </b>{' '}
            Must be declared outside any function, and kernel code cannot change it. It is stored in global memory but
            cached, which is where its speed comes from. The book gives the total size of all constant variables as
            65,536 bytes "currently".
          </li>
        </ul>
      </Translated>
      <Source section="5.2" pages="102–104" />
      <Translated
        vi={
          <p>
            Một giới hạn mà sách gọi là "hiện tại" là thứ cần truy vấn, không phải để học thuộc. Mọi dung lượng trong chương này
            cũng vậy: bộ nhớ hằng ở đây, bộ nhớ chia sẻ mỗi khối ở phần 5.6, thanh ghi mỗi luồng ở chương 4. Một kernel viết cứng
            một trong số đó chỉ chắc chắn vừa với thiết bị mà nó được viết trên đó.
          </p>
        }
      >
        <p>
          A limit that a book calls current is one to query, not to memorise. The same goes for every capacity in this
          chapter: constant memory here, shared memory per block in section 5.6, registers per thread in chapter 4. A
          kernel that hard-codes one of them is only known to fit the device it was written on.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the CUDA 13.5 guide (notes/today/ch05-facts.md, A4)', '2026-11-20, đối chiếu với hướng dẫn CUDA 13.5 (notes/today/ch05-facts.md, A4)')}>
        <Translated
          vi={
            <p>
              Vẫn là 64 KB, mà hướng dẫn của NVIDIA định nghĩa là 65,536 byte, cho mọi compute capability nó liệt kê. Hướng dẫn
              nói "typically", nên hãy đọc <code>totalConstMem</code> từ <code>cudaDeviceProp</code>.
            </p>
          }
        >
          <p>
            Still 64 KB, which NVIDIA's guide defines as 65,536 bytes, for every compute capability it lists. The guide
            says "typically", so read <code>totalConstMem</code> from <code>cudaDeviceProp</code>.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Nó nằm ở đâu?">Where does it live?</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Giờ hãy áp dụng bảng. Kernel bên dưới làm mượt một tín hiệu: mỗi luồng kết hợp bốn mẫu liên tiếp với bốn trọng số
            rồi nhân kết quả với một hệ số gain thuộc về khối của nó. Với mỗi cái tên trong sáu cái tên, hãy nói nó nằm ở bộ nhớ
            nào, ai truy cập được nó, và có bao nhiêu bản sao trong lần khởi chạy duy nhất ở cuối đoạn mã. Hãy tự tính số lượng.
            Khi một câu trả lời sai, phản hồi sẽ nêu quy tắc bạn đã vấp phải.
          </p>
        }
      >
        <p>
          Now apply the table. The kernel below smooths a signal: each thread combines four consecutive samples with four
          weights and scales the result by a gain that belongs to its block. For each of six names, say which memory it
          is in, who can access it, and how many copies exist during the one launch shown at the bottom. Work the counts
          out. When an answer is wrong, the feedback names the rule you tripped over.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các đáp án theo quy tắc khai báo của sách. Một trình biên dịch thật có thể làm tốt hơn, chẳng hạn giữ một mảng nhỏ
              trong thanh ghi. Kernel là của chúng tôi, theo kiểu bài tập 11 của sách (tr. 120–121), bài có đoạn mã in bị lỗi dàn
              trang.
            </p>
          }
        >
          <p>
            The answers follow the book's rules for declarations. A real compiler may do better, for example by keeping a
            small array in registers. The kernel is ours, in the manner of the book's exercise 11 (pp. 120–121), whose
            printed listing has typesetting errors.
          </p>
        </Translated>
      </Simplification>
      <WhereItLives />
      <Translated vi={<p>Cùng cách đếm đó ở quy mô một lần khởi chạy thật, rồi đến câu hỏi mà phần này được viết ra để dẫn tới.</p>}>
        <p>The same counting at the scale of a real launch, and then the question this section exists to set up.</p>
      </Translated>
      <Ask q={questionsS52['ch05/5.2/q-copies-thread']} />
      <Ask q={questionsS52['ch05/5.2/q-copies-block']} />
      <Ask q={questionsS52['ch05/5.2/q-why-shared']} />

      <h2>
        <Translated vi="Cùng các ý tưởng trong HIP và Metal">The same ideas in HIP and Metal</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mọi thứ đến giờ là CUDA, vì sách là vậy. Các ý tưởng chuyển được sang hai đích còn lại của khóa học này. Cách viết
            thì chỉ chuyển được sang một trong hai.
          </p>
        }
      >
        <p>
          Everything so far is CUDA, because the book is. The ideas carry over to the other two targets of this course.
          The spellings carry over to one of them.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ đề cập CUDA. Bảng được xây từ tài liệu của các hãng đọc ngày 2026-11-20 (notes/today/ch05-facts.md: A1, B2,
              B4, B9, C1, C2, C3, C6, C8), không phải từ việc chạy mã.
            </p>
          }
        >
          <p>
            The book covers CUDA only. The table is built from vendor documentation read on 2026-11-20
            (notes/today/ch05-facts.md: A1, B2, B4, B9, C1, C2, C3, C6, C8), not from running code.
          </p>
        </Translated>
      </Beyond>
      <div className="wide" style={{ maxWidth: '62rem' }}>
        <table className="data" data-compare>
          <thead>
            <tr>
              <th style={{ width: '14%' }}>
                <Translated vi="Khái niệm">Concept</Translated>
              </th>
              <th style={{ width: '28%' }}>
                <Translated vi="CUDA (sách)">CUDA (the book)</Translated>
              </th>
              <th style={{ width: '25%' }}>HIP</th>
              <th>Metal</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row" style={ROW_HEAD}>
                <Translated vi="Bộ nhớ nháp cho một khối">Scratch memory for a block</Translated>
              </th>
              <td>
                <Translated
                  vi={
                    <>
                      <code>__shared__</code> trên một biến trong kernel. Nhóm luồng là một khối.
                    </>
                  }
                >
                  <code>__shared__</code> on a variable in the kernel. The group of threads is a block.
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      <code>__shared__</code>, cùng cách viết và cùng nghĩa: các luồng của một khối thấy được, trong suốt thời gian sống của
                      khối đó.
                    </>
                  }
                >
                  <code>__shared__</code>, same spelling and same meaning: visible to the threads of one block, for the
                  lifetime of that block.
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      Không gian địa chỉ <code>threadgroup</code>: một biến <code>threadgroup</code> trong thân kernel, hoặc một tham số
                      con trỏ <code>threadgroup</code> được đánh dấu <code>[[threadgroup(n)]]</code> mà host đặt kích thước. Nhóm là một
                      threadgroup.
                    </>
                  }
                >
                  The <code>threadgroup</code> address space: a <code>threadgroup</code> variable in the kernel body, or
                  a <code>threadgroup</code> pointer parameter marked <code>[[threadgroup(n)]]</code> that the host
                  sizes. The group is a threadgroup.
                </Translated>
              </td>
            </tr>
            <tr>
              <th scope="row" style={ROW_HEAD}>
                <Translated vi="Rào chắn">Barrier</Translated>
              </th>
              <td>
                <code>__syncthreads()</code>
              </td>
              <td>
                <code>__syncthreads()</code>
              </td>
              <td>
                <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>
              </td>
            </tr>
            <tr>
              <th scope="row" style={ROW_HEAD}>
                <Translated vi="Buffer toàn cục">Global buffers</Translated>
              </th>
              <td>
                <Translated vi="Các tham số con trỏ của kernel trỏ vào bộ nhớ toàn cục của thiết bị.">
                  Pointer parameters of the kernel that point into the device's global memory.
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      Giống hệt. <code>__global__</code> và <code>__device__</code> giữ nguyên cách viết, và llama.cpp dựng backend HIP từ
                      cùng các tệp kernel với backend CUDA.
                    </>
                  }
                >
                  The same. <code>__global__</code> and <code>__device__</code> keep their spelling, and llama.cpp
                  builds its HIP backend from the same kernel files as its CUDA one.
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      Các buffer truyền vào làm đối số kernel, nằm trong thứ mà Metal Shading Language gọi là device memory. Ở phía host
                      mỗi buffer là một <code>MTLBuffer</code>. Các kernel trong lab cho thấy cách khai báo.
                    </>
                  }
                >
                  Buffers passed as kernel arguments, in what the Metal Shading Language calls device memory. On the
                  host each one is an <code>MTLBuffer</code>. The lab kernels show the declaration.
                </Translated>
              </td>
            </tr>
            <tr>
              <th scope="row" style={ROW_HEAD}>
                <Translated vi="Từ host sang device">Host to device</Translated>
              </th>
              <td>
                <Translated
                  vi={
                    <>
                      Một lần sao chép tường minh. Chiều chép được gọi tên bằng <code>cudaMemcpyHostToDevice</code>.
                    </>
                  }
                >
                  An explicit copy. The direction is named by <code>cudaMemcpyHostToDevice</code>.
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      Một lần sao chép tường minh, <code>hipMemcpyHostToDevice</code>.
                    </>
                  }
                >
                  An explicit copy, <code>hipMemcpyHostToDevice</code>.
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      Trên Apple silicon, không có. Một buffer ở chế độ lưu trữ shared, mặc định ở đó, là một vùng cấp phát mà cả hai bộ
                      xử lý truy cập; <code>contents()</code> là con trỏ của CPU tới nó. Sắp thứ tự các truy cập là việc của bạn.
                    </>
                  }
                >
                  On Apple silicon, none. A buffer in shared storage mode, the default there, is one allocation both
                  processors address; <code>contents()</code> is the CPU's pointer to it. Ordering the accesses is your
                  job.
                </Translated>
              </td>
            </tr>
            <tr>
              <th scope="row" style={ROW_HEAD}>
                <Translated vi="Hỏi các giới hạn">Asking for limits</Translated>
              </th>
              <td>
                <Translated
                  vi={
                    <>
                      <code>cudaGetDeviceProperties</code> điền vào một <code>cudaDeviceProp</code>: <code>sharedMemPerBlock</code>,{' '}
                      <code>sharedMemPerMultiprocessor</code>, <code>totalConstMem</code>, <code>totalGlobalMem</code>.
                    </>
                  }
                >
                  <code>cudaGetDeviceProperties</code> fills a <code>cudaDeviceProp</code>:{' '}
                  <code>sharedMemPerBlock</code>, <code>sharedMemPerMultiprocessor</code>, <code>totalConstMem</code>,{' '}
                  <code>totalGlobalMem</code>.
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      <code>hipGetDeviceProperties</code> điền vào một <code>hipDeviceProp_t</code> với cùng tên trường. Thiết bị AMD báo
                      gì trong đó thì chưa được kiểm tra.
                    </>
                  }
                >
                  <code>hipGetDeviceProperties</code> fills a <code>hipDeviceProp_t</code> with the same field names.
                  What AMD devices report in them was not checked.
                </Translated>
              </td>
              <td>
                <Translated
                  vi={
                    <>
                      <code>MTLDevice.maxThreadgroupMemoryLength</code> và <code>MTLDevice.hasUnifiedMemory</code>; theo từng kernel đã
                      biên dịch, <code>staticThreadgroupMemoryLength</code> và <code>maxTotalThreadsPerThreadgroup</code> trên pipeline
                      state.
                    </>
                  }
                >
                  <code>MTLDevice.maxThreadgroupMemoryLength</code> and <code>MTLDevice.hasUnifiedMemory</code>; per
                  compiled kernel, <code>staticThreadgroupMemoryLength</code> and{' '}
                  <code>maxTotalThreadsPerThreadgroup</code> on the pipeline state.
                </Translated>
              </td>
            </tr>
          </tbody>
        </table>
        <YourDevices />
      </div>
      <Translated
        vi={
          <p>
            HIP là cột dễ. Nó giữ cách viết của CUDA cho <code>__shared__</code>, <code>__syncthreads()</code>,{' '}
            <code>__global__</code>, <code>__device__</code>, <code>__constant__</code> và các biến dựng sẵn về chỉ số, nên các
            kernel của chương này, vốn chỉ dùng luồng, khối, bộ nhớ chia sẻ, bộ nhớ hằng và rào chắn (barrier), là kernel HIP
            nguyên trạng. Một khác biệt cần nhớ: trình biên dịch của HIP không ngầm định nghĩa các biến dựng sẵn, nên mã nguồn
            phải include <code>hip/hip_runtime.h</code>. Sự giống nhau dừng lại ở các intrinsic của warp, cooperative groups, các
            header nửa độ chính xác và các intrinsic của matrix core, những thứ mà các chương sau gặp lần lượt. Bản dựng HIP của
            khóa học này là unverified: chưa có máy HIP nào biên dịch nó.
          </p>
        }
      >
        <p>
          HIP is the easy column. It keeps CUDA's spellings for <code>__shared__</code>, <code>__syncthreads()</code>,{' '}
          <code>__global__</code>, <code>__device__</code>, <code>__constant__</code> and the index built-ins, so the
          kernels of this chapter, which use only threads, blocks, shared and constant memory and barriers, are HIP
          kernels as they stand. One difference to remember: HIP's compiler does not define the built-ins implicitly, so the source has
          to include <code>hip/hip_runtime.h</code>. The sameness stops at warp intrinsics, cooperative groups,
          half-precision headers and matrix-core intrinsics, which later chapters meet one at a time. This course's HIP
          build is unverified: no HIP machine has compiled it yet.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Metal đổi bộ từ vựng, và trên Apple silicon nó đổi phần dưới của bản đồ bộ nhớ. Trong bản đồ, host nằm dưới một ranh
            giới, và trong CUDA dữ liệu vượt ranh giới đó bằng một lần sao chép. Trên Apple silicon, CPU và GPU dùng chung bộ nhớ hệ
            thống. Một buffer tạo ở chế độ lưu trữ shared là một vùng cấp phát duy nhất mà cả hai đều truy cập được, nên một lần
            chép từ host sang device không có gì để di chuyển, và không lời gọi nào thay thế nó. Điều còn lại với bạn là thứ tự:
            khi một bên có công việc đã lập lịch làm thay đổi buffer, bên kia không được chạm vào nó cho đến khi công việc đó hoàn
            tất. Trong thực tế, điều đó nghĩa là chờ command buffer chạy xong trước khi CPU đọc kết quả.
          </p>
        }
      >
        <p>
          Metal changes the vocabulary, and on Apple silicon it changes the bottom of the memory map. In the map the
          host sits below a boundary, and in CUDA data crosses that boundary by a copy. On Apple silicon the CPU and the
          GPU share system memory. A buffer created in shared storage mode is a single allocation that both can address, so
          a host-to-device copy has nothing to move, and no call replaces it. What stays with you is ordering: when one
          side has work scheduled that changes the buffer, the other side must not touch it until that work has
          completed. In practice that means waiting for the command buffer to finish before the CPU reads a result.
        </p>
      </Translated>
      <Beyond title={tr("Apple's wording", 'Lời của Apple')}>
        <Translated
          vi={
            <p>
              "Ensure that all changes you schedule on either the CPU or GPU for a resource that uses shared memory complete
              before accessing that resource on the other processor." (Hãy bảo đảm mọi thay đổi bạn lập lịch trên CPU hoặc GPU cho
              một tài nguyên dùng bộ nhớ chung đều hoàn tất trước khi truy cập tài nguyên đó trên bộ xử lý kia.) Cách hiểu theo
              command buffer là của chúng tôi (notes/today/ch05-facts.md, C6).
            </p>
          }
        >
          <p>
            "Ensure that all changes you schedule on either the CPU or GPU for a resource that uses shared memory
            complete before accessing that resource on the other processor." The command-buffer reading of it is ours
            (notes/today/ch05-facts.md, C6).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Phần trên của bản đồ không đổi. Một kernel Metal vẫn chọn giữa bộ nhớ riêng của một luồng, bộ nhớ threadgroup, và
            buffer, và đặc tả Metal Shading Language đưa ra cùng lý do như sách: chia sẻ qua bộ nhớ threadgroup "is faster on most
            devices" so với chia sẻ qua device memory. Unified memory ảnh hưởng thế nào đến chi phí đọc một buffer từ kernel thì
            tài liệu không phân định. Bạn đã đo nó ở <a href={href('ch05', 'labs')}>lab 5.1</a>, hoặc sẽ đo.
          </p>
        }
      >
        <p>
          The top of the map does not change. A Metal kernel still chooses between memory private to a thread,
          threadgroup memory, and buffers, and the Metal Shading Language specification gives the same reason the book
          does: sharing through threadgroup memory "is faster on most devices" than sharing through device memory. What
          unified memory does to the cost of reading a buffer from a kernel is not something the documentation settles.
          You measured it in <a href={href('ch05', 'labs')}>lab 5.1</a>, or will.
        </p>
      </Translated>
      <Ask q={questionsS52['ch05/5.2/q-unified']} />
      <Translated
        vi={
          <p>
            Hãy mang một điều sang phần 5.3. Trong năm loại bộ nhớ, đúng một loại nằm trên chip, kernel ghi được và hiển thị với
            nhiều hơn một luồng. Kernel ngây thơ hoàn toàn không dùng nó.
          </p>
        }
      >
        <p>
          Carry one thing into section 5.3. Of the five memories, exactly one is on the chip, writable by a kernel and
          visible to more than one thread. The naive kernel does not use it at all.
        </p>
      </Translated>
    </>
  );
}
