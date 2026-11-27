import { useState, type MouseEvent } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BIN_LABEL,
  CARDS,
  STRIP_TOP,
  VARIABLES,
  WINDOWS,
  classify,
  judge,
  type Bin,
  type Card,
  type Rules,
} from '../../sim/addrspace';

// Lesson copies of labs/appC/snippets/c1_helper_generic.cu and c1_helper_overloads.metal;
// SC1.snippets.test.ts fails if they drift from the files.
const GENERIC_CU = `#define TILE 256

// A plain pointer: the signature names no memory. Under generic addressing
// the address it is given decides whether these loads read shared or global memory.
__device__ float rowMax(const float* p, int n) {
    float m = p[0];
    for (int i = 1; i < n; ++i) m = p[i] > m ? p[i] : m;
    return m;
}

// One block per row of x (row-major, n columns, 1 <= n <= TILE). Thread 0 takes
// the row's maximum twice: from global memory, and from the block's shared copy.
__global__ void rowMaxTwice(const float* x, float* fromGlobal, float* fromShared, int n) {
    __shared__ float tile[TILE];
    const float* row = x + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) tile[i] = row[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        fromGlobal[blockIdx.x] = rowMax(row, n);    // an address in global memory
        fromShared[blockIdx.x] = rowMax(tile, n);   // an address in the shared window
    }
}

void launchRowMaxTwice(const float* x, float* fromGlobal, float* fromShared, int rows, int n) {
    rowMaxTwice<<<rows, TILE>>>(x, fromGlobal, fromShared, n);
}`;

const OVERLOADS_METAL = `#include <metal_stdlib>
using namespace metal;

#define TILE 256

// MSL has no generic address space: every pointer names its memory, and the
// address space is part of the overload. One definition per memory you call it on.
float rowMax(device const float* p, uint n) {
    float m = p[0];
    for (uint i = 1; i < n; ++i) m = p[i] > m ? p[i] : m;
    return m;
}

float rowMax(threadgroup const float* p, uint n) {
    float m = p[0];
    for (uint i = 1; i < n; ++i) m = p[i] > m ? p[i] : m;
    return m;
}

// One threadgroup per row of x (row-major, n columns, 1 <= n <= TILE). Thread 0
// takes the row's maximum twice: from device memory, and from the threadgroup copy.
kernel void row_max_twice(device const float* x            [[buffer(0)]],
                          device float* fromDevice         [[buffer(1)]],
                          device float* fromThreadgroup    [[buffer(2)]],
                          constant uint& n                 [[buffer(3)]],
                          uint row  [[threadgroup_position_in_grid]],
                          uint t    [[thread_position_in_threadgroup]],
                          uint size [[threads_per_threadgroup]]) {
    threadgroup float tile[TILE];
    device const float* r = x + row * n;
    for (uint i = t; i < n; i += size) tile[i] = r[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        fromDevice[row] = rowMax(r, n);            // the device overload
        fromThreadgroup[row] = rowMax(tile, n);    // the threadgroup overload
    }
}`;

// `p` has no colour on purpose: which memory it reads is decided by the address it is given.
const CU_TIERS: Record<string, Tier> = {
  tile: 'shared', x: 'global', row: 'global', fromGlobal: 'global', fromShared: 'global', m: 'reg', i: 'reg',
};
const METAL_TIERS: Record<string, Tier> = {
  tile: 'shared', x: 'global', r: 'global', fromDevice: 'global', fromThreadgroup: 'global', m: 'reg', i: 'reg', t: 'reg',
};

const hex = (a: number) => '0x' + a.toString(16).padStart(4, '0');
const WINDOW_NAME = {
  get local() {
    return tr('local window', 'cửa sổ local');
  },
  get shared() {
    return tr('shared window', 'cửa sổ shared');
  },
  get global() {
    return tr('outside every window: global', 'nằm ngoài mọi cửa sổ: global');
  },
};

// ---------------------------------------------------------------- explorer: which window?
function WhichWindow() {
  const [addr, setAddr] = useState(VARIABLES[1].addr);
  const k = classify(addr);
  const picked = VARIABLES.find((v) => v.addr === addr);
  const W = 640;
  const H = 100;
  const pad = 12;
  const X = (a: number) => pad + (a / STRIP_TOP) * (W - 2 * pad);
  const onStrip = (ev: MouseEvent<SVGSVGElement>) => {
    const box = ev.currentTarget.getBoundingClientRect();
    const frac = ((ev.clientX - box.left) / box.width) * W;
    const a = Math.round((((frac - pad) / (W - 2 * pad)) * STRIP_TOP) / 4) * 4;
    setAddr(Math.max(0, Math.min(STRIP_TOP - 4, a)));
  };
  return (
    <Bench
      title={tr('Which window?', 'Cửa sổ nào?')}
      task={tr(
        'Pick a variable, or move the address along the generic address space.',
        'Chọn một biến, hoặc di chuyển địa chỉ dọc theo không gian địa chỉ tổng quát (generic).',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="ui muted">
          <Translated vi="Hàm phụ trợ nhận một con trỏ tới">The helper is given a pointer to</Translated>
        </span>
        <span className="seg" role="group" aria-label={tr('Variable', 'Biến')}>
          {VARIABLES.map((v) => (
            <button key={v.id} data-var={v.id} aria-pressed={picked?.id === v.id} onClick={() => setAddr(v.addr)}>
              {v.code}
            </button>
          ))}
        </span>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={tr(
          'The generic address space, with the local and shared windows',
          'Không gian địa chỉ tổng quát, với các cửa sổ local và shared',
        )}
        style={{ width: '100%', maxWidth: W, display: 'block', cursor: 'pointer' }}
        onClick={onStrip}
      >
        <rect x={pad} y={18} width={W - 2 * pad} height={28} fill="var(--global-wash)" stroke="var(--global)" />
        {WINDOWS.map((w) => {
          // Local memory lives in global memory, so its window keeps the global colour
          // (as in chapter 5) and is told apart by a dashed outline and its label.
          const local = w.window === 'local';
          const ink = local ? 'var(--global)' : 'var(--shared)';
          return (
            <g key={w.window} data-window-rect={w.window}>
              <rect
                x={X(w.lo)}
                y={18}
                width={X(w.hi) - X(w.lo)}
                height={28}
                fill={local ? 'var(--global-wash)' : 'var(--shared-wash)'}
                stroke={ink}
                strokeWidth={local ? 2 : 1}
                strokeDasharray={local ? '4 3' : undefined}
              />
              <text x={(X(w.lo) + X(w.hi)) / 2} y={62} fontSize="13" textAnchor="middle" style={{ fill: ink }}>
                {w.window}
              </text>
              {local ? (
                <text x={(X(w.lo) + X(w.hi)) / 2} y={76} fontSize="10" textAnchor="middle" style={{ fill: ink }}>
                  <Translated vi="trong bộ nhớ toàn cục">in global memory</Translated>
                </text>
              ) : null}
            </g>
          );
        })}
        <text x={X(0x4000)} y={62} fontSize="13" textAnchor="middle" style={{ fill: 'var(--global)' }}>
          <Translated vi="global">global</Translated>
        </text>
        <text x={pad} y={62} fontSize="11">0x0000</text>
        <text x={W - pad} y={62} fontSize="11" textAnchor="end">0xffff{/* i18n-ignore: address */}</text>
        <line x1={X(addr)} x2={X(addr)} y1={10} y2={52} stroke="var(--ink)" strokeWidth="2" />
        <polygon points={`${X(addr) - 5},4 ${X(addr) + 5},4 ${X(addr)},11`} fill="var(--ink)" />
        <text x={Math.min(W - 40, Math.max(40, X(addr)))} y={95} fontSize="13" textAnchor="middle" fontWeight="600">
          {hex(addr)}
        </text>
      </svg>
      <label className="field ui" style={{ marginTop: '0.5rem', width: '100%', maxWidth: W }}>
        <Translated vi={<>Địa chỉ {hex(addr)}</>}>Address {hex(addr)}</Translated>
        <input type="range" min={0} max={STRIP_TOP - 4} step={4} value={addr} aria-label={tr('address', 'địa chỉ')} onChange={(e) => setAddr(+e.target.value)} />
      </label>
      <div className="narrate" data-window={k.window} style={{ marginTop: '0.7rem' }}>
        <p style={{ margin: 0 }}>
          {picked ? (
            <>
              <Translated
                vi={
                  <>
                    <code>{picked.code}</code>, {picked.label}, nằm ở <code>{hex(addr)}</code>.{' '}
                  </>
                }
              >
                <code>{picked.code}</code>, {picked.label}, is at <code>{hex(addr)}</code>.{' '}
              </Translated>
            </>
          ) : (
            <>
              <Translated
                vi={
                  <>
                    Địa chỉ <code>{hex(addr)}</code>.{' '}
                  </>
                }
              >
                Address <code>{hex(addr)}</code>.{' '}
              </Translated>
            </>
          )}
          <Translated
            vi={
              <>
                Nó rơi {k.window === 'global' ? '' : 'vào '}
                <span className={'chip ' + k.tier}>{WINDOW_NAME[k.window]}</span>.
              </>
            }
          >
            It falls {k.window === 'global' ? '' : 'in the '}
            <span className={'chip ' + k.tier}>{WINDOW_NAME[k.window]}</span>.
          </Translated>
          {k.window === 'local' ? (
            <>
              {' '}
              <Translated
                vi={
                  <>
                    Bộ nhớ cục bộ là riêng của một luồng nhưng nằm trong bộ nhớ toàn cục, nên nó giữ màu của global, như ở{' '}
                    <a href={href('ch05', '5.2')}>chương 5</a>.
                  </>
                }
              >
                Local memory is private to one thread but lives in global memory, so it keeps the global colour, as in{' '}
                <a href={href('ch05', '5.2')}>chapter 5</a>.
              </Translated>
            </>
          ) : null}
        </p>
        <Translated
          vi={
            <p style={{ margin: '0.35rem 0 0' }}>
              Lệnh nạp của hàm phụ trợ là một lệnh <code>ld</code> tổng quát, không có state space trong lệnh. Với địa chỉ
              này, nó đi tới state space <code data-state>{k.state}</code>, như <code data-load>{k.load}</code> sẽ làm.
            </p>
          }
        >
          <p style={{ margin: '0.35rem 0 0' }}>
            The helper's load is a generic <code>ld</code> with no state space in the instruction. For this address it
            goes to the <code data-state>{k.state}</code> state space, as <code data-load>{k.load}</code> would.
          </p>
        </Translated>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- classification
const LANG_LABEL = { cuda: 'CUDA', hip: 'HIP', msl: 'MSL' };
const BINS: Bin[] = ['global', 'shared', 'error'];
const BIN_SHORT: Record<Bin, string> = {
  get global() {
    return tr('reads global / device', 'đọc global / device');
  },
  get shared() {
    return tr('reads shared / threadgroup', 'đọc shared / threadgroup');
  },
  get error() {
    return tr('does not compile', 'không biên dịch được');
  },
};

function CardBox({ card, rules, pick, onPick }: { card: Card; rules: Rules; pick: Bin | undefined; onPick: (b: Bin) => void }) {
  const o = judge(card, rules);
  const done = pick !== undefined;
  const right = pick === o.bin;
  return (
    <div className="panel" data-card={card.id} data-picked={pick ?? ''} data-result={done ? (right ? 'right' : 'wrong') : ''}>
      <p className="ui" style={{ margin: '0 0 0.3rem', fontSize: '0.8rem' }}>
        <b>{LANG_LABEL[card.lang]}</b>
      </p>
      <pre className="mono" style={{ margin: '0 0 0.4rem', fontSize: '0.76rem', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
        {card.helper}
      </pre>
      <pre className="mono muted" style={{ margin: '0 0 0.6rem', fontSize: '0.76rem', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
        {card.call}
      </pre>
      <div className="row" style={{ gap: '0.35rem' }}>
        {BINS.map((b) => (
          <button key={b} data-bin={b} aria-pressed={pick === b} className="quiet" style={{ fontSize: '0.78rem', padding: '0.2rem 0.45rem' }} onClick={() => onPick(b)}>
            {BIN_SHORT[b]}
          </button>
        ))}
      </div>
      {done ? (
        <p className="ui" style={{ margin: '0.55rem 0 0', fontSize: '0.82rem' }} data-feedback>
          <span className={'chip ' + (right ? 'ok' : 'fail')}>{right ? tr('right', 'đúng') : tr('not this bin', 'không phải ô này')}</span>{' '}
          {right ? null : (
            <Translated vi={<>Nó {o.bin === 'error' ? 'không biên dịch được' : BIN_LABEL[o.bin].replace('compiles, ', 'compiles and ')}. </>}>
              <>It {o.bin === 'error' ? 'does not compile' : BIN_LABEL[o.bin].replace('compiles, ', 'compiles and ')}. </>
            </Translated>
          )}
          {o.rule} <span className="muted">({o.source})</span>
        </p>
      ) : null}
    </div>
  );
}

function SortCards() {
  const [rules, setRules] = useState<Rules>('spec');
  const [picks, setPicks] = useState<Record<string, Bin>>({});
  const answered = CARDS.filter((c) => picks[c.id] !== undefined);
  const right = answered.filter((c) => picks[c.id] === judge(c, rules).bin);
  return (
    <Bench
      title={tr('Does it compile, and which memory does it read?', 'Nó có biên dịch được không, và nó đọc bộ nhớ nào?')}
      task={tr('Sort each card into one of the three bins.', 'Xếp mỗi thẻ vào một trong ba ô.')}
      foot={
        <Translated
          vi={
            <>
              Các quy tắc của đặc tả là MSL 4.1 mục 1.5.1 và 4 như được trích trong appC-facts C1; trình biên dịch Metal trên
              máy build đồng ý với mọi thẻ khi được yêu cầu MSL 4.0 trở về trước. Ở mặc định của nó, MSL 4.1, nó bất đồng
              ở một thẻ (labs/appC/snippets/check.py biên dịch mỗi thẻ MSL theo cả hai cách). Không có gì ở đây được biên
              dịch cho GPU AMD.
            </>
          }
        >
          <>
            The specification rules are MSL 4.1 sections 1.5.1 and 4 as quoted in appC-facts C1; the Metal compiler on
            the build machine agreed with every card when asked for MSL 4.0 or earlier. At its default, MSL 4.1, it
            disagreed on one card (labs/appC/snippets/check.py compiles each MSL card both ways). Nothing here was
            compiled for an AMD GPU.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="ui muted">
          <Translated vi="Quy tắc Metal:">Metal rules:</Translated>
        </span>
        <span className="seg" role="group" aria-label={tr('Metal rules', 'Quy tắc Metal')}>
          <button aria-pressed={rules === 'spec'} data-rules="spec" onClick={() => setRules('spec')}>
            <Translated vi="đặc tả (và -std=metal4.0)">the specification (and -std=metal4.0)</Translated>
          </button>
          <button aria-pressed={rules === 'observed-4.1'} data-rules="observed-4.1" onClick={() => setRules('observed-4.1')}>
            <Translated vi="mặc định của trình biên dịch ở đây, MSL 4.1 (quan sát được)">
              the compiler's default here, MSL 4.1 (observed)
            </Translated>
          </button>
        </span>
        <span className="ui" data-score>
          <Translated vi={<>{right.length} trên {CARDS.length} thẻ xếp đúng</>}>
            {right.length} of {CARDS.length} sorted correctly
          </Translated>
        </span>
      </div>
      <div className="cols">
        {CARDS.map((c) => (
          <CardBox key={c.id} card={c} rules={rules} pick={picks[c.id]} onPick={(b) => setPicks({ ...picks, [c.id]: b })} />
        ))}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC1: Record<string, Question> = {
  'appC/C.1/q-who-decides': {
    kind: 'choice',
    id: 'appC/C.1/q-who-decides',
    prompt: (
      <Translated
        vi={
          <>
            Một hàm phụ trợ CUDA nhận <code>const float* p</code> và nạp <code>p[i]</code>. Một lời gọi truyền vào một tile{' '}
            <code>__shared__</code>, lời gọi kia truyền một mảng global. Điều gì khiến lệnh nạp của lời gọi thứ nhất đọc bộ
            nhớ chia sẻ?
          </>
        }
      >
        <>
          A CUDA helper takes <code>const float* p</code> and loads <code>p[i]</code>. One call passes a{' '}
          <code>__shared__</code> tile, another a global array. What makes the first call's load read shared memory?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Địa chỉ: nó nằm trong cửa sổ shared của không gian địa chỉ tổng quát.">
            The address: it lies in the shared window of the generic address space.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Trình biên dịch sinh một bản riêng của hàm phụ trợ cho mỗi loại bộ nhớ, và nơi gọi chọn bản shared.">
            The compiler emits a separate copy of the helper for each memory, and the call site picks the shared copy.
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì đòi hỏi điều đó. Một hàm đã biên dịch, với một lệnh nạp tổng quát, xử lý được cả hai.">
            Nothing requires that. One compiled function, with one generic load, handles both.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Driver kiểm tra con trỏ lúc khởi chạy và vá lại hàm phụ trợ.">
            The driver checks the pointer at launch and patches the helper.
          </Translated>
        ),
        why: (
          <Translated vi="Hàm phụ trợ thậm chí không được gọi lúc khởi chạy; quyết định được đưa ra cho từng lệnh nạp, bởi địa chỉ.">
            The helper is not even called at launch; the decision is made per load, by the address.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một phép ép kiểu đối số sang __shared__ float* tại lời gọi; không có nó thì lệnh nạp đọc global.">
            A cast of the argument to __shared__ float* at the call; without it the load reads global.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là tình trạng trước Fermi, khi một con trỏ chỉ có thể trỏ vào bộ nhớ toàn cục.">
            That was the situation before Fermi, when a pointer could point only into global memory.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Định địa chỉ tổng quát (generic addressing): một lệnh nạp không có state space trong lệnh sẽ đọc bất kỳ bộ nhớ
            nào mà địa chỉ của nó nằm trong. Bộ nhớ chia sẻ và bộ nhớ cục bộ là các cửa sổ trong một không gian địa chỉ, và
            mọi thứ nằm ngoài chúng là global.
          </p>
        }
      >
        <p>
          Generic addressing: a load with no state space in the instruction reads whatever memory its address lies in.
          Shared and local memory are windows in one address space, and everything outside them is global.
        </p>
      </Translated>
    ),
    source: 'book p. 624; appC-facts A2',
    vi: { source: 'sách p. 624; appC-facts A2' },
  },
  'appC/C.1/q-overloads': {
    kind: 'numeric',
    id: 'appC/C.1/q-overloads',
    prompt: (
      <Translated
        vi={
          <>
            Bạn chuyển sang MSL một hàm phụ trợ mà kernel gọi cả trên một tile <code>threadgroup</code> lẫn trên một buffer{' '}
            <code>device</code>. Nó phải biên dịch được dưới mọi phiên bản MSL, kể cả 4.0 trở về trước. Số định nghĩa hàm
            ít nhất bạn cần là bao nhiêu?
          </>
        }
      >
        <>
          You port to MSL a helper that a kernel calls both on a <code>threadgroup</code> tile and on a{' '}
          <code>device</code> buffer. It must compile under every MSL version, 4.0 and earlier included. What is the
          smallest number of function definitions you need?
        </>
      </Translated>
    ),
    answer: 2,
    unit: 'definitions',
    explain: (
      <Translated
        vi={
          <p>
            Một hàm nhận <code>device const float*</code> và một hàm nhận <code>threadgroup const float*</code>. MSL mở rộng
            nạp chồng (overloading) để bao gồm cả address space của đối số, nên hai hàm có thể dùng chung một tên và mỗi
            lời gọi chọn hàm của mình. Trong CUDA và HIP, đáp án là 1.
          </p>
        }
      >
        <p>
          One taking <code>device const float*</code> and one taking <code>threadgroup const float*</code>. MSL extends
          overloading to include the address space of an argument, so the two can share a name and each call picks its
          own. In CUDA and HIP the answer is 1.
        </p>
      </Translated>
    ),
    source: 'MSL specification §1.5.1 and §4, appC-facts C1; compiled on the build machine with -std=metal4.0',
    vi: {
      unit: 'định nghĩa',
      source: 'đặc tả MSL §1.5.1 và §4, appC-facts C1; biên dịch trên máy build với -std=metal4.0',
    },
  },
  'appC/C.1/q-missing-space': {
    kind: 'choice',
    id: 'appC/C.1/q-missing-space',
    prompt: (
      <Translated
        vi={
          <>
            MSL, biên dịch với <code>-std=metal4.0</code>: <code>{'float rowMax(const float* p, uint n) { ... }'}</code>.
            Điều gì xảy ra?
          </>
        }
      >
        <>
          MSL, compiled with <code>-std=metal4.0</code>: <code>{'float rowMax(const float* p, uint n) { ... }'}</code>.
          What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Lỗi biên dịch: con trỏ không nêu address space nào.">
            A compile error: the pointer names no address space.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó biên dịch được, và p mặc định là device, address space của các đối số buffer của kernel.">
            It compiles, and p defaults to device, the address space of kernel buffer arguments.
          </Translated>
        ),
        why: (
          <Translated vi='Không có mặc định trong MSL 4.0: trình biên dịch báo "pointer type must have explicit address space qualifier".'>
            There is no default in MSL 4.0: the compiler says "pointer type must have explicit address space qualifier".
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó biên dịch được, và p mặc định là thread, giống một biến cục bộ không có qualifier.">
            It compiles, and p defaults to thread, like an unqualified local variable.
          </Translated>
        ),
        why: (
          <Translated vi='Không có mặc định trong MSL 4.0: trình biên dịch báo "pointer type must have explicit address space qualifier".'>
            There is no default in MSL 4.0: the compiler says "pointer type must have explicit address space qualifier".
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó biên dịch được, và p trở thành generic, như trong CUDA, nên một hàm phụ trợ phục vụ mọi address space.">
            It compiles, and p becomes generic, as in CUDA, so one helper serves every address space.
          </Translated>
        ),
        why: (
          <Translated vi="Không phải dưới 4.0. Ở mặc định của nó, MSL 4.1, trình biên dịch trên máy build đã chấp nhận nó như generic; đó là điều quan sát được, không có trong tài liệu (appC-facts C1).">
            Not under 4.0. At its default, MSL 4.1, the compiler on the build machine did accept it as generic; that is
            observed, not documented (appC-facts C1).
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đặc tả viết: "If an address space attribute is missing on a pointer or reference type declaration, a
        compilation error occurs." (nếu khai báo kiểu con trỏ hoặc tham chiếu thiếu thuộc tính address space thì sẽ có
            lỗi biên dịch). Trình biên dịch trên máy build áp dụng điều này cho tới MSL 4.0.
          </p>
        }
      >
        <p>
        The specification: "If an address space attribute is missing on a pointer or reference type declaration, a
        compilation error occurs." The compiler on the build machine enforces this up to MSL 4.0.
        </p>
      </Translated>
    ),
    source: 'MSL specification §4, appC-facts C1; compiled on the build machine',
    vi: { source: 'đặc tả MSL §4, appC-facts C1; biên dịch trên máy build' },
  },
  'appC/C.1/q-speed': {
    kind: 'choice',
    id: 'appC/C.1/q-speed',
    prompt: (
      <Translated
        vi={
          <>
            Cùng một hàm phụ trợ CUDA, với cùng n, được gọi một lần trên dữ liệu shared và một lần trên dữ liệu global. Thời
            gian có như nhau không?
          </>
        }
      >
        <>The same CUDA helper, with the same n, is called once on shared data and once on global data. Same time?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không: cùng các lệnh, nhưng lệnh nạp đi tới các bộ nhớ khác nhau, và shared nhanh hơn.">
            No: the same instructions, but the loads go to different memories, and shared is faster.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có: định địa chỉ tổng quát che đi việc con trỏ trỏ tới bộ nhớ nào, nên hai lời gọi tốn như nhau.">
            Yes: generic addressing hides which memory a pointer refers to, so both calls cost the same.
          </Translated>
        ),
        why: <Translated vi="Nó che đi cách viết, không che đi chi phí.">It hides the spelling, not the cost.</Translated>,
      },
      {
        label: (
          <Translated vi="Có: hai lời gọi chạy cùng mã đã biên dịch với cùng các lệnh nạp tổng quát, nên mất cùng thời gian.">
            Yes: both calls run the same compiled code with the same generic loads, so they take the same time.
          </Translated>
        ),
        why: (
          <Translated vi="Cùng các lệnh, nhưng phía sau lệnh nạp là các bộ nhớ khác nhau.">
            Same instructions, different memories behind the loads.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó phụ thuộc vào n, số lệnh nạp, chứ không phụ thuộc vào nơi chúng đi tới.">
            It depends on n, the number of loads, not on where they go.
          </Translated>
        ),
        why: (
          <Translated vi="n quyết định có bao nhiêu lệnh nạp; bộ nhớ quyết định mỗi lệnh tốn bao nhiêu.">
            n sets how many loads there are; the memory sets what each one costs.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Việc đặt dữ liệu vẫn là việc của bạn, và phép tối ưu của chương 5 cũng vậy: đưa những gì một khối dùng lại vào
            bộ nhớ chia sẻ, rồi để một hàm phụ trợ đọc nó ở đó. Các trang của NVIDIA không đưa ra con số cho khác biệt này.
          </p>
        }
      >
        <p>
          Placement stays your job, and so does the optimisation of chapter 5: stage what a block reuses in shared memory,
          then let one helper read it there. NVIDIA's pages do not put a number on the difference.
        </p>
      </Translated>
    ),
    source: 'book p. 624; appC-facts A2',
    vi: { source: 'sách p. 624; appC-facts A2' },
  },
};

// ---------------------------------------------------------------- the page
export function SC1() {
  return (
    <>
      <Translated
        vi={
          <p>
            Trên những GPU CUDA đầu tiên, bộ nhớ chia sẻ (shared memory), bộ nhớ cục bộ (local memory) và bộ nhớ toàn cục
            (global memory) là ba không gian địa chỉ riêng biệt, mỗi cái có lệnh nạp và ghi riêng. Một con trỏ chỉ có thể
            trỏ vào bộ nhớ toàn cục. Vì vậy một hàm device nhận con trỏ cần một phiên bản riêng cho mỗi loại bộ nhớ mà nó có
            thể nhận, cho từng đối số con trỏ.
          </p>
        }
      >
        <p>
          On the first CUDA GPUs, shared memory, local memory and global memory were three separate address spaces, each
          with its own load and store instructions. A pointer could point only into global memory. A device function that
          took a pointer therefore needed a separate version for each kind of memory it might be given, for each pointer
          argument.
        </p>
      </Translated>
      <Source section="C.1" pages="624" />
      <Translated
        vi={
          <p>
            Fermi gộp ba không gian đó thành một không gian địa chỉ, truy cập bằng một bộ lệnh nạp và ghi. Bạn quyết định bộ
            nhớ nào chứa dữ liệu một lần, khi khai báo hoặc cấp phát nó, và không bao giờ phải quyết định lại. Một hàm nhận
            con trỏ chấp nhận dữ liệu ở bất kỳ bộ nhớ nào trong ba, nên mã device kết hợp được với nhau theo cách C++ vẫn làm
            ở nơi khác. Sách nói thêm rằng điều này đã hạ chi phí xây dựng các thư viện CUDA chất lượng sản phẩm và mang hỗ
            trợ con trỏ C và C++ đầy đủ tới kernel.
          </p>
        }
      >
        <p>
          Fermi put the three into one address space, reached by one set of load and store instructions. You decide which
          memory holds the data once, when you declare or allocate it, and never again. A function that takes a pointer
          accepts data in any of the three, so device code composes the way C++ does elsewhere. The book adds that this
          lowered the cost of building production-quality CUDA libraries and brought full C and C++ pointer support to
          kernels.
        </p>
      </Translated>
      <Source section="C.1" pages="624" />
      <Translated
        vi={
          <p>
            Điều mà một không gian địa chỉ không làm được là khiến các bộ nhớ nhanh như nhau. Theo lời sách, lời gọi nhanh hơn
            khi con trỏ trỏ tới bộ nhớ chia sẻ và chậm hơn khi nó trỏ tới bộ nhớ toàn cục. Quyết định dữ liệu nằm ở đâu, và
            đưa nó vào bộ nhớ chia sẻ như chương 5 đã làm, vẫn là việc của bạn.
          </p>
        }
      >
        <p>
          What one address space does not do is make the memories equally fast. In the book's words, the call is faster
          when the pointer refers to shared memory and slower when it refers to global memory. Deciding where data lives,
          and staging it in shared memory as chapter 5 did, is still your job.
        </p>
      </Translated>
      <Today
        checked={tr(
          "2026-11-20 against NVIDIA's Fermi whitepaper and the PTX ISA (notes/today/appC-facts.md, A2)",
          'ngày 2026-11-20, đối chiếu với whitepaper Fermi của NVIDIA và PTX ISA (notes/today/appC-facts.md, A2)',
        )}
      >
        <Translated
          vi={
            <p>
              Đây là định địa chỉ tổng quát của PTX (PTX 2.0, <code>sm_20</code> trở lên). <code>.shared</code>,{' '}
              <code>.local</code>, <code>.const</code> và <code>.param</code> là các cửa sổ trong một không gian tổng quát;
              một địa chỉ nằm ngoài mọi cửa sổ là global. <code>cvta</code> chuyển đổi giữa địa chỉ tổng quát và địa chỉ
              state space, còn <code>isspacep</code> kiểm tra một địa chỉ nằm trong cửa sổ nào. Whitepaper ra năm 2009;
              NVIDIA nói Fermi lên kệ vào tháng 3 năm 2010. NVIDIA không định lượng khác biệt về tốc độ.
            </p>
          }
        >
          <p>
            This is PTX's generic addressing (PTX 2.0, <code>sm_20</code> and later). <code>.shared</code>,{' '}
            <code>.local</code>, <code>.const</code> and <code>.param</code> are windows in one generic space; an address
            outside every window is global. <code>cvta</code> converts between generic and state-space addresses, and{' '}
            <code>isspacep</code> tests which window an address is in. The whitepaper is from 2009; NVIDIA says Fermi
            reached the market in March 2010. NVIDIA does not quantify the speed difference.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Cửa sổ nào?">Which window?</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đây là ý nghĩa của điều đó với một lệnh nạp. Lệnh nạp của hàm phụ trợ không nêu bộ nhớ nào. Hãy chọn một biến,
            hoặc di chuyển địa chỉ dọc theo dải, và xem địa chỉ rơi vào cửa sổ nào và do đó lệnh nạp tới bộ nhớ nào.
          </p>
        }
      >
        <p>
          Here is what that means for one load. The helper's load instruction names no memory. Pick a variable, or move
          the address along the strip, and see which window the address falls in and which memory the load therefore
          reaches.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Vị trí và kích thước các cửa sổ được bịa ra cho hình minh họa; giá trị thật do phần cứng đặt và không có trong
              các nguồn của chúng tôi. Một chiều, không có bộ nhớ đệm, và các cửa sổ <code>.const</code> và{' '}
              <code>.param</code> bị lược đi.
            </p>
          }
        >
          <p>
            Window positions and sizes are invented for the picture; the real ones are set by the hardware and not given
            in our sources. One dimension, no caches, and the <code>.const</code> and <code>.param</code> windows left
            out.
          </p>
        </Translated>
      </Simplification>
      <WhichWindow />
      <Translated
        vi={
          <p>
            Địa chỉ, chứ không phải mã nguồn, chọn bộ nhớ. Đó là lý do một <code>rowMax</code> đã biên dịch phục vụ được một
            hàng global, một tile shared và một mảng cục bộ của một luồng.
          </p>
        }
      >
        <p>
          The address, not the source code, picks the memory. That is why one compiled <code>rowMax</code> serves a global
          row, a shared tile and a thread's local array.
        </p>
      </Translated>
      <Ask q={questionsSC1['appC/C.1/q-who-decides']} />

      <h2>
        <Translated vi="HIP: cùng một cách sắp xếp">HIP: the same arrangement</Translated>
      </h2>
      <Translated
        vi={
          <p>
            GPU AMD hoạt động theo cùng cách dưới những cái tên khác. Trình biên dịch mà HIP dùng có tài liệu về một không
            gian địa chỉ "flat" tổng quát với hai khe (aperture), một cho LDS (tên AMD gọi bộ nhớ chia sẻ) và một cho scratch
            (bộ nhớ riêng). Một lệnh nạp flat tới bộ nhớ toàn cục, LDS hoặc scratch tùy theo địa chỉ của nó nằm trong dải
            nào. Vì vậy một hàm phụ trợ nhận <code>const float*</code> chạy được trong HIP mà không cần sửa. Một điều kiện
            từ cùng tài liệu đó: chỉ không gian địa chỉ 64-bit được hỗ trợ cho target <code>amdgcn</code>.
          </p>
        }
      >
        <p>
          AMD GPUs work the same way under other names. The compiler HIP uses documents a generic "flat" address space
          with two apertures, one for LDS (AMD's name for shared memory) and one for scratch (private memory). A flat load
          reaches global memory, LDS or scratch depending on which range its address is in. So a helper that takes{' '}
          <code>const float*</code> works unchanged in HIP. One condition from the same document: only 64-bit address
          spaces are supported for the <code>amdgcn</code> target.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Nguồn là tài liệu LLVM AMDGPU, không phải tài liệu của chính HIP, vốn không bàn tới điều này (appC-facts B1).
            </p>
          }
        >
          <p>
            The source is the LLVM AMDGPU documentation, not HIP's own, which does not discuss this (appC-facts B1).
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Metal: mỗi address space một hàm phụ trợ">Metal: one helper per address space</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đặc tả Metal Shading Language đi theo hướng ngược lại. Nó mô tả <code>device</code>, <code>constant</code>,{' '}
            <code>thread</code> và <code>threadgroup</code> là các address space tách rời và đòi mọi con trỏ và tham chiếu
            phải nêu một trong số đó; thiếu address space là lỗi biên dịch. Điều làm cho cách này khả thi là nạp chồng. MSL
            đưa address space của đối số vào quy tắc nạp chồng, nên bạn viết <code>rowMax</code> một lần cho{' '}
            <code>device const float*</code> và một lần cho <code>threadgroup const float*</code>, và mỗi lời gọi chọn hàm
            của mình. Đó chính là cách sắp xếp mỗi bộ nhớ một phiên bản mà sách nói Fermi đã loại bỏ. Thêm một quy tắc: một
            hàm phụ trợ chỉ được nhận con trỏ <code>threadgroup</code> nếu nó được một kernel gọi.
          </p>
        }
      >
        <p>
          The Metal Shading Language specification goes the other way. It describes <code>device</code>,{' '}
          <code>constant</code>, <code>thread</code> and <code>threadgroup</code> as disjoint address spaces and requires
          every pointer and reference to name one; a missing address space is a compile error. What makes this workable is
          overloading. MSL includes an argument's address space in the overloading rules, so you write{' '}
          <code>rowMax</code> once for <code>device const float*</code> and once for <code>threadgroup const float*</code>,
          and each call picks its own. That is exactly the one-version-per-memory arrangement the book says Fermi removed.
          One more rule: a helper may take a <code>threadgroup</code> pointer only if a kernel calls it.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Đặc tả MSL 4.1, mục 1.5.1 và 4, như được trích trong appC-facts C1.</p>}>
          <p>MSL specification 4.1, sections 1.5.1 and 4, as quoted in appC-facts C1.</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Có một điều đặc tả không nói với bạn. Trên máy dùng để xây dựng khóa học này, trình biên dịch Metal của Apple ở
            phiên bản ngôn ngữ mặc định, MSL 4.1, đã chấp nhận <code>rowMax(const float* p, uint n)</code> không có address
            space, và một kernel gọi nó trên bộ nhớ threadgroup, device và thread đã trả về đúng các giá trị lớn nhất trên
            GPU. Khi được yêu cầu MSL 4.0 trở về trước bằng <code>-std=metal4.0</code>, cùng trình biên dịch đó từ chối dòng
            này: "pointer type must have explicit address space qualifier". Đặc tả nói thiếu address space là lỗi biên
            dịch, nên hãy coi đây là điều quan sát được, không có trong tài liệu, và hãy viết các bản nạp chồng. Chúng biên
            dịch được dưới mọi phiên bản.
          </p>
        }
      >
        <p>
          One thing the specification does not tell you. On the machine this course was built on, Apple's Metal compiler
          at its default language version, MSL 4.1, accepted <code>rowMax(const float* p, uint n)</code> with no address
          space, and a kernel that called it on threadgroup, device and thread memory returned the right maxima on the GPU.
          Asked for MSL 4.0 or earlier with <code>-std=metal4.0</code>, the same compiler rejects the line: "pointer type
          must have explicit address space qualifier". The specification says a missing address space is a compile error,
          so treat this as observed, not documented, and write the overloads. They compile under every version.
        </p>
      </Translated>
      <Beyond title={tr('Observed on the build machine', 'Quan sát trên máy build')}>
        <Translated
          vi={
            <p>
              Được ghi trong appC-facts C1 là quan sát được, không có trong tài liệu: Apple metal 32023.921, macOS 27 SDK,
              2026-11-20. <code>labs/appC/snippets/check.py</code> lặp lại các lần biên dịch và lần chạy.
            </p>
          }
        >
          <p>
            Recorded in appC-facts C1 as observed, not documented: Apple metal 32023.921, macOS 27 SDK, 2026-11-20.{' '}
            <code>labs/appC/snippets/check.py</code> repeats the compiles and the run.
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Nó có biên dịch được không, và nó đọc bộ nhớ nào?">
          Does it compile, and which memory does it read?
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Tám nơi gọi trong CUDA, HIP và MSL. Với mỗi nơi, hãy quyết định nó có biên dịch được không và, nếu có, lệnh nạp
            của hàm phụ trợ đọc bộ nhớ nào. Công tắc ở trên cùng đổi quy tắc Metal từ đặc tả sang những gì trình biên dịch ở
            đây đã làm ở mặc định của nó. Trước khi bật nó, hãy dự đoán những thẻ nào sẽ đổi chỗ.
          </p>
        }
      >
        <p>
          Eight call sites in CUDA, HIP and MSL. For each, decide whether it compiles and, if it does, which memory the
          helper's loads read. The switch at the top changes the Metal rules from the specification to what the compiler
          here did at its default. Before you flip it, predict which cards will move.
        </p>
      </Translated>
      <SortCards />
      <Translated
        vi={
          <p>
            CUDA và HIP cần một hàm phụ trợ; Metal, theo đặc tả của nó, cần mỗi address space một hàm. Một bản nạp chồng sai
            không phải là đường chạy chậm hay một phép chuyển đổi âm thầm: nó là lỗi biên dịch.
          </p>
        }
      >
        <p>
          CUDA and HIP need one helper; Metal, by its specification, needs one per address space. A wrong overload is not
          a slow path or a silent conversion: it is a compile error.
        </p>
      </Translated>
      <Ask q={questionsSC1['appC/C.1/q-overloads']} />
      <Ask q={questionsSC1['appC/C.1/q-missing-space']} />

      <h2>
        <Translated vi="Hàm phụ trợ, trong cả hai ngôn ngữ">The helper, in both languages</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đây là hàm phụ trợ với hai nơi gọi của nó. Trong tab CUDA, <code>p</code> không mang màu nào: nó đọc bộ nhớ nào
            là do địa chỉ được truyền vào quyết định. Tab Metal có cặp bản nạp chồng.
          </p>
        }
      >
        <p>
          Here is the helper with its two call sites. In the CUDA tab, <code>p</code> carries no colour: which memory it
          reads is decided by the address it is given. The Metal tab has the pair of overloads.
        </p>
      </Translated>
      <Code
        tabs={[
          { label: tr('CUDA and HIP', 'CUDA và HIP'), code: GENERIC_CU, tiers: CU_TIERS, status: 'front-end-checked' },
          { label: 'Metal', code: OVERLOADS_METAL, tiers: METAL_TIERS, status: 'verified on Metal' },
        ]}
        caption={
          <Translated
            vi={
              <>
                labs/appC/snippets/c1_helper_generic.cu và c1_helper_overloads.metal. Tệp Metal biên dịch được với mọi phiên
                bản MSL từ 2.0 tới 4.1 trên máy build, và <code>python3 labs/appC/snippets/check.py</code> chạy kernel của nó
                trên GPU, đối chiếu với một bản tham chiếu trên CPU. Tệp CUDA chưa chạy trên GPU NVIDIA hay AMD nào.
              </>
            }
          >
            <>
              labs/appC/snippets/c1_helper_generic.cu and c1_helper_overloads.metal. The Metal file compiles with every MSL
              version from 2.0 to 4.1 on the build machine, and <code>python3 labs/appC/snippets/check.py</code> runs its
              kernel on the GPU against a CPU reference. The CUDA file has not run on an NVIDIA or AMD GPU.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Vì sao điều này quan trọng với kernel suy luận: attention và lấy mẫu đầy những phép rút gọn nhỏ, giá trị lớn nhất
            và tổng bên trong một softmax, một tích vô hướng, một argmax. Bạn sẽ muốn mỗi phép được viết một lần và gọi cả
            trên một tile đã đưa vào bộ nhớ chia sẻ lẫn trên một hàng đọc thẳng từ bộ nhớ toàn cục. Trong CUDA và HIP, đó là
            một hàm. Trong MSL, theo đặc tả, đó là mỗi address space một hàm.
          </p>
        }
      >
        <p>
          Why this matters for inference kernels: attention and sampling are full of small reductions, the maximum and the
          sum inside a softmax, a dot product, an argmax. You will want each written once and called both on a tile staged
          in shared memory and on a row read straight from global memory. In CUDA and HIP that is one function. In MSL,
          as specified, it is one per address space.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Mối liên hệ với attention và lấy mẫu này là của chúng tôi.</p>}>
          <p>This connection to attention and sampling is ours.</p>
        </Translated>
      </Beyond>
      <Ask q={questionsSC1['appC/C.1/q-speed']} />
      <Translated
        vi={
          <p>
            Quay lại <a href={href('appC', 'start')}>bản đồ</a>: bước này là bước được đánh dấu C.1, và nó chỉ thay đổi hàng
            của một biến <code>__shared__</code>.
          </p>
        }
      >
        <p>
          Back to <a href={href('appC', 'start')}>the map</a>: this step is the one marked C.1, and it changes only the
          row for a <code>__shared__</code> variable.
        </p>
      </Translated>
    </>
  );
}
