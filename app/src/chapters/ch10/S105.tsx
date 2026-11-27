import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr, useLocale } from '../../lib/i18n';
import {
  ACCESS_LABEL,
  KIND_LABEL,
  bookFormula,
  requests,
  type AccessName,
  type Kind,
  type RequestStep,
} from '../../sim/reduction';
import { widthChoices } from '../../sim/warps';

import { NoteGroup } from './NoteGroup';
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const SIZES = [16, 64, 256, 1024, 2048];
const KINDS: Kind[] = ['simple', 'convergent', 'shared'];
const SEGMENT = 32;
const ACCESSES: Record<Kind, AccessName[]> = {
  simple: ['own', 'partner', 'write'],
  convergent: ['own', 'partner', 'write'],
  shared: ['load-lo', 'load-hi', 'store'],
};
/** How the two read styles are drawn: the first access of a pair filled, the second outlined. */
const FIRST: AccessName[] = ['own', 'load-lo'];

/** The aligned pieces of 32 floats, one row each, with the elements one warp touches in one step marked. */
function SegmentRows({ step, warp, N, kind }: { step: RequestStep; warp: number; N: number; kind: Kind }) {
  const w = step.warps.find((x) => x.warp === warp);
  const marks = new Map<number, Set<AccessName>>();
  for (const a of w?.accesses ?? []) if (a.array === 'input') for (const c of a.cells) marks.set(c, new Set([...(marks.get(c) ?? []), a.name]));
  const total = Math.ceil(N / SEGMENT);
  const touched = [...new Set((w?.accesses ?? []).filter((a) => a.array === 'input').flatMap((a) => a.segments))].sort((a, b) => a - b);
  const rows = total <= 8 ? Array.from({ length: total }, (_, i) => i) : touched;
  const cw = 12;
  const rh = 18;
  const left = 92;
  const W = left + SEGMENT * cw + 4;
  const H = Math.max(1, rows.length) * (rh + 4) + 2;
  if (!rows.length)
    return <p className="muted" data-part="segments" data-rows="0"><Translated vi="Warp này không chạm phần tử nào của đầu vào ở bước này.">This warp touches no element of the input in this step.</Translated></p>;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Pieces of 32 floats of the input, with the elements this warp touches marked', 'Các mẩu 32 float của đầu vào, đánh dấu các phần tử warp này chạm tới')}
      data-part="segments" data-rows={rows.length} style={{ width: W, maxWidth: '100%', minWidth: Math.min(W, 420), display: 'block' }}>
      {rows.map((seg, r) => {
        const hit = touched.includes(seg);
        return (
          <g key={seg} data-segment={seg} data-hit={hit ? '1' : '0'}>
            <text x={left - 6} y={r * (rh + 4) + rh - 5} fontSize="10" textAnchor="end" style={{ fill: hit ? 'var(--ink)' : 'var(--ink-3)', fontWeight: hit ? 650 : 400 }}>
              {seg * SEGMENT}–{seg * SEGMENT + SEGMENT - 1}
            </text>
            <rect x={left - 1} y={r * (rh + 4)} width={SEGMENT * cw + 1} height={rh} fill="none" stroke={hit ? 'var(--global)' : 'var(--rule)'} strokeWidth={hit ? 1.5 : 1} rx="2" />
            {Array.from({ length: SEGMENT }, (_, i) => {
              const c = seg * SEGMENT + i;
              if (c >= N) return null;
              const m = marks.get(c);
              const first = m && [...m].some((x) => FIRST.includes(x));
              const second = m && [...m].some((x) => !FIRST.includes(x) && x !== 'write');
              return (
                <rect key={i} x={left + i * cw + 1} y={r * (rh + 4) + 3} width={cw - 2} height={rh - 6} rx="1" data-cell={c}
                  data-mark={m ? [...m].join(' ') : ''}
                  fill={first ? 'var(--global)' : second ? 'var(--global-wash)' : 'var(--cell)'}
                  stroke={second ? 'var(--global)' : 'var(--rule)'} strokeWidth={second ? 1.4 : 0.5} />
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

function SegmentExplorer() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const [kind, setKind] = useState<Kind>('simple');
  const [N, setN] = useState(256);
  const [widthKey, setWidthKey] = useState('book');
  const [stepIx, setStepIx] = useState(0);
  const [warp, setWarp] = useState(0);
  const width = (choices.find((c) => c.key === widthKey) ?? choices[0]).width;
  const locale = useLocale();
  const trace = useMemo(() => requests(kind, N, width), [kind, N, width, locale]);
  const k = Math.min(stepIx, trace.steps.length - 1);
  const step = trace.steps[k];
  const warpCount = Math.ceil(N / 2 / width);
  const wsel = Math.min(warp, warpCount - 1);
  const mine = step.warps.find((x) => x.warp === wsel);
  const names = ACCESSES[kind];
  const book = width === 32 && N >= 64 ? (kind === 'simple' ? bookFormula.simpleRequests(N) : kind === 'shared' ? bookFormula.sharedRequests(N) : null) : null;
  const reset = () => { setStepIx(0); setWarp(0); };

  return (
    <Bench
      title={tr('Which pieces does a warp touch?', 'Một warp chạm những mẩu nào?')}
      task={tr('Pick a warp and step through the loop: count the pieces of 32 floats it touches in each access.', 'Chọn một warp và chạy từng bước qua vòng lặp: đếm số mẩu 32 float nó chạm tới ở mỗi lần truy cập.')}
      foot={
        <>
          <Translated
            vi={
              <>
                Mỗi hàng là một mẩu 32 float (128 byte) đã căn chỉnh của đầu vào. Ô tô đặc là các phần tử mà các lane đang hoạt
                động của warp được chọn đọc trước ({kind === 'shared' ? 'input[t]' : 'phần tử của chính chúng, cũng là phần tử chúng ghi'}); ô
                viền là lần đọc thứ hai ({kind === 'shared' ? 'input[t + blockDim.x]' : 'đối tác'}). Một yêu cầu (request) là một mẩu được chạm
                trong một lần truy cập. Khi đầu vào có hơn 8 mẩu, chỉ những mẩu warp này chạm tới mới được vẽ.
              </>
            }
          >
            Each row is an aligned piece of 32 floats (128 bytes) of the input. Filled cells are the elements the chosen
            warp's active lanes read first ({kind === 'shared' ? 'input[t]' : 'their own element, which they also write'}); outlined cells
            are the second read ({kind === 'shared' ? 'input[t + blockDim.x]' : 'the partner'}). A request is one piece touched in one access. With
            more than 8 pieces in the input, only the ones this warp touches are drawn.
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span style={FIELD}>
          <Translated vi="Kernel">Kernel</Translated>
          <span className="seg" role="group" aria-label={tr('Kernel', 'Kernel')}>
            {KINDS.map((x) => <button key={x} aria-pressed={kind === x} onClick={() => { setKind(x); reset(); }}>{x === 'shared' ? tr('shared memory', 'bộ nhớ chia sẻ') : x}</button>)}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi="N (giá trị)">N (values)</Translated>
          <span className="seg" role="group" aria-label="N">
            {SIZES.map((x) => <button key={x} aria-pressed={N === x} onClick={() => { setN(x); reset(); }}>{x}</button>)}
          </span>
        </span>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={widthKey} onChange={(e) => { setWidthKey(e.target.value); reset(); }} aria-label={tr('Warp width', 'Độ rộng warp')}>
            {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <button onClick={() => setStepIx(Math.max(0, k - 1))} disabled={k === 0}><Translated vi="Lùi lại">Back</Translated></button>
        <button className="primary" onClick={() => setStepIx(Math.min(trace.steps.length - 1, k + 1))} disabled={k >= trace.steps.length - 1}><Translated vi="Bước">Step</Translated></button>
        <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
          <Translated vi="bước">step</Translated>
          <input type="range" min={0} max={trace.steps.length - 1} value={k} onChange={(e) => setStepIx(Number(e.target.value))} aria-label={tr('Step', 'Bước')} />
        </label>
        <label className="field">
          <Translated vi="Warp">Warp</Translated>
          <select value={wsel} onChange={(e) => setWarp(Number(e.target.value))} aria-label={tr('Warp', 'Warp')}>
            {Array.from({ length: warpCount }, (_, i) => <option key={i} value={i}><Translated vi={<>warp {i} (luồng {i * width} đến {Math.min(N / 2, (i + 1) * width) - 1})</>}>warp {i} (threads {i * width} to {Math.min(N / 2, (i + 1) * width) - 1})</Translated></option>)}
          </select>
        </label>
      </div>
      <div className="narrate" role="status" data-readout="warp" data-requests={mine?.requests ?? 0} style={{ marginBottom: '0.9rem' }}>
        <b>{KIND_LABEL[kind]}, {step.label}.</b>{' '}
        {step.phase === 'iteration' && kind === 'shared' ? (
          <><Translated vi="Vòng lặp này chỉ đọc và ghi bộ nhớ chia sẻ: không warp nào tạo yêu cầu tới bộ nhớ toàn cục.">This iteration reads and writes shared memory only: no warp makes a global-memory request.</Translated></>
        ) : !mine ? (
          <><Translated vi={<>Warp {wsel} không có lane nào hoạt động ở bước này, nên nó không tạo yêu cầu nào.</>}>Warp {wsel} has no active lane in this step, so it makes no request.</Translated></>
        ) : (
          <>
            <Translated vi={<>Warp {wsel}:</>}>Warp {wsel}:</Translated>{' '}
            {mine.accesses.map((a, i) => (
              <span key={a.name}>
                {i ? '; ' : ''}
                <Translated vi={<>{ACCESS_LABEL[a.name]}, {a.cells.length} phần tử trong {a.segments.length} mẩu</>}>{ACCESS_LABEL[a.name]}, {a.cells.length} element{a.cells.length === 1 ? '' : 's'} in {a.segments.length} piece{a.segments.length === 1 ? '' : 's'}</Translated>
              </span>
            ))}
            . <b><Translated vi={<>{mine.requests} yêu cầu.</>}>{mine.requests} requests.</Translated></b>
          </>
        )}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ overflowX: 'auto' }}>
            <SegmentRows step={step} warp={wsel} N={N} kind={kind} />
          </div>
          {mine?.accesses.some((a) => a.array === 'output') ? <p className="muted" style={{ margin: '0.5rem 0 0' }}><Translated vi={<>Luồng 0 ghi <span className="mono">*output</span>: thêm một yêu cầu, tới mảng đầu ra.</>}>Thread 0 writes <span className="mono">*output</span>: one more request, to the output array.</Translated></p> : null}
          {Math.ceil(N / SEGMENT) > 8 && mine ? (
            <p className="muted" style={{ margin: '0.5rem 0 0' }}><Translated vi={<>Đầu vào có {Math.ceil(N / SEGMENT)} mẩu; các mẩu khác không bị warp này chạm tới ở bước này.</>}>The input has {Math.ceil(N / SEGMENT)} pieces; the others are not touched by this warp in this step.</Translated></p>
          ) : null}
        </div>
        <div style={{ minWidth: 0, overflowX: 'auto' }}>
          <table className="threads" aria-label={tr('Requests per step', 'Số yêu cầu mỗi bước')} data-part="request-table">
            <thead>
              <tr>
                <th><Translated vi="bước">step</Translated></th><th><Translated vi="warp">warps</Translated></th>
                {names.map((a) => <th key={a}>{a === 'own' ? tr('own', 'của mình') : a === 'partner' ? tr('partner', 'đối tác') : a === 'write' ? tr('write', 'ghi') : a === 'load-lo' ? 'in[t]' : a === 'load-hi' ? 'in[t+B]' : tr('store', 'lưu')}</th>)}
                <th><Translated vi="tổng">total</Translated></th>
              </tr>
            </thead>
            <tbody>
              {trace.steps.map((s, i) => (
                <tr key={i} className={i === k ? 'cur' : ''}>
                  <td>{s.phase === 'iteration' ? tr(`${s.n} (stride ${s.stride})`, `${s.n} (bước nhảy ${s.stride})`) : s.phase === 'load' ? tr('load', 'nạp') : tr('store', 'lưu')}</td>
                  <td>{s.warps.length}</td>
                  {names.map((a) => <td key={a}>{s.byAccess[a] ?? 0}</td>)}
                  <td>{s.requests}</td>
                </tr>
              ))}
              <tr>
                <td><b><Translated vi="Tổng">Total</Translated></b></td><td /><td colSpan={names.length} /><td data-readout="total"><b>{trace.total}</b></td>
              </tr>
            </tbody>
          </table>
          {book != null ? (
            <p className="muted" style={{ margin: '0.5rem 0 0' }} data-readout="book">
              <Translated
                vi={
                  <>
                    Công thức của sách ({kind === 'shared' ? 'N/32 + 1, pp. 236–237' : 'p. 235'}) cho ra {book}.
                  </>
                }
              >
                The book's formula ({kind === 'shared' ? 'N/32 + 1, pp. 236–237' : 'p. 235'}) gives {book}.
              </Translated>
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS105: Record<string, Question> = {
  'ch10/10.5/q-half-used': {
    kind: 'choice',
    id: 'ch10/10.5/q-half-used',
    prompt: (
      <Translated vi={<>Ở vòng lặp đầu tiên của kernel đơn giản, warp dùng bao nhiêu phần số byte được chuyển tới cho các lần đọc phần tử của chính nó?</>}>
        <>In the simple kernel's first iteration, what fraction of the bytes delivered for a warp's own-element reads does the warp use?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Toàn bộ.">All of them.</Translated>, why: <Translated vi="32 phần tử warp đọc cách nhau 2, nên chúng trải trên 64 float.">The 32 elements the warp reads are 2 apart, so they span 64 floats.</Translated> },
      { label: <Translated vi="Một nửa.">Half of them.</Translated>, correct: true },
      { label: <Translated vi="Một phần tư.">A quarter of them.</Translated>, why: <Translated vi="Đó là vòng lặp thứ hai, khi phần tử của các lane đang hoạt động cách nhau 4.">That is the second iteration, when the active lanes' elements are 4 apart.</Translated> },
      { label: <Translated vi="Một trên 32.">One in 32.</Translated>, why: <Translated vi="Điều đó đến sau, khi mỗi warp chỉ còn một lane hoạt động.">That comes later, once one lane per warp is active.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Luồng t của warp đọc phần tử 2t, nên 32 lần đọc của warp phủ các phần tử 0, 2, ..., 62: hai mẩu 32 float, trong
            đó cứ cách một float mới có một float được dùng. Gấp đôi số giao dịch của một lần đọc được hợp nhất, chỉ dùng một
            nửa dữ liệu. Ở vòng lặp thứ hai, phần tử của các lane đang hoạt động cách nhau 4, và tình hình tệ dần cho đến khi
            chỉ còn một lane.
          </p>
        }
      >
        <p>
          Thread t of the warp reads element 2t, so the warp's 32 reads cover elements 0, 2, ..., 62: two pieces of 32
          floats, of which every second float is used. Twice the transactions of a coalesced read, half the data used. In
          the second iteration the active lanes' elements are 4 apart, and it gets worse until one lane is left.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.5, pp. 234–235',
  },
  'ch10/10.5/q-req-2048': {
    kind: 'numeric',
    id: 'ch10/10.5/q-req-2048',
    prompt: (
      <Translated vi={<>Kernel đơn giản trên 2048 giá trị với warp rộng 32. Theo công thức của sách, nó tạo bao nhiêu yêu cầu tới bộ nhớ toàn cục?</>}>
        <>The simple kernel on 2048 values with 32-wide warps. How many global-memory requests does it make, by the book's formula?</>
      </Translated>
    ),
    answer: 1149,
    explain: (
      <Translated
        vi={
          <p>
            N/64 = 32 warp. Năm vòng lặp trong đó mọi warp tạo hai yêu cầu cho mỗi lần truy cập: 32 × 5 × 2. Rồi một yêu cầu
            cho mỗi lần truy cập với mỗi warp còn một lane hoạt động: 32 + 16 + 8 + 4 + 2 + 1. Mỗi lượt ba lần truy cập:
            (320 + 63) × 3 = 1149. Kernel hội tụ cần 204, ít hơn năm lần rưỡi, và kernel bộ nhớ chia sẻ của mục 10.6 cần 65.
          </p>
        }
      >
        <p>
          N/64 = 32 warps. Five iterations in which every warp makes two requests per access: 32 × 5 × 2. Then one request
          per access for each warp that still has an active lane: 32 + 16 + 8 + 4 + 2 + 1. Three accesses each:
          (320 + 63) × 3 = 1149. The convergent kernel needs 204, five and a half times fewer, and the shared-memory kernel of
          section 10.6 needs 65.
        </p>
      </Translated>
    ),
    source: 'ch. 10 §10.5, p. 235 (the formula); the 2048 case is ours',
    vi: { source: 'ch. 10 §10.5, p. 235 (công thức); trường hợp 2048 là của khóa học này' },
  },
  'ch10/10.5/q-softmax-loop': {
    kind: 'choice',
    id: 'ch10/10.5/q-softmax-loop',
    prompt: (
      <Translated
        vi={
          <>
            Một khối 256 luồng đi qua một hàng 32,000 logits để tìm max. Luồng t đọc hoặc (A){' '}
            <code>x[t + k*256]</code> với k = 0, 1, 2, ..., hoặc (B) <code>x[t*125 + k]</code> với k = 0 đến 124. Cách nào được
            hợp nhất truy cập?
          </>
        }
      >
        <>
          A block of 256 threads walks a row of 32,000 logits to find its max. Thread t reads either (A){' '}
          <code>x[t + k*256]</code> for k = 0, 1, 2, ..., or (B) <code>x[t*125 + k]</code> for k = 0 to 124. Which coalesces?
        </>
      </Translated>
    ),
    options: [
      { label: 'A', correct: true },
      { label: 'B', why: <Translated vi="Ở mỗi bước k, 32 lần đọc của một warp cách nhau 125 float: 32 mẩu riêng cho 32 float.">In each step k, a warp's 32 reads are 125 floats apart: 32 separate pieces for 32 floats.</Translated> },
      { label: <Translated vi="Cả hai: ở B mỗi luồng đọc 125 giá trị liên tiếp, bộ nhớ đệm sẽ đón được.">Both: each thread reads 125 consecutive values in B, which a cache will catch.</Translated>, why: <Translated vi="Hợp nhất truy cập nói về những gì 32 lane của một warp đọc trong cùng một bước, không phải những gì một luồng đọc theo thời gian.">Coalescing is about what the 32 lanes of a warp read in the same step, not what one thread reads over time.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Quy tắc là của kernel hội tụ: ở mỗi bước, các luồng liên tiếp phải đọc các phần tử liên tiếp. Ở A, bước k cho warp
            đọc 32 logits kề nhau, một mẩu. Ở B, luồng t bắt đầu tại 125t, nên ở bất kỳ bước nào các lane của warp cũng cách nhau
            125 float. A là cách phân chia xen kẽ của chương 9 (§9.5), và là vòng lặp bước theo khối mà mục 10.10 dùng để gộp
            việc.
          </p>
        }
      >
        <p>
          The rule is the convergent kernel's: in each step, consecutive threads must read consecutive elements. In A, step k
          has the warp read 32 neighbouring logits, one piece. In B, thread t starts at 125t, so in any one step the warp's
          lanes are 125 floats apart. A is the interleaved partitioning of chapter 9 (§9.5), and the stride-by-block loop
          that section 10.10 uses to coarsen.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.1; ch. 9 §9.5; ch. 10 §10.10, Fig. 10.20', // i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- the page
export function S105() {
  const c256 = requests('convergent', 256, 32);
  const warpIters = c256.steps.reduce((a, s) => a + s.warps.length, 0);
  return (
    <>
      <Translated
        vi={
          <p>
            Kernel đơn giản còn một vấn đề thứ hai, và nó đến từ cùng lựa chọn ai sở hữu cái gì. Ở mỗi vòng lặp, mỗi luồng đang
            hoạt động thực hiện hai lần đọc bộ nhớ toàn cục (phần tử của chính nó, và phần tử đối tác cách một bước nhảy) và một
            lần ghi (phần tử của chính nó). Chương 6 đã nói điều gì làm chúng rẻ: các lane của một warp đọc các phần tử kề nhau
            trong cùng một lệnh, để phần cứng gộp chúng lại (<a href={href('ch06', '6.1')}>mục 6.1</a>). Ở đây các luồng kề
            nhau sở hữu các phần tử cách nhau 2.
          </p>
        }
      >
        <p>
          The simple kernel has a second problem, and it comes from the same choice of who owns what. In every iteration each
          active thread makes two reads of global memory (its own element, and the partner one stride away) and one write
          (its own element). Chapter 6 said what makes those cheap: the lanes of a warp reading neighbouring elements in
          the same instruction, so that the hardware combines them (<a href={href('ch06', '6.1')}>section 6.1</a>). Here
          neighbouring threads own elements 2 apart.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.5" pages="234–235" />
        <Aside>
          <p><Translated vi="Ở p. 234 sách nói hợp nhất truy cập thuộc chương 5; ấn bản này dạy nó ở chương 6.">On p. 234 the book credits coalescing to chapter 5; this edition teaches it in chapter 6.</Translated></p>
        </Aside>
      </NoteGroup>
      <Translated
        vi={
          <p>
            Vì vậy ở vòng lặp đầu tiên, các lần đọc của một warp trải trên gấp đôi số mẩu bộ nhớ so với một lần đọc được hợp
            nhất, và một nửa dữ liệu đến nơi không được dùng. Ở vòng lặp thứ hai, phần tử của các lane đang hoạt động cách nhau
            4: gấp bốn số giao dịch, dùng một phần tư. Nó cứ tệ dần khi bước nhảy tăng, cho đến khi mỗi warp chỉ còn một lane
            hoạt động và tạo một yêu cầu cho mỗi lần truy cập.
          </p>
        }
      >
        <p>
          So in the first iteration a warp's reads span twice as many pieces of memory as a coalesced read would, and half the
          data that arrives is not used. In the second iteration the active lanes' elements are 4 apart: four times the
          transactions, a quarter used. It keeps getting worse as the stride grows, until each warp has one active lane and
          makes one request per access.
        </p>
      </Translated>
      <Source section="10.5" pages="235" />
      <Ask q={questionsS105['ch10/10.5/q-half-used']} />

      <h2><Translated vi="Đếm số yêu cầu">Counting requests</Translated></h2>
      <Translated
        vi={
          <p>
            Sách đếm số yêu cầu bộ nhớ toàn cục của kernel đơn giản cho 256 giá trị là (N/64 × 5 × 2 + N/64 × 1 + N/64 ×
            1/2 + ... + 1) × 3. Số hạng đầu phủ năm vòng lặp trong đó mọi warp có ít nhất hai luồng hoạt động và các truy cập
            rải rác của nó cần hai yêu cầu; sau đó mỗi warp có một luồng hoạt động cần một yêu cầu, và số warp đó giảm một nửa
            sau mỗi vòng lặp; × 3 cho hai lần đọc và một lần ghi. Với N = 256, đó là (40 + 4 + 2 + 1) × 3 = 141.
          </p>
        }
      >
        <p>
          The book counts the simple kernel's global-memory requests for 256 values as (N/64 × 5 × 2 + N/64 × 1 + N/64 ×
          1/2 + ... + 1) × 3. The first term covers five iterations in which every warp has at least two active threads and
          its scattered accesses need two requests; after that each warp with one active thread needs one, and their number
          halves every iteration; × 3 for the two reads and the write. For N = 256 that is (40 + 4 + 2 + 1) × 3 = 141.
        </p>
      </Translated>
      <Source section="10.5" pages="235" />
      <Translated
        vi={
          <p>
            Sách không nói một yêu cầu là gì. Để vẽ nó, chúng tôi dùng một quy tắc: trong một lần truy cập, một warp tạo một yêu
            cầu cho mỗi mẩu 32 float (128 byte) đã căn chỉnh mà các lane đang hoạt động của nó chạm tới. Quy tắc đó tái hiện cả
            ba con số của sách, 141 ở đây, 36 cho kernel hội tụ và 9 cho kernel của mục 10.6.
          </p>
        }
      >
        <p>
          The book does not say what a request is. To draw it, we use one rule: in one access, a warp makes one request per
          aligned piece of 32 floats (128 bytes) that its active lanes touch. That rule reproduces all three of the book's
          counts, 141 here, 36 for the convergent kernel and 9 for section 10.6's.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của chúng tôi, không phải của sách')}>
        <Translated
          vi={
            <p>
              Các hệ thống bộ nhớ thật khác nhau về kích thước giao dịch và bộ nhớ đệm (chương 6). Như trong sách, lần đọc cuối{' '}
              <code>input[0]</code> và lần ghi <code>*output</code> của các kernel tại chỗ bị bỏ ra, còn một lần ghi của kernel
              bộ nhớ chia sẻ thì được tính.
            </p>
          }
        >
          <p>
            Real memory systems differ in transaction sizes and caches (chapter 6). As in the book, the in-place kernels'
            last read of <code>input[0]</code> and write of <code>*output</code> are left out, and the shared-memory
            kernel's one write is counted.
          </p>
        </Translated>
      </Beyond>
      <Predict
        id="ch10/10.5/p-requests"
        prompt={<Translated vi={<>Trước khi chọn kernel hội tụ bên dưới: nó tạo bao nhiêu yêu cầu bộ nhớ toàn cục cho 256 giá trị?</>}><>Before you select the convergent kernel below: how many global-memory requests does it make for 256 values?</></Translated>}
        unit={tr('requests', 'yêu cầu')}
      >
        {(v) => (
          <p data-readout="p-requests">
            <Translated
              vi={
                <>
                  Mô hình đếm được {c256.total}: {warpIters} lượt warp-vòng lặp, mỗi lượt có ba lần truy cập, mỗi lần chạm một mẩu.{' '}
                  {v === c256.total ? 'Đúng như bạn dự đoán.' : v > c256.total ? 'Ít hơn bạn dự đoán.' : 'Nhiều hơn bạn dự đoán.'} Sách
                  cũng cho ra 36 ở p. 237, không kèm phép tính.
                </>
              }
            >
              The model counts {c256.total}: {warpIters} warp-iterations, each with three accesses that touch one piece apiece.{' '}
              {v === c256.total ? 'Exactly what you predicted.' : v > c256.total ? 'Fewer than you predicted.' : 'More than you predicted.'} The book
              gives the same 36 on p. 237, without the arithmetic.
            </Translated>
          </p>
        )}
      </Predict>
      <Simplification>
        <Translated
          vi={
            <p>
              Quy tắc yêu cầu là của chúng tôi (xem ghi chú ở trên). Mô hình không biết gì về bộ nhớ đệm, thời gian hay kích thước
              giao dịch, và các trình giả lập (emulator) CPU của khóa học cũng vậy, chúng không mô hình hóa hợp nhất truy cập chút
              nào; lab 10.1 đo các kernel trên thiết bị của bạn.
            </p>
          }
        >
          <p>
            The request rule is ours (see the note above). The model knows nothing about caches, timing or transaction sizes,
            and neither do the course's CPU emulators, which do not model coalescing at all; lab 10.1 measures the kernels on
            your device.
          </p>
        </Translated>
      </Simplification>
      <SegmentExplorer />

      <h2><Translated vi="Kernel hội tụ được hợp nhất truy cập">The convergent kernel coalesces</Translated></h2>
      <Translated
        vi={
          <p>
            Trong kernel hội tụ, các luồng kề nhau luôn chạm các phần tử kề nhau: phần tử của mình là t, đối tác là t + stride.
            Mọi truy cập của một warp đang hoạt động rơi vào một mẩu, và một warp không có luồng hoạt động nào thì không tạo yêu
            cầu nào. Lãng phí duy nhất còn lại là khi số lane hoạt động xin dữ liệu ít hơn một mẩu đầy, ở các vòng lặp cuối.
            Sách cho 36 yêu cầu cho 256 giá trị so với 141 của kernel đơn giản.
          </p>
        }
      >
        <p>
          In the convergent kernel neighbouring threads always touch neighbouring elements: own elements t, partners t +
          stride. Every access of an active warp falls in one piece, and a warp with no active thread makes no request at
          all. The only waste left is when fewer active lanes than a whole piece's worth ask for data, in the last
          iterations. The book gives 36 requests for 256 values against the simple kernel's 141.
        </p>
      </Translated>
      <Source section="10.5" pages="235, 237" />
      <Translated
        vi={
          <p>
            Vậy ba dòng thay đổi của mục 10.4 thắng hai lần: ít khe thực thi hơn, nhờ ít phân kỳ hơn, và ít yêu cầu DRAM hơn,
            nhờ truy cập được hợp nhất. Băng thông DRAM có hạn, nên sách kỳ vọng kernel hội tụ tốt hơn đáng kể ở điểm này nữa.
          </p>
        }
      >
        <p>
          So the three changed lines of section 10.4 win twice: fewer execution slots, from less divergence, and fewer DRAM
          requests, from coalesced accesses. DRAM bandwidth is limited, so the book expects the convergent kernel to be
          significantly better on this count too.
        </p>
      </Translated>
      <NoteGroup>
        <Source section="10.5" pages="235" />
        <Aside>
          <Translated
            vi={
              <p>
                Ở p. 235, "the kernel in Fig. 10.7" (hai lần) nghĩa là kernel của Hình 10.8; Hình 10.7 là sơ đồ của nó.
              </p>
            }
          >
            <p>
              On p. 235, "the kernel in Fig. 10.7" (twice) means the kernel of Fig. 10.8; Fig. 10.7 is its diagram.
            </p>
          </Translated>
        </Aside>
        <Today checked="2026-11-20 (notes/today/infra-facts.md C13)">{/* i18n-ignore: date and citation */}
          <Translated
            vi={
              <p>
                Apple không công bố kích thước bộ nhớ đệm, kích thước cache line hay con số băng thông bộ nhớ nào cho GPU của mình,
                chỉ nói rằng các lần đọc buffer đi qua một bộ nhớ đệm có kích thước hạn chế. Trên Mac, khác biệt giữa các kernel này
                là thứ cần đo.
              </p>
            }
          >
            <p>
              Apple documents no cache size, cache-line size or memory bandwidth figure for its GPUs, only that buffer reads
              go through a cache of limited size. On a Mac, the difference between these kernels is something to measure.
            </p>
          </Translated>
        </Today>
      </NoteGroup>
      <Ask q={questionsS105['ch10/10.5/q-req-2048']} label={tr('Work it out', 'Tự tính ra')} />

      <h2><Translated vi="Cùng quy tắc đó trong một softmax">The same rule in a softmax</Translated></h2>
      <Translated
        vi={
          <p>
            Quy tắc bạn vừa đếm quyết định nhiều hơn các phép rút gọn. Một softmax trên một hàng logits bắt đầu bằng một phép max
            trên hàng: mỗi luồng của một khối đi qua một phần của hàng, và điều quan trọng với bộ nhớ là những gì các lane của
            một warp đọc trong cùng một bước. Các luồng liên tiếp phải đọc các logit liên tiếp. Đó là cách phân chia xen kẽ của{' '}
            <a href={href('ch09', '9.5')}>mục 9.5</a>, và là vòng lặp mà mục 10.10 dùng để gộp việc.
          </p>
        }
      >
        <p>
          The rule you just counted decides more than reductions. A softmax over a row of logits starts with a max over the
          row: each thread of a block walks part of the row, and what matters for memory is what a warp's lanes read in the
          same step. Consecutive threads must read consecutive logits. That is the interleaved partitioning of{' '}
          <a href={href('ch09', '9.5')}>section 9.5</a>, and the loop section 10.10 uses to coarsen.
        </p>
      </Translated>
      <Source section="10.10" pages="247" figs={tr('Fig. 10.20', 'Hình 10.20')} />
      <Ask q={questionsS105['ch10/10.5/q-softmax-loop']} />
    </>
  );
}
