import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { LabsPage, Reveal, defineLabs, measure, solutionTabStatus } from '../../components/LabKit';
import { Beyond, Bench, Simplification, Source, Today, type Tier } from '../../components/notes';
import { fmt, num, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  SECTORS_PER_REQUEST, bytesPerWarpLoad, instructions, misalignedChunks, plan, widestVector, type Plan, type ThreadPlan,
} from '../../sim/vectorize';
import { Translated, tr } from '../../lib/i18n';

// ---------------------------------------------------------------- lab 6.2, the code this section shows and measures
const LABS = defineLabs('ch06', [{ num: '6.2', dir: 'lab2_vector', name: 'vector loads', vi: { name: 'nạp dạng vector' } }] as const);
const [VEC] = LABS;
const SOL = `${VEC.id}/solution`;

// x, y and z are left uncoloured: the highlighter colours by name, and .x, .y, .z are also float4 fields and
// blockIdx components.
const VEC_TIERS: Record<string, Tier> = {
  x4: 'global', y4: 'global', z4: 'global',
  i: 'reg', j: 'reg', n: 'reg', first: 'reg', iStart: 'reg', c: 'reg', gid: 'reg', chunks: 'reg', threads: 'reg',
};
// In the CUDA file x4, y4, z4 are registers (local float4 copies); in the Metal file they are the float4 views of
// the buffers. Each tab gets the colours that are true for it.
const CU_TIERS: Record<string, Tier> = { ...VEC_TIERS, x4: 'reg', y4: 'reg', z4: 'reg' };

const linesWith = (text: string, word: string) =>
  text.split('\n').flatMap((l, i) => (l.includes(word) && !l.trim().startsWith('//') ? [i + 1] : []));

function LabListing() {
  const cu = useLabFile(SOL + '/kernel.cu');
  const metal = useLabFile(SOL + '/kernel.metal');
  const mm = useLabFile(SOL + '/launch.mm');
  const { reports, loading } = useReports();
  // The same words as the labs page's status table (LabKit's solutionTabStatus).
  const reading = loading ? tr('reading reports', 'đang đọc báo cáo') : null;
  const cuda = reading ?? solutionTabStatus(reports, VEC, 'kernel.cu');
  const onMetal = reading ?? solutionTabStatus(reports, VEC, 'kernel.metal');
  if (!cu || !metal || !mm) return (
      <p className="muted">
        <Translated vi="Đang tải lời giải tham chiếu (solution) của lab 6.2.">Loading lab 6.2's reference solution.</Translated>
      </p>
    );
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: CU_TIERS, status: cuda, highlight: linesWith(cu, 'float4') },
        { label: tr('Metal kernel', 'Kernel Metal'), code: metal, tiers: VEC_TIERS, status: onMetal, highlight: linesWith(metal, 'float4') },
        { label: tr('Metal host', 'Host Metal'), code: mm, tiers: {}, status: onMetal, highlight: linesWith(mm, 'atIndex') },
      ]}
      caption={
        <Translated
          vi={
            <>
              Lời giải tham chiếu của lab 6.2, tải từ <code>labs/{SOL}/</code>; trạng thái cạnh các tab được đọc từ các
              lần chạy lab đã ghi lại trên máy này. Khác biệt so với sách: Fig. 6.12 như được in không có kiểm tra biên và
              không bao giờ dùng tham số <code>N</code>, nên nó đọc và ghi vượt quá cuối mảng mỗi khi độ dài không phải
              bội của bốn. Lời giải thêm hai biên của Exercise 4 và khai báo các đầu vào là <code>const</code>. Kernel thứ
              hai của nó được gộp việc theo hệ số bốn mà không dùng truy cập vector, để lab có thể tách bốn phần tử mỗi
              luồng khỏi bốn phần tử mỗi lệnh. Bản build HIP được thiết kế để đưa cùng tệp này qua header ánh xạ tên của
              dự án: tệp dùng luồng, khối, <code>float4</code>,{' '}
              <code>#pragma unroll</code> và cách khởi chạy <code>{'<<<…>>>'}</code>, không cái nào nằm trong
              các trường hợp mà CUDA và HIP viết khác nhau. Trừ khi trạng thái cạnh tab CUDA cho thấy HIP đã chạy qua, con
              đường đó là unverified.
            </>
          }
        >
          Lab 6.2's reference solution, loaded from <code>labs/{SOL}/</code>; the status beside the tabs is read from
          this machine's recorded lab runs. How it differs from the page: Fig. 6.12 as printed has no bounds check and
          never uses its parameter <code>N</code>, so it reads and writes past the end whenever the length is not a
          multiple of four. The solution adds Exercise 4's two boundaries and makes the inputs <code>const</code>. Its
          second kernel is coarsened by four without vector accesses, so that the lab can separate four elements per
          thread from four elements per instruction. The HIP build is meant to take the same file through the
          project&rsquo;s name-mapping header: the file uses threads, blocks, <code>float4</code>,{' '}
          <code>#pragma unroll</code> and the <code>{'<<<…>>>'}</code> launch, none of them among
          the cases where CUDA and HIP spellings differ. Unless the status beside the CUDA tab shows a HIP pass, that
          path is unverified.
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- the chunk planner
const DRAW = 80;
const ROLE_STYLE: Record<ThreadPlan['role'], { bg: string; border: string; dash?: boolean }> = {
  vector: { bg: 'var(--global-wash)', border: 'var(--global)' },
  tail: { bg: 'var(--cell)', border: 'var(--global)', dash: true },
  idle: { bg: 'transparent', border: 'var(--ink-3)', dash: true },
};

function Chunk({ r, p, bad }: { r: ThreadPlan; p: Plan; bad: boolean }) {
  const st = ROLE_STYLE[r.role];
  const cells = r.role === 'idle' ? 1 : r.count;
  return (
    <span
      data-thread={r.thread}
      data-role={r.role}
      data-bad={bad ? 'yes' : 'no'}
      style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'stretch', margin: '0 3px 6px 0' }}
    >
      <span style={{ fontSize: '0.64rem', color: bad ? 'var(--fail)' : 'var(--ink-2)', textAlign: 'center', whiteSpace: 'nowrap' }}>
        t{r.thread}
        {r.role === 'tail' ? tr(' tail', ' đuôi') : ''}
      </span>
      <span
        style={{
          display: 'inline-flex',
          gap: 1,
          padding: 1,
          border: `1.5px ${st.dash || bad ? 'dashed' : 'solid'} ${bad ? 'var(--fail)' : st.border}`,
          background: bad ? 'var(--fail-wash)' : 'transparent',
          borderRadius: 3,
        }}
      >
        {Array.from({ length: cells }, (_, j) => (
          <span
            key={j}
            style={{
              width: p.n > 40 ? '1.15rem' : '1.45rem',
              height: '1.35rem',
              display: 'grid',
              placeItems: 'center',
              fontSize: '0.6rem',
              background: st.bg,
              color: r.role === 'idle' ? 'var(--ink-3)' : 'var(--ink)',
              borderRadius: 2,
            }}
          >
            {r.role === 'idle' ? '–' : r.first + j}
          </span>
        ))}
      </span>
    </span>
  );
}

function ChunkPlanner() {
  const [n, setN] = useState(37);
  const [block, setBlock] = useState(8);
  const [w, setW] = useState(4);
  const [offset, setOffset] = useState(0);
  const p = useMemo(() => plan(n, block, w), [n, block, w]);
  const ins = instructions(n, w);
  const bad = new Set(misalignedChunks(p, offset));
  const widest = widestVector(offset, 4);
  // Long arrays: draw the last elements, where the boundaries are.
  const firstDrawn = p.roles.findIndex((r) => r.role !== 'idle' && r.first + r.count > Math.max(0, n - DRAW));
  const drawn = p.roles.slice(Math.max(0, firstDrawn));
  const cut = firstDrawn > 0;
  const warpBytes = bytesPerWarpLoad(32, w);
  const isCase = n === 1003 && block === 256;
  return (
    <Bench
      title={tr('Who owns which floats', 'Luồng nào sở hữu những số float nào')}
      task={tr('Make n a non-multiple of four and find the tail; then move the base pointer off a 16-byte boundary.', 'Đặt n không phải bội của bốn và tìm phần đuôi; sau đó dời con trỏ gốc ra khỏi biên 16 byte.')}
      foot={tr('One thread per chunk of the chosen width, launched in whole blocks. Instruction counts are for vector addition (two loads and one store per element) summed over all threads; they say nothing about time.', 'Mỗi luồng một đoạn có độ rộng đã chọn, khởi chạy theo khối nguyên. Số lệnh là của phép cộng vector (hai lần nạp và một lần ghi mỗi phần tử) cộng trên mọi luồng; chúng không nói gì về thời gian.')}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Số phần tử n">Elements n</Translated>
          <input type="number" min={1} max={100000000} value={n} aria-label={tr('Elements n', 'Số phần tử n')} onChange={(e) => setN(Math.max(1, Math.min(1e8, Math.floor(Number(e.target.value) || 1))))} />
        </label>
        <label className="field">
          <Translated vi="Số luồng mỗi khối">Threads per block</Translated>
          <select value={block} aria-label={tr('Threads per block', 'Số luồng mỗi khối')} onChange={(e) => setBlock(+e.target.value)}>
            {[4, 8, 16, 32, 64, 128, 256].map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số float mỗi lần truy cập">Floats per access</Translated>
          <span className="seg">
            {[1, 2, 4].map((v) => (
              <button key={v} aria-pressed={w === v} onClick={() => setW(v)}>
                {v === 1 ? '1 (float)' : v === 2 ? '2 (float2)' : '4 (float4)' /* i18n-ignore: counts and type names */}
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Con trỏ gốc, số byte vượt qua một biên 256 byte">Base pointer, bytes past a 256-byte boundary</Translated>
          <span className="seg">
            {[0, 4, 8, 12, 16].map((o) => (
              <button key={o} aria-pressed={offset === o} onClick={() => setOffset(o)}>{o}</button>
            ))}
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }} role="group" aria-label={tr('Presets', 'Thiết lập sẵn')}>
        <span className="muted">
          <Translated vi="Thiết lập sẵn">Presets</Translated>
        </span>
        <button className="quiet" data-preset="small" aria-pressed={n === 37 && block === 8} onClick={() => { setN(37); setBlock(8); }}><Translated vi="n = 37, 8 luồng mỗi khối">n = 37, 8 threads per block</Translated></button>
        <button className="quiet" data-preset="q-tail" aria-pressed={isCase} onClick={() => { setN(1003); setBlock(256); setW(4); }}><Translated vi="n = 1003, 256 luồng mỗi khối">n = 1003, 256 threads per block</Translated></button>
      </div>
      <div className="panel" style={{ marginBottom: '0.9rem' }} data-out="drawing">
        {cut ? (
          <p className="muted" style={{ margin: '0 0 0.4rem' }}>
            <Translated vi={<>Các phần tử 0 đến {drawn[0].first - 1} là các đoạn đầy đủ giống những đoạn được vẽ; hình vẽ bắt đầu từ luồng {drawn[0].thread}.</>}>
              Elements 0 to {drawn[0].first - 1} are full chunks like the ones drawn; the drawing starts at thread {drawn[0].thread}.
            </Translated>
          </p>
        ) : null}
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'flex-end' }}>
          {drawn.map((r) => (
            <Chunk key={r.thread} r={r} p={p} bad={bad.has(r.thread)} />
          ))}
        </div>
        <div className="legend" style={{ marginTop: '0.3rem' }}>
          <span><i style={{ background: 'var(--global-wash)', outline: '1.5px solid var(--global)' }} />{w > 1 ? tr(`one ${w * 4}-byte access per array`, `một lần truy cập ${w * 4} byte mỗi mảng`) : tr('one 4-byte access per array', 'một lần truy cập 4 byte mỗi mảng')}</span>
          {w > 1 ? <span><i style={{ background: 'var(--cell)', outline: '1.5px dashed var(--global)' }} /><Translated vi="đuôi: mỗi lần một float">tail: one float at a time</Translated></span> : null}
          <span><i style={{ background: 'transparent', outline: '1.5px dashed var(--ink-3)' }} /><Translated vi="luồng rảnh">idle thread</Translated></span>
          {bad.size ? <span><i style={{ background: 'var(--fail-wash)', outline: '1.5px dashed var(--fail)' }} /><Translated vi="truy cập vector không căn chỉnh">misaligned vector access</Translated></span> : null}
        </div>
      </div>
      <div className="cols">
        <div className="panel" aria-live="polite">
          <Translated
            vi={
              <p style={{ margin: '0 0 0.4rem' }}>
                <b><output data-out="threads">{p.threads}</output></b> luồng trong <output data-out="blocks">{p.blocks}</output>{' '}
                khối cho <output data-out="chunks">{p.chunks}</output> đoạn.
              </p>
            }
          >
            <p style={{ margin: '0 0 0.4rem' }}>
              <b><output data-out="threads">{p.threads}</output></b> threads in <output data-out="blocks">{p.blocks}</output>{' '}
              {p.blocks === 1 ? 'block' : 'blocks'} for <output data-out="chunks">{p.chunks}</output> chunks.
            </p>
          </Translated>
          <p style={{ margin: '0 0 0.4rem' }}>
            <Translated vi="Đuôi:">Tail:</Translated>{' '}
            <output data-out="tail" data-value={p.tailThread ?? 'none'}>
              {p.tailThread == null ? tr('none, n divides evenly', 'không có, n chia hết') : tr(`thread ${p.tailThread}, ${p.tailCount} ${p.tailCount === 1 ? 'element' : 'elements'} one at a time`, `luồng ${p.tailThread}, ${p.tailCount} phần tử, mỗi lần một phần tử`)}
            </output>
            .
          </p>
          <Translated
            vi={
              <p style={{ margin: 0 }}>
                Rảnh trong khối cuối: <b><output data-out="idle">{p.idle}</output></b> luồng.
              </p>
            }
          >
            <p style={{ margin: 0 }}>
              Idle in the last block: <b><output data-out="idle">{p.idle}</output></b> {p.idle === 1 ? 'thread' : 'threads'}.
            </p>
          </Translated>
        </div>
        <div className="panel">
          <Translated
            vi={
              <p style={{ margin: '0 0 0.4rem' }}>
                Lệnh nạp: <output data-out="loads">{fmt(ins.loads)}</output> so với {fmt(ins.scalarLoads)} khi nạp mỗi lần một float,{' '}
                <b>ít hơn <output data-out="fewer" data-value={Math.round(ins.fewerLoadsPct * 100) / 100}>{fmt(ins.fewerLoadsPct)}%</output></b>.
                Lệnh ghi: {fmt(ins.stores)} so với {fmt(ins.scalarStores)}.
              </p>
            }
          >
            <p style={{ margin: '0 0 0.4rem' }}>
              Load instructions: <output data-out="loads">{fmt(ins.loads)}</output> against {fmt(ins.scalarLoads)} one float at a time,{' '}
              <b><output data-out="fewer" data-value={Math.round(ins.fewerLoadsPct * 100) / 100}>{fmt(ins.fewerLoadsPct)}%</output> fewer</b>.
              Stores: {fmt(ins.stores)} against {fmt(ins.scalarStores)}.
            </p>
          </Translated>
          <Translated
            vi={
              <p style={{ margin: '0 0 0.4rem' }}>
                Một lần nạp trên toàn warp gồm 32 lane chuyển {warpBytes} byte
                {SECTORS_PER_REQUEST[w * 4] ? `, ${SECTORS_PER_REQUEST[w * 4]} sector 32 byte của NVIDIA (mức tối ưu theo Nsight Compute)` : ''}.
              </p>
            }
          >
            <p style={{ margin: '0 0 0.4rem' }}>
              One warp-wide load of 32 lanes moves {warpBytes} bytes
              {SECTORS_PER_REQUEST[w * 4] ? `, ${SECTORS_PER_REQUEST[w * 4]} of NVIDIA's 32-byte sectors (Nsight Compute's optimum)` : ''}.
            </p>
          </Translated>
          <p style={{ margin: 0 }} data-out="align">
            {w > 1 && bad.size ? (
              <Translated
                vi={
                  <>
                    <span className="chip fail">không được phép</span> Ở vị trí {offset} byte sau biên, cả {bad.size} lần truy
                    cập {w * 4} byte đều không căn chỉnh.{' '}
                  </>
                }
              >
                <span className="chip fail">not allowed</span> At {offset} bytes past the boundary, all {bad.size} {w * 4}-byte
                accesses are misaligned.{' '}
              </Translated>
            ) : null}
            <Translated
              vi={
                <>
                  Vector rộng nhất mà quy tắc của PyTorch cho phép ở đây: <b><output data-out="widest">{widest}</output> float</b>{' '}
                  <span className="muted">(cách làm của PyTorch, can_vectorize_up_to)</span>.
                </>
              }
            >
              Widest vector PyTorch's rule allows here: <b><output data-out="widest">{widest}</output> floats</b>{' '}
              <span className="muted">(PyTorch's practice, can_vectorize_up_to)</span>.
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 6.2's measurement, for the prediction
function speedups(reports: Reports) {
  return measure(reports, VEC, (set) => {
    const row = (v: string) => set.rows.find((r) => r.name === 'vecadd' && r.variant === v);
    const vec4 = row('vec4');
    const coarse = row('coarse4');
    const s = num(vec4?.speedup_vs_scalar);
    if (!vec4 || !s) return null;
    const c = num(coarse?.speedup_vs_scalar);
    return {
      value: s,
      detail:
        tr(`${fmt(num(vec4.n) ?? NaN)} floats per array, ${fmt(num(vec4.gb_per_s) ?? NaN)} GB/s`, `${fmt(num(vec4.n) ?? NaN)} float mỗi mảng, ${fmt(num(vec4.gb_per_s) ?? NaN)} GB/s`) +
        (c ? tr(`; the coarsened kernel without float4 measured ${fmt(c)} ×`, `; kernel đã gộp việc không dùng float4 đo được ${fmt(c)} ×`) : ''),
    };
  });
}

// ---------------------------------------------------------------- questions
export const questionsS63: Record<string, Question> = {
  'ch06/6.3/q-tail': {
    kind: 'numeric',
    id: 'ch06/6.3/q-tail',
    prompt: (
      <Translated
        vi={
          <>
            n = 1003 float, 256 luồng mỗi khối, mỗi luồng sở hữu một đoạn bốn phần tử. Chỉ số luồng nào xử lý đoạn cuối chưa đầy?
          </>
        }
      >
        <>
          n = 1003 floats, 256 threads per block, each thread owning a chunk of four. Which thread index handles the
          partial last chunk?
        </>
      </Translated>
    ),
    answer: 250,
    explain: (
      <Translated
        vi={
          <p>
            ⌈1003 / 4⌉ = 251 đoạn, đánh số 0 đến 250. Luồng 250 sở hữu các phần tử 1000 đến 1002, ba phần tử, và phải dùng truy cập vô hướng: một truy cập float4 sẽ chạm tới phần tử 1003, vốn không tồn tại. Các luồng 251 đến 255 của khối duy nhất bắt đầu ở ngoài dữ liệu và không làm gì.
          </p>
        }
      >
        <p>
          ⌈1003 / 4⌉ = 251 chunks, numbered 0 to 250. Thread 250 owns elements 1000 to 1002, three of them, and must use
          scalar accesses: a float4 access would touch element 1003, which does not exist. Threads 251 to 255 of the one
          block start past the data and do nothing.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.3, pp. 138–139; Exercise 4, p. 155',
    vi: { source: 'ch. 6 §6.3, pp. 138–139; bài tập 4, p. 155' },
  },
  'ch06/6.3/q-instr': {
    kind: 'numeric',
    id: 'ch06/6.3/q-instr',
    prompt: (
      <Translated
        vi={
          <>
            Một luồng cộng bốn cặp float. Thay vì tám lần nạp vô hướng, nó dùng hai lần nạp float4. Nó thực thi ít hơn bao nhiêu phần trăm lệnh nạp?
          </>
        }
      >
        <>
          A thread adds four pairs of floats. Instead of eight scalar loads it uses two float4 loads. How many percent fewer
          load instructions does it execute?
        </>
      </Translated>
    ),
    answer: 75,
    tol: 0.01,
    unit: '%',
    explain: (
      <Translated vi={<p>1 − 2/8 = 75 %. Vẫn chừng ấy byte được di chuyển; số lệnh di chuyển chúng giảm còn một phần tư, và lệnh ghi cũng vậy.</p>}>
        <p>1 − 2/8 = 75 %. The same bytes move; the instructions that move them fall to a quarter, and stores likewise.</p>
      </Translated>
    ),
    source: 'ch. 6 §6.3, p. 139',
  },
  'ch06/6.3/q-when': {
    kind: 'choice',
    id: 'ch06/6.3/q-when',
    prompt: <Translated vi={<>Chuyển sang nạp vector 16 byte sẽ giúp ít nhất ở đâu?</>}>Where should switching to 16-byte vector loads help least?</Translated>,
    options: [
      { label: <Translated vi="Một kernel dạng streaming ở occupancy tối đa, đã gần với băng thông bạn đo được.">A streaming kernel at full occupancy, already close to the bandwidth you measured.</Translated>, correct: true },
      { label: <Translated vi="Một kernel có occupancy bị giới hạn bởi bộ nhớ chia sẻ của nó, nên chỉ có ít warp thường trú.">A kernel whose occupancy is limited by its shared memory, so that few warps are resident.</Translated>, why: <Translated vi="Đó là lợi ích thứ hai của sách: khi có ít warp thường trú, các lần nạp rộng hơn giữ đủ byte đang trên đường truyền.">That is the book's second benefit: with few warps resident, wider loads keep enough bytes in flight.</Translated> },
      { label: <Translated vi="Một kernel phát ra ít lần nạp mỗi luồng và chạy ở occupancy thấp.">A kernel that issues few loads per thread and runs at low occupancy.</Translated>, why: <Translated vi="Đó là lợi ích thứ hai của sách: khi có quá ít dữ liệu đang trên đường truyền, nhiều byte hơn mỗi lần nạp sẽ có ích, miễn là các thanh ghi tăng thêm không làm mất occupancy.">That is the book's second benefit: with too little in flight, more bytes per load help, as long as the extra registers do not cost occupancy.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Nếu bộ nhớ đã gần bão hòa, chẳng còn mấy để thu được từ việc di chuyển cùng số byte với ít lệnh hơn. Lợi ích thứ hai của sách (p. 139) nói về độ trễ khi occupancy thấp. NVIDIA thêm một cái giá: nạp dạng vector làm tăng áp lực thanh ghi (register), nên một kernel vốn đã bị giới hạn bởi thanh ghi có thể mất occupancy.
          </p>
        }
      >
        <p>
          If the memory is already nearly saturated, there is little left to gain from moving the same bytes with fewer
          instructions. The book's second benefit (p. 139) is about latency when occupancy is low. NVIDIA adds a cost:
          vectorised loads raise register pressure, so a kernel that is already register-limited can lose occupancy.
        </p>
      </Translated>
    ),
    source: "ch. 6 §6.3, p. 139; NVIDIA's developer blog (facts A11)",
    vi: { source: 'ch. 6 §6.3, p. 139; blog dành cho nhà phát triển của NVIDIA (facts A11)' },
  },
  'ch06/6.3/q-align': {
    kind: 'choice',
    id: 'ch06/6.3/q-align',
    prompt: (
      <Translated
        vi={
          <>
            <code>x</code> đến từ <code>cudaMalloc</code>. <code>reinterpret_cast&lt;float4*&gt;(x + 1)[i]</code> có phải là một truy cập hợp lệ không?
          </>
        }
      >
        <>
          <code>x</code> came from <code>cudaMalloc</code>. Is <code>reinterpret_cast&lt;float4*&gt;(x + 1)[i]</code> a valid
          access?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Không: một truy cập float4 cần một biên 16 byte, còn x + 1 nằm sau một biên như vậy 4 byte.">No: a float4 access needs a 16-byte boundary, and x + 1 is 4 bytes past one.</Translated>, correct: true },
      { label: <Translated vi="Có: phần cứng tách một truy cập không căn chỉnh thành các truy cập nhỏ hơn được căn chỉnh.">Yes: the hardware splits a misaligned access into smaller aligned ones.</Translated>, why: <Translated vi="Không nhà sản xuất nào hứa điều đó. NVIDIA nói các lệnh này yêu cầu dữ liệu được căn chỉnh.">No vendor promises that. NVIDIA says these instructions require aligned data.</Translated> },
      { label: <Translated vi="Có: cudaMalloc căn chỉnh theo 256 byte, nên mọi độ lệch vào vùng cấp phát đều dùng được.">Yes: cudaMalloc aligns to 256 bytes, so any offset into the allocation works.</Translated>, why: <Translated vi="Vùng cấp phát được căn chỉnh; nhưng thứ cần căn chỉnh là địa chỉ bạn đọc.">The allocation is aligned; the address you read at is what has to be.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Blog dành cho nhà phát triển của NVIDIA: bộ nhớ device được căn chỉnh cho các kiểu vector, “but if you offset the pointer, the offset must also be aligned” (nhưng nếu bạn dịch con trỏ, độ dịch cũng phải được căn chỉnh), và đưa <code>reinterpret_cast&lt;int2*&gt;(d_in+1)</code> làm ví dụ không hợp lệ. PyTorch kiểm tra căn chỉnh của từng con trỏ lúc chạy và lùi về các truy cập hẹp hơn. Đặt con trỏ gốc của trình lập kế hoạch thành 4 byte để xem điều đó loại bỏ những truy cập nào.
          </p>
        }
      >
        <p>
          NVIDIA's developer blog: device memory is aligned for the vector types, "but if you offset the pointer, the
          offset must also be aligned", and gives <code>reinterpret_cast&lt;int2*&gt;(d_in+1)</code> as invalid. PyTorch
          checks each pointer's alignment at run time and falls back to narrower accesses. Set the planner's base pointer
          to 4 bytes to see which accesses that rules out.
        </p>
      </Translated>
    ),
    source: "NVIDIA's developer blog (facts A11); PyTorch MemoryAccess.cuh (facts D4)",
    vi: { source: 'blog dành cho nhà phát triển của NVIDIA (facts A11); PyTorch MemoryAccess.cuh (facts D4)' },
  },
  'ch06/6.3/q-metal-index': {
    kind: 'choice',
    id: 'ch06/6.3/q-metal-index',
    prompt: (
      <Translated
        vi={
          <>
            Một đồng nghiệp chuyển phép cộng vector sang Metal. Bản cơ sở mỗi luồng một float của họ đánh chỉ số bằng vị trí luồng kiểu <code>uint</code>; kernel <code>float4</code> của họ chuyển nó sang <code>int</code> trước. Kernel float4 thắng vượt xa mức bạn dự đoán cho một kernel ở băng thông của nó. Trước khi ghi công cho các lần nạp 16 byte, bạn kiểm tra gì?
          </>
        }
      >
        <>
          A colleague ports vector addition to Metal. Their one-float-per-thread baseline indexes with the{' '}
          <code>uint</code> thread position; their <code>float4</code> kernel converts it to <code>int</code> first. The
          float4 kernel wins by far more than you predicted for a kernel at its bandwidth. Before crediting the 16-byte
          loads, what do you check?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Liệu chỉ số uint có kìm bản cơ sở lại không: đánh chỉ số cả hai bằng int rồi đo lại.">Whether the uint index held the baseline back: index both with int and measure again.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Không gì cả: kernel float4 phát ra một phần tư số lệnh nạp, nên phần lợi là của chúng.">Nothing: the float4 kernel issues a quarter of the load instructions, so the gain is theirs.</Translated>,
        why: <Translated vi="Mục 6.3 vừa lập luận rằng một kernel bị giới hạn bởi băng thông ở occupancy tối đa chẳng được lợi mấy từ việc giảm số lệnh. Và hai kernel còn khác nhau ở một điểm thứ hai.">Section 6.3 just argued that a bandwidth-bound kernel at full occupancy has little to gain from fewer instructions. And the two kernels differ in a second way.</Translated>,
      },
      {
        label: <Translated vi="Liệu phép cộng uint có chậm hơn phép cộng int trên GPU của Apple không.">Whether uint additions are slower than int additions on Apple GPUs.</Translated>,
        why: <Translated vi="Chi phí của phép cộng không phải vấn đề. Điều thay đổi là những gì trình biên dịch được phép giả định về chỉ số, và do đó cách nó có thể nạp.">The cost of the addition is not the issue. What changes is what the compiler may assume about the index, and so how it can load.</Translated>,
      },
      {
        label: <Translated vi="Liệu buffer float4 có được gắn ở một độ lệch là bội của 16 byte không.">Whether the float4 buffer is bound at an offset that is a multiple of 16 bytes.</Translated>,
        why: <Translated vi="Điều đó quyết định kernel float4 có hợp lệ không, chứ không giải thích vì sao bản cơ sở chậm.">That decides whether the float4 kernel is valid, not why the baseline is slow.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bài nói của Apple “Metal Compute on MacBook Pro”: “Due to the wrapping characteristics of uint in the shader language spec, this disables vectorized loads” (do đặc tính quay vòng của uint trong đặc tả ngôn ngữ shader, điều này vô hiệu hóa nạp dạng vector), còn với int thì “the load will be vectorized” (lần nạp sẽ được vector hóa). Một bản cơ sở đánh chỉ số bằng uint có thể chậm vì một lý do chẳng liên quan gì đến mục này, nên phép so sánh chỉ công bằng khi cả hai kernel đánh chỉ số theo cùng một cách. Bản cơ sở Metal mà lab 6.2 cung cấp và lời giải tham chiếu đều chuyển vị trí luồng sang int một lần rồi đánh chỉ số bằng nó.
          </p>
        }
      >
        <p>
          Apple's talk "Metal Compute on MacBook Pro": "Due to the wrapping characteristics of uint in the shader language
          spec, this disables vectorized loads", while with int "the load will be vectorized". A uint-indexed baseline can
          be slow for a reason that has nothing to do with this section, so a comparison is fair only when both kernels
          index the same way. Lab 6.2's provided Metal baseline and the reference solution both convert the thread
          position to int once and index with that.
        </p>
      </Translated>
    ),
    source: 'Apple tech talk, 2021 (facts C3)',
    vi: { source: 'bài nói kỹ thuật của Apple, 2021 (facts C3)' },
  },
  'ch06/6.3/q-int4': {
    kind: 'numeric',
    id: 'ch06/6.3/q-int4',
    prompt: <Translated vi={<>Một lần nạp 16 byte mang được bao nhiêu trọng số lượng tử hóa 4 bit?</>}>How many 4-bit quantised weights does one 16-byte load carry?</Translated>,
    answer: 32,
    explain: (
      <Translated vi={<p>16 byte × 8 bit / 4 bit = 32 (16 với int8). Một lệnh mang vào cả một nhóm trọng số đã đóng gói.</p>}>
        <p>16 bytes × 8 bits / 4 bits = 32 (16 for int8). One instruction brings in a whole group of packed weights.</p>
      </Translated>
    ),
    source: 'our arithmetic (the book shows only float4)',
    vi: { source: 'phép tính của khóa học này (sách chỉ trình bày float4)' },
  },
};

// ---------------------------------------------------------------- page
export function S63() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 6.2 kết thúc với hai cách giữ cho nhiều byte hơn được di chuyển: nhiều luồng hơn cùng phát lệnh nạp, hoặc nhiều byte hơn trong mỗi lần nạp. Mục này là cách thứ hai. Bắt đầu từ phép cộng vector như chương 2 đã viết (Fig. 2.10): mỗi luồng nạp hai float và ghi một. Các truy cập được hợp nhất (coalesced), và vì kernel dùng ít tài nguyên nên nó chạy ở occupancy (mức chiếm dụng) tối đa. Dù vậy, mỗi lệnh nạp hoặc ghi chỉ di chuyển 4 byte.
          </p>
        }
      >
        <p>
          Section 6.2 ended with two ways to keep more bytes moving: more threads issuing loads at once, or more bytes in
          each load. This section is the second. Start from vector addition as chapter 2 wrote it (Fig. 2.10): each thread
          loads two floats and stores one. The accesses are coalesced, and since the kernel uses few resources it runs at
          full occupancy. Even so, every load or store instruction moves 4 bytes.
        </p>
      </Translated>
      <Source section="6.3" pages="138" />

      <h2>
        <Translated vi="Bốn float mỗi lệnh">Four floats per instruction</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Fig. 6.12 làm cho mỗi lệnh di chuyển 16 byte. Khởi chạy số luồng bằng một phần tư. Để chỉ số <code>i</code> chỉ một đoạn bốn float thay vì một float. Và đọc qua một con trỏ tới <code>float4</code>, một kiểu 16 byte với các trường <code>.x</code>, <code>.y</code>, <code>.z</code> và <code>.w</code>: <code>float4 x4 = ((float4*)x)[i];</code>. Theo sách, việc giải tham chiếu nó khiến trình biên dịch phát ra một lệnh nạp, hoặc ghi, 16 byte cho cả đoạn; blog dành cho nhà phát triển của NVIDIA cho thấy cùng cách viết này được biên dịch thành các lệnh nạp và ghi 128 bit. Đây là nó trong lời giải tham chiếu của lab 6.2; bạn tự viết bản của mình trên <a href={href('ch06', 'labs')}>trang lab</a>.
          </p>
        }
      >
        <p>
          Fig. 6.12 makes each instruction move 16 bytes. Launch a quarter as many threads. Let the index <code>i</code>{' '}
          name a chunk of four floats instead of one float. And read through a pointer to <code>float4</code>, a 16-byte
          type with fields <code>.x</code>, <code>.y</code>, <code>.z</code> and <code>.w</code>:{' '}
          <code>float4 x4 = ((float4*)x)[i];</code>. Dereferencing it, the book says, makes the compiler emit one 16-byte
          load, or store, for the whole chunk; NVIDIA's developer blog shows the same idiom compiling to 128-bit loads and
          stores. Here it is as lab 6.2's reference solution; you write your own on the{' '}
          <a href={href('ch06', 'labs')}>labs page</a>.
        </p>
      </Translated>
      <Source section="6.3" pages="138" figs="Fig. 6.12" />
      <LabListing />

      <h2>
        <Translated vi="Hai biên">The two boundaries</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách để phần biên lại cho bạn (Exercise 4), và có hai biên. Nếu số phần tử không phải bội của bốn, luồng sở hữu đoạn cuối chưa đầy phải lùi về xử lý mỗi lần một float, vì một truy cập 16 byte sẽ chạy quá cuối mảng. Và những luồng của khối cuối có đoạn bắt đầu ở ngoài dữ liệu thì không được làm gì.
          </p>
        }
      >
        <p>
          The book leaves the edges to you (Exercise 4), and there are two. If the element count is not a multiple of
          four, the thread that owns the last, partial chunk must fall back to one float at a time, because a 16-byte
          access would run past the end of the array. And threads of the last block whose chunk starts past the data must
          do nothing.
        </p>
      </Translated>
      <Source section="6.3" pages="138–139" figs="Exercise 4, p. 155" />
      <ChunkPlanner />
      <Translated
        vi={
          <p>
            Trên GPU, một truy cập vượt quá cuối mảng vài byte có thể không bị phát hiện, đó là lý do các bài kiểm tra của lab 6.2 chạy trên trình giả lập (emulator) trước: các mảng của nó không có gì phía sau, và một float4 vươn quá cuối mảng sẽ bị báo cáo kèm dòng mã của nó.
          </p>
        }
      >
        <p>
          On a GPU an access a few bytes past the end of an array may go unnoticed, which is why lab 6.2's tests run on
          the emulator first: its arrays have nothing after them, and a float4 that reaches past the end is reported with
          its line.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trình lập kế hoạch đếm số lệnh mà các luồng thực thi. Nó không mô hình hóa thời gian, occupancy, bộ nhớ đệm (cache), hay cách các truy cập của một warp được gom thành giao dịch bộ nhớ.
            </p>
          }
        >
          <p>
            The planner counts instructions executed by threads. It does not model time, occupancy, caches, or how a warp's
            accesses are grouped into memory transactions.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsS63['ch06/6.3/q-tail']} />

      <h2>
        <Translated vi="Nó mang lại gì">What it buys</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách nêu hai lợi ích. Thứ nhất là ít lệnh hơn cho cùng số byte: hai lần nạp vector thay cho tám lần nạp vô hướng, ít hơn 75 % lệnh nạp, và lệnh ghi cũng giảm như vậy. Bộ đếm của trình lập kế hoạch cho thấy điều đó, và giảm ít hơn một chút khi có phần đuôi vô hướng.
          </p>
        }
      >
        <p>
          The book names two benefits. The first is fewer instructions for the same bytes: two vector loads replace eight
          scalar ones, 75 % fewer load instructions, and the stores fall the same way. The planner's counter shows that,
          and slightly less when there is a scalar tail.
        </p>
      </Translated>
      <Source section="6.3" pages="139" />
      <Ask q={questionsS63['ch06/6.3/q-instr']} />
      <Translated
        vi={
          <p>
            Lợi ích thứ hai quan trọng hơn với những kernel bạn đang hướng tới. Khi một kernel không thể chạy ở occupancy tối đa, các truy cập rộng hơn vẫn đưa đủ byte lên đường truyền để che giấu độ trễ bộ nhớ và tận dụng băng thông: trong trình khám phá byte đang trên đường truyền của mục 6.2, chuyển sang nạp 16 byte giảm số lần nạp trên toàn warp cần thiết đi bốn lần. Sách hứa một ví dụ ở chương 15, nơi một phép nhân ma trận đánh đổi occupancy lấy những lợi ích khác.
          </p>
        }
      >
        <p>
          The second matters more for the kernels you are headed for. When a kernel cannot run at full occupancy, wider
          accesses still put enough bytes in flight to hide the memory latency and use the bandwidth: in section 6.2's
          bytes-in-flight explorer, switching to 16-byte loads cuts the warp-wide loads needed by four. The book promises an
          example in chapter 15, where a matrix multiplication trades occupancy for other gains.
        </p>
      </Translated>
      <Source section="6.3" pages="139" />
      <Translated
        vi={
          <p>
            Phép cộng vector là trường hợp còn lại. Nó bị giới hạn bởi bộ nhớ, chạy ở occupancy tối đa, và các lần nạp của nó đã được hợp nhất. Hãy quyết định bao nhiêu phần trong 75 % đó biến thành thời gian trước khi bạn đo.
          </p>
        }
      >
        <p>
          Vector addition is the other case. It is memory-bound, it runs at full occupancy, and its loads are already
          coalesced. Decide how much of the 75 % becomes time before you measure.
        </p>
      </Translated>
      <LabsPage labs={LABS}>
        <Predict
          id="ch06/6.3/p-speedup"
          prompt={
            <Translated
              vi={
                <>
                  Trên thiết bị của bạn, ở kích thước mảng mà lab 6.2 đo hiệu năng, kernel float4 nhanh gấp bao nhiêu lần so với mỗi luồng một float?
                </>
              }
            >
              <>
                On your device, at the array size lab 6.2 benchmarks, how many times faster is the float4 kernel than one
                float per thread?
              </>
            </Translated>
          }
          unit="×"
        >
          {(v) => (
            <Reveal
              committed={v}
              measure={speedups}
              unit={tr('× one float per thread', '× mỗi luồng một float')}
              reading={
                <Translated
                  vi={
                    <>
                      Lab 6.2 đo thời gian ba kernel trên cùng các mảng: mỗi luồng một float, mỗi luồng bốn float xử lý từng cái một (gộp việc, mục 6.5), và bốn float mỗi lệnh. Đây là số đo trên máy của bạn, không phải giá trị kỳ vọng. So sánh kernel thứ hai và thứ ba để thấy bao nhiêu phần lợi ích là do chính truy cập vector.
                    </>
                  }
                >
                  <>
                    Lab 6.2 times three kernels on the same arrays: one float per thread, four floats per thread one at a
                    time (coarsened, section 6.5), and four per instruction. These are measurements on your machine, not
                    expected values. Compare the second and third to see how much of any gain is the vector access itself.
                  </>
                </Translated>
              }
              missing={
                <Translated
                  vi={
                    <>
                      Chạy <code>python3 labs/pmpp.py run 6.2 --solution --bench</code> trên một máy có GPU cho lời giải tham chiếu (hoặc <code>run 6.2 --bench</code> khi các kernel của bạn đã qua), rồi tải lại trang.
                    </>
                  }
                >
                  <>
                    Run <code>python3 labs/pmpp.py run 6.2 --solution --bench</code> on a machine with a GPU for the
                    reference solution (or <code>run 6.2 --bench</code> once your own kernels pass), then reload.
                  </>
                </Translated>
              }
            />
          )}
        </Predict>
      </LabsPage>
      <Ask q={questionsS63['ch06/6.3/q-when']} />

      <h2>
        <Translated vi="Phép ép kiểu có được phép không?">Is the cast allowed?</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phép ép kiểu từ <code>float*</code> sang <code>float4*</code> là kỹ thuật của sách, và mã production có dùng nó. Sách không nêu điều kiện nào cho con trỏ.
          </p>
        }
      >
        <p>
          The cast from <code>float*</code> to <code>float4*</code> is the book's technique, and production code uses it.
          The book states no condition on the pointer.
        </p>
      </Translated>
      <Source section="6.3" pages="138" />
      <Translated
        vi={
          <p>
            Các tài liệu tham khảo của NVIDIA không mô tả phép ép kiểu này, nhưng blog dành cho nhà phát triển của họ dạy cùng cách viết và thêm điều kiện: các lệnh này “require aligned data” (yêu cầu dữ liệu được căn chỉnh). Một <code>float4</code> có 16 byte với căn chỉnh 16 byte, nên địa chỉ của mọi truy cập vector phải là bội của 16. Con trỏ từ bộ cấp phát được căn chỉnh vượt xa mức đó; một con trỏ vào giữa mảng thì có thể không. Đặt con trỏ gốc của trình lập kế hoạch thành 4 byte, đúng như <code>x + 1</code> cho bạn, và mọi truy cập float4 đều bị loại.
          </p>
        }
      >
        <p>
          NVIDIA's reference guides do not describe the cast, but its developer blog teaches the same idiom and adds the
          condition: these instructions "require aligned data". A <code>float4</code> is 16 bytes with 16-byte alignment,
          so the address of every vector access must be a multiple of 16. An allocator's pointer is aligned far beyond
          that; a pointer into the middle of an array may not be. Set the planner's base pointer to 4 bytes, which is what{' '}
          <code>x + 1</code> gives you, and every float4 access is ruled out.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch06-facts.md A11, B5; infra-facts.md A6, B6)">{/* i18n-ignore: date and notes reference */}
        <Translated
          vi={
            <p>
              Blog của NVIDIA (cập nhật 2025-08-21) cũng cảnh báo rằng nạp dạng vector làm tăng áp lực thanh ghi. PTX có lệnh nạp 32 byte trên sm_100 trở về sau; <code>long4</code> và <code>double4</code> bị đánh dấu lỗi thời trong CUDA 13. AMD: <code>float4</code> được căn chỉnh 16 byte; không trang nào của AMD mô tả phép ép kiểu.
            </p>
          }
        >
          <p>
            NVIDIA's blog (updated 2025-08-21) also warns that vector loads raise register pressure. PTX has 32-byte loads
            on sm_100 and later; <code>long4</code> and <code>double4</code> are deprecated in CUDA 13. AMD: <code>float4</code>{' '}
            is 16-byte aligned; no AMD page describes the cast.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS63['ch06/6.3/q-align']} />
      <Translated
        vi={
          <p>
            Metal có con đường riêng được ghi trong tài liệu, và các tab Metal của lời giải tham chiếu đi theo nó. Apple mô tả việc đọc bốn float một lúc bằng cách khai báo buffer là <code>device float4*</code> (căn chỉnh 16 byte) hoặc <code>device packed_float4*</code> (căn chỉnh 4 byte), nên không cần ép kiểu; khi đó độ lệch gắn (bind offset) của một buffer device phải là bội của 16. Kernel của lab nhận cùng các buffer đó lần thứ hai dưới dạng <code>float</code> thường cho phần đuôi, và đánh chỉ số bằng <code>int</code>, vì một lý do mà câu hỏi tiếp theo đề cập.
          </p>
        }
      >
        <p>
          Metal has its own documented route, and the reference solution's Metal tabs take it. Apple documents reading four
          floats at once by declaring the buffer <code>device float4*</code> (16-byte aligned) or{' '}
          <code>device packed_float4*</code> (4-byte aligned), so no cast is needed; the bind offset of a device buffer must
          then be a multiple of 16. The lab's kernel receives the same buffers a second time as plain <code>float</code> for
          the tail, and indexes with <code>int</code>, for a reason the next question asks about.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch06-facts.md C3; infra-facts.md C7)">{/* i18n-ignore: date and notes reference */}
        <Translated
          vi={
            <p>
              MLX và llama.cpp có ép kiểu con trỏ trong MSL; đặc tả không mô tả điều này. Chưa kiểm chứng (unverified): việc phép ép kiểu như vậy trở thành một lần nạp 16 byte trên GPU (bằng chứng chỉ ở mức biên dịch).
            </p>
          }
        >
          <p>
            MLX and llama.cpp do cast pointers in MSL; the specification does not describe it. Unverified: that such a cast
            becomes one 16-byte load on the GPU (the evidence is compile-only).
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS63['ch06/6.3/q-metal-index']} />

      <h2>
        <Translated vi="Bạn sẽ gặp nó ở đâu">Where you will meet it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các kernel elementwise của PyTorch chính là Exercise 4 trong production. Chúng đọc qua một kiểu vector được căn chỉnh với cùng loại ép kiểu, nhưng chọn độ rộng lúc chạy: một hàm phụ trợ kiểm tra địa chỉ của con trỏ và trả về độ rộng lớn nhất trong 8, 4, 2 hoặc 1 phần tử mà nó được căn chỉnh cho, và kernel lùi về các truy cập hẹp hơn khi buộc phải vậy. Bộ đếm cuối cùng của trình lập kế hoạch áp dụng quy tắc đó cho con trỏ gốc của nó.
          </p>
        }
      >
        <p>
          PyTorch's elementwise kernels are Exercise 4 in production. They read through an aligned vector type with the
          same kind of cast, but choose its width at run time: a helper tests the pointer's address and returns the widest
          of 8, 4, 2 or 1 elements it is aligned for, and the kernel falls back to narrower accesses when it must. The
          planner's last counter applies that rule to its base pointer.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch06-facts.md D4)">{/* i18n-ignore: date and notes reference */}
        <Translated
          vi={
            <p>
              <code>aten/src/ATen/native/cuda/MemoryAccess.cuh</code>: <code>aligned_vector</code> được đọc qua <code>reinterpret_cast</code>; <code>can_vectorize_up_to</code> trả về 8, 4, 2 hoặc 1. Đọc trên nhánh mặc định.
            </p>
          }
        >
          <p>
            <code>aten/src/ATen/native/cuda/MemoryAccess.cuh</code>: <code>aligned_vector</code> read through{' '}
            <code>reinterpret_cast</code>; <code>can_vectorize_up_to</code> returns 8, 4, 2 or 1. Read on the default branch.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Ý tưởng này càng quan trọng với các mô hình đã lượng tử hóa, có trọng số được đóng gói chỉ vài bit mỗi cái: khi đó một lần nạp 16 byte mang vào cả một nhóm trọng số chỉ với một lệnh.
          </p>
        }
      >
        <p>
          The idea matters even more for quantised models, whose weights are packed into a few bits each: one 16-byte load
          then brings in a whole group of weights for a single instruction.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Phép tính trọng số đóng gói là của khóa học này; sách chỉ trình bày nạp dạng vector cho float4.</p>}>
          <p>The packed-weights arithmetic is ours; the book shows vector loads only for float4.</p>
        </Translated>
      </Beyond>
      <Ask q={questionsS63['ch06/6.3/q-int4']} />
    </>
  );
}
