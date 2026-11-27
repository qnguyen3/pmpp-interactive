import { useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import {
  CATEGORIES,
  CATEGORY_AIM,
  CATEGORY_NAME,
  OUTCOME_LABEL,
  ROWS,
  SCENARIOS,
  SYMPTOMS,
  categoryCounts,
  countLine,
  rankRows,
  rowById,
  sortScore,
  type Outcome,
  type Placement,
  type Row,
  type SymptomId,
} from '../../sim/checklist';
import { Translated, tr } from '../../lib/i18n';

export const questionsS68: Record<string, Question> = {
  'ch06/6.8/q-tiling-occ': {
    kind: 'choice',
    id: 'ch06/6.8/q-tiling-occ',
    prompt: (
      <Translated
        vi={
          <>
            Tốc độ của một matmul phân ô bị giới hạn bởi occupancy, occupancy của nó đã bị giới hạn bởi mức dùng bộ nhớ chia sẻ, và profiler cho thấy nó hiếm khi phải chờ bộ nhớ toàn cục. Ai đó cho mỗi khối T × T luồng tính một ô đầu ra 2T × 2T, bốn kết quả mỗi luồng, để giảm thêm lưu lượng toàn cục. Mỗi khối giờ giữ các ô của M và N lớn gấp đôi: gấp đôi bộ nhớ chia sẻ mỗi khối, cùng số luồng, một nửa lưu lượng toàn cục. Bạn kỳ vọng điều gì?
          </>
        }
      >
        <>
          A tiled matmul&rsquo;s speed is limited by occupancy, its occupancy is already limited by its shared-memory
          use, and the profiler shows it rarely waits on global memory. Someone has each block of T × T threads compute a 2T × 2T tile of the output, four
          results per thread, to cut global traffic further. Each block now keeps tiles of M and N twice the size: twice
          the shared memory per block, the same threads, half the global traffic. What do you expect?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Chậm hơn: nó tiêu thêm thứ đang giới hạn kernel để giải tỏa thứ không giới hạn.">Slower: it spends more of what limits the kernel to relieve what does not.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Nhanh hơn: giảm một nửa lưu lượng toàn cục giải phóng băng thông cho mọi khối thường trú.">Faster: halving the global traffic frees bandwidth for every resident block.</Translated>,
        why: <Translated vi="Ít lưu lượng hơn chỉ có ích nếu lưu lượng là giới hạn. Ở đây thì không, và phần bộ nhớ chia sẻ tăng thêm làm giảm số khối thường trú.">Less traffic helps only if traffic was the limit. Here it was not, and the extra shared memory removes resident blocks.</Translated>,
      },
      {
        label: <Translated vi="Không thay đổi: lưu lượng giảm, nhưng kernel vốn không chờ bộ nhớ toàn cục.">No change: the traffic falls, but the kernel was not waiting on global memory.</Translated>,
        why: <Translated vi="Có thứ khác thực sự thay đổi: gấp đôi bộ nhớ chia sẻ mỗi khối với cùng số luồng chỉ còn chỗ cho khoảng một nửa số khối mỗi SM, nên occupancy giảm thêm.">Something else does move: twice the shared memory per block at the same thread count leaves room for about half as many blocks per SM, so occupancy falls further.</Translated>,
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Đây chính là lời cảnh báo của sách ở p. 153, và là lý do danh sách kiểm tra có cột chi phí trong khóa học này. Ít khối thường trú hơn nghĩa là ít warp hơn để che giấu độ trễ, và ít yêu cầu bộ nhớ đang chờ phục vụ hơn (6.2). Mục 6.9 cho bạn xem điều đó xảy ra trong một mô hình.
            </p>
          }
        >
          <p>
            This is the book&rsquo;s own warning on p. 153, and the reason the checklist has a cost column in this
            course. Fewer resident blocks means fewer warps to hide latency with, and fewer memory requests in flight
            (6.2). Section 6.9 lets you watch it happen in a model.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Chỉ nhân đôi <code>TILE_WIDTH</code> trong kernel của chương 5 sẽ không kiểm tra được điều này. Khối của nó có T × T luồng với một ô T × T của M và một của N, nên một ô rộng gấp đôi có bộ nhớ chia sẻ gấp bốn và số luồng gấp bốn. Bộ nhớ chia sẻ mỗi luồng vẫn là hai float, 8 byte, và một occupancy bị giới hạn bởi bộ nhớ chia sẻ vẫn ở nguyên chỗ cũ, trừ sai số làm tròn. Thứ tiêu thêm tài nguyên đang giới hạn là nhiều bộ nhớ chia sẻ hơn trên mỗi luồng.
            </p>
          }
        >
          <p>
            Simply doubling <code>TILE_WIDTH</code> in chapter 5&rsquo;s kernel would not test this. Its block is T × T
            threads with one T × T tile of M and one of N, so a tile twice as wide has four times the shared memory and
            four times the threads. Shared memory per thread stays at two floats, 8 bytes, and an occupancy limited by
            shared memory stays where it was, apart from rounding. What spends more of the limiting resource is more
            shared memory per thread.
          </p>
        </Translated>
      </>
    ),
    source: 'ch. 6 §6.9, p. 153; occupancy budgets from ch. 5 §5.6',
    vi: { source: 'ch. 6 §6.9, p. 153; ngân sách occupancy từ ch. 5 §5.6' },
  },
  'ch06/6.8/q-sectors': {
    kind: 'choice',
    id: 'ch06/6.8/q-sectors',
    prompt: (
      <Translated
        vi={
          <>
            Trên một GPU NVIDIA, Nsight Compute báo cáo 32 sector mỗi yêu cầu cho các lần nạp toàn cục 4 byte của một kernel; mức lý tưởng cho các lần nạp 4 byte của cả một warp là 4. Hàng nào của danh sách kiểm tra, và chiến lược nào của hàng đó?
          </>
        }
      >
        <>
          On an NVIDIA GPU, Nsight Compute reports 32 sectors per request for a kernel&rsquo;s 4-byte global loads; the
          ideal for 4-byte loads by a full warp is 4. Which row of the checklist, and which of its strategies?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Hợp nhất truy cập: ánh xạ lại luồng với dữ liệu, đổi bố cục (chuyển vị, cấu trúc của các mảng), hoặc trung chuyển qua bộ nhớ chia sẻ (xoay góc).">Coalescing: remap threads to data, change the layout (transpose, structure of arrays), or stage through shared memory (corner turning).</Translated>,
        correct: true,
      },
      { label: <Translated vi="Tinh chỉnh occupancy: thêm warp, để có nhiều yêu cầu đang chờ phục vụ hơn.">Occupancy tuning: more warps, so that more requests are in flight.</Translated>, why: <Translated vi="Thêm warp chỉ đưa thêm những yêu cầu lãng phí y như vậy lên đường truyền. Mỗi yêu cầu vẫn nạp 32 sector cho 128 byte hữu ích.">More warps put more of the same wasteful requests in flight. Each request still fetches 32 sectors for 128 useful bytes.</Translated> },
      { label: <Translated vi="Đệm để chống xung đột bank, để các lane thôi dùng chung một bank.">Padding against bank conflicts, so lanes stop sharing a bank.</Translated>, why: <Translated vi="Xung đột bank là vấn đề của bộ nhớ chia sẻ. Số sector mỗi yêu cầu nói về các lần nạp toàn cục.">Bank conflicts are a shared-memory problem. Sectors per request is about global loads.</Translated> },
      { label: <Translated vi="Trải vòng lặp, để ít lệnh hơn đưa ra cùng các yêu cầu.">Loop unrolling, so fewer instructions make the same requests.</Translated>, why: <Translated vi="Trải vòng lặp thay đổi số lệnh chạy, không thay đổi địa chỉ mỗi lệnh chạm tới.">Unrolling changes how many instructions run, not which addresses each one touches.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            32 sector cho 32 luồng nghĩa là 4 byte của mỗi luồng đến từ một sector 32 byte khác nhau: các luồng kề nhau trong warp cách nhau ít nhất 32 byte, đúng mẫu <code>N[col*Width + k]</code> của 6.1. Bảy phần tám mỗi sector bị nạp mà không dùng. Cách sửa nằm ở địa chỉ, nên nó thuộc hàng hợp nhất truy cập. Đơn vị sector và chỉ số đo là của NVIDIA (facts A2, A11, A17); AMD đếm theo dòng cache còn Apple không công bố đơn vị nào.
          </p>
        }
      >
        <p>
          32 sectors for 32 threads means every thread&rsquo;s 4 bytes came from a different 32-byte sector: the
          neighbours in the warp are at least 32 bytes apart, the <code>N[col*Width + k]</code> pattern of 6.1. Seven
          eighths of every sector is fetched and not used. The fix is in the addresses, so it is in the coalescing row.
          The sector unit and the metric are NVIDIA&rsquo;s (facts A2, A11, A17); AMD counts cache lines and Apple
          documents no unit.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.1 and §6.8, pp. 128–129, 150–151; notes/today/ch06-facts.md A2, A11, A17',
    vi: { source: 'ch. 6 §6.1 và §6.8, pp. 128–129, 150–151; notes/today/ch06-facts.md A2, A11, A17' },
  },
  'ch06/6.8/q-general': {
    kind: 'choice',
    id: 'ch06/6.8/q-general',
    prompt: (
      <Translated vi={<>Vì sao sách xếp gộp việc cho luồng vào mục &ldquo;chung&rdquo; thay vì một trong ba nhóm kia?</>}>
        Why does the book file thread coarsening under &ldquo;general&rdquo; instead of one of the three categories?
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Chi phí mà nó loại bỏ tùy thuộc vào kernel: nạp thừa trong matmul, bản sao riêng trong histogram, đồng bộ hóa trong phép rút gọn, hợp nhất truy cập kém trong sắp xếp.">The overhead it removes depends on the kernel: redundant loads in matmul, private copies in histogram, synchronisation in reduction, poor coalescing in sorting.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Nó không có lợi ích gì cho bộ nhớ, chỉ cho các lõi tính toán.">It has no benefit to memory, only to the compute cores.</Translated>, why: <Translated vi="Trong matmul lợi ích của nó hoàn toàn về bộ nhớ: ít lần nạp ô thừa hơn.">In matmul its benefit is entirely about memory: fewer redundant tile loads.</Translated> },
      { label: <Translated vi="Nó quá mới để phân loại: sách chỉ giới thiệu nó trong chương này.">It is too new to classify: the book introduces it only in this chapter.</Translated>, why: <Translated vi="Nó được dùng xuyên suốt Phần 2 và 3; lý do là tác dụng của nó thay đổi từ kernel này sang kernel khác.">It is used throughout Parts 2 and 3; the reason is that its effect changes from kernel to kernel.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mọi hàng khác được xếp theo mục đích chính của nó (p. 149). Gộp việc không có mục đích riêng: nó tuần tự hóa một phần công việc song song, và thứ nó mua được là bất kỳ chi phí nào mà phiên bản song song đang phải trả. Đó cũng là lý do nó có thể chẳng mua được gì (cạm bẫy 1 ở p. 142). Danh sách đầy đủ của sách: nạp thừa trong stencil và matmul, bản sao riêng trong histogram, đồng bộ hóa và phân kỳ trong phép rút gọn và scan, hợp nhất truy cập kém trong lọc và sắp xếp.
          </p>
        }
      >
        <p>
          Every other row is filed under its primary intent (p. 149). Coarsening has none of its own: it serialises some
          of the parallel work, and what that buys you is whatever overhead the parallel version was paying. That is
          also why it can buy nothing (pitfall 1 on p. 142). The book&rsquo;s full list: redundant loads in stencil and
          matmul, private copies in histogram, synchronisation and divergence in reduction and scan, poor coalescing in
          filter and sorting.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.8, pp. 149, 152–153',
  },
  'ch06/6.8/q-warp': {
    kind: 'choice',
    id: 'ch06/6.8/q-warp',
    prompt: <Translated vi={<>Các primitive cấp warp giúp một khối tránh được điều gì?</>}>What do warp-level primitives let a block avoid?</Translated>,
    options: [
      {
        label: <Translated vi="Một số rào chắn trên toàn khối, và vòng đi về qua bộ nhớ chia sẻ bên trong một warp.">Some block-wide barriers, and the shared-memory round trip within a warp.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Toàn bộ lưu lượng bộ nhớ toàn cục, vì các giá trị ở trong thanh ghi từ đầu đến cuối.">All global memory traffic, since values stay in registers from start to finish.</Translated>, why: <Translated vi="Dữ liệu vẫn đến từ và đi về bộ nhớ toàn cục. Điều thay đổi là cách các luồng của một warp chia sẻ nó.">The data still comes from and goes to global memory. What changes is how threads of a warp share it.</Translated> },
      { label: <Translated vi="Phân kỳ điều khiển, vì khi đó mọi lane của một warp đi cùng một đường.">Control divergence, since all lanes of a warp then take the same path.</Translated>, why: <Translated vi="Chúng không thay đổi đường mà các luồng đi; đó là hàng phân kỳ.">They do not change which path threads take; that is the divergence row.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các luồng của một warp có thể trao giá trị trực tiếp cho nhau và đồng bộ hóa với nhau (chương 4 đã nêu điều này; các chương 10 đến 12 dùng nó). Khối vẫn cần một rào chắn ở chỗ các warp trao đổi kết quả. Ghi chú khi chuyển mã: mặt nạ lane là 32 bit trong CUDA và phải là 64 bit trong HIP, còn các hàm SIMD-group của Metal không nhận mặt nạ nào (infra facts B3, C2).
          </p>
        }
      >
        <p>
          Threads of a warp can hand values to each other directly and synchronise among themselves (chapter 4 named
          this; chapters 10 to 12 use it). The block still needs a barrier where warps exchange results. Porting note:
          the lane mask is 32 bits in CUDA and must be 64 bits in HIP, and Metal&rsquo;s SIMD-group functions take no
          mask (infra facts B3, C2).
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.8, p. 152; ch. 4 §4.4, pp. 77–78', // i18n-ignore: citation only
  },
  'ch06/6.8/q-vector-memory': {
    kind: 'choice',
    id: 'ch06/6.8/q-vector-memory',
    prompt: (
      <Translated vi={<>Vì sao bảng ghi cho nạp dạng vector lợi ích về bộ nhớ là &ldquo;nhiều truy cập bộ nhớ song song hơn&rdquo;?</>}>
        Why does the table give vector loads the memory benefit &ldquo;more parallel memory accesses&rdquo;?
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Mỗi lệnh đưa nhiều byte hơn lên đường truyền, che độ trễ DRAM khi có ít warp.">Each instruction puts more bytes in flight, covering DRAM latency when warps are few.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Nạp dạng vector di chuyển tổng cộng ít byte hơn, để lại nhiều chỗ trống trên bus cho các truy cập khác.">Vector loads move fewer bytes in total, which leaves more of the bus free for other accesses.</Translated>, why: <Translated vi="Số byte là như nhau. Chỉ số lệnh yêu cầu chúng giảm đi.">The bytes are the same. Only the number of instructions that ask for them falls.</Translated> },
      { label: <Translated vi="Nạp dạng vector tránh xung đột bank, nên nhiều truy cập diễn ra cùng lúc hơn.">Vector loads avoid bank conflicts, so more accesses proceed at once.</Translated>, why: <Translated vi="Bank thuộc về bộ nhớ chia sẻ; hàng này nói về các lần nạp toàn cục.">Banks belong to shared memory; this row is about global loads.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mục 6.2: bus chỉ được giữ bận nếu có đủ yêu cầu đang chờ, và các warp thường trú là một cách để có chúng. Mục 6.3, p. 139: khi một kernel không đạt được occupancy tối đa, các lần nạp rộng hơn là cách còn lại. Cùng lợi ích đó xuất hiện ở hàng occupancy, từ hướng ngược lại.
          </p>
        }
      >
        <p>
          Section 6.2: the bus is kept busy only if enough requests are outstanding, and resident warps are one way to
          have them. Section 6.3, p. 139: when a kernel cannot reach full occupancy, wider loads are the other way. The
          same benefit appears in the occupancy row, from the other direction.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.2, pp. 135–136; §6.3, p. 139; Fig. 6.13, p. 148', // i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- the decision tool
const ROLE_CHIP: Record<'relieves' | 'spends', ReactNode> = {
  relieves: (
    <span className="chip ok">
      <Translated vi="giải tỏa nó">relieves it</Translated>
    </span>
  ),
  spends: (
    <span className="chip fail">
      <Translated vi="tiêu thứ đang thiếu">spends what is short</Translated>
    </span>
  ),
};

function CostLine({ row }: { row: Row }) {
  return (
    <p style={{ margin: '0.35rem 0 0' }}>
      <b>
        <Translated vi="Chi phí:">Costs:</Translated>
      </b>{' '}
      {row.cost.text}{' '}
      {row.cost.source ? <span className="muted">({row.cost.source})</span> : null}
    </p>
  );
}

function RowDetails({ row }: { row: Row }) {
  return (
    <div style={{ marginTop: '0.6rem', borderTop: '1px dashed var(--rule)', paddingTop: '0.6rem' }} data-details={row.id}>
      <table className="data" style={{ margin: 0 }}>
        <tbody>
          <tr>
            <th scope="row"><Translated vi="Lợi ích cho các lõi">Benefit to the cores</Translated></th>
            <td>{row.benefitCompute}</td>
          </tr>
          <tr>
            <th scope="row"><Translated vi="Lợi ích cho bộ nhớ">Benefit to memory</Translated></th>
            <td>{row.benefitMemory ?? <span className="muted"><Translated vi="bảng không liệt kê">none listed in the table</Translated></span>}</td>
          </tr>
          <tr>
            <th scope="row"><Translated vi="Chiến lược">Strategies</Translated></th>
            <td>
              {row.strategies.map((s, i) => (
                <div key={i}>{s}</div>
              ))}
            </td>
          </tr>
          <tr>
            <th scope="row"><Translated vi="Phần văn bản bổ sung">What the text adds</Translated></th>
            <td>
              {row.commentary} <span className="muted">(pp. {row.pages})</span>
            </td>
          </tr>
          <tr>
            <th scope="row"><Translated vi="Xuất hiện lại ở">Returns in</Translated></th>
            <td>
              {row.returnsIn.map((r, i) => (
                <span key={r.ch}>
                  {i ? '; ' : ''}
                  <a href={href('ch' + String(r.ch).padStart(2, '0'))}>ch. {r.ch}</a> {r.what}
                </span>
              ))}
            </td>
          </tr>
          <tr>
            <th scope="row"><Translated vi="Thấy trong các backend thực tế">Seen in real backends</Translated></th>
            <td>
              {row.seenIn.length ? (
                row.seenIn.map((s, i) => (
                  <div key={i} data-seen={s.fact}>
                    {s.absent ? <b>{tr('Not in ', 'Không có trong ')}</b> : null}
                    <code>{s.where}</code>: {s.what} <span className="muted">(facts {s.fact})</span>{/* i18n-ignore: facts-file reference */}
                  </div>
                ))
              ) : (
                <span className="muted"><Translated vi="chưa kiểm tra">not checked</Translated></span>
              )}
            </td>
          </tr>
          <tr>
            <th scope="row"><Translated vi="Được dạy ở">Taught in</Translated></th>
            <td>
              {row.taught.length ? (
                row.taught.map(([c, s], i) => (
                  <span key={c + s}>
                    {i ? ', ' : ''}
                    <a href={href(c, s)}>{c === 'ch06' ? `§${s}` : `ch. ${Number(c.slice(2))} §${s}`}</a>{/* i18n-ignore: chapter citation */}
                  </span>
                ))
              ) : (
                <span className="muted"><Translated vi="từ chương 9 trở đi">chapter 9 onwards</Translated></span>
              )}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function DecisionTool() {
  const [symptom, setSymptom] = useState<SymptomId | null>('global-traffic');
  const [open, setOpen] = useState<Set<string>>(new Set());
  const ranked = rankRows(symptom);
  const s = SYMPTOMS.find((x) => x.id === symptom);
  const toggle = (id: string) =>
    setOpen((o) => {
      const n = new Set(o);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  const allOpen = open.size === ROWS.length;
  return (
    <Bench
      title={tr('Checklist as a decision tool', 'Danh sách kiểm tra như một công cụ ra quyết định')}
      task={tr('Pick what you observe. The rows that relieve it rise to the top; open a row for the rest of it.', 'Chọn điều bạn quan sát được. Các hàng giải tỏa nó sẽ nổi lên trên cùng; mở một hàng để xem phần còn lại.')}
      foot={
        <Translated vi={<>Mười hai hàng của Fig. 6.13 theo cách diễn đạt của chúng tôi, với các cột của sách (nhóm, lợi ích cho các lõi, lợi ích cho bộ nhớ, chiến lược) và ba cột của chúng tôi: chi phí, nơi nó xuất hiện lại, và nơi nó được thấy trong các backend thực tế (chỉ từ <code>notes/today/ch06-facts.md</code> D1 đến D4). Các triệu chứng và thứ hạng của chúng là của chúng tôi; mỗi liên kết dẫn tới trang sách hỗ trợ nó.</>}>
          The twelve rows of Fig. 6.13 in our wording, with the book&rsquo;s columns (category, benefit to the cores,
          benefit to memory, strategies) and three of ours: what it costs, where it returns, and where it is seen in
          real backends (from <code>notes/today/ch06-facts.md</code> D1 to D4 only). The symptoms and their ranking are
          ours; each link cites the page that supports it.
        </Translated>
      }
    >
      <div className="row" role="radiogroup" aria-label={tr('What you observe', 'Điều bạn quan sát được')} style={{ marginBottom: '0.8rem' }}>
        {SYMPTOMS.map((x) => (
          <button
            key={x.id}
            className="quiet"
            aria-pressed={symptom === x.id}
            data-symptom={x.id}
            onClick={() => setSymptom(x.id)}
          >
            {x.label}
          </button>
        ))}
        <button className="quiet" aria-pressed={symptom === null} data-symptom="none" onClick={() => setSymptom(null)}>
          <Translated vi="Thứ tự của sách">Book order</Translated>
        </button>
      </div>
      <p className="status-line" data-looks>
        {s ? (
          <Translated
            vi={
              <>
                <b>Có vẻ là:</b> {s.looksLike} {s.relief.length} hàng giải tỏa nó
                {s.spends.length ? `; ${s.spends.length} hàng sẽ tiêu thứ vốn đã thiếu` : ''}.
              </>
            }
          >
            <b>Looks like:</b> {s.looksLike} {s.relief.length} row{s.relief.length > 1 ? 's' : ''} relieve
            {s.relief.length > 1 ? '' : 's'} it
            {s.spends.length ? `; ${s.spends.length} would spend what is already short` : ''}.
          </Translated>
        ) : (
          <Translated vi={<>Bảng theo thứ tự của sách: ba hàng cho tính toán, sáu cho bộ nhớ, hai cho đồng bộ hóa, một hàng chung.</>}>
            <>The table in the book&rsquo;s order: three rows for compute, six for memory, two for synchronisation, one general.</>
          </Translated>
        )}
      </p>
      <div className="row" style={{ margin: '0.4rem 0 0.8rem' }}>
        <button className="quiet" onClick={() => setOpen(allOpen ? new Set() : new Set(ROWS.map((r) => r.id)))}>
          {allOpen ? tr('Close all rows', 'Đóng tất cả các hàng') : tr('Open all rows', 'Mở tất cả các hàng')}
        </button>
      </div>
      <ol style={{ listStyle: 'none', padding: 0, margin: 0, display: 'grid', gap: '0.5rem' }}>
        {ranked.map(({ row, role, relief }) => (
          <li
            key={row.id}
            className="panel"
            data-row={row.id}
            data-role={role}
            style={{
              opacity: role === 'other' && symptom ? 0.72 : 1,
              borderLeft:
                role === 'relieves' ? '3px solid var(--ok)' : role === 'spends' ? '3px solid var(--fail)' : undefined,
            }}
          >
            <div className="row" style={{ justifyContent: 'space-between', gap: '0.4rem 1rem' }}>
              <div>
                <b>{row.name}</b> <span className="muted">· {CATEGORY_NAME[row.category]}</span>{' '}
                {role !== 'other' ? ROLE_CHIP[role] : null}
              </div>
              <button className="quiet" aria-expanded={open.has(row.id)} onClick={() => toggle(row.id)}>
                {open.has(row.id) ? tr('Less', 'Thu gọn') : tr('More', 'Xem thêm')}
              </button>
            </div>
            {relief ? (
              <p style={{ margin: '0.35rem 0 0' }} data-how>
                <b>{role === 'relieves' ? tr('How:', 'Cách làm:') : tr('Why it hurts here:', 'Vì sao nó có hại ở đây:')}</b> {relief.how}{' '}
                <span className="muted">({relief.page})</span>
              </p>
            ) : null}
            <CostLine row={row} />
            {open.has(row.id) ? <RowDetails row={row} /> : null}
          </li>
        ))}
      </ol>
    </Bench>
  );
}

// ---------------------------------------------------------------- sorting drill
function SortDrill() {
  const [place, setPlace] = useState<Placement>({});
  const score = sortScore(place);
  const book = categoryCounts();
  return (
    <Bench
      title={tr('Sort the twelve', 'Phân loại mười hai hàng')}
      task={tr('File each optimisation under the category the book gives it. Each card says at once whether you agree with the book.', 'Xếp mỗi tối ưu hóa vào nhóm mà sách gán cho nó. Mỗi thẻ cho biết ngay bạn có đồng ý với sách hay không.')}
      foot={
        <Translated vi={<>Quy tắc của sách (p. 149): một hàng có ích theo nhiều cách được xếp theo mục đích chính của nó. Lý do trên các thẻ là của chúng tôi.</>}>
          The book&rsquo;s rule (p. 149): a row that helps in several ways is filed under its primary intent. The
          reasons on the cards are ours.
        </Translated>
      }
    >
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))' }}>
        {ROWS.map((r) => {
          const c = place[r.id];
          const verdict = c ? (c === r.category ? 'right' : 'wrong') : undefined;
          return (
            <div key={r.id} className="panel" data-card={r.id} data-verdict={verdict ?? 'open'}>
              <div>
                <b>{r.name}</b>
              </div>
              <div className="seg" style={{ display: 'flex', marginTop: '0.45rem', flexWrap: 'wrap' }}>
                {CATEGORIES.map((k) => (
                  <button
                    key={k}
                    aria-pressed={c === k}
                    data-cat={k}
                    onClick={() => setPlace((p) => ({ ...p, [r.id]: k }))}
                  >
                    {k === 'compute' ? tr('Compute', 'Tính toán') : k === 'memory' ? tr('Memory', 'Bộ nhớ') : k === 'sync' ? tr('Sync', 'Đồng bộ') : tr('General', 'Chung')}
                  </button>
                ))}
              </div>
              {verdict ? (
                <p style={{ margin: '0.45rem 0 0' }}>
                  {verdict === 'right' ? (
                    <span className="t-ok"><Translated vi="Giống sách.">Same as the book.</Translated></span>
                  ) : (
                    <span className="t-fail"><Translated vi={<>Sách xếp nó vào nhóm {CATEGORY_NAME[r.category].toLowerCase()}.</>}>The book files it under {CATEGORY_NAME[r.category].toLowerCase()}.</Translated></span>
                  )}{' '}
                  {r.why}
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <table className="data" style={{ marginTop: '1rem', marginBottom: '0.4rem' }} aria-label={tr('Tally', 'Bảng đếm')}>
        <thead>
          <tr>
            <th><Translated vi="Nhóm">Category</Translated></th>
            <th><Translated vi="Nó nhắm vào gì (p. 149)">What it targets (p. 149)</Translated></th>
            <th className="num"><Translated vi="Của bạn">Yours</Translated></th>
            <th className="num"><Translated vi="Của sách">The book&rsquo;s</Translated></th>
          </tr>
        </thead>
        <tbody>
          {CATEGORIES.map((k) => (
            <tr key={k} data-tally={k}>
              <td>{CATEGORY_NAME[k]}</td>
              <td>{CATEGORY_AIM[k]}</td>
              <td className="num">{score.tally[k].yours}</td>
              <td className="num">{score.done ? book[k] : '?'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="row">
        <p className="status-line" data-sort-result style={{ margin: 0 }}>
          {score.done
            ? tr(`All twelve placed. ${score.right} of 12 where the book puts them. The book's count: ${countLine(book)} = 12.`, `Đã xếp cả mười hai. ${score.right} trên 12 đúng chỗ sách xếp. Cách đếm của sách: ${countLine(book)} = 12.`)
            : tr(`${score.placed} of 12 placed.`, `Đã xếp ${score.placed} trên 12.`)}
        </p>
        <button className="quiet" onClick={() => setPlace({})} disabled={!score.placed}>
          <Translated vi="Làm lại từ đầu">Start again</Translated>
        </button>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- scenario cards
const OUTCOMES: Outcome[] = ['faster', 'same', 'slower'];
const OUTCOME_CODE: Record<Outcome, number> = { faster: 0, same: 1, slower: 2 };

function Scenarios() {
  const { progress, answer } = useProgress();
  const picks = SCENARIOS.map((s) => {
    const v = progress.answers[`ch06/6.8/p-${s.id}`]?.value;
    return typeof v === 'number' ? OUTCOMES[v] : undefined;
  });
  const done = picks.filter(Boolean).length;
  const right = SCENARIOS.filter((s, i) => picks[i] === s.expect).length;
  return (
    <Bench
      title={tr('Scenario cards', 'Thẻ tình huống')}
      task={tr('For each proposal, predict: faster, about the same, or slower. Then read why.', 'Với mỗi đề xuất, hãy dự đoán: nhanh hơn, gần như như cũ, hay chậm hơn. Rồi đọc lý do.')}
      foot={
        <Translated vi={<>Dự đoán đầu tiên của bạn cho mỗi thẻ được lưu cùng tiến độ học. Các tình huống là của chúng tôi; mỗi tình huống được quyết định bởi quy tắc ở p. 153 và trang sách đi kèm.</>}>
          Your first prediction for each card is kept with your progress. The cases are ours; each is decided by the
          rule of p. 153 and the page given with it.
        </Translated>
      }
    >
      <div style={{ display: 'grid', gap: '0.7rem' }}>
        {SCENARIOS.map((s, i) => {
          const pick = picks[i];
          const row = rowById(s.row);
          return (
            <div key={s.id} className="panel" data-scenario={s.id} data-picked={pick ?? 'none'}>
              <p style={{ margin: '0 0 0.3rem' }}>
                <b>
                  <Translated vi={<>Thẻ {i + 1}.</>}>Card {i + 1}.</Translated>
                </b>{' '}
                {s.situation}
              </p>
              <p style={{ margin: '0 0 0.5rem' }}>
                <b>
                  <Translated vi="Đề xuất:">Proposal:</Translated>
                </b>{' '}
                {s.change}
              </p>
              <div className="seg">
                {OUTCOMES.map((o) => (
                  <button
                    key={o}
                    aria-pressed={pick === o}
                    data-outcome={o}
                    disabled={!!pick}
                    onClick={() => answer(`ch06/6.8/p-${s.id}`, OUTCOME_CODE[o], o === s.expect)}
                  >
                    {OUTCOME_LABEL[o]}
                  </button>
                ))}
              </div>
              {pick ? (
                <div style={{ marginTop: '0.5rem' }} data-reveal>
                  <p style={{ margin: '0 0 0.3rem' }}>
                    {pick === s.expect ? (
                      <span className="t-ok">
                        <b>
                          <Translated vi={<>{OUTCOME_LABEL[s.expect]}, đúng như bạn dự đoán.</>}>{OUTCOME_LABEL[s.expect]}, as you predicted.</Translated>
                        </b>
                      </span>
                    ) : (
                      <span className="t-fail">
                        <b>
                          <Translated vi={<>Bạn dự đoán {OUTCOME_LABEL[pick].toLowerCase()}; quy tắc nói {OUTCOME_LABEL[s.expect].toLowerCase()}.</>}>
                            You predicted {OUTCOME_LABEL[pick].toLowerCase()}; the rule says{' '}
                            {OUTCOME_LABEL[s.expect].toLowerCase()}.
                          </Translated>
                        </b>
                      </span>
                    )}
                  </p>
                  <p style={{ margin: 0 }}>
                    {s.why} <span className="muted"><Translated vi={<>Hàng: {row.name}. Nguồn: {s.page}.</>}>Row: {row.name}. Source: {s.page}.</Translated></span>
                  </p>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
      <p className="status-line" data-scenario-result style={{ margin: '0.8rem 0 0' }}>
        {done === SCENARIOS.length
          ? tr(`${right} of ${SCENARIOS.length} predicted as the rule says.`, `${right} trên ${SCENARIOS.length} dự đoán đúng như quy tắc nói.`)
          : tr(`${done} of ${SCENARIOS.length} predicted.`, `Đã dự đoán ${done} trên ${SCENARIOS.length}.`)}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function S68() {
  const q = questionsS68;
  return (
    <>
      <Translated
        vi={
          <p>
            Chương này khép lại Phần 1 bằng cách gom những gì chương 4, 5 và 6 đã dạy vào một bảng gồm mười hai tối ưu hóa. Bảng không tự nhận là đầy đủ. Đây là những tối ưu hóa đáng cân nhắc trước tiên cho gần như mọi kernel; các chương sau bổ sung thêm của riêng chúng, như bộ nhớ hằng (constant memory) ở chương 7. Với mỗi hàng, bảng cho biết nó làm gì cho các lõi, làm gì cho bộ nhớ, và cách thực hiện. Phần 2 và 3 của sách liên tục nhắc lại bảng này.
          </p>
        }
      >
        <p>
          The chapter closes Part 1 by gathering what chapters 4, 5 and 6 taught into one table of twelve
          optimisations. It does not claim to be complete. These are the ones worth considering first for almost any
          kernel; later chapters add their own, such as constant memory in chapter 7. For each row the table gives
          what it does for the cores, what it does for memory, and how to do it. Parts 2 and 3 of the book refer back
          to it constantly.
        </p>
      </Translated>
      <Source section="6.8" pages="147–149, 153" figs="Fig. 6.13" />

      <h2>
        <Translated vi="Bốn nhóm, mười hai hàng">Four categories, twelve rows</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách sắp xếp các hàng theo khía cạnh hiệu năng mà mỗi hàng nhắm tới: giữ cho các lõi bận, dùng tốt hệ thống bộ nhớ, hoặc chờ ít hơn ở các rào chắn. Nhiều hàng có ích theo nhiều cách; mỗi hàng được xếp theo mục đích chính của nó. Một hàng có nhóm riêng. Hãy tự phân loại mười hai hàng trước khi bảng bên dưới cho bạn xem đáp án của sách.
          </p>
        }
      >
        <p>
          The book sorts the rows by the aspect of performance each one targets: keeping the cores busy, using the
          memory system well, or waiting less at barriers. Many rows help in more than one way; each is filed under its
          primary intent. One row gets a category of its own. Sort the twelve yourself before the table below shows
          you the book&rsquo;s answer.
        </p>
      </Translated>
      <Source section="6.8" pages="149" />
      <SortDrill />
      <Ask q={q['ch06/6.8/q-general']} />

      <h2>
        <Translated vi="Từ một triệu chứng đến một hàng">From a symptom to a row</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đọc như một bảng, danh sách kiểm tra trả lời câu &ldquo;phân ô làm gì?&rdquo;. Câu hỏi bạn thực sự sẽ có thì theo chiều ngược lại: &ldquo;kernel của tôi đang chờ thứ này; tôi nên thử gì?&rdquo;. Vì vậy ở đây bảng bắt đầu từ một triệu chứng. Mỗi hàng cũng mang một chi phí, vì toàn bộ lập luận của mục 6.9 là mỗi tối ưu hóa đều tiêu tốn một thứ gì đó.
          </p>
        }
      >
        <p>
          Read as a table, the checklist answers &ldquo;what does tiling do?&rdquo;. The question you will actually
          have runs the other way: &ldquo;my kernel waits on this; what do I try?&rdquo;. So here the table starts from
          a symptom. Every row also carries a cost, because section 6.9&rsquo;s whole argument is that each
          optimisation spends something.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Danh sách triệu chứng, thứ hạng, cách diễn đạt chi phí ở chỗ ghi &ldquo;not stated in the book&rdquo; (sách không nêu), và cột &ldquo;seen in&rdquo; (thấy trong) là của chúng tôi. Các chi phí mà chương nêu được dẫn tới trang của chúng; một chi phí từ nhà sản xuất (nạp dạng vector) được dẫn tới <code>ch06-facts.md</code> A11.
            </p>
          }
        >
          <p>
            The symptom list, the ranking, the cost wording where it says &ldquo;not stated in the book&rdquo;, and the
            &ldquo;seen in&rdquo; column are ours. Costs the chapter states are cited to their page; one vendor cost
            (vector loads) to <code>ch06-facts.md</code> A11.
          </p>
        </Translated>
      </Beyond>
      <DecisionTool />

      <Translated
        vi={
          <p>
            Ba điều cần để ý khi bạn bấm. Một nửa số hàng giải tỏa nhiều hơn một triệu chứng, nên cùng một tối ưu hóa cứ quay lại vì những lý do khác nhau. Gộp việc giải tỏa sáu trong tám triệu chứng, đó là ý nghĩa thực tế của nhãn &ldquo;chung&rdquo; của nó. Và với ba triệu chứng, có một hàng được đánh dấu đỏ: nó sẽ tiêu đúng thứ mà kernel đang thiếu.
          </p>
        }
      >
        <p>
          Three things to notice while you click. Half the rows relieve more than one symptom, so the same optimisation
          keeps coming back for different reasons. Coarsening relieves six of the eight symptoms, which is what its
          &ldquo;general&rdquo; label means in practice. And for three symptoms a row is marked in red: it would spend
          exactly what the kernel is short of.
        </p>
      </Translated>
      <Ask q={q['ch06/6.8/q-sectors']} />
      <Ask q={q['ch06/6.8/q-vector-memory']} />
      <Ask q={q['ch06/6.8/q-warp']} />

      <Translated
        vi={
          <p>
            Bảng tóm tắt các trang 149 đến 153, và ở hai hàng, phần tóm tắt và phần văn bản không hoàn toàn khớp nhau. Chỗ nào chúng khác nhau, hãy theo phần văn bản và chờ chương sau làm rõ.
          </p>
        }
      >
        <p>
          The table summarises pages 149 to 153, and in two rows the summary and the text do not quite agree. Where
          they differ, go by the text and expect the later chapter to settle it.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Bảng và văn bản khác nhau.</b> Phân ô thanh ghi: bảng nói dữ liệu di chuyển một lần giữa bộ nhớ chia sẻ và thanh ghi; văn bản (p. 151) nói giữa bộ nhớ toàn cục hoặc bộ nhớ chia sẻ và thanh ghi. Các primitive cấp warp: bảng khẳng định ít lưu lượng toàn cục và chia sẻ hơn; văn bản (p. 152) chỉ nhắc đến việc tránh bộ nhớ chia sẻ, và không nói vì sao lưu lượng toàn cục lại giảm.
            </p>
          }
        >
          <p>
            <b>Table and text differ.</b> Register tiling: the table says data moves once between shared memory and
            registers; the text (p. 151) says between global or shared memory and registers. Warp-level primitives: the
            table claims less global and shared traffic; the text (p. 152) mentions only avoiding shared memory, and
            does not say why global traffic would fall.
          </p>
        </Translated>
      </Aside>

      <h2>
        <Translated vi="Dự đoán trước khi áp dụng">Predict before you apply</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một hàng cho biết một tối ưu hóa có thể giải tỏa điều gì. Nó có làm kernel của bạn nhanh hơn hay không tùy thuộc vào việc đó có phải là thứ đang giới hạn kernel không, và tài nguyên nó tiêu có vốn đã thiếu không. Đó là quy tắc của mục <a href={href('ch06', '6.9')}>6.9</a>, nêu ở p. 153. Năm đề xuất; hãy quyết định từng cái trước khi xem.
          </p>
        }
      >
        <p>
          A row says what an optimisation can relieve. Whether it speeds up your kernel depends on whether that is what
          limits the kernel, and on whether the resource it spends was already short. That is the rule of section{' '}
          <a href={href('ch06', '6.9')}>6.9</a>, stated on p. 153. Five proposals; decide each one before you look.
        </p>
      </Translated>
      <Source section="6.9" pages="153" />
      <Scenarios />
      <Ask q={q['ch06/6.8/q-tiling-occ']} />

      <h2>
        <Translated vi="Bạn sẽ gặp các hàng này ở đâu">Where you will meet these rows</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Khi bạn đọc các backend GPU của PyTorch, MLX hay llama.cpp, các hàng này rất dễ nhận ra một khi bạn biết hình dạng của chúng. Các kernel elementwise của PyTorch dùng nạp dạng vector và chọn độ rộng cho từng tensor dựa trên căn chỉnh của con trỏ: phương án lùi về vô hướng mà sách để lại cho bài tập 4, được làm lúc chạy. llama.cpp đệm các ô matmul lượng tử hóa trong bộ nhớ chia sẻ để chống xung đột bank, và kernel flash-attention của nó nạp khối tiếp theo vào một buffer thứ hai trong khi tính toán trên buffer thứ nhất. MLX cũng đệm các ô GEMM Metal của nó, nhưng vòng lặp chính của nó chỉ dùng một buffer với hai rào chắn mỗi bước: vòng lặp &ldquo;trước&rdquo; của 6.7, trong mã production. Dòng &ldquo;seen in&rdquo; (thấy trong) của mỗi hàng ở trên nêu tên các tệp.
          </p>
        }
      >
        <p>
          When you read the GPU backends of PyTorch, MLX or llama.cpp, the rows are easy to spot once you know their
          shapes. PyTorch&rsquo;s elementwise kernels use vector loads and choose the width per tensor from the
          pointer&rsquo;s alignment: the scalar fallback the book leaves to exercise 4, done at run time. llama.cpp pads
          its quantised-matmul tiles in shared memory against bank conflicts, and its flash-attention kernel loads the
          next block into a second buffer while computing on the first. MLX pads its Metal GEMM tiles too, but its main
          loop is single-buffered with two barriers per step: the &ldquo;before&rdquo; loop of 6.7, in production
          code. The &ldquo;seen in&rdquo; line of each row above names the files.
        </p>
      </Translated>
      <Today checked="2026-11-20">{/* i18n-ignore: date and notes reference */}
        <Translated
          vi={
            <p>
              Đọc trên nhánh mặc định của các dự án (<code>notes/today/ch06-facts.md</code> D1 đến D4). Chi tiết trong tệp thay đổi theo thời gian; tệp facts ghi ngày cho từng mục. Một hàng được đánh dấu &ldquo;chưa kiểm tra&rdquo; là chưa được tìm, điều đó không giống với không có.
            </p>
          }
        >
          <p>
            Read on the projects&rsquo; default branches (<code>notes/today/ch06-facts.md</code> D1 to D4). File details
            drift; the facts file dates each one. A row marked &ldquo;not checked&rdquo; was not looked for, which is
            not the same as absent.
          </p>
        </Translated>
      </Today>
    </>
  );
}
