import { Ask, Predict, type Question } from '../../components/Ask';
import { LabsPage, Reveal, benchRow, defineLabs, measure, preferredBench, useLabs, type BenchSet } from '../../components/LabKit';
import { Beyond, Bench, Source } from '../../components/notes';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// Prerequisite questions drawn from the notes of chapters 4 to 6. Each explanation names
// where the idea was taught and which section of this chapter builds on it.
export const questionsS150: Record<string, Question> = {
  'ch15/start/q-tile-ratio': {
    kind: 'numeric',
    id: 'ch15/start/q-tile-ratio',
    prompt: (
      <Translated
        vi={
          <>
            Kernel matmul ngây thơ thực hiện 0.25 FLOP cho mỗi byte nó yêu cầu từ bộ nhớ toàn cục, và phân ô với các ô T × T chia
            lưu lượng đó cho T. Kernel của lab 5.3 dùng ô 16 × 16. Trong mô hình đó, nó đạt tỉ số bao nhiêu?
          </>
        }
      >
        <>
          The naive matmul kernel does 0.25 FLOP per byte it asks global memory for, and tiling with T × T tiles divides
          that traffic by T. Lab 5.3&rsquo;s kernel uses 16 × 16 tiles. What ratio does it reach in that model?
        </>
      </Translated>
    ),
    answer: 4,
    tol: 0.01,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <p>
            0.25 × 16 = 4 FLOP/B. Ô 32 × 32 của sách cho 8, vẫn dưới mức 20 của thiết bị ví dụ trong sách, và chương 5 để việc đi
            xa hơn lại cho chương này. Mục <a href={href('ch15', '15.2')}>15.2</a> đếm lại cho ô có hình dạng bất kỳ và thấy rằng
            chỉ ô đầu ra là quan trọng: 128 × 128 cho 32. Đã học ở{' '}
            <a href={href('ch05', '5.4')}>ch. 5 §5.4</a>.
          </p>
        }
      >
        <p>
          0.25 × 16 = 4 FLOP/B. The book&rsquo;s 32 × 32 tiles give 8, still under its example device&rsquo;s 20, and
          chapter 5 leaves going further to this chapter. Section <a href={href('ch15', '15.2')}>15.2</a> redoes the count
          for tiles of any shape and finds that only the output tile matters: 128 × 128 gives 32. Taught in{' '}
          <a href={href('ch05', '5.4')}>ch. 5 §5.4</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.1, p. 97, and §5.4, p. 111',
    vi: { source: 'ch. 5 §5.1, p. 97, và §5.4, p. 111' },
  },
  'ch15/start/q-local-array': {
    kind: 'choice',
    id: 'ch15/start/q-local-array',
    prompt: (
      <Translated
        vi={
          <>
            Một luồng giữ tám tổng riêng phần trong <code>float acc[8]</code> và cập nhật chúng trong{' '}
            <code>{'for (int j = 0; j < 8; ++j) acc[j] += ...;'}</code>, một vòng lặp mà trình biên dịch không trải ra (unroll).{' '}
            <code>acc</code> nằm ở đâu?
          </>
        }
      >
        <>
          A thread keeps eight partial sums in <code>float acc[8]</code> and updates them in{' '}
          <code>{'for (int j = 0; j < 8; ++j) acc[j] += ...;'}</code>, a loop the compiler does not unroll. Where does{' '}
          <code>acc</code> live?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Trong thanh ghi (register), như mọi biến tự động.">In registers, like any automatic variable.</Translated>,
        why: (
          <Translated vi="Thanh ghi không thể được đánh chỉ số bằng một giá trị tính lúc chạy. acc[j] với j là biến vòng lặp cần một địa chỉ.">
            Registers cannot be indexed by a value computed at run time. acc[j] with j a loop variable needs an address.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Trong bộ nhớ cục bộ (local memory): riêng của luồng, nhưng về vật lý nằm trong bộ nhớ thiết bị.">
            In local memory: private to the thread, but physically in device memory.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Trong bộ nhớ chia sẻ (shared memory), vì nó là một mảng.">In shared memory, because it is an array.</Translated>,
        why: (
          <Translated vi="Bộ nhớ chia sẻ là thứ bạn khai báo __shared__, và cả khối đều thấy nó. acc là riêng của một luồng.">
            Shared memory is what you declare __shared__, and it is visible to the whole block. acc is private to one thread.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Trong bộ nhớ hằng (constant memory), vì kích thước của nó là hằng số.">In constant memory, because its size is a constant.</Translated>,
        why: (
          <Translated vi="Bộ nhớ hằng là chỉ đọc đối với device và được khai báo __constant__. Kích thước hằng không đưa một mảng vào đó.">
            Constant memory is read-only for the device and declared __constant__. A constant size does not put an array there.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một mảng tự động sẽ nằm trong bộ nhớ cục bộ trừ khi mọi lần truy cập đều dùng chỉ số hằng. Hãy trải hết vòng lặp (số
            lần lặp của nó là hằng) và <code>acc[j]</code> trở thành <code>acc[0]</code> đến <code>acc[7]</code>, những thứ trình
            biên dịch có thể giữ trong thanh ghi. Kernel của chương này giữ 64 biến tích lũy mỗi luồng theo cách đó, và mọi vòng
            lặp chạm vào chúng đều mang <code>#pragma unroll</code>: mục{' '}
            <a href={href('ch15', '15.3')}>15.3</a>. Đã học ở <a href={href('ch05', '5.2')}>ch. 5 §5.2</a> và{' '}
            <a href={href('ch06', '6.6')}>ch. 6 §6.6</a>.
          </p>
        }
      >
        <p>
          An automatic array goes to local memory unless every access uses a constant index. Fully unroll the loop (its
          trip count is a constant) and <code>acc[j]</code> becomes <code>acc[0]</code> to <code>acc[7]</code>, which the
          compiler can keep in registers. This chapter&rsquo;s kernel keeps 64 accumulators per thread that way, and every
          loop that touches them carries <code>#pragma unroll</code>: section{' '}
          <a href={href('ch15', '15.3')}>15.3</a>. Taught in <a href={href('ch05', '5.2')}>ch. 5 §5.2</a> and{' '}
          <a href={href('ch06', '6.6')}>ch. 6 §6.6</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.2, pp. 98 and 103; ch. 6 §6.6, pp. 145–146',
    vi: { source: 'ch. 5 §5.2, pp. 98 và 103; ch. 6 §6.6, pp. 145–146' },
  },
  'ch15/start/q-bank-stride': {
    kind: 'numeric',
    id: 'ch15/start/q-bank-stride',
    prompt: (
      <Translated
        vi={
          <>
            Một khối khai báo <code>__shared__ float t[32][16];</code> và mỗi lane trong 32 lane của một warp đọc{' '}
            <code>t[lane][0]</code>. Với 32 bank rộng 4 byte, bank bận nhất phải phục vụ bao nhiêu lane?
          </>
        }
      >
        <>
          A block declares <code>__shared__ float t[32][16];</code> and the 32 lanes of a warp each read{' '}
          <code>t[lane][0]</code>. With 32 banks of 4 bytes, how many lanes land in the busiest bank?
        </>
      </Translated>
    ),
    answer: 16,
    unit: 'lanes',
    vi: { unit: 'lane', source: 'ch. 6 §6.4, pp. 139–141, và bài tập 5, p. 155' },
    explain: (
      <Translated
        vi={
          <p>
            Lane l đọc word 16·l, và 16·l mod 32 bằng 0 với các lane chẵn và 16 với các lane lẻ: hai bank, mỗi bank mười sáu lane,
            một xung đột 16 chiều. Một hàng 17 float sẽ đặt lane l vào bank 17·l mod 32, mỗi lane một bank khác nhau. Mục{' '}
            <a href={href('ch15', '15.6')}>15.6</a> gặp lại đúng vấn đề này ở ô A, rộng 8 float và được đọc bởi các nhóm lane cách
            nhau 4 hàng. Đã học ở <a href={href('ch06', '6.4')}>ch. 6 §6.4</a>.
          </p>
        }
      >
        <p>
          Lane l reads word 16·l, and 16·l mod 32 is 0 for even lanes and 16 for odd ones: two banks, sixteen lanes on
          each, a 16-way conflict. A row of 17 floats would put lane l in bank 17·l mod 32, a different bank for every
          lane. Section <a href={href('ch15', '15.6')}>15.6</a> meets the same problem in the A tile, which is 8 floats
          wide and read by groups of lanes 4 rows apart. Taught in <a href={href('ch06', '6.4')}>ch. 6 §6.4</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.4, pp. 139–141, and exercise 5, p. 155',
  },
  'ch15/start/q-registers': {
    kind: 'numeric',
    id: 'ch15/start/q-registers',
    prompt: (
      <Translated
        vi={
          <>
            Trên SM ví dụ của sách, với 65,536 thanh ghi và chỗ cho 2048 luồng, một kernel cần 128 thanh ghi mỗi luồng. Tối đa
            bao nhiêu phần trăm số chỗ luồng của SM có thể được lấp đầy?
          </>
        }
      >
        <>
          On the book&rsquo;s example SM, with 65,536 registers and room for 2048 threads, a kernel needs 128 registers per
          thread. At most what percentage of the SM&rsquo;s thread slots can be filled?
        </>
      </Translated>
    ),
    answer: 25,
    tol: 0.01,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            65,536 / 128 = 512 luồng, và 512 / 2048 = 25%. Ví dụ của chính sách là 64 thanh ghi cho 50%. Kernel của chương này đi
            xa hơn: gần mức tối đa 255 thanh ghi, mỗi SM một khối 256 luồng, 12.5%, và mục{' '}
            <a href={href('ch15', '15.7')}>15.7</a> lập luận rằng đây là sự đánh đổi đúng. Đã học ở{' '}
            <a href={href('ch04', '4.7')}>ch. 4 §4.7</a>.
          </p>
        }
      >
        <p>
          65,536 / 128 = 512 threads, and 512 / 2048 = 25%. The book&rsquo;s own example is 64 registers for 50%. This
          chapter&rsquo;s kernel goes further: close to the 255-register maximum, one block of 256 threads per SM, 12.5%,
          and section <a href={href('ch15', '15.7')}>15.7</a> argues that this is the right trade. Taught in{' '}
          <a href={href('ch04', '4.7')}>ch. 4 §4.7</a>.
        </p>
      </Translated>
    ),
    source: "ch. 4 §4.7, p. 86 (the device figures are the book's Hopper H100 example)",
    vi: { source: 'ch. 4 §4.7, p. 86 (số liệu thiết bị là ví dụ Hopper H100 của sách)' },
  },
};

// ---------------------------------------------------------------- your starting point
// Lab 15.1's benchmark times lab 5.3's reference kernel (row "tiled_5_3") and the vendor's
// GEMM (row "vendor") beside the stage itself, at the same shapes (labs/ch15/shared/stage_driver.hpp).
const [L151] = defineLabs('ch15', [{ num: '15.1', dir: 'lab1_coarse', name: 'coarsened', vi: { name: 'gộp việc cho luồng' } }]);
const SIDE = 2048;
const CUBE = { m: SIDE, k: SIDE, n: SIDE };
const WHOSE = () =>
  tr(
    "lab 15.1's benchmark (lab 5.3's reference kernel, T = 16, and the vendor library)",
    'benchmark của lab 15.1 (kernel tham chiếu của lab 5.3, T = 16, và thư viện của nhà cung cấp)',
  );
const gflops = (r?: BenchRow) => num(r?.gflops);

/** Lab 5.3's speed as a percentage of the library's, at 2048 cubed. */
export const gapAt2048 = (reports: Reports) =>
  measure(reports, L151, (s: BenchSet) => {
    const mine = gflops(benchRow(s, 'tiled_5_3', CUBE));
    const lib = gflops(benchRow(s, 'vendor', CUBE));
    if (!mine || !lib) return null;
    return { value: (100 * mine) / lib, detail: tr(`${fmt(mine)} against ${fmt(lib)} GFLOP/s`, `${fmt(mine)} so với ${fmt(lib)} GFLOP/s`), whose: WHOSE() };
  });

/** Every shape lab 15.1 timed: lab 5.3's kernel against the library, one table per backend.
 *  Shown once a prediction is committed. */
function GapTable() {
  const { reports } = useLabs();
  const sets = preferredBench(reports, L151);
  const tables = sets
    .map((s) => ({
      s,
      rows: s.rows
        .filter((r) => r.name === 'tiled_5_3')
        .map((r) => ({ r, mine: gflops(r), lib: gflops(benchRow(s, 'vendor', { m: num(r.m) ?? -1, k: num(r.k) ?? -1, n: num(r.n) ?? -1 })) })),
    }))
    .filter((t) => t.rows.length);
  if (!tables.length) return null;
  return (
    <div className="panel" style={{ overflowX: 'auto', marginTop: '0.6rem' }}>
      {tables.map(({ s, rows }) => (
        <table className="data" style={{ margin: 0 }} data-gap-table={s.backend} key={s.backend}>
          <thead>
            <tr>
              <th>
                {s.backend} <span className="muted">({s.device})</span>, M × K × N
              </th>
              <th className="num">{tr('Lab 5.3, GFLOP/s', 'Lab 5.3, GFLOP/s')}</th>
              <th className="num">{tr('Library, GFLOP/s', 'Thư viện, GFLOP/s')}</th>
              <th className="num">{tr('Lab 5.3 as % of the library', 'Lab 5.3 tính theo % của thư viện')}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ r, mine, lib }, i) => (
              <tr key={i}>
                <td style={{ whiteSpace: 'nowrap' }}>
                  {String(r.m)} × {String(r.k)} × {String(r.n)}
                </td>
                <td className="num">{mine ? fmt(mine) : ''}</td>
                <td className="num">{lib ? fmt(lib) : ''}</td>
                <td className="num">{mine && lib ? fmt((100 * mine) / lib) : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ))}
    </div>
  );
}

function StartingPoint() {
  return (
    <LabsPage labs={[L151]}>
      <Bench
        title={tr('Your starting point', 'Điểm xuất phát của bạn')}
        task={tr("Commit a number, then see this machine's measurement.", 'Chốt một con số, rồi xem số đo của máy này.')}
        foot={
          <Translated
            vi={
              <>
                Đọc từ các lần chạy lab đã ghi lại trên máy này (<code>reports/runs/</code>). Không con số nào ở đây là giá trị kỳ vọng.
              </>
            }
          >
            <>Read from this machine&rsquo;s recorded lab runs (<code>reports/runs/</code>). Nothing here is an expected value.</>
          </Translated>
        }
      >
        <Predict
          id="ch15/start/p-gap"
          prompt={
            <Translated
              vi={
                <>
                  Ở {SIDE} × {SIDE} × {SIDE}, kernel phân ô của chương 5 đạt bao nhiêu phần trăm tốc độ của thư viện nhà cung cấp trên
                  thiết bị của bạn?
                </>
              }
            >
              <>
                At {SIDE} × {SIDE} × {SIDE}, what percentage of the vendor library&rsquo;s speed does chapter 5&rsquo;s tiled
                kernel reach on your device?
              </>
            </Translated>
          }
          unit={tr('% of the library', '% của thư viện')}
        >
          {(v) => (
            <>
              <Reveal
                committed={v}
                measure={gapAt2048}
                unit={tr('% of the library', '% của thư viện')}
                reading={
                  <Translated
                    vi={
                      <>
                        100% là tốc độ của thư viện. Cả hai bên đều tính bằng FP32 (mục{' '}
                        <a href={href('ch15', '15.1')}>15.1</a>), nên khoảng cách nằm ở kỹ thuật, không phải độ chính xác. Bảng có mọi hình
                        dạng mà benchmark đo; hãy xem lớp với batch 1 trước khi bạn đọc mục{' '}
                        <a href={href('ch15', '15.2')}>15.2</a>.
                      </>
                    }
                  >
                    <>
                      100% would be the library&rsquo;s speed. Both sides compute in FP32 (section{' '}
                      <a href={href('ch15', '15.1')}>15.1</a>), so the gap is technique, not precision. The table has every
                      shape the benchmark times; look at the batch-1 layer before you read section{' '}
                      <a href={href('ch15', '15.2')}>15.2</a>.
                    </>
                  </Translated>
                }
                missing={
                  <Translated
                    vi={
                      <>
                        Chạy <code>python3 labs/pmpp.py run 15.1 --solution --bench</code> trên một máy có GPU. Lệnh này đo kernel tham chiếu
                        của lab 5.3 và thư viện, dù bạn đã viết lab 15.1 hay chưa.
                      </>
                    }
                  >
                    <>
                      Run <code>python3 labs/pmpp.py run 15.1 --solution --bench</code> on a machine with a GPU. It times
                      lab 5.3&rsquo;s reference kernel and the library whether or not you have written lab 15.1 yet.
                    </>
                  </Translated>
                }
              />
              <GapTable />
            </>
          )}
        </Predict>
      </Bench>
    </LabsPage>
  );
}

// ---------------------------------------------------------------- the chapter at a glance
// Functions, not constants: the labels are read in the selected language when the table renders.
const MAP: () => { part: string; rows: [string, string, string][] }[] = () => [
  {
    part: tr('The model', 'Mô hình'),
    rows: [
      ['15.1', tr('Background', 'Bối cảnh'), tr('Map a linear layer or an attention product onto GEMM and onto a library call', 'Ánh xạ một lớp tuyến tính hay một tích attention vào GEMM và vào một lời gọi thư viện')],
      ['15.2', tr('Data reuse analysis', 'Phân tích tái sử dụng dữ liệu'), tr('Compute the intensity of any output tile; see why only its size matters', 'Tính cường độ tính toán của bất kỳ ô đầu ra nào; thấy vì sao chỉ kích thước của nó là quan trọng')],
    ],
  },
  {
    part: tr('The kernel, one bottleneck at a time', 'Kernel, từng nút thắt một'),
    rows: [
      ['15.3', tr('Larger tiles with thread coarsening', 'Ô lớn hơn với gộp việc cho luồng'), tr('A 128 × 128 tile per block, an 8 × 8 patch per thread, in registers', 'Mỗi khối một ô 128 × 128, mỗi luồng một mảng 8 × 8, nằm trong thanh ghi')],
      ['15.4', tr('Register tiling of the input tiles', 'Phân ô thanh ghi cho các ô đầu vào'), tr('Read each shared value once per thread, by swapping two loops', 'Mỗi luồng đọc mỗi giá trị chia sẻ một lần, bằng cách đổi chỗ hai vòng lặp')],
      ['15.5', tr('Coalesced storing of the output tile', 'Ghi ô đầu ra được hợp nhất'), tr('Re-map threads so that a warp’s stores land side by side', 'Ánh xạ lại các luồng để các lần ghi của một warp nằm sát cạnh nhau')],
      ['15.6', tr('Eliminating bank conflicts', 'Loại bỏ xung đột bank'), tr('One padding column, and the line of code the book leaves out', 'Một cột đệm, và dòng mã mà sách bỏ sót')],
    ],
  },
  {
    part: tr('Paying for it', 'Cái giá phải trả'),
    rows: [
      ['15.7', tr('Occupancy considerations', 'Cân nhắc về occupancy'), tr('Why one block per SM is the right trade here', 'Vì sao mỗi SM một khối là sự đánh đổi đúng ở đây')],
      ['15.8', tr('Software pipelining', 'Pipeline phần mềm'), tr('Load the next tiles while computing on these, with one barrier', 'Nạp các ô kế tiếp trong khi tính trên các ô hiện tại, với một rào chắn')],
    ],
  },
  {
    part: tr('Beyond your kernel', 'Vượt ra ngoài kernel của bạn'),
    rows: [['15.9', tr('Specialized software and hardware', 'Phần mềm và phần cứng chuyên biệt'), tr('Libraries, tensor cores, asynchronous copies, and what each backend has', 'Thư viện, tensor core, sao chép bất đồng bộ, và mỗi backend có những gì')]],
  },
];
const LADDER: () => [string, string, string][] = () => [
  ['15.1', tr('The 128 × 128 coarsened kernel', 'Kernel gộp việc 128 × 128'), '15.3'],
  ['15.2', tr('Register tiling: the loops interchanged', 'Phân ô thanh ghi: hoán đổi các vòng lặp'), '15.4'],
  ['15.3', tr('Coalesced stores: four 4 × 4 sub-tiles and float4 stores', 'Ghi được hợp nhất: bốn ô con 4 × 4 và lệnh ghi float4'), '15.5'],
  ['15.4', tr('Padding and vector loads', 'Phần đệm và lệnh nạp vector'), '15.6, 15.7'],
  ['15.5', tr('Double buffering and software pipelining', 'Đệm kép và pipeline phần mềm'), '15.8'],
  ['15.6', tr('The fast linear layer (milestone M3)', 'Lớp tuyến tính nhanh (mốc dự án M3)'), tr('15.3 to 15.8', '15.3 đến 15.8')],
  ['15.7', tr('Bug hunt: three defects in a fast linear layer', 'Săn lỗi: ba lỗi trong một lớp tuyến tính nhanh'), '15.6, 15.8'],
];

export function S150() {
  const q = questionsS150;
  return (
    <>
      <Translated
        vi={
          <p>
            Phép nhân ma trận làm rất nhiều phép tính trên một lượng dữ liệu tương đối ít, nên mỗi phần tử được dùng nhiều lần, và
            có nhiều thứ để thu được từ việc quyết định ai nạp cái gì. Chương 3 dùng nó để ánh xạ luồng lên một lưới (grid), chương
            5 để giới thiệu phân ô (tiling). Chương này đi xa hơn cả hai, với những tối ưu hóa đặc thù cho GPU.
          </p>
        }
      >
        <p>
          Matrix multiplication does a great deal of arithmetic on comparatively little data, so every element is used many
          times, and there is a lot to gain from deciding who loads what. Chapter 3 used it to map threads onto a grid,
          chapter 5 to introduce tiling. This chapter goes further than either, with optimisations that are specific to
          GPUs.
        </p>
      </Translated>
      <Source pages="349" />
      <Translated
        vi={
          <p>
            Mục 5.4 khép lại bằng một nhận xét: trên GPU ví dụ của sách, ô 32 × 32 vẫn khiến kernel thiếu dữ liệu, chỉ ở mức 40%
            khả năng tính toán số học của thiết bị. Việc đi xa hơn được để lại cho chương này và cho các thư viện như cuBLAS và
            CUTLASS.
          </p>
        }
      >
        <p>
          Section 5.4 closed with a remark: on the book&rsquo;s example GPU, 32 × 32 tiles still leave the kernel short of
          data, at 40% of the arithmetic the device can do. Going further was left to this chapter and to libraries such
          as cuBLAS and CUTLASS.
        </p>
      </Translated>
      <Source section="5.4" pages="111" />
      <Translated
        vi={
          <p>
            Vì vậy chương này tháo rời kernel phân ô và dựng lại nó, từng nút thắt một, thành bộ khung của một GEMM trong thư
            viện: mỗi khối một ô đầu ra 128 × 128, mỗi luồng giữ một mảng 8 × 8 của ô đó trong thanh ghi của mình, các dải đầu vào
            cũng nằm trong thanh ghi, các lần ghi được sắp xếp để những gì một warp ghi nằm sát cạnh nhau, một ô chia sẻ có phần
            đệm, một khối cố ý chạy một mình trên SM của nó, và một vòng lặp nạp các ô kế tiếp trong khi tính trên các ô hiện tại.
            Sách lập luận mọi bước trên giấy và không đưa ra số đo thời gian nào. Các lab cung cấp chúng: bạn xây kernel từng
            bước, đo mỗi bước so với kernel của chương 5 và thư viện của nhà cung cấp trên chính GPU của bạn, và kết thúc bằng
            lớp tuyến tính của dự án, mốc dự án M3.
          </p>
        }
      >
        <p>
          So the chapter takes the tiled kernel apart and rebuilds it, one bottleneck at a time, into the skeleton of a
          library GEMM: a 128 × 128 output tile per block, an 8 × 8 patch of it in each thread&rsquo;s registers, input
          strips in registers too, stores arranged so that a warp&rsquo;s writes land side by side, a padded shared tile, a
          block that runs alone on its SM on purpose, and a loop that loads the next tiles while it computes on the
          current ones. The book argues every step on paper and gives no timings. The labs supply them: you build the
          kernel stage by stage, time each stage against chapter 5&rsquo;s kernel and the vendor&rsquo;s library on your
          own GPU, and finish with the project&rsquo;s linear layer, milestone M3.
        </p>
      </Translated>

      <h2>
        <Translated vi="Bên dưới framework của bạn">Underneath your framework</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Khi PyTorch nhân hai tensor float trên GPU NVIDIA, nó gọi cuBLAS hoặc cuBLASLt, còn trên ROCm nó có thể gọi Composable
            Kernel. Những thư viện đó được xây từ các ý tưởng của chương này cộng với phần cứng mà chương chỉ điểm qua. llama.cpp
            cho thấy các ý tưởng ngừng có lợi ở đâu: backend CUDA của nó chuyển sang kernel ma trận-vector khi một ma trận trọng
            số phục vụ 8 token trở xuống, đúng là điều xảy ra khi sinh từng token một, và dùng kernel ma trận phân ô khi nhiều hơn.
            Mục <a href={href('ch15', '15.2')}>15.2</a> đưa ra lý do chỉ bằng một dòng tính toán.
          </p>
        }
      >
        <p>
          When PyTorch multiplies two float tensors on an NVIDIA GPU it calls cuBLAS or cuBLASLt, and on ROCm it can call
          Composable Kernel. Those libraries are built from this chapter&rsquo;s ideas plus hardware the chapter only
          surveys. llama.cpp shows where the ideas stop paying: its CUDA backend switches to matrix-vector kernels when a
          weight matrix serves 8 tokens or fewer, which is what generating one token at a time does, and uses tiled matrix
          kernels above that. Section <a href={href('ch15', '15.2')}>15.2</a> gives the reason in one line of arithmetic.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đọc từ <code>CUDABlas.cpp</code> của PyTorch và <code>mmvq.cuh</code>, <code>mmvf.cuh</code> của llama.cpp ngày
              2026-11-20 (<code>notes/today/ch15-facts.md</code> D1, D4; ch06-facts D1).
            </p>
          }
        >
          <p>
            Read from PyTorch&rsquo;s <code>CUDABlas.cpp</code> and llama.cpp&rsquo;s <code>mmvq.cuh</code> and{' '}
            <code>mmvf.cuh</code> on 2026-11-20 (<code>notes/today/ch15-facts.md</code> D1, D4; ch06-facts D1).
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Điểm xuất phát của bạn">Your starting point</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Benchmark của lab 15.1 đo ba kernel cạnh nhau: lời giải tham chiếu của lab 5.3 (kernel phân ô của chương 5 với T =
            16), bước bạn đang xây, và GEMM độ chính xác đơn của nhà cung cấp (cuBLAS, hipBLAS hoặc Metal Performance Shaders).
            Trước khi xem cái đầu và cái cuối, hãy chốt một con số.
          </p>
        }
      >
        <p>
          Lab 15.1&rsquo;s benchmark times three kernels side by side: lab 5.3&rsquo;s reference solution (chapter 5&rsquo;s
          tiled kernel with T = 16), the stage you are building, and the vendor&rsquo;s single-precision GEMM (cuBLAS,
          hipBLAS or Metal Performance Shaders). Before you look at the first and the last, commit a number.
        </p>
      </Translated>
      <StartingPoint />

      <h2>
        <Translated vi="Toàn cảnh chương">The chapter at a glance</Translated>
      </h2>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-map>
          <thead>
            <tr>
              <th>
                <Translated vi="Về">About</Translated>
              </th>
              <th>
                <Translated vi="Mục">Section</Translated>
              </th>
              <th>
                <Translated vi="Bạn sẽ làm được gì">What you will be able to do</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {MAP().flatMap((g) =>
              g.rows.map(([id, title, what], i) => (
                <tr key={id}>
                  <td>{i === 0 ? g.part : ''}</td>
                  <td>
                    <a href={href('ch15', id)}>
                      {id} {title}
                    </a>
                  </td>
                  <td>{what}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Bảy lab, trên <a href={href('ch15', 'labs')}>trang lab</a>. Năm lab đầu là một chiếc thang: mỗi lab bắt đầu từ lời
            giải của lab trước với một phần bị gỡ đi, và mỗi benchmark đặt bước của bạn cạnh bước trước nó.
          </p>
        }
      >
        <p>
          Seven labs, on the <a href={href('ch15', 'labs')}>labs page</a>. The first five are one ladder: each starts from
          the previous lab&rsquo;s solution with one part removed, and each benchmark puts your stage beside the one
          before it.
        </p>
      </Translated>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data" data-labs>
          <thead>
            <tr>
              <th>
                <Translated vi="Lab">Lab</Translated>
              </th>
              <th>
                <Translated vi="Bạn xây gì">What you build</Translated>
              </th>
              <th>
                <Translated vi="Các mục">Sections</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {LADDER().map(([n, t, s]) => (
              <tr key={n} data-lab={n}>
                <td>{n}</td>
                <td>{t}</td>
                <td>{s}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bốn câu hỏi về những gì chương này giả định từ chương 4 đến 6. Mỗi lời giải thích chỉ ra nơi ý tưởng đã được dạy và mục
            ở đây xây tiếp trên nó. Chúng sẽ quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Four questions on what this chapter assumes from chapters 4 to 6. Each explanation points to where the idea was
          taught and to the section here that builds on it. They come back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={q['ch15/start/q-tile-ratio']} label={tr('From chapter 5', 'Từ chương 5')} />
      <Ask q={q['ch15/start/q-local-array']} label={tr('From chapters 5 and 6', 'Từ chương 5 và 6')} />
      <Ask q={q['ch15/start/q-bank-stride']} label={tr('From chapter 6', 'Từ chương 6')} />
      <Ask q={q['ch15/start/q-registers']} label={tr('From chapter 4', 'Từ chương 4')} />
    </>
  );
}
