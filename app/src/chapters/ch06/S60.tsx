import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Source } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// Prerequisite questions drawn from the notes of chapters 3 to 5. Each explanation
// names where the idea was taught and which section of this chapter builds on it.
export const questionsS60: Record<string, Question> = {
  'ch06/start/q-neighbours': {
    kind: 'numeric',
    id: 'ch06/start/q-neighbours',
    prompt: (
      <Translated
        vi={
          <>
            Trong vòng lặp matmul ngây thơ <code>Pvalue += M[row*Width + k] * N[k*Width + col];</code> các luồng (thread){' '}
            <code>(tx, ty)</code> và <code>(tx + 1, ty)</code> của một khối thực thi cùng lần lặp <code>k</code>{' '}
            với nhau. Hai địa chỉ chúng đọc từ <code>N</code> cách nhau bao nhiêu phần tử của mảng?
          </>
        }
      >
        In the naive matmul loop <code>Pvalue += M[row*Width + k] * N[k*Width + col];</code> the threads{' '}
        <code>(tx, ty)</code> and <code>(tx + 1, ty)</code> of one block execute the same iteration <code>k</code>{' '}
        together. How many array elements apart are the two addresses they read from <code>N</code>?
      </Translated>
    ),
    answer: 1,
    unit: 'elements',
    explain: (
      <Translated
        vi={
          <p>
            <code>col</code> khác nhau một đơn vị và mọi thứ khác trong <code>k*Width + col</code> đều như nhau, nên hai
            lần đọc nằm kề nhau. Các lần đọc <code>M</code> của chúng giống hệt nhau: cùng <code>row</code>, cùng{' '}
            <code>k</code>. Vậy trên một hàng luồng, một toán hạng là một dải địa chỉ liên tiếp còn toán hạng kia là một giá
            trị chung duy nhất. Được dạy ở <a href={href('ch03', '3.4')}>ch. 3 §3.4</a>; mục{' '}
            <a href={href('ch06', '6.1')}>6.1</a> biến nó thành một chi phí bạn đếm được.
          </p>
        }
      >
        <p>
          <code>col</code> differs by one and everything else in <code>k*Width + col</code> is the same, so the two reads
          are neighbours. Their <code>M</code> reads are identical: same <code>row</code>, same <code>k</code>. So across a
          row of threads one operand is a run of consecutive addresses and the other is a single shared value. Taught
          in <a href={href('ch03', '3.4')}>ch. 3 §3.4</a>; section <a href={href('ch06', '6.1')}>6.1</a> turns it into
          a cost you can count.
        </p>
      </Translated>
    ),
    source: 'ch. 3 §3.4, pp. 61–62',
    vi: { unit: 'phần tử' },
  },
  'ch06/start/q-warp2d': {
    kind: 'choice',
    id: 'ch06/start/q-warp2d',
    prompt: (
      <Translated vi={<>Một kernel chạy các khối 16 × 16 trên phần cứng có warp rộng 32. Những luồng nào của một khối tạo thành warp 0?</>}>
        <>A kernel runs 16 × 16 blocks on hardware with 32-wide warps. Which threads of a block form warp 0?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="threadIdx.x từ 0 đến 15 ở các hàng threadIdx.y = 0 và 1.">threadIdx.x 0 to 15 in rows threadIdx.y = 0 and 1.</Translated>,
        correct: true,
      },
      {
        label: (
          <Translated vi="threadIdx.x từ 0 đến 31 của hàng 0: một warp là 32 giá trị x liên tiếp.">
            threadIdx.x 0 to 31 of row 0: a warp is 32 consecutive x values.
          </Translated>
        ),
        why: (
          <Translated vi="Khối chỉ rộng 16; không có threadIdx.x nào lớn hơn 15. Liên tiếp theo thứ tự tuyến tính hóa ở đây nghĩa là hai hàng.">
            The block is only 16 wide; there is no threadIdx.x above 15. Consecutive in the linearised order means two rows here.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="threadIdx.y từ 0 đến 15 ở các cột threadIdx.x = 0 và 1.">
            threadIdx.y 0 to 15 in columns threadIdx.x = 0 and 1.
          </Translated>
        ),
        why: (
          <Translated vi="Cách đó tuyến tính hóa y chạy nhanh nhất. Thứ tự là x chạy nhanh nhất, nên một warp lấy trọn các hàng, không phải các cột.">
            That linearises y fastest. The order is x fastest, so a warp takes whole rows, not columns.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phần cứng chọn chúng lúc chạy, nên không kernel nào dựa vào điều đó được.">
            The hardware picks them at run time, so no kernel can rely on it.
          </Translated>
        ),
        why: (
          <Translated vi="Cách chia là cố định: tuyến tính hóa theo thứ tự hàng (x chạy nhanh nhất), rồi cắt thành từng nhóm 32.">
            The partition is fixed: linearise row-major (x fastest), then cut into 32s.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khối được tuyến tính hóa với x chạy nhanh nhất, rồi cắt thành các nhóm 32, nên một warp là hai hàng rộng 16.
            Được dạy ở <a href={href('ch04', '4.4')}>ch. 4 §4.4</a>. Điều này quan trọng hai lần trong chương này: một hàng
            ô rộng 16 là nửa warp, nên phép nạp của một warp chạm vào hai dải địa chỉ riêng biệt (
            <a href={href('ch06', '6.1')}>6.1</a>), và các truy cập bộ nhớ chia sẻ của nó trải qua các bank khác với một
            hàng rộng 32 (<a href={href('ch06', '6.4')}>6.4</a>). Con số 32 là độ rộng của sách; GPU của AMD và Apple có
            thể khác.
          </p>
        }
      >
        <p>
          The block is linearised x fastest, then cut into groups of 32, so a warp is two 16-wide rows. Taught in{' '}
          <a href={href('ch04', '4.4')}>ch. 4 §4.4</a>. It matters twice in this chapter: a 16-wide tile row is half a
          warp, so one warp&rsquo;s load touches two separate runs of addresses (
          <a href={href('ch06', '6.1')}>6.1</a>), and its shared-memory accesses spread over banks differently from a
          32-wide row&rsquo;s (<a href={href('ch06', '6.4')}>6.4</a>). The 32 is the book&rsquo;s width; AMD and Apple
          GPUs can differ.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.4, pp. 75–76',
  },
  'ch06/start/q-hide': {
    kind: 'numeric',
    id: 'ch06/start/q-hide',
    prompt: (
      <Translated
        vi={
          <>
            Số liệu minh họa. Một SM có một warp cư trú. Nó phát 4 lệnh, rồi chờ 396 chu kỳ cho một lần nạp, rồi lặp lại.
            Đơn vị thực thi phát lệnh trong bao nhiêu phần trăm số chu kỳ?
          </>
        }
      >
        <>
          Toy numbers. An SM has one resident warp. It issues 4 instructions, then waits 396 cycles for a load, then
          repeats. For what percentage of cycles does the execution unit issue an instruction?
        </>
      </Translated>
    ),
    answer: 1,
    tol: 0.01,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            4 chu kỳ bận trong mỗi 400: 1%. Với nhiều warp cư trú hơn, bộ lập lịch lấp khoảng chờ bằng lệnh của các warp
            khác, đó là che giấu độ trễ (<a href={href('ch04', '4.6')}>ch. 4 §4.6</a>). Mục{' '}
            <a href={href('ch06', '6.2')}>6.2</a> thêm một lý do thứ hai để muốn có những warp đó: mỗi warp giữ các yêu
            cầu bộ nhớ của riêng nó đang bay, và hệ thống bộ nhớ cần nhiều yêu cầu đang bay mới đạt được băng thông của nó.
          </p>
        }
      >
        <p>
          4 busy cycles in every 400: 1%. With more resident warps the scheduler fills the wait with other warps&rsquo;
          instructions, which is latency hiding (<a href={href('ch04', '4.6')}>ch. 4 §4.6</a>). Section{' '}
          <a href={href('ch06', '6.2')}>6.2</a> adds a second reason to want those warps: each one keeps its own memory
          requests in flight, and the memory system needs many in flight to deliver its bandwidth.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.6, pp. 82–84 (the numbers are ours)',
    vi: { source: 'ch. 4 §4.6, pp. 82–84 (số liệu là của chúng tôi)' },
  },
  'ch06/start/q-false': {
    kind: 'choice',
    id: 'ch06/start/q-false',
    prompt: (
      <Translated
        vi={
          <>
            Kernel matmul phân ô có hai <code>__syncthreads()</code> mỗi pha. Cái nào canh giữ một phụ thuộc <em>giả</em>{' '}
            (ghi-sau-đọc)?
          </>
        }
      >
        <>
          The tiled matmul kernel has two <code>__syncthreads()</code> per phase. Which one guards a <em>false</em>{' '}
          (write-after-read) dependence?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Cái sau vòng lặp tích vô hướng (rào chắn 2).">The one after the dot-product loop (barrier 2).</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Cái sau các phép nạp ô (rào chắn 1).">The one after the tile loads (barrier 1).</Translated>,
        why: (
          <Translated vi="Cái đó là một phụ thuộc thật: tích vô hướng thực sự cần các giá trị vừa được ghi.">
            That one is a true dependence: the dot product really needs the values just written.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Cả hai.">Both.</Translated>,
        why: (
          <Translated vi="Chỉ một trong hai bảo vệ dữ liệu được cần đến; cái kia chỉ bảo vệ một vị trí khỏi bị dùng lại quá sớm.">
            Only one of them protects data that is needed; the other only protects a location from being reused too early.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không cái nào: cả hai đều cần cho tính đúng đắn, nên cả hai đều là phụ thuộc thật.">
            Neither: both are needed for correctness, so both are true dependences.
          </Translated>
        ),
        why: (
          <Translated vi="Như được viết thì cả hai đều cần, nhưng vì những lý do khác nhau, và chỉ một lý do liên quan đến dữ liệu.">
            Both are needed as written, but for different reasons, and only one reason is about the data.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Rào chắn 2 ngăn một luồng nhanh ghi đè một ô nhớ của tile cho pha kế tiếp trong khi một luồng chậm vẫn đang đọc
            nó ở pha này. Không có gì được truyền từ bên đọc sang bên ghi: hai bên chỉ dùng chung một vị trí. Được dạy ở{' '}
            <a href={href('ch05', '5.4')}>ch. 5 §5.4</a>. Mục <a href={href('ch06', '6.7')}>6.7</a> bỏ loại rào chắn đó bằng
            cách ghi vào một vị trí khác.
          </p>
        }
      >
        <p>
          Barrier 2 stops a fast thread from overwriting a tile slot for the next phase while a slow thread is still
          reading it this phase. Nothing is passed from the reader to the writer: the two only share a location. Taught
          in <a href={href('ch05', '5.4')}>ch. 5 §5.4</a>. Section <a href={href('ch06', '6.7')}>6.7</a> removes that
          kind of barrier by writing to a different location.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.4, pp. 110–111',
  },
  'ch06/start/q-vecadd': {
    kind: 'choice',
    id: 'ch06/start/q-vecadd',
    prompt: (
      <Translated
        vi={
          <>
            Phép cộng vector làm một phép toán cho mỗi 12 byte nó di chuyển, và phiên bản của bạn chạy ở 60% băng thông bạn
            đã đo. Loại thay đổi nào vẫn có thể làm nó nhanh hơn?
          </>
        }
      >
        <>
          Vector addition does one operation per 12 bytes it moves, and your version runs at 60% of the bandwidth you
          measured. Which kind of change can still make it faster?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Ít yêu cầu hơn, rộng hơn, đúng dạng hơn cho cùng số byte, với đủ nhiều yêu cầu đang bay.">
            Fewer, wider, well-formed requests for the same bytes, with enough of them in flight.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Phân ô nó qua bộ nhớ chia sẻ, để mỗi phần tử chỉ đi qua bus bộ nhớ một lần.">
            Tiling it through shared memory, so that each element crosses the memory bus only once.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi phần tử vốn đã đi qua một lần: nó được một luồng dùng một lần, nên không có sự tái sử dụng nào để một ô nắm bắt.">
            Each element already crosses once: it is used once by one thread, so there is no reuse for a tile to capture.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Làm nhiều phép tính số học hơn trên mỗi phần tử được nạp, để nâng số phép toán trên mỗi byte.">
            Doing more arithmetic per element loaded, to raise its operations per byte.
          </Translated>
        ),
        why: (
          <Translated vi="Phép tính cần một phép cộng mỗi phần tử. Số học thêm vào là việc thêm, không phải tốc độ.">
            The computation needs one add per element. Extra arithmetic is extra work, not speed.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một ô lớn hơn, để mỗi khối tái sử dụng nhiều hơn các phần tử nó nạp.">
            A larger tile, so that each block reuses more of the elements it loads.
          </Translated>
        ),
        why: (
          <Translated vi="Không có ô nào, và không phần tử nào được dùng hai lần, nên không kích thước ô nào thay đổi được một tỉ lệ đã bị phép tính cố định.">
            There is no tile, and no element is used twice, so no tile size changes a ratio fixed by the computation.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tỉ lệ của nó, 1 phép toán mỗi 12 byte, bị cố định bởi thứ nó tính (
            <a href={href('ch05', '5.1')}>ch. 5 §5.1</a>), nên nó bị giới hạn bởi bộ nhớ trên mọi GPU. Điều không cố định
            là các byte đó được nạp hiệu quả đến đâu, và đó là phần lớn chương này: các yêu cầu của một warp (6.1), đủ nhiều
            yêu cầu đang bay (6.2), những yêu cầu rộng hơn (6.3).
          </p>
        }
      >
        <p>
          Its ratio, 1 operation per 12 bytes, is fixed by what it computes (
          <a href={href('ch05', '5.1')}>ch. 5 §5.1</a>), so it is memory-bound on every GPU. What is not fixed is how
          efficiently those bytes are fetched, and that is most of this chapter: one warp&rsquo;s requests (6.1), enough
          of them in flight (6.2), wider ones (6.3).
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.1, p. 96',
  },
};

// ---------------------------------------------------------------- your measurements so far
const GRAN = 'ch03/lab4_granularity';
const TILED = 'ch05/lab2_tiled';
const MODES = ['element', 'row', 'col'] as const;
const MODE_WORDS: Record<(typeof MODES)[number], string> = {
  get element() {
    return tr('one thread per element', 'một luồng mỗi phần tử');
  },
  get row() {
    return tr('one thread per row', 'một luồng mỗi hàng');
  },
  get col() {
    return tr('one thread per column', 'một luồng mỗi cột');
  },
};

/** GPU backends only: realBackends() leaves out 'emu' but not the race-detection emulator. */
const gpuBackends = (r: Reports) => realBackends(r).filter((b) => b !== 'race');

/** The reader's own rows when they have run their work with --bench, else the reference solution's. */
function rowsFor(reports: Reports, backend: string, lab: string): { rows: BenchRow[]; whose: string } {
  const mine = benchRows(reports[backend], lab, 'work');
  return mine.length
    ? { rows: mine, whose: tr('your kernels', 'các kernel của bạn') }
    : { rows: benchRows(reports[backend], lab, 'solution'), whose: tr('the reference solution', 'lời giải tham chiếu') };
}

function Measurements() {
  const { reports, loading, reload } = useReports();
  const backends = gpuBackends(reports);
  const per = backends.map((b) => {
    const g = rowsFor(reports, b, GRAN);
    const gran = MODES.map((m) => g.rows.find((r) => r.name === 'granularity' && r.mode === m));
    const t = rowsFor(reports, b, TILED);
    const tiled = t.rows.find((r) => r.name === 'tiled' && num(r.tile) === 16);
    return { b, device: String(reports[b].device.name ?? b), g, gran, t, tiled };
  });
  const any = per.some((p) => p.gran.some(Boolean) || p.tiled);
  return (
    <Bench
      title={tr('Your measurements so far', 'Các phép đo của bạn cho đến nay')}
      task={tr(
        'Two numbers from earlier labs that this chapter explains or starts from.',
        'Hai con số từ các lab trước mà chương này giải thích hoặc lấy làm điểm xuất phát.',
      )}
      foot={
        <Translated
          vi={
            <>
              Đọc từ các lần chạy lab đã ghi lại của máy này (<code>reports/runs/</code>). Không có gì ở đây là giá trị kỳ
              vọng.
            </>
          }
        >
          <>Read from this machine&rsquo;s recorded lab runs (<code>reports/runs/</code>). Nothing here is an expected value.</>
        </Translated>
      }
    >
      {loading ? (
        <p data-measured="loading">
          <Translated vi="Đang đọc báo cáo.">Reading reports.</Translated>
        </p>
      ) : !any ? (
        <p data-measured="none">
          <Translated
            vi={
              <>
                Máy này chưa có phép đo nào. Chạy <code>python3 labs/pmpp.py run 3.4 --solution --bench</code> và{' '}
                <code>python3 labs/pmpp.py run 5.2 --solution --bench</code> (hoặc bản làm của chính bạn, không có{' '}
                <code>--solution</code>), rồi tải lại.{' '}
              </>
            }
          >
            <>
              No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 3.4 --solution --bench</code> and{' '}
              <code>python3 labs/pmpp.py run 5.2 --solution --bench</code> (or your own work without{' '}
              <code>--solution</code>), then reload.{' '}
            </>
          </Translated>
          <button className="quiet" onClick={reload}>
            <Translated vi="Tải lại kết quả">Reload results</Translated>
          </button>
        </p>
      ) : (
        per.map((p) => {
          const [el, row, col] = p.gran;
          const sec = (r?: BenchRow) => num(r?.seconds);
          const rowSlower = sec(row) && sec(col) ? sec(row)! > sec(col)! : null;
          return (
            <div key={p.b} data-measured={p.b} style={{ marginBottom: '0.8rem' }}>
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>{p.device}</b> ({p.b})
              </p>
              {el || row || col ? (
                <>
                  <table className="data" style={{ marginBottom: '0.4rem' }}>
                    <thead>
                      <tr>
                        <th>
                          <Translated vi={<>Lab 3.4, {p.g.whose}</>}>
                            <>Lab 3.4, {p.g.whose}</>
                          </Translated>
                        </th>
                        <th className="num">
                          <Translated vi="Thời gian (ms)">Time (ms)</Translated>
                        </th>
                        <th className="num">
                          <Translated vi="Thời gian ÷ một luồng mỗi phần tử">Time ÷ one per element</Translated>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {MODES.map((m, i) => {
                        const r = p.gran[i];
                        return (
                          <tr key={m} data-mode={m}>
                            <td>{MODE_WORDS[m]}</td>
                            <td className="num">{sec(r) ? fmt(sec(r)! * 1000, 3) : '–'}</td>
                            <td className="num">{num(r?.time_vs_element) ? fmt(num(r?.time_vs_element)!, 3) : '–'}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <p style={{ margin: '0 0 0.6rem' }} data-row-vs-col={rowSlower == null ? 'unknown' : rowSlower ? 'row-slower' : 'row-not-slower'}>
                    {rowSlower == null ? (
                      <Translated vi={<>Lab 3.4 chưa có thời gian đo theo hàng hay theo cột ở đây.</>}>
                        <>Lab 3.4 has no row or column timing here yet.</>
                      </Translated>
                    ) : rowSlower ? (
                      <Translated
                        vi={
                          <>
                            Một luồng mỗi hàng chậm hơn một luồng mỗi cột, dù cả hai làm cùng lượng số học. Mục{' '}
                            <a href={href('ch06', '6.1')}>6.1</a> giải thích vì sao: hãy nhìn xem các luồng kề nhau chạm vào
                            những địa chỉ nào trong mỗi kernel.
                          </>
                        }
                      >
                        <>
                          One thread per row was slower than one per column, though both do the same arithmetic. Section{' '}
                          <a href={href('ch06', '6.1')}>6.1</a> explains why: look at which addresses neighbouring threads
                          touch in each kernel.
                        </>
                      </Translated>
                    ) : (
                      <Translated
                        vi={
                          <>
                            Ở đây một luồng mỗi hàng không chậm hơn một luồng mỗi cột. Mục{' '}
                            <a href={href('ch06', '6.1')}>6.1</a> cho biết quy tắc hợp nhất truy cập bộ nhớ dự đoán cái nào
                            chậm hơn, và lab 6.1 đo riêng hiệu ứng đó.
                          </>
                        }
                      >
                        <>
                          Here one thread per row was not slower than one per column. Section{' '}
                          <a href={href('ch06', '6.1')}>6.1</a> says which one the coalescing rule predicts to be slower,
                          and lab 6.1 measures the effect on its own.
                        </>
                      </Translated>
                    )}
                  </p>
                </>
              ) : (
                <p className="status-line" data-gran="none">
                  <Translated
                    vi={
                      <>
                        Lab 3.4: máy này chưa có phép đo nào (<code>python3 labs/pmpp.py run 3.4 --solution --bench</code>).
                      </>
                    }
                  >
                    <>
                      Lab 3.4: no measurement on this machine yet (<code>python3 labs/pmpp.py run 3.4 --solution --bench</code>).
                    </>
                  </Translated>
                </p>
              )}
              {p.tiled ? (
                <p style={{ margin: 0 }} data-tiled={String(num(p.tiled.speedup_vs_naive) ?? '')}>
                  <Translated
                    vi={
                      <>
                        Lab 5.2, {p.t.whose}: kernel phân ô ở độ rộng ô 16 chạy{' '}
                        <b>{num(p.tiled.speedup_vs_naive) ? `nhanh gấp ${fmt(num(p.tiled.speedup_vs_naive)!, 3)} lần` : 'ở một tốc độ chưa được ghi lại so với'}</b>{' '}
                        kernel ngây thơ
                        {num(p.tiled.gflops) ? ` (${fmt(num(p.tiled.gflops)!, 4)} GFLOP/s)` : ''}. Đó là điểm xuất phát của chương này; mục{' '}
                        <a href={href('ch06', '6.9')}>6.9</a> đặt nó cạnh mái của bạn.
                      </>
                    }
                  >
                    <>
                      Lab 5.2, {p.t.whose}: the tiled kernel at tile width 16 ran{' '}
                      <b>{num(p.tiled.speedup_vs_naive) ? `${fmt(num(p.tiled.speedup_vs_naive)!, 3)} times` : 'at an unrecorded speed relative to'}</b>{' '}
                      {num(p.tiled.speedup_vs_naive) ? 'as fast as the naive kernel' : 'the naive kernel'}
                      {num(p.tiled.gflops) ? ` (${fmt(num(p.tiled.gflops)!, 4)} GFLOP/s)` : ''}. That is this chapter&rsquo;s
                      starting point; section <a href={href('ch06', '6.9')}>6.9</a> places it against your roof.
                    </>
                  </Translated>
                </p>
              ) : (
                <p className="status-line" style={{ margin: 0 }} data-tiled="none">
                  <Translated
                    vi={
                      <>
                        Lab 5.2: máy này chưa có phép đo nào (<code>python3 labs/pmpp.py run 5.2 --solution --bench</code>).
                      </>
                    }
                  >
                    <>
                      Lab 5.2: no measurement on this machine yet (<code>python3 labs/pmpp.py run 5.2 --solution --bench</code>).
                    </>
                  </Translated>
                </p>
              )}
            </div>
          );
        })
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the chapter at a glance
// Getters, so the text is read when rendered, in the reader's language.
const MAP: { part: string; rows: [string, string, string][] }[] = [
  {
    get part() {
      return tr('Global memory', 'Bộ nhớ toàn cục');
    },
    get rows(): [string, string, string][] {
      return [
      [
        '6.1',
        tr('Global memory access coalescing', 'Hợp nhất truy cập bộ nhớ toàn cục'),
        tr(
          'How one warp’s loads become few or many memory transactions',
          'Các phép nạp của một warp trở thành ít hay nhiều giao dịch bộ nhớ như thế nào',
        ),
      ],
      [
        '6.2',
        tr('Hiding memory latency', 'Che giấu độ trễ bộ nhớ'),
        tr(
          'Why the memory system needs many requests in flight, and where they come from',
          'Vì sao hệ thống bộ nhớ cần nhiều yêu cầu đang bay, và chúng đến từ đâu',
        ),
      ],
      ['6.3', tr('Vector loads and stores', 'Nạp và ghi vector'), tr('Fewer, wider requests for the same bytes', 'Ít yêu cầu hơn, rộng hơn cho cùng số byte')],
    ];
    },
  },
  {
    get part() {
      return tr('Shared memory', 'Bộ nhớ chia sẻ');
    },
    get rows(): [string, string, string][] {
      return [
      [
        '6.4',
        tr('Shared memory bank conflicts', 'Xung đột bank của bộ nhớ chia sẻ'),
        tr(
          'When a warp’s shared-memory access is split into passes, and padding',
          'Khi nào truy cập bộ nhớ chia sẻ của một warp bị chia thành nhiều lượt, và chèn đệm',
        ),
      ],
    ];
    },
  },
  {
    get part() {
      return tr('Work per thread', 'Lượng việc mỗi luồng');
    },
    get rows(): [string, string, string][] {
      return [
      [
        '6.5',
        tr('Thread coarsening', 'Gộp việc cho luồng'),
        tr('Several units of work per thread, and when that backfires', 'Nhiều đơn vị việc mỗi luồng, và khi nào điều đó phản tác dụng'),
      ],
      [
        '6.6',
        tr('Loop unrolling', 'Trải vòng lặp'),
        tr(
          'Fewer branches, more independent instructions, arrays kept in registers',
          'Ít nhánh hơn, nhiều lệnh độc lập hơn, mảng được giữ trong thanh ghi',
        ),
      ],
    ];
    },
  },
  {
    get part() {
      return tr('Synchronisation', 'Đồng bộ hóa');
    },
    get rows(): [string, string, string][] {
      return [['6.7', tr('Double buffering', 'Buffer kép'), tr('Removing the barrier that only protects a location', 'Bỏ rào chắn chỉ bảo vệ một vị trí')]];
    },
  },
  {
    get part() {
      return tr('Method', 'Phương pháp');
    },
    get rows(): [string, string, string][] {
      return [
      [
        '6.8',
        tr('A checklist of optimisations', 'Danh sách kiểm tra các tối ưu hóa'),
        tr(
          'Twelve rows, from a symptom to what relieves it and what it costs',
          'Mười hai hàng, từ một triệu chứng đến thứ làm giảm nó và cái giá của nó',
        ),
      ],
      [
        '6.9',
        tr('Optimisation strategy', 'Chiến lược tối ưu hóa'),
        tr(
          'Find the bottleneck, fix it, measure, and know when to stop',
          'Tìm điểm nghẽn, sửa nó, đo, và biết khi nào dừng',
        ),
      ],
    ];
    },
  },
];
const LABS = (): [string, string, string][] => [
  ['6.1', tr('Coalescing on your device', 'Hợp nhất truy cập bộ nhớ trên thiết bị của bạn'), '6.1'],
  ['6.2', tr('Vector loads, with the boundaries done right', 'Nạp vector, xử lý biên cho đúng'), '6.3'],
  ['6.3', tr('Provoke a bank conflict, then pad it away', 'Gây ra xung đột bank, rồi chèn đệm để khử nó'), '6.4'],
  [
    '6.4',
    tr(
      'The linear layer in the framework layout, corner-turned (milestone M2, “tuned”)',
      'Lớp tuyến tính trong bố cục của framework, xoay góc (mốc dự án M2, “tuned”)',
    ),
    '6.1, 6.4',
  ],
  ['6.5', tr('Double buffering, with a race detector watching', 'Buffer kép, với một bộ phát hiện race condition theo dõi'), '6.7'],
];

export function S60() {
  const q = questionsS60;
  return (
    <>
      <Translated
        vi={
          <p>
            Chương 5 cắt giảm số byte mà một kernel yêu cầu từ bộ nhớ toàn cục (global memory). Chương này nói về cách nó yêu
            cầu chúng. Cùng những byte đó có thể đến với tốc độ đầy đủ hoặc chỉ một phần nhỏ của nó, tùy vào những địa chỉ
            mà các luồng của một warp cùng yêu cầu, bao nhiêu yêu cầu đang bay cùng lúc, và mỗi yêu cầu rộng bao nhiêu.
            Chương bắt đầu từ chính các chip nhớ (vì sao chúng giao dữ liệu theo từng loạt, và qua các kênh và bank song
            song), chuyển sang các bank của bộ nhớ chia sẻ (shared memory), rồi thêm ba công cụ thay đổi lượng việc mỗi luồng
            làm và mức độ thường xuyên nó phải chờ: gộp việc, trải vòng lặp và buffer kép. Chương kết thúc bằng một danh
            sách kiểm tra mọi tối ưu hóa trong Phần 1, và một phương pháp để chọn giữa chúng.
          </p>
        }
      >
        <p>
          Chapter 5 cut how many bytes a kernel asks global memory for. This chapter is about how it asks for them. The
          same bytes can arrive at full speed or at a fraction of it, depending on which addresses the threads of a warp
          request together, how many requests are in flight at once, and how wide each one is. The chapter starts
          from the memory chips themselves (why they deliver data in bursts, and in parallel channels and banks), turns
          to the banks of shared memory, then adds three tools that change how much work each thread does and how often
          it waits: coarsening, unrolling and double buffering. It ends with a checklist of every optimisation in
          Part 1, and a method for choosing among them.
        </p>
      </Translated>
      <Source section="6" pages="123–124" />
      <Translated
        vi={
          <p>
            Phương pháp đó dựa trên một ý tưởng từ trang đầu tiên của chương. Tinh chỉnh một kernel gần như luôn đổi một tài
            nguyên lấy một tài nguyên khác, và cuộc trao đổi chỉ có lợi nếu nó giải tỏa tài nguyên đang thực sự giới hạn
            kernel, tức điểm nghẽn của nó. Mục <a href={href('ch06', '6.9')}>6.9</a> khép lại vòng đó.
          </p>
        }
      >
        <p>
          The method rests on one idea from the chapter&rsquo;s first page. Tuning a kernel almost always trades one
          resource for another, and the trade pays only if it relieves the resource that is actually limiting the
          kernel, its bottleneck. Section <a href={href('ch06', '6.9')}>6.9</a> closes that loop.
        </p>
      </Translated>

      <h2>
        <Translated vi="Bên dưới một lời gọi framework">Under a framework call</Translated>
      </h2>
      <Translated
        vi={
          <p>
            <code>torch.nn.Linear</code> lưu trọng số dạng <code>(out_features, in_features)</code> và tính{' '}
            <code>y = x W^T</code>: mỗi đầu ra là tích vô hướng của một hàng của <code>x</code> với một hàng của{' '}
            <code>W</code>. ggml, bên dưới llama.cpp, cũng làm vậy. Các lớp tuyến tính của bạn trong lab 3.3 và 5.3 giữ bố cục
            của sách, với trọng số lưu dạng <code>(in, out)</code> và được đọc dọc theo một cột. Lab 6.4 chuyển lớp tuyến tính
            của dự án sang bố cục của framework, và mục 6.1 cho thấy vì sao bố cục đó cần một cách nạp ô khác. Điều này quan
            trọng nhất ở nơi bạn có thể đoán: một bước decode batch 1 bị giới hạn bởi bộ nhớ dù được viết thế nào (
            <a href={href('ch05', '5.1')}>mục 5.1</a>), nên các kỹ thuật bộ nhớ của chương này chính là những kỹ thuật thay
            đổi tốc độ decode.
          </p>
        }
      >
        <p>
          <code>torch.nn.Linear</code> stores its weight as <code>(out_features, in_features)</code> and computes{' '}
          <code>y = x W^T</code>: each output is the dot product of a row of <code>x</code> with a row of{' '}
          <code>W</code>. ggml, under llama.cpp, does the same. Your linear layers in labs 3.3 and 5.3 kept the
          book&rsquo;s layout, with the weight stored <code>(in, out)</code> and read down a column. Lab 6.4 moves the
          project&rsquo;s linear layer to the framework layout, and section 6.1 shows why that layout needs a different
          tile load. It matters most where you would expect: a batch-1 decode step is memory-bound however it is written
          (<a href={href('ch05', '5.1')}>section 5.1</a>), so the memory techniques of this chapter are the ones that move
          decode speed.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cây cầu nối này là của chúng tôi. Các quy ước của PyTorch và ggml lấy từ <code>notes/today/ch03-facts.md</code>{' '}
              D2 và D6 (tài liệu và mã nguồn PyTorch 2.15; <code>ggml_mul_mat</code> của llama.cpp).
            </p>
          }
        >
          <p>
            The bridge is ours. The PyTorch and ggml conventions are from <code>notes/today/ch03-facts.md</code> D2 and
            D6 (PyTorch 2.15 documentation and source; llama.cpp&rsquo;s <code>ggml_mul_mat</code>).
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Các phép đo của bạn cho đến nay">Your measurements so far</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai kết quả từ các lab trước thuộc về chương này. Lab 3.4 đo thời gian ba cách chia một phép nhân ma trận cho
            các luồng, tất cả làm cùng lượng số học; khác biệt về thời gian giữa chúng là chủ đề của 6.1. Lab 5.2 đo kernel
            phân ô của bạn so với kernel ngây thơ, đó là nơi chương này bắt đầu.
          </p>
        }
      >
        <p>
          Two results from earlier labs belong to this chapter. Lab 3.4 timed three ways of splitting a matrix product
          among threads, all doing the same arithmetic; their difference in time is the subject of 6.1. Lab 5.2 timed
          your tiled kernel against the naive one, which is where this chapter starts.
        </p>
      </Translated>
      <Measurements />

      <h2>
        <Translated vi="Toàn cảnh chương">The chapter at a glance</Translated>
      </h2>
      <table className="data" data-map>
        <thead>
          <tr>
            <th>
              <Translated vi="Chủ đề">About</Translated>
            </th>
            <th>
              <Translated vi="Mục">Section</Translated>
            </th>
            <th>
              <Translated vi="Những gì bạn sẽ làm được">What you will be able to do</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {MAP.flatMap((g) =>
            g.rows.map(([id, title, what], i) => (
              <tr key={id}>
                <td>{i === 0 ? g.part : ''}</td>
                <td>
                  <a href={href('ch06', id)}>
                    {id} {title}
                  </a>
                </td>
                <td>{what}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Năm lab, trên <a href={href('ch06', 'labs')}>trang lab</a>, mỗi lab có một dự đoán bạn phải chốt trước khi đo:
          </p>
        }
      >
        <p>Five labs, on the <a href={href('ch06', 'labs')}>labs page</a>, each with a prediction to commit before you measure:</p>
      </Translated>
      <table className="data" data-labs>
        <thead>
          <tr>
            <th>Lab</th>
            <th>
              <Translated vi="Những gì bạn xây dựng">What you build</Translated>
            </th>
            <th>
              <Translated vi="Các mục">Sections</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {LABS().map(([n, t, s]) => (
            <tr key={n} data-lab={n}>
              <td>{n}</td>
              <td>{t}</td>
              <td>{s}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Năm câu hỏi về những gì chương này giả định từ chương 3 đến 5. Mỗi lời giải thích chỉ tới nơi ý tưởng đó được
            dạy và tới mục ở đây xây tiếp trên nó. Chúng sẽ quay lại trong phần <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Five questions on what this chapter assumes from chapters 3 to 5. Each explanation points to where the idea was
          taught and to the section here that builds on it. They come back in <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={q['ch06/start/q-neighbours']} label={tr('From chapter 3', 'Từ chương 3')} />
      <Ask q={q['ch06/start/q-warp2d']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={q['ch06/start/q-hide']} label={tr('From chapter 4', 'Từ chương 4')} />
      <Ask q={q['ch06/start/q-false']} label={tr('From chapter 5', 'Từ chương 5')} />
      <Ask q={q['ch06/start/q-vecadd']} label={tr('From chapter 5', 'Từ chương 5')} />
    </>
  );
}
