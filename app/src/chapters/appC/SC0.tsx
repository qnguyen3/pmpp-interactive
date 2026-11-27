import { Ask, Predict, type Question } from '../../components/Ask';
import { Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { DerefMap } from './DerefMap';

const P_WEIGHTS = 'appC/start/p-weights';
/** 8 billion parameters, 2 bytes each. */
export const WEIGHT_BYTES = 8e9 * 2;

// ---------------------------------------------------------------- your device memory
interface MemRow {
  backend: string;
  name: string;
  /** [label, bytes] pairs, as the device reported them. */
  figures: [string, number][];
  /** The figure the weights are compared against. */
  budget: number;
  budgetLabel: string;
}

export function memoryRows(reports: Reports): MemRow[] {
  return Object.keys(reports)
    .filter((b) => b !== 'emu' && b !== 'race')
    .flatMap((b): MemRow[] => {
      const d = reports[b].device ?? {};
      const n = (k: string) => {
        const v = Number(d[k]);
        return isFinite(v) && v > 0 ? v : undefined;
      };
      const name = String(d.name ?? b);
      if (b === 'metal') {
        const ws = n('recommendedMaxWorkingSetSize');
        const phys = n('physicalMemory');
        if (!ws) return [];
        return [
          {
            backend: b,
            name,
            figures: [['recommendedMaxWorkingSetSize', ws], ...(phys ? [['physicalMemory', phys] as [string, number]] : [])],
            budget: ws,
            budgetLabel: tr('the recommended working set', 'working set được khuyến nghị'),
          },
        ];
      }
      const total = n('totalGlobalMem');
      if (!total) return [];
      return [{ backend: b, name, figures: [['totalGlobalMem', total]], budget: total, budgetLabel: tr('device memory', 'bộ nhớ device') }];
    });
}

const gb = (bytes: number) => (bytes / 1e9).toFixed(1) + ' GB';

function YourMemory() {
  const { reports, loading } = useReports();
  if (loading)
    return (
      <p className="muted">
        <Translated vi="Đang đọc báo cáo thiết bị của bạn…">Reading your device reports…</Translated>
      </p>
    );
  const rows = memoryRows(reports);
  if (!rows.length)
    return (
      <p data-yours="none" style={{ marginBottom: '1em' }}>
        <Translated
          vi={
            <>
              <b>Chưa có thiết bị nào được ghi nhận trên máy này.</b> Hãy chạy{' '}
              <code>python3 labs/pmpp.py run appC/lab1 --bench</code> rồi tải lại trang này: lần chạy đầu tiên trên một
              backend sẽ ghi lại những gì thiết bị báo cáo.
            </>
          }
        >
          <b>No device recorded on this machine.</b> Run <code>python3 labs/pmpp.py run appC/lab1 --bench</code> and
          reload this page: the first run on a backend records what the device reports.
        </Translated>
      </p>
    );
  return (
    <div data-yours="some" style={{ marginBottom: '1em' }}>
      {rows.map((r) => {
        const fits = WEIGHT_BYTES <= r.budget;
        return (
          <p key={r.backend} data-backend={r.backend} data-fits={fits ? 'yes' : 'no'}>
            <b>
              {r.name} ({r.backend})
            </b>{' '}
            <Translated vi="báo cáo">reports</Translated>{' '}
            {r.figures.map(([k, v], i) => (
              <span key={k}>
                {i ? tr(' and ', ' và ') : ''}
                <code>{k}</code> = {gb(v)}
              </span>
            ))}
            <Translated
              vi={
                <>
                  . 16 GB trọng số {fits ? 'vừa trong' : 'không vừa trong'} {r.budgetLabel}
                  {fits ? ', trước khi activation và KV cache đòi phần của chúng.' : '.'}
                </>
              }
            >
              . The 16 GB of weights {fits ? 'fit under' : 'do not fit in'} {r.budgetLabel}
              {fits ? ', before activations and the KV cache ask for their share.' : '.'}
            </Translated>
          </p>
        );
      })}
      {rows.some((r) => r.backend === 'metal') ? (
        <Translated
          vi={
            <p className="muted">
              Trên Apple silicon, trọng số nằm trong một bộ nhớ duy nhất mà cả hai bộ xử lý cùng dùng. Apple gọi{' '}
              <code>recommendedMaxWorkingSetSize</code> là lượng bộ nhớ GPU có thể dùng "mà không ảnh hưởng tới hiệu năng
              lúc chạy": đó là lời khuyên, không phải giới hạn cứng, và Apple không ghi lại cách nào để dùng vượt quá nó
              (appC-facts C5).
            </p>
          }
        >
          <p className="muted">
            On Apple silicon the weights sit in the one memory both processors use. Apple calls{' '}
            <code>recommendedMaxWorkingSetSize</code> the amount the GPU can use "without affecting its runtime
            performance": advice, not a hard limit, and Apple documents no way to oversubscribe it (appC-facts C5).
          </p>
        </Translated>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC0: Record<string, Question> = {
  'appC/start/q-hostderef': {
    kind: 'choice',
    id: 'appC/start/q-hostderef',
    prompt: (
      <Translated
        vi={
          <>
            Mã host CUDA. Sau <code>cudaMalloc((void**)&amp;A_d, n*sizeof(float))</code> và một lệnh{' '}
            <code>cudaMemcpy</code> vào <code>A_d</code>, host chạy <code>printf("%f", A_d[0])</code>. Điều gì xảy ra?
          </>
        }
      >
        <>
          CUDA host code. After <code>cudaMalloc((void**)&amp;A_d, n*sizeof(float))</code> and a <code>cudaMemcpy</code>{' '}
          into <code>A_d</code>, the host runs <code>printf("%f", A_d[0])</code>. What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không xác định: mã host không được giải tham chiếu bộ nhớ cudaMalloc, và thường thì chương trình crash.">
            Undefined: host code may not dereference cudaMalloc memory, and a crash is typical.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó in ra giá trị, chậm, vì lần đọc phải đi qua đường kết nối.">
            It prints the value, slowly, because the read crosses the interconnect.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là điều AMD ghi trong tài liệu cho bộ nhớ hipMalloc dưới HIP (appC-facts B3). CUDA không cho phép điều đó.">
            That is what AMD documents for hipMalloc memory under HIP (appC-facts B3). CUDA does not allow it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó in ra giá trị: từ CUDA 4, định địa chỉ hợp nhất đặt host và device vào một không gian địa chỉ, nên mã host có thể đi theo con trỏ.">
            It prints the value: since CUDA 4, unified addressing puts host and device in one address space, so host code can follow the pointer.
          </Translated>
        ),
        why: (
          <Translated vi="Một không gian địa chỉ không có nghĩa là một quyền truy cập. Mục C.3 nói chính về khác biệt này.">
            One address space is not one access right. Section C.3 is about exactly this difference.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó chạy được sau khi đã gọi cudaDeviceSynchronize().">
            It works once cudaDeviceSynchronize() has been called.
          </Translated>
        ),
        why: (
          <Translated vi="Đồng bộ hóa sắp xếp thứ tự công việc. Nó không làm cho bộ nhớ device đọc được từ mã host.">
            Synchronising orders the work. It does not make device memory readable from host code.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>A_d</code> chứa một địa chỉ trong bộ nhớ device. Mã host có thể truyền nó cho các lời gọi API và kernel,
            và không gì khác (<a href={href('ch02', '2.4')}>chương 2, mục 2.4</a>). Không thời kỳ nào của phụ lục này thay
            đổi điều đó: ngay cả trên các hệ thống có tính nhất quán (coherence) bằng phần cứng, NVIDIA vẫn nói mã host không
            thể với tới bộ nhớ <code>cudaMalloc</code>.
          </p>
        }
      >
        <p>
          <code>A_d</code> holds an address in device memory. Host code may pass it to API calls and kernels, and nothing
          else (<a href={href('ch02', '2.4')}>chapter 2, section 2.4</a>). No era of this appendix changes that: even on
          systems with hardware coherence, NVIDIA says host code cannot reach <code>cudaMalloc</code> memory.
        </p>
      </Translated>
    ),
    source: 'ch. 2 §2.4, p. 30; notes/today/ch02-facts.md A17; appC-facts A12, B3', // i18n-ignore: citation only
  },
  'appC/start/q-membound': {
    kind: 'numeric',
    id: 'appC/start/q-membound',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel bị giới hạn bởi bộ nhớ đọc 1 GiB một lần và mất 1 ms để đọc nó từ bộ nhớ device. Giờ cùng những byte
            đó đến qua một đường truyền chỉ có một phần mười băng thông ấy. Bây giờ mất bao nhiêu mili giây?
          </>
        }
      >
        <>
          A memory-bound kernel reads 1 GiB once and takes 1 ms reading it from device memory. The same bytes come
          instead over a link with one tenth of that bandwidth. How many milliseconds now?
        </>
      </Translated>
    ),
    answer: 10,
    tol: 0.01,
    unit: 'ms',
    explain: (
      <Translated
        vi={
          <p>
            Với một kernel bị giới hạn bởi bộ nhớ, thời gian bằng số byte chia cho băng thông (
            <a href={href('ch05', '5.1')}>chương 5, mục 5.1</a>). Cùng số byte, một phần mười băng thông: thời gian gấp
            mười. Một phần mười là quy tắc ước lượng của chính sách cho PCIe so với bộ nhớ device (p. 625).{' '}
            <a href={href('appC', 'C.2')}>Mục C.2</a> hỏi điều đó gây ra gì cho một kernel chỉ đọc một phần số byte của nó
            qua đường truyền.
          </p>
        }
      >
        <p>
          For a memory-bound kernel, time is bytes divided by bandwidth (
          <a href={href('ch05', '5.1')}>chapter 5, section 5.1</a>). Same bytes, a tenth of the bandwidth: ten times the
          time. One tenth is the book's own rule of thumb for PCIe against device memory (p. 625).{' '}
          <a href={href('appC', 'C.2')}>Section C.2</a> asks what that does to a kernel that reads only some of its bytes
          over the link.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.1, pp. 93–97; appendix C, p. 625',
    vi: { source: 'ch. 5 §5.1, pp. 93–97; phụ lục C, p. 625' },
  },
  'appC/start/q-sharedscope': {
    kind: 'choice',
    id: 'appC/start/q-sharedscope',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel khai báo <code>__shared__ float tile[256]</code>. Luồng 5 của khối 2 ghi <code>tile[5]</code>. Ai
            đọc được giá trị đó?
          </>
        }
      >
        <>
          A kernel declares <code>__shared__ float tile[256]</code>. Thread 5 of block 2 writes <code>tile[5]</code>. Who
          can read that value?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mọi luồng của khối 2, một khi có rào chắn tách lần ghi khỏi lần đọc.">
            Every thread of block 2, once a barrier separates the write from the read.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Mọi luồng của lưới, vì mọi khối đều gọi cùng một mảng tile.">
            Every thread of the grid, since every block names the same tile array.
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ chia sẻ có phạm vi khối: mỗi khối có tile riêng của nó.">
            Shared memory has block scope: every block has its own tile.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ luồng 5 của khối 2, luồng sở hữu tile[5] giống như sở hữu một biến cục bộ.">
            Thread 5 of block 2 alone, which owns tile[5] the way it owns a local variable.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là mô tả một thanh ghi hoặc một mảng cục bộ. Một biến __shared__ hiển thị với cả khối của nó.">
            That describes a register or a local array. A __shared__ variable is visible to its whole block.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Host, thông qua cudaMemcpy, một khi kernel đã chạy xong.">
            The host, through cudaMemcpy, once the kernel has finished.
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ chia sẻ chỉ tồn tại khi khối của nó đang chạy. Muốn giữ một giá trị, hãy ghi nó vào bộ nhớ toàn cục.">
            Shared memory lives only while its block runs. To keep a value, write it to global memory.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi khối một bản sao, hiển thị với các luồng của khối đó trong suốt vòng đời của khối (
            <a href={href('ch05', '5.2')}>chương 5, mục 5.2</a>). Hãy nhớ điều này cho{' '}
            <a href={href('appC', 'C.1')}>mục C.1</a>: một con trỏ tới <code>tile</code> chỉ có nghĩa bên trong khối sở
            hữu nó.
          </p>
        }
      >
        <p>
          One copy per block, visible to that block's threads for the block's lifetime (
          <a href={href('ch05', '5.2')}>chapter 5, section 5.2</a>). Keep this in mind for{' '}
          <a href={href('appC', 'C.1')}>section C.1</a>: a pointer to <code>tile</code> means something only inside the
          block that owns it.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.2, pp. 98–104',
  },
  'appC/start/q-ordering': {
    kind: 'choice',
    id: 'appC/start/q-ordering',
    prompt: (
      <Translated
        vi={
          <>
            Metal trên Apple silicon, buffer dùng chung (shared). CPU điền <code>input.contents</code>, commit một command
            buffer có kernel đọc <code>input</code>, rồi ngay lập tức ghi batch tiếp theo vào cùng buffer đó. Sai ở đâu?
          </>
        }
      >
        <>
          Metal on Apple silicon, shared buffers. The CPU fills <code>input.contents</code>, commits a command buffer whose
          kernel reads <code>input</code>, then immediately writes the next batch into the same buffer. What is wrong?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="GPU có thể chưa đọc batch đầu tiên, nên CPU có thể ghi đè lên nó ngay dưới chân kernel.">
            The GPU may not have read the first batch yet, so the CPU can overwrite it under the kernel.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không sai gì: unified memory có tính nhất quán, nên kernel thấy đúng dữ liệu.">
            Nothing: unified memory is coherent, so the kernel sees the right data.
          </Translated>
        ),
        why: (
          <Translated vi="Lưu trữ shared bỏ đi lần sao chép, không bỏ đi thứ tự. Apple bắt bạn chịu trách nhiệm về thứ tự.">
            Shared storage removes the copy, not the ordering. Apple makes you responsible for the ordering.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Lần ghi thất bại, vì buffer chỉ đọc khi một command buffer dùng nó đang thực thi.">
            The write fails, because a buffer is read-only while a command buffer that uses it is in flight.
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì ngăn lần ghi. Giữ hai phía tách khỏi nhau là việc của bạn.">
            Nothing stops the write. Keeping the two sides apart is your job.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Kernel đọc dữ liệu cũ từ bộ nhớ đệm GPU, nên hãy gọi didModifyRange trước.">
            The kernel reads stale data from a GPU cache, so call didModifyRange first.
          </Translated>
        ),
        why: (
          <Translated vi="didModifyRange thuộc về lưu trữ managed, vốn không có tác dụng trên Apple silicon.">
            didModifyRange belongs to managed storage, which has no effect on Apple silicon.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>commit</code> trả về trong khi công việc vẫn đang xếp hàng hoặc đang chạy. Với lưu trữ shared, không có
            lần sao chép nào để nấp sau: hãy chờ command buffer hoàn tất trước khi dùng lại buffer, hoặc ghi batch tiếp theo vào
            một buffer thứ hai. Bạn đã gặp quy tắc này ở <a href={href('ch02', '2.4')}>chương 2, mục 2.4</a>.
          </p>
        }
      >
        <p>
          <code>commit</code> returns while the work is still queued or running. With shared storage there is no copy to
          hide behind: wait for the command buffer to complete before you reuse the buffer, or write the next batch into
          a second buffer. You met the same rule in <a href={href('ch02', '2.4')}>chapter 2, section 2.4</a>.
        </p>
      </Translated>
    ),
    source: 'not in the book; notes/today/ch05-facts.md C6',
    vi: { source: 'không có trong sách; notes/today/ch05-facts.md C6' },
  },
};

// ---------------------------------------------------------------- the page
export function SC0() {
  return (
    <>
      <Translated
        vi={
          <p>
            Ở chương 2, bạn đã viết mã host theo ba phần: cấp phát bộ nhớ device và sao chép đầu vào sang, khởi chạy kernel,
            rồi sao chép kết quả ra và giải phóng. Đằng sau đó là một hình dung đơn giản về cỗ máy. CPU có bộ nhớ của nó, GPU
            có bộ nhớ riêng, và không có gì đi qua lại giữa chúng ngoài những gì <code>cudaMemcpy</code> mang theo. Sách nói
            các GPU đời đầu không hỗ trợ gì khác.
          </p>
        }
      >
        <p>
          In chapter 2 you wrote host code in three parts: allocate device memory and copy the inputs in, launch the
          kernel, then copy the result out and free. Behind it sits a simple picture of the machine. The CPU has its
          memory, the GPU has its own, and nothing crosses between them except what <code>cudaMemcpy</code> carries. The
          book says early GPUs supported nothing else.
        </p>
      </Translated>
      <Source pages="623" />
      <Translated
        vi={
          <p>
            Bạn đã dùng mô hình này mà không phải tự viết ra. Trong PyTorch, <code>tensor.to('cuda')</code> kết thúc bằng
            một lệnh <code>cudaMemcpyAsync</code> từ host sang device. Trên máy Mac, <code>tensor.to('mps')</code> vẫn sao
            chép, dù ở đó CPU và GPU dùng chung một bộ nhớ.
          </p>
        }
      >
        <p>
          You have used this model without writing it. In PyTorch, <code>tensor.to('cuda')</code> ends in a
          host-to-device <code>cudaMemcpyAsync</code>. On a Mac, <code>tensor.to('mps')</code> still copies, although the
          CPU and the GPU share one memory there.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Mối liên hệ với PyTorch là của chúng tôi, đọc từ mã nguồn PyTorch ngày 2026-11-20 (notes/today/ch02-facts.md
              D4, như được trích dẫn trong appC-facts).
            </p>
          }
        >
          <p>
            The PyTorch connection is ours, read from PyTorch's source on 2026-11-20 (notes/today/ch02-facts.md D4, as
            cited in appC-facts).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Phụ lục này là lịch sử mà sách kể lại về cách CUDA nới lỏng hình dung đó, từng bước một. Nó dài bảy trang, không
            có bài tập, và là nền tảng cho một câu hỏi bạn sẽ còn gặp mãi trong mã suy luận: tensor này nằm ở đâu, và ai đọc
            được nó từ đó?
          </p>
        }
      >
        <p>
          This appendix is the book's history of how CUDA loosened that picture, one step at a time. It is seven pages
          with no exercises, and it is the background to a question you will keep meeting in inference code: where does
          this tensor live, and who can read it from there?
        </p>
      </Translated>
      <Today
        checked={tr(
          "2026-11-20 against NVIDIA's and AMD's documentation (notes/today/appC-facts.md, A1)",
          'ngày 2026-11-20, đối chiếu với tài liệu của NVIDIA và AMD (notes/today/appC-facts.md, A1)',
        )}
      >
        <Translated
          vi={
            <p>
              Vẫn là mặc định trên GPU rời. Hướng dẫn của NVIDIA mở đầu chương bộ nhớ bằng các bộ nhớ vật lý tách biệt, và
              cả NVIDIA lẫn AMD vẫn mô tả bộ nhớ GPU là nhanh hơn nhưng nhỏ hơn bộ nhớ của CPU.
            </p>
          }
        >
          <p>
            Still the default on discrete GPUs. NVIDIA's guide opens its memory chapter with separate physical memories,
            and NVIDIA and AMD both still describe GPU memory as faster but smaller than the CPU's.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Vì sao sao chép mọi thứ không còn đủ">Why copying everything stops being enough</Translated>
      </h2>
      <Translated vi={<p>Sách nêu hai vấn đề của mô hình sao chép.</p>}>
        <p>The book names two problems with the copy model.</p>
      </Translated>
      <Source pages="623" />
      <Translated
        vi={
          <p>
            <b>Thiết bị vào và ra.</b> Bộ điều khiển ổ đĩa và card mạng được thiết kế để đọc và ghi bộ nhớ host. Dữ liệu
            vào từ ổ đĩa rơi vào bộ nhớ host rồi phải được sao chép sang GPU, và dữ liệu ra đi cùng con đường đó theo chiều
            ngược lại. Mỗi chuyến đi cộng thêm độ trễ và làm giảm thông lượng. Để các thiết bị đó làm việc thẳng trên bộ nhớ
            GPU sẽ nhanh hơn và đơn giản hơn.
          </p>
        }
      >
        <p>
          <b>Input and output devices.</b> Disk controllers and network cards are built to read and write host memory.
          Input from a disk lands in host memory and then has to be copied to the GPU, and output makes the same trip back.
          Each trip adds latency and costs throughput. Letting those devices work on GPU memory directly would be faster
          and simpler.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Câu trả lời của NVIDIA ngày nay là họ GPUDirect: ổ lưu trữ và bộ điều hợp mạng đọc và ghi trực tiếp bộ nhớ
              GPU (appC-facts A19). Khóa học này không dùng nó.
            </p>
          }
        >
          <p>
            NVIDIA's answer today is the GPUDirect family: storage drives and network adapters that read and write GPU
            memory directly (appC-facts A19). This course does not use it.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            <b>Dữ liệu lớn hơn GPU.</b> Chương trình giữ dữ liệu của chúng trong bộ nhớ host, và bộ nhớ GPU thường nhanh hơn
            nhưng nhỏ hơn. Dữ liệu không vừa phải được cắt thành những phần vừa và đưa qua từng phần một. Chương 21 của sách
            làm đúng như vậy, đẩy một lưới 3D qua GPU theo từng lát 2D (khóa học này dành cho nó hai câu trong{' '}
            <a href={href('ch21', '21.2')}>mục 21.2</a>). Một số cấu trúc không có cách nào tốt để cắt cả. Những câu trả
            lời tốt hơn của sách: để GPU đọc trực tiếp bộ nhớ host, hoặc để runtime chỉ di chuyển phần dữ liệu mà kernel
            thực sự chạm tới.
          </p>
        }
      >
        <p>
          <b>Data larger than the GPU.</b> Programs keep their data in host memory, and GPU memory tends to be faster but
          smaller. Data that does not fit has to be cut into pieces that do and moved through piece by piece. The book's
          chapter 21 does exactly this, pushing a 3D grid through the GPU in 2D slices (this course gives it two sentences
          in <a href={href('ch21', '21.2')}>section 21.2</a>). Some structures have no good way to be cut
          at all. The book's better answers: let the GPU read host memory directly, or let the runtime move only the data
          a kernel actually touches.
        </p>
      </Translated>
      <Source pages="623" figs={tr("chapter 21's slices are on p. 517", 'các lát của chương 21 ở p. 517')} />
      <Translated
        vi={
          <p>
            Vấn đề thứ hai là vấn đề bạn sẽ gặp trước, dưới dạng một mô hình không vừa bộ nhớ. Hãy chốt một con số trước khi
            nhìn vào thiết bị của chính bạn.
          </p>
        }
      >
        <p>
          The second problem is the one you will meet first, as a model that does not fit. Commit to a number before you
          look at your own device.
        </p>
      </Translated>
      <Predict
        id={P_WEIGHTS}
        prompt={
          <Translated
            vi={
              <>
                Một mô hình 8 tỉ tham số lưu bằng giá trị 16-bit: trọng số chiếm bao nhiêu GB (10<sup>9</sup> byte)?
              </>
            }
          >
            <>
              An 8-billion-parameter model stored in 16-bit values: how many GB (10<sup>9</sup> bytes) of weights?
            </>
          </Translated>
        }
        unit="GB"
      >
        {(v) => (
          <>
            <Translated
              vi={
                <p>
                  {Math.abs(v - 16) < 0.05 ? 'Đúng: ' : 'Đáp án là 16 GB: '}8 × 10<sup>9</sup> tham số × 2 byte mỗi tham số
                  = 1.6 × 10<sup>10</sup> byte. Đây là những gì các thiết bị của bạn báo cáo.
                </p>
              }
            >
              <p>
                {Math.abs(v - 16) < 0.05 ? 'Right: ' : 'The answer is 16 GB: '}8 × 10<sup>9</sup> parameters × 2 bytes
                each = 1.6 × 10<sup>10</sup> bytes. Here is what your devices report.
              </p>
            </Translated>
            <YourMemory />
            <Translated
              vi={
                <p>
                  Khi một mô hình không vừa, phụ lục đưa ra ba câu trả lời. Cắt nó thành những phần vừa và cho chúng chạy
                  qua lần lượt, như chương 21 làm. Đọc nó tại chỗ từ bộ nhớ host qua đường kết nối (
                  <a href={href('appC', 'C.2')}>mục C.2</a>). Hoặc để runtime di chuyển trang theo nhu cầu và dùng vượt quá
                  bộ nhớ GPU (<a href={href('appC', 'C.8')}>mục C.8</a>).
                </p>
              }
            >
              <p>
                When a model does not fit, the appendix offers three answers. Cut it into pieces that fit and stream them
                through, as chapter 21 does. Read it in place from host memory over the interconnect (
                <a href={href('appC', 'C.2')}>section C.2</a>). Or let the runtime migrate pages on demand and
                oversubscribe GPU memory (<a href={href('appC', 'C.8')}>section C.8</a>).
              </p>
            </Translated>
          </>
        )}
      </Predict>

      <h2>
        <Translated vi="Bản đồ">The map</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phần còn lại của phụ lục, các mục C.1 đến C.9, là lịch sử trong bảy bước; hai bước trong số đó chiếm hai mục mỗi
            bước. Mỗi bước thay đổi câu trả lời cho một câu hỏi: cho một con trỏ, ai được giải tham chiếu nó, và tốn bao
            nhiêu? Bản đồ dưới đây là câu trả lời đó, qua từng thời kỳ, cộng thêm một bước mà sách không đi tới. Hãy đi qua
            nó bằng nút Trước và Tiếp, và để ý các ô được đánh dấu "đã đổi". Ba điều cần tìm ra:
          </p>
        }
      >
        <p>
          The rest of the appendix, sections C.1 to C.9, is the history in seven steps; two of them take two sections
          each. Each step changes the answer to one question: given a pointer, who may dereference it, and what does it
          cost? The map below is that answer, era by era, plus one step the book does not reach. Step through it with Previous and Next, and look at the cells marked
          "changed". Three things to find out:
        </p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>khi nào các địa chỉ trở thành một (C.3), và quyền truy cập theo sau bao nhiêu bước;</li>
            <li>
              liệu kernel có đọc được bộ nhớ <code>malloc</code> thường hay không, một khi Pascal mang lại lỗi trang;
            </li>
            <li>
              liệu bộ nhớ <code>cudaMalloc</code> có bao giờ trở nên đọc được từ mã host, trên CUDA và trên HIP.
            </li>
          </ul>
        }
      >
        <ul>
          <li>when the addresses became one (C.3), and how many steps later access followed;</li>
          <li>
            whether a kernel can read plain <code>malloc</code> memory once Pascal brings page faults;
          </li>
          <li>
            whether <code>cudaMalloc</code> memory ever becomes readable from host code, on CUDA and on HIP.
          </li>
        </ul>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi ô là một quy tắc có tài liệu; chi phí là các hạng, không phải con số. Windows, WSL và một số hệ thống Tegra
              ngày nay có chung các ô với thời kỳ giới hạn của CUDA 6, và thời kỳ đó được ghi nhãn như vậy. "Không có tài
              liệu" nghĩa là các nguồn của chúng tôi không nói tới.
            </p>
          }
        >
          <p>
            One cell is one documented rule; costs are classes, not numbers. Windows, WSL and some Tegra systems today
            share the cells of the CUDA 6 limited era, and that era is labelled so. "Not documented" means our sources do
            not say.
          </p>
        </Translated>
      </Simplification>
      <Translated
        vi={
          <p>
            Ở chỗ các nguồn của chúng tôi ghi ngày của một bước khác với sách, bản đồ cho thấy cả hai. Whitepaper của Fermi
            ra năm 2009, nhưng các sản phẩm của nó lên kệ vào tháng 3 năm 2010. Kepler ra mắt năm 2012, không phải 2013, và
            nó không phải GPU đầu tiên có địa chỉ 64-bit. CUDA 6 được công bố vào tháng 11 năm 2013 và phát hành vào tháng
            4 năm 2014. Các lời gọi đặt trước và ánh xạ của C.9 xuất hiện trong CUDA 10.2, không phải CUDA 11.
          </p>
        }
      >
        <p>
          Where our sources date a step differently from the book, the map shows both. Fermi's whitepaper is from 2009,
          but its products reached the market in March 2010. Kepler launched in 2012, not 2013, and it was not the first
          GPU with 64-bit addresses. CUDA 6 was announced in November 2013 and shipped in April 2014. The reserve-and-map
          calls of C.9 arrived in CUDA 10.2, not CUDA 11.
        </p>
      </Translated>
      <Today
        checked={tr(
          "2026-11-20 against NVIDIA's documents and press releases (notes/today/appC-facts.md, A2, A6, A9, A15)",
          'ngày 2026-11-20, đối chiếu với tài liệu và thông cáo báo chí của NVIDIA (notes/today/appC-facts.md, A2, A6, A9, A15)',
        )}
      >
        <Translated vi={<p>Bốn mốc ngày của sách cần sửa. Bản đồ ghi nhãn mỗi mốc bằng mục facts của nó.</p>}>
          <p>Four of the book's dates need a correction. The map labels each one with its facts item.</p>
        </Translated>
      </Today>
      <DerefMap />
      <Translated
        vi={
          <p>
            Mỗi trang mục của phụ lục này đi sâu vào một thời kỳ và liên kết ngược về bản đồ này. Công tắc nền tảng cho
            thấy các hàng đó trông như thế nào trên HIP và trên Metal ngày nay. Hóa ra HIP cũng có các thời kỳ như trong
            sách, nhưng được chọn theo từng GPU và từng tiến trình thay vì theo ngày. Metal phần lớn chỉ có một câu trả lời,
            vì nó chỉ có một bộ nhớ.
          </p>
        }
      >
        <p>
          Each section page of this appendix goes deeper into one era and links back to this map. The platform switch
          shows what the same rows look like on HIP and on Metal today. HIP turns out to have the book's eras too, chosen
          per GPU and per process instead of by date. Metal mostly has one answer, because it has one memory.
        </p>
      </Translated>

      <h2>
        <Translated vi="Một cái tên, hai thứ">One name, two things</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một lời cảnh báo trước khi bạn đi tiếp, vì nếu không thì từ này sẽ đánh lừa bạn. <b>Unified Memory</b> của CUDA
            (mục C.6) là một vùng các trang mà runtime CUDA di chuyển giữa hai bộ nhớ vật lý. <b>unified memory</b>{' '}
            của Apple là một bộ nhớ vật lý duy nhất mà CPU và GPU dùng chung, không có gì phải di chuyển. Chúng chung một cái
            tên và gần như không chung gì khác. Trong phụ lục này, chữ U viết hoa luôn chỉ tính năng của CUDA, và{' '}
            <a href={href('appC', 'C.6')}>mục C.6</a> đặt hai thứ cạnh nhau.
          </p>
        }
      >
        <p>
          A warning before you go on, because the word will mislead you otherwise. CUDA's <b>Unified Memory</b> (section
          C.6) is a pool of pages that the CUDA runtime moves between two physical memories. Apple's <b>unified memory</b>{' '}
          is one physical memory that the CPU and the GPU share, with nothing to move. They share a name and little else.
          In this appendix the capital U always means CUDA's feature, and <a href={href('appC', 'C.6')}>section C.6</a>{' '}
          sets the two side by side.
        </p>
      </Translated>
      <Beyond title={tr('The two definitions', 'Hai định nghĩa')}>
        <Translated
          vi={
            <p>
              NVIDIA: "A CUDA feature that enables managed memory which can be automatically migrated between CPU and
              GPUs." (một tính năng của CUDA cho phép bộ nhớ được quản lý tự động di chuyển giữa CPU và các GPU). Apple:
              "the CPU and the GPU share system memory" (CPU và GPU dùng chung bộ nhớ hệ thống) (appC-facts, thuật ngữ;
              ch05-facts C6).
            </p>
          }
        >
          <p>
            NVIDIA: "A CUDA feature that enables managed memory which can be automatically migrated between CPU and
            GPUs." Apple: "the CPU and the GPU share system memory" (appC-facts, terminology; ch05-facts C6).
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bốn câu hỏi về những gì phụ lục này giả định từ chương 2 và 5, và từ phần Metal của chương 2. Mỗi lời giải thích
            liên kết ngược về nơi ý tưởng xuất phát, và mỗi câu hỏi sẽ quay lại trong phần{' '}
            <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Four questions on what this appendix assumes from chapters 2 and 5, and from the Metal side of chapter 2. Each
          explanation links back to where the idea comes from, and each question returns in{' '}
          <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={questionsSC0['appC/start/q-hostderef']} label={tr('From chapter 2', 'Từ chương 2')} />
      <Ask q={questionsSC0['appC/start/q-membound']} label={tr('From chapter 5', 'Từ chương 5')} />
      <Ask q={questionsSC0['appC/start/q-sharedscope']} label={tr('From chapter 5', 'Từ chương 5')} />
      <Ask q={questionsSC0['appC/start/q-ordering']} label={tr('From chapter 2, Metal', 'Từ chương 2, Metal')} />
    </>
  );
}
