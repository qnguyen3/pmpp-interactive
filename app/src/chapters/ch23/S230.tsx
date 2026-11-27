import { useState, type CSSProperties, type KeyboardEvent, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  EXERCISE1,
  bestTiles,
  coverage,
  factorisations,
  firstRows,
  launches,
  partition,
  pointerExpr,
  rowKind,
  rowShares,
  tileOf,
  tileSplit,
  visibleRows,
  type Launch,
  type Partition,
  type TileSplit,
} from '../../sim/halopart';

// Prerequisite questions drawn from chapters 2, 4, 8, 10 and 22 and appendix C (their notes
// and facts files), plus one of this chapter's exercises. Each explanation says where the idea
// was taught and where this chapter uses it.
export const questionsS230: Record<string, Question> = {
  'ch23/start/q-queued': {
    kind: 'choice',
    id: 'ch23/start/q-queued',
    prompt: (
      <Translated
        vi={
          <>
            <pre className="mono" style={{ margin: '0 0 0.6rem', whiteSpace: 'pre-wrap' }}>
              {'t0 = now();\nscale<<<g, b>>>(d, n);\nt1 = now();\ncudaDeviceSynchronize();\nt2 = now();'}
            </pre>
            Kernel chạy lâu hơn nhiều so với thời gian một lần khởi chạy. <code>t1 − t0</code> đo cái gì?
          </>
        }
      >
        <pre className="mono" style={{ margin: '0 0 0.6rem', whiteSpace: 'pre-wrap' }}>
          {'t0 = now();\nscale<<<g, b>>>(d, n);\nt1 = now();\ncudaDeviceSynchronize();\nt2 = now();'}
        </pre>
        The kernel runs for far longer than a launch takes. What does <code>t1 − t0</code> measure?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chỉ thời gian đưa lần khởi chạy vào hàng đợi; kernel có thể còn chưa bắt đầu.">
            Only the time to queue the launch; the kernel may not even have started.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Thời gian chạy của kernel, vì lời gọi chỉ trả về khi khối cuối cùng đã xong.">
            The kernel’s run time, because the call returns once the last block has finished.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là t2 − t0, cộng thêm thời gian xếp hàng. Một lần khởi chạy trả về ngay khi kernel được đưa vào hàng đợi (p. 556).">
            That is t2 − t0, plus the queuing. A launch returns as soon as the kernel is queued (p. 556).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không đo được gì hữu ích: lần khởi chạy chờ đến khi GPU rảnh, nên đó là kernel trước đó.">
            Nothing useful: a launch waits until the GPU is idle, so it is the previous kernel.
          </Translated>
        ),
        why: (
          <Translated vi="Lần khởi chạy không chờ GPU. Công việc xếp hàng trước đó trong cùng stream chạy trước, trên GPU, trong khi host đi tiếp.">
            A launch does not wait for the GPU. Work queued earlier in the same stream runs first, on the GPU, while the host moves on.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={
              <>
                Lần sao chép <code>d</code> lên GPU mà lần khởi chạy thực hiện trước khi kernel bắt đầu.
              </>
            }
          >
            The copy of <code>d</code> to the GPU that the launch makes before the kernel starts.
          </Translated>
        ),
        why: (
          <Translated vi="Lần khởi chạy không sao chép gì. d là con trỏ device; dữ liệu của nó đã được sao chép trước đó, bằng cudaMemcpy.">
            A launch copies nothing. d is a device pointer; its data was copied earlier, with cudaMemcpy.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trả về từ một lần khởi chạy nghĩa là kernel đã được đưa vào hàng đợi của một stream, và không hơn: nó có thể còn
            chưa bắt đầu (p. 556). Host được tự do xếp thêm việc, và phần{' '}
            <a href={href('ch23', '23.3')}>23.3</a> được xây trên điều đó: một lần sao chép hay một kernel khác có thể được xếp
            hàng trong khi kernel này chạy. Để đo thời gian một kernel, hãy chờ nó bên trong vùng được đo, như <code>t2</code>{' '}
            làm, hoặc ghi event quanh nó và đọc khoảng thời gian giữa chúng (trang{' '}
            <a href={href('ch23', '23.4')}>23.3–23.5</a> chỉ dùng event để sắp thứ tự). Đã học ở{' '}
            <a href={href('ch02', '2.6')}>ch. 2 §2.6</a>.
          </p>
        }
      >
        <p>
          Returning from a launch means the kernel has been put in a stream&rsquo;s queue, and nothing more: it may not
          even have started (p. 556). The host is free to queue more work, and section{' '}
          <a href={href('ch23', '23.3')}>23.3</a> is built on that: a copy or another kernel can be queued while this one
          runs. To time a kernel, wait for it inside the timed region, as <code>t2</code> does, or record events around it
          and read the time between them (page <a href={href('ch23', '23.4')}>23.3–23.5</a> uses events only for ordering). Taught in <a href={href('ch02', '2.6')}>ch. 2 §2.6</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 23 §23.3, p. 556; notes/today/ch02-facts.md A5', // i18n-ignore: citation only
  },
  'ch23/start/q-halo-rows': {
    kind: 'numeric',
    id: 'ch23/start/q-halo-rows',
    prompt: (
      <Translated
        vi={
          <>
            Một stencil đọc bốn điểm lân cận của mỗi điểm chạy trên một lưới (grid) 128 hàng, chia theo hàng cho 4 GPU.
            GPU 1 sở hữu các hàng toàn cục 32 đến 63. Mảng cục bộ của nó phải chứa bao nhiêu hàng?
          </>
        }
      >
        A stencil that reads the four neighbours of each point runs on a grid of 128 rows, split by rows over 4 GPUs.
        GPU 1 owns global rows 32 to 63. How many rows must its local array hold?
      </Translated>
    ),
    answer: 34,
    unit: 'rows',
    explain: (
      <Translated
        vi={
          <p>
            32 hàng của chính nó, cộng hàng 31 ở trên (hàng cuối của GPU 0) và hàng 64 ở dưới (hàng đầu của GPU 2): 34. Đó
            là <code>ny</code> của sách cho GPU này, tính cả hai hàng vùng biên (halo). Stencil với lên một hàng và xuống
            một hàng, nên mỗi phía một hàng vùng biên là đủ. Trình khám phá bên dưới bắt đầu với cách chia này. Ô vùng biên
            đã được học ở <a href={href('ch08', 'start')}>chương 8</a>.
          </p>
        }
      >
        <p>
          Its own 32 rows, plus row 31 above (the last row of GPU 0) and row 64 below (the first row of GPU 2): 34. That
          is the book&rsquo;s <code>ny</code> for this GPU, which counts the two halo rows. The stencil reaches one row up
          and one row down, so one halo row on each side is enough. The explorer below starts on this split. Halo cells
          were taught in <a href={href('ch08', 'start')}>chapter 8</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 23 §23.1, p. 544; halo cells: ch. 8, p. 183',
    vi: { unit: 'hàng', source: 'ch. 23 §23.1, p. 544; ô vùng biên (halo): ch. 8, p. 183' },
  },
  'ch23/start/q-blockreduce': {
    kind: 'choice',
    id: 'ch23/start/q-blockreduce',
    prompt: (
      <Translated
        vi={
          <>
            Trong kernel của Fig. 23.1, mọi luồng, kể cả những luồng không qua phép kiểm tra biên, đều gọi{' '}
            <code>blockReduce(residue*residue)</code>, và sau đó chỉ luồng (0,0) của mỗi khối gọi{' '}
            <code>atomicAdd</code>. Vì sao lời gọi <code>blockReduce</code> phải nằm ngoài phép kiểm tra biên?
          </>
        }
      >
        In the kernel of Fig. 23.1 every thread, including those that fail the bounds test, calls{' '}
        <code>blockReduce(residue*residue)</code>, and only thread (0,0) of each block then calls{' '}
        <code>atomicAdd</code>. Why must the <code>blockReduce</code> call sit outside the bounds test?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mọi luồng của khối phải đến được các rào chắn của nó; một luồng không có điểm nào thì góp 0.">
            Its barriers must be reached by every thread of the block; a thread with no point brings 0.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Để mỗi luồng có thể tự atomicAdd vào chuẩn (norm), sau khi phép rút gọn của khối xong.">
            So that each thread can make its own atomicAdd to the norm, once the block’s reduction is done.
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ luồng (0,0) cộng, mỗi khối một lần. Rút gọn trước chính là điều tiết kiệm các phép nguyên tử còn lại.">
            Only thread (0,0) adds, once per block. Reducing first is what saves the other atomics.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Để tránh cái giá của phân kỳ luồng: phép rút gọn chạy nhanh hơn khi mọi luồng đi cùng một nhánh.">
            To save the cost of divergence: a reduction runs faster when every thread takes one path.
          </Translated>
        ),
        why: (
          <Translated vi="Phân kỳ sẽ tốn thời gian. Một rào chắn mà vài luồng bỏ qua còn tệ hơn: kết quả sai hoặc treo (ch. 4, p. 72).">
            Divergence would cost time. A barrier that some threads skip is worse: wrong results or a hang (ch. 4, p. 72).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Vì trình biên dịch từ chối lời gọi hàm device bên trong câu lệnh if.">
            Because the compiler rejects a call to a device function inside an if statement.
          </Translated>
        ),
        why: (
          <Translated vi="Nó vẫn biên dịch được. Giữ cho cả khối đều đến được mọi rào chắn là việc của lập trình viên (ch. 4, p. 72).">
            It compiles. Keeping every barrier reachable by the whole block is the programmer’s job (ch. 4, p. 72).
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách không trình bày <code>blockReduce</code>; nó chỉ sang chương 10 hoặc CUB để lấy một bản (p. 543). Phép
            rút gọn toàn khối của chương 10 đồng bộ hóa các luồng bằng <code>__syncthreads</code>, và mọi luồng của khối
            phải đến được rào chắn (<a href={href('ch04', '4.3')}>ch. 4 §4.3</a>). Một khối ở rìa có những luồng vượt quá
            hàng hoặc cột cuối, nên phép rút gọn nằm ngoài <code>if</code> và những luồng đó góp phần dư (residue) bằng 0.
            Một <code>atomicAdd</code> cho mỗi khối là cách ghi kết quả nhiều khối của chương 10 (
            <a href={href('ch10', '10.9')}>§10.9</a>).
          </p>
        }
      >
        <p>
          The book does not show <code>blockReduce</code>; it points to chapter 10 or to CUB for one (p. 543). A
          block-wide reduction from chapter 10 synchronises its threads with <code>__syncthreads</code>, and a barrier must
          be reached by every thread of the block (<a href={href('ch04', '4.3')}>ch. 4 §4.3</a>). An edge block has
          threads past the last row or column, so the reduction stays outside the <code>if</code> and those threads bring a
          residue of 0. The one <code>atomicAdd</code> per block is chapter 10&rsquo;s multi-block commit (
          <a href={href('ch10', '10.9')}>§10.9</a>).
        </p>
      </Translated>
    ),
    source: 'ch. 23 §23.1, pp. 542–543; ch. 4 §4.3, p. 72; ch. 10 §10.9, pp. 244–245', // i18n-ignore: citation only
  },
  'ch23/start/q-amdahl': {
    kind: 'numeric',
    id: 'ch23/start/q-amdahl',
    prompt: (
      <Translated
        vi={
          <>
            Ví dụ của chương 22: 95% thời gian chạy của một chương trình được tăng tốc 100 lần trên GPU và 5% ở lại trên
            CPU. Nếu phần CPU có thể chạy trong khi GPU làm việc, cả chương trình được tăng tốc bao nhiêu lần?
          </>
        }
      >
        Chapter 22&rsquo;s example: 95% of a program&rsquo;s run time is made 100 times faster on the GPU and 5% stays
        on the CPU. If the CPU part can run while the GPU works, what speed-up does the whole program get?
      </Translated>
    ),
    answer: 20,
    tol: 0.01,
    unit: '×',
    explain: (
      <Translated
        vi={
          <p>
            Khi chồng lấp, lần chạy kéo dài bằng phần dài hơn: max(5%, 0.95%) = 5% thời gian ban đầu, nên 1 / 0.05 = 20×.
            Không chồng lấp thì các phần cộng lại: 1 / (5% + 0.95%) ≈ 16.8×, mà sách in là 17×. Phần{' '}
            <a href={href('ch23', '23.3')}>23.3</a> làm đúng bước này bên trong một bước của stencil: gửi các hàng rìa
            trong khi tính các hàng bên trong, để một bước tốn bằng phần dài hơn trong hai thay vì tổng của chúng. Đã học
            ở <a href={href('ch22', '22.3')}>ch. 22 §22.3</a>.
          </p>
        }
      >
        <p>
          With overlap the run lasts as long as the longer part: max(5%, 0.95%) = 5% of the original, so 1 / 0.05 = 20×.
          Without overlap the parts add: 1 / (5% + 0.95%) ≈ 16.8×, which the book prints as 17×. Section{' '}
          <a href={href('ch23', '23.3')}>23.3</a> makes the same move inside one step of the stencil: send the boundary
          rows while the internal rows are computed, so a step costs the longer of the two instead of their sum. Taught
          in <a href={href('ch22', '22.3')}>ch. 22 §22.3</a>.
        </p>
      </Translated>
    ),
    source: 'ch. 22 §22.3, pp. 537–538; ch. 23 §23.3, p. 554', // i18n-ignore: citation only
  },
  'ch23/start/q-pinned': {
    kind: 'choice',
    id: 'ch23/start/q-pinned',
    prompt: (
      <Translated
        vi={
          <>
            Phụ lục C nói bộ nhớ zero-copy, tức bộ nhớ host mà kernel đọc và ghi trực tiếp, phải được ghim (pinned). Vì sao?
          </>
        }
      >
        Appendix C says zero-copy memory, host memory that a kernel reads and writes directly, must be pinned. Why?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Kernel truy cập nó qua đường kết nối vào bất kỳ lúc nào, nên hệ thống không được di chuyển nó.">
            A kernel reaches it over the interconnect at any moment, so the system must not move it.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Bộ nhớ ghim đọc nhanh hơn, và kernel không thể chịu được những lần đọc chậm từ host.">
            Pinned memory is faster to read, and a kernel cannot afford slow reads from the host.
          </Translated>
        ),
        why: (
          <Translated vi="Tốc độ không phải lý do. Đọc zero-copy luôn chậm dù bạn làm gì, bị giới hạn bởi đường kết nối (p. 625).">
            Speed is not the reason. Zero-copy reads are slow whatever you do, limited by the interconnect (p. 625).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="CPU không thể ghi vào bộ nhớ ghim, nên GPU không bao giờ thấy một giá trị ghi dở.">
            The CPU cannot write pinned memory, so the GPU never sees a half-written value.
          </Translated>
        ),
        why: (
          <Translated vi="Host đọc và ghi nó qua con trỏ riêng của mình; dùng chung nó chính là mục đích (p. 624).">
            The host reads and writes it through its own pointer; sharing it is the point (p. 624).
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Việc ghim là thứ cho buffer một địa chỉ mà kernel có thể dùng.">
            Pinning is what gives the buffer an address that a kernel is able to use.
          </Translated>
        ),
        why: (
          <Translated vi="Địa chỉ đó đến từ một lời gọi riêng, cudaHostGetDevicePointer (p. 624). Việc ghim giữ cho vùng nhớ phía sau nó đứng yên.">
            That address comes from a separate call, cudaHostGetDevicePointer (p. 624). Pinning keeps the memory behind it in place.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hệ điều hành có thể di chuyển một trang bộ nhớ thường, hoặc ghi nó ra đĩa, vào bất cứ lúc nào. Một GPU đang dùng
            vùng nhớ đó đúng lúc ấy sẽ đọc hoặc ghi vào bất cứ thứ gì đang nằm ở đó. Chương 23 đưa ra cùng lý do cho các
            lần sao chép: phần cứng DMA phía sau <code>cudaMemcpy</code> làm việc trên địa chỉ vật lý (pp. 562–563). Phần{' '}
            <a href={href('ch23', '23.3')}>23.3</a> đi tiếp từ đó: vì sao một lần sao chép bất đồng bộ cần buffer host được
            ghim. Đã học ở <a href={href('appC', 'C.2')}>phụ lục C §C.2</a>.
          </p>
        }
      >
        <p>
          The operating system may move a page of ordinary memory, or write it out to disk, at any time. A GPU using the
          memory at that moment would read or write whatever sits there now. Chapter 23 gives the same reason for copies:
          the DMA hardware behind <code>cudaMemcpy</code> works on physical addresses (pp. 562–563). Section{' '}
          <a href={href('ch23', '23.3')}>23.3</a> takes it from there: why an asynchronous copy wants its host buffer
          pinned. Taught in <a href={href('appC', 'C.2')}>appendix C §C.2</a>.
        </p>
      </Translated>
    ),
    source: 'appendix C §C.2, pp. 624–625; ch. 23 §23.3, pp. 562–563',
    vi: { source: 'phụ lục C §C.2, pp. 624–625; ch. 23 §23.3, pp. 562–563' },
  },
  'ch23/start/q-ex1-bytes': {
    kind: 'numeric',
    id: 'ch23/start/q-ex1-bytes',
    prompt: (
      <Translated
        vi={
          <>
            Stencil của chương này chạy trên một lưới rộng 64 điểm và cao 512, chia theo hàng cho 16 GPU, với số float
            4 byte. Mỗi GPU gửi bao nhiêu byte cho các lân cận của nó mỗi bước?
          </>
        }
      >
        This chapter&rsquo;s stencil runs on a grid 64 points wide and 512 tall, split by rows over 16 GPUs, with
        4-byte floats. How many bytes does each GPU send to its neighbours per step?
      </Translated>
    ),
    answer: 512,
    unit: 'bytes',
    explain: (
      <Translated
        vi={
          <p>
            512 / 16 = 32 hàng mỗi GPU. Mỗi bước một GPU gửi hàng trên cùng của nó lên và hàng dưới cùng xuống: 2 × 64 × 4 =
            512 byte, 256 cho mỗi lân cận. Cả số 16 lẫn 512 hàng đều không xuất hiện trong tích đó. Thiết lập sẵn bài tập
            của trình khám phá cho thấy các phần 1a đến 1d, với cả hai cách hiểu của 1a, 1c và 1d: bài tập không nói hai
            cột rìa mà kernel bỏ qua có được tính là đã tính hay không.
          </p>
        }
      >
        <p>
          512 / 16 = 32 rows each. Every step a GPU sends its top row up and its bottom row down: 2 × 64 × 4 = 512 bytes,
          256 to each neighbour. Neither the 16 nor the 512 rows appear in that product. The explorer&rsquo;s exercise
          preset shows parts 1a to 1d, with both readings of 1a, 1c and 1d: the exercise does not say whether the two edge
          columns the kernel skips count as computed.
        </p>
      </Translated>
    ),
    source: 'ch. 23 §23.7, exercise 1, p. 576',
    vi: { unit: 'byte', source: 'ch. 23 §23.7, bài tập 1, p. 576' },
  },
};

// ---------------------------------------------------------------- formatting

const n = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => `${(100 * v).toLocaleString('en-US', { maximumFractionDigits: 1 })}%`;
const range = (a: number, b: number) => (a === b ? `${a}` : tr(`${a} to ${b}`, `${a} đến ${b}`));

// Hatching for the two edge columns the kernel skips, and for stale halo rows. Ink only.
const HATCH = 'repeating-linear-gradient(135deg, var(--rule) 0 1.5px, transparent 1.5px 5px)'; // i18n-ignore: CSS value

// ---------------------------------------------------------------- the domain, one band per GPU

function DomainPicture({ p, onPick }: { p: Partition; onPick: (g: number) => void }) {
  const shares = rowShares(p.rows, p.gpus);
  const firsts = firstRows(shares);
  const X = 58;
  const Y = 26;
  const W = 120;
  const H = 300;
  const y = (r: number) => Y + (r / p.rows) * H;
  // An edge row drawn at its true height would vanish; give it at least 4 px, kept
  // inside its owner's band (above the band's bottom for a last row, below its top for a first).
  const rowStrip = (r: number, last: boolean) => {
    const t = Math.max(4, y(r + 1) - y(r));
    const y0 = last ? y(r + 1) - t : y(r);
    return { y0: Math.max(Y, Math.min(y0, Y + H - t)), h: t };
  };
  const order = shares.map((_, g) => g).sort((a, b) => (a === p.gpu ? 1 : b === p.gpu ? -1 : a - b));
  const key = (g: number) => (e: KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onPick(g);
    }
  };
  const topRow = firsts[p.top] + shares[p.top] - 1;
  const bottomRow = firsts[p.bottom];
  const halo = [rowStrip(topRow, true), rowStrip(bottomRow, false)];
  const edge = [rowStrip(p.first, false), rowStrip(p.last, true)];
  return (
    <svg viewBox="0 0 200 346" width="100%" style={{ maxWidth: 250, display: 'block', margin: '0 auto' }} data-domain>
      <title>
        <Translated vi="Lưới được cắt thành một dải hàng cho mỗi GPU">The grid cut into one band of rows per GPU</Translated>
      </title>
      <text x={X + W / 2} y={15} fontSize="11" textAnchor="middle">
        nx = {n(p.nx)}
      </text>
      <text x={X + W + 10} y={Y + H / 2} fontSize="11" textAnchor="middle" transform={`rotate(90 ${X + W + 10} ${Y + H / 2})`}>
        <Translated vi={<>{n(p.rows)} hàng</>}>{n(p.rows)} rows</Translated>
      </text>
      {order.map((g) => {
        const y0 = y(firsts[g]);
        const h = y(firsts[g] + shares[g]) - y0;
        const sel = g === p.gpu;
        return (
          <g
            key={g}
            data-gpu={g}
            role="button"
            tabIndex={0}
            aria-pressed={sel}
            aria-label={tr(`GPU ${g}: rows ${range(firsts[g], firsts[g] + shares[g] - 1)}`, `GPU ${g}: các hàng ${range(firsts[g], firsts[g] + shares[g] - 1)}`)}
            onClick={() => onPick(g)}
            onKeyDown={key(g)}
            style={{ cursor: 'pointer' }}
          >
            <rect
              x={X}
              y={y0}
              width={W}
              height={h}
              fill={sel ? 'var(--global-wash)' : g % 2 ? 'var(--rule-soft)' : 'var(--cell)'}
              stroke={sel ? 'var(--global)' : 'var(--rule)'}
              strokeWidth={sel ? 1.5 : 0.6}
            />
            {h >= 13 || sel ? (
              <text x={X - 6} y={Math.min(Math.max(y0 + h / 2 + 4, Y + 9), Y + H)} fontSize="10.5" textAnchor="end" fontWeight={sel ? 650 : 400}>
                GPU {g}
              </text>
            ) : null}
          </g>
        );
      })}
      {halo.map((s, i) => (
        <rect key={'h' + i} x={X + 1} y={s.y0} width={W - 2} height={s.h} fill="var(--ink-2)" pointerEvents="none" />// i18n-ignore: SVG attribute
      ))}
      {edge.map((s, i) => (
        <rect key={'e' + i} x={X + 1} y={s.y0} width={W - 2} height={s.h} fill="var(--global)" pointerEvents="none" />// i18n-ignore: SVG attribute
      ))}
      <rect x={X} y={Y} width={W} height={H} fill="none" stroke="var(--ink-3)" strokeWidth={1} pointerEvents="none" />{/* i18n-ignore: SVG attribute */}
    </svg>
  );
}

// ---------------------------------------------------------------- one GPU's local array

function Strip({ kind, stale }: { kind: 'halo' | 'boundary' | 'internal'; stale: boolean }) {
  const body: CSSProperties =
    kind === 'halo'
      ? stale
        ? { background: HATCH, border: '1px dashed var(--ink-3)' } // i18n-ignore: CSS value
        : { background: 'var(--rule-soft)', border: '1px dashed var(--ink-3)' } // i18n-ignore: CSS value
      : kind === 'boundary'
        ? { background: 'var(--global)', opacity: 0.75 }
        : { background: 'var(--global-wash)', border: '1px solid var(--rule)' }; // i18n-ignore: CSS value
  const ends: CSSProperties = { width: '8%', background: HATCH, border: '1px solid var(--rule)' }; // i18n-ignore: CSS value
  return (
    <div style={{ display: 'flex', height: 16, minWidth: 56 }} aria-hidden>
      {kind === 'halo' ? (
        <div style={{ flex: 1, ...body }} />
      ) : (
        <>
          <div style={ends} />
          <div style={{ flex: 1, ...body }} />
          <div style={ends} />
        </>
      )}
    </div>
  );
}

type Phase = 'before' | 'after';

/** What row r of the chosen GPU's array is, in words. */
function rowRole(p: Partition, r: number, phase: Phase): ReactNode {
  const g = (r - 1 + p.first + p.rows) % p.rows; // the global row this local row holds
  const kind = rowKind(p.ny, r);
  if (kind === 'halo') {
    const owner = r === 0 ? p.top : p.bottom;
    const side = r === 0 ? tr('Top', 'Trên') : tr('Bottom', 'Dưới');
    const wraps = r === 0 ? p.first === 0 : p.last === p.rows - 1;
    const note = p.self ? tr(' (itself, by the wrap-around)', ' (chính nó, do vòng quanh)') : wraps ? tr(' (by the wrap-around)', ' (do vòng quanh)') : '';
    return phase === 'before' ? (
      <Translated
        vi={
          <>
            Vùng biên {r === 0 ? 'trên' : 'dưới'}: bản sao của hàng toàn cục {g}, từ GPU {owner}
            {note}
          </>
        }
      >
        {side} halo: a copy of global row {g}, from GPU {owner}
        {note}
      </Translated>
    ) : (
      <Translated
        vi={
          <>
            Vùng biên {r === 0 ? 'trên' : 'dưới'}, <b>cũ</b>: GPU {owner} vừa tính lại hàng {g}
          </>
        }
      >
        {side} halo, <b>stale</b>: GPU {owner} has just computed row {g} again
      </Translated>
    );
  }
  if (kind === 'boundary') {
    const both = r === 1 && r === p.ny - 2;
    const to = both ? tr(`GPUs ${p.top} and ${p.bottom}`, `GPU ${p.top} và ${p.bottom}`) : r === 1 ? tr(`GPU ${p.top}`, `GPU ${p.top}`) : tr(`GPU ${p.bottom}`, `GPU ${p.bottom}`);
    const side = both ? tr('Top and bottom', 'Trên và dưới') : r === 1 ? tr('Top', 'Trên') : tr('Bottom', 'Dưới');
    return phase === 'before' ? (
      <Translated
        vi={
          <>
            Hàng rìa {both ? 'trên và dưới' : r === 1 ? 'trên' : 'dưới'}: hàng toàn cục {g}; sau bước này nó được gửi tới {to}
          </>
        }
      >
        {side} boundary: global row {g}; after the step it goes to {to}
      </Translated>
    ) : (
      <Translated
        vi={
          <>
            Hàng rìa {both ? 'trên và dưới' : r === 1 ? 'trên' : 'dưới'}, <b>mới</b>: phải tới {to} trước bước tiếp theo
          </>
        }
      >
        {side} boundary, <b>new</b>: must reach {to} before the next step
      </Translated>
    );
  }
  return <Translated vi={<>Bên trong: hàng toàn cục {g}</>}>Internal: global row {g}</Translated>;
}

function LaunchCell({ l, r, gap, twice }: { l: Launch; r: number; gap?: { from: number; to: number }; twice: number[] }) {
  if (gap) {
    const w = l.writes.filter((x) => x >= gap.from && x <= gap.to);
    if (w.length === 0) return <td data-launch={l.name} />;
    const t0 = w[0] - l.base - 1;
    const t1 = w[w.length - 1] - l.base - 1;
    return (
      <td data-launch={l.name} data-cell="write">
        <span className="chip global">
          t{t0}–t{t1}
        </span>
      </td>
    );
  }
  if (l.writes.includes(r))
    return (
      <td data-launch={l.name} data-cell="write">
        <span className={'chip ' + (twice.includes(r) ? 'fail' : 'global')}>t{r - l.base - 1}</span>
      </td>
    );
  if (l.reads.includes(r))
    return (
      <td data-launch={l.name} data-cell="read" className="muted">
        <Translated vi="đọc">read</Translated>
      </td>
    );
  return <td data-launch={l.name} />;
}

function LocalArray({ p, phase, showLaunches }: { p: Partition; phase: Phase; showLaunches: boolean }) {
  const ls = launches(p.ny, p.nx);
  const cov = coverage(p.ny);
  return (
    <table className="data" data-local style={{ margin: 0, minWidth: showLaunches ? '27rem' : '17rem' }}>
      <thead>
        <tr>
          <th className="num">
            <Translated vi="hàng">row</Translated>
          </th>
          <th style={{ width: '22%', minWidth: 64 }}>
            <Translated vi={<>Mảng của GPU {p.gpu}</>}>GPU {p.gpu}&rsquo;s array</Translated>
          </th>
          <th>
            <Translated vi="con trỏ, và hàng đó chứa gì">pointer, and what the row holds</Translated>
          </th>
          {showLaunches ? ls.map((l) => <th key={l.name}>{l.name}</th>) : null}
        </tr>
      </thead>
      <tbody>
        {visibleRows(p.ny).map((it) =>
          it.kind === 'gap' ? (
            <tr key="gap" data-gap>
              <td className="num muted">⋮</td>
              <td>
                <div style={{ height: 22, background: 'var(--global-wash)', border: '1px dashed var(--rule)' }} aria-hidden />
              </td>
              <td className="muted" data-role>
                <Translated vi={<>Các hàng {range(it.from, it.to)}: thêm {n(it.to - it.from + 1)} hàng bên trong</>}>
                  Rows {range(it.from, it.to)}: {n(it.to - it.from + 1)} more internal {it.to === it.from ? 'row' : 'rows'}
                </Translated>
              </td>
              {showLaunches ? ls.map((l) => <LaunchCell key={l.name} l={l} r={-1} gap={it} twice={cov.twice} />) : null}
            </tr>
          ) : (
            <tr key={it.row} data-row={it.row} data-kind={rowKind(p.ny, it.row)}>
              <td className="num">{it.row}</td>
              <td>
                <Strip kind={rowKind(p.ny, it.row)} stale={phase === 'after'} />
              </td>
              <td>
                <code style={{ whiteSpace: 'nowrap' }}>{pointerExpr(p.ny, it.row)}</code>
                <div data-role style={{ fontSize: '0.78rem', marginTop: '0.15rem' }}>
                  {rowRole(p, it.row, phase)}
                </div>
              </td>
              {showLaunches ? ls.map((l) => <LaunchCell key={l.name} l={l} r={it.row} twice={cov.twice} />) : null}
            </tr>
          ),
        )}
      </tbody>
    </table>
  );
}

function LaunchTable({ p }: { p: Partition }) {
  const ls = launches(p.ny, p.nx);
  const cov = coverage(p.ny);
  return (
    <>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-launches style={{ marginTop: '0.8rem', minWidth: '34rem' }}>
          <thead>
            <tr>
              <th>
                <Translated vi="lần khởi chạy">launch</Translated>
              </th>
              <th>
                <Translated vi="stream">stream</Translated>
              </th>
              <th>
                <Translated vi="con trỏ nó nhận">pointer it gets</Translated>
              </th>
              <th className="num">
                <Translated vi="phần tử">element</Translated>
              </th>
              <th>
                <Translated vi="số hàng được báo">rows it is told</Translated>
              </th>
              <th>
                <Translated vi="các hàng nó ghi">rows it writes</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {ls.map((l) => (
              <tr key={l.name} data-launch={l.name}>
                <td>{l.name}</td>
                <td>
                  <code style={{ whiteSpace: 'nowrap' }}>{l.stream}</code>
                </td>
                <td>
                  <code style={{ whiteSpace: 'nowrap' }}>{l.pointer}</code>
                </td>
                <td className="num" data-offset={l.offset}>
                  {n(l.offset)}
                </td>
                <td data-rows={l.rows}>
                  {l.rowsExpr === String(l.rows) ? (
                    l.rows
                  ) : (
                    <>
                      <code>{l.rowsExpr}</code> = {l.rows}
                    </>
                  )}
                </td>
                <td data-writes={l.writes.join(',')}>{l.writes.length ? range(l.writes[0], l.writes[l.writes.length - 1]) : tr('none', 'không có')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={cov.exact ? '' : 'narrate bad'} data-coverage={cov.exact ? 'exact' : 'overlap'} style={{ margin: '0 0 0.4rem' }}>
        {cov.exact ? (
          <Translated
            vi={
              <>
                <span className="chip ok">phủ đúng một lần</span> Cùng nhau, ba lần khởi chạy ghi các hàng 1 đến {p.ny - 2}{' '}
                mỗi hàng một lần và không bao giờ chạm vào các hàng vùng biên 0 và {p.ny - 1}. Hàng luồng t của một lần
                khởi chạy được trao <code>output + base*nx</code> sẽ ghi hàng base + t + 1: phần <code>+ 1</code> của
                kernel và điều kiện chặn <code>y &lt; ny - 1</code> lo phần còn lại.
              </>
            }
          >
            <span className="chip ok">covered once</span> Together the three launches write rows 1 to {p.ny - 2} once
            each and never touch the halo rows 0 and {p.ny - 1}. Thread row t of a launch handed{' '}
            <code>output + base*nx</code> writes row base + t + 1: the kernel&rsquo;s <code>+ 1</code> and its guard{' '}
            <code>y &lt; ny - 1</code> do the rest.
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                <span className="chip fail">hàng {cov.twice.join(', ')} hai lần</span> Với một hàng mỗi GPU, ny = 3: lần
                khởi chạy trên và lần khởi chạy dưới đều tính hàng 1, nên phần dư của nó bị cộng vào chuẩn hai lần. Cách
                chia của phần 23.3 cần ít nhất hai hàng mỗi GPU (cách đọc của chúng tôi về điều kiện chặn của kernel).
              </>
            }
          >
            <span className="chip fail">row {cov.twice.join(', ')} twice</span> With one row per GPU, ny = 3: the top
            and the bottom launch both compute row 1, so its residue is added to the norm twice. Section 23.3&rsquo;s
            split needs at least two rows per GPU (our reading of the kernel&rsquo;s guard).
          </Translated>
        )}
      </p>
    </>
  );
}

// ---------------------------------------------------------------- the 2D alternative (ours)

function TilesPicture({ nx, rows, t, gpu, onPick }: { nx: number; rows: number; t: TileSplit; gpu: number; onPick: (g: number) => void }) {
  const ratio = nx / rows;
  const S = 170;
  const bw = ratio >= 1 ? S : Math.max(56, S * ratio);
  const bh = ratio >= 1 ? Math.max(56, S / ratio) : S;
  const X = 10;
  const Y = 10;
  const hs = rowShares(rows, t.p);
  const ws = rowShares(nx, t.q);
  const hy = firstRows(hs).map((r) => Y + (r / rows) * bh);
  const wx = firstRows(ws).map((c) => X + (c / nx) * bw);
  const at = tileOf(gpu, t.q);
  const cells = [];
  for (let i = 0; i < t.p; i++)
    for (let j = 0; j < t.q; j++) {
      const g = i * t.q + j;
      const x0 = wx[j];
      const y0 = hy[i];
      const w = (ws[j] / nx) * bw;
      const h = (hs[i] / rows) * bh;
      cells.push(
        <rect
          key={g}
          data-tile={g}
          x={x0}
          y={y0}
          width={w}
          height={h}
          fill={g === gpu ? 'var(--global-wash)' : 'var(--cell)'}
          stroke="var(--rule)"
          strokeWidth={0.6}
          style={{ cursor: 'pointer' }}
          onClick={() => onPick(g)}
        />,
      );
    }
  const sx = wx[at.col];
  const sy = hy[at.row];
  const sw = (ws[at.col] / nx) * bw;
  const sh = (hs[at.row] / rows) * bh;
  return (
    <svg viewBox={`0 0 ${bw + 2 * X} ${bh + 2 * Y}`} width="100%" style={{ maxWidth: 200, display: 'block', margin: '0 auto' }} data-tilespic>
      <title>
        <Translated vi={<>Lưới được cắt thành {t.p} × {t.q} ô</>}>
          The grid cut into {t.p} by {t.q} tiles
        </Translated>
      </title>
      {cells}
      <rect x={sx} y={sy} width={sw} height={sh} fill="none" stroke="var(--global)" strokeWidth={1.5} pointerEvents="none" />{/* i18n-ignore: SVG attribute */}
      {t.p > 1 ? (
        <>
          <line x1={sx} x2={sx + sw} y1={sy} y2={sy} stroke="var(--ink)" strokeWidth={3} />
          <line x1={sx} x2={sx + sw} y1={sy + sh} y2={sy + sh} stroke="var(--ink)" strokeWidth={3} />
        </>
      ) : null}
      {t.q > 1 ? (
        <>
          <line x1={sx} x2={sx} y1={sy} y2={sy + sh} stroke="var(--ink)" strokeWidth={3} strokeDasharray="2 3" />
          <line x1={sx + sw} x2={sx + sw} y1={sy} y2={sy + sh} stroke="var(--ink)" strokeWidth={3} strokeDasharray="2 3" />
        </>
      ) : null}
      <rect x={X} y={Y} width={bw} height={bh} fill="none" stroke="var(--ink-3)" pointerEvents="none" />{/* i18n-ignore: SVG attribute */}
    </svg>
  );
}

function TilesPanel({ p, pick, setPick, onGpu }: { p: Partition; pick: string | null; setPick: (k: string) => void; onGpu: (g: number) => void }) {
  if (p.gpus === 1)
    return (
      <div className="panel" data-tiles="none" style={{ marginTop: '1rem' }}>
        <Translated vi={<p style={{ margin: 0 }}>Một GPU: không có gì để cắt. Thêm GPU để so sánh chia theo hàng với chia ô.</p>}>
          <p style={{ margin: 0 }}>One GPU: there is nothing to cut. Add GPUs to compare rows with tiles.</p>
        </Translated>
      </div>
    );
  const options = factorisations(p.gpus).filter(({ p: tp, q }) => p.rows >= tp && p.nx >= q);
  const best = bestTiles(p.nx, p.rows, p.gpus);
  const chosen = options.find((o) => `${o.p}x${o.q}` === pick) ?? best;
  const t = tileSplit(p.nx, p.rows, chosen.p, chosen.q);
  const rowsT = tileSplit(p.nx, p.rows, p.gpus, 1);
  const maxShare = Math.max(...rowShares(p.rows, p.gpus));
  const memory = (s: TileSplit) => {
    const parts: string[] = [];
    if (s.rowMessages) parts.push(tr(`${s.rowMessages} row pieces of ${n(s.w)} values, each one contiguous run`, `${s.rowMessages} mảnh hàng gồm ${n(s.w)} giá trị, mỗi mảnh là một dải liền nhau`));
    if (s.columnMessages) parts.push(tr(`${s.columnMessages} columns of ${n(s.h)} single values, ${n(p.nx)} apart in memory`, `${s.columnMessages} cột gồm ${n(s.h)} giá trị lẻ, cách nhau ${n(p.nx)} trong bộ nhớ`));
    return parts.join('; ');
  };
  const saving = 1 - t.bytes / rowsT.bytes;
  return (
    <div className="panel" data-tiles={`${t.p}x${t.q}`} style={{ marginTop: '1rem' }}>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span>
          <Translated
            vi={
              <>
                Xếp {p.gpus} GPU thành <i>P</i> hàng ô × <i>Q</i> cột ô:
              </>
            }
          >
            Arrange the {p.gpus} GPUs as <i>P</i> tile rows × <i>Q</i> tile columns:
          </Translated>
        </span>
        <span className="seg" style={{ flexWrap: 'wrap', rowGap: 4 }}>
          {options.map((o) => {
            const k = `${o.p}x${o.q}`;
            return (
              <button key={k} data-pq={k} aria-pressed={o.p === t.p && o.q === t.q} onClick={() => setPick(k)}>
                {o.p} × {o.q}
              </button>
            );
          })}
        </span>
        <span className="muted" data-best={`${best.p}x${best.q}`}>
          <Translated vi={<>Gửi ít nhất ở đây: {best.p} × {best.q}.</>}>
            Sends least here: {best.p} × {best.q}.
          </Translated>
        </span>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 12rem) minmax(0, 1fr)' }}>
        <div>
          <TilesPicture nx={p.nx} rows={p.rows} t={t} gpu={p.gpu} onPick={onGpu} />
          <Translated
            vi={
              <p className="muted" style={{ fontSize: '0.74rem', margin: '0.4rem 0 0' }}>
                Ô của GPU {p.gpu}. Cạnh liền: các mảnh hàng nó gửi lên và xuống. Cạnh chấm: các cột nó gửi sang trái và
                phải.
              </p>
            }
          >
            <p className="muted" style={{ fontSize: '0.74rem', margin: '0.4rem 0 0' }}>
              GPU {p.gpu}&rsquo;s tile. Solid edges: row pieces it sends up and down. Dotted edges: columns it sends left and
              right.
            </p>
          </Translated>
        </div>
        <table className="data" data-tilecompare style={{ margin: 0 }}>
          <thead>
            <tr>
              <th>
                <Translated vi="mỗi GPU, mỗi bước">per GPU, each step</Translated>
              </th>
              <th>
                <Translated vi={<>theo hàng ({p.gpus} × 1)</>}>rows ({p.gpus} × 1)</Translated>
              </th>
              <th>
                <Translated vi={<>theo ô ({t.p} × {t.q})</>}>tiles ({t.p} × {t.q})</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <Translated vi="mảnh lớn nhất">largest piece</Translated>
              </td>
              <td>
                {n(maxShare)} × {n(p.nx)}
              </td>
              <td data-t="piece">
                {n(t.h)} × {n(t.w)}
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi="số lân cận, một cho mỗi cạnh bị cắt">neighbours, one per cut side</Translated>
              </td>
              <td data-t="rows-neighbours">{rowsT.neighbours}</td>
              <td data-t="tiles-neighbours">{t.neighbours}</td>
            </tr>
            <tr>
              <td>
                <Translated vi="số giá trị gửi đi">values sent</Translated>
              </td>
              <td data-t="rows-points">{n(rowsT.points)}</td>
              <td data-t="tiles-points">{n(t.points)}</td>
            </tr>
            <tr>
              <td>
                <Translated vi="số byte gửi đi">bytes sent</Translated>
              </td>
              <td data-t="rows-bytes">{n(rowsT.bytes)}</td>
              <td data-t="tiles-bytes">{n(t.bytes)}</td>
            </tr>
            <tr>
              <td>
                <Translated vi="cách nó nằm trong bộ nhớ">how it sits in memory</Translated>
              </td>
              <td>{memory(rowsT)}</td>
              <td>{memory(t)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p data-tileverdict style={{ margin: '0.7rem 0 0' }}>
        {t.q === 1 ? (
          <Translated vi={<>Đây chính là cách chia theo hàng.</>}>
            <>This is the row split itself.</>
          </Translated>
        ) : saving > 0 ? (
          <Translated
            vi={
              <>
                {t.p} × {t.q} gửi ít hơn chia theo hàng {pct(saving)}, và trả giá bằng {t.neighbours} lân cận và{' '}
                {t.columnMessages} cột có bước nhảy (stride): một cột phải được gom từ {n(t.h)} chỗ trước khi có thể gửi
                đi, hoặc gửi thành nhiều mảnh nhỏ.
              </>
            }
          >
            {t.p} × {t.q} sends {pct(saving)} less than rows, and pays for it with {t.neighbours} neighbours and{' '}
            {t.columnMessages} strided {t.columnMessages === 1 ? 'column' : 'columns'}: a column has to be gathered from{' '}
            {n(t.h)} places before it can travel, or sent as many small pieces.
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                Chia theo hàng gửi ít hơn trên lưới này: {n(rowsT.points)} giá trị so với {n(t.points)}, tất cả nằm trong
                các hàng liền nhau.
              </>
            }
          >
            Rows send less on this grid: {n(rowsT.points)} values against {n(t.points)}, all of them in contiguous rows.
          </Translated>
        )}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- the explorer

const NX = { min: 8, max: 1024, step: 8 };
const ROWS = { min: 32, max: 1024, step: 8 };
const GPUS = { min: 1, max: 32 };

function HaloBench() {
  const [nx, setNx] = useState(64);
  const [rows, setRows] = useState(128);
  const [gpus, setGpus] = useState(4);
  const [gpu, setGpu] = useState(1);
  const [phase, setPhase] = useState<Phase>('before');
  const [showLaunches, setShowLaunches] = useState(false);
  const [showTiles, setShowTiles] = useState(false);
  const [tilePick, setTilePick] = useState<string | null>(null);

  const g = Math.min(gpu, gpus - 1);
  const p = partition(nx, rows, gpus, g);
  const isExercise = nx === EXERCISE1.nx && rows === EXERCISE1.rows && gpus === EXERCISE1.gpus;
  const slider = (label: ReactNode, value: number, set: (v: number) => void, r: { min: number; max: number; step?: number }, name: string) => (
    <label className="field" style={{ minWidth: '11rem' }}>
      <span>
        {label} <b data-value={name}>{n(value)}</b>
      </span>
      <input type="range" min={r.min} max={r.max} step={r.step ?? 1} value={value} aria-label={name} data-slider={name} onChange={(e) => set(+e.target.value)} />
    </label>
  );

  return (
    <Bench
      title={tr('Rows, halos and launches', 'Hàng, vùng biên và các lần khởi chạy')}
      task={tr(
        'Load the exercise, then drag the GPU count: watch what each GPU sends against what it computes.',
        'Nạp bài tập, rồi kéo số GPU: theo dõi mỗi GPU gửi bao nhiêu so với bao nhiêu nó tính.',
      )}
      foot={
        <Translated
          vi={
            <>
              Các con số theo kernel của Fig. 23.1 và phép trao đổi của Fig. 23.7, với số float 4 byte. Các lân cận vòng
              quanh như trong mã của sách. Phần so sánh chia ô là của khóa học này.
            </>
          }
        >
          <>
            Counts follow the kernel of Fig. 23.1 and the exchange of Fig. 23.7, with 4-byte floats. Neighbours wrap
            around as in the book&rsquo;s code. The tile comparison is ours.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        {slider(tr('grid width nx', 'chiều rộng lưới nx'), nx, setNx, NX, 'nx')}
        {slider(tr('rows in the grid', 'số hàng của lưới'), rows, setRows, ROWS, 'rows')}
        {slider(tr('GPUs', 'Số GPU'), gpus, setGpus, GPUS, 'gpus')}
        <button
          data-preset="exercise1"
          aria-pressed={isExercise}
          onClick={() => {
            setNx(EXERCISE1.nx);
            setRows(EXERCISE1.rows);
            setGpus(EXERCISE1.gpus);
          }}
        >
          <Translated vi="Bài tập 1 (p. 576)">Exercise 1 (p. 576)</Translated>
        </button>
      </div>
      <div className="row" style={{ marginBottom: '1rem' }}>
        <span className="seg" role="group" aria-label={tr('When', 'Khi nào')}>
          <button data-phase="before" aria-pressed={phase === 'before'} onClick={() => setPhase('before')}>
            <Translated vi="trước bước">before the step</Translated>
          </button>
          <button data-phase="after" aria-pressed={phase === 'after'} onClick={() => setPhase('after')}>
            <Translated vi="sau bước">after the step</Translated>
          </button>
        </span>
        <label className="check">
          <input type="checkbox" data-toggle="launches" checked={showLaunches} onChange={(e) => setShowLaunches(e.target.checked)} />
          <Translated vi="ba lần khởi chạy của phần 23.3">the three launches of section 23.3</Translated>
        </label>
        <label className="check">
          <input type="checkbox" data-toggle="tiles" checked={showTiles} onChange={(e) => setShowTiles(e.target.checked)} />
          <Translated vi="chia ô (chia 2D)">tiles (2D split)</Translated>
        </label>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 13rem) minmax(0, 1fr)' }}>
        <div>
          <DomainPicture p={p} onPick={setGpu} />
          <div className="legend" style={{ marginTop: '0.5rem', fontSize: '0.74rem' }}>
            <span>
              <i style={{ background: 'var(--global)' }} />
              <Translated vi={<>Các hàng rìa của GPU {p.gpu}: nó gửi những hàng này</>}>
                GPU {p.gpu}&rsquo;s edge rows: it sends these
              </Translated>
            </span>
            <span>
              <i style={{ background: 'var(--ink-2)' }} />
              <Translated vi="các hàng của lân cận mà nó chép vào làm vùng biên">
                neighbours&rsquo; rows it copies in as halos
              </Translated>
            </span>
          </div>
          <Translated
            vi={
              <p className="muted" style={{ fontSize: '0.74rem', margin: '0.4rem 0 0' }}>
                Bấm vào một dải để chọn GPU.
              </p>
            }
          >
            <p className="muted" style={{ fontSize: '0.74rem', margin: '0.4rem 0 0' }}>
              Click a band to pick a GPU.
            </p>
          </Translated>
        </div>
        <div style={{ minWidth: 0 }}>
          <div style={{ overflowX: 'auto' }}>
            <LocalArray p={p} phase={phase} showLaunches={showLaunches} />
          </div>
          <p className="muted" style={{ fontSize: '0.74rem', margin: '0.4rem 0 0' }}>
            <Translated vi="Hai đầu gạch chéo: cột 0 và cột nx − 1, mà kernel không bao giờ tính.">
              Hatched ends: columns 0 and nx − 1, which the kernel never computes.
            </Translated>
            {showLaunches
              ? tr(
                  ' t k: written by thread row k of that launch; read: in its slice, not written.',
                  ' t k: được ghi bởi hàng luồng k của lần khởi chạy đó; đọc: nằm trong phần của nó, không được ghi.',
                )
              : ''}
          </p>
        </div>
      </div>

      {showLaunches ? <LaunchTable p={p} /> : null}

      <div className="cols plot" style={{ marginTop: '1rem' }}>
        <table className="data" data-readouts style={{ margin: 0 }}>
          <thead>
            <tr>
              <th>
                <Translated vi={<>GPU {p.gpu}, mỗi bước</>}>GPU {p.gpu}, every step</Translated>
              </th>
              <th className="num">
                <Translated vi="tất cả nx cột">all nx columns</Translated>
              </th>
              <th className="num">
                <Translated vi="nx − 2 cột của kernel">the kernel&rsquo;s nx − 2</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <Translated vi={<>các hàng nó sở hữu (toàn cục {range(p.first, p.last)})</>}>
                  rows it owns (global {range(p.first, p.last)})
                </Translated>
              </td>
              <td className="num" data-readout="owned">
                {n(p.owned)}
              </td>
              <td />
            </tr>
            <tr>
              <td>
                <Translated
                  vi={
                    <>
                      <code>ny</code>: các hàng đó cộng hai hàng vùng biên
                    </>
                  }
                >
                  <code>ny</code>: those rows plus two halo rows
                </Translated>
              </td>
              <td className="num" data-readout="ny">
                {n(p.ny)}
              </td>
              <td />
            </tr>
            <tr>
              <td>
                <Translated vi="số điểm nó tính">points it computes</Translated>
              </td>
              <td className="num" data-readout="output-all">
                {n(p.output.all)}
              </td>
              <td className="num" data-readout="output-kernel">
                {n(p.output.kernel)}
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi="số điểm vùng biên nó nhận (cả hàng)">halo points it receives (whole rows)</Translated>
              </td>
              <td className="num" data-readout="halo">
                {n(p.halo)}
              </td>
              <td />
            </tr>
            <tr>
              <td>
                <Translated vi="số điểm rìa (hàng 1 và ny − 2)">boundary points (rows 1 and ny − 2)</Translated>
              </td>
              <td className="num" data-readout="boundary-all">
                {n(p.boundary.all)}
              </td>
              <td className="num" data-readout="boundary-kernel">
                {n(p.boundary.kernel)}
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi="số điểm bên trong (hàng 2 đến ny − 3)">internal points (rows 2 to ny − 3)</Translated>
              </td>
              <td className="num" data-readout="internal-all">
                {n(p.internal.all)}
              </td>
              <td className="num" data-readout="internal-kernel">
                {n(p.internal.kernel)}
              </td>
            </tr>
            <tr>
              <td>
                <Translated vi="số byte nó gửi">bytes it sends</Translated>
              </td>
              <td className="num" data-readout="bytes">
                {n(p.bytesSent)}
              </td>
              <td className="muted">{p.self ? tr('to itself', 'cho chính nó') : tr(`${n(p.bytesPerNeighbour)} to each neighbour`, `${n(p.bytesPerNeighbour)} cho mỗi lân cận`)}</td>
            </tr>
          </tbody>
        </table>
        <div className="narrate" data-share={p.exchangeShare}>
          {p.self ? (
            <Translated
              vi={
                <p style={{ margin: 0 }}>
                  Một GPU. Việc vòng quanh của sách khiến GPU 0 là lân cận trên và dưới của chính nó, nên hai lần trao đổi
                  chép các hàng rìa của chính nó vào các hàng vùng biên của chính nó và không có gì đi qua đường kết nối.
                </p>
              }
            >
              <p style={{ margin: 0 }}>
                One GPU. The book&rsquo;s wrap-around makes GPU 0 its own top and bottom neighbour, so its two exchanges
                copy its own edge rows into its own halo rows and nothing crosses a link.
              </p>
            </Translated>
          ) : (
            <Translated
              vi={
                <p style={{ margin: 0 }}>
                  GPU {p.gpu} tính {n(p.owned)} hàng và gửi 2 hàng mỗi bước: phép trao đổi bằng{' '}
                  <b>{pct(p.exchangeShare)}</b> số hàng nó tính. Tăng gấp đôi số GPU thì số hàng của nó giảm một nửa,
                  nhưng nó vẫn gửi một hàng lên và một hàng xuống.
                </p>
              }
            >
              <p style={{ margin: 0 }}>
                GPU {p.gpu} computes {n(p.owned)} {p.owned === 1 ? 'row' : 'rows'} and sends 2 every step: the exchange is{' '}
                <b>{pct(p.exchangeShare)}</b> of the rows it computes. Double the GPUs and its rows halve, but it still
                sends one row up and one down.
              </p>
            </Translated>
          )}
        </div>
      </div>

      {showTiles ? <TilesPanel p={p} pick={tilePick} setPick={setTilePick} onGpu={setGpu} /> : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the page

export function S230() {
  const q = questionsS230;
  const at16 = partition(EXERCISE1.nx, EXERCISE1.rows, 16, 0);
  const at32 = partition(EXERCISE1.nx, EXERCISE1.rows, 32, 0);
  return (
    <>
      <Translated
        vi={
          <p>
            Chương 23 đưa một stencil từ một GPU lên một cụm GPU, và dùng nó để giới thiệu ba cách chuyển dữ liệu giữa
            chúng: MPI, NCCL và NVSHMEM. Khóa học này không giữ lại những thư viện cụm đó. Thứ nó giữ là bộ máy mà chương
            cần trên đường đi, thứ bạn sẽ dùng trên một GPU:
          </p>
        }
      >
        <p>
          Chapter 23 takes one stencil from a single GPU to a cluster of GPUs, and uses it to introduce three ways of
          moving data between them: MPI, NCCL and NVSHMEM. Those cluster libraries are not what this course keeps. What it
          keeps is the machinery the chapter needs on the way, which you will use on one GPU:
        </p>
      </Translated>
      <Source pages="541" />
      <Translated
        vi={
          <ul>
            <li>
              một lần khởi chạy hay một lần sao chép bất đồng bộ được <b>xếp hàng, chưa xong</b>: khi lời gọi trả về, công việc
              có thể còn chưa bắt đầu;
            </li>
            <li>
              <b>stream</b>, những hàng đợi công việc GPU có thứ tự, có thể chạy song song với nhau;
            </li>
            <li>
              <b>event</b>, cho phép một hàng đợi chờ một điểm trong hàng đợi khác mà không chặn host;
            </li>
            <li>
              <b>lần sao chép bất đồng bộ</b> và bộ nhớ host <b>được ghim</b> (pinned memory) mà chúng phụ thuộc vào;
            </li>
            <li>
              và khuôn mẫu gắn kết tất cả: tính trước thứ người khác đang chờ, rồi chồng lấp việc truyền nó với phần còn lại
              của công việc.
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            a launch or an asynchronous copy is <b>queued, not done</b>: when the call returns, the work may not even have
            started;
          </li>
          <li>
            <b>streams</b>, ordered queues of GPU work that may run alongside each other;
          </li>
          <li>
            <b>events</b>, which let one queue wait for a point in another without stopping the host;
          </li>
          <li>
            <b>asynchronous copies</b> and the <b>pinned</b> host memory they depend on;
          </li>
          <li>
            and the pattern that ties them together: compute first what someone else is waiting for, then overlap its
            transfer with the rest of the work.
          </li>
        </ul>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Chương này không nêu tên GPU nào và phiên bản thư viện nào. Mọi con số trên trang này là số hàng, số điểm
              hoặc số byte của một lưới bạn chọn, nên không con số nào phụ thuộc vào thiết bị.
            </p>
          }
        >
          <p>
            The chapter names no GPU and no library version. Every number on this page is rows, points or bytes of a grid
            you choose, so none of it depends on a device.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Đó là phần 23.3 của sách, ở đây được chia thành hai trang. Trang <a href={href('ch23', '23.3')}>23.3</a>{' '}
            bao quát nửa đầu (chồng lấp, stream, lần sao chép bất đồng bộ, bộ nhớ ghim, pp. 553–556 và 560–563); trang{' '}
            <a href={href('ch23', '23.4')}>23.3–23.5</a> bao quát nửa sau (ba lần khởi chạy, độ ưu tiên, event, pp.
            556–562). MPI, NCCL và NVSHMEM, các phần 23.2, 23.4 và 23.5 của sách, mỗi thứ có một đoạn trên trang đó và một
            bảng duy nhất so sánh nơi vòng lặp của chúng phải chờ, kèm số trang trong sách cho khi bạn cần. Các trình khám
            phá chỉ mô hình hóa hai vòng lặp MPI, và không câu hỏi nào về NCCL hay NVSHMEM vào phần ôn tập của bạn. Chương
            này không có mốc dự án. Lab duy nhất của nó,{' '}
            <a href={href('ch23', 'labs')}>Giấu các lần sao chép sau kernel</a>, đo mức chồng lấp trên GPU của chính bạn.
          </p>
        }
      >
        <p>
          That is the book&rsquo;s section 23.3, split over two pages here. Page <a href={href('ch23', '23.3')}>23.3</a>{' '}
          covers its first half (overlap, streams, asynchronous copies, pinned memory, pp. 553–556 and 560–563); page{' '}
          <a href={href('ch23', '23.4')}>23.3–23.5</a> covers its second half (the three launches, priorities, events, pp.
          556–562). MPI, NCCL and NVSHMEM, the book&rsquo;s sections 23.2, 23.4 and 23.5, get one paragraph each on that
          page and a single table that compares where their loops wait, with their pages in the book for when you need them.
          The explorers model only the two MPI loops, and no question on NCCL or NVSHMEM enters your review. The chapter has
          no project milestone. Its one lab,{' '}
          <a href={href('ch23', 'labs')}>Hide the copies behind the kernel</a>, measures the overlap on your own GPU.
        </p>
      </Translated>
      <Source section="23.3" pages="553–563" />

      <h2>
        <Translated vi="Ví dụ xuyên suốt">The running example</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ví dụ của sách là phương pháp Jacobi trên một lưới 2D, một stencil giống những stencil ở chương 8. Mỗi bước
            tính giá trị mới của mọi điểm bên trong bằng trung bình các giá trị cũ của bốn điểm lân cận, đọc một mảng và
            ghi một mảng khác. Các bước lặp lại cho đến khi giá trị không còn thay đổi nhiều. Để quyết định khi nào dừng,
            kernel cũng đo mỗi điểm đã dịch chuyển bao xa, tức phần dư (residue) của nó (mới trừ cũ). Bình phương các phần
            dư được cộng lại trong mỗi khối, và mỗi khối có một luồng cộng tổng của khối vào một tổng duy nhất bằng{' '}
            <code>atomicAdd</code>: cách ghi kết quả nhiều khối của chương 10. Tổng đó là bình phương của chuẩn L2; host
            lấy căn bậc hai.
          </p>
        }
      >
        <p>
          The book&rsquo;s example is the Jacobi method on a 2D grid, a stencil like those of chapter 8. Each step computes
          every interior point&rsquo;s new value as the average of its four neighbours&rsquo; old values, reading one array
          and writing another. Steps repeat until the values stop changing much. To decide when, the kernel also measures
          how far each point moved, its residue (new minus old). The squared residues are summed inside each block, and one
          thread per block adds the block&rsquo;s sum to a single total with <code>atomicAdd</code>: chapter 10&rsquo;s
          multi-block commit. The total is the square of the L2 norm; the host takes the root.
        </p>
      </Translated>
      <Source section="23.1" pages="542–543" figs="Fig. 23.1" />
      <Translated
        vi={
          <p>
            Lưới được lưu theo thứ tự hàng (row-major) với x thay đổi nhanh nhất: toàn bộ hàng 0, rồi toàn bộ hàng 1, cứ
            thế. Cắt nó theo y thì mỗi GPU nhận một dải các hàng nguyên, liền nhau trong bộ nhớ. Để cập nhật hàng đầu và
            hàng cuối của mình, một GPU cần hàng ngay trên phần của nó và hàng ngay dưới, và những hàng đó thuộc về các
            lân cận. Vì vậy mảng của mỗi GPU mang thêm hai hàng, các <b>hàng vùng biên (halo)</b>, mỗi hàng là bản sao một
            hàng rìa của lân cận. <code>ny</code> của sách là chiều cao của các mảng đó tính cả các hàng vùng biên.
          </p>
        }
      >
        <p>
          The grid is stored row-major with x fastest: all of row 0, then all of row 1, and so on. Cut it along y and each
          GPU gets a run of whole rows, contiguous in memory. To update its first and last rows a GPU needs the row just
          above its share and the row just below, and those belong to its neighbours. So each GPU&rsquo;s arrays carry two
          extra rows, the <b>halo rows</b>, each a copy of a neighbour&rsquo;s edge row. The book&rsquo;s{' '}
          <code>ny</code> is the height of those arrays with the halo rows included.
        </p>
      </Translated>
      <Source section="23.1" pages="543–544" figs="Figs. 23.2, 23.3" />
      <Translated
        vi={
          <p>
            Một GPU đọc các hàng vùng biên của nó và không bao giờ ghi chúng. Sau một bước, các lân cận đã có giá trị mới
            cho những hàng đó, và các bản sao đã cũ. Trước bước tiếp theo, mỗi GPU gửi hàng trên cùng và dưới cùng của chính
            nó (các <b>hàng rìa</b>) cho các lân cận và nhận về các hàng vùng biên mới: đó là <b>phép trao đổi vùng biên</b>.
            Vì vậy mỗi bước có hai pha, tính rồi trao đổi, và pha trao đổi cũng cộng chuẩn của các GPU thành một.
          </p>
        }
      >
        <p>
          A GPU reads its halo rows and never writes them. After one step its neighbours have new values for those rows,
          and the copies are stale. Before the next step every GPU sends its own top and bottom rows (its{' '}
          <b>boundary rows</b>) to its neighbours and receives fresh halo rows in return: the <b>halo exchange</b>. So
          every step has two phases, compute and then communicate, and the communicate phase also sums the GPUs&rsquo;
          norms into one.
        </p>
      </Translated>
      <Source section="23.1" pages="544–545" />
      <Ask q={q['ch23/start/q-halo-rows']} label={tr('From chapter 8', 'Từ chương 8')} />

      <h2>
        <Translated vi="Phần lưới của một GPU">One GPU&rsquo;s share of the grid</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trình khám phá bên dưới là Figs. 23.2 và 23.3 của sách kèm con số. Chọn một lưới, một số GPU và một GPU, rồi
            so sánh hai đại lượng: số hàng nó tính và số hàng nó gửi. Một nút bật xem trước phần 23.3, nơi cùng một kernel
            được khởi chạy ba lần trên các phần khác nhau của mảng; một nút khác so sánh cách chia theo hàng của sách với
            cách chia thành ô.
          </p>
        }
      >
        <p>
          The explorer below is the book&rsquo;s Figs. 23.2 and 23.3 with numbers attached. Pick a grid, a GPU count and
          one GPU, and compare two quantities: the rows it computes and the rows it sends. One toggle previews section
          23.3, where the same kernel is launched three times on different slices of the array; another compares the
          book&rsquo;s split by rows with a split into tiles.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một lưới thứ tự hàng gồm số float 4 byte, cắt thành các hàng nguyên; khi chia không đều, GPU cuối nhận phần
              dư (các ví dụ của sách chia đều). Các lân cận vòng quanh như trong mã của sách (p. 551), dù hình vẽ của sách
              vẽ GPU 0 không có lân cận phía trên. Chỉ có điểm và byte, không có thời gian. Phần so sánh chia ô là của khóa
              học này và cho mỗi ô một lân cận ở mỗi cạnh bị cắt.
            </p>
          }
        >
          <p>
            One row-major grid of 4-byte floats, cut into whole rows; when they do not divide evenly the last GPU takes the
            remainder (the book&rsquo;s examples divide evenly). Neighbours wrap around as in the book&rsquo;s code (p.
            551), although its figures draw GPU 0 with no neighbour above. Points and bytes only, no time. The tile
            comparison is ours and gives every tile a neighbour on each side that is cut.
          </p>
        </Translated>
      </Simplification>
      <Ask q={q['ch23/start/q-ex1-bytes']} label={tr('Book exercise 1e', 'Bài tập 1e của sách')} />
      <Predict
        id="ch23/start/p-32gpus"
        prompt={
          <Translated
            vi={
              <>
                Lưới của bài tập, rộng 64 và cao 512, giờ chạy trên 32 GPU thay vì 16. Mỗi GPU gửi bao nhiêu byte mỗi bước?
              </>
            }
          >
            <>
              The exercise&rsquo;s grid, 64 wide and 512 tall, now runs on 32 GPUs instead of 16. How many bytes does each
              GPU send per step?
            </>
          </Translated>
        }
        unit={tr('bytes', 'byte')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                <b>{n(at32.bytesSent)} byte</b>, bằng với trên 16 GPU. Mỗi GPU vẫn gửi một hàng {EXERCISE1.nx} số float
                lên và một hàng xuống. Thứ giảm một nửa là khối lượng việc của nó, từ {at16.owned} hàng xuống {at32.owned},
                nên phép trao đổi tăng từ {pct(at16.exchangeShare)} lên {pct(at32.exchangeShare)} số hàng nó tính.{' '}
                {v === at32.bytesSent
                  ? 'Đúng như bạn đã đoán.'
                  : v < at32.bytesSent
                    ? 'Giảm một nửa là phỏng đoán tự nhiên, và là điều cần bỏ: số byte đi theo chiều rộng của một hàng, không theo số hàng.'
                    : 'Thêm GPU không làm tăng số thông điệp mỗi GPU khi chia theo hàng: mỗi GPU vẫn có đúng hai lân cận.'}
              </p>
            }
          >
            <p>
              <b>{n(at32.bytesSent)} bytes</b>, the same as on 16 GPUs. Each GPU still sends one row of {EXERCISE1.nx}{' '}
              floats up and one down. What halved is its work, from {at16.owned} rows to {at32.owned}, so the exchange went
              from {pct(at16.exchangeShare)} to {pct(at32.exchangeShare)} of the rows it computes.{' '}
              {v === at32.bytesSent
                ? 'That is what you had.'
                : v < at32.bytesSent
                  ? 'Halving is the natural guess, and the one to unlearn: the bytes follow the width of a row, not the number of rows.'
                  : 'More GPUs do not add messages per GPU in a row split: each still has exactly two neighbours.'}
            </p>
          </Translated>
        )}
      </Predict>
      <HaloBench />
      <Translated
        vi={
          <p>
            Ba điều cần tìm. Thứ nhất, nạp bài tập và kéo số GPU: số byte mỗi GPU gửi giữ nguyên trong khi số hàng nó tính
            co lại, nên giao tiếp chiếm phần ngày càng lớn của mỗi bước. Thứ hai, bật ba lần khởi chạy: mỗi lần nhận một
            con trỏ vào cùng một mảng và một số hàng, điều kiện chặn của kernel biến điều đó thành các hàng nó ghi, và
            cùng nhau chúng ghi các hàng 1 đến <code>ny − 2</code> đúng một lần. Rồi kéo số hàng xuống cho đến khi mỗi GPU
            chỉ sở hữu một hàng, và xem cách chia đó hỏng. Thứ ba, làm lưới vuông, hoặc rộng, rồi bật chia ô: chia 2D gửi ít hơn,
            nhưng mỗi GPU nói chuyện với nhiều lân cận hơn và một phần thứ nó gửi là các cột, mỗi hàng một giá trị, rải
            rác trong bộ nhớ.
          </p>
        }
      >
        <p>
          Three things to find. First, load the exercise and drag the GPU count: the bytes each GPU sends stay put while
          the rows it computes shrink, so communication takes a growing share of every step. Second, turn on the three
          launches: each gets a pointer into the same array and a row count, the kernel&rsquo;s guard turns that into
          the rows it writes, and together they write rows 1 to <code>ny − 2</code> exactly once. Then drag the rows
          down until each GPU owns a single row, and watch that break. Third, make the grid square, or wide, and turn on
          tiles: a 2D split sends less, but each GPU talks to more neighbours and some of what it sends are columns, one
          value per row, scattered through memory.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Các con số chia ô là của khóa học này. Sách nêu sự đánh đổi trong một đoạn: ít dữ liệu hơn, bốn lân cận thay
              vì hai, dữ liệu không liền nhau (p. 545). Việc đếm trùng khi mỗi GPU có một hàng cũng là của khóa học này,
              đọc ra từ điều kiện chặn của kernel.
            </p>
          }
        >
          <p>
            The tile numbers are ours. The book states the trade-off in one paragraph: less data, four neighbours instead
            of two, data that is not contiguous (p. 545). The double count at one row per GPU is also ours, read off the
            kernel&rsquo;s guard.
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Vòng lặp đầu tiên của sách, bằng lời">The book&rsquo;s first loop, in words</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Phần 23.2 xây phiên bản nhiều GPU bằng MPI, mỗi GPU một tiến trình. Để các lời gọi MPI cho trang{' '}
            <a href={href('ch23', '23.4')}>23.3–23.5</a>; ở đây hình dạng của vòng lặp mới là điều quan trọng. Mỗi vòng
            lặp, mỗi GPU:
          </p>
        }
      >
        <p>
          Section 23.2 builds the multi-GPU version with MPI, one process per GPU. Leave the MPI calls for page{' '}
          <a href={href('ch23', '23.4')}>23.3–23.5</a>; the shape of the loop is what matters here. Every iteration, each GPU:
        </p>
      </Translated>
      <Translated
        vi={
          <ol>
            <li>đặt chuẩn trên device về 0;</li>
            <li>
              chạy kernel Jacobi trên <code>ny</code> hàng của nó;
            </li>
            <li>chờ kernel chạy xong;</li>
            <li>
              hoán đổi các hàng rìa với hai lân cận: nó gửi hàng 1 lên trong khi nhận hàng <code>ny − 1</code> từ dưới,
              rồi gửi hàng <code>ny − 2</code> xuống trong khi nhận hàng 0 từ trên;
            </li>
            <li>chép chuẩn của nó về host;</li>
            <li>cộng chuẩn của tất cả GPU và lấy căn bậc hai, để quyết định có tiếp tục hay không;</li>
            <li>hoán đổi con trỏ đầu vào và đầu ra, để đầu ra của bước này là đầu vào của bước sau.</li>
          </ol>
        }
      >
        <ol>
          <li>zeroes the norm on the device;</li>
          <li>
            runs the Jacobi kernel over its <code>ny</code> rows;
          </li>
          <li>waits for the kernel to finish;</li>
          <li>
            swaps edge rows with its two neighbours: it sends row 1 up while receiving row <code>ny − 1</code> from below,
            then sends row <code>ny − 2</code> down while receiving row 0 from above;
          </li>
          <li>copies its norm to the host;</li>
          <li>sums the norms of all GPUs and takes the square root, to decide whether to go on;</li>
          <li>swaps its input and output pointers, so this step&rsquo;s output is the next step&rsquo;s input.</li>
        </ol>
      </Translated>
      <Source section="23.2" pages="548–549" figs="Fig. 23.7" />
      <Translated
        vi={
          <p>
            Bước 3 tách bước 2 và bước 4: phép trao đổi không thể bắt đầu cho đến khi cả kernel xong. Vì vậy trong khi
            các GPU tính, đường kết nối giữa chúng không mang gì, và trong khi các hàng đang truyền, các GPU không có việc
            gì làm. Phần <a href={href('ch23', '23.3')}>23.3</a> bắt đầu từ ba lần khởi chạy của trình khám phá: tính hai
            hàng rìa trước, bắt đầu gửi chúng, và tính các hàng bên trong trong khi chúng đang truyền. Điều đó cần những
            lần khởi chạy không chờ nhau và những lần sao chép không chặn host, chính là thứ stream, event và bộ nhớ ghim cung
            cấp.
          </p>
        }
      >
        <p>
          Step 3 keeps steps 2 and 4 apart: the exchange cannot start until the whole kernel is done. So while the GPUs
          compute, the links between them carry nothing, and while the rows travel, the GPUs have nothing to do. Section{' '}
          <a href={href('ch23', '23.3')}>23.3</a> starts from the explorer&rsquo;s three launches: compute the two boundary
          rows first, start sending them, and compute the internal rows while they travel. That needs launches that do not
          wait for each other and copies that do not stop the host, which is what streams, events and pinned memory
          provide.
        </p>
      </Translated>
      <Source section="23.3" pages="553–555" figs="Figs. 23.12, 23.13" />

      <h2>
        <Translated vi="Trước khi bắt đầu">Before you start</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bốn câu hỏi về những gì chương này dựa vào từ các chương 2, 4, 10 và 22 và phụ lục C. Mỗi lời giải thích chỉ
            ra nơi ý tưởng đó được dạy và trang ở đây dùng nó. Chúng sẽ quay lại trong phần{' '}
            <a href={href('review')}>ôn tập</a>.
          </p>
        }
      >
        <p>
          Four questions on what this chapter leans on from chapters 2, 4, 10 and 22 and appendix C. Each explanation
          points to where the idea was taught and to the page here that uses it. They come back in{' '}
          <a href={href('review')}>review</a>.
        </p>
      </Translated>
      <Ask q={q['ch23/start/q-queued']} label={tr('From chapter 2', 'Từ chương 2')} />
      <Ask q={q['ch23/start/q-blockreduce']} label={tr('From chapters 4 and 10', 'Từ chương 4 và 10')} />
      <Ask q={q['ch23/start/q-amdahl']} label={tr('From chapter 22', 'Từ chương 22')} />
      <Ask q={q['ch23/start/q-pinned']} label={tr('From appendix C', 'Từ phụ lục C')} />
    </>
  );
}
