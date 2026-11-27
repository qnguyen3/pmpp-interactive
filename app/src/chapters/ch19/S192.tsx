import { useMemo, useState, type MouseEvent, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import type { Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { FIG_19_4_LOOPS, directRatioMaps, gradeLoops, lenetLayer, lessonRegion, type LoopRole, type LoopVar } from '../../sim/cnnlayer';
import { SEGMENT_MODELS, type SegmentModelId } from '../../sim/coalesce';
import {
  CUDA_GRID,
  HIP_GRID,
  LAYOUTS,
  coverage,
  geometry,
  limits,
  mapThread,
  warpReads,
  writersOf,
  type LaneMap,
  type LaunchParams,
  type Layout,
  type ThreadPos,
} from '../../sim/convgrid';
import { DirectReadsScene } from './DirectReadsScene';

// Section 19.2: the direct kernel. Which loops parallelise, the 3D grid of Figs. 19.5 and
// 19.6 and the index recovery of Fig. 19.7 (with the book's integer division, a bounds check
// and a planted bug), the corrected kernel from lab 19.1, what it reads (a narrated
// animation), Exercise 2's layouts at the level of one warp, and the HIP and Metal side.

const LAB = 'ch19/lab1_convgemm';
const SOL = LAB + '/solution';
const big = (v: number) => v.toLocaleString('en-US');
/** A documented limit, written as 2^k − 1 when it is one and too long to read as digits. */
const limitText = (v: number) => {
  const k = Math.log2(v + 1);
  return Number.isInteger(k) && k > 16 ? <>2<sup>{k}</sup> − 1</> : big(v);
};
const CONV_TIERS: Record<string, Tier> = {
  X: 'global', F: 'global', Y: 'global',
  acc: 'reg', n: 'reg', m: 'reg', h: 'reg', w: 'reg', c: 'reg', p: 'reg', q: 'reg',
};

// ---------------------------------------------------------------- questions

export const questionsS192: Record<string, Question> = {
  'ch19/19.2/q-loops': {
    kind: 'choice',
    id: 'ch19/19.2/q-loops',
    prompt: (
      <Translated
        vi={
          <>
            Bạn tách vòng lặp c trong Hình 19.4 cho lớp C3 của LeNet ra sáu bản đồ đầu vào của nó: sáu luồng, mỗi luồng
            cộng tổng riêng phần của mình vào cùng một đầu ra bằng một lệnh <code>Y[i] += part</code> thông thường, và
            Y[i] bắt đầu từ 0. Khi cả sáu luồng xong, Y[i] có thể chứa giá trị nào?
          </>
        }
      >
        You split Fig. 19.4&rsquo;s c loop for LeNet&rsquo;s C3 across its six input maps: six threads each add their
        own partial sum into the same output with a plain <code>Y[i] += part</code>, and Y[i] starts at 0. What can
        Y[i] hold when all six are done?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Tổng đầy đủ, hoặc tổng của chỉ một số trong sáu phần.">
            The full sum, or the sum of only some of the six parts.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Tổng đầy đủ; chỉ thứ tự của sáu phép cộng là thay đổi giữa các lần chạy.">
            The full sum; just the order of the six additions varies from run to run.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là điều một phép cộng nguyên tử (atomic) bảo đảm. Một lệnh += thông thường gồm một lần nạp, một phép cộng và một lần ghi, và hai luồng có thể nạp cùng một giá trị cũ.">
            That is what an atomic add guarantees. A plain += is a load, an add and a store, and two threads can load the
            same old value.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Tổng đầy đủ, hoặc 0.">The full sum, or 0.</Translated>,
        why: (
          <Translated vi="Luồng nào ghi sau cùng sẽ ghi phần của chính nó cộng với giá trị nó đã nạp, nên luôn có ít nhất một phần được giữ lại.">
            Whichever thread stores last stores its own part plus the value it loaded, so at least one part always
            survives.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tổng đầy đủ, miễn là sáu luồng nằm trong cùng một warp.">
            The full sum, as long as the six threads are in one warp.
          </Translated>
        ),
        why: (
          <Translated vi="Nằm trong cùng một warp không sắp thứ tự gì ở đây: cả sáu lane đều có thể nạp giá trị cũ trước khi bất kỳ lane nào ghi.">
            Being in one warp orders nothing here: all six lanes can load the old value before any of them stores.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi lệnh <code>+=</code> gồm ba bước: nạp, cộng, ghi. Nếu hai luồng cùng nạp 0 rồi cùng ghi, lần ghi sau sẽ xóa
            phần của luồng ghi trước. Theo cách này, bao nhiêu phần cũng có thể bị mất, nhưng không bao giờ mất phần của
            luồng ghi cuối. Đó là lý do sách giữ c, p và q bên trong mỗi luồng: tách chúng ra cần đến phép toán nguyên tử
            (p. 460).
          </p>
        }
      >
        <p>
          Each <code>+=</code> is three steps: load, add, store. If two threads both load 0 and both store, the later store
          erases the earlier thread&rsquo;s part. Any number of parts can be lost this way, but never the last
          writer&rsquo;s. That is why the book keeps c, p and q inside each thread: splitting them needs atomic operations
          (p. 460).
        </p>
      </Translated>
    ),
    source: 'ch. 19 §19.2, p. 460; the lost update is our illustration',
    vi: { source: 'ch. 19 §19.2, p. 460; lần cập nhật bị mất là minh họa của khóa học này' },
  },
  'ch19/19.2/q-mapping': {
    kind: 'choice',
    id: 'ch19/19.2/q-mapping',
    prompt: (
      <Translated
        vi={
          <>
            N = 2, M = 4, bản đồ đầu ra 32 × 32, <code>TILE_WIDTH</code> 16. Luồng <code>(x 5, y 9)</code> của khối{' '}
            <code>(3, 2, 1)</code> tính (n, m, h, w) nào?
          </>
        }
      >
        N = 2, M = 4, 32 × 32 output maps, <code>TILE_WIDTH</code> 16. Thread <code>(x 5, y 9)</code> of block{' '}
        <code>(3, 2, 1)</code> computes which (n, m, h, w)?
      </Translated>
    ),
    options: [
      { label: '(1, 3, 25, 5)', correct: true },
      {
        label: '(1, 3, 9, 21)',
        why: (
          <Translated vi="Đó là / và % bị đổi chỗ: hàng ô 2 % 2 = 0, cột ô 2 / 2 = 1.">
            That is / and % swapped: tile row 2 % 2 = 0, tile column 2 / 2 = 1.
          </Translated>
        ),
      },
      {
        label: '(1, 2, 25, 5)',
        why: (
          <Translated vi="m lấy từ blockIdx.x, không phải blockIdx.y; y là ô.">
            m comes from blockIdx.x, not blockIdx.y; y is the tile.
          </Translated>
        ),
      },
      {
        label: '(3, 2, 25, 5)',
        why: (
          <Translated vi="n lấy từ blockIdx.z, tức mẫu; x là bản đồ đầu ra.">
            n comes from blockIdx.z, the sample; x is the output map.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            W_grid = 32 / 16 = 2. m = blockIdx.x = 3 và n = blockIdx.z = 1. Hàng ô là 2 / 2 = 1 và cột ô là 2 % 2 = 0, nên
            h = 1 · 16 + 9 = 25 và w = 0 · 16 + 5 = 5. Hãy nạp cấu hình có sẵn của câu hỏi này trong trình khám phá, nhấp
            vào khối (3, 2) với mẫu 1, rồi luồng (5, 9).
          </p>
        }
      >
        <p>
          W_grid = 32 / 16 = 2. m = blockIdx.x = 3 and n = blockIdx.z = 1. The tile row is 2 / 2 = 1 and the tile column
          2 % 2 = 0, so h = 1 · 16 + 9 = 25 and w = 0 · 16 + 5 = 5. Load the question&rsquo;s preset in the explorer,
          click block (3, 2) with sample 1, then thread (5, 9).
        </p>
      </Translated>
    ),
    source: 'ch. 19 §19.2, pp. 461, 463',
  },
  'ch19/19.2/q-swap': {
    kind: 'choice',
    id: 'ch19/19.2/q-swap',
    prompt: (
      <Translated
        vi={
          <>
            Một đồng nghiệp đổi chỗ <code>/</code> và <code>%</code> ở dòng 4 và 5 của kernel, và các bài kiểm tra trên
            đầu ra 64 × 64 đều qua. Vì sao, và bài kiểm tra nào lẽ ra đã thất bại?
          </>
        }
      >
        A colleague swaps <code>/</code> and <code>%</code> in lines 4 and 5 of the kernel, and the tests on 64 × 64
        outputs all pass. Why, and which test would have failed?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Trên một lưới ô vuông, việc đổi chỗ chỉ thay đổi khối nào sở hữu ô nào; một đầu ra 32 × 64 sẽ thất bại.">
            On a square grid of tiles the swap only changes which block owns which tile; a 32 × 64 output would fail.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Việc đổi chỗ không thay đổi gì, vì / và % cho cùng kết quả mỗi khi W_grid chia hết blockIdx.y. Không bài kiểm tra nào thất bại.">
            The swap changes nothing, because / and % agree whenever W_grid divides blockIdx.y. No test would fail.
          </Translated>
        ),
        why: (
          <Translated vi="Chúng chỉ trùng nhau với vài khối. Ở 64 × 64, việc đổi chỗ chuyển phần lớn các khối sang một ô khác; chỉ là chúng tình cờ được chuyển lên một tập ô đầy đủ.">
            They agree for few blocks. At 64 × 64 the swap moves most blocks to another tile; it just happens to move
            them onto a full set of tiles.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các bài kiểm tra qua nhờ may mắn về thời điểm chạy của các luồng; chạy lại chúng trên một GPU khác sẽ thất bại.">
            The tests passed by luck of thread timing; the same tests run again on another GPU would fail.
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì ở đây phụ thuộc vào thời điểm. Phép tính chỉ số giống hệt nhau ở mọi lần chạy.">
            Nothing here depends on timing. The index arithmetic is the same on every run.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ những đầu ra lớn hơn 64 × 64 mới thất bại, vì chúng cần nhiều ô hơn số ô vừa trong một hàng.">
            Only outputs larger than 64 × 64 would fail, because they need more tiles than fit in one row.
          </Translated>
        ),
        why: (
          <Translated vi="Kích thước không quan trọng; hình dạng mới quan trọng. Bất kỳ đầu ra nào có lưới ô không vuông đều hỏng.">
            Size does not matter; shape does. Any output whose tile grid is not square breaks.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Khi H_grid = W_grid, việc đổi chỗ là một hoán vị của các ô: mỗi ô vẫn được một khối, và mỗi luồng ghi đúng
              pixel nó đã thực sự tính, nên mọi kết quả đều đúng. Ở 32 × 64 (H_grid 2, W_grid 4), mã bị đổi chỗ lấy hàng ô
              từ <code>blockIdx.y % 4</code>, vốn đạt tới 3, và cột từ <code>blockIdx.y / 4</code>, vốn không bao giờ vượt
              quá 1. Các cột ô 2 và 3 không bao giờ được tính, và một nửa số khối bị đưa tới các hàng 32 đến 63, vượt ra
              ngoài bản đồ.
            </p>
            <p>
              Có kiểm tra biên thì các luồng đó không làm gì và bài kiểm tra thất bại vì các cột bị thiếu. Không có kiểm
              tra biên thì chúng ghi vào bản đồ kế tiếp, và các luồng đi lạc của bản đồ cuối ghi vượt quá cuối Y: một truy
              cập ngoài biên đúng loại mà trình giả lập (emulator) được xây dựng để bắt.
            </p>
          </>
        }
      >
        <p>
          With H_grid = W_grid the swap is a permutation of the tiles: every tile still gets one block, and each thread
          stores the pixel it actually computed, so every result is right. At 32 × 64 (H_grid 2, W_grid 4) the
          swapped code takes the tile row from <code>blockIdx.y % 4</code>, which reaches 3, and the column from{' '}
          <code>blockIdx.y / 4</code>, which never passes 1. Tile columns 2 and 3 are never computed, and half the
          blocks are sent to rows 32 to 63, past the map.
        </p>
        <p>
          With a bounds check those threads do nothing and the test fails on the missing columns. Without one they
          store into the next map, and the last map&rsquo;s strays store past the end of Y: an out-of-bounds access of
          the kind the emulator is built to catch.
        </p>
      </Translated>
    ),
    source: 'ch. 19 §19.2, p. 463; our analysis, checked by convgrid.test.ts',
    vi: { source: 'ch. 19 §19.2, p. 463; phân tích của khóa học này, được kiểm tra bằng convgrid.test.ts' },
  },
  'ch19/19.2/q-ratio': {
    kind: 'numeric',
    id: 'ch19/19.2/q-ratio',
    prompt: (
      <Translated
        vi={
          <>
            Một đồng nghiệp sửa kernel trực tiếp để mỗi luồng tính hai bản đồ đầu ra, m và m + 1, tại pixel của nó. Nó nạp
            mỗi giá trị X một lần và dùng cho cả hai bản đồ, mỗi bản đồ với trọng số F riêng. Trên lớp C3 của LeNet (C =
            6, K = 5), nó thực hiện bao nhiêu phép toán dấu phẩy động cho mỗi byte nó yêu cầu từ bộ nhớ toàn cục (global
            memory)?
          </>
        }
      >
        A colleague changes the direct kernel so that each thread computes two output maps, m and m + 1, at its
        pixel. It loads each X value once and uses it for both maps, each with its own F weight. On LeNet&rsquo;s C3 (C
        = 6, K = 5), how many floating-point operations does it perform per byte it requests from global memory?
      </Translated>
    ),
    answer: Number(directRatioMaps(lenetLayer('C3')!, 2).toFixed(2)), // 1/3, shown as 0.33
    tol: 0.03,
    unit: 'FLOP/B',
    explain: (
      <Translated
        vi={
          <p>
            Mỗi luồng: 2 · 150 = 300 phép nhân-cộng, 600 FLOP. Số lần nạp: 150 giá trị X, nay dùng chung cho cả hai bản đồ,
            và 300 giá trị F, mỗi phép nhân-cộng một giá trị. Tổng cộng 450 lần nạp 4 byte, 1800 byte, và 600 / 1800 ≈
            0.33. Tốt hơn 0.25, nhưng vẫn không phụ thuộc vào hình dạng của lớp. Với r bản đồ mỗi luồng, tỉ số là r / (2(1
            + r)), không bao giờ đạt 0.5, vì mỗi phép nhân-cộng vẫn nạp trọng số riêng của nó. Muốn vượt qua mức đó thì
            trọng số cũng phải được dùng lại, giữa các pixel: các ô của mục 19.3.
          </p>
        }
      >
        <p>
          Per thread: 2 · 150 = 300 multiply-adds, 600 FLOP. Loads: 150 X values, now shared by both maps, and 300 F
          values, one per multiply-add. That is 450 loads of 4 bytes, 1800 bytes, and 600 / 1800 ≈ 0.33. Better than 0.25,
          and still independent of the layer&rsquo;s shape. With r maps per thread the ratio is r / (2(1 + r)), which
          never reaches 0.5, because every multiply-add still loads its own weight. Going past that needs the weights
          reused as well, across pixels: the tiles of section 19.3.
        </p>
      </Translated>
    ),
    source: 'ch. 19 §19.2, p. 463 (bandwidth-bound); the variation and its count are ours (notes, §19.2)',
    vi: {
      source:
        'ch. 19 §19.2, p. 463 (bị giới hạn bởi băng thông); biến thể này và phép đếm của nó là của khóa học này (notes, §19.2)',
    },
  },
  'ch19/19.2/q-layout': {
    kind: 'choice',
    id: 'ch19/19.2/q-layout',
    prompt: (
      <Translated
        vi={
          <>
            Với Hình 19.7 như được in, bố cục nào của X cho phép các lần nạp của một warp tại một (c, p, q) nhất định được
            hợp nhất?
          </>
        }
      >
        For Fig. 19.7 as written, which layout of X lets one warp&rsquo;s loads at a given (c, p, q) coalesce?
      </Translated>
    ),
    options: [
      { label: 'NCHW', correct: true },
      {
        label: 'NHWC',
        why: (
          <Translated vi="Các lane kề nhau đọc các w kề nhau, và trong NHWC những phần tử đó cách nhau C số float.">
            Neighbouring lanes read neighbouring w, and in NHWC those are C floats apart.
          </Translated>
        ),
      },
      {
        label: 'CHWN',
        why: (
          <Translated vi="Các w kề nhau cách nhau N số float. CHWN chỉ có lợi cho thiết kế mà ở đó các luồng kề nhau xử lý các mẫu khác nhau.">
            Neighbouring w are N floats apart. CHWN helps only a design whose neighbouring threads take different
            samples.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Cả ba như nhau">All three equally</Translated>,
        why: (
          <Translated vi="Bố cục không thay đổi số phần tử được đọc, nhưng nó thay đổi những phần tử nào nằm cạnh nhau.">
            The layout does not change how many elements are read, but it does change which of them sit next to each
            other.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Các <code>threadIdx.x</code> kề nhau xử lý các w kề nhau với cùng (c, p, q), và NCHW đặt những phần tử đó
              cạnh nhau. NHWC đặt chúng cách nhau C số float. CHWN đặt cùng một pixel của mọi mẫu cạnh nhau, điều chỉ có
              lợi nếu các luồng kề nhau xử lý các mẫu khác nhau, và khi đó chúng cũng dùng chung một trọng số bộ lọc.
            </p>
            <p>
              Các thư viện ưa NHWC vì những lý do khác: các đơn vị ma trận cần nó (cuDNN chuyển sang NHWC khi dùng Tensor
              Cores), và nó làm cho phép rút gọn (reduction) của cách diễn đạt GEMM ở mục 19.3 nằm liền mạch. MIOpen ghi
              CHWN là không được hỗ trợ.
            </p>
          </>
        }
      >
        <p>
          Neighbouring <code>threadIdx.x</code> take neighbouring w for the same (c, p, q), and NCHW puts those next to
          each other. NHWC puts them C floats apart. CHWN puts the same pixel of all samples together, which pays only
          if neighbouring threads take different samples, and then they also share one filter weight.
        </p>
        <p>
          Libraries prefer NHWC for other reasons: the matrix units need it (cuDNN converts to NHWC when it uses Tensor
          Cores), and it makes the reduction of the GEMM formulation in section 19.3 contiguous. MIOpen lists CHWN as not
          supported.
        </p>
      </Translated>
    ),
    source: 'ch. 19 Exercise 2, p. 475 (the book gives no answer); notes/today/ch19-facts.md A5, B3',
    vi: { source: 'ch. 19 Exercise 2, p. 475 (sách không đưa ra đáp án); notes/today/ch19-facts.md A5, B3' },
  },
  'ch19/19.2/q-reuse': {
    kind: 'choice',
    id: 'ch19/19.2/q-reuse',
    prompt: (
      <Translated
        vi={
          <>
            Vì sao kernel trực tiếp không thể giữ một ô đầu vào trong bộ nhớ chia sẻ (shared memory) và dùng nó cho cả M
            bản đồ đầu ra?
          </>
        }
      >
        Why can&rsquo;t the direct kernel keep one input tile in shared memory and use it for all M output maps?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi khối phục vụ một bản đồ đầu ra, và bộ nhớ chia sẻ là riêng của từng khối.">
            Each block serves one output map, and shared memory is private to a block.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Bộ nhớ chia sẻ quá nhỏ để chứa một ô đầu vào, vùng biên của nó và cả M bộ lọc.">
            Shared memory is too small to hold an input tile, its halo and all M filters.
          </Translated>
        ),
        why: (
          <Translated vi="Chương 7 giữ đúng một ô như vậy, và các bộ lọc không cần nằm trong bộ nhớ chia sẻ. Kích thước không phải là thứ ngăn việc dùng lại giữa các bản đồ.">
            Chapter 7 holds exactly such a tile, and the filters need not sit in shared memory. Size is not what stops
            reuse across maps.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Mỗi bản đồ đầu ra đọc một phần khác nhau của đầu vào, nên không có ô nào chung cho hai bản đồ.">
            Each output map reads a different part of the input, so no tile is common to two maps.
          </Translated>
        ),
        why: (
          <Translated vi="Mọi bản đồ đầu ra đều đọc mọi bản đồ đầu vào, toàn bộ: cùng một X.">
            Every output map reads every input map, all of it: the same X.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các rào chắn cần để chia sẻ ô sẽ gây deadlock.">
            The barriers needed to share the tile would deadlock.
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn hoạt động trong một khối; chúng không cản trở gì ở đây.">
            Barriers work within a block; nothing about them is in the way.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trục x của lưới là bản đồ đầu ra, nên các khối cho bản đồ 0, 1, 2, ... là những khối khác nhau, và không khối
            nào thấy được bộ nhớ chia sẻ của khối khác. Mỗi khối lại nạp cùng một đầu vào. Muốn có sự dùng lại đó cần một
            thiết kế khác, trong đó một khối tạo đầu ra cho nhiều bản đồ từ một ô đầu vào. Đó là điều cách diễn đạt GEMM ở
            mục 19.3 làm.
          </p>
        }
      >
        <p>
          Grid x is the output map, so the blocks for maps 0, 1, 2, ... are different blocks, and none can see another&rsquo;s
          shared memory. Each fetches the same input again. Getting that reuse needs a different design, in which one
          block produces outputs for several maps from one tile of input. That is what the GEMM formulation of section 19.3
          does.
        </p>
      </Translated>
    ),
    source: 'ch. 19 §19.2, p. 463',
  },
};

// ---------------------------------------------------------------- which loops parallelise

function LoopSort() {
  const [marks, setMarks] = useState<Partial<Record<LoopVar, LoopRole>>>({});
  const [checked, setChecked] = useState(false);
  const wrong = gradeLoops(marks);
  const all = FIG_19_4_LOOPS.every((l) => marks[l.v]);
  const set = (v: LoopVar, r: LoopRole) => {
    setMarks({ ...marks, [v]: r });
    setChecked(false);
  };
  return (
    <Bench
      title={tr('Which loops can be split across threads?', 'Những vòng lặp nào có thể chia cho nhiều luồng?')}
      task={tr('Mark each loop of Fig. 19.4, then check.', 'Đánh dấu từng vòng lặp của Hình 19.4, rồi kiểm tra.')}
      foot={tr(
        'The seven loops of Fig. 19.4, outermost first, with the line each opens.',
        'Bảy vòng lặp của Hình 19.4, từ ngoài vào trong, kèm dòng mà mỗi vòng mở ra.',
      )}
    >
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(15.5rem, 1fr))', gap: '0.7rem' }}>
        {FIG_19_4_LOOPS.map((l) => {
          const bad = checked && wrong.includes(l.v);
          const good = checked && !wrong.includes(l.v);
          return (
            <div key={l.v} className="panel" data-loop={l.v} data-result={checked ? (bad ? 'wrong' : 'right') : ''}
              style={{ borderColor: bad ? 'var(--fail)' : good ? 'var(--ok)' : undefined }}>
              <p style={{ margin: '0 0 0.4rem' }}>
                <code>for {l.v}</code>{' '}
                <span className="muted">
                  <Translated vi="(dòng">(line</Translated> {l.line}):{' '}
                  {tr(
                    l.over,
                    l.v === 'n'
                      ? 'các mẫu trong batch'
                      : l.v === 'm'
                        ? 'các bản đồ đặc trưng đầu ra'
                        : l.v === 'h'
                          ? 'các hàng đầu ra'
                          : l.v === 'w'
                            ? 'các cột đầu ra'
                            : l.v === 'c'
                              ? 'các bản đồ đặc trưng đầu vào'
                              : l.v === 'p'
                                ? 'các hàng của bộ lọc'
                                : 'các cột của bộ lọc',
                  )}
                </span>
              </p>
              <span className="seg" style={{ display: 'flex' }}>
                <button style={{ flex: 1 }} aria-pressed={marks[l.v] === 'independent'} onClick={() => set(l.v, 'independent')}>
                  <Translated vi="các đầu ra độc lập">independent outputs</Translated>
                </button>
                <button style={{ flex: 1 }} aria-pressed={marks[l.v] === 'accumulates'} onClick={() => set(l.v, 'accumulates')}>
                  <Translated vi="cộng dồn vào một phần tử">accumulates into one</Translated>
                </button>
              </span>
            </div>
          );
        })}
      </div>
      <div className="row" style={{ marginTop: '0.9rem' }}>
        <button className="primary" disabled={!all} onClick={() => setChecked(true)}>
          <Translated vi="Kiểm tra">Check</Translated>
        </button>
        {checked ? (
          wrong.length ? (
            <span className="chip fail" data-out="loops">
              <Translated vi={<>{wrong.length} trên 7 sai: {wrong.join(', ')}</>}>
                {wrong.length} of 7 wrong: {wrong.join(', ')}
              </Translated>
            </span>
          ) : (
            <span className="chip ok" data-out="loops">
              <Translated vi="cả 7 đều đúng">all 7 right</Translated>
            </span>
          )
        ) : null}
      </div>
      {checked ? (
        <p style={{ margin: '0.7rem 0 0' }} data-out="loops-why">
          <Translated
            vi={
              <>
                Mỗi lần lặp của n, m, h và w ghi một phần tử khác nhau của Y. Mỗi lần lặp của c, p và q đọc{' '}
                <code>Y[n, m, h, w]</code>, cộng thêm một tích rồi ghi lại: chia những vòng đó cho nhiều luồng thì hai luồng
                có thể đọc cùng một giá trị cũ, mỗi luồng cộng tích của mình, và một trong hai phép cộng bị mất. Phép cộng
                nguyên tử sẽ ngăn điều đó, nhưng phải trả giá; sách giữ c, p và q bên trong một luồng.
              </>
            }
          >
            Iterations of n, m, h and w each write a different element of Y. Every iteration of c, p and q reads{' '}
            <code>Y[n, m, h, w]</code>, adds one product and writes it back: split those across threads and two of them
            can read the same old value, each add their own product, and one of the two additions is lost. Atomic additions
            would prevent that, at a price; the book keeps c, p and q inside one thread.
          </Translated>
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- from blockIdx to pixel

type Host = 'floor' | 'ceil-raw' | 'ceil-guard';
const HOST_LABEL: Record<Host, string> = {
  get floor() {
    return tr("the book's host code (integer division)", 'mã host của sách (phép chia nguyên)');
  },
  get 'ceil-raw'() {
    return tr('ceiling division, no bounds check', 'phép chia làm tròn lên, không kiểm tra biên');
  },
  get 'ceil-guard'() {
    return tr('ceiling division with a bounds check', 'phép chia làm tròn lên, có kiểm tra biên');
  },
};
interface Shape {
  N: number;
  M: number;
  Hout: number;
  Wout: number;
  tile: number;
}
const PRESETS: { label: string; s: Shape }[] = [
  {
    get label() {
      return tr('Fig. 19.6: M = 4, 32 × 32 maps', 'Hình 19.6: M = 4, bản đồ 32 × 32');
    },
    s: { N: 1, M: 4, Hout: 32, Wout: 32, tile: 16 },
  },
  {
    get label() {
      return tr('LeNet C1: M = 6, 28 × 28 maps', 'LeNet C1: M = 6, bản đồ 28 × 28');
    },
    s: { N: 1, M: 6, Hout: 28, Wout: 28, tile: 16 },
  },
  {
    get label() {
      return tr('Rectangular: M = 2, 32 × 64 maps', 'Hình chữ nhật: M = 2, bản đồ 32 × 64');
    },
    s: { N: 1, M: 2, Hout: 32, Wout: 64, tile: 16 },
  },
  {
    get label() {
      return tr('The question: N = 2, M = 4, 32 × 32', 'Câu hỏi: N = 2, M = 4, 32 × 32');
    },
    s: { N: 2, M: 4, Hout: 32, Wout: 32, tile: 16 },
  },
  {
    get label() {
      return tr('One 4096 × 4096 map', 'Một bản đồ 4096 × 4096');
    },
    s: { N: 1, M: 1, Hout: 4096, Wout: 4096, tile: 16 },
  },
];
const TILES = [2, 4, 8, 16, 32];
const MAX_MAPS = 8;

function NumField({ label, value, min, max, onChange }: { label: string; value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <label className="field">
      {label}
      <input type="number" min={min} max={max} value={value} aria-label={label}
        onChange={(e) => {
          const v = Math.round(Number(e.target.value));
          if (Number.isFinite(v)) onChange(Math.max(min, Math.min(max, v)));
        }} />
    </label>
  );
}

/** One row of the grid-limits table: a verdict chip over the documented limit, per vendor. */
function LimitRow({ label, value, cuda, hip }: {
  label: ReactNode;
  value: number;
  cuda: [boolean, number] | null;
  hip: [boolean, number, string?];
}) {
  const pad = { padding: '0.4rem 0.45rem 0.4rem 0' }; // i18n-ignore: CSS value
  const cell = ([ok, limit, mark]: [boolean, number, string?]) => (
    <td style={pad}>
      <span className={'chip ' + (ok ? 'ok' : 'fail')}>{ok ? tr('within', 'trong giới hạn') : tr('over', 'vượt')}</span>
      <br />
      <span style={{ whiteSpace: 'nowrap' }}>≤ {limitText(limit)}{mark ?? ''}</span>
    </td>
  );
  return (
    <tr>
      <td style={{ ...pad, paddingLeft: '0.6rem' }}>{label}</td>
      <td className="num" style={pad}>{big(value)}</td>
      {cuda ? cell(cuda) : <td className="muted" style={pad}><Translated vi="không giới hạn">no cap</Translated></td>}
      {cell(hip)}
    </tr>
  );
}

function GridMap() {
  const [s, setS] = useState<Shape>(PRESETS[0].s);
  const [host, setHost] = useState<Host>('floor');
  const [swap, setSwap] = useState(false);
  const [sample, setSample] = useState(0);
  const [blk, setBlk] = useState<{ bx: number; by: number } | null>(null);
  const [thr, setThr] = useState<{ tx: number; ty: number } | null>(null);
  const [mode, setMode] = useState<'thread' | 'pixel'>('thread');
  const [pix, setPix] = useState<{ m: number; h: number; w: number } | null>(null);
  const [guess, setGuess] = useState({ bx: '', by: '', bz: '', tx: '', ty: '' });
  const [revealed, setRevealed] = useState(false);

  const p: LaunchParams = { ...s, division: host === 'floor' ? 'floor' : 'ceil' };
  const opts = { swap, guard: host === 'ceil-guard' };
  const g = geometry(p);
  const lim = limits(g);
  const bz = Math.min(sample, s.N - 1);
  const cov = useMemo(
    () => coverage({ ...s, division: host === 'floor' ? 'floor' : 'ceil' }, { swap, guard: host === 'ceil-guard' }, bz),
    [s, host, swap, bz],
  );
  const drawable = cov != null && s.Hout <= 256 && s.Wout <= 256;
  const gridDrawable = s.M <= 32 && g.T <= 64 && g.T > 0;
  const block = blk && blk.bx < g.grid.x && blk.by < g.grid.y ? blk : null;
  const thread = block && thr && thr.tx < s.tile && thr.ty < s.tile ? thr : null;
  const pos: ThreadPos | null = block && thread ? { ...block, bz, ...thread } : null;
  const mapped = pos ? mapThread(p, g, pos, swap) : null;
  const corner = block ? mapThread(p, g, { ...block, bz, tx: 0, ty: 0 }, swap) : null;

  const reset = () => {
    setBlk(null);
    setThr(null);
    setPix(null);
    setRevealed(false);
  };
  const update = (o: Partial<Shape>) => {
    setS({ ...s, ...o });
    reset();
  };

  // ---- the grid of blocks
  const cs = gridDrawable ? Math.max(8, Math.min(22, Math.floor(Math.min(260 / s.M, 300 / g.T)))) : 0;
  const strayIn = (bx: number, by: number) => {
    const c = mapThread(p, g, { bx, by, bz, tx: 0, ty: 0 }, swap);
    return c.h + s.tile > s.Hout || c.w + s.tile > s.Wout;
  };

  // ---- the maps
  const shown = Math.min(s.M, MAX_MAPS);
  const scale = Math.max(1, Math.min(6, 168 / Math.max(s.Hout, s.Wout)));
  const pickPixel = (m: number) => (e: MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const w = Math.min(s.Wout - 1, Math.max(0, Math.floor(((e.clientX - r.left) / r.width) * s.Wout)));
    const h = Math.min(s.Hout - 1, Math.max(0, Math.floor(((e.clientY - r.top) / r.height) * s.Hout)));
    if (mode === 'pixel') {
      setPix({ m, h, w });
      setRevealed(false);
      setGuess({ bx: '', by: '', bz: '', tx: '', ty: '' });
    }
  };
  const writers = useMemo(
    () => (pix && revealed && cov ? writersOf({ ...s, division: host === 'floor' ? 'floor' : 'ceil' }, { swap, guard: host === 'ceil-guard' }, bz, pix.m, pix.h, pix.w) : []),
    [pix, revealed, cov, s, host, swap, bz],
  );
  const guessPos = { bx: Number(guess.bx), by: Number(guess.by), bz: Number(guess.bz), tx: Number(guess.tx), ty: Number(guess.ty) };
  const guessRight = writers.some((t) => t.bx === guessPos.bx && t.by === guessPos.by && t.bz === guessPos.bz && t.tx === guessPos.tx && t.ty === guessPos.ty);
  const guessFilled = Object.values(guess).every((v) => v.trim() !== '');

  const rowRuns = (m: number, h: number) => {
    const out: { w0: number; w1: number; st: number }[] = [];
    if (!cov) return out;
    for (let w = 0; w < s.Wout; w++) {
      const c = cov.counts[(m * s.Hout + h) * s.Wout + w];
      const st = c === 0 ? 0 : c === 1 ? 1 : 2;
      const last = out[out.length - 1];
      if (last && last.st === st) last.w1 = w + 1;
      else out.push({ w0: w, w1: w + 1, st });
    }
    return out;
  };
  const FILL = ['var(--fail-wash)', 'var(--ok-wash)', 'var(--fail)'];


  return (
    <Bench
      title={tr('From blockIdx to pixel', 'Từ blockIdx đến pixel')}
      task={tr(
        'Pick a preset and a host code, click a block, then a thread. Then switch to reverse mode.',
        'Chọn một cấu hình có sẵn và một mã host, nhấp vào một khối, rồi một luồng. Sau đó chuyển sang chế độ ngược.',
      )}
      foot={
        <Translated
          vi={
            <>
              Độ phủ: <span className="chip ok">được tính một lần</span> <span className="chip fail">không bao giờ được tính</span>{' '}
              <span className="chip fail" style={{ background: 'var(--fail)', color: 'var(--page)' }}>được ghi từ hai lần trở lên</span>. Giới
              hạn lưới là các giá trị được CUDA và HIP công bố trong tài liệu (ch05-facts A5, ch19-facts A11, B2); Metal
              không công bố giới hạn nào.
            </>
          }
        >
          Coverage: <span className="chip ok">computed once</span> <span className="chip fail">never computed</span>{' '}
          <span className="chip fail" style={{ background: 'var(--fail)', color: 'var(--page)' }}>stored twice or more</span>. Grid limits are
          CUDA&rsquo;s and HIP&rsquo;s documented values (ch05-facts A5, ch19-facts A11, B2); Metal documents none.
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Cấu hình có sẵn">Preset</Translated>
          <select aria-label={tr('Preset', 'Cấu hình có sẵn')} value="" onChange={(e) => { const k = Number(e.target.value); if (k >= 0) { setS(PRESETS[k].s); setSample(0); reset(); } }}>
            <option value="">{tr('choose…', 'chọn…')}</option>
            {PRESETS.map((x, i) => <option key={i} value={i}>{x.label}</option>)}
          </select>
        </label>
        <NumField label={tr('N (samples)', 'N (số mẫu)')} value={s.N} min={1} max={1000000} onChange={(v) => update({ N: v })} />
        <NumField label={tr('M (output maps)', 'M (số bản đồ đầu ra)')} value={s.M} min={1} max={100000} onChange={(v) => update({ M: v })} />
        <NumField label="H_out" value={s.Hout} min={1} max={100000} onChange={(v) => update({ Hout: v })} />{/* i18n-ignore: variable name */}
        <NumField label="W_out" value={s.Wout} min={1} max={100000} onChange={(v) => update({ Wout: v })} />{/* i18n-ignore: variable name */}
        <label className="field">
          TILE_WIDTH
          <select aria-label="TILE_WIDTH" value={s.tile} onChange={(e) => update({ tile: Number(e.target.value) })}>
            {TILES.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Mã host và kernel">Host code and kernel</Translated>
          <select aria-label={tr('Host code', 'Mã host')} value={host} onChange={(e) => { setHost(e.target.value as Host); reset(); }}>
            {(Object.keys(HOST_LABEL) as Host[]).map((h) => <option key={h} value={h}>{HOST_LABEL[h]}</option>)}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={swap} onChange={(e) => { setSwap(e.target.checked); reset(); }} />{' '}
          <Translated vi="lỗi cài sẵn: đổi chỗ / và % ở dòng 4 và 5">
            planted bug: swap / and % in
            lines 4 and 5
          </Translated>
        </label>
        <NumField label={tr('sample (blockIdx.z)', 'mẫu (blockIdx.z)')} value={bz} min={0} max={Math.max(0, s.N - 1)} onChange={(v) => { setSample(v); reset(); }} />
        <span className="seg" role="group" aria-label={tr('Mode', 'Chế độ')}>
          <button aria-pressed={mode === 'thread'} onClick={() => { setMode('thread'); setPix(null); }}>
            <Translated vi="chọn một luồng">pick a thread</Translated>
          </button>
          <button aria-pressed={mode === 'pixel'} onClick={() => { setMode('pixel'); setPix(null); setRevealed(false); }}>
            <Translated vi="ngược: chọn một pixel">reverse: pick a pixel</Translated>
          </button>
        </span>
      </div>

      <p style={{ margin: '0 0 0.7rem' }} data-out="geometry" data-gx={g.grid.x} data-gy={g.grid.y} data-gz={g.grid.z} data-threads={g.threads}>
        <Translated
          vi={
            <>
              W_grid = {host === 'floor' ? `${s.Wout} / ${s.tile}` : `⌈${s.Wout} / ${s.tile}⌉`} = <b>{g.Wgrid}</b>, H_grid ={' '}
              {host === 'floor' ? `${s.Hout} / ${s.tile}` : `⌈${s.Hout} / ${s.tile}⌉`} = <b>{g.Hgrid}</b>, T ={' '}
              <b>{g.T}</b>. <code>gridDim({big(g.grid.x)}, {big(g.grid.y)}, {big(g.grid.z)})</code>,{' '}
              <code>blockDim({s.tile}, {s.tile}, 1)</code>: {big(g.threadsPerBlock)} luồng mỗi khối, tổng cộng{' '}
              {big(g.threads)}.
            </>
          }
        >
          W_grid = {host === 'floor' ? `${s.Wout} / ${s.tile}` : `⌈${s.Wout} / ${s.tile}⌉`} = <b>{g.Wgrid}</b>, H_grid ={' '}
          {host === 'floor' ? `${s.Hout} / ${s.tile}` : `⌈${s.Hout} / ${s.tile}⌉`} = <b>{g.Hgrid}</b>, T ={' '}
          <b>{g.T}</b>. <code>gridDim({big(g.grid.x)}, {big(g.grid.y)}, {big(g.grid.z)})</code>,{' '}
          <code>blockDim({s.tile}, {s.tile}, 1)</code>: {big(g.threadsPerBlock)} threads per block, {big(g.threads)} in all.
        </Translated>
      </p>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div>
          <div className="mx-label">
            <Translated vi={<>Lưới cho mẫu {bz}: x = bản đồ đầu ra, y = ô</>}>
              Grid for sample {bz}: x = output map, y = tile
            </Translated>
          </div>
          {gridDrawable ? (
            <svg width={cs * s.M + 2} height={cs * g.T + 2} role="img" aria-label={tr('Grid of blocks', 'Lưới các khối')} data-grid="blocks" style={{ display: 'block', maxWidth: '100%' }}>
              {Array.from({ length: s.M * g.T }, (_, i) => {
                const bx = i % s.M;
                const by = Math.floor(i / s.M);
                const sel = block?.bx === bx && block?.by === by;
                return (
                  <rect key={i} x={1 + bx * cs} y={1 + by * cs} width={cs - 2} height={cs - 2} rx="2"
                    data-block={`${bx},${by}`}
                    fill={sel ? 'var(--ink)' : strayIn(bx, by) ? 'var(--rule)' : 'var(--cell)'}
                    stroke="var(--ink-3)" style={{ cursor: 'pointer' }}
                    onClick={() => { setBlk({ bx, by }); setThr(null); }} />
                );
              })}
            </svg>
          ) : (
            <p className="muted" data-out="grid-too-big">
              {g.T === 0
                ? tr('No blocks at all: T is 0.', 'Không có khối nào: T bằng 0.')
                : tr(
                    `${big(s.M)} × ${big(g.T)} blocks: too many to draw. The counts and limits still apply.`,
                    `${big(s.M)} × ${big(g.T)} khối: quá nhiều để vẽ. Các con số và giới hạn vẫn áp dụng.`,
                  )}
            </p>
          )}
          {gridDrawable ? (
            <p className="muted" style={{ margin: '0.3rem 0 0' }}>
              <Translated vi="Khối màu xám có luồng nằm ngoài bản đồ của nó.">Grey blocks have threads outside their map.</Translated>
            </p>
          ) : null}

          {block ? (
            <div style={{ marginTop: '0.8rem' }}>
              <div className="mx-label">
                <Translated
                  vi={
                    <>
                      Khối ({block.bx}, {block.by}, {bz}): {s.tile} × {s.tile} luồng của nó
                      {mode === 'thread' ? ' (nhấp vào một luồng)' : ''}
                    </>
                  }
                >
                  Block ({block.bx}, {block.by}, {bz}): its {s.tile} × {s.tile} threads{mode === 'thread' ? ' (click one)' : ''}
                </Translated>
              </div>
              {(() => {
                const tc = Math.max(5, Math.min(14, Math.floor(224 / s.tile)));
                return (
                  <svg width={tc * s.tile + 2} height={tc * s.tile + 2} role="img" aria-label={tr('Threads of the block', 'Các luồng của khối')} data-grid="threads" style={{ display: 'block' }}>
                    {Array.from({ length: s.tile * s.tile }, (_, i) => {
                      const tx = i % s.tile;
                      const ty = Math.floor(i / s.tile);
                      const r = mapThread(p, g, { ...block, bz, tx, ty }, swap);
                      const sel = thread?.tx === tx && thread?.ty === ty;
                      const fill = sel ? 'var(--ink)' : r.inRange ? 'var(--cell)' : opts.guard ? 'var(--rule)' : 'var(--fail-wash)';
                      return (
                        <rect key={i} x={1 + tx * tc} y={1 + ty * tc} width={tc - 1} height={tc - 1} fill={fill} stroke="var(--rule)"
                          data-thread={`${tx},${ty}`} style={{ cursor: 'pointer' }} onClick={() => setThr({ tx, ty })} />
                      );
                    })}
                  </svg>
                );
              })()}
              {corner && (corner.h >= s.Hout || corner.w >= s.Wout) ? (
                <p className="t-fail" style={{ margin: '0.3rem 0 0' }} data-out="tile-outside">
                  <Translated
                    vi={
                      <>
                        Ô của khối này bắt đầu ở hàng {corner.h}, cột {corner.w}: nằm ngoài bản đồ {s.Hout} × {s.Wout}.
                      </>
                    }
                  >
                    This block&rsquo;s tile starts at row {corner.h}, column {corner.w}: outside the {s.Hout} × {s.Wout} map.
                  </Translated>
                </p>
              ) : null}
            </div>
          ) : null}
        </div>

        <div>
          <div className="mx-label">
            <Translated
              vi={
                <>
                  Các bản đồ đầu ra của mẫu {bz}
                  {s.M > MAX_MAPS ? ` (${MAX_MAPS} bản đồ đầu tiên trong ${s.M})` : ''}
                  {mode === 'pixel' ? ': nhấp vào một pixel' : ''}
                </>
              }
            >
              Output maps of sample {bz}
              {s.M > MAX_MAPS ? ` (the first ${MAX_MAPS} of ${s.M})` : ''}
              {mode === 'pixel' ? ': click a pixel' : ''}
            </Translated>
          </div>
          {drawable ? (
            <div className="row" style={{ alignItems: 'flex-start', gap: '0.6rem' }}>
              {Array.from({ length: shown }, (_, m) => (
                <div key={m}>
                  <svg viewBox={`0 0 ${s.Wout} ${s.Hout}`} width={s.Wout * scale} height={s.Hout * scale} role="img"
                    aria-label={tr(`Output map ${m}`, `Bản đồ đầu ra ${m}`)} data-map={m} shapeRendering="crispEdges"
                    style={{ display: 'block', border: '1px solid var(--rule)', cursor: mode === 'pixel' ? 'crosshair' : 'default', maxWidth: '100%', height: 'auto' }}
                    onClick={pickPixel(m)}>
                    {Array.from({ length: s.Hout }, (_, h) =>
                      rowRuns(m, h).map((r) => <rect key={`${h}:${r.w0}`} x={r.w0} y={h} width={r.w1 - r.w0} height={1} fill={FILL[r.st]} />),
                    )}
                    {Array.from({ length: Math.ceil(s.Wout / s.tile) - 1 }, (_, i) => (
                      <line key={'v' + i} x1={(i + 1) * s.tile} x2={(i + 1) * s.tile} y1={0} y2={s.Hout} stroke="var(--ink-3)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                    ))}
                    {Array.from({ length: Math.ceil(s.Hout / s.tile) - 1 }, (_, i) => (
                      <line key={'h' + i} y1={(i + 1) * s.tile} y2={(i + 1) * s.tile} x1={0} x2={s.Wout} stroke="var(--ink-3)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
                    ))}
                    {block && corner && block.bx === m && corner.h < s.Hout && corner.w < s.Wout ? (
                      <rect x={corner.w} y={corner.h} width={s.tile} height={s.tile} fill="none" stroke="var(--ink)" strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
                    ) : null}
                    {mapped && mapped.m === m && mapped.inRange ? (
                      <rect x={mapped.w} y={mapped.h} width={1} height={1} fill="var(--ink)" />
                    ) : null}
                    {pix && pix.m === m ? (
                      <rect x={pix.w - 0.5} y={pix.h - 0.5} width={2} height={2} fill="none" stroke="var(--ink)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
                    ) : null}
                  </svg>
                  <div className="muted" style={{ fontSize: '0.72rem' }}>
                    <Translated vi={<>bản đồ {m}</>}>map {m}</Translated>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted" data-out="maps-too-big">
              <Translated vi="Quá lớn để vẽ ở đây; kích thước hình học và các giới hạn bên dưới vẫn áp dụng.">
                Too large to draw here; the geometry and the limits below still apply.
              </Translated>
            </p>
          )}
          {cov ? (
            <p style={{ margin: '0.6rem 0 0' }} data-out="coverage" data-never={cov.never} data-once={cov.once} data-multi={cov.multi}
              data-idle={cov.idle} data-stray={cov.stray} data-pastend={cov.pastEnd}>
              <Translated
                vi={
                  <>
                    Trong {big(s.M * s.Hout * s.Wout)} pixel của mẫu này: <b>{big(cov.once)}</b> được tính một lần,{' '}
                    <b className={cov.never ? 't-fail' : ''}>{big(cov.never)}</b> không bao giờ được tính
                    {cov.multi ? <>, <b className="t-fail">{big(cov.multi)}</b> được ghi từ hai lần trở lên</> : null}. Các
                    luồng nằm ngoài bản đồ của chúng: {big(cov.idle)} nhàn rỗi{opts.guard ? '' : ' (không có kiểm tra, nên không luồng nào nhàn rỗi)'},{' '}
                    {big(cov.stray)} vẫn ghi
                    {cov.pastEnd ? <>, trong đó <b className="t-fail">{big(cov.pastEnd)}</b> ghi vượt quá cuối Y</> : null}.
                  </>
                }
              >
                Of the {big(s.M * s.Hout * s.Wout)} pixels of this sample: <b>{big(cov.once)}</b> computed once,{' '}
                <b className={cov.never ? 't-fail' : ''}>{big(cov.never)}</b> never
                {cov.multi ? <>, <b className="t-fail">{big(cov.multi)}</b> stored twice or more</> : null}. Threads outside
                their map: {big(cov.idle)} idle{opts.guard ? '' : ' (no check, so none idle)'}, {big(cov.stray)} storing
                anyway{cov.pastEnd ? <>, of which <b className="t-fail">{big(cov.pastEnd)}</b> past the end of Y</> : null}.
              </Translated>
            </p>
          ) : (
            <p className="muted" data-out="coverage-too-big">
              <Translated vi="Hơn hai triệu luồng mỗi mẫu: độ phủ không được mô phỏng ở đây.">
                More than two million threads per sample: coverage is not simulated here.
              </Translated>
            </p>
          )}
        </div>
      </div>

      {mode === 'thread' && pos && mapped ? (
        <div className={'narrate' + (!mapped.inRange && !opts.guard ? ' bad' : '')} style={{ marginTop: '0.8rem' }} data-out="map"
          data-n={mapped.n} data-m={mapped.m} data-h={mapped.h} data-w={mapped.w} data-y={mapped.yIndex}>
          <Translated
            vi={
              <>
                blockIdx ({pos.bx}, {pos.by}, {pos.bz}), threadIdx ({pos.tx}, {pos.ty}): m = blockIdx.x = {mapped.m}; hàng ô ={' '}
                {pos.by} {swap ? '%' : '/'} {g.Wgrid} = {Math.floor((mapped.h - pos.ty) / s.tile)}, cột ô = {pos.by} {swap ? '/' : '%'}{' '}
                {g.Wgrid} = {Math.floor((mapped.w - pos.tx) / s.tile)}; h = {Math.floor((mapped.h - pos.ty) / s.tile)}·{s.tile} + {pos.ty} ={' '}
                {mapped.h}, w = {Math.floor((mapped.w - pos.tx) / s.tile)}·{s.tile} + {pos.tx} = {mapped.w}; n = {mapped.n}.{' '}
                {mapped.inRange ? (
                  <>Ghi Y[{mapped.n}, {mapped.m}, {mapped.h}, {mapped.w}], chỉ số phẳng (({mapped.n}·{s.M} + {mapped.m})·{s.Hout} + {mapped.h})·{s.Wout} + {mapped.w} = {big(mapped.yIndex)}.</>
                ) : opts.guard ? (
                  <>Nằm ngoài bản đồ {s.Hout} × {s.Wout}: kiểm tra biên chặn nó lại, và nó không ghi gì.</>
                ) : (
                  <b className="t-fail">
                    Nằm ngoài bản đồ, và không có gì chặn nó: nó ghi vào chỉ số phẳng {big(mapped.yIndex)}
                    {mapped.yIndex >= s.N * s.M * s.Hout * s.Wout ? ', vượt quá cuối Y.' : ', vốn thuộc về một pixel khác.'}
                  </b>
                )}
              </>
            }
          >
            blockIdx ({pos.bx}, {pos.by}, {pos.bz}), threadIdx ({pos.tx}, {pos.ty}): m = blockIdx.x = {mapped.m}; tile row ={' '}
            {pos.by} {swap ? '%' : '/'} {g.Wgrid} = {Math.floor((mapped.h - pos.ty) / s.tile)}, tile column = {pos.by} {swap ? '/' : '%'}{' '}
            {g.Wgrid} = {Math.floor((mapped.w - pos.tx) / s.tile)}; h = {Math.floor((mapped.h - pos.ty) / s.tile)}·{s.tile} + {pos.ty} ={' '}
            {mapped.h}, w = {Math.floor((mapped.w - pos.tx) / s.tile)}·{s.tile} + {pos.tx} = {mapped.w}; n = {mapped.n}.{' '}
            {mapped.inRange ? (
              <>Stores Y[{mapped.n}, {mapped.m}, {mapped.h}, {mapped.w}], flat index (({mapped.n}·{s.M} + {mapped.m})·{s.Hout} + {mapped.h})·{s.Wout} + {mapped.w} = {big(mapped.yIndex)}.</>
            ) : opts.guard ? (
              <>Outside the {s.Hout} × {s.Wout} map: the bounds check stops it, and it stores nothing.</>
            ) : (
              <b className="t-fail">
                Outside the map, and nothing stops it: it stores to flat index {big(mapped.yIndex)}
                {mapped.yIndex >= s.N * s.M * s.Hout * s.Wout ? ', past the end of Y.' : ', which belongs to another pixel.'}
              </b>
            )}
          </Translated>
        </div>
      ) : null}
      {mode === 'thread' && !pos ? (
        <p className="muted" style={{ margin: '0.8rem 0 0' }}>
          <Translated vi="Nhấp vào một khối trong lưới, rồi một luồng của nó.">Click a block in the grid, then one of its threads.</Translated>
        </p>
      ) : null}

      {mode === 'pixel' ? (
        pix ? (
          <div className="panel" style={{ marginTop: '0.8rem' }} data-out="reverse">
            <p style={{ margin: '0 0 0.5rem' }}>
              <Translated vi={<>Luồng nào ghi Y[{bz}, {pix.m}, {pix.h}, {pix.w}]? Dự đoán, rồi xem đáp án.</>}>
                Which thread stores Y[{bz}, {pix.m}, {pix.h}, {pix.w}]? Predict, then reveal.
              </Translated>
            </p>
            <div className="row">
              {(['bx', 'by', 'bz', 'tx', 'ty'] as const).map((k) => (
                <label key={k} className="field">
                  {k === 'bx' ? 'blockIdx.x' : k === 'by' ? 'blockIdx.y' : k === 'bz' ? 'blockIdx.z' : k === 'tx' ? 'threadIdx.x' : 'threadIdx.y'}
                  <input type="number" aria-label={k} value={guess[k]} style={{ width: '5.5rem' }}
                    onChange={(e) => { setGuess({ ...guess, [k]: e.target.value }); setRevealed(false); }} />
                </label>
              ))}
              <button className="primary" disabled={!guessFilled} onClick={() => setRevealed(true)}>
                <Translated vi="Xem đáp án">Reveal</Translated>
              </button>
            </div>
            {revealed ? (
              <p style={{ margin: '0.6rem 0 0' }} data-out="reveal" data-right={guessRight ? 'yes' : 'no'} data-writers={writers.length}>
                {guessRight ? (
                  <span className="chip ok">
                    <Translated vi="đúng">right</Translated>
                  </span>
                ) : (
                  <span className="chip fail">
                    <Translated vi="không phải luồng đó">not that one</Translated>
                  </span>
                )}{' '}
                {writers.length === 0
                  ? tr('No thread stores here: no tile covers this pixel.', 'Không luồng nào ghi vào đây: không ô nào phủ pixel này.')
                  : `${writers.length === 1 ? tr('The thread that stores here is', 'Luồng ghi vào đây là') : tr('These threads all store here:', 'Tất cả các luồng này đều ghi vào đây:')} ${writers
                      .map((t) => `blockIdx (${t.bx}, ${t.by}, ${t.bz}), threadIdx (${t.tx}, ${t.ty})`) // i18n-ignore: CUDA index notation
                      .join('; ')}.`}
              </p>
            ) : null}
          </div>
        ) : (
          <p className="muted" style={{ margin: '0.8rem 0 0' }}>
            <Translated vi="Nhấp vào một pixel trong một bản đồ.">Click a pixel in one of the maps.</Translated>
          </p>
        )
      ) : null}

      <div style={{ overflowX: 'auto', margin: '0.9rem 0 0' }} data-out="limits-wrap">
        <table className="data" style={{ margin: 0, background: 'var(--page)' }} data-out="limits"
          data-cuda={lim.cuda.ok ? 'ok' : 'over'} data-hip={lim.hip.ok ? 'ok' : 'over'}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>
                <Translated vi="lưới">grid</Translated>
              </th>
              <th className="num">
                <Translated vi="lần khởi chạy này">this launch</Translated>
              </th>
              <th>CUDA</th>
              <th>HIP</th>
            </tr>
          </thead>
          <tbody>
            <LimitRow label={<>x<br /><span className="muted"><Translated vi="bản đồ">maps</Translated></span></>} value={g.grid.x} cuda={[lim.cuda.x, CUDA_GRID.x]} hip={[g.grid.x * g.threadsPerBlock <= HIP_GRID.threads, HIP_GRID.threads, '*']} />
            <LimitRow label={<>y<br /><span className="muted"><Translated vi="ô">tiles</Translated></span></>} value={g.grid.y} cuda={[lim.cuda.y, CUDA_GRID.y]} hip={[lim.hip.y, HIP_GRID.y]} />
            <LimitRow label={<>z<br /><span className="muted"><Translated vi="mẫu">samples</Translated></span></>} value={g.grid.z} cuda={[lim.cuda.z, CUDA_GRID.z]} hip={[lim.hip.z, HIP_GRID.z]} />
            <LimitRow label={tr('threads', 'luồng')} value={g.threads} cuda={null} hip={[lim.hip.threads, HIP_GRID.threads]} />
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ margin: '0.4rem 0 0', fontSize: '0.9em' }}>
        <Translated
          vi={
            <>
              * HIP ghi trong tài liệu 2<sup>32</sup> − 1 cho trục x mà không nói con số đó đếm khối hay luồng; phần kiểm tra
              hiểu nó là luồng, cách hiểu chặt hơn. CUDA không công bố giới hạn nào cho số luồng trong một lưới.
            </>
          }
        >
          * HIP documents 2<sup>32</sup> − 1 for x without saying whether it counts blocks or threads; the check reads it as
          threads, the stricter reading. CUDA documents no cap on the threads in a grid.
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- one warp's reads

const WIDTHS = [
  {
    w: 32,
    get label() {
      return tr('32 (NVIDIA warp)', '32 (warp của NVIDIA)');
    },
  },
  {
    w: 64,
    get label() {
      return tr('64 (AMD wavefront, gfx9)', '64 (wavefront của AMD, gfx9)');
    },
  },
];
const SEGS: SegmentModelId[] = ['nvidia', 'amd64', 'amd128'];
const DIMS = { H: 18, W: 18, K: 3, tile: 16 };

function WarpReadsBench() {
  const [layout, setLayout] = useState<Layout>('NCHW');
  const [C, setC] = useState(4);
  const [N, setN] = useState(1);
  const [width, setWidth] = useState(32);
  const [seg, setSeg] = useState<SegmentModelId>('nvidia');
  const [warp, setWarp] = useState(0);
  const [cpq, setCpq] = useState({ c: 0, p: 0, q: 0 });
  const [lanes, setLanes] = useState<LaneMap>('w');
  const model = SEGMENT_MODELS[seg];
  const segBytes = model.segmentBytes!;
  const maxWarp = lanes === 'w' ? (DIMS.tile * DIMS.tile) / width - 1 : 255;
  const wsel = Math.min(warp, maxWarp);
  const c = Math.min(cpq.c, C - 1);
  const r = warpReads({
    layout, dims: { N, C, H: DIMS.H, W: DIMS.W }, K: DIMS.K, tile: DIMS.tile, warpWidth: width, warp: wsel,
    c, p: cpq.p, q: cpq.q, lanes, segmentBytes: segBytes,
  });
  const slots = segBytes / 4;
  const used = new Map<number, number[]>(); // segment -> slots used
  for (const l of r.lanes) {
    const sgi = Math.floor(l.byte / segBytes);
    const slot = (l.byte % segBytes) / 4;
    used.set(sgi, [...(used.get(sgi) ?? []), slot]);
  }
  const eff = r.access.efficiency;
  const first = r.lanes[0];
  return (
    <Bench
      title={tr("One warp's input reads", 'Các lần đọc đầu vào của một warp')}
      task={tr(
        'Find the layout in which one warp of the direct kernel reads X with the fewest segments.',
        'Tìm bố cục mà trong đó một warp của kernel trực tiếp đọc X với ít đoạn (segment) nhất.',
      )}
      foot={
        <Translated
          vi={
            <>
              Đầu vào 18 × 18 mỗi bản đồ, bộ lọc 3 × 3, nên đầu ra là 16 × 16: một khối 16 × 16 phủ trọn một bản đồ. Đơn
              vị:{' '}
              {seg === 'nvidia'
                ? 'NVIDIA: sector 32 byte trong dòng 128 byte'
                : seg === 'amd64'
                  ? 'AMD: dòng cache 64 byte'
                  : seg === 'amd128'
                    ? 'AMD: dòng cache 128 byte'
                    : 'Apple: không có tài liệu'}{' '}
              ({model.facts}). Apple không công bố quy tắc hợp nhất truy cập bộ nhớ nào (ch06-facts C1), nên không có gì ở
              đây mô tả một thiết bị Metal.
            </>
          }
        >
          Input 18 × 18 per map, 3 × 3 filters, so 16 × 16 outputs: one 16 × 16 block covers a whole map. Unit:{' '}
          {model.label} ({model.facts}). Apple documents no coalescing rule (ch06-facts C1), so nothing here describes a
          Metal device.
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Layout', 'Bố cục')}>
          {LAYOUTS.map((l) => <button key={l} aria-pressed={layout === l} onClick={() => setLayout(l)}>{l}</button>)}
        </span>
        <label className="field">
          <Translated vi={<>C (số bản đồ đầu vào) = {C}</>}>C (input maps) = {C}</Translated>
          <input type="range" min="1" max="16" value={C} aria-label="C" onChange={(e) => setC(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi={<>N (số mẫu) = {N}</>}>N (samples) = {N}</Translated>
          <input type="range" min="1" max="64" value={N} aria-label="N" onChange={(e) => setN(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select aria-label={tr('Warp width', 'Độ rộng warp')} value={width} onChange={(e) => { setWidth(+e.target.value); setWarp(0); }}>
            {WIDTHS.map((x) => <option key={x.w} value={x.w}>{x.label}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Đơn vị bộ nhớ">Memory unit</Translated>
          <select aria-label={tr('Memory unit', 'Đơn vị bộ nhớ')} value={seg} onChange={(e) => setSeg(e.target.value as SegmentModelId)}>
            {SEGS.map((x) => (
              <option key={x} value={x}>
                {tr(
                  SEGMENT_MODELS[x].label,
                  x === 'nvidia'
                    ? 'NVIDIA: sector 32 byte trong dòng 128 byte'
                    : x === 'amd64'
                      ? 'AMD: dòng cache 64 byte'
                      : x === 'amd128'
                        ? 'AMD: dòng cache 128 byte'
                        : 'Apple: không có tài liệu',
                )}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Lanes', 'Các lane')}>
          <button aria-pressed={lanes === 'w'} onClick={() => setLanes('w')}>
            <Translated vi="các lane xử lý các w kề nhau (Hình 19.7)">lanes take neighbouring w (Fig. 19.7)</Translated>
          </button>
          <button aria-pressed={lanes === 'n'} onClick={() => setLanes('n')}>
            <Translated vi="các lane xử lý các mẫu khác nhau (một thiết kế lại)">lanes take different samples (a redesign)</Translated>
          </button>
        </span>
        <label className="field">
          {lanes === 'w'
            ? tr(`warp of the block (0 to ${maxWarp})`, `warp của khối (0 đến ${maxWarp})`)
            : tr('output pixel, row-major (0 to 255)', 'pixel đầu ra, thứ tự hàng (0 đến 255)')}
          <input type="number" min={0} max={maxWarp} value={wsel} aria-label={tr('warp', 'warp')} style={{ width: '5rem' }}
            onChange={(e) => setWarp(Math.max(0, Math.min(maxWarp, Math.round(+e.target.value) || 0)))} />
        </label>
        {(['c', 'p', 'q'] as const).map((k) => (
          <label key={k} className="field">
            {k}
            <select aria-label={k} value={k === 'c' ? c : cpq[k]} onChange={(e) => setCpq({ ...cpq, [k]: +e.target.value })}>
              {Array.from({ length: k === 'c' ? C : DIMS.K }, (_, i) => <option key={i} value={i}>{i}</option>)}
            </select>
          </label>
        ))}
      </div>

      <div className="mx-label t-global">
        <Translated
          vi={
            <>
              X trong bộ nhớ: {r.access.transactions} {model.unit === 'sector' ? 'sector' : model.unit === 'line' ? 'dòng' : 'đoạn'} mà warp
              này chạm tới (ô được tô là ô được đọc)
            </>
          }
        >
          X in memory: the {r.access.transactions} {model.unit}s this warp touches (filled slots are read)
        </Translated>
      </div>
      <div className="row" style={{ gap: '4px 0', alignItems: 'center', marginBottom: '0.8rem' }} data-out="strip">
        {r.access.segments.map((sg, i) => {
          const gap = i > 0 && sg !== r.access.segments[i - 1] + 1;
          const u = new Set(used.get(sg) ?? []);
          return (
            <span key={sg} style={{ display: 'inline-flex', alignItems: 'center' }}>
              {gap ? <span className="muted" style={{ padding: '0 6px' }} title={tr('addresses skipped', 'các địa chỉ bị bỏ qua')}>⋯</span> : null}
              <span data-seg={sg} title={tr(`bytes ${sg * segBytes} to ${sg * segBytes + segBytes - 1}`, `byte ${sg * segBytes} đến ${sg * segBytes + segBytes - 1}`)}
                style={{ display: 'inline-flex', border: '1px solid var(--global)', marginLeft: gap || i === 0 ? 0 : 3, background: 'var(--cell)' }}>
                {Array.from({ length: slots }, (_, k) => (
                  <i key={k} style={{ width: slots > 16 ? 5 : 9, height: 16, background: u.has(k) ? 'var(--global)' : 'transparent', borderRight: k < slots - 1 ? '1px solid var(--rule-soft)' : undefined }} />
                ))}
              </span>
            </span>
          );
        })}
      </div>
      <div className="cols">
        <div className="panel" data-out="warp" data-segments={r.access.transactions} data-lanes={r.lanes.length} data-runs={r.runs}
          data-stride={r.stride ?? ''} data-efficiency={eff}>
          <p style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  {r.lanes.length} lane đang hoạt động yêu cầu {r.access.bytesNeeded} byte. Chúng chạm tới{' '}
                  <b>{r.access.transactions}</b> {model.unit === 'sector' ? 'sector' : model.unit === 'line' ? 'dòng' : 'đoạn'},{' '}
                  {r.access.bytesFetched} byte: {Math.round(eff * 100)}% những gì được nạp về được dùng.{' '}
                </>
              }
            >
              {r.lanes.length} active lanes ask for {r.access.bytesNeeded} bytes. They touch <b>{r.access.transactions}</b>{' '}
              {model.unit}s, {r.access.bytesFetched} bytes: {Math.round(eff * 100)}% of what is fetched is used.{' '}
            </Translated>
            {eff >= 0.99 ? (
              <span className="chip ok">
                <Translated vi="được dùng hết">fully used</Translated>
              </span>
            ) : eff < 0.5 ? (
              <span className="chip fail">
                <Translated vi="phần lớn bị lãng phí">mostly wasted</Translated>
              </span>
            ) : null}
          </p>
          <p style={{ margin: '0.5rem 0 0' }} className="muted">
            <Translated
              vi={
                <>
                  {r.stride != null ? <>Các lane kề nhau cách nhau {r.stride} byte; </> : null}
                  {r.runs} dải địa chỉ liên tiếp riêng biệt.
                  {first ? <> Lane 0 đọc X[{first.n}, {c}, {first.h + cpq.p}, {first.w + cpq.q}] tại phần tử {first.index}.</> : null}
                </>
              }
            >
              {r.stride != null ? <>Neighbouring lanes are {r.stride} bytes apart; </> : null}
              {r.runs} separate run{r.runs === 1 ? '' : 's'} of consecutive addresses.
              {first ? <> Lane 0 reads X[{first.n}, {c}, {first.h + cpq.p}, {first.w + cpq.q}] at element {first.index}.</> : null}
            </Translated>
          </p>
        </div>
        <div className="panel">
          <p style={{ margin: 0 }} data-out="f">
            <Translated
              vi={
                <>
                  F: mọi lane đều đọc <code>F[0, {c}, {cpq.p}, {cpq.q}]</code>, phần tử {r.fIndex}. Một địa chỉ cho cả warp,
                  nên nó tốn một {model.unit === 'sector' ? 'sector' : model.unit === 'line' ? 'dòng' : 'đoạn'}, bất kể bố cục
                  của X.
                </>
              }
            >
              F: every lane reads <code>F[0, {c}, {cpq.p}, {cpq.q}]</code>, element {r.fIndex}. One address for the whole warp,
              so it costs one {model.unit}, whatever the layout of X.
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the corrected kernel

function useLabStatusText() {
  const { reports, loading } = useReports();
  if (loading) return { cuda: tr('reading reports', 'đang đọc báo cáo'), metal: tr('reading reports', 'đang đọc báo cáo') };
  // The labs page's words (LabKit's solutionTabStatus), so a failed run reads as failed here too.
  return { cuda: solutionTabStatus(reports, LAB, 'kernel.cu'), metal: solutionTabStatus(reports, LAB, 'kernel.metal') };
}

function DirectKernel() {
  const cu = useLabFile(SOL + '/kernel.cu');
  const metal = useLabFile(SOL + '/kernel.metal');
  const mm = useLabFile(SOL + '/launch.mm');
  const status = useLabStatusText();
  if (!cu || !metal || !mm) return null;
  const parts = [cu, metal, mm].map((t) => lessonRegion(t, 'direct'));
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: parts[0].code, tiers: CONV_TIERS, status: status.cuda },
        { label: tr('Metal kernel', 'Metal kernel'), code: parts[1].code, tiers: CONV_TIERS, status: status.metal },
        { label: tr('Metal host', 'Metal host'), code: parts[2].code, tiers: {}, status: status.metal },
      ]}
      caption={
        <Translated
          vi={
            <>
              Kernel trực tiếp và lần khởi chạy nó từ lời giải tham chiếu (solution) của lab 19.1, được nạp từ{' '}
              <code>labs/{SOL}/</code>
              {parts.every((x) => x.found) ? '' : ' (không tìm thấy một dấu đánh dấu vùng, nên cả tệp được hiển thị)'}. Hãy
              thử nó trong <a href={href('ch19', 'labs')}>lab</a> trước. Bản dựng HIP được thiết lập để nhận nguyên tệp CUDA
              qua <code>pmpp/vendors/hip.h</code>; nó chưa được biên dịch hay chạy trên phần cứng AMD ở đây. Trạng thái bên
              cạnh các tab được đọc từ các lần chạy lab đã ghi lại trên máy này.
            </>
          }
        >
          The direct kernel and its launch from lab 19.1&rsquo;s reference solution, loaded from{' '}
          <code>labs/{SOL}/</code>
          {parts.every((x) => x.found) ? '' : ' (a region marker was not found, so a whole file is shown)'}. Try it in the{' '}
          <a href={href('ch19', 'labs')}>lab</a> first. The HIP build is set up to take the CUDA file unchanged through <code>pmpp/vendors/hip.h</code>; it has not been compiled or run on AMD hardware here. The status beside the
          tabs is read from this machine&rsquo;s recorded lab runs.
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- the page

export function S192() {
  return (
    <>
      <h2>
        <Translated vi="Những vòng lặp nào có thể chạy song song">Which loops can run in parallel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hãy xem lại Hình 19.4, bảy vòng lặp của lớp chạy theo batch. Một số vòng chuyển sang một phần tử khác của Y ở
            mỗi lần lặp; số khác thì tiếp tục cộng dồn vào phần tử đã chọn. Hãy phân loại chúng trước khi đọc tiếp: chỉ loại
            thứ nhất mới có thể giao nguyên trạng cho các luồng riêng biệt.
          </p>
        }
      >
        <p>
          Look at Fig. 19.4 again, the seven loops of the batched layer. Some of them move to a different element of Y on
          each iteration; others keep adding into the element already chosen. Sort them before reading on: only the first
          kind can be handed to separate threads as it stands.
        </p>
      </Translated>
      <Source section="19.2" pages="460" figs={tr('Fig. 19.4', 'Hình 19.4')} />
      <LoopSort />
      <Translated
        vi={
          <p>
            Bốn vòng ngoài, n, m, h và w, mỗi vòng chọn một phần tử khác nhau của Y để ghi, nên các lần lặp của chúng có
            thể chạy theo bất kỳ thứ tự nào, trên bất kỳ luồng nào. Ba vòng trong, c, p và q, đều cộng vào cùng một phần
            tử. Chúng cũng có thể được song song hóa, nhưng khi đó mọi luồng đều phải đọc, cộng vào và ghi một phần tử Y
            dùng chung. Sách giữ chúng tuần tự trừ khi thực sự cần thêm độ song song.
          </p>
        }
      >
        <p>
          The four outer loops, n, m, h and w, each pick a different element of Y to write, so their iterations can run in
          any order, on any thread. The three inner ones, c, p and q, all add into the same element. They could be
          parallelised too, but every thread would then be reading, adding to and writing one shared Y element. The book
          keeps them serial unless more parallelism is really needed.
        </p>
      </Translated>
      <Ask q={questionsS192['ch19/19.2/q-loops']} />
      <Translated
        vi={
          <p>
            Như vậy vẫn còn N · M · H_out · W_out đầu ra độc lập. Với lớp C1 của LeNet ở batch 1, con số đó là 6 · 28 · 28
            = 4704; với một lớp có 64 bản đồ đầu ra 56 × 56 ở batch 32, nó là hơn sáu triệu.
          </p>
        }
      >
        <p>
          That still leaves N · M · H_out · W_out independent outputs. For LeNet&rsquo;s C1 at batch 1 that is 6 · 28 · 28
          = 4704; for a layer with 64 output maps of 56 × 56 at batch 32 it is over six million.
        </p>
      </Translated>
      <Source section="19.2" pages="460" />

      <h2>
        <Translated vi="Một luồng cho mỗi pixel đầu ra">One thread per output pixel</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Thiết kế: một luồng tính một phần tử đầu ra. Các luồng tạo thành những khối vuông{' '}
            <code>TILE_WIDTH × TILE_WIDTH</code>, 256 luồng với giá trị 16 của sách, và mỗi khối tính một ô của một bản đồ
            đầu ra. Như vậy có bốn thứ cần trải lên một lưới có ba trục: mẫu, bản đồ đầu ra, và hàng cùng cột của ô.
          </p>
        }
      >
        <p>
          The design: one thread computes one output element. Threads form square blocks of{' '}
          <code>TILE_WIDTH × TILE_WIDTH</code>, 256 of them at the book&rsquo;s 16, and each block computes one tile of one
          output map. That leaves four things to spread over a grid with three axes: the sample, the output map, and the
          tile&rsquo;s row and column.
        </p>
      </Translated>
      <Source section="19.2" pages="460" />
      <Translated
        vi={
          <p>
            Sách dùng x cho bản đồ đầu ra m và z cho mẫu n. Hai tọa độ của ô lẽ ra sẽ lấy y và một trục khác, nhưng x và z
            đã được dùng, nên chúng chia nhau y: ô ở hàng <code>r</code> và cột <code>c</code>{' '}
            trở thành <code>blockIdx.y = r · W_grid + c</code>, theo thứ tự hàng (row-major), trong đó W_grid là số ô theo
            chiều ngang của một bản đồ. Trong ví dụ của Hình 19.6, M = 4 bản đồ, mỗi bản đồ 2 × 2 ô, cho T = 4 ô mỗi bản đồ
            và một lần khởi chạy <code>gridDim(4, 4, N)</code>.
          </p>
        }
      >
        <p>
          The book uses x for the output map m and z for the sample n. The two tile coordinates would naturally take y and
          another axis, but x and z are gone, so they share y: the tile at row <code>r</code> and column <code>c</code>{' '}
          becomes <code>blockIdx.y = r · W_grid + c</code>, row-major, where W_grid is the number of tiles across a map. In
          Fig. 19.6&rsquo;s example, M = 4 maps of 2 × 2 tiles give T = 4 tiles per map and a launch of{' '}
          <code>gridDim(4, 4, N)</code>.
        </p>
      </Translated>
      <Source section="19.2" pages="460–462" figs={tr('Figs. 19.5, 19.6', 'Hình 19.5, 19.6')} />

      <h2>
        <Translated vi="Lấy lại (h, w)">Getting (h, w) back</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bên trong kernel, mỗi luồng tháo ngược việc đóng gói. Phép chia nguyên cho W_grid cho ra hàng của ô và phần dư
            cho ra cột của nó; nhân mỗi giá trị với <code>TILE_WIDTH</code> rồi cộng <code>threadIdx.y</code> hoặc{' '}
            <code>threadIdx.x</code>. Chia để lấy hàng, lấy phần dư để có cột: đổi chỗ hai phép này thì các ô rơi vào sai
            chỗ, nhưng không phải lúc nào cũng thấy được, như trình khám phá cho thấy.
          </p>
        }
      >
        <p>
          Inside the kernel each thread undoes the packing. Integer division by W_grid gives the tile&rsquo;s row and the
          remainder gives its column; multiply each by <code>TILE_WIDTH</code> and add <code>threadIdx.y</code> or{' '}
          <code>threadIdx.x</code>. Divide for the row, take the remainder for the column: swap the two and the tiles land
          in the wrong places, but not always visibly, as the explorer shows.
        </p>
      </Translated>
      <Source section="19.2" pages="463" figs={tr('Fig. 19.7', 'Hình 19.7')} />
      <Translated
        vi={
          <p>
            Hình 19.5 tính W_grid và H_grid là <code>W_out / TILE_WIDTH</code> và <code>H_out / TILE_WIDTH</code>, bằng số
            học số nguyên, và không nói gì về các kích thước không phải bội số của ô.
          </p>
        }
      >
        <p>
          Fig. 19.5 computes W_grid and H_grid as <code>W_out / TILE_WIDTH</code> and <code>H_out / TILE_WIDTH</code>, in
          integer arithmetic, and says nothing about sizes that are not multiples of the tile.
        </p>
      </Translated>
      <Source section="19.2" pages="461" figs={tr('Fig. 19.5', 'Hình 19.5')} />
      <Predict
        id="ch19/19.2/p-cover"
        prompt={
          <Translated
            vi={
              <>
                Lớp C1 của LeNet có các bản đồ đầu ra 28 × 28. Dùng Hình 19.5 đúng như được in với <code>TILE_WIDTH</code> 16.
                Bao nhiêu trong 784 pixel của mỗi bản đồ đầu ra được tính?
              </>
            }
          >
            LeNet&rsquo;s C1 has 28 × 28 output maps. Use Fig. 19.5 exactly as printed with <code>TILE_WIDTH</code> 16.
            How many of the 784 pixels of each output map get computed?
          </Translated>
        }
        unit={tr('pixels', 'pixel')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                <b>256.</b> W_grid = H_grid = 28 / 16 = 1 trong số học số nguyên, nên mỗi bản đồ được một ô 16 × 16 và 528
                pixel còn lại, một dải hình chữ L dọc theo cạnh phải và cạnh dưới, không bao giờ được ghi.{' '}
                {v === 256 ? 'Chính xác.' : 'Hãy chọn cấu hình LeNet trong trình khám phá và nhìn vào các bản đồ.'} Không có gì
                thất bại một cách ồn ào: Y chỉ giữ nguyên những gì đã có trong bộ nhớ.
              </p>
            }
          >
            <p>
              <b>256.</b> W_grid = H_grid = 28 / 16 = 1 in integer arithmetic, so each map gets one 16 × 16 tile and the
              other 528 pixels, an L-shaped band along the right and bottom, are never written.{' '}
              {v === 256 ? 'Exactly.' : 'Choose the LeNet preset in the explorer and look at the maps.'} Nothing fails
              loudly: Y just keeps whatever was in memory.
            </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá chạy Hình 19.5 và 19.7 trên bất kỳ hình dạng nào bạn đưa vào. Nhấp vào một khối trong lưới để
            xem ô mà nó sở hữu, rồi một luồng để xem phép tính cho ra pixel của nó; ở chế độ ngược, hãy nhấp vào một pixel
            và nói luồng nào tính nó trước khi xem đáp án.
          </p>
        }
      >
        <p>
          The explorer runs Figs. 19.5 and 19.7 on any shape you give it. Click a block in the grid to see the tile it
          owns, then a thread to see the arithmetic that gives its pixel; in reverse mode, click a pixel and say which
          thread computes it before revealing the answer.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ có chỉ số, không có thời gian: luồng nào tính pixel nào và mỗi lần ghi rơi vào đâu. Một luồng đi lạc
              không có kiểm tra biên cũng đọc X bên ngoài vùng của nó; mô hình chỉ theo dõi lần ghi của nó. Độ phủ được mô
              phỏng cho tối đa hai triệu luồng mỗi mẫu.
            </p>
          }
        >
          <p>
            Indices only, no timing: which thread computes which pixel and where each store lands. A stray thread with no
            bounds check also reads X outside its patch; the model tracks only its store. Coverage is simulated for up to
            two million threads per sample.
          </p>
        </Translated>
      </Simplification>
      <GridMap />
      <Translated
        vi={
          <p>
            Có ba điều cần tìm trong đó. Với mã host của sách và cấu hình LeNet, phần lớn mỗi bản đồ không bao giờ được
            tính. Với phép chia làm tròn lên nhưng không có kiểm tra biên, các luồng vượt quá cạnh phải của một bản đồ ghi
            vào đầu hàng kế tiếp, các luồng vượt quá cạnh dưới ghi vào bản đồ kế tiếp, và các luồng đi lạc của bản đồ cuối
            ghi vượt quá cuối Y. Còn lỗi cài sẵn thì không thay đổi gì bạn có thể thấy trên lưới ô vuông của Hình 19.6,
            rồi làm hỏng cấu hình hình chữ nhật.
          </p>
        }
      >
        <p>
          Three things to find in it. With the book&rsquo;s host code and the LeNet preset, most of each map is never
          computed. With ceiling division but no bounds check, the threads past the right edge of a map store into the
          start of the next row, those past the bottom edge into the next map, and the last map&rsquo;s strays past the
          end of Y. And the planted bug changes
          nothing you can see on Fig. 19.6&rsquo;s square grid of tiles, then breaks the rectangular preset.
        </p>
      </Translated>
      <Ask q={questionsS192['ch19/19.2/q-mapping']} />
      <Ask q={questionsS192['ch19/19.2/q-swap']} />

      <h2>
        <Translated vi="Kernel, đã sửa">The kernel, corrected</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hình 19.7 là kernel, viết bằng chỉ số nhiều chiều dưới dạng mã giả. Có hai vấn đề trên trang sách trở nên quan
            trọng khi bạn viết nó thật. Chữ ký của nó truyền C, W_grid và K nhưng không truyền H, W hay M, những thứ các chỉ
            số phẳng cần. Và nó không có kiểm tra biên, điều này cùng với phép chia nguyên của Hình 19.5 có nghĩa là nó chỉ
            đúng khi cả hai cạnh đầu ra đều là bội số của ô.
          </p>
        }
      >
        <p>
          Fig. 19.7 is the kernel, written with multi-dimensional indices as pseudo-code. Two problems on the page matter
          once you write it for real. Its signature passes C, W_grid and K but not H, W or M, which the flat indices need.
          And it has no bounds check, which together with Fig. 19.5&rsquo;s integer division means it is correct only when
          both output sides are multiples of the tile.
        </p>
      </Translated>
      <Source section="19.2" pages="461–462" figs={tr('Figs. 19.5, 19.7', 'Hình 19.5, 19.7')} />
      <DirectKernel />
      <p>
        <Translated vi="Những chỗ kernel và lần khởi chạy của lab khác với trang sách:">
          Where the lab&rsquo;s kernel and launch differ from the page:
        </Translated>
      </p>
      <Translated
        vi={
          <ul>
            <li>Mã host làm tròn W_grid và H_grid lên, nên các ô phủ toàn bộ bản đồ, kể cả 28 × 28.</li>
            <li>
              Kernel còn nhận H, W và M, những thứ các chỉ số phẳng cần, và một luồng có pixel không tồn tại thì không làm
              gì cả.
            </li>
            <li>
              <code>X[n, c, h + p, w + q]</code> và các chỉ số khác được viết ra thành các offset phẳng theo thứ tự hàng,
              như ở chương 3.
            </li>
            <li>
              Lần khởi chạy đặt tên các biến <code>dim3</code> của nó là <code>dimBlock</code> và <code>dimGrid</code>.
              Các tên <code>blockDim</code> và <code>gridDim</code> của trang sách sẽ che khuất các biến dựng sẵn cùng tên
              của CUDA bên trong hàm đó.
            </li>
            <li>
              Kernel Metal nhận vị trí threadgroup và vị trí luồng làm tham số <code>uint3</code>, nhận các kích thước
              trong một struct nhỏ, và được khởi chạy bằng <code>dispatchThreadgroups</code> trên cùng lưới đã làm tròn
              lên.
            </li>
          </ul>
        }
      >
        <ul>
          <li>The host rounds W_grid and H_grid up, so the tiles cover the whole map, 28 × 28 included.</li>
          <li>
            The kernel also receives H, W and M, which the flat indices need, and a thread whose pixel does not exist does
            no work at all.
          </li>
          <li>
            <code>X[n, c, h + p, w + q]</code> and the other indices are written out as flat row-major offsets, as in
            chapter 3.
          </li>
          <li>
            The launch names its <code>dim3</code> variables <code>dimBlock</code> and <code>dimGrid</code>. The
            page&rsquo;s <code>blockDim</code> and <code>gridDim</code> would hide CUDA&rsquo;s built-in variables of the
            same names inside that function.
          </li>
          <li>
            The Metal kernel takes its threadgroup and thread positions as <code>uint3</code> parameters, receives the
            sizes in a small struct, and is launched with <code>dispatchThreadgroups</code> on the same rounded-up grid.
          </li>
        </ul>
      </Translated>
      <Translated
        vi={
          <p>
            Có một điều trang sách làm hoàn toàn đúng: mỗi luồng cộng các tích của nó vào một biến cục bộ, <code>acc</code>,
            nằm trong một thanh ghi (register), và ghi Y một lần ở cuối. Không luồng nào khác chạm vào phần tử đó, nên không
            cần phép toán nguyên tử hay ghi lặp lại.
          </p>
        }
      >
        <p>
          One thing the page gets exactly right: each thread adds its products into a local variable, <code>acc</code>,
          which lives in a register, and writes Y once at the end. No other thread touches that element, so no atomics
          and no repeated stores are needed.
        </p>
      </Translated>
      <Source section="19.2" pages="463" />

      <h2>
        <Translated vi="Cái giá của nó">What it costs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đếm một đầu ra: C · K² phép nhân-cộng, và với mỗi phép, một lần nạp từ X và một từ F. Tức là 2 FLOP cho 8 byte,
            0.25 FLOP mỗi byte, bất kể hình dạng của lớp: cùng tỉ số với kernel tích chập (convolution) cơ bản của chương
            7 và phép nhân ma trận ngây thơ của chương 5. Kết luận của sách là kernel này có rất nhiều độ song song nhưng
            tiêu tốn quá nhiều băng thông bộ nhớ toàn cục.
          </p>
        }
      >
        <p>
          Count one output: C · K² multiply-adds, and for each, one load from X and one from F. That is 2 FLOP for 8 bytes,
          0.25 FLOP per byte, whatever the layer&rsquo;s shape: the same ratio as chapter 7&rsquo;s basic convolution
          kernel and chapter 5&rsquo;s naive matrix multiplication. The book&rsquo;s verdict is that the kernel has plenty
          of parallelism but consumes too much global memory bandwidth.
        </p>
      </Translated>
      <Source section="19.2" pages="463" />
      <Ask q={questionsS192['ch19/19.2/q-ratio']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Sự lãng phí đến từ đâu? Các luồng kề nhau đọc những vùng chồng lấn nhau, và bộ nhớ đệm (cache) có thể phục vụ
            một phần những lần đọc lặp lại đó. Nhưng mọi khối tính một bản đồ đầu ra khác lại đọc đúng đầu vào đó thêm lần
            nữa, và bộ nhớ chia sẻ không giúp được, vì nó thuộc về một khối.
          </p>
        }
      >
        <p>
          Where does the waste come from? Neighbouring threads read overlapping patches, and caches may serve some of those
          repeats. But every block that computes another output map reads the very same input again, and shared memory
          cannot help, because it belongs to one block.
        </p>
      </Translated>
      <Source section="19.2" pages="463" />
      <Translated
        vi={
          <p>
            Hoạt cảnh đếm cả hai loại trên ví dụ của sách: ba bản đồ đầu vào 3 × 3, bộ lọc 2 × 2, hai bản đồ đầu ra, và các
            ô gồm 2 × 2 luồng.
          </p>
        }
      >
        <p>
          The animation counts both kinds on the book&rsquo;s example: three 3 × 3 input maps, 2 × 2 filters, two output
          maps, and tiles of 2 × 2 threads.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Đếm các yêu cầu đọc, mỗi luồng một yêu cầu cho mỗi bước của vòng lặp c, p, q. Các khối chạy lần lượt từng
              khối cho rõ ràng. Không có cache, không có hợp nhất truy cập, không có thời gian.
            </p>
          }
        >
          <p>
            Counts read requests, one per thread per step of the c, p, q loop. The blocks run one after the other for
            clarity. No caches, no coalescing, no timing.
          </p>
        </Translated>
      </Simplification>
      <DirectReadsScene />
      <Ask q={questionsS192['ch19/19.2/q-reuse']} />
      <Translated
        vi={
          <p>
            Phân ô (tiling) đầu vào của một (n, m) trong bộ nhớ chia sẻ, như chương 7 đã làm, sẽ loại bỏ loại lặp lại thứ
            nhất và được sách để lại làm bài tập; mục <a href={href('ch07', '7.5')}>7.5</a> có kỹ thuật đó. Loại thứ hai
            cần một thiết kế hoàn toàn khác, chính là mục 19.3.
          </p>
        }
      >
        <p>
          Tiling the input of one (n, m) in shared memory, as chapter 7 did, would remove the first kind of repeat and is
          left as an exercise in the book; section <a href={href('ch07', '7.5')}>7.5</a> has the technique. The second kind
          needs a different design altogether, which is section 19.3.
        </p>
      </Translated>
      <Source section="19.2" pages="463" />

      <h2>
        <Translated vi="Bài tập 2: bố cục có quan trọng không?">Exercise 2: does the layout matter?</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chương này lưu X dưới dạng N × C × H × W. Bài tập 2 hỏi liệu N × H × W × C có thể giảm băng thông bộ nhớ không,
            và C × H × W × N có thể mang lại gì. Một bố cục không thay đổi số phần tử nào; nó thay đổi những phần tử nào nằm
            cạnh nhau, và do đó các lần nạp của một warp kết hợp tốt đến đâu. Hãy trả lời cho kernel này bằng cách nhìn vào
            một warp.
          </p>
        }
      >
        <p>
          The chapter stores X as N × C × H × W. Exercise 2 asks whether N × H × W × C could reduce memory bandwidth, and
          what C × H × W × N might offer. A layout changes no count of elements; it changes which of them sit next to each
          other, and so how well one warp&rsquo;s loads combine. Answer it for this kernel by looking at one warp.
        </p>
      </Translated>
      <Source section="19.6" pages="475" figs={tr('Exercise 2', 'Bài tập 2')} />
      <Translated
        vi={
          <p>
            Trình khám phá lấy một warp của kernel trực tiếp tại một bước (c, p, q) và trải ra trong bộ nhớ các địa chỉ X
            mà các lane của nó đọc, đếm theo đơn vị bộ nhớ mà NVIDIA hoặc AMD công bố trong tài liệu. Hãy bắt đầu với NCHW,
            rồi chuyển sang NHWC và tăng C.
          </p>
        }
      >
        <p>
          The explorer takes one warp of the direct kernel at one step (c, p, q) and lays out in memory the X addresses
          its lanes read, counted in the memory unit NVIDIA or AMD documents. Start with NCHW, then switch to NHWC and
          raise C.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một lệnh nạp của một warp, đếm theo đơn vị được nhà sản xuất công bố. Không có cache: một đoạn mà một lệnh
              nạp về không được lệnh kế tiếp ghi nhớ. Các lane là các luồng của khối theo thứ tự hàng.
            </p>
          }
        >
          <p>
            One load instruction of one warp, counted in the vendor&rsquo;s documented unit. No cache: a segment one
            instruction fetches is not remembered by the next. Lanes are the block&rsquo;s threads in row-major order.
          </p>
        </Translated>
      </Simplification>
      <WarpReadsBench />
      <Ask q={questionsS192['ch19/19.2/q-layout']} />
      <Translated
        vi={
          <p>
            Các thư viện chọn bố cục vì nhiều lý do hơn là các lần nạp của kernel này. Các đơn vị ma trận của NVIDIA muốn
            kênh nằm cuối, và thư viện của AMD hoàn toàn không nhận bố cục CHWN của Bài tập 2.
          </p>
        }
      >
        <p>
          Libraries choose a layout for more reasons than this kernel&rsquo;s loads. NVIDIA&rsquo;s matrix units want
          channels last, and AMD&rsquo;s library does not take Exercise 2&rsquo;s CHWN at all.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, A5, B3)">{/* i18n-ignore: date and fact citation */}
        <Translated
          vi={
            <p>
              cuDNN: &ldquo;Tensor Cores require that the tensors be in the NHWC data layout&rdquo; (Tensor Cores yêu cầu
              các tensor ở bố cục dữ liệu NHWC), và nó chỉ chuyển đổi giữa NCHW và NHWC khi chúng được dùng. MIOpen đánh dấu
              NCHW và NHWC được hỗ trợ đầy đủ và CHWN là &ldquo;Not supported&rdquo; (không được hỗ trợ).
            </p>
          }
        >
          <p>
            cuDNN: &ldquo;Tensor Cores require that the tensors be in the NHWC data layout&rdquo;, and it converts between
            NCHW and NHWC only when they are used. MIOpen marks NCHW and NHWC fully supported and CHWN &ldquo;Not
            supported&rdquo;.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Về HIP và Metal">On HIP and Metal</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel CUDA được thiết lập để dựng cho HIP mà không cần sửa qua <code>pmpp/vendors/hip.h</code> (có trong tài
            liệu, ch19-facts B1; nó chưa được biên dịch hay chạy trên phần cứng AMD ở đây), vì nó chỉ dùng{' '}
            <code>__global__</code>, <code>blockIdx</code>, <code>threadIdx</code>, <code>dim3</code> và cú pháp khởi
            chạy ba dấu ngoặc nhọn, những thứ HIP viết y hệt. Điều đó không đúng với mọi tệp CUDA: các intrinsic warp{' '}
            <code>_sync</code>, cooperative groups, các header bán chính xác (half-precision), functor của CUB và intrinsic
            của lõi ma trận đều khác trên HIP, và <code>__dp4a</code> không có bản tương ứng trên HIP. Một khối 16 × 16 là
            bốn wavefront 64 lane trên các thiết bị gfx9, tám wavefront 32 lane trên gfx10 trở về sau.
          </p>
        }
      >
        <p>
          The CUDA kernel is set up to build for HIP unchanged through <code>pmpp/vendors/hip.h</code> (documented,
          ch19-facts B1; it has not been compiled or run on AMD hardware here), because it uses only{' '}
          <code>__global__</code>, <code>blockIdx</code>, <code>threadIdx</code>, <code>dim3</code> and the
          triple-angle-bracket launch, which HIP spells the same way. That is not true of every CUDA file: warp{' '}
          <code>_sync</code> intrinsics, cooperative groups, half-precision headers, CUB functors and matrix-core intrinsics
          all differ on HIP, and <code>__dp4a</code> has no HIP counterpart. A 16 × 16 block is four wavefronts of 64 on gfx9 devices, eight of 32 on gfx10 and later.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Các thông tin về HIP và Metal ở đây lấy từ notes/today/ch19-facts.md (B1, B2, C1) và
              ch05-facts.md (B5, C4, C5).
            </p>
          }
        >
          <p>
            The book uses CUDA only. HIP and Metal facts here come from notes/today/ch19-facts.md (B1, B2, C1) and
            ch05-facts.md (B5, C4, C5).
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Metal cần viết lại phần chỉ số và lần khởi chạy. Kernel nhận vị trí khối và vị trí luồng làm tham số{' '}
            <code>uint3</code> được đánh dấu <code>[[threadgroup_position_in_grid]]</code> và{' '}
            <code>[[thread_position_in_threadgroup]]</code>. Host có thể dispatch trọn các threadgroup bằng{' '}
            <code>dispatchThreadgroups</code>, cách này cần cùng phép chia làm tròn lên và kiểm tra biên như CUDA. Hoặc nó
            có thể dùng <code>dispatchThreads</code>, cách này định kích thước lưới theo luồng và để Metal tạo các
            threadgroup không đầy đủ ở các cạnh trên GPU Apple4 trở về sau; cách đó hợp với một lưới mà các trục chính là
            tọa độ pixel, chứ không phải chỉ số ô được đóng gói của Hình 19.6.
          </p>
        }
      >
        <p>
          Metal needs a rewrite of the indices and the launch. The kernel receives its block and thread positions as{' '}
          <code>uint3</code> parameters marked <code>[[threadgroup_position_in_grid]]</code> and{' '}
          <code>[[thread_position_in_threadgroup]]</code>. The host can dispatch whole threadgroups with{' '}
          <code>dispatchThreadgroups</code>, which needs the same ceiling division and bounds check as CUDA. Or it can use{' '}
          <code>dispatchThreads</code>, which sizes the grid in threads and lets Metal make partial threadgroups at the
          edges on Apple4 and later GPUs; that fits a grid whose axes are the pixel coordinates themselves, not Fig.
          19.6&rsquo;s packed tile index.
        </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr>
            <th>
              <Translated vi="Giới hạn lưới">Grid limit</Translated>
            </th>
            <th>CUDA</th>
            <th>HIP</th>
            <th>Metal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>x</td>
            <td>
              <Translated vi={<>2<sup>31</sup> − 1 khối</>}>
                2<sup>31</sup> − 1 blocks
              </Translated>
            </td>
            <td>
              <Translated vi={<>2<sup>32</sup> − 1; con số đó đếm khối hay luồng thì unverified</>}>
                2<sup>32</sup> − 1; whether that counts blocks or threads is unverified
              </Translated>
            </td>
            <td rowSpan={2}>
              <Translated
                vi={
                  <>
                    không có tài liệu về mức tối đa của lưới tính toán (unverified); các vị trí là <code>uint</code> 32-bit
                  </>
                }
              >
                no compute-grid maximum documented (unverified); positions are 32-bit <code>uint</code>
              </Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="y và z">y and z</Translated>
            </td>
            <td>65,535</td>
            <td>2<sup>16</sup> − 1 = 65,535</td>
          </tr>
          <tr>
            <td>
              <Translated vi="số luồng trong một lưới">threads in one grid</Translated>
            </td>
            <td>
              <Translated vi="không có giới hạn trong tài liệu">no cap documented</Translated>
            </td>
            <td>
              <Translated vi={<>tối đa 2<sup>32</sup> − 1</>}>
                at most 2<sup>32</sup> − 1
              </Translated>
            </td>
            <td>
              <Translated vi="không có tài liệu">not documented</Translated>
            </td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Với kernel này, giới hạn y là thứ cần để ý: nó chặn số ô mỗi bản đồ, và một đầu ra 4096 × 4096 duy nhất với ô
            16 × 16 cần 65,536 ô, thừa đúng một. Trên HIP, tích N · M · T · 256 cũng phải nhỏ hơn 2<sup>32</sup>. Hãy nạp
            cấu hình cuối cùng trong trình khám phá để xem cả hai phép kiểm tra.
          </p>
        }
      >
        <p>
          For this kernel the y limit is the one to watch: it bounds the tiles per map, and a single 4096 × 4096 output in
          16 × 16 tiles needs 65,536 of them, one too many. On HIP, the product N · M · T · 256 must also stay below
          2<sup>32</sup>. Load the last preset in the explorer to see both checks.
        </p>
      </Translated>
      <Source section="19.2" pages="461" />

      <h2>
        <Translated vi="Nơi kernel trực tiếp vẫn còn tồn tại">Where the direct kernel survives</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Backend CUDA của llama.cpp, cũng chính là backend HIP của nó, vẫn có một kernel tích chập trực tiếp. Nó chỉ gửi
            một lớp tới đó khi lớp đó nhỏ ở cả hai mặt: ít hơn 64 số hạng mỗi đầu ra (số kênh đầu vào nhân diện tích bộ lọc)
            và ít hơn 512 pixel đầu ra. Các lớp lớn hơn đi tới một kernel implicit-GEMM, chủ đề của mục 19.3, và tích chập
            1 × 1 đi thẳng tới cuBLAS. Đó là lập luận của sách về các ma trận nhỏ (p. 472), được viết thành một dòng mã
            production.
          </p>
        }
      >
        <p>
          llama.cpp&rsquo;s CUDA backend, which is also its HIP backend, still has a direct convolution kernel. It sends a
          layer there only when it is small on both counts: fewer than 64 terms per output (input channels times filter
          area) and fewer than 512 output pixels. Larger layers go to an implicit-GEMM kernel, the subject of section 19.3,
          and 1 × 1 convolutions straight to cuBLAS. That is the book&rsquo;s argument about small matrices (p. 472), written
          as one line of production code.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch19-facts.md, D1)">{/* i18n-ignore: date and fact citation */}
        <Translated
          vi={
            <p>
              <code>ggml/src/ggml-cuda/conv2d.cu</code>: <code>small_conv = IC * KW * KH &lt; 64 &amp;&amp; OW * OH &lt; 512</code>.
              Số dòng sẽ thay đổi theo thời gian.
            </p>
          }
        >
          <p>
            <code>ggml/src/ggml-cuda/conv2d.cu</code>: <code>small_conv = IC * KW * KH &lt; 64 &amp;&amp; OW * OH &lt; 512</code>.
            Line numbers will drift.
          </p>
        </Translated>
      </Today>
    </>
  );
}
