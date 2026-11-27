import type { Question } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  ReferenceSolution,
  RunningLabs,
  VerificationStatus,
  benchRow,
  defineLabs,
  measure,
  msCol,
  numCol,
  preferredBench,
  rawCol,
  useLabs,
  type Col,
} from '../../components/LabKit';
import { fmt, num, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 7's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch07', [
  { num: '7.1', dir: 'lab1_constconv', name: 'constant filter', vi: { name: 'bộ lọc hằng' } },
  { num: '7.2', dir: 'lab2_tiledconv', name: 'tiled with halos', vi: { name: 'phân ô với vùng biên' } },
]);
const [l1, l2] = LABS;
// Lab 5.1 measured each device's bandwidth: the model ceilings of lab 7.1 are built on it.
const [l51] = defineLabs('ch05', [{ num: '5.1', dir: 'lab1_roofline', name: 'roofline' }]);

// Identifier colours per lab. In lab 7.1 `F` is a global pointer in the basic kernel and a
// constant buffer in Metal's conv_const, so it is left uncoloured; `F_c` and the by-value
// struct `f` are constant memory (CUDA passes kernel arguments through it, ch07-facts A4).
const REG: Record<string, Tier> = {
  Pvalue: 'reg', outRow: 'reg', outCol: 'reg', inRow: 'reg', inCol: 'reg', fRow: 'reg', fCol: 'reg',
  row: 'reg', col: 'reg', tx: 'reg', ty: 'reg', tileRow: 'reg', tileCol: 'reg',
};
const TIERS_71: Record<string, Tier> = { ...REG, N: 'global', P: 'global', F_c: 'const', f: 'const' };
const TIERS_72: Record<string, Tier> = { ...REG, N: 'global', P: 'global', F_c: 'const', F: 'const', N_s: 'shared' };

// ---------------------------------------------------------------- what each prediction is measured against
/** Lab 7.1: the constant-memory kernel's speed-up over the basic one at r = 2. */
const constSpeedup = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const r = benchRow(s, 'const', { radius: 2 });
    const v = num(r?.speedup_vs_basic);
    if (!r || !v) return null;
    return {
      value: v,
      detail: tr(`${fmt(num(r.gflops) ?? NaN)} GFLOP/s against ${fmt(num(benchRow(s, 'basic', { radius: 2 })?.gflops) ?? NaN)} for the basic kernel, ${String(r.width)} × ${String(r.height)} image`, `${fmt(num(r.gflops) ?? NaN)} GFLOP/s so với ${fmt(num(benchRow(s, 'basic', { radius: 2 })?.gflops) ?? NaN)} của kernel cơ bản, ảnh ${String(r.width)} × ${String(r.height)}`),
    };
  });
/** Lab 7.1: the by-value struct over the constant buffer, at r = 2. */
const byValueOverConst = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const a = num(benchRow(s, 'byvalue', { radius: 2 })?.gflops);
    const b = num(benchRow(s, 'const', { radius: 2 })?.gflops);
    if (!a || !b) return null;
    return { value: a / b, detail: tr(`${fmt(a)} GFLOP/s by value against ${fmt(b)} GFLOP/s through the constant buffer or symbol`, `${fmt(a)} GFLOP/s khi truyền theo giá trị so với ${fmt(b)} GFLOP/s qua buffer hoặc symbol hằng`) };
  });
/** Lab 7.2: the halo-tiled kernel's speed-up over the constant-memory kernel, 32-wide tile, r = 2. */
const haloSpeedup = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const r = benchRow(s, 'tiled_halo', { tile: 32, radius: 2 });
    const v = num(r?.speedup_vs_const);
    if (!r || !v) return null;
    return {
      value: v,
      detail: tr(`${fmt(num(r.gflops) ?? NaN)} GFLOP/s tiled; the formula's traffic ratio there is ${fmt(num(r.model_traffic_ratio) ?? NaN)}`, `${fmt(num(r.gflops) ?? NaN)} GFLOP/s khi phân ô; tỷ lệ lưu lượng theo công thức ở đó là ${fmt(num(r.model_traffic_ratio) ?? NaN)}`),
    };
  });
/** Lab 7.2: the cached-halo kernel over the halo-tiled one, 32-wide tile, r = 2. */
const cachedOverHalo = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const c = num(benchRow(s, 'tiled_cached', { tile: 32, radius: 2 })?.gflops);
    const h = num(benchRow(s, 'tiled_halo', { tile: 32, radius: 2 })?.gflops);
    if (!c || !h) return null;
    return { value: c / h, detail: tr(`${fmt(c)} GFLOP/s cached against ${fmt(h)} GFLOP/s with the halo in shared memory`, `${fmt(c)} GFLOP/s khi dùng cache so với ${fmt(h)} GFLOP/s khi vùng biên nằm trong bộ nhớ chia sẻ`) };
  });

// ---------------------------------------------------------------- results tables
const image: Col = { get head() { return tr('Image', 'Ảnh'); }, num: true, cell: (r) => (r.width != null ? `${String(r.width)} × ${String(r.height)}` : '') };
const blank = (v: unknown) => (v == null ? '' : String(v));

/** Lab 7.1's rows, with each kernel set against the ceiling its model gives it on the same device. */
function Lab71Results() {
  const { reports } = useLabs();
  // Lab 5.1's bandwidth, per backend, from the reader's own run where it passed.
  const bandwidth = new Map(
    preferredBench(reports, l51).flatMap((s) => {
      const v = num(benchRow(s, 'roofline')?.bandwidth_gb_per_s);
      return v ? [[s.backend, v] as const] : [];
    }),
  );
  const ceiling = (r: BenchRow) => {
    const bw = bandwidth.get(String(r.backend));
    const ai = num(r.model_intensity);
    return bw && ai ? ai * bw : undefined;
  };
  return (
    <LabResults
      lab={l1}
      cols={[
        {
          head: tr('Variant', 'Biến thể'),
          cell: (r) =>
            r.name === 'pipeline'
              ? tr(`conv_const pipeline: execution width ${blank(r.threadExecutionWidth)}, at most ${blank(r.maxTotalThreadsPerThreadgroup)} threads`, `pipeline conv_const: độ rộng thực thi ${blank(r.threadExecutionWidth)}, tối đa ${blank(r.maxTotalThreadsPerThreadgroup)} luồng`)
              : String(r.name),
        },
        rawCol('r', 'radius'),
        image,
        msCol,
        numCol('GFLOP/s', 'gflops'),
        numCol(tr('Speed-up over basic', 'Tăng tốc so với cơ bản'), 'speedup_vs_basic'),
        numCol(tr('Model FLOP/B', 'FLOP/B theo mô hình'), 'model_intensity'),
        {
          head: tr('× model ceiling', '× trần mô hình'),
          num: true,
          cell: (r) => {
            const c = ceiling(r);
            const g = num(r.gflops);
            return c && g ? fmt(g / c) : '';
          },
        },
      ]}
    />
  );
}

const KERNEL_72: Record<string, string> = {
  get const() {
    return tr('constant memory (provided)', 'bộ nhớ hằng (cho sẵn)');
  },
  get tiled_halo() {
    return tr('halo tiles (part A)', 'ô có vùng biên (phần A)');
  },
  get tiled_cached() {
    return tr('cached halo (part B)', 'vùng biên qua cache (phần B)');
  },
};

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <Translated vi={<LabsIntro>Hai lab, mỗi lab trong thư mục riêng dưới <code>labs/ch07/</code>. Lab thứ nhất viết phép tích chập cơ bản ba lần, chỉ thay đổi nơi bộ lọc nằm. Lab thứ hai phân ô (tiling) nó: bạn viết kernel vùng biên (halo) của mục 7.5 và sửa kernel vùng biên qua cache của mục 7.6, rồi đo cả hai đối chiếu với công thức.</LabsIntro>}>
<LabsIntro>
        Two labs, each in its own directory under <code>labs/ch07/</code>. The first writes the basic convolution three
        times, changing only where the filter lives. The second tiles it: you write the halo kernel of section 7.5 and
        repair the cached-halo kernel of section 7.6, then measure both against the formula.
      </LabsIntro>
</Translated>
      <RunningLabs
        example="7.1"
        alsoShownIn={
          <Translated vi={<>bảng mức trần của <a href={href('ch07', '7.3')}>mục 7.3</a>, biểu đồ tăng tốc của{' '}<a href={href('ch07', '7.5')}>mục 7.5</a> và lưới thời gian của <a href={href('ch07', '7.6')}>mục 7.6</a></>}>
          <>
            the ceiling panel of <a href={href('ch07', '7.3')}>section 7.3</a>, the speed-up plot of{' '}
            <a href={href('ch07', '7.5')}>section 7.5</a> and the timing grid of <a href={href('ch07', '7.6')}>section 7.6</a>
          </>
          </Translated>
        }
      />
      <BackendsAndEmulator />
      <VerificationStatus />

      {/* ------------------------------------------------------------ 7.1 */}
      <h2><Translated vi="Lab 7.1: bộ lọc trong bộ nhớ hằng, theo ba cách">Lab 7.1: the filter in constant memory, three ways</Translated></h2>
      <Translated vi={<p>Bạn viết kernel mỗi-đầu-ra-một-luồng của mục 7.2 ba lần, và thứ duy nhất thay đổi là nơi bộ lọc nằm. Đầu tiên là một con trỏ tới bộ nhớ toàn cục (global memory), tức Fig. 7.7 được sửa cho biên dịch được và có thêm phép chặn đầu ra. Sau đó là một mảng <code>__constant__</code> mà phần khởi chạy điền bằng <code>cudaMemcpyToSymbol</code>, tức Fig. 7.9. Rồi đến một struct truyền theo giá trị cùng lần khởi chạy, phương án thay thế mà sách nêu ở p. 172. Cả ba áp dụng bộ lọc không lật, dùng ô ma (ghost cell) bằng 0 và nhận bán kính lúc chạy từ 0 đến 7.</p>}>
<p>
        You write the one-thread-per-output kernel of section 7.2 three times, and the only thing that changes is where
        the filter lives. First as a pointer to global memory, which is Fig. 7.7 made to compile and given an output
        guard. Then in a <code>__constant__</code> array that the launch fills with <code>cudaMemcpyToSymbol</code>, which
        is Fig. 7.9. Then in a struct passed by value with the launch, the alternative the book names on p. 172. All
        three apply the filter without flipping it, use zero ghost cells and take a run-time radius from 0 to 7.
      </p>
</Translated>
      <Source section="7.2–7.4" pages="164–172" figs={tr('Figs. 7.7 and 7.9', 'Hình 7.7 và 7.9')} />
      <Translated vi={<p>Mục đích là một phép đo. Mô hình nói kernel thứ hai và thứ ba cần một nửa lưu lượng bộ nhớ toàn cục so với kernel đầu, 0.5 FLOP/B so với 0.25, nên chúng phải chạy nhanh gấp đôi nếu bộ nhớ là yếu tố quyết định. Mục 7.3 đã giải thích vì sao điều đó không nhất thiết xảy ra: mô hình tính mọi lần đọc vào DRAM, còn bộ nhớ đệm của thiết bị bạn có thể đã đang phục vụ bộ lọc.</p>}>
<p>
        The point is a measurement. The model says the second and third kernels need half the global traffic of the
        first, 0.5 FLOP/B against 0.25, so they should run twice as fast if memory decides. Section 7.3 explained why
        that need not happen: the model charges every read to DRAM, and your device&rsquo;s caches may already be
        serving the filter.
      </p>
</Translated>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <Translated vi={<><code>convBasicKernel</code>, <code>convConstKernel</code> với <code>F_c</code>, và{' '}<code>convByValueKernel</code>, mỗi kernel có hàm <code>launch</code> của nó. Giữ mảng{' '}<code>__constant__</code>, kernel của nó và lệnh sao chép trong tệp này: một khai báo <code>extern __constant__</code>{' '}ở tệp khác, như sách gợi ý ở p. 169, là lỗi ở chế độ mặc định của nvcc (ch07-facts A1).</>}>
            <>
              <code>convBasicKernel</code>, <code>convConstKernel</code> with <code>F_c</code>, and{' '}
              <code>convByValueKernel</code>, each with its <code>launch</code> function. Keep the{' '}
              <code>__constant__</code> array, its kernel and the copy in this file: an <code>extern __constant__</code>{' '}
              declaration in another file, as the book suggests on p. 169, is an error in nvcc&rsquo;s default mode (ch07-facts A1).
            </>
            </Translated>,
          ],
          ['kernel.metal', <Translated vi={<><code>conv_basic</code>, <code>conv_const</code> và <code>conv_by_value</code></>}><><code>conv_basic</code>, <code>conv_const</code> and <code>conv_by_value</code></></Translated>],
          ['launch.mm', <Translated vi={<><code>encodeConvBasic</code>, <code>encodeConvConst</code> và <code>encodeConvByValue</code></>}><><code>encodeConvBasic</code>, <code>encodeConvConst</code> and <code>encodeConvByValue</code></></Translated>],
        ]}
      />
      <Translated vi={<p>Metal không có symbol để sao chép vào: một mảng <code>constant</code> ở phạm vi chương trình phải được khởi tạo trong mã nguồn shader. Vì vậy trên Metal, ba cách là ba loại đối số kernel: một buffer <code>device const float*</code>, một buffer <code>constant float*</code>, và một <code>constant Filter&amp;</code> mà host điền bằng{' '}<code>setBytes</code>. Struct có 904 byte, dưới giới hạn của <code>setBytes</code>.</p>}>
<p>
        Metal has no symbol to copy into: a program-scope <code>constant</code> array has to be initialised in the shader
        source. So on Metal the three ways are three kinds of kernel argument: a <code>device const float*</code>{' '}
        buffer, a <code>constant float*</code> buffer, and a <code>constant Filter&amp;</code> that the host fills with{' '}
        <code>setBytes</code>. The struct is 904 bytes, under <code>setBytes</code>&rsquo;s limit.
      </p>
</Translated>
      <Today checked="2026-11-20 (ch07-facts A4, B2, C1)" /* i18n-ignore: citation */>
        <Translated vi={<p><code>setBytes</code> nhận dưới 4 KB. AMD không ghi trong tài liệu giới hạn đối số nào và không có <code>__grid_constant__</code>, nên struct là một tham số <code>const</code> thông thường.</p>}>
<p>
          <code>setBytes</code> takes under 4 KB. AMD documents no argument limit and no <code>__grid_constant__</code>,
          so the struct is a plain <code>const</code> parameter.
        </p>
</Translated>
      </Today>
      <Translated vi={<p>Chạy <code>python3 labs/pmpp.py run 7.1 --backend emu</code> trước, rồi chạy không có <code>--backend</code>. Với mỗi kernel, các test lấy ảnh 7×7 và bộ lọc 5×5 của sách và kiểm tra <code>P[2][2] = 321</code> và{' '}<code>P[1][0] = 112</code>, rồi đến ảnh ngẫu nhiên với bộ lọc ngẫu nhiên ở các bán kính 0, 1, 2, 3 và 7, trên các kích thước từ 1×1 đến 257×129 mà không kích thước khối nào chia hết. Bộ lọc ngẫu nhiên không đối xứng, nên một bộ lọc bị lật hoặc bị đọc thành{' '}<code>F[fCol][fRow]</code> sẽ hỏng ở đó dù nó qua được ví dụ của sách. <code>P</code> bắt đầu là NaN, nên một đầu ra bạn quên ghi sẽ bị phát hiện. Trên trình giả lập (emulator), thiếu phép kiểm tra ô ma hoặc phép chặn đầu ra sẽ dừng lần chạy với một báo cáo AddressSanitizer chỉ ra dòng lỗi.</p>}>
<p>
        Run <code>python3 labs/pmpp.py run 7.1 --backend emu</code> first, then without <code>--backend</code>. For each
        kernel the tests take the book&rsquo;s 7×7 image and 5×5 filter and check <code>P[2][2] = 321</code> and{' '}
        <code>P[1][0] = 112</code>, then random images with random filters at radii 0, 1, 2, 3 and 7, on shapes from 1×1
        to 257×129 that no block size divides. A random filter is not symmetric, so a flipped filter or one read as{' '}
        <code>F[fCol][fRow]</code> fails there although it passes the book&rsquo;s example. <code>P</code> starts as NaN,
        so an output you never write is caught. On the emulator, a missing ghost test or output guard stops the run with
        an AddressSanitizer report that names the line.
      </p>
</Translated>
      <Translated vi={<p>Với <code>--bench</code>, mỗi bán kính trong 1, 2, 3, 5 và 7 được đo thời gian trên một ảnh lớn dần cho tới khi kernel cơ bản mất khoảng 50 ms. GFLOP/s đếm mọi tap, kể cả tap rơi vào ô ma, như sách làm ở p. 167. Cột cuối của bảng chia GFLOP/s của mỗi kernel cho mức trần mà mô hình của nó cho trên cùng thiết bị: FLOP/B của nó nhân với băng thông mà lab 5.1 đo được ở đó. Trên 1 nghĩa là có thứ gì đó đã phục vụ phần tái sử dụng mà mô hình tính vào DRAM.</p>}>
<p>
        With <code>--bench</code>, each radius in 1, 2, 3, 5 and 7 is timed on an image that grows until the basic kernel
        takes about 50 ms. GFLOP/s counts every tap, ghost taps included, as the book does on p. 167. The last column of
        the table divides each kernel&rsquo;s GFLOP/s by the ceiling its model gives it on the same device: its FLOP/B
        times the bandwidth lab 5.1 measured there. Above 1, something served the reuse that the model charges to DRAM.
      </p>
</Translated>
      <LabPrediction
        id="ch07/labs/p-lab1-const-speedup"
        prompt={
          <Translated vi={<>Trước khi chạy benchmark. Ở r = 2 (bộ lọc 5×5), kernel dùng bộ nhớ hằng sẽ chạy nhanh gấp bao nhiêu lần kernel cơ bản trên thiết bị của bạn? Câu trả lời của mô hình là 2, nếu cả hai kernel đều bị giới hạn bởi bộ nhớ và không có gì khác được cache.</>}>
          <>
            Before you run the benchmark. At r = 2 (a 5×5 filter), the constant-memory kernel will run how many times as
            fast as the basic kernel on your device? The model&rsquo;s answer is 2, if both kernels are limited by memory
            and nothing else is cached.
          </>
          </Translated>
        }
        unit={tr('× the basic kernel', '× kernel cơ bản')}
        measure={constSpeedup}
        reading={
          <Translated vi={<>Con số 2 của mô hình đếm những gì mỗi kernel đòi hỏi ở bộ nhớ. Nếu các lần đọc bộ lọc của kernel cơ bản đã được một cache trả lời, bộ nhớ hằng chẳng còn mấy để tiết kiệm, và hệ số co lại về phía 1. Hãy xem các bán kính khác trong bảng trước khi giải thích con số.</>}>
          <>
            The model&rsquo;s 2 counts what each kernel asks of memory. If the basic kernel&rsquo;s filter reads were
            already being answered by a cache, constant memory has little left to save, and the factor shrinks towards
            1. Look at the other radii in the table before you explain the number.
          </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch07/labs/p-lab1-byvalue-vs-const"
        prompt={
          <Translated vi={<>Trước khi chạy benchmark. Ở r = 2, GFLOP/s của kernel truyền theo giá trị chia cho GFLOP/s của kernel dùng bộ nhớ hằng sẽ bằng bao nhiêu? Trên CUDA cả hai đều đọc qua bộ nhớ hằng. Trên Metal, một bên là buffer{' '}<code>constant</code> còn bên kia là một struct kích thước cố định được điền bằng <code>setBytes</code>; Apple không ghi trong tài liệu bộ nhớ đệm hằng nào.</>}>
          <>
            Before you run the benchmark. At r = 2, the by-value kernel&rsquo;s GFLOP/s divided by the constant-memory
            kernel&rsquo;s will be what? On CUDA both read through constant memory. On Metal one is a{' '}
            <code>constant</code> buffer and the other a fixed-size struct filled with <code>setBytes</code>; Apple
            documents no constant cache.
          </>
          </Translated>
        }
        unit={tr('× the constant-memory kernel', '× kernel dùng bộ nhớ hằng')}
        measure={byValueOverConst}
        reading={
          <Translated vi={<>1 nghĩa là không khác biệt. Điều duy nhất Apple nói về chuyện này là một bài nói năm 2016: dữ liệu trong không gian địa chỉ <code>constant</code>{' '}có các lần đọc được giới hạn tĩnh, chẳng hạn một struct kích thước cố định truyền theo tham chiếu, có thể được nạp sẵn vào thanh ghi (register). Máy Mac của bạn có làm vậy hay không thì không được ghi trong tài liệu, và một lần chạy không đủ để phân biệt vài phần trăm với nhiễu: hãy chạy benchmark hai lần.</>}>
          <>
            1 means no difference. Apple&rsquo;s only words on this are a 2016 talk: data in the <code>constant</code>{' '}
            address space whose reads are statically bounded, such as a fixed-size struct passed by reference, can be
            preloaded into registers. Whether your Mac does that is not documented, and one run is not enough to tell a
            few percent from noise: run the benchmark twice.
          </>
          </Translated>
        }
      />
      <Lab71Results />
      <Translated vi={<p>Trên CUDA còn một chỗ nữa để xem. <code>nvcc -Xptxas=-v</code> liệt kê thanh ghi, bộ nhớ chia sẻ và bộ nhớ hằng của mỗi kernel, và README chỉ ra nơi 900 byte của <code>F_c</code> và 904 byte của struct sẽ xuất hiện. Lệnh đó chưa được chạy cho khóa học này, vì khóa học không có CUDA toolkit.</p>}>
<p>
        On CUDA there is one more place to look. <code>nvcc -Xptxas=-v</code> lists each kernel&rsquo;s register, shared
        and constant memory, and the README shows where <code>F_c</code>&rsquo;s 900 bytes and the struct&rsquo;s 904
        should appear. That command has not been run for this course, which had no CUDA toolkit.
      </p>
</Translated>
      <Translated vi={<p>Và một câu hỏi để mang theo, từ phần cuối của README: llama.cpp giữ các codebook lượng tử hóa của nó trong bộ nhớ toàn cục trên CUDA và HIP, không phải trong <code>__constant__</code>. Mỗi luồng tra mã của riêng nó.{' '}<a href={href('ch07', '7.4')}>Mục 7.4</a> giải thích vì sao đó là trường hợp tệ nhất của bộ nhớ đệm hằng.</p>}>
<p>
        And one question to take away, from the README&rsquo;s last section: llama.cpp keeps its quantisation codebooks
        in global memory on CUDA and HIP, not in <code>__constant__</code>. Each thread looks up its own code.{' '}
        <a href={href('ch07', '7.4')}>Section 7.4</a> explains why that is the constant cache&rsquo;s worst case.
      </p>
</Translated>
      <Beyond>
        <Translated vi={<p>Các bảng của llama.cpp là mảng <code>static const __device__</code> trên CUDA và HIP và là mảng{' '}<code>constant</code> ở phạm vi chương trình trên Metal (ch07-facts D4, kiểm tra ngày 2026-11-20).</p>}>
<p>
          llama.cpp&rsquo;s tables are <code>static const __device__</code> arrays on CUDA and HIP and program-scope{' '}
          <code>constant</code> arrays on Metal (ch07-facts D4, checked 2026-11-20).
        </p>
</Translated>
      </Beyond>
      <ReferenceSolution lab={l1} tiers={TIERS_71} />

      {/* ------------------------------------------------------------ 7.2 */}
      <h2><Translated vi="Lab 7.2: tích chập phân ô với vùng biên">Lab 7.2: tiled convolution with halos</Translated></h2>
      <Translated vi={<p>Hai kernel phân ô, cả hai đều đọc bộ lọc từ bộ nhớ hằng. Ở phần A bạn viết kernel vùng biên của Fig. 7.12: khối là ô đầu vào, mỗi luồng nạp một phần tử và ghi số 0 cho một ô ma, một rào chắn (barrier), rồi vòng luồng ngoài cùng ngồi không trong khi các luồng khác tính. Ở phần B bạn sửa kernel vùng biên qua cache của Fig. 7.15, kernel này biên dịch được và có hai lỗi cài sẵn. Một trong hai lỗi được in trong sách.</p>}>
<p>
        Two tiled kernels, both reading the filter from constant memory. In part A you write the halo kernel of Fig.
        7.12: the block is the input tile, every thread loads one element and stores a zero for a ghost cell, one
        barrier, and then the outer ring of threads sits idle while the others compute. In part B you repair the
        cached-halo kernel of Fig. 7.15, which compiles and has two planted defects. One of them is printed in the book.
      </p>
</Translated>
      <Source section="7.5–7.6" pages="172–179" figs={tr('Figs. 7.12 and 7.15', 'Hình 7.12 và 7.15')} />
      <Translated vi={<p>Bán kính là một tham số template, <code>template &lt;int FILTER_RADIUS&gt;</code>, nên các vòng lặp bộ lọc có số vòng cố định, và một switch cho sẵn chọn kernel đã biên dịch cho bán kính được yêu cầu, giống như kernel thật trong ví dụ phục vụ LLM của <a href={href('ch07', '7.5')}>mục 7.5</a>. Chiều rộng ô là{' '}<code>TILE_WIDTH</code> của bản build: mặc định 32, và 8 cùng 16 cũng được build cho benchmark. Trên Metal nó là một macro tiền xử lý, vì một function constant không thể định kích thước cho mảng <code>threadgroup</code>.</p>}>
<p>
        The radius is a template parameter, <code>template &lt;int FILTER_RADIUS&gt;</code>, so the filter loops have
        fixed trip counts, and a provided switch picks the compiled kernel for the radius asked for, as the real kernel
        in <a href={href('ch07', '7.5')}>section 7.5</a>&rsquo;s LLM-serving example does. The tile width is the build&rsquo;s{' '}
        <code>TILE_WIDTH</code>: 32 by default, with 8 and 16 also built for the benchmark. On Metal it is a
        preprocessor macro, because a function constant cannot size a <code>threadgroup</code> array.
      </p>
</Translated>
      <Today checked="2026-11-20 (ch07-facts B6, C3)" /* i18n-ignore: citation */>
        <Translated vi={<p>HIP từ chối một khối lớn hơn launch bound của kernel, mặc định là mức tối đa của thiết bị.</p>}>
<p>
          HIP refuses a block larger than the kernel&rsquo;s launch bound, the device maximum by default.
        </p>
</Translated>
      </Today>
      <LabFiles
        lab={l2}
        files={[
          [
            'kernel.cu',
            <Translated vi={<>phần A: <code>convTiledHaloKernel</code> cùng lưới (grid) và khối của nó trong <code>launchHalo</code>; phần B:{' '}<code>convCachedKernel</code>, cần sửa. Đừng động vào phần mã sau <code>// @part provided</code>.</>}>
            <>
              part A: <code>convTiledHaloKernel</code> and its grid and block in <code>launchHalo</code>; part B:{' '}
              <code>convCachedKernel</code>, to repair. Leave the code after <code>// @part provided</code> alone.
            </>
            </Translated>,
          ],
          ['kernel.metal', <Translated vi={<><code>conv_tiled_halo</code> cần viết, <code>conv_tiled_cached</code> với cùng hai lỗi</>}><><code>conv_tiled_halo</code> to write, <code>conv_tiled_cached</code> with the same two defects</></Translated>],
          ['launch.mm', <Translated vi={<>kích thước lưới và threadgroup trong <code>encodeConvTiledHalo</code></>}><>the grid and threadgroup sizes in <code>encodeConvTiledHalo</code></></Translated>],
        ]}
      />
      <Translated vi={<p>Chạy <code>python3 labs/pmpp.py run 7.2</code>, bắt đầu với <code>--backend emu</code>. Các test chạy cả hai kernel ở mọi bán kính từ 1 đến giá trị lớn nhất còn để lại ô đầu ra ít nhất là 2: ví dụ 7×7 của sách (lại là 321 và 112), rồi dữ liệu ngẫu nhiên với bộ lọc ngẫu nhiên trên các kích thước tới 1000×999, được chọn để các ô thò ra ngoài cạnh phải và cạnh dưới. Các test của phần A chạy trước và hỏng cho tới khi bạn viết xong nó; điều đó không dừng lần chạy. Một ô chỉ được test ở nơi thiết bị nhận khối kích thước đó: trên Metal là nơi <code>maxTotalThreadsPerThreadgroup</code> của pipeline cho phép, còn trên HIP, nơi có thể từ chối khối 32×32, chương trình điều khiển (driver) sẽ báo như vậy.</p>}>
<p>
        Run <code>python3 labs/pmpp.py run 7.2</code>, starting with <code>--backend emu</code>. The tests run both
        kernels at every radius from 1 to the largest that leaves an output tile of at least 2: the book&rsquo;s 7×7
        example (321 and 112 again), then random data with random filters on shapes up to 1000×999, chosen so that tiles
        overhang the right and bottom edges. Part A&rsquo;s tests come first and fail until you have written it; that
        does not stop the run. A tile is tested only where the device takes a block that size: on Metal where the
        pipeline&rsquo;s <code>maxTotalThreadsPerThreadgroup</code> allows it, and on HIP, which may refuse a 32×32
        block, the driver says so.
      </p>
</Translated>
      <Translated vi={<p>Phần A có ba điều cần làm đúng, và trình giả lập bắt được từng điều:</p>}>
<p>Part A has three things to get right, and the emulator catches each:</p>
</Translated>
      <ul>
        <Translated vi={<li>Chép <code>threadIdx</code> vào các biến <code>int</code> trước khi trừ bán kính. Các biến dựng sẵn là số không dấu, và các thuộc tính vị trí của Metal cũng vậy: với luồng 0, <code>threadIdx.x - FILTER_RADIUS</code> là khoảng bốn tỷ, không phải &minus;2.</li>}>
<li>
          Copy <code>threadIdx</code> into <code>int</code>s before subtracting the radius. The built-ins are unsigned, and
          so are Metal&rsquo;s position attributes: for thread 0, <code>threadIdx.x - FILTER_RADIUS</code> is about four
          billion, not &minus;2.
        </li>
</Translated>
        <Translated vi={<li>Mọi luồng đều tới được rào chắn. Các luồng ở vòng ngoài không tính gì, nhưng chúng nạp các phần tử vùng biên mà các luồng khác đọc, nên phép chặn phần tính đặt sau <code>__syncthreads()</code>.</li>}>
<li>
          Every thread reaches the barrier. The ring threads compute nothing, but they load halo elements that the others
          read, so the compute guard goes after <code>__syncthreads()</code>.
        </li>
</Translated>
        <Translated vi={<li>Ghi các số 0 cho ô ma. Vòng lặp tính không có kiểm tra biên. Trình giả lập cho mỗi khối bộ nhớ chia sẻ mới điền đầy byte <code>0xCD</code>, nên thiếu một số 0 sẽ hỏng ở đó mọi lần; trên GPU, vùng nhớ chưa ai ghi có thể tình cờ đọc ra 0, và lỗi có thể lọt qua.</li>}>
<li>
          Store the ghost zeros. The compute loop has no bounds test. The emulator gives every block fresh shared
          memory filled with <code>0xCD</code> bytes, so a missing zero fails there every time; on a GPU, memory nobody
          wrote may happen to read as zero, and the mistake can pass.
        </li>
</Translated>
      </ul>
      <Translated vi={<p>Với phần B, dùng phương pháp của lab 5.4:</p>}>
<p>For part B, the method of lab 5.4:</p>
</Translated>
      <ol>
        <Translated vi={<li>Chạy trên trình giả lập trước. AddressSanitizer dừng lần chạy ở lần truy cập đầu tiên ra ngoài một mảng và chỉ ra dòng lỗi; các test sau đó không chạy.</li>}>
<li>
          Run on the emulator first. AddressSanitizer stops the run at the first access outside an array and names the
          line; the tests after that do not run.
        </li>
</Translated>
        <Translated vi={<li>Sửa một chỗ, chạy lại, và ghi lại các lỗi thay đổi thế nào.</li>}>
<li>Fix one thing, run again, and note how the failures change.</li>
</Translated>
        <Translated vi={<li>Rồi chạy trên GPU của bạn, và ghi lại nó cho bạn thấy lỗi nào và bằng cách nào. Một lỗi cho đầu ra sai gần như khắp nơi; lỗi kia chỉ ở gần cạnh phải, trong các ô thò ra ngoài cạnh đó.</li>}>
<li>
          Then run on your GPU, and note which defect it showed you and how. One defect gives wrong outputs almost
          everywhere; the other only near the right edge, in tiles that overhang it.
        </li>
</Translated>
      </ol>
      <Translated vi={<p>Điều thứ ba cần để ý, mà không test nào bắt được: phép kiểm tra trong-ô của mã khởi đầu (starter) là của sách,{' '}<code>threadIdx.x - FILTER_RADIUS + fCol &gt;= 0</code>. Vế trái của nó là số không dấu, nên phép so sánh luôn đúng, và kernel đúng chỉ vì nửa <code>&lt; TILE_DIM</code> loại bỏ các giá trị bị quấn vòng. Dù vậy vẫn hãy sửa nó (<a href={href('ch07', '7.6')}>mục 7.6</a> cho thấy vì sao).</p>}>
<p>
        A third thing to notice, which no test can catch: the starter&rsquo;s in-tile test is the book&rsquo;s,{' '}
        <code>threadIdx.x - FILTER_RADIUS + fCol &gt;= 0</code>. Its left side is unsigned, so the comparison is always
        true, and the kernel is correct only because the <code>&lt; TILE_DIM</code> half rejects the wrapped values.
        Fix it anyway (<a href={href('ch07', '7.6')}>section 7.6</a> shows why).
      </p>
</Translated>
      <Translated vi={<p>Với <code>--bench</code>, mỗi chiều rộng ô và mỗi bán kính trong 1, 2, 3, 5 và 7 còn để lại ô đầu ra ít nhất là 2 được đo thời gian so với một kernel dùng bộ nhớ hằng cho sẵn có cùng bán kính lúc biên dịch. Mỗi kernel được kiểm tra trên một ảnh 100×61 trước, và kernel sai sẽ được báo cáo, không được đo thời gian. Bên cạnh mỗi hàng vùng biên, chương trình điều khiển ghi lại FLOP/B theo công thức của mục 7.5 và tỷ lệ của nó so với 0.5 của kernel dùng bộ nhớ hằng: hệ số lưu lượng yêu cầu mà phân ô loại bỏ được.</p>}>
<p>
        With <code>--bench</code>, each tile width and each radius in 1, 2, 3, 5 and 7 that leaves an output tile of at
        least 2 is timed against a provided constant-memory kernel with the same compile-time radius. Each kernel is
        checked on a 100×61 image first, and a wrong one is reported, not timed. Beside every halo row the driver records
        the section 7.5 formula&rsquo;s FLOP/B and its ratio to the constant-memory kernel&rsquo;s 0.5: the factor of
        requested traffic the tiling removes.
      </p>
</Translated>
      <LabPrediction
        id="ch07/labs/p-lab2-halo-speedup"
        prompt={
          <Translated vi={<>Trước khi chạy benchmark. Với ô đầu vào 32×32 và bộ lọc 5×5, công thức nói kernel vùng biên đòi ít lưu lượng bộ nhớ toàn cục hơn kernel dùng bộ nhớ hằng 5.42 / 0.5, khoảng 10.8 lần. Trên thiết bị của bạn nó sẽ chạy nhanh gấp bao nhiêu lần?</>}>
          <>
            Before you run the benchmark. With a 32×32 input tile and a 5×5 filter, the formula says the halo kernel asks
            for 5.42 / 0.5, about 10.8 times, less global traffic than the constant-memory kernel. How many times as fast
            will it run on your device?
          </>
          </Translated>
        }
        unit={tr('× the constant-memory kernel', '× kernel dùng bộ nhớ hằng')}
        measure={haloSpeedup}
        reading={
          <Translated vi={<>Con số 10.8 là về số byte được yêu cầu. Bộ nhớ đệm đã phục vụ nhiều lần đọc phần tử lân cận của kernel dùng bộ nhớ hằng (p. 168), và một kernel đòi ít lưu lượng khi đó bị giới hạn bởi thứ khác, chẳng hạn rào chắn và vòng luồng ngồi không. Mục 7.5 vẽ mọi ô và bán kính đối chiếu với công thức.</>}>
          <>
            The 10.8 is about requested bytes. Caches already serve many of the constant-memory kernel&rsquo;s neighbour
            reads (p. 168), and a kernel that asks for little traffic is then limited by something else, such as the
            barrier and the idle ring. Section 7.5 plots every tile and radius against the formula.
          </>
          </Translated>
        }
      />
      <LabPrediction
        id="ch07/labs/p-lab2-cached-vs-halo"
        prompt={
          <Translated vi={<>Trước khi chạy benchmark. Ở ô 32 và r = 2, GFLOP/s của kernel vùng biên qua cache chia cho GFLOP/s của kernel vùng biên sẽ bằng bao nhiêu? Trên 1 nghĩa là cache phục vụ vùng biên đủ tốt để thắng việc nạp sẵn nó.</>}>
          <>
            Before you run the benchmark. At tile 32 and r = 2, the cached-halo kernel&rsquo;s GFLOP/s divided by the
            halo kernel&rsquo;s will be what? Above 1 means the cache served the halo well enough to beat staging it.
          </>
          </Translated>
        }
        unit={tr('× the halo kernel', '× kernel vùng biên')}
        measure={cachedOverHalo}
        reading={
          <Translated vi={<>Kernel của mục 7.6 đặt cược rằng vùng biên của một khối vẫn còn trong cache, và trả giá bằng một nhánh rẽ ba hướng trong vòng lặp bộ lọc. Chỉ thời gian thôi không cho biết hiệu ứng nào thắng; tỷ lệ trúng L2 của một profiler sẽ giúp. Mục 7.6 cho thấy tỷ lệ này với mọi ô và bán kính bạn đã đo.</>}>
          <>
            Section 7.6&rsquo;s kernel bets that a block&rsquo;s halo is still in a cache, and pays for a three-way branch
            inside the filter loop. Timing alone does not say which effect won; a profiler&rsquo;s L2 hit rate helps.
            Section 7.6 shows the ratio for every tile and radius you measured.
          </>
          </Translated>
        }
      />
      <LabResults
        lab={l2}
        cols={[
          { head: tr('Variant', 'Biến thể'), cell: (r) => KERNEL_72[String(r.name)] ?? String(r.name) },
          { head: tr('Tile', 'Ô'), num: true, cell: (r) => (r.name === 'const' ? '' : blank(r.tile)) },
          { head: tr('Output tile', 'Ô đầu ra'), num: true, cell: (r) => (r.name === 'const' ? '' : blank(r.out_tile)) },
          rawCol('r', 'radius'),
          image,
          msCol,
          numCol('GFLOP/s', 'gflops'),
          numCol(tr('× const', '× hằng'), 'speedup_vs_const'),
          numCol(tr('Model FLOP/B', 'FLOP/B theo mô hình'), 'model_intensity'),
          numCol(tr('Traffic ratio', 'Tỷ lệ lưu lượng'), 'model_traffic_ratio'),
        ]}
      />
      <Translated vi={<p className="muted">Kernel dùng bộ nhớ hằng cho sẵn dùng khối 16×16 trong mọi bản build, nên các hàng của nó không có ô. Trên CUDA và HIP, mỗi lần chạy được đo thời gian cũng bao gồm việc sao chép bộ lọc vào bộ nhớ hằng, như nhau cho cả ba kernel.</p>}>
<p className="muted">
        The provided constant-memory kernel uses 16×16 blocks in every build, so its rows have no tile. On CUDA and HIP
        each timed run also includes the copy of the filter into constant memory, for all three kernels alike.
      </p>
</Translated>
      <ReferenceSolution lab={l2} tiers={TIERS_72} />

      <h2><Translated vi="Chương này không có mốc dự án">No project milestone in this chapter</Translated></h2>
      <Translated vi={<p><code>PLAN.md</code> không giao cho chương 7 mốc dự án nào của dự án xuyên suốt. Cả hai lab vẫn theo các quy ước của nó, để các chương sau có thể dùng lại: tensor <code>float</code> theo thứ tự hàng (row-major), con trỏ device thô và kích thước tường minh làm đối số, một hàm <code>launch</code> bên cạnh mỗi kernel, và bản tham chiếu CPU trong{' '}<code>main.cpp</code>. Phần <a href={href('ch07', 'wrap')}>tóm tắt</a> liệt kê những gì của chương này bạn sẽ dùng lại.</p>}>
<p>
        <code>PLAN.md</code> gives chapter 7 no milestone of the through-line project. Both labs still follow its
        conventions, so later chapters can reuse them: row-major <code>float</code> tensors, raw device pointers and
        explicit sizes as arguments, a <code>launch</code> function beside each kernel, and the CPU reference in{' '}
        <code>main.cpp</code>. The <a href={href('ch07', 'wrap')}>summary</a> lists what of this chapter you will use
        again.
      </p>
</Translated>
    </LabsPage>
  );
}
