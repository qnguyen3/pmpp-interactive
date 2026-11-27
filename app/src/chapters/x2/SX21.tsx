import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_DEVICE, FORMATS, LAYERS, PIPELINES, bitsPerWeight, byteRatio, bytesPerWeight, ceiling, gemv, kneeBatch, layer,
  mbu, measuredBandwidth, measuredDevices, measuredPipelines, modelBytes, pickRuns, pipelineBytesPerWeight, pipelineRelative, qgemvRows, speedupOverF32,
  thresholdOf, weightOnlyIntensity, type Device, type FormatId, type PipelineId, type QgemvRow,
} from '../../sim/qtraffic';
import { ThreeTrafficsScene } from './ThreeTrafficsScene';
import { Translated, tr } from '../../lib/i18n';

// Section X2.1 (ours): bytes per weight as the roofline knob of decode. A roofline
// explorer by weight format, a Predict and a bar explorer for fused and unfused
// dequantisation, the narrated scene "Three ways to feed one GEMV", and a bandwidth
// utilisation calculator over the reader's own lab X2.1 and X2.2 rows. Every number comes
// from qtraffic.ts.

const FACTS = 'notes/today/x2-facts.md';
const us = (s: number) => fmt(s * 1e6) + ' µs';

export const questionsSX21: Record<string, Question> = {
  'x2/X2.1/q-q8-ceiling': {
    kind: 'choice',
    id: 'x2/X2.1/q-q8-ceiling',
    prompt: (
      <Translated vi={<>Q8_0 lưu mã 8 bit, bằng một phần tư 32 bit của float32. Vì sao trần tăng tốc của nó so với float32 là 3.76 chứ không phải 4?</>}>
        <>Q8_0 stores 8-bit codes, a quarter of float32's 32 bits. Why is its speed-up ceiling over float32 3.76 and not 4?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Mỗi khối 32 mã mang theo một hệ số tỉ lệ 2 byte: 34 byte, không phải 32.">{'Each block of 32 codes carries a 2-byte scale: 34 bytes, not 32.'}</Translated>, correct: true },
      { label: <Translated vi="Phép nhân int8 chậm hơn phép nhân float trên GPU.">{'The int8 multiply is slower than a float multiply on GPUs.'}</Translated>, why: <Translated vi="Trần này là trần về byte. Trong một GEMV bị giới hạn bởi bộ nhớ, phép tính số học không phải thứ quyết định thời gian.">{'The ceiling is a bytes ceiling. In a memory-bound GEMV the arithmetic is not what sets the time.'}</Translated> },
      { label: <Translated vi="Activation vẫn là float32, và chúng tốn một phần tư số byte.">{'The activations are still float32, and they cost a quarter of the bytes.'}</Translated>, why: <Translated vi="Activation vào khoảng in + out giá trị so với in × out trọng số: một phần rất nhỏ, không phải một phần tư.">{'The activations are about in + out values against in × out weights: a sliver, not a quarter.'}</Translated> },
      { label: <Translated vi="Mỗi mã được lưu trong 9 bit để có thể chứa dấu của nó.">{'Each code is stored in 9 bits, so that it can hold its sign.'}</Translated>, why: <Translated vi="Mã Q8_0 là int8, từ -127 đến 127. Nửa bit thêm cho mỗi trọng số là hệ số tỉ lệ.">{'Q8_0 codes are int8, from -127 to 127. The extra half bit per weight is the scale.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            34 byte cho 32 trọng số là 1.0625 byte mỗi trọng số, và 4 / 1.0625 = 3.76. Số bit mỗi trọng số đã gồm cả hệ số
            tỉ lệ: Q8_0 là 8.5, Q4_0 là 4.5. Hệ số tỉ lệ của các khối là một phần của những gì mỗi token phải đọc, và chúng là
            lý do kích thước khối vừa là quyết định về băng thông vừa là quyết định về độ chính xác.
          </p>
        }
      >
        <p>
          34 bytes per 32 weights is 1.0625 bytes per weight, and 4 / 1.0625 = 3.76. Bits per weight include the scales:
          Q8_0 is 8.5, Q4_0 4.5. The block scales are part of what every token reads,
          and they are why block size is a bandwidth decision as well as an accuracy one.
        </p>
      </Translated>
    ),
    source: `${FACTS} D1, E1`,
  },
  'x2/X2.1/q-knee-batch': {
    kind: 'numeric',
    id: 'x2/X2.1/q-knee-batch',
    prompt: (
      <Translated
        vi={
          <>
            Trên thiết bị ví dụ của sách (ngưỡng 20 FLOP/B), một lớp 4096 sang 4096 với trọng số Q4_0 và activation 16 bit:
            batch nhỏ nhất để nó đạt 20 FLOP/B là bao nhiêu? Đếm trọng số, đầu vào và đầu ra mỗi thứ một lần.
          </>
        }
      >
        <>
          On the book's example device (threshold 20 FLOP/B), a 4096-to-4096 layer with Q4_0 weights and 16-bit
          activations: what is the smallest batch at which it reaches 20 FLOP/B? Count weights, inputs and outputs once each.
        </>
      </Translated>
    ),
    answer: 6,
    unit: 'rows',
    explain: (
      <Translated
        vi={
          <p>
            Batch 6 cho {gemv('q4_0', layer('4096x4096'), 6, 2).intensity.toFixed(1)} FLOP/B; batch 5 cho{' '}
            {gemv('q4_0', layer('4096x4096'), 5, 2).intensity.toFixed(1)}. Với trọng số và activation float32, cùng lớp đó
            cần batch {kneeBatch('f32', layer('4096x4096'), 4, 20)}. Ít byte hơn cho mỗi trọng số khiến một batch nhỏ sớm bị
            giới hạn bởi tính toán hơn, và từ đó trở đi phần việc giải lượng tử hóa, thứ mô hình này bỏ qua, bắt đầu có ý nghĩa.
          </p>
        }
      >
        <p>
          Batch 6 gives {gemv('q4_0', layer('4096x4096'), 6, 2).intensity.toFixed(1)} FLOP/B; batch 5 gives{' '}
          {gemv('q4_0', layer('4096x4096'), 5, 2).intensity.toFixed(1)}. With float32 weights and activations the same layer
          needs batch {kneeBatch('f32', layer('4096x4096'), 4, 20)}. Fewer bytes per weight make a small batch compute-bound
          sooner, and from there on the dequantise work, which this model ignores, starts to count.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.1, p. 94 (the threshold); our arithmetic (x2-facts E1)',
    vi: { unit: 'hàng', source: 'ch. 5 §5.1, p. 94 (ngưỡng); phép tính của khóa học này (x2-facts E1)' },
  },
  'x2/X2.1/q-unfused': {
    kind: 'choice',
    id: 'x2/X2.1/q-unfused',
    prompt: (
      <Translated vi={<>Giải lượng tử hóa trọng số Q4_0 sang float32 trong một kernel rồi chạy GEMV float32, so với chạy GEMV float32 trên trọng số float32. Cách nào nhanh hơn?</>}>
        <>Dequantise Q4_0 weights to float32 in one kernel, then run the float32 GEMV, against running the float32 GEMV on float32 weights. Which is faster?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Riêng GEMV float32: cách kia di chuyển gấp 2.1 lần số byte.">{'The float32 GEMV alone: the other path moves 2.1 times the bytes.'}</Translated>, correct: true },
      { label: <Translated vi="Cách hai kernel, nhanh hơn khoảng 7 lần: nó đọc Q4_0.">{'The two-kernel path, by about 7 times: it reads Q4_0.'}</Translated>, why: <Translated vi="Nó đọc Q4_0 một lần, rồi ghi và đọc một bản sao float32: 0.5625 + 4 + 4 byte mỗi trọng số.">{'It reads Q4_0 once, then writes and reads a float32 copy: 0.5625 + 4 + 4 bytes per weight.'}</Translated> },
      { label: <Translated vi="Như nhau: cả hai đều kết thúc bằng cùng một GEMV float32.">{'They are the same: both end in the same float32 GEMV.'}</Translated>, why: <Translated vi="Cùng một GEMV, cộng thêm cả một lượt nữa ghi ra đầu vào của nó.">{'The same GEMV, plus a whole extra pass that writes its input.'}</Translated> },
      { label: <Translated vi="Cách hai kernel, nhanh hơn một chút, vì giải lượng tử hóa rẻ.">{'The two-kernel path, slightly, because dequantising is cheap.'}</Translated>, why: <Translated vi="Rẻ về tính toán, không rẻ về byte: bản sao lớn bằng trọng số float32.">{'Cheap in arithmetic, not in bytes: the copy is as large as the float32 weights.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            8.5625 so với 4 byte mỗi trọng số là {(pipelineBytesPerWeight('unfused-f32', 'q4_0') / 4).toFixed(2)} lần, và
            thời gian đi theo số byte vì cả hai kernel đều bị giới hạn bởi băng thông: một phép chuyển đổi từng phần tử và
            một GEMV float32 ở batch 1 hầu như không tính toán gì trên mỗi byte. Bản sao đã giải lượng tử hóa được ghi ra bộ
            nhớ rồi đọc lại, nên lượng tử hóa không tiết kiệm được gì trên đường đi. Chỉ một kernel giải nén trong thanh ghi
            mới giữ được tỉ lệ byte.
          </p>
        }
      >
        <p>
          8.5625 against 4 bytes per weight is {(pipelineBytesPerWeight('unfused-f32', 'q4_0') / 4).toFixed(2)} times, and
          time follows bytes because both kernels are bandwidth-bound: an elementwise conversion and a float32 GEMV at
          batch 1 do almost no arithmetic per byte. The dequantised copy is written to memory and read back, so the quantisation saves nothing on the way. Only a kernel
          that unpacks in registers keeps the byte ratio.
        </p>
      </Translated>
    ),
    source: `our bytes model; PyTorch's blog reports the same effect (${FACTS} A1)`,
    vi: { source: `mô hình byte của khóa học này; blog của PyTorch báo cáo cùng hiệu ứng (${FACTS} A1)` },
  },
  'x2/X2.1/q-mbu': {
    kind: 'numeric',
    id: 'x2/X2.1/q-mbu',
    prompt: (
      <Translated
        vi={
          <>
            Một thiết bị giả định có đỉnh 400 GB/s decode một mô hình 7 tỉ trọng số ở Q4_0 với tốc độ 30 token mỗi giây. Mức
            sử dụng băng thông của mô hình (model bandwidth utilisation) là bao nhiêu phần trăm?
          </>
        }
      >
        <>
          A hypothetical device with a peak of 400 GB/s decodes a 7-billion-weight model in Q4_0 at 30 tokens per second.
          What is its model bandwidth utilisation, in percent?
        </>
      </Translated>
    ),
    answer: Number((100 * mbu(modelBytes(7e9, 'q4_0'), 1 / 30, 400)).toFixed(1)),
    tol: 0.02,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            7 × 10⁹ × 0.5625 = 3.94 GB mỗi token, × 30 = 118 GB/s, và 118 / 400 = 29.5 %. Chỉ đếm trọng số là định nghĩa của
            PyTorch; nó bỏ qua KV cache và activation, điều này ổn với ngữ cảnh ngắn và tô đẹp con số với ngữ cảnh dài.
          </p>
        }
      >
        <p>
          7 × 10⁹ × 0.5625 = 3.94 GB per token, × 30 = 118 GB/s, and 118 / 400 = 29.5 %. Counting the weights only is
          PyTorch's definition; it ignores the KV cache and the activations, which is fine at short context and flattering
          at long context.
        </p>
      </Translated>
    ),
    source: `${FACTS} A1 (the definition)`,
    vi: { source: `${FACTS} A1 (định nghĩa)` },
  },
  'x2/X2.1/q-int-mm': {
    kind: 'choice',
    id: 'x2/X2.1/q-int-mm',
    prompt: (
      <Translated vi={<>Vì sao decode batch 1 trong PyTorch trên CUDA không dùng được <code>torch._int_mm</code>, và thứ gì phục vụ nó thay vào đó?</>}>
        <>Why can batch-1 decode in PyTorch on CUDA not use <code>torch._int_mm</code>, and what serves it instead?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Trên CUDA nó từ chối chiều thứ nhất từ 16 trở xuống; các op chỉ lượng tử hóa trọng số phục vụ decode.">{'On CUDA it refuses a first dimension of 16 or less; weight-only ops serve decode.'}</Translated>, correct: true },
      { label: <Translated vi="_int_mm không có bản cài đặt CUDA, chỉ có bản CPU và MPS.">{'_int_mm has no CUDA implementation, only CPU and MPS ones.'}</Translated>, why: <Translated vi="Nó dispatch cho CPU, CUDA, MPS và XPU. Trên CUDA nó kiểm tra kích thước và từ chối M nhỏ.">{'It dispatches CPU, CUDA, MPS and XPU. On CUDA it checks the sizes and refuses small M.'}</Translated> },
      { label: <Translated vi="Nó sẽ chạy, nhưng activation int8 mất quá nhiều độ chính xác cho decode.">{'It would run, but int8 activations lose too much accuracy for decode.'}</Translated>, why: <Translated vi="Nó không chạy: phép kiểm tra kích thước chặn nó trước mọi phép tính.">{'It does not work: the size check stops it before any arithmetic.'}</Translated> },
      { label: <Translated vi="Nó cần trọng số int4, được đóng gói trước bằng _convert_weight_to_int4pack.">{'It needs int4 weights, packed by _convert_weight_to_int4pack first.'}</Translated>, why: <Translated vi="Nó nhân int8 với int8 ra int32.">{'It multiplies int8 by int8 into int32.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            GEMV lượng tử hóa và GEMM lượng tử hóa là hai kernel khác nhau. Một thư viện GEMM được xây cho nhiều hàng; decode
            chỉ có một. Kernel chỉ lượng tử hóa trọng số giữ activation ở dạng float và chỉ thu nhỏ những gì đi qua bus.
          </p>
        }
      >
        <p>
          The quantised GEMV and the quantised GEMM are different kernels. A GEMM library is built for many rows; decode has
          one. Weight-only kernels keep the activations in float and only shrink what crosses the bus.
        </p>
      </Translated>
    ),
    source: `${FACTS} A6, D6`,
  },
  'x2/X2.1/q-regime-q8': {
    kind: 'choice',
    id: 'x2/X2.1/q-regime-q8',
    prompt: (
      <Translated
        vi={
          <>
            Một GEMV Q8_0 ở batch 1 đạt 2 / 1.0625 = 1.88 FLOP/B. Thiết bị A: 200 GFLOP/s và 100 GB/s. Thiết bị B: 300 GFLOP/s
            và 250 GB/s. Roofline nói gì?
          </>
        }
      >
        <>
          A Q8_0 GEMV at batch 1 does 2 / 1.0625 = 1.88 FLOP/B. Device A: 200 GFLOP/s and 100 GB/s. Device B: 300 GFLOP/s
          and 250 GB/s. What does the roofline say?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Bị giới hạn bởi bộ nhớ trên A; bị giới hạn bởi tính toán trên B, nơi ngưỡng thấp hơn.">{'Memory-bound on A; compute-bound on B, whose threshold is lower.'}</Translated>, correct: true },
      { label: <Translated vi="Bị giới hạn bởi bộ nhớ trên cả hai: decode bị giới hạn bởi bộ nhớ trên mọi thiết bị.">{'Memory-bound on both: decode is memory-bound on any device.'}</Translated>, why: <Translated vi="Ngưỡng của B là 300 / 250 = 1.2 FLOP/B, thấp hơn 1.88.">{"B's threshold is 300 / 250 = 1.2 FLOP/B, below 1.88."}</Translated> },
      { label: <Translated vi="Bị giới hạn bởi tính toán trên cả hai, vì Q8_0 gần như gấp đôi cường độ tính toán của FP16.">{'Compute-bound on both, since Q8_0 nearly doubles FP16\'s intensity.'}</Translated>, why: <Translated vi="Ngưỡng của A là 2.0 FLOP/B, cao hơn 1.88.">{"A's threshold is 2.0 FLOP/B, above 1.88."}</Translated> },
      { label: <Translated vi="Chỉ bị giới hạn bởi bộ nhớ trên B, vì B có nhiều băng thông hơn cho mỗi phép toán.">{'Memory-bound on B only, since B has more bandwidth per operation.'}</Translated>, why: <Translated vi="B có nhiều băng thông hơn cho mỗi phép toán, nên ngưỡng của nó thấp hơn.">{'B has more bandwidth per operation, so its threshold is lower.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            "Decode bị giới hạn bởi bộ nhớ" là một nhận định về cặp kernel và thiết bị. Trên một thiết bị có ít phép tính trên
            mỗi byte, một GEMV lượng tử hóa có thể vượt ngưỡng ngay ở batch 1, và khi đó phần việc số nguyên và các hệ số tỉ
            lệ, thứ mô hình của trang này coi là miễn phí, chính là thứ bạn phải chờ.
          </p>
        }
      >
        <p>
          "Decode is memory-bound" is a statement about the pair of kernel and device. On a device with little arithmetic
          per byte, a quantised GEMV can cross the threshold at batch 1, and then the integer work and the scales, which
          this page's model counts as free, are what you are waiting for.
        </p>
      </Translated>
    ),
    source: 'adapted from ch. 5 exercise 9, p. 119',
    vi: { source: 'phỏng theo ch. 5 bài tập 9, p. 119' },
  },
};

// ---------------------------------------------------------------- hardware
interface Hardware extends Device {
  key: string;
  label: string;
  note: string;
}
const BOOK: Hardware = {
  ...BOOK_DEVICE, key: 'book',
  get label() {
    return tr("The book's example (H100)", 'Ví dụ của sách (H100)');
  },
  get note() {
    return tr(
      "The book's example device, p. 94: 66.9 TFLOPS, 3.35 TB/s, threshold 20 FLOP/B.",
      'Thiết bị ví dụ của sách, p. 94: 66.9 TFLOPS, 3.35 TB/s, ngưỡng 20 FLOP/B.',
    );
  },
};

function useHardware() {
  const { reports } = useReports();
  const mine = useMemo(
    () => measuredDevices(reports).map((d): Hardware => ({
      ...d, key: d.backend,
      get label() {
        return tr(`Yours: ${d.name} (${d.backend})`, `Của bạn: ${d.name} (${d.backend})`);
      },
      get note() {
        return tr(
          `Measured by lab 5.1 on ${d.date}: two probe kernels, not the vendor's figures.`,
          `Đo bằng lab 5.1 vào ${d.date}: hai kernel thăm dò, không phải số liệu của nhà sản xuất.`,
        );
      },
    })),
    [reports],
  );
  return { reports, options: [BOOK, ...mine] };
}

// ---------------------------------------------------------------- the decode roofline
const EXPLORER_FORMATS: FormatId[] = ['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'q4_k', 'mxfp4', 'mlx4_g32', 'mlx4_g64', 'mlx4_g128', 'mlx4_g32_f32', 'mlx4_g64_f32', 'mlx4_g128_f32'];
const fmtLabel = (f: FormatId) => (f.startsWith('mlx') ? FORMATS[f].label : `${FORMATS[f].label}${FORMATS[f].owner.startsWith('llama') || f === 'mxfp4' ? ` (${FORMATS[f].owner})` : ''}`);

interface Mark {
  label: string;
  x: number;
  y?: number;
  kind: 'model' | 'ref' | 'measured';
}

function Roofline({ hw, marks }: { hw: Device; marks: Mark[] }) {
  const W = 680;
  const H = 360;
  const pad = { l: 64, r: 16, t: 14, b: 44 };
  const T = thresholdOf(hw);
  const peak = T * hw.bw;
  const xMin = 0.1;
  const xMax = 1000;
  const yMax = peak * 3;
  const yMin = hw.bw * xMin * 0.5;
  const X = (v: number) => pad.l + ((Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const roof = (x: number) => Math.min(peak, hw.bw * x);
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e++) yt.push(10 ** e);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Roofline of the decode GEMV', 'Roofline của GEMV decode')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      <rect x={pad.l} y={pad.t} width={X(T) - pad.l} height={H - pad.t - pad.b} fill="var(--global-wash)" />
      <text x={pad.l + 8} y={pad.t + 16} fontSize="11" style={{ fill: 'var(--global)' }}><Translated vi="bị giới hạn bởi bộ nhớ">memory-bound</Translated></text>
      <text x={X(T) + 8} y={pad.t + 16} fontSize="11" style={{ fill: 'var(--reg)' }}><Translated vi="bị giới hạn bởi tính toán">compute-bound</Translated></text>
      {[0.1, 1, 10, 100, 1000].map((v) => (
        <g key={'x' + v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle">{v}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{v >= 1000 ? v / 1000 + 'k' : v}</text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="11.5" textAnchor="middle"><Translated vi="FLOP trên mỗi byte lưu lượng">FLOP per byte of traffic</Translated></text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}>GFLOP/s</text>
      <polyline points={`${X(xMin)},${Y(hw.bw * xMin)} ${X(T)},${Y(peak)} ${X(xMax)},${Y(peak)}`} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
      <line x1={X(T)} x2={X(T)} y1={Y(peak)} y2={H - pad.b} stroke="var(--ink)" strokeDasharray="3 3" />
      <text x={X(T)} y={H - pad.b - 6} fontSize="11" textAnchor="middle"><Translated vi={<>ngưỡng {fmt(T)}</>}>threshold {fmt(T)}</Translated></text>
      {marks.map((m, i) => {
        const x = Math.min(Math.max(m.x, xMin), xMax);
        const cx = X(x);
        if (m.kind !== 'measured') {
          const cy = Y(roof(x));
          return (
            <g key={i} data-mark={m.kind} data-x={m.x.toFixed(3)}>
              <circle cx={cx} cy={cy} r="5" fill={m.kind === 'model' ? 'var(--ink)' : 'var(--page)'} stroke="var(--ink)" strokeWidth="1.8" />
              <text x={m.kind === 'model' ? cx + 8 : cx - 8} y={cy - 8} fontSize="11" textAnchor={m.kind === 'model' ? 'start' : 'end'}>{m.label}</text>
            </g>
          );
        }
        const above = (m.y ?? 0) > roof(x) * 1.02;
        return (
          <g key={i} data-mark="measured">
            <rect x={cx - 4.5} y={Y(m.y!) - 4.5} width="9" height="9" transform={`rotate(45 ${cx} ${Y(m.y!)})`} fill={above ? 'var(--fail)' : 'var(--ok)'} />
            <text x={cx + 9} y={Y(m.y!) + 14} fontSize="11" style={{ fill: above ? 'var(--fail)' : 'var(--ok)' }}>{m.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function DecodeRoofline() {
  const { reports, options } = useHardware();
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState({ peak: 10000, bw: 500 });
  const [f, setF] = useState<FormatId>('q4_0');
  const [batch, setBatch] = useState(1);
  const [lid, setLid] = useState('4096x4096');
  const [act, setAct] = useState(2);
  const hw: Hardware = key === 'custom' ? { key, label: tr('Custom', 'Tùy chỉnh'), ...custom, note: tr('Numbers you typed.', 'Các số bạn đã nhập.') } : (options.find((o) => o.key === key) ?? BOOK);
  const L = layer(lid);
  const t = gemv(f, L, batch, act);
  const c = ceiling(t, hw);
  const ref = gemv('f32', L, batch, act);
  const cRef = ceiling(ref, hw);
  const sp = speedupOverF32(f, L, batch, act, hw);
  const T = thresholdOf(hw);
  const knee = kneeBatch(f, L, act, T);
  const kneeF32 = kneeBatch('f32', L, 4, T);
  const rows = pickRuns(qgemvRows(reports), hw.key, L, batch, f);
  const runF = rows.find((r) => r.format === f);
  const runRef = rows.find((r) => r.format === 'f32');
  const marks: Mark[] = [
    { label: tr(`${FORMATS[f].label}, batch ${batch}`, `${FORMATS[f].label}, batch ${batch}`), x: t.intensity, kind: 'model' },
    ...(f === 'f32' ? [] : [{ label: tr(`float32 weights, batch ${batch}`, `trọng số float32, batch ${batch}`), x: ref.intensity, kind: 'ref' as const }]),
    ...rows.map((r): Mark => ({ label: tr(`yours, ${FORMATS[r.format].label} (lab ${r.labNum})`, `của bạn, ${FORMATS[r.format].label} (lab ${r.labNum})`), x: r.intensity, y: r.gflops, kind: 'measured' })),
  ];
  return (
    <Bench
      title={tr('Decode GEMV roofline by weight format', 'Roofline của GEMV decode theo định dạng trọng số')}
      task={tr('Pick a format and drag the batch. Find where the speed-up stops being the byte ratio.', 'Chọn một định dạng và kéo batch. Tìm chỗ mà mức tăng tốc không còn bằng tỉ lệ byte.')}
      foot={
        <Translated
          vi={
            <>
              Thang logarit. Hình tròn đặc là định dạng bạn chọn, hình tròn rỗng là trọng số float32 với cùng activation và
              batch. Hình thoi đặc là các lần chạy lab X2.1 hoặc X2.2 của chính bạn với lớp, định dạng và batch này, có float32 bên cạnh, khi thiết bị này có chúng. {hw.note}
            </>
          }
        >
          <>
            Log scales. The filled circle is the format you picked, the open circle float32 weights with the same
            activations and batch. Filled diamonds are your own lab X2.1 or X2.2 runs of this layer, format and batch, with float32 beside them, when this device has them. {hw.note}
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Phần cứng">Hardware</Translated>
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label={tr('Hardware', 'Phần cứng')}>
            {options.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
            <option value="custom">{tr('Custom numbers', 'Số tùy chỉnh')}</option>
          </select>
        </label>
        {key === 'custom' ? (
          <>
            <label className="field">
              <Translated vi="Đỉnh (GFLOP/s)">Peak (GFLOP/s)</Translated>
              <input type="number" min="1" value={custom.peak} aria-label={tr('Peak', 'Đỉnh')} onChange={(e) => setCustom({ ...custom, peak: Math.max(1, +e.target.value) })} />
            </label>
            <label className="field">
              <Translated vi="Băng thông (GB/s)">Bandwidth (GB/s)</Translated>
              <input type="number" min="1" value={custom.bw} aria-label={tr('Bandwidth', 'Băng thông')} onChange={(e) => setCustom({ ...custom, bw: Math.max(1, +e.target.value) })} />
            </label>
          </>
        ) : null}
        <label className="field">
          <Translated vi="Định dạng trọng số">Weight format</Translated>
          <select value={f} onChange={(e) => setF(e.target.value as FormatId)} aria-label={tr('Weight format', 'Định dạng trọng số')}>
            {EXPLORER_FORMATS.map((x) => <option key={x} value={x}>{fmtLabel(x)}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Lớp">Layer</Translated>
          <select value={lid} onChange={(e) => setLid(e.target.value)} aria-label={tr('Layer', 'Lớp')}>
            {LAYERS.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Activation">Activations</Translated>
          <span className="seg">
            {[4, 2].map((a) => (
              <button key={a} aria-pressed={act === a} onClick={() => setAct(a)}>{a === 4 ? 'float32' : tr('16-bit', '16 bit')}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Batch = {batch}
          <input type="range" min="1" max="64" value={batch} aria-label={tr('Batch', 'Batch')} onChange={(e) => setBatch(+e.target.value)} />
        </label>
      </div>
      <div className="cols plot">
        <Roofline hw={hw} marks={marks} />
        <div className="panel" data-panel="roofline">
          <Translated
            vi={
              <p style={{ margin: 0 }}>
                {FORMATS[f].label}: {bytesPerWeight(f)} byte, {bitsPerWeight(f)} bit mỗi trọng số. Trọng số {fmt(t.weightBytes / 1e6)} MB,
                activation và đầu ra {fmt((t.inputBytes + t.outputBytes) / 1e3)} kB.
              </p>
            }
          >
            <p style={{ margin: 0 }}>
              {FORMATS[f].label}: {bytesPerWeight(f)} bytes, {bitsPerWeight(f)} bits per weight. Weights {fmt(t.weightBytes / 1e6)} MB,
              activations and outputs {fmt((t.inputBytes + t.outputBytes) / 1e3)} kB.
            </p>
          </Translated>
          <p style={{ fontSize: '1.4rem', margin: '0.5rem 0 0.2rem', fontWeight: 600 }} data-intensity={t.intensity.toFixed(3)}>
            {fmt(t.intensity)} FLOP/B
          </p>
          <p style={{ margin: '0 0 0.6rem' }}>
            {c.bound === 'memory' ? <span className="chip global"><Translated vi="bị giới hạn bởi bộ nhớ">memory-bound</Translated></span> : <span className="chip reg"><Translated vi="bị giới hạn bởi tính toán">compute-bound</Translated></span>}{' '}
            <span className="muted"><Translated vi={<>ngưỡng {fmt(T)} FLOP/B</>}>threshold {fmt(T)} FLOP/B</Translated></span>
          </p>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr><td><Translated vi="Thời gian trần, lớp này">Ceiling time, this layer</Translated></td><td className="num nowrap" data-time>{us(c.seconds)}</td></tr>
              <tr><td><Translated vi="Trọng số float32, cùng activation">Float32 weights, same activations</Translated></td><td className="num nowrap">{us(cRef.seconds)}</td></tr>
              <tr><td><Translated vi="Mức tăng tốc dự đoán so với float32">Predicted speed-up over float32</Translated></td><td className="num" data-speedup={sp.toFixed(3)}><b>{sp.toFixed(2)}×</b></td></tr>
              <tr><td><Translated vi="Tỉ lệ byte của riêng trọng số">Byte ratio of the weights alone</Translated></td><td className="num" data-ratio={byteRatio(f).toFixed(3)}>{byteRatio(f).toFixed(2)}×</td></tr>
              <tr><td><Translated vi="Batch mà tại đó định dạng này chuyển sang bị giới hạn bởi tính toán">Batch where this format turns compute-bound</Translated></td><td className="num" data-knee={knee ?? 'none'}>{knee ?? tr('never', 'không bao giờ')}</td></tr>
              {runF && runRef && f !== 'f32' ? (
                <tr>
                  <td><Translated vi={<>Mức tăng tốc bạn đo được (lab {runF.labNum})</>}>Your measured speed-up (lab {runF.labNum})</Translated></td>
                  <td className="num" data-measured-speedup={(runRef.seconds / runF.seconds).toFixed(3)}><b>{(runRef.seconds / runF.seconds).toFixed(2)}×</b></td>
                </tr>
              ) : null}
              <tr><td><Translated vi="Điều tương tự cho trọng số và activation float32">The same for float32 weights and activations</Translated></td><td className="num" data-knee-f32={kneeF32 ?? 'none'}>{kneeF32 ?? tr('never', 'không bao giờ')}</td></tr>
            </tbody>
          </table>
          {rows.length === 0 && key !== 'book' && key !== 'custom' ? (
            <Translated
              vi={
                <p className="muted" style={{ margin: '0.6rem 0 0' }} data-no-runs>
                  Thiết bị này chưa có lần chạy nào của lớp, định dạng và batch này. Lab X2.1 và X2.2 đo thời gian float32, Q8_0
                  và Q4_0 ở batch 1, 4 và 8: <code>python3 labs/pmpp.py run X2.2 --bench</code>.
                </p>
              }
            >
              <p className="muted" style={{ margin: '0.6rem 0 0' }} data-no-runs>
                No run of this layer, format and batch on this device. Labs X2.1 and X2.2 time float32, Q8_0 and Q4_0 at
                batches 1, 4 and 8: <code>python3 labs/pmpp.py run X2.2 --bench</code>.
              </p>
            </Translated>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- fused or unfused
const PIPE_FORMATS: FormatId[] = ['q4_0', 'q8_0', 'mlx4_g64'];

function FusedOrUnfused() {
  const { reports } = useReports();
  const [f, setF] = useState<FormatId>('q4_0');
  const [on, setOn] = useState<Record<PipelineId, boolean>>({ f32: true, 'unfused-f32': true, 'unfused-f16': false, fused: false });
  const L = layer('4096x4096');
  const maxBpw = Math.max(...PIPELINES.map((p) => pipelineBytesPerWeight(p.id, f)));
  const rows = qgemvRows(reports);
  const measured = f === 'q4_0'
    ? [...new Set(rows.map((r) => r.backend))].flatMap((b) => { const m = measuredPipelines(rows, b, L, 1); return m ? [{ b, m }] : []; })
    : [];
  return (
    <Bench
      title={tr('Fused or unfused', 'Gộp (fused) hay không gộp (unfused)')}
      task={tr('Switch the four pipelines on and off. Which ones keep the byte ratio?', 'Bật và tắt bốn pipeline. Pipeline nào giữ được tỉ lệ byte?')}
      foot={<Translated vi={<>Mô hình byte: một lớp 4096 → 4096 ở batch 1 với activation float32, mọi byte cùng một băng thông, không có bộ nhớ đệm. Thời gian tính tương đối so với GEMV float32. Với Q4_0, thời gian lab X2.2 của chính bạn cho cùng lớp xuất hiện bên cạnh mô hình khi có.</>}><>Bytes model: a 4096 → 4096 layer at batch 1 with float32 activations, every byte at the same bandwidth, no caches. Time is relative to the float32 GEMV. With Q4_0, your own lab X2.2 times for the same layer appear beside the model where they exist.</></Translated>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Định dạng lượng tử hóa">Quantised format</Translated>
          <select value={f} onChange={(e) => setF(e.target.value as FormatId)} aria-label={tr('Quantised format', 'Định dạng lượng tử hóa')}>
            {PIPE_FORMATS.map((x) => <option key={x} value={x}>{fmtLabel(x)}</option>)}
          </select>
        </label>
        {PIPELINES.map((p) => (
          <label key={p.id} className="check">
            <input type="checkbox" checked={on[p.id]} onChange={(e) => setOn({ ...on, [p.id]: e.target.checked })} /> {p.label}
          </label>
        ))}
      </div>
      <div className="panel" data-panel="pipelines">
        {PIPELINES.filter((p) => on[p.id]).map((p) => {
          const bpw = pipelineBytesPerWeight(p.id, f);
          const rel = pipelineRelative(p.id, f, L, 1, 4);
          return (
            <div key={p.id} data-pipeline={p.id} data-bpw={bpw} data-rel={rel.toFixed(3)} style={{ margin: '0 0 0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
                <span>{p.label}{p.kernels > 1 ? <span className="muted"><Translated vi=" (hai kernel)"> (two kernels)</Translated></span> : null}</span>
                <span className="num">
                  <Translated vi={<>{bpw} byte mỗi trọng số, <b>{rel.toFixed(2)}×</b> thời gian float32</>}>
                    {bpw} bytes per weight, <b>{rel.toFixed(2)}×</b> the float32 time
                  </Translated>
                  {measured.map(({ b, m }) => (m[p.id] != null ? (
                    <span key={b} data-measured={b} data-value={m[p.id]!.toFixed(3)}><Translated vi={<>; đo trên {b} của bạn: <b>{m[p.id]!.toFixed(2)}×</b></>}>; measured on your {b}: <b>{m[p.id]!.toFixed(2)}×</b></Translated></span>
                  ) : null))}
                </span>
              </div>
              <svg viewBox="0 0 600 16" preserveAspectRatio="none" /* i18n-ignore: an SVG attribute value */ style={{ width: '100%', height: 16, display: 'block' }} aria-hidden="true">
                <rect x="0" y="2" width={(600 * bpw) / maxBpw} height="12" fill={rel > 1.001 ? 'var(--fail)' : rel < 0.999 ? 'var(--ok)' : 'var(--ink-3)'} />
              </svg>
            </div>
          );
        })}
        {PIPELINES.every((p) => !on[p.id]) ? <p className="muted" style={{ margin: 0 }}><Translated vi="Hãy bật một pipeline.">Switch a pipeline on.</Translated></p> : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- bandwidth utilisation
function UtilisationCalc() {
  const { reports } = useReports();
  const rows = qgemvRows(reports);
  const [sel, setSel] = useState(0);
  const r: QgemvRow | undefined = rows[Math.min(sel, rows.length - 1)];
  const bw = r ? measuredBandwidth(reports, r.backend) : null;
  const [hand, setHand] = useState({ params: 8, f: 'q8_0' as FormatId, tps: 20, bw: 300 });
  const handBytes = modelBytes(hand.params * 1e9, hand.f);
  const handMbu = mbu(handBytes, 1 / Math.max(hand.tps, 1e-9), Math.max(hand.bw, 1e-9));
  return (
    <Bench
      title={tr('Your bandwidth utilisation', 'Mức sử dụng băng thông của bạn')}
      task={tr('Pick one of your runs, or type a model, a format, a speed and a bandwidth.', 'Chọn một lần chạy của bạn, hoặc nhập một mô hình, một định dạng, một tốc độ và một băng thông.')}
      foot={<Translated vi={<>Số byte trọng số chia cho thời gian, chia cho băng thông mà lab 5.1 đo trên cùng thiết bị: định nghĩa của PyTorch về mức sử dụng băng thông của mô hình ({FACTS} A1). Không bao giờ là số liệu của nhà sản xuất.</>}><>Weight bytes over time, over the bandwidth lab 5.1 measured on the same device: PyTorch's definition of model bandwidth utilisation ({FACTS} A1). Never a vendor's figure.</></Translated>}
    >
      <div className="cols">
        <div className="panel" data-panel="mbu-runs">
          <p style={{ margin: '0 0 0.5rem' }}><b><Translated vi="Từ các lab của bạn">From your labs</Translated></b></p>
          {rows.length ? (
            <>
              <label className="field">
                <Translated vi="Lần chạy">Run</Translated>
                <select value={sel} onChange={(e) => setSel(+e.target.value)} aria-label={tr('Run', 'Lần chạy')}>
                  {rows.map((x, i) => (
                    <option key={i} value={i}>lab {x.labNum}, {x.backend}, {FORMATS[x.format].label} ({x.kernel}), {x.in} → {x.out}, batch {x.batch}{/* i18n-ignore: lab and batch stay English per the glossary; the rest are values */}</option>
                  ))}
                </select>
              </label>
              {r ? (
                <table className="data" style={{ margin: '0.5rem 0 0' }}>
                  <tbody>
                    <tr><td><Translated vi="Số byte trọng số">Weight bytes</Translated></td><td className="num">{fmt(r.weightBytes / 1e6)} MB</td></tr>
                    <tr><td><Translated vi="Thời gian">Time</Translated></td><td className="num">{us(r.seconds)}</td></tr>
                    <tr><td><Translated vi="Băng thông trọng số">Weight bandwidth</Translated></td><td className="num">{fmt(r.weightGBs)} GB/s</td></tr>
                    <tr>
                      <td><Translated vi="Trên băng thông lab 5.1 của bạn">Of your lab 5.1 bandwidth</Translated></td>
                      <td className="num" data-mbu-run>{bw ? `${(100 * mbu(r.weightBytes, r.seconds, bw)).toFixed(1)} %` : tr('run lab 5.1 on this device', 'hãy chạy lab 5.1 trên thiết bị này')}</td>
                    </tr>
                  </tbody>
                </table>
              ) : null}
            </>
          ) : (
            <Translated
              vi={
                <p style={{ margin: 0 }} data-empty="runs">
                  Máy này chưa có lần chạy nào của lab X2.1 hay X2.2. <code>python3 labs/pmpp.py run X2.1 --bench</code> và{' '}
                  <code>python3 labs/pmpp.py run X2.2 --bench</code> tạo ra chúng, còn <code>python3 labs/pmpp.py run 5.1 --bench</code>{' '}
                  đo băng thông để so sánh với chúng (<a href={href('x2', 'labs')}>labs</a>).
                </p>
              }
            >
              <p style={{ margin: 0 }} data-empty="runs">
                No runs of labs X2.1 or X2.2 on this machine yet. <code>python3 labs/pmpp.py run X2.1 --bench</code> and{' '}
                <code>python3 labs/pmpp.py run X2.2 --bench</code> produce them, and <code>python3 labs/pmpp.py run 5.1 --bench</code>{' '}
                measures the bandwidth they are compared with (<a href={href('x2', 'labs')}>labs</a>).
              </p>
            </Translated>
          )}
        </div>
        <div className="panel" data-panel="mbu-hand">
          <p style={{ margin: '0 0 0.5rem' }}><b><Translated vi="Tính tay">By hand</Translated></b> <span className="muted"><Translated vi="(một thiết bị giả định)">(a hypothetical device)</Translated></span></p>
          <div className="row">
            <label className="field">
              <Translated vi="Trọng số, tỉ">Weights, billions</Translated>
              <input type="number" min="0.1" step="0.1" value={hand.params} aria-label={tr('Weights, billions', 'Trọng số, tỉ')} onChange={(e) => setHand({ ...hand, params: +e.target.value })} />
            </label>
            <label className="field">
              <Translated vi="Định dạng">Format</Translated>
              <select value={hand.f} aria-label={tr('Format', 'Định dạng')} onChange={(e) => setHand({ ...hand, f: e.target.value as FormatId })}>
                {(['f16', 'q8_0', 'q4_0', 'q4_k', 'mlx4_g64'] as FormatId[]).map((x) => <option key={x} value={x}>{FORMATS[x].label.replace('MLX affine 4-bit', 'MLX 4-bit')}</option>)}
              </select>
            </label>
            <label className="field">
              <Translated vi="Token mỗi giây">Tokens per second</Translated>
              <input type="number" min="0.1" step="0.1" value={hand.tps} aria-label={tr('Tokens per second', 'Token mỗi giây')} onChange={(e) => setHand({ ...hand, tps: +e.target.value })} />
            </label>
            <label className="field">
              <Translated vi="Băng thông đỉnh, GB/s">Peak bandwidth, GB/s</Translated>
              <input type="number" min="1" value={hand.bw} aria-label={tr('Peak bandwidth', 'Băng thông đỉnh')} onChange={(e) => setHand({ ...hand, bw: +e.target.value })} />
            </label>
          </div>
          <p style={{ margin: '0.5rem 0 0' }}>
            <Translated
              vi={
                <>
                  {fmt(handBytes / 1e9)} GB mỗi token × {fmt(hand.tps)} = {fmt((handBytes * hand.tps) / 1e9)} GB/s:{' '}
                  <b data-mbu-hand={(100 * handMbu).toFixed(1)}>{(100 * handMbu).toFixed(1)} %</b> của đỉnh.
                </>
              }
            >
              {fmt(handBytes / 1e9)} GB per token × {fmt(hand.tps)} = {fmt((handBytes * hand.tps) / 1e9)} GB/s:{' '}
              <b data-mbu-hand={(100 * handMbu).toFixed(1)}>{(100 * handMbu).toFixed(1)} %</b> of the peak.
            </Translated>
            {handMbu > 1 ? <span className="chip fail" style={{ marginLeft: '0.4rem' }}><Translated vi="trên 100 %: có thứ gì đó được lưu đệm, hoặc các số không khớp nhau">above 100 %: something is cached or the numbers disagree</Translated></span> : null}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the section
const TABLE_FORMATS: FormatId[] = ['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'q4_k', 'mxfp4', 'mlx4_g32', 'mlx4_g64', 'mlx4_g128', 'mlx4_g64_f32'];

export function SX21() {
  const q = questionsSX21;
  const book = gemv('f16', layer('1024x4096'), 1, 2);
  return (
    <>
      <Translated
        vi={
          <p>
            Một phép nhân ma trận với vector thực hiện một phép nhân và một phép cộng cho mỗi trọng số và đọc mỗi trọng số một
            lần. Cường độ tính toán của nó là 2 chia cho số byte mỗi trọng số, bất kể mã trông thế nào. Với lớp của sách, 1024
            đầu vào sang 4096 đầu ra ở 16 bit với batch 1, con số đó là {book.intensity.toFixed(3)} FLOP/B: "khoảng 1" của sách.
            Decode ở batch 1 được tạo nên từ những phép nhân như vậy, và nó nằm xa về bên trái ngưỡng của bất kỳ GPU nào.
          </p>
        }
      >
        <p>
          A matrix-vector product does one multiply and one add for each weight and reads each weight once. Its intensity is
          2 divided by the bytes per weight, whatever the code looks like. For the book's layer, 1024 inputs to 4096 outputs in
          16-bit at batch 1, that is {book.intensity.toFixed(3)} FLOP/B: the book's "about 1". Decode at batch 1 is made of
          such products, and it sits far to the left of any GPU's threshold.
        </p>
      </Translated>
      <Source section="5.1" pages="93–97" />
      <Translated
        vi={
          <p>
            Chương 5 đưa ra hai cách để dịch sang phải trên roofline: dùng lại mỗi byte cho nhiều phép toán hơn, hoặc nạp ít
            byte hơn cho mỗi phép toán. Chương 20 và 22 đi theo cách thứ nhất, batching. Trang này đi theo cách thứ hai.
          </p>
        }
      >
        <p>
          Chapter 5 gave two ways to the right on the roofline: reuse each byte for more operations, or fetch fewer bytes per
          operation. Chapters 20 and 22 took the first, batching. This page takes the second.
        </p>
      </Translated>
      <Source section="20.6" pages="504–505" />

      <h2><Translated vi="Ít byte hơn cho mỗi phép toán">Fewer bytes per operation</Translated></h2>
      <Translated
        vi={
          <p>
            Phụ lục A nêu lập luận trong một câu: một định dạng nhỏ hơn nạp ít byte hơn cho mỗi phép toán số học, nên cường độ
            tính toán của ứng dụng tăng lên. Sách áp dụng nó cho phép toán dấu phẩy động.
          </p>
        }
      >
        <p>
          Appendix A gives the argument in one sentence: a smaller format fetches fewer bytes for each arithmetic operation, so
          the application's intensity rises. The book makes it about floating-point arithmetic.
        </p>
      </Translated>
      <Source section="A.5" pages="593" />
      <Translated
        vi={
          <p>
            Ở đây nó được áp dụng cho riêng trọng số, được lưu dưới dạng số nguyên nhỏ với một hệ số tỉ lệ cho mỗi khối, trong
            khi phép tính vẫn ở float hoặc int32. Activation giữ nguyên định dạng; chỉ những gì đi qua bus là thu nhỏ lại.
          </p>
        }
      >
        <p>
          Here it is applied to the weights alone, stored as small integers with a scale per block, while the arithmetic
          stays in float or int32. The activations keep their format; only what crosses the bus shrinks.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Việc áp dụng §A.5 cho các định dạng trọng số số nguyên là của khóa học này. Chính blog của PyTorch đưa ra cùng
              lập luận theo cách nói quen thuộc với bạn: số tham số và băng thông là cố định, "But, we can change how many bytes each parameter is stored in!" (
              {FACTS} A1).
            </p>
          }
        >
          <p>
            Applying §A.5 to integer weight formats is ours. PyTorch's own blog makes the same argument in your terms: the
            parameter count and the bandwidth are fixed, "But, we can change how many bytes each parameter is stored in!" (
            {FACTS} A1).
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Số byte mỗi trọng số, gồm cả hệ số tỉ lệ">Bytes per weight, scales included</Translated></h2>
      <Translated
        vi={
          <p>
            Một định dạng lượng tử hóa không bao giờ chỉ là độ rộng mã của nó. Mỗi khối mã mang theo những gì cần để biến chúng
            trở lại thành số: một hệ số tỉ lệ, đôi khi một giá trị nhỏ nhất hoặc một độ lệch. Những byte đó cũng đi qua bus ở
            mỗi token.
          </p>
        }
      >
        <p>
          A quantised format is never just its code width. Every block of codes carries what is needed to turn them back into
          numbers: a scale, sometimes a minimum or a bias. Those bytes cross the bus on every token too.
        </p>
      </Translated>
      <table className="data wide" data-table="bpw" style={{ maxWidth: '50rem' }}>
        <thead>
          <tr>
            <th><Translated vi="Định dạng">Format</Translated></th>
            <th><Translated vi="Của ai">Whose</Translated></th>
            <th className="num"><Translated vi="Khối">Block</Translated></th>
            <th className="num"><Translated vi="Byte mỗi khối">Bytes per block</Translated></th>
            <th className="num"><Translated vi="Byte mỗi trọng số">Bytes per weight</Translated></th>
            <th className="num">GEMV FLOP/B{/* i18n-ignore: a kernel name and a unit */}</th>
          </tr>
        </thead>
        <tbody>
          {TABLE_FORMATS.map((f) => (
            <tr key={f} data-format={f}>
              <td>{FORMATS[f].label.replace('MLX affine 4-bit, ', 'affine 4-bit, ')}</td>
              <td>{FORMATS[f].owner}</td>
              <td className="num">{FORMATS[f].blockWeights}</td>
              <td className="num">{FORMATS[f].blockBytes}</td>
              <td className="num" data-bpw>{bytesPerWeight(f)}</td>
              <td className="num">{fmt(weightOnlyIntensity(f))}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Q4_K đạt 4.5 bit với một hệ số tỉ lệ và một giá trị nhỏ nhất cho mỗi 32 trọng số bằng cách lưu chúng trong 6 bit mỗi
            giá trị, dưới một cặp 16 bit cho mỗi 256. Hệ số tỉ lệ của MXFP4 là một byte duy nhất, một lũy thừa của hai. Mục{' '}
            <a href={href('x2', 'X2.2')}>X2.2</a> xây dựng các định dạng này; lúc này chúng chỉ là những con số đếm byte.
          </p>
        }
      >
        <p>
          Q4_K gets to 4.5 bits with a scale and a minimum for every 32 weights by storing those in 6 bits each under one
          16-bit pair per 256. MXFP4's scale is a single byte, a power of two. Section{' '}
          <a href={href('x2', 'X2.2')}>X2.2</a> builds these formats; for now they are just byte counts.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${FACTS}, D1, D2, D5, E1; appA-facts D2)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, D1, D2, D5, E1; appA-facts D2)`)}>
        <Translated
          vi={
            <p>
              llama.cpp: <code>block_q8_0</code> 34 byte, <code>block_q4_0</code> 18, <code>block_q4_1</code> 20 cho mỗi 32
              trọng số; Q4_K 144 cho mỗi 256. MLX: nhóm affine 32, 64 hoặc 128, hệ số tỉ lệ và độ lệch cùng kiểu với đầu vào.
            </p>
          }
        >
          <p>
            llama.cpp: <code>block_q8_0</code> 34 bytes, <code>block_q4_0</code> 18, <code>block_q4_1</code> 20 per 32
            weights; Q4_K 144 per 256. MLX: affine groups of 32, 64 or 128, scales and biases of the input's type.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Roofline của decode, theo định dạng">The roofline of decode, by format</Translated></h2>
      <Translated
        vi={
          <p>
            Chọn một định dạng và để batch ở 1. Điểm dịch sang phải khi số byte giảm, nhưng trên mọi thiết bị thật nó vẫn nằm
            trên đoạn dốc, nên thời gian là số byte chia cho băng thông, và mức tăng tốc dự đoán so với float32 là tỉ lệ số
            byte. Sau đó hãy kéo batch.
          </p>
        }
      >
        <p>
          Pick a format and leave the batch at 1. The point moves right as the bytes fall, but on any real device it stays on
          the slope, so time is bytes over bandwidth, and the predicted speed-up over float32 is the ratio of bytes. Then
          drag the batch.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trọng số, đầu vào và đầu ra mỗi thứ được đếm một lần, ở băng thông đầy đủ. Không có bộ nhớ đệm, không có chi phí
              khởi chạy, và việc giải nén cùng nhân hệ số tỉ lệ là miễn phí. Điều cuối cùng không còn đúng khi gần ngưỡng.
            </p>
          }
        >
          <p>
            Weights, inputs and outputs are each counted once, at full bandwidth. No caches, no launch overhead, and the
            unpacking and scaling are free. That last one stops being true near the threshold.
          </p>
        </Translated>
      </Simplification>
      <DecodeRoofline />
      <Translated
        vi={
          <p>
            Ba điều cần tìm. Thứ nhất, ở batch 1 mức tăng tốc bằng tỉ lệ byte, chỉ sai khác phần nhỏ của activation: 3.76 với
            Q8_0, 7.11 với Q4_0. Thứ hai, hệ số tỉ lệ có ảnh hưởng: 4 bit của MLX với hệ số tỉ lệ float32 theo nhóm 32 kém xa
            Q4_0. Thứ ba, điểm gãy dịch chuyển: trên ví dụ của sách, một lớp 4096 sang 4096 chuyển sang bị giới hạn bởi tính
            toán ở batch 41 với float32, và ở một batch chỉ một chữ số với Q4_0 và activation 16 bit. Qua điểm gãy, mô hình
            không còn nói được gì hữu ích, vì phần việc nó coi là miễn phí khi đó chính là thứ bạn phải chờ.
          </p>
        }
      >
        <p>
          Three things to find. First, at batch 1 the speed-up is the byte ratio to within the activations' sliver: 3.76 for
          Q8_0, 7.11 for Q4_0. Second, the scales show: MLX's 4-bit with float32 scales in groups of 32 is a long way short
          of Q4_0. Third, the knee moves: on the book's example a 4096-to-4096 layer turns compute-bound at batch 41 with
          float32, and at a single-digit batch with Q4_0 and 16-bit activations. Past the knee the model says nothing
          useful, because the work it treats as free is then what you wait for.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Nếu bạn đã chạy lab X2.1 và X2.2, hãy chọn thiết bị của bạn và batch 1, 4 hoặc 8. Mức tăng tốc bạn đo được có thể
            thấp hơn tỉ lệ byte, hoặc cao hơn. Cao hơn nghĩa là kernel float32 của bạn nằm xa dưới đường dốc băng thông hơn
            kernel lượng tử hóa: hãy so hai hình thoi với mái, chứ không so chúng với nhau.
          </p>
        }
      >
        <p>
          If you have run labs X2.1 and X2.2, pick your device and a batch of 1, 4 or 8. Your measured speed-up can land
          below the byte ratio, or above it. Above it means your float32 kernel sat further below the bandwidth slope than
          your quantised one: compare the two diamonds with the roof, not with each other.
        </p>
      </Translated>
      <Ask q={q['x2/X2.1/q-q8-ceiling']} />
      <Ask q={q['x2/X2.1/q-knee-batch']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['x2/X2.1/q-regime-q8']} label={tr('From chapter 5', 'Từ chương 5')} />

      <h2><Translated vi="Gộp (fused) hay không gộp (unfused)">Fused or unfused</Translated></h2>
      <Translated
        vi={
          <p>
            Tỉ lệ byte là một trần, và nó giả định rằng trọng số lượng tử hóa là thứ GEMV đọc. Cách cài đặt hiển nhiên đầu
            tiên lại làm khác: một kernel biến các mã trở lại thành số float và ghi chúng ra, rồi GEMV float bạn đã có đọc
            chúng.
          </p>
        }
      >
        <p>
          The byte ratio is a ceiling, and it assumes the quantised weights are what the GEMV reads. The obvious first
          implementation does something else: one kernel turns the codes back into floats and writes them out, and the
          float GEMV you already have reads them.
        </p>
      </Translated>
      <Predict
        id="x2/X2.1/p-unfused"
        unit={tr('× the float32 time', '× thời gian float32')}
        prompt={
          <Translated
            vi={
              <>
                Giải lượng tử hóa trọng số Q4_0 sang float32 trong một kernel, rồi chạy GEMV float32 của lab 22.1 trên kết quả.
                Việc đó mất bao lâu, tính bằng bội số của GEMV float32 trên trọng số float32? (1 nghĩa là như nhau; 0.14 sẽ là
                toàn bộ tỉ lệ byte.)
              </>
            }
          >
            <>
              Dequantise Q4_0 weights to float32 in one kernel, then run lab 22.1's float32 GEMV on the result. How long does
              that take, as a multiple of the float32 GEMV on float32 weights? (1 means the same; 0.14 would be the full
              byte ratio.)
            </>
          </Translated>
        }
      >
        {(c) => {
          const rel = pipelineRelative('unfused-f32', 'q4_0', layer('4096x4096'), 1, 4);
          return (
            <Translated
              vi={
                <p data-reveal="unfused">
                  <b>{rel.toFixed(2)} lần</b>, còn chậm hơn cả khi không lượng tử hóa
                  {Math.abs(c - rel) < 0.15 ? ', đúng như bạn nói' : c < 1 ? '. Nếu bạn dự đoán là nhanh hơn, hãy lần theo số byte' : ''}. Kernel
                  thứ nhất đọc 0.5625 byte mỗi trọng số và ghi 4; kernel thứ hai đọc lại 4 byte đó. Tức là{' '}
                  {pipelineBytesPerWeight('unfused-f32', 'q4_0')} byte mỗi trọng số so với 4 của float32. Trình khám phá bên
                  dưới có các pipeline khác.
                </p>
              }
            >
              <p data-reveal="unfused">
                <b>{rel.toFixed(2)} times</b>, slower than not quantising at all
                {Math.abs(c - rel) < 0.15 ? ', as you said' : c < 1 ? '. If you predicted a speed-up, follow the bytes' : ''}. The
                first kernel reads 0.5625 bytes per weight and writes 4; the second reads those 4 back. That is{' '}
                {pipelineBytesPerWeight('unfused-f32', 'q4_0')} bytes per weight against float32's 4. The explorer below
                has the other pipelines.
              </p>
            </Translated>
          );
        }}
      </Predict>
      <FusedOrUnfused />
      <Translated
        vi={
          <p>
            Giải lượng tử hóa sang FP16 thay vào đó giảm một nửa bản sao mà vẫn thua float32. Chỉ kernel gộp, thứ đọc mã
            và hệ số tỉ lệ rồi giải nén chúng trong thanh ghi trên đường tới phép nhân, mới di chuyển ít byte hơn so với ban đầu.
          </p>
        }
      >
        <p>
          Dequantising to FP16 instead halves the copy and still loses to float32. Only the fused kernel, which reads codes
          and scales and unpacks them in registers on the way to the multiply, moves fewer bytes than it started with.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${FACTS}, A1)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, A1)`)}>
        <Translated
          vi={
            <p>
              Blog của PyTorch thấy một đường int8 không gộp "much worse than even the fp16 performance", trên thiết bị của họ.
            </p>
          }
        >
          <p>
            PyTorch's blog found an unfused int8 path "much worse than even the fp16 performance", on their device.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Hãy xem ba dòng dữ liệu chảy qua cùng một đường ống. Tỉ lệ byte là một trần, không phải một lời hứa: các kernel hợp
            nhất của chính blog đó vẫn còn cách xa nó.
          </p>
        }
      >
        <p>
          Watch the three streams drain through the same pipe. The byte ratio is a ceiling, not a promise: the same blog's
          fused kernels stayed well short of it.
        </p>
      </Translated>
      <Simplification>
        <Translated vi={<p>Chỉ có byte, ở một băng thông cố định: không có chi phí giải nén, không có bộ nhớ đệm, không có chi phí khởi chạy.</p>}>
          <p>Bytes only, at one fixed bandwidth: no unpacking cost, no caches, no launch overhead.</p>
        </Translated>
      </Simplification>
      <ThreeTrafficsScene />
      <Ask q={q['x2/X2.1/q-unfused']} />

      <h2><Translated vi="Báo cáo một phép đo">Reporting a measurement</Translated></h2>
      <Translated
        vi={
          <p>
            Một con số thời gian riêng lẻ nói được rất ít. Hãy báo cáo trọng số đã dùng bao nhiêu phần băng thông của bạn: số
            byte trọng số mỗi lượt, nhân số lượt mỗi giây, chia cho băng thông đỉnh của bạn. PyTorch gọi nó là mức sử dụng băng
            thông của mô hình (model bandwidth utilisation). Hãy dùng đỉnh mà lab 5.1 đo trên cùng thiết bị, không bao giờ dùng
            số của datasheet. Một GEMV gần 100 % không còn gì để cải thiện bằng mã; một GEMV ở 30 % thì còn.
          </p>
        }
      >
        <p>
          A time on its own says little. Report what fraction of your bandwidth the weights used: weight bytes per pass,
          times passes per second, over your peak bandwidth. PyTorch calls it model bandwidth utilisation. Use the peak lab
          5.1 measured on the same device, never a datasheet's. A GEMV near 100 % has nothing left to gain from code; one at
          30 % does.
        </p>
      </Translated>
      <Ask q={q['x2/X2.1/q-mbu']} label={tr('Work it out', 'Tự tính')} />
      <UtilisationCalc />
      <Translated
        vi={
          <p>
            <a href={href('x2', 'labs')}>Trang lab</a> tính cùng con số này cho lab X2.1, X2.2 và X2.4 từ các dòng benchmark
            của chúng.
          </p>
        }
      >
        <p>
          The <a href={href('x2', 'labs')}>labs page</a> computes the same number for labs X2.1, X2.2 and X2.4 from their
          benchmark rows.
        </p>
      </Translated>

      <h2><Translated vi="GEMM của thư viện không phục vụ decode">The library GEMM does not serve decode</Translated></h2>
      <Translated
        vi={
          <p>
            Thêm một cái bẫy trước khi viết kernel. Phép nhân ma trận int8 có trong mọi stack, và nó không phải kernel mà decode
            cần: với một token chỉ có một hàng.
          </p>
        }
      >
        <p>
          One more trap before writing kernels. An int8 matrix multiply exists in every stack, and it is not the kernel decode
          needs: at one token there is one row.
        </p>
      </Translated>
      <Today checked={tr(`2026-11-20 in the facts pass (${FACTS}, A6, D6)`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, A6, D6)`)}>
        <Translated
          vi={
            <p>
              <code>torch._int_mm</code> trên CUDA: <code>TORCH_CHECK(self.size(0) &gt; 16, ...)</code>. cuBLAS int8 cần căn
              chỉnh 4 byte và leading dimension là bội số của 4. Các op chỉ lượng tử hóa trọng số phục vụ batch 1.
            </p>
          }
        >
          <p>
            <code>torch._int_mm</code> on CUDA: <code>TORCH_CHECK(self.size(0) &gt; 16, ...)</code>. cuBLAS int8 needs
            4-byte alignment and leading dimensions that are multiples of 4. Weight-only ops serve batch 1.
          </p>
        </Translated>
      </Today>
      <Ask q={q['x2/X2.1/q-int-mm']} />
    </>
  );
}
