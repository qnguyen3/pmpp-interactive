import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import { Q4_0, Q4_1, Q4_K, bitsPerWeight, mlxAffine } from '../../sim/qformats';
import {
  DIST_LABEL,
  GRANS,
  MIRROR_AT,
  OUTLIER_AT,
  RULES,
  RULE_FACT,
  RULE_LABEL,
  SCHEME_LABEL,
  SWEEP_BITS,
  bestUnder,
  demoBlock,
  errorSweep,
  fp16,
  granLabel,
  quantTrace,
  scaleMinSources,
  type Dist,
  type Gran,
  type OutlierSign,
  type Rule,
  type Scheme,
  type SweepPoint,
  type Trace,
} from '../../sim/quant';
import { Translated, tr } from '../../lib/i18n';

const FACTS = 'notes/today/x2-facts.md';
const CHECKED = (items: string) => tr(`2026-11-20 in the facts pass (${FACTS}, ${items})`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, ${items})`);
const SNIPPET = 'x2/snippets/q4_0_ref.h';
const SNIPPET_STATUS = () => tr('host C++; compiled and compared bit for bit with the stepper’s model by a unit test; not a GPU kernel', 'C++ phía host; được biên dịch và so sánh từng bit với mô hình của trình chạy từng bước bằng một unit test; không phải kernel GPU');
const n3 = (v: number) => (Object.is(v, -0) ? '-0' : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));
const n4 = (v: number) => (Object.is(v, -0) ? '-0' : v.toPrecision(4));

// ---------------------------------------------------------------- questions
export const questionsSX22: Record<string, Question> = {
  'x2/X2.2/q-q40-clip': {
    kind: 'numeric',
    id: 'x2/X2.2/q-q40-clip',
    prompt: (
      <Translated
        vi={
          <>
            Giá trị có độ lớn lớn nhất của một khối Q4_0 là <code>-0.8</code>. Một trọng số khác trong cùng khối là{' '}
            <code>+0.8</code>. Nó được giải lượng tử hóa thành bao nhiêu? (Hai chữ số thập phân là đủ.)
          </>
        }
      >
        <>
          A Q4_0 block's value of largest magnitude is <code>-0.8</code>. Another weight in the same block is{' '}
          <code>+0.8</code>. What does it dequantise to? (Two decimals are enough.)
        </>
      </Translated>
    ),
    answer: 0.7,
    tol: 0.01,
    explain: (
      <Translated
        vi={
          <p>
            d = -0.8 / -8 = 0.1, nên cực trị thành mã 0 (giá trị -8 × 0.1). Với +0.8: 0.8 × 10 + 8.5 = 16.5, cắt phần lẻ
            còn 16, chặn về 15. (15 - 8) × 0.1 = 0.7, hoặc 0.6998 khi d đã được lưu dưới dạng FP16 (0.0999756). 16 mức của
            khối chạy từ -8d đến +7d: phía đối diện với cực trị mất một bước.
          </p>
        }
      >
        <p>
          d = -0.8 / -8 = 0.1, so the extreme becomes code 0 (value -8 × 0.1). For +0.8: 0.8 × 10 + 8.5 = 16.5, truncated
          to 16, clamped to 15. (15 - 8) × 0.1 = 0.7, or 0.6998 once d has been stored as FP16 (0.0999756). The block's
          16 levels run from -8d to +7d: the side opposite the extreme loses one step.
        </p>
      </Translated>
    ),
    source: `${FACTS} D1 (quantize_row_q4_0_ref)`, // i18n-ignore: a citation of a function name
  },
  'x2/X2.2/q-q40-zero': {
    kind: 'choice',
    id: 'x2/X2.2/q-q40-zero',
    prompt: (
      <Translated vi={<>Một khối gồm 32 số 0 đi qua quy tắc tham chiếu Q4_0 của llama.cpp. Điều gì xảy ra?</>}>
        <>A block of 32 zeros goes through llama.cpp's Q4_0 reference rule. What happens?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Mọi mã đều là 8 và mọi trọng số giải mã thành 0: nghịch đảo của d có được bảo vệ.">{'Every code is 8 and every weight decodes to 0: the inverse of d is guarded.'}</Translated>, correct: true },
      { label: <Translated vi="Chia cho không: 1 / d là vô cùng, 0 nhân với nó là NaN, và mọi trọng số giải mã thành NaN.">{'Division by zero: 1 / d is infinite, 0 times that is NaN, and every weight decodes to NaN.'}</Translated>, why: <Translated vi="Mã viết id = d ? 1.0f/d : 0.0f, nên nó không bao giờ chia cho một d bằng không.">{'The code writes id = d ? 1.0f/d : 0.0f, so it never divides by a zero d.'}</Translated> },
      { label: <Translated vi="Mọi mã đều là 0, mức thấp nhất, vì mỗi trọng số đều bằng với cực trị của khối.">{'Every code is 0, the lowest level, because each weight ties with the block\'s extreme.'}</Translated>, why: <Translated vi="x × 0 + 8.5 cắt phần lẻ còn 8, mã ở giữa, tức (8 - 8) × d = 0.">{'x × 0 + 8.5 truncates to 8, the middle code, which is (8 - 8) × d = 0.'}</Translated> },
      { label: <Translated vi="Quy tắc bỏ qua các khối rỗng và không ghi byte nào.">{'The rule skips empty blocks and writes no bytes.'}</Translated>, why: <Translated vi="Mọi khối đều là 18 byte, bất kể nội dung.">{'Every block is 18 bytes, whatever its contents.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            amax và max bắt đầu từ 0 và giữ nguyên ở đó, nên d = 0 / -8, tức là âm không trong số học IEEE (lưu dưới dạng mẫu
            bit FP16 0x8000). Phép bảo vệ làm nghịch đảo bằng 0, mỗi trọng số thành 0 + 8.5, cắt phần lẻ còn 8, và (8 - 8)
            × d bằng không. Không có gì hỏng, đó là lý do llama.cpp có thể viết phép bảo vệ thay vì một trường hợp đặc biệt.
          </p>
        }
      >
        <p>
          amax and max start at 0 and stay there, so d = 0 / -8, which is minus zero in IEEE arithmetic (stored as the
          FP16 pattern 0x8000). The guard makes the inverse 0, each weight becomes 0 + 8.5, truncated to 8, and (8 - 8)
          × d is zero. Nothing breaks, which is why llama.cpp can write the guard instead of a special case.
        </p>
      </Translated>
    ),
    source: `${FACTS} D1; the sign of the zero is our reading of IEEE division`,
    vi: { source: `${FACTS} D1; dấu của số không là cách chúng tôi đọc phép chia IEEE` },
  },
  'x2/X2.2/q-bits': {
    kind: 'numeric',
    id: 'x2/X2.2/q-bits',
    prompt: (
      <Translated
        vi={
          <>
            Lượng tử hóa affine mặc định của MLX: 4 bit, nhóm 64, với một hệ số tỉ lệ và một độ lệch cho mỗi nhóm lưu ở FP16.
            Đó là bao nhiêu bit mỗi trọng số?
          </>
        }
      >
        <>
          MLX's default affine quantisation: 4 bits, groups of 64, with a scale and a bias per group stored in FP16. How
          many bits per weight is that?
        </>
      </Translated>
    ),
    answer: bitsPerWeight(mlxAffine(4, 64, 'f16')),
    tol: 0.001,
    unit: 'bits',
    explain: (
      <Translated
        vi={
          <p>
            4 + (16 + 16) / 64 = 4.5. Hệ số tỉ lệ và độ lệch có cùng kiểu với đầu vào, nên trọng số float32 cho hệ số tỉ lệ
            và độ lệch float32: 4 + 64 / 64 = 5.0, nhiều bit bằng nhóm 32 ở FP16.
          </p>
        }
      >
        <p>
          4 + (16 + 16) / 64 = 4.5. The scale and bias have the input's type, so float32 weights give float32 scales and
          biases: 4 + 64 / 64 = 5.0, as many bits as group 32 in FP16.
        </p>
      </Translated>
    ),
    source: `${FACTS} D5, E1`,
    vi: { unit: 'bit' },
  },
  'x2/X2.2/q-convert': {
    kind: 'choice',
    id: 'x2/X2.2/q-convert',
    prompt: (
      <Translated
        vi={
          <>
            Bạn có một nhóm MLX, value = s·q + bias, và muốn dạng của MPSGraph, value = scale·(q - zeroPoint), trong đó zero
            point phải có kiểu số nguyên của đầu vào. scale và zeroPoint là gì, và khi nào cách này dùng được?
          </>
        }
      >
        <>
          You have one MLX group, value = s·q + bias, and want MPSGraph's form, value = scale·(q - zeroPoint), where the
          zero point must have the input's integer type. What are scale and zeroPoint, and when does it work?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="scale = s, zeroPoint = -bias / s; chỉ khi đó là một số nguyên, như mã của MLX sắp xếp.">
            {'scale = s, zeroPoint = -bias / s; only if that is an integer, as MLX’s code arranges.'}
          </Translated>
        ),
        correct: true,
      },
      { label: <Translated vi="scale = s, zeroPoint = bias; dùng được cho mọi nhóm, vì cả hai đều là độ dời.">{'scale = s, zeroPoint = bias; it works for every group, as both are offsets.'}</Translated>, why: <Translated vi="Khai triển scale·(q - zp) = s·q - s·zp: bạn cần -s·zp = bias, nên zp = -bias / s, không phải bias.">{'Expand scale·(q - zp) = s·q - s·zp: you need -s·zp = bias, so zp = -bias / s, not bias.'}</Translated> },
      { label: <Translated vi="scale = 1 / s, zeroPoint = bias·s; dùng được bất cứ khi nào s là lũy thừa của hai.">{'scale = 1 / s, zeroPoint = bias·s; it works whenever s is a power of two.'}</Translated>, why: <Translated vi="Hệ số tỉ lệ nhân với các mã ở cả hai dạng; nó là cùng một s.">{'The scale multiplies the codes in both forms; it is the same s.'}</Translated> },
      { label: <Translated vi="scale = s, zeroPoint = -bias / s; dùng được với mọi s khác không, vì zeroPoint có thể là số float.">{'scale = s, zeroPoint = -bias / s; it works for any nonzero s, as zeroPoint may be a float.'}</Translated>, why: <Translated vi="Header nói tensor zero-point phải khớp kiểu của tensor đầu vào: Int4, UInt4, Int8 hoặc UInt8. Cần s khác không, nhưng -bias / s cũng phải là số nguyên.">{'The header says the zero-point tensor must match the input tensor’s type: Int4, UInt4, Int8 or UInt8. A nonzero s is needed, but -bias / s must also be an integer.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            s·q + bias = s·(q - (-bias / s)). Với quy tắc được ghi trong tài liệu bias = min và s = (max - min) / 15, thì -min / s
            là bất cứ giá trị nào nó tình cờ nhận. Mã của MLX thay vào đó chọn cạnh có độ lớn lớn hơn, làm tròn edge / s thành
            một số nguyên q0, rồi đặt s = edge / q0, bias = edge; khi đó -bias / s = -q0, một số nguyên. Quy tắc min/max trong
            tài liệu nói chung không cho ra số nguyên. Trình chạy từng bước cho thấy mã đó (nó
            ghi "mã … nằm trên không").
          </p>
        }
      >
        <p>
          s·q + bias = s·(q - (-bias / s)). With the documented rule bias = min and s = (max - min) / 15, so -min / s is
          whatever it happens to be. MLX's code instead picks the edge of larger magnitude, rounds edge / s to an integer
          q0, and sets s = edge / q0, bias = edge; then -bias / s = -q0, an integer. The documented min/max rule
          generally does not give one. The stepper shows that code (it
          marks "code on zero").
        </p>
      </Translated>
    ),
    source: `${FACTS} C4, D5 (our algebra)`,
    vi: { source: `${FACTS} C4, D5 (phép biến đổi đại số của chúng tôi)` },
  },
  'x2/X2.2/q-outlier': {
    kind: 'choice',
    id: 'x2/X2.2/q-outlier',
    prompt: (
      <Translated
        vi={
          <>
            Một hàng trọng số 4096 phần tử có một giá trị lệch 12 độ lệch chuẩn. Ở 4 bit, sai số của 4095 trọng số còn lại so
            sánh thế nào giữa một hệ số tỉ lệ cho mỗi hàng và một hệ số tỉ lệ cho mỗi 32?
          </>
        }
      >
        <>
          A weight row of 4096 has one value 12 standard deviations out. At 4 bits, how does the error of the other 4095
          weights compare between one scale per row and one scale per 32?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Theo hàng, bước của mọi trọng số tăng theo giá trị ngoại lai; theo 32, chỉ 31 trọng số láng giềng của nó chịu thiệt.">{'Per row, every weight’s step grows with the outlier; per 32, only its own 31 neighbours pay.'}</Translated>, correct: true },
      { label: <Translated vi="Như nhau ở cả hai cách: chỉ riêng giá trị ngoại lai bị làm tròn tệ.">{'The same either way: only the outlier itself is badly rounded.'}</Translated>, why: <Translated vi="Giá trị ngoại lai quyết định hệ số tỉ lệ, và hệ số tỉ lệ là bước của mọi trọng số.">{'The outlier sets the scale, and the scale is every weight’s step.'}</Translated> },
      { label: <Translated vi="Theo hàng tốt hơn: một hệ số tỉ lệ của nó nắm đúng khoảng giá trị của cả hàng.">{'Per row is better: its one scale gets the whole row\'s range right.'}</Translated>, why: <Translated vi="Một hệ số tỉ lệ khớp khoảng giá trị bằng cách kéo giãn bước cho cả 4096 trọng số. Khoảng giá trị thì đúng, còn bước thì quá thô cho mọi thứ trừ giá trị ngoại lai.">{'One scale fits the range by stretching the step for all 4096 weights. The range is right and the step is too coarse for everything but the outlier.'}</Translated> },
      { label: <Translated vi="Theo 32 tệ hơn, vì 128 hệ số tỉ lệ thêm 128 lần làm tròn nữa.">{'Per 32 is worse, because 128 scales add 128 more roundings.'}</Translated>, why: <Translated vi="Mỗi hệ số tỉ lệ FP16 thêm một sai số tương đối khoảng 2^-11, nhỏ so với một bước 4 bit.">{'Each FP16 scale adds a relative error near 2^-11, small next to a 4-bit step.'}</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với cách chia tỉ lệ kiểu absmax, bước là cực trị chia cho số mức ở phía của nó. Giá trị lớn nhất trong 32 trọng số
            điển hình thường là 2 đến 3σ, nên một trọng số ở 12σ làm bước lớn gấp bốn đến sáu lần, và sai số làm tròn của mọi
            trọng số khác tăng theo. Giới hạn hệ số tỉ lệ trong 32 trọng số thì 31 trọng số chịu thiệt thay vì 4095. Hãy kéo
            thanh trượt giá trị ngoại lai trong trình chạy từng bước, rồi so sánh các đường "block 32" và "one per row" trong
            trình khám phá với phân phối đuôi nặng.
          </p>
        }
      >
        <p>
          With absmax-style scaling the step is the extreme divided by the number of levels on its side. The largest of
          32 typical weights is usually 2 to 3σ, so one weight at 12σ makes the step four to six times larger, and every
          other weight's rounding error grows with it. Confine the scale to 32 weights and 31 weights pay instead of 4095. Move
          the outlier slider in the stepper, then compare the curves for "block 32" and "one per row" in the explorer
          with heavy tails.
        </p>
      </Translated>
    ),
    source: `ours; ${FACTS} E2 for the measured trend`,
    vi: { source: `của khóa học này; ${FACTS} E2 cho xu hướng đo được` },
  },
  'x2/X2.2/q-q4k': {
    kind: 'numeric',
    id: 'x2/X2.2/q-q4k',
    prompt: (
      <Translated
        vi={
          <>
            Q4_K của llama.cpp lưu 256 trọng số trong 144 byte, với một hệ số tỉ lệ và một giá trị min cho mỗi 32 trọng số.
            Bao nhiêu bit mỗi trọng số?
          </>
        }
      >
        <>
          llama.cpp's Q4_K stores 256 weights in 144 bytes, with a scale and a min for every 32 weights. How many bits per
          weight?
        </>
      </Translated>
    ),
    answer: bitsPerWeight(Q4_K),
    tol: 0.001,
    unit: 'bits',
    explain: (
      <Translated
        vi={
          <p>
            144 × 8 / 256 = 4.5. Q4_1 có cùng hệ số tỉ lệ và min cho mỗi 32 và tốn 5.0, vì mỗi giá trị là một số FP16. Q4_K
            lưu tám hệ số tỉ lệ và tám min dưới dạng mã 6 bit (tổng cộng 12 byte) dưới hai số FP16 của siêu khối, d và dmin:
            2 + 2 + 12 + 128 byte.
          </p>
        }
      >
        <p>
          144 × 8 / 256 = 4.5. Q4_1 has the same per-32 scale and min and costs 5.0, because each is an FP16 number. Q4_K
          stores the eight scales and eight mins as 6-bit codes (12 bytes in all) under two FP16 super-block numbers, d
          and dmin: 2 + 2 + 12 + 128 bytes.
        </p>
      </Translated>
    ),
    source: `${FACTS} D2, E1`,
    vi: { unit: 'bit' },
  },
};

// ---------------------------------------------------------------- stepper: one block, six rules
const STEPS = () => [
  tr('the weights', 'các trọng số'),
  tr('find the extreme', 'tìm cực trị'),
  tr('derive the scale', 'suy ra hệ số tỉ lệ'),
  tr('divide', 'chia'),
  tr('round', 'làm tròn'),
  tr('clamp', 'chặn'),
  tr('store', 'lưu'),
] as const;

function stepText(t: Trace, k: number, fp16Scale: boolean): string {
  const e = t.extreme;
  const sym = t.rule === 'q8_0' || t.rule === 'sym4';
  const qmax = t.rule === 'q8_0' ? 127 : 7;
  const clips = t.clipped.flatMap((c, i) => (c ? [i] : []));
  switch (k) {
    case 0:
      return tr(`32 float32 weights of one block. Weight ${OUTLIER_AT} is the outlier you control.`, `32 trọng số float32 của một khối. Trọng số ${OUTLIER_AT} là giá trị ngoại lai bạn điều khiển.`);
    case 1:
      if (t.rule === 'q4_0') return tr(`Q4_0 looks for the largest magnitude and keeps its sign: weight ${e.index}, ${n3(e.value)}.`, `Q4_0 tìm độ lớn lớn nhất và giữ dấu của nó: trọng số ${e.index}, ${n3(e.value)}.`);
      if (sym) return tr(`The largest magnitude, sign dropped: |${n3(e.value)}| at weight ${e.index}.`, `Độ lớn lớn nhất, bỏ dấu: |${n3(e.value)}| ở trọng số ${e.index}.`);
      return tr(`Both ends: min ${n3(e.min)} at weight ${e.minIndex}, max ${n3(e.max)} at weight ${e.maxIndex}.`, `Cả hai đầu: min ${n3(e.min)} ở trọng số ${e.minIndex}, max ${n3(e.max)} ở trọng số ${e.maxIndex}.`);
    case 2:
      if (t.rule === 'q4_0') return tr(`d = ${n3(e.value)} / -8 = ${n4(t.scale)}. The extreme will sit exactly on code 0 (value -8d); the codes run 0 to 15 for -8d to +7d.`, `d = ${n3(e.value)} / -8 = ${n4(t.scale)}. Cực trị sẽ nằm đúng trên mã 0 (giá trị -8d); các mã chạy từ 0 đến 15 cho -8d đến +7d.`);
      if (sym) return tr(`d = ${n3(Math.abs(e.value))} / ${qmax} = ${n4(t.scale)}. Codes run -${qmax} to +${qmax}: ${2 * qmax + 1} levels, symmetric around zero.`, `d = ${n3(Math.abs(e.value))} / ${qmax} = ${n4(t.scale)}. Các mã chạy từ -${qmax} đến +${qmax}: ${2 * qmax + 1} mức, đối xứng quanh không.`);
      if (t.rule === 'mlx-code')
        return tr(
          `s from (max - min) / 15, signed toward the edge of larger magnitude (${n3(t.offset)}), then nudged to edge / q0 = ${n4(t.scale)}; bias = ${n3(t.offset)}.${t.zeroCode != null ? ` Code ${t.zeroCode} now lands on zero.` : ''}`,
          `s lấy từ (max - min) / 15, mang dấu theo cạnh có độ lớn lớn hơn (${n3(t.offset)}), rồi chỉnh nhẹ thành edge / q0 = ${n4(t.scale)}; bias = ${n3(t.offset)}.${t.zeroCode != null ? ` Giờ mã ${t.zeroCode} rơi đúng vào không.` : ''}`,
        );
      return tr(
        `${t.rule === 'q4_1' ? 'd' : 's'} = (max - min) / 15 = ${n4(t.scale)}, offset = min = ${n3(t.offset)}. Sixteen levels from min to max, none wasted outside them.`,
        `${t.rule === 'q4_1' ? 'd' : 's'} = (max - min) / 15 = ${n4(t.scale)}, offset = min = ${n3(t.offset)}. Mười sáu mức từ min đến max, không mức nào lãng phí bên ngoài.`,
      );
    case 3:
      return t.rule === 'q4_0' || sym
        ? tr('Each weight times 1 / d.', 'Mỗi trọng số nhân với 1 / d.')
        : t.rule === 'q4_1'
          ? tr('Each weight minus m, times 1 / d.', 'Mỗi trọng số trừ m, nhân với 1 / d.')
          : tr('Each weight minus the bias, divided by s.', 'Mỗi trọng số trừ bias, chia cho s.');
    case 4:
      return t.rule === 'q4_0'
        ? tr('Add 8.5 and truncate (the (int8_t) cast): round half up, and shifted so the codes start at 0.', 'Cộng 8.5 và cắt phần lẻ (phép ép kiểu (int8_t)): làm tròn nửa lên, và dời đi để các mã bắt đầu từ 0.')
        : t.rule === 'q4_1'
          ? tr('Add 0.5 and truncate.', 'Cộng 0.5 và cắt phần lẻ.')
          : tr('Round to the nearest integer, halves away from zero.', 'Làm tròn về số nguyên gần nhất, giá trị nửa làm tròn ra xa không.');
    case 5:
      return clips.length
        ? tr(`Clamp to ${t.codeMin}..${t.codeMax}: weight${clips.length > 1 ? 's' : ''} ${clips.join(', ')} clipped (shown in red).`, `Chặn về ${t.codeMin}..${t.codeMax}: trọng số ${clips.join(', ')} bị chặn (tô đỏ).`)
        : tr(`Clamp to ${t.codeMin}..${t.codeMax}: nothing needed clamping.`, `Chặn về ${t.codeMin}..${t.codeMax}: không có gì cần chặn.`);
    default:
      return tr(
        `Store the codes and ${fp16Scale ? 'the scale rounded to FP16' : 'the scale as a float (FP16 rounding off)'}${t.offset !== 0 || t.rule === 'q4_1' ? ' and offset' : ''}: ${n4(t.scaleStored)}${t.rule === 'q4_0' || sym ? '' : `, ${n4(t.offsetStored)}`}. Relative RMS error of this block: ${(100 * t.rel).toFixed(2)}%.`,
        `Lưu các mã và ${fp16Scale ? 'hệ số tỉ lệ làm tròn về FP16' : 'hệ số tỉ lệ dưới dạng float (tắt làm tròn FP16)'}${t.offset !== 0 || t.rule === 'q4_1' ? ' cùng độ dời' : ''}: ${n4(t.scaleStored)}${t.rule === 'q4_0' || sym ? '' : `, ${n4(t.offsetStored)}`}. Sai số RMS tương đối của khối này: ${(100 * t.rel).toFixed(2)}%.`,
      );
  }
}

function NumberLine({ t, k }: { t: Trace; k: number }) {
  const W = 720;
  const H = 150;
  const R = Math.max(...t.values.map(Math.abs), ...(k >= 2 ? t.levels.map(Math.abs) : [0])) * 1.08 || 1;
  const X = (v: number) => 20 + ((v + R) / (2 * R)) * (W - 40);
  const showLevels = k >= 2 && t.levels.length <= 32;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Weights, the grid and the dequantised values on a number line', 'Các trọng số, lưới giá trị và các giá trị đã giải lượng tử hóa trên một trục số')} style={{ width: '100%', display: 'block' }}>
      <line x1={20} x2={W - 20} y1={70} y2={70} stroke="var(--rule)" />
      <line x1={X(0)} x2={X(0)} y1={60} y2={80} stroke="var(--ink-3)" />
      <text x={X(0)} y={95} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>0</text>
      {showLevels
        ? t.levels.map((v, i) => (
            <line key={i} data-level={v} x1={X(v)} x2={X(v)} y1={62} y2={78} stroke="var(--global)" strokeWidth={1.4} />
          ))
        : k >= 2 ? <text x={20} y={56} fontSize="12" style={{ fill: 'var(--global)' }}><Translated vi={<>{t.levels.length} mức, quá dày để vẽ</>}>{t.levels.length} levels, too dense to draw</Translated></text> : null}
      {t.values.map((v, i) => {
        const hot = k >= 1 && (i === t.extreme.index || ((t.rule === 'q4_1' || t.rule.startsWith('mlx')) && (i === t.extreme.minIndex || i === t.extreme.maxIndex)));
        const bad = k >= 5 && t.clipped[i];
        return (
          <g key={i}>
            <circle cx={X(v)} cy={40} r={hot ? 5 : 3.4} fill={bad ? 'var(--fail)' : hot ? 'var(--ink)' : 'var(--page)'} stroke={bad ? 'var(--fail)' : 'var(--ink)'} />
            {k >= 6 ? (
              <>
                <line x1={X(v)} y1={44} x2={X(t.dequant[i])} y2={100} stroke={bad ? 'var(--fail)' : 'var(--ink-3)'} strokeWidth={bad ? 1.6 : 0.8} />
                <circle cx={X(t.dequant[i])} cy={104} r={3.4} fill={bad ? 'var(--fail)' : 'var(--global)'} />
              </>
            ) : null}
          </g>
        );
      })}
      <text x={W - 20} y={14} fontSize="12.5" textAnchor="end"><Translated vi="trọng số (float32)">weights (float32)</Translated></text>
      {k >= 2 ? <text x={20} y={132} fontSize="12.5" style={{ fill: 'var(--global)' }}>{k >= 6 ? tr('dequantised values, on the grid', 'các giá trị đã giải lượng tử hóa, trên lưới') : tr('the grid: every value the block can hold', 'lưới: mọi giá trị mà khối có thể chứa')}</text> : null}
      {k >= 1 ? <text x={X(t.extreme.value)} y={27} fontSize="12" textAnchor="middle">{t.rule === 'q4_0' ? tr('signed extreme', 'cực trị có dấu') : tr('extreme', 'cực trị')}</text> : null}
    </svg>
  );
}

function OneBlock() {
  const [dist, setDist] = useState<Dist>('normal');
  const [outlier, setOutlier] = useState(0.8);
  const [sign, setSign] = useState<OutlierSign>('±');
  const [fp16Scale, setFp16] = useState(true);
  const [rule, setRule] = useState<Rule>('q4_0');
  const [k, setK] = useState(0);
  const values = useMemo(() => demoBlock(dist, outlier, sign), [dist, outlier, sign]);
  const t = useMemo(() => quantTrace(rule, values, fp16Scale), [rule, values, fp16Scale]);
  const last = STEPS().length - 1;
  const codeShown = (i: number) => (rule === 'q4_0' ? `${t.codes[i]} (${t.codes[i] - 8})` : String(t.codes[i]));
  return (
    <Bench
      title={tr('One block, six rules', 'Một khối, sáu quy tắc')}
      task={tr('Step Q4_0 to the end with the outlier mirrored, then try the other rules on the same block.', 'Chạy từng bước Q4_0 đến cuối với giá trị ngoại lai được lấy đối xứng, rồi thử các quy tắc khác trên cùng khối.')}
      foot={
        <Translated
          vi={
            <>
              32 trọng số sinh từ seed cố định, độ lệch chuẩn 0.25. Trọng số {OUTLIER_AT} là giá trị ngoại lai; với ± ảnh
              đối xứng của nó nằm ở trọng số {MIRROR_AT}, nửa còn lại của cùng byte Q4_0. Màu đỏ: trọng số bị phép chặn thay
              đổi. Mã của Q4_0 được hiển thị như khi lưu, với mã - 8 trong ngoặc. Các quy tắc: {RULES.map((r) => `${RULE_LABEL[r]} (${RULE_FACT[r]})`).join('; ')}.
            </>
          }
        >
          <>
            32 seeded weights, standard deviation 0.25. Weight {OUTLIER_AT} is the outlier; with ± its mirror image is at
            weight {MIRROR_AT}, the other half of the same Q4_0 byte. Red: a weight the clamp changed. Codes for Q4_0 are
            shown as stored, with code - 8 in brackets. Rules: {RULES.map((r) => `${RULE_LABEL[r]} (${RULE_FACT[r]})`).join('; ')}.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Quy tắc">Rule</Translated>
          <select aria-label={tr('rule', 'quy tắc')} value={rule} onChange={(e) => setRule(e.target.value as Rule)}>
            {RULES.map((r) => <option key={r} value={r}>{RULE_LABEL[r]}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Trọng số">Weights</Translated>
          <span className="seg" role="group" aria-label={tr('distribution', 'phân phối')}>
            {(['normal', 't3'] as Dist[]).map((d) => (
              <button key={d} aria-pressed={dist === d} onClick={() => setDist(d)}>{d === 'normal' ? tr('normal', 'chuẩn') : tr('Student t (3)', 'Student t (3)')}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi={<>Giá trị ngoại lai ở trọng số {OUTLIER_AT}: {outlier.toFixed(2)}</>}>
            Outlier at weight {OUTLIER_AT}: {outlier.toFixed(2)}
          </Translated>
          <input type="range" aria-label={tr('outlier magnitude', 'độ lớn giá trị ngoại lai')} min="0" max="3" step="0.05" value={outlier} onChange={(e) => setOutlier(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Dấu">Sign</Translated>
          <span className="seg" role="group" aria-label={tr('outlier sign', 'dấu của giá trị ngoại lai')}>
            {(['-', '+', '±'] as OutlierSign[]).map((s) => (
              <button key={s} aria-pressed={sign === s} onClick={() => setSign(s)}>{s === '±' ? tr('− and its mirror', '− và ảnh đối xứng') : s === '-' ? tr('negative', 'âm') : tr('positive', 'dương')}</button>
            ))}
          </span>
        </label>
        <label className="check"><input type="checkbox" checked={fp16Scale} onChange={(e) => setFp16(e.target.checked)} /> <Translated vi="lưu hệ số tỉ lệ dưới dạng FP16">store the scale as FP16</Translated></label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setK(0)} disabled={k === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setK(k - 1)} disabled={k === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setK(k + 1)} disabled={k >= last}><Translated vi="Bước tiếp">Step</Translated></button>
        <button onClick={() => setK(last)} disabled={k >= last}><Translated vi="Đến cuối">To the end</Translated></button>
        <span className="muted" aria-live="polite"><Translated vi={<>bước {k} trên {last}: {STEPS()[k]}</>}>step {k} of {last}: {STEPS()[k]}</Translated></span>
      </div>
      <div className={'narrate' + (k >= 5 && t.clipped.some(Boolean) ? ' bad' : '')} role="status" data-step={k} style={{ marginBottom: '0.8rem' }}>
        {stepText(t, k, fp16Scale)}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <div style={{ minWidth: 640 }}><NumberLine t={t} k={k} /></div>
      </div>
      <div data-part="cells" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(5.4rem, 1fr))', gap: '4px', marginTop: '0.6rem' }}>
        {t.values.map((v, i) => (
          <div key={i} data-w={i} data-clipped={k >= 5 && t.clipped[i] ? 'yes' : 'no'} className="panel"
            style={{ padding: '0.25rem 0.4rem', fontVariantNumeric: 'tabular-nums', lineHeight: 1.3, borderColor: k >= 5 && t.clipped[i] ? 'var(--fail)' : undefined }}>
            <div className="muted">w{i}{k >= 1 && i === t.extreme.index ? ' ★' : ''}</div>
            <div>{n3(v)}</div>
            {k >= 3 ? <div className="muted">÷ {n3(t.divided[i])}</div> : null}
            {k >= 5 ? <div className={t.clipped[i] ? 't-fail' : 't-global'}>q {codeShown(i)}</div> : k >= 4 ? <div className="t-global">→ {t.rounded[i]}</div> : null}
            {k >= 6 ? <div data-deq={t.dequant[i]}>{n3(t.dequant[i])}</div> : null}
            {k >= 6 ? <div className={t.clipped[i] ? 't-fail' : 'muted'}><Translated vi={<>sai số {n3(t.err[i])}</>}>err {n3(t.err[i])}</Translated></div> : null}
          </div>
        ))}
      </div>
      {k >= last ? (
        <p style={{ margin: '0.8rem 0 0' }} data-rel={t.rel}>
          <span className={'chip ' + (t.clipped.some(Boolean) ? 'fail' : 'ok')}>
            <Translated vi={<>{t.clipped.filter(Boolean).length} bị chặn</>}>{t.clipped.filter(Boolean).length} clipped</Translated>
          </span>{' '}
          <Translated
            vi={
              <>
                sai số RMS tương đối {(100 * t.rel).toFixed(2)}%; {t.levels.length} mức
                {t.zeroCode != null && t.zeroValue != null ? <>; mã {t.zeroCode} nằm trên không ({t.zeroValue.toExponential(1)} sau khi lưu)</> : null}.
              </>
            }
          >
            relative RMS error {(100 * t.rel).toFixed(2)}%; {t.levels.length} levels
            {t.zeroCode != null && t.zeroValue != null ? <>; code {t.zeroCode} on zero ({t.zeroValue.toExponential(1)} after storage)</> : null}.
          </Translated>
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: error against block size and bit width
const DASH: Record<string, string> = { '16': '', '32': '6 3', '64': '2 3', '128': '8 3 2 3', '256': '1 5', row: '10 4', tensor: '3 6' };
const gkey = (g: Gran) => String(g);

function SweepPlot({ pts }: { pts: SweepPoint[] }) {
  const W = 720;
  const H = 330;
  const pad = { l: 52, r: 14, t: 12, b: 38 };
  const xs = pts.map((p) => p.bpw);
  const x0 = 2;
  const x1 = Math.max(...xs);
  const lo = Math.min(...pts.map((p) => p.rel)) / 1.3;
  const hi = Math.min(2, Math.max(...pts.map((p) => p.rel)) * 1.3);
  const X = (v: number) => pad.l + ((v - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(Math.min(v, hi)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (H - pad.t - pad.b);
  const ticks: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= 0; e++) for (const m of [1, 2, 5]) if (m * 10 ** e >= lo && m * 10 ** e <= hi) ticks.push(m * 10 ** e);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Relative error against bits per weight, one curve per block size', 'Sai số tương đối theo số bit mỗi trọng số, mỗi kích thước khối một đường')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {ticks.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 4} y={Y(v) + 3.5} fontSize="10" textAnchor="end">{v >= 0.01 ? `${+(v * 100).toFixed(1)}%` : `${+(v * 100).toPrecision(1)}%`}</text>
        </g>
      ))}
      {Array.from({ length: Math.floor(x1) - 1 }, (_, i) => i + 2).map((b) => (
        <g key={b}>
          <line x1={X(b)} x2={X(b)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(b)} y={H - pad.b + 13} fontSize="10" textAnchor="middle">{b}</text>
        </g>
      ))}
      <text x={(W - pad.r + pad.l) / 2} y={H - 6} fontSize="11" textAnchor="middle"><Translated vi="số bit mỗi trọng số, gồm cả hệ số tỉ lệ">bits per weight, scales included</Translated></text>
      <text x={12} y={(H - pad.b) / 2} fontSize="11" textAnchor="middle" transform={`rotate(-90 12 ${(H - pad.b) / 2})`}><Translated vi="sai số RMS tương đối (log)">relative RMS error (log)</Translated></text>
      {GRANS.map((g) => {
        const line = pts.filter((p) => p.gran === g).sort((a, b) => a.bits - b.bits);
        return (
          <g key={gkey(g)} data-curve={gkey(g)}>
            <polyline points={line.map((p) => `${X(p.bpw).toFixed(1)},${Y(p.rel).toFixed(1)}`).join(' ')} fill="none" stroke="var(--ink)" strokeWidth={1.6} strokeDasharray={DASH[gkey(g)]} />
            {line.map((p) => <circle key={p.bits} cx={X(p.bpw)} cy={Y(p.rel)} r={2.4} fill="var(--ink)" />)}
          </g>
        );
      })}
      <g transform={`translate(${W - pad.r - 128}, ${pad.t + 8})`}>
        <rect x={-6} y={-6} width={130} height={GRANS.length * 15 + 6} fill="var(--page)" stroke="var(--rule)" />
        {GRANS.map((g, i) => (
          <g key={gkey(g)} transform={`translate(0, ${i * 15})`}>
            <line x1={0} x2={34} y1={5} y2={5} stroke="var(--ink)" strokeWidth={1.6} strokeDasharray={DASH[gkey(g)]} />
            <text x={40} y={9} fontSize="10">{granLabel(g)}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}

const BUDGETS = [4.5, 5, 5.5, 6, 8.5];

function Sweep() {
  const [dist, setDist] = useState<Dist>('normal');
  const [scheme, setScheme] = useState<Scheme>('signed-extreme');
  const [budget, setBudget] = useState(5);
  const pts = useMemo(() => errorSweep({ dist, scheme }), [dist, scheme]);
  const best = bestUnder(pts, budget);
  const at = (g: Gran, b: number) => pts.find((p) => p.gran === g && p.bits === b)!;
  return (
    <Bench
      title={tr('Error against block size and bit width', 'Sai số theo kích thước khối và độ rộng bit')}
      task={tr('Switch to heavy tails and watch which curves move. Then pick a budget and see where it is best spent.', 'Chuyển sang phân phối đuôi nặng và xem đường nào dịch chuyển. Sau đó chọn một ngân sách và xem nên dùng nó vào đâu là tốt nhất.')}
      foot={
        <Translated
          vi={
            <>
              Một ma trận 16 × 4096 sinh từ seed cố định cho mỗi phân phối, được lượng tử hóa dọc theo từng hàng với một hệ số
              tỉ lệ FP16 (và, với affine, một độ lệch FP16) cho mỗi khối. Số bit mỗi trọng số tính cả chúng: 16 hoặc 32 bit mỗi khối. {SCHEME_LABEL[scheme]}.
              Phép tính ở độ chính xác kép trừ các hệ số tỉ lệ FP16: một mô hình sai số, không phải một kernel.
            </>
          }
        >
          <>
            One seeded 16 × 4096 matrix per distribution, quantised along each row with one FP16 scale (and, for affine,
            one FP16 bias) per block. Bits per weight count those: 16 or 32 bits per block. {SCHEME_LABEL[scheme]}.
            Arithmetic in double precision apart from the FP16 scales: an error model, not a kernel.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Trọng số">Weights</Translated>
          <span className="seg" role="group" aria-label={tr('sweep distribution', 'phân phối của lượt quét')}>
            {(['normal', 't3'] as Dist[]).map((d) => (
              <button key={d} aria-pressed={dist === d} onClick={() => setDist(d)}>{d === 'normal' ? tr('normal', 'chuẩn') : tr('Student t (3)', 'Student t (3)')}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Lược đồ">Scheme</Translated>
          <select aria-label={tr('scheme', 'lược đồ')} value={scheme} onChange={(e) => setScheme(e.target.value as Scheme)}>
            {(Object.keys(SCHEME_LABEL) as Scheme[]).map((s) => <option key={s} value={s}>{SCHEME_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Ngân sách, bit mỗi trọng số">Budget, bits per weight</Translated>
          <span className="seg" role="group" aria-label={tr('budget', 'ngân sách')}>
            {BUDGETS.map((b) => <button key={b} aria-pressed={budget === b} onClick={() => setBudget(b)}>{b}</button>)}
          </span>
        </label>
      </div>
      <div>
        <SweepPlot pts={pts} />
        <div style={{ marginTop: '0.8rem' }}>
          <p data-part="best" data-bits={best?.bits} data-gran={best ? gkey(best.gran) : ''} style={{ marginTop: 0 }}>
            {best ? (
              <Translated
                vi={
                  <>
                    Trong phạm vi {budget} bit mỗi trọng số, sai số thấp nhất trên trọng số {DIST_LABEL[dist]} là{' '}
                    <b>{best.bits} bit, {granLabel(best.gran)}</b>: {(100 * best.rel).toFixed(2)}% ở {best.bpw.toFixed(3)} bit.
                  </>
                }
              >
                <>
                  Within {budget} bits per weight, the lowest error on {DIST_LABEL[dist]} weights is{' '}
                  <b>{best.bits} bits, {granLabel(best.gran)}</b>: {(100 * best.rel).toFixed(2)}% at {best.bpw.toFixed(3)} bits.
                </>
              </Translated>
            ) : (
              <Translated vi={<>Không có gì vừa {budget} bit mỗi trọng số với lược đồ này.</>}>
                <>Nothing fits {budget} bits per weight with this scheme.</>
              </Translated>
            )}
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table className="data" data-part="sweep" style={{ margin: 0 }}>
              <thead>
                <tr><th><Translated vi="một hệ số tỉ lệ cho mỗi">one scale per</Translated></th>{SWEEP_BITS.map((b) => <th key={b} className="num" style={{ whiteSpace: 'nowrap' }}><Translated vi={<>{b} bit</>}>{b} bits</Translated></th>)}</tr>
              </thead>
              <tbody>
                {GRANS.map((g) => (
                  <tr key={gkey(g)}>
                    <td style={{ whiteSpace: 'nowrap' }}>{granLabel(g)}</td>
                    {SWEEP_BITS.map((b) => (
                      <td key={b} className="num" data-cell={`${gkey(g)}:${b}`} data-rel={at(g, b).rel}>{(100 * at(g, b).rel).toFixed(1)}%</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- predict: where to spend a bit
const SPEND_ID = 'x2/X2.2/p-spend';
const SPEND = () => [tr('halve the group to 32 (5.0 bits per weight)', 'giảm nửa nhóm xuống 32 (5.0 bit mỗi trọng số)'), tr('keep group 64, use 5 bits (5.5 bits per weight)', 'giữ nhóm 64, dùng 5 bit (5.5 bit mỗi trọng số)')];
function SpendPredict() {
  const { progress, answer } = useProgress();
  const prior = progress.answers[SPEND_ID]?.value;
  const committed = typeof prior === 'number' ? prior : null;
  const pts = useMemo(() => errorSweep({ dist: 't3', scheme: 'affine', bits: [4, 5], grans: [32, 64] }), []);
  const get = (b: number, g: number) => pts.find((p) => p.bits === b && p.gran === g)!;
  const base = get(4, 64);
  const opts = [get(4, 32), get(5, 64)];
  const lower = opts[0].rel < opts[1].rel ? 0 : 1;
  const perBit = opts.map((o) => (base.rel - o.rel) / (o.bpw - base.bpw));
  return (
    <div className="q" data-part="spend">
      <div className="kind"><Translated vi="Dự đoán trước">Predict first</Translated></div>
      <Translated
        vi={
          <div className="prompt">
            Trọng số đuôi nặng (Student t), affine kiểu MLX, 4 bit, nhóm 64, hệ số tỉ lệ và độ lệch FP16: 4.5 bit mỗi trọng số.
            Bạn có thể chi thêm. Cách nào giảm sai số nhiều hơn?
          </div>
        }
      >
        <div className="prompt">
          Heavy-tailed (Student t) weights, MLX-style affine, 4 bits, group 64, FP16 scale and bias: 4.5 bits per weight.
          You may spend more. Which lowers the error more?
        </div>
      </Translated>
      {committed == null ? (
        <div className="row">
          {SPEND().map((s, i) => (
            <button key={i} onClick={() => answer(SPEND_ID, i)}>{s}</button>
          ))}
        </div>
      ) : (
        <div className="verdict info" data-lower={lower}>
          <p><b><Translated vi={<>Bạn đã chọn: {SPEND()[committed]}.</>}>You chose: {SPEND()[committed]}.</Translated></b></p>
          <Translated
            vi={
              <p>
                Theo mô hình: {(100 * base.rel).toFixed(2)}% hiện tại; nhóm 32 cho {(100 * opts[0].rel).toFixed(2)}%; 5 bit ở
                nhóm 64 cho {(100 * opts[1].rel).toFixed(2)}%. {lower === 0 ? 'Nhóm nhỏ hơn' : 'Bit thêm'} giảm sai số
                nhiều hơn{committed === lower ? ', đúng như bạn dự đoán' : ''}. Trên mỗi bit chi ra: {(100 * perBit[0]).toFixed(2)} điểm cho
                nhóm, {(100 * perBit[1]).toFixed(2)} cho bit.
              </p>
            }
          >
            <p>
              From the model: {(100 * base.rel).toFixed(2)}% now; group 32 gives {(100 * opts[0].rel).toFixed(2)}%; 5 bits at
              group 64 gives {(100 * opts[1].rel).toFixed(2)}%. The {lower === 0 ? 'smaller group' : 'extra bit'} lowers the error
              more{committed === lower ? ', as you predicted' : ''}. Per bit spent: {(100 * perBit[0]).toFixed(2)} points for the
              group, {(100 * perBit[1]).toFixed(2)} for the bit.
            </p>
          </Translated>
          <Translated
            vi={
              <p className="muted">
                Cả hai đều là lựa chọn thật trong MLX (nhóm 32, 64, 128; bit từ 1 đến 6 và 8). Bit thêm giảm một nửa mọi bước;
                nhóm nhỏ hơn chỉ giúp những trọng số mà khối của chúng chứa giá trị ngoại lai, và với phần đuôi này đó là thiểu số.
              </p>
            }
          >
            <p className="muted">
              Both are real options in MLX (groups 32, 64, 128; bits 1 to 6 and 8). The bit halves every step; the smaller
              group only helps the weights whose block holds an outlier, and on these tails that is a minority.
            </p>
          </Translated>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the page
export function SX22() {
  const q = questionsSX22;
  const src = useLabFile(SNIPPET);
  const d32 = 0.1;
  return (
    <>
      <Translated
        vi={
          <p>
            Mục X2.1 cho thấy ít byte hơn cho mỗi trọng số giúp decode nhanh hơn. Mục này nói về thứ bạn phải đánh đổi để có
            chúng. Một mã 4 bit chứa được 16 giá trị; một trọng số thật có thể là bất kỳ số float nào. Những lựa chọn quyết định
            bạn tiến gần đến đâu không nhiều: một hệ số tỉ lệ hay một hệ số tỉ lệ cùng một độ dời, bao nhiêu trọng số dùng chung
            chúng, và hệ số tỉ lệ được chọn và lưu thế nào. Bạn sẽ gặp từng lựa chọn trong các quy tắc mà llama.cpp và MLX thực
            sự chạy.
          </p>
        }
      >
        <p>
          Section X2.1 showed that fewer bytes per weight make decode faster. This one is about what you give up for them.
          A 4-bit code can hold 16 values; a real weight can hold any float. The choices that decide how close you get are
          few: one scale or a scale and an offset, how many weights share them, and how the scale is chosen and stored.
          You will meet each one in the rules llama.cpp and MLX actually run.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Sách dừng lại ở các định dạng dấu phẩy động. Lượng tử hóa số nguyên với hệ số tỉ lệ và zero point không có trong
              sách; mọi quy tắc ở đây đều được đối chiếu với mã nguồn của các dự án trong lượt kiểm tra dữ kiện.
            </p>
          }
        >
          <p>
            The book stops at floating-point formats. Integer quantisation with scales and zero points is not in it; every
            rule here is checked against the projects' source in the facts pass.
          </p>
        </Translated>
      </Beyond>
      <Source section="A.5" pages="592–593" />

      <h2><Translated vi="Tấm bản đồ">The map</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi lược đồ ở đây lưu một mã số nguyên q cho mỗi trọng số và khôi phục w ≈ s·(q - z): một hệ số tỉ lệ s, và một zero
            point z cho biết mã nào nghĩa là không. Các lược đồ <b>đối xứng</b> cố định z (nên lưới được căn giữa quanh không) và
            chỉ lưu s;{' '}
            các lược đồ <b>bất đối xứng</b> hay <b>affine</b> lưu thêm một độ dời, nên 16 mức có thể nằm ở bất cứ đâu trọng số
            nằm. Trục thứ hai là độ mịn: một hệ số tỉ lệ cho mỗi tensor, mỗi hàng, hoặc mỗi <b>khối</b> (MLX gọi là group) gồm g
            trọng số liên tiếp dọc theo một hàng. Theo cách nói của PyTorch, hệ số tỉ lệ theo khối là một tensor có shape (out,
            in / g) nằm cạnh các mã.
          </p>
        }
      >
        <p>
          Every scheme here stores an integer code q per weight and recovers w ≈ s·(q - z): a scale s, and a zero point z
          that says which code means zero. <b>Symmetric</b> schemes fix z (so the grid is centred on zero) and store only s;{' '}
          <b>asymmetric</b> or <b>affine</b> ones store an offset too, so the 16 levels can sit wherever the weights are.
          The second axis is granularity: one scale per tensor, per row, or per <b>block</b> (MLX says group) of g
          consecutive weights along a row. In PyTorch terms, a block scale is a tensor of shape (out, in / g) beside the
          codes.
        </p>
      </Translated>

      <h2><Translated vi="Q8_0: absmax chia 127">Q8_0: absmax over 127</Translated></h2>
      <Translated
        vi={
          <p>
            Định dạng 8 bit của llama.cpp lấy khối 32, tìm độ lớn lớn nhất, và đặt d = amax / 127. Mỗi trọng số trở thành{' '}
            <code>roundf(x / d)</code>, một mã từ -127 đến 127, và d được lưu dưới dạng FP16 cạnh 32 byte: 34 byte mỗi khối.
            Không bao giờ có gì bị chặn, và với 255 mức sai số rất nhỏ. Mã -128 không bao giờ được dùng.
          </p>
        }
      >
        <p>
          llama.cpp's 8-bit format takes the block of 32, finds the largest magnitude, and sets d = amax / 127. Each weight
          becomes <code>roundf(x / d)</code>, a code from -127 to 127, and d is stored as FP16 beside the 32 bytes: 34 bytes
          per block. Nothing ever clips, and with 255 levels the error is tiny. Code -128 is never used.
        </p>
      </Translated>
      <Today checked={CHECKED('D1')}>
        <Translated
          vi={
            <p>
              Các quy tắc trên trang này là các bộ lượng tử hóa <code>_ref</code> của llama.cpp, những bộ mà nó gọi là "reference
              implementation for deterministic creation of model files". Khi có ma trận độ quan trọng, nó chọn d theo cách
              khác (<code>make_qx_quants</code>); bố cục thì giống nhau.
            </p>
          }
        >
          <p>
            The rules on this page are llama.cpp's <code>_ref</code> quantisers, the ones it calls the "reference
            implementation for deterministic creation of model files". Given an importance matrix it chooses d
            differently (<code>make_qx_quants</code>); the layout is the same.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Q4_0: cực trị có dấu chia -8">Q4_0: the signed extreme over -8</Translated></h2>
      <Translated
        vi={
          <p>
            Định dạng 4 bit tinh tế hơn. Nó tìm giá trị có độ lớn lớn nhất <i>kèm theo dấu</i>, gọi là max, và đặt d = max / -8.
            Sau đó mỗi trọng số trở thành <code>MIN(15, (int8_t)(x/d + 8.5))</code> và được giải lượng tử hóa thành (q - 8)·d.
            Vậy cực trị rơi đúng vào mã 0, mức âm nhất, và các mã phủ từ -8d đến +7d. Một trọng số ở phía bên kia với cùng độ
            lớn sẽ cần +8 nhưng nhận +7: nó bị chặn mất một bước. Chính một bước đó đổi lấy mức thứ mười sáu. Người ta gọi Q4_0
            là đối xứng; hệ số tỉ lệ của nó thì đối xứng, còn khoảng mã thì không.
          </p>
        }
      >
        <p>
          The 4-bit format is subtler. It finds the value of largest magnitude <i>with its sign</i>, call it max, and sets
          d = max / -8. Then each weight becomes <code>MIN(15, (int8_t)(x/d + 8.5))</code> and dequantises as (q - 8)·d.
          So the extreme lands exactly on code 0, the most negative level, and the codes cover -8d to +7d. A weight on the
          other side with the same magnitude would need +8 and gets +7: it clips by one step. That one step is what buys
          the sixteenth level. People call Q4_0 symmetric; its scale is, its code range is not.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Cộng 8.5 rồi cắt phần lẻ là làm tròn nửa lên, dời đi để các mã bắt đầu từ không. Hãy chạy nó trong trình chạy từng
            bước bên dưới với khối mặc định, có trọng số 5 là -0.8 và trọng số 21 là ảnh đối xứng của nó, +0.8: d = {d32}, và
            trọng số 21 quay lại thành 0.7.
          </p>
        }
      >
        <p>
          Adding 8.5 and truncating is round half up, shifted so the codes start at zero. Run it in the stepper below with
          the default block, whose weight 5 is -0.8 and whose weight 21 is its mirror, +0.8: d = {d32}, and weight 21 comes
          back as 0.7.
        </p>
      </Translated>
      {src ? (
        <Code
          tabs={[{ label: 'q4_0_ref.h', code: src, status: SNIPPET_STATUS(), highlight: [49, 50, 53, 54, 55, 58, 59, 60, 64, 66] }]}
          caption={
            <Translated
              vi={
                <>
                  Một tham chiếu CPU cho một khối, viết theo <code>quantize_row_q4_0_ref</code> của llama.cpp, nạp từ{' '}
                  <code>labs/x2/snippets/q4_0_ref.h</code>. Đây là dạng mà tham chiếu CPU của một lab có. Những dòng quyết định
                  quy tắc được đánh dấu.
                </>
              }
            >
              <>
                A CPU reference for one block, written from llama.cpp's <code>quantize_row_q4_0_ref</code>, loaded from{' '}
                <code>labs/x2/snippets/q4_0_ref.h</code>. It is the shape a lab's CPU reference takes. Lines that decide the
                rule are marked.
              </>
            </Translated>
          }
        />
      ) : null}
      <Ask q={q['x2/X2.2/q-q40-clip']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={q['x2/X2.2/q-q40-zero']} />

      <h2><Translated vi="Q4_1 và MLX: một hệ số tỉ lệ và một độ dời">Q4_1 and MLX: a scale and an offset</Translated></h2>
      <Translated
        vi={
          <p>
            Q4_1 là người anh em bất đối xứng của llama.cpp: d = (max - min) / 15, m = min, value = q·d + m, cả hai lưu dưới dạng
            FP16, 20 byte mỗi khối. Không mức nào rơi ra ngoài khoảng giá trị của trọng số, nên khi một khối bị lệch (mọi giá trị
            lớn đều ở một phía) không mức nào bị lãng phí.
          </p>
        }
      >
        <p>
          Q4_1 is llama.cpp's asymmetric twin: d = (max - min) / 15, m = min, value = q·d + m, both stored as FP16, 20 bytes
          per block. No level falls outside the weights' range, so when a block is lopsided (all its large values on one
          side) none is wasted.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chế độ <code>affine</code> của MLX ghi trong tài liệu cùng ý tưởng: s = (max - min) / (2^b - 1), bias = min, mã từ 0
            đến 2^b - 1, với b từ 1 đến 6 hoặc 8 và nhóm 32, 64 hoặc 128. Mã của nó làm hơi khác. Nó lấy cạnh có độ lớn lớn hơn
            làm bias (nên hệ số tỉ lệ âm khi cạnh đó là max), làm tròn edge / s thành một số nguyên q0, và đặt lại s = edge / q0.
            Hiệu quả: mã -q0 giải lượng tử hóa thành không, sai khác chỉ do làm tròn một phép chia. Một trọng số vốn đúng bằng 0
            vẫn là 0, còn cạnh xa có thể bị chặn một phần của một bước. Hãy chọn cả hai quy tắc MLX trong trình chạy từng bước và
            so sánh bước "suy ra hệ số tỉ lệ".
          </p>
        }
      >
        <p>
          MLX's <code>affine</code> mode documents the same idea: s = (max - min) / (2^b - 1), bias = min, codes from 0 to
          2^b - 1, with b from 1 to 6 or 8 and groups of 32, 64 or 128. Its code does something slightly different. It takes
          the edge of larger magnitude as the bias (so the scale is negative when that edge is the max), rounds edge / s to
          an integer q0, and resets s = edge / q0. The effect: code -q0 dequantises to zero, up to the rounding of one
          division. A weight that was exactly 0 stays 0, and the far edge may clip by a fraction of a step. Pick both MLX
          rules in the stepper and compare the step "derive the scale".
        </p>
      </Translated>
      <Today checked={CHECKED('D5')}>
        <Translated
          vi={
            <p>
              <code>mx.quantize(w, group_size=64, bits=4)</code> trả về <code>(w_q, scales, biases)</code>: các mã đóng gói
              vào word uint32, và một hệ số tỉ lệ cùng một độ lệch cho mỗi nhóm theo kiểu của đầu vào. <code>mlx.nn.QuantizedLinear</code>{' '}
              dùng cùng giá trị mặc định. Tài liệu của MLX đưa công thức min/max; kernel tính phần tinh chỉnh ở trên.
            </p>
          }
        >
          <p>
            <code>mx.quantize(w, group_size=64, bits=4)</code> returns <code>(w_q, scales, biases)</code>: codes packed into
            uint32 words, and one scale and one bias per group in the input's type. <code>mlx.nn.QuantizedLinear</code>{' '}
            uses the same defaults. MLX's docs give the min/max formula; the kernel computes the refinement above.
          </p>
        </Translated>
      </Today>
      <Ask q={q['x2/X2.2/q-bits']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Một khối, sáu quy tắc">One block, six rules</Translated></h2>
      <Translated
        vi={
          <p>
            Chạy từng bước Q4_0 qua khối mặc định và tìm trọng số bị chặn. Sau đó chỉ đổi quy tắc. Trước khi bấm Bước tiếp với mỗi
            quy tắc, hãy dự đoán: có gì bị chặn không, và mức gần không nhất nằm ở đâu? Rồi kéo giá trị ngoại lai đến 2.5 và
            xem điều gì xảy ra với sai số của mọi trọng số khác dưới các quy tắc đối xứng.
          </p>
        }
      >
        <p>
          Step Q4_0 through the default block and find the clipped weight. Then change only the rule. Before you press
          Step on each, predict: does anything clip, and where is the level nearest zero? Then drag the outlier to 2.5 and
          watch what happens to every other weight's error under the symmetric rules.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trọng số tổng hợp, độc lập: một phân phối chuẩn, hoặc một Student t với 3 bậc tự do cho phần đuôi nặng, được
              chia tỉ lệ về cùng phương sai. Ma trận trọng số thật có cấu trúc (chẳng hạn các kênh ngoại lai) mà mô hình này bỏ
              qua. Sai số được đo trên trọng số, không phải trên đầu ra của mô hình; mục X2.7 làm việc đó.
            </p>
          }
        >
          <p>
            Synthetic, independent weights: a normal distribution, or a Student t with 3 degrees of freedom for heavy
            tails, scaled to the same variance. Real weight matrices have structure (outlier channels, for one) that this
            leaves out. Error is measured on the weights, not on a model's output; section X2.7 does that.
          </p>
        </Translated>
      </Simplification>
      <OneBlock />

      <h2><Translated vi="Bốn quy ước cho một ý tưởng">Four conventions for one idea</Translated></h2>
      <Translated
        vi={
          <p>
            Mọi định dạng bất đối xứng đều lưu cùng hai số cho mỗi nhóm, nhưng mỗi dự án viết công thức theo cách riêng. Khi bạn
            chuyển trọng số giữa chúng, đây là bảng bạn cần.
          </p>
        }
      >
        <p>
          Every asymmetric format stores the same two numbers per group, but each project writes the formula its own way.
          When you move weights between them, this is the table you need.
        </p>
      </Translated>
      <div style={{ overflowX: 'auto' }} className="wide">
        <table className="data" style={{ maxWidth: '52rem' }}>
          <thead>
            <tr><th><Translated vi="Dự án">Project</Translated></th><th><Translated vi="Giá trị của mã q">Value of code q</Translated></th><th><Translated vi="Những gì được lưu">What is stored</Translated></th><th><Translated vi="Dữ kiện">Fact</Translated></th></tr>
          </thead>
          <tbody>
            <tr><td>llama.cpp Q4_1{/* i18n-ignore: product and format names */}</td><td><code>d·q + m</code></td><td><Translated vi="d và m dưới dạng FP16, mã 0..15">d and m as FP16, codes 0..15</Translated></td><td>D1</td></tr>
            <tr><td><Translated vi="MLX affine">MLX affine</Translated></td><td><code>s·q + bias</code></td><td><Translated vi="hệ số tỉ lệ và độ lệch theo kiểu của đầu vào">scale and bias in the input's type</Translated></td><td>D5</td></tr>
            <tr><td>MPSGraph{/* i18n-ignore: a product name */}</td><td><code>scale·(q - zeroPoint)</code></td><td><Translated vi="hệ số tỉ lệ float, zero point số nguyên theo kiểu của đầu vào (Int4, UInt4, Int8, UInt8)">float scale, integer zero point of the input's type (Int4, UInt4, Int8, UInt8)</Translated></td><td>C4</td></tr>
            <tr><td>PyTorch int4 (tinygemm){/* i18n-ignore: product and kernel names */}</td><td><code>q·scale + zero</code></td><td><Translated vi="mã dạng int4 trong -8..7, scale và zero ở bf16; zero được cộng sau khi nhân tỉ lệ">codes as int4 in -8..7, scale and zero in bf16; the zero is added after scaling</Translated></td><td>D6</td></tr>
          </tbody>
        </table>
      </div>
      <Ask q={q['x2/X2.2/q-convert']} />

      <h2><Translated vi="Độ mịn và sai số">Granularity and error</Translated></h2>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chỉ là một khối. Trình khám phá bên dưới lượng tử hóa cả một ma trận 16 × 4096 và vẽ sai số
            tương đối theo cái giá bạn trả, số bit mỗi trọng số gồm cả hệ số tỉ lệ. Bốn điều cần tìm: thêm một bit làm được gì;
            khối nhỏ giúp trọng số phân phối chuẩn hay phân phối đuôi nặng nhiều hơn; mức thứ mười sáu của Q4_0 có đáng với việc
            bị chặn không; và một hệ số tỉ lệ cho mỗi tensor làm gì với phân phối đuôi nặng ở 4 bit.
          </p>
        }
      >
        <p>
          The stepper is one block. The explorer below quantises a whole 16 × 4096 matrix and plots relative error
          against what you pay, bits per weight with the scales included. Four things to find: what one more bit does;
          whether small blocks help normal weights or heavy tails more; whether Q4_0's sixteenth level is worth its
          clipping; and what one scale per tensor does to heavy tails at 4 bits.
        </p>
      </Translated>
      <Today checked={CHECKED('D6')}>
        <Translated
          vi={
            <p>
              Lượng tử hóa của PyTorch giờ nằm trong torchao. <code>Int4WeightOnlyConfig</code> của nó chỉ theo nhóm, với kích
              thước nhóm 256, 128, 64 và 32 và mặc định là 128.
            </p>
          }
        >
          <p>
            PyTorch's quantisation lives in torchao now. Its <code>Int4WeightOnlyConfig</code> is group-wise only, with
            group sizes 256, 128, 64 and 32 and a default of 128.
          </p>
        </Translated>
      </Today>
      <Sweep />
      <SpendPredict />
      <Ask q={q['x2/X2.2/q-outlier']} />

      <h2><Translated vi="Hệ số tỉ lệ được lưu">The stored scale</Translated></h2>
      <Translated
        vi={
          <p>
            Hãy nhìn lại bộ lượng tử hóa: các mã được tính với d dạng float, rồi khối lưu FP16(d). Vậy phép giải lượng tử hóa
            nhân với một số hơi khác so với số mà các mã được chọn theo. Với khối mặc định d = 0.1 nhưng FP16 giữ {fp16(0.1)},
            một thay đổi tương đối khoảng {(100 * Math.abs(fp16(0.1) - 0.1) / 0.1).toFixed(3)}%. Con số đó nhỏ so với một bước 4
            bit, đó là lý do xét về độ chính xác thì không ai bận tâm. Nó quan trọng cho việc kiểm thử: một tham chiếu CPU bỏ qua bước
            khứ hồi FP16 sẽ không khớp từng bit với các giá trị giải lượng tử hóa của GPU. Bỏ chọn "lưu hệ số tỉ lệ dưới dạng FP16"
            trong trình chạy từng bước và xem chữ số cuối của các giá trị giải lượng tử hóa thay đổi. Bản thân FP16 là chủ đề
            của phụ lục A (<a href={href('appA', 'A.2')}>mục A.2</a>).
          </p>
        }
      >
        <p>
          Look again at the quantiser: the codes are computed with the float d, and then the block stores FP16(d). So
          dequantisation multiplies by a slightly different number from the one the codes were chosen for. For the
          default block d = 0.1 but FP16 keeps {fp16(0.1)}, a relative change of about {(100 * Math.abs(fp16(0.1) - 0.1) / 0.1).toFixed(3)}%.
          That is small next to a 4-bit step, which is why nobody cares for accuracy. It matters for testing: a CPU
          reference that skips the FP16 round trip will not match the GPU's dequantised values bit for bit. Untick "store
          the scale as FP16" in the stepper and watch the last digit of the dequantised values change. FP16 itself is
          appendix A's subject (<a href={href('appA', 'A.2')}>section A.2</a>).
        </p>
      </Translated>
      <Source section="A.2" pages="586–590" />

      <h2><Translated vi="Hệ số tỉ lệ hai cấp: Q4_K">Two-level scales: Q4_K</Translated></h2>
      <Translated
        vi={
          <p>
            Một hệ số tỉ lệ và một min cho mỗi 32 trọng số khiến Q4_1 tốn trọn một bit mỗi trọng số. Q4_K của llama.cpp giữ cả
            hai mà chỉ trả một nửa: một siêu khối 256 trọng số có hai số FP16, d và dmin, và mỗi khối con 32 trọng số trong
            tám khối con của nó có một hệ số tỉ lệ 6 bit và một min 6 bit nhân với chúng. Mười sáu số 6 bit vừa trong 12 byte,
            nên khối là 2 + 2 + 12 + 128 = 144 byte: {bitsPerWeight(Q4_K)} bit mỗi trọng số so với {bitsPerWeight(Q4_1)} của
            Q4_1, và bằng {bitsPerWeight(Q4_0)} của Q4_0, vốn không có min. Giá trị là d·sc·q - dmin·m. Cách đóng gói bit của 12
            byte đó là loại chi tiết mà một kernel phải làm đúng tuyệt đối:
          </p>
        }
      >
        <p>
          A scale and a min per 32 weights cost Q4_1 a full bit per weight. llama.cpp's Q4_K keeps both and pays half:
          a super-block of 256 weights has two FP16 numbers, d and dmin, and each of its eight 32-weight sub-blocks gets a
          6-bit scale and a 6-bit min that multiply them. Sixteen 6-bit numbers fit in 12 bytes, so the block is 2 + 2 +
          12 + 128 = 144 bytes: {bitsPerWeight(Q4_K)} bits per weight against Q4_1's {bitsPerWeight(Q4_1)}, and the same as
          Q4_0's {bitsPerWeight(Q4_0)}, which has no min at all. The value is d·sc·q - dmin·m. The bit-packing of those 12
          bytes is the kind of detail a kernel must get exactly right:
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Bảng này là <code>get_scale_min_k4</code> đọc như dữ liệu; một unit test đóng gói các hệ số tỉ lệ và min ngẫu
              nhiên bằng hàm nghịch đảo của nó rồi đọc lại từng giá trị.
            </p>
          }
        >
          <p>
            The table is <code>get_scale_min_k4</code> read as data; a unit test packs random scales and mins with its
            inverse and reads every one back.
          </p>
        </Translated>
      </Aside>
      <div style={{ overflowX: 'auto' }} className="wide">
        <table className="data" data-part="q4k" style={{ maxWidth: '52rem' }}>
          <thead><tr><th><Translated vi="Khối con j">Sub-block j</Translated></th><th><Translated vi="Hệ số tỉ lệ 6 bit lấy từ">6-bit scale from</Translated></th><th><Translated vi="Min 6 bit lấy từ">6-bit min from</Translated></th></tr></thead>
          <tbody>
            {Array.from({ length: 8 }, (_, j) => (
              <tr key={j}><td>{j}</td><td>{scaleMinSources(j).sc}</td><td>{scaleMinSources(j).m}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <Ask q={q['x2/X2.2/q-q4k']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Tiếp theo, mục <a href={href('x2', 'X2.3')}>X2.3</a> đặt hai mã vào mỗi byte và cho thấy vị trí của mỗi nibble
            được chọn cho kernel đọc nó.
          </p>
        }
      >
        <p>
          Next, section <a href={href('x2', 'X2.3')}>X2.3</a> puts two codes in each byte and shows that where each
          nibble goes is chosen for the kernel that reads it.
        </p>
      </Translated>
    </>
  );
}
