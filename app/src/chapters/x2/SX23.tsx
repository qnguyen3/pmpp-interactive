import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr, useLocale } from '../../lib/i18n';
import {
  LAYOUT_LABEL,
  LAYOUT_WEIGHTS,
  SCENE_BLOCK,
  blockAddress,
  demoBlock,
  isAligned,
  packMlx4,
  packQ4K,
  packQ4_0,
  placeOf,
  qsLoads,
  quantTrace,
  quantizeQ4_0,
  quantizeQ4_1,
  sceneActivations,
  unpackTrace,
  weightsInByte,
  type Layout,
  type UnpackVariant,
} from '../../sim/quant';
import { PackedBlockScene } from './PackedBlockScene';

const FACTS = 'notes/today/x2-facts.md';
const CHECKED = (items: string) => tr(`2026-11-20 in the facts pass (${FACTS}, ${items})`, `2026-11-20 trong lượt kiểm tra dữ kiện (${FACTS}, ${items})`);
const SNIPPET = 'x2/snippets/q4_0_ref.h';
const hex = (v: number, digits: number) => (v >>> 0).toString(16).toUpperCase().padStart(digits, '0');

// ---------------------------------------------------------------- questions
export const questionsSX23: Record<string, Question> = {
  'x2/X2.3/q-byte9': {
    kind: 'numeric',
    id: 'x2/X2.3/q-byte9',
    prompt: <Translated vi={<>Trong một khối Q4_0 của llama.cpp, trọng số nào (0 đến 31) nằm ở nibble cao của byte mã số 9?</>}><>In a llama.cpp Q4_0 block, which weight (0 to 31) is in the high nibble of quant byte 9?</></Translated>,
    answer: weightsInByte('q4_0', 9)[1],
    explain: <Translated vi={<p>Byte j chứa trọng số j ở nửa thấp và trọng số j + 16 ở nửa cao, nên byte 9 chứa các trọng số 9 và 25.</p>}><p>Byte j holds weight j low and weight j + 16 high, so byte 9 holds weights 9 and 25.</p></Translated>,
    source: `${FACTS} D1 (quantize_row_q4_0_ref, dequantize_row_q4_0)`, // i18n-ignore: a citation of code names
  },
  'x2/X2.3/q-mlx13': {
    kind: 'choice',
    id: 'x2/X2.3/q-mlx13',
    prompt: <Translated vi={<>MLX 4 bit: trọng số 13 của một nhóm nằm ở đâu?</>}><>MLX 4-bit: where is weight 13 of a group?</></Translated>,
    options: [
      { label: <Translated vi="word uint32 số 1, các bit 20 đến 23">uint32 word 1, bits 20 to 23</Translated>, correct: true },
      { label: <Translated vi="word uint32 số 1, các bit 8 đến 11">uint32 word 1, bits 8 to 11</Translated>, why: <Translated vi="Trọng số 13 là trọng số thứ sáu của word 1 (13 - 8 = 5), và 5 × 4 = 20.">Weight 13 is the sixth of word 1 (13 - 8 = 5), and 5 × 4 = 20.</Translated> },
      { label: <Translated vi="byte 13, nibble thấp">byte 13, low nibble</Translated>, why: <Translated vi="Như thế là một mã mỗi byte. MLX đóng gói 8 mã vào mỗi word 32 bit.">That would be one code per byte. MLX packs 8 codes per 32-bit word.</Translated> },
      { label: <Translated vi="word uint32 số 0, các bit 28 đến 31">uint32 word 0, bits 28 to 31</Translated>, why: <Translated vi="Word 0 chỉ chứa các trọng số 0 đến 7.">Word 0 holds weights 0 to 7 only.</Translated> },
    ],
    explain: <Translated vi={<p>8 mã mỗi uint32, mã đầu tiên ở 4 bit thấp nhất: word 13 / 8 = 1, vị trí 13 mod 8 = 5, các bit 4 × 5 = 20 đến 23. Các trọng số kề nhau vẫn kề nhau.</p>}><p>8 codes per uint32, the first in the 4 least significant bits: word 13 / 8 = 1, position 13 mod 8 = 5, bits 4 × 5 = 20 to 23. Neighbours stay neighbours.</p></Translated>,
    source: `${FACTS} D5 (mlx.core.quantize docs)`,
    vi: { source: `${FACTS} D5 (tài liệu mlx.core.quantize)` },
  },
  'x2/X2.3/q-why-j16': {
    kind: 'choice',
    id: 'x2/X2.3/q-why-j16',
    prompt: <Translated vi={<>Vì sao llama.cpp đặt trọng số j + 16, chứ không phải trọng số j + 1, cạnh trọng số j trong một byte?</>}><>Why does llama.cpp put weight j + 16, rather than weight j + 1, beside weight j in a byte?</></Translated>,
    options: [
      {
        label: <Translated vi="Khi đó một mặt nạ cho ra các mã j..j+3 theo thứ tự activation, sẵn sàng cho dp4a.">One mask then gives codes j..j+3 in activation order, ready for dp4a.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Nó nén tốt hơn: các nibble cách nhau 16 thường có độ lớn tương tự.">It compresses better: nibbles 16 apart tend to be similar in size.</Translated>, why: <Translated vi="Cả hai thứ tự đều lưu đúng 4 bit mỗi trọng số.">Both orders store exactly 4 bits per weight.</Translated> },
      { label: <Translated vi="Nó giữ cho mã của mỗi khối căn chỉnh 2 byte, để đọc theo word 16 bit.">It keeps each block's codes 2-byte aligned, for 16-bit word reads.</Translated>, why: <Translated vi="Căn chỉnh do d FP16 và kích thước khối quyết định, không phải do thứ tự nibble.">Alignment is set by the FP16 d and the block size, not by the nibble order.</Translated> },
      { label: <Translated vi="GPU nạp nibble cao của một byte trước, nên j + 16 đến tiếp theo.">GPUs load the high nibble of a byte first, so j + 16 comes next.</Translated>, why: <Translated vi="Không lệnh nạp nào đọc nibble; các lệnh nạp đọc byte và word.">No load reads nibbles; loads read bytes and words.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một mặt nạ 32 bit cho ra các mã j..j+3, gặp bốn activation int8 liên tiếp trong một dp4a; word đã dịch bit cho ra
            j+16..j+19, gặp các activation xa hơn 16 vị trí. Nếu các trọng số kề nhau nằm chung một byte (thứ tự của MLX), lọc
            một word bằng mặt nạ cho ra các mã j, j+2, j+4, j+6 ở các nibble thấp và các trọng số lẻ kề chúng ở nibble cao, nên
            một tích vô hướng byte sẽ cần sắp xếp lại các activation. Với j và j + 16, các byte đã lọc nằm sẵn theo thứ tự
            activation. Bố cục được chọn cho kernel sẽ đọc nó.
          </p>
        }
      >
        <p>
          One 32-bit mask yields codes j..j+3, which meet four consecutive int8 activations in one dp4a; the shifted word
          yields j+16..j+19, which meet the activations 16 further on. With neighbours in a byte (MLX's order), masking a
          word gives codes j, j+2, j+4, j+6 in the low nibbles and
          their odd neighbours in the high ones, so a byte dot product would need the activations rearranged. With j and
          j + 16 the masked bytes are already in activation order. The layout is chosen for the kernel that reads it.
        </p>
      </Translated>
    ),
    source: `${FACTS} D3 (vecdotq.cuh); the comparison with neighbour packing is ours`,
    vi: { source: `${FACTS} D3 (vecdotq.cuh); phần so sánh với cách đóng gói kề nhau là của khóa học này` },
  },
  'x2/X2.3/q-align': {
    kind: 'choice',
    id: 'x2/X2.3/q-align',
    prompt: <Translated vi={<>Một hàng khối Q4_0 bắt đầu tại một địa chỉ căn chỉnh 16 byte. Mã của khối nào có thể đọc dưới dạng <code>int</code> (lần nạp 4 byte): các khối 0, 1, 2, 3?</>}><>A row of Q4_0 blocks starts at a 16-byte aligned address. Whose quants can be read as <code>int</code> (4-byte loads): blocks 0, 1, 2, 3?</></Translated>,
    options: [
      { label: <Translated vi="Các khối 1 và 3">Blocks 1 and 3</Translated>, correct: true },
      { label: <Translated vi="Cả bốn">All four</Translated>, why: <Translated vi="Mã bắt đầu sau 2 byte trong một khối 18 byte; 18i + 2 là bội số của 4 chỉ khi i lẻ.">The quants start 2 bytes into an 18-byte block; 18i + 2 is a multiple of 4 only for odd i.</Translated> },
      { label: <Translated vi="Các khối 0 và 2">Blocks 0 and 2</Translated>, why: <Translated vi="Mã của khối 0 bắt đầu tại byte 2, tức 2 mod 4.">Block 0’s quants start at byte 2, which is 2 mod 4.</Translated> },
      { label: <Translated vi="Không khối nào">None</Translated>, why: <Translated vi="Mã của khối 1 bắt đầu tại 18 + 2 = 20, một bội số của 4.">Block 1’s quants start at 18 + 2 = 20, a multiple of 4.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mã của khối i bắt đầu tại 18i + 2: các offset 2, 20, 38, 56. Chỉ 20 và 56 là bội số của 4. Một kernel đọc mã của mọi
            khối qua <code>int*</code> có hành vi không xác định với một nửa số khối; llama.cpp thay vào đó đọc hai nửa 16 bit.
          </p>
        }
      >
        <p>
          The quants of block i start at 18i + 2: offsets 2, 20, 38, 56. Only 20 and 56 are multiples of 4. A kernel that
          reads every block's quants through <code>int*</code> is undefined for half of them; llama.cpp reads two 16-bit
          halves instead.
        </p>
      </Translated>
    ),
    source: `${FACTS} A5, C5`,
  },
  'x2/X2.3/q-int4b': {
    kind: 'choice',
    id: 'x2/X2.3/q-int4b',
    prompt: <Translated vi={<>MSL giờ đã có <code>uint4b_format</code> và <code>unpack</code>. Vì sao kernel Metal của lab X2.2 vẫn giải nén bằng mặt nạ?</>}><>MSL now has <code>uint4b_format</code> and <code>unpack</code>. Why does lab X2.2's Metal kernel unpack with masks instead?</></Translated>,
    options: [
      {
        label: <Translated vi="Thứ tự nibble và mức hỗ trợ GPU của chúng không được tài liệu hóa.">Their nibble order and GPU support are undocumented.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Các kiểu đóng gói chậm hơn việc tự lọc byte bằng mặt nạ.">The packed types are slower than masking the bytes by hand.</Translated>, why: <Translated vi="Chưa ai công bố phép đo nào theo hướng nào, và ở đây không có khẳng định nào như vậy.">Nobody has published a measurement either way, and nothing here claims one.</Translated> },
      { label: <Translated vi="Chúng chỉ hoạt động với giá trị nằm trong bộ nhớ threadgroup.">They work only on values held in threadgroup memory.</Translated>, why: <Translated vi="Đặc tả không giới hạn chúng như vậy.">The spec does not restrict them so.</Translated> },
      { label: <Translated vi="Chúng chỉ có dấu, nên các mã không dấu của Q4_0 không dùng được.">They are signed only, so Q4_0's unsigned codes cannot use them.</Translated>, why: <Translated vi="Cả int4b_format và uint4b_format đều tồn tại.">Both int4b_format and uint4b_format exist.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các kiểu này còn cần Metal 4 với SDK 26.4 (và 4.1 cho <code>unpack</code>). Một kernel phải giải mã các byte của
            llama.cpp cần biết nibble nào là phần tử 0; đặc tả không nói, và Feature Set Tables không có hàng nào cho các kiểu số
            đóng gói. Cả MLX lẫn llama.cpp đều không dùng chúng (theo một lần tìm trong mã, không phải một chứng minh). Một lab
            muốn thử sẽ phải kiểm tra thứ tự nibble trên thiết bị so với một bản đóng gói trên CPU trước.
          </p>
        }
      >
        <p>
          The types also need Metal 4 with SDK 26.4 (and 4.1 for <code>unpack</code>). A kernel that must decode
          llama.cpp's bytes needs to know which nibble is element 0; the spec does not say, and
          the Feature Set Tables have no row for packed numeric types. Neither MLX nor llama.cpp uses them (a code search,
          not a proof). A lab that tried would have to test the nibble order on the device against a CPU pack first.
        </p>
      </Translated>
    ),
    source: `${FACTS} C2`,
  },
};

// ---------------------------------------------------------------- explorer: where each weight lives
const codesFor = (layout: Layout): number[] => {
  if (layout === 'q4_0') return quantizeQ4_0(SCENE_BLOCK).codes;
  if (layout === 'mlx4') return quantTrace('mlx-code', SCENE_BLOCK).codes;
  return [...quantizeQ4_1(SCENE_BLOCK).codes, ...quantizeQ4_1(demoBlock('normal', 0, '-', 8)).codes];
};
const bytesFor = (layout: Layout, codes: number[]): number[] => {
  if (layout === 'q4_0') return packQ4_0(codes);
  if (layout === 'q4_k') return packQ4K(codes);
  return packMlx4(codes).flatMap((w) => [w & 255, (w >>> 8) & 255, (w >>> 16) & 255, w >>> 24]);
};

type Sel = { kind: 'w'; j: number } | { kind: 'b'; b: number };

function WhereLives() {
  const [layout, setLayout] = useState<Layout>('q4_0');
  const [sel, setSel] = useState<Sel>({ kind: 'w', j: 5 });
  const codes = useMemo(() => codesFor(layout), [layout]);
  const bytes = useMemo(() => bytesFor(layout, codes), [layout, codes]);
  const n = LAYOUT_WEIGHTS[layout];
  const pair: [number, number] = sel.kind === 'w' ? (() => {
    const p = placeOf(layout, Math.min(sel.j, n - 1));
    return weightsInByte(layout, p.byte);
  })() : weightsInByte(layout, sel.b);
  const byte = sel.kind === 'w' ? placeOf(layout, Math.min(sel.j, n - 1)).byte : sel.b;
  const focusW = sel.kind === 'w' ? Math.min(sel.j, n - 1) : null;
  const pl = focusW != null ? placeOf(layout, focusW) : null;
  const readout =
    focusW != null && pl
      ? tr(
          `Weight ${focusW} (code ${codes[focusW]}) is in byte ${pl.byte}, ${pl.high ? 'high' : 'low'} nibble${pl.word != null ? `: uint32 word ${pl.word}, bits ${pl.bits![0]} to ${pl.bits![1]}` : ''}. That byte also holds weight ${pl.high ? pair[0] : pair[1]}.`,
          `Trọng số ${focusW} (mã ${codes[focusW]}) nằm ở byte ${pl.byte}, nibble ${pl.high ? 'cao' : 'thấp'}${pl.word != null ? `: word uint32 số ${pl.word}, các bit ${pl.bits![0]} đến ${pl.bits![1]}` : ''}. Byte đó cũng chứa trọng số ${pl.high ? pair[0] : pair[1]}.`,
        )
      : tr(
          `Byte ${byte} = 0x${hex(bytes[byte], 2)}: weight ${pair[0]} in the low nibble (${codes[pair[0]]}), weight ${pair[1]} in the high nibble (${codes[pair[1]]}).`,
          `Byte ${byte} = 0x${hex(bytes[byte], 2)}: trọng số ${pair[0]} ở nibble thấp (${codes[pair[0]]}), trọng số ${pair[1]} ở nibble cao (${codes[pair[1]]}).`,
        );
  return (
    <Bench
      title={tr('Where each weight lives', 'Mỗi trọng số nằm ở đâu')}
      task={<Translated vi="Bấm vào trọng số 5 trong mỗi bố cục, rồi bấm vào một byte. Bố cục nào giữ các trọng số kề nhau ở cạnh nhau?">Click weight 5 in each layout, then click a byte. Which layouts keep neighbours together?</Translated>}
      foot={
        <Translated
          vi={
            <>
              Mã lấy từ khối có seed cố định của trình chạy từng bước ở X2.2: quy tắc của Q4_0 cho Q4_0, quy tắc affine đã mã hóa của
              MLX cho MLX, và quy tắc của Q4_1 trên hai khối cho phần Q4_K (cách Q4_K tự chọn hệ số tỉ lệ không được mô phỏng; chỉ có
              thứ tự của nó). Các byte MLX giả định máy little-endian. Thứ tự: {`${FACTS}`} D1, D2, D5.
            </>
          }
        >
          <>
            Codes from the seeded block of the X2.2 stepper: Q4_0's rule for Q4_0, MLX's coded affine rule for MLX, and
            Q4_1's rule on two blocks for the Q4_K chunk (Q4_K's own choice of scales is not modelled; only its order is).
            MLX bytes assume a little-endian machine. Orders: {`${FACTS}`} D1, D2, D5.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Bố cục">Layout</Translated>
          <span className="seg" role="group" aria-label={tr('layout', 'bố cục')}>
            {(['q4_0', 'mlx4', 'q4_k'] as Layout[]).map((l) => (
              <button key={l} aria-pressed={layout === l} onClick={() => setLayout(l)}>{l === 'q4_0' ? tr('llama.cpp Q4_0', 'llama.cpp Q4_0') : l === 'mlx4' ? tr('MLX 4-bit', 'MLX 4 bit') : tr('llama.cpp Q4_K chunk', 'phần Q4_K của llama.cpp')}</button>
            ))}
          </span>
        </label>
        <span className="muted">{tr(LAYOUT_LABEL[layout], layout === 'q4_0' ? 'llama.cpp Q4_0 (32 trọng số, 16 byte mã)' : layout === 'mlx4' ? 'MLX 4 bit (32 trọng số, bốn word uint32)' : 'llama.cpp Q4_K (một phần 64 trọng số, 32 byte)')}</span>
      </div>
      <div className="mx-label"><Translated vi="trọng số (bấm vào một trọng số)">weights (click one)</Translated></div>
      <div data-part="weights" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(2.4rem, 1fr))', gap: '3px', marginBottom: '0.9rem' }}>
        {codes.map((c, j) => {
          const on = pair.includes(j);
          return (
            <button key={j} data-w={j} aria-pressed={on} onClick={() => setSel({ kind: 'w', j })}
              style={{ padding: '0.15rem 0', fontVariantNumeric: 'tabular-nums', lineHeight: 1.15 }}>
              <span style={{ fontSize: '0.68rem', opacity: 0.75 }}>w{j}</span>
              <br />
              {c}
            </button>
          );
        })}
      </div>
      <div className="mx-label t-global"><Translated vi="các byte trong bộ nhớ (bấm vào một byte)">bytes in memory (click one)</Translated></div>
      <div data-part="bytes" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(3.3rem, 1fr))', gap: '3px' }}>
        {bytes.map((b, i) => {
          const on = i === byte;
          const [lo, hi] = weightsInByte(layout, i);
          return (
            <button key={i} data-b={i} aria-pressed={on} onClick={() => setSel({ kind: 'b', b: i })}
              style={{ padding: '0.15rem 0', fontVariantNumeric: 'tabular-nums', lineHeight: 1.15, borderColor: 'var(--global)' }}>
              <span style={{ fontSize: '0.68rem', opacity: 0.75 }}>{layout === 'mlx4' && i % 4 === 0 ? `w${i / 4}·` : ''}b{i}</span>
              <br />
              {hex(b, 2)}
              <br />
              <span style={{ fontSize: '0.66rem', opacity: 0.75 }}>{hi}|{lo}</span>
            </button>
          );
        })}
      </div>
      <div className="narrate" role="status" data-part="readout" style={{ marginTop: '0.9rem' }}>{readout}</div>
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: unpack one word
function BitRow({ word, width, fields }: { word: number; width: 16 | 32; fields: { lo: number; hi: number; weight: number }[] }) {
  return (
    <div data-part="bits" style={{ display: 'grid', gridTemplateColumns: `repeat(${width}, minmax(0, 1fr))`, gap: '1px', maxWidth: width === 32 ? '46rem' : '26rem' }}>
      {Array.from({ length: width }, (_, k) => {
        const bit = width - 1 - k;
        const f = fields.find((x) => bit >= x.lo && bit <= x.hi);
        const v = (word >>> bit) & 1;
        return (
          <span key={bit} data-bit={bit} title={tr(`bit ${bit}`, `bit ${bit}`)}
            style={{
              textAlign: 'center', fontVariantNumeric: 'tabular-nums', fontSize: '0.74rem', padding: '0.18rem 0',
              background: f ? 'var(--reg-wash)' : 'var(--cell)', borderStyle: 'solid', borderColor: f ? 'var(--reg)' : 'var(--rule)',
              borderWidth: `1px 1px 1px ${bit % 4 === 3 ? 2 : 1}px`,
            }}>
            {v}
          </span>
        );
      })}
    </div>
  );
}

function Unpack() {
  const [variant, setVariant] = useState<UnpackVariant>('cuda32');
  const [wi, setWi] = useState(1);
  const [k, setK] = useState(0);
  const qs = useMemo(() => quantizeQ4_0(SCENE_BLOCK).bytes.slice(2), []);
  const y = useMemo(() => sceneActivations(), []);
  const words = variant === 'cuda32' ? 4 : 8;
  const locale = useLocale();
  const steps = useMemo(() => unpackTrace(variant, qs, Math.min(wi, words - 1), y), [variant, qs, wi, words, y, locale]);
  const st = steps[Math.min(k, steps.length - 1)];
  const pick = (v: UnpackVariant) => { setVariant(v); setK(0); setWi(v === 'cuda32' ? 1 : 2); };
  return (
    <Bench
      title={tr('Unpack one word', 'Giải nén một word')}
      task={<Translated vi="Chạy từng bước cả hai biến thể trên word chứa các trọng số 5 và 21. Biến thể nào không bao giờ dịch bit?">Step through both variants on the word that holds weights 5 and 21. Which one never shifts?</Translated>}
      foot={
        <Translated
          vi={
            <>
              Khối là khối của trình chạy từng bước, lượng tử hóa theo quy tắc tham chiếu của Q4_0; activation là các số thực có seed
              cố định. Biến thể CUDA: <code>get_int_b2</code> của llama.cpp, <code>&amp; 0x0F0F0F0F</code> và <code>&gt;&gt; 4</code>{' '}
              (D3, A5). Biến thể Metal: <code>block_q_n_dot_y</code> của llama.cpp lọc bằng mặt nạ với activation được chia trước
              (D4); MLX và PyTorch MPS cũng làm như vậy (D5, D6). Các bit được tô sáng: thứ mà bước này tách ra.
            </>
          }
        >
          <>
            The block is the stepper's, quantised by Q4_0's reference rule; activations are seeded floats. CUDA variant:
            llama.cpp's <code>get_int_b2</code>, <code>&amp; 0x0F0F0F0F</code> and <code>&gt;&gt; 4</code> (D3, A5). Metal
            variant: llama.cpp's <code>block_q_n_dot_y</code> masks with activations pre-divided (D4); MLX and PyTorch MPS do
            the same (D5, D6). Highlighted bits: what this step singles out.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Cách viết kernel">Kernel idiom</Translated>
          <span className="seg" role="group" aria-label={tr('variant', 'biến thể')}>
            <button aria-pressed={variant === 'cuda32'} onClick={() => pick('cuda32')}><Translated vi="CUDA: word 32 bit">CUDA: 32-bit word</Translated></button>
            <button aria-pressed={variant === 'metal16'} onClick={() => pick('metal16')}><Translated vi="Metal: word 16 bit">Metal: 16-bit word</Translated></button>
          </span>
        </label>
        <label className="field">
          <Translated vi="Word">Word</Translated>
          <select aria-label={tr('word', 'word')} value={Math.min(wi, words - 1)} onChange={(e) => { setWi(+e.target.value); setK(0); }}>
            {Array.from({ length: words }, (_, i) => (
              <option key={i} value={i}>{variant === 'cuda32' ? tr(`${i}: bytes ${4 * i}–${4 * i + 3}`, `${i}: các byte ${4 * i}–${4 * i + 3}`) : tr(`${i}: bytes ${2 * i}–${2 * i + 1}`, `${i}: các byte ${2 * i}–${2 * i + 1}`)}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setK(0)} disabled={k === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setK(k - 1)} disabled={k === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setK(k + 1)} disabled={k >= steps.length - 1}><Translated vi="Bước tiếp">Step</Translated></button>
        <span className="muted" aria-live="polite"><Translated vi={<>bước {k + 1} trên {steps.length}: {st.op}</>}>step {k + 1} of {steps.length}: {st.op}</Translated></span>
      </div>
      <div className="narrate" role="status" data-op={st.op} style={{ marginBottom: '0.8rem' }}>{st.text}</div>
      <BitRow word={st.word} width={st.width} fields={st.fields} />
      {st.out.length ? (
        <table className="data" data-part="out" style={{ marginTop: '0.8rem', maxWidth: '34rem' }}>
          <thead><tr><th><Translated vi="trọng số">weight</Translated></th><th className="num"><Translated vi="giá trị">value</Translated></th><th><Translated vi="từ">from</Translated></th></tr></thead>
          <tbody>
            {st.out.map((o) => (
              <tr key={o.weight} data-weight={o.weight} data-value={o.value}>
                <td>{o.weight}</td>
                <td className="num t-reg">{Number.isInteger(o.value) ? o.value : o.value.toPrecision(5)}</td>
                <td className="muted">{o.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: blocks end to end
const WIDTHS = [1, 2, 4, 16];
function Alignment() {
  const [i, setI] = useState(0);
  const [width, setWidth] = useState(4);
  const blocks = 8;
  const B = 4.6; // px per byte
  const x0 = 6;
  const W = x0 * 2 + blocks * 18 * B;
  const a = blockAddress(i);
  const loads = qsLoads(i, width);
  const bad = loads.filter((l) => !l.ok).length;
  const okBlocks = Array.from({ length: blocks }, (_, k) => qsLoads(k, width).every((l) => l.ok));
  return (
    <Bench
      title={tr('Blocks end to end', 'Các khối nối tiếp nhau')}
      task={<Translated vi="Kéo qua các khối với độ rộng lần nạp 4 byte, rồi thử 2 và 16.">Drag through the blocks at a 4-byte load width, then try 2 and 16.</Translated>}
      foot={
        <Translated
          vi={
            <>
              Một mảng các khối Q4_0 18 byte bắt đầu từ một địa chỉ căn chỉnh 16 byte: một d 2 byte (tô đặc), rồi 16 byte mã. Vạch
              chia mỗi 4 byte. Một lần nạp w byte được căn chỉnh khi địa chỉ của nó là bội số của w (A5: <code>char4</code> và{' '}
              <code>int</code> căn chỉnh 4 byte; C5: trên Metal cũng vậy). Chỉ có địa chỉ; không đo thời gian.
            </>
          }
        >
          <>
            An array of 18-byte Q4_0 blocks from a 16-byte aligned start: a 2-byte d (solid), then 16 bytes of quants.
            Ticks every 4 bytes. A load of w bytes is aligned when its address is a multiple of w (A5: <code>char4</code> and{' '}
            <code>int</code> are 4-byte aligned; C5: the same on Metal). Addresses only; no timing.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi={<>Khối {i}</>}>Block {i}</Translated>
          <input type="range" aria-label={tr('block index', 'chỉ số khối')} min="0" max={blocks - 1} value={i} onChange={(e) => setI(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Độ rộng lần nạp (byte)">Load width (bytes)</Translated>
          <span className="seg" role="group" aria-label={tr('load width', 'độ rộng lần nạp')}>
            {WIDTHS.map((w) => <button key={w} aria-pressed={width === w} onClick={() => setWidth(w)}>{w}</button>)}
          </span>
        </label>
      </div>
      <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${W} 92`} role="img" aria-label={tr('Byte strip of eight blocks', 'Dải byte của tám khối')} style={{ width: '100%', minWidth: 600, display: 'block' }}>
        {Array.from({ length: blocks * 18 / 4 + 1 }, (_, k) => (
          <line key={k} x1={x0 + 4 * k * B} x2={x0 + 4 * k * B} y1={8} y2={44} stroke="var(--rule)" />
        ))}
        {Array.from({ length: blocks }, (_, k) => {
          const b = blockAddress(k);
          return (
            <g key={k} opacity={k === i ? 1 : 0.45}>
              <rect x={x0 + b.d * B} y={14} width={2 * B} height={24} fill="var(--global)" />
              <rect x={x0 + b.qs * B} y={14} width={16 * B} height={24} fill="var(--global-wash)" stroke="var(--global)" strokeWidth={0.8} />
              <text x={x0 + (b.start + 9) * B} y={30} fontSize="9" textAnchor="middle">{k}</text>
            </g>
          );
        })}
        {loads.map((l, k) => (
          <rect key={k} data-load={l.addr} data-ok={l.ok ? 'yes' : 'no'} x={x0 + l.addr * B + 0.5} y={52} width={width * B - 1} height={14}
            fill={l.ok ? 'var(--ok-wash)' : 'var(--fail-wash)'} stroke={l.ok ? 'var(--ok)' : 'var(--fail)'} />
        ))}
        <text x={x0} y={84} fontSize="10"><Translated vi={<>các lần nạp mã của khối {i}, mỗi lần {width} byte</>}>loads of block {i}'s quants, {width} bytes each</Translated></text>
      </svg>
      </div>
      <div className={'narrate' + (bad ? ' bad' : '')} role="status" data-part="align" style={{ marginTop: '0.6rem' }}>
        <Translated
          vi={
            <>
              Khối {i}: d tại byte {a.d} ({isAligned(a.d, 2) ? 'căn chỉnh cho lần nạp 2 byte' : 'lệch căn chỉnh'}), mã tại byte {a.qs} = {a.qs % 4} mod 4.{' '}
              {bad ? `${bad} trên ${loads.length} lần nạp ${width} byte bị lệch căn chỉnh.` : `Cả ${loads.length} lần nạp ${width} byte đều được căn chỉnh.`}{' '}
              Ở độ rộng này, mã của các khối {okBlocks.flatMap((o, k) => (o ? [k] : [])).join(', ') || 'không khối nào'} trong số 0 đến {blocks - 1} đọc được trọn vẹn.
            </>
          }
        >
          Block {i}: d at byte {a.d} ({isAligned(a.d, 2) ? 'aligned for a 2-byte load' : 'misaligned'}), quants at byte {a.qs} = {a.qs % 4} mod 4.{' '}
          {bad ? `${bad} of ${loads.length} ${width}-byte loads are misaligned.` : `All ${loads.length} ${width}-byte loads are aligned.`}{' '}
          At this width the quants of blocks {okBlocks.flatMap((o, k) => (o ? [k] : [])).join(', ') || 'none'} of 0 to {blocks - 1} read cleanly.
        </Translated>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function SX23() {
  const q = questionsSX23;
  const src = useLabFile(SNIPPET);
  return (
    <>
      <Translated
        vi={
          <p>
            Một mã 4 bit là nửa byte, và không lệnh nạp nào đọc nửa byte. Vì vậy mọi kernel lượng tử hóa đều phải giải nén: nó nạp
            các byte hoặc word chứa nhiều mã rồi tách chúng ra bằng mặt nạ và phép dịch bit. Mỗi mã nằm ở đâu trong các byte đó là
            một lựa chọn, và ba bố cục bạn hay gặp nhất chọn theo ba cách khác nhau. Mục này nói về lý do, và về cái bẫy duy nhất
            khi đọc các khối của llama.cpp.
          </p>
        }
      >
        <p>
          A 4-bit code is half a byte, and no load instruction reads half a byte. So every quantised kernel unpacks: it
          loads bytes or words that hold several codes and pulls them apart with masks and shifts. Where each code sits in
          those bytes is a choice, and the three layouts you are most likely to read make three different ones. This
          section is about why, and about the one trap in reading llama.cpp's blocks.
        </p>
      </Translated>
      <Source section="A.5" pages="592" />
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Sách chỉ nhắc tên dữ liệu 4 bit (phụ lục A, §A.5, p. 592) và không hơn. Các thứ tự, căn chỉnh và cách viết ở đây lấy
              từ mã nguồn của các dự án và tài liệu tham khảo của các nhà cung cấp, đã kiểm tra trong lượt kiểm tra dữ kiện.
            </p>
          }
        >
          <p>
            The book names 4-bit data (appendix A, §A.5, p. 592) and nothing more. The orders, alignments and idioms here
            are from the projects' source and the vendors' references, checked in the facts pass.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            CUDA hay HIP C++ đều không có kiểu số nguyên 4 bit: bạn lưu mã trong <code>uint8_t</code> hoặc{' '}
            <code>uint32_t</code> và tự đóng gói. Metal gần đây đã thêm các kiểu 4 bit đóng gói, sẽ được nói ở cuối trang này.
          </p>
        }
      >
        <p>
          There is no 4-bit integer type in CUDA or HIP C++: you store codes in <code>uint8_t</code> or{' '}
          <code>uint32_t</code> and pack them by hand. Metal has recently added packed 4-bit types, which come at the end
          of this page.
        </p>
      </Translated>
      <Beyond title={tr('Below C++', 'Bên dưới C++')}>
        <Translated
          vi={
            <p>
              PTX có <code>.s4</code> và <code>.u4</code>, nhưng chỉ bên trong các lệnh ma trận (<code>mma</code> từ sm_75 và
              sm_80), dạng <code>.s8.s4</code> của <code>ldmatrix</code> (PTX 9.4; sm_90a và các họ sm_100, sm_110 và sm_120) và
              phép chuyển đổi đóng gói <code>cvt.pack</code>. Đã kiểm tra {CHECKED('A4')}.
            </p>
          }
        >
          <p>
            PTX does have <code>.s4</code> and <code>.u4</code>, but only inside matrix instructions (<code>mma</code> from
            sm_75 and sm_80), <code>ldmatrix</code>'s <code>.s8.s4</code> form (PTX 9.4; sm_90a and the sm_100, sm_110 and sm_120 families) and the
            packing conversion <code>cvt.pack</code>. Checked {CHECKED('A4')}.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Ba thứ tự">Three orders</Translated></h2>
      <Translated
        vi={
          <p>
            <b>llama.cpp Q4_0</b>: byte j trong 16 byte mã chứa trọng số j ở nibble thấp và trọng số j + 16 ở nibble cao. Hai
            mã trong một byte không kề nhau. <b>MLX</b>: 8 mã mỗi <code>uint32</code>, mã đầu tiên ở 4 bit thấp nhất; các trọng
            số kề nhau vẫn kề nhau. <b>llama.cpp Q4_K</b>: trong mỗi phần 64 trọng số, các nibble thấp của 32 byte là 32 trọng
            số đầu và các nibble cao là 32 trọng số tiếp theo: ý tưởng của Q4_0 ở kích thước lớn hơn.
          </p>
        }
      >
        <p>
          <b>llama.cpp Q4_0</b>: byte j of the 16 quant bytes holds weight j in its low nibble and weight j + 16 in its
          high nibble. The two codes in a byte are not neighbours. <b>MLX</b>: 8 codes per <code>uint32</code>, the first
          in the 4 least significant bits; neighbours stay neighbours. <b>llama.cpp Q4_K</b>: within each 64-weight chunk,
          the low nibbles of 32 bytes are the first 32 weights and the high nibbles the next 32: Q4_0's idea at a larger
          size.
        </p>
      </Translated>
      <Predict
        id="x2/X2.3/p-byte5"
        prompt={<Translated vi={<>Trước khi dùng trình khám phá: trong một khối Q4_0, trọng số 5 dùng chung byte với trọng số nào khác?</>}><>Before you use the explorer: in a Q4_0 block, weight 5 shares its byte with which other weight?</></Translated>}
        unit={tr('(weight index)', '(chỉ số trọng số)')}
      >
        {(v) => (
          <p data-part="p-byte5">
            <Translated
              vi={
                <>
                  Trọng số {weightsInByte('q4_0', 5)[1]}: byte 5 chứa các trọng số {weightsInByte('q4_0', 5).join(' và ')}.{' '}
                  {v === weightsInByte('q4_0', 5)[1] ? 'Đúng như bạn nói.' : v === 4 || v === 6 ? 'Đó là thứ tự của MLX, nơi các trọng số kề nhau dùng chung một byte.' : ''}
                </>
              }
            >
              Weight {weightsInByte('q4_0', 5)[1]}: byte 5 holds weights {weightsInByte('q4_0', 5).join(' and ')}.{' '}
              {v === weightsInByte('q4_0', 5)[1] ? 'As you said.' : v === 4 || v === 6 ? 'That is MLX’s order, where neighbours share a byte.' : ''}
            </Translated>
          </p>
        )}
      </Predict>
      <WhereLives />
      <Ask q={q['x2/X2.3/q-byte9']} label={tr('Work it out', 'Tự tính ra')} />
      <Ask q={q['x2/X2.3/q-mlx13']} />

      <h2><Translated vi="Vì sao là j và j + 16">Why j and j + 16</Translated></h2>
      <Translated
        vi={
          <p>
            Kernel decode CUDA của llama.cpp đọc một word mã 32 bit, bốn byte, và tính{' '}
            <code>v &amp; 0x0F0F0F0F</code>. Mỗi byte của kết quả là một nibble thấp: mã của các trọng số j đến j + 3, theo thứ
            tự. Chúng gặp bốn activation int8 liên tiếp trong một tích vô hướng byte bốn chiều. Sau đó{' '}
            <code>(v &gt;&gt; 4) &amp; 0x0F0F0F0F</code> cho các trọng số j + 16 đến j + 19, gặp các activation xa hơn 16 vị trí.
            Một mặt nạ, một lần dịch bit, không xáo trộn byte: bố cục phục vụ kernel. Mục{' '}
            <a href={href('x2', 'X2.5')}>X2.5</a> đi tiếp trên đường số nguyên đó.
          </p>
        }
      >
        <p>
          llama.cpp's CUDA decode kernel reads a 32-bit word of quants, four bytes, and computes{' '}
          <code>v &amp; 0x0F0F0F0F</code>. Each byte of the result is one low nibble: the codes of weights j to j + 3, in
          order. Those meet four consecutive int8 activations in one four-way byte dot product. Then{' '}
          <code>(v &gt;&gt; 4) &amp; 0x0F0F0F0F</code> gives weights j + 16 to j + 19, which meet the activations 16 further
          on. One mask, one shift, no shuffling of bytes: the layout serves the kernel. Section{' '}
          <a href={href('x2', 'X2.5')}>X2.5</a> takes that integer path further.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Cách viết còn lại, được dùng trong các kernel Metal của llama.cpp, MLX và backend MPS của PyTorch, không bao giờ dịch
            bit. Nó lọc một word 16 bit theo bốn cách: <code>0x000F</code>, <code>0x0F00</code>, <code>0x00F0</code>,{' '}
            <code>0xF000</code>. Một nibble đã lọc mà để nguyên chỗ là mã nhân với 1, 256, 16 hoặc 4096, nên kernel chia các
            activation tương ứng cho cùng các lũy thừa của hai đó khi nạp chúng. Chia một số thực cho một lũy thừa của hai là
            chính xác (trừ khi bị underflow), nên mỗi tích đúng bằng mã × activation. Metal còn có <code>extract_bits</code>, có
            mở rộng dấu cho các kiểu có dấu; các kernel này không cần đến nó.
          </p>
        }
      >
        <p>
          The other idiom, used by llama.cpp's Metal kernels, MLX and PyTorch's MPS backend, never shifts. It masks a
          16-bit word four ways: <code>0x000F</code>, <code>0x0F00</code>, <code>0x00F0</code>, <code>0xF000</code>. A masked
          nibble left in place is the code times 1, 256, 16 or 4096, so the kernel divides the matching activations by
          the same powers of two when it loads them. Dividing a float by a power of two is exact (unless it underflows), so
          each product is exactly code × activation. Metal also has <code>extract_bits</code>, which sign-extends for signed
          types; these kernels do not need it.
        </p>
      </Translated>
      <Today checked={CHECKED('D3, D4, D5, D6, C1, E4')}>
        <Translated
          vi={
            <p>
              llama.cpp CUDA: <code>vi0 = (v[i] &gt;&gt; 0) &amp; 0x0F0F0F0F</code>, <code>vi1 = (v[i] &gt;&gt; 4) &amp;
              0x0F0F0F0F</code>. llama.cpp Metal: "we assume that the yl's have been multiplied with the appropriate scale
              factor that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)" (các yl được giả định đã nhân với hệ
              số tỉ lệ bù cho các phép dịch bit bị bỏ qua). <code>qdot</code> của MLX và <code>int4pack_mm</code> của PyTorch trên
              MPS cũng làm như vậy.
            </p>
          }
        >
          <p>
            llama.cpp CUDA: <code>vi0 = (v[i] &gt;&gt; 0) &amp; 0x0F0F0F0F</code>, <code>vi1 = (v[i] &gt;&gt; 4) &amp;
            0x0F0F0F0F</code>. llama.cpp Metal: "we assume that the yl's have been multiplied with the appropriate scale
            factor that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)". MLX's <code>qdot</code> and
            PyTorch's <code>int4pack_mm</code> on MPS do the same.
          </p>
        </Translated>
      </Today>
      <Unpack />
      <Ask q={q['x2/X2.3/q-why-j16']} />

      <h2><Translated vi="Cái bẫy căn chỉnh">The alignment trap</Translated></h2>
      <Translated
        vi={
          <p>
            Một khối Q4_0 là một d FP16 2 byte theo sau là 16 byte mã: 18 byte, và chỉ căn chỉnh 2 byte. Đặt các khối nối tiếp
            nhau thì mã của khối i bắt đầu tại 18i + 2. Với i chẵn, đó là 2 mod 4, nên đọc chúng qua{' '}
            <code>int*</code> hoặc <code>char4*</code> (cả hai đều là kiểu căn chỉnh 4 byte) là một truy cập lệch căn chỉnh, tức
            hành vi không xác định trong C++ trên mọi backend. llama.cpp đọc mã thành hai nửa 16 bit trên CUDA và HIP (hàm{' '}
            <code>get_int_b2</code> của nó) và thành các word <code>uint16_t</code> trên Metal. MLX tránh hẳn vấn đề: hệ số tỉ
            lệ, bias và mã nằm trong các mảng riêng, nên mã luôn căn chỉnh 4 byte.
          </p>
        }
      >
        <p>
          A Q4_0 block is a 2-byte FP16 d followed by 16 bytes of quants: 18 bytes, and only 2-byte aligned. Lay blocks end
          to end and block i's quants start at 18i + 2. For even i that is 2 mod 4, so reading them through{' '}
          <code>int*</code> or <code>char4*</code> (both 4-byte aligned types) is a misaligned access, which is undefined
          behaviour in C++ on every backend. llama.cpp reads the quants as two 16-bit halves on CUDA and HIP (its{' '}
          <code>get_int_b2</code>) and as <code>uint16_t</code> words on Metal. MLX sidesteps the problem: scales, biases and
          codes live in separate arrays, so the codes stay 4-byte aligned.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ có địa chỉ: trình khám phá đánh dấu lần nạp nào được căn chỉnh, không phải một lần nạp lệch căn chỉnh tốn bao
              nhiêu hay làm gì trên một GPU cụ thể, điều mà không nhà cung cấp nào ở đây tài liệu hóa thành một hành vi duy nhất.
            </p>
          }
        >
          <p>
            Addresses only: the explorer marks which loads are aligned, not what a misaligned load costs or does on a given
            GPU, which no vendor here documents as a single behaviour.
          </p>
        </Translated>
      </Simplification>
      <Alignment />
      <Ask q={q['x2/X2.3/q-align']} />
      {src ? (
        <Code
          tabs={[{ label: 'q4_0_ref.h', code: src, status: tr('host C++; compiled and compared bit for bit with the model by a unit test; not a GPU kernel', 'C++ phía host; được biên dịch và so sánh từng bit với mô hình bằng một unit test; không phải kernel GPU'), highlight: [75, 76, 84, 85, 86, 87, 88, 102] }]}
          caption={
            <Translated
              vi={
                <>
                  Cùng bản tham chiếu CPU như ở mục X2.2, với phần giải nén được đánh dấu: mặt nạ và phép dịch bit của bộ giải lượng
                  tử hóa, word được dựng từ hai lần đọc 16 bit, và tích vô hướng của khối với số 8 được tách ra. Nạp từ{' '}
                  <code>labs/x2/snippets/q4_0_ref.h</code>.
                </>
              }
            >
              <>
                The same CPU reference as in section X2.2, with the unpacking marked: the dequantiser's mask and shift, the
                word built from two 16-bit reads, and the block's dot product with the 8 folded out. Loaded from{' '}
                <code>labs/x2/snippets/q4_0_ref.h</code>.
              </>
            </Translated>
          }
        />
      ) : null}

      <h2><Translated vi="Một khối, từ đầu đến cuối">One block, end to end</Translated></h2>
      <Translated
        vi={
          <p>
            Hoạt ảnh theo khối 32 trọng số của trình chạy từng bước, từ số thực sang mã, vào byte, qua bộ nhớ rồi ra khỏi một word
            16 bit, và kiểm tra tích vô hướng của khối theo cách các kernel Metal tính nó.
          </p>
        }
      >
        <p>
          The animation follows the stepper's block of 32 weights from floats to codes, into bytes, through memory and back
          out of one 16-bit word, and checks the block's dot product the way the Metal kernels compute it.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Cảnh này chỉ vẽ địa chỉ và giá trị. Nó chạy quy tắc tham chiếu từ mã CPU của llama.cpp; các kernel GPU đọc cùng các
              byte đó nhưng không được mô phỏng, và không có gì được đo thời gian.
            </p>
          }
        >
          <p>
            The scene draws addresses and values only. It runs the reference rule from llama.cpp's CPU code; the GPU
            kernels read the same bytes but are not simulated, and nothing is timed.
          </p>
        </Translated>
      </Simplification>
      <PackedBlockScene />

      <h2><Translated vi="Mỗi backend cho bạn gì với dữ liệu 4 bit">What each backend gives you for 4-bit data</Translated></h2>
      <Translated
        vi={
          <p>
            Trên CUDA và HIP: byte, word và mặt nạ của chính bạn. Trên Metal cũng vậy, và gần đây có thêm chút ít. MSL định nghĩa
            các định dạng số nguyên 4 bit đóng gói có chuyển đổi sang dấu phẩy động, và MPSGraph có một kiểu dữ liệu 4 bit với
            phép giải lượng tử hóa theo khối. Các lab không dùng cái nào, vì những lý do trong câu hỏi bên dưới.
          </p>
        }
      >
        <p>
          On CUDA and HIP: bytes, words and your own masks. On Metal, the same, and since recently a little more. MSL
          defines packed 4-bit integer formats with conversion to floating point, and MPSGraph has a 4-bit data type with a
          blockwise dequantise. The labs do not use either, for the reasons in the question below.
        </p>
      </Translated>
      <Today checked={CHECKED('C2, C4')}>
        <Translated
          vi={
            <p>
              MSL: <code>int4b_format</code> và <code>uint4b_format</code> từ Metal 4.0 với SDK 26.4,{' '}
              <code>unpack</code> sang float, half hoặc bfloat từ Metal 4.1. Nibble nào của một byte là phần tử 0, và họ GPU nào hỗ
              trợ các kiểu này, đều không được tài liệu hóa; cả MLX lẫn llama.cpp đều không dùng chúng (theo một lần tìm trong mã).
              MPSGraph: <code>MPSDataTypeInt4</code> và <code>dequantizeTensor</code> với hệ số tỉ lệ số thực theo khối từ
              macOS 15.
            </p>
          }
        >
          <p>
            MSL: <code>int4b_format</code> and <code>uint4b_format</code> from Metal 4.0 with SDK 26.4,{' '}
            <code>unpack</code> to float, half or bfloat from Metal 4.1. Which nibble of a byte is element 0, and which GPU
            families support the types, are not documented; neither MLX nor llama.cpp uses them (a code search).
            MPSGraph: <code>MPSDataTypeInt4</code> and <code>dequantizeTensor</code> with blockwise float scales from
            macOS 15.
          </p>
        </Translated>
      </Today>
      <Ask q={q['x2/X2.3/q-int4b']} />
      <Translated
        vi={
          <p>
            Một cầu nối để đọc chính llama.cpp. Các byte của một tensor Q4_0 trong tệp GGUF chính là các khối 18 byte này, và gói{' '}
            <code>gguf-py</code> của llama.cpp lượng tử hóa theo cùng quy tắc bằng NumPy, nên nó là một nguồn đối chiếu dùng được
            khi bạn viết test của riêng mình. Lab X2.2 (trên <a href={href('x2', 'labs')}>trang lab</a>) đọc các khối này trên GPU.
          </p>
        }
      >
        <p>
          One bridge to reading llama.cpp itself. The bytes of a Q4_0 tensor in a GGUF file are exactly these 18-byte blocks, and
          llama.cpp's <code>gguf-py</code> package quantises with the same rule in NumPy, so it is a usable oracle when you
          write your own tests. Lab X2.2 (on the <a href={href('x2', 'labs')}>labs page</a>) reads these blocks on the GPU.
        </p>
      </Translated>
    </>
  );
}
