import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  PIXELS,
  PRESETS,
  SIDE,
  WIDTHS,
  binCount,
  binOf,
  binRange,
  flatIndex,
  histogram,
  hottest,
  preset,
  type PresetId,
} from '../../sim/histogram';

const SEQ_TIERS: Record<string, Tier> = { image: 'global', bins: 'global', i: 'reg', b: 'reg' };
const grey = (v: number) => `rgb(${v},${v},${v})`;
const range = (b: number, w: number) => {
  const [lo, hi] = binRange(b, w);
  return lo === hi ? String(lo) : `${lo}–${hi}`;
};

// ---------------------------------------------------------------- the explorer
function BarChart({ counts, width, mark }: { counts: number[]; width: number; mark: number }) {
  const W = 540;
  const H = 230;
  const pad = { l: 34, r: 8, t: 10, b: 40 };
  const n = counts.length;
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const bw = pw / n;
  const Y = (c: number) => pad.t + ph - (c / PIXELS) * ph;
  const ticks = [0, 16, 32, 48, 64];
  const labels = n <= 16 ? counts.map((_, b) => b) : [0, 64, 128, 192, 255];
  const busiest = counts.indexOf(Math.max(...counts));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Histogram of the image', 'Histogram của ảnh')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(t)} y2={Y(t)} stroke="var(--rule-soft)" />
          <text x={pad.l - 5} y={Y(t) + 4} fontSize="10.5" textAnchor="end">{t}</text>
        </g>
      ))}
      {counts.map((c, b) => (
        <rect
          key={b}
          data-bin={b}
          data-count={c}
          x={pad.l + b * bw + (n <= 16 ? 2 : 0)}
          y={Y(c)}
          width={Math.max(2.5, bw - (n <= 16 ? 4 : 0))}
          height={Y(0) - Y(c)}
          fill={b === mark ? 'var(--ink)' : 'var(--ink-3)'}
        />
      ))}
      {counts.map((c, b) =>
        c > 0 && (n <= 16 || b === mark || b === busiest) ? (
          <text key={'c' + b} x={Math.min(pad.l + (b + 0.5) * bw, W - pad.r - 4)} y={Y(c) - 4} fontSize="10.5"
            textAnchor={n <= 16 ? 'middle' : 'end'} fontWeight={b === mark ? 650 : 400}>
            {n <= 16 ? c : `${b}: ${c}`}
          </text>
        ) : null,
      )}
      {labels.map((b) => (
        <text
          key={'l' + b}
          x={pad.l + (b + 0.5) * bw}
          y={H - pad.b + 14}
          fontSize={n <= 8 ? 10.5 : 9}
          textAnchor="middle"
        >
          {n <= 8 ? range(b, width) : n <= 16 ? binRange(b, width)[0] : b}
        </text>
      ))}
      <text x={pad.l + pw / 2} y={H - 6} fontSize="11" textAnchor="middle">
        {n === 256 ? tr('pixel value (one bin per value)', 'giá trị điểm ảnh (mỗi giá trị một ô)') : tr(`bin (interval of ${width} values)`, `ô (khoảng gồm ${width} giá trị)`)}
      </text>
    </svg>
  );
}

function FlatStrip({ img, focus }: { img: number[]; focus: number }) {
  const C = 8;
  const lx = Math.min(Math.max(focus * C + C / 2, 30), PIXELS * C - 30);
  return (
    <svg viewBox={`0 0 ${PIXELS * C} 46`} role="img" aria-label={tr('The image as one flat array', 'Ảnh dưới dạng một mảng phẳng')} style={{ width: '100%', display: 'block' }}>
      {img.map((v, i) => (
        <rect key={i} x={i * C + 0.5} y={2} width={C - 1} height={20} fill={grey(v)} stroke={i === focus ? 'var(--ink)' : 'var(--rule)'} strokeWidth={i === focus ? 2 : 0.6} />
      ))}
      {Array.from({ length: SIDE }, (_, r) => (
        <line key={r} x1={r * SIDE * C} x2={r * SIDE * C} y1={0} y2={26} stroke="var(--ink-2)" strokeWidth={0.8} />
      ))}
      <path d={`M ${focus * C + C / 2} 24 l -4 6 h 8 z`} fill="var(--ink)" />
      <text x={lx} y={43} fontSize="13" textAnchor="middle" data-strip-index={focus}><Translated vi={<>chỉ số {focus}</>}>index {focus}</Translated></text>
    </svg>
  );
}

function PaintAndBin() {
  const [pid, setPid] = useState<PresetId>('sky');
  const [img, setImg] = useState<number[]>(() => preset('sky').values.slice());
  const [width, setWidth] = useState(64);
  const [paint, setPaint] = useState(255);
  const [focus, setFocus] = useState(0);
  const [down, setDown] = useState(false);
  const counts = useMemo(() => histogram(img, width), [img, width]);
  const hot = hottest(counts);
  const sum = counts.reduce((s, c) => s + c, 0);
  const r = Math.floor(focus / SIDE);
  const c = focus % SIDE;
  const v = img[focus];
  const b = binOf(v, width);

  useEffect(() => {
    const up = () => setDown(false);
    window.addEventListener('pointerup', up);
    return () => window.removeEventListener('pointerup', up);
  }, []);

  const set = (i: number) => setImg((im) => (im[i] === paint ? im : im.map((x, k) => (k === i ? paint : x))));
  const choose = (id: PresetId) => {
    setPid(id);
    setImg(preset(id).values.slice());
  };

  return (
    <Bench
      title={tr('Paint and bin', 'Tô và chia ô')}
      task={tr(
        'Paint some sky, then switch to one bin per value: which bar grows, and how many threads would collide on it?',
        'Tô thêm một ít bầu trời, rồi chuyển sang mỗi giá trị một ô: cột nào cao lên, và bao nhiêu luồng sẽ va nhau trên nó?',
      )}
      foot={
        <Translated
          vi={
            <>
              Các ảnh 8 × 8 của khóa học này, không phải của sách. Nhấp hoặc kéo trên ảnh để tô bằng giá trị đã chọn; trỏ vào một
              điểm ảnh để xem nó nằm ở đâu trong mảng phẳng và nó cập nhật ô nào. Thang của biểu đồ cột cố định ở 64, tức cả
              ảnh.
            </>
          }
        >
          <>
            Our own 8 × 8 images, not the book's. Click or drag on the image to paint with the chosen value; point at a
            pixel to see where it lands in the flat array and which bin it updates. The bar chart's scale is fixed at 64,
            the whole image.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Ảnh">Image</Translated>
          <span className="seg" role="group" aria-label={tr('Image', 'Ảnh')}>
            {PRESETS.map((p) => (
              <button key={p.id} aria-pressed={pid === p.id} onClick={() => choose(p.id)}>{p.label}</button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Độ rộng khoảng">Interval width</Translated>
          <span className="seg" role="group" aria-label={tr('Interval width', 'Độ rộng khoảng')}>
            {WIDTHS.map((w) => (
              <button key={w} aria-pressed={width === w} onClick={() => setWidth(w)}>
                <Translated vi={<>{w} ({binCount(w)} ô)</>}>
                  {w} ({binCount(w)} bins)
                </Translated>
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          <span>
            <Translated vi={<>Giá trị tô = <span data-paint>{paint}</span></>}>
              Paint value = <span data-paint>{paint}</span>
            </Translated>
          </span>
          <span className="row" style={{ gap: '0.5rem' }}>
            <input type="range" min="0" max="255" value={paint} aria-label={tr('Paint value', 'Giá trị tô')} onChange={(e) => setPaint(+e.target.value)} />
            <i style={{ display: 'inline-block', width: '1.4rem', height: '1.4rem', background: grey(paint), border: '1px solid var(--rule)', borderRadius: 2 }} />
          </span>
        </label>
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div>
          <div
            role="grid"
            aria-label={tr('The image', 'Ảnh')}
            data-part="image"
            style={{ display: 'grid', gridTemplateColumns: `repeat(${SIDE}, 1fr)`, gap: 2, maxWidth: 300, touchAction: 'none' }}
          >
            {img.map((val, i) => (
              <button
                key={i}
                data-index={i}
                aria-label={tr(`row ${Math.floor(i / SIDE)}, column ${i % SIDE}, value ${val}`, `hàng ${Math.floor(i / SIDE)}, cột ${i % SIDE}, giá trị ${val}`)}
                onPointerDown={(e) => {
                  e.preventDefault();
                  setDown(true);
                  setFocus(i);
                  set(i);
                }}
                onPointerEnter={() => {
                  setFocus(i);
                  if (down) set(i);
                }}
                onFocus={() => setFocus(i)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    set(i);
                  }
                }}
                style={{
                  aspectRatio: '1',
                  padding: 0,
                  minWidth: 0,
                  background: grey(val),
                  border: i === focus ? '2px solid var(--ink)' : '1px solid var(--rule)',
                  borderRadius: 2,
                }}
              />
            ))}
          </div>
          <Translated
            vi={<p className="muted" style={{ margin: '0.8rem 0 0.2rem' }}>Cũng 64 byte đó theo cách kernel nhìn thấy, hết hàng này đến hàng khác:</p>}
          >
            <p className="muted" style={{ margin: '0.8rem 0 0.2rem' }}>The same 64 bytes as the kernel sees them, row after row:</p>
          </Translated>
          <FlatStrip img={img} focus={focus} />
        </div>
        <div>
          <div className="narrate" role="status" data-part="readout" style={{ marginBottom: '0.8rem' }}>
            <Translated
              vi={
                <>
                  Điểm ảnh ở hàng {r}, cột {c}: chỉ số phẳng {r} × {SIDE} + {c} = <b data-flat>{flatIndex(r, c, SIDE)}</b>. Giá trị{' '}
                  <b>{v}</b>, nên nó cập nhật ô <b data-bin-of>{b}</b> = ⌊{v} / {width}⌋
                  {width > 1 ? <>, ô chứa các giá trị {range(b, width)}</> : null}.
                </>
              }
            >
              Pixel at row {r}, column {c}: flat index {r} × {SIDE} + {c} = <b data-flat>{flatIndex(r, c, SIDE)}</b>. Value{' '}
              <b>{v}</b>, so it updates bin <b data-bin-of>{b}</b> = ⌊{v} / {width}⌋
              {width > 1 ? <>, which holds the values {range(b, width)}</> : null}.
            </Translated>
          </div>
          <BarChart counts={counts} width={width} mark={b} />
          <Translated
            vi={
              <p style={{ margin: '0.6rem 0 0' }} data-part="hot" data-hot-count={hot.count}>
                Các cột cộng lại được <b data-sum>{sum}</b>, mỗi điểm ảnh một lần đếm. Ô bận nhất:{' '}
                <b>{width === 1 ? `giá trị ${hot.bin}` : `${range(hot.bin, width)}`}</b>, với {hot.count} điểm ảnh. Với mỗi điểm ảnh
                một luồng, <b>{hot.count} luồng</b> sẽ cập nhật đúng ô đó.
              </p>
            }
          >
            <p style={{ margin: '0.6rem 0 0' }} data-part="hot" data-hot-count={hot.count}>
              The bars add up to <b data-sum>{sum}</b>, one count per pixel. Busiest bin:{' '}
              <b>{width === 1 ? `value ${hot.bin}` : `${range(hot.bin, width)}`}</b>, with {hot.count} pixels. With one thread
              per pixel, <b>{hot.count} threads</b> would update that one bin.
            </p>
          </Translated>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS91: Record<string, Question> = {
  'ch09/9.1/q-owner': {
    kind: 'choice',
    id: 'ch09/9.1/q-owner',
    prompt: (
      <Translated vi={<>Trường hợp nào dưới đây phá vỡ quy tắc owner-computes?</>}>
        <>Which of these breaks the owner-computes rule?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Chuyển ảnh màu sang ảnh xám, mỗi điểm ảnh một luồng.">Colour to grayscale, one thread per pixel.</Translated>, why: <Translated vi="Mỗi luồng ghi điểm ảnh đầu ra của riêng nó và không gì khác.">Each thread writes its own output pixel and nothing else.</Translated> },
      { label: <Translated vi="Phép nhân ma trận, mỗi phần tử đầu ra một luồng.">Matrix multiplication, one thread per output element.</Translated>, why: <Translated vi="Nhiều luồng đọc cùng các đầu vào, nhưng mỗi đầu ra chỉ có một người ghi.">Many threads read the same inputs, but each output has one writer.</Translated> },
      { label: <Translated vi="Với mỗi token id trong một batch, cộng 1 vào số đếm của token đó.">For every token id in a batch, add 1 to that token's count.</Translated>, correct: true },
      {
        label: <Translated vi="Một khối chuẩn hóa một hàng của softmax.">One block normalising one row of a softmax.</Translated>,
        why: (
          <Translated vi="Khối dùng chung một tổng hàng, nhưng mỗi phần tử đầu ra vẫn do đúng một luồng ghi. Tính tổng dùng chung là một phép rút gọn (reduction), chương 10.">
            The block shares a row sum, but each output element is still written by exactly one thread. Computing the shared sum is a reduction, chapter 10.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong kernel đếm, phần tử mà một luồng ghi phụ thuộc vào giá trị nó đọc: hai luồng đọc cùng một token id sẽ ghi
            cùng một số đếm. Không gì trong cấu hình khởi chạy cho bạn biết những luồng nào sẽ va nhau; chỉ dữ liệu mới cho biết.
            Đó là tính chất định nghĩa của mẫu histogram.
          </p>
        }
      >
        <p>
          In the counting kernel, which element a thread writes depends on the value it read: two threads that read the
          same token id write the same count. Nothing in the launch configuration tells you which threads will collide;
          only the data does. That is the defining property of the histogram pattern.
        </p>
      </Translated>
    ),
    source: 'ch. 9, p. 201',
  },
  'ch09/9.1/q-contention': {
    kind: 'numeric',
    id: 'ch09/9.1/q-contention',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một ảnh có 1,000,000 điểm ảnh, và 40% trong số đó là bầu trời trắng tinh. Với mỗi điểm ảnh một luồng và mỗi giá trị
              một ô, bao nhiêu luồng cập nhật ô 255?
            </>
          }
        >
          An image has 1,000,000 pixels, and 40% of them are pure white sky. With one thread per pixel and one bin per
          value, how many threads update bin 255?
        </Translated>
      </>
    ),
    answer: 400000,
    unit: 'threads',
    vi: { unit: 'luồng', source: 'phép tính của chúng tôi trên thiết lập của §9.1 (pp. 202–203)' },
    explain: (
      <Translated
        vi={
          <p>
            400,000: chiều cao của một cột là số luồng tranh chấp ô đó. Trên CPU nó chỉ là một con số đếm. Trên GPU nó là một
            hàng đợi, và mục 9.3 biến nó thành thời gian.
          </p>
        }
      >
        <p>
          400,000: the height of a bar is the number of threads that contend for that bin. On the CPU it is only a count.
          On the GPU it is a queue, and section 9.3 turns it into time.
        </p>
      </Translated>
    ),
    source: 'our arithmetic on the §9.1 setup (pp. 202–203)',
  },
  'ch09/9.1/q-cpu': {
    kind: 'choice',
    id: 'ch09/9.1/q-cpu',
    prompt: (
      <Translated vi={<>Vì sao trên CPU vòng lặp tuần tự bị giới hạn bởi bộ nhớ, chứ không phải bởi các phép tăng của nó?</>}>
        <>Why is the sequential loop limited by memory on a CPU, rather than by its increments?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Mỗi điểm ảnh cần một phép chia nguyên cho độ rộng khoảng, và phép chia nguyên thì chậm trên mọi CPU.">Each pixel needs an integer division by the interval width, and integer division is slow on every CPU.</Translated>, why: <Translated vi="Đoạn mã dùng mỗi giá trị một ô và không chia cho gì cả. Kể cả có phép chia, đó cũng chỉ là vài chu kỳ tính toán.">The listing uses one bin per value and divides by nothing. Even with a division, it is a few cycles of arithmetic.</Translated> },
      { label: <Translated vi="Nhánh của vòng lặp thường bị dự đoán sai, vì các giá trị điểm ảnh nó đọc không thể đoán trước.">The loop branch mispredicts often, because the pixel values it reads are unpredictable.</Translated>, why: <Translated vi="Vòng lặp có một nhánh, phép kiểm tra vòng lặp, và nó không phụ thuộc vào giá trị điểm ảnh.">The loop has one branch, the loop test, and it does not depend on the pixel values.</Translated> },
      { label: <Translated vi="Các ô nằm trong L1, nên các phép tăng rẻ; việc truyền ảnh vào từ DRAM quyết định nhịp độ.">The bins sit in L1, so increments are cheap; streaming the image in from DRAM sets the pace.</Translated>, correct: true },
      { label: <Translated vi="Các phép tăng trượt bộ nhớ đệm, vì các điểm ảnh liền kề thường cập nhật những ô cách xa nhau.">The increments miss in the cache, because neighbouring pixels usually update bins that are far apart.</Translated>, why: <Translated vi="256 bộ đếm 4 byte dễ dàng vừa trong bộ nhớ đệm cấp một.">256 counters of 4 bytes fit easily in a level-one cache.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lập luận của sách: ảnh được đọc một lần, tuần tự, nên mọi dòng bộ nhớ đệm được nạp đều được dùng trọn; các ô đủ nhỏ
            để nằm trong bộ nhớ đệm dữ liệu cấp một của CPU. Điều còn lại là tốc độ DRAM có thể đưa ảnh tới.
          </p>
        }
      >
        <p>
          The book's argument: the image is read once, sequentially, so every cache line fetched is fully used; the bins
          are small enough to sit in the CPU's level-one data cache. What is left is the rate at which DRAM can deliver
          the image.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.1, p. 203',
  },
};

export function S91() {
  const seq = useLabFile('ch09/snippets/histogram_sequential.cpp');
  return (
    <>
      <Translated
        vi={
          <p>
            Một histogram đếm có bao nhiêu giá trị trong một tập dữ liệu rơi vào mỗi khoảng và vẽ mỗi khoảng một cột, chiều cao
            là số đếm. Với một ảnh xám 8-bit, các giá trị là cường độ điểm ảnh từ 0 (đen) đến 255 (trắng). Cắt dải đó thành bốn
            khoảng 64 và bạn có bốn cột: bao nhiêu phần bức ảnh gần đen, xám tối, xám sáng và gần trắng.
          </p>
        }
      >
        <p>
          A histogram counts how many values in a data set fall in each interval and draws one bar per interval, its
          height the count. For an 8-bit grayscale image the values are pixel intensities from 0 (black) to 255 (white).
          Cut that range into four intervals of 64 and you get four bars: how much of the picture is near black, dark
          grey, light grey and near white.
        </p>
      </Translated>
      <Source section="9.1" pages="202" />
      <Translated
        vi={
          <p>
            Hình dạng các cột nói lên điều gì đó về dữ liệu nói chung, vì thế sách gọi nó là một đặc trưng của tập dữ liệu: một
            histogram dồn về bên phải là một bức ảnh dư sáng. Sách liệt kê các ứng dụng từ phát hiện gian lận thẻ tín dụng và
            thị giác máy tính đến nhận dạng giọng nói, gợi ý mua hàng và thiên văn học. Ở đâu cần tóm tắt nhiều dữ liệu, ở đó
            histogram nhiều khả năng là một trong những phép tính đầu tiên.
          </p>
        }
      >
        <p>
          The shape of the bars says something about the data as a whole, which is why the book calls it a feature of the
          data set: a histogram crowded at the right is an overexposed photo. The book lists uses from credit-card fraud
          detection and computer vision to speech recognition, purchase recommendations and astronomy. Wherever a lot of
          data has to be summarised, a histogram is likely to be one of the first computations.
        </p>
      </Translated>
      <Source section="9.1" pages="202–203" />

      <h2>
        <Translated vi="Tô và chia ô">Paint and bin</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trình khám phá bên dưới bắt đầu với hình một bầu trời sáng trên nền đất tối. Hãy tô lên nó, đổi độ rộng khoảng, và
            theo dõi hai thứ: con số dưới ô của mỗi điểm ảnh, và cột bận nhất.
          </p>
        }
      >
        <p>
          The explorer below starts with a picture of a bright sky over dark ground. Paint on it, change the interval
          width, and watch two things: the number under each pixel's bin, and the busiest bar.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Chỉ ảnh xám 8-bit, và 64 điểm ảnh để mọi điểm ảnh đều nhìn thấy và tô được. Một ảnh thật có hàng triệu điểm ảnh,
              điều đó chỉ làm các cột cao hơn.
            </p>
          }
        >
          <p>
            8-bit grayscale only, and 64 pixels so that every pixel can be seen and painted. A real image has millions,
            which only makes the bars taller.
          </p>
        </Translated>
      </Simplification>
      <PaintAndBin />

      <h2>
        <Translated vi="Một phép chia, một mảng phẳng">One division, one flat array</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chia ô là một phép chia nguyên: một điểm ảnh có giá trị <i>v</i> rơi vào ô ⌊<i>v</i> / <i>w</i>⌋ với các khoảng
            rộng <i>w</i>. Mã của sách lấy trường hợp đơn giản nhất, mỗi giá trị một ô (<i>w</i> = 1, 256 ô), nên bản thân giá
            trị điểm ảnh là chỉ số vào mảng các ô. Vị trí của điểm ảnh không bao giờ quan trọng, chỉ giá trị của nó, nên mã coi
            ảnh như một mảng phẳng gồm <code>width*height</code> byte, xếp hết hàng này đến hàng khác như trong{' '}
            <a href={href('ch03', '3.2')}>chương 3</a>.
          </p>
        }
      >
        <p>
          Binning is one integer division: a pixel of value <i>v</i> goes in bin ⌊<i>v</i> / <i>w</i>⌋ for intervals of
          width <i>w</i>. The book's code takes the simplest case, one bin per value (<i>w</i> = 1, 256 bins), so the pixel
          value is itself the index into the bins array. The position of a pixel never matters, only its value, so the
          code treats the image as one flat array of <code>width*height</code> bytes, laid out row after row as in{' '}
          <a href={href('ch03', '3.2')}>chapter 3</a>.
        </p>
      </Translated>
      <Source section="9.1" pages="203" />
      <Translated vi={<p>Toàn bộ thuật toán tuần tự là một vòng lặp:</p>}>
        <p>The whole sequential algorithm is one loop:</p>
      </Translated>
      {seq ? (
        <Code
          tabs={[{ label: 'histogram_sequential.cpp', code: seq, tiers: SEQ_TIERS }]}
          highlight={[6, 7]}
          caption={
            <Translated
              vi={
                <>
                  Fig. 9.2, nạp từ <code>labs/ch09/snippets/histogram_sequential.cpp</code>. Nó khác trang sách chỉ ở{' '}
                  <code>const</code> trên ảnh. Một unit test biên dịch nó và kiểm tra rằng nó đếm mọi ảnh trong trình khám phá đúng
                  như trình khám phá đếm.
                </>
              }
            >
              <>
                Fig. 9.2, loaded from <code>labs/ch09/snippets/histogram_sequential.cpp</code>. It differs from the page
                only in <code>const</code> on the image. A unit test compiles it and checks that it counts every image in the
                explorer exactly as the explorer does.
              </>
            </Translated>
          }
        />
      ) : null}
      <Translated
        vi={
          <p>
            Hãy đọc nó với các ô trong đầu: hàm không bao giờ đặt chúng về 0. Giống đoạn mã của sách, nó giả định bên gọi đã làm
            việc đó. Mọi phiên bản GPU trong chương này đều giả định như vậy, và mã host của các lab xóa các ô trước mỗi lần
            khởi chạy.
          </p>
        }
      >
        <p>
          Read it with the bins in mind: the function never sets them to zero. Like the book's listing, it assumes the
          caller has done that. Every GPU version in this chapter makes the same assumption, and the labs' host code
          clears the bins before each launch.
        </p>
      </Translated>
      <Ask q={questionsS91['ch09/9.1/q-cpu']} label={tr('Predict', 'Dự đoán')} />
      <Translated
        vi={
          <p>
            Trên CPU, vòng lặp này bị giới hạn bởi bộ nhớ, không phải bởi phép tính. Nó đọc ảnh một lần, từ đầu đến cuối, nên
            mọi dòng bộ nhớ đệm nạp từ DRAM đều được dùng trọn. Mảng các ô, 256 bộ đếm 4 byte, nằm trong bộ nhớ đệm dữ liệu cấp
            một, nên mỗi phép tăng đều rẻ. Điều còn lại là DRAM có thể đưa ảnh tới nhanh đến đâu.
          </p>
        }
      >
        <p>
          On a CPU this loop is limited by memory, not by its arithmetic. It reads the image once, from start to end, so
          every cache line fetched from DRAM is used in full. The bins array, 256 counters of 4 bytes, sits in the
          level-one data cache, so each increment is cheap. What remains is how fast DRAM can deliver the image.
        </p>
      </Translated>
      <Source section="9.1" pages="203" />

      <h2>
        <Translated vi="Mỗi cột là một hàng đợi">Each bar is a queue</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Giờ hãy hình dung phiên bản song song hiển nhiên: mỗi điểm ảnh một luồng, mỗi luồng cộng một vào ô của điểm ảnh
            mình. Khi đó chiều cao của mỗi cột đúng bằng số luồng sẽ cố cập nhật ô đó. Trên một ảnh phân bố đều, các luồng trải
            khắp các ô. Trên một ảnh thật thì không: trong hình bầu trời, gần một nửa số luồng muốn cùng một ô. Tô một mảng lớn
            cùng một giá trị, hoặc chọn ảnh phẳng, làm một cột chứa mọi luồng có mặt.
          </p>
        }
      >
        <p>
          Now picture the obvious parallel version: one thread per pixel, each adding one to its pixel's bin. Then the
          height of each bar is exactly the number of threads that will try to update that bin. On a uniform image the
          threads spread over all the bins. On a real one they do not: in the sky picture, almost half of the threads
          want the same bin. Painting a large patch of one value, or choosing the flat image, makes one bar hold every
          thread there is.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đọc chiều cao cột như số lần va chạm là cách nhìn của chúng tôi; sách nêu cùng ý đó về ảnh bị lệch ở p. 210.
            </p>
          }
        >
          <p>
            Reading bar heights as collision counts is our framing; the book makes the same point about skewed images on
            p. 210.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS91['ch09/9.1/q-contention']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS91['ch09/9.1/q-owner']} />

      <h2>
        <Translated vi="Bên dưới torch.bincount">Under torch.bincount</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn đã từng gọi phép tính này từ Python. <code>torch.bincount</code> đếm các số nguyên không âm, và đầu ra của nó có
            nhiều hơn giá trị lớn nhất nó thấy một ô; <code>torch.histc</code> là người anh em của nó cho một số ô cho trước
            trên một khoảng. Cả hai đều chạy trên GPU. Đưa token id vào <code>bincount</code> thì các ô là bộ từ vựng: từ khoảng
            năm mươi nghìn đến hai trăm nghìn ô trong các mô hình hiện nay, quá nhiều cho những thủ thuật trên chip giúp một
            histogram ảnh 256 ô chạy nhanh. Giới hạn đó quay lại ở mục 9.4 và trong <a href={href('ch09', 'labs')}>lab 9.3</a>.
          </p>
        }
      >
        <p>
          You have called this computation from Python. <code>torch.bincount</code> counts non-negative integers, and its
          output has one more bin than the largest value it saw; <code>torch.histc</code> is its sibling for a given
          number of bins over a range. Both run on the GPU. Feed <code>bincount</code> token ids and the bins are the
          vocabulary: from about fifty thousand to two hundred thousand of them in current models, far too many for the
          on-chip tricks that make a 256-bin image histogram fast. That limit
          comes back in section 9.4 and in <a href={href('ch09', 'labs')}>lab 9.3</a>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md, C6, D1, D5, D7)', '2026-11-20 (notes/today/ch09-facts.md, các mục C6, D1, D5, D7)')}>
        <Translated
          vi={
            <p>
              Thư viện của Apple có một histogram ảnh, <code>MPSImageHistogram</code>: số ô phải là lũy thừa của hai, các số đếm
              là 32-bit, và nó làm việc trên texture. Nó không phục vụ một buffer token id.
            </p>
          }
        >
          <p>
            Apple's library has an image histogram, <code>MPSImageHistogram</code>: bin counts must be a power of two,
            counts are 32-bit, and it works on textures. It does not serve a buffer of token ids.
          </p>
        </Translated>
      </Today>
    </>
  );
}
