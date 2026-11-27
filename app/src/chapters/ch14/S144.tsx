import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Beyond, Bench, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { href } from '../../lib/router';
import { useSolutionStatus } from './solutionStatus';
import {
  BOOK_KEYS,
  bin,
  lsdSort,
  parseKeys,
  promptKey,
  randomKeys,
  type SortPass,
  type SortRun,
} from '../../sim/radixsort';
import {
  PRESETS,
  TRANSFORM_LABEL,
  bitsOf,
  explore,
  hex,
  items,
  rawAscending,
  show,
  type DescBy,
  type Transform,
} from '../../sim/floatkey';
import { LsdPassesScene } from './LsdPassesScene';

// The key transform as the lesson shows it: the region "floatkey" of
// labs/ch14/snippets/s144_floatkey.cu, which is front-end-checked and run on the emulator
// against src/sim/floatkey.ts. S144.snippets.test.ts fails if this copy drifts.
const FLOATKEY = `__host__ __device__ inline unsigned int float_to_key(unsigned int u) {
    if(u == 0x80000000u) u = 0u;                    // -0.0 sorts as +0.0
    unsigned int mask = (u & 0x80000000u) ? 0xFFFFFFFFu  // negative: invert every bit
                                          : 0x80000000u; // otherwise: set the sign bit
    return u ^ mask;
}

__host__ __device__ inline unsigned int float_to_key_desc(unsigned int u) {
    return ~float_to_key(u);                        // descending: invert, sort ascending
}

__global__ void logits_to_keys(const unsigned int* logitBits, unsigned int* keys,
                               unsigned int n, bool descending) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) {
        unsigned int u = logitBits[i];              // the float's bits, never a float
        keys[i] = descending ? float_to_key_desc(u) : float_to_key(u);
    }
}`;
const SNIPPET_STATUS = () =>
  tr(
    'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware',
    'front-end-checked; emulator-checked; unverified trên phần cứng CUDA và HIP',
  );
const LAB3_METAL = 'ch14/lab3_sortrows/solution/kernel.metal';
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);

// ---------------------------------------------------------------- the LSD stepper
type KeysMode = 'book' | 'shuffle' | 'own';
const OWN_DEFAULT = '0101 0100 1110 0011 1010 0001 0110 1001';
const CW = 44;
const AX = 12;
const TOP = 30;
const BOT = 146;

const digitName = (p: SortPass) => (p.bits === 1 ? tr(`bit ${p.shift}`, `bit ${p.shift}`) : tr(`bits ${p.shift + p.bits - 1} and ${p.shift}`, `các bit ${p.shift + p.bits - 1} và ${p.shift}`));
const digitShort = (p: SortPass) => (p.bits === 1 ? tr(`bit ${p.shift}`, `bit ${p.shift}`) : tr(`bits ${p.shift + p.bits - 1}–${p.shift}`, `các bit ${p.shift + p.bits - 1}–${p.shift}`));
const sortedText = (k: number) =>
  k === 4
    ? tr('sorted', 'đã sắp xếp')
    : k === 0
      ? tr('not in order by its low bits', 'chưa đúng thứ tự theo các bit thấp')
      : tr(`in order by its low ${['', 'bit', 'two bits', 'three bits'][k]}`, `đúng thứ tự theo ${['', 'bit', 'hai bit', 'ba bit'][k]} thấp nhất`);
/** A halo so that a label stays readable where arrows cross it. */
const HALO = { paintOrder: 'stroke', stroke: 'var(--bench)', strokeWidth: 4, strokeLinejoin: 'round' } as const;

/** What the passes so far order the list by, primary first (true only when all were stable). */
function orderedBy(run: SortRun, upto: number): string {
  const ds = run.passes.slice(0, upto).map(digitShort).reverse();
  if (ds.length === 1) return ds[0];
  return ds[0] + tr(', then ', ', rồi ') + ds.slice(1).join(tr(', then ', ', rồi ')) + tr(' among equals', ' giữa các khóa bằng nhau');
}

function describe(run: SortRun, k: number): string {
  const p = run.passes[k - 1];
  const n = p.input.length;
  const buckets = p.counts
    .map((c, d) => (c ? tr(`${c} with ${bin(d, p.bits)} to positions ${p.starts[d]}–${p.starts[d] + c - 1}`, `${c} khóa có ${bin(d, p.bits)} tới các vị trí ${p.starts[d]}–${p.starts[d] + c - 1}`) : null))
    .filter(Boolean)
    .join('; ');
  let s = tr(`Pass ${k} used ${digitName(p)}: ${buckets}.`, `Lượt ${k} dùng ${digitName(p)}: ${buckets}.`);
  s += p.unstable ? tr(' Its top bucket was filled backwards.', ' Bucket trên cùng của nó bị điền ngược.') : tr(' Each bucket kept the order its keys arrived in.', ' Mỗi bucket giữ thứ tự các khóa đi vào nó.');
  const anyUnstable = run.passes.slice(0, k).some((q) => q.unstable);
  if (!anyUnstable && run.order === 'msd')
    s += tr(
      ` The last pass has the final word: the list is now ordered by ${orderedBy(run, k)}. It is ${sortedText(p.sortedBy)}.`,
      ` Lượt cuối có tiếng nói quyết định: danh sách giờ được sắp theo ${orderedBy(run, k)}. Nó ${sortedText(p.sortedBy)}.`,
    );
  else s += tr(` The list of ${n} keys is now ${sortedText(p.sortedBy)}.`, ` Danh sách ${n} khóa giờ ${sortedText(p.sortedBy)}.`);
  return s;
}

function KeyBox({ x, y, k, mark, markBits = 1, shade = 0, hot = false, ring = false, bad = false, ghost = false, row, slot }: {
  x: number; y: number; k: number; mark?: number; markBits?: number; shade?: number; hot?: boolean; ring?: boolean; bad?: boolean; ghost?: boolean;
  row: string; slot: number;
}) {
  const s = bin(k, 4);
  if (ghost) return <rect x={x} y={y} width={CW - 6} height={26} rx="2" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />;
  return (
    <g data-row={row} data-slot={slot} data-key={s} data-hot={hot ? '1' : undefined} data-ring={ring ? '1' : undefined} data-bad={bad ? '1' : undefined}>
      <rect x={x} y={y} width={CW - 6} height={26} rx="2"
        fill={bad ? 'var(--fail-wash)' : hot ? 'var(--ink)' : 'var(--cell)'} stroke={bad ? 'var(--fail)' : hot || ring ? 'var(--ink)' : 'var(--rule)'}
        strokeWidth={ring && !hot ? 2.4 : 1} />
      {shade > 0 && !hot ? <rect x={x + 3 + (4 - shade) * 8} y={y + 3} width={shade * 8} height={20} rx="1" fill="var(--rule)" /> : null}
      {s.split('').map((ch, c) => {
        const bitIx = 3 - c;
        const isMark = mark != null && bitIx >= mark && bitIx < mark + markBits;
        return (
          <g key={c}>
            <text x={x + 7 + c * 8} y={y + 17.5} fontSize="12.5" textAnchor="middle" fontWeight={isMark ? 700 : 400}
              style={{ fill: hot ? 'var(--page)' : bad ? 'var(--fail)' : 'var(--ink)' }}>{ch}</text>
            {isMark ? <rect x={x + 3.5 + c * 8} y={y + 21} width={7} height={1.8} style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }} /> : null}
          </g>
        );
      })}
    </g>
  );
}

/** askId: the key the next question is about; lastId: the key the last question was about. */
function PassPicture({ run, at, askId, lastId }: { run: SortRun; at: number; askId: number | null; lastId: number | null }) {
  const n = run.keys.length;
  const w = AX * 2 + n * CW;
  const cx = (j: number) => AX + j * CW + (CW - 6) / 2;
  const p = at > 0 ? run.passes[at - 1] : null;
  const next = at < run.passes.length ? run.passes[at] : null;
  const topKeys = p ? p.input : run.keys;
  const topIds = p ? p.inIds : run.keys.map((_, i) => i);
  const topMark = p ?? next!;
  const done = at === run.passes.length;
  const badFrom = done && !run.sorted ? run.firstDisorder : -1;
  const topBucket = p ? p.counts.length - 1 - [...p.counts].reverse().findIndex((c) => c > 0) : -1;
  return (
    <svg viewBox={`0 0 ${w} 236`} role="img" aria-label={tr('Keys before and after the current pass', 'Các khóa trước và sau lượt hiện tại')} style={{ width: '100%', maxWidth: w, minWidth: Math.min(w, 560), display: 'block' }}>
      <defs>
        <marker id="lsdst-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute value */}
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink-2)' }} />
        </marker>
        <marker id="lsdst-arrow-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute value */}
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--fail)' }} />
        </marker>
      </defs>
      <text x={AX} y={TOP - 9} fontSize="11.5" fontWeight="600" style={HALO}>
        {p
          ? (p.index === 1 ? tr('input', 'đầu vào') : tr(`after pass ${p.index - 1}`, `sau lượt ${p.index - 1}`)) +
            tr(`, ${digitShort(p)} marked`, `, đánh dấu ${digitShort(p)}`)
          : tr(`input, ${digitShort(next!)} marked: pass 1 reads it`, `đầu vào, đánh dấu ${digitShort(next!)}: lượt 1 đọc nó`)}
      </text>
      {topKeys.map((k, j) => (
        <KeyBox key={j} x={AX + j * CW} y={TOP} k={k} mark={topMark.shift} markBits={topMark.bits} row="top" slot={j} // i18n-ignore: row id for checks
          hot={!p && askId === topIds[j]} />
      ))}
      {p
        ? p.dst.map((d, i) => {
            const bad = p.unstable && ((p.input[i] >>> p.shift) & ((1 << p.bits) - 1)) === topBucket;
            const hot = lastId != null && p.inIds[i] === lastId;
            return (
              <line key={i} data-arrow={i} data-to={d} x1={cx(i)} y1={TOP + 28} x2={cx(d)} y2={BOT - 3}
                stroke={bad ? 'var(--fail)' : hot ? 'var(--ink)' : 'var(--ink-2)'} strokeWidth={hot ? 2.4 : bad ? 1.5 : 1}
                opacity={hot || bad ? 1 : 0.75} markerEnd={bad ? 'url(#lsdst-arrow-bad)' : 'url(#lsdst-arrow)'} />
            );
          })
        : null}
      <text x={AX} y={BOT - 9} fontSize="11.5" fontWeight="600" opacity={p ? 1 : 0.5} style={HALO}>
        {p
          ? tr(`after pass ${p.index}${next ? `; next: ${digitShort(next)}` : ''}`, `sau lượt ${p.index}${next ? `; tiếp theo: ${digitShort(next)}` : ''}`)
          : tr('after pass 1', 'sau lượt 1')}
      </text>
      {(p ? p.output : topKeys).map((k, j) => (
        <KeyBox key={j} x={AX + j * CW} y={BOT} k={k} row="bottom" slot={j} ghost={!p} // i18n-ignore: row id for checks
          mark={next && p ? next.shift : undefined} markBits={next?.bits}
          shade={p && !next ? p.sortedBy : 0}
          hot={!!p && askId === p.outIds[j]} ring={!!p && lastId === p.outIds[j]} bad={badFrom >= 0 && (j === badFrom || j === badFrom + 1)} />
      ))}
      {p
        ? p.counts.map((c, d) =>
            c ? (
              <g key={d} data-bucket={d}>
                <line x1={AX + p.starts[d] * CW + 2} x2={AX + (p.starts[d] + c) * CW - 8} y1={BOT + 36} y2={BOT + 36} stroke="var(--ink-2)" />
                <text x={AX + (p.starts[d] * CW + (p.starts[d] + c) * CW) / 2 - 3} y={BOT + 50} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                  {p.bits === 1 ? `${digitShort(p)} = ${d}` : bin(d, p.bits)}
                </text>
              </g>
            ) : null,
          )
        : null}
      {p ? (
        <text x={w - AX} y={BOT + 72} fontSize="12" textAnchor="end" data-sorted-by={p.sortedBy} style={{ fill: 'var(--ink-2)' }}>
          {sortedText(p.sortedBy)}
        </text>
      ) : null}
    </svg>
  );
}

function LsdStepper() {
  const [mode, setMode] = useState<KeysMode>('book');
  const [seed, setSeed] = useState(7);
  const [ownText, setOwnText] = useState(OWN_DEFAULT);
  const [radix, setRadix] = useState(1);
  const [unstable, setUnstable] = useState(0);
  const [msd, setMsd] = useState(false);
  const [at, setAt] = useState(0);
  const [guess, setGuess] = useState('');
  const [last, setLast] = useState<{ pass: number; guess: number | null; id: number; key: number; landed: number } | null>(null);

  const locale = useLocale();
  const parsed = useMemo(() => (mode === 'own' ? parseKeys(ownText, 4, 8) : null), [mode, ownText, locale]);
  const keys = useMemo(
    () => (mode === 'book' ? BOOK_KEYS : mode === 'shuffle' ? randomKeys(16, 4, seed) : parsed && 'keys' in parsed ? parsed.keys : BOOK_KEYS.slice(0, 8)),
    [mode, seed, parsed],
  );
  const passes = 4 / radix;
  const unstablePass = unstable > passes ? 0 : unstable;
  const run = useMemo(() => lsdSort(keys, 4, radix, { order: msd ? 'msd' : 'lsd', unstablePass: unstablePass || null }), [keys, radix, msd, unstablePass]);
  useEffect(() => {
    setAt(0);
    setLast(null);
    setGuess('');
  }, [run]);

  const next = at < run.passes.length ? run.passes[at] : null;
  const promptPos = next ? promptKey(next) : null;
  const askId = next && promptPos != null ? next.inIds[promptPos] : null;

  const step = () => {
    if (!next || promptPos == null) return;
    const g = guess.trim() === '' ? null : Number(guess);
    setLast({ pass: next.index, guess: g != null && !Number.isNaN(g) ? g : null, id: next.inIds[promptPos], key: next.input[promptPos], landed: next.dst[promptPos] });
    setGuess('');
    setAt((x) => Math.min(run.passes.length, x + 1));
  };

  const done = at === run.passes.length;
  return (
    <Bench
      title={tr('Radix sort, pass by pass', 'Radix sort, từng lượt một')}
      task={tr(
        'Before each pass, say where the highlighted key will land. Then break a pass and watch the result.',
        'Trước mỗi lượt, hãy nói khóa được tô sáng sẽ rơi vào đâu. Sau đó làm hỏng một lượt và xem kết quả.',
      )}
      foot={
        <Translated
          vi={
            <>
              Khóa dài 4 bit. Các bit gạch chân là chữ số mà lượt đó đọc; các bit thấp được tô ở hàng cuối là những bit
              mà danh sách đã đúng thứ tự theo. Mô hình: <code>src/sim/radixsort.ts</code> (các bài kiểm thử của nó tái
              hiện mọi hàng của Hình 14.4 và 14.10).
            </>
          }
        >
          <>
            Keys are 4 bits. The underlined bits are the digit the pass reads; the shaded low bits of the final row are
            the ones the list is in order by. Model: <code>src/sim/radixsort.ts</code> (its tests reproduce every row of
            Figs. 14.4 and 14.10).
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Khóa">Keys</Translated>
          <select value={mode} onChange={(e) => setMode(e.target.value as KeysMode)}>
            <option value="book">{tr("the book's sixteen (Fig. 14.4)", 'mười sáu khóa của sách (Hình 14.4)')}</option>
            <option value="shuffle">{tr('sixteen random keys', 'mười sáu khóa ngẫu nhiên')}</option>
            <option value="own">{tr('your own eight keys', 'tám khóa của riêng bạn')}</option>
          </select>
        </label>
        {mode === 'shuffle' ? <button onClick={() => setSeed(seed + 1)}>{tr('Shuffle', 'Xáo trộn')}</button> : null}
        {mode === 'own' ? (
          <label className="field">
            <Translated vi="Tám khóa, dạng nhị phân hoặc 0 đến 15">Eight keys, binary or 0 to 15</Translated>
            <input type="text" value={ownText} aria-label={tr('your eight keys', 'tám khóa của bạn')} onChange={(e) => setOwnText(e.target.value)} style={{ width: '19rem' }} />
          </label>
        ) : null}
        <label className="field">
          <Translated vi="Độ rộng radix">Radix width</Translated>
          <span className="seg" role="group" aria-label={tr('Radix width', 'Độ rộng radix')}>
            {[1, 2].map((r) => (
              <button key={r} aria-pressed={radix === r} onClick={() => setRadix(r)}><Translated vi={<>{r} bit</>}>{r} bit{r > 1 ? 's' : ''}</Translated></button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Làm một lượt không ổn định">Make a pass unstable</Translated>
          <select value={unstablePass} onChange={(e) => setUnstable(Number(e.target.value))} aria-label={tr('Make a pass unstable', 'Làm một lượt không ổn định')}>
            <option value={0}>{tr('none', 'không lượt nào')}</option>
            {Array.from({ length: passes }, (_, k) => (
              <option key={k} value={k + 1}><Translated vi={<>lượt {k + 1}</>}>pass {k + 1}</Translated></option>
            ))}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={msd} onChange={(e) => setMsd(e.target.checked)} />{' '}<Translated vi="bit có nghĩa lớn nhất trước">most significant bit first</Translated>
        </label>
      </div>
      {parsed && 'error' in parsed ? <p className="t-fail" role="alert"><Translated vi={<>Đang dùng tám khóa đầu của sách: {parsed.error}.</>}>Using the book's first eight keys: {parsed.error}.</Translated></p> : null}

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setAt(0); setLast(null); }} disabled={at === 0}>{tr('Restart', 'Bắt đầu lại')}</button>
        <button onClick={() => { setAt((x) => Math.max(0, x - 1)); setLast(null); }} disabled={at === 0}>{tr('Back', 'Lùi')}</button>
        <button className="primary" onClick={step} disabled={!next}><Translated vi={<>Chạy lượt {Math.min(at + 1, run.passes.length)}</>}>Run pass {Math.min(at + 1, run.passes.length)}</Translated></button>
        <button onClick={() => { setAt(run.passes.length); setLast(null); }} disabled={done}>{tr('Run all', 'Chạy tất cả')}</button>
        <span className="muted" aria-live="polite"><Translated vi={<>{at} / {run.passes.length} lượt</>}>{at} of {run.passes.length} passes</Translated></span>
      </div>

      {next && promptPos != null ? (
        <div className="row" style={{ marginBottom: '0.8rem' }} data-part="prompt">
          <Translated
            vi={
              <span>
                Lượt {next.index} đọc {digitName(next)}. Khóa được tô sáng <b>{bin(next.input[promptPos], 4)}</b> đang ở vị
                trí {promptPos}. Nó sẽ rơi vào đâu?
              </span>
            }
          >
            <span>
              Pass {next.index} reads {digitName(next)}. The highlighted key <b>{bin(next.input[promptPos], 4)}</b> is at
              position {promptPos}. Where will it land?
            </span>
          </Translated>
          <input type="number" min={0} max={run.keys.length - 1} value={guess} aria-label={tr('your guess', 'dự đoán của bạn')}
            onChange={(e) => setGuess(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && step()} />
        </div>
      ) : null}

      <div className={'narrate' + (done && !run.sorted ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {at === 0 ? (
          <Translated
            vi={<>Chưa có gì chạy. {run.passes.length} lượt, mỗi lượt {radix} bit, sẽ phủ hết các khóa 4 bit, bắt đầu từ chữ số có nghĩa {msd ? 'lớn' : 'nhỏ'} nhất.</>}
          >
            <>Nothing has run. {run.passes.length} passes of {radix} bit{radix > 1 ? 's' : ''} will cover the 4-bit keys, {msd ? 'most' : 'least'} significant digit first.</>
          </Translated>
        ) : (
          <>
            {last && last.pass === at ? (
              <Translated
                vi={
                  <>
                    Khóa bạn được hỏi, {bin(last.key, 4)} (có viền), đã rơi vào vị trí <b>{last.landed}</b>
                    {last.guess != null ? (last.guess === last.landed ? <>; bạn nói {last.guess}: <span className="t-ok">đúng</span>. </> : <>; bạn nói {last.guess}. </>) : '. '}
                  </>
                }
              >
                <>
                  The key you were asked about, {bin(last.key, 4)} (outlined), landed at position <b>{last.landed}</b>
                  {last.guess != null ? (last.guess === last.landed ? <>; you said {last.guess}: <span className="t-ok">right</span>. </> : <>; you said {last.guess}. </>) : '. '}
                </>
              </Translated>
            ) : null}
            {describe(run, at)}
          </>
        )}
      </div>

      <PassPicture run={run} at={at} askId={askId} lastId={last && last.pass === at ? last.id : null} />

      {done ? (
        <p style={{ margin: '0.6rem 0 0' }} data-part="outcome">
          {run.sorted ? <span className="chip ok"><Translated vi="đã sắp xếp">sorted</Translated></span> : <span className="chip fail"><Translated vi="chưa sắp xếp">not sorted</Translated></span>}{' '}
          {!run.sorted ? (
            <Translated
              vi={
                <>
                  {bin(run.result[run.firstDisorder], 4)} ở vị trí {run.firstDisorder} đứng trước{' '}
                  {bin(run.result[run.firstDisorder + 1], 4)}.
                </>
              }
            >
              <>
                {bin(run.result[run.firstDisorder], 4)} at position {run.firstDisorder} comes before{' '}
                {bin(run.result[run.firstDisorder + 1], 4)}.
              </>
            </Translated>
          ) : !run.stable ? (
            <Translated
              vi={
                <>
                  nhưng hai khóa bằng nhau ({bin(run.result[run.firstTieSwap], 4)}, vị trí {run.firstTieSwap} và {run.firstTieSwap + 1}) bị lệch
                  khỏi thứ tự đầu vào. Một danh sách chỉ có khóa không thể cho thấy điều đó; một danh sách các cặp (logit, id token) thì có.
                </>
              }
            >
              <>
                but two equal keys ({bin(run.result[run.firstTieSwap], 4)}, positions {run.firstTieSwap} and {run.firstTieSwap + 1}) are out of
                their input order. A list of bare keys cannot show it; a list of (logit, token id) pairs would.
              </>
            </Translated>
          ) : (
            tr('Equal keys are in their input order too.', 'Các khóa bằng nhau cũng giữ thứ tự đầu vào.')
          )}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the float explorer
function FloatExplorer() {
  const [presetId, setPresetId] = useState('ties');
  const [tf, setTf] = useState<Transform>('none');
  const [zero, setZero] = useState(false);
  const [descBy, setDescBy] = useState<DescBy>('invert');
  const pr = PRESETS.find((p) => p.id === presetId)!;
  const xs = useMemo(() => items(pr.bits), [pr]);
  const r = useMemo(() => explore(xs, { transform: tf, zero, descBy }), [xs, tf, zero, descBy]);
  const label = (token: number) =>
    tr(
      pr.labels?.[token] ?? show(pr.bits[token]),
      ({ 'smallest negative subnormal': 'số subnormal âm nhỏ nhất', 'smallest positive subnormal': 'số subnormal dương nhỏ nhất' } as Record<string, string>)[pr.labels?.[token] ?? ''] ??
        pr.labels?.[token] ??
        show(pr.bits[token]),
    );
  const ok = r.firstMismatch < 0;
  return (
    <Bench
      title={tr('Logits to keys', 'Từ logits sang khóa')}
      task={tr(
        'Find the settings that make every preset match the reference, and say what each one fixes.',
        'Tìm các thiết lập làm mọi bộ giá trị mẫu khớp với tham chiếu, và nói mỗi thiết lập sửa điều gì.',
      )}
      foot={
        <Translated
          vi={
            <>
              Sắp xếp lớn nhất trước bằng một phép sắp xếp ổn định trên các khóa. Tham chiếu: {r.rule === "CUB's bits rule" ? 'quy tắc của CUB (phép biến đổi đầy đủ, bước −0.0, đảo bit; NaN theo bit của nó)' : 'std::stable_sort với > trên các giá trị'}.
              Mô hình: <code>src/sim/floatkey.ts</code> (các bài kiểm thử của nó kiểm tra bảng E1 đến từng bit).
            </>
          }
        >
          <>
            Sorted largest first by a stable sort of the keys. Reference: {r.rule === "CUB's bits rule" ? "CUB's rule (the full transform, the −0.0 step, inverted; NaN by its bits)" : 'std::stable_sort with > on the values'}.
            Model: <code>src/sim/floatkey.ts</code> (its tests check E1's table bit for bit).
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Hàng logits">Row of logits</Translated>
          <select value={presetId} onChange={(e) => setPresetId(e.target.value)}>
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>{tr(p.label, ({ ties: 'logits có giá trị bằng nhau', zeros: 'dấu và số không', edges: 'các giá trị biên (E1 cộng hai NaN)' } as Record<string, string>)[p.id] ?? p.label)}</option>
            ))}
          </select>
        </label>
        <label className="field">
          <Translated vi="Phép biến đổi">Transform</Translated>
          <span className="seg" role="group" aria-label={tr('Transform', 'Phép biến đổi')}>
            {(['none', 'sign', 'full'] as Transform[]).map((t) => (
              <button key={t} aria-pressed={tf === t} onClick={() => setTf(t)} title={tr(TRANSFORM_LABEL[t], { none: 'không: bit thô', sign: 'chỉ lật bit dấu', full: 'đầy đủ: đặt bit dấu, hoặc đảo mọi bit của số âm' }[t])}>
                {t === 'none' ? tr('raw bits', 'bit thô') : t === 'sign' ? tr('flip sign bit', 'lật bit dấu') : tr('full', 'đầy đủ')}
              </button>
            ))}
          </span>
        </label>
        <label className="check">
          <input type="checkbox" checked={zero} onChange={(e) => setZero(e.target.checked)} />{' '}<Translated vi="bước −0.0">−0.0 step</Translated>
        </label>
        <label className="field">
          <Translated vi="Giảm dần bằng cách">Descending by</Translated>
          <span className="seg" role="group" aria-label={tr('Descending by', 'Giảm dần bằng cách')}>
            {(['invert', 'reverse'] as DescBy[]).map((d) => (
              <button key={d} aria-pressed={descBy === d} onClick={() => setDescBy(d)}>
                {d === 'invert' ? tr('inverting the keys', 'đảo bit các khóa') : tr('reversing an ascending sort', 'lật ngược một phép sắp xếp tăng dần')}
              </button>
            ))}
          </span>
        </label>
      </div>
      <table className="threads" aria-label={tr('Each logit and its key', 'Mỗi logit và khóa của nó')} style={{ marginBottom: '0.9rem' }}>
        <thead>
          <tr>
            <th><Translated vi="token">token</Translated></th><th><Translated vi="giá trị">value</Translated></th><th><Translated vi="bit">bits</Translated></th><th><Translated vi="sau bước −0.0">after −0.0 step</Translated></th><th><Translated vi="khóa">key</Translated></th><th><Translated vi="sắp tăng dần theo">sorted ascending by</Translated></th>
          </tr>
        </thead>
        <tbody>
          {r.rows.map((row) => (
            <tr key={row.token} data-token={row.token}>
              <td>{row.token}</td>
              <td>{label(row.token)}</td>
              <td className="mono">{hex(row.bits)}</td>
              <td className="mono" style={{ color: zero && row.stepped !== row.bits ? 'var(--ink)' : 'var(--ink-3)' }}>{hex(row.stepped)}</td>
              <td className="mono">{hex(row.key)}</td>
              <td className="mono" data-sortkey={hex(row.sortKey)}>{hex(row.sortKey)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mx-label"><Translated vi="Đã sắp xếp, lớn nhất trước">Sorted, largest first</Translated></div>
      <div className="mx" style={{ display: 'flex', flexWrap: 'wrap', marginBottom: '0.5rem' }} data-part="output">
        {r.order.map((t, k) => (
          <span key={k} className={r.match[k] ? 'good' : 'bad'} data-token={t} style={{ padding: '0 0.35rem', minWidth: '3.4rem' }}>
            {show(pr.bits[t])}
            <sub>{t}</sub>
          </span>
        ))}
      </div>
      <div className="mx-label"><Translated vi="Tham chiếu">Reference</Translated></div>
      <div className="mx" style={{ display: 'flex', flexWrap: 'wrap', marginBottom: '0.8rem' }} data-part="reference">
        {r.reference.map((t, k) => (
          <span key={k} data-token={t} style={{ padding: '0 0.35rem', minWidth: '3.4rem' }}>
            {show(pr.bits[t])}
            <sub>{t}</sub>
          </span>
        ))}
      </div>
      <p style={{ margin: 0 }} data-part="verdict">
        {ok ? (
          <span className="chip ok"><Translated vi="khớp với tham chiếu">matches the reference</Translated></span>
        ) : (
          <Translated
            vi={
              <>
                <span className="chip fail">khác với tham chiếu</span> lần đầu ở vị trí {r.firstMismatch}: token{' '}
                {r.order[r.firstMismatch]} trong khi tham chiếu có token {r.reference[r.firstMismatch]}.
              </>
            }
          >
            <>
              <span className="chip fail">differs from the reference</span> first at position {r.firstMismatch}: token{' '}
              {r.order[r.firstMismatch]} where the reference has token {r.reference[r.firstMismatch]}.
            </>
          </Translated>
        )}{' '}
        <span className="muted"><Translated vi="Các số nhỏ là id token.">Small numbers are token ids.</Translated></span>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS144: Record<string, Question> = {
  'ch14/14.4/q-pass2': {
    kind: 'numeric',
    id: 'ch14/14.4/q-pass2',
    prompt: (
      <Translated
        vi={
          <>
            Lấy mười sáu khóa của sách, <code>1100 0011 0110 1001 1111 1000 0101 1010 1001 0110 1011 1101 0100 1010 0111
            0000</code>. Sau lượt 2 (bit 0 và 1), <code>0110</code> đầu tiên nằm ở vị trí nào, đếm từ 0?
          </>
        }
      >
        <>
          Take the book's sixteen keys, <code>1100 0011 0110 1001 1111 1000 0101 1010 1001 0110 1011 1101 0100 1010 0111
          0000</code>. After pass 2 (bits 0 and 1), at which position, counting from 0, is the first <code>0110</code>?
        </>
      </Translated>
    ),
    answer: 8,
    explain: (
      <Translated
        vi={
          <p>
            Sau lượt 2, danh sách đúng thứ tự theo hai bit thấp nhất. Bốn khóa kết thúc bằng 00 và bốn khóa bằng 01, nên các khóa
            kết thúc bằng 10 bắt đầu ở vị trí 8, theo thứ tự lượt 1 để lại; <code>0110</code> là khóa đầu tiên trong số đó (đầu ra
            của lượt 1: 1100 <b>0110</b> 1000 1010 …). Trình chạy từng bước cho thấy điều này.
          </p>
        }
      >
        <p>
          After pass 2 the list is in order by its low two bits. Four keys end in 00 and four in 01, so the keys ending
          in 10 start at position 8, in the order pass 1 left them; <code>0110</code> is the first of them there (pass 1
          output: 1100 <b>0110</b> 1000 1010 …). The stepper shows it.
        </p>
      </Translated>
    ),
    source: 'ch. 14 §14.4, pp. 335–336, Fig. 14.4',
  },
  'ch14/14.4/q-msb': {
    kind: 'choice',
    id: 'ch14/14.4/q-msb',
    prompt: (
      <Translated vi={<>Chạy cùng bốn lượt 1 bit ổn định đó nhưng từ bit có nghĩa lớn nhất xuống (bit 3, rồi 2, 1, 0). Kết quả được sắp theo cái gì?</>}>
        <>Run the same four stable 1-bit passes from the most significant bit down (bit 3, then 2, 1, 0). What is the result in order by?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Theo khóa: mọi lượt đều ổn định, nên thứ tự các lượt không tạo ra khác biệt.">By the key: every pass is stable, so the order of the passes makes no difference.</Translated>,
        why: <Translated vi="Mỗi lượt ổn định biến chữ số của nó thành chữ số quan trọng nhất cho đến lúc đó. Lượt cuối thắng.">Each stable pass makes its digit the most important one so far. The last pass wins.</Translated>,
      },
      {
        label: <Translated vi="Không theo gì cụ thể: không bắt đầu từ các bit thấp thì kết quả là ngẫu nhiên.">By nothing in particular: without the low bits first, the result is random.</Translated>,
        why: <Translated vi="Nó hoàn toàn xác định, và đã được sắp xếp, chỉ là theo sai khóa.">It is fully determined, and sorted, just by the wrong key.</Translated>,
      },
      {
        label: <Translated vi="Theo khóa đảo ngược thứ tự bit: bit 0, được lượt cuối đọc, trở thành khóa chính.">By the bit-reversed key: bit 0, read by the last pass, becomes the primary key.</Translated>,
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một lượt ổn định chỉ giữ thứ tự cũ giữa các khóa có chữ số bằng nhau, nên lượt cuối đọc gì thì cái đó chi phối. Bắt
            đầu từ chữ số có nghĩa nhỏ nhất đặt chữ số có nghĩa lớn nhất ở cuối, đúng chỗ của nó. Đánh dấu "bit có nghĩa lớn nhất
            trước" trong trình chạy từng bước: mọi khóa chẵn đều đứng trước mọi khóa lẻ. (Radix sort MSD của mục 14.9 thì khác:
            nó sắp xếp từng bucket riêng thay vì cả danh sách ở mỗi lượt.)
          </p>
        }
      >
        <p>
          A stable pass keeps the earlier order only among keys with equal digits, so whatever the last pass reads
          dominates. Least significant first puts the most significant digit last, where it belongs. Tick "most
          significant bit first" in the stepper: every even key ends up before every odd key. (Section 14.9's MSD radix
          sort is different: it sorts each bucket separately instead of the whole list each pass.)
        </p>
      </Translated>
    ),
    source: 'our analysis of ch. 14 §14.4, p. 335; §14.9, p. 346',
    vi: { source: 'phân tích của khóa học này về ch. 14 §14.4, p. 335; §14.9, p. 346' },
  },
  'ch14/14.4/q-negatives': {
    kind: 'choice',
    id: 'ch14/14.4/q-negatives',
    prompt: (
      <Translated
        vi={
          <>
            −1.0 là <code>0xBF800000</code> và −2.0 là <code>0xC0000000</code>. Chỉ lật bit dấu của mọi giá trị rồi sắp xếp tăng
            dần như số nguyên không dấu. Chuyện gì xảy ra với hai giá trị này?
          </>
        }
      >
        <>
          −1.0 is <code>0xBF800000</code> and −2.0 is <code>0xC0000000</code>. Flip only the sign bit of every value and
          sort ascending as unsigned integers. What happens to these two?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Chúng ra đúng thứ tự: khi bit dấu đã lật, cả hai được đọc như các khóa xếp theo độ lớn.">They come out in the right order: with the sign bit flipped, both read as keys ordered by magnitude.</Translated>,
        why: <Translated vi="Khóa của chúng thành 0x3F800000 và 0x40000000: −2.0 có khóa lớn hơn.">Their keys become 0x3F800000 and 0x40000000: −2.0 has the larger key.</Translated>,
      },
      {
        label: <Translated vi="−2.0 vẫn rơi phía trên −1.0: các bit độ lớn của số âm cũng phải được đảo.">−2.0 still lands above −1.0: a negative's magnitude bits must be inverted too.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Cả hai rơi phía trên mọi giá trị dương, vì các bit số mũ của chúng lớn hơn.">Both land above every positive value, because their exponent bits are larger.</Translated>,
        why: <Translated vi="Lật bit dấu đã sửa phần đó: số âm giờ bắt đầu bằng 0.">Flipping the sign bit fixed that part: negatives now start with 0.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lật bit dấu đặt mọi số âm xuống dưới mọi số dương, nhưng các bit còn lại của một số âm vẫn tăng theo độ lớn, nên giữa
            các số âm thứ tự bị ngược. Đảo cả 32 bit của một số âm sửa cả hai cùng lúc: −2.0 thành <code>0x3FFFFFFF</code> và
            −1.0 thành <code>0x407FFFFF</code>.
          </p>
        }
      >
        <p>
          Flipping the sign bit puts every negative below every positive, but a negative's remaining bits still grow with
          its magnitude, so among negatives the order is backwards. Inverting all 32 bits of a negative fixes both at
          once: −2.0 becomes <code>0x3FFFFFFF</code> and −1.0 <code>0x407FFFFF</code>.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch14-facts.md A3, E1', // i18n-ignore: citation (file path)
  },
  'ch14/14.4/q-ties-desc': {
    kind: 'choice',
    id: 'ch14/14.4/q-ties-desc',
    prompt: (
      <Translated
        vi={
          <>
            Token 3 và 7 có xác suất bằng nhau hoàn toàn. Hàng được sắp giảm dần bằng cách sắp tăng dần, ổn định, rồi lật ngược
            kết quả. Token nào trong hai token đứng trước?
          </>
        }
      >
        <>
          Tokens 3 and 7 have exactly equal probability. The row is sorted descending by sorting ascending, stably, and
          then reversing the result. Which of the two comes first?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Token 7: lật ngược cả danh sách cũng lật ngược thứ tự của các giá trị bằng nhau.">Token 7: reversing the whole list reverses the order of the ties too.</Translated>, correct: true },
      {
        label: <Translated vi="Token 3: phép sắp xếp tăng dần ổn định đặt token 3 trước, và việc lật ngược giữ thứ tự các giá trị bằng nhau.">Token 3: the stable ascending sort put token 3 first, and a reversal keeps the order of ties.</Translated>,
        why: <Translated vi="Đúng là vậy, cho đến khi việc lật ngược xoay cả danh sách, kể cả các giá trị bằng nhau.">It did, until the reversal turned the whole list round, ties included.</Translated>,
      },
      {
        label: <Translated vi="Cái nào cũng được: như mọi thứ tự giữa các khóa bằng nhau, nó phụ thuộc vào lần chạy.">Either: as with any order among equal keys, it depends on the run.</Translated>,
        why: <Translated vi="Cả hai bước đều tất định.">Both steps are deterministic.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đảo bit các khóa và sắp tăng dần cho thứ tự giảm dần mà các giá trị bằng nhau vẫn theo thứ tự đầu vào, token 3 trước.
            Đó là quy tắc của CUB ("the keys are inverted ... and the resulting keys are sorted in ascending order") và là điều{' '}
            <code>std::stable_sort</code> với <code>&gt;</code> cho ra. Hãy thử cả hai thiết lập trên hàng "logits có giá trị bằng
            nhau".
          </p>
        }
      >
        <p>
          Inverting the keys and sorting ascending gives descending order with ties still in input order, token 3 first.
          That is CUB's rule ("the keys are inverted ... and the resulting keys are sorted in ascending order") and what{' '}
          <code>std::stable_sort</code> with <code>&gt;</code> gives. Try both settings on the "logits with ties" row.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch14-facts.md A3, E1', // i18n-ignore: citation (file path)
  },
  'ch14/14.4/q-zero': {
    kind: 'choice',
    id: 'ch14/14.4/q-zero',
    prompt: (
      <Translated
        vi={
          <>
            Không có bước −0.0, một hàng chứa −0.0 ở token 2 và +0.0 ở token 9. Khi sắp giảm dần (đảo bit khóa, sắp tăng dần ổn
            định), cái nào đứng trước?
          </>
        }
      >
        <>
          Without the −0.0 step, a row holds −0.0 at token 2 and +0.0 at token 9. Sorted descending (keys inverted,
          stable ascending sort), which comes first?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="−0.0 (token 2): các giá trị bằng nhau giữ thứ tự đầu vào trong một phép sắp xếp ổn định.">−0.0 (token 2): equal values keep their input order in a stable sort.</Translated>,
        why: <Translated vi="Chúng bằng nhau khi là float, nhưng khóa của chúng khác nhau nếu thiếu bước này.">They are equal as floats, but their keys differ without the step.</Translated>,
      },
      {
        label: <Translated vi="Tùy vào phép so sánh float của GPU, vốn có thể coi hoặc không coi hai số không là bằng nhau.">It depends on the GPU's float compare, which may or may not treat the two zeros as equal.</Translated>,
        why: <Translated vi="Phép biến đổi là số học số nguyên; nó cho cùng các khóa ở mọi nơi.">The transform is integer arithmetic; it gives the same keys everywhere.</Translated>,
      },
      {
        label: <Translated vi="+0.0 (token 9): khóa của nó, 0x80000000, lớn hơn 0x7FFFFFFF của −0.0.">+0.0 (token 9): its key, 0x80000000, is above −0.0's 0x7FFFFFFF.</Translated>,
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Có bước này, −0.0 thành +0.0 trước phép biến đổi, cả hai nhận khóa <code>0x80000000</code>, và phép sắp xếp ổn định
            giữ token 2 trước, như <code>std::stable_sort</code> với <code>&gt;</code>. CUB và rocPRIM có bước này; khóa top-k của
            llama.cpp và khóa CUDA, MPS của PyTorch thì không, theo cách chúng tôi đọc mã nguồn của chúng.
          </p>
        }
      >
        <p>
          With the step, −0.0 becomes +0.0 before the transform, both get the key <code>0x80000000</code>, and the stable
          sort keeps token 2 first, as <code>std::stable_sort</code> with <code>&gt;</code> does. CUB and rocPRIM take this
          step; llama.cpp's top-k keys and PyTorch's CUDA and MPS keys do not, by our reading of their source.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch14-facts.md A3, B2, D2, D5, D6, E1', // i18n-ignore: citation (file path)
  },
};

const RAW = [-2.0, -1.0, 0.5, 1.0];

export function S144() {
  const q = questionsS144;
  const metal = loaded(useLabFile(LAB3_METAL));
  const lab3Status = useSolutionStatus('ch14/lab3_sortrows');
  const metalLines = metal
    ? metal.split('\n').flatMap((l, k) => (/as_type<\s*uint\s*>|0x80000000u|~key\b/.test(l) ? [k + 1] : []))
    : [];
  const raw = rawAscending(RAW);
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi phép sắp xếp từ đầu chương đến giờ đều so sánh khóa. Radix sort thì không bao giờ. Nó nhìn một chữ số của mọi
            khóa, chia các khóa vào các bucket theo chữ số đó, rồi lặp lại với chữ số tiếp theo cho đến khi dùng hết. Chính việc
            không so sánh cho phép nó vượt qua cận O(N log N) của các phép sắp xếp có so sánh.
          </p>
        }
      >
        <p>
          Every sort so far in this chapter compares keys. Radix sort never does. It looks at one digit of every key,
          deals the keys into buckets by that digit, and repeats with the next digit until it has used them all. Not
          comparing is what lets it beat the O(N log N) bound of comparison sorts.
        </p>
      </Translated>
      <Source section="14.4" pages="334–335" />
      <Translated
        vi={
          <p>
            Với khóa nhị phân, radix tiện nhất là một lũy thừa của hai, để một chữ số là một lát bit cố định. Chương bắt đầu với
            radix nhỏ nhất: radix 1 bit, hai bucket mỗi lượt.
          </p>
        }
      >
        <p>
          With binary keys the convenient radix is a power of two, so that a digit is a fixed slice of bits. The chapter
          starts with the smallest: a 1-bit radix, two buckets per pass.
        </p>
      </Translated>
      <Source pages="335" />

      <h2>
        <Translated vi="Bốn lượt qua bốn bit">Four passes over four bits</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ví dụ của sách sắp xếp mười sáu khóa 4 bit từ bit có nghĩa nhỏ nhất trở lên. Lượt 1 đưa mọi khóa có bit 0 bằng 0 lên
            đầu và mọi khóa có bit 0 bằng 1 về cuối, mỗi nhóm theo thứ tự đến. Lượt 2 làm tương tự với bit 1, trên đầu ra của lượt
            1; rồi bit 2, rồi bit 3. Sau lượt k, danh sách đúng thứ tự theo k bit thấp nhất, nên sau lượt 4 nó đã được sắp xếp.
          </p>
        }
      >
        <p>
          The book's example sorts sixteen 4-bit keys from the least significant bit up. Pass 1 sends every key whose
          bit 0 is 0 to the front and every key whose bit 0 is 1 to the back, each group in the order it arrived. Pass 2
          does the same with bit 1, on pass 1's output; then bit 2, then bit 3. After pass k the list is in order by its
          low k bits, so after pass 4 it is sorted.
        </p>
      </Translated>
      <Source section="14.4" pages="335–336" figs={tr('Fig. 14.4', 'Hình 14.4')} />
      <Ask q={q['ch14/14.4/q-pass2']} label={tr('Work it out, then check it in the stepper', 'Tự tính, rồi kiểm tra trong trình chạy từng bước')} />
      <Translated
        vi={
          <p>
            Trước mỗi lượt, trình chạy từng bước tô sáng một khóa và hỏi nó sẽ rơi vào đâu. Hãy đếm các khóa đi vào những bucket
            đứng trước, rồi đếm các khóa trong bucket của chính nó đứng trước nó.
          </p>
        }
      >
        <p>
          Before each pass the stepper highlights a key and asks where it will land. Count the keys that go to earlier
          buckets, then the keys of its own bucket that are ahead of it.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trình chạy từng bước di chuyển nguyên các khóa qua từng lượt. Nó không nói gì về luồng hay bộ nhớ: mục 14.5 và 14.6
              mới nói. Một lượt "không ổn định" ở đây điền bucket trên cùng theo chiều ngược, một cách cụ thể để không ổn định.
            </p>
          }
        >
          <p>
            The stepper moves whole keys pass by pass. It says nothing about threads or memory: sections 14.5 and 14.6
            do. An "unstable" pass here fills its top bucket backwards, one specific way of not being stable.
          </p>
        </Translated>
      </Simplification>
      <LsdStepper />

      <h2>
        <Translated vi="Vì sao mọi lượt phải ổn định">Why every pass must be stable</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lượt 2 chỉ xây tiếp được trên lượt 1 nếu nó giữ những gì lượt 1 đã làm. Hai khóa có cùng bit 1 đã được lượt 1 xếp thứ
            tự theo bit 0; một lượt giữ mỗi bucket theo thứ tự đến sẽ để nguyên chúng như vậy. Đó là toàn bộ lập luận về tính đúng
            đắn: một lượt ổn định mang theo thứ tự của mọi thứ trước nó, và chữ số của chính nó trở thành chữ số có nghĩa lớn nhất
            đã được sắp cho đến lúc đó.
          </p>
        }
      >
        <p>
          Pass 2 can only build on pass 1 if it keeps what pass 1 did. Two keys with the same bit 1 were put in order by
          bit 0 in pass 1; a pass that keeps each bucket in arrival order leaves them that way. That is the whole
          correctness argument: a stable pass carries forward the order of everything before it, and its own digit
          becomes the most significant one sorted so far.
        </p>
      </Translated>
      <Source section="14.4" pages="335" />
      <LsdPassesScene />
      <Translated
        vi={
          <p>
            Trong trình chạy từng bước, hãy làm lượt 4 không ổn định và chạy tất cả các lượt. Rồi thay vào đó làm lượt 1 không ổn
            định. Các khóa ra đã sắp xếp, vì lượt 1 không có thứ tự nào trước đó để đánh mất; chỉ hai khóa <code>1001</code> đổi
            chỗ cho nhau, điều mà một danh sách chỉ có khóa không cho thấy được còn một danh sách các cặp (logit, id token) thì có.
            Một bộ lấy mẫu cần tính ổn định đó vì một lý do thứ hai, của khóa học này chứ không phải của sách: khi đó các xác suất
            bằng nhau ra theo một thứ tự cố định, tái lập được.
          </p>
        }
      >
        <p>
          In the stepper, make pass 4 unstable and run all the passes. Then make pass 1 unstable instead. The keys come
          out sorted, because pass 1 had no earlier order to lose; only the two <code>1001</code>s swap places, which a
          list of bare keys cannot show and a list of (logit, token id) pairs would. A sampler needs that stability for a
          second reason, ours rather than the book's: equal probabilities then come out in a fixed, reproducible
          order.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Thứ tự các lượt quan trọng vì cùng lý do đó. Lượt cuối có tiếng nói quyết định, nên nó phải đọc chữ số có nghĩa lớn
            nhất: bắt đầu từ chữ số có nghĩa nhỏ nhất. Đánh dấu "bit có nghĩa lớn nhất trước" và dự đoán kết quả trước khi chạy.
          </p>
        }
      >
        <p>
          The order of the passes matters for the same reason. The last pass has the final word, so it must read the most
          significant digit: least significant first. Tick "most significant bit first" and predict the result before you
          run it.
        </p>
      </Translated>
      <Ask q={q['ch14/14.4/q-msb']} />
      <Translated
        vi={
          <p>
            Chữ số rộng hơn nghĩa là ít lượt hơn. Với 2 bit mỗi lượt, mỗi lượt bốn bucket, cùng các khóa đó được sắp xếp trong hai
            lượt; chuyển trình chạy từng bước sang 2 bit và so với Hình 14.10. Mục 14.7 cân nhắc chữ số rộng hơn tốn gì.
          </p>
        }
      >
        <p>
          A wider digit means fewer passes. With 2 bits per pass, four buckets each, the same keys sort in two passes;
          switch the stepper to 2 bits and compare with Fig. 14.10. Section 14.7 weighs what the wider digit costs.
        </p>
      </Translated>
      <Source section="14.7" pages="342" figs={tr('Fig. 14.10', 'Hình 14.10')} />

      <h2>
        <Translated vi="Một lượt là một phép phân hoạch ổn định">A pass is a stable partition</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bộ lọc ổn định của chương 12 đưa các khóa vượt qua một phép kiểm tra lên đầu đầu ra, theo thứ tự đầu vào, và bỏ phần
            còn lại. Một lượt radix giữ cả phần còn lại: khóa có bit bằng 0 lên đầu theo thứ tự đầu vào, khóa có bit bằng 1 về cuối
            theo thứ tự đầu vào. Sách gọi đây là mẫu phân hoạch ổn định (stable partition), và cách song song hóa cũng như tối ưu
            nó bám sát bộ lọc ổn định. Đặc biệt, đích đến của mọi khóa đến từ một phép scan loại trừ trên các bit:{' '}
            <a href={href('ch14', '14.5')}>mục 14.5</a>.
          </p>
        }
      >
        <p>
          Chapter 12's stable filter moves the keys that pass a test to the front of the output, in input order, and
          drops the rest. A radix pass keeps the rest too: keys whose bit is 0 go to the front in input order, keys whose
          bit is 1 go to the back in input order. The book calls this the stable partition pattern, and its
          parallelisation and optimisation follow the stable filter's closely. In particular, every key's destination
          comes from one exclusive scan of the bits: <a href={href('ch14', '14.5')}>section 14.5</a>.
        </p>
      </Translated>
      <Source section="14.4" pages="336" />

      <h2>
        <Translated vi="Sắp xếp logits">Sorting logits</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Radix sort làm việc trên số nguyên không dấu. Một bộ lấy mẫu sắp xếp logits hoặc xác suất, vốn là float, và muốn giá
            trị lớn nhất đứng trước. Đọc 32 bit của một float như một số nguyên không dấu không tốn gì. Câu hỏi là các số nguyên đó
            có ra theo thứ tự của các float hay không.
          </p>
        }
      >
        <p>
          Radix sort works on unsigned integers. A sampler sorts logits or probabilities, which are floats, and wants the
          largest first. Reading a float's 32 bits as an unsigned integer costs nothing. The question is whether those
          integers come out in the floats' order.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Từ đây đến cuối trang là của khóa học này. Sách chỉ cảnh báo rằng các phép sắp xếp không so sánh "may not generalize to
              arbitrary types of keys" (có thể không tổng quát hóa được cho mọi kiểu khóa) (p. 330).
            </p>
          }
        >
          <p>
            From here to the end of the page is ours. The book only warns that non-comparison sorts "may not generalize
            to arbitrary types of keys" (p. 330).
          </p>
        </Translated>
      </Beyond>
      <Predict
        id="ch14/14.4/p-raw-order"
        prompt={
          <Translated
            vi={
              <>
                Sắp xếp −2.0, −1.0, 0.5 và 1.0 theo bit thô của chúng, đọc như số nguyên không dấu, tăng dần. −1.0 kết thúc ở vị trí
                nào (0 đến 3)?
              </>
            }
          >
            <>
              Sort −2.0, −1.0, 0.5 and 1.0 by their raw bits, read as unsigned integers, ascending. At which position
              (0 to 3) does −1.0 end up?
            </>
          </Translated>
        }

      >
        {(v) => (
          <>
            <Translated
              vi={
                <p>
                  Nó kết thúc ở vị trí <b>{raw.indexOf(-1)}</b>{v === raw.indexOf(-1) ? ', đúng như bạn nói' : ''}. Các bit đã sắp
                  xếp: {raw.map((x) => `${show(bitsOf(x))} = ${hex(bitsOf(x))}`).join(', ')}.
                </p>
              }
            >
              <p>
                It ends up at position <b>{raw.indexOf(-1)}</b>{v === raw.indexOf(-1) ? ', as you said' : ''}. The sorted
                bits: {raw.map((x) => `${show(bitsOf(x))} = ${hex(bitsOf(x))}`).join(', ')}.
              </p>
            </Translated>
            <Translated
              vi={
                <p>
                  Các float dương xếp đúng với nhau: số mũ nằm trên phần định trị, nên độ lớn lớn hơn có bit lớn hơn. Nhưng mọi float âm
                  đều có bit cao nhất được đặt, nên tất cả chúng đứng sau mọi số dương, và giữa chúng với nhau thì chạy ngược, vì độ lớn
                  của chúng vẫn tăng theo bit.
                </p>
              }
            >
              <p>
                Positive floats are fine among themselves: the exponent sits above the mantissa, so a larger magnitude has
                larger bits. But every negative float has its top bit set, so all of them come after all the positives,
                and among themselves they run backwards, because their magnitudes still grow with their bits.
              </p>
            </Translated>
          </>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Cách sửa có hai quy tắc. Đặt bit dấu của một giá trị không âm, để nó rơi phía trên mọi số âm. Đảo mọi bit của một giá
            trị âm, để độ lớn lớn hơn cho khóa nhỏ hơn. Đọc như số nguyên không dấu, kết quả theo đúng thứ tự của các float, kể cả
            vô cực và số subnormal.
          </p>
        }
      >
        <p>
          The fix has two rules. Set the sign bit of a non-negative value, so that it lands above every negative. Invert
          every bit of a negative value, so that a larger magnitude gives a smaller key. Read as unsigned integers, the
          results are in the order of the floats, infinities and subnormals included.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md, A3, B2, E1)', '2026-11-20 (notes/today/ch14-facts.md, mục A3, B2, E1)')}>
        <Translated
          vi={
            <p>
              CUB ghi trong tài liệu đúng điều này cho khóa float: "For positive floating point values, the sign bit is inverted. For
              negative floating point values, the full key is inverted." (với giá trị dương, bit dấu bị đảo; với giá trị âm, cả khóa
              bị đảo). rocPRIM dùng cùng phép biến đổi.
            </p>
          }
        >
          <p>
            CUB documents exactly this for float keys: "For positive floating point values, the sign bit is inverted. For
            negative floating point values, the full key is inverted." rocPRIM uses the same transform.
          </p>
        </Translated>
      </Today>
      <Translated vi={<p>Một bộ lấy mẫu cần thêm hai bước, và một quyết định.</p>}>
        <p>A sampler needs two more steps, and one decision.</p>
      </Translated>
      <ul>
        <Translated
          vi={
            <li>
              <b>Bước −0.0.</b> −0.0 và +0.0 bằng nhau khi là float, nhưng bit của chúng khác nhau, và phép biến đổi giữ chúng tách
              biệt: −0.0 thành <code>0x7FFFFFFF</code>, +0.0 thành <code>0x80000000</code>. Nên +0.0 luôn xếp trên −0.0, bất kể thứ
              tự của chúng trong hàng. Hãy ánh xạ −0.0 thành +0.0 trước khi trích chữ số.
            </li>
          }
        >
          <li>
            <b>The −0.0 step.</b> −0.0 and +0.0 are equal as floats, but their bits differ, and the transform keeps them
            apart: −0.0 becomes <code>0x7FFFFFFF</code>, +0.0 becomes <code>0x80000000</code>. So +0.0 always sorts
            above −0.0, whatever their order in the row. Map −0.0 to +0.0 before extracting digits.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Giảm dần bằng cách đảo bit.</b> Có giá trị lớn nhất trước bằng cách đảo bit khóa và sắp tăng dần, chứ không phải
              bằng cách lật ngược một kết quả tăng dần. Lật ngược cũng lật luôn thứ tự của các giá trị bằng nhau.
            </li>
          }
        >
          <li>
            <b>Descending by inversion.</b> Get the largest first by inverting the key and sorting ascending, not by
            reversing an ascending result. Reversal also reverses the order of ties.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>NaN.</b> CUB không áp dụng quy tắc đặc biệt nào: sau phép biến đổi, +NaN xếp trên +inf và −NaN xếp dưới −inf. MLX
              đặt NaN ở cuối, còn PyTorch cho NaN một khóa cố định riêng. Các lab của khóa học này dùng quy tắc của CUB, và trình
              khám phá lấy nó làm tham chiếu cho các hàng có NaN.
            </li>
          }
        >
          <li>
            <b>NaN.</b> CUB applies no special rule: after the transform +NaN sorts above +inf and −NaN below −inf. MLX
            puts NaN at the end, and PyTorch gives NaN a fixed key of its own. This course's labs use CUB's rule, and the
            explorer takes it as the reference for rows with NaN.
          </li>
        </Translated>
      </ul>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md, A3, B2, D2, D4 to D6)', '2026-11-20 (notes/today/ch14-facts.md, mục A3, B2, D2, D4 đến D6)')}>
        <Translated
          vi={
            <p>
              CUB và rocPRIM có bước −0.0. Khóa top-k của llama.cpp và khóa CUDA, MPS của PyTorch thì không, theo cách chúng tôi đọc
              mã nguồn của chúng. MLX ghi trong tài liệu "NaN values are placed at the end" (giá trị NaN được đặt ở cuối); các hàm
              tạo khóa của PyTorch kiểm tra NaN và trả về một hằng số.
            </p>
          }
        >
          <p>
            CUB and rocPRIM take the −0.0 step. llama.cpp's top-k keys and PyTorch's CUDA and MPS keys do not, by our
            reading of their source. MLX documents "NaN values are placed at the end"; PyTorch's key functions test for
            NaN and return a constant.
          </p>
        </Translated>
      </Today>
      <FloatExplorer />
      <Ask q={q['ch14/14.4/q-negatives']} />
      <Ask q={q['ch14/14.4/q-ties-desc']} />
      <Ask q={q['ch14/14.4/q-zero']} />
      <Translated
        vi={
          <p>
            Đây là phép biến đổi dưới dạng một hàm device, cùng một kernel áp dụng nó lên một hàng. Nó nhận bit của float dưới dạng{' '}
            <code>unsigned int</code> và không bao giờ giữ một float, vì những lý do mà bảng tiếp theo đưa ra.
          </p>
        }
      >
        <p>
          Here is the transform as a device function, with a kernel that applies it to a row. It takes the float's bits
          as an <code>unsigned int</code> and never holds a float, for reasons the next table gives.
        </p>
      </Translated>
      <Code
        tabs={[{ label: tr('CUDA (and HIP)', 'CUDA (và HIP)'), code: FLOATKEY, status: SNIPPET_STATUS() }]}
        caption={
          <Translated
            vi={
              <>
                Từ <code>labs/ch14/snippets/s144_floatkey.cu</code>. Bài kiểm thử đơn vị của nó chạy nó trên trình giả lập CPU của
                khóa học với các giá trị E1, hai NaN và 600 mẫu bit khác, rồi so mọi khóa với mô hình, đến từng bit.
              </>
            }
          >
            <>
              From <code>labs/ch14/snippets/s144_floatkey.cu</code>. Its unit test runs it on the course's CPU emulator over
              E1's values, two NaNs and 600 other bit patterns and compares every key with the model, bit for bit.
            </>
          </Translated>
        }
      />

      <h2>
        <Translated vi="Trên từng backend">On each backend</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lấy được bit của một float là phần duy nhất khác nhau, và trên Metal các quy tắc về float quyết định cách làm.
          </p>
        }
      >
        <p>
          Getting at a float's bits is the only part that differs, and on Metal the float rules decide how to do it.
        </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-part="backends">
        <thead>
          <tr><th><Translated vi="Backend">Backend</Translated></th><th><Translated vi="Các bit của float">The float's bits</Translated></th><th><Translated vi="Cần lưu ý gì">What to watch</Translated></th></tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <td><code>__float_as_uint(x)</code>, <Translated vi="hoặc">or</Translated> <code>cuda::std::bit_cast&lt;unsigned int&gt;(x)</code></td>
            <td><Translated vi="Cả hai sao chép nguyên các bit (ch14-facts A8).">Both copy the bits unchanged (ch14-facts A8).</Translated></td>
          </tr>
          <tr>
            <td>HIP</td>
            <td><code>__float_as_uint(x)</code>, <Translated vi="cùng cách viết">same spelling</Translated></td>
            <td><Translated vi="Nó có trong header device của HIP nhưng không có trong tài liệu tham khảo toán học AMD công bố (B6).">It is in HIP's device header but not in AMD's published math reference (B6).</Translated></td>
          </tr>
          <tr>
            <td>Metal</td>
            <td>
              <code>as_type&lt;uint&gt;(x)</code>:{' '}
              <Translated vi={'"Các bit của toán hạng được trả về trực tiếp, không sửa đổi"'}>"The bits in the operand are returned directly without modification"</Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    Cả fast math và relaxed math đều cho phép "no signed zeros", và fast math để NaN và INF không xác định.
                    Hãy làm phép kiểm tra −0.0 và phép biến đổi trên <code>uint</code>, hoặc bind buffer dưới dạng{' '}
                    <code>device const uint*</code> và không bao giờ giữ một float (C3).
                  </>
                }
              >
                Fast and relaxed math both allow "no signed zeros", and fast math leaves NaN and INF undefined. Do the
                −0.0 test and the transform on the <code>uint</code>, or bind the buffer as <code>device const uint*</code>{' '}
                and never hold a float (C3).
              </Translated>
            </td>
          </tr>
          <tr>
            <td><Translated vi="Các lab">The labs</Translated></td>
            <td><Translated vi="đọc buffer float qua một góc nhìn không dấu trên mọi backend">read the float buffer through an unsigned view on every backend</Translated></td>
            <td><Translated vi="Cách này hoàn toàn không cần intrinsic bit-cast nào, và trên Metal nó giữ mọi phép kiểm tra ở phía số nguyên.">It needs no bit-cast intrinsic at all, and on Metal it keeps every test on the integer side.</Translated></td>
          </tr>
        </tbody>
      </table>
      {metal ? (
        <Code
          tabs={[{ label: tr('Metal (lab 14.3)', 'Metal (lab 14.3)'), code: metal, highlight: metalLines, status: lab3Status.metal }]}
          caption={
            <Translated
              vi={
                <>
                  Kernel Metal của lab 14.3, nạp từ <code>labs/{LAB3_METAL}</code>. Các dòng được đánh dấu là phép biến đổi khóa của
                  nó{metal.includes('device const uint* probs') ? <>: nó bind các xác suất dưới dạng <code>device const uint*</code>, nên không bao giờ giữ một float</> : metal.includes('as_type') ? <>: nó đọc bit của mỗi float bằng <code>as_type&lt;uint&gt;</code></> : null}.
                  Hãy thử làm <a href={href('ch14', 'labs')}>lab</a> trước khi đọc nó.
                </>
              }
            >
              <>
                Lab 14.3's Metal kernel, loaded from <code>labs/{LAB3_METAL}</code>. The marked lines are its key
                transform{metal.includes('device const uint* probs') ? <>: it binds the probabilities as <code>device const uint*</code>, so no float is ever held</> : metal.includes('as_type') ? <>: it reads each float's bits with <code>as_type&lt;uint&gt;</code></> : null}.
                Try the <a href={href('ch14', 'labs')}>lab</a> before you read it.
              </>
            </Translated>
          }
        />
      ) : null}
      <Translated
        vi={
          <p>
            Phép biến đổi này là dòng đầu tiên trong phép sắp xếp của một bộ lấy mẫu trên GPU. Top-k CUDA của llama.cpp tính nó
            trong một hàm tên <code>top_k_float_to_ordered</code>, và top-k Metal của nó làm điều tương tự với{' '}
            <code>as_type&lt;uint&gt;</code>; cả hai đều không có bước −0.0.
          </p>
        }
      >
        <p>
          This transform is the first line of a GPU sampler's sort. llama.cpp's CUDA top-k computes it in a function
          called <code>top_k_float_to_ordered</code>, and its Metal top-k does the same with <code>as_type&lt;uint&gt;</code>;
          neither takes the −0.0 step.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch14-facts.md, D2)', '2026-11-20 (notes/today/ch14-facts.md, mục D2)')}>
        <Translated
          vi={
            <p>
              Đọc trên nhánh master của llama.cpp: <code>ggml/src/ggml-cuda/top-k.cu</code> và{' '}
              <code>ggml/src/ggml-metal/kernels/argsort.metal</code>. Cách chúng tôi đọc: không cái nào gộp −0.0 với +0.0, nên ở đó
              +0.0 xếp trên −0.0, khác với CUB.
            </p>
          }
        >
          <p>
            Read on llama.cpp's master branch: <code>ggml/src/ggml-cuda/top-k.cu</code> and{' '}
            <code>ggml/src/ggml-metal/kernels/argsort.metal</code>. Our reading: neither merges −0.0 with +0.0, so +0.0
            ranks above −0.0 there, unlike CUB.
          </p>
        </Translated>
      </Today>
    </>
  );
}
