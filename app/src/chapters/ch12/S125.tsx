import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import { functionSource } from '../../sim/toolchain';
import {
  BOOK_KEEP,
  BOOK_KEYS,
  blockBinaryScan,
  cutIndex,
  earlierKeepers,
  exclusiveScan,
  isPrefix,
  keepAtLeast,
  keepTopP,
  lanesBelow,
  makeLogits,
  makeSortedProbs,
  maskBits,
  massBefore,
  minPThreshold,
  total,
  traffic,
} from '../../sim/stablescan';

// Section 12.5: the stable filter by scan (Figs. 12.7 and 12.8, pp. 297-298). The explorer
// and the block stepper run on src/sim/stablescan.ts.

const TIERS: Record<string, Tier> = {
  input: 'global', keep: 'global', offset: 'global', output: 'global', outputSize: 'global',
  i: 'reg', gid: 'reg', n: 'reg',
};

type Skin = 'keys' | 'logits' | 'topp';
const SKIN_LABEL: Record<Skin, string> = {
  get keys() {
    return tr('keys (Fig. 12.7)', 'khóa (Hình 12.7)');
  },
  get logits() {
    return tr('logits, min-p', 'logits, min-p');
  },
  get topp() {
    return tr('sorted probabilities, top-p', 'xác suất đã sắp xếp, top-p');
  },
};
const LOGITS = makeLogits();
const PROBS = makeSortedProbs();
const MASS = massBefore(PROBS);

function Cells({ label, values, cls, onClick, data, width = '2.1rem', title }: {
  label: string;
  values: string[];
  cls?: (i: number) => string;
  onClick?: (i: number) => void;
  data?: string;
  width?: string;
  title?: (i: number) => string;
}) {
  return (
    <div className="row" style={{ gap: '0.5rem', flexWrap: 'nowrap', marginBottom: '0.35rem' }}>
      <div className="ui muted" style={{ width: '4.6rem', flex: 'none', fontSize: '0.78rem' }}>{label}</div>
      <div className="mx global" style={{ gridTemplateColumns: `repeat(${values.length}, ${width})` }} data-row={data}>
        {values.map((v, i) => (
          <span
            key={i}
            className={cls?.(i) ?? ''}
            data-i={i}
            title={title?.(i)}
            role={onClick ? 'button' : undefined}
            tabIndex={onClick ? 0 : undefined}
            style={onClick ? { cursor: 'pointer' } : undefined}
            onClick={onClick ? () => onClick(i) : undefined}
            onKeyDown={onClick ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(i); } } : undefined}
          >
            {v}
          </span>
        ))}
      </div>
    </div>
  );
}

function Explorer() {
  const [skin, setSkin] = useState<Skin>('keys');
  const [flags, setFlags] = useState<number[]>([...BOOK_KEEP]);
  const [minp, setMinp] = useState(0.1);
  const [topp, setTopp] = useState(0.8);
  const [sel, setSel] = useState<number | null>(null);

  const threshold = minPThreshold(LOGITS, minp);
  const keep = skin === 'keys' ? flags : skin === 'logits' ? keepAtLeast(LOGITS, threshold) : keepTopP(PROBS, topp);
  const offsets = exclusiveScan(keep);
  const tot = total(keep, offsets);
  const n = keep.length;
  const inputs = skin === 'keys' ? [...BOOK_KEYS] : skin === 'logits' ? LOGITS.map((l) => l.toFixed(1)) : PROBS.map((p) => p.toFixed(3));
  const name = (i: number) => (skin === 'keys' ? BOOK_KEYS[i] : `t${i}`);
  const output = Array.from({ length: tot }, () => '');
  keep.forEach((k, i) => {
    if (k) output[offsets[i]] = skin === 'keys' ? BOOK_KEYS[i] : `t${i}`;
  });
  const behind = sel != null && keep[sel] ? new Set(earlierKeepers(keep, sel)) : new Set<number>();
  const width = skin === 'topp' ? '2.7rem' : '2.1rem';

  return (
    <Bench
      title={tr('Keep, scan, scatter', 'Giữ, scan, rải')}
      task={tr(
        'Click a kept key to see which earlier keys its offset counts. Then switch to top-p and look at the shape of the keep row.',
        'Bấm vào một khóa được giữ để xem offset của nó đếm những khóa nào đứng trước. Sau đó chuyển sang top-p và nhìn hình dạng của hàng keep.',
      )}
      foot={
        <Translated
          vi={
            <>
              Hàng offset là scan loại trừ của hàng keep. Nó được tính cho mọi khóa, nhưng chỉ các khóa được giữ dùng offset
              của mình (các ô xám không bao giờ được đọc). Logits là một hàng có seed gồm {n} token theo thứ tự id; các xác
              suất là một softmax có seed, sắp xếp giảm dần. Mô hình: src/sim/stablescan.ts.
            </>
          }
        >
          <>
            The offset row is the exclusive scan of the keep row. It is computed for every key, but only kept keys use theirs
            (the greyed ones are never read). The logits are a seeded row of {n} tokens in id order; the probabilities are a
            seeded softmax, sorted in descending order. Model: src/sim/stablescan.ts.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Skin', 'Giao diện')}>
          {(['keys', 'logits', 'topp'] as Skin[]).map((s) => (
            <button key={s} aria-pressed={skin === s} onClick={() => { setSkin(s); setSel(null); }}>{SKIN_LABEL[s]}</button>
          ))}
        </span>
        {skin === 'keys' ? (
          <button className="quiet" onClick={() => setFlags([...BOOK_KEEP])}><Translated vi="Các cờ của sách">The book's flags</Translated></button>
        ) : skin === 'logits' ? (
          <label className="field">
            <Translated vi={<>min-p: p = {minp.toFixed(2)}, giữ logit ≥ max + ln p = {threshold.toFixed(2)}</>}>
              min-p: p = {minp.toFixed(2)}, keep logit ≥ max + ln p = {threshold.toFixed(2)}
            </Translated>
            <input type="range" min="1" max="100" value={Math.round(minp * 100)} aria-label="min-p" /* i18n-ignore: parameter name */ onChange={(e) => setMinp(Number(e.target.value) / 100)} />
          </label>
        ) : (
          <label className="field">
            <Translated vi={<>top-p: p = {topp.toFixed(2)}, giữ khi khối lượng xác suất trước một token còn dưới p</>}>
              top-p: p = {topp.toFixed(2)}, keep while the mass before a token is below p
            </Translated>
            <input type="range" min="1" max="100" value={Math.round(topp * 100)} aria-label="top-p" /* i18n-ignore: parameter name */ onChange={(e) => setTopp(Number(e.target.value) / 100)} />
          </label>
        )}
      </div>

      <Cells label={tr('index', 'chỉ số')} values={keep.map((_, i) => String(i))} width={width} data="index" cls={() => 'void'} /* i18n-ignore: row key */ />
      <Cells
        label={skin === 'keys' ? tr('input', 'đầu vào') : skin === 'logits' ? 'logit' : tr('probability', 'xác suất')}
        values={inputs}
        width={width}
        data="input" // i18n-ignore: row key
        onClick={(i) => setSel(sel === i ? null : i)}
        cls={(i) => (sel === i ? 'hot' : behind.has(i) ? 'good' : keep[i] ? 'on' : '')}
        title={(i) => tr(`${name(i)}: click to see its offset's earlier keepers`, `${name(i)}: bấm để xem các khóa được giữ đứng trước mà offset của nó đếm`)}
      />
      {skin === 'topp' ? (
        <Cells label={tr('mass before', 'khối lượng trước')} values={MASS.map((m) => m.toFixed(3))} width={width} data="mass" cls={(i) => (keep[i] ? 'on' : '')} /* i18n-ignore: row key */ />
      ) : null}
      <Cells
        label={tr('keep', 'giữ')}
        values={keep.map(String)}
        width={width}
        data="keep" // i18n-ignore: row key
        onClick={skin === 'keys' ? (i) => setFlags((f) => f.map((x, j) => (j === i ? 1 - x : x))) : undefined}
        cls={(i) => (keep[i] ? 'on' : '')}
        title={skin === 'keys' ? () => tr('click to flip', 'bấm để lật') : undefined}
      />
      <Cells
        label={tr('offset', 'offset')}
        values={offsets.map(String)}
        width={width}
        data="offset" // i18n-ignore: row key
        cls={(i) => (sel === i ? 'hot' : keep[i] ? '' : 'void')}
      />
      <div className="row" style={{ gap: '0.5rem', flexWrap: 'nowrap', marginTop: '0.6rem' }}>
        <div className="ui muted" style={{ width: '4.6rem', flex: 'none', fontSize: '0.78rem' }}><Translated vi="đầu ra">output</Translated></div>
        <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.max(1, tot)}, ${width})` }} data-row="output">
          {tot ? output.map((v, j) => (
            <span key={j} className={sel != null && keep[sel] && offsets[sel] === j ? 'hot' : 'on'} data-j={j}>{v}<sub>{j}</sub></span>
          )) : <span className="void"><Translated vi="trống">empty</Translated></span>}
        </div>
      </div>

      <p className="ui" style={{ margin: '0.9rem 0 0' }} data-total={tot}>
        <Translated vi={<>Kích thước đầu ra = offset cuối + keep cuối = {offsets[n - 1]} + {keep[n - 1]} = <b>{tot}</b>.</>}>
          Output size = last offset + last keep = {offsets[n - 1]} + {keep[n - 1]} = <b>{tot}</b>.
        </Translated>
        {sel != null ? (
          keep[sel] ? (
            <Translated vi={<> {name(sel)} có offset {offsets[sel]}: các khóa được giữ đứng trước nó là {behind.size ? [...behind].map(name).join(', ') : 'không có'}.</>}>
              <> {name(sel)} has offset {offsets[sel]}: the kept keys before it are {behind.size ? [...behind].map(name).join(', ') : 'none'}.</>
            </Translated>
          ) : (
            <Translated vi={<> {name(sel)} không được giữ, nên offset của nó, {offsets[sel]}, không bao giờ được dùng.</>}>
              <> {name(sel)} is not kept, so its offset, {offsets[sel]}, is never used.</>
            </Translated>
          )
        ) : null}
      </p>
      {skin === 'topp' ? (
        <p className="ui" style={{ margin: '0.4rem 0 0' }} data-prefix={isPrefix(keep) ? 'yes' : 'no'}>
          {isPrefix(keep) ? (
            <Translated vi={<>Các token được giữ là {cutIndex(keep)} token đầu tiên, dù p là bao nhiêu: một tiền tố. Tìm chỉ số cắt là tất cả những gì bộ lọc phải làm ở đây.</>}>
              <>The kept tokens are the first {cutIndex(keep)}, whatever p is: a prefix. Finding the cut index is all the filter has to do here.</>
            </Translated>
          ) : (
            <Translated vi={<>Không phải tiền tố.</>}>
              <>Not a prefix.</>
            </Translated>
          )}
        </p>
      ) : null}
    </Bench>
  );
}

const PHASES = () => [
  tr('Nothing has run yet.', 'Chưa có gì chạy.'),
  tr(
    'Each warp votes: every lane, kept or not, contributes one bit, and every lane of the warp receives the whole mask.',
    'Mỗi warp bỏ phiếu: mọi lane, được giữ hay không, đóng góp một bit, và mọi lane của warp nhận được toàn bộ mặt nạ.',
  ),
  tr(
    'Each lane counts the kept lanes below it: the popcount of the mask ANDed with the lanes-below mask. That is its rank inside the warp.',
    'Mỗi lane đếm các lane được giữ đứng dưới nó: popcount của mặt nạ AND với mặt nạ các lane bên dưới. Đó là hạng của nó trong warp.',
  ),
  tr(
    'Lane 0 of each warp writes the warp\'s total, the popcount of the whole mask, to shared memory.',
    'Lane 0 của mỗi warp ghi tổng của warp, tức popcount của cả mặt nạ, vào bộ nhớ chia sẻ (shared memory).',
  ),
  tr(
    'Barrier: every warp\'s total is in shared memory before anyone reads them.',
    'Rào chắn (barrier): tổng của mọi warp đã nằm trong bộ nhớ chia sẻ trước khi bất kỳ ai đọc chúng.',
  ),
  tr(
    'The warp totals are scanned: an exclusive scan of a handful of numbers gives where each warp\'s kept keys start in the block.',
    'Các tổng warp được scan: một scan loại trừ trên vài con số cho biết các khóa được giữ của mỗi warp bắt đầu ở đâu trong khối.',
  ),
  tr(
    'Each lane\'s offset in the block is its warp\'s start plus its rank. Only kept lanes use theirs.',
    'Offset của mỗi lane trong khối là điểm bắt đầu của warp nó cộng với hạng của nó. Chỉ các lane được giữ dùng offset của mình.',
  ),
];

function seeded(n: number, seed: number, f: number): number[] {
  let r = (Math.imul(seed, 2654435761) + 7) >>> 0;
  return Array.from({ length: n }, () => {
    r = (Math.imul(r, 1664525) + 1013904223) >>> 0;
    return (r >>> 8) / 16777216 < f ? 1 : 0;
  });
}

function BlockStepper() {
  const [width, setWidth] = useState<8 | 32>(8);
  const [keepBy, setKeepBy] = useState<Record<number, number[]>>({ 8: seeded(32, 4, 0.45), 32: seeded(128, 9, 0.45) });
  const [phase, setPhase] = useState(0);
  const [sel, setSel] = useState<number>(-1);
  const keep = keepBy[width];
  const s = useMemo(() => blockBinaryScan(keep, width), [keep, width]);
  const plain = exclusiveScan(keep);
  const same = s.offsets.every((o, t) => o === plain[t]);
  const flip = (t: number) => setKeepBy((k) => ({ ...k, [width]: k[width].map((x, j) => (j === t ? 1 - x : x)) }));
  const w = sel >= 0 ? Math.floor(sel / width) : -1;
  const l = sel >= 0 ? sel % width : -1;

  return (
    <Bench
      title={tr('Inside one block: votes, then a scan of warp totals', 'Bên trong một khối: bỏ phiếu, rồi scan các tổng warp')}
      task={tr(
        'Step through the six steps. Then click a lane and check its offset by hand.',
        'Đi qua sáu bước. Sau đó bấm vào một lane và tự kiểm tra offset của nó.',
      )}
      foot={
        <Translated
          vi={
            <>
              Bốn warp. Bấm vào cờ của một lane để lật nó; bấm vào con số của một lane để theo dõi phép tính của nó. Mặt nạ
              được in như sách in, lane 0 ở bên phải. Mô hình: blockBinaryScan trong src/sim/stablescan.ts, được các bài kiểm
              thử so với phép scan thông thường.
            </>
          }
        >
          <>
            Four warps. Click a lane's flag to flip it; click a lane's number to follow its arithmetic. A mask is printed as
            the book prints one, lane 0 on the right. Model: blockBinaryScan in src/sim/stablescan.ts, which the tests check
            against the plain scan.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Warp width', 'Độ rộng warp')}>
          {([8, 32] as const).map((x) => (
            <button key={x} aria-pressed={width === x} onClick={() => { setWidth(x); setSel(-1); }}>
              {x === 8 ? tr('toy: warps of 8 lanes', 'đồ chơi: warp 8 lane') : tr('warps of 32 lanes, 128 threads', 'warp 32 lane, 128 luồng')}
            </button>
          ))}
        </span>
        <button onClick={() => setPhase(0)} disabled={phase === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setPhase(phase - 1)} disabled={phase === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setPhase(phase + 1)} disabled={phase >= PHASES().length - 1}><Translated vi="Bước">Step</Translated></button>
        <span className="muted" aria-live="polite"><Translated vi={<>bước {phase} / {PHASES().length - 1}</>}>step {phase} of {PHASES().length - 1}</Translated></span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }} data-phase={phase}>
        {PHASES()[phase]}
        {sel >= 0 && phase >= 2 ? (
          <>
            <br />
            <span data-lane-arith>
              <Translated
                vi={
                  <>
                    Lane {l} của warp {w}: hạng = popcount(mask &amp; các lane dưới {l}) = popcount({maskBits(s.masks[w] & lanesBelow(l), width)}) ={' '}
                    {s.ranks[sel]}
                    {phase >= 6 ? <>; offset = điểm bắt đầu của warp {w} + hạng = {s.warpStarts[w]} + {s.ranks[sel]} = <b>{s.offsets[sel]}</b>{keep[sel] ? '' : ' (không dùng: lane này không giữ gì)'}</> : null}.
                  </>
                }
              >
                Lane {l} of warp {w}: rank = popcount(mask &amp; lanes below {l}) = popcount({maskBits(s.masks[w] & lanesBelow(l), width)}) ={' '}
                {s.ranks[sel]}
                {phase >= 6 ? <>; offset = start of warp {w} + rank = {s.warpStarts[w]} + {s.ranks[sel]} = <b>{s.offsets[sel]}</b>{keep[sel] ? '' : ' (unused: this lane keeps nothing)'}</> : null}.
              </Translated>
            </span>
          </>
        ) : null}
      </div>
      <div style={{ overflowX: 'auto' }}>
        {Array.from({ length: s.warps }, (_, wi) => (
          <div key={wi} className="row" style={{ gap: '0.6rem', flexWrap: 'nowrap', alignItems: 'flex-start', marginBottom: '0.55rem' }} data-warp={wi}>
            <div className="ui" style={{ width: '4.2rem', flex: 'none', fontSize: '0.78rem', paddingTop: '0.3rem' }}><Translated vi={<>warp {wi}</>}>warp {wi}</Translated>{/* i18n-ignore: warp is kept in English */}</div>
            <div>
              <div className="mx shared" style={{ display: 'grid', gridTemplateColumns: `repeat(${width}, ${width === 8 ? '2.1rem' : '1.05rem'})` }}>
                {Array.from({ length: width }, (_, li) => {
                  const t = wi * width + li;
                  return (
                    <span
                      key={li} role="button" tabIndex={0} aria-pressed={!!keep[t]} data-t={t}
                      className={keep[t] ? 'on' : ''}
                      style={{ cursor: 'pointer', minWidth: width === 8 ? undefined : '1rem', fontSize: width === 8 ? undefined : '0.62rem' }}
                      onClick={() => flip(t)}
                      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); flip(t); } }}
                      title={tr(`lane ${li}: keep = ${keep[t]} (click to flip)`, `lane ${li}: keep = ${keep[t]} (bấm để lật)`)}
                    >
                      {keep[t]}
                    </span>
                  );
                })}
              </div>
              {phase >= 2 ? (
                <div className="mx" style={{ display: 'grid', gridTemplateColumns: `repeat(${width}, ${width === 8 ? '2.1rem' : '1.05rem'})`, marginTop: 2 }} data-ranks={wi}>
                  {Array.from({ length: width }, (_, li) => {
                    const t = wi * width + li;
                    const v = phase >= 6 ? s.offsets[t] : s.ranks[t];
                    return (
                      <span
                        key={li} data-t={t}
                        className={sel === t ? 'hot' : keep[t] ? '' : 'void'}
                        style={{ cursor: 'pointer', minWidth: width === 8 ? undefined : '1rem', fontSize: width === 8 ? undefined : '0.62rem' }}
                        onClick={() => setSel(sel === t ? -1 : t)}
                        title={phase >= 6 ? tr('offset in the block', 'offset trong khối') : tr('rank in the warp', 'hạng trong warp')}
                      >
                        {v}
                      </span>
                    );
                  })}
                </div>
              ) : null}
            </div>
            <div className="ui" style={{ fontSize: '0.74rem', minWidth: '10rem' }}>
              {phase >= 1 ? <div data-mask={wi}><Translated vi="mặt nạ">mask</Translated> <code>{maskBits(s.masks[wi], width)}</code></div> : null}
              {phase >= 3 ? <div data-total={wi}><Translated vi="tổng">total</Translated> <b className="t-shared">{s.warpTotals[wi]}</b></div> : null}
              {phase >= 5 ? <div data-start={wi}><Translated vi="bắt đầu tại">starts at</Translated> <b className="t-shared">{s.warpStarts[wi]}</b></div> : null}
            </div>
          </div>
        ))}
      </div>
      <p className="ui" style={{ margin: '0.4rem 0 0' }}>
        {phase >= 2 && phase < 6 ? tr('Second row: each lane\'s rank in its warp. ', 'Hàng thứ hai: hạng của mỗi lane trong warp của nó. ') : phase >= 6 ? tr('Second row: each lane\'s offset in the block. ', 'Hàng thứ hai: offset của mỗi lane trong khối. ') : ''}
        {phase >= 6 ? (
          <span data-same={same ? 'yes' : 'no'}>
            <Translated vi={<>Tổng của khối {s.total}. Các offset này bằng scan loại trừ thông thường của các cờ: <b>{same ? 'đúng' : 'không'}</b>.</>}>
              Block total {s.total}. These offsets equal the plain exclusive scan of the flags: <b>{same ? 'yes' : 'no'}</b>.
            </Translated>
          </span>
        ) : null}
      </p>
    </Bench>
  );
}

const T1 = traffic('three-kernels', 0.5);
const T2 = traffic('single-pass', 0.5);

export const questionsS125: Record<string, Question> = {
  'ch12/12.5/q-counter': {
    kind: 'choice',
    id: 'ch12/12.5/q-counter',
    prompt: (
      <Translated vi={<>Vì sao một bộ lọc ổn định không thể lấy ô của nó từ <code>fetch_add</code> trên một bộ đếm, ngay cả khi chỉ có một phép atomic mỗi warp?</>}>
        <>Why can a stable filter not take its slot from <code>fetch_add</code> on a counter, even with only one atomic per warp?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Atomic quá chậm: ngay cả một phép mỗi warp cũng tuần tự hóa cả lưới trên bộ đếm, điều mà bộ lọc ổn định không chịu nổi.">
            Atomics are too slow: even one per warp serialises the whole grid on the counter, which a stable filter cannot afford.
          </Translated>
        ),
        why: (
          <Translated vi="Vấn đề không phải tốc độ mà là thứ tự. Một bộ đếm nhanh vẫn phát ô theo thứ tự hoàn thành.">
            Speed is not the problem; order is. A fast counter still hands out slots in completion order.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các atomic hoàn thành không theo thứ tự cố định, nhưng một ô ổn định phải đếm các khóa được giữ đứng trước nó: một phép scan.">
            Atomics complete in no fixed order, but a stable slot must count the kept keys before it: a scan.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có thể, với một atomic mỗi khối: luồng 0 đặt trước phạm vi của khối, và các khối đặt trước theo thứ tự chỉ số.">
            It can, with one atomic per block: thread 0 reserves the block's range, and blocks reserve in index order.
          </Translated>
        ),
        why: (
          <Translated vi="Các khối cũng đặt trước theo thứ tự tùy ý: Hình 12.5 chính là điều đó.">
            Blocks also reserve in arbitrary order: Fig. 12.5 is exactly that.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một bộ đếm trả lời "đến giờ có bao nhiêu luồng đã lấy ô", điều phụ thuộc vào thời điểm. Tính ổn định cần "có bao nhiêu
            khóa được giữ đứng trước tôi trong đầu vào", điều chỉ phụ thuộc vào dữ liệu. Câu thứ hai là scan loại trừ của các cờ
            keep.
          </p>
        }
      >
        <p>
          A counter answers "how many have taken a slot so far", which depends on timing. Stability needs "how many kept keys
          come before me in the input", which depends only on the data. The second is an exclusive scan of the keep flags.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.5, p. 297',
  },
  'ch12/12.5/q-everycall': {
    kind: 'choice',
    id: 'ch12/12.5/q-everycall',
    prompt: (
      <Translated vi={<>Trong Hình 12.8 mọi luồng đều gọi <code>gridExclusiveScan(keep)</code>, kể cả những luồng có khóa không qua kiểm tra. Vì sao phải vậy?</>}>
        <>In Fig. 12.8 every thread calls <code>gridExclusiveScan(keep)</code>, including those whose key fails. Why must they?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không cần: chỉ các luồng giữ khóa mới cần offset, nên các luồng khác có thể return trước phép scan.">
            They need not: only threads that keep a key need an offset, so the others can return before the scan.
          </Translated>
        ),
        why: (
          <Translated vi="Offset của các luồng kia có phụ thuộc vào các số 0 của những luồng này, và một phép scan dựng từ rào chắn sẽ chờ mọi luồng của khối.">
            Their offsets do depend on the zeros of the others, and a scan built from barriers waits for every thread of the block.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Để các khóa bị bỏ cũng được ghi, tại các offset mà phép scan cho chúng.">
            So that the dropped keys are written too, at the offsets the scan gives them.
          </Translated>
        ),
        why: (
          <Translated vi="Các khóa bị bỏ không bao giờ được ghi; chỉ các khóa được giữ dùng offset của mình.">
            Dropped keys are never written; only kept keys use their offsets.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phép scan là tập thể: mỗi offset phía sau đều đếm cờ của luồng này, dù là 0 hay không, và mọi luồng phải tới được các rào chắn của phép scan.">
            A scan is collective: each later offset counts this thread's flag, 0 or not, and every thread must reach the scan's barriers.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bỏ qua lời gọi sẽ hoặc khiến phép scan chờ ở rào chắn những luồng không bao giờ tới, hoặc để lại lỗ hổng trong đầu
            vào của nó. Quy tắc tương tự áp dụng cho các lần bỏ phiếu warp bên dưới: mọi lane đều bỏ phiếu, được giữ hay không,
            trong luồng điều khiển đồng nhất.
          </p>
        }
      >
        <p>
          Skipping the call would either leave the scan waiting at a barrier for threads that never come, or leave holes in
          its input. The same rule applies to the warp votes below: every lane votes, kept or not, in uniform control flow.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.5, p. 298; notes/ch12.md, "Code listings"',
    vi: { source: 'ch. 12 §12.5, p. 298; notes/ch12.md, mục "Code listings"' },
  },
  'ch12/12.5/q-blockrank': {
    kind: 'numeric',
    id: 'ch12/12.5/q-blockrank',
    prompt: (
      <Translated
        vi={
          <>
            Một khối có bốn warp. Các warp giữ 3, 0, 5 và 2 khóa. Một luồng trong warp 2 có hạng trong warp là 4. Offset của nó
            trong đầu ra của khối là bao nhiêu?
          </>
        }
      >
        <>
          A block of four warps. The warps keep 3, 0, 5 and 2 keys. A thread in warp 2 has an in-warp rank of 4. What is its
          offset within the block's output?
        </>
      </Translated>
    ),
    answer: 7,
    explain: (
      <Translated
        vi={
          <p>
            Scan loại trừ của các tổng warp 3, 0, 5, 2 là 0, 3, 3, 8: các khóa được giữ của warp 2 bắt đầu ở 3. Cộng hạng: 3 + 4 =
            7. Warp 1 không giữ gì, nên warp 1 và warp 2 bắt đầu ở cùng một chỗ.
          </p>
        }
      >
        <p>
          The exclusive scan of the warp totals 3, 0, 5, 2 is 0, 3, 3, 8: warp 2's kept keys start at 3. Add the rank: 3 + 4 =
          7. Warp 1 keeps nothing, so warps 1 and 2 start at the same place.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.5, p. 298 (binary prefix sum); ch. 11 §11.4, pp. 261–265',
    vi: { source: 'ch. 12 §12.5, p. 298 (tổng tiền tố nhị phân); ch. 11 §11.4, pp. 261–265' },
  },
  'ch12/12.5/q-ex1-bytes': {
    kind: 'numeric',
    id: 'ch12/12.5/q-ex1-bytes',
    prompt: (
      <Translated
        vi={
          <>
            Ba kernel của bài tập 1 với khóa, cờ và offset 4 byte: kernel 1 đọc mỗi khóa và ghi cờ của nó; kernel 2 đọc các cờ
            và ghi các offset (chỉ đếm phần đó, không tính lưu lượng nội bộ của phép scan); kernel 3 đọc mỗi khóa, cờ và offset
            của nó, rồi ghi các khóa được giữ. Với một nửa số khóa được giữ, mỗi khóa đầu vào tốn bao nhiêu byte lưu lượng bộ
            nhớ toàn cục?
          </>
        }
      >
        <>
          Exercise 1's three kernels with 4-byte keys, flags and offsets: kernel 1 reads each key and writes its flag; kernel 2
          reads the flags and writes the offsets (count only that, not the scan's internal traffic); kernel 3 reads each key,
          its flag and its offset, and writes the kept keys. With half the keys kept, how many bytes of global traffic per
          input key?
        </>
      </Translated>
    ),
    answer: T1.total,
    unit: 'bytes',
    vi: { unit: 'byte', source: 'ch. 12 bài tập 1 và 2, p. 302; src/sim/stablescan.ts (traffic)' },
    explain: (
      <Translated
        vi={
          <p>
            8 + 8 + 12 + 2 = {T1.total}. Một kernel duy nhất (bài tập 2) đọc mỗi khóa một lần và ghi mỗi khóa được giữ một lần: 4
            + 2 = {T2.total} byte, ít hơn {T1.total / T2.total} lần, cho một phép tính bị giới hạn bởi bộ nhớ. Điều đó, hai lần
            khởi chạy nó tiết kiệm, và việc thứ tự các khối của nó cho phép dùng tại chỗ (mục 12.7) là những "ưu điểm chính" mà
            bài tập 2 hỏi tới. Sách không liệt kê chúng; cách tính này là của khóa học này.
          </p>
        }
      >
        <p>
          8 + 8 + 12 + 2 = {T1.total}. A single kernel (exercise 2) reads each key once and writes each kept key once: 4 + 2 ={' '}
          {T2.total} bytes, {T1.total / T2.total} times less, for a computation that is limited by memory. That, the two
          launches it saves, and the fact that its block ordering makes it usable in place (section 12.7) are the "major
          advantages" exercise 2 asks about. The book does not list them; this accounting is ours.
        </p>
      </Translated>
    ),
    source: 'ch. 12 exercises 1 and 2, p. 302; src/sim/stablescan.ts (traffic)',
  },
  'ch12/12.5/q-topp': {
    kind: 'choice',
    id: 'ch12/12.5/q-topp',
    prompt: (
      <Translated
        vi={
          <>
            Với top-p trên các xác suất sắp xếp giảm dần, keep = (khối lượng tích lũy trước tôi) &lt; p. Hàng keep trông như
            thế nào, và điều đó biến bộ lọc thành gì?
          </>
        }
      >
        <>
          For top-p over probabilities sorted in descending order, keep = (cumulative mass before me) &lt; p. What does the
          keep row look like, and what does that make the filter?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một tiền tố: các số 1, rồi các số 0. Việc cần làm là tìm một chỉ số cắt, và sao chép một tiền tố không cần scan.">
            A prefix: ones, then zeros. The job is finding one cut index, and copying a prefix needs no scan.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Các số 1 rải khắp hàng ở bất cứ đâu xác suất riêng của một token lớn: cần một bộ lọc ổn định đầy đủ.">
            Ones scattered through the row wherever a token's own probability is large: a full stable filter is needed.
          </Translated>
        ),
        why: (
          <Translated vi="Khối lượng trước một token chỉ tăng dọc theo một hàng đã sắp xếp, nên khi nó đạt tới p thì nó ở lại đó.">
            The mass before a token only grows along a sorted row, so once it reaches p it stays there.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Một hậu tố.">A suffix.</Translated>,
        why: (
          <Translated vi="Token đầu tiên luôn có khối lượng 0 đứng trước, nên nó luôn được giữ.">
            The first token always has mass 0 before it, so it is always kept.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Giao diện thứ ba của trình khám phá cho thấy điều đó với mọi p. Bộ lọc ổn định tổng quát chứng minh giá trị khi các
            tiêu chí được kết hợp trên một hàng chưa sắp xếp: chẳng hạn top-k và min-p cùng một mặt nạ các token bị cấm, trên bộ
            từ vựng theo thứ tự id.
          </p>
        }
      >
        <p>
          The explorer's third skin shows it for every p. The general stable filter earns its keep when criteria are combined
          on a row that is not sorted: top-k and min-p and a mask of banned tokens, say, over the vocabulary in id order.
        </p>
      </Translated>
    ),
    source: 'notes/ch12.md, "Relevance to LLM-inference kernels"; ch. 11 §11.1, p. 253',
    vi: { source: 'notes/ch12.md, mục "Relevance to LLM-inference kernels"; ch. 11 §11.1, p. 253' },
  },
};

/** A lab's reference-solution status, in the labs page's words (LabKit's solutionTabStatus), from this machine's runs. */
function useLabStatus(lab: string) {
  const { reports } = useReports();
  return {
    cu: solutionTabStatus(reports, lab, 'kernel.cu'),
    metal: solutionTabStatus(reports, lab, 'kernel.metal'),
    mm: solutionTabStatus(reports, lab, 'launch.mm'),
  };
}

export function S125() {
  const status = useLabStatus('ch12/lab3_stable');
  const cu = useLabFile('ch12/lab3_stable/solution/kernel.cu');
  const metal = useLabFile('ch12/lab3_stable/solution/kernel.metal');
  // The provided scan, at the path lab 12.3's kernel.cu names.
  const scanFile = useLabFile('ch12/shared-g12/scan_u32.cu');
  const pick = (src: string | null, names: string[]) => {
    if (!src) return null;
    const parts = names.map((nm) => functionSource(src, nm)?.text).filter((x): x is string => !!x);
    return parts.length ? parts.join('\n\n') : src;
  };
  const scan = scanFile && !scanFile.startsWith('// could not load') ? scanFile : null;
  const scanHeader = scan ? scan.split('\n').filter((_, i, a) => a.slice(0, i + 1).every((x) => x.startsWith('//'))).join('\n') : null;
  const cuPart = pick(cu, ['keepFlagsKernel', 'scatterKernel', 'launchStableFilterFlags']);
  const metalPart = pick(metal, ['keep_flags', 'scatter_flags']);
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi bộ lọc cho đến giờ đều phát ô từ một bộ đếm, và bộ đếm cho ô theo thứ tự các phép atomic hoàn thành. Với bộ lọc
            ổn định thì không được. Một khóa được giữ cần đúng ô của nó, và ô đó chính là số khóa được giữ đứng trước nó trong
            đầu vào. Con số đó chỉ phụ thuộc vào dữ liệu, không phụ thuộc vào thời điểm.
          </p>
        }
      >
        <p>
          Every filter so far hands out slots from a counter, and a counter gives slots in the order the atomics complete.
          For a stable filter that will not do. A kept key needs its exact slot, and its exact slot is the number of kept
          keys that come before it in the input. That number depends only on the data, not on timing.
        </p>
      </Translated>
      <Source section="12.5" pages="297" />
      <Ask q={questionsS125['ch12/12.5/q-counter']} />
      <Translated
        vi={
          <p>
            Tính ổn định quan trọng khi thứ tự mang ý nghĩa. Ví dụ của sách là xóa hoặc trích phần tử khỏi một danh sách đã sắp
            xếp: kết quả vẫn phải được sắp xếp. Trong một bộ lấy mẫu, danh sách ứng viên sắp theo xác suất cũng là trường hợp đó.
          </p>
        }
      >
        <p>
          Stability matters when the order means something. The book's example is deleting or extracting elements from a
          sorted list: the result must still be sorted. In a sampler, a candidate list sorted by probability is the same case.
        </p>
      </Translated>
      <Source section="12.5" pages="297" />

      <h2><Translated vi="Giữ, scan, rải">Keep, scan, scatter</Translated></h2>
      <Translated
        vi={
          <p>
            Công thức có ba bước. Mỗi luồng biến khóa của mình thành một cờ keep, 1 hoặc 0. Một scan loại trừ trên các cờ cho mỗi
            luồng số khóa được giữ đứng trước nó: offset của nó. Các luồng có cờ bằng 1 ghi khóa của mình tại offset của mình.
            Luồng cuối cùng biết kích thước đầu ra: offset của nó cộng với cờ của chính nó.
          </p>
        }
      >
        <p>
          The recipe has three steps. Each thread turns its key into a keep flag, 1 or 0. An exclusive scan of the flags gives
          each thread the number of kept keys before it: its offset. Threads whose flag is 1 write their key at their offset.
          The last thread knows the output size: its offset plus its own flag.
        </p>
      </Translated>
      <Source section="12.5" pages="297–298" figs={tr('Fig. 12.7', 'Hình 12.7')} />
      <Predict
        id="ch12/12.5/p-size"
        prompt={
          <Translated
            vi={
              <>
                Các cờ keep <code>1 0 0 1 1 0 1 1</code>. Luồng cuối cùng của Hình 12.8 ghi kích thước đầu ra là bao nhiêu?
              </>
            }
          >
            <>
              Keep flags <code>1 0 0 1 1 0 1 1</code>. What does the last thread of Fig. 12.8 write as the output size?
            </>
          </Translated>
        }
      >
        {(v) => (
          <p>
            <Translated
              vi={
                <>
                  Scan loại trừ là <code>0 1 1 1 2 3 3 4</code>, nên offset của luồng cuối là 4, và cờ của chính nó là 1: kích
                  thước là <b>5</b>.
                </>
              }
            >
              The exclusive scan is <code>0 1 1 1 2 3 3 4</code>, so the last thread's offset is 4, and its own flag is 1: the
              size is <b>5</b>.
            </Translated>{' '}
            {v === 5 ? tr('As you said.', 'Đúng như bạn nói.') : v === 4 ? tr('The last offset alone, 4, misses the last key\'s own flag.', 'Chỉ offset cuối, 4, thì bỏ sót cờ của chính khóa cuối.') : tr('Count the ones: five keys are kept.', 'Đếm các số 1: năm khóa được giữ.')}
          </p>
        )}
      </Predict>
      <Explorer />
      <Translated
        vi={
          <p>
            Hai điều cần rút ra. Một offset là số khóa được giữ đứng trước, không hơn: bấm vào bất kỳ khóa được giữ nào và các
            khóa sáng lên phía sau nó chính là những gì offset của nó đếm. Và mọi khóa đều có offset, được giữ hay không, vì phép
            scan chạy qua tất cả; chỉ các khóa được giữ đọc offset của mình.
          </p>
        }
      >
        <p>
          Two things to take from it. An offset is a count of earlier keepers, nothing more: click any kept key and the keys
          lit up behind it are exactly what its offset counts. And every key gets an offset, kept or not, because the scan
          runs over all of them; only kept keys read theirs.
        </p>
      </Translated>

      <h2><Translated vi="Hình 12.8, và mã dùng ở đây thay thế">Fig. 12.8, and the code here instead</Translated></h2>
      <Translated
        vi={
          <p>
            Kernel của sách làm tất cả trong một lần khởi chạy: tính <code>keep</code>, gọi{' '}
            <code>gridExclusiveScan(keep)</code>, ghi. Nó ngắn vì phần khó bị giấu đi. <code>gridExclusiveScan</code> không bao
            giờ được định nghĩa; nó đại diện cho phép scan một lượt của chương 11 chạy bên trong bộ lọc. Đoạn mã cũng đọc{' '}
            <code>input[i]</code> mà không kiểm tra <code>i &lt; N</code>, khác với mọi kernel khác trong chương.
          </p>
        }
      >
        <p>
          The book's kernel does all of this in one launch: compute <code>keep</code>, call{' '}
          <code>gridExclusiveScan(keep)</code>, write. It is short because the hard part is hidden.{' '}
          <code>gridExclusiveScan</code> is never defined; it stands for chapter 11's single-pass scan run inside the filter.
          The listing also reads <code>input[i]</code> with no <code>i &lt; N</code> check, unlike every other kernel in the
          chapter.
        </p>
      </Translated>
      <Source section="12.5" pages="298" figs={tr('Fig. 12.8', 'Hình 12.8')} />
      <Ask q={questionsS125['ch12/12.5/q-everycall']} />
      <Translated
        vi={
          <p>
            Phép scan một lượt bắt các khối chờ các khối trước, điều mà chỉ CUDA ghi nhận là an toàn, và chỉ khi chỉ số khối lấy
            từ một bộ đếm (ch12-facts A10; B6 và C4 cho AMD và Apple). Vì vậy mã hiển thị ở đây là dạng của bài tập 1, ba lần
            khởi chạy không chờ đợi, cũng là phần A của lab 12.3: một kernel ghi các cờ, một phép scan được cung cấp sẵn, và một
            kernel rải.
          </p>
        }
      >
        <p>
          A single-pass scan has blocks wait for earlier blocks, which only CUDA documents as safe, and only with a block index
          taken from a counter (ch12-facts A10; B6 and C4 for AMD and Apple). So the code shown here is exercise 1's form, three
          launches with no waiting, which is lab 12.3 part A: a kernel that writes the flags, a provided scan, and a kernel
          that scatters.
        </p>
      </Translated>
      {cuPart && metalPart ? (
        <Code
          tabs={[
            { label: 'CUDA', code: cuPart, tiers: TIERS, status: status.cu },
            { label: tr('Metal kernel', 'Kernel Metal'), code: metalPart, tiers: TIERS, status: status.metal },
            ...(scanHeader ? [{ label: tr('the provided scan', 'phép scan được cung cấp'), code: scanHeader, tiers: {} }] : []),
          ]}
          caption={
            <Translated
              vi={
                <>
                  Lab 12.3 phần A, từ <code>labs/ch12/lab3_stable/solution/</code>. Hãy tự làm trong lab trước khi đọc phần này.
                  Phép scan giữa hai kernel được cung cấp sẵn và dựng từ các scan khối theo cách mục 11.9 dựng một scan toàn cục;
                  phần đầu tệp của nó là tab thứ ba.
                </>
              }
            >
              <>
                Lab 12.3 part A, from <code>labs/ch12/lab3_stable/solution/</code>. Try it yourself in the lab before reading
                this. The scan between the two kernels is provided and built from block scans the way section 11.9 builds a
                global scan; its header is the third tab.
              </>
            </Translated>
          }
        />
      ) : null}
      <Translated
        vi={
          <p>
            Một phép scan toàn lưới trong một kernel đến từ đâu, và nó cần gì ở phần cứng, là nội dung chương 11:{' '}
            <a href={href('ch11', '11.9')}>mục 11.9</a> so sánh scan ba kernel với scan một lượt.
          </p>
        }
      >
        <p>
          Where a one-kernel grid-wide scan comes from, and what it needs from the hardware, is chapter 11:{' '}
          <a href={href('ch11', '11.9')}>section 11.9</a> compares the three-kernel scan with the single pass.
        </p>
      </Translated>

      <h2><Translated vi="Scan các số 0 và 1, bên trong một khối">A scan of 0s and 1s, inside a block</Translated></h2>
      <Translated
        vi={
          <p>
            Các cờ là nhị phân, và điều đó làm phép scan bên trong một warp trở nên rẻ. Mục 12.3 đã tính nó: hạng của một lane
            trong số các lane được giữ của warp nó là popcount của phiếu bầu warp, che lại chỉ còn các lane bên dưới nó. Một lần
            bỏ phiếu và một popcount mỗi lane thay cho cả một phép scan cấp warp. Một khối khi đó cần thêm một cấp: tổng của các
            warp, được scan, cho mỗi warp biết các khóa được giữ của nó bắt đầu ở đâu. Đó là phép scan của chương 11: scan, rồi
            scan các tổng, rồi cộng.
          </p>
        }
      >
        <p>
          The flags are binary, and that makes the scan inside a warp cheap. Section 12.3 already computed it: a lane's rank
          among the kept lanes of its warp is the popcount of the warp's vote masked to the lanes below it. One vote and one
          popcount per lane replace a whole warp-level scan. A block then needs one more level: the totals of its warps, scanned,
          tell each warp where its kept keys start. That is chapter 11's scan, then scan the totals, then add.
        </p>
      </Translated>
      <Source section="12.5" pages="298" />
      <Translated
        vi={
          <p>
            Trình chạy từng bước đưa một khối đi qua quy trình đó: phiếu bầu của mỗi warp, các hạng trong mỗi warp, các tổng
            warp, phép scan của chúng, và ô của mọi khóa được giữ. Đổi độ rộng warp để thấy không gì trong phương pháp phụ thuộc
            vào nó.
          </p>
        }
      >
        <p>
          The stepper runs one block through it: each warp's vote, the ranks inside each warp, the warp totals, their scan,
          and the slot of every kept key. Switch the warp width to see that nothing in the method depends on it.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các bước được vẽ lần lượt cho tất cả warp; trên GPU các warp của một khối tới đó vào những thời điểm khác nhau, và
              chỉ rào chắn mới bắt chúng chờ nhau. Phiếu bầu được hiển thị là một giá trị mỗi warp, đúng như những gì một lệnh
              ballot trong luồng điều khiển đồng nhất trả về.
            </p>
          }
        >
          <p>
            The steps are drawn one after another for all warps; on a GPU the warps of a block reach them at different times,
            and only the barrier makes them wait for each other. The vote is shown as one value per warp, which is what a
            ballot in uniform control flow returns.
          </p>
        </Translated>
      </Simplification>
      <BlockStepper />
      <Ask q={questionsS125['ch12/12.5/q-blockrank']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Đây là cách lab 12.3 phần B và lab 12.4 tính các offset của một khối: một lần bỏ phiếu và một popcount mỗi lane, các
            tổng warp trong bộ nhớ chia sẻ, một rào chắn, một phép scan trên vài con số. Trên HIP mặt nạ rộng 64 bit và cần{' '}
            <code>__popcll</code>; trên Metal, <code>simd_prefix_exclusive_sum</code> và <code>simd_sum</code> tính hạng và tổng,
            mỗi thứ bằng một lời gọi (ch12-facts B1, C1).
          </p>
        }
      >
        <p>
          This is how lab 12.3 part B and lab 12.4 compute a block's offsets: a vote and a popcount per lane, warp totals in
          shared memory, one barrier, a scan of a few numbers. On HIP the mask is 64 bits wide and needs <code>__popcll</code>;
          on Metal, <code>simd_prefix_exclusive_sum</code> and <code>simd_sum</code> do the rank and the total in one call each
          (ch12-facts B1, C1).
        </p>
      </Translated>

      <h2><Translated vi="Một kernel duy nhất sẽ tiết kiệm được gì">What one kernel would save</Translated></h2>
      <Translated
        vi={
          <p>
            Bài tập 2 yêu cầu viết bộ lọc ổn định thành một kernel duy nhất, dựng bên trong một phép scan một lượt, và nêu các ưu
            điểm chính của nó. Sách không đưa ra đáp án. Hãy đếm số byte.
          </p>
        }
      >
        <p>
          Exercise 2 asks for the stable filter as a single kernel, built into a one-pass scan, and for its major advantages.
          The book gives no answer. Count the bytes.
        </p>
      </Translated>
      <Source section="12.10" pages="302" />
      <Ask q={questionsS125['ch12/12.5/q-ex1-bytes']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Điều này xuất hiện ở đâu trong một bộ lấy mẫu">Where this shows up in a sampler</Translated></h2>
      <Translated
        vi={
          <p>
            Bộ lấy mẫu min-p phía host của llama.cpp chính là bộ lọc này, chạy tuần tự: nó giữ các cặp (token id, logit) có logit
            ít nhất bằng giá trị lớn nhất cộng ln p, theo thứ tự sẵn có, rồi chép chúng lại với kích thước mới. Lab 12.4 là dạng
            GPU của nó. Giao diện thứ hai của trình khám phá là phép kiểm tra đó trên một hàng logits; giao diện thứ ba, top-p trên
            các xác suất đã sắp xếp, cho thấy vì sao trường hợp đó dễ hơn vẻ ngoài.
          </p>
        }
      >
        <p>
          llama.cpp's host min-p sampler is this filter, run sequentially: it keeps (token id, logit) pairs whose logit is at
          least the maximum plus ln p, in their existing order, and copies them back with a new size. Lab 12.4 is its GPU form.
          The explorer's second skin is that test on a row of logits; the third, top-p over sorted probabilities, shows why that
          case is easier than it looks.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cách đối chiếu với bộ lấy mẫu là của khóa học này. Mã của llama.cpp:{' '}
              <code>min_logit = max_logit + logf(ctx-&gt;p)</code>, rồi <code>push_back</code> mọi ứng viên có{' '}
              <code>logit &gt;= min_logit</code> (ch12-facts D3).
            </p>
          }
        >
          <p>
            The sampler mapping is ours. llama.cpp's code: <code>min_logit = max_logit + logf(ctx-&gt;p)</code>, then a{' '}
            <code>push_back</code> of every candidate with <code>logit &gt;= min_logit</code> (ch12-facts D3).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS125['ch12/12.5/q-topp']} />

      <h2><Translated vi="Các thư viện cam kết gì">What the libraries promise</Translated></h2>
      <Translated
        vi={
          <p>
            Lời khuyên của sách cho môi trường production là gọi Thrust hoặc CUB. Biết các biến thể cho bạn biết mình cần lời gọi
            nào và nó tốn gì: một thư viện phải trả về một số đếm mà bạn sau đó chép về host đã phải trả giá cho một lần đồng bộ
            hóa.
          </p>
        }
      >
        <p>
          The book's advice for production is to call Thrust or CUB. Knowing the variants tells you which call you need and
          what it costs: a library that must return a count you then copy to the host has paid for a synchronisation.
        </p>
      </Translated>
      <Source section="12.9" pages="302" />
      <Translated
        vi={
          <p>
            Các bộ lọc của CUB và Thrust là ổn định, và PyTorch trả giá cho số đếm, thứ phụ thuộc vào dữ liệu, theo một trong hai
            cách: chờ nó trên host, hoặc nhận kích thước đầu ra ngay từ đầu.
          </p>
        }
      >
        <p>
          CUB's and Thrust's filters are stable, and PyTorch pays for the count, which depends on the data, in one of two
          ways: it waits for it on the host, or it takes the output size up front.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md, A8, D4, B5, C5)', '2026-11-20 (notes/today/ch12-facts.md, mục A8, D4, B5, C5)')}>
        <Translated
          vi={
            <p>
              <code>DeviceSelect::If</code> và <code>::Flagged</code> của CUB (hàm nhận mảng keep của bài tập 1) và{' '}
              <code>copy_if</code> của Thrust là ổn định. <code>nonzero</code> của PyTorch trên CUDA là một phép rút gọn CUB, một
              lần sao chép số đếm về host, rồi <code>DeviceSelect::Flagged</code>; <code>nonzero_static</code> là bài tập 1 có gộp
              việc cho luồng và không đồng bộ với host. <code>topk</code> của PyTorch gom bằng ballot và popcount, và tránh{' '}
              <code>atomicAdd</code> để giữ thứ tự chỉ số. hipCUB chuyển tiếp sang rocPRIM, mà tài liệu chỉ cho thấy thứ tự qua
              ví dụ. Trên Apple, thứ tương ứng tìm thấy trong thư viện là <code>nonZeroIndices</code> của MPSGraph.
            </p>
          }
        >
          <p>
            CUB's <code>DeviceSelect::If</code> and <code>::Flagged</code> (which takes exercise 1's keep array) and Thrust's{' '}
            <code>copy_if</code> are stable. PyTorch's <code>nonzero</code> on CUDA is a CUB reduction, a copy of the count to
            the host, then <code>DeviceSelect::Flagged</code>; <code>nonzero_static</code> is exercise 1 with coarsening and no
            host sync. PyTorch's <code>topk</code> gathers with ballots and popcounts and rejects <code>atomicAdd</code> to keep
            index order. hipCUB forwards to rocPRIM, whose docs show the order only by example. On Apple, the library
            counterpart found is MPSGraph's <code>nonZeroIndices</code>.
          </p>
        </Translated>
      </Today>
    </>
  );
}
