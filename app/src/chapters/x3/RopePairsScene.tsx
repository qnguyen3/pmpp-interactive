import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { compare, makeCase, matvec, pairOf, permute, permuteSource, rope, ropeTables, seeded, unpermuteVec, type Pairing } from '../../sim/ropepair';

// "Two ways to pair a head" (section X3.7): one head of 8 values in the two RoPE pairings,
// turned by position, then a half-split-trained head run through a consecutive-pair kernel
// with and without llama.cpp's row permutation. Six beats, one clip each
// (content/x3/narration/rope-pairs-N.txt). Every number drawn comes from src/sim/ropepair.ts.
const HEADN = 8;
const T = ropeTables(64, HEADN);
const CASE = makeCase(HEADN);
const Q = matvec(CASE.Wq, CASE.xq, HEADN, CASE.din);
const POS = 15;
const MAXD = 15;
const MISMATCH = compare(CASE, POS, 'norm', false, T, MAXD);
const PERMUTED = compare(CASE, POS, 'norm', true, T, MAXD);

// The GQA beat: 4 query heads over 2 key heads of 8, a K weight of 16 rows.
const GQA = { nHead: 4, nKv: 2, din: 6 };
const KROWS = GQA.nKv * HEADN;
const GOOD_SRC = permuteSource(KROWS, GQA.nHead, GQA.nKv);
const BAD_SRC = permuteSource(KROWS, GQA.nHead);
const gqaOk = (useKv: boolean) => {
  const r = seeded(11);
  const W = Float32Array.from({ length: KROWS * GQA.din }, () => Math.fround(r()));
  const x = Float32Array.from({ length: GQA.din }, () => Math.fround(r()));
  const ref = rope(matvec(W, x, KROWS, GQA.din), 5, 'neox', T);
  const Pw = useKv ? permute(W, KROWS, GQA.din, GQA.nHead, GQA.nKv) : permute(W, KROWS, GQA.din, GQA.nHead);
  const ker = rope(matvec(Pw, x, KROWS, GQA.din), 5, 'norm', T);
  const back = useKv ? unpermuteVec(ker, GQA.nHead, GQA.nKv) : unpermuteVec(ker, GQA.nHead);
  return back.every((v, i) => v === ref[i]);
};
const GOOD = gqaOk(true);
const BAD = gqaOk(false);

const W = 720;
const H = 330;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const MONO = { fontVariationSettings: "'MONO' 1" } as const; // i18n-ignore: CSS value

function Frame({ beat, headline, children }: { beat: number; headline: string; children: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={headline} data-beat={beat} style={{ display: 'block' }}>
      <text x={14} y={22} fontSize="13" fontWeight="600" data-headline="" style={{ fill: 'var(--ink)' }}>{headline}</text>
      {children}
    </svg>
  );
}

/** The head's elements as a row of cells, each labelled with the pair it belongs to. */
function Elements({ mode, y, dim = 1 }: { mode: Pairing; y: number; dim?: number }) {
  const pairIdx = (e: number) => [0, 1, 2, 3].find((j) => pairOf(j, HEADN, mode).includes(e))!;
  return (
    <g opacity={dim} data-elements={mode}>
      {Array.from({ length: HEADN }, (_, e) => (
        <g key={e} data-el={e} data-pair={pairIdx(e)}>
          <rect x={40 + e * 52} y={y} width={46} height={26} rx="3" fill="var(--cell)" stroke="var(--rule)" />
          <text x={63 + e * 52} y={y + 17} fontSize="11.5" textAnchor="middle" style={{ ...MONO, fill: 'var(--ink)' }}>x{e}</text>
          <text x={63 + e * 52} y={y + 40} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{tr(`pair ${pairIdx(e)}`, `cặp ${pairIdx(e)}`)}</text>
        </g>
      ))}
    </g>
  );
}

/** Four circles, one per pair, with the pair's vector at position pos. */
function Circles({ mode, pos, y, dim = 1 }: { mode: Pairing; pos: number; y: number; dim?: number }) {
  const v = rope(Q, pos, mode, T);
  const R = 34;
  const scale = R / Math.max(...[0, 1, 2, 3].map((j) => Math.hypot(...pairOf(j, HEADN, mode).map((e) => Q[e]))));
  return (
    <g opacity={dim}>
      {[0, 1, 2, 3].map((j) => {
        const [a, b] = pairOf(j, HEADN, mode);
        const cx = 90 + j * 150;
        const cy = y + R + 6;
        return (
          <g key={j} data-circle={j} data-angle={Math.atan2(v[b], v[a]).toFixed(4)}>
            <circle cx={cx} cy={cy} r={R} fill="none" stroke="var(--rule)" />
            <line x1={cx} y1={cy} x2={cx + v[a] * scale} y2={cy - v[b] * scale} stroke="var(--ink)" strokeWidth="2.2" />
            <circle cx={cx + v[a] * scale} cy={cy - v[b] * scale} r="3" fill="var(--ink)" />
            <text x={cx} y={cy + R + 16} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
              (x{a}, x{b})
            </text>
          </g>
        );
      })}
    </g>
  );
}

/** Score against distance: the reference (ink) and a kernel's curve. */
function Scores({ kernel, ok, show }: { kernel: number[] | null; ok: boolean; show: number }) {
  const ref = MISMATCH.ref.scores;
  const all = [...ref, ...MISMATCH.ker.scores];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const X0 = 60, X1 = 680, Y0 = 290, Y1 = 90;
  const px = (d: number) => X0 + (d / MAXD) * (X1 - X0);
  const py = (s: number) => Y0 - ((s - lo) / (hi - lo || 1)) * (Y0 - Y1);
  const path = (s: number[]) => s.map((v, d) => `${d ? 'L' : 'M'}${px(d).toFixed(1)},${py(v).toFixed(1)}`).join(' ');
  return (
    <g>
      <line x1={X0} y1={Y0} x2={X1} y2={Y0} stroke="var(--rule)" />
      <text x={X0} y={Y0 + 18} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{tr('distance 0', 'khoảng cách 0')}</text>
      <text x={X1} y={Y0 + 18} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{tr(`distance ${MAXD}`, `khoảng cách ${MAXD}`)}</text>
      <text x={X0} y={Y1 - 12} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>q · k cho một query tại vị trí {POS} và các key lùi lại 0 đến {MAXD} vị trí</>}>q · k for a query at position {POS} and keys 0 to {MAXD} positions back</Translated>
      </text>
      <path d={path(ref)} fill="none" stroke="var(--ink)" strokeWidth="2" data-curve="reference" />
      {kernel ? (
        <path d={path(kernel.slice(0, Math.max(2, Math.round(show * (MAXD + 1)))))} fill="none"
          stroke={ok ? 'var(--ok)' : 'var(--fail)'} strokeWidth="2.4" strokeDasharray={ok ? '6 4' : undefined} data-curve={ok ? 'match' : 'mismatch'} />
      ) : null}
      <g transform={`translate(${X1 - 230}, ${Y1 - 4})`}>
        <line x1={0} y1={0} x2={22} y2={0} stroke="var(--ink)" strokeWidth="2" />
        <text x={28} y={4} fontSize="10.5" style={{ fill: 'var(--ink)' }}>{tr('reference: half split', 'tham chiếu: chia đôi')}</text>
        {kernel ? (
          <>
            <line x1={0} y1={16} x2={22} y2={16} stroke={ok ? 'var(--ok)' : 'var(--fail)'} strokeWidth="2.4" strokeDasharray={ok ? '6 4' : undefined} />
            <text x={28} y={20} fontSize="10.5" style={{ fill: 'var(--ink)' }}>{ok ? tr('consecutive, permuted weights', 'liên tiếp, trọng số đã hoán vị') : tr('consecutive, same weights', 'liên tiếp, trọng số giữ nguyên')}</text>
          </>
        ) : null}
      </g>
    </g>
  );
}

function RowMap({ src, y, label, ok, show }: { src: number[]; y: number; label: string; ok: boolean; show: number }) {
  return (
    <g data-rowmap={label} data-ok={ok ? 1 : 0}>
      <text x={14} y={y - 8} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{label}</text>
      {src.map((s, r) => (
        <g key={r} opacity={r < Math.round(show * src.length) ? 1 : 0.2}>
          <rect x={14 + r * 40} y={y} width={36} height={24} rx="3" fill={r < HEADN ? 'var(--cell)' : 'var(--rule-soft)'} stroke="var(--rule)" />
          <text x={32 + r * 40} y={y + 16} fontSize="11" textAnchor="middle" style={{ ...MONO, fill: 'var(--ink)' }}>{s}</text>
        </g>
      ))}
      <text x={14} y={y + 42} fontSize="11" opacity={show > 0.6 ? 1 : 0} style={{ fill: 'var(--ink)' }} data-pairs="">
        <Translated
          vi={
            <>
              các cặp liên tiếp của kernel trong key head 0 giờ chứa các hàng cũ{' '}
              {[0, 1, 2, 3].map((j) => `(${src[2 * j]}, ${src[2 * j + 1]})`).join(', ')}
            </>
          }
        >
          the kernel's consecutive pairs in key head 0 now hold old rows{' '}
          {[0, 1, 2, 3].map((j) => `(${src[2 * j]}, ${src[2 * j + 1]})`).join(', ')}
        </Translated>
      </text>
      <text x={14} y={y + 62} fontSize="11.5" fontWeight="600" opacity={show > 0.9 ? 1 : 0} style={{ fill: ok ? 'var(--ok)' : 'var(--fail)' }}>
        {ok ? tr('scores equal the reference', 'điểm số bằng tham chiếu') : tr('scores differ from the reference', 'điểm số khác tham chiếu')}
      </text>
    </g>
  );
}

const beats: Beat[] = [
  {
    clip: 'rope-pairs-1',
    get label() { return tr('Consecutive pairs', 'Các cặp liên tiếp'); },
    draw: (p) => (
      <Frame beat={1} headline={tr('one head of 8 values: consecutive pairing groups (0, 1), (2, 3), (4, 5), (6, 7)', 'một head gồm 8 giá trị: ghép cặp liên tiếp nhóm (0, 1), (2, 3), (4, 5), (6, 7)')}>
        <Elements mode="norm" y={52} dim={ramp(p, 0, 0.3)} />
        <Circles mode="norm" pos={0} y={130} dim={ramp(p, 0.35, 0.65)} />
      </Frame>
    ),
  },
  {
    clip: 'rope-pairs-2',
    get label() { return tr('Position turns them', 'Vị trí làm chúng xoay'); },
    draw: (p) => {
      const pos = Math.min(3, Math.floor(p * 4));
      return (
        <Frame beat={2} headline={tr(`position ${pos}: pair j turns by ${pos} × 10000^(−2j/8) radians`, `vị trí ${pos}: cặp j xoay ${pos} × 10000^(−2j/8) radian`)}>
          <Elements mode="norm" y={52} dim={0.45} />
          <g data-pos={pos}><Circles mode="norm" pos={pos} y={130} /></g>
          <text x={14} y={318} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
            {tr('angles per step: 1, 0.1, 0.01, 0.001 radians. The lengths do not change.', 'góc mỗi bước: 1, 0.1, 0.01, 0.001 radian. Độ dài không đổi.')}
          </text>
        </Frame>
      );
    },
  },
  {
    clip: 'rope-pairs-3',
    get label() { return tr('Half split', 'Chia đôi'); },
    draw: (p) => (
      <Frame beat={3} headline={tr('half split: element j goes with element j + 4; the same rotation, other partners', 'chia đôi: phần tử j đi cặp với phần tử j + 4; cùng phép xoay, khác bạn cặp')}>
        <Elements mode="neox" y={52} dim={ramp(p, 0, 0.3)} />
        <Circles mode="neox" pos={Math.min(3, Math.floor(ramp(p, 0.4, 1) * 4))} y={130} dim={ramp(p, 0.3, 0.5)} />
      </Frame>
    ),
  },
  {
    clip: 'rope-pairs-4',
    get label() { return tr('The wrong pairing', 'Ghép cặp sai'); },
    draw: (p) => (
      <Frame beat={4} headline={tr('half-split weights through a consecutive-pair kernel: finite, plausible, wrong', 'trọng số kiểu chia đôi qua một kernel ghép cặp liên tiếp: hữu hạn, trông hợp lý, nhưng sai')}>
        <Scores kernel={MISMATCH.ker.scores} ok={MISMATCH.same} show={ramp(p, 0.2, 0.8)} />
        <text x={60} y={320} fontSize="11" opacity={ramp(p, 0.8, 0.95)} style={{ fill: 'var(--fail)' }} data-diff={MISMATCH.maxScoreDiff.toFixed(4)}>
          <Translated vi={<>chênh lệch lớn nhất so với tham chiếu: {MISMATCH.maxScoreDiff.toFixed(3)}. Không lỗi, không cảnh báo.</>}>
            largest difference from the reference: {MISMATCH.maxScoreDiff.toFixed(3)}. No error, no warning.
          </Translated>
        </text>
      </Frame>
    ),
  },
  {
    clip: 'rope-pairs-5',
    get label() { return tr('Permute the rows', 'Hoán vị các hàng'); },
    draw: (p) => (
      <Frame beat={5} headline={tr("permute the rows of Wq and Wk as llama.cpp's converter does: the scores match", 'hoán vị các hàng của Wq và Wk như bộ chuyển đổi của llama.cpp làm: điểm số khớp')}>
        <Scores kernel={PERMUTED.ker.scores} ok={PERMUTED.same} show={ramp(p, 0.2, 0.8)} />
        <text x={60} y={320} fontSize="11" opacity={ramp(p, 0.8, 0.95)} style={{ fill: PERMUTED.same ? 'var(--ok)' : 'var(--fail)' }} data-same={PERMUTED.same ? 1 : 0}>
          {PERMUTED.same ? tr('the rotated vector, put back in order, equals the reference exactly', 'vector đã xoay, khi xếp lại đúng thứ tự, bằng đúng tham chiếu') : tr('still differs', 'vẫn khác')}
        </text>
      </Frame>
    ),
  },
  {
    clip: 'rope-pairs-6',
    get label() { return tr('Keys and GQA', 'Key và GQA'); },
    draw: (p) => (
      <Frame beat={6} headline={tr('K has 2 heads of 8 rows in a model with 4 query heads: permute by the KV head count', 'K có 2 head 8 hàng trong một mô hình có 4 head query: hoán vị theo số head KV')}>
        <text x={14} y={48} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr("the half split trained each key head's pairs as old rows (0, 4), (1, 5), (2, 6), (3, 7)", 'kiểu chia đôi đã huấn luyện các cặp của mỗi key head là các hàng cũ (0, 4), (1, 5), (2, 6), (3, 7)')}</text>
        <RowMap src={GOOD_SRC} y={84} label={tr('permute(Wk, n_head = 4, n_head_kv = 2): new row r takes old row', 'permute(Wk, n_head = 4, n_head_kv = 2): hàng mới r lấy hàng cũ')} ok={GOOD} show={ramp(p, 0.05, 0.4)} />
        <RowMap src={BAD_SRC} y={204} label={tr('grouped by the query head count, 4: new row r takes old row', 'nhóm theo số head query, 4: hàng mới r lấy hàng cũ')} ok={BAD} show={ramp(p, 0.45, 0.8)} />
      </Frame>
    ),
  },
];

export function RopePairsScene() {
  return <Scene chapter="x3" title={tr('Two ways to pair a head', 'Hai cách ghép cặp một head')} beats={beats} />;
}
