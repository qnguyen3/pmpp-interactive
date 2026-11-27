import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { countsAfter, gather, route } from '../../sim/sparseroute';

// "Routing tokens to experts": exercise 3's COO-to-CSR conversion (count, scan, scatter)
// as mixture-of-experts routing runs it, then the expert GEMMs on padded slices and the
// inverse permutation back to token order. Every number drawn comes from
// src/sim/sparseroute.ts (8 tokens, top-2, 4 experts, expert 2 unused, tiles of 4).
// Six beats, one narration clip each (content/ch17/narration/route-N.txt). Ours, not the
// book's: ch17-facts D5 to D7.
const R = route();
const N = R.pairs.length; // 16
const SORTED = gather(R.pairs, R.scatter.order);

const W = 720;
const H = 450;
// the matrix of router picks
const MX = 64;
const MY = 46;
const MC = 24;
// counters and offsets
const CX = 300;
const OX = 500;
const OW = 40;
// strips
const SX = 40; // token-order strip and the compact expert-order strip: 16 slots of 40
const SW = 40;
const TOP = 176; // token-order strip
const MID = 266; // expert-order strip
const BOT = 352; // back in token order
const PX = 48; // the padded layout: 24 slots of 26
const PW = 26;
const BH = 34; // box height

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const ease = (t: number) => t * t * (3 - 2 * t);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const list = (xs: readonly number[]) => '[' + xs.join(', ') + ']';

/** Where pair i sits in each layout. */
const topX = (i: number) => SX + i * SW;
const compactX = (i: number) => SX + R.scatter.pos[i] * SW;
const paddedX = (i: number) => PX + R.paddedPos[i] * PW;

interface Box {
  i: number; // pair index
  x: number;
  y: number;
  w: number;
  hot?: boolean;
  dim?: boolean;
  note?: string; // small text under the box
}

function PairBox({ b }: { b: Box }) {
  const p = R.pairs[b.i];
  const narrow = b.w < 32;
  return (
    <g data-pair={b.i} opacity={b.dim ? 0.35 : 1}>
      <rect x={b.x + 1} y={b.y} width={b.w - 2} height={BH} rx="3"
        fill={b.hot ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
      <text x={b.x + b.w / 2} y={b.y + 14} fontSize={narrow ? 10 : 12} textAnchor="middle" fontWeight="600"
        style={{ fill: b.hot ? 'var(--page)' : 'var(--ink)' }}>t{p.token}</text>
      <text x={b.x + b.w / 2} y={b.y + 28} fontSize={narrow ? 9 : 10} textAnchor="middle"
        style={{ fill: b.hot ? 'var(--page)' : 'var(--ink-2)' }}>E{p.expert}</text>
      {b.note ? (
        <text x={b.x + b.w / 2} y={b.y + BH + 12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{b.note}</text>
      ) : null}
    </g>
  );
}

function StripLabel({ y, text }: { y: number; text: string }) {
  return <text x={SX} y={y - 8} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>{text}</text>;
}

/** The experts × tokens matrix, with the first `shown` pairs marked and pair `hot` highlighted. */
function Matrix({ shown, hot }: { shown: number; hot?: number }) {
  const on = new Map<string, number>();
  R.pairs.slice(0, shown).forEach((p, i) => on.set(`${p.expert},${p.token}`, i));
  return (
    <g data-matrix={shown}>
      <text x={MX} y={MY - 26} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="lựa chọn của router: expert × token">router picks: experts × tokens</Translated></text>
      {Array.from({ length: R.tokens }, (_, t) => (
        <text key={t} x={MX + t * MC + MC / 2} y={MY - 6} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>t{t}</text>
      ))}
      {Array.from({ length: R.experts }, (_, e) => (
        <g key={e}>
          <text x={MX - 8} y={MY + e * MC + 16} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>E{e}</text>
          {Array.from({ length: R.tokens }, (_, t) => {
            const i = on.get(`${e},${t}`);
            const isOn = i != null;
            const isHot = isOn && i === hot;
            return (
              <rect key={t} x={MX + t * MC} y={MY + e * MC} width={MC - 2} height={MC - 2} rx="2"
                fill={isHot ? 'var(--global)' : isOn ? 'var(--global-wash)' : 'var(--cell)'}
                stroke={isOn ? 'var(--global)' : 'var(--rule)'} />
            );
          })}
        </g>
      ))}
    </g>
  );
}

function Counters({ counts, show = 1, final = false }: { counts: number[]; show?: number; final?: boolean }) {
  return (
    <g opacity={show} data-counts={counts.join(',')}>
      <text x={CX} y={MY - 26} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="đếm (histogram)">count (histogram)</Translated></text>
      {counts.map((c, e) => (
        <g key={e}>
          <text x={CX} y={MY + e * MC + 16} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>E{e}</text>
          {Array.from({ length: c }, (_, k) => (
            <rect key={k} x={CX + 22 + k * 14} y={MY + e * MC + 5} width="11" height="12" rx="1.5" fill="var(--global)" />
          ))}
          <text x={CX + 22 + Math.max(c, 0) * 14 + 4} y={MY + e * MC + 16} fontSize="11" fontWeight="600">{c}</text>
          {final && c === 0 ? (
            <text x={CX + 40} y={MY + e * MC + 16} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="không có token: bỏ qua GEMM của nó">no tokens: skip its GEMM</Translated></text>
          ) : null}
        </g>
      ))}
    </g>
  );
}

function Offsets({ shown, show = 1 }: { shown: number; show?: number }) {
  const labels = [...Array.from({ length: R.experts }, (_, e) => `E${e}`), tr('end', 'cuối')];
  return (
    <g opacity={show} data-offsets={R.offsets.slice(0, shown).join(',')}>
      <text x={OX} y={MY - 26} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>offsets = rowPtrs</text>
      {R.offsets.map((v, k) => (
        <g key={k}>
          <text x={OX + k * OW + OW / 2} y={MY - 6} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{labels[k]}</text>
          <rect x={OX + k * OW} y={MY} width={OW - 3} height={28} rx="2"
            fill={k < shown ? 'var(--global-wash)' : 'var(--cell)'} stroke={k < shown ? 'var(--global)' : 'var(--rule)'}
            strokeDasharray={k === R.experts ? '4 3' : undefined} />
          {k < shown ? (
            <text x={OX + k * OW + (OW - 3) / 2} y={MY + 19} fontSize="12.5" textAnchor="middle" fontWeight="600">{v}</text>
          ) : null}
        </g>
      ))}
    </g>
  );
}

/** Brackets under the expert-order strip, compact (16 slots) or padded (24 slots), blended by t. */
function Slices({ t, y, hot, show = 1 }: { t: number; y: number; hot?: number; show?: number }) {
  return (
    <g opacity={show}>
      {Array.from({ length: R.experts }, (_, e) => {
        const a = lerp(SX + R.offsets[e] * SW, PX + R.slices.starts[e] * PW, t);
        const b = lerp(SX + R.offsets[e + 1] * SW, PX + R.slices.starts[e + 1] * PW, t);
        const empty = R.counts[e] === 0;
        const yb = y + BH + 6;
        return (
          <g key={e} data-slice={e}>
            {empty ? (
              <>
                <line x1={a} x2={a} y1={y - 4} y2={yb + 8} stroke="var(--ink-3)" strokeDasharray="3 3" />
                <text x={a} y={yb + 32} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>E{e}: rỗng</>}>E{e}: empty</Translated></text>
              </>
            ) : (
              <>
                <path d={`M${a + 2},${yb} v6 H${b - 2} v-6`} fill="none" stroke={hot === e ? 'var(--ink)' : 'var(--ink-2)'} strokeWidth={hot === e ? 2 : 1.2} />
                <text x={(a + b) / 2} y={yb + 20} fontSize="10.5" textAnchor="middle" fontWeight={hot === e ? 700 : 500}>
                  <Translated vi={<>E{e}: {R.counts[e]} token{t > 0.5 ? `, đệm lên ${R.slices.padded[e]}` : ''}</>}>
                    E{e}: {R.counts[e]} tokens{t > 0.5 ? tr(`, padded to ${R.slices.padded[e]}`, `, đệm lên ${R.slices.padded[e]}`) : ''}
                  </Translated>
                </text>
              </>
            )}
          </g>
        );
      })}
    </g>
  );
}

function PadBoxes({ show }: { show: number }) {
  const used = new Set(R.paddedPos);
  return (
    <g opacity={show} data-pads={R.slices.pads}>
      {Array.from({ length: R.slices.total }, (_, k) =>
        used.has(k) ? null : (
          <rect key={k} x={PX + k * PW + 1} y={MID} width={PW - 2} height={BH} rx="3" fill="none" stroke="var(--ink-3)" strokeDasharray="3 3" />
        ),
      )}
    </g>
  );
}

function Stage({ beat, children, headline, data }: { beat: number; children: ReactNode; headline: string; data?: Record<string, string> }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Routing tokens to experts: count, scan, scatter, run, put back', 'Định tuyến token tới expert: đếm, scan, rải, chạy, đặt lại')}
      style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <g data-beat={beat} {...Object.fromEntries(Object.entries(data ?? {}).map(([k, v]) => [`data-${k}`, v]))}>{/* i18n-ignore: attribute name */}
        {children}
        <text x={SX} y={H - 8} fontSize="13" fontWeight="600">{headline}</text>
      </g>
    </svg>
  );
}

const allTop = (dim = false, hot?: number) => R.pairs.map((_, i) => <PairBox key={i} b={{ i, x: topX(i), y: TOP, w: SW, dim, hot: hot === i }} />);
const ghosts = () =>
  R.pairs.map((_, i) => (
    <rect key={'g' + i} x={topX(i) + 1} y={TOP} width={SW - 2} height={BH} rx="3" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />
  ));

const beats: Beat[] = [
  {
    clip: 'route-1',
    get label() {
      return tr('Pairs', 'Các cặp');
    },
    draw: (p) => {
      // the first token's two picks are there from the start, so the opening frame is not empty
      const shown = Math.max(R.topk, Math.min(N, Math.ceil(ramp(p, 0.05, 0.7) * N)));
      return (
        <Stage beat={1} headline={tr(`${N} token–expert pairs in token order: a COO list (row = expert, column = token)`, `${N} cặp token–expert theo thứ tự token: một danh sách COO (hàng = expert, cột = token)`)} data={{ shown: String(shown) }}>
          <Matrix shown={shown} hot={shown - 1} />
          <StripLabel y={TOP} text={tr('pairs in token order (slot = 2 × token + choice)', 'các cặp theo thứ tự token (vị trí = 2 × token + lựa chọn)')} />
          {ghosts()}
          {R.pairs.slice(0, shown).map((_, i) => <PairBox key={i} b={{ i, x: topX(i), y: TOP, w: SW, hot: i === shown - 1 && p < 0.75 }} />)}
        </Stage>
      );
    },
  },
  {
    clip: 'route-2',
    get label() {
      return tr('Count', 'Đếm');
    },
    draw: (p) => {
      const k = Math.min(N, Math.floor(ramp(p, 0.08, 0.72) * (N + 0.999)));
      const counts = countsAfter(R, k);
      const hot = k < N ? k : undefined;
      return (
        <Stage beat={2} headline={k < N ? tr(`counting pair ${k}: one atomic add on its expert's counter`, `đếm cặp ${k}: một phép cộng nguyên tử trên bộ đếm của expert của nó`) : tr(`counts ${list(R.counts)}: expert ${R.empty.join(', ')} has no tokens`, `số đếm ${list(R.counts)}: expert ${R.empty.join(', ')} không có token nào`)}>
          <Matrix shown={N} hot={hot} />
          <Counters counts={counts} final={k >= N} />
          <StripLabel y={TOP} text={tr('pairs in token order', 'các cặp theo thứ tự token')} />
          {allTop(false, hot)}
        </Stage>
      );
    },
  },
  {
    clip: 'route-3',
    get label() {
      return tr('Scan', 'Scan');
    },
    draw: (p) => {
      const shown = Math.min(R.offsets.length, Math.floor(ramp(p, 0.08, 0.7) * (R.offsets.length + 0.999)));
      return (
        <Stage beat={3} headline={shown >= R.offsets.length ? tr(`exclusive scan + total = ${list(R.offsets)}: CSR's rowPtrs`, `scan loại trừ + tổng = ${list(R.offsets)}: rowPtrs của CSR`) : tr('exclusive scan: each expert starts where the previous ones end', 'scan loại trừ: mỗi expert bắt đầu ở chỗ các expert trước kết thúc')}>
          <Matrix shown={N} />
          <Counters counts={R.counts} final />
          <Offsets shown={shown} />
          <StripLabel y={TOP} text={tr('pairs in token order', 'các cặp theo thứ tự token')} />
          {allTop()}
        </Stage>
      );
    },
  },
  {
    clip: 'route-4',
    get label() {
      return tr('Scatter', 'Rải');
    },
    draw: (p) => {
      const ts = R.pairs.map((_, i) => {
        const start = (i / N) * 0.55 + 0.05;
        return ease(ramp(p, start, start + 0.3));
      });
      const done = ts.filter((t) => t >= 1).length;
      // while pairs land, each shows its start + rank; once all have, the slice brackets take over
      const boxes = ts.map((t, i) => ({
        i, x: lerp(topX(i), compactX(i), t), y: lerp(TOP, MID, t), w: SW, hot: t > 0 && t < 1,
        note: t >= 1 && done < N ? `${R.offsets[R.pairs[i].expert]}+${R.scatter.rank[i]}` : undefined,
      }));
      return (
        <Stage beat={4} headline={done < N ? tr('position = offsets[expert] + rank (earlier pairs of the same expert)', 'vị trí = offsets[expert] + rank (số cặp trước đó của cùng expert)') : tr('each expert\'s tokens are contiguous, still in token order', 'các token của mỗi expert nằm liền nhau, vẫn theo thứ tự token')}
          data={{ order: SORTED.map((x) => x.slot).join(',') }}>
          <Matrix shown={N} />
          <Counters counts={R.counts} final />
          <Offsets shown={R.offsets.length} />
          <StripLabel y={TOP} text={tr('pairs in token order', 'các cặp theo thứ tự token')} />
          {ghosts()}
          <StripLabel y={MID} text={tr('pairs in expert order (stable scatter)', 'các cặp theo thứ tự expert (rải ổn định)')} />
          {done === N ? <Slices t={0} y={MID + 10} /> : null}
          {boxes.map((b) => <PairBox key={b.i} b={b} />)}
        </Stage>
      );
    },
  },
  {
    clip: 'route-5',
    get label() {
      return tr('Run', 'Chạy');
    },
    draw: (p) => {
      const m = ease(ramp(p, 0.05, 0.3));
      // the GEMMs run one expert after another; the empty one is skipped
      const run = R.counts.map((c, e) => (c === 0 ? null : e)).filter((e): e is number => e != null);
      const at = ramp(p, 0.38, 0.98) * run.length;
      const hot = p >= 0.38 ? run[Math.min(run.length - 1, Math.floor(at))] : undefined;
      const boxes = R.pairs.map((pr, i) => ({ i, x: lerp(compactX(i), paddedX(i), m), y: MID, w: lerp(SW, PW, m), hot: hot === pr.expert }));
      return (
        <Stage beat={5} headline={hot == null ? tr(`slices padded to multiples of ${R.tile}: ${R.slices.pads} pad slots, every store unconditional`, `các lát được đệm lên bội của ${R.tile}: ${R.slices.pads} vị trí đệm, mọi lần ghi đều không điều kiện`) : tr(`GEMM: the ${R.counts[hot]} tokens of E${hot} × E${hot}'s weights; E${R.empty.join(', E')} skipped`, `GEMM: ${R.counts[hot]} token của E${hot} × trọng số của E${hot}; bỏ qua E${R.empty.join(', E')}`)}
          data={{ starts: R.slices.starts.join(','), pads: String(R.slices.pads), skip: R.empty.join(',') }}>
          <Matrix shown={N} />
          <Counters counts={R.counts} final />
          <Offsets shown={R.offsets.length} />
          <StripLabel y={TOP} text={tr('pairs in token order', 'các cặp theo thứ tự token')} />
          {ghosts()}
          <StripLabel y={MID} text={tr(`pairs in expert order, slices padded to tiles of ${R.tile}`, `các cặp theo thứ tự expert, các lát được đệm theo ô (tile) cỡ ${R.tile}`)} />
          <PadBoxes show={m} />
          <Slices t={m} y={MID + 10} hot={hot} />
          {boxes.map((b) => <PairBox key={b.i} b={b} />)}
        </Stage>
      );
    },
  },
  {
    clip: 'route-6',
    get label() {
      return tr('Put back', 'Đặt lại');
    },
    draw: (p) => {
      const boxes = R.pairs.map((_, i) => {
        const start = (R.paddedPos[i] / R.slices.total) * 0.35;
        const t = ease(ramp(p, start, start + 0.25));
        return { i, x: lerp(paddedX(i), topX(i), t), y: lerp(MID, BOT, t), w: lerp(PW, SW, t), hot: t > 0 && t < 1 };
      });
      const combine = ramp(p, 0.68, 0.85);
      return (
        <Stage beat={6} headline={combine < 1 ? tr('the inverse permutation (JDS\'s row array) sends each output to its token\'s slot', 'hoán vị ngược (mảng row của JDS) gửi mỗi đầu ra về vị trí của token của nó') : tr('each token combines its two experts\' outputs', 'mỗi token kết hợp đầu ra của hai expert của nó')}
          data={{ back: R.pairs.map((x) => x.slot).join(',') }}>
          <Matrix shown={N} />
          <Counters counts={R.counts} final />
          <Offsets shown={R.offsets.length} />
          <StripLabel y={TOP} text={tr('pairs in token order', 'các cặp theo thứ tự token')} />
          {ghosts()}
          <StripLabel y={MID} text={tr('expert outputs, in expert order', 'đầu ra của các expert, theo thứ tự expert')} />
          <PadBoxes show={1 - ramp(p, 0, 0.3)} />
          <StripLabel y={BOT} text={tr('expert outputs back in token order', 'đầu ra của các expert, trở lại thứ tự token')} />
          {boxes.map((b) => <PairBox key={b.i} b={b} />)}
          <g opacity={combine} data-combine={combine >= 1 ? 1 : 0}>
            {Array.from({ length: R.tokens }, (_, t) => {
              const a = topX(t * R.topk) + 3;
              const b = topX(t * R.topk + R.topk - 1) + SW - 3;
              const [e1, e2] = R.pairs.slice(t * R.topk, t * R.topk + R.topk).map((x) => x.expert);
              return (
                <g key={t}>
                  <path d={`M${a},${BOT + BH + 4} v6 H${b} v-6`} fill="none" stroke="var(--ink)" strokeWidth="1.3" />
                  <text x={(a + b) / 2} y={BOT + BH + 22} fontSize="10" textAnchor="middle">t{t} = E{e1} + E{e2}</text>
                </g>
              );
            })}
          </g>
        </Stage>
      );
    },
  },
];

export function RouteScene() {
  return <Scene chapter="ch17" title={tr('Routing tokens to experts', 'Định tuyến token tới expert')} beats={beats} />;
}
