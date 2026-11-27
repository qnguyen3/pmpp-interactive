import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { SCENE_S, STEPS, V, compare, type StepRow } from '../../sim/tokencompare';

// "A test that fails late" (section X3.9): the toy model of src/sim/tokencompare.ts at
// perturbation size SCENE_S, where exactly one step (19) flips. Five beats, one narration
// clip each (content/x3/narration/late-divergence-N.txt). Every logit, token and margin drawn
// comes from compare(SCENE_S), pinned in tokencompare.test.ts.
const C = compare(SCENE_S);
const FLIP = C.tfDiffers[0];

const W = 720;
const H = 402;
const PANEL_W = 300;
const PX = [30, 390]; // left edges of the reference and GPU panels
const PY = 34; // top of the bar area
const PH = 130; // bar area height
const LO = -2.5;
const HI = 7.5;
const MONO = { fontVariationSettings: "'MONO' 1" } as const; // i18n-ignore: CSS value
const TX = 128; // token rows
const TW = 24;
const ROWS_Y = [268, 300, 332];

const by = (v: number) => PY + PH - ((v - LO) / (HI - LO)) * PH;
const clampStep = (x: number, a: number, b: number) => Math.max(a, Math.min(b, Math.floor(x)));
const f7 = (v: number) => v.toFixed(7);

function Bars({ x, logits, label, pick, other }: { x: number; logits: number[]; label: string; pick: number; other?: number }) {
  const bw = PANEL_W / V;
  return (
    <g>
      <text x={x} y={PY - 12} fontSize="12" fontWeight="600" style={{ fill: 'var(--ink)' }}>{label}</text>
      <line x1={x} y1={by(0)} x2={x + PANEL_W} y2={by(0)} stroke="var(--rule)" />
      {logits.map((v, i) => {
        const y0 = by(0);
        const y1 = by(v);
        const top = Math.min(y0, y1);
        const fill = i === pick ? (other != null && other !== pick ? 'var(--fail)' : 'var(--ink)') : 'var(--ink-3)';
        return (
          <rect key={i} x={x + i * bw + 1.5} y={top} width={bw - 3} height={Math.max(1, Math.abs(y1 - y0))} rx="1.5"
            fill={fill} opacity={i === pick ? 1 : 0.45} data-token={i} />
        );
      })}
      {logits.map((_, i) => (
        <text key={'n' + i} x={x + i * bw + bw / 2} y={PY + PH + 13} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)', ...MONO }}>{i}</text>
      ))}
    </g>
  );
}

interface Frame {
  t: number;
  /** Which GPU run the right-hand panel shows. */
  gpu: 'fr' | 'tf';
  /** Tokens shown up to and including this step, per row (-1: none). */
  upto: [number, number, number];
  /** Show the teacher-forced row at all. */
  tfRow: boolean;
  zoom?: boolean;
  note?: ReactNode;
  headline: ReactNode;
}

function Tokens({ f }: { f: Frame }) {
  const rows: { label: string; tok: (r: StepRow) => number; upto: number; show: boolean }[] = [
    { label: tr('reference', 'tham chiếu'), tok: (r) => r.refTok, upto: f.upto[0], show: true },
    { label: tr('GPU, free running', 'GPU, chạy tự do'), tok: (r) => r.frTok, upto: f.upto[1], show: true },
    { label: tr('GPU, teacher forced', 'GPU, teacher forcing'), tok: (r) => r.tfTok, upto: f.upto[2], show: f.tfRow },
  ];
  return (
    <g>
      {C.steps.map((r) => (
        <text key={'s' + r.t} x={TX + r.t * TW + TW / 2} y={ROWS_Y[0] - 14} fontSize="9" textAnchor="middle"
          style={{ fill: r.t === f.t ? 'var(--ink)' : 'var(--ink-3)', fontWeight: r.t === f.t ? 700 : 400, ...MONO }}>{r.t}</text>
      ))}
      {rows.map((row, k) =>
        row.show ? (
          <g key={row.label}>
            <text x={TX - 8} y={ROWS_Y[k] + 15} fontSize="11" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{row.label}</text>
            {C.steps.map((r) => {
              const shown = r.t <= row.upto;
              const bad = shown && k > 0 && row.tok(r) !== r.refTok;
              return (
                <g key={r.t}>
                  <rect x={TX + r.t * TW + 1} y={ROWS_Y[k]} width={TW - 2} height={22} rx="2" data-row={k} data-t={r.t} data-shown={shown ? 'yes' : 'no'} data-bad={bad ? 'yes' : 'no'}
                    fill={bad ? 'var(--fail-wash)' : 'var(--cell)'} stroke={bad ? 'var(--fail)' : 'var(--rule)'} />
                  {shown ? (
                    <text x={TX + r.t * TW + TW / 2} y={ROWS_Y[k] + 15} fontSize="11" textAnchor="middle"
                      style={{ fill: bad ? 'var(--fail)' : 'var(--ink)', ...MONO }}>{row.tok(r)}</text>
                  ) : null}
                </g>
              );
            })}
          </g>
        ) : null,
      )}
      {f.tfRow && f.upto[2] >= FLIP ? (
        <text x={TX + FLIP * TW + TW / 2} y={ROWS_Y[2] + 36} fontSize="10" textAnchor="middle" style={{ fill: 'var(--fail)' }}>
          {tr('flagged: margin below 2 × error', 'bị gắn cờ: khoảng cách nhỏ hơn 2 × sai số')}
        </text>
      ) : null}
    </g>
  );
}

function Stage(f: Frame) {
  const r = C.steps[f.t];
  const gpu = f.gpu === 'fr' ? r.fr : r.tf;
  const gpuTok = f.gpu === 'fr' ? r.frTok : r.tfTok;
  const runnerRef = [...r.ref.keys()].sort((a, b) => r.ref[b] - r.ref[a])[1];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={tr('Two runs of a toy model compared step by step', 'Hai lần chạy của một mô hình đồ chơi được so sánh từng bước')}
      data-step={f.t} data-ref-tok={r.refTok} data-gpu-tok={gpuTok} data-gpu={f.gpu} style={{ display: 'block' }}>
      <Bars x={PX[0]} logits={r.ref} label={tr(`reference, step ${f.t}: input ${r.refIn}`, `tham chiếu, bước ${f.t}: đầu vào ${r.refIn}`)} pick={r.refTok} />
      <Bars x={PX[1]} logits={gpu} label={tr(`GPU, ${f.gpu === 'fr' ? 'free running' : 'teacher forced'}, step ${f.t}: input ${f.gpu === 'fr' ? r.frIn : r.refIn}`, `GPU, ${f.gpu === 'fr' ? 'chạy tự do' : 'teacher forcing'}, bước ${f.t}: đầu vào ${f.gpu === 'fr' ? r.frIn : r.refIn}`)}
        pick={gpuTok} other={r.refTok} />
      {f.zoom ? (
        <g style={MONO}>
          {[0, 1].map((side) => {
            const l = side === 0 ? r.ref : gpu;
            const pick = side === 0 ? r.refTok : gpuTok;
            return [r.refTok, runnerRef].map((tok, j) => (
              <text key={side + '-' + tok} x={PX[side]} y={PY + PH + 50 + j * 15} fontSize="11.5"
                style={{ fill: tok === pick ? (side === 1 && pick !== r.refTok ? 'var(--fail)' : 'var(--ink)') : 'var(--ink-2)', fontWeight: tok === pick ? 700 : 400 }}>
                <Translated vi={<>token {tok}: {f7(l[tok])}</>}>token {tok}: {f7(l[tok])}</Translated>
              </text>
            ));
          })}
        </g>
      ) : null}
      <text x={PX[0]} y={PY + PH + 32} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated
          vi={<>chọn <tspan style={{ fill: 'var(--ink)', fontWeight: 700 }}>{r.refTok}</tspan>; khoảng cách {r.margin < 1e-3 ? r.margin.toExponential(1) : r.margin.toFixed(3)}</>}
        >
          picks <tspan style={{ fill: 'var(--ink)', fontWeight: 700 }}>{r.refTok}</tspan>; margin {r.margin < 1e-3 ? r.margin.toExponential(1) : r.margin.toFixed(3)}
        </Translated>
      </text>
      <text x={PX[1]} y={PY + PH + 32} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="chọn ">picks </Translated>
        <tspan style={{ fill: gpuTok === r.refTok ? 'var(--ink)' : 'var(--fail)', fontWeight: 700 }}>{gpuTok}</tspan>
        {f.note ? <>; {f.note}</> : null}
      </text>
      <Tokens f={f} />
      <text x={20} y={H - 8} fontSize="13" fontWeight="600" style={{ fill: 'var(--ink)' }}>{f.headline}</text>
    </svg>
  );
}

const err = (r: StepRow, which: 'fr' | 'tf') => Math.max(...r.ref.map((v, i) => Math.abs(v - (which === 'fr' ? r.fr : r.tf)[i])));

const beats: Beat[] = [
  {
    clip: 'late-divergence-1',
    get label() { return tr('Two runs', 'Hai lần chạy'); },
    draw: (p) => {
      const t = clampStep(p * 4, 0, 3);
      return <Stage t={t} gpu="fr" upto={[t, t, -1]} tfRow={false} headline={tr('same model, two runs: the bars differ only by rounding', 'cùng mô hình, hai lần chạy: các cột chỉ khác nhau do làm tròn')} />;
    },
  },
  {
    clip: 'late-divergence-2',
    get label() { return tr('Wide margins', 'Khoảng cách rộng'); },
    draw: (p) => {
      const t = clampStep(4 + p * 15, 4, FLIP - 1);
      const r = C.steps[t];
      return (
        <Stage t={t} gpu="fr" upto={[t, t, -1]} tfRow={false} note={<Translated vi={<>chênh lệch lớn nhất {err(r, 'fr').toExponential(1)}</>}>largest difference {err(r, 'fr').toExponential(1)}</Translated>}
          headline={tr(`margin ${r.margin.toFixed(2)} against a difference of about ${SCENE_S.toExponential(0)}: the tokens match`, `khoảng cách ${r.margin.toFixed(2)} so với chênh lệch khoảng ${SCENE_S.toExponential(0)}: các token khớp nhau`)} />
      );
    },
  },
  {
    clip: 'late-divergence-3',
    get label() { return tr('One close step', 'Một bước sát nút'); },
    draw: () => {
      const r = C.steps[FLIP];
      return (
        <Stage t={FLIP} gpu="fr" upto={[FLIP, FLIP, -1]} tfRow={false} zoom note={<Translated vi={<>chênh lệch lớn nhất {err(r, 'fr').toExponential(1)}</>}>largest difference {err(r, 'fr').toExponential(1)}</Translated>}
          headline={tr(`step ${FLIP}: top two ${r.margin.toExponential(0)} apart, the GPU picks the runner-up`, `bước ${FLIP}: hai giá trị lớn nhất cách nhau ${r.margin.toExponential(0)}, GPU chọn giá trị đứng thứ hai`)} />
      );
    },
  },
  {
    clip: 'late-divergence-4',
    get label() { return tr('Free running', 'Chạy tự do'); },
    draw: (p) => {
      const t = clampStep(FLIP + 1 + p * (STEPS - FLIP - 1), FLIP + 1, STEPS - 1);
      const r = C.steps[t];
      return (
        <Stage t={t} gpu="fr" upto={[t, t, -1]} tfRow={false} note={<Translated vi={<>đầu vào khác, NMSE {r.frNmse.toFixed(1)}</>}>different input, NMSE {r.frNmse.toFixed(1)}</Translated>}
          headline={tr(`free running: ${C.frDiffers.filter((x) => x <= t).length} tokens differ from step ${FLIP} on; no kernel is wrong`, `chạy tự do: ${C.frDiffers.filter((x) => x <= t).length} token khác nhau kể từ bước ${FLIP}; không kernel nào sai`)} />
      );
    },
  },
  {
    clip: 'late-divergence-5',
    get label() { return tr('Teacher forced', 'Teacher forcing'); },
    draw: (p) => {
      const t = clampStep(p * STEPS * 1.15, 0, STEPS - 1);
      return (
        <Stage t={t} gpu="tf" upto={[STEPS - 1, STEPS - 1, t]} tfRow
          note={<>NMSE {C.steps[t].tfNmse.toExponential(1)}</>}
          headline={tr(`teacher forced: only step ${FLIP} differs, and it is flagged`, `teacher forcing: chỉ bước ${FLIP} khác, và nó bị gắn cờ`)} />
      );
    },
  },
];

export function LateDivergenceScene() {
  return <Scene chapter="x3" title={tr('A test that fails late', 'Một bài kiểm thử hỏng muộn')} beats={beats} />;
}
