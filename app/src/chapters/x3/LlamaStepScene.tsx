import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { JOB_LABEL, SYNTHETIC, stepLaunches, totals, type Launch } from '../../sim/llamastep';

// "One token through the step" (section X3.7): the launches of one decode step of the module's
// synthetic model, as lab X3.2 runs them, with the supplying lab lit at each launch. Seven beats,
// one clip each (content/x3/narration/llama-step-N.txt). Every box, shape and byte count comes
// from stepLaunches and totals in src/sim/llamastep.ts.
const CFG = SYNTHETIC;
const POS = 5;
const STEP = stepLaunches(CFG, POS);
const LAYER0 = STEP.filter((l) => l.layer === 0);
const HEAD_ = STEP.filter((l) => l.layer === null && l.n === 0);
const TAIL = STEP.filter((l) => l.layer === null && l.n > 0);
const LATE = CFG.seqLen - 1;

const W = 720;
const H = 330;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const MB = (b: number) => (b / 2 ** 20).toFixed(2) + ' MiB';
const KB = (b: number) => (b / 1024).toFixed(1) + ' KiB';

function Frame({ beat, headline, children }: { beat: number; headline: string; children: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={headline} data-beat={beat} style={{ display: 'block' }}>
      <text x={14} y={22} fontSize="13" fontWeight="600" data-headline="" style={{ fill: 'var(--ink)' }}>{headline}</text>
      {children}
    </svg>
  );
}

/** Where each launch's box sits: embed on the left, layer 0's 16 launches in two rows, the tail on the right. */
function place(l: Launch): { x: number; y: number } {
  if (l.layer === null && l.n === 0) return { x: 14, y: 70 };
  if (l.layer === 0) {
    const k = LAYER0.indexOf(l);
    return { x: 104 + (k % 8) * 66, y: k < 8 ? 70 : 150 };
  }
  const k = TAIL.indexOf(l);
  return { x: 104 + k * 132, y: 236 };
}

function Box({ l, lit, done }: { l: Launch; lit: boolean; done: boolean }) {
  const { x, y } = place(l);
  const wide = l.layer === null && l.n > 0;
  const w = wide ? 124 : 60;
  return (
    <g data-launch={l.n} data-job={l.job} data-lit={lit ? 1 : 0}>
      <rect x={x} y={y} width={w} height={54} rx="4"
        fill={lit ? 'var(--rule)' : 'var(--cell)'} stroke={lit ? 'var(--ink)' : 'var(--rule)'} strokeWidth={lit ? 2 : 1}
        opacity={done || lit ? 1 : 0.55} />
      <text x={x + w / 2} y={y + 18} fontSize="10.5" textAnchor="middle" fontWeight={lit ? 700 : 400} style={{ fill: 'var(--ink)' }}>{wide ? JOB_LABEL[l.job] : JOB_LABEL[l.job].replace('residual add', 'residual')}</text>
      <text x={x + w / 2} y={y + 33} fontSize="9" textAnchor="middle" fontWeight={l.exists ? 400 : 700} style={{ fill: 'var(--ink-2)' }}>{l.lab.replace(' (or 6.4, 15.6)', '').replace(' (yours)', '')}</text>
      <text x={x + w / 2} y={y + 46} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{l.milestone}</text>
    </g>
  );
}

function Pipeline({ lit, upto }: { lit: string[]; upto: number }) {
  const all = [...HEAD_, ...LAYER0, ...TAIL];
  return (
    <g>
      <text x={104} y={62} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>lớp 0 (rồi tương tự cho các lớp 1 đến {CFG.layers - 1})</>}>layer 0 (then the same for layers 1 to {CFG.layers - 1})</Translated>
      </text>
      <text x={104} y={228} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{tr('after the last layer', 'sau lớp cuối cùng')}</text>
      {all.map((l) => <Box key={l.n} l={l} lit={lit.includes(String(l.n))} done={l.n < upto} />)}
      <text x={14} y={318} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
        <Translated
          vi={
            <>
              các lab in đậm là mới trong học phần này (X3.1, X3.2); d {CFG.d}, {CFG.heads} head query trên {CFG.kvHeads} head KV kích thước 128, hidden {CFG.hidden}, từ vựng {CFG.vocab}
            </>
          }
        >
          labs in bold are new in this module (X3.1, X3.2); d {CFG.d}, {CFG.heads} query heads over {CFG.kvHeads} KV heads of 128, hidden {CFG.hidden}, vocabulary {CFG.vocab}
        </Translated>
      </text>
    </g>
  );
}

const nOf = (layer: number | null, jobs: string[]) => STEP.filter((l) => l.layer === layer && jobs.includes(l.job)).map((l) => String(l.n));
const shapeLine = (l: Launch) => ((f: Launch) => `${f.fn}: ${f.inputs} → ${f.outputs}`)(stepLaunches(CFG, POS)[l.n]);

const beats: Beat[] = [
  {
    clip: 'llama-step-1',
    get label() { return tr('Embed', 'Embedding'); },
    draw: (p) => (
      <Frame beat={1} headline={tr(`token in: the embedding row is copied into x (${CFG.d} floats, ${KB(STEP[0].weightBytes)} of weights)`, `token vào: hàng embedding được chép vào x (${CFG.d} số float, ${KB(STEP[0].weightBytes)} trọng số)`)}>
        <Pipeline lit={ramp(p, 0.1, 0.2) > 0 ? ['0'] : []} upto={0} />
        <text x={14} y={300} fontSize="10.5" opacity={ramp(p, 0.3, 0.5)} style={{ fill: 'var(--ink)' }} data-shape="embed">{shapeLine(STEP[0])}</text>
      </Frame>
    ),
  },
  {
    clip: 'llama-step-2',
    get label() { return tr('Norm and q, k, v', 'Chuẩn hóa và q, k, v'); },
    draw: (p) => {
      const lit = [...nOf(0, ['rmsnorm']).slice(0, 1), ...(p > 0.35 ? nOf(0, ['q', 'k', 'v']) : [])];
      const q = STEP.find((l) => l.layer === 0 && l.job === 'q')!;
      return (
        <Frame beat={2} headline={tr("RMSNorm (lab X3.1), then three GEMVs (lab 22.1's gather at batch 1)", 'RMSNorm (lab X3.1), rồi ba GEMV (bản gather của lab 22.1 ở batch 1)')}>
          <Pipeline lit={lit} upto={1} />
          <text x={14} y={300} fontSize="10.5" opacity={ramp(p, 0.4, 0.6)} style={{ fill: 'var(--ink)' }} data-shape="q">{shapeLine(q)}, {MB(q.weightBytes)}</text>
        </Frame>
      );
    },
  },
  {
    clip: 'llama-step-3',
    get label() { return tr('RoPE and append', 'RoPE và nối thêm'); },
    draw: (p) => {
      const ap = STEP.find((l) => l.layer === 0 && l.job === 'append')!;
      return (
        <Frame beat={3} headline={tr(`RoPE on q and k at position ${POS}; k and v go to cache row ${POS}`, `RoPE trên q và k tại vị trí ${POS}; k và v vào hàng cache ${POS}`)}>
          <Pipeline lit={p < 0.5 ? nOf(0, ['rope-q', 'rope-k']) : nOf(0, ['append'])} upto={4} />
          <text x={14} y={300} fontSize="10.5" opacity={ramp(p, 0.5, 0.7)} style={{ fill: 'var(--ink)' }} data-shape="append">{shapeLine(ap)}</text>
        </Frame>
      );
    },
  },
  {
    clip: 'llama-step-4',
    get label() { return tr('Attention', 'Attention'); },
    draw: (p) => {
      const at = STEP.find((l) => l.layer === 0 && l.job === 'attention')!;
      return (
        <Frame beat={4} headline={tr(`decode attention over ${POS + 1} cached rows per KV head (lab 20.5, grouped), then Wo and a residual add`, `attention decode trên ${POS + 1} hàng trong cache mỗi head KV (lab 20.5, theo nhóm), rồi Wo và một phép cộng residual`)}>
          <Pipeline lit={p < 0.55 ? nOf(0, ['attention']) : nOf(0, ['wo', 'residual']).slice(0, 2)} upto={7} />
          <text x={14} y={300} fontSize="10.5" opacity={ramp(p, 0.2, 0.4)} style={{ fill: 'var(--ink)' }} data-shape="attention">
            <Translated vi={<>{at.fn}: đọc {KB(at.cacheBytes)} cache tại vị trí {POS}</>}>
              {at.fn}: reads {KB(at.cacheBytes)} of cache at pos {POS}
            </Translated>
          </text>
        </Frame>
      );
    },
  },
  {
    clip: 'llama-step-5',
    get label() { return tr('Feed-forward', 'Feed-forward'); },
    draw: (p) => {
      const jobs = ['rmsnorm', 'w1', 'w3', 'swiglu', 'w2', 'residual'];
      const ffn = LAYER0.slice(10);
      const k = Math.min(ffn.length - 1, Math.floor(p * ffn.length * 1.2));
      return (
        <Frame beat={5} headline={tr(`the feed-forward: RMSNorm, W1 and W3, SwiGLU, W2, residual; then ${CFG.layers - 1} more layers`, `phần feed-forward: RMSNorm, W1 và W3, SwiGLU, W2, residual; rồi thêm ${CFG.layers - 1} lớp nữa`)}>
          <Pipeline lit={[String(ffn[k].n)]} upto={ffn[k].n} />
          <text x={14} y={300} fontSize="10.5" style={{ fill: 'var(--ink)' }} data-ffn={jobs.join(',')}>{shapeLine(ffn[k])}</text>
        </Frame>
      );
    },
  },
  {
    clip: 'llama-step-6',
    get label() { return tr('Logits and argmax', 'Logits và argmax'); },
    draw: (p) => {
      const lm = TAIL.find((l) => l.job === 'lm-head')!;
      return (
        <Frame beat={6} headline={tr(`final RMSNorm, the tied LM head (${MB(lm.weightBytes)}), lab 10.3's argmax`, `RMSNorm cuối, LM head dùng chung trọng số (${MB(lm.weightBytes)}), argmax của lab 10.3`)}>
          <Pipeline lit={p < 0.3 ? nOf(null, ['final-norm']) : p < 0.65 ? nOf(null, ['lm-head']) : nOf(null, ['argmax', 'advance'])} upto={STEP.length - 4} />
          <text x={14} y={300} fontSize="10.5" opacity={ramp(p, 0.3, 0.5)} style={{ fill: 'var(--ink)' }} data-shape="lm-head">{shapeLine(lm)}</text>
        </Frame>
      );
    },
  },
  {
    clip: 'llama-step-7',
    get label() { return tr('Totals', 'Tổng cộng'); },
    draw: (p) => {
      const pos = Math.round(ramp(p, 0.2, 0.9) * LATE);
      const t = totals(stepLaunches(CFG, pos));
      const all = t.weightBytes + t.actBytes + t.cacheBytes + t.tableBytes;
      const parts: [string, number][] = [
        [tr('matrix products', 'phép nhân ma trận'), t.byKind.matmul],
        [tr('attention (cache reads)', 'attention (đọc cache)'), t.byKind.attention],
        [tr('everything else', 'mọi thứ khác'), all - t.byKind.matmul - t.byKind.attention],
      ];
      return (
        <Frame beat={7} headline={tr(`${t.launches} launches per token; ${MB(t.weightBytes)} of weights read per token`, `${t.launches} lần khởi chạy mỗi token; ${MB(t.weightBytes)} trọng số được đọc mỗi token`)}>
          <text x={14} y={64} fontSize="11.5" style={{ fill: 'var(--ink)' }} data-pos={pos}>
            <Translated vi={<>số byte di chuyển mỗi token tại vị trí {pos}: {MB(all)}</>}>bytes moved per token at pos {pos}: {MB(all)}</Translated>
          </text>
          {parts.map(([name, b], i) => (
            <g key={name} data-part={name} data-bytes={b}>
              <text x={14} y={110 + i * 56} fontSize="11" style={{ fill: 'var(--ink)' }}>{name}</text>
              <rect x={200} y={96 + i * 56} width={480} height={20} fill="none" stroke="var(--rule)" />
              <rect x={200} y={96 + i * 56} width={(480 * b) / all} height={20} fill="var(--ink-2)" />
              <text x={200} y={132 + i * 56} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{((100 * b) / all).toFixed(1)} % ({MB(b)})</text>
            </g>
          ))}
          <text x={14} y={300} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
            {tr('the weights are the same at every pos; the cache that attention reads grows by one row per layer per step', 'trọng số như nhau ở mọi vị trí; cache mà attention đọc lớn thêm một hàng mỗi lớp mỗi bước')}
          </text>
        </Frame>
      );
    },
  },
];

export function LlamaStepScene() {
  return <Scene chapter="x3" title={tr('One token through the step', 'Một token đi qua một bước')} beats={beats} />;
}
