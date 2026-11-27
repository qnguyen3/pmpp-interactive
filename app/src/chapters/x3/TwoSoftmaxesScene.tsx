import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { KERNELS, ROW16, maxUlps, online, twoPass, type RowStep, type RowTrace } from '../../sim/rowreduce';

// "One framework, two softmaxes": section X3.3's row through lab 10.3's three loops (the
// order of reductions MLX uses on Metal) and through MLX CUDA's online kernel. Every number
// drawn is a step of src/sim/rowreduce.ts; nothing is typed in by hand.
// Clips: content/x3/narration/two-softmaxes-N.txt.
export const TP = twoPass(ROW16);
export const ON = online(ROW16);
export const SCENE_ULPS = maxUlps(TP.out, ON.out);
const N = ROW16.length;

const W = 720;
const H = 300;
const X0 = 40;
const CW = 36; // cell width
const GAP = 12; // gap between chunks of 4
const ROW_Y = 78;
const OUT_Y = 218;
const cellX = (i: number) => X0 + i * CW + Math.floor(i / 4) * GAP;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const f2 = (v: number) => v.toFixed(2);
const f3 = (v: number) => v.toFixed(3);
const show = (v: number) => (v === -Infinity ? '−∞' : f2(v));

/** The step of `steps[from..to)` that part p of a beat shows. */
const pick = (t: RowTrace, from: number, to: number, p: number) => t.steps[Math.min(to - 1, from + Math.floor(clamp(p) * (to - from)))];
const firstOf = (t: RowTrace, pass: number) => t.steps.findIndex((s) => s.pass === pass);
const lastOf = (t: RowTrace, pass: number) => t.steps.map((s) => s.pass).lastIndexOf(pass) + 1;

function Frame({ headline, sub, children }: { headline: string; sub?: string; children: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('One softmax row, two ways', 'Một hàng softmax, hai cách')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <text x={X0 - 4} y={22} fontSize="14" fontWeight="650" data-headline>{headline}</text>
      {sub ? <text x={X0 - 4} y={40} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{sub}</text> : null}
      {children}
    </svg>
  );
}

/** The row in global memory; the step's chunk highlighted. */
function Row({ step, label }: { step?: RowStep; label: string }) {
  const hot = new Set(step?.cols ?? []);
  return (
    <g data-part="row">
      <text x={X0} y={ROW_Y - 8} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>{label}</text>
      {ROW16.map((v, i) => (
        <g key={i} data-col={i} data-hot={hot.has(i) ? 1 : 0}>
          <rect x={cellX(i)} y={ROW_Y} width={CW - 3} height={28} rx="2"
            fill={hot.has(i) ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
          <text x={cellX(i) + (CW - 3) / 2} y={ROW_Y + 18} fontSize="11" textAnchor="middle"
            style={{ fill: hot.has(i) ? 'var(--page)' : 'var(--ink)' }}>{v.toFixed(1)}</text>
        </g>
      ))}
    </g>
  );
}

/** What the threads keep in registers, and the traffic so far. */
function State({ step, sumName, factor }: { step: RowStep; sumName: string; factor?: boolean }) {
  const y = 140;
  return (
    <g data-part="state" data-max={step.max} data-sum={step.sum} data-reads={step.counters.reads / N} data-writes={step.counters.writes / N}>
      <text x={X0} y={y} fontSize="11" fontWeight="600" style={{ fill: 'var(--reg)' }}>{tr('in registers', 'trong thanh ghi')}</text>
      <text x={X0} y={y + 20} fontSize="13" style={{ fill: 'var(--reg)' }}>max = {show(step.max)}</text>
      <text x={X0 + 130} y={y + 20} fontSize="13" style={{ fill: 'var(--reg)' }}>{sumName} = {f3(step.sum)}</text>
      {factor && step.factor != null ? (
        <text x={X0 + 300} y={y + 20} fontSize="13" data-factor={step.factor} style={{ fill: 'var(--reg)' }}>
          <Translated vi={<>co giãn lại × exp({show(step.prevMax!)} − {show(step.max)}) = {f2(step.factor)}</>}>
            rescale × exp({show(step.prevMax!)} − {show(step.max)}) = {f2(step.factor)}
          </Translated>
        </text>
      ) : null}
      <text x={X0} y={y + 44} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated
          vi={
            <>
              số lần đọc hàng tới giờ: {(step.counters.reads / N).toFixed(2).replace(/\.00$/, '')} · số lần ghi: {(step.counters.writes / N).toFixed(2).replace(/\.00$/, '')} · số phép mũ: {step.counters.exps}
            </>
          }
        >
          reads of the row so far: {(step.counters.reads / N).toFixed(2).replace(/\.00$/, '')} · writes: {(step.counters.writes / N).toFixed(2).replace(/\.00$/, '')} · exponentials: {step.counters.exps}
        </Translated>
      </text>
    </g>
  );
}

function Outputs({ out, label = tr('output (global memory)', 'đầu ra (bộ nhớ toàn cục)') }: { out: (number | null)[]; label?: string }) {
  return (
    <g data-part="out">
      <text x={X0} y={OUT_Y - 8} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>{label}</text>
      {out.map((v, i) => (
        <g key={i} data-out={v == null ? '' : v.toFixed(4)}>
          <rect x={cellX(i)} y={OUT_Y} width={CW - 3} height={28} rx="2" fill={v == null ? 'transparent' : 'var(--cell)'}
            stroke="var(--global)" strokeDasharray={v == null ? '3 2' : undefined} />
          {v != null ? <text x={cellX(i) + (CW - 3) / 2} y={OUT_Y + 18} fontSize="9.5" textAnchor="middle">{v.toFixed(3)}</text> : null}
        </g>
      ))}
    </g>
  );
}

const describe = (s: RowStep) =>
  s.op === 'max' ? tr(`pass 1: max over chunk ${s.chunk}`, `lượt 1: max trên đoạn ${s.chunk}`)
    : s.op === 'sum' ? tr(`pass 2: add exp(x − max) for chunk ${s.chunk}`, `lượt 2: cộng exp(x − max) cho đoạn ${s.chunk}`)
      : s.op === 'inv' ? tr('one reciprocal: 1 / sum', 'một phép nghịch đảo: 1 / sum')
        : s.op === 'write' ? tr(`pass ${s.pass}: write exp(x − max) × (1 / sum), chunk ${s.chunk}`, `lượt ${s.pass}: ghi exp(x − max) × (1 / sum), đoạn ${s.chunk}`)
          : s.op === 'chunk-max' ? tr(`chunk ${s.chunk}: new max ${show(s.max)}`, `đoạn ${s.chunk}: max mới ${show(s.max)}`)
            : s.op === 'rescale' ? tr(`chunk ${s.chunk}: rescale the sum by exp(${show(s.prevMax!)} − ${show(s.max)})`, `đoạn ${s.chunk}: co giãn lại tổng theo exp(${show(s.prevMax!)} − ${show(s.max)})`)
              : tr(`chunk ${s.chunk}: add its 4 terms`, `đoạn ${s.chunk}: cộng 4 số hạng của nó`);

const mlxMetal = KERNELS.find((k) => k.id === 'mlx-metal-softmax')!;
const mlxCuda = KERNELS.find((k) => k.id === 'mlx-cuda-softmax')!;

const beats: Beat[] = [
  {
    clip: 'two-softmaxes-1',
    get label() { return tr('Find the max', 'Tìm max'); },
    draw: (p) => {
      const s = pick(TP, firstOf(TP, 1), lastOf(TP, 1), p);
      return (
        <Frame headline={tr('max, then sum: sweep 1 looks only for the maximum', 'max, rồi sum: lượt quét 1 chỉ tìm giá trị lớn nhất')} sub={describe(s)}>
          <Row step={s} label={tr('one row of 16 logits (global memory), one block', 'một hàng 16 logits (bộ nhớ toàn cục), một khối')} />
          <State step={s} sumName="sum" />{/* i18n-ignore: variable name in the kernel */}
          <Outputs out={s.out} />
        </Frame>
      );
    },
  },
  {
    clip: 'two-softmaxes-2',
    get label() { return tr('Sum, then write', 'Cộng, rồi ghi'); },
    draw: (p) => {
      const s = pick(TP, firstOf(TP, 2), TP.steps.length, p);
      return (
        <Frame headline={s.pass === 2 ? tr('sweep 2: the sum of exponentials', 'lượt quét 2: tổng các phép mũ') : tr('sweep 3: compute each exponential again and write', 'lượt quét 3: tính lại từng phép mũ và ghi')} sub={describe(s)}>
          <Row step={s} label={tr('the same row, read again', 'cùng hàng đó, đọc lại')} />
          <State step={s} sumName="sum" />{/* i18n-ignore: variable name in the kernel */}
          <Outputs out={s.out} />
        </Frame>
      );
    },
  },
  {
    clip: 'two-softmaxes-3',
    get label() { return tr('Online', 'Online'); },
    draw: (p) => {
      const s = pick(ON, 0, lastOf(ON, 1), p);
      return (
        <Frame headline={tr('MLX on CUDA: chunks of 4, a running max and a running sum', 'MLX trên CUDA: các đoạn 4 phần tử, một max chạy và một tổng chạy')} sub={describe(s)}>
          <Row step={s} label={tr('the row, one chunk of 4 at a time', 'hàng đó, mỗi lần một đoạn 4 phần tử')} />
          <State step={s} sumName={tr('normaliser', 'hệ số chuẩn hóa')} factor={s.op === 'rescale' || s.op === 'add'} />
          <Outputs out={s.out} />
        </Frame>
      );
    },
  },
  {
    clip: 'two-softmaxes-4',
    get label() { return tr('Write', 'Ghi'); },
    draw: (p) => {
      const s = pick(ON, lastOf(ON, 1) - 1, ON.steps.length, p);
      const done = s.out.every((v) => v != null);
      return (
        <Frame headline={done ? tr(`two reads and one write; the outputs agree to within ${SCENE_ULPS} units in the last place`, `hai lần đọc và một lần ghi; các đầu ra khớp nhau trong phạm vi ${SCENE_ULPS} đơn vị ở chữ số cuối (ULP)`) : tr('the second sweep writes each exponential times 1 / normaliser', 'lượt quét thứ hai ghi từng phép mũ nhân 1 / hệ số chuẩn hóa')}
          sub={describe(s)}>
          <Row step={s} label={tr('the row, read a second time', 'hàng đó, đọc lần thứ hai')} />
          <State step={s} sumName={tr('normaliser', 'hệ số chuẩn hóa')} />
          <Outputs out={s.out} label={tr('output (online)', 'đầu ra (online)')} />
          {done ? (
            <g data-part="compare" data-ulps={SCENE_ULPS}>
              <text x={X0} y={OUT_Y + 50} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>
                <Translated
                  vi={
                    <>
                      chênh lệch lớn nhất so với max-rồi-sum: {SCENE_ULPS} đơn vị ở chữ số cuối của một float32 (tổng {TP.sum.toPrecision(8)} và {ON.sum.toPrecision(8)})
                    </>
                  }
                >
                  largest difference from max-then-sum: {SCENE_ULPS} units in the last place of a float32 (sums {TP.sum.toPrecision(8)} and {ON.sum.toPrecision(8)})
                </Translated>
              </text>
            </g>
          ) : null}
        </Frame>
      );
    },
  },
  {
    clip: 'two-softmaxes-5',
    get label() { return tr('Compared', 'So sánh'); },
    draw: (p) => {
      const rows: [string, string, string][] = [
        [tr('reads of the row', 'số lần đọc hàng'), String(TP.rowReads), String(ON.rowReads)],
        [tr('writes', 'số lần ghi'), '1', '1'],
        [tr('exponentials', 'số phép mũ'), String(TP.counters.exps), String(ON.counters.exps)],
        [tr('max rises after the first chunk', 'số lần max tăng sau đoạn đầu'), '–', String(ON.rises)],
        [tr('MLX uses it on', 'MLX dùng nó trên'), 'Metal', 'CUDA'],
        [tr('its exponential', 'hàm mũ của nó'), mlxMetal.exp, mlxCuda.exp],
      ];
      const shown = Math.max(1, Math.ceil(clamp(p / 0.7) * rows.length));
      return (
        <Frame headline={tr("chapter 20's online softmax, applied to one row", 'online softmax của chương 20, áp dụng cho một hàng')} sub={tr('one read saved, one extra exponential per chunk', 'tiết kiệm một lần đọc, thêm một phép mũ mỗi đoạn')}>
          <g data-part="table">
            <text x={X0 + 250} y={80} fontSize="12" fontWeight="650">{tr('max, then sum', 'max, rồi sum')}</text>
            <text x={X0 + 440} y={80} fontSize="12" fontWeight="650">{tr('online', 'online')}</text>
            {rows.slice(0, shown).map(([name, a, b], k) => (
              <g key={name} data-row={name} data-a={a} data-b={b}>
                <text x={X0} y={110 + k * 28} fontSize="12.5" style={{ fill: 'var(--ink-2)' }}>{name}</text>
                <text x={X0 + 250} y={110 + k * 28} fontSize="13">{a}</text>
                <text x={X0 + 440} y={110 + k * 28} fontSize="13">{b}</text>
              </g>
            ))}
          </g>
        </Frame>
      );
    },
  },
];

export function TwoSoftmaxesScene() {
  return <Scene chapter="x3" title={tr('One framework, two softmaxes', 'Một framework, hai softmax')} beats={beats} />;
}
