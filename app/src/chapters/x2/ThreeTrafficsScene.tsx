import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { byteRatio, bytesPerWeight, layer, passBytes, pipelinePasses, pipelineRelative, type Pass, type PipelineId } from '../../sim/qtraffic';
import { Translated, tr } from '../../lib/i18n';

// "Three ways to feed one GEMV": the same 4096 → 4096 layer at batch 1 with float32
// activations, fed by float32 weights, by Q4_0 weights dequantised in a separate kernel, and
// by Q4_0 weights unpacked inside the GEMV. At a fixed bandwidth, bytes are time: every
// stream's length and every finishing time is computed by qtraffic.ts (pipelinePasses), on a
// clock whose unit is the float32 GEMV's time. Five beats, one narration clip each
// (content/x2/narration/three-traffics-N.txt).

const L = layer('4096x4096');
const B = 1;
const ACT = 4;
const F: 'q4_0' = 'q4_0';
const ROWS: { id: PipelineId; label: string }[] = [
  { id: 'f32', get label() { return tr('float32 weights', 'trọng số float32'); } },
  { id: 'unfused-f32', get label() { return tr('Q4_0, dequantised first', 'Q4_0, giải lượng tử hóa trước'); } },
  { id: 'fused', get label() { return tr('Q4_0, fused', 'Q4_0, gộp'); } },
];
const passes = (id: PipelineId) => pipelinePasses(id, F, L, B, ACT);
const F32_BYTES = passBytes(passes('f32'));
/** Finishing time of a pipeline, in units of the float32 GEMV. */
const finish = (id: PipelineId) => pipelineRelative(id, F, L, B, ACT);
const T_MAX = finish('unfused-f32');

const W = 720;
const H = 350;
const X0 = 150; // where time 0 sits on the tracks
const SCALE = (W - X0 - 40) / T_MAX; // pixels per float32-GEMV time
const tx = (t: number) => X0 + t * SCALE;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const f2 = (v: number) => v.toFixed(2);

const SHORT: Partial<Record<Pass['kind'], string>> = {
  get 'read-weights'() { return tr('read weights', 'đọc trọng số'); },
  get 'write-copy'() { return tr('write the copy', 'ghi bản sao'); },
  get 'read-copy'() { return tr('read it back', 'đọc lại'); },
};
const fillOf = (k: Pass['kind']) => (k === 'write-copy' || k === 'read-copy' ? 'var(--global-wash)' : 'var(--global)');

/** The pipe between memory and the kernel, with what is crossing it now. */
function Pipe({ label, toMemory, flow, unpack }: { label: string; toMemory?: boolean; flow: number; unpack?: number }) {
  const y = 44;
  const dash = (flow * 120) % 24;
  return (
    <g>
      <rect x={12} y={y - 26} width={104} height={52} rx="4" fill="var(--global-wash)" stroke="var(--global)" />
      <text x={64} y={y + 4} fontSize="12" textAnchor="middle" style={{ fill: 'var(--global)' }}><Translated vi="bộ nhớ toàn cục">global memory</Translated></text>
      <rect x={W - 150} y={y - 26} width={138} height={52} rx="4" fill="var(--page)" stroke="var(--ink-2)" />
      <text x={W - 81} y={y - 6} fontSize="12" textAnchor="middle"><Translated vi="kernel">the kernel</Translated></text>
      {unpack ? (
        <g opacity={unpack} data-unpack>
          <rect x={W - 142} y={y + 2} width={56} height={18} rx="3" fill="var(--reg-wash)" stroke="var(--reg)" />
          <text x={W - 114} y={y + 15} fontSize="10" textAnchor="middle" style={{ fill: 'var(--reg)' }}><Translated vi="giải nén">unpack</Translated></text>
          <text x={W - 80} y={y + 15} fontSize="11" textAnchor="middle">→</text>
          <rect x={W - 72} y={y + 2} width={54} height={18} rx="3" fill="var(--reg-wash)" stroke="var(--reg)" />
          <text x={W - 45} y={y + 15} fontSize="10" textAnchor="middle" style={{ fill: 'var(--reg)' }}><Translated vi="nhân">multiply</Translated></text>
        </g>
      ) : null}
      <rect x={116} y={y - 9} width={W - 266} height={18} fill="var(--page)" stroke="var(--rule)" />
      <line x1={118} x2={W - 152} y1={y} y2={y} stroke="var(--global)" strokeWidth="6" strokeDasharray="12 12"
        strokeDashoffset={toMemory ? dash : -dash} opacity={flow > 0 ? 0.7 : 0.15} />
      <text x={W / 2 - 10} y={y + 26} fontSize="11.5" textAnchor="middle" data-crossing={label}>
        {label ? (toMemory ? '← ' : '') + label + (toMemory ? '' : ' →') : tr('nothing crossing', 'không có gì đi qua')}
      </text>
    </g>
  );
}

/** One pipeline as segments on the time axis, filled up to the clock. */
function Track({ id, y, clock, show = 1, finishLabel }: { id: PipelineId; y: number; clock: number; show?: number; finishLabel?: ReactNode }) {
  const ps = passes(id);
  let t = 0;
  const end = finish(id);
  return (
    <g opacity={show} data-pipeline={id} data-finish={f2(end)}>
      <text x={X0 - 8} y={y + 13} fontSize="11.5" textAnchor="end">{ROWS.find((r) => r.id === id)!.label}</text>
      <rect x={tx(0)} y={y} width={end * SCALE} height={18} fill="none" stroke="var(--rule)" />
      {ps.map((p, i) => {
        const d = p.bytes / F32_BYTES;
        const x = tx(t);
        const filled = clamp((clock - t) / Math.max(d, 1e-9));
        t += d;
        return (
          <g key={i} data-pass={p.kind} data-bytes={p.bytes}>
            <rect x={x} y={y} width={Math.max(d * SCALE * filled, filled > 0 ? 0.8 : 0)} height={18}
              fill={fillOf(p.kind)} stroke={p.kind === 'write-copy' || p.kind === 'read-copy' ? 'var(--global)' : 'none'} />
            {d * SCALE > 70 && filled > 0.5 && SHORT[p.kind] ? (
              <text x={x + (d * SCALE) / 2} y={y + 13} fontSize="10.5" textAnchor="middle"
                style={{ fill: p.kind === 'read-weights' ? 'var(--page)' : 'var(--ink)' }}>{SHORT[p.kind]}</text>
            ) : null}
          </g>
        );
      })}
      {finishLabel && clock >= end ? (
        <text x={tx(end) + 6} y={y + 13} fontSize="11.5" fontWeight="600" data-finish-label>{finishLabel}</text>
      ) : null}
    </g>
  );
}

function Stage({ crossing, toMemory, flow, unpack, tracks, clock, headline, legend = true }: {
  crossing: string; toMemory?: boolean; flow: number; unpack?: number;
  tracks: { id: PipelineId; show?: number; label?: ReactNode }[];
  clock: number; headline: ReactNode; legend?: boolean;
}) {
  const top = 112;
  const gap = 46;
  const bottom = top + gap * 3 - 10;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Byte streams through one memory pipe', 'Các luồng byte đi qua một đường ống bộ nhớ')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <Pipe label={crossing} toMemory={toMemory} flow={flow} unpack={unpack} />
      {[0, 0.5, 1, 1.5, 2].map((v) => (
        <g key={v}>
          <line x1={tx(v)} x2={tx(v)} y1={top - 8} y2={bottom} stroke="var(--rule-soft)" />
          <text x={tx(v)} y={bottom + 14} fontSize="10.5" textAnchor="middle">{v}</text>
        </g>
      ))}
      <text x={tx(T_MAX / 2)} y={bottom + 30} fontSize="11" textAnchor="middle"><Translated vi="thời gian, theo đơn vị thời gian của GEMV float32 (băng thông cố định)">time, in units of the float32 GEMV (fixed bandwidth)</Translated></text>
      <line x1={tx(1)} x2={tx(1)} y1={top - 14} y2={bottom} stroke="var(--ink)" strokeDasharray="4 3" />
      <text x={tx(1) + 4} y={top - 16} fontSize="10.5"><Translated vi="vạch đích float32">float32 finish line</Translated></text>
      {tracks.map((t, i) => (
        <Track key={t.id} id={t.id} y={top + i * gap} clock={clock} show={t.show} finishLabel={t.label} />
      ))}
      <line x1={tx(Math.min(clock, T_MAX))} x2={tx(Math.min(clock, T_MAX))} y1={top - 6} y2={bottom} stroke="var(--reg)" strokeWidth="2" data-clock={f2(clock)} />
      <text x={tx(Math.min(clock, T_MAX))} y={top - 4} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--reg)' }}>{f2(clock)}</text>
      {legend ? (
        <g>
          <rect x={14} y={H - 46} width={12} height={10} fill="var(--global)" />
          <text x={30} y={H - 37} fontSize="10.5"><Translated vi="trọng số, activation, đầu ra">weights, activations, outputs</Translated></text>
          <rect x={196} y={H - 46} width={12} height={10} fill="var(--global-wash)" stroke="var(--global)" />
          <text x={212} y={H - 37} fontSize="10.5"><Translated vi="một bản sao đã giải lượng tử hóa, được ghi rồi đọc lại">a dequantised copy, written and read back</Translated></text>
        </g>
      ) : null}
      <text x={14} y={H - 12} fontSize="12.5" fontWeight="600" data-headline>{headline}</text>
    </svg>
  );
}

/** Which pass of a pipeline is crossing the pipe at time t (in float32 units). */
function crossingAt(id: PipelineId, t: number): Pass | null {
  let s = 0;
  for (const p of passes(id)) {
    const d = p.bytes / F32_BYTES;
    if (t < s + d) return p;
    s += d;
  }
  return null;
}
const unfused = passes('unfused-f32');
const ioShare = (passes('fused').filter((p) => p.kind === 'read-x' || p.kind === 'write-y').reduce((s, p) => s + p.bytes, 0) / passBytes(passes('fused'))) * 100;

const beats: Beat[] = [
  {
    clip: 'three-traffics-1',
    get label() { return tr('Float32', 'Float32'); },
    draw: (p) => {
      const clock = ramp(p, 0.05, 0.9) * finish('f32');
      const c = crossingAt('f32', clock);
      return (
        <Stage crossing={c && clock < finish('f32') ? tr(c.label, c.kind === 'read-weights' ? 'đọc trọng số float32' : c.kind === 'read-x' ? 'đọc activation X' : 'ghi đầu ra Y') : ''} flow={c ? clock : 0} clock={clock}
          tracks={[{ id: 'f32', label: tr(`done at ${f2(finish('f32'))}`, `xong tại ${f2(finish('f32'))}`) }]}
          headline={tr(`4 bytes per weight, ${(L.in * L.out / 1e6).toFixed(1)} million weights`, `4 byte mỗi trọng số, ${(L.in * L.out / 1e6).toFixed(1)} triệu trọng số`)} />
      );
    },
  },
  {
    clip: 'three-traffics-2',
    get label() { return tr('Q4_0 is short', 'Q4_0 ngắn'); },
    draw: (p) => {
      const clock = ramp(p, 0.15, 0.55) * finish('fused');
      return (
        <Stage crossing={clock < finish('fused') && clock > 0 ? tr(`read Q4_0 weights, ${bytesPerWeight(F)} bytes each`, `đọc trọng số Q4_0, mỗi trọng số ${bytesPerWeight(F)} byte`) : ''} flow={clock} clock={clock}
          tracks={[{ id: 'f32', show: 0.35 }, { id: 'fused', show: 1, label: tr(`${f2(finish('fused'))}: about 1/${Math.round(byteRatio(F))}`, `${f2(finish('fused'))}: khoảng 1/${Math.round(byteRatio(F))}`) }]}
          headline={p < 0.6 ? tr('the same weights, 4.5 bits each', 'cùng các trọng số đó, mỗi trọng số 4.5 bit') : tr(`activations and outputs: ${ioShare.toFixed(2)} % of the bytes`, `activation và đầu ra: ${ioShare.toFixed(2)} % số byte`)} legend={false} />
      );
    },
  },
  {
    clip: 'three-traffics-3',
    get label() { return tr('Unfused', 'Không gộp'); },
    draw: (p) => {
      const clock = ramp(p, 0.04, 0.92) * finish('unfused-f32');
      const c = crossingAt('unfused-f32', clock);
      return (
        <Stage crossing={c && clock < finish('unfused-f32') ? tr(`kernel ${c.kernel}: ${c.label}`, `kernel ${c.kernel}: ${c.kind === 'read-weights' ? 'đọc trọng số Q4_0' : c.kind === 'write-copy' ? 'ghi một bản sao float32' : c.kind === 'read-copy' ? 'đọc lại bản sao float32' : c.kind === 'read-x' ? 'đọc activation X' : 'ghi đầu ra Y'}`) : ''} toMemory={c?.kind === 'write-copy' || c?.kind === 'write-y'}
          flow={c ? clock : 0} clock={clock}
          tracks={[{ id: 'f32', show: 0.35 }, { id: 'unfused-f32', label: `${f2(finish('unfused-f32'))}` }]}
          headline={clock > 1 ? tr('past the float32 finish line', 'đã quá vạch đích float32') : tr(`${unfused[0].label}, then ${unfused[1].label}, then ${unfused[2].label}`, 'đọc trọng số Q4_0, rồi ghi một bản sao float32, rồi đọc lại bản sao float32')} />
      );
    },
  },
  {
    clip: 'three-traffics-4',
    get label() { return tr('Fused', 'Gộp'); },
    draw: (p) => {
      const clock = ramp(p, 0.1, 0.6) * finish('fused');
      return (
        <Stage crossing={clock > 0 && clock < finish('fused') ? tr('read Q4_0 weights once', 'đọc trọng số Q4_0 một lần') : ''} flow={clock} clock={clock} unpack={ramp(p, 0.05, 0.3)}
          tracks={[{ id: 'f32', show: 0.35 }, { id: 'unfused-f32', show: 0.35 }, { id: 'fused', label: f2(finish('fused')) }]}
          headline={tr('unpacked between load and multiply: nothing extra crosses the pipe', 'giải nén giữa lúc nạp và lúc nhân: không có gì thêm đi qua đường ống')} />
      );
    },
  },
  {
    clip: 'three-traffics-5',
    get label() { return tr('Side by side', 'Đặt cạnh nhau'); },
    draw: (p) => {
      const clock = ramp(p, 0.02, 0.5) * T_MAX;
      return (
        <Stage crossing="" flow={0} clock={clock}
          tracks={[
            { id: 'f32', label: '1.00' },
            { id: 'unfused-f32', label: tr(`${f2(finish('unfused-f32'))} × float32`, `${f2(finish('unfused-f32'))} × float32`) },
            { id: 'fused', label: tr(`${f2(1 / finish('fused'))} × faster`, `nhanh hơn ${f2(1 / finish('fused'))} ×`) },
          ]}
          headline={p > 0.6 ? tr('bytes model only: your device decides how much of the lead survives', 'chỉ là mô hình theo byte: thiết bị của bạn quyết định giữ lại được bao nhiêu phần lợi thế') : tr('three finishing times, from the bytes alone', 'ba thời điểm về đích, chỉ tính từ số byte')} />
      );
    },
  },
];

export function ThreeTrafficsScene() {
  return <Scene chapter="x2" title={tr('Three ways to feed one GEMV', 'Ba cách nạp dữ liệu cho một GEMV')} beats={beats} />;
}
