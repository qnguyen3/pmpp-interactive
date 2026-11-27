import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import {
  SCENE,
  blockCount,
  blockWork,
  distinctResults,
  globalIndex,
  launch,
  launchWith,
  makeData,
  runKernel,
  schedule,
  showValue,
} from '../../sim/launch';

// "One launch, start to finish": the launch statement of section 2.6 as a sequence
// of events. 22 elements, blocks of 8, a device that runs 2 blocks at once. Every
// number drawn comes from src/sim/launch.ts, the model behind the section's three
// interactives. Seven beats, one narration clip each
// (content/ch02/narration/one-launch-N.txt).
const { n, B, S, seed } = SCENE;
const count = blockCount('int-ceil', n, B);
const good = launch(n, B, count.blocks);
const sched = schedule(count.blocks, S, 'shuffled', seed);
const data = makeData(n, B, NaN); // NaN: C holds whatever the allocation held
const run = runKernel('independent', sched, data);
const sums = run.expected;
const every = distinctResults('independent', count.blocks, makeData(n, B));
const badCount = blockCount('trunc', n, B);
const bad = launchWith('trunc', n, B);
const FIRST = sched.waves[0];
const SECOND = sched.waves[1];

// Geometry: thread box i and array cell i share an x, so the grid sits over the data.
const VW = 760;
const CW = 27;
const GAP = 14;
const X0 = 46;
const xOf = (i: number) => X0 + i * CW + Math.floor(i / B) * GAP;
const GY = 162; // thread boxes
const TH = 36;
const AY = 252; // first array row
const RH = 26;
const MONO = { fontVariationSettings: "'MONO' 1, 'CASL' 0" } as const; // i18n-ignore: CSS font settings

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

type BlockState = 'hidden' | 'queued' | 'running' | 'done' | 'ghost';

interface Frame {
  /** The launch statement on the host's first line. */
  statement: string;
  /** Which of the three host lines the host thread is on. */
  hostLine: 0 | 1 | 2;
  hostNote?: string;
  /** Lines of the strip between the host and the grid. */
  strip?: ReactNode[];
  /** Runtime panel: one status line, the queue and the two device slots. */
  runtime?: string;
  queue?: number[];
  slots?: (number | null)[];
  blocks: BlockState[];
  gridOpacity?: number;
  /** Per block: how many of its threads have computed i. */
  indexed?: number[];
  /** Threads the guard has stopped, by global index. */
  stopped?: Set<number>;
  /** Elements being loaded from A and B right now. */
  hot?: Set<number>;
  /** Elements of C that have been stored. */
  stored?: Set<number>;
  /** Elements of C to mark as never written. */
  missing?: Set<number>;
  headline?: ReactNode;
}

const HOST_LINES = ['', 'CHECK(cudaGetLastError());', 'CHECK(cudaDeviceSynchronize());'];
const STATE_WORD: Record<BlockState, string> = {
  hidden: '',
  get queued() {
    return tr('in the queue', 'trong hàng đợi');
  },
  get running() {
    return tr('running', 'đang chạy');
  },
  get done() {
    return tr('finished', 'đã xong');
  },
  get ghost() {
    return tr('never created', 'không bao giờ được tạo');
  },
};

function Stage(f: Frame) {
  const lines = [f.statement, HOST_LINES[1], HOST_LINES[2]];
  const shownBlocks = f.blocks.map((s, b) => ({ s, b })).filter((x) => x.s !== 'hidden');
  // On a phone the drawing keeps a legible size and scrolls sideways inside its frame.
  return (
    <div style={{ overflowX: 'auto' }}>
    <svg viewBox={`0 0 ${VW} 392`} role="img" aria-label={tr('Animation of one kernel launch', 'Hoạt hình một lần khởi chạy kernel')} style={{ width: '100%', minWidth: 600, maxWidth: VW, display: 'block' }}>
      {/* host thread */}
      <rect x={10} y={8} width={476} height={86} rx="3" fill="var(--cell)" stroke="var(--rule)" />
      <text x={20} y={24} fontSize="11" fontWeight="600" style={{ fill: 'var(--ink-2)' }}>{tr('host thread', 'luồng host')}</text>
      {lines.map((ln, k) => (
        <g key={k}>
          {k === f.hostLine ? <rect x={14} y={31 + k * 19} width={468} height={18} rx="2" fill="var(--rule-soft)" /> : null}
          {k === f.hostLine ? <text x={20} y={44 + k * 19} fontSize="11">▶</text> : null}
          <text x={36} y={44 + k * 19} fontSize="11.5" style={{ ...MONO, fill: k === f.hostLine ? 'var(--ink)' : 'var(--ink-3)' }}>{ln}</text>
        </g>
      ))}
      {f.hostNote ? (
        <text x={300} y={24} fontSize="11" textAnchor="start" style={{ fill: 'var(--ink-2)' }}>{f.hostNote}</text>
      ) : null}

      {/* runtime and device */}
      <rect x={498} y={8} width={252} height={86} rx="3" fill="var(--cell)" stroke="var(--rule)" />
      <text x={508} y={24} fontSize="11" fontWeight="600" style={{ fill: 'var(--ink-2)' }}>{tr('runtime and device', 'runtime và device')}</text>
      <text x={508} y={42} fontSize="11.5">{f.runtime ?? tr('nothing queued', 'chưa có gì trong hàng đợi')}</text>
      <text x={508} y={63} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('queue', 'hàng đợi')}</text>
      {(f.queue ?? []).map((b, k) => (
        <g key={b}>
          <rect x={548 + k * 30} y={51} width={26} height={17} rx="2" fill="var(--page)" stroke="var(--ink-2)" />
          <text x={561 + k * 30} y={63.5} fontSize="10.5" textAnchor="middle">b{b}</text>
        </g>
      ))}
      <text x={508} y={85} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>chạy {S} cùng lúc</>}>runs {S} at once</Translated>
      </text>
      {Array.from({ length: S }, (_, k) => {
        const b = f.slots?.[k] ?? null;
        return (
          <g key={k}>
            <rect x={596 + k * 34} y={73} width={30} height={17} rx="2" fill={b == null ? 'var(--page)' : 'var(--ink)'} stroke="var(--ink-2)" strokeDasharray={b == null ? '3 2' : undefined} />
            {b != null ? <text x={611 + k * 34} y={85.5} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>b{b}</text> : null}
          </g>
        );
      })}

      {/* strip: what is being worked out */}
      {(f.strip ?? []).map((s, k) => (
        <text key={k} x={X0} y={114 + k * 17} fontSize="12" style={MONO}>{s}</text>
      ))}

      {/* the grid */}
      <g opacity={f.gridOpacity ?? 1}>
        {shownBlocks.map(({ s, b }) => {
          const x = xOf(b * B) - 4;
          const w = B * CW + 6;
          const ghost = s === 'ghost';
          return (
            <g key={b}>
              <rect x={x} y={GY - 20} width={w} height={TH + 26} rx="3" fill="none"
                stroke={ghost ? 'var(--fail)' : s === 'running' ? 'var(--ink)' : 'var(--rule)'}
                strokeWidth={s === 'running' ? 2 : 1} strokeDasharray={ghost ? '5 4' : undefined} />
              <text x={x + 5} y={GY - 7} fontSize="11" fontWeight="600" style={{ fill: ghost ? 'var(--fail)' : 'var(--ink)' }}>
                {tr(`block ${b}`, `khối ${b}`)}
                <tspan fontWeight="400" style={{ fill: ghost ? 'var(--fail)' : 'var(--ink-2)' }}> · {STATE_WORD[s]}</tspan>
              </text>
              {ghost
                ? null
                : Array.from({ length: B }, (_, t) => {
                    const i = globalIndex(b, B, t);
                    const hasI = (f.indexed?.[b] ?? 0) > t;
                    const stop = f.stopped?.has(i);
                    return (
                      <g key={t}>
                        <rect x={xOf(i)} y={GY} width={CW - 3} height={TH} rx="2"
                          fill={stop ? 'var(--cell)' : hasI ? 'var(--reg-wash)' : 'var(--cell)'}
                          stroke={stop ? 'var(--ink-3)' : hasI ? 'var(--reg)' : 'var(--rule)'} strokeDasharray={stop ? '3 2' : undefined} />
                        <text x={xOf(i) + (CW - 3) / 2} y={GY + 13} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{b},{t}</text>
                        {hasI ? (
                          <text x={xOf(i) + (CW - 3) / 2} y={GY + 29} fontSize="11.5" textAnchor="middle" fontWeight="600"
                            style={{ fill: stop ? 'var(--ink-3)' : 'var(--reg)', textDecoration: stop ? 'line-through' : undefined }}>{i}</text>
                        ) : null}
                      </g>
                    );
                  })}
            </g>
          );
        })}
      </g>

      {/* device global memory */}
      {(['A', 'B', 'C'] as const).map((name, r) => (
        <g key={name}>
          <text x={14} y={AY + r * RH + 16} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>{name}_d</text>
          {Array.from({ length: n }, (_, i) => {
            const hot = name !== 'C' && f.hot?.has(i);
            const stored = name === 'C' && f.stored?.has(i);
            const miss = name === 'C' && f.missing?.has(i);
            const v = name === 'A' ? data.A[i] : name === 'B' ? data.Bv[i] : stored ? sums[i] : null;
            return (
              <g key={i}>
                <rect x={xOf(i)} y={AY + r * RH} width={CW - 3} height={RH - 4} rx="2"
                  fill={miss ? 'var(--fail-wash)' : hot ? 'var(--global)' : stored ? 'var(--global-wash)' : 'var(--cell)'}
                  stroke={miss ? 'var(--fail)' : hot || stored ? 'var(--global)' : 'var(--rule)'} />
                <text x={xOf(i) + (CW - 3) / 2} y={AY + r * RH + 15.5} fontSize="11" textAnchor="middle"
                  style={{ fill: miss ? 'var(--fail)' : hot ? 'var(--page)' : v == null ? 'var(--ink-3)' : 'var(--ink)' }}>{v == null ? '?' : v}</text>
              </g>
            );
          })}
        </g>
      ))}
      <text x={X0} y={AY + 3 * RH + 12} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>bộ nhớ toàn cục của device, phần tử 0 đến {n - 1}. Dấu hỏi là ô nhớ chưa có gì ghi vào.</>}>
          device global memory, elements 0 to {n - 1}. A question mark is memory nothing has written yet.
        </Translated>
      </text>
      {f.headline ? <text x={X0} y={382} fontSize="13" fontWeight="600">{f.headline}</text> : null}
    </svg>
    </div>
  );
}

const GOOD_STATEMENT = `vecAddKernel<<<(n + ${B} - 1) / ${B}, ${B}>>>(A_d, B_d, C_d, n);`;
const BAD_STATEMENT = `vecAddKernel<<<n / ${B}, ${B}>>>(A_d, B_d, C_d, n);`;
const QUEUED: BlockState[] = Array.from({ length: count.blocks }, () => 'queued');
const range = (a: number, b: number) => Array.from({ length: Math.max(0, b - a) }, (_, k) => a + k);
/** Elements the first `t` threads of the given blocks have reached, guard applied. */
const reached = (blocks: number[], t: number) =>
  new Set(blocks.flatMap((b) => range(0, Math.min(t, blockWork(n, B, b).working)).map((k) => globalIndex(b, B, k))));
const STOPPED = new Set(range(n, good.threads));
const [s0, s1] = count.steps;
const KERNEL_LINE = 'int i = blockIdx.x * blockDim.x + threadIdx.x;  if (i < n) C[i] = A[i] + B[i];';

const beats: Beat[] = [
  {
    clip: 'one-launch-1',
    get label() {
      return tr('The statement', 'Câu lệnh');
    },
    draw: (p) => (
      <Stage
        statement={GOOD_STATEMENT}
        hostLine={0}
        blocks={QUEUED.map(() => 'hidden')}
        strip={[
          p > 0.08
            ? tr(`1. the configuration:  ${s0.text} = ${s0.value}`, `1. cấu hình:  ${s0.text} = ${s0.value}`) +
              (p > 0.26
                ? tr(
                    `,  ${s1.text} = ${s1.value}  (whole numbers: ${showValue(s0.value / B, 'double')} becomes ${s1.value})`,
                    `,  ${s1.text} = ${s1.value}  (số nguyên: ${showValue(s0.value / B, 'double')} thành ${s1.value})`,
                  )
                : '')
            : '',
          p > 0.62
            ? tr(
                `2. the arguments:  A_d, B_d, C_d are three device addresses;  n = ${n}, passed by value`,
                `2. các đối số:  A_d, B_d, C_d là ba địa chỉ trên device;  n = ${n}, truyền theo giá trị`,
              )
            : '',
        ]}
        headline={p > 0.42 ? tr(`${count.blocks} blocks of ${B} threads`, `${count.blocks} khối, mỗi khối ${B} luồng`) : undefined}
      />
    ),
  },
  {
    clip: 'one-launch-2',
    get label() {
      return tr('It returns', 'Nó trả về');
    },
    draw: (p) => (
      <Stage
        statement={GOOD_STATEMENT}
        hostLine={p > 0.45 ? 1 : 0}
        hostNote={p > 0.45 ? tr('the statement has returned', 'câu lệnh đã trả về') : undefined}
        runtime={p > 0.12 ? tr('configuration accepted', 'cấu hình được chấp nhận') : tr('checking the configuration', 'đang kiểm tra cấu hình')}
        queue={p > 0.28 ? range(0, count.blocks) : []}
        blocks={QUEUED.map(() => 'hidden')}
        strip={[
          p > 0.55 ? tr('No thread has run yet.', 'Chưa luồng nào chạy.') : '',
          p > 0.72
            ? tr(
                'A configuration the device cannot satisfy is reported here, on the next line.',
                'Một cấu hình mà device không đáp ứng được sẽ được báo ở đây, trên dòng tiếp theo.',
              )
            : '',
        ]}
      />
    ),
  },
  {
    clip: 'one-launch-3',
    get label() {
      return tr('The grid', 'Lưới');
    },
    draw: (p) => (
      <Stage
        statement={GOOD_STATEMENT}
        hostLine={1}
        runtime={tr('configuration accepted', 'cấu hình được chấp nhận')}
        queue={range(0, count.blocks)}
        blocks={QUEUED}
        gridOpacity={ramp(p, 0.02, 0.22)}
        strip={[p > 0.35 ? KERNEL_LINE : '', p > 0.6 ? tr('Each box: block index, thread index.', 'Mỗi ô: chỉ số khối, chỉ số luồng.') : '']}
        headline={tr(
          `${good.blocks} blocks × ${B} threads = ${good.threads} threads, one kernel`,
          `${good.blocks} khối × ${B} luồng = ${good.threads} luồng, một kernel`,
        )}
      />
    ),
  },
  {
    clip: 'one-launch-4',
    get label() {
      return tr('Two at a time', 'Mỗi lần hai khối');
    },
    draw: (p) => {
      const taken = p > 0.14;
      const done = Math.floor(ramp(p, 0.5, 0.92) * B + 0.001);
      return (
        <Stage
          statement={GOOD_STATEMENT}
          hostLine={1}
          runtime={
            taken
              ? tr(`took block ${FIRST[0]} and block ${FIRST[1]}`, `đã lấy khối ${FIRST[0]} và khối ${FIRST[1]}`)
              : tr('choosing blocks', 'đang chọn khối')
          }
          queue={taken ? SECOND : range(0, count.blocks)}
          slots={taken ? FIRST : []}
          blocks={QUEUED.map((_, b) => (taken && FIRST.includes(b) ? 'running' : 'queued'))}
          indexed={QUEUED.map((_, b) => (FIRST.includes(b) ? done : 0))}
          strip={[
            p > 0.45
              ? tr(
                  `i = blockIdx.x * ${B} + threadIdx.x, in a register of its own for every thread`,
                  `i = blockIdx.x * ${B} + threadIdx.x, trong một thanh ghi riêng của mỗi luồng`,
                )
              : '',
          ]}
        />
      );
    },
  },
  {
    clip: 'one-launch-5',
    get label() {
      return tr('The guard', 'Phép chặn');
    },
    draw: (p) => {
      const second = p > 0.66;
      const t1 = Math.floor(ramp(p, 0.26, 0.58) * B + 0.001); // threads of the first wave that have stored
      const t2 = Math.floor(ramp(p, 0.78, 0.96) * B + 0.001);
      const stored = new Set([...reached(FIRST, t1), ...(second ? reached(SECOND, t2) : [])]);
      const hot = second
        ? new Set(t2 < B ? SECOND.map((b) => globalIndex(b, B, t2)) : [])
        : new Set(p > 0.22 && t1 < B ? FIRST.map((b) => globalIndex(b, B, t1)).filter((i) => i < n) : []);
      return (
        <Stage
          statement={GOOD_STATEMENT}
          hostLine={1}
          runtime={
            second
              ? tr(`took block ${SECOND[0]}`, `đã lấy khối ${SECOND[0]}`)
              : tr(`running block ${FIRST[0]} and block ${FIRST[1]}`, `đang chạy khối ${FIRST[0]} và khối ${FIRST[1]}`)
          }
          queue={second ? [] : SECOND}
          slots={second ? SECOND : FIRST}
          blocks={QUEUED.map((_, b) => (FIRST.includes(b) ? (second ? 'done' : 'running') : second ? 'running' : 'queued'))}
          indexed={QUEUED.map((_, b) => (FIRST.includes(b) || second ? B : 0))}
          stopped={p > 0.1 ? STOPPED : undefined}
          hot={hot}
          stored={stored}
          strip={[
            p > 0.1
              ? tr(
                  `if (i < n): threads with i = ${[...STOPPED].join(' and ')} fail the test and stop`,
                  `if (i < n): các luồng có i = ${[...STOPPED].join(' và ')} không qua phép kiểm tra và dừng`,
                )
              : '',
            p > 0.26 ? tr('C[i] = A[i] + B[i] for every other thread', 'C[i] = A[i] + B[i] với mọi luồng còn lại') : '',
          ]}
        />
      );
    },
  },
  {
    clip: 'one-launch-6',
    get label() {
      return tr('Only now', 'Chỉ đến lúc này');
    },
    draw: (p) => (
      <Stage
        statement={GOOD_STATEMENT}
        hostLine={p > 0.3 ? 2 : 1}
        hostNote={p > 0.3 ? tr('the blocking call returns now', 'lời gọi chặn giờ mới trả về') : undefined}
        runtime={tr('the grid has finished', 'lưới đã chạy xong')}
        blocks={QUEUED.map(() => 'done')}
        indexed={QUEUED.map(() => B)}
        stopped={STOPPED}
        stored={new Set(range(0, n))}
        strip={[
          p > 0.55
            ? tr(
                `Every order of ${count.blocks} blocks, 1 to ${count.blocks} at a time: ${every.runs} runs of the model.`,
                `Mọi thứ tự của ${count.blocks} khối, mỗi lần 1 đến ${count.blocks} khối: ${every.runs} lần chạy mô hình.`,
              )
            : '',
        ]}
        headline={p > 0.62 ? tr(`different results: ${every.distinct}`, `số kết quả khác nhau: ${every.distinct}`) : tr('C is complete', 'C đã hoàn chỉnh')}
      />
    ),
  },
  {
    clip: 'one-launch-7',
    get label() {
      return tr('The bug', 'Lỗi');
    },
    draw: (p) => {
      const t = Math.floor(ramp(p, 0.42, 0.62) * B + 0.001);
      const missing = new Set(bad.unprocessed.flatMap((r) => range(r.first, r.last + 1)));
      const ran = range(0, bad.blocks);
      return (
        <Stage
          statement={BAD_STATEMENT}
          hostLine={p > 0.36 ? 2 : 0}
          runtime={p > 0.3 ? tr('configuration accepted', 'cấu hình được chấp nhận') : tr('nothing queued', 'chưa có gì trong hàng đợi')}
          blocks={QUEUED.map((_, b) => (b < bad.blocks ? (p > 0.62 ? 'done' : p > 0.36 ? 'running' : p > 0.24 ? 'queued' : 'hidden') : p > 0.24 ? 'ghost' : 'hidden'))}
          indexed={QUEUED.map((_, b) => (b < bad.blocks && p > 0.4 ? B : 0))}
          stored={reached(ran, p > 0.36 ? t : 0)}
          missing={p > 0.66 ? missing : undefined}
          strip={[
            p > 0.08
              ? tr(
                  `${badCount.steps[0].text} = ${badCount.steps[0].value}  (whole numbers: ${showValue(n / B, 'double')} becomes ${badCount.blocks})`,
                  `${badCount.steps[0].text} = ${badCount.steps[0].value}  (số nguyên: ${showValue(n / B, 'double')} thành ${badCount.blocks})`,
                )
              : '',
            p > 0.66
              ? tr(
                  `elements ${bad.unprocessed[0].first} to ${bad.unprocessed[0].last}: no thread exists for them`,
                  `phần tử ${bad.unprocessed[0].first} đến ${bad.unprocessed[0].last}: không có luồng nào cho chúng`,
                )
              : '',
          ]}
          headline={
            p > 0.24
              ? tr(
                  `${bad.blocks} blocks, ${bad.threads} threads, ${bad.unprocessedCount} elements never written`,
                  `${bad.blocks} khối, ${bad.threads} luồng, ${bad.unprocessedCount} phần tử không bao giờ được ghi`,
                )
              : undefined
          }
        />
      );
    },
  },
];

export function OneLaunchScene() {
  return <Scene chapter="ch02" title={tr('One launch, start to finish', 'Một lần khởi chạy, từ đầu đến cuối')} beats={beats} />;
}
