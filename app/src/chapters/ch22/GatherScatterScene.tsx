import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { SCENE, sceneModel } from '../../sim/decompose';
import { Translated, tr } from '../../lib/i18n';

// "Gather and scatter, in time": a valid 1-D convolution of 8 inputs into 6 outputs, each
// the sum of 3 neighbours, cut both ways. Every number drawn (arrows, writer counts, the
// queue at each output, read counts, the histogram preview) comes from src/sim/decompose.ts.
// Six beats, one narration clip each (content/ch22/narration/gather-scatter-N.txt).
const M = sceneModel();
const IN = SCENE.inputs;
const NI = M.p.nIn; // 8
const NO = M.p.nOut; // 6
const TICKS = M.scatterTicks.ticks.length; // 3

const W = 720;
const PITCH = 72;
const CW = 48; // cell width
const CH = 34; // cell height
const IN_Y = 44;
const TH_Y = 160;
const OUT_Y = 262;
const inX = (i: number) => (W - NI * PITCH) / 2 + i * PITCH + (PITCH - CW) / 2; // left edge of input i
const inC = (i: number) => inX(i) + CW / 2;
const outC = (o: number) => inC(o + 1); // out[o] sits under its middle input
const outX = (o: number) => outC(o) - CW / 2;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

function Cell({ x, y, v, fill = 'var(--cell)', stroke = 'var(--rule)', ink = 'var(--ink)', dash, data }: {
  x: number; y: number; v: ReactNode; fill?: string; stroke?: string; ink?: string; dash?: boolean; data?: Record<string, string>;
}) {
  return (
    <g {...data}>
      <rect x={x} y={y} width={CW} height={CH} rx="3" fill={fill} stroke={stroke} strokeWidth="1.4" strokeDasharray={dash ? '3 3' : undefined} />
      <text x={x + CW / 2} y={y + CH / 2 + 5} fontSize="14" fontWeight="600" textAnchor="middle" style={{ fill: ink }}>{v}</text>
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, t = 1, color = 'var(--ink-3)', width = 1.4 }: {
  x1: number; y1: number; x2: number; y2: number; t?: number; color?: string; width?: number;
}) {
  if (t <= 0) return null;
  const x = x1 + (x2 - x1) * t;
  const y = y1 + (y2 - y1) * t;
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const h = 6;
  return (
    <g>
      <line x1={x1} y1={y1} x2={x} y2={y} stroke={color} strokeWidth={width} />
      {t >= 1 ? (
        <path d={`M ${x2} ${y2} L ${x2 - h * Math.cos(ang - 0.4)} ${y2 - h * Math.sin(ang - 0.4)} L ${x2 - h * Math.cos(ang + 0.4)} ${y2 - h * Math.sin(ang + 0.4)} Z`} fill={color} />
      ) : null}
    </g>
  );
}

function Thread({ x, label, hot }: { x: number; label: string; hot?: boolean }) {
  return (
    <g>
      <circle cx={x} cy={TH_Y} r="15" fill="var(--page)" stroke={hot ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={hot ? 2 : 1.3} />
      <text x={x} y={TH_Y + 4} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{label}</text>
    </g>
  );
}

function Labels({ outs = true, threads = true }: { outs?: boolean; threads?: boolean }) {
  return (
    <g>
      <text x={14} y={IN_Y + CH / 2 + 4} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="vào">in</Translated></text>
      {threads ? <text x={14} y={TH_Y + 4} fontSize="12" style={{ fill: 'var(--ink-2)' }}><Translated vi="luồng">threads</Translated></text> : null}
      {outs ? <text x={14} y={OUT_Y + CH / 2 + 4} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="ra">out</Translated></text> : null}
    </g>
  );
}

function Inputs({ wash, counts, countColor = 'var(--ink-2)' }: { wash?: (i: number) => string | undefined; counts?: (i: number) => string | null; countColor?: string }) {
  return (
    <g data-row="in" data-values={IN.join(' ')}>
      {IN.map((v, i) => (
        <g key={i}>
          <Cell x={inX(i)} y={IN_Y} v={v} fill={wash?.(i) ?? 'var(--global-wash)'} stroke="var(--global)" />
          {counts?.(i) ? (
            <text x={inC(i)} y={IN_Y - 8} fontSize="11.5" fontWeight="600" textAnchor="middle" style={{ fill: countColor }}>{counts(i)}</text>
          ) : null}
        </g>
      ))}
    </g>
  );
}

function Headline({ children }: { children: ReactNode }) {
  return <text data-headline x={W / 2} y={344} fontSize="14" fontWeight="600" textAnchor="middle">{children}</text>;
}

function Stage({ beat, children }: { beat: number; children: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} 360`} role="img" aria-label={tr('Gather and scatter on a small convolution', 'Gather và scatter trên một phép tích chập nhỏ')} style={{ width: '100%', maxWidth: W, display: 'block', margin: '0 auto' }}>
      <g data-beat={beat}>{children}</g>
    </svg>
  );
}

// ---------------------------------------------------------------- beat 1: the contribution graph
function Graph({ p }: { p: number }) {
  const shown = Math.round(ramp(p, 0.08, 0.7) * M.p.edges.length);
  return (
    <Stage beat={1}>
      <Labels threads={false} />
      <Inputs counts={(i) => (p > 0.6 ? tr(`feeds ${M.fanOut[i]}`, `cấp cho ${M.fanOut[i]}`) : null)} />
      {M.p.edges.slice(0, shown).map((e, k) => (
        <Arrow key={k} x1={inC(e.i)} y1={IN_Y + CH} x2={outC(e.o)} y2={OUT_Y - 2} color="var(--ink-3)" />
      ))}
      <g data-row="out">
        {Array.from({ length: NO }, (_, o) => <Cell key={o} x={outX(o)} y={OUT_Y} v="" />)}
      </g>
      <g data-edges={shown} data-fanout={M.fanOut.join(' ')} />
      <Headline>{p > 0.6 ? tr(`${NI} inputs, ${NO} outputs: every input feeds up to ${Math.max(...M.fanOut)} outputs`, `${NI} đầu vào, ${NO} đầu ra: mỗi đầu vào cấp cho tối đa ${Math.max(...M.fanOut)} đầu ra`) : tr(`each output is the sum of ${SCENE.w} neighbouring inputs`, `mỗi đầu ra là tổng của ${SCENE.w} đầu vào lân cận`)}</Headline>
    </Stage>
  );
}

// ---------------------------------------------------------------- beat 2: gather
function Gather({ p }: { p: number }) {
  const read = ramp(p, 0.05, 0.5); // arrows in → thread grow
  const adds = Math.min(SCENE.w, Math.floor(ramp(p, 0.15, 0.55) * (SCENE.w + 0.999)));
  const write = ramp(p, 0.62, 0.8);
  const written = write >= 1;
  return (
    <Stage beat={2}>
      <Labels />
      <Inputs />
      {M.gather.workers.map((w) => (
        <g key={w.id}>
          {w.useful.map((i) => <Arrow key={i} x1={inC(i)} y1={IN_Y + CH} x2={outC(w.id)} y2={TH_Y - 16} t={read} color="var(--ink-3)" />)}
          <Thread x={outC(w.id)} label={`t${w.id}`} hot={p < 0.6} />
          <g>
            <rect x={outC(w.id) + 18} y={TH_Y - 11} width={30} height={22} rx="2" fill="var(--reg-wash)" stroke="var(--reg)" />
            <text x={outC(w.id) + 33} y={TH_Y + 4} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--reg)' }}>
              {w.useful.slice(0, adds).reduce((s, i) => s + IN[i], 0)}
            </text>
          </g>
          <Arrow x1={outC(w.id)} y1={TH_Y + 16} x2={outC(w.id)} y2={OUT_Y - 2} t={write} color="var(--ink)" width={1.8} />
        </g>
      ))}
      <g data-row="out" data-writers={M.gather.writers.join(' ')} data-written={written ? NO : 0}>
        {M.sums.map((s, o) => (
          <Cell key={o} x={outX(o)} y={OUT_Y} v={written ? s : ''} stroke={written ? 'var(--ok)' : 'var(--rule)'} fill={written ? 'var(--ok-wash)' : 'var(--cell)'} />
        ))}
      </g>
      <text x={W - 14} y={TH_Y + 40} fontSize="11" textAnchor="end" style={{ fill: 'var(--reg)' }}><Translated vi="tổng tích lũy: một thanh ghi">running sum: a register</Translated></text>
      <Headline>{written ? tr(`tick 1: ${NO} writes, ${NO} different cells, no atomics`, `nhịp 1: ${NO} lần ghi, ${NO} ô khác nhau, không cần atomic`) : tr(`${M.gather.metrics.threads} threads, one per output, each adds its ${SCENE.w} inputs in a register`, `${M.gather.metrics.threads} luồng, mỗi luồng một đầu ra, mỗi luồng cộng ${SCENE.w} đầu vào của nó trong một thanh ghi`)}</Headline>
    </Stage>
  );
}

// ---------------------------------------------------------------- beats 3 and 4: scatter
function ScatterArrows({ t, color = 'var(--ink-3)' }: { t: number; color?: string }) {
  return (
    <g>
      {M.scatter.workers.map((w) =>
        w.writes.map((x) => <Arrow key={`${w.id}-${x.o}`} x1={inC(w.id)} y1={TH_Y + 16} x2={outC(x.o)} y2={OUT_Y - 2} t={t} color={color} />),
      )}
    </g>
  );
}

function Scatter({ p }: { p: number }) {
  const t = ramp(p, 0.1, 0.55);
  const flash = p > 0.55;
  return (
    <Stage beat={3}>
      <Labels />
      <Inputs />
      {M.scatter.workers.map((w) => (
        <g key={w.id}>
          <Arrow x1={inC(w.id)} y1={IN_Y + CH} x2={inC(w.id)} y2={TH_Y - 16} t={ramp(p, 0, 0.1)} color="var(--ink-3)" />
          <Thread x={inC(w.id)} label={`t${w.id}`} />
        </g>
      ))}
      <ScatterArrows t={t} color={flash ? 'var(--fail)' : 'var(--ink-3)'} />
      <g data-row="out" data-writers={M.scatter.writers.join(' ')} data-atomic={flash ? 'yes' : 'no'}>
        {M.scatter.writers.map((n, o) => (
          <g key={o}>
            <Cell x={outX(o)} y={OUT_Y} v="" stroke={flash && n > 1 ? 'var(--fail)' : 'var(--rule)'} fill={flash && n > 1 ? 'var(--fail-wash)' : 'var(--cell)'} />
            {flash ? (
              <text x={outC(o)} y={OUT_Y + CH + 16} fontSize="11.5" fontWeight="600" textAnchor="middle" style={{ fill: 'var(--fail)' }}><Translated vi={<>{n} luồng ghi</>}>{n} writers</Translated></text>
            ) : null}
          </g>
        ))}
      </g>
      <Headline>{flash ? tr(`${M.scatter.metrics.outputsNeedingAtomics} outputs, ${M.scatter.metrics.worstWriters} writers each: ${M.scatter.metrics.atomicUpdates} atomic updates`, `${M.scatter.metrics.outputsNeedingAtomics} đầu ra, mỗi đầu ra ${M.scatter.metrics.worstWriters} luồng ghi: ${M.scatter.metrics.atomicUpdates} lần cập nhật atomic`) : tr(`${M.scatter.metrics.threads} threads, one per input, each pushes into up to ${M.scatter.metrics.worstWriters} outputs`, `${M.scatter.metrics.threads} luồng, mỗi luồng một đầu vào, mỗi luồng đẩy vào tối đa ${M.scatter.metrics.worstWriters} đầu ra`)}</Headline>
    </Stage>
  );
}

function Queue({ p }: { p: number }) {
  const tick = Math.min(TICKS, Math.floor(ramp(p, 0.12, 0.88) * (TICKS + 1)));
  const waiting = tick === 0 ? M.scatterTicks.issued : M.scatterTicks.ticks[tick - 1].waiting;
  // What each output holds after `tick` ticks: the contributions served so far, in worker order.
  const held = Array.from({ length: NO }, (_, o) =>
    M.scatterTicks.ticks.slice(0, tick).reduce((s, tk) => s + tk.served.filter((x) => x.o === o).reduce((a, x) => a + IN[x.worker], 0), 0),
  );
  const box = 13;
  return (
    <Stage beat={4}>
      <Labels />
      <Inputs />
      {M.scatter.workers.map((w) => <Thread key={w.id} x={inC(w.id)} label={`t${w.id}`} />)}
      <g opacity="0.35"><ScatterArrows t={1} color="var(--ink-3)" /></g>
      <g data-row="out" data-tick={tick} data-waiting={waiting.join(' ')}>
        {held.map((v, o) => (
          <g key={o}>
            {/* the queue: updates still waiting at this address */}
            {Array.from({ length: waiting[o] }, (_, k) => (
              <rect key={k} x={outC(o) - box / 2} y={OUT_Y - 8 - (k + 1) * (box + 3)} width={box} height={box} rx="2" fill="var(--fail-wash)" stroke="var(--fail)" />
            ))}
            <Cell x={outX(o)} y={OUT_Y} v={tick ? v : ''} stroke={tick === TICKS ? 'var(--ok)' : 'var(--fail)'} fill={tick === TICKS ? 'var(--ok-wash)' : 'var(--fail-wash)'} />
          </g>
        ))}
      </g>
      <g>
        <rect x={W - 210} y={OUT_Y + CH + 8} width={196} height={46} rx="3" fill="var(--page)" stroke="var(--rule)" />
        <text x={W - 200} y={OUT_Y + CH + 27} fontSize="12" fontWeight="600" data-scatter-tick={tick}><Translated vi={<>scatter: nhịp {tick} trên {TICKS}</>}>scatter: tick {tick} of {TICKS}</Translated></text>
        <text x={W - 200} y={OUT_Y + CH + 45} fontSize="12" style={{ fill: 'var(--ink-2)' }} data-gather-done={M.gatherTicks.ticks[0].served.length}>
          <Translated vi={<>gather: {M.gatherTicks.ticks[0].served.length} trên {NO} đã ghi ở nhịp 1</>}>
            gather: {M.gatherTicks.ticks[0].served.length} of {NO} written at tick 1
          </Translated>
        </text>
      </g>
      <Headline>{tick < TICKS ? tr('one address serves one update per tick', 'một địa chỉ phục vụ một lần cập nhật mỗi nhịp') : tr(`${TICKS} ticks to drain ${M.scatter.metrics.atomicUpdates} updates; gather needed 1`, `${TICKS} nhịp để xử lý hết ${M.scatter.metrics.atomicUpdates} lần cập nhật; gather chỉ cần 1`)}</Headline>
    </Stage>
  );
}

// ---------------------------------------------------------------- beat 5: gather is not free
function NotFree({ p }: { p: number }) {
  const hist = p >= 0.55;
  if (!hist) {
    const show = ramp(p, 0.05, 0.3);
    const total = M.reads.reduce((a, b) => a + b, 0);
    return (
      <Stage beat={5}>
        <Labels outs={false} />
        <Inputs
          wash={(i) => (show > 0 && M.reads[i] > 1 ? 'var(--shared-wash)' : undefined)}
          counts={(i) => (show > 0 ? tr(`read ${M.reads[i]}×`, `đọc ${M.reads[i]}×`) : null)}
          countColor="var(--shared)"
        />
        {M.gather.workers.map((w) => (
          <g key={w.id}>
            {w.useful.map((i) => <Arrow key={i} x1={inC(i)} y1={IN_Y + CH} x2={outC(w.id)} y2={TH_Y - 16} t={1} color={M.reads[i] > 1 ? 'var(--shared)' : 'var(--ink-3)'} />)}
            <Thread x={outC(w.id)} label={`t${w.id}`} />
          </g>
        ))}
        <g data-reads={M.reads.join(' ')} data-reads-total={total}>
          {p > 0.3 ? (
            <g>
              <rect x={inX(0) - 6} y={IN_Y - 26} width={inX(NI - 1) + CW + 12 - (inX(0) - 6)} height={CH + 32} rx="5" fill="none" stroke="var(--shared)" strokeWidth="2" strokeDasharray="5 4" />
              <text x={W / 2} y={OUT_Y + 8} fontSize="12.5" textAnchor="middle" style={{ fill: 'var(--shared)' }}>
                <Translated vi={<>{total} lần đọc {NI} đầu vào: một khối có thể nạp {NI} đầu vào đó một lần vào bộ nhớ chia sẻ</>}>
                  {total} reads of {NI} inputs: a block could load the {NI} once into shared memory
                </Translated>
              </text>
            </g>
          ) : null}
        </g>
        <Headline><Translated vi="gather chia sẻ đầu vào giữa các luồng">gather shares inputs between threads</Translated></Headline>
      </Stage>
    );
  }
  // The histogram preview: bin b's thread looks at every input to find its own.
  const bin = 1;
  const hp = M.hp;
  const w = M.histGather.workers[bin];
  const cursor = Math.min(hp.nIn, Math.floor(ramp(p, 0.6, 0.92) * (hp.nIn + 0.999)));
  const found = hp.values.slice(0, cursor).filter((v) => v === bin).length;
  return (
    <Stage beat={5}>
      <text x={14} y={IN_Y + CH / 2 + 4} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="vào: bin">in: bin</Translated></text>
      <g data-row="hist" data-values={hp.values.join(' ')} data-inspected={cursor} data-found={found} data-inspects={w.inspects}>
        {hp.values.map((v, i) => {
          const seen = i < cursor;
          const mine = seen && v === bin;
          return (
            <g key={i}>
              <Cell x={inX(i)} y={IN_Y} v={v} fill={mine ? 'var(--global)' : seen ? 'var(--global-wash)' : 'var(--cell)'} stroke="var(--global)" ink={mine ? 'var(--page)' : 'var(--ink)'} />
              {seen ? <text x={inC(i)} y={IN_Y - 8} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{mine ? tr('mine', 'của tôi') : tr('no', 'không')}</text> : null}
            </g>
          );
        })}
      </g>
      {cursor > 0 && cursor <= hp.nIn ? (
        <Arrow x1={W / 2} y1={TH_Y - 16} x2={inC(Math.min(cursor, hp.nIn) - 1)} y2={IN_Y + CH + 2} color="var(--ink)" width={1.6} />
      ) : null}
      <Thread x={W / 2} label={`b${bin}`} hot />
      <rect x={W / 2 + 20} y={TH_Y - 11} width={30} height={22} rx="2" fill="var(--reg-wash)" stroke="var(--reg)" />
      <text x={W / 2 + 35} y={TH_Y + 4} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--reg)' }}>{found}</text>
      <text x={W / 2} y={OUT_Y - 30} fontSize="12.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>Một histogram, mỗi bin một luồng: {SCENE.histBins} bin trên {hp.nIn} đầu vào (giá trị của mỗi đầu vào mod {SCENE.histBins}).</>}>
          A histogram, one thread per bin: {SCENE.histBins} bins of {hp.nIn} inputs (each input's value mod {SCENE.histBins}).
        </Translated>
      </text>
      <Headline>{tr(`inspected ${cursor} of ${w.inspects} inputs, found ${found} of its ${w.useful.length}`, `đã xem ${cursor} trên ${w.inspects} đầu vào, tìm thấy ${found} trên ${w.useful.length} của nó`)}</Headline>
    </Stage>
  );
}

// ---------------------------------------------------------------- beat 6: the questions
const QUESTIONS = () => [
  tr('Who writes each output?', 'Ai ghi mỗi đầu ra?'),
  tr('Who reads each input?', 'Ai đọc mỗi đầu vào?'),
  tr('How many threads does the cut expose?', 'Cách cắt này lộ ra bao nhiêu luồng?'),
  tr('How much work does each thread get?', 'Mỗi luồng nhận bao nhiêu việc?'),
];
const CRITERIA = () => [
  tr('gather or scatter: are atomics needed?', 'gather hay scatter: có cần atomic không?'),
  tr('the parallelism exposed', 'mức song song được lộ ra'),
  tr('how easily a thread finds its inputs', 'một luồng tìm đầu vào của nó dễ đến đâu'),
  tr('load balance', 'cân bằng tải'),
  tr('others that depend on the application', 'các tiêu chí khác tùy ứng dụng'),
];

function Summary({ p }: { p: number }) {
  const q = Math.ceil(ramp(p, 0.05, 0.6) * QUESTIONS().length);
  const c = Math.ceil(ramp(p, 0.55, 0.9) * CRITERIA().length);
  return (
    <Stage beat={6}>
      <rect x={24} y={24} width={W - 48} height={290} rx="6" fill="var(--page)" stroke="var(--rule)" />
      <text x={48} y={60} fontSize="14" fontWeight="600"><Translated vi="Trước khi viết một kernel, hãy hỏi">Before writing a kernel, ask</Translated></text>
      <g data-questions={q}>
        {QUESTIONS().slice(0, q).map((s, k) => (
          <text key={k} x={48} y={98 + k * 34} fontSize="14">{k + 1}. {s}</text>
        ))}
      </g>
      <line x1={W / 2 + 10} x2={W / 2 + 10} y1={44} y2={294} stroke="var(--rule)" />
      <text x={W / 2 + 32} y={60} fontSize="14" fontWeight="600"><Translated vi="Các tiêu chí của sách (p. 533)">The book's criteria (p. 533)</Translated></text>
      <g data-criteria={c}>
        {CRITERIA().slice(0, c).map((s, k) => (
          <text key={k} x={W / 2 + 32} y={98 + k * 34} fontSize="13.5">{s}</text>
        ))}
      </g>
      <Headline><Translated vi="atomic chỉ là một trong nhiều câu hỏi">atomics are one question of several</Translated></Headline>
    </Stage>
  );
}

const beats: Beat[] = [
  { clip: 'gather-scatter-1', get label() { return tr('Two ways to cut', 'Hai cách cắt'); }, draw: (p) => <Graph p={p} /> },
  { clip: 'gather-scatter-2', get label() { return tr('Gather', 'Gather'); }, draw: (p) => <Gather p={p} /> },
  { clip: 'gather-scatter-3', get label() { return tr('Scatter', 'Scatter'); }, draw: (p) => <Scatter p={p} /> },
  { clip: 'gather-scatter-4', get label() { return tr('The queue', 'Hàng đợi'); }, draw: (p) => <Queue p={p} /> },
  { clip: 'gather-scatter-5', get label() { return tr('Gather is not free', 'Gather không miễn phí'); }, draw: (p) => <NotFree p={p} /> },
  { clip: 'gather-scatter-6', get label() { return tr('Four questions', 'Bốn câu hỏi'); }, draw: (p) => <Summary p={p} /> },
];

export function GatherScatterScene() {
  return <Scene chapter="ch22" title={tr('Gather and scatter, in time', 'Gather và scatter, theo thời gian')} beats={beats} />;
}
