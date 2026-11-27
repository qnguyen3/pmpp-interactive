import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  BOOK_OVERSUBSCRIPTION, PRESETS, estimate, idleCycles, period, runs, simulate, utilisation, warpsNeeded, type Cell,
  type LatencyConfig, type Timeline,
} from '../../sim/latency';
import { LatencyScene } from './LatencyScene';
import { Translated, tr } from '../../lib/i18n';

const pct = (fraction: number) => Math.round(fraction * 1000) / 10;
const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

// ---------------------------------------------------------------- the timeline drawing
const FILL: Record<Cell, string> = { exec: 'var(--ink)', wait: 'var(--global-wash)', ready: 'var(--rule)' };
const STROKE: Record<Cell, string> = { exec: 'var(--ink)', wait: 'var(--global)', ready: 'var(--ink-3)' };
const CELL_NAME: Record<Cell, () => string> = {
  exec: () => tr('executing', 'đang thực thi'),
  wait: () => tr('waiting on memory', 'đang chờ bộ nhớ'),
  ready: () => tr('ready, not chosen', 'sẵn sàng, chưa được chọn'),
};

function CellLegend() {
  return (
    <span className="legend">
      {(['exec', 'wait', 'ready'] as Cell[]).map((c) => (
        <span key={c}>
          <i style={{ background: FILL[c], outline: `1px solid ${STROKE[c]}` }} />
          {CELL_NAME[c]()}
        </span>
      ))}
    </span>
  );
}

/** One row per warp, the execution unit underneath, from a cold start. */
function WarpTimeline({ tl, label }: { tl: Timeline; label: string }) {
  const R = tl.rows.length;
  const W = 720;
  const X = 44;
  const plotW = W - X - 8;
  const cw = plotW / tl.cycles;
  // Few warps get tall rows, so that the drawing stays legible when a narrow screen scales it down.
  const rh = Math.max(3, Math.min(R <= 8 ? 26 : 16, Math.floor(232 / R)));
  const ready = tl.rows.reduce((s, row) => s + row.filter((c) => c === 'ready').length, 0);
  const gap = rh >= 8 ? 2 : rh >= 5 ? 1 : 0.5;
  const top = 6;
  const uy = top + R * rh + 10;
  const uh = 16;
  const H = uy + uh + 22;
  const slots = tl.config.slots;
  // The unit strip as runs of cycles with the same number of slots in use.
  const used: { start: number; length: number; n: number }[] = [];
  tl.chosen.forEach((pick, t) => {
    const last = used[used.length - 1];
    if (last && last.n === pick.length) last.length += 1;
    else used.push({ start: t, length: 1, n: pick.length });
  });
  const labelled = R <= 16 ? tl.rows.map((_, w) => w) : [0, R - 1];
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={label}
      data-rows={R}
      data-cycles={tl.cycles}
      data-idle={idleCycles(tl)}
      data-ready={ready}
      style={{ width: '100%', display: 'block' }}
    >
      {tl.rows.map((row, w) => (
        <g key={w} data-warp={w}>
          {runs(row).map((r) => (
            <rect
              key={r.start}
              x={X + r.start * cw}
              y={top + w * rh}
              width={Math.max(0.6, r.length * cw - (cw > 3 ? 0.6 : 0))}
              height={rh - gap}
              fill={FILL[r.state]}
              stroke={rh >= 8 ? STROKE[r.state] : 'none'}
              strokeWidth="0.6"
            />
          ))}
        </g>
      ))}
      {labelled.map((w) => (
        <text key={w} x={X - 6} y={top + w * rh + Math.max(rh / 2 + 3, 8)} fontSize={rh >= 20 ? 12 : 9.5} textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
          w{w}
        </text>
      ))}
      <text x={X - 6} y={uy + 11} fontSize="9.5" textAnchor="end" fontWeight="600"><Translated vi="đơn vị">unit</Translated></text>
      <rect x={X} y={uy} width={plotW} height={uh} fill="none" stroke="var(--ink-3)" strokeDasharray="2 2" strokeWidth="0.8" />
      {used.filter((u) => u.n > 0).map((u) => (
        <rect key={u.start} x={X + u.start * cw} y={uy + uh * (1 - u.n / slots)} width={u.length * cw} height={(uh * u.n) / slots} fill="var(--ink)" />
      ))}
      <text x={X} y={H - 4} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="chu kỳ 0">cycle 0</Translated></text>
      <text x={X + plotW} y={H - 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>chu kỳ {tl.cycles}</>}>cycle {tl.cycles}</Translated></text>
    </svg>
  );
}

/** How many cycles of a configuration to draw: three trips round one warp's loop, within limits. */
const drawnCycles = (c: LatencyConfig) => Math.max(48, Math.min(300, 3 * period(c)));

// ---------------------------------------------------------------- explorer
interface PresetButton {
  key: string;
  label: string;
  set: Partial<LatencyConfig>;
}
const PRESET_BUTTONS: PresetButton[] = [
  { key: 'load', get label() { return tr('A load every other instruction', 'Cứ hai lệnh có một lệnh nạp'); }, set: { ...PRESETS.loadHeavy } },
  { key: 'compute', get label() { return tr('Long compute between loads', 'Tính toán dài giữa các lần nạp'); }, set: { ...PRESETS.computeHeavy } },
  // 64 warp slots (p. 85) and four processing blocks (p. 76) on the book's example SM.
  { key: 'book', get label() { return tr("The book's example SM: 64 warps, 4 at a time", 'SM ví dụ của sách: 64 warp, mỗi lần 4'); }, set: { warps: 64, slots: 4 } },
];

function WarpsExplorer() {
  const [cfg, setCfg] = useState<LatencyConfig>({ warps: 4, ...PRESETS.loadHeavy, slots: 1 });
  const tl = useMemo(() => simulate(cfg, drawnCycles(cfg)), [cfg]);
  const use = useMemo(() => utilisation(cfg), [cfg]);
  const est = estimate(cfg);
  const need = warpsNeeded(cfg.work, cfg.latency, cfg.slots);
  const idle = idleCycles(tl);
  const active = (b: PresetButton) => (Object.keys(b.set) as (keyof LatencyConfig)[]).every((k) => cfg[k] === b.set[k]);

  const slider = (name: string, key: keyof LatencyConfig, min: number, max: number) => (
    <label className="field">
      {name} = {cfg[key]}
      <input
        type="range"
        min={min}
        max={max}
        value={cfg[key]}
        aria-label={name}
        style={{ width: '11rem' }}
        onChange={(e) => setCfg({ ...cfg, [key]: +e.target.value })}
      />
    </label>
  );

  const short = need - cfg.warps;
  return (
    <Bench
      title={tr('How many warps hide a wait?', 'Bao nhiêu warp thì che được một lần chờ?')}
      task={tr('Raise R until the unit stops idling. Then keep going, and see what the extra warps do.', 'Tăng R cho đến khi đơn vị hết ngồi không. Rồi tăng tiếp, và xem các warp thừa làm gì.')}
      foot={tr('Cycle counts here are illustrative inputs, not measurements of any device. The drawing starts cold, with every warp ready at cycle 0; the percentage is taken after that start-up has passed.', 'Số chu kỳ ở đây là đầu vào minh họa, không phải số đo của thiết bị nào. Hình bắt đầu từ trạng thái nguội, mọi warp đều sẵn sàng ở chu kỳ 0; tỉ lệ phần trăm được tính sau khi giai đoạn khởi động đó đã qua.')}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        {slider(tr('Resident warps R', 'Số warp thường trú R'), 'warps', 1, 64)}
        {slider(tr('Cycles of work between memory accesses c', 'Số chu kỳ làm việc giữa các lần truy cập bộ nhớ c'), 'work', 1, 32)}
        {slider(tr('Memory latency L, in cycles', 'Độ trễ bộ nhớ L, tính bằng chu kỳ'), 'latency', 1, 120)}
        {slider(tr('Issue slots', 'Số ô phát lệnh'), 'slots', 1, 4)}
      </div>
      <div className="row" style={{ marginBottom: '1rem' }} role="group" aria-label={tr('Presets', 'Cấu hình mẫu')}>
        <span className="muted"><Translated vi="Cấu hình mẫu">Presets</Translated></span>
        {PRESET_BUTTONS.map((b) => (
          <button key={b.key} className="quiet" data-preset={b.key} aria-pressed={active(b)} onClick={() => setCfg({ ...cfg, ...b.set })}>
            {b.label}
          </button>
        ))}
      </div>
      <div className="cols plot">
        <div className="panel">
          <div style={{ marginBottom: '0.5rem' }}>
            <CellLegend />
          </div>
          <WarpTimeline tl={tl} label={tr(`Timeline of ${cfg.warps} warps over ${tl.cycles} cycles`, `Dòng thời gian của ${cfg.warps} warp qua ${tl.cycles} chu kỳ`)} />
          <Translated
            vi={
              <p className="muted" style={{ margin: '0.5rem 0 0' }}>
                Mỗi warp một hàng. Dải có nhãn "đơn vị" được tô khi đơn vị thực thi phát một lệnh trong chu kỳ
                đó{cfg.slots > 1 ? `, theo tỉ lệ số ô đã dùng trong ${cfg.slots} ô của nó` : ''}, và để trống khi nó không
                có gì để phát. Trong {tl.cycles} chu kỳ được vẽ, nó không phát lệnh nào trong{' '}
                <span data-out="idle">{idle}</span> chu kỳ.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>
              One row per warp. The strip marked "unit" is filled when the execution unit issued an instruction in that
              cycle{cfg.slots > 1 ? `, in proportion to how many of its ${cfg.slots} slots were used` : ''}, and empty when
              it had nothing to issue. In the {tl.cycles} cycles drawn it issued nothing in{' '}
              <span data-out="idle">{idle}</span>.
            </p>
          </Translated>
        </div>
        <div className="panel" aria-live="polite">
          <p className="muted" style={{ margin: 0 }}>
            <Translated vi="Đơn vị thực thi bận">Execution unit busy</Translated>
          </p>
          <p style={{ fontSize: '2rem', fontWeight: 600, margin: '0 0 0.2rem', lineHeight: 1.1 }}>
            <output data-out="util" data-value={pct(use)}>
              {pct(use)}%
            </output>
          </p>
          <p style={{ margin: '0 0 0.6rem' }}>
            <Translated vi="Ước lượng bằng cách đếm:">Estimate by counting:</Translated>{' '}
            {cfg.slots > 1
              ? `min(1, R × c / (slots × (c + L))) = min(1, ${cfg.warps} × ${cfg.work} / (${cfg.slots} × ${period(cfg)}))`
              : `min(1, R × c / (c + L)) = min(1, ${cfg.warps} × ${cfg.work} / ${period(cfg)})`}{' '}
            ={' '}
            <b>
              <span data-out="estimate" data-value={pct(est)}>
                {pct(est)}%
              </span>
            </b>
          </p>
          <p style={{ margin: '0 0 0.6rem' }}>
            <Translated vi={<>Số warp cần để giữ {cfg.slots > 1 ? `cả ${cfg.slots} ô` : 'đơn vị'} luôn bận:</>}>
              Warps needed to keep {cfg.slots > 1 ? `all ${cfg.slots} slots` : 'the unit'} busy:
            </Translated>{' '}
            {cfg.slots > 1 ? `${cfg.slots} × ` : ''}({cfg.work} + {cfg.latency}) / {cfg.work}<Translated vi=", làm tròn lên">, rounded up</Translated> ={' '}
            <b>
              <span data-out="needed">{need}</span>
            </b>
          </p>
          <p style={{ margin: 0 }} data-out="verdict">
            {short > 0 ? (
              <Translated
                vi={
                  <>
                    <b>Thiếu {short} warp.</b> Đơn vị không có gì để phát trong {pct(1 - use)}%
                    {cfg.slots > 1 ? ' số ô' : ' số chu kỳ'} của nó. Khi giai đoạn khởi động nguội đã qua, mọi warp hoặc
                    đang thực thi hoặc đang chờ bộ nhớ; không warp nào phải chờ đến lượt dùng đơn vị.
                  </>
                }
              >
                <>
                  <b>{plural(short, 'warp')} short.</b> The unit has nothing to issue for {pct(1 - use)}% of its
                  {cfg.slots > 1 ? ' slots' : ' cycles'}. Once the cold start has passed, every warp is either executing
                  or waiting on memory; none is kept waiting for the unit.
                </>
              </Translated>
            ) : short === 0 ? (
              <Translated
                vi={
                  <>
                    <b>Vừa đủ.</b> Khi một warp bắt đầu chờ, dữ liệu của warp khác đã về.
                  </>
                }
              >
                <>
                  <b>Exactly enough.</b> As one warp starts to wait, another's data is back.
                </>
              </Translated>
            ) : (
              <Translated
                vi={
                  <>
                    <b>Dư {-short} warp so với mức cần.</b> Chúng không thêm được gì: đơn vị đã đầy. Dữ liệu của chúng về
                    và chúng xếp hàng chờ lượt, đó là phần màu xám trong hình. Mỗi warp vẫn giữ thanh ghi của mình.
                  </>
                }
              >
                <>
                  <b>{plural(-short, 'warp')} more than needed.</b> They add nothing: the unit was already full. Their
                  data comes back and they queue for a turn, which is the grey in the drawing. Each one still holds its
                  registers.
                </>
              </Translated>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the prediction's reveal
const PREDICTED: LatencyConfig = { warps: 16, ...PRESETS.loadHeavy, slots: 1 };
const PREDICTED_TL = simulate(PREDICTED, 96);
const NEEDED = warpsNeeded(PREDICTED.work, PREDICTED.latency);

// ---------------------------------------------------------------- questions
export const questionsS46: Record<string, Question> = {
  'ch04/4.6/q-util': {
    kind: 'numeric',
    id: 'ch04/4.6/q-util',
    prompt: (
      <Translated
        vi={
          <>
            Trong mô hình của phần này, 8 warp thường trú trên một đơn vị có một ô phát lệnh. Mỗi warp tính trong 4 chu kỳ
            giữa các lần truy cập bộ nhớ rồi chờ 60 chu kỳ. Đơn vị thực thi dành bao nhiêu phần trăm số chu kỳ của nó để
            phát lệnh?
          </>
        }
      >
        <>
          In the model of this section, 8 warps are resident on a unit with one issue slot. Each computes for 4 cycles
          between memory accesses and then waits 60 cycles. What fraction of its cycles does the execution unit spend
          issuing instructions, in percent?
        </>
      </Translated>
    ),
    answer: 50,
    tol: 0.005,
    unit: '%',
    explain: (
      <Translated
        vi={
          <>
            <p>
              Một warp dùng đơn vị trong 4 trên mỗi 4 + 60 = 64 chu kỳ. Tám warp dùng 32 trên mỗi 64 chu kỳ, và không warp
              nào phải xếp hàng, nên đơn vị bận một nửa thời gian.
            </p>
            <p>
              Cần 64 / 4 = 16 warp để lấp đầy nó. Kernel không đổi giữa 8 và 16, phần cứng cũng vậy. Chỉ có số warp thường
              trú thay đổi, và đó là chủ đề của phần tiếp theo.
            </p>
          </>
        }
      >
        <>
          <p>
            One warp uses the unit for 4 of every 4 + 60 = 64 cycles. Eight of them use 32 of every 64, and none of them
            ever has to queue, so the unit is busy half the time.
          </p>
          <p>
            It would take 64 / 4 = 16 warps to fill it. The kernel did not change between 8 and 16, and neither did the
            hardware. Only the number of warps resident did, and that is what the next section is about.
          </p>
        </>
      </Translated>
    ),
    source: 'our model of ch. 4 §4.6, pp. 83–85; a simplification, tested in app/src/sim/latency.ts',
    vi: { source: 'mô hình của chúng tôi cho ch. 4 §4.6, pp. 83–85; một sự đơn giản hóa, được kiểm thử trong app/src/sim/latency.ts' },
  },
  'ch04/4.6/q-zero': {
    kind: 'choice',
    id: 'ch04/4.6/q-zero',
    prompt: (
      <Translated
        vi={
          <>
            Một CPU phải trả giá cho mỗi lần chuyển ngữ cảnh. Một SM đổi warp mà nó phát lệnh thường xuyên tới mức mỗi lệnh
            một lần mà không tốn gì. Vì sao?
          </>
        }
      >
        <>
          A CPU pays for every context switch. An SM changes the warp it issues from as often as every instruction and
          pays nothing. Why?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Bộ lập lịch dự đoán warp nào sẽ sẵn sàng tiếp theo và nạp trước trạng thái của nó.">
            {'The scheduler predicts which warp will be ready next and loads its state ahead of time.'}
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì được nạp cả, dù sớm hay muộn. NVIDIA còn nói các SM hoàn toàn không thực thi suy đoán.">
            {'Nothing is loaded, early or late. NVIDIA also says SMs do no speculative execution at all.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Bộ đếm chương trình và thanh ghi của mọi warp thường trú nằm trong thanh ghi trên chip suốt thời gian warp đó tồn tại.">
            {"Every resident warp's program counter and registers stay in on-chip registers for as long as it lives."}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Mọi warp của một kernel chạy cùng một mã, nên chúng có thể dùng chung một bộ thanh ghi và không có gì phải hoán đổi.">
            {'All warps of a kernel run the same code, so they can share one set of registers and there is nothing to swap.'}
          </Translated>
        ),
        why: (
          <Translated vi="Chúng dùng chung mã, không dùng chung dữ liệu. Mỗi luồng của mỗi warp có giá trị thanh ghi riêng (phần 4.4).">
            {'They share the code, not the data. Each thread of each warp has its own register values (section 4.4).'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Trạng thái của warp bị thay ra được lưu vào bộ nhớ chia sẻ, đủ nhanh để chi phí coi như bằng không.">
            {"The outgoing warp's state is saved to shared memory, which is fast enough for the cost to round to zero."}
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì được lưu ở đâu cả. Trạng thái của một warp thường trú không bao giờ rời khỏi các thanh ghi chứa nó.">
            {'Nothing is saved anywhere. The state of a resident warp never leaves the registers it lives in.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một lần chuyển ngữ cảnh tốn thời gian vì trạng thái phải di chuyển: ra bộ nhớ cho luồng này, từ bộ nhớ vào cho
            luồng kia. Một SM giữ trạng thái của mọi warp được gán cho nó trong phần cứng suốt thời gian warp ở đó, nên chọn
            một warp khác không di chuyển gì cả. Chi phí không biến mất. Nó trở thành chi phí về không gian: mọi warp thường
            trú giữ thanh ghi của mình dù có đang thực thi hay không, và một kernel dùng nhiều thanh ghi mỗi luồng sẽ chừa
            chỗ cho ít warp hơn.
          </p>
        }
      >
        <p>
          A context switch costs time because state moves: out to memory for one thread, in from memory for the other.
          An SM keeps the state of every warp assigned to it in hardware for as long as the warp is there, so choosing
          another warp moves nothing. The cost has not vanished. It has become a cost in space: every resident warp
          holds its registers whether or not it is executing, and a kernel that uses many registers per thread leaves
          room for fewer warps.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.6, p. 84 and its sidebar; NVIDIA wording checked against the CUDA 13.5 guide, 2026-11-20',
    vi: { source: 'ch. 4 §4.6, p. 84 và khung bên lề của nó; cách diễn đạt của NVIDIA đã đối chiếu với CUDA 13.5 guide, 2026-11-20' },
  },
  'ch04/4.6/q-area': {
    kind: 'choice',
    id: 'ch04/4.6/q-area',
    prompt: (
      <Translated vi={<>Khả năng che độ trễ cho phép người thiết kế GPU bỏ đi thứ gì, và nó không làm được gì cho một luồng riêng lẻ?</>}>
        <>What does latency tolerance let a GPU designer leave out, and what does it not do for a single thread?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Bộ nhớ đệm lớn và dự đoán rẽ nhánh. Một luồng phải chờ vẫn chờ lâu như cũ; các warp khác dùng đơn vị.">
            {'Large caches and branch prediction. A thread that waits still waits just as long; other warps use the unit.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Bộ nhớ đệm lớn và dự đoán rẽ nhánh. Mỗi luồng cũng xong sớm hơn, vì chuyển giữa các warp không tốn gì.">
            {'Large caches and branch prediction. Each thread also finishes sooner, because switching between warps is free.'}
          </Translated>
        ),
        why: (
          <Translated vi="Chuyển đổi miễn phí giữ cho đơn vị luôn bận. Nó không rút ngắn thời gian chờ của bất kỳ luồng nào.">
            {'Free switching keeps the unit busy. It does not shorten any one thread\'s wait.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các đơn vị dấu phẩy động. Phép tính số học có thể bắt chờ, nên cần ít đơn vị hơn, và một luồng riêng lẻ không bị ảnh hưởng.">
            {'Floating-point units. Arithmetic can be made to wait, so fewer units are needed, and a single thread is unaffected.'}
          </Translated>
        ),
        why: (
          <Translated vi="Ngược lại: các đơn vị số học chính là thứ mà diện tích tiết kiệm được dành cho.">
            {'The other way round: arithmetic units are what the saved area is spent on.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các kênh bộ nhớ. Với đủ warp thường trú, bộ nhớ chậm không còn quan trọng với luồng nào hay với SM.">
            {'Memory channels. With enough warps resident, a slow memory no longer matters to any thread or to the SM.'}
          </Translated>
        ),
        why: (
          <Translated vi="Kênh bộ nhớ là thứ còn lại mà diện tích được dành cho. Che một lần chờ không giống với loại bỏ lưu lượng; chương 5 nói về giới hạn đó.">
            {'Memory channels are the other thing the area is spent on. Hiding a wait is not the same as removing the traffic; chapter 5 is about that limit.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một lõi CPU chỉ có một luồng để chạy, nên nó dành diện tích để lệnh kế tiếp của luồng đó đến nhanh: bộ nhớ đệm
            để dữ liệu ở gần, dự đoán để một nhánh không làm kẹt. Một SM luôn có warp khác để phát lệnh, nên nó có thể để
            một lệnh chậm cứ chậm và dành diện tích cho các đơn vị số học và kênh bộ nhớ. Điều đó tăng lượng việc cả SM làm
            được mỗi giây. Nó không giúp gì cho thời gian một luồng cần để chạy hết các lệnh của chính nó, đó là sự đánh đổi
            mà chương 1 đã mô tả.
          </p>
        }
      >
        <p>
          A CPU core has one thread to run, so it spends area on making that thread's next instruction arrive quickly:
          caches so that data is near, prediction so that a branch does not stall. An SM always has another warp to
          issue from, so it can leave a slow instruction slow and spend the area on arithmetic units and memory
          channels. That raises the work done per second by the whole SM. It does nothing for the time one thread takes
          to get through its own instructions, which is the trade chapter 1 described.
        </p>
      </Translated>
    ),
    source: 'ch. 4 §4.6, p. 84',
  },
  'ch04/4.6/q-lowocc': {
    kind: 'choice',
    id: 'ch04/4.6/q-lowocc',
    prompt: (
      <Translated
        vi={
          <>
            Phép nhân ma trận ở chương 15 giữ các ô (tile) lớn trong thanh ghi và chạy với một khối 256 luồng mỗi SM, bằng
            một phần tám số ô luồng trong ví dụ của sách. Một kernel nhân ma trận với vector lúc decode trên cùng thiết bị
            chạy tốt nhất khi SM đầy. Vì sao kernel đầu tiên chỉ cần ít warp như vậy?
          </>
        }
      >
        <>
          Chapter 15's matrix multiplication keeps large tiles in registers and runs with one block of 256 threads per
          SM, an eighth of the thread slots on the book's example. A decode-time matrix-vector kernel on the same
          device does best with the SM full. Why can the first get away with so few warps?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Giữa các lần truy cập bộ nhớ, các warp của nó tính nhiều chu kỳ trên thanh ghi, nên vài warp đủ giữ các đơn vị luôn bận.">
            {'Between memory accesses its warps compute for many cycles on registers, so a few keep the units busy.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Phép nhân ma trận không có phân kỳ điều khiển, nên mỗi warp của nó cần ít chu kỳ hơn cho mỗi kết quả.">
            {'Matrix multiplication has no control divergence, so each of its warps needs fewer cycles per result.'}
          </Translated>
        ),
        why: (
          <Translated vi="Cả hai kernel đều không phân kỳ nhiều. Phân kỳ nói về số lượt bên trong một warp, không phải về số warp cần để che một lần chờ.">
            {'Neither kernel diverges much. Divergence is about passes within a warp, not about how many warps are needed to cover a wait.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Khi ít warp thường trú, mỗi warp được nhiều băng thông bộ nhớ hơn, và đó là thứ giới hạn cả hai kernel.">
            {'With fewer warps resident each one gets more memory bandwidth, and that is what limits both kernels.'}
          </Translated>
        ),
        why: (
          <Translated vi="Nếu vậy thì occupancy thấp cũng có lợi cho kernel bị giới hạn bởi bộ nhớ, trong khi chính nó lại là kernel muốn SM đầy.">
            {'That would favour low occupancy for the memory-bound kernel too, and it is the one that wants the SM full.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các khối lớn được lập lịch trước, nên một khối 256 luồng không bao giờ phải chờ SM trống.">
            {'Large blocks are scheduled first, so a 256-thread block is never kept waiting for a free SM.'}
          </Translated>
        ),
        why: (
          <Translated vi="Không nhà cung cấp nào mô tả thứ tự kiểu đó, và nó cũng không giải thích được mức sử dụng bên trong SM.">
            {'No vendor documents an order of that kind, and it would not explain utilisation inside the SM.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Theo ngôn ngữ của phần này, kernel đầu tiên có c lớn: những đoạn tính toán số học dài trên thanh ghi cho mỗi
              lần đi tới bộ nhớ. Số warp cần là khoảng (c + L) / c cho mỗi ô phát lệnh, gần bằng 1 khi c lớn. Kernel thứ hai
              có c nhỏ, nên cùng công thức đó đòi nhiều warp.
            </p>
            <p>
              Dù sao kernel nặng thanh ghi cũng không thể có thêm warp: chính thanh ghi giới hạn nó ở một khối. Điểm chính là
              nó mất rất ít vì điều đó. Hướng dẫn của AMD cũng nói vậy về các GEMM dùng matrix core của họ: chúng thường chạy
              ở occupancy thấp vì vài warp đã làm bão hòa các pipeline.
            </p>
          </>
        }
      >
        <>
          <p>
            In this section's terms the first kernel has a large c: long stretches of arithmetic on registers for each
            trip to memory. The warps needed are about (c + L) / c for each issue slot, which is close to 1 when c is
            large. The second has a small c, so the same formula asks for many warps.
          </p>
          <p>
            The register-heavy kernel could not have more warps anyway: its registers are what limit it to one block.
            The point is that it loses little by that. AMD's guide says the same of its matrix-core GEMMs: they often
            run at low occupancy because a few warps saturate the pipelines.
          </p>
        </>
      </Translated>
    ),
    source: 'ch. 15 §15.7, p. 363; AMD HIP performance guide, checked 2026-11-20; the model is ours',
    vi: { source: 'ch. 15 §15.7, p. 363; AMD HIP performance guide, đối chiếu ngày 2026-11-20; mô hình là của chúng tôi' },
  },
};

// ---------------------------------------------------------------- the section
export function S46() {
  return (
    <>
      <Translated
        vi={
          <p>
            Phần 4.4 đặt một warp lên một processing block: một lần nạp lệnh, mọi lane. Nó để lại một điều chưa giải thích.
            SM ví dụ của sách có {BOOK_OVERSUBSCRIPTION.coresPerSm} streaming processor và chỗ cho{' '}
            {BOOK_OVERSUBSCRIPTION.threadsPerSm} luồng. Mười sáu luồng cho mỗi bộ xử lý thì không thể cùng thực thi một lúc.
            Mười lăm luồng còn lại để làm gì?
          </p>
        }
      >
        <p>
          Section 4.4 put a warp on a processing block: one fetch, every lane. It left a mismatch unexplained. The
          book's example SM has {BOOK_OVERSUBSCRIPTION.coresPerSm} streaming processors and room for{' '}
          {BOOK_OVERSUBSCRIPTION.threadsPerSm} threads. Sixteen threads for every processor cannot all execute at once.
          What are the other fifteen for?
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Chúng ở đó để chờ, và đó chính là thiết kế. Tại mỗi thời điểm, một SM chỉ thực thi một số warp được gán cho nó.
            Các thiết kế ban đầu phát một lệnh, cho một warp, mỗi lần. Các thiết kế sau có nhiều processing block và phát
            lệnh cho vài warp cùng lúc, như SM hai block trong Fig. 4.8 của sách. Dù theo cách nào, số warp thường trú vẫn
            nhiều hơn hẳn số warp đang thực thi.
          </p>
        }
      >
        <p>
          They are there to wait, and that is the design. At any instant an SM executes only some of the warps assigned
          to it. Early designs issued one instruction, for one warp, at a time. Later ones have several processing
          blocks and issue for a few warps at once, like the two-block SM of the book's Fig. 4.8. Either way the warps
          that are resident far outnumber the ones executing.
        </p>
      </Translated>
      <Source section="4.6" pages="82–83" figs={tr('refers back to Fig. 4.8', 'nhắc lại Fig. 4.8')} />

      <h2><Translated vi="Các warp thừa để làm gì">What the extra warps are for</Translated></h2>
      <Translated
        vi={
          <p>
            Một số lệnh cần nhiều thời gian mới trả kết quả. Ví dụ chính của sách là một lần đọc từ bộ nhớ toàn cục (global
            memory): yêu cầu rời SM tới DRAM ngoài chip của phần 4.1, và dữ liệu quay về sau nhiều chu kỳ. Một warp có lệnh
            kế tiếp cần dữ liệu đó không thể làm gì hữu ích cho đến khi nó về.
          </p>
        }
      >
        <p>
          Some instructions take a long time to deliver their result. The book's main example is a read from global
          memory: the request leaves the SM for the off-chip DRAM of section 4.1, and the data comes back many cycles
          later. A warp whose next instruction needs that data can do nothing useful until it arrives.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            SM không chờ cùng nó. Một warp đang chờ một thao tác trước đó sẽ không được chọn. Mọi warp khác đều{' '}
            <b>sẵn sàng</b>, và khi có nhiều warp sẵn sàng, một logic ưu tiên chọn một warp để phát lệnh. Việc chọn giữa các
            warp sẵn sàng ở mỗi lần phát lệnh được gọi là <b>đa luồng mịn (fine-grained multithreading)</b>. Tác dụng của nó
            là các đơn vị thực thi luôn bận dù từng lệnh riêng lẻ chậm, điều mà sách gọi là <b>khả năng che độ trễ (latency
            tolerance)</b> hay latency hiding.
          </p>
        }
      >
        <p>
          The SM does not wait with it. A warp that is waiting on an earlier operation is not selected. Every other warp
          is <b>ready</b>, and when several are ready a priority logic picks one to issue from. Choosing among the ready
          warps at every issue is called <b>fine-grained multithreading</b>. Its effect is that the execution units stay
          busy although individual instructions are slow, which the book calls <b>latency tolerance</b> or latency
          hiding.
        </p>
      </Translated>
      <Source section="4.6" pages="83" />
      <Translated
        vi={
          <p>
            Tác dụng này là chuyện thời gian, nên hãy xem nó diễn ra theo thời gian trước khi đọc tiếp. Hoạt hình dùng một
            trường hợp nhỏ: 2 chu kỳ làm việc, rồi 6 chu kỳ chờ.
          </p>
        }
      >
        <p>
          The effect is about time, so watch it in time before reading on. The animation uses a small case: 2 cycles of
          work, then a 6-cycle wait.
        </p>
      </Translated>
      <LatencyScene />

      <h2><Translated vi="Những lần chờ khác nó che được">Other waits it hides</Translated></h2>
      <Translated
        vi={
          <p>
            Bộ nhớ là lần chờ dài, nhưng không phải duy nhất. Phép tính dấu phẩy động được pipeline, nên kết quả không có
            ngay ở chu kỳ sau khi được yêu cầu, và một nhánh có điều kiện cần thời gian để phân giải. Cùng cơ chế lập lịch đó
            bao cả hai. Trong khi kết quả của một warp đang trên đường về, đơn vị phát lệnh từ warp khác.
          </p>
        }
      >
        <p>
          Memory is the long wait, not the only one. Floating-point arithmetic is pipelined, so a result is not
          available the cycle after it is asked for, and a conditional branch takes time to resolve. The same
          scheduling covers both. While one warp's result is on its way, the unit issues from another.
        </p>
      </Translated>
      <Source section="4.6" pages="83–84" />

      <h2><Translated vi="Vì sao việc chuyển đổi không tốn gì">Why the switch is free</Translated></h2>
      <Translated
        vi={
          <p>
            Trên CPU, chuyển một lõi từ luồng này sang luồng khác là một lần chuyển ngữ cảnh. Bộ đếm chương trình và thanh
            ghi của luồng bị thay ra được lưu vào bộ nhớ, và của luồng được đưa vào thì được nạp lên, việc đó tốn thời gian.
            Một thiết kế phải trả cái giá đó ở mỗi lần chuyển thì không thể chuyển ở mỗi lệnh.
          </p>
        }
      >
        <p>
          On a CPU, moving a core from one thread to another is a context switch. The program counter and registers of
          the outgoing thread are saved to memory and those of the incoming thread are loaded, and that takes time. A
          design that paid that price at every switch could not afford to switch at every instruction.
        </p>
      </Translated>
      <Source section="4.6" pages="84" figs={tr('and its sidebar on context switching', 'và khung bên lề về chuyển ngữ cảnh')} />
      <Translated
        vi={
          <p>
            Một SM không bao giờ làm vậy. Trạng thái thực thi của mọi warp được gán cho nó nằm trong thanh ghi phần cứng từ
            lúc warp đến cho tới khi nó xong. Chọn một warp khác chỉ thay đổi nơi lệnh kế tiếp được lấy, không gì khác, nên
            không có thời gian rỗi nào xuất hiện giữa lệnh của warp này và lệnh của warp kia. Sách gọi điều này là{' '}
            <b>lập lịch không tốn chi phí (zero-overhead scheduling)</b>.
          </p>
        }
      >
        <p>
          An SM never does it. The execution state of every warp assigned to it stays in hardware registers from the
          moment the warp arrives until it finishes. Selecting a different warp changes where the next instruction comes
          from and nothing else, so no idle time appears between one warp's instruction and another's. The book's name
          for this is <b>zero-overhead scheduling</b>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the CUDA 13.5 guide (ch04-facts.md, A11)', '2026-11-20, đối chiếu với CUDA 13.5 guide (ch04-facts.md, A11)')}>
        <Translated
          vi={
            <p>
              NVIDIA: ngữ cảnh của mỗi warp được "maintained on-chip throughout the warp's lifetime" (giữ trên chip suốt
              vòng đời của warp), nên chuyển đổi không tốn gì; SM phát lệnh theo thứ tự, không có dự đoán rẽ nhánh hay thực
              thi suy đoán. Cách chọn một warp sẵn sàng không được mô tả trong tài liệu.
            </p>
          }
        >
          <p>
            NVIDIA: each warp's context is "maintained on-chip throughout the warp's lifetime", so switching costs
            nothing; SMs issue in order, with no branch prediction or speculation. How a ready warp is chosen is not
            documented.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Không tốn thời gian không có nghĩa là không tốn không gian. Mọi warp thường trú giữ thanh ghi của mình suốt thời
            gian nó thường trú, dù nó đang thực thi, đang chờ bộ nhớ, hay sẵn sàng mà chưa được chọn. Một kernel dùng nhiều
            thanh ghi mỗi luồng hơn sẽ chừa chỗ cho ít warp hơn, và phần 4.7 tính xem bao nhiêu.
          </p>
        }
      >
        <p>
          Free in time is not free in space. Every resident warp holds its registers for as long as it is resident,
          whether it is executing, waiting on memory, or ready and not chosen. A kernel that uses more registers per
          thread leaves room for fewer warps, and section 4.7 works out how many.
        </p>
      </Translated>
      <Beyond title={tr('On HIP and Metal', 'Trên HIP và Metal')}>
        <Translated
          vi={
            <p>
              AMD mô tả điều tương tự cho một compute unit (chuyển trong một chu kỳ, mọi ngữ cảnh đều thường trú) và công bố
              chính sách của họ: warp cũ nhất nạp lệnh trước, việc phát lệnh lần lượt xoay vòng qua các pool. Không tìm thấy
              tuyên bố nào của Apple: với Metal, điều này là unverified.
            </p>
          }
        >
          <p>
            AMD documents the same for a compute unit (single-cycle switching, all contexts resident) and publishes its
            policy: oldest warp fetches first, issue goes round the pools in turn. No Apple statement found: for Metal
            this is unverified.
          </p>
        </Translated>
        <p className="muted"><Translated vi="Đối chiếu ngày 2026-11-20 (ch04-facts.md, B8).">Checked 2026-11-20 (ch04-facts.md, B8).</Translated></p>
      </Beyond>
      <Ask q={questionsS46['ch04/4.6/q-zero']} />

      <h2><Translated vi="Thiết kế dùng phần tiết kiệm vào đâu">What the design spends the savings on</Translated></h2>
      <Translated
        vi={
          <p>
            Khả năng che độ trễ là lý do chính khiến GPU có thể bỏ đi hai thứ mà CPU tốn rất nhiều diện tích chip: bộ nhớ đệm
            (cache) lớn và dự đoán rẽ nhánh. Một lõi CPU chỉ có một luồng để chạy và phải làm cho lệnh kế tiếp của luồng đó
            đến nhanh. Một SM có các warp khác để chạy, nên nó có thể để một lệnh chậm cứ chậm. Diện tích được dành cho các
            đơn vị thực thi dấu phẩy động và các kênh truy cập bộ nhớ.
          </p>
        }
      >
        <p>
          Latency tolerance is the main reason a GPU can do without two things a CPU spends a great deal of chip area
          on: large caches and branch prediction. A CPU core has one thread to run and must make that thread's next
          instruction arrive quickly. An SM has other warps to run, so it can leave a slow instruction slow. The area
          goes to floating-point execution units and memory access channels.
        </p>
      </Translated>
      <Source section="4.6" pages="84" />
      <Ask q={questionsS46['ch04/4.6/q-area']} />

      <h2><Translated vi="Bao nhiêu warp là đủ">How many warps is enough</Translated></h2>
      <Translated
        vi={
          <p>
            Để điều này hoạt động, một SM phải giữ nhiều luồng hơn hẳn số nó có thể thực thi cùng lúc. Ví dụ của sách giữ tới{' '}
            {BOOK_OVERSUBSCRIPTION.threadsPerSm} luồng cho {BOOK_OVERSUBSCRIPTION.coresPerSm} streaming processor, gấp{' '}
            {BOOK_OVERSUBSCRIPTION.ratio} lần số có thể thực thi trong một chu kỳ, và sách gọi sự đăng ký vượt mức
            (oversubscription) đó là thiết yếu.
          </p>
        }
      >
        <p>
          For this to work an SM has to hold many more threads than it can execute at once. The book's example holds up
          to {BOOK_OVERSUBSCRIPTION.threadsPerSm} threads for {BOOK_OVERSUBSCRIPTION.coresPerSm} streaming processors,{' '}
          {BOOK_OVERSUBSCRIPTION.ratio} times more than can execute in any one cycle, and the book calls that
          oversubscription essential.
        </p>
      </Translated>
      <Source section="4.6" pages="84–85" />
      <Translated
        vi={
          <p>
            Một kernel cụ thể cần bao nhiêu thì tùy vào kernel đó, và một mô hình nhỏ là đủ để thấy vì sao. Một warp phát c
            lệnh, mỗi chu kỳ một lệnh, và lệnh cuối cùng bắt đầu một lần truy cập bộ nhớ. Rồi nó chờ L chu kỳ. Rồi lặp lại.
            Một mình, nó giữ đơn vị bận trong c trên mỗi c + L chu kỳ. R warp giữ nó bận gấp R lần, cho đến khi đơn vị đầy:
          </p>
        }
      >
        <p>
          How many a particular kernel needs depends on the kernel, and a small model is enough to see how. A warp
          issues c instructions, one per cycle, and the last of them starts a memory access. Then it waits L cycles.
          Then it repeats. Alone, it keeps the unit busy for c of every c + L cycles. R warps keep it busy R times as
          much, until the unit is full:
        </p>
      </Translated>
      <Translated
        vi={
          <p className="ui" style={{ fontSize: '1rem' }}>
            mức sử dụng = min(1, R × c / (c + L)), nên đơn vị đầy từ R = (c + L) / c warp, làm tròn lên.
          </p>
        }
      >
        <p className="ui" style={{ fontSize: '1rem' }}>
          utilisation = min(1, R × c / (c + L)), so the unit is full from R = (c + L) / c warps, rounded up.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Thời gian chờ cố định, không giới hạn số warp chờ cùng lúc: ở đây bộ nhớ có băng thông vô hạn. Mỗi chu kỳ một
              lệnh. Mọi warp chạy cùng một vòng lặp.
            </p>
          }
        >
          <p>
            A fixed wait, with no limit on how many warps wait at once: memory has unlimited bandwidth here. One
            instruction per cycle. Every warp runs the same loop.
          </p>
        </Translated>
        <Translated vi={<p>Warp sẵn sàng được chọn theo thứ tự cũ nhất trước. Đó là lựa chọn của chúng tôi; NVIDIA không mô tả chính sách của họ.</p>}>
          <p>Ready warps are chosen oldest first. That is our choice; NVIDIA does not document its policy.</p>
        </Translated>
      </Simplification>
      <Predict
        id="ch04/4.6/p-warps-needed"
        prompt={
          <Translated
            vi={
              <>
                Mỗi warp của một kernel làm việc {PREDICTED.work} chu kỳ rồi chờ bộ nhớ toàn cục {PREDICTED.latency} chu kỳ.
                Cần bao nhiêu warp thường trú để giữ một đơn vị thực thi luôn bận?
              </>
            }
          >
            <>
              A kernel's warps each do {PREDICTED.work} cycles of work and then wait {PREDICTED.latency} cycles on global
              memory. How many warps have to be resident to keep one execution unit busy all the time?
            </>
          </Translated>
        }
        unit={tr('warps', 'warp')}
      >
        {(v) => (
          <>
            <p>
              <Translated
                vi={
                  <>
                    <b>{NEEDED}.</b> Một warp dùng đơn vị trong {PREDICTED.work} trên mỗi {period(PREDICTED)} chu kỳ, nên cần{' '}
                    {period(PREDICTED)} / {PREDICTED.work} = {NEEDED} warp để lấp đầy nó.{' '}
                  </>
                }
              >
                <b>{NEEDED}.</b> One warp uses the unit for {PREDICTED.work} of every {period(PREDICTED)} cycles, so it
                takes {period(PREDICTED)} / {PREDICTED.work} = {NEEDED} of them to fill it.{' '}
              </Translated>
              {v === NEEDED
                ? tr('That is what you had.', 'Đúng như bạn đã đoán.')
                : v < NEEDED
                  ? tr(`With ${Math.max(1, Math.floor(v))} the unit would be busy ${pct(utilisation({ ...PREDICTED, warps: Math.max(1, Math.floor(v)) }))}% of the time.`, `Với ${Math.max(1, Math.floor(v))} warp, đơn vị sẽ bận ${pct(utilisation({ ...PREDICTED, warps: Math.max(1, Math.floor(v)) }))}% thời gian.`)
                  : tr(`With ${Math.floor(v)} it would be no busier than with ${NEEDED}: the extra warps queue.`, `Với ${Math.floor(v)} warp, nó cũng không bận hơn so với ${NEEDED}: các warp thừa phải xếp hàng.`)}{' '}
              <Translated vi={<>Bên dưới là {NEEDED} warp từ trạng thái nguội. Dải của đơn vị không có khoảng trống nào.</>}>
                Below, {NEEDED} warps from a cold start. The unit strip has no gap.
              </Translated>
            </p>
            <div className="panel" data-out="predicted">
              <WarpTimeline tl={PREDICTED_TL} label={tr(`Timeline of ${NEEDED} warps, each working ${PREDICTED.work} cycles and waiting ${PREDICTED.latency}`, `Dòng thời gian của ${NEEDED} warp, mỗi warp làm việc ${PREDICTED.work} chu kỳ và chờ ${PREDICTED.latency}`)} />
            </div>
          </>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá chạy mô hình và vẽ mỗi warp đang làm gì ở mỗi chu kỳ. Có ba điều đáng tìm ra. Bắt đầu dưới ngưỡng
            và tăng R từng warp một. Rồi vượt qua ngưỡng và xem các warp thừa làm gì với thời gian của chúng. Rồi chuyển qua
            lại giữa hai cấu hình kernel mẫu mà không chạm vào R. Thanh trượt thứ tư cho phép đơn vị phát lệnh cho nhiều hơn
            một warp mỗi chu kỳ, như một SM có nhiều processing block; cấu hình mẫu cuối cùng là SM ví dụ của sách, 64 ô warp
            và bốn processing block.
          </p>
        }
      >
        <p>
          The explorer runs the model and draws what each warp is doing in each cycle. Three things are worth finding.
          Start below the threshold and raise R one warp at a time. Then go past the threshold and look at what the
          extra warps do with their time. Then switch between the two kernel presets without touching R. The fourth
          slider lets the unit issue for more than one warp per cycle, as an SM with several processing blocks does;
          the last preset is the book's example SM, 64 warp slots and four processing blocks.
        </p>
      </Translated>
      <WarpsExplorer />
      <Ask q={questionsS46['ch04/4.6/q-util']} />
      <Translated
        vi={
          <p>
            Dưới ngưỡng, đơn vị ngồi không, và mỗi warp bạn thêm vào đều được dùng trọn. Tại ngưỡng, việc ngồi không dừng lại.
            Vượt qua nó, các warp thừa không thay đổi gì bạn đo được ở đơn vị: dữ liệu của chúng về và chúng chờ lượt. Và
            chính ngưỡng được quyết định bởi tỉ lệ giữa chờ và làm việc, thứ thuộc về kernel. Cấu hình mẫu cuối cùng đưa SM ví
            dụ của sách vào mô hình: 64 ô warp, bốn processing block. Ở đó nó luôn bận miễn là một warp chờ không quá 15 chu
            kỳ cho mỗi chu kỳ làm việc, đó chính là tỉ lệ 16 trên 1 của sách nhìn từ phía kernel.
          </p>
        }
      >
        <p>
          Below the threshold the unit idles, and every warp you add is put to full use. At the threshold the idling
          stops. Past it, extra warps change nothing you can measure at the unit: their data comes back and they wait
          for a turn. And the threshold itself is set by the ratio of waiting to working, which belongs to the kernel.
          The last preset puts the book's example SM into the model: 64 warp slots, four processing blocks. There it
          stays busy as long as a warp waits no more than 15 cycles for each cycle it works, which is the book's 16 to
          1 seen from the kernel's side.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Có một giới hạn mô hình bỏ qua lại quan trọng với những kernel bạn quan tâm. Nhiều warp đang chờ hơn nghĩa là
            nhiều yêu cầu đang bay hơn, và bộ nhớ chỉ phục vụ được một số byte nhất định mỗi giây. Khi băng thông đó đã dùng
            hết, thêm warp sẽ không còn giúp ích dù các đơn vị thực thi có rảnh đến đâu. Chương 5 gọi trần đó là roofline.
          </p>
        }
      >
        <p>
          One limit the model leaves out matters for the kernels you care about. More waiting warps means more requests
          in flight, and memory can serve only so many bytes per second. Once that bandwidth is used up, adding warps
          stops helping however idle the execution units are. Chapter 5 calls that ceiling the roofline.
        </p>
      </Translated>

      <h2><Translated vi="Kernel nào cần một SM đầy">Which kernels need a full SM</Translated></h2>
      <Translated
        vi={
          <p>
            Hãy đưa hai kernel từ một hệ thống suy luận vào mô hình. Khi một mô hình decode từng token một, phần lớn công
            việc là các phép nhân ma trận với vector và tra cứu embedding: nạp một trọng số, nhân và cộng, nạp trọng số tiếp
            theo. Đó là c nhỏ. Một luồng dành gần như toàn bộ thời gian chờ bộ nhớ toàn cục, nên thông lượng phụ thuộc vào việc
            có nhiều warp thường trú, và những kernel như vậy muốn SM đầy hết mức có thể.
          </p>
        }
      >
        <p>
          Put two kernels from an inference stack into the model. When a model decodes one token at a time, most of the
          work is matrix-vector products and embedding lookups: load a weight, multiply and add, load the next. That is
          a small c. A thread spends almost all of its time waiting on global memory, so throughput depends on having
          many warps resident, and such kernels want the SM as full as it will go.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Một phép nhân ma trận được viết để giữ các ô làm việc trong thanh ghi là trường hợp ngược lại. Mỗi luồng nạp một
            dải đầu vào rồi tính trên thanh ghi trong thời gian dài trước khi lại cần đến bộ nhớ. Đó là c lớn, và vài warp đủ
            giữ các đơn vị luôn bận. Chương 15 dựng một kernel như vậy và cố ý chạy nó với một khối 256 luồng mỗi SM, tức
            12.5% trong số 2048 ô luồng mà nó giả định.
          </p>
        }
      >
        <p>
          A matrix multiplication written to keep its working tiles in registers is the opposite case. Each thread
          loads a strip of inputs and then computes on registers for a long time before it needs memory again. That is
          a large c, and a few warps keep the units busy. Chapter 15 builds such a kernel and runs it on purpose with
          one block of 256 threads per SM, which is 12.5% of the 2048 thread slots it assumes.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Việc ghép hai kernel này với mô hình là của chúng tôi. Con số 12.5% là của sách (ch. 15, p. 363).</p>}>
          <p>The pairing of these two kernels with the model is ours. The 12.5% is the book's (ch. 15, p. 363).</p>
        </Translated>
        <Translated
          vi={
            <p>
              Hướng dẫn của AMD cũng nêu cùng điểm này về các GEMM dùng matrix core của họ: occupancy thấp, vì vài warp đã
              làm bão hòa các pipeline.
            </p>
          }
        >
          <p>
            AMD's guide makes the same point about its matrix-core GEMMs: low occupancy, because a few warps saturate
            the pipelines.
          </p>
        </Translated>
        <p className="muted"><Translated vi="Đối chiếu ngày 2026-11-20 (ch04-facts.md, B10).">Checked 2026-11-20 (ch04-facts.md, B10).</Translated></p>
      </Beyond>
      <Ask q={questionsS46['ch04/4.6/q-lowocc']} />

      <h2><Translated vi="Nó tốn gì">What it costs</Translated></h2>
      <Translated
        vi={
          <p>
            Giữ warp thường trú là điều làm lần chờ biến mất, và nó được trả bằng hai loại tiền mà SM có số lượng cố định: ô
            để giữ warp và khối, và thanh ghi để giữ trạng thái của chúng. Những thứ đó được chia ra thế nào, và một thay đổi
            nhỏ trong kernel có thể làm số warp vừa chỗ giảm một nửa ra sao, là nội dung{' '}
            <a href={href('ch04', '4.7')}>phần 4.7</a>.
          </p>
        }
      >
        <p>
          Keeping warps resident is what makes the wait disappear, and it is paid for in two currencies the SM has a
          fixed amount of: slots to hold warps and blocks, and registers to hold their state. How those are shared
          out, and how a small change in a kernel can halve the number of warps that fit, is{' '}
          <a href={href('ch04', '4.7')}>section 4.7</a>.
        </p>
      </Translated>
    </>
  );
}
