import { useMemo, useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  DEVICE,
  OPS,
  PRESETS,
  RESOURCES,
  RESOURCE_NAME,
  STOP_LABS,
  canApply,
  evaluate,
  headroom,
  labNum,
  presetById,
  roofFraction,
  run,
  stopPoints,
  stopVerdict,
  type LabPoint,
  type OpId,
  type Resource,
  type StepVerdict,
} from '../../sim/bottleneck';
import { Translated, tr } from '../../lib/i18n';

export const questionsS69: Record<string, Question> = {
  'ch06/6.9/q-nonbottleneck': {
    kind: 'choice',
    id: 'ch06/6.9/q-nonbottleneck',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel bị giới hạn bởi bộ nhớ đã di chuyển dữ liệu ở mức 90% băng thông bạn đo được trong lab 5.1. Ai đó trải các vòng lặp của nó để giảm chi phí lệnh. Bạn kỳ vọng điều gì?
          </>
        }
      >
        <>
          A memory-bound kernel already moves its bytes at 90% of the bandwidth you measured in lab 5.1. Someone unrolls
          its loops to cut instruction overhead. What do you expect?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Thay đổi rất ít hoặc không có: tài nguyên nó giải tỏa không phải điểm nghẽn.">Little or no change: the resource it relieves is not the bottleneck.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Lợi ích lớn: ít nhánh hơn và nhiều lệnh độc lập hơn nghĩa là ít lần chờ hơn.">A large gain: fewer branches and more independent instructions mean fewer stalls.</Translated>,
        why: <Translated vi="Nó loại bỏ nhánh và làm lộ ra các lệnh độc lập. Cả hai đều không phải thứ kernel này đang chờ.">It removes branches and exposes independent instructions. Neither is what this kernel waits on.</Translated>,
      },
      {
        label: <Translated vi="Chậm đi: các vòng lặp đã trải cần nhiều thanh ghi hơn và vì thế mất occupancy.">A slowdown: the unrolled loops need more registers and so lose occupancy.</Translated>,
        why: <Translated vi="Nó có thể tốn thanh ghi, nhưng không có gì ở đây cho thấy kernel thiếu thanh ghi hay thiếu occupancy. Tác động khả dĩ nhất là không có gì.">It can cost registers, but nothing here says the kernel is short of registers or of occupancy. The likely effect is none.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ở mức 90% của mái, kernel chỉ có thể nhanh hơn tối đa 1/0.9 = 1.11 lần, dù bạn làm gì, trừ khi bạn thay đổi số byte nó di chuyển. Trải vòng lặp thay đổi dòng lệnh, mà dòng lệnh không phải thứ giới hạn nó. Đây là nửa đầu của p. 153: giải tỏa một tài nguyên không phải điểm nghẽn thì chẳng được gì.
          </p>
        }
      >
        <p>
          At 90% of the roof the kernel can get at most 1/0.9 = 1.11 times faster, whatever you do, unless you change
          how many bytes it moves. Unrolling changes the instruction stream, and the instruction stream is not what
          limits it. This is the first half of p. 153: relieving a resource that is not the bottleneck gains nothing.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.9, p. 153; roofline from ch. 5 §5.1',
    vi: { source: 'ch. 6 §6.9, p. 153; roofline từ ch. 5 §5.1' },
  },
  'ch06/6.9/q-stop': {
    kind: 'numeric',
    id: 'ch06/6.9/q-stop',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel bị giới hạn bởi bộ nhớ di chuyển 12 GB trong 5 ms. Lab 5.1 đo được 3000 GB/s trên thiết bị đó. Nó đạt bao nhiêu phần trăm của mái? (Số liệu minh họa.)
          </>
        }
      >
        <>
          A memory-bound kernel moves 12 GB in 5 ms. Lab 5.1 measured 3000 GB/s on that device. What percentage of the
          roof does it reach? (Toy numbers.)
        </>
      </Translated>
    ),
    answer: 80,
    tol: 0.01,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            12 GB / 5 ms = 2400 GB/s, và 2400 / 3000 = 80%. Nếu không di chuyển ít byte hơn, kernel này nhiều nhất chỉ nhanh hơn 1 / 0.8 = 1.25 lần, dù thay đổi tiếp theo khéo đến đâu. Có đáng theo đuổi 25% đó không giờ là quyết định bạn có thể đưa ra dựa trên một con số. Chương 5 đã làm cùng phép tính trên thiết bị ví dụ của sách (p. 96).
          </p>
        }
      >
        <p>
          12 GB / 5 ms = 2400 GB/s, and 2400 / 3000 = 80%. Without moving fewer bytes, the best this kernel can do is 1 /
          0.8 = 1.25 times faster, however clever the next change. Whether 25% is worth chasing is now a decision you
          can make with a number. Chapter 5 worked the same arithmetic on the book&rsquo;s example device (p. 96).
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.9, p. 154; ch. 5 §5.1, p. 96', // i18n-ignore: citation only
  },
  'ch06/6.9/q-sol': {
    kind: 'choice',
    id: 'ch06/6.9/q-sol',
    prompt: (
      <Translated
        vi={
          <>
            Sau một thay đổi, phần trăm &ldquo;tốc độ ánh sáng&rdquo; (speed of light) cho bộ nhớ của profiler giảm, và thời gian chạy của kernel cũng giảm. Bạn tin cái nào?
          </>
        }
      >
        <>
          After a change, the profiler&rsquo;s &ldquo;speed of light&rdquo; percentage for memory drops, and the
          kernel&rsquo;s duration also drops. Which do you trust?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Thời gian chạy: thay đổi đó có thể đã loại bớt công việc, làm mức sử dụng cũng giảm theo.">The duration: the change may have removed work, lowering utilisation too.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Phần trăm: đó là con số chính phần cứng đếm về mức bận của bộ nhớ.">The percentage: it is the hardware’s own count of how busy the memory was.</Translated>,
        why: <Translated vi="Nó đo một bộ phận bận đến mức nào, không đo cần bao nhiêu công việc. Di chuyển ít byte hơn với cùng tốc độ sẽ hiện ra thành phần trăm thấp hơn.">It measures how busy a unit was, not how much work was needed. Fewer bytes moved at the same rate shows up as a lower percentage.</Translated>,
      },
      { label: <Translated vi="Không cái nào: hai con số mâu thuẫn nhau, nên phép đo bị hỏng.">Neither: the two contradict each other, so the measurement is broken.</Translated>, why: <Translated vi="Chúng đo những thứ khác nhau và không nhất thiết thay đổi cùng nhau.">They measure different things and need not move together.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phân ô là ví dụ tiêu biểu: nó cắt giảm số byte kernel di chuyển, nên bộ nhớ trông bớt bận, và kernel xong sớm hơn. Hướng dẫn phân loại vấn đề hiện tại của NVIDIA nói thẳng điều này: thời gian chạy mới là sự thật, không phải phần trăm mức sử dụng. Các lab của khóa học báo cáo thời gian đo được cũng vì lý do đó.
          </p>
        }
      >
        <p>
          Tiling is the standard example: it cuts the bytes a kernel moves, so memory looks less busy, and the kernel
          finishes sooner. NVIDIA&rsquo;s current triage guide says it in so many words: duration is the ground truth,
          not the utilisation percentage. The course&rsquo;s labs report measured time for the same reason.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch06-facts.md A17 (Nsight Compute Compute Triage Guide)', // i18n-ignore: citation only
  },
  'ch06/6.9/q-hw': {
    kind: 'choice',
    id: 'ch06/6.9/q-hw',
    prompt: <Translated vi={<>Cùng một kernel bị giới hạn bởi tính toán trên GPU này và bị giới hạn bởi bộ nhớ trên GPU khác. Điều đó có thể xảy ra không?</>}>The same kernel is compute-bound on one GPU and memory-bound on another. Is that possible?</Translated>,
    options: [
      {
        label: <Translated vi="Có. Kernel quyết định cường độ tính toán còn thiết bị quyết định ngưỡng, nên điểm nghẽn, và tối ưu hóa đúng, phải được tìm lại cho từng thiết bị.">Yes. The kernel fixes the intensity and the device fixes the threshold, so the bottleneck, and the right optimisation, have to be found again per device.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Không: cường độ tính toán chỉ là thuộc tính của kernel, nên chế độ của nó như nhau trên mọi thiết bị.">No: intensity is a property of the kernel alone, so its regime is the same on every device.</Translated>, why: <Translated vi="Cường độ thì đúng là vậy. Còn nó nằm trên hay dưới ngưỡng thì phụ thuộc vào tỉ lệ giữa khả năng tính toán và băng thông của thiết bị.">Intensity is. Whether it is above or below the threshold depends on the device’s ratio of compute to bandwidth.</Translated> },
      { label: <Translated vi="Chỉ khi profiler sai trên một trong hai.">Only if the profiler is wrong on one of them.</Translated>, why: <Translated vi="Không có gì sai cả: hai thiết bị có ngưỡng khác nhau.">Nothing is wrong: the two devices have different thresholds.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một kernel ở 5 FLOP/B bị giới hạn bởi bộ nhớ trên một thiết bị có ngưỡng 20 FLOP/B và bị giới hạn bởi tính toán trên một thiết bị có ngưỡng 3 (các ngưỡng giả định). Ngoài roofline, giới hạn occupancy, số bank và kích thước cache đều khác nhau giữa các thiết bị. Sách: điểm nghẽn có thể phụ thuộc phần cứng (p. 154). Hướng dẫn của NVIDIA: tìm lại yếu tố giới hạn chủ đạo trên mỗi thế hệ GPU mới.
          </p>
        }
      >
        <p>
          A kernel at 5 FLOP/B is memory-bound on a device whose threshold is 20 FLOP/B and compute-bound on one whose
          threshold is 3 (made-up thresholds). Beyond the roofline, occupancy limits, bank counts and cache sizes all
          differ between devices. The book: bottlenecks can be hardware-specific (p. 154). NVIDIA&rsquo;s guide: find the
          dominant limiter again on each new GPU generation.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.9, p. 154; ch. 5 §5.1, p. 95; notes/today/ch06-facts.md A17', // i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- the bottleneck loop
const BAR: Record<Resource, string> = {
  global: 'var(--global)',
  shared: 'var(--shared)',
  compute: 'var(--ink-2)',
  barrier: 'var(--ink-3)',
};
const VERDICT_CHIP: Record<StepVerdict, ReactNode> = {
  faster: (
    <span className="chip ok">
      <Translated vi="nhanh hơn">faster</Translated>
    </span>
  ),
  same: (
    <span className="chip">
      <Translated vi="không đổi">unchanged</Translated>
    </span>
  ),
  slower: (
    <span className="chip fail">
      <Translated vi="chậm hơn">slower</Translated>
    </span>
  ),
};
const pct = (x: number) => `${Math.round(x * 100)}%`;
const LIMITER_WORDS = {
  get blocks() {
    return tr('block slots', 'chỗ cho khối');
  },
  get shared() {
    return tr('shared memory', 'bộ nhớ chia sẻ');
  },
  get regs() {
    return tr('registers', 'thanh ghi');
  },
} as const;

function BottleneckLoop() {
  const [presetId, setPresetId] = useState('shared-limited');
  const [ops, setOps] = useState<OpId[]>([]);
  const preset = presetById(presetId);
  const { kernel, steps } = useMemo(() => run(preset.kernel, ops), [preset, ops]);
  const now = evaluate(kernel);
  const start = evaluate(preset.kernel);
  const prev = steps.length ? steps[steps.length - 1].before : null;
  const scale = Math.max(start.total, now.total, ...steps.map((s) => s.after.total), ...(prev ? [prev.total] : [])) * 1.05;
  const choose = (id: string) => {
    setPresetId(id);
    setOps([]);
  };
  return (
    <Bench
      title={tr('Bottleneck loop', 'Vòng lặp điểm nghẽn')}
      task={tr('Pick a starting kernel, apply optimisations from the checklist, and watch which bar sets the time.', 'Chọn một kernel ban đầu, áp dụng các tối ưu hóa từ danh sách kiểm tra, và xem thanh nào quyết định thời gian.')}
      foot={
        <Translated vi={<>Thời gian chạy là thanh dài nhất. Vạch mảnh trên mỗi thanh là độ dài của nó trước bước cuối cùng của bạn. Mọi con số đều đến từ một mô hình minh họa (<code>src/sim/bottleneck.ts</code>) với các hệ số tự đặt ra; chỉ có chiều hướng của chúng là của sách.</>}>
          Run time is the longest bar. The thin mark on each bar is its length before your last step. Every number
          comes from a toy model (<code>src/sim/bottleneck.ts</code>) whose factors are invented; only their
          directions are the book&rsquo;s.
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }} role="radiogroup" aria-label={tr('Starting kernel', 'Kernel ban đầu')}>
        {PRESETS.map((p) => (
          <button key={p.id} className="quiet" aria-pressed={p.id === presetId} data-preset={p.id} onClick={() => choose(p.id)}>
            {p.label}
          </button>
        ))}
      </div>
      <p className="status-line" style={{ margin: '0 0 0.9rem' }}>
        {preset.about}
      </p>
      <div className="cols plot">
        <div>
          <div style={{ display: 'grid', gap: '0.45rem' }} data-bars>
            {RESOURCES.map((r) => {
              const t = now.times[r];
              const isB = r === now.bottleneck;
              return (
                <div key={r} data-resource={r} data-time={t.toFixed(3)} data-bottleneck={isB ? 'yes' : 'no'}>
                  <div className="row" style={{ justifyContent: 'space-between', gap: '0.2rem 0.6rem' }}>
                    <span>
                      {RESOURCE_NAME[r]} {isB ? <b>{tr('· bottleneck', '· điểm nghẽn')}</b> : null}
                    </span>
                    <span className="mono">{fmt(t, 3)}</span>
                  </div>
                  <div
                    style={{
                      position: 'relative',
                      height: '0.95rem',
                      background: 'var(--cell)',
                      border: '1px solid var(--rule)',
                      borderRadius: 2,
                    }}
                  >
                    <div
                      style={{
                        width: `${Math.min(100, (100 * t) / scale)}%`,
                        height: '100%',
                        background: BAR[r],
                        opacity: isB ? 1 : 0.55,
                        transition: 'width 220ms ease',
                      }}
                    />
                    {prev ? (
                      <div
                        title={tr('before your last step', 'trước bước cuối cùng của bạn')}
                        style={{
                          position: 'absolute',
                          top: -3,
                          bottom: -3,
                          width: 2,
                          left: `calc(${Math.min(100, (100 * prev.times[r]) / scale)}% - 1px)`,
                          background: 'var(--ink)',
                        }}
                      />
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>
          <Translated
            vi={
              <p style={{ margin: '0.9rem 0 0' }} data-total={now.total.toFixed(3)}>
                <b>Thời gian chạy {fmt(now.total, 3)}</b> (ban đầu là {fmt(start.total, 3)}). Điểm nghẽn:{' '}
                <b data-bottleneck-label>{RESOURCE_NAME[now.bottleneck].toLowerCase()}</b>.
              </p>
            }
          >
            <p style={{ margin: '0.9rem 0 0' }} data-total={now.total.toFixed(3)}>
              <b>Run time {fmt(now.total, 3)}</b> (started at {fmt(start.total, 3)}). Bottleneck:{' '}
              <b data-bottleneck-label>{RESOURCE_NAME[now.bottleneck].toLowerCase()}</b>.
            </p>
          </Translated>
          <Translated
            vi={
              <p className="status-line" style={{ margin: '0.3rem 0 0' }} data-occupancy>
                {now.blocksPerSM} trên {DEVICE.maxBlocksPerSM} khối mỗi SM, bị giới hạn bởi {LIMITER_WORDS[now.limiter]}: occupancy{' '}
                {pct(now.occupancy)}. Byte đang trên đường truyền: {now.memRate >= 1 ? 'đủ để giữ bus bận' : `${pct(now.memRate)} mức bus cần`}.
                Lưới: {fmt(now.waves, 3)} wave{now.tail > 1.0001 ? `, wave cuối chưa đầy (mọi thời gian × ${fmt(now.tail, 3)})` : ''}.
              </p>
            }
          >
            <p className="status-line" style={{ margin: '0.3rem 0 0' }} data-occupancy>
              {now.blocksPerSM} of {DEVICE.maxBlocksPerSM} blocks per SM, limited by {LIMITER_WORDS[now.limiter]}: occupancy{' '}
              {pct(now.occupancy)}. Bytes in flight: {now.memRate >= 1 ? 'enough to keep the bus busy' : `${pct(now.memRate)} of what the bus needs`}.
              Grid: {fmt(now.waves, 3)} waves{now.tail > 1.0001 ? `, the last one partial (every time × ${fmt(now.tail, 3)})` : ''}.
            </p>
          </Translated>
        </div>
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(8.5rem, auto) 1fr', gap: '0.45rem 0.7rem', alignItems: 'start' }}>
            {OPS.map((o) => {
              const ok = canApply(kernel, o.id);
              return (
                <div key={o.id} style={{ display: 'contents' }}>
                  <button data-op={o.id} disabled={!ok.ok} title={o.effect} onClick={() => setOps((x) => [...x, o.id])}>
                    {o.label}
                  </button>
                  <span className="muted" style={{ fontSize: '0.78rem' }}>
                    {ok.ok ? `${o.effect} (${o.page})` : ok.why}
                  </span>
                </div>
              );
            })}
          </div>
          <div className="row" style={{ marginTop: '0.7rem' }}>
            <button className="quiet" disabled={!ops.length} onClick={() => setOps((x) => x.slice(0, -1))}>
              <Translated vi="Hoàn tác">Undo</Translated>
            </button>
            <button className="quiet" disabled={!ops.length} onClick={() => setOps([])}>
              <Translated vi="Đặt lại">Reset</Translated>
            </button>
          </div>
        </div>
      </div>
      <h4 style={{ margin: '1rem 0 0.4rem', fontFamily: 'var(--sans)', fontSize: '0.9rem' }}><Translated vi="Lịch sử">History</Translated></h4>
      {steps.length ? (
        <ol style={{ margin: 0, paddingLeft: '1.3rem' }} data-history>
          {steps.map((s, i) => (
            <li key={i} data-step={s.op} data-verdict={s.verdict} style={{ marginBottom: '0.25rem' }}>
              {OPS.find((o) => o.id === s.op)!.label}: {fmt(s.before.total, 3)} → {fmt(s.after.total, 3)}{' '}
              {VERDICT_CHIP[s.verdict]}
              <Translated
                vi={
                  <>
                    . Điểm nghẽn {RESOURCE_NAME[s.before.bottleneck].toLowerCase()}
                    {s.after.bottleneck !== s.before.bottleneck ? (
                      <> → <b>{RESOURCE_NAME[s.after.bottleneck].toLowerCase()}</b></>
                    ) : null}
                    ; occupancy {pct(s.before.occupancy)}
                    {s.after.occupancy !== s.before.occupancy ? ` → ${pct(s.after.occupancy)}` : ''}.
                  </>
                }
              >
                . Bottleneck {RESOURCE_NAME[s.before.bottleneck].toLowerCase()}
                {s.after.bottleneck !== s.before.bottleneck ? (
                  <> → <b>{RESOURCE_NAME[s.after.bottleneck].toLowerCase()}</b></>
                ) : null}
                ; occupancy {pct(s.before.occupancy)}
                {s.after.occupancy !== s.before.occupancy ? ` → ${pct(s.after.occupancy)}` : ''}.
              </Translated>
              {s.relievedNonBottleneck ? <span className="muted">{tr(' It relieved a resource that was not the bottleneck.', ' Nó đã giải tỏa một tài nguyên không phải điểm nghẽn.')}</span> : null}
            </li>
          ))}
        </ol>
      ) : (
        <p className="status-line" style={{ margin: 0 }}>
          <Translated vi="Chưa có bước nào.">No steps yet.</Translated>
        </p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the stopping rule on your measurements
interface Roof {
  key: string;
  label: string;
  peak: number;
  bw: number;
  when: string;
}
/** GPU backends only: realBackends() leaves out 'emu' but not the race-detection emulator. */
const gpuBackends = (r: Reports) => realBackends(r).filter((b) => b !== 'race');
function measuredRoofs(reports: Reports): Roof[] {
  return gpuBackends(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const peak = num(r?.compute_gflops);
    const bw = num(r?.bandwidth_gb_per_s);
    if (!peak || !bw) return [];
    return [{ key: b, label: `${String(reports[b].device.name ?? b)} (${b})`, peak, bw, when: reports[b].generated.slice(0, 10) }];
  });
}
/** The benchmark rows of the stopping-rule labs on one backend: the reader's work when it has rows, else the reference. */
function labRows(reports: Reports, backend: string) {
  const runOn = reports[backend];
  if (!runOn) return [];
  const labs = [...new Set(runOn.results.map((r) => r.lab))].filter((l) => STOP_LABS.some((s) => l.startsWith(s.prefix)));
  return labs.flatMap((lab) => {
    const work = benchRows(runOn, lab, 'work');
    const variant = work.length ? 'work' : 'solution';
    const rows = work.length ? work : benchRows(runOn, lab, 'solution');
    return rows.map((row) => ({ lab, row: row as Record<string, unknown>, variant }));
  });
}
const BOOK_ROOF = { peak: 66900, bw: 3350 };

function StopRule() {
  const { reports, loading, reload } = useReports();
  const roofs = useMemo(() => measuredRoofs(reports), [reports]);
  const [key, setKey] = useState<string | null>(null);
  const [thr, setThr] = useState(80);
  const roof = roofs.find((r) => r.key === key) ?? roofs[0];
  const rows = roof ? labRows(reports, roof.key) : [];
  const points: (LabPoint & { variant: string })[] = rows.flatMap((r) =>
    stopPoints([{ lab: r.lab, row: r.row }]).map((p) => ({ ...p, variant: r.variant })),
  );
  const have = new Set(points.map((p) => p.lab));
  const missing = STOP_LABS.filter((s) => !have.has(labNum(s.prefix.replace(/_$/, ''))));
  // A kernel of your own, against your roof or the book's example.
  const [mine, setMine] = useState({ gflop: 2, gb: 8, ms: 4, against: 'book' as 'book' | 'yours' });
  const mineRoof = mine.against === 'yours' && roof ? roof : { ...BOOK_ROOF };
  const minePt = roofFraction(mine.gflop * 1e9, mine.gb * 1e9, mine.ms / 1000, mineRoof.peak, mineRoof.bw);
  const verdictText = (f: number) => {
    const v = stopVerdict(f, thr / 100);
    return v === 'stop' ? (
      <span className="chip ok">
        <Translated vi="gần mái: dừng">close to the roof: stop</Translated>
      </span>
    ) : v === 'room' ? (
      <span>
        <Translated vi={<>còn khoảng trống: tối đa ×{fmt(headroom(f), 3)}</>}>room left: at most ×{fmt(headroom(f), 3)}</Translated>
      </span>
    ) : (
      <span>
        <Translated vi={<><b>vượt trên mái của mô hình</b>: số byte được đếm quá cao</>}>
          <b>above the model&rsquo;s roof</b>: the byte count is too high
        </Translated>
      </span>
    );
  };
  return (
    <Bench
      title={tr('Stopping rule on your measurements', 'Quy tắc dừng trên số đo của bạn')}
      task={tr('Move the threshold. Which of your kernels are worth another optimisation?', 'Di chuyển ngưỡng. Kernel nào của bạn đáng được tối ưu hóa thêm?')}
      foot={
        <Translated vi={<>Cường độ tính toán là cách đếm của sách: mọi phần tử một khối dùng được nạp một lần mỗi khối, bỏ qua cache (chương 5). Phép cộng vector là 1 FLOP cho mỗi 12 byte. Mái là min(đỉnh, băng thông × cường độ) từ hai phép đo thăm dò của lab 5.1, không phải số liệu của nhà sản xuất. Mọi con số ở đây là số đo từ máy này hoặc phép tính trên số đo đó.</>}>
          Intensity is the book&rsquo;s count: every element a block uses is loaded once per block, caches ignored
          (chapter 5). Vector addition is 1 FLOP per 12 bytes. The roof is min(peak, bandwidth × intensity) from lab
          5.1&rsquo;s two probes, not the vendor&rsquo;s figures. Every number here is a measurement from this machine
          or arithmetic on one.
        </Translated>
      }
    >
      {roof ? (
        <>
          <div className="row" style={{ marginBottom: '0.8rem' }}>
            {roofs.length > 1 ? (
              <label className="field">
                <Translated vi="Thiết bị">Device</Translated>
                <select value={roof.key} onChange={(e) => setKey(e.target.value)}>
                  {roofs.map((r) => (
                    <option key={r.key} value={r.key}>
                      {r.label}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            <label className="field">
              <Translated vi={<>Đủ gần để dừng ở {thr}% của mái</>}>Close enough to stop at {thr}% of the roof</Translated>
              <input type="range" min="50" max="100" step="5" value={thr} onChange={(e) => setThr(+e.target.value)} aria-label={tr('threshold', 'ngưỡng')} />
            </label>
          </div>
          <Translated
            vi={
              <p className="status-line" style={{ margin: '0 0 0.6rem' }} data-roof={roof.key}>
                Mái trên {roof.label}, đo bởi lab 5.1 vào {roof.when}: {fmt(roof.bw, 4)} GB/s, {fmt(roof.peak, 4)} GFLOP/s, ngưỡng {fmt(roof.peak / roof.bw, 3)} FLOP/B. Mỗi lab hiển thị kernel của chính bạn khi bạn đã chạy nó với <code>--bench</code> (đánh dấu &ldquo;của bạn&rdquo;), còn trước đó là lời giải tham chiếu.
              </p>
            }
          >
            <p className="status-line" style={{ margin: '0 0 0.6rem' }} data-roof={roof.key}>
              Roof on {roof.label}, measured by lab 5.1 on {roof.when}: {fmt(roof.bw, 4)} GB/s, {fmt(roof.peak, 4)} GFLOP/s,
              threshold {fmt(roof.peak / roof.bw, 3)} FLOP/B. Each lab shows your own kernel once you have run it with{' '}
              <code>--bench</code> (marked &ldquo;yours&rdquo;), the reference solution until then.
            </p>
          </Translated>
          {points.length ? (
            <div style={{ overflowX: 'auto' }}>
              <table className="data" data-stop-table>
                <thead>
                  <tr>
                    <th><Translated vi="Lab">Lab</Translated></th>
                    <th><Translated vi="Kernel">Kernel</Translated></th>
                    <th className="num"><Translated vi="% của mái">% of roof</Translated></th>
                    <th><Translated vi="Kết luận">Verdict</Translated></th>
                    <th className="num">FLOP/B</th>
                    <th className="num">GFLOP/s</th>
                    <th className="num"><Translated vi="Mái tại đó">Roof there</Translated></th>
                  </tr>
                </thead>
                <tbody>
                  {points.map((p, i) => {
                    const r = roofFraction(p.flop, p.bytes, p.seconds, roof.peak, roof.bw);
                    return (
                      <tr key={i} data-point={p.lab} data-fraction={r.fraction.toFixed(4)} data-verdict={stopVerdict(r.fraction, thr / 100)}>
                        <td className="nowrap">
                          <a href={href('ch' + p.lab.split('.')[0].padStart(2, '0'), 'labs')}>{p.lab}</a>
                        </td>
                        <td>
                          {p.label}
                          {p.variant === 'work' ? <b>{tr(' (yours)', ' (của bạn)')}</b> : null}
                        </td>
                        <td className="num">{fmt(100 * r.fraction, 3)}%</td>
                        <td>{verdictText(r.fraction)}</td>
                        <td className="num">{fmt(r.intensity, 3)}</td>
                        <td className="num">{fmt(r.achieved, 4)}</td>
                        <td className="num">{fmt(r.roof, 4)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p data-stop="none">
              <Translated vi="Chưa có số đo nào trên máy này cho các kernel bên dưới.">No measurement on this machine yet for the kernels below.</Translated>
            </p>
          )}
          {points.some((p) => stopVerdict(roofFraction(p.flop, p.bytes, p.seconds, roof.peak, roof.bw).fraction, 1) === 'above') ? (
            <Translated
              vi={
                <p className="status-line" data-above-note>
                  Một điểm nằm trên mái phá vỡ quy tắc của sách rằng các điểm đo nằm dưới cả hai đường (ch. 5, pp. 95–96). Kernel không vượt được phần cứng: mô hình đã đếm cả những byte do cache phục vụ, nên cường độ thật cao hơn con số đếm được. Số byte DRAM đo bởi profiler sẽ đặt nó đúng chỗ; cho đến lúc đó, hãy coi kết luận của nó là chưa biết.
                </p>
              }
            >
              <p className="status-line" data-above-note>
                A point above the roof breaks the book&rsquo;s rule that measured points lie under both lines (ch. 5,
                pp. 95–96). The kernel did not beat the hardware: the model counted bytes that the caches served, so
                the true intensity is higher than the count. A profiler&rsquo;s measured DRAM bytes would place it
                properly; until then, read its verdict as unknown.
              </p>
            </Translated>
          ) : null}
          {missing.length ? (
            <ul className="status-line" data-missing style={{ paddingLeft: '1.2rem' }}>
              {missing.map((s) => {
                const n = labNum(s.prefix.replace(/_$/, ''));
                return (
                  <li key={s.prefix}>
                    <Translated vi={<>Lab {n} ({s.why}): chưa có số đo nào trên máy này. Chạy <code>python3 labs/pmpp.py run {n} --bench</code> khi kernel của bạn đã qua (thêm <code>--solution</code> cho lời giải tham chiếu).</>}>
                      Lab {n} ({s.why}): no measurement on this machine yet. Run <code>python3 labs/pmpp.py run {n} --bench</code>{' '}
                      once your kernel passes (add <code>--solution</code> for the reference).
                    </Translated>
                  </li>
                );
              })}
            </ul>
          ) : null}
          <button className="quiet" onClick={reload}>
            <Translated vi="Tải lại kết quả">Reload results</Translated>
          </button>
        </>
      ) : loading ? (
        <p data-stop="loading">
          <Translated vi="Đang đọc báo cáo.">Reading reports.</Translated>
        </p>
      ) : (
        <p data-stop="none">
          <Translated vi={<>Chưa có số đo nào trên máy này: mái lấy từ lab 5.1. Chạy <code>python3 labs/pmpp.py run 5.1 --solution --bench</code>, rồi các lab của chương 6 với <code>--bench</code>.{' '}</>}>
            No measurement on this machine yet: the roof comes from lab 5.1. Run{' '}
            <code>python3 labs/pmpp.py run 5.1 --solution --bench</code>, then the chapter 6 labs with <code>--bench</code>.{' '}
          </Translated>
          <button className="quiet" onClick={reload}>
            <Translated vi="Tải lại kết quả">Reload results</Translated>
          </button>
        </p>
      )}

      <h4 style={{ margin: '1.2rem 0 0.4rem', fontFamily: 'var(--sans)', fontSize: '0.9rem' }}><Translated vi="Một kernel của riêng bạn">A kernel of your own</Translated></h4>
      <div className="row" data-mine>
        <label className="field">
          <Translated vi="Số GFLOP đã làm">GFLOP done</Translated>
          <input type="number" min="0" step="any" value={mine.gflop} onChange={(e) => setMine({ ...mine, gflop: Math.max(1e-6, +e.target.value || 0) })} />
        </label>
        <label className="field">
          <Translated vi="Số GB đã di chuyển">GB moved</Translated>
          <input type="number" min="0" step="any" value={mine.gb} onChange={(e) => setMine({ ...mine, gb: Math.max(1e-6, +e.target.value || 0) })} />
        </label>
        <label className="field">
          <Translated vi="Thời gian (ms)">Time (ms)</Translated>
          <input type="number" min="0" step="any" value={mine.ms} onChange={(e) => setMine({ ...mine, ms: Math.max(1e-6, +e.target.value || 0) })} />
        </label>
        <label className="field">
          <Translated vi="So với">Against</Translated>
          <select value={mine.against} onChange={(e) => setMine({ ...mine, against: e.target.value as 'book' | 'yours' })}>
            <option value="book"><Translated vi="ví dụ của sách (H100, p. 94)">the book&rsquo;s example (H100, p. 94)</Translated></option>
            {roof ? <option value="yours">{tr('your device (lab 5.1)', 'thiết bị của bạn (lab 5.1)')}</option> : null}
          </select>
        </label>
      </div>
      <p style={{ margin: '0.6rem 0 0' }} data-mine-result={minePt.fraction.toFixed(4)}>
        <Translated vi={<>{fmt(minePt.intensity, 3)} FLOP/B, bị giới hạn bởi {minePt.regime === 'memory' ? 'bộ nhớ' : 'tính toán'} ở đó. Đạt {fmt(minePt.achieved, 4)} GFLOP/s so với mái {fmt(minePt.roof, 4)}: <b>{fmt(100 * minePt.fraction, 3)}%</b>. Kết luận ở mức {thr}%: </>}>
          {fmt(minePt.intensity, 3)} FLOP/B, {minePt.regime}-bound there. Achieved {fmt(minePt.achieved, 4)} GFLOP/s against
          a roof of {fmt(minePt.roof, 4)}: <b>{fmt(100 * minePt.fraction, 3)}%</b>. Verdict at {thr}%:{' '}
        </Translated>
        {verdictText(minePt.fraction)}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function S69() {
  const q = questionsS69;
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi tối ưu hóa trong danh sách kiểm tra đều tiêu thêm một tài nguyên để giải tỏa tài nguyên khác. Phân ô tiêu bộ nhớ chia sẻ để giải tỏa băng thông toàn cục; buffer kép tiêu bộ nhớ chia sẻ để bỏ một rào chắn; gộp việc tiêu tính song song để loại bỏ một chi phí. Hai hệ quả theo sau, và chúng là toàn bộ chiến lược. Nếu tài nguyên được giải tỏa không phải thứ đang giới hạn kernel, chẳng có gì nhanh hơn. Nếu tài nguyên bị tiêu chính là thứ đang giới hạn nó, kernel sẽ chậm đi.
          </p>
        }
      >
        <p>
          Every optimisation in the checklist spends more of one resource to relieve another. Tiling spends shared
          memory to relieve global bandwidth; double buffering spends shared memory to remove a barrier; coarsening
          spends parallelism to remove an overhead. Two consequences follow, and they are the whole of the strategy.
          If the resource relieved is not the one limiting the kernel, nothing gets faster. If the resource spent is
          the one limiting it, the kernel gets slower.
        </p>
      </Translated>
      <Source section="6.9" pages="153" />
      <Translated
        vi={
          <p>
            Ví dụ của sách là phân ô. Khi băng thông toàn cục là giới hạn, đó là nước đi đúng. Khi kernel bị giới hạn bởi occupancy, và occupancy đã bị giới hạn bởi lượng bộ nhớ chia sẻ mỗi khối chiếm, phân ô thêm sẽ làm giảm số khối thường trú và khiến mọi thứ tệ hơn. Mô hình bên dưới cho bạn thử cả hai, cùng năm hàng khác bạn đã gặp trong chương này.
          </p>
        }
      >
        <p>
          The book&rsquo;s example is tiling. When global bandwidth is the limit, it is the right move. When the kernel
          is limited by occupancy, and occupancy is already limited by the shared memory each block takes, more tiling
          removes resident blocks and makes things worse. The model below lets you try both, and the other five rows
          you have met in this chapter.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Của chúng tôi, và chỉ để minh họa: thời gian chạy là thời gian của tài nguyên chậm nhất trong bốn tài nguyên; occupancy đếm các khối nguyên trên một SM giả định; bộ nhớ toàn cục bị giới hạn bởi độ trễ khi số byte đang trên đường truyền thấp hơn một con số giả định. Mỗi tối ưu hóa nhân vài con số với một hệ số tự đặt ra, theo chiều hướng mà chương đưa ra. Nó cho thấy hình dạng của lập luận. Nó không dự đoán gì về GPU của bạn.
            </p>
          }
        >
          <p>
            Ours, and illustrative: run time is the time of the slowest of four resources; occupancy counts whole blocks
            on a made-up SM; global memory is latency-bound below a made-up number of bytes in flight. Each
            optimisation multiplies a few numbers by an invented factor, in the direction the chapter gives. It shows
            the shape of the reasoning. It predicts nothing about your GPU.
          </p>
        </Translated>
      </Simplification>
      <ul>
        <Translated
          vi={
            <li>
              Từ <b>Tiled, limited by shared memory</b>, áp dụng phân ô. Hoàn tác, rồi thử nạp dạng vector và trải vòng lặp. Mỗi cái đã tiêu tài nguyên nào, và giải tỏa tài nguyên nào?
            </li>
          }
        >
          <li>
            From <b>Tiled, limited by shared memory</b>, apply tiling. Undo, then try vector loads and unrolling. Which
            resource did each spend, and which did it relieve?
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Từ <b>Vector addition</b>, áp dụng nạp dạng vector. Rồi bấm Đặt lại và áp dụng gộp việc ba lần. Giải thích các bước không đổi và bước cuối cùng.
            </li>
          }
        >
          <li>
            From <b>Vector addition</b>, apply vector loads. Then press Reset and apply coarsening three times.
            Explain the unchanged steps and the last one.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              Từ <b>Naive matmul</b>, tiếp tục cho đến khi điểm nghẽn đã dịch chuyển hai lần.
            </li>
          }
        >
          <li>
            From <b>Naive matmul</b>, keep going until the bottleneck has moved twice.
          </li>
        </Translated>
      </ul>
      <BottleneckLoop />
      <Ask q={q['ch06/6.9/q-nonbottleneck']} />

      <h2>
        <Translated vi="Vòng lặp, và công cụ dẫn dắt nó">The loop, and the tool that drives it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Vậy quy trình là một vòng lặp: tìm điểm nghẽn, áp dụng một tối ưu hóa nhắm vào nó, đo, và tìm điểm nghẽn mới, thường nằm ở chỗ khác.
          </p>
        }
      >
        <p>
          So the process is a loop: find the bottleneck, apply one optimisation aimed at it, measure, and find the new
          bottleneck, which is often somewhere else.
        </p>
      </Translated>
      <Source section="6.9" pages="153–154" />
      <Translated
        vi={
          <p>
            Bạn không tìm ra nó bằng cách đọc mã. Bạn tìm ra nó bằng profiler; sách chỉ tới Nsight Compute của NVIDIA. Và câu trả lời phụ thuộc vào thiết bị: cùng một kernel có thể chạm những giới hạn khác nhau trên các GPU khác nhau, nên vòng lặp phải được chạy lại trên từng GPU.
          </p>
        }
      >
        <p>
          You do not find it by reading the code. You find it with a profiler; the book points to NVIDIA&rsquo;s Nsight
          Compute. And the answer depends on the device: the same kernel can hit different limits on different GPUs, so
          the loop has to be run again on each one.
        </p>
      </Translated>
      <Today checked="2026-11-20">{/* i18n-ignore: date */}
        <Translated
          vi={
            <p>
              Tài liệu tham khảo [1] của sách vẫn còn truy cập được, dẫn tới Nsight Compute v2026.4.0. Compute Triage Guide của nó, mới hơn tài liệu sách dẫn, mô tả vòng lặp này gần như từng bước, và thêm ba quy tắc đáng ghi nhớ: thời gian chạy mới là sự thật, không phải phần trăm mức sử dụng; tìm lại yếu tố giới hạn chủ đạo trên mỗi thế hệ GPU mới; dừng khi lợi ích kỳ vọng nhỏ so với độ phức tạp và rủi ro. (ch06-facts A17.)
            </p>
          }
        >
          <p>
            The book&rsquo;s reference [1] still resolves, to Nsight Compute v2026.4.0. Its Compute Triage Guide, newer
            than the book&rsquo;s citation, states the loop almost step for step, and adds three rules worth keeping:
            duration is the ground truth, not utilisation percentages; find the dominant limiter again on each new GPU
            generation; stop when the expected gain is small next to the complexity and risk. (ch06-facts A17.)
          </p>
        </Translated>
      </Today>
      <Ask q={q['ch06/6.9/q-sol']} />
      <Ask q={q['ch06/6.9/q-hw']} />

      <h2>
        <Translated vi="Khi nào dừng">When to stop</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Vòng lặp cần một lối ra, và sách đưa ra nó bằng <a href={href('ch05', '5.1')}>roofline</a> của chương 5. Đếm số phép toán một kernel làm và số byte nó di chuyển; tỉ lệ giữa chúng cho biết trần của nó do băng thông hay do phép tính quyết định. Rồi so sánh mức nó đạt được với trần đó. Gần trần thì chẳng còn bao nhiêu để thu. Còn xa dưới trần thì tối ưu hóa thêm nhiều khả năng có lợi.
          </p>
        }
      >
        <p>
          The loop needs an exit, and the book gives it with chapter 5&rsquo;s{' '}
          <a href={href('ch05', '5.1')}>roofline</a>. Count the operations a kernel does and the bytes it moves; their
          ratio says whether its ceiling is set by bandwidth or by arithmetic. Then compare what it achieves with that
          ceiling. Close to it, there is little left to gain. Well below it, more optimisation is likely to pay.
        </p>
      </Translated>
      <Source section="6.9" pages="154" />
      <Translated
        vi={
          <p>
            Bên dưới, các kernel bạn đã đo đến giờ được đặt cạnh mái mà lab 5.1 đo trên thiết bị của bạn. Một trong số đó là cầu nối tới công việc của chính bạn: lớp tuyến tính ở batch 1 là một bước decode, mà mục 5.1 đã cho thấy phải luôn bị giới hạn bởi bộ nhớ. Phần trăm của nó cho bạn biết kernel decode của bạn còn cách mái bộ nhớ bao xa, và do đó các kỹ thuật của 6.1 đến 6.3 còn gì để đóng góp không.
          </p>
        }
      >
        <p>
          Below, the kernels you have measured so far are placed against the roof lab 5.1 measured on your device. One
          of them is the bridge to your own work: the linear layer at batch 1 is a decode step, which section 5.1 showed
          must stay memory-bound. Its percentage tells you how far your decode kernel is from the memory roof, and so
          whether the techniques of 6.1 to 6.3 have anything left to give.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Việc coi batch 1 là một bước decode là của chúng tôi (ch. 5, câu hỏi q-decode). Lab 6.4 đo thời gian cùng lớp đó với bố cục mà PyTorch và llama.cpp dùng.
            </p>
          }
        >
          <p>
            Reading batch 1 as a decode step is ours (ch. 5, question q-decode). Lab 6.4 times the same layer in the
            layout PyTorch and llama.cpp use.
          </p>
        </Translated>
      </Beyond>
      <StopRule />
      <Ask q={q['ch06/6.9/q-stop']} />

      <h2>
        <Translated vi="Profiler nào, trên backend nào">Which profiler, on which backend</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách nêu một công cụ. Trên ba backend của khóa học này, các công cụ khác nhau, và mức độ trực tiếp mà chúng trả lời hai câu hỏi ở trên cũng khác: điểm nghẽn là gì, và tôi còn cách mái bao xa.
          </p>
        }
      >
        <p>
          The book names one tool. On the three backends of this course the tools differ, and so does how directly
          they answer the two questions above: what is the bottleneck, and how far from the roof am I.
        </p>
      </Translated>
      <div style={{ overflowX: 'auto' }} className="wide">
        <table className="data" data-profilers>
          <thead>
            <tr>
              <th><Translated vi="Backend">Backend</Translated></th>
              <th><Translated vi="Profiler">Profiler</Translated></th>
              <th><Translated vi="Roofline">Roofline</Translated></th>
              <th><Translated vi="Lần nạp không được hợp nhất">Uncoalesced loads</Translated></th>
              <th><Translated vi="Xung đột bank">Bank conflicts</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>CUDA</td>
              <td>Nsight Compute{/* i18n-ignore: product name */}</td>
              <td><Translated vi="Có: roofline tốc độ ánh sáng, với các trần L1, L2 và DRAM riêng">Yes: speed-of-light rooflines, with separate L1, L2 and DRAM ceilings</Translated></td>
              <td><Translated vi="Số sector mỗi yêu cầu cao hơn mức lý tưởng (4 cho các lần nạp 4 byte của cả một warp)">Sectors per request above the ideal (4 for 4-byte loads by a full warp)</Translated></td>
              <td>
                <Translated vi={<><code>l1tex__data_bank_conflicts_pipe_lsu.sum</code>, và một chỉ số xung đột N chiều trung bình</>}>
                  <code>l1tex__data_bank_conflicts_pipe_lsu.sum</code>, and an average N-way conflict metric
                </Translated>
              </td>
            </tr>
            <tr>
              <td>HIP</td>
              <td>
                <code>rocprofv3</code>; ROCm Compute Profiler (<code>rocprof-compute</code>){/* i18n-ignore: product names */}
              </td>
              <td>
                <Translated vi={<><code>rocprof-compute</code> vẽ một roofline theo mặc định, nhưng chỉ liệt kê các GPU Instinct và APU Ryzen AI Max là đã được kiểm thử; trên một card Radeon hãy dùng <code>rocprofv3</code></>}>
                  <code>rocprof-compute</code> draws one by default, but lists only Instinct GPUs and Ryzen AI Max APUs as
                  tested; on a Radeon card expect <code>rocprofv3</code>
                </Translated>
              </td>
              <td><Translated vi="Chưa kiểm tra theo tên">Not checked by name</Translated></td>
              <td><Translated vi="LDS &ldquo;Bank Conflict Rate&rdquo; (tỉ lệ xung đột bank)">LDS &ldquo;Bank Conflict Rate&rdquo;</Translated></td>
            </tr>
            <tr>
              <td>Metal</td>
              <td><Translated vi="Metal debugger của Xcode, Instruments">Xcode&rsquo;s Metal debugger, Instruments</Translated></td>
              <td>
                <Translated vi="Không tìm thấy (việc vắng mặt là unverified); các limiter đọc và ghi bộ nhớ tính theo phần trăm của đỉnh là thứ gần nhất">
                  None found (unverified absence); memory read and write limiters as a percentage of peak are the
                  nearest thing
                </Translated>
              </td>
              <td><Translated vi="Các limiter của cache cấp cuối và MMU (gián tiếp)">Last-level-cache and MMU limiters (indirect)</Translated></td>
              <td><Translated vi="Các limiter của tile memory (gián tiếp)">Tile-memory limiters (indirect)</Translated></td>
            </tr>
          </tbody>
        </table>
      </div>
      <Translated
        vi={
          <p>
            Vì vậy trên Metal, câu hỏi &ldquo;còn cách mái bao xa&rdquo; là việc bạn tự tính, đúng như bảng ở trên làm: từ một thời gian đo được, số byte và số phép toán bạn đếm, và một mái bạn đã đo.
          </p>
        }
      >
        <p>
          On Metal the question &ldquo;how far from the roof&rdquo; is therefore yours to compute, exactly as the panel
          above does: from a measured time, your byte and operation counts, and a roof you measured.
        </p>
      </Translated>
      <Today checked="2026-11-20">{/* i18n-ignore: date */}
        <Translated
          vi={
            <p>
              Từ <code>notes/today/ch06-facts.md</code> A17 (Nsight Compute), B9 và B4 (ROCm Compute Profiler, tài liệu 3.10.0), C5 (tài liệu Xcode), và bảng &ldquo;Where the backends really differ&rdquo; (Các backend thực sự khác nhau ở đâu). Chúng tôi chưa chạy thứ gì ở đây trên thiết bị.
            </p>
          }
        >
          <p>
            From <code>notes/today/ch06-facts.md</code> A17 (Nsight Compute), B9 and B4 (ROCm Compute Profiler, docs
            3.10.0), C5 (Xcode documentation), and the table &ldquo;Where the backends really differ&rdquo;. Nothing
            here was run on a device by us.
          </p>
        </Translated>
      </Today>
    </>
  );
}
