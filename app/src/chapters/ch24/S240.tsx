import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type ChoiceQ, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useNarrow } from '../../lib/narrow';
import { useProgress } from '../../lib/progress';
import { fmt, useReports } from '../../lib/reports';
import { Translated, localized, tr, useLocale } from '../../lib/i18n';
import { href } from '../../lib/router';
import { PLAN, useAllChapters } from '../registry';
import { GOALS, STEPS, goal, lanes, rollup, stepOf, stepsOf, type GoalId, type Step } from '../../sim/booksteps';
import {
  BOOK,
  CORRECTIONS,
  DEFAULTS,
  DEVICES,
  LIMITS,
  LIMIT_ROWS,
  VENDOR,
  auditPrinted,
  built,
  figures,
  fmtInt,
  fmtX,
  marks,
  nvlinkOverHostLink,
  printedCells,
  ratio,
  regime,
  threshold,
  tileFor,
  type FigureSet,
  type LimitId,
  type Options,
  type RowId,
} from '../../sim/hwtable';

const P_HOSTLINK = 'ch24/start/p-hostlink';
const P_KNEE = 'ch24/start/p-knee';

/** True below 600 px of viewport width, where the two wide widgets lay themselves out again. */
const useNarrow600 = () => useNarrow(600);

// ---------------------------------------------------------------- 1. the steps map
// What each goal's set of chapters shows, in one or two sentences (PLAN.md sections 2, 3 and 5).
const GOAL_NOTE: Record<GoalId, string> = {
  get matmul() {
    return tr(
      'All three steps. Step one writes the kernel and tunes it, chapter 15 in step two takes it towards a library-grade one, and chapter 19 in step three shows how a framework lowers other work onto it.',
      'Cả ba bước. Bước một viết kernel và tinh chỉnh nó, chương 15 ở bước hai đưa nó tới mức của một thư viện, và chương 19 ở bước ba cho thấy một framework hạ các việc khác xuống nó thế nào.',
    );
  },
  get attention() {
    return tr(
      'Step two supplies the softmax, which is a reduction. Step three supplies the attention kernels and the batching argument.',
      'Bước hai cung cấp softmax, vốn là một phép rút gọn (reduction). Bước ba cung cấp các kernel attention và lập luận về batching.',
    );
  },
  get sampling() {
    return tr(
      'Every book chapter here is a step-two pattern. The book never assembles them into a sampler; module X1 does.',
      'Mỗi chương của sách ở đây là một mẫu của bước hai. Sách không bao giờ ghép chúng thành một bộ lấy mẫu; module X1 làm việc đó.',
    );
  },
  get quantised() {
    return tr(
      "The book offers two paragraphs on lower-precision formats in appendix A and chapter 15's intensity analysis. The kernels themselves are ours, in module X2.",
      'Sách có hai đoạn về các định dạng độ chính xác thấp hơn ở phụ lục A và phân tích cường độ tính toán của chương 15. Bản thân các kernel là của chúng tôi, trong module X2.',
    );
  },
  get backends() {
    return tr(
      "Appendix C gives the book's memory models, which we set against unified memory, and module X3 reads real HIP and Metal backends. Every lab counts as well: one .cu source serves CUDA and HIP through a header for threads, blocks, shared memory, barriers, legacy atomics and runtime calls, but warp _sync intrinsics need a typed lane mask, and cooperative groups, half-precision headers, CUB functors, matrix-core intrinsics and __dp4a (which HIP lacks) need guards or a substitute; Metal gets its own kernel wherever it can express the lab.",
      'Phụ lục C đưa ra các mô hình bộ nhớ của sách, mà chúng tôi đặt cạnh unified memory, và module X3 đọc các backend HIP và Metal thật. Mọi lab cũng đều tính: một nguồn .cu phục vụ CUDA và HIP qua một header cho luồng, khối, bộ nhớ chia sẻ, rào chắn, atomic kiểu cũ và lời gọi runtime, nhưng các intrinsic warp _sync cần một lane mask có kiểu, còn cooperative groups, header half-precision, functor CUB, intrinsic matrix-core và __dp4a (HIP không có) cần guard hoặc một thứ thay thế; Metal có kernel riêng ở bất cứ đâu nó diễn đạt được lab.',
    );
  },
  get reading() {
    return tr(
      'Chapter 19 shows how a framework hands work to a library. Module X3 reads llama.cpp, MLX and PyTorch.',
      'Chương 19 cho thấy một framework giao việc cho thư viện thế nào. Module X3 đọc llama.cpp, MLX và PyTorch.',
    );
  },
};
const STEP_WORD: Record<string, string> = {
  get 1() {
    return tr('step one', 'bước một');
  },
  get 2() {
    return tr('step two', 'bước hai');
  },
  get 3() {
    return tr('step three', 'bước ba');
  },
  get outside() {
    return tr("outside the book's steps", 'ngoài các bước của sách');
  },
};

function StepsMap() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const chapters = useAllChapters();
  const [goalId, setGoalId] = useState<GoalId>('matmul');
  const g = goal(goalId);
  const lit = new Set(g.chapters);
  const record = useMemo(() => rollup(PLAN, progress.sections, reports), [progress.sections, reports]);
  const builtCh = new Map((chapters ?? []).filter((c) => c.sections.length).map((c) => [c.id, c]));
  const row = (id: string) => PLAN.find((r) => r.id === id)!;
  const nums = (step: Step) => g.chapters.filter((c) => stepOf(c) === step).map((c) => row(c).num);

  return (
    <Bench
      title={tr("The book's three steps", 'Ba bước của sách')}
      task={tr('Pick one of your goals. Which steps does it draw on?', 'Chọn một mục tiêu của bạn. Nó dựa vào những bước nào?')}
      foot={tr(
        "Steps as the book assigns them on pp. 577–578. Which chapters feed a goal comes from this course's plan, not from the book. A chapter either feeds a goal or does not; the map says nothing about how much of it you need. Your record counts sections you marked done and labs whose work/ copy passes on a real backend.",
        'Các bước theo cách sách phân chia ở pp. 577–578. Chương nào phục vụ một mục tiêu là theo kế hoạch của khóa học này, không phải của sách. Một chương hoặc phục vụ một mục tiêu hoặc không; bản đồ không nói bạn cần bao nhiêu phần của nó. Bản ghi của bạn đếm các mục bạn đã đánh dấu hoàn thành và các lab có bản work/ qua trên một backend thật.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="muted"><Translated vi="Mục tiêu">Goal</Translated></span>
        <span className="seg" style={{ flexWrap: 'wrap' }} data-goals>
          {GOALS.map((x) => (
            <button key={x.id} aria-pressed={goalId === x.id} onClick={() => setGoalId(x.id)}>
              {x.label}
            </button>
          ))}
        </span>
      </div>
      {lanes(PLAN).map(({ step, rows }) => {
        const s = STEPS.find((x) => x.step === step)!;
        return (
          <div
            key={String(step)}
            className="cols"
            data-lane={String(step)}
            style={{ gridTemplateColumns: 'minmax(0, 13rem) minmax(0, 1fr)', gap: '0.5rem 1rem', padding: '0.6rem 0', borderTop: '1px solid var(--rule)' }}
          >
            <div>
              <b>{s.name}</b>
              <div className="muted" style={{ fontSize: '0.76rem' }}>
                {s.chapters}
              </div>
              <div className="muted" style={{ fontSize: '0.76rem', marginTop: '0.2rem' }}>
                {s.what}
              </div>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
              {rows.map((r) => {
                const on = lit.has(r.id!);
                const ch = builtCh.get(r.id!);
                const rec = record[r.id!];
                const chip: CSSProperties = {
                  flex: '1 1 8.6rem', // i18n-ignore: CSS value
                  maxWidth: '12.5rem',
                  padding: '0.35rem 0.5rem 0.4rem', // i18n-ignore: CSS value
                  borderRadius: 4,
                  border: `1px solid ${on ? 'var(--ink)' : 'var(--rule)'}`, // i18n-ignore: CSS value
                  boxShadow: on ? 'inset 0 3px 0 var(--ink)' : undefined, // i18n-ignore: CSS value
                  background: on ? 'var(--page)' : 'transparent',
                  color: on ? 'var(--ink)' : 'var(--ink-3)',
                  fontSize: '0.78rem',
                  lineHeight: 1.3,
                };
                return (
                  <div key={r.id} data-chip={r.id} data-lit={on ? 'true' : 'false'} style={chip}>
                    <div>
                      <b>{r.num}</b> {ch ? <a href={href(ch.id, ch.sections[0].id)}>{localized(r.title, r.vi?.title)}</a> : localized(r.title, r.vi?.title)}
                      {r.ours ? <span className="muted"><Translated vi=" (của khóa học này)"> (ours)</Translated></span> : null}
                    </div>
                    {on && r.milestone ? <div style={{ fontWeight: 600, marginTop: '0.15rem' }}>{localized(r.milestone, r.vi?.milestone)}</div> : null}
                    <div style={{ fontSize: '0.7rem', marginTop: '0.15rem', color: on ? 'var(--ink-2)' : 'var(--ink-3)' }} data-record>
                      {ch ? (
                        <Translated
                          vi={
                            <>
                              {rec.sectionsDone} trên {ch.sections.length} mục đã xong
                              {rec.labs ? `, ${rec.labsPassing} trên ${rec.labs} lab qua` : ''}
                            </>
                          }
                        >
                        <>
                          {rec.sectionsDone} of {ch.sections.length} sections done
                          {rec.labs ? `, ${rec.labsPassing} of ${rec.labs} labs passing` : ''}
                        </>
                        </Translated>
                      ) : (
                        tr('planned', 'dự kiến')
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
      <div className="panel" style={{ marginTop: '0.6rem' }} data-readout="goal" aria-live="polite">
        <Translated
          vi={
            <p style={{ margin: '0 0 0.4rem' }}>
              <b>{g.label}</b> dựa vào {g.chapters.length} chương:{' '}
              {stepsOf(goalId)
                .map((s) => `${STEP_WORD[String(s)]} (${nums(s).join(', ')})`)
                .join('; ')}
              {g.everyLab ? '; và mọi lab' : ''}.
            </p>
          }
        >
        <p style={{ margin: '0 0 0.4rem' }}>
          <b>{g.label}</b> draws on {g.chapters.length} chapters:{' '}
          {stepsOf(goalId)
            .map((s) => `${STEP_WORD[String(s)]} (${nums(s).join(', ')})`)
            .join('; ')}
          {g.everyLab ? '; and every lab' : ''}.
        </p>
        </Translated>
        <p style={{ margin: '0 0 0.5rem' }}>{GOAL_NOTE[goalId]}</p>
        <table className="data" style={{ margin: 0 }}>
          <tbody>
            {g.chapters.map((c) => (
              <tr key={c}>
                <td className="nowrap">
                  <b>{row(c).num}</b>
                </td>
                <td>{localized(row(c).title, row(c).vi?.title)}</td>
                <td>{localized(row(c).why, row(c).vi?.why)}</td>
                <td>{localized(row(c).milestone ?? '', row(c).vi?.milestone)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- 2. what limits what
// One line per wrong placement. The right row and its reason live in the model (LIMITS).
const NOT_THIS: Record<string, Partial<Record<LimitId, string>>> = {
  'ch24/start/q-limit-load': {
    get compute() {
      return tr('No arithmetic is done on the way. The bytes only move.', 'Không có phép tính số học nào trên đường đi. Byte chỉ di chuyển.');
    },
    get membw() {
      return tr('Device memory can take the bytes faster than they arrive. Ask which path they arrive on.', 'Bộ nhớ device nhận byte nhanh hơn tốc độ chúng đến. Hãy hỏi chúng đến bằng đường nào.');
    },
    get memcap() {
      return tr('Capacity decides whether the weights fit, not how long the copy takes.', 'Dung lượng quyết định trọng số có vừa không, không quyết định việc sao chép mất bao lâu.');
    },
    get gpulink() {
      return tr('The copy starts in host memory, so it does not travel between GPUs.', 'Việc sao chép bắt đầu ở bộ nhớ host, nên nó không đi giữa các GPU.');
    },
  },
  'ch24/start/q-limit-decode': {
    get compute() {
      return tr('Two operations per weight fetched leave the arithmetic units waiting. Ask what they wait for.', 'Hai phép toán cho mỗi trọng số được nạp khiến các đơn vị số học phải chờ. Hãy hỏi chúng chờ cái gì.');
    },
    get memcap() {
      return tr('Capacity decides whether the model fits. Once it does, something else sets the time per token.', 'Dung lượng quyết định mô hình có vừa không. Một khi đã vừa, thứ khác quyết định thời gian cho mỗi token.');
    },
    get hostlink() {
      return tr('The weights and the KV cache are already on the device. Nothing crosses to the host per weight.', 'Trọng số và KV cache đã nằm sẵn trên device. Không có gì đi sang host cho mỗi trọng số.');
    },
    get gpulink() {
      return tr('One GPU is doing this.', 'Chỉ một GPU làm việc này.');
    },
  },
  'ch24/start/q-limit-prefill': {
    get membw() {
      return tr('A matrix-matrix product reuses each operand for a whole row or column of outputs, so a well-written kernel does not wait on memory.', 'Một phép nhân ma trận-ma trận tái sử dụng mỗi toán hạng cho cả một hàng hay một cột đầu ra, nên một kernel viết tốt không phải chờ bộ nhớ.');
    },
    get memcap() {
      return tr('Capacity decides whether the prompt fits, not how fast the products run.', 'Dung lượng quyết định prompt có vừa không, không quyết định các phép nhân chạy nhanh đến đâu.');
    },
    get hostlink() {
      return tr("The prompt's tokens cross once. The products then run on the device.", 'Các token của prompt đi qua một lần. Sau đó các phép nhân chạy trên device.');
    },
    get gpulink() {
      return tr('One GPU is doing this.', 'Chỉ một GPU làm việc này.');
    },
  },
  'ch24/start/q-limit-fit': {
    get compute() {
      return tr('That row is a rate. This card asks a yes-or-no question about bytes.', 'Hàng đó là một tốc độ. Thẻ này hỏi một câu có-hoặc-không về byte.');
    },
    get membw() {
      return tr('That row is a rate. This card asks a yes-or-no question about bytes.', 'Hàng đó là một tốc độ. Thẻ này hỏi một câu có-hoặc-không về byte.');
    },
    get hostlink() {
      return tr('A link has a rate. This card asks whether bytes fit.', 'Một đường truyền có tốc độ. Thẻ này hỏi byte có vừa không.');
    },
    get gpulink() {
      return tr('A link has a rate. This card asks whether bytes fit.', 'Một đường truyền có tốc độ. Thẻ này hỏi byte có vừa không.');
    },
  },
  'ch24/start/q-limit-allreduce': {
    get compute() {
      return tr("The arithmetic is one elementwise reduction. The card is about getting each GPU's data to the others.", 'Phép tính số học chỉ là một phép rút gọn theo từng phần tử. Thẻ này nói về việc đưa dữ liệu của mỗi GPU tới các GPU khác.');
    },
    get membw() {
      return tr('Each GPU reads its own memory quickly. The data then has to reach the other GPUs.', 'Mỗi GPU đọc bộ nhớ của chính nó rất nhanh. Sau đó dữ liệu phải tới được các GPU khác.');
    },
    get memcap() {
      return tr('Nothing here asks whether data fits.', 'Không có gì ở đây hỏi dữ liệu có vừa không.');
    },
    get hostlink() {
      return tr('These GPUs are joined to each other directly, and the book calls NCCL topology-aware (p. 564), so their traffic need not cross the link to the host. Without NVLink it would go over PCIe.', 'Các GPU này nối trực tiếp với nhau, và sách gọi NCCL là nhận biết topology (p. 564), nên lưu lượng của chúng không cần đi qua đường truyền tới host. Không có NVLink thì nó sẽ đi qua PCIe.');
    },
  },
};

const limitQuestion = (qid: string): ChoiceQ => {
  const c = LIMITS.find((x) => x.qid === qid)!;
  return {
    kind: 'choice',
    id: qid,
    // Getters: the card, row names, explanation and source come from hwtable getters, so they are
    // read in the selected language when the question renders, not once when this module loads.
    get prompt() {
      return (
        <Translated vi={<>Hàng nào trong bảng cuối cùng của sách giới hạn việc này? {c.card}</>}>
          <>Which row of the book's last table limits this? {c.card}</>
        </Translated>
      );
    },
    get options() {
      return LIMIT_ROWS.map((r) => ({
        label: r.name,
        correct: r.id === c.row,
        get why() {
          return r.id === c.row ? undefined : NOT_THIS[qid][r.id];
        },
      }));
    },
    get explain() {
      return <p>{c.why}</p>;
    },
    get source() {
      return c.source;
    },
  };
};

function LimitsBench() {
  const { progress, answer, recordReview } = useProgress();
  const prior = () =>
    Object.fromEntries(
      LIMITS.map((c) => {
        const v = progress.answers[c.qid]?.value;
        return [c.qid, typeof v === 'number' && LIMIT_ROWS[v] ? LIMIT_ROWS[v].id : ''];
      }),
    ) as Record<string, LimitId | ''>;
  const [picks, setPicks] = useState<Record<string, LimitId | ''>>(prior);
  // The placements as they stood at the last press of Check. Feedback shows only while a row is unchanged since.
  const [checked, setChecked] = useState<Record<string, LimitId | ''> | null>(() =>
    LIMITS.every((c) => progress.answers[c.qid]) ? prior() : null,
  );
  const ready = LIMITS.every((c) => picks[c.qid] !== '');
  const allRight = checked !== null && LIMITS.every((c) => checked[c.qid] === c.row && picks[c.qid] === c.row);
  const check = () => {
    for (const c of LIMITS) {
      const pick = picks[c.qid];
      if (pick === '' || (checked && checked[c.qid] === pick)) continue;
      const correct = pick === c.row;
      if (!progress.answers[c.qid]) recordReview(c.qid, correct); // first answer schedules it
      answer(c.qid, LIMIT_ROWS.findIndex((r) => r.id === pick), correct);
    }
    setChecked(picks);
  };
  return (
    <Bench
      title={tr('What limits what', 'Cái gì giới hạn cái gì')}
      task={tr(
        'Five things you have met. Place each on the row of the table that limits it, then check.',
        'Năm thứ bạn đã gặp. Đặt mỗi thứ vào hàng của bảng giới hạn nó, rồi kiểm tra.',
      )}
      foot={tr(
        'Each card is limited by a different row. Each is labelled with the chapters it draws on, and the feedback gives sections and pages; the placement of each card is how those chapters reason, not something chapter 24 says.',
        'Mỗi thẻ bị giới hạn bởi một hàng khác nhau. Mỗi thẻ ghi các chương mà nó dựa vào, và phần phản hồi cho biết mục và trang; vị trí của mỗi thẻ là cách các chương đó lập luận, không phải điều chương 24 nói.',
      )}
    >
      <div className="panel">
        {LIMITS.map((c, i) => {
          const pick = picks[c.qid];
          const fresh = checked !== null && checked[c.qid] === pick && pick !== '';
          const ok = pick === c.row;
          return (
            <div
              key={c.qid}
              data-card={c.qid}
              data-result={fresh ? (ok ? 'right' : 'wrong') : ''}
              style={{ padding: '0.65rem 0', borderTop: i ? '1px solid var(--rule-soft)' : undefined }}
            >
              <div className="muted" style={{ fontSize: '0.74rem', fontWeight: 600, marginBottom: '0.2rem' }} data-from>
                {c.from}
              </div>
              <div className="row">
                <p style={{ margin: 0, flex: '1 1 18rem', minWidth: 0 }}>{c.card}</p>
                <label className="field" style={{ maxWidth: '100%' }}>
                  <select
                    style={{ maxWidth: '100%' }}
                    aria-label={tr(`Limit for: ${c.card}`, `Giới hạn cho: ${c.card}`)}
                    value={pick}
                    onChange={(e) => setPicks({ ...picks, [c.qid]: e.target.value as LimitId | '' })}
                  >
                    <option value="">{tr('limited by…', 'bị giới hạn bởi…')}</option>
                    {LIMIT_ROWS.map((r) => (
                      <option key={r.id} value={r.id}>
                        {r.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              {fresh ? (
                <p style={{ margin: '0.4rem 0 0' }} role="status">
                  {ok ? (
                    <>
                      <span className="chip ok"><Translated vi="đúng">right</Translated></span> {c.why} <span className="muted"><Translated vi={<>Nguồn: {c.source}.</>}>Source: {c.source}.</Translated></span>
                    </>
                  ) : (
                    <>
                      <span className="chip fail"><Translated vi={<>không phải {LIMIT_ROWS.find((r) => r.id === pick)!.name}</>}>not {LIMIT_ROWS.find((r) => r.id === pick)!.name}</Translated></span>{' '}
                      {NOT_THIS[c.qid][pick as LimitId]}
                    </>
                  )}
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" onClick={check} disabled={!ready}>
          <Translated vi="Kiểm tra">Check</Translated>
        </button>
        <span data-readout="limits" aria-live="polite">
          {checked === null ? (
            <span className="muted">{ready ? tr('All five placed.', 'Đã đặt đủ năm thẻ.') : tr('Place all five first.', 'Hãy đặt đủ năm thẻ trước.')}</span>
          ) : allRight ? (
            <span className="chip ok"><Translated vi="Đúng cả năm.">All five right.</Translated></span>
          ) : (
            <Translated vi={<>{LIMITS.filter((c) => checked[c.qid] === c.row).length} trên {LIMITS.length} đúng ở lần kiểm tra gần nhất.</>}>
            <>
              {LIMITS.filter((c) => checked[c.qid] === c.row).length} of {LIMITS.length} right at the last check.
            </>
            </Translated>
          )}
        </span>
      </div>
      {allRight ? (
        <div className="panel" style={{ marginTop: '0.8rem' }} data-limits-done>
          <Translated
            vi={
              <p style={{ margin: 0 }}>
                Các hàng của bảng chính là năm giới hạn bạn vẫn dùng để lập luận. Một đường suy luận trên một GPU chạm vào bốn
                trong số đó: đường truyền tới host một lần, khi nạp trọng số; dung lượng khi bạn chọn kích thước batch; tính
                toán trong prefill; băng thông cho mỗi token được sinh ra.
              </p>
            }
          >
          <p style={{ margin: 0 }}>
            The table's rows are the five limits you already reason with. An inference path on one GPU touches four of
            them: the host link once, when the weights are loaded; capacity when you size the batch; compute in
            prefill; bandwidth for every generated token.
          </p>
          </Translated>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- 3. check the table
const ROW_NAME: Record<RowId, string> = {
  get fp32() {
    return tr('Compute throughput, FP32', 'Thông lượng tính toán, FP32');
  },
  get tensor() {
    return tr('Compute throughput, tensor-core figure', 'Thông lượng tính toán, số liệu tensor core');
  },
  get membw() {
    return tr('Memory bandwidth', 'Băng thông bộ nhớ');
  },
  get memcap() {
    return tr('Memory capacity', 'Dung lượng bộ nhớ');
  },
  get pcie() {
    return tr('Host link (PCIe x16, each way)', 'Đường truyền tới host (PCIe x16, mỗi chiều)');
  },
  get nvlink() {
    return tr('GPU-to-GPU link (NVLink)', 'Đường truyền GPU-GPU (NVLink)');
  },
};
const ROWS: RowId[] = BOOK.rows.map((r) => r.id);
const lower = (s: string) => (/^[A-Z][a-z]/.test(s) ? s[0].toLowerCase() + s.slice(1) : s);
/** Thresholds: three significant digits, but 137.5 stays 137.5. */
const thrFmt = (v: number) => (v >= 100 ? String(Number(v.toFixed(1))) : fmt(v));
const show = (v: number) => (v >= 1000 ? fmtInt(v) : Number(v.toPrecision(v < 1 ? 4 : 3)).toString());

function Seg<T extends string | number>({ label, value, options, onChange }: { label: string; value: T; options: [T, string][]; onChange: (v: T) => void }) {
  return (
    <label className="field" data-seg={label}>
      {label}
      <span className="seg">
        {options.map(([v, text]) => (
          <button key={String(v)} aria-pressed={value === v} onClick={() => onChange(v)}>
            {text}
          </button>
        ))}
      </span>
    </label>
  );
}

// A printed ratio its own printed figures contradict (only 638x): the one that earns the failure colour.
const CONTRADICTED = new Set(auditPrinted().filter((a) => !a.matches).map((a) => a.row));

function Recomputed() {
  const narrow = useNarrow600();
  const [set, setSet] = useState<FigureSet>('book');
  const [o, setO] = useState<Options>(DEFAULTS.book);
  const [sel, setSel] = useState<RowId>('fp32');
  const pick = (s: FigureSet) => {
    setSet(s);
    setO({ ...o, g80Pcie: DEFAULTS[s].g80Pcie });
  };
  const rated = ROWS.filter((r) => r !== 'nvlink').map((r) => ({ r, x: ratio(r, set, o)! }));
  const order = [...rated].sort((a, b) => b.x - a.x);
  const least = order[order.length - 1];
  const next = order[order.length - 2];
  const LOG_MAX = 4; // the axis runs from 1x to 10,000x
  return (
    <div data-explorer style={{ marginTop: '1rem' }}>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <Seg label={tr('Figures', 'Số liệu')} value={set} onChange={pick} options={[['book', tr('the book, as printed', 'sách, như được in')], ['vendor', tr(`NVIDIA's documents (${VENDOR.boards.g80}, ${VENDOR.boards.b200})`, `tài liệu của NVIDIA (${VENDOR.boards.g80}, ${VENDOR.boards.b200})`)]]} />
        <Seg label={tr('Tensor-core figure', 'Số liệu tensor core')} value={o.tensor} onChange={(v) => setO({ ...o, tensor: v })} options={[['sparse', tr('with sparsity', 'có sparsity')], ['dense', tr('dense', 'dense (không sparsity)')]]} />
        <Seg label={tr('PCIe generation of the 2007 board', 'Thế hệ PCIe của bo mạch 2007')} value={o.g80Pcie} onChange={(v) => setO({ ...o, g80Pcie: v })} options={[[1, '1'], [2, '2']]} />
        <Seg label={tr('NVLink counted', 'NVLink được đếm')} value={o.nvlink} onChange={(v) => setO({ ...o, nvlink: v })} options={[['both', tr('both directions', 'cả hai chiều')], ['eachWay', tr('each way', 'mỗi chiều')]]} />
      </div>
      <div className="panel" style={{ overflowX: 'auto' }}>
        {/* Below 600 px the two figure columns move under the row name, so growth, printed ratio and bar stay in view. */}
        <table className="data" style={{ margin: 0, minWidth: narrow ? 0 : '42rem', fontSize: narrow ? '0.8rem' : undefined }} data-recomputed>
          <thead>
            <tr>
              <th><Translated vi="Hàng (chọn một)">Row (select one)</Translated></th>
              {narrow ? null : (
                <>
                  <th className="num">2007</th>
                  <th className="num">2025</th>
                </>
              )}
              <th className="num"><Translated vi="Mức tăng">Growth</Translated></th>
              <th className="num" style={{ paddingRight: narrow ? undefined : '1.2rem' }}><Translated vi="Bản in">Printed</Translated></th>
              <th style={{ width: narrow ? '22%' : '36%', minWidth: narrow ? '3.5rem' : '15rem' }}>
                {narrow ? (
                  <span className="muted" style={{ fontWeight: 450, fontSize: '0.7rem' }}>
                    <Translated vi={<>log, {fmtX(1)} đến {fmtX(10 ** LOG_MAX)}</>}>
                    log, {fmtX(1)} to {fmtX(10 ** LOG_MAX)}
                    </Translated>
                  </span>
                ) : (
                  <div style={{ position: 'relative', height: '1.1em', fontWeight: 450, fontSize: '0.7rem' }} className="muted">
                    {[0, 1, 2, 3, 4].map((e) => (
                      <span key={e} style={{ position: 'absolute', left: `${(e / LOG_MAX) * 100}%`, transform: e === LOG_MAX ? 'translateX(-100%)' : e ? 'translateX(-50%)' : undefined, whiteSpace: 'nowrap' }}>
                        {fmtX(10 ** e)}
                      </span>
                    ))}
                  </div>
                )}
              </th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => {
              const f = figures(r, set, o);
              const x = ratio(r, set, o);
              const printed = BOOK.rows.find((b) => b.id === r)!.ratio;
              const differs = x != null && fmtX(x) !== printed;
              const fromText = f.from == null ? tr('N/A', 'không có') : `${show(f.from)} ${f.unit}`;
              const toText = `${show(f.to)} ${f.unit}`;
              return (
                <tr key={r} data-row={r} style={sel === r ? { background: 'var(--rule-soft)' } : undefined}>
                  <td>
                    <label className="check" style={{ alignItems: 'baseline' }}>
                      <input type="radio" name="ch24-row" checked={sel === r} onChange={() => setSel(r)} />{/* i18n-ignore: form group name */}
                      <span style={sel === r ? { fontWeight: 600 } : undefined}>{ROW_NAME[r]}</span>
                    </label>
                    {narrow ? (
                      <div className="muted" style={{ fontSize: '0.74rem' }}>
                        {fromText} <Translated vi="đến">to</Translated> <span data-to>{toText}</span>
                      </div>
                    ) : null}
                  </td>
                  {narrow ? null : (
                    <>
                      <td className="num nowrap">{fromText}</td>
                      <td className="num nowrap" data-to>
                        {toText}
                      </td>
                    </>
                  )}
                  <td className="num" data-ratio>
                    <b>{x == null ? '' : fmtX(x)}</b>
                  </td>
                  <td
                    className="num nowrap"
                    data-printed
                    data-differs={!differs ? undefined : CONTRADICTED.has(r) ? 'contradicted' : 'figures'}
                    style={{ paddingRight: narrow ? undefined : '1.2rem' }}
                  >
                    {!differs ? (
                      <span className="muted">{printed}</span>
                    ) : CONTRADICTED.has(r) ? (
                      <span className="t-fail" title={tr('Its own printed figures do not give this ratio', 'Chính các số liệu được in của nó không cho ra tỉ lệ này')}>
                        {printed}
                      </span>
                    ) : (
                      <span title={tr('Follows from the printed figures, not from the figures selected', 'Suy ra từ các số liệu được in, không phải từ các số liệu đang chọn')}>
                        <span className="muted">≠ </span>
                        {printed}
                      </span>
                    )}
                  </td>
                  <td>
                    {x == null ? (
                      <span className="muted">{narrow ? tr('no 2007 figure', 'không có số 2007') : tr('no 2007 figure to divide by', 'không có số 2007 để chia')}</span>
                    ) : (
                      <div
                        style={{
                          position: 'relative',
                          height: '0.95rem',
                          marginTop: '0.15rem',
                          backgroundImage: 'linear-gradient(90deg, var(--rule) 1px, transparent 1px)',
                          backgroundSize: `${100 / LOG_MAX}% 100%`,
                          borderRight: '1px solid var(--rule)',
                        }}
                      >
                        <div data-bar style={{ width: `${Math.min(100, (Math.log10(x) / LOG_MAX) * 100)}%`, height: '100%', background: 'var(--ink)', borderRadius: '0 2px 2px 0' }} />
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="cols" style={{ marginTop: '0.8rem' }}>
        <div className="panel" data-readout="order" aria-live="polite">
          <Translated
            vi={
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>Tăng ít nhất: {lower(ROW_NAME[least.r])}, {fmtX(least.x)}.</b> Hàng kế trên là{' '}
                {lower(ROW_NAME[next.r])} với {fmtX(next.x)}, gấp {fmt(next.x / least.x, 2)} lần.
              </p>
            }
          >
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>Least growth: {lower(ROW_NAME[least.r])}, {fmtX(least.x)}.</b> The next row up is{' '}
            {lower(ROW_NAME[next.r])} at {fmtX(next.x)}, {fmt(next.x / least.x, 2)} times as much.
          </p>
          </Translated>
          <Translated
            vi={
              <p style={{ margin: '0 0 0.4rem' }} className="muted">
                Các thanh dùng thang log: mỗi vạch lưới gấp mười lần vạch trước. NVLink không có số 2007; theo một chiều
                nó mang {VENDOR.b200.nvlinkEachWay.value} GB/s, gấp {nvlinkOverHostLink().toFixed(1)} lần một đường truyền
                tới host Gen5 x16.
              </p>
            }
          >
          <p style={{ margin: '0 0 0.4rem' }} className="muted">
            Bars are on a log scale: each gridline is ten times the one before. NVLink has no 2007 figure; in one
            direction it carries {VENDOR.b200.nvlinkEachWay.value} GB/s, {nvlinkOverHostLink().toFixed(1)} times a Gen5
            x16 host link.
          </p>
          </Translated>
          <Translated
            vi={
              <p style={{ margin: 0 }} className="muted">
                “≠” đánh dấu một tỉ lệ được in suy ra từ số liệu của sách nhưng không từ số liệu đang chọn. Màu đỏ đánh dấu
                tỉ lệ mà chính các số liệu được in của nó mâu thuẫn.
              </p>
            }
          >
          <p style={{ margin: 0 }} className="muted">
            “≠” marks a printed ratio that follows from the book's figures but not from the ones selected. Red marks
            the one its own printed figures contradict.
          </p>
          </Translated>
        </div>
        <div className="panel" data-readout="built" aria-live="polite">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b><Translated vi={<>{ROW_NAME[sel]}: các số liệu được dựng thế nào</>}>{ROW_NAME[sel]}: how the figures are built</Translated></b>
          </p>
          {built(sel, set, o).map((b) => (
            <p key={b.when} style={{ margin: '0 0 0.4rem' }}>
              <b>{b.when}.</b> {b.text} <span className="muted"><Translated vi={<>[mục facts {b.fact}]</>}>[facts {b.fact}]</Translated></span>
            </p>
          ))}
        </div>
      </div>
    </div>
  );
}

// Cell padding for the printed table, so its four columns fit a phone without scrolling.
const TIGHT: CSSProperties = { padding: '0.4rem 0.25rem' }; // i18n-ignore: CSS value

const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;

function TableAudit({ onCheck }: { onCheck: () => void }) {
  const { progress } = useProgress();
  const committed = typeof progress.answers[P_HOSTLINK]?.value === 'number';
  const locale = useLocale();
  const cells = useMemo(printedCells, [locale]);
  const byId = useMemo(() => Object.fromEntries(cells.map((c) => [c.id, c])), [cells]);
  const [ticks, setTicks] = useState<string[]>([]);
  // The ticks as they stood at the last press of Check.
  const [checked, setChecked] = useState<string[] | null>(null);
  const toggle = (id: string) => {
    setTicks(ticks.includes(id) ? ticks.filter((t) => t !== id) : [...ticks, id]);
    setChecked(null);
  };
  const wrong = cells.filter((c) => c.kind === 'wrong');
  const found = checked ? wrong.filter((c) => checked.includes(c.id)).length : 0;
  const others = checked ? checked.filter((id) => byId[id].kind !== 'wrong') : [];
  const othersDiffer = others.filter((id) => byId[id].kind === 'differs').length;
  const othersHold = others.length - othersDiffer;
  const listed = checked ? cells.filter((c) => checked.includes(c.id) || c.kind === 'wrong') : [];

  if (!committed)
    return (
      <Bench title={tr('Check the table', 'Kiểm tra bảng')} task={tr('Commit your prediction above first.', 'Hãy chốt dự đoán của bạn ở trên trước.')}>
        <Translated
          vi={
            <p style={{ margin: 0 }} data-gate="table">
              Bảng trả lời câu hỏi bạn vừa được hỏi, nên nó được ẩn cho tới khi bạn chốt một con số.
            </p>
          }
        >
        <p style={{ margin: 0 }} data-gate="table">
          The table answers the question you were just asked, so it stays hidden until you have committed a number.
        </p>
        </Translated>
      </Bench>
    );

  const cell = (id: string, text: string | null) => {
    if (text == null) return <td className="muted" style={TIGHT}><Translated vi="(ô phía trên)">(the cell above)</Translated></td>;
    const c = byId[id];
    if (!c) return <td style={TIGHT}>{text}</td>;
    const ticked = ticks.includes(id);
    // Red: the table contradicts itself. Green: its arithmetic holds and nothing corrects it.
    // Neutral: its arithmetic holds, but NVIDIA's documents give another figure or label.
    const state = !checked ? '' : c.kind === 'wrong' ? 'wrong' : !ticked ? '' : c.kind;
    const wash = { wrong: 'var(--fail-wash)', consistent: 'var(--ok-wash)', differs: 'var(--rule-soft)', '': undefined }[state];
    return (
      <td data-cell={id} data-state={state} style={{ ...TIGHT, background: wash }}>
        <label className="check" style={{ alignItems: 'baseline', gap: '0.3rem' }}>
          <input type="checkbox" checked={ticked} onChange={() => toggle(id)} aria-label={tr(`${text}: the table contradicts itself here`, `${text}: bảng tự mâu thuẫn ở đây`)} />
          <span style={state === 'wrong' ? { color: 'var(--fail)', fontWeight: 600 } : undefined}>{text}</span>
        </label>
      </td>
    );
  };

  return (
    <Bench
      title={tr('Check the table', 'Kiểm tra bảng')}
      task={tr(
        'Tick every cell the table itself contradicts: a ratio its two figures do not give, or a unit that cannot be meant. Then check. A calculator is fair.',
        'Đánh dấu mọi ô mà chính bảng tự mâu thuẫn: một tỉ lệ mà hai số liệu của nó không cho ra, hoặc một đơn vị không thể là ý định. Rồi kiểm tra. Dùng máy tính là hợp lệ.',
      )}
      foot={tr(
        "The cells are the book's, as printed on p. 578; the layout is ours. The book prints the two compute figures in one cell (“80 TFLOPS / 2200 TFLOPS (FP32 / Tensor Core FP32)”, “232x / 638x”); they are on two lines here so you can tick them separately.",
        'Các ô là của sách, như được in ở p. 578; cách trình bày là của chúng tôi. Sách in hai số liệu tính toán trong một ô (“80 TFLOPS / 2200 TFLOPS (FP32 / Tensor Core FP32)”, “232x / 638x”); ở đây chúng nằm trên hai dòng để bạn đánh dấu riêng từng số.',
      )}
    >
      <div className="panel" style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, minWidth: '16rem', fontSize: 'clamp(0.7rem, 2.7vw, 0.84rem)' }} data-printed-table>
          <thead>
            <tr>
              <th />
              {BOOK.columns.map((c) => (
                <th key={c} style={TIGHT}>
                  {/* the heading as printed, allowed to break after its slash on a narrow screen */}
                  {c.split('/').map((part, i) => (
                    <span key={i}>
                      {i ? (
                        <>
                          /<wbr />
                        </>
                      ) : null}
                      {part}
                    </span>
                  ))}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {BOOK.rows.map((r) => (
              <tr key={r.id}>
                <td style={TIGHT}>
                  <b>{r.id === 'tensor' ? '' : r.label}</b>
                </td>
                {cell(`${r.id}.g80`, r.g80)}
                {cell(`${r.id}.b200`, r.b200)}
                {cell(`${r.id}.ratio`, r.ratio)}{/* i18n-ignore: cell id */}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button
          className="primary"
          onClick={() => {
            setChecked(ticks);
            onCheck();
          }}
        >
          <Translated vi="Kiểm tra">Check</Translated>
        </button>
        <button className="quiet" onClick={() => { setTicks([]); setChecked(null); }} disabled={!ticks.length}>
          <Translated vi="Xóa các dấu">Clear ticks</Translated>
        </button>
        <span data-readout="audit" aria-live="polite">
          {checked === null ? (
            <span className="muted">
              <Translated vi={<>{ticks.length} trên {cells.length} ô đã đánh dấu.</>}>
              {ticks.length} of {cells.length} cells ticked.
              </Translated>
            </span>
          ) : (
            <Translated
              vi={
                <>
                  <b>Bảng tự mâu thuẫn ở {wrong.length === 2 ? 'hai' : wrong.length} ô.</b> Bạn tìm được {found} trên{' '}
                  {wrong.length}
                  {others.length
                    ? `, và đã đánh dấu ${others.length} ô khác: ${[
                        othersDiffer ? `${othersDiffer} ô mà tài liệu của NVIDIA sửa lại` : '',
                        othersHold ? `${othersHold} ô kiểm tra là đúng` : '',
                      ]
                        .filter(Boolean)
                        .join(' và ')}`
                    : ''}
                  .
                </>
              }
            >
            <>
              <b>The table contradicts itself in {wrong.length === 2 ? 'two' : wrong.length} cells.</b> You found {found} of{' '}
              {wrong.length}
              {others.length
                ? `, and ticked ${plural(others.length, 'other cell', 'other cells')}: ${[
                    othersDiffer ? `${othersDiffer} that NVIDIA's documents correct` : '',
                    othersHold ? `${othersHold} that ${othersHold === 1 ? 'checks' : 'check'} out` : '',
                  ]
                    .filter(Boolean)
                    .join(' and ')}`
                : ''}
              .
            </>
            </Translated>
          )}
        </span>
      </div>
      {checked ? (
        <>
          <div className="panel" style={{ marginTop: '0.8rem' }} data-verdicts>
            <ul style={{ margin: 0, paddingLeft: 0, listStyle: 'none' }}>
              {listed.map((c) => (
                <li key={c.id} data-verdict={c.id} style={{ marginBottom: '0.45rem' }}>
                  {c.kind === 'wrong' ? (
                    <span className="chip fail">{checked.includes(c.id) ? tr('contradicted', 'mâu thuẫn') : tr('missed', 'bỏ sót')}</span>
                  ) : c.kind === 'differs' ? (
                    <span className="chip" style={{ background: 'var(--rule-soft)' }}>
                      <Translated vi="tài liệu của NVIDIA khác">NVIDIA's documents differ</Translated>
                    </span>
                  ) : (
                    <span className="chip ok"><Translated vi="nhất quán">consistent</Translated></span>
                  )}{' '}
                  <b>{c.text}</b>: <span className="mono">{c.arithmetic}</span>. {c.verdict}
                  {c.fact ? <span className="muted"><Translated vi={<> [mục facts {c.fact}]</>}> [facts {c.fact}]</Translated></span> : null}
                </li>
              ))}
            </ul>
            <Translated
              vi={
                <p className="muted" style={{ margin: '0.3rem 0 0' }}>
                  Nhất quán nghĩa là phép tính của bảng đúng: {auditPrinted().filter((a) => a.matches).length} trên 5 tỉ lệ
                  được in suy ra từ hai số liệu bên cạnh. Mỗi số liệu có đúng với phần cứng hay không là một câu hỏi khác:
                  tài liệu của NVIDIA sửa {Object.keys(CORRECTIONS).length} ô không phải là mâu thuẫn, một trong số đó chỉ ở
                  nhãn. Các nút điều khiển dưới đây tính lại mọi tỉ lệ trên số liệu của NVIDIA.
                </p>
              }
            >
            <p className="muted" style={{ margin: '0.3rem 0 0' }}>
              Consistent means the table's arithmetic holds: {auditPrinted().filter((a) => a.matches).length} of the 5
              printed ratios follow from the two figures beside them. Whether each figure is the right one for the
              hardware is a separate question: NVIDIA's documents correct {Object.keys(CORRECTIONS).length} cells that
              are not contradictions, one of them only in its label. The controls below recompute every ratio on
              NVIDIA's figures.
            </p>
            </Translated>
          </div>
          <Recomputed />
        </>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- 4. where the threshold went
function Strip({ thr, tile, narrow }: { thr: number; tile: number; narrow: boolean }) {
  // Below 600 px the strip is drawn on a narrower canvas with larger type, one label per
  // level, so it scales to the box instead of scrolling inside it.
  const W = narrow ? 400 : 760;
  const fs = narrow ? 13 : 11;
  const pad = { l: 18, r: 18 };
  const level = narrow ? [0, 1, 2, 3, 4, 5] : [2, 1, 0, 2, 1, 0]; // label rows, so neighbours do not collide
  const step = narrow ? 20 : 30;
  const top = narrow ? 48 : 50;
  const axisY = narrow ? top + 5 * step + 24 : 150;
  const H = axisY + 72;
  const xMin = 0.05;
  const xMax = 2000;
  const X = (v: number) =>
    pad.l + ((Math.log10(Math.min(Math.max(v, xMin), xMax)) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * (W - pad.l - pad.r);
  const ms = marks(tile);
  const tx = X(thr);
  const edge = narrow ? 90 : 110;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Kernels on a FLOP per byte axis with the device threshold', 'Các kernel trên trục FLOP trên mỗi byte cùng ngưỡng của thiết bị')} style={{ width: '100%', display: 'block' }} data-strip>
      <rect x={pad.l} y={14} width={W - pad.l - pad.r} height={axisY - 14} fill="var(--page)" />
      <rect x={pad.l} y={14} width={Math.max(0, tx - pad.l)} height={axisY - 14} fill="var(--global-wash)" />
      <rect x={tx} y={14} width={Math.max(0, W - pad.r - tx)} height={axisY - 14} fill="var(--reg-wash)" />
      <text x={pad.l + 6} y={28} fontSize={fs} style={{ fill: 'var(--global)' }}><Translated vi="bị giới hạn bởi bộ nhớ">memory-bound</Translated></text>
      <text x={W - pad.r - 6} y={28} fontSize={fs} textAnchor="end" style={{ fill: 'var(--reg)' }}><Translated vi="bị giới hạn bởi tính toán">compute-bound</Translated></text>
      <line x1={pad.l} x2={W - pad.r} y1={axisY} y2={axisY} stroke="var(--ink)" strokeWidth="1.4" />
      {[0.1, 1, 10, 100, 1000].map((v) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={axisY} y2={axisY + 5} stroke="var(--ink)" />
          <text x={X(v)} y={axisY + 18} fontSize={fs} textAnchor="middle">{v}</text>
        </g>
      ))}
      <text x={W / 2} y={H - 6} fontSize={fs + 0.5} textAnchor="middle">
        {narrow ? tr('FLOP per byte of global memory, log scale', 'FLOP trên mỗi byte bộ nhớ toàn cục, thang log') : tr('operations per byte of global-memory traffic (FLOP/B), log scale', 'số phép toán trên mỗi byte lưu lượng bộ nhớ toàn cục (FLOP/B), thang log')}
      </text>
      {ms.map((m, i) => {
        const cx = X(m.ratio);
        const ly = top + level[i] * step;
        const tone = regime(m.ratio, thr) === 'memory-bound' ? 'var(--global)' : 'var(--reg)';
        const anchor = cx < edge ? 'start' : cx > W - edge ? 'end' : 'middle';
        return (
          <g key={m.id} data-mark={m.id}>
            <line x1={cx} x2={cx} y1={ly + 5} y2={axisY} stroke={tone} strokeWidth={m.tiled ? 1.8 : 1} />
            <circle cx={cx} cy={axisY} r={m.tiled ? 6 : 4.5} fill={m.tiled ? tone : 'var(--page)'} stroke={tone} strokeWidth="1.8" />
            <text x={anchor === 'start' ? cx - 4 : anchor === 'end' ? cx + 4 : cx} y={ly} fontSize={fs} textAnchor={anchor} fontWeight={m.tiled ? 650 : 400} style={{ paintOrder: 'stroke', stroke: 'var(--page)', strokeWidth: 4, strokeLinejoin: 'round' }}>
              {m.label}
            </text>
          </g>
        );
      })}
      <line x1={tx} x2={tx} y1={8} y2={axisY + 30} stroke="var(--ink)" strokeWidth="2" strokeDasharray="5 3" />
      <text x={tx + (tx > W - 150 ? -6 : 6)} y={axisY + 42} fontSize={fs + 0.5} fontWeight="650" textAnchor={tx > W - 150 ? 'end' : 'start'} data-threshold-label>
        <Translated vi={<>ngưỡng {thrFmt(thr)}</>}>threshold {thrFmt(thr)}</Translated>
      </text>
    </svg>
  );
}

function ThresholdBench() {
  const { progress } = useProgress();
  const narrow = useNarrow600();
  const committed = typeof progress.answers[P_KNEE]?.value === 'number';
  const [key, setKey] = useState('book2007');
  const [dense, setDense] = useState(false);
  const [tile, setTile] = useState(15);
  if (!committed)
    return (
      <Bench title={tr('Where the threshold went', 'Ngưỡng đã đi đâu')} task={tr('Commit your prediction above first.', 'Hãy chốt dự đoán của bạn ở trên trước.')}>
        <Translated
          vi={
            <p style={{ margin: 0 }} data-gate="knee">
              Dải này cho thấy con số bạn vừa được hỏi, nên nó được ẩn cho tới khi bạn chốt một con số.
            </p>
          }
        >
        <p style={{ margin: 0 }} data-gate="knee">
          The strip shows the number you were just asked for, so it stays hidden until you have committed one.
        </p>
        </Translated>
      </Bench>
    );
  const offered = DEVICES.filter((d) => d.key !== 'book2025dense');
  const dev = key === 'book2025tc' && dense ? DEVICES.find((d) => d.key === 'book2025dense')! : (offered.find((d) => d.key === key) ?? DEVICES[0]);
  const thr = threshold(dev.peak, dev.bw);
  const base = DEVICES[0];
  const baseThr = threshold(base.peak, base.bw);
  const need = tileFor(thr);
  const ms = marks(tile);
  return (
    <Bench
      title={tr('Where the threshold went', 'Ngưỡng đã đi đâu')}
      task={tr(
        'Start on the 2007 column and find the first T that crosses. Then change the column.',
        'Bắt đầu ở cột 2007 và tìm T đầu tiên vượt qua ngưỡng. Rồi đổi cột.',
      )}
      foot={
        <Translated
          vi={
            <>
              Thang log. Các dấu là tỉ lệ của các kernel từ những chương trước; dấu tô đặc di chuyển theo T. Bên trái đường
              nét đứt, một kernel bị giới hạn bởi bộ nhớ trên các số liệu đã chọn. {dev.note} Các thiết bị khác, kể cả thiết
              bị của bạn, là roofline của{' '}
              <a href={href('ch05', '5.1')}>chương 5, §5.1</a>.
            </>
          }
        >
        <>
          Log scale. Marks are kernel ratios from earlier chapters; the filled one moves with T. Left of the dashed
          line a kernel is memory-bound on the chosen figures. {dev.note} Other devices, your own among them, are the
          roofline of{' '}
          <a href={href('ch05', '5.1')}>chapter 5, §5.1</a>.
        </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Số liệu thiết bị">Device figures</Translated>
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label={tr('Device figures', 'Số liệu thiết bị')} style={{ maxWidth: '100%' }}>
            {offered.map((d) => (
              <option key={d.key} value={d.key}>
                {d.label}
              </option>
            ))}
          </select>
        </label>
        {key === 'book2025tc' ? (
          <label className="check">
            <input type="checkbox" checked={dense} onChange={(e) => setDense(e.target.checked)} />
            <Translated vi="dùng số liệu TF32 dense của NVIDIA">use NVIDIA's dense TF32 figure</Translated>
          </label>
        ) : null}
        <label className="field">
          <Translated vi={<>Độ rộng ô T = {tile}</>}>Tile width T = {tile}</Translated>
          <input type="range" min="1" max="128" value={tile} onChange={(e) => setTile(+e.target.value)} aria-label={tr('Tile width T', 'Độ rộng ô T')} />
        </label>
      </div>
      <Strip thr={thr} tile={tile} narrow={narrow} />
      <div className="cols" style={{ marginTop: '0.8rem' }}>
        <div className="panel" data-readout="knee" aria-live="polite">
          <p style={{ fontSize: '1.4rem', margin: '0 0 0.2rem', fontWeight: 600 }}><Translated vi={<>ngưỡng {thrFmt(thr)} FLOP/B</>}>threshold {thrFmt(thr)} FLOP/B</Translated></p>
          <Translated
            vi={
              <p style={{ margin: '0 0 0.5rem' }} className="muted">
                {fmtInt(dev.peak)} GFLOP/s chia cho {show(dev.bw)} GB/s
                {dev.key !== 'book2007' ? <>, gấp {fmt(thr / baseThr, 2)} lần mức {thrFmt(baseThr)} của cột 2007</> : null}.
              </p>
            }
          >
          <p style={{ margin: '0 0 0.5rem' }} className="muted">
            {fmtInt(dev.peak)} GFLOP/s over {show(dev.bw)} GB/s
            {dev.key !== 'book2007' ? <>, {fmt(thr / baseThr, 2)} times the 2007 column's {thrFmt(baseThr)}</> : null}.
          </p>
          </Translated>
          <Translated
            vi={
              <p style={{ margin: '0 0 0.4rem' }}>
                T nhỏ nhất có 0.25 × T bằng hoặc vượt ngưỡng: <b data-need>{fmtInt(need)}</b>
                {need > 128 ? ', vượt xa thanh trượt này' : ''}.
              </p>
            }
          >
          <p style={{ margin: '0 0 0.4rem' }}>
            Smallest T with 0.25 × T at or above it: <b data-need>{fmtInt(need)}</b>
            {need > 128 ? ', far off this slider' : ''}.
          </p>
          </Translated>
          <p style={{ margin: 0 }}>
            <Translated vi={<>Với T = {tile}, kernel phân ô đạt {fmt(0.25 * tile)} FLOP/B:</>}>
            At T = {tile} the tiled kernel reaches {fmt(0.25 * tile)} FLOP/B:
            </Translated>{' '}
            {regime(0.25 * tile, thr) === 'memory-bound' ? <span className="chip global"><Translated vi="bị giới hạn bởi bộ nhớ">memory-bound</Translated></span> : <span className="chip reg"><Translated vi="bị giới hạn bởi tính toán">compute-bound</Translated></span>}
          </p>
        </div>
        <div className="panel" style={{ overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0 }} data-marks>
            <thead>
              <tr>
                <th><Translated vi="Kernel">Kernel</Translated></th>
                <th className="num">FLOP/B</th>
                <th><Translated vi="Trên các số liệu này">On these figures</Translated></th>
                <th><Translated vi="Nguồn">From</Translated></th>
              </tr>
            </thead>
            <tbody>
              {[...ms].sort((a, b) => a.ratio - b.ratio).map((m) => (
                <tr key={m.id} data-mark-row={m.id}>
                  <td>{m.label}</td>
                  <td className="num">{fmt(m.ratio)}</td>
                  <td>{regime(m.ratio, thr) === 'memory-bound' ? <span className="chip global"><Translated vi="bị giới hạn bởi bộ nhớ">memory-bound</Translated></span> : <span className="chip reg"><Translated vi="bị giới hạn bởi tính toán">compute-bound</Translated></span>}</td>
                  <td className="muted">{m.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const BOOK_2007 = DEVICES.find((d) => d.key === 'book2007')!;
const BOOK_2025 = DEVICES.find((d) => d.key === 'book2025')!;
const BOOK_TC = DEVICES.find((d) => d.key === 'book2025tc')!;
const THR_2007 = threshold(BOOK_2007.peak, BOOK_2007.bw);
const THR_2025 = threshold(BOOK_2025.peak, BOOK_2025.bw);
const THR_TC = threshold(BOOK_TC.peak, BOOK_TC.bw);
const THR_DENSE = threshold(DEVICES.find((d) => d.key === 'book2025dense')!.peak, BOOK_TC.bw);
const TENSOR_AUDIT = auditPrinted().find((a) => a.row === 'tensor')!;
const TENSOR_ROW = BOOK.rows.find((r) => r.id === 'tensor')!;
/** The 2007 figure that would make the printed 638x right against the printed 2200. */
const FROM_FOR_638 = TENSOR_ROW.to / TENSOR_AUDIT.printed;
/** Shared memory, in MB, for one T x T tile of M and one of N as 4-byte floats. */
export const tileBytesMB = (t: number) => (2 * t * t * 4) / 1e6;
const T_TC = tileFor(THR_TC);
const T_DENSE = tileFor(THR_DENSE);
const T_2007 = tileFor(THR_2007);

export const questions240: Record<string, Question> = {
  'ch24/start/q-compose': {
    kind: 'choice',
    id: 'ch24/start/q-compose',
    prompt: (
      <Translated
        vi={
          <>
            Lấy mẫu min-p giữ mọi token có xác suất ít nhất bằng một tỉ lệ cố định của xác suất của token có khả năng
            cao nhất, rồi lấy mẫu trong số đó. Bản đồ làm sáng năm chương về mẫu cho việc lấy mẫu. Bản thân ngưỡng cắt
            min-p cần chương nào?
          </>
        }
      >
        Min-p sampling keeps every token whose probability is at least a fixed fraction of the most likely token's,
        then samples among them. The map lights five pattern chapters for sampling. Which does the min-p cut-off
        itself need?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Sắp xếp (chương 14) hàng đó, rồi scan (chương 11) các xác suất đã sắp xếp để tìm chỗ ngưỡng cắt rơi vào, như top-p làm.">
            {'A sort (chapter 14) of the row, then a scan (chapter 11) of the sorted probabilities to find where the cut-off falls, as top-p does.'}
          </Translated>
        ),
        why: (
          <Translated vi="Top-p cần chúng vì ngưỡng cắt của nó là một tổng tích lũy trên các xác suất theo thứ tự đã sắp xếp. Ngưỡng cắt của min-p phụ thuộc vào một con số, xác suất lớn nhất, nên không cần thứ tự.">
            {"Top-p needs them because its cut-off is a running sum over the probabilities in sorted order. Min-p's cut-off depends on one number, the largest probability, so no order is needed."}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một phép rút gọn (reduction, chương 10) để tìm xác suất lớn nhất, rồi lọc (chương 12) để gom các token còn lại.">
            {'A reduction (chapter 10) for the largest probability, then a filter (chapter 12) to gather the survivors.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không cần gì ngoài bước một: mỗi luồng so xác suất của chính nó với ngưỡng.">
            {'Nothing beyond step one: each thread compares its own probability with the threshold.'}
          </Translated>
        ),
        why: (
          <Translated vi="Ngưỡng là một tỉ lệ của xác suất lớn nhất trong hàng, và không luồng nào tự mình biết giá trị đó. Tìm nó là một phép rút gọn.">
            {'The threshold is a fraction of the largest probability in the row, and no thread knows that alone. Finding it is a reduction.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một histogram (chương 9) của các xác suất, rồi scan (chương 11) số đếm của các bin để tìm bin chứa ngưỡng cắt.">
            {'A histogram (chapter 9) of the probabilities, then a scan (chapter 11) of the bin counts to find the cut-off bin.'}
          </Translated>
        ),
        why: (
          <Translated vi="Đó là cách radix select tìm ngưỡng cắt top-k mà không cần sắp xếp toàn bộ (module X1); top-p sẽ cần tổng xác suất của mỗi bin, không phải số đếm của nó. Ngưỡng cắt của min-p chỉ cần một giá trị, giá trị lớn nhất.">
            {'That is how radix select finds a top-k cut-off without a full sort (module X1); top-p would need each bin\'s probability mass, not its count. Min-p\'s cut-off needs one value, the maximum.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
        <p>
          Khi một phép rút gọn max (chương 10) đã có xác suất lớn nhất, phép kiểm tra cho mỗi token là theo từng phần tử,
          và gom các token còn lại mà không để lỗ hổng là một phép lọc (chương 12), có dạng ổn định chạy một scan trên các
          cờ giữ lại. Không cần sắp xếp. Vì mọi xác suất dùng chung một hệ số chuẩn hóa, phép kiểm tra thậm chí có thể chạy
          trên logits: giữ một token nếu logit của nó ít nhất bằng logit lớn nhất cộng với log của tỉ lệ. Min-p không có
          trong sách; module X1 xây dựng các bộ lấy mẫu từ những chương này.
        </p>
        }
      >
      <p>
        Once a max reduction (chapter 10) has the largest probability, each token's test is elementwise, and gathering
        the survivors without holes is a filter (chapter 12), whose stable form runs a scan over keep flags. No sort is
        needed. Because every probability shares one normaliser, the test can even run on logits: keep a token if its
        logit is at least the largest logit plus the log of the fraction. Min-p is not in the book; module X1 builds
        samplers from these chapters.
      </p>
      </Translated>
    ),
    source: 'ch. 24 §24.1, pp. 577–578, for the steps; ch. 10 (pp. 221–250) and ch. 12 (pp. 289–302) for the patterns; min-p is ours, not the book\'s',
    vi: {
      source: 'ch. 24 §24.1, pp. 577–578, cho các bước; ch. 10 (pp. 221–250) và ch. 12 (pp. 289–302) cho các mẫu; min-p là của khóa học này, không phải của sách',
    },
  },
  ...Object.fromEntries(LIMITS.map((c) => [c.qid, limitQuestion(c.qid)])),
  'ch24/start/q-638': {
    kind: 'numeric',
    id: 'ch24/start/q-638',
    prompt: (
      <Translated
        vi={
          <>
            Fig. 24.1 in 0.345 TFLOPS cho 2007, 2200 TFLOPS cho số liệu tensor core 2025, và 638x giữa hai số đó. Giữ
            nguyên 2200, thông lượng năm 2007 bằng bao nhiêu thì 638x mới đúng?
          </>
        }
      >
        Fig. 24.1 prints 0.345 TFLOPS for 2007, 2200 TFLOPS for the 2025 tensor-core figure, and 638x between them.
        Keeping the 2200, what 2007 throughput would make 638x correct?
      </Translated>
    ),
    answer: Number(FROM_FOR_638.toFixed(2)),
    tol: 0.01,
    unit: 'TFLOPS',
    explain: (
      <Translated
        vi={
        <p>
          2200 / 638 = {FROM_FOR_638.toFixed(2)} TFLOPS, gấp mười lần con số 0.345 được in. Chính các hệ số của NVIDIA đặt
          G80 ở mức 128 phép nhân-cộng mỗi chu kỳ × 2 × 1.35 GHz = 345.6 GFLOPS, và NVIDIA in 2.2 petaFLOPS cho hàng 2025. Khi
          cả hai đầu vào đều được xác nhận, tỉ lệ là ô duy nhất còn lại: nó phải là khoảng {fmtX(TENSOR_AUDIT.recomputed)}.
        </p>
        }
      >
      <p>
        2200 / 638 = {FROM_FOR_638.toFixed(2)} TFLOPS, ten times the printed 0.345. NVIDIA's own factors put the G80 at
        128 multiply-adds per clock × 2 × 1.35 GHz = 345.6 GFLOPS, and NVIDIA prints 2.2 petaFLOPS for the 2025 row. With
        both inputs confirmed, the ratio is the only cell left: it should read about {fmtX(TENSOR_AUDIT.recomputed)}.
      </p>
      </Translated>
    ),
    source: 'ch. 24 §24.2, Fig. 24.1, p. 578; notes/today/ch24-facts.md, A2 and A9',
    vi: { source: 'ch. 24 §24.2, Fig. 24.1, p. 578; notes/today/ch24-facts.md, A2 và A9' },
  },
  'ch24/start/q-tile': {
    kind: 'numeric',
    id: 'ch24/start/q-tile',
    prompt: (
      <Translated
        vi={
          <>
            So với số liệu tensor core 2025 của sách, ngưỡng là {fmt(THR_TC)} FLOP/B, mức mà kernel nhân ma trận phân ô của
            chương 5 chỉ đạt tới khi T = {fmtInt(T_TC)}. Kernel đó giữ một ô T × T của M và một ô của N trong bộ nhớ chia sẻ
            (shared memory), dưới dạng float 4 byte. Một khối (block) sẽ cần bao nhiêu megabyte?
          </>
        }
      >
        Against the book's 2025 tensor-core figure the threshold is {fmt(THR_TC)} FLOP/B, which chapter 5's tiled matmul
        reaches only at T = {fmtInt(T_TC)}. That kernel keeps one T × T tile of M and one of N in shared memory, as
        4-byte floats. How many megabytes would one block need?
      </Translated>
    ),
    answer: Number(tileBytesMB(T_TC).toFixed(2)),
    // 9.68 MB, or 9.23 if counted in units of 2^20 bytes: both are right.
    tol: 0.05,
    unit: 'MB',
    explain: (
      <Translated
        vi={
          <>
            <p>
              2 × {fmtInt(T_TC)} × {fmtInt(T_TC)} × 4 byte = {fmt(tileBytesMB(T_TC))} MB mỗi khối. Hãy so với giới hạn mà{' '}
              <code>python3 labs/pmpp.py doctor</code> in ra cho thiết bị của bạn (<code>sharedMemPerBlock</code> trên CUDA
              và HIP). Với số liệu TF32 dense của NVIDIA, T = {T_DENSE} vẫn cần {fmt(tileBytesMB(T_DENSE), 2)} MB; với số liệu
              2007, T = {T_2007} cần {fmt((tileBytesMB(T_2007) * 1e6) / 1024, 2)} KB.
            </p>
            <p>
              Việc tái sử dụng phải đến từ chỗ khác: các ô đầu ra lớn hơn giữ trong thanh ghi (chương 15), batching, hoặc ít
              byte hơn cho mỗi giá trị.
            </p>
          </>
        }
      >
      <>
        <p>
          2 × {fmtInt(T_TC)} × {fmtInt(T_TC)} × 4 bytes = {fmt(tileBytesMB(T_TC))} MB per block. Compare that with the
          limit <code>python3 labs/pmpp.py doctor</code> prints for your device (<code>sharedMemPerBlock</code> on CUDA
          and HIP). On NVIDIA's dense TF32 figure T = {T_DENSE} still needs {fmt(tileBytesMB(T_DENSE), 2)} MB; on the
          2007 figures T = {T_2007} needed {fmt((tileBytesMB(T_2007) * 1e6) / 1024, 2)} KB.
        </p>
        <p>
          The reuse has to come from somewhere else: larger output tiles held in registers (chapter 15), batching, or
          fewer bytes per value.
        </p>
      </>
      </Translated>
    ),
    source: 'our analysis with the §5.1 method (ch. 5, pp. 94–97) and the tiled kernel of §5.4 on the figures of Fig. 24.1, p. 578',
    vi: {
      source: 'phân tích của chúng tôi theo phương pháp §5.1 (ch. 5, pp. 94–97) và kernel phân ô của §5.4 trên số liệu của Fig. 24.1, p. 578',
    },
  },
  'ch24/start/q-norow': {
    kind: 'choice',
    id: 'ch24/start/q-norow',
    prompt: (
      <Translated
        vi={
          <>
            Một đường suy luận trên một GPU chạm vào bốn hàng của bảng trong sách. Hàng nào trong bốn hàng đó không có đối
            ứng trên Apple silicon, và chi phí nào thế chỗ nó?
          </>
        }
      >
        An inference path on one GPU touches four rows of the book's table. Which of the four has no counterpart on
        Apple silicon, and what cost takes its place?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Băng thông bộ nhớ. CPU và GPU dùng chung một bộ nhớ, nên GPU không có băng thông riêng; tranh chấp với CPU thế chỗ nó.">
            {'Memory bandwidth. CPU and GPU share one memory, so the GPU has no bandwidth of its own; contention with the CPU takes its place.'}
          </Translated>
        ),
        why: (
          <Translated vi="Apple công bố một số liệu băng thông bộ nhớ ở cấp chip, và một kernel batch 1 vẫn bị nó giới hạn. Hàng này tồn tại.">
            {'Apple publishes a chip-level memory bandwidth figure, and a batch-1 kernel is still limited by it. The row exists.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Dung lượng bộ nhớ. GPU không có bộ nhớ riêng để cạn, nên một mô hình kích thước bất kỳ được ánh xạ và nạp trang theo nhu cầu.">
            {'Memory capacity. The GPU has no memory of its own to run out of, so a model of any size is mapped and paged in on demand.'}
          </Translated>
        ),
        why: (
          <Translated vi="Unified memory vẫn có kích thước, và trọng số cùng KV cache vẫn phải vừa trong đó.">
            {'Unified memory still has a size, and the weights and KV cache still have to fit in it.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Đường truyền từ host tới device. Buffer được chia sẻ mà không cần sao chép; chi phí còn lại là chờ một command buffer xong trước khi CPU đọc.">
            {'The host-to-device link. Buffers are shared with no copy; the cost left is waiting for a command buffer before the CPU reads.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Thông lượng tính toán. Apple không công bố số FLOPS nào, nên ngưỡng của mục 5.1 nhường chỗ cho occupancy (mức chiếm dụng) như chi phí cần theo dõi.">
            {'Compute throughput. Apple publishes no FLOPS figure, so the threshold of section 5.1 gives way to occupancy as the cost to watch.'}
          </Translated>
        ),
        why: (
          <Translated vi="Con số công bố thì thiếu, nhưng giới hạn đối với kernel vẫn còn đó.">
            {'The published number is missing, but the limit on a kernel is still there.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
        <p>
          Trên Apple silicon, CPU và GPU dùng chung bộ nhớ hệ thống, nên không có bus nào để đi qua và không có số liệu nào
          để điền vào hàng đó. Nguyên tắc mà hàng đó đại diện đổi hình dạng: giữ kết quả ở phía GPU qua các kernel, vì mỗi
          lần CPU cần chúng nó phải chờ công việc hoàn tất. Sự đối chiếu này là của chúng tôi, không phải của Apple và không
          phải của sách.
        </p>
        }
      >
      <p>
        On Apple silicon the CPU and the GPU share system memory, so there is no bus to cross and no figure to put in
        that row. The discipline the row stands for changes shape: keep results on the GPU side across kernels, because
        each time the CPU needs them it has to wait for the work to finish. The contrast is ours, not Apple's and not
        the book's.
      </p>
      </Translated>
    ),
    source: 'notes/today/ch05-facts.md, C6; notes/today/ch24-facts.md, C2', // i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- the page
const BACKEND_ROWS: { role: string; cuda: ReactNode; hip: ReactNode; metal: ReactNode; facts: string }[] = [
  {
    get role() {
      return tr('Matrix-multiply hardware', 'Phần cứng nhân ma trận');
    },
    cuda: 'Tensor Cores', // i18n-ignore: NVIDIA product name
    get hip() {
      return tr('Matrix cores, formally MFMA units; documented for CDNA parts only', 'Matrix core, tên chính thức là đơn vị MFMA; chỉ được tài liệu hóa cho các chip CDNA');
    },
    get metal() {
      return tr("None named in Metal documentation; Apple's spec page lists “Neural Accelerators” in the GPU, with no link to MSL stated", 'Tài liệu Metal không nêu tên phần cứng nào; trang thông số của Apple liệt kê “Neural Accelerators” trong GPU, không nói gì về liên hệ với MSL');
    },
    facts: 'A9, B1, C1',
  },
  {
    get role() {
      return tr('How kernel code reaches it', 'Mã kernel tiếp cận nó thế nào');
    },
    cuda: (
      <Translated
        vi={
          <>
            CUDA Tile (<code>cuda::tiles</code> trong <code>cuda_tile.h</code>, từ Toolkit 13.3 trở đi), hoặc thư viện. §15.9
            của sách (p. 369) cũng nêu tên các lệnh WMMA và WGMMA.
          </>
        }
      >
      <>
        CUDA Tile (<code>cuda::tiles</code> in <code>cuda_tile.h</code>, Toolkit 13.3 onward), or libraries. The book's
        §15.9 (p. 369) also names the WMMA and WGMMA instructions.
      </>
      </Translated>
    ),
    get hip() {
      return tr('Libraries, compiler intrinsics or inline assembly; no tile-level model found', 'Thư viện, intrinsic của trình biên dịch hoặc inline assembly; không tìm thấy mô hình cấp tile nào');
    },
    metal: (
      <Translated
        vi={
          <>
            Lời gọi thư viện MSL: <code>mpp::tensor_ops::matmul2d</code>, lựa chọn Apple nói là ưu tiên từ Metal 4, hoặc{' '}
            <code>&lt;metal_simdgroup_matrix&gt;</code>
          </>
        }
      >
      <>
        MSL library calls: <code>mpp::tensor_ops::matmul2d</code>, Apple's stated preference since Metal 4, or{' '}
        <code>&lt;metal_simdgroup_matrix&gt;</code>
      </>
      </Translated>
    ),
    facts: 'A19, B1, C1',
  },
  {
    get role() {
      return tr('Host-to-device link', 'Đường truyền từ host tới device');
    },
    cuda: 'PCIe',
    hip: 'PCIe',
    get metal() {
      return tr('None: memory is unified', 'Không có: bộ nhớ là hợp nhất');
    },
    facts: 'A12, B4, C2',
  },
  {
    get role() {
      return tr('GPU-to-GPU link', 'Đường truyền GPU-GPU');
    },
    cuda: 'NVLink',
    get hip() {
      return tr('Infinity Fabric links', 'Các liên kết Infinity Fabric');
    },
    get metal() {
      return tr('None documented for Apple silicon', 'Không có tài liệu nào cho Apple silicon');
    },
    facts: 'A13, B2, C3',
  },
  {
    get role() {
      return tr('Collective library', 'Thư viện collective');
    },
    cuda: 'NCCL',
    hip: (
      <Translated vi={<>RCCL, với cùng tên hàm <code>ncclXxx</code></>}>
      <>
        RCCL, with the same <code>ncclXxx</code> function names
      </>
      </Translated>
    ),
    metal: (
      <Translated vi={<>Metal không có; MLX có <code>mx.distributed</code></>}>
      <>
        None in Metal; MLX has <code>mx.distributed</code>
      </>
      </Translated>
    ),
    facts: 'A16, B3, C4',
  },
];

export function S240() {
  const { progress } = useProgress();
  const committed = typeof progress.answers[P_HOSTLINK]?.value === 'number';
  // What the audit finds is discussed below it, so that text waits for its Check.
  const [audited, setAudited] = useState(false);
  return (
    <>
      <Translated
        vi={
        <p>
          Chương 24 dài bốn trang, không có mã và không có bài tập: nhìn lại các mục tiêu của sách, rồi triển vọng của các
          tác giả. Trang này giữ ba thứ: ba bước như một tuyên bố về những gì bạn giờ đã làm được, bảng duy nhất của chương
          được kiểm tra từng dòng, và một cách đọc bảng đó cho công việc viết kernel.
        </p>
        }
      >
      <p>
        Chapter 24 is four pages with no code and no exercises: the book's goals revisited, then the authors' outlook.
        This page keeps three things: the three steps as a statement of what you can now do, the chapter's one table
        checked line by line, and a reading of that table for kernel work.
      </p>
      </Translated>
      <Source pages="577" />

      <h2><Translated vi="Trước khi bắt đầu">Before you start</Translated></h2>
      <Translated
        vi={
        <p>
          Năm thẻ lấy từ các chương 2, 5, 20 và 23. Bảng cuối cùng của sách có năm hàng, mỗi hàng là một giới hạn bạn vẫn
          dùng để lập luận: đặt mỗi thẻ vào hàng giới hạn nó. Mỗi thẻ sẽ quay lại trong phần{' '}
          <a href={href('review')}>ôn tập</a>.
        </p>
        }
      >
      <p>
        Five cards drawn from chapters 2, 5, 20 and 23. The book's last table has five rows, each a limit you already
        reason with: place each card on the row that limits it. Each card comes back in{' '}
        <a href={href('review')}>review</a>.
      </p>
      </Translated>
      <LimitsBench />

      <h2><Translated vi="Những gì bạn giờ đã làm được">What you can now do</Translated></h2>
      <Translated
        vi={
        <p>
          Sách hứa hai điều: tư duy tính toán, và một cách có hệ thống để tấn công bất cứ thứ gì giới hạn hiệu năng. Bước
          một, chương 2 đến 6: viết một kernel, ánh xạ luồng (thread) sang dữ liệu, lập luận về warp, occupancy (mức chiếm
          dụng), các tầng bộ nhớ và hợp nhất truy cập bộ nhớ (coalescing), và kết thúc bằng danh sách tối ưu hóa (§6.8, pp.
          147–153). Bước hai, chương 7 đến 15: các mẫu, mỗi mẫu là một phương tiện cho những tối ưu hóa đó. Bước ba, chương
          16 đến 23: các cách phân rã thay thế và cấu trúc dữ liệu của chúng, với chương 22 là phần tóm lại cách tư duy. Với
          điều này, các tác giả khẳng định (p. 578), công việc hiệu năng là một quá trình lập luận. Cách nhóm khá lỏng lẻo:
          chương 22 và 23 được tính là “advanced patterns and applications”.
        </p>
        }
      >
      <p>
        The book promised two things: computational thinking, and a systematic way to attack whatever limits
        performance. Step one, chapters 2 to 6: write a kernel, map threads to data, reason about warps, occupancy,
        memory tiers and coalescing, and end with the optimisation checklist (§6.8, pp. 147–153). Step two, chapters 7
        to 15: the patterns, each a vehicle for those optimisations. Step three, chapters 16 to 23: alternative
        decompositions and their data structures, with chapter 22 as the recap of the thinking. With this, the authors
        claim (p. 578), performance work is a reasoning process. The grouping is loose: chapters 22 and 23 count as
        “advanced patterns and applications”.
      </p>
      </Translated>
      <Source section="24.1" pages="577–578" />
      <StepsMap />
      <Ask q={questions240['ch24/start/q-compose']} />

      <h2><Translated vi="Bảng duy nhất của sách">The book's one table</Translated></h2>
      <Translated
        vi={
        <p>
          Mục 24.2 mở đầu bằng một bảng đặt GPU CUDA đầu tiên cạnh một GPU năm 2025, với một cột tỉ lệ. Trước khi xem nó,
          hãy chốt một con số cho một trong các hàng của nó.
        </p>
        }
      >
      <p>
        Section 24.2 opens with a table that sets the first CUDA GPU beside a 2025 one, with a ratio column. Before you
        see it, commit to a number for one of its rows.
      </p>
      </Translated>
      <Source section="24.2" pages="578" figs={tr('Fig. 24.1', 'Hình 24.1')} />
      <Predict
        id={P_HOSTLINK}
        unit="x"
        prompt={
          <Translated
            vi={
              <>
                Từ 2007 đến 2025, bảng của sách cho thông lượng tính toán tăng 232x, băng thông bộ nhớ tăng 93x và dung lượng
                bộ nhớ tăng 128x. Theo bảng, đường truyền tới host (PCIe x16, mỗi chiều) tăng bao nhiêu lần?
              </>
            }
          >
          <>
            From 2007 to 2025 the book's table has compute throughput up 232x, memory bandwidth up 93x and memory
            capacity up 128x. By what factor does it say the host link (PCIe x16, each way) grew?
          </>
          </Translated>
        }
      >
        {(v) => (
          <Translated
            vi={
              <p>
                <b>8x</b> như được in: 8 GB/s lên 64 GB/s.{' '}
                {Math.abs(v - 8) <= 1
                  ? 'Đó đúng là con số bảng in.'
                  : v >= 50
                    ? 'Một phỏng đoán gần 100 giả định đường truyền theo kịp device. Nó không theo kịp.'
                    : v >= 12 && v <= 20
                      ? 'Con số đó gần với những gì chính tài liệu của NVIDIA cho bo mạch 2007.'
                      : v < 8
                        ? 'Bạn đoán thấp hơn bảng, và đúng về chiều hướng: đường truyền bị tụt lại.'
                        : 'Đường truyền tụt lại sau device xa hơn bạn nghĩ.'}{' '}
                Trên số liệu của NVIDIA cho bo mạch 2007, hệ số là khoảng 16x; bạn sẽ đến phần đó ở dưới.
              </p>
            }
          >
          <p>
            <b>8x</b> as printed: 8 GB/s to 64 GB/s.{' '}
            {Math.abs(v - 8) <= 1
              ? 'That is what the table prints.'
              : v >= 50
                ? 'A guess near 100 assumes the link kept pace with the device. It did not.'
                : v >= 12 && v <= 20
                  ? "That is close to what NVIDIA's own documents give for the 2007 board."
                  : v < 8
                    ? 'You were lower than the table, and right about the direction: the link fell behind.'
                    : 'The link fell further behind the device than you expected.'}{' '}
            On NVIDIA's figures for the 2007 board the factor is about 16x; you will get to that below.
          </p>
          </Translated>
        )}
      </Predict>
      <TableAudit onCheck={() => setAudited(true)} />
      {audited ? (
        <>
          <Translated
            vi={
            <p data-after-audit>
              Bảng tự mâu thuẫn hai lần. 2200 / 0.345 là 6,377, không phải 638, và con số 2,200 là của chính NVIDIA, nên ô
              sai là ô tỉ lệ. “1,800 TB/s” là 1,800 GB/s, và nó đếm gộp cả hai chiều trong khi hàng PCIe đếm mỗi chiều.
            </p>
            }
          >
          <p data-after-audit>
            The table contradicts itself twice. 2200 / 0.345 is 6,377, not 638, and the 2,200 figure is NVIDIA's own,
            so the ratio is the wrong cell. “1,800 TB/s” is 1,800 GB/s, and it counts both directions together while
            the PCIe row counts each way.
          </p>
          </Translated>
          <Aside>
            <Translated
              vi={
              <p>
                Facts A9 và A13 cho hai điểm này, A9 và A6 cho hai điểm tiếp theo, A3, A4, A8 và A11 cho phần còn lại
                (<code>notes/today/ch24-facts.md</code>).
              </p>
              }
            >
            <p>
              Facts A9 and A13 for these two, A9 and A6 for the next two, A3, A4, A8 and A11 for the rest
              (<code>notes/today/ch24-facts.md</code>).
            </p>
            </Translated>
          </Aside>
          <Translated
            vi={
            <p>
              Hai ô nữa sai so với tài liệu của NVIDIA. Hàng có nhãn “Tensor Core FP32” là TF32 trên Tensor Core tính có
              sparsity; dense là một nửa. Bo mạch 1.5 GB là PCI Express thế hệ 1 theo thông số bo mạch của NVIDIA, nên trên
              các số liệu đó đường truyền tới host tăng khoảng 16x, không phải 8x. Phần còn lại lỏng lẻo hơn: cột 2007 lấy
              băng thông từ một bo mạch và dung lượng từ một bo mạch khác, và 80 TFLOPS và 192 GB của cột 2025 khớp với
              những số liệu cụ thể của NVIDIA chứ không phải với các bo mạch được bán ra, vốn có 75 TFLOPS và 180 GB.
            </p>
            }
          >
          <p>
            Two more cells are wrong against NVIDIA's documents. The row labelled “Tensor Core FP32” is TF32 on Tensor
            Cores counted with sparsity; dense is half. The 1.5 GB board is PCI Express generation 1 in NVIDIA's board
            specification, so on those figures the host link grew about 16x, not 8x. The rest is looser: the 2007
            column takes its bandwidth from one board and its capacity from another, and the 2025 column's 80 TFLOPS
            and 192 GB match particular NVIDIA figures rather than the boards that ship, which have 75 TFLOPS and 180
            GB.
          </p>
          </Translated>
          <Today checked="2026-11-20 (ch24-facts A2, A8, A11)">{/* i18n-ignore: date and facts citation */}
            <Translated
              vi={
              <p>
                Mọi số liệu đều là giá trị đỉnh được công bố, không phải phép đo: 0.345 TFLOPS là 128 phép nhân-cộng mỗi chu
                kỳ × 2 × 1.35 GHz.
              </p>
              }
            >
            <p>
              Every figure is a published peak, not a measurement: 0.345 TFLOPS is 128 multiply-adds per clock × 2 ×
              1.35 GHz.
            </p>
            </Translated>
          </Today>
          <Translated
            vi={
            <p>
              Một điểm vẫn đứng vững qua mọi sửa chữa: mọi thứ trên device tăng khoảng hai bậc độ lớn trở lên, còn đường
              truyền tới host chỉ một bậc. Cách đọc của chúng tôi (sách chỉ in các con số): giữ trọng số, KV cache và việc
              lấy mẫu trên device.
            </p>
            }
          >
          <p>
            One point survives every correction: everything on the device grew by about two orders of magnitude or
            more, the host link by one. Our reading (the book only prints the numbers): keep weights, the KV cache and
            sampling on the device.
          </p>
          </Translated>
          <Ask q={questions240['ch24/start/q-638']} />
        </>
      ) : committed ? (
        <p className="muted" data-gate="audit">
          <Translated vi="Kết quả của phép kiểm tra, và một câu hỏi về nó, sẽ hiện ra sau khi bạn bấm Kiểm tra ở trên.">
            What the check finds, and a question on it, follow once you press Check above.
          </Translated>
        </p>
      ) : null}

      <h2><Translated vi="Đọc bảng như một người viết kernel">Reading the table as a kernel writer</Translated></h2>
      <Translated
        vi={
        <p>
          Sách dừng ở các tỉ lệ. Mục 5.1 cho hai hàng trên cùng một cách dùng tốt hơn: ngưỡng của một thiết bị là thông
          lượng đỉnh chia cho băng thông đỉnh của nó, và một kernel có FLOP/B dưới ngưỡng đó thì bị giới hạn bởi bộ nhớ.
          Việc áp dụng điều đó vào Fig. 24.1 là của chúng tôi; chương 24 không làm vậy. Dải bên dưới đặt các kernel từ những
          chương trước lên một trục FLOP/B. Chọn một cột thì ngưỡng di chuyển; mọi thứ bên trái nó bị giới hạn bởi bộ nhớ
          trên các số liệu đó.
        </p>
        }
      >
      <p>
        The book stops at the ratios. Section 5.1 gives the top two rows a better use: a device's threshold is its peak
        throughput over its peak bandwidth, and a kernel whose FLOP/B is below it is memory-bound. Applying that to Fig.
        24.1 is ours; chapter 24 does not do it. The strip puts kernels from earlier chapters on one FLOP/B axis. Choose
        a column and the threshold moves; everything left of it is memory-bound on those figures.
      </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
          <p>
            Giá trị đỉnh là số liệu của nhà sản xuất, bộ nhớ đệm (cache) bị bỏ qua, và hai dấu lớp tuyến tính là kernel 16-bit
            đặt cạnh các đỉnh FP32 và TF32, nên hãy coi phía của đường mà chúng nằm chỉ mang tính tham khảo. Batch 1 bị giới
            hạn bởi bộ nhớ trên bất kỳ đỉnh nào.
          </p>
          }
        >
        <p>
          Peaks are vendor figures, caches are ignored, and the two linear-layer marks are 16-bit kernels set against
          FP32 and TF32 peaks, so read their side of the line as indicative. Batch 1 is memory-bound on any peak.
        </p>
        </Translated>
      </Simplification>
      <Predict
        id={P_KNEE}
        unit="FLOP/B"
        prompt={
          <Translated
            vi={
              <>
                Theo số liệu 2007 của sách, ngưỡng của mục 5.1 là 0.345 TFLOPS chia cho 86.4 GB/s, khoảng{' '}
                {THR_2007.toFixed(1)} FLOP/B. Sau đó thông lượng FP32 tăng 232x và băng thông tăng 93x. Ngưỡng ở cột 2025 là
                bao nhiêu?
              </>
            }
          >
          <>
            By the book's 2007 figures the threshold of section 5.1 is 0.345 TFLOPS over 86.4 GB/s, about{' '}
            {THR_2007.toFixed(1)} FLOP/B. FP32 throughput then grew 232x and bandwidth 93x. What is the threshold in
            the 2025 column?
          </>
          </Translated>
        }
      >
        {(v) => (
          <Translated
            vi={
              <p>
                80,000 GFLOP/s / 8,000 GB/s = <b>{fmt(THR_2025)} FLOP/B</b>, gấp {fmt(THR_2025 / THR_2007, 2)} lần giá trị
                2007. So với số liệu tensor core thì nó là {fmt(THR_TC)}.{' '}
                {Math.abs(v - THR_2025) <= 0.5
                  ? 'Đúng như bạn đoán.'
                  : v < THR_2007
                    ? 'Một phỏng đoán dưới 4 đã lộn ngược thương số: tính toán tăng nhanh hơn băng thông, nên ngưỡng tăng lên.'
                    : 'Cách tính tắt là 4.0 × 232 / 93.'}
              </p>
            }
          >
          <p>
            80,000 GFLOP/s / 8,000 GB/s = <b>{fmt(THR_2025)} FLOP/B</b>, {fmt(THR_2025 / THR_2007, 2)} times the 2007
            value. Against the tensor-core figure it is {fmt(THR_TC)}.{' '}
            {Math.abs(v - THR_2025) <= 0.5
              ? 'That is what you had.'
              : v < THR_2007
                ? 'A guess below 4 has the quotient upside down: compute grew faster than bandwidth, so the threshold rose.'
                : 'The shortcut is 4.0 × 232 / 93.'}
          </p>
          </Translated>
        )}
      </Predict>
      <ThresholdBench />
      <Translated
        vi={
        <p>
          Tính toán tăng nhanh hơn băng thông, nên ngưỡng tăng lên. Một ô rộng 16 đã đạt tới nó trên số liệu 2007. So với
          số liệu tensor core, cả ô rộng 32 của chương 5 (8 FLOP/B) lẫn ô đầu ra 128 × 128 của chương 15 (32 FLOP/B, §15.2,
          pp. 350–352) đều không đạt.
        </p>
        }
      >
      <p>
        Compute grew faster than bandwidth, so the threshold rose. A 16-wide tile reached it on the 2007 figures.
        Against the tensor-core figure neither chapter 5's 32-wide tile (8 FLOP/B) nor chapter 15's 128 × 128 output
        tile (32 FLOP/B, §15.2, pp. 350–352) does.
      </p>
      </Translated>
      <Ask q={questions240['ch24/start/q-tile']} />
      <Translated
        vi={
        <p>
          Với công việc ma trận-ma trận, nguồn tái sử dụng tiếp theo sau một ô bộ nhớ chia sẻ là các ô thanh ghi của chương
          15, dù so với số liệu tensor core thì ngay cả ô 128 × 128 cũng chưa đủ trong mô hình này. Sinh token với batch 1,
          khoảng 1 FLOP/B trong ví dụ 16-bit của sách (§20.6, p. 505), bị giới hạn bởi bộ nhớ ở mọi cột; tỉ lệ của nó chỉ
          thay đổi nhờ batching (chương 20 và 22) hoặc ít byte hơn cho mỗi trọng số (module X2). Phép tính đứng sau con số
          tính toán lớn hơn, TF32 trên Tensor Core (dense, nên tối đa một nửa con số được in), là thứ mà{' '}
          <code>torch.backends.cuda.matmul.allow_tf32</code> bật lên cho matmul.
        </p>
        }
      >
      <p>
        For matrix-matrix work, the next source of reuse after a shared-memory tile is chapter 15's register tiles,
        though against the tensor-core figure even the 128 × 128 tile falls short in this model. Batch-1 generation,
        about 1 FLOP/B in the book's 16-bit example (§20.6, p. 505), is memory-bound in every column; its ratio moves
        only with batching (chapters 20 and 22) or fewer bytes per weight (module X2). The arithmetic behind the larger
        compute number, TF32 on Tensor Cores (dense, so at most half the printed figure), is what{' '}
        <code>torch.backends.cuda.matmul.allow_tf32</code> switches on for matmul.
      </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the PyTorch 2.15 docs (ch24-facts A9, D4)', '2026-11-20 so với tài liệu PyTorch 2.15 (ch24-facts A9, D4)')}>
        <Translated
          vi={
          <p>
            Mặc định PyTorch để cờ đó tắt cho matmul. TF32 làm tròn đầu vào xuống 10 bit phần định trị và tích lũy ở FP32.
          </p>
          }
        >
        <p>
          PyTorch leaves that flag off for matmul by default. TF32 rounds inputs to 10 bits of mantissa and accumulates
          in FP32.
        </p>
        </Translated>
      </Today>

      <h2><Translated vi="Cùng các hàng đó trên AMD và Apple">The same rows on AMD and Apple</Translated></h2>
      <Translated
        vi={
        <p>
          Từ vựng của bảng là của NVIDIA. Đây là tên gọi của mỗi vai trò, hoặc nó có tồn tại hay không, trên hai backend
          khác mà khóa học này xây dựng cho. Cột cuối cho các mục của <code>notes/today/ch24-facts.md</code>{' '}
          đứng sau mỗi hàng; WMMA và WGMMA là tên riêng của sách (ch. 15 §15.9, p. 369).
        </p>
        }
      >
      <p>
        The table's vocabulary is NVIDIA's. Here is what each role is called, or whether it exists, on the two other
        backends this course builds for. The last column gives the items of <code>notes/today/ch24-facts.md</code>{' '}
        behind each row; WMMA and WGMMA are the book's own names (ch. 15 §15.9, p. 369).
      </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Chương 24 không nêu tên nhà sản xuất nào khác. Bảng này là của chúng tôi.</p>}>
          <p>Chapter 24 names no other vendor. This table is ours.</p>
        </Translated>
      </Beyond>
      <table className="data wide" data-backends>
        <thead>
          <tr>
            <th><Translated vi="Vai trò">Role</Translated></th>
            <th>CUDA (NVIDIA)</th>{/* i18n-ignore: backend and vendor names */}
            <th>HIP (AMD)</th>{/* i18n-ignore: backend and vendor names */}
            <th>Metal (Apple)</th>{/* i18n-ignore: backend and vendor names */}
            <th><Translated vi="Mục facts">Facts</Translated></th>
          </tr>
        </thead>
        <tbody>
          {BACKEND_ROWS.map((r) => (
            <tr key={r.role}>
              <td>
                <b>{r.role}</b>
              </td>
              <td>{r.cuda}</td>
              <td>{r.hip}</td>
              <td>{r.metal}</td>
              <td className="muted nowrap">{r.facts}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
        <p>
          TF32 không khả chuyển: một chip AMD hiện tại cung cấp nó qua giả lập bằng phần mềm, và nó không có trong bảng kiểu
          của <code>matmul2d</code> trên Metal. Apple không công bố số FLOPS nào, nên không thể dựng một cột số liệu cho
          Metal. Ở chỗ một ô ghi “not found” hay “none documented”, chúng tôi không tìm thấy gì trong các trang đã đọc; nhà
          sản xuất không nói là không có.
        </p>
        }
      >
      <p>
        TF32 is not portable: one current AMD part offers it through software emulation, and it is absent from the
        type table of Metal's <code>matmul2d</code>. Apple publishes no FLOPS figure, so no Metal column of numbers can
        be built. Where a cell says “not found” or “none documented”, we found none in the pages read; the vendor does
        not say there is none.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch24-facts B1, B4, C1, C3, C5)">{/* i18n-ignore: date and facts citation */}
        <Translated vi={<p>Trang MI355X của AMD chú thích “TF32 support through software emulation”.</p>}>
          <p>AMD's MI355X page footnotes “TF32 support through software emulation”.</p>
        </Translated>
      </Today>
      <Ask q={questions240['ch24/start/q-norow']} />

      <h2><Translated vi="Phần còn lại của triển vọng">The rest of the outlook</Translated></h2>
      <Translated
        vi={
        <p>
          Hai trang cuối là triển vọng của các tác giả, và tất cả là của họ. Các tác vụ giờ trải trên nhiều GPU, do việc huấn
          luyện LLM thúc đẩy, với NCCL là thư viện được nêu tên. Đổi mới giờ được dẫn dắt bởi tính toán có thiết bị đo hỗ
          trợ, không còn theo chiều ngược lại; các ví dụ của họ là lái xe, y học và quang khắc. Truy cập song song vào bộ lưu
          trữ là cơ hội tiếp theo mà họ thấy.
        </p>
        }
      >
      <p>
        The last two pages are the authors' outlook, and all of it is theirs. Jobs now span many GPUs, driven by LLM
        training, with NCCL as the library named. Innovation is now led by computing assisted by instruments, no longer
        the reverse; their examples are driving, medicine and lithography. Parallel access to storage is the next
        opening they see.
      </p>
      </Translated>
      <Today checked="2026-11-20 (ch24-facts A16, A17, A18)">{/* i18n-ignore: date and facts citation */}
        <Translated
          vi={
          <p>
            Kho mã công khai của NCCL có từ năm 2015, nên “mới” chỉ là tương đối. Các khẳng định về quy mô tác vụ và quang
            khắc là nhận định của các tác giả.
          </p>
          }
        >
        <p>
          NCCL's public repository dates from 2015, so “new” is relative. The claims on job sizes and lithography are
          the authors' judgement.
        </p>
        </Translated>
      </Today>
      <Translated
        vi={
        <p>
          Để lại trong sách: ba ví dụ đầy đủ và các lĩnh vực mà các tác giả kỳ vọng sẽ thay đổi tiếp theo (pp. 579–580).{' '}
          <a href={href('ch24', 'wrap')}>Tóm tắt và đi tiếp đâu</a> gom lại những gì cần mang theo.
        </p>
        }
      >
      <p>
        Left in the book: the three examples in full and the fields the authors expect to change next (pp. 579–580).{' '}
        <a href={href('ch24', 'wrap')}>Summary and where next</a> collects what to carry forward.
      </p>
      </Translated>
      <Source section="24.2" pages="579–580" />
    </>
  );
}
