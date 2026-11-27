import { useEffect, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useProgress } from '../../lib/progress';
import { href } from '../../lib/router';
import { ATTENTION_HOME } from '../../sim/attnroute';
import { ATTN_HOME_OPTIONS, OPS, PROJECTS, PROJECT_NAME, attentionHome, type AttnHome, type Project } from '../../sim/backendmap';
import { PINNED, READ_ON, permalink, shortCommit, type Excerpt as ExcerptData } from '../../sim/x3sources';
import { loadChapter } from '../registry';
import { Excerpt } from './Excerpt';

// Module X3's start page: what reading a real backend means here, the snapshot every claim
// rests on (rendered from sim/x3sources.ts PINNED), the method in one paragraph, the tour's
// ops and where the course taught each (from sim/backendmap.ts OPS), how excerpts are shown,
// milestone M10, a prediction X3.5 settles, and a prerequisite check that reuses earlier
// chapters' questions by id (loaded on demand through the registry, never imported).

const FACTS = 'notes/today/x3-facts.md';
const P_HOME = 'x3/start/p-attention-home';

export const EXCERPTS_SX30: Record<string, ExcerptData> = {
  addH: {
    id: 'x30-ufunc-add', project: 'pytorch', path: 'aten/src/ATen/native/ufunc/add.h', lines: null, factsItem: 'A3', lang: 'cpp',
    code: `return self + alpha * other;`,
  },
};

export const questionsSX30: Record<string, Question> = {
  'x3/start/q-hip-same': {
    kind: 'choice',
    id: 'x3/start/q-hip-same',
    prompt: (
      <Translated
        vi={
          <>
            llama.cpp có một thư mục <code>ggml/src/ggml-hip</code>. Kernel softmax HIP nằm ở đâu?
          </>
        }
      >
        <>
          llama.cpp has a directory <code>ggml/src/ggml-hip</code>. Where is the HIP softmax kernel?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Trong ggml-cuda/softmax.cu: ggml-hip chứa các tệp build biên dịch ../ggml-cuda/*.cu thành HIP.">
            In ggml-cuda/softmax.cu: ggml-hip holds build files that compile ../ggml-cuda/*.cu as HIP.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Trong một bản sao softmax.cu đã qua hipify mà bước build ghi vào ggml-hip khi bạn cấu hình cho AMD.">
            In a hipified copy of softmax.cu that the build writes into ggml-hip when you configure it for AMD.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là cách của PyTorch. llama.cpp đổi tên các định danh CUDA bằng #define trong vendors/hip.h và biên dịch chính tệp CUDA.">
            That is PyTorch’s approach. llama.cpp renames CUDA identifiers with #defines in vendors/hip.h and compiles the CUDA file itself.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Trong ggml-hip/softmax.hip, một bản port giữ bên cạnh bản CUDA.">
            In ggml-hip/softmax.hip, a port kept beside the CUDA one.
          </Translated>
        ),
        why: (
          <Translated vi="Không có tệp nào như vậy: ggml-hip chỉ chứa CMakeLists.txt.">
            There is no such file: ggml-hip contains only CMakeLists.txt.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không ở đâu cả: llama.cpp chạy softmax trên CPU khi GPU là AMD.">
            Nowhere: llama.cpp runs softmax on the CPU when the GPU is AMD.
          </Translated>
        ),
        why: (
          <Translated vi="Backend HIP nhận softmax; đó là kernel CUDA được biên dịch cho AMD.">
            The HIP backend accepts softmax; it is the CUDA kernel compiled for AMD.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>ggml-hip/CMakeLists.txt</code> glob <code>../ggml-cuda/*.cu</code> và đặt <code>LANGUAGE HIP</code>. Nhánh
            riêng theo nhà cung cấp duy nhất thấy được trong <code>softmax.cu</code> là một <code>#ifdef GGML_USE_HIP</code> include{' '}
            <code>&lt;hip/hip_cooperative_groups.h&gt;</code>.
          </p>
        }
      >
        <p>
          <code>ggml-hip/CMakeLists.txt</code> globs <code>../ggml-cuda/*.cu</code> and sets <code>LANGUAGE HIP</code>. The one
          visible vendor branch in <code>softmax.cu</code> is an <code>#ifdef GGML_USE_HIP</code> that includes{' '}
          <code>&lt;hip/hip_cooperative_groups.h&gt;</code>.
        </p>
      </Translated>
    ),
    source: `${FACTS} B9; notes/today/infra-facts.md D2; notes/today/ch05-facts.md B9`,// i18n-ignore: citation only
  },
  'x3/start/q-pinned': {
    kind: 'choice',
    id: 'x3/start/q-pinned',
    prompt: (
      <Translated
        vi={<>Vì sao mọi liên kết trong học phần này đều trỏ tới một mã băm commit thay vì <code>master</code> hay <code>main</code>?</>}
      >
        <>Why does every link in this module point at a commit hash rather than at <code>master</code> or <code>main</code>?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các đoạn trích và số dòng được đọc ở commit đó, còn master ngày mai có thể chứa mã khác.">
            The excerpts and line numbers were read at that commit, and master may hold other code tomorrow.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Giấy phép của các dự án yêu cầu mọi trích dẫn mã nguồn phải liên kết tới một phiên bản cố định đã phát hành.">
            The projects’ licences require any quotation of their source to link to a fixed, released version.
          </Translated>
        ),
        why: (
          <Translated vi="Không văn bản giấy phép nào đã đọc nói vậy. Lý do là độ chính xác, không phải giấy phép.">
            None of the licence texts read says so. The reason is accuracy, not licensing.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="GitHub không phục vụ neo dòng như #L120 trên liên kết nhánh, chỉ trên permalink của commit.">
            GitHub does not serve line anchors such as #L120 on branch links, only on commit permalinks.
          </Translated>
        ),
        why: (
          <Translated vi="Có đấy. Chỉ là chúng trỏ tới bất cứ thứ gì nhánh đang chứa hôm nay.">
            It does. They just point at whatever the branch holds today.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Master là riêng tư với các dự án này; chỉ các commit là công khai.">
            Master is private for these projects; only commits are public.
          </Translated>
        ),
        why: (
          <Translated vi="Cả ba nhánh mặc định đều công khai.">All three default branches are public.</Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lượt kiểm tra dữ kiện đọc mỗi tệp ở một commit. Nhiều tệp trong chuyến tham quan đã thay đổi ngay trong ngày đọc, nên
            số dòng trên <code>master</code> có thể đã lệch. Một permalink giữ cho đoạn trích, số dòng và trang khớp với nhau.
            Hãy tự đọc <code>master</code> khi bạn muốn mã của hôm nay, và hãy lường trước rằng nó sẽ khác.
          </p>
        }
      >
        <p>
          The facts pass read each file at one commit. Several files in the tour changed on the day of reading, so a line number
          on <code>master</code> could already be off. A permalink keeps the excerpt, the line numbers and the page in agreement.
          Read <code>master</code> yourself when you want today’s code, and expect it to differ.
        </p>
      </Translated>
    ),
    source: `${FACTS}, header (“Snapshot”)`,
    vi: { source: `${FACTS}, phần đầu (“Snapshot”)` },
  },
};

/** Earlier chapters' questions this page asks again, by id. */
const PRIOR: { id: string; label: string }[] = [
  { id: 'ch10/10.6/q-two-reductions', get label() { return tr('From chapter 10', 'Từ chương 10'); } },
  { id: 'ch10/10.2/q-argmax-rule', get label() { return tr('From chapter 10', 'Từ chương 10'); } },
  { id: 'ch04/4.5/q-amd32', get label() { return tr('From chapter 4', 'Từ chương 4'); } },
  { id: 'ch15/15.5/q-float4-alone', get label() { return tr('From chapter 15', 'Từ chương 15'); } },
  { id: 'ch20/20.1/q-rope', get label() { return tr('From chapter 20', 'Từ chương 20'); } },
  { id: 'ch20/20.5/q-init', get label() { return tr('From chapter 20', 'Từ chương 20'); } },
  { id: 'ch22/start/q-gemv-ceiling', get label() { return tr('From chapter 22', 'Từ chương 22'); } },
  { id: 'appA/A.6/q-fastmath', get label() { return tr('From appendix A', 'Từ phụ lục A'); } },
  { id: 'x1/X1.2/q-greedy-ties', get label() { return tr('From module X1', 'Từ học phần X1'); } },
];

/** Ask an earlier chapter's question by its id: its chapter and section load on demand. */
function PriorAsk({ id, label }: { id: string; label: string }) {
  const [q, setQ] = useState<Question | null | undefined>(undefined);
  useEffect(() => {
    let live = true;
    const [ch, sec] = id.split('/');
    const p = loadChapter(ch);
    if (!p) {
      setQ(null);
      return;
    }
    p.then((c) => c.sections.find((s) => s.id === sec)?.load())
      .then((m) => live && setQ(m?.questions[id] ?? null))
      .catch(() => live && setQ(null));
    return () => {
      live = false;
    };
  }, [id]);
  if (q === undefined)
    return (
      <p className="muted ui" data-prior={id}>
        <Translated vi={<>Đang tải một câu hỏi từ {id.split('/')[0]}…</>}>Loading a question from {id.split('/')[0]}…</Translated>
      </p>
    );
  if (q === null)
    return (
      <p className="muted ui" data-prior={id} data-missing>
        <Translated vi={<>Câu hỏi {id} không có sẵn.</>}>The question {id} is not available.</Translated>
      </p>
    );
  const [ch, sec] = id.split('/');
  return (
    <div data-prior={id}>
      <Ask q={q} label={label} />
      <p className="muted ui" style={{ margin: '-0.6rem 0 1.2rem', fontSize: '0.8rem' }}>
        <Translated
          vi={
            <>
              Được hỏi lần đầu ở <a href={href(ch, sec)}>{ch} {sec === 'start' ? 'trang mở đầu' : sec}</a>.
            </>
          }
        >
          Asked first in <a href={href(ch, sec)}>{ch} {sec === 'start' ? 'start page' : sec}</a>.
        </Translated>
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- the prediction X3.5 settles
type HomePick = Partial<Record<Project, AttnHome>>;
function AttentionHomePredict() {
  const { progress, answer } = useProgress();
  const prior = progress.answers[P_HOME]?.value as HomePick | undefined;
  const [pick, setPick] = useState<HomePick>({});
  const [show, setShow] = useState(false);
  const committed = prior && PROJECTS.every((p) => prior[p]) ? prior : null;
  const all = PROJECTS.every((p) => pick[p]);
  const cudaRows = ATTENTION_HOME.filter((r) => r.backend.includes('CUDA'));
  return (
    <div className="q" data-part="predict-home">
      <div className="kind"><Translated vi="Dự đoán trước">Predict first</Translated></div>
      <div className="prompt">
        <Translated
          vi={
            <p style={{ margin: '0 0 0.5rem' }}>
              Bạn sẽ đọc mã attention của ba dự án trên GPU NVIDIA. Với mỗi dự án, các kernel attention CUDA mà nó chạy đến từ
              đâu? Hãy chốt một câu trả lời cho mỗi dự án ngay bây giờ; mục <a href={href('x3', 'X3.5')}>X3.5</a> sẽ phân xử.
            </p>
          }
        >
          <p style={{ margin: '0 0 0.5rem' }}>
            You will read three projects’ attention code on NVIDIA GPUs. For each, where do the CUDA attention kernels it runs
            come from? Commit one answer per project now; section <a href={href('x3', 'X3.5')}>X3.5</a> settles it.
          </p>
        </Translated>
      </div>
      {committed == null ? (
        <>
          <div style={{ display: 'grid', gap: '0.5rem', marginBottom: '0.6rem' }}>
            {PROJECTS.map((p) => (
              <fieldset key={p} style={{ border: '1px solid var(--rule)', borderRadius: 4, padding: '0.4rem 0.6rem', margin: 0 }} data-project={p}>{/* i18n-ignore: CSS values */}
                <legend className="ui" style={{ fontWeight: 600, padding: '0 0.3rem' }}>{PROJECT_NAME[p]}</legend>{/* i18n-ignore: CSS value */}
                {ATTN_HOME_OPTIONS.map((o) => (
                  <label key={o.id} className="check" style={{ display: 'flex', marginBottom: '0.15rem' }}>
                    <input type="radio" name={`${P_HOME}-${p}`} checked={pick[p] === o.id} onChange={() => setPick({ ...pick, [p]: o.id })} data-home={o.id} />
                    <span>{o.label}</span>
                  </label>
                ))}
              </fieldset>
            ))}
          </div>
          <button className="primary" disabled={!all} onClick={() => answer(P_HOME, pick)}>
            <Translated vi="Chốt dự đoán">Commit prediction</Translated>
          </button>
        </>
      ) : (
        <div className="verdict info" data-readout="home-committed">
          <p>
            <b><Translated vi="Bạn đã dự đoán:">You predicted:</Translated></b>{' '}
            {PROJECTS.map((p, i) => (
              <span key={p}>
                {i ? '; ' : ''}
                {PROJECT_NAME[p]}, {ATTN_HOME_OPTIONS.find((o) => o.id === committed[p])!.label}
              </span>
            ))}
            .
          </p>
          {show ? (
            <div data-readout="home-reveal">
              {PROJECTS.map((p) => {
                const right = attentionHome(p);
                const row = cudaRows.find((r) => PROJECT_NAME[p] === r.project);
                return (
                  <p key={p} data-project={p} data-right={committed[p] === right ? '1' : '0'}>
                    <b className={committed[p] === right ? 't-ok' : 't-fail'}>{PROJECT_NAME[p]}:</b>{' '}
                    {ATTN_HOME_OPTIONS.find((o) => o.id === right)!.label}. {row ? <span className="muted">({row.home}.)</span> : null}
                  </p>
                );
              })}
              <Translated
                vi={
                  <p className="muted">
                    Mục <a href={href('x3', 'X3.5')}>X3.5</a> lần theo từng cái đến các tệp của nó.
                  </p>
                }
              >
                <p className="muted">Section <a href={href('x3', 'X3.5')}>X3.5</a> follows each one to its files.</p>
              </Translated>
            </div>
          ) : (
            <p>
              <Translated vi="Mục X3.5 tiết lộ đáp án khi bạn đọc mã.">Section X3.5 reveals it as you read the code.</Translated>{' '}
              <button className="quiet" onClick={() => setShow(true)}>
                <Translated vi="Cho tôi xem ngay">Show me now</Translated>
              </button>
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the section
export function SX30() {
  const q = questionsSX30;
  const tour = [...new Map(OPS.map((o) => [o.tour, o])).keys()].map((t) => {
    const rows = OPS.filter((o) => o.tour === t);
    const labs = [...new Map(rows.flatMap((o) => o.course).map((l) => [l.num, l])).values()];
    return { tour: t, section: rows[0].section, labs };
  });
  return (
    <>
      <Translated
        vi={
          <p>
            Khóa học đã cho bạn tự viết mọi kernel của một bước decode kiểu Llama. Học phần này đọc các kernel mà llama.cpp, MLX
            và PyTorch chạy cho cùng các op đó, trên CUDA, HIP và Metal. Mục tiêu là mục tiêu thứ ba bạn đã đặt ra: đọc và đóng
            góp cho các backend đó. Đến cuối, bạn sẽ có thể gọi tên tệp và hàm chạy cho bất kỳ op nào trong chuyến tham quan,
            trên bất kỳ backend nào trong ba backend, và nói nó làm gì khác với kernel của chính bạn.
          </p>
        }
      >
        <p>
          The course has had you write every kernel of a Llama-style decode step yourself. This module reads the kernels that
          llama.cpp, MLX and PyTorch run for the same ops, on CUDA, HIP and Metal. The aim is the third goal you set: to read and
          contribute to those backends. By the end you should be able to name the file and function that runs for any op in
          the tour, on any of the three backends, and say what it does differently from your own kernel.
        </p>
      </Translated>
      <Beyond title={tr("Ours, not the book's", 'Của khóa học này, không phải của sách')}>
        <Translated
          vi={
            <p>
              Sách không có chương nào về backend thực tế, nên học phần này không có trang sách và không có tệp ghi chú. Mọi khẳng
              định về một codebase đều được đọc trong một lượt kiểm tra dữ kiện vào ngày {READ_ON}, ở các commit bên dưới, và được
              trích dẫn theo mục của nó trong <code>{FACTS}</code>. Không có gì được build hay chạy: điều một kernel “làm” là cách
              đọc mã nguồn của nó.
            </p>
          }
        >
          <p>
            The book has no chapter on real backends, so there are no book pages and no notes file for this module. Every claim
            about a codebase was read in one facts pass on {READ_ON}, at the commits below, and is cited by its item in{' '}
            <code>{FACTS}</code>. Nothing was built or run: what a kernel “does” is a reading of its source.
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Bản chụp nhanh">The snapshot</Translated></h2>
      <Translated
        vi={
          <p>
            Các dự án thực tế thay đổi hằng ngày. Nhiều tệp mà học phần này đọc đã thay đổi ngay trong ngày chúng được đọc. Vì
            vậy học phần cố định một commit cho mỗi dự án, và mọi liên kết là một permalink tới commit đó: đoạn trích bạn thấy,
            số dòng của nó và tệp sau liên kết đều khớp nhau.
          </p>
        }
      >
        <p>
          Real projects move daily. Several of the files this module reads changed on the day they were read. So the module
          fixes one commit per project, and every link is a permalink to it: the excerpt you see, its line numbers and the file
          behind the link agree.
        </p>
      </Translated>
      <Today checked={tr(`${READ_ON} in the facts pass (${FACTS}, header and A4)`, `${READ_ON} trong lượt kiểm tra dữ kiện (${FACTS}, phần đầu và A4)`)}>
        <Translated
          vi={
            <p>
              Các tệp được lấy ở những commit này; ngày commit cuối của mỗi tệp là ngày GitHub ghi cho đường dẫn đó tại hoặc trước
              commit. Không kiểm tra: bất cứ điều gì cần build hay chạy các dự án.
            </p>
          }
        >
          <p>
            Files were fetched at these commits; each file’s last-commit date is GitHub’s for that path at or before the commit.
            Not checked: anything by building or running the projects.
          </p>
        </Translated>
      </Today>
      <table className="data wide" data-part="snapshot" style={{ maxWidth: '62rem' }}>
        <thead>
          <tr>
            <th><Translated vi="Dự án">Project</Translated></th>
            <th><Translated vi="Commit">Commit</Translated></th>
            <th><Translated vi="Ngày commit">Committed</Translated></th>
            <th><Translated vi="Giấy phép">Licence</Translated></th>
            <th><Translated vi="Dòng bản quyền">Copyright line</Translated></th>
          </tr>
        </thead>
        <tbody>
          {PINNED.map((p) => (
            <tr key={p.id} data-project={p.id}>
              <td><a href={permalink(p.id, '', null, true)} target="_blank" rel="noreferrer">{p.name}</a> <span className="muted">{p.repo}</span></td>
              <td className="nowrap"><code data-commit={p.commit}>{shortCommit(p.id)}</code></td>
              <td className="nowrap">{p.date ?? '—'}</td>
              <td>{p.licence}{p.id === 'pytorch' ? tr(' (GitHub reports NOASSERTION)', ' (GitHub báo NOASSERTION)') : ''}</td>
              <td>{p.copyright ?? <span className="muted"><Translated vi="không được trích trong lượt kiểm tra dữ kiện">not quoted in the facts pass</Translated></span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Một số ô của chuyến tham quan dựa vào các mục đã được xác lập trong tệp dữ kiện của các chương trước (các kernel
            attention của llama.cpp ở chương 20, switch GEMV của nó ở chương 22, các bộ lấy mẫu ở học phần X1). Những mục đó được
            đọc ở <code>master</code> trong cùng ngày. Học phần liên kết chúng theo mục dữ kiện và URL riêng của chúng, và không
            bao giờ gán cho chúng số dòng ở commit được ghim.
          </p>
        }
      >
        <p>
          Some cells of the tour rest on items settled in earlier chapters’ facts files (llama.cpp’s attention kernels in
          chapter 20, its GEMV switch in chapter 22, the samplers in module X1). Those were read at <code>master</code> on the
          same day. The module links them by their facts item and their own URL, and never gives them line numbers at the
          pinned commit.
        </p>
      </Translated>
      <Ask q={q['x3/start/q-pinned']} />

      <h2><Translated vi="Phương pháp, trong một đoạn">The method, in one paragraph</Translated></h2>
      <Translated
        vi={
          <p>
            Mỗi dự án có một nơi mà một op trở thành một lời gọi backend. Trong llama.cpp đó là một switch trên enum op của đồ
            thị, mỗi backend một cái. Trong MLX đó là một phương thức, <code>eval_gpu</code>, mà mỗi primitive cài đặt một lần
            cho mỗi thư mục backend. Trong PyTorch đó là bảng dispatch trong <code>native_functions.yaml</code>. Bắt đầu từ đó và
            lần theo một cái tên. Trên Metal, cái tên là một chuỗi do host dựng, nên hãy lần theo chuỗi. Mục{' '}
            <a href={href('x3', 'X3.1')}>X3.1</a> làm điều này cho mọi op, kể cả hai chỗ cách này đổ vỡ: gộp op (fusion),
            chạy trước switch, và mã được sinh ra lúc build hoặc lúc chạy.
          </p>
        }
      >
        <p>
          Each project has one place where an op becomes a backend call. In llama.cpp it is a switch over the graph’s op enum,
          one per backend. In MLX it is a method, <code>eval_gpu</code>, that each primitive implements once per backend
          directory. In PyTorch it is the dispatch table in <code>native_functions.yaml</code>. Start there and follow one
          name. On Metal the name is a string the host builds, so follow the string. Section{' '}
          <a href={href('x3', 'X3.1')}>X3.1</a> does this for every op, including the two places it breaks: fusion, which runs
          before the switch, and code that is generated at build time or at run time.
        </p>
      </Translated>
      <Today checked={tr(`${READ_ON} in the facts pass (${FACTS}, B9)`, `${READ_ON} trong lượt kiểm tra dữ kiện (${FACTS}, B9)`)}>
        <Translated
          vi={
            <p>
              llama.cpp và PyTorch biên dịch mã nguồn CUDA của chúng cho AMD. MLX không có backend AMD: <code>mlx/backend</code>{' '}
              liệt kê common, cpu, cuda, gpu, metal, no_cpu, no_gpu; pull request #2300, “[Experiment] ROCm backend”, đang mở.
            </p>
          }
        >
          <p>
            llama.cpp and PyTorch compile their CUDA sources for AMD. MLX has no AMD backend: <code>mlx/backend</code> lists
            common, cpu, cuda, gpu, metal, no_cpu, no_gpu; pull request #2300, “[Experiment] ROCm backend”, is open.
          </p>
        </Translated>
      </Today>
      <Ask q={q['x3/start/q-hip-same']} />

      <h2><Translated vi="Chuyến tham quan">The tour</Translated></h2>
      <Translated
        vi={
          <p>
            Tám op tạo nên một bước decode của Llama. Mỗi op có một mục ở đây và một chỗ trong khóa học nơi bạn đã gặp kỹ thuật
            của nó. Ba op bạn chưa viết, RMSNorm, RoPE và SwiGLU, bạn sẽ xây trong lab đầu tiên của học phần này.
          </p>
        }
      >
        <p>
          Eight ops make a Llama decode step. Each has a section here and a place in the course where you met its technique.
          Three you have not written yet, RMSNorm, RoPE and SwiGLU, you build in this module’s first lab.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Một cầu nối để ghi nhớ: <code>torch.nn.functional.rms_norm</code> trong một lớp Llama của Hugging Face. Đến cuối
              X3.1 bạn sẽ tìm ra kernel mà nó chạm tới trên CUDA và trên MPS.
            </p>
          }
        >
          <p>
            A bridge to keep in mind: <code>torch.nn.functional.rms_norm</code> in a Hugging Face Llama layer. By the end of
            X3.1 you will have found the kernel it reaches on CUDA and on MPS.
          </p>
        </Translated>
      </Aside>
      <table className="data" data-part="tour" style={{ maxWidth: '44rem' }}>
        <thead>
          <tr>
            <th><Translated vi="Op">Op</Translated></th>
            <th><Translated vi="Nơi bạn đã viết nó">Where you wrote it</Translated></th>
            <th><Translated vi="Đọc ở">Read in</Translated></th>
          </tr>
        </thead>
        <tbody>
          {tour.map((t) => (
            <tr key={t.tour} data-tour={t.tour}>
              <td>{tr(t.tour, ({ 'GEMV and GEMM': 'GEMV và GEMM', 'RoPE and SwiGLU': 'RoPE và SwiGLU', elementwise: 'theo từng phần tử', sampling: 'lấy mẫu' } as Record<string, string>)[t.tour] ?? t.tour)}</td>
              <td>
                {t.labs.map((l, i) => (
                  <span key={l.num}>
                    {i ? ', ' : ''}
                    <a href={href(l.chapter, 'labs')}>
                      <Translated vi={<>lab {l.num}</>}>lab {l.num}</Translated>
                    </a>
                    {l.planned ? tr(' (new)', ' (mới)') : ''}
                  </span>
                ))}
              </td>
              <td><a href={href('x3', t.section)}>{t.section}</a></td>
            </tr>
          ))}
        </tbody>
      </table>
      <AttentionHomePredict />

      <h2><Translated vi="Cách trình bày đoạn trích">How excerpts are shown</Translated></h2>
      <Translated
        vi={
          <p>
            Mã từ ba dự án xuất hiện dưới dạng các đoạn trích ngắn, tối đa khoảng một chục dòng, chép từ lượt kiểm tra dữ kiện.
            Mỗi đoạn nêu tệp và các dòng, liên kết tới commit được ghim, và mang giấy phép cùng dòng bản quyền của dự án. Một
            khoảng trống đánh dấu các dòng được lược bỏ. Khoảng trắng có thể khác với tệp, vì tệp dữ kiện ghi một số trích dẫn
            trên một dòng.
          </p>
        }
      >
        <p>
          Code from the three projects appears in short excerpts, at most about a dozen lines, copied from the facts pass. Each
          names its file and lines, links to the pinned commit, and carries the project’s licence and copyright line. A gap
          marks lines left out. Spacing can differ from the file, because the facts file records some quotes on one line.
        </p>
      </Translated>
      <Today checked={tr(`${READ_ON} in the facts pass (${FACTS}, A4)`, `${READ_ON} trong lượt kiểm tra dữ kiện (${FACTS}, A4)`)}>
        <Translated
          vi={
            <p>
              llama.cpp và MLX dùng giấy phép MIT (“…shall be included in all copies or substantial portions of the Software”);
              PyTorch dùng giấy phép kiểu BSD ba điều khoản. Một đoạn trích ngắn có phải là một “substantial portion” hay không là
              câu hỏi pháp lý mà lượt kiểm tra không thể phân xử: unverified.
            </p>
          }
        >
          <p>
            llama.cpp and MLX are MIT (“…shall be included in all copies or substantial portions of the Software”); PyTorch is
            three-clause BSD-style. Whether a short excerpt is a “substantial portion” is a legal question the pass could not
            settle: unverified.
          </p>
        </Translated>
      </Today>
      <Excerpt
        x={EXCERPTS_SX30.addH}
        caption={
          <Translated vi="Một ví dụ: phần mã nguồn riêng cho op duy nhất của kernel add CUDA trong PyTorch, thân vô hướng của nó; torchgen sinh phần còn lại lúc build, như X3.1 giải thích.">
            An example: the only op-specific source of PyTorch’s CUDA add kernel, its scalar body; torchgen generates the rest at build time, as X3.1 explains.
          </Translated>
        }
      />

      <h2><Translated vi="Mốc dự án">The milestone</Translated></h2>
      <Translated
        vi={
          <p>
            Mốc dự án M10 ghép các mốc trước thành một bước decode kiểu Llama, thông qua lời giải tham chiếu của các lab tương
            ứng: luôn có M0, M2, M4, M7 và M8, còn M3 và M5 khi có lời giải của các lab đó (GEMM ngây thơ của M1 và kernel prefill
            của M6 không được dùng). Nó thêm các phần khóa học chưa xây (RMSNorm, RoPE, feed-forward có cổng SiLU), và chạy bước
            đó trên một mô hình nhỏ mà lab sinh ra trong repository từ một seed cố định, được kiểm tra với một tham chiếu CPU. Quá
            trình build không tải gì về. Cách trỏ cùng pipeline đó vào một checkpoint thực tế mà bạn tự tải về được mô tả trong
            README của lab và trong mục <a href={href('x3', 'X3.8')}>X3.8</a>.
          </p>
        }
      >
        <p>
          Milestone M10 composes the earlier milestones into one Llama-style decode step, through their labs' reference
          solutions: M0, M2, M4, M7 and M8 always, M3 and M5 when their labs' solutions are present (M1's naive GEMM and M6's
          prefill kernel are not used). It adds the pieces the course has not built (RMSNorm, RoPE, the SiLU-gated
          feed-forward), and runs the step on a small model that the lab generates in the
          repository from a fixed seed, checked against a CPU reference. The build downloads nothing. Pointing the same pipeline
          at a real checkpoint you download yourself is described in the lab’s README and in section{' '}
          <a href={href('x3', 'X3.8')}>X3.8</a>.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            M10 chạy trọng số float32, và cả lớp lượng tử hóa của học phần X2 (M9, lab X2.4). Hàm khởi chạy của M9 nhận một bộ mô
            tả ma trận lượng tử hóa thay vì trọng số float mà khe lớp tuyến tính của M10 cần, nên một adapter có sẵn nối hai
            thứ đó mà không đổi bước của bạn. <a href={href('x3', 'labs')}>Trang lab</a> nói cách làm.
          </p>
        }
      >
        <p>
          M10 runs float32 weights, and also module X2’s quantised layer (M9, lab X2.4). M9’s launch function takes a
          quantised matrix descriptor rather than the float weights M10’s linear-layer slot expects, so a provided adapter
          joins the two without changing your step. The <a href={href('x3', 'labs')}>labs page</a> says how.
        </p>
      </Translated>
      <Beyond title={tr('Project note', 'Ghi chú dự án')}>
        <Translated
          vi={
            <p>
              Ghi chú dự án trong tệp dữ kiện liệt kê những gì M0 đến M8 cung cấp và những gì còn thiếu cho một bước Llama:
              RMSNorm, RoPE, SwiGLU, và một kích thước head khác 128 (không checkpoint thực tế nhỏ nào có 128).
            </p>
          }
        >
          <p>
            The facts file’s project notes list what M0 to M8 provide and what is missing for a Llama step: RMSNorm, RoPE,
            SwiGLU, and a head size other than 128 (none of the small real checkpoints has 128).
          </p>
        </Translated>
      </Beyond>
      <ul>
        <Translated vi={<li><b>Lab X3.1</b>, các phần Llama mà khóa học chưa xây, và một bài review bản port.</li>}>
          <li><b>Lab X3.1</b>, the Llama pieces the course has not built, and a port review.</li>
        </Translated>
        <Translated vi={<li><b>Lab X3.2</b>, mốc dự án M10: một mô hình kiểu Llama chạy trọn vẹn trên các kernel của khóa học.</li>}>
          <li><b>Lab X3.2</b>, milestone M10: a Llama-style model end to end on the course’s kernels.</li>
        </Translated>
      </ul>
      <Translated vi={<p>Cả hai đều ở <a href={href('x3', 'labs')}>trang lab</a>, mỗi lab có một dự đoán để bạn đưa ra trước khi đo.</p>}>
        <p>Both are on the <a href={href('x3', 'labs')}>labs page</a>, each with a prediction to make before you measure.</p>
      </Translated>

      <h2><Translated vi="Trước khi bắt đầu">Before you start</Translated></h2>
      <Translated
        vi={
          <p>
            Chín câu hỏi từ các chương trước, về những điều học phần này giả định: phép rút gọn hai giai đoạn và các rào chắn của
            chúng, quy tắc phá hòa, độ rộng warp, ghi vector, RoPE trong KV cache, online softmax, trần của GEMV, chế độ toán học
            của Metal, và các trường hợp hòa greedy giữa các backend. Mỗi câu liên kết tới nơi nó được hỏi lần đầu.
          </p>
        }
      >
        <p>
          Nine questions from earlier chapters, on the things this module assumes: two-stage reductions and their barriers, tie
          rules, warp width, vector stores, RoPE in a KV cache, the online softmax, the GEMV ceiling, Metal’s math mode, and
          greedy ties across backends. Each links to where it was first asked.
        </p>
      </Translated>
      <Bench
        title={tr('Prerequisite check', 'Kiểm tra kiến thức nền')}
        task={tr('Answer each; wrong answers come back in review.', 'Trả lời từng câu; câu trả lời sai sẽ quay lại trong phần ôn tập.')}
      >
        {PRIOR.map((p) => <PriorAsk key={p.id} id={p.id} label={p.label} />)}
      </Bench>
    </>
  );
}
