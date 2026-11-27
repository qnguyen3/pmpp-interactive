import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { solutionTabStatus } from '../../components/LabKit';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BROKEN,
  GARBAGE,
  HUNT_FILE,
  HUNT_PIXELS,
  HUNT_THREADS,
  SCHEDULES,
  SCHEDULE_LABEL,
  SCHEMES,
  SCHEME_LABEL,
  SITE_TEXT,
  VARIANT,
  VOCABS,
  caughtBy,
  diffLines,
  expectedBins,
  kib,
  makeValues,
  plan,
  races,
  region,
  simulate,
  siteLine,
  type Plan,
  type Scheme,
  type Variant,
} from '../../sim/histpriv';
import { functionSource, type FunctionSource } from '../../sim/toolchain';
import type { ImageKind } from '../../sim/histogram';
import { PrivatiseScene } from './PrivatiseScene';

const LAB1 = 'ch09/lab1_racehunt';
const LAB2 = 'ch09/lab2_privatised';
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const int = (v: number) => v.toLocaleString('en-US');
/** Table cells that may wrap on a narrow screen (.threads cells do not by default). */
const WRAP = { whiteSpace: 'normal' } as const;

// Identifiers coloured by the memory they live in.
const TIERS: Record<string, Tier> = {
  image: 'global', bins: 'global', binsPool: 'global', binsPriv: 'global', bins_pool: 'global', bins_priv: 'global',
  bins_s: 'shared', binsS: 'shared', localBins: 'shared',
  i: 'reg', b: 'reg', bin: 'reg', count: 'reg', binValue: 'reg', segment: 'reg', c: 'reg',
  tid: 'reg', bid: 'reg', bdim: 'reg', coarseFactor: 'reg',
};

/** Real GPUs only: realBackends() leaves out 'emu' but not the race-detection emulator. */
const gpuBackends = (r: Reports) => realBackends(r).filter((b) => b !== 'race');

function bytesText(b: number): string {
  if (b < 1024) return `${int(b)} B`;
  if (b < 1024 * 1024) return `${fmt(b / 1024, 4)} KiB`; // i18n-ignore: number and unit
  return `${fmt(b / (1024 * 1024), 4)} MiB`; // i18n-ignore: number and unit
}

/** The per-block on-chip budget each of the reader's GPUs reported (doctor / verify.sh). */
function useBudgets() {
  const { reports } = useReports();
  return useMemo(
    () =>
      gpuBackends(reports).flatMap((b) => {
        const d = reports[b].device;
        const metal = 'maxThreadgroupMemoryLength' in d;
        const bytes = Number(metal ? d.maxThreadgroupMemoryLength : d.sharedMemPerBlock);
        if (!(bytes > 0)) return [];
        const field = metal ? 'maxThreadgroupMemoryLength' : 'sharedMemPerBlock';
        return [{ id: b, device: String(d.name ?? b), field, bytes, label: `${String(d.name ?? b)} (${b}): ${field} ${int(bytes)} B` }];
      }),
    [reports],
  );
}

function useListingStatus(lab: string) {
  const { reports } = useReports();
  // The same words as the labs page's status table and reference-solution tabs.
  return { cu: solutionTabStatus(reports, lab, 'kernel.cu'), metal: solutionTabStatus(reports, lab, 'kernel.metal') };
}

// ---------------------------------------------------------------- listings from the labs
function LabRegion({ lab, name, what, caption }: { lab: string; name: string; what: string; caption: ReactNode }) {
  const cu = useLabFile(`${lab}/solution/kernel.cu`);
  const metal = useLabFile(`${lab}/solution/kernel.metal`);
  const status = useListingStatus(lab);
  const k = missing(cu) ? null : region(cu!, name);
  const m = missing(metal) ? null : region(metal!, name);
  if (!k)
    return (
      <Translated
        vi={
          <p className="muted" data-listing="missing">
            {what} được nạp từ <code>labs/{lab}/solution/kernel.cu</code> (vùng <code>{name}</code>) khi lab đã có
            trên máy này.
          </p>
        }
      >
        <p className="muted" data-listing="missing">
          {what} is loaded from <code>labs/{lab}/solution/kernel.cu</code> (region <code>{name}</code>) once the lab is
          on this machine.
        </p>
      </Translated>
    );
  return (
    <Code
      tabs={[
        { label: tr('CUDA and HIP', 'CUDA và HIP'), code: k.text, tiers: TIERS, status: status.cu },
        ...(m ? [{ label: 'Metal', code: m.text, tiers: TIERS, status: status.metal }] : []),
      ]}
      caption={caption}
    />
  );
}

// ---------------------------------------------------------------- explorer: copies and contenders
const KINDS: ImageKind[] = ['uniform', 'bright', 'constant'];
const KIND_LABEL = (kind: ImageKind, tokens: boolean) =>
  tokens
    ? {
        uniform: tr('uniform token ids', 'token id phân bố đều'),
        bright: tr('skewed token ids (a few very frequent)', 'token id lệch (vài token rất thường gặp)'),
        constant: tr('one token repeated', 'một token lặp lại'),
      }[kind]
    : {
        uniform: tr('uniform noise', 'nhiễu phân bố đều'),
        bright: tr('a bright, skewed image', 'một ảnh sáng, lệch'),
        constant: tr('a constant image (every pixel 255)', 'một ảnh hằng (mọi điểm ảnh bằng 255)'),
      }[kind];
const PIXEL_CHOICES = [4096, 65_536, 524_288];
const TPB_CHOICES = [64, 128, 256, 512, 1024];
type BinsChoice = '256' | '128' | 'custom' | (typeof VOCABS)[number]['id'];
const SCHEME_SHORT: Record<Scheme, string> = {
  get public() {
    return tr('one public copy', 'một bản công khai');
  },
  get two() {
    return tr('two copies', 'hai bản sao');
  },
  get four() {
    return tr('four copies', 'bốn bản sao');
  },
  get block() {
    return tr('one per block', 'một bản mỗi khối');
  },
};

function Explorer() {
  const budgets = useBudgets();
  const [kind, setKind] = useState<ImageKind>('bright');
  const [n, setN] = useState(65_536);
  const [tpb, setTpb] = useState(256);
  const [scheme, setScheme] = useState<Scheme>('block');
  const [binsChoice, setBinsChoice] = useState<BinsChoice>('256');
  const [customBins, setCustomBins] = useState(1000);
  const [budgetId, setBudgetId] = useState<string>('auto');
  const [customKiB, setCustomKiB] = useState(48);

  const vocab = VOCABS.find((v) => v.id === binsChoice);
  const bins = vocab ? vocab.bins : binsChoice === 'custom' ? customBins : Number(binsChoice);
  const chosen = budgetId === 'auto' ? (budgets[0]?.id ?? 'custom') : budgetId;
  const device = budgets.find((b) => b.id === chosen);
  const budget = device ? device.bytes : customKiB * 1024;
  const values = useMemo(() => makeValues(kind, n, bins), [kind, n, bins]);
  const plans = useMemo(() => SCHEMES.map((s) => plan(values, tpb, s, 1, bins, 4, budget)), [values, tpb, bins, budget]);
  const p = plans[SCHEMES.indexOf(scheme)];
  const pub = plans[0];
  const tokens = bins > 256;

  return (
    <Bench
      title={tr('Copies and contenders', 'Bản sao và các luồng tranh chấp')}
      task={
        <Translated vi="Dự đoán trước: khi chuyển từ một bản công khai sang một bản mỗi khối, hàng đợi dài nhất và bước ghi (commit) thay đổi thế nào?">
          Predict first: going from one public copy to one per block, what happens to the longest queue, and to the commit?
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Số lượng công việc từ <code>src/sim/histpriv.ts</code>, không phải thời gian. Các điểm ảnh và token id được sinh
              từ seed cố định và là của khóa học này. Ngân sách là những gì thiết bị của bạn báo cho{' '}
              <code>python3 labs/pmpp.py doctor</code>, hoặc một giá trị bạn nhập.
            </>
          }
        >
          <>
            Counts of work from <code>src/sim/histpriv.ts</code>, not times. The pixels and token ids are seeded and ours.
            Budgets are what your devices reported to <code>python3 labs/pmpp.py doctor</code>, or one you enter.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Đầu vào">Input</Translated>
          <select value={kind} aria-label={tr('input', 'đầu vào')} onChange={(e) => setKind(e.target.value as ImageKind)}>
            {KINDS.map((k) => <option key={k} value={k}>{KIND_LABEL(k, tokens)}</option>)}
          </select>
        </label>
        <label className="field">
          {tokens ? tr('Tokens', 'Token') : tr('Pixels', 'Điểm ảnh')}
          <select value={n} aria-label={tr('elements', 'số phần tử')} onChange={(e) => setN(Number(e.target.value))}>
            {PIXEL_CHOICES.map((x) => <option key={x} value={x}>{int(x)}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số luồng mỗi khối">Threads per block</Translated>
          <select value={tpb} aria-label={tr('threads per block', 'số luồng mỗi khối')} onChange={(e) => setTpb(Number(e.target.value))}>
            {TPB_CHOICES.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số bin">Bins</Translated>
          <select value={binsChoice} aria-label={tr('bins', 'số bin')} onChange={(e) => setBinsChoice(e.target.value as BinsChoice)}>
            <option value="256">{tr('256 (one per 8-bit value)', '256 (một cho mỗi giá trị 8-bit)')}</option>
            <option value="128">{tr('128 (exercise 5)', '128 (bài tập 5)')}</option>
            <option value="custom">{tr('custom', 'tùy chỉnh')}</option>
            {VOCABS.map((v) => <option key={v.id} value={v.id}>{v.name}{tr(' vocabulary: ', ' (bộ từ vựng): ')}{int(v.bins)}</option>)}
          </select>
        </label>
        {binsChoice === 'custom' ? (
          <label className="field">
            <Translated vi="Số bin tùy chỉnh">Custom bins</Translated>
            <input type="number" min={1} max={1_000_000} value={customBins} aria-label={tr('custom bins', 'số bin tùy chỉnh')}
              onChange={(e) => { const v = Math.floor(Number(e.target.value)); if (v >= 1 && v <= 1_000_000) setCustomBins(v); }} />
          </label>
        ) : null}
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Ngân sách trên chip mỗi khối">On-chip budget per block</Translated>
          <select value={chosen} aria-label={tr('budget', 'ngân sách')} style={{ maxWidth: '100%' }} onChange={(e) => setBudgetId(e.target.value)}>
            {budgets.map((b) => <option key={b.id} value={b.id}>{b.label}</option>)}
            <option value="custom">{tr('custom', 'tùy chỉnh')}</option>
          </select>
        </label>
        {chosen === 'custom' ? (
          <label className="field">
            <Translated vi="Ngân sách tùy chỉnh (KiB)">Custom budget (KiB)</Translated>
            <input type="number" min={1} value={customKiB} aria-label={tr('custom budget', 'ngân sách tùy chỉnh')}
              onChange={(e) => { const v = Number(e.target.value); if (v > 0) setCustomKiB(v); }} />
            <Translated
              vi={
                <span className="muted" style={{ fontSize: '0.8rem' }} data-part="custom-default">
                  Bắt đầu ở 48: mặc định mỗi khối mà CUDA ghi trong tài liệu khi không opt-in (ch09-facts A8). Hãy nhập con số{' '}
                  <code>doctor</code> của bạn thay vào.
                </span>
              }
            >
              <span className="muted" style={{ fontSize: '0.8rem' }} data-part="custom-default">
                Starts at 48: CUDA's documented per-block default without opt-in (ch09-facts A8). Enter your{' '}
                <code>doctor</code> figure instead.
              </span>
            </Translated>
          </label>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="muted"><Translated vi="Ai dùng chung một bản sao:">Who shares a copy:</Translated></span>
        <span className="seg" role="group" aria-label={tr('copy scheme', 'cách chia bản sao')}>
          {SCHEMES.map((s) => (
            <button key={s} aria-pressed={s === scheme} onClick={() => setScheme(s)}>{SCHEME_SHORT[s]}</button>
          ))}
        </span>
      </div>

      <div className="panel" data-readout data-scheme={scheme} data-contention={p.updateContention} data-commit={p.commitAtomics}
        data-commit-contention={p.commitContention} data-copies={p.copies} data-fits={p.fits ? 'yes' : 'no'} style={{ marginBottom: '0.9rem' }}>
        <Translated
          vi={
            <p style={{ margin: '0 0 0.5rem' }}>
              <b>{SCHEME_LABEL[scheme]}.</b> {int(n)} {tokens ? 'token' : 'điểm ảnh'} trong {int(p.blocks)} khối, mỗi khối {tpb} luồng,{' '}
              {int(bins)} bin.
            </p>
          }
        >
          <p style={{ margin: '0 0 0.5rem' }}>
            <b>{SCHEME_LABEL[scheme]}.</b> {int(n)} {tokens ? 'tokens' : 'pixels'} in {int(p.blocks)} blocks of {tpb} threads,{' '}
            {int(bins)} bins.
          </p>
        </Translated>
        <table className="threads wrap" aria-label={tr('This scheme', 'Cách chia này')}>
          <tbody>
            <tr>
              <td><Translated vi="số bản sao nhận cập nhật">copies the updates go to</Translated></td>
              <td><b>{int(p.copies)}</b>{scheme === 'public' ? tr(' (the public histogram itself)', ' (chính histogram công khai)') : tr(`, ${bytesText(p.privateBytes)} of private counters in all`, `, tổng cộng ${bytesText(p.privateBytes)} bộ đếm riêng`)}</td>
            </tr>
            <tr>
              <td><Translated vi="hàng đợi dài nhất trong pha cập nhật">longest queue in the update phase</Translated></td>
              <Translated
                vi={
                  <td>
                    <b className={scheme === 'public' ? 't-global' : p.where === 'shared' ? 't-shared' : 't-global'}>{int(p.updateContention)}</b> cập nhật
                    nhắm vào một bộ đếm (bin {int(p.hottest.bin)}{scheme === 'public' ? '' : ` của bản sao ${int(p.hottest.copy)}`})
                    {scheme !== 'public' && pub.updateContention > p.updateContention ? <>, ngắn hơn {fmt(pub.updateContention / p.updateContention)}× so với khi chỉ có một bản công khai</> : null}
                  </td>
                }
              >
                <td>
                  <b className={scheme === 'public' ? 't-global' : p.where === 'shared' ? 't-shared' : 't-global'}>{int(p.updateContention)}</b> updates
                  aimed at one counter (bin {int(p.hottest.bin)}{scheme === 'public' ? '' : ` of copy ${int(p.hottest.copy)}`})
                  {scheme !== 'public' && pub.updateContention > p.updateContention ? <>, {fmt(pub.updateContention / p.updateContention)}× shorter than with one public copy</> : null}
                </td>
              </Translated>
            </tr>
            <tr>
              <td><Translated vi="phép toán nguyên tử toàn cục trong bước ghi">global atomics in the commit</Translated></td>
              <Translated
                vi={
                  <td>
                    {scheme === 'public' ? 'không có: không có gì để trộn' : (
                      <><b>{int(p.commitAtomics)}</b> khi có kiểm tra bằng không (nhiều nhất bản sao × bin = {int(p.commitBound)})</>
                    )}
                  </td>
                }
              >
                <td>
                  {scheme === 'public' ? 'none: there is nothing to merge' : (
                    <><b>{int(p.commitAtomics)}</b> with the zero test (at most copies × bins = {int(p.commitBound)})</>
                  )}
                </td>
              </Translated>
            </tr>
            <tr>
              <td><Translated vi="số luồng tranh chấp trên mỗi bin công khai trong bước ghi">contenders per public bin in the commit</Translated></td>
              <Translated vi={<td>{scheme === 'public' ? 'không có' : <>nhiều nhất <b>{int(p.commitContention)}</b>, một cho mỗi bản sao đã gặp bin đó</>}</td>}>
                <td>{scheme === 'public' ? 'none' : <>at most <b>{int(p.commitContention)}</b>, one per copy that saw the bin</>}</td>
              </Translated>
            </tr>
            <tr>
              <td><Translated vi="một bản sao trên chip">one copy on chip</Translated></td>
              <Translated
                vi={
                  <td>
                    {int(bins)} × 4 byte = {bytesText(p.copyBytes)} so với {bytesText(budget)}:{' '}
                    {p.fits ? <span className="chip ok">vừa</span> : <span className="chip fail">không vừa</span>}
                    {scheme === 'block' ? (p.fits ? <> nên bản sao của mỗi khối có thể nằm trong bộ nhớ chia sẻ (Hình 9.10)</> : <> nên các bản sao nằm lại trong bộ nhớ toàn cục (Hình 9.9)</>) : null}
                  </td>
                }
              >
                <td>
                  {int(bins)} × 4 bytes = {bytesText(p.copyBytes)} against {bytesText(budget)}:{' '}
                  {p.fits ? <span className="chip ok">fits</span> : <span className="chip fail">does not fit</span>}
                  {scheme === 'block' ? (p.fits ? <> so each block's copy can live in shared memory (Fig. 9.10)</> : <> so the copies stay in global memory (Fig. 9.9)</>) : null}
                </td>
              </Translated>
            </tr>
          </tbody>
        </table>
      </div>

      <table className="data" style={{ background: 'var(--page)', margin: 0 }} aria-label={tr('All four schemes', 'Cả bốn cách chia')}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Ai dùng chung một bản sao">Who shares a copy</Translated></th>
            <th className="num"><Translated vi="bản sao">copies</Translated></th>
            <th className="num"><Translated vi="hàng đợi dài nhất">longest queue</Translated></th>
            <th className="num"><Translated vi="nguyên tử khi ghi">commit atomics</Translated></th>
            <th className="num"><Translated vi="mỗi bin công khai">per public bin</Translated></th>
            <th className="num"><Translated vi="bộ đếm riêng">private counters</Translated></th>
          </tr>
        </thead>
        <tbody>
          {plans.map((q: Plan) => (
            <tr key={q.scheme} data-row={q.scheme} style={q.scheme === scheme ? { background: 'var(--rule-soft)', fontWeight: 600 } : undefined}>
              <td style={{ paddingLeft: '0.6rem' }}>{SCHEME_SHORT[q.scheme]}</td>
              <td className="num">{int(q.copies)}</td>
              <td className="num">{int(q.updateContention)}</td>
              <td className="num">{q.scheme === 'public' ? '·' : int(q.commitAtomics)}</td>
              <td className="num">{q.scheme === 'public' ? '·' : int(q.commitContention)}</td>
              <td className="num">{q.scheme === 'public' ? '·' : bytesText(q.privateBytes)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- bug hunt
type Tool = 'emu' | 'race' | 'gpu' | 'none';
const TOOLS: { id: Tool; label: string }[] = [
  { id: 'emu', label: 'emu' },
  {
    id: 'race',
    get label() {
      return tr('race backend', 'backend race');
    },
  },
  {
    id: 'gpu',
    get label() {
      return tr('a GPU run', 'một lần chạy trên GPU');
    },
  },
  {
    id: 'none',
    get label() {
      return tr('nothing here', 'không có gì ở đây');
    },
  },
];

type Outcome = { runs: ReturnType<typeof simulate>[]; races: ReturnType<typeof races>; caught: { emu: boolean; race: boolean } };
const binsList = (bs: number[]) =>
  bs.length === 1
    ? tr(`bin ${bs[0]}`, `bin ${bs[0]}`)
    : tr(`bins ${bs.slice(0, -1).join(', ')} and ${bs[bs.length - 1]}`, `các bin ${bs.slice(0, -1).join(', ')} và ${bs[bs.length - 1]}`);

/** What each reveal says. Numbers that depend on the hunt's pixels come from the model's results. */
function why(v: Variant, r: Outcome): string {
  const wrongSets = r.runs.map((x) => x.wrong);
  const ex = r.races[0]?.example;
  switch (v) {
    case 'no-zeroing':
      return tr(
        `Every private bin starts at the emulator's leftover 0x${GARBAGE.toString(16).toUpperCase()}, so both blocks commit leftovers plus counts and ${wrongSets[0].length} of 8 public bins are wrong in every order. The race backend has nothing to report: every access is ordered by a barrier or atomic.`,
        `Mọi bin riêng bắt đầu từ giá trị sót lại 0x${GARBAGE.toString(16).toUpperCase()} của trình giả lập, nên cả hai khối ghi giá trị sót lại cộng với số đếm và ${wrongSets[0].length} trên 8 bin công khai sai ở mọi thứ tự. Backend race không có gì để báo cáo: mọi truy cập đều được sắp thứ tự bởi một rào chắn hoặc một phép toán nguyên tử.`,
      );
    case 'no-barrier-after-zeroing':
      return tr(
        'A thread can count into a bin that another thread zeroes afterwards, which wipes the count. Which counts are lost depends on the order. The zeroing store and another thread\'s atomic update of the same bin are unordered: a race.',
        'Một luồng có thể đếm vào một bin mà một luồng khác xóa về không sau đó, làm mất số đếm. Số đếm nào bị mất phụ thuộc vào thứ tự. Lần ghi xóa về không và phép cập nhật nguyên tử của luồng khác trên cùng bin không có thứ tự: một race.',
      );
    case 'no-barrier-before-commit':
      return tr(
        'A thread can commit a bin before another thread of its block has counted into it, and that late count is never committed. Which counts are lost depends on the order. The commit\'s plain read and another thread\'s atomic update are unordered: a race.',
        'Một luồng có thể ghi một bin trước khi một luồng khác trong khối của nó đếm vào bin đó, và số đếm muộn đó không bao giờ được ghi. Số đếm nào bị mất phụ thuộc vào thứ tự. Lần đọc thường của bước ghi và phép cập nhật nguyên tử của luồng khác không có thứ tự: một race.',
      );
    case 'plain-shared-update':
      return tr(
        `In emu no thread is interrupted inside ++, so every order gives the right answer. Threads ${ex?.a.thread} and ${ex?.b.thread} of block ${ex?.a.block} both count into bin ${ex?.addr} with a plain read-modify-write between the same two barriers: a race, which is what the race backend is for. A caveat: on this tiny launch the real race backend misses it in most runs, a limit of the emulator's barrier bookkeeping. On blocks of 256 threads, the size lab 9.2 launches, it reported this line in every one of our runs.`,
        `Trong emu không luồng nào bị ngắt giữa ++, nên mọi thứ tự đều cho đáp án đúng. Luồng ${ex?.a.thread} và ${ex?.b.thread} của khối ${ex?.a.block} cùng đếm vào bin ${ex?.addr} bằng một thao tác đọc-sửa-ghi thường giữa cùng hai rào chắn: một race, đúng là thứ backend race sinh ra để bắt. Một lưu ý: với lần khởi chạy nhỏ xíu này, backend race thật bỏ sót nó trong phần lớn các lần chạy, một giới hạn của cách trình giả lập theo dõi rào chắn. Với các khối 256 luồng, kích thước mà lab 9.2 khởi chạy, nó đã báo dòng này trong mọi lần chạy của chúng tôi.`,
      );
    case 'plain-commit':
      return tr(
        `Blocks ${ex?.a.block} and ${ex?.b.block} both commit to bin ${ex?.addr} with a plain +=. Blocks run one after another in emu, so the answer is right; nothing orders two blocks, so the race backend reports it.`,
        `Khối ${ex?.a.block} và ${ex?.b.block} cùng ghi vào bin ${ex?.addr} bằng một lệnh += thường. Trong emu các khối chạy lần lượt, nên đáp án đúng; không gì sắp thứ tự hai khối, nên backend race báo cáo nó.`,
      );
    case 'block-scope-commit':
      return tr(
        'Right in every order and clean on the race backend, because neither emulator gives scopes any effect, and Metal has no scope to get wrong. Under CUDA\'s memory model it is still a data race, and so undefined behaviour: two blocks add to one public bin with an atomic whose scope does not include the other block.',
        'Đúng ở mọi thứ tự và sạch trên backend race, vì cả hai trình giả lập đều không cho phạm vi tác dụng gì, và Metal không có phạm vi để mà viết sai. Theo mô hình bộ nhớ của CUDA nó vẫn là một data race, và do đó là hành vi không xác định: hai khối cộng vào một bin công khai bằng một phép toán nguyên tử có phạm vi không bao gồm khối kia.',
      );
    case 'commit-bound-blockDim':
      return tr(
        `With 8 bins and ${HUNT_THREADS} threads, the loop stops at blockDim.x, so bins ${HUNT_THREADS} to 7 are never committed. ${binsList(wrongSets[0]).replace(/^b/, 'B')} had counts: ${wrongSets[0].length} bins wrong, in every order and on every device. Nothing races.`,
        `Với 8 bin và ${HUNT_THREADS} luồng, vòng lặp dừng ở blockDim.x, nên các bin từ ${HUNT_THREADS} đến 7 không bao giờ được ghi. ${binsList(wrongSets[0]).replace(/^([bc])/, (c) => c.toUpperCase())} đã có số đếm: ${wrongSets[0].length} bin sai, ở mọi thứ tự và trên mọi thiết bị. Không có race nào.`,
      );
    default:
      return '';
  }
}

function BugHunt() {
  const src = useLabFile(HUNT_FILE);
  const kernels = useMemo(() => {
    if (missing(src)) return null;
    const out = Object.fromEntries((['correct', ...BROKEN] as Variant[]).map((v) => [v, functionSource(src!, VARIANT[v].kernel)]));
    return Object.values(out).every(Boolean) ? (out as Record<Variant, FunctionSource>) : null;
  }, [src]);
  const results = useMemo(
    () => Object.fromEntries(BROKEN.map((v) => [v, { runs: SCHEDULES.map((s) => simulate(v, s)), races: races(v), caught: caughtBy(v) }])) as Record<Variant, Outcome>,
    [],
  );
  const [ticks, setTicks] = useState<Record<string, Tool[]>>({});
  const [shown, setShown] = useState<Record<string, boolean>>({});
  const expected = expectedBins();

  if (!kernels)
    return (
      <Translated
        vi={
          <p className="muted" data-hunt="missing">
            Các kernel của cuộc săn lỗi được nạp từ <code>labs/{HUNT_FILE}</code>.
          </p>
        }
      >
        <p className="muted" data-hunt="missing">
          The hunt's kernels are loaded from <code>labs/{HUNT_FILE}</code>.
        </p>
      </Translated>
    );
  const ref = kernels.correct;
  const toggle = (v: Variant, t: Tool) =>
    setTicks((m) => {
      const cur = m[v] ?? [];
      return { ...m, [v]: cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t] };
    });
  const lineOf = (v: Variant, site: Parameters<typeof siteLine>[1]) => kernels[v].firstLine + siteLine(kernels[v].text, site);
  const raceText = (v: Variant) => {
    const rs = results[v].races;
    if (!rs.length) return tr('clean', 'sạch');
    return rs
      .map((r) => {
        const e = r.example;
        const where = r.space === 'shared'
          ? tr(`bins_s[${e.addr}] of block ${e.a.block}, threads ${e.a.thread} and ${e.b.thread}`, `bins_s[${e.addr}] của khối ${e.a.block}, luồng ${e.a.thread} và ${e.b.thread}`)
          : tr(`bins[${e.addr}], blocks ${e.a.block} and ${e.b.block}`, `bins[${e.addr}], khối ${e.a.block} và ${e.b.block}`);
        const [a, b] = r.sites;
        return a === b
          ? tr(
              `reported: ${SITE_TEXT[a]} (line ${lineOf(v, a)}) against itself in another thread, e.g. ${where}`,
              `đã báo: ${SITE_TEXT[a]} (dòng ${lineOf(v, a)}) với chính nó ở một luồng khác, ví dụ ${where}`,
            )
          : tr(
              `reported: ${SITE_TEXT[a]} (line ${lineOf(v, a)}) and ${SITE_TEXT[b]} (line ${lineOf(v, b)}), e.g. ${where}`,
              `đã báo: ${SITE_TEXT[a]} (dòng ${lineOf(v, a)}) và ${SITE_TEXT[b]} (dòng ${lineOf(v, b)}), ví dụ ${where}`,
            );
      })
      .join('; ');
  };
  const emuCell = (v: Variant, k: number) => {
    const r = results[v].runs[k];
    return r.correct ? (
      <span className="chip ok"><Translated vi="đúng">correct</Translated></span>
    ) : (
      <Translated vi={<span className="chip fail">{r.wrong.length} trên 8 bin sai</span>}>
        <span className="chip fail">{r.wrong.length} of 8 bins wrong</span>
      </Translated>
    );
  };

  return (
    <Bench
      title={tr('Seven broken privatised kernels', 'Bảy kernel riêng tư hóa bị hỏng')}
      task={
        <Translated vi="Với mỗi kernel, đánh dấu những công cụ bạn nghĩ sẽ cho thấy lỗi, rồi bấm hiện đáp án.">
          For each kernel, tick the tools you expect to show the defect, then reveal.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Các kernel từ <code>labs/{HUNT_FILE}</code> (front-end-checked). Cột emu và race lấy từ{' '}
              <code>src/sim/histpriv.ts</code>, mà bài kiểm tra đơn vị của nó chạy tệp này trên cả hai trình giả lập và so sánh. Câu về GPU
              là lập luận của chúng tôi từ tài liệu nhà cung cấp, nên không được chấm điểm.
            </>
          }
        >
          <>
            Kernels from <code>labs/{HUNT_FILE}</code> (front-end-checked). The emu and race columns come from{' '}
            <code>src/sim/histpriv.ts</code>, whose unit test runs this file on both emulators and compares. The GPU sentence
            is our reasoning from vendor documents, so it is not scored.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-start' }}>
        <div>
          <div className="mx-label t-global"><Translated vi="ảnh (số bin): khối 0 | khối 1">image (bin numbers): block 0 | block 1</Translated></div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${HUNT_PIXELS.length}, auto)` }}>
            {HUNT_PIXELS.map((v, i) => <span key={i} style={i === HUNT_THREADS ? { marginLeft: '0.6rem' } : undefined}>{v}</span>)}
          </div>
        </div>
        <div>
          <div className="mx-label t-global"><Translated vi="các bin công khai đúng, từ 0 đến 7">the right public bins 0 to 7</Translated></div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${expected.length}, auto)` }}>
            {expected.map((v, i) => <span key={i}>{v}</span>)}
          </div>
        </div>
      </div>
      <Code
        tabs={[{ label: tr('histPrivatised (the reference)', 'histPrivatised (bản tham chiếu)'), code: ref.text, tiers: TIERS, status: 'front-end-checked; emulator-checked' }]}
        firstLine={ref.firstLine}
        caption={
          <Translated vi={<>Hình 9.10 thu nhỏ còn 8 bin và một điểm ảnh mỗi luồng, khởi chạy thành 2 khối 4 luồng. Số dòng là số dòng của tệp.</>}>
            <>Fig. 9.10 shrunk to 8 bins and one pixel per thread, launched as 2 blocks of 4 threads. Line numbers are those of the file.</>
          </Translated>
        }
      />
      <div style={{ display: 'grid', gap: '0.8rem', marginTop: '0.4rem' }}>
        {BROKEN.map((v) => {
          const d = diffLines(ref.text, kernels[v].text);
          const mine = ticks[v] ?? [];
          const open = !!shown[v];
          const c = results[v].caught;
          const truth = { emu: c.emu, race: c.race, none: !c.emu && !c.race };
          const graded = (['emu', 'race', 'none'] as const).filter((t) => mine.includes(t) !== truth[t]);
          return (
            <div key={v} className="panel" data-variant={v} data-revealed={open ? 'yes' : 'no'}>
              <p style={{ margin: '0 0 0.4rem' }}>
                <b><Translated vi="Kernel">Kernel</Translated> {VARIANT[v].label}</b>
                {open ? <> <code>{VARIANT[v].kernel}</code></> : null}
              </p>
              <div className="mono" style={{ fontSize: '0.8rem', marginBottom: '0.5rem', overflowX: 'auto' }}>
                {d.removed.map((l, k) => (
                  <div key={'r' + k}>
                    <Translated vi={<span className="muted">{d.added.length ? 'thay' : 'bỏ'} dòng {ref.firstLine + d.removedAt[k] - 1}: </span>}>
                      <span className="muted">{d.added.length ? 'replaces' : 'removes'} line {ref.firstLine + d.removedAt[k] - 1}: </span>
                    </Translated>
                    <code>{l}</code>
                  </div>
                ))}
                {d.added.map((l, k) => (
                  <div key={'a' + k}>
                    <span className="muted">{d.removed.length ? tr('with', 'bằng') : tr('adds', 'thêm')}: </span>
                    <code>{l}</code>
                  </div>
                ))}
              </div>
              <div className="row">
                <span className="muted"><Translated vi="Cho thấy lỗi:">Shows it:</Translated></span>
                {TOOLS.map((t) => (
                  <label key={t.id} className="check">
                    <input type="checkbox" checked={mine.includes(t.id)} disabled={open} onChange={() => toggle(v, t.id)} /> {t.label}
                  </label>
                ))}
                <button onClick={() => setShown((m) => ({ ...m, [v]: true }))} disabled={open}><Translated vi="Hiện đáp án">Reveal</Translated></button>
              </div>
              {open ? (
                <div style={{ marginTop: '0.6rem' }} data-outcome-detail>
                  <table className="threads" aria-label={tr(`Outcome of kernel ${VARIANT[v].label}`, `Kết quả của kernel ${VARIANT[v].label}`)}>
                    <tbody>
                      {SCHEDULES.map((s, k) => (
                        <tr key={s} data-order={s}>
                          <td>emu, {SCHEDULE_LABEL[s]}{/* i18n-ignore: backend name */}</td>
                          <td style={WRAP}>
                            {emuCell(v, k)}{' '}
                            {results[v].runs[k].wrong.length && results[v].runs[k].wrong.length < 8 ? (
                              <span className="muted">
                                {results[v].runs[k].wrong
                                  .map((b) =>
                                    tr(
                                      `bin ${b} holds ${results[v].runs[k].bins[b]}, not ${expected[b]}`,
                                      `bin ${b} giữ ${results[v].runs[k].bins[b]}, không phải ${expected[b]}`,
                                    ),
                                  )
                                  .join('; ')}
                              </span>
                            ) : null}
                          </td>
                        </tr>
                      ))}
                      <tr data-order="race">
                        <td><Translated vi="backend race">race backend</Translated></td>
                        <td style={WRAP}>
                          {results[v].races.length ? <span className="chip fail">race</span> : <span className="chip ok"><Translated vi="sạch">clean</Translated></span>}{' '}
                          <span className="muted">{raceText(v)}</span>
                        </td>
                      </tr>
                      <tr>
                        <td><Translated vi="một lần chạy trên GPU">a GPU run</Translated></td>
                        <td style={WRAP}>{VARIANT[v].gpu}</td>
                      </tr>
                    </tbody>
                  </table>
                  <p style={{ margin: '0.5rem 0 0' }} data-why>{why(v, results[v])}</p>
                  <p className="muted" style={{ margin: '0.4rem 0 0' }} data-graded={mine.length === 0 ? 'none' : graded.length ? 'off' : 'right'}>
                    {mine.length === 0
                      ? tr('You ticked nothing for this one, so there is nothing to score.', 'Bạn không đánh dấu gì cho kernel này, nên không có gì để chấm.')
                      : graded.length === 0
                        ? tr('Your emu, race and "nothing here" calls were right.', 'Các lựa chọn emu, race và "không có gì ở đây" của bạn đều đúng.')
                        : tr(
                            `Not quite: check ${graded.map((t) => (t === 'none' ? '"nothing here"' : t === 'race' ? 'the race backend' : 'emu')).join(' and ')}.`,
                            `Chưa hẳn: hãy xem lại ${graded.map((t) => (t === 'none' ? '"không có gì ở đây"' : t === 'race' ? 'backend race' : 'emu')).join(' và ')}.`,
                          )}
                  </p>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="row" style={{ margin: '0.9rem 0 0.5rem' }}>
        <button className="quiet" onClick={() => setShown(Object.fromEntries(BROKEN.map((v) => [v, true])))}><Translated vi="Hiện tất cả">Reveal all</Translated></button>
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: 0 }} aria-label={tr('Outcome table', 'Bảng kết quả')}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Kernel">Kernel</Translated></th>
            {SCHEDULES.map((s) => <th key={s}>emu, {SCHEDULE_LABEL[s]}{/* i18n-ignore: backend name */}</th>)}
            <th><Translated vi="backend race">race backend</Translated></th>
          </tr>
        </thead>
        <tbody>
          {BROKEN.map((v) => (
            <tr key={v} data-summary={v}>
              <td style={{ paddingLeft: '0.6rem' }}>{VARIANT[v].label}{shown[v] ? <span className="muted"> {VARIANT[v].kernel.replace(/^hist/, '')}</span> : null}</td>
              {SCHEDULES.map((s, k) => (
                <td key={s} data-outcome={shown[v] ? (results[v].runs[k].correct ? 'correct' : `wrong-${results[v].runs[k].wrong.length}`) : 'hidden'}>
                  {shown[v] ? emuCell(v, k) : <span className="muted">?</span>}
                </td>
              ))}
              <td data-outcome={shown[v] ? (results[v].races.length ? 'race' : 'clean') : 'hidden'}>
                {shown[v] ? (results[v].races.length ? <span className="chip fail">race</span> : <span className="chip ok"><Translated vi="sạch">clean</Translated></span>) : <span className="muted">?</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- does a copy fit?
/** GPT-2's row would print the answer to q-vocab-kib, so its size waits until that is answered. */
const VOCAB_Q = 'ch09/9.4/q-vocab-kib';

function FitTable() {
  const budgets = useBudgets();
  const { progress } = useProgress();
  const answered = !!progress.answers[VOCAB_Q];
  const rows = [
    { label: tr('256 bins (an 8-bit image)', '256 bin (một ảnh 8-bit)'), bins: 256, hide: false },
    ...VOCABS.map((v) => ({ label: tr(`${v.name} vocabulary`, `bộ từ vựng ${v.name}`), bins: v.bins, hide: v.id === 'gpt2' && !answered })),
  ];
  return (
    <table className="data wide" style={{ maxWidth: '58rem' }} data-part="fit">
      <thead>
        <tr>
          <th><Translated vi="Bin">Bins</Translated></th>
          <th className="num"><Translated vi="số lượng">count</Translated></th>
          <th className="num"><Translated vi="một bản sao, bộ đếm 32-bit">one copy, 32-bit counts</Translated></th>
          {budgets.map((b) => <th key={b.id}>{b.device} ({b.id}), {bytesText(b.bytes)}</th>)}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.label} data-bins={r.bins}>
            <td>{r.label}</td>
            <td className="num">{int(r.bins)}</td>
            <td className="num" data-hidden={r.hide ? 'yes' : undefined}>
              {r.hide ? <span className="muted"><Translated vi="hãy trả lời câu hỏi ở trên trước">answer the question above first</Translated></span> : `${fmt(kib(r.bins * 4), 4)} KiB` /* i18n-ignore: number and unit */}
            </td>
            {budgets.map((b) => (
              <td key={b.id} data-fits={r.bins * 4 <= b.bytes ? 'yes' : 'no'}>
                {r.bins * 4 <= b.bytes ? <span className="chip ok"><Translated vi="vừa">fits</Translated></span> : <span className="chip fail"><Translated vi="không vừa">does not fit</Translated></span>}
              </td>
            ))}
          </tr>
        ))}
        {budgets.length === 0 ? (
          <tr>
            <Translated
              vi={
                <td colSpan={3} className="muted">
                  Chưa có báo cáo thiết bị: chạy <code>./verify.sh</code>, hoặc bất kỳ lab nào bằng <code>python3 labs/pmpp.py run</code>, và
                  ngân sách mỗi khối của các GPU của bạn sẽ hiện ở đây thành các cột.
                </td>
              }
            >
              <td colSpan={3} className="muted">
                No device report yet: run <code>./verify.sh</code>, or any lab with <code>python3 labs/pmpp.py run</code>, and
                your GPUs' per-block budgets appear here as columns.
              </td>
            </Translated>
          </tr>
        ) : null}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------- the measured speed-up
function MeasuredSharedSpeedup({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = gpuBackends(reports).flatMap((b) => {
    const rows = benchRows(reports[b], LAB2);
    const basic = num(rows.find((r) => r.name === 'basic' && r.data === 'constant')?.gpixels_per_s);
    const shared = num(rows.find((r) => r.name === 'shared' && num(r.coarse) === 16 && r.data === 'constant')?.gpixels_per_s);
    return basic && shared ? [{ b, device: String(reports[b].device.name ?? b), basic, shared }] : [];
  });
  if (!found.length)
    return (
      <Translated
        vi={
          <p>
            Chưa có số đo nào trên máy này. Chạy <code>python3 labs/pmpp.py run 9.2 --solution --bench</code>, lệnh này
            ghi lại số đo, rồi tải lại trang: ô này so sánh dự đoán của bạn với những gì GPU của bạn đã làm.
          </p>
        }
      >
        <p>
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 9.2 --solution --bench</code>, which
          records it, then reload this page: this box compares your prediction with what your GPU did.
        </p>
      </Translated>
    );
  return (
    <>
      {found.map((f) => {
        const s = f.shared / f.basic;
        return (
          <Translated
            key={f.b}
            vi={
              <p data-speedup={s}>
                Trên <b>{f.device}</b> ({f.b}), với mọi điểm ảnh như nhau, kernel bộ nhớ chia sẻ (hệ số gộp việc 16) chạy ở{' '}
                {fmt(f.shared)} Gpixel/s so với {fmt(f.basic)} của kernel cơ bản: <b>{fmt(s)}×</b>.{' '}
                {predicted > s * 2
                  ? 'Bạn đã kỳ vọng nhiều hơn đáng kể. Kernel cơ bản có những trợ giúp mà có thể bạn chưa tính: các phép toán nguyên tử của nó chạy ở bộ nhớ đệm cấp cuối (mục 9.3), và NVIDIA từng nói trình biên dịch của họ gộp các phép toán nguyên tử cùng địa chỉ của một warp "in many cases" (trong nhiều trường hợp), mà không nói là những trường hợp nào.'
                  : predicted < s / 2
                    ? 'Bạn đã kỳ vọng ít hơn đáng kể. Trên đầu vào này, mọi phép toán nguyên tử của kernel cơ bản xếp hàng trên một bộ đếm; các bản sao riêng chia hàng đợi đó theo số khối đang chạy.'
                    : 'Con số đó gần với dự đoán của bạn.'}
              </p>
            }
          >
            <p key={f.b} data-speedup={s}>
              On <b>{f.device}</b> ({f.b}), with every pixel the same, the shared-memory kernel (coarsening factor 16) ran at{' '}
              {fmt(f.shared)} Gpixel/s against the basic kernel's {fmt(f.basic)}: <b>{fmt(s)}×</b>.{' '}
              {predicted > s * 2
                ? 'You expected considerably more. The basic kernel had help you may not have counted: its atomics run at the last-level cache (section 9.3), and NVIDIA has said its compiler combines a warp\'s same-address atomics "in many cases", without saying which.'
                : predicted < s / 2
                  ? 'You expected considerably less. On this input every atomic of the basic kernel queues on one counter; the private copies split that queue by the number of blocks in flight.'
                  : 'That is close to what you predicted.'}
            </p>
          </Translated>
        );
      })}
      <Translated vi={<p className="muted">Đây là số đo từ máy của bạn, không phải giá trị kỳ vọng. Một thiết bị khác có thể khác rất nhiều theo cả hai hướng.</p>}>
        <p className="muted">These are measurements from your machine, not expected values. Another device can differ widely in either direction.</p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS94: Record<string, Question> = {
  'ch09/9.4/q-commit-atomic': {
    kind: 'choice',
    id: 'ch09/9.4/q-commit-atomic',
    prompt: (
      <Translated
        vi={
          <>
            Trong vòng lặp ghi, chỉ một luồng của một khối từng chạm vào một bin công khai nhất định. Vì sao phép cộng của nó vào{' '}
            <code>bins[b]</code> vẫn phải là nguyên tử?
          </>
        }
      >
        <>
          In the commit loop, only one thread of a block ever touches a given public bin. Why does its add to{' '}
          <code>bins[b]</code> still have to be atomic?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Vì số đếm riêng mà nó đọc có thể vẫn đang thay đổi, nếu một luồng chậm vẫn đang đếm.">
            Because the private count it reads might still be changing, if a slow thread is still counting.
          </Translated>
        ),
        why: (
          <Translated vi="Rào chắn trước bước ghi loại trừ điều đó: mọi cập nhật của khối đã xong.">
            The barrier before the commit rules that out: every update of the block has finished.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Vì cùng luồng đó của mọi khối khác cũng ghi vào bin công khai đó, có thể vào cùng một lúc.">
            Because the same thread of every other block commits to that public bin, maybe at the same moment.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không cần: một người ghi cho mỗi bin trong mỗi khối là đủ, và phép kiểm tra bằng không bỏ qua phần còn lại.">
            It does not have to: one writer per bin per block is enough, and the zero test skips the rest.
          </Translated>
        ),
        why: (
          <Translated vi="Một người ghi mỗi khối không phải là một người ghi. Các khối không có thứ tự với nhau.">
            One writer per block is not one writer. The blocks are not ordered with respect to each other.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Vì các lệnh ghi thường vào bộ nhớ toàn cục không hiển thị cho host cho đến khi một phép toán nguyên tử đẩy chúng ra.">
            Because plain stores to global memory are not visible to the host until an atomic flushes them.
          </Translated>
        ),
        why: (
          <Translated vi="Chúng có hiển thị, một khi kernel đã chạy xong. Vấn đề ở đây không phải tính hiển thị; mà là cập nhật bị mất.">
            They are, once the kernel has finished. Visibility is not the problem here; lost updates are.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một luồng mỗi khối nghĩa là có bao nhiêu khối đã gặp bin thì có bấy nhiêu luồng tranh chấp, điều mà sách gọi là tranh
            chấp vừa phải, chứ không phải không có. Hai khối đọc cùng một số đếm cũ và mỗi khối ghi lại tổng của mình sẽ làm mất
            toàn bộ phần đóng góp của một khối.
          </p>
        }
      >
        <p>
          One thread per block means as many contenders as there are blocks that saw the bin, which the book calls modest
          contention, not none. Two blocks that read the same old count and each write back their sum lose one block's
          whole contribution.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.4, p. 213',
  },
  'ch09/9.4/q-scope-commit': {
    kind: 'choice',
    id: 'ch09/9.4/q-scope-commit',
    prompt: (
      <Translated
        vi={
          <>
            Một đồng nghiệp viết bước ghi với <code>cuda::thread_scope_block</code>, "vì mỗi khối chỉ có một luồng ghi cho
            mỗi bin". Mọi bài kiểm tra đều qua trên <code>emu</code>, trên <code>race</code> và trên Metal. Kernel có
            đúng không?
          </>
        }
      >
        <>
          A colleague writes the commit with <code>cuda::thread_scope_block</code>, "since each block has a single
          committing thread per bin". Every test passes on <code>emu</code>, on <code>race</code> and on Metal. Is the
          kernel correct?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: ba backend đồng ý, và phạm vi khối là đủ khi mỗi khối có một người ghi cho mỗi bin.">
            Yes: three backends agree, and block scope is enough when each block has one writer per bin.
          </Translated>
        ),
        why: (
          <Translated vi="Các trình giả lập hoàn toàn không cho phạm vi tác dụng gì, và Metal không có phạm vi trên thao tác đọc-sửa-ghi. Sự đồng thuận giữa các công cụ không thể thấy lỗi không phải là bằng chứng.">
            The emulators give scopes no effect at all, and Metal has no scope on a read-modify-write. Agreement among tools that cannot see the defect is no evidence.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: các khối khác cộng vào cùng bin công khai, nên CUDA gọi nó là data race, thứ mà không bài kiểm tra nào ở đây thấy được.">
            No: other blocks add to the same public bin, so CUDA calls it a data race, which no test here can see.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có trên NVIDIA: phép toán nguyên tử toàn cục được thực hiện trong bộ nhớ đệm L2 bất kể phạm vi, nên ở đó nó không thay đổi gì.">
            Yes on NVIDIA: global atomics are performed in the L2 cache whatever the scope, so it changes nothing there.
          </Translated>
        ),
        why: (
          <Translated vi="Nơi phần cứng thực hiện chúng không phải điều ngôn ngữ hứa hẹn. Mô hình bộ nhớ của NVIDIA gọi đây là data race, bất kể các con số cho thấy gì.">
            Where the hardware performs them is not what the language promises. NVIDIA's memory model calls this a data race, whatever the numbers show.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không, và backend race lẽ ra đã gắn cờ hai phép cộng của hai khối là không có thứ tự.">
            No, and the race backend would have flagged the two blocks' adds as unordered.
          </Translated>
        ),
        why: (
          <Translated vi="Nó sẽ không làm vậy: trình giả lập coi mọi phép toán nguyên tử là nguyên tử, bất kể phạm vi của nó.">
            It would not: the emulator treats every atomic as an atomic, whatever its scope.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            CUDA định nghĩa một data race là hai truy cập xung đột trong đó một truy cập không nguyên tử ở phạm vi bao gồm luồng
            kia, và coi mọi data race là hành vi không xác định. Một phép cộng phạm vi khối từ khối 0 và một từ khối 1 trên
            cùng bin chính là như vậy. Đây là phát hiện khi review mã, không phải phát hiện khi kiểm thử.
          </p>
        }
      >
        <p>
          CUDA defines a data race as two conflicting accesses where one is not atomic at a scope that includes the other
          thread, and makes every data race undefined behaviour. A block-scope add from block 0 and one from block 1 on
          the same bin is exactly that. This is a review finding, not a test finding.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch09-facts.md A2 (CUDA programming guide 5.8.4); course emulator header (scopes not modelled); C1 (Metal)',
    vi: { source: 'notes/today/ch09-facts.md A2 (CUDA programming guide 5.8.4); header trình giả lập của khóa học (không mô hình hóa phạm vi); C1 (Metal)' },
  },
  'ch09/9.4/q-why-block': {
    kind: 'choice',
    id: 'ch09/9.4/q-why-block',
    prompt: (
      <Translated vi={<>Vì sao sách cho mỗi khối một bản sao riêng, thay vì mỗi luồng?</>}>
        <>Why does the book give each block a private copy, rather than each thread?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mọi khối của một lưới chạy cùng lúc, nên các bản sao theo khối có thể được trộn trong khi chúng đếm.">
            All the blocks of a grid run at the same time, so per-block copies can be merged while they count.
          </Translated>
        ),
        why: (
          <Translated vi="Không phải vậy: các khối chạy khi phần cứng có chỗ cho chúng, không theo thứ tự nào được hứa hẹn.">
            They do not: blocks run when the hardware has room for them, in no promised order.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phép toán nguyên tử không được phép trên dữ liệu riêng của một luồng, nên một bản sao theo luồng không thể cập nhật được.">
            Atomics are not allowed on data private to one thread, so a per-thread copy could not be updated.
          </Translated>
        ),
        why: (
          <Translated vi="Chúng được phép, chỉ là không cần: không ai khác chạm vào nó.">
            They are allowed, just unnecessary: nobody else touches it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một khối có một rào chắn cho biết khi nào việc đếm xong, và một bản sao nhỏ vừa trong bộ nhớ chia sẻ của nó.">
            A block has a barrier that says when its counting is done, and a small copy fits in its shared memory.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách cân nhắc tranh chấp với chi phí trộn và chọn một tập con các luồng, phổ biến nhất là một khối. Khối
            mang lại miễn phí hai thứ: <code>__syncthreads()</code> làm tín hiệu "mọi người đã xong" trước bước
            ghi, và bộ nhớ chia sẻ làm chỗ ở nhanh cho bản sao. Bản sao theo luồng sẽ nhân bộ nhớ và bước
            ghi lên theo số luồng. Mục 9.6 có quay lại riêng tư hóa theo luồng, cho từng bin một.
          </p>
        }
      >
        <p>
          The book weighs contention against merge cost and settles on a subset of threads, most commonly a block. The
          block brings two things for free: <code>__syncthreads()</code> as the "everyone is done" signal before the
          commit, and shared memory as a fast home for the copy. Per-thread copies would multiply the memory and the
          commit by the number of threads. Section 9.6 does come back to per-thread privatisation, for one bin at a time.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.4, pp. 211, 213',
  },
  'ch09/9.4/q-ex5a': {
    kind: 'numeric',
    id: 'ch09/9.4/q-ex5a',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 5a: 524,288 phần tử, 128 bin, 1024 luồng mỗi khối. Kernel cơ bản (Hình 9.6) thực hiện bao nhiêu phép
            toán nguyên tử trên bộ nhớ toàn cục?
          </>
        }
      >
        <>
          Exercise 5a: 524,288 elements, 128 bins, 1024 threads per block. How many global-memory atomic operations does the
          basic kernel (Fig. 9.6) perform?
        </>
      </Translated>
    ),
    answer: 524_288,
    explain: (
      <Translated vi={<p>Một cho mỗi phần tử: mỗi luồng cộng 1 vào một bin công khai. 524,288.</p>}>
        <p>One per element: every thread adds 1 to a public bin. 524,288.</p>
      </Translated>
    ),
    source: 'ch. 9 §9.8, exercise 5a, p. 220 (our answer; the book prints none)',
    vi: { source: 'ch. 9 §9.8, bài tập 5a, p. 220 (đáp án của khóa học này; sách không in đáp án)' },
  },
  'ch09/9.4/q-ex5b': {
    kind: 'numeric',
    id: 'ch09/9.4/q-ex5b',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 5b: cùng đầu vào. Kernel bộ nhớ chia sẻ (Hình 9.10, không gộp việc) có thể thực hiện <em>nhiều nhất</em>{' '}
            bao nhiêu phép toán nguyên tử trên bộ nhớ toàn cục?
          </>
        }
      >
        <>
          Exercise 5b: same input. What is the <em>most</em> global-memory atomics the shared-memory kernel (Fig. 9.10, no
          coarsening) can perform?
        </>
      </Translated>
    ),
    answer: 65_536,
    explain: (
      <Translated
        vi={
          <p>
            524,288 / 1024 = 512 khối. Mỗi khối ghi nhiều nhất 128 bin khác không: 512 × 128 = 65,536, một phần tám của 5a. Có
            phép kiểm tra bằng không thì sẽ ít hơn mỗi khi một khối để trống bin. Mọi phép toán nguyên tử khác đều nằm trong bộ nhớ chia sẻ.
          </p>
        }
      >
        <p>
          524,288 / 1024 = 512 blocks. Each commits at most 128 non-zero bins: 512 × 128 = 65,536, an eighth of 5a. With
          the zero test it is fewer whenever a block leaves bins empty. All the other atomics are in shared memory.
        </p>
      </Translated>
    ),
    source: 'ch. 9 §9.8, exercise 5b, p. 220 (our answer)',
    vi: { source: 'ch. 9 §9.8, bài tập 5b, p. 220 (đáp án của khóa học này)' },
  },
  'ch09/9.4/q-vocab-kib': {
    kind: 'numeric',
    id: 'ch09/9.4/q-vocab-kib',
    prompt: (
      <Translated
        vi={
          <>
            Bộ từ vựng của GPT-2 có 50,257 token. Một bản sao riêng với bộ đếm 32-bit chiếm bao nhiêu KiB?
          </>
        }
      >
        <>
          GPT-2's vocabulary has 50,257 tokens. How many KiB does one private copy take with 32-bit counts?
        </>
      </Translated>
    ),
    answer: 196.3,
    tol: 0.005,
    unit: 'KiB',
    explain: (
      <Translated
        vi={
          <p>
            50,257 × 4 = 201,028 byte = 196.3 KiB. Con số đó vượt quá 32 KB mà Metal ghi trong tài liệu và mặc định 48 KB mỗi
            khối của CUDA; chỉ giới hạn opt-in của một số thiết bị CUDA mới chứa được nó.
          </p>
        }
      >
        <p>
          50,257 × 4 = 201,028 bytes = 196.3 KiB. That is beyond Metal's documented 32 KB and CUDA's 48 KB default per
          block; only the opt-in limits of some CUDA devices hold it.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch09-facts.md D7 (vocabulary from the model config; arithmetic ours), A8, C3',
    vi: { source: 'notes/today/ch09-facts.md D7 (bộ từ vựng lấy từ cấu hình mô hình; phép tính của khóa học này), A8, C3' },
  },
};

// ---------------------------------------------------------------- the page
export function S94() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 9.3 làm mỗi phép toán nguyên tử ngắn hơn bằng cách thực hiện nó trong bộ nhớ đệm cấp cuối thay vì DRAM. Mục này nhắm
            vào yếu tố còn lại: bao nhiêu luồng xếp hàng cho cùng một bộ đếm. Cách chữa là <b>riêng tư hóa</b> (privatization). Cho các tập con luồng
            bản sao histogram riêng của chúng, để mỗi tập con đếm vào bản sao của mình, và cuối cùng cộng các bản sao vào histogram
            công khai. Phép cộng có tính giao hoán và kết hợp, nên các bản sao có thể được trộn theo bất kỳ thứ tự nào.
          </p>
        }
      >
        <p>
          Section 9.3 made each atomic shorter by performing it in the last-level cache instead of DRAM. This section goes
          after the other factor: how many threads queue for the same counter. The remedy is <b>privatisation</b>. Give subsets of threads their own
          copies of the histogram, let each subset count into its own copy, and at the end add the copies into the public
          histogram. Addition is commutative and associative, so the copies can be merged in any order.
        </p>
      </Translated>
      <Source section="9.4" pages="211" figs={tr('Fig. 9.8', 'Hình 9.8')} />
      <Translated
        vi={
          <p>
            Riêng tư hóa mua được ít tranh chấp hơn, và thường là một bộ nhớ nhanh hơn cho bản sao. Cái giá là một lần trộn. Càng nhiều bản sao,
            mỗi hàng đợi càng ngắn và lần trộn càng dài, đó là lý do trên một máy song song quy mô lớn, sách cho một
            bản sao cho một tập con luồng thay vì cho mỗi luồng. Sách nêu ba cách chọn tập con: hai bản sao,
            một cho các khối số chẵn và một cho các khối số lẻ; bốn bản sao, với khối <i>i</i> dùng bản sao <i>i</i> mod 4; và,
            phổ biến nhất, một bản sao mỗi khối.
          </p>
        }
      >
        <p>
          Privatisation buys less contention, and often a faster memory for the copy. It costs a merge. The more copies,
          the shorter each queue and the longer the merge, which is why on a massively parallel machine the book gives a
          copy to a subset of threads rather than to each thread. It names three ways to choose the subsets: two copies,
          one for even-numbered blocks and one for odd; four copies, with block <i>i</i> using copy <i>i</i> mod 4; and,
          most commonly, one copy per block.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              p. 211 viết "In our text histogram example" (trong ví dụ histogram văn bản của chúng tôi). Ví dụ xuyên suốt của ấn bản
              này là một ảnh; cụm từ đó có vẻ sót lại từ ví dụ văn bản của một ấn bản trước. Không gì trên trang phụ thuộc vào nó.
            </p>
          }
        >
          <p>
            p. 211 says "In our text histogram example". This edition's running example is an image; the phrase looks left
            over from an earlier edition's text example. Nothing on the page depends on it.
          </p>
        </Translated>
      </Aside>

      <h2><Translated vi="Bản sao và các luồng tranh chấp">Copies and contenders</Translated></h2>
      <Translated
        vi={
          <p>
            Trình khám phá đếm, với mỗi cách dùng chung, hàng đợi cập nhật dài nhất trên một bộ đếm và chi phí của lần
            trộn. Bắt đầu với ảnh hằng, nơi mọi cập nhật trong kernel cơ bản rơi vào một bin, rồi chuyển sang
            nhiễu phân bố đều, rồi chọn một bộ từ vựng.
          </p>
        }
      >
        <p>
          The explorer counts, for each way of sharing, the longest queue of updates on one counter and what the merge
          costs. Start with the constant image, where every update in the basic kernel lands on one bin, then switch to
          uniform noise, then pick a vocabulary.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Nó đếm công việc, không phải thời gian. Mỗi luồng dùng chung một bản sao được tính là một luồng tranh chấp, kể cả khi khối của nó chạy vào
              một thời điểm khác. Các bản sao được nhiều khối dùng chung (hai hoặc bốn) cũng cần một tín hiệu "mọi người đã xong" trên toàn lưới trước
              khi trộn, chẳng hạn một kernel thứ hai; sách không trình bày điều này.
            </p>
          }
        >
          <p>
            It counts work, not time. Every thread that shares a copy counts as a contender, even if its block runs at
            another moment. Copies shared by several blocks (two or four) also need a grid-wide "everyone is done" before
            their merge, such as a second kernel; the book does not show one.
          </p>
        </Translated>
      </Simplification>
      <Explorer />
      <Translated
        vi={
          <p>
            Hai điều cần rút ra. Hàng đợi dài nhất giảm theo số bản sao, nhưng lần trộn tăng theo số bản sao ×
            số bin, nên một bản sao mỗi khối chỉ là trao đổi tốt khi số bin còn ít. Và 256 bin vừa với ngân sách trên chip của bất kỳ khối nào,
            trong khi một bộ từ vựng không vừa với ngân sách mỗi khối mặc định nào; chỉ GPT-2 vừa, và chỉ khi opt-in trên CUDA ở
            một số compute capability. Hãy giữ ý đó cho đến phần "Một bản sao có vừa trên chip không?" ở dưới.
          </p>
        }
      >
        <p>
          Two things to take from it. The longest queue falls by the number of copies, but the merge grows with copies ×
          bins, so a copy per block is a good trade only while bins are few. And 256 bins fit any block's on-chip budget,
          while a vocabulary fits none of the default per-block budgets; only GPT-2's fits, and only behind CUDA's opt-in on
          some compute capabilities. Hold that thought until "Does a copy fit on chip?" below.
        </p>
      </Translated>

      <h2><Translated vi="Phiên bản 1: bản sao riêng trong bộ nhớ toàn cục">Version 1: private copies in global memory</Translated></h2>
      <Translated
        vi={
          <p>
            Phiên bản đầu tiên giữ mọi bản sao riêng trong bộ nhớ toàn cục. Host cấp phát một vùng gồm{' '}
            <code>gridDim.x × NUM_BINS</code> bộ đếm, và khối <i>b</i> tìm bản sao của mình ở vị trí <code>b × NUM_BINS</code>.
            Sách kỳ vọng các bản sao đó nằm lại trong bộ nhớ đệm. Vùng này phải bắt đầu từ không; sách không nói ai xóa nó, còn
            mã host của lab thì làm việc đó.
          </p>
        }
      >
        <p>
          The first version keeps every private copy in global memory. The host allocates a pool of{' '}
          <code>gridDim.x × NUM_BINS</code> counters, and block <i>b</i> finds its copy at offset <code>b × NUM_BINS</code>.
          The book expects those copies to stay in the cache. The pool has to start at zero; the book does not say who zeroes it, and the
          lab's host code does.
        </p>
      </Translated>
      <Source section="9.4" pages="212" figs={tr('Fig. 9.9', 'Hình 9.9')} />
      <Translated
        vi={
          <p>
            Mỗi luồng cộng 1 vào bản sao của khối mình bằng một phép toán nguyên tử <b>phạm vi khối</b>. Rồi <code>__syncthreads()</code> chờ
            cho đến khi mọi luồng của khối đã đếm xong. Rồi đến bước ghi: luồng <i>t</i> lấy các bin <i>t</i>, <i>t</i> +{' '}
            <code>blockDim.x</code>, <i>t</i> + 2·<code>blockDim.x</code> và cứ thế, nên số bin có thể nhiều hơn số luồng;
            nó bỏ qua các bin vẫn bằng không và cộng phần còn lại vào các bin công khai bằng phép toán nguyên tử <b>phạm vi device</b>. Giờ tối đa
            1024 luồng dùng chung một bản sao thay vì cả lưới dùng chung một, nên tranh chấp giảm xấp xỉ theo số
            khối hoạt động cùng lúc.
          </p>
        }
      >
        <p>
          Each thread adds 1 to its block's copy with a <b>block-scope</b> atomic. Then <code>__syncthreads()</code> waits
          until every thread of the block has counted. Then the commit: thread <i>t</i> takes bins <i>t</i>, <i>t</i> +{' '}
          <code>blockDim.x</code>, <i>t</i> + 2·<code>blockDim.x</code> and so on, so the bins may outnumber the threads;
          it skips bins that are still zero and adds the rest to the public bins with <b>device-scope</b> atomics. Up to
          1024 threads now share a copy instead of the whole grid sharing one, so contention falls by roughly the number of
          blocks active at once.
        </p>
      </Translated>
      <Source section="9.4" pages="212–213" />
      <LabRegion
        lab={LAB1}
        name="pool" // i18n-ignore: region name in the lab file
        what={tr('The pool kernel', 'Kernel vùng chung (pool)')}
        caption={
          <Translated
            vi={
              <>
                <code>histogramPool</code> từ lời giải tham chiếu của lab 9.1 (<code>labs/{LAB1}/solution/</code>), là Hình 9.9
                của sách với các tên đã đổi. Hãy thử lab trước khi bạn nghiên cứu nó.
              </>
            }
          >
            <>
              <code>histogramPool</code> from lab 9.1's reference solution (<code>labs/{LAB1}/solution/</code>), the book's
              Fig. 9.9 with its names changed. Try the lab before you study it.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Chuyển sang tab Metal. Ba thứ đổi hình dạng. Metal không có góc nhìn nguyên tử lên một mảng thường, nên vùng chung được
            khai báo <code>device atomic_uint*</code> trong khi host vẫn cấp phát bộ đếm thường. Lần đọc thường số đếm riêng
            trong bước ghi trở thành <code>atomic_load_explicit</code>. Và vì bản sao nằm trong bộ nhớ device,
            rào chắn phải sắp thứ tự bộ nhớ device: <code>threadgroup_barrier(mem_flags::mem_device)</code>, không phải{' '}
            <code>mem_threadgroup</code> bạn đã dùng cho các ô.
          </p>
        }
      >
        <p>
          Switch to the Metal tab. Three things change shape. Metal has no atomic view of a plain array, so the pool is
          declared <code>device atomic_uint*</code> while the host still allocates plain counters. The commit's plain read
          of a private count becomes <code>atomic_load_explicit</code>. And because the copy lives in device memory, the
          barrier has to order device memory: <code>threadgroup_barrier(mem_flags::mem_device)</code>, not the{' '}
          <code>mem_threadgroup</code> you used for tiles.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md, C1; ch05-facts.md, C2)', '2026-11-20 (notes/today/ch09-facts.md, mục C1; ch05-facts.md, C2)')}>
        <Translated
          vi={
            <p>
              MSL: các hàm atomic "can only use Metal atomic data types" (chỉ dùng được các kiểu dữ liệu atomic của Metal).{' '}
              <code>mem_device</code> "ensures the GPU correctly orders the memory operations to device memory for threads in
              the threadgroup" (đảm bảo GPU sắp đúng thứ tự các thao tác bộ nhớ tới bộ nhớ device cho các luồng trong threadgroup).
            </p>
          }
        >
          <p>
            MSL: atomic functions "can only use Metal atomic data types". <code>mem_device</code> "ensures the GPU correctly
            orders the memory operations to device memory for threads in the threadgroup".
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS94['ch09/9.4/q-commit-atomic']} />

      <h2><Translated vi="Phạm vi, một cách cẩn thận">Scope, carefully</Translated></h2>
      <Translated vi={<p>Phiên bản 1 dùng hai phạm vi trong một kernel. Trước khi đọc lý do, hãy quyết định xem bước ghi có thể dùng phạm vi hẹp hơn không.</p>}>
        <p>Version 1 uses two scopes in one kernel. Before reading why, decide whether the commit could use the narrower one.</p>
      </Translated>
      <Ask q={questionsS94['ch09/9.4/q-scope-commit']} label={tr('Predict', 'Dự đoán')} />
      <Translated
        vi={
          <p>
            Một kernel dùng hai phạm vi, mỗi phạm vi được chọn cho những luồng dùng chung vị trí đó. Một bản sao riêng chỉ được chạm bởi
            các luồng của một khối, nên phạm vi khối là đủ ở đó, và sách thêm rằng một phạm vi hẹp hơn có thể, tùy
            phần cứng, xong sớm hơn. Các bin công khai được mọi khối chạm vào, nên bước ghi cần phạm vi device.
          </p>
        }
      >
        <p>
          One kernel uses two scopes, each chosen for the threads that share the location. A private copy is touched only by
          the threads of one block, so block scope is enough there, and the book adds that a narrower scope may, depending on
          the hardware, finish sooner. The public bins are touched by every block, so the commit needs device scope.
        </p>
      </Translated>
      <Source section="9.4" pages="212–213" />
      <Translated
        vi={
          <p>
            Viết sai cái thứ hai thì bạn không viết ra một kernel chậm hơn; bạn viết ra một kernel sai. Mô hình bộ nhớ của CUDA
            gọi hai truy cập xung đột là một data race khi một trong hai không nguyên tử ở phạm vi bao gồm luồng
            kia, và mọi data race là hành vi không xác định. Một phép cộng phạm vi khối vào một bin mà một khối khác cũng cộng
            vào chính là như vậy, kể cả ở những lần chạy mà các con số ra đúng.
          </p>
        }
      >
        <p>
          Get the second one wrong and you have not written a slower kernel; you have written a wrong one. CUDA's memory
          model calls two conflicting accesses a data race when one of them is not atomic at a scope that includes the
          other thread, and every data race is undefined behaviour. A block-scope add to a bin that another block also adds
          to is exactly that, even on the runs where the numbers come out right.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md, A2, B1, C1)', '2026-11-20 (notes/today/ch09-facts.md, mục A2, B1, C1)')}>
        <Translated
          vi={
            <p>
              Bảng phạm vi của NVIDIA cho L1 là điểm nhất quán (point of coherency) của phạm vi khối và L2 cho phạm vi device. Lời khuyên của họ,
              "block-scoped atomics are much faster than system-scoped atomics", so sánh phạm vi khối với phạm vi system; không có
              con số khối-so-với-device nào được công bố, mà đó lại là phép so sánh Hình 9.9 dựa vào. HIP giữ đối số
              template qua libhipcxx, dù các trang của AMD mô tả phạm vi khối là "shared memory" và phạm vi device là
              "global memory", vốn không phải nghĩa của CUDA. Metal không có phạm vi trên thao tác đọc-sửa-ghi: không gian địa chỉ,
              threadgroup hay device, là sự phân biệt duy nhất.
            </p>
          }
        >
          <p>
            NVIDIA's scope table gives L1 as the point of coherency for block scope and L2 for device scope. Its advice,
            "block-scoped atomics are much faster than system-scoped atomics", compares block with system scope; no
            block-versus-device figure is published, and that is the comparison Fig. 9.9 relies on. HIP keeps the template
            argument through libhipcxx, though AMD's pages describe block scope as "shared memory" and device scope as
            "global memory", which is not CUDA's meaning. Metal has no scope on a read-modify-write: the address space,
            threadgroup or device, is the only distinction.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Vì sao một bản sao mỗi khối">Why one copy per block</Translated></h2>
      <Ask q={questionsS94['ch09/9.4/q-why-block']} label={tr('Predict', 'Dự đoán')} />
      <Translated
        vi={
          <p>
            Một khối là chủ sở hữu tự nhiên của một bản sao riêng, vì hai lý do. Nó có một rào chắn, nên nó biết khi nào mọi luồng
            đã đếm xong và bước ghi có thể bắt đầu. Và nếu bản sao đủ nhỏ, nó có thể nằm trong bộ nhớ chia sẻ
            của khối, trên SM, nơi một phép toán nguyên tử xong sớm hơn nhiều so với trong bộ nhớ toàn cục. Vì thông lượng của phép toán nguyên tử trên
            một vị trí được quyết định bởi độ trễ của chúng (mục 9.3), đó là một mức lợi lớn.
          </p>
        }
      >
        <p>
          A block is the natural owner of a private copy, for two reasons. It has a barrier, so it knows when every thread
          has finished counting and the commit may start. And if the copy is small enough it can live in the block's shared
          memory, on the SM, where an atomic finishes far sooner than in global memory. Since the throughput of atomics on
          one location is set by their latency (section 9.3), that is a large gain.
        </p>
      </Translated>
      <Source section="9.4" pages="213" />

      <h2><Translated vi="Phiên bản 2: bản sao trong bộ nhớ chia sẻ">Version 2: the copy in shared memory</Translated></h2>
      <Translated
        vi={
          <p>
            Phiên bản thứ hai khai báo bản sao là <code>__shared__ unsigned int bins_s[NUM_BINS]</code>. Bộ nhớ chia sẻ bắt đầu
            với bất cứ thứ gì còn sót lại trong nó, nên trước tiên khối xóa bản sao của mình về không bằng cùng vòng lặp nhảy bước, rồi chờ ở một
            rào chắn trước khi ai đó đếm. Từ đó trở đi nó là Phiên bản 1 với <code>bins_s</code> thay cho vùng chung:
            cập nhật phạm vi khối, rào chắn thứ hai, bước ghi. Hai rào chắn, mỗi cái canh một lần bàn giao khác nhau.
          </p>
        }
      >
        <p>
          The second version declares the copy <code>__shared__ unsigned int bins_s[NUM_BINS]</code>. Shared memory starts
          with whatever was left in it, so the block first zeroes its copy with the same strided loop, then waits at a
          barrier before anyone counts. From there it is Version 1 with <code>bins_s</code> in place of the pool:
          block-scope updates, a second barrier, the commit. Two barriers, each guarding a different handover.
        </p>
      </Translated>
      <Source section="9.4" pages="213–214" figs={tr('Fig. 9.10', 'Hình 9.10')} />
      <LabRegion
        lab={LAB2}
        name="shared" // i18n-ignore: region name in the lab file
        what={tr('The shared-memory kernel', 'Kernel bộ nhớ chia sẻ')}
        caption={
          <Translated
            vi={
              <>
                <code>histogramShared</code> từ lời giải tham chiếu của lab 9.2 (<code>labs/{LAB2}/solution/</code>). Với
                hệ số gộp việc bằng 1, nó là Hình 9.10 của sách; <a href={href('ch09', '9.5')}>mục 9.5</a> giải thích
                vòng lặp quanh phép cập nhật.
              </>
            }
          >
            <>
              <code>histogramShared</code> from lab 9.2's reference solution (<code>labs/{LAB2}/solution/</code>). With a
              coarsening factor of 1 it is the book's Fig. 9.10; <a href={href('ch09', '9.5')}>section 9.5</a> explains the
              loop around the update.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Trên Metal, bản sao là <code>threadgroup atomic_uint bins_s[NUM_BINS]</code>, được xóa về không bằng{' '}
            <code>atomic_store_explicit</code>, được canh bởi <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>, và
            được ghi bằng cách đọc với <code>atomic_load_explicit</code> và cộng với <code>atomic_fetch_add_explicit</code>{' '}
            vào bộ nhớ device. Histogram MPS của PyTorch dùng cùng các phép toán nguyên tử này, trên một tham số{' '}
            <code>threadgroup atomic_uint*</code> mà kích thước do host đặt, thay vì một mảng khai báo trong
            kernel.
          </p>
        }
      >
        <p>
          On Metal the copy is <code>threadgroup atomic_uint bins_s[NUM_BINS]</code>, zeroed with{' '}
          <code>atomic_store_explicit</code>, guarded by <code>threadgroup_barrier(mem_flags::mem_threadgroup)</code>, and
          committed by reading with <code>atomic_load_explicit</code> and adding with <code>atomic_fetch_add_explicit</code>{' '}
          into device memory. PyTorch's MPS histogram uses the same atomic operations, on a{' '}
          <code>threadgroup atomic_uint*</code> parameter whose size the host sets, rather than an array declared in the
          kernel.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md, A5, A6, C1, D2)', '2026-11-20 (notes/today/ch09-facts.md, mục A5, A6, C1, D2)')}>
        <Translated
          vi={
            <p>
              "A few cycles" (vài chu kỳ) không xuất hiện trong tài liệu hiện hành nào của NVIDIA. NVIDIA nói phép toán nguyên tử trên bộ nhớ chia sẻ "are served inside the SM
              and are significantly faster than global atomics" (được phục vụ bên trong SM và nhanh hơn đáng kể so với atomic toàn cục), là lệnh gốc cho số nguyên 32-bit kể từ Maxwell. Hướng dẫn phân loại vấn đề của
              Nsight Compute liệt kê gộp ở mức warp, rồi phép rút gọn ở mức khối, rồi riêng tư hóa theo khối trong bộ nhớ
              chia sẻ, chính là kernel của mục này.
            </p>
          }
        >
          <p>
            "A few cycles" appears in no current NVIDIA document. NVIDIA says shared-memory atomics "are served inside the SM
            and are significantly faster than global atomics", native for 32-bit integers since Maxwell. Its Nsight Compute
            triage guide lists warp-level combining, then block-level reduction, then per-block privatisation in shared
            memory, which is this section's kernel.
          </p>
        </Translated>
      </Today>
      <Translated vi={<p>Hãy đếm xem bản sao chia sẻ tiết kiệm được gì. Bài tập 5 của sách hỏi đúng điều này.</p>}>
        <p>Count what the shared copy saves. Exercise 5 of the book asks exactly this.</p>
      </Translated>
      <Ask q={questionsS94['ch09/9.4/q-ex5a']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS94['ch09/9.4/q-ex5b']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Xem nó diễn ra">Watch it happen</Translated></h2>
      <Translated
        vi={
          <p>
            Riêng tư hóa là hai pha với một rào chắn ở giữa, và điểm mấu chốt là số luồng nhắm vào một
            bộ đếm thay đổi thế nào từ pha này sang pha kia. Hoạt ảnh chạy theo hình minh họa Hình 9.8 của sách, 24 điểm ảnh và ba khối
            tám luồng, với giá trị điểm ảnh của riêng chúng tôi.
          </p>
        }
      >
        <p>
          Privatisation is two phases with a barrier between them, and the point is how the number of threads aiming at one
          counter changes from one phase to the next. The animation runs the book's Fig. 9.8 picture, 24 pixels and three blocks
          of eight threads, with pixel values of our own.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Số lượng, không phải thời gian: nó cho thấy bao nhiêu luồng nhắm vào một bộ đếm, không phải chúng chờ bao lâu. Một điểm ảnh mỗi luồng và
              8 bin; khối thật lớn hơn nhiều.
            </p>
          }
        >
          <p>
            Counts, not time: it shows how many threads aim at one counter, not how long they wait. One pixel per thread and
            8 bins; real blocks are far larger.
          </p>
        </Translated>
      </Simplification>
      <PrivatiseScene />

      <h2><Translated vi="Bảy kernel riêng tư hóa bị hỏng">Seven broken privatised kernels</Translated></h2>
      <Translated
        vi={
          <p>
            Mỗi rào chắn, phép toán nguyên tử và cận vòng lặp trong Phiên bản 2 đều có lý do, và mỗi lỗi lộ ra ở một chỗ
            khác nhau. Dưới đây là Phiên bản 2 thu nhỏ còn tám bin và hai khối bốn luồng, với bảy bản sao mỗi bản thay đổi
            một thứ. Trước khi hiện đáp án, hãy quyết định công cụ nào sẽ cho thấy mỗi lỗi: backend <code>emu</code>, backend{' '}
            <code>race</code>, một lần chạy trên GPU, hoặc không có gì ở đây làm được.
          </p>
        }
      >
        <p>
          Every barrier, atomic and loop bound in Version 2 is there for a reason, and each mistake shows up in a different
          place. Below is Version 2 shrunk to eight bins and two blocks of four threads, with seven copies that each change
          one thing. Before revealing, decide which tools would show each defect: the <code>emu</code> backend, the{' '}
          <code>race</code> backend, a run on a GPU, or nothing available here.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các cột emu tái hiện chính xác trình giả lập: từng luồng một cho đến một rào chắn, ba thứ tự của nó, bộ nhớ
              chia sẻ đầy byte 0xCD (0x{GARBAGE.toString(16).toUpperCase()} khi đọc như một số đếm 32-bit). Cột race là một
              phép kiểm tra happens-before kiểu mà ThreadSanitizer chạy: các luồng của một khối chỉ có thứ tự qua một rào chắn,
              các khối thì không bao giờ, và phạm vi bị bỏ qua giống như trình giả lập bỏ qua chúng. Backend race thật, chạy trên chính các
              kernel này, đã báo mọi race trong cột đó, ngoại trừ race của kernel D chỉ xuất hiện trong một số lần chạy của chúng tôi: nó
              chỉ xem xét những lịch chạy đã thực sự xảy ra, nên một lần chạy sạch không chứng minh điều gì.
            </p>
          }
        >
          <p>
            The emu columns replay the emulator exactly: one thread at a time until a barrier, its three orders, shared
            memory filled with 0xCD bytes (0x{GARBAGE.toString(16).toUpperCase()} as a 32-bit count). The race column is a
            happens-before check of the kind ThreadSanitizer runs: threads of a block are ordered only across a barrier,
            blocks never, and scopes are ignored as the emulator ignores them. The real race backend, run on these exact
            kernels, reported every race in that column, except that kernel D's showed up in only some of our runs: it
            examines only the schedules that occurred, so a clean run proves nothing.
          </p>
        </Translated>
      </Simplification>
      <BugHunt />
      <Translated
        vi={
          <p>
            Đọc bảng kết quả theo cột. <code>emu</code> bắt các lỗi về thứ tự: thiếu bước xóa về không, thiếu một trong hai
            rào chắn, sai cận vòng lặp. Backend race bắt các phép toán nguyên tử bị thiếu, và lại cả các rào chắn bị thiếu, vì
            nó báo cặp không có thứ tự ngay cả khi các con số tình cờ đúng. Một phạm vi quá hẹp thì vô hình với
            mọi công cụ trong khóa học này, dù CUDA gọi nó là hành vi không xác định. Lỗi đó bạn bắt khi review mã.
          </p>
        }
      >
        <p>
          Read the outcome table by column. <code>emu</code> catches ordering mistakes: the missing zeroing, either missing
          barrier, the wrong loop bound. The race backend catches missing atomics, and the missing barriers again, because
          it reports the unordered pair even when the numbers happen to be right. A scope that is too narrow is invisible to
          every tool in this course, although CUDA calls it undefined behaviour. That one you catch in review.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Cuộc săn lỗi là của khóa học này. Trên CUDA, <code>compute-sanitizer --tool racecheck</code> chỉ xem xét bộ nhớ chia sẻ; AMD
              không ghi trong tài liệu trình phát hiện race nào và shader validation của Metal không có (notes/today/ch09-facts.md A10, B6, C5).
            </p>
          }
        >
          <p>
            The hunt is ours. On CUDA, <code>compute-sanitizer --tool racecheck</code> examines shared memory only; AMD
            documents no race detector and Metal's shader validation has none (notes/today/ch09-facts.md A10, B6, C5).
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Một bản sao có vừa trên chip không?">Does a copy fit on chip?</Translated></h2>
      <Translated
        vi={
          <p>
            Phiên bản 2 chỉ hoạt động khi một bản sao vừa trong bộ nhớ chia sẻ mà một khối được dùng. Một bản sao bộ đếm 32-bit chiếm
            số bin × 4 byte: 1 KiB cho 256 bin của một ảnh. Với histogram ảnh, đó không bao giờ là vấn đề. Khi các bin
            là một bộ từ vựng, như khi bạn đếm token cho phạt lặp (repetition penalty), đó là vấn đề duy nhất.
          </p>
        }
      >
        <p>
          Version 2 works only while one copy fits in the shared memory a block may use. One copy of 32-bit counts takes
          bins × 4 bytes: 1 KiB for an image's 256 bins. For an image histogram that is never the question. When the bins
          are a vocabulary, as they are when you count tokens for a repetition penalty, it is the only question.
        </p>
      </Translated>
      <Ask q={questionsS94['ch09/9.4/q-vocab-kib']} label={tr('Work it out', 'Tự tính')} />
      <FitTable />
      <Translated
        vi={
          <p>
            Các cột thiết bị của bạn lấy từ những gì chúng báo cho <code>doctor</code>: <code>sharedMemPerBlock</code> trên
            CUDA và HIP, <code>maxThreadgroupMemoryLength</code> trên Metal. Các giới hạn trong tài liệu nói cùng điều đó một cách
            rộng hơn: một bản sao cỡ GPT-2 chỉ vừa khi opt-in trên CUDA ở một số compute capability, còn các bộ từ vựng
            hiện nay không vừa với giới hạn mỗi khối nào được ghi trong tài liệu. Giảm bộ đếm xuống 16 bit gần như không giúp gì: phép toán
            nguyên tử 16-bit được giả lập trên CUDA và HIP và không có trên Metal.
          </p>
        }
      >
        <p>
          Your devices' columns come from what they reported to <code>doctor</code>: <code>sharedMemPerBlock</code> on
          CUDA and HIP, <code>maxThreadgroupMemoryLength</code> on Metal. The documented limits say the same thing more
          broadly: a GPT-2-sized copy fits only behind CUDA's opt-in on some compute capabilities, and current
          vocabularies fit no documented per-block limit at all. Halving the counters to 16 bits barely helps: 16-bit
          atomics are emulated on CUDA and HIP and absent on Metal.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md, A1, A8, B4, C2, C3, D7)', '2026-11-20 (notes/today/ch09-facts.md, mục A1, A8, B4, C2, C3, D7)')}>
        <Translated
          vi={
            <p>
              CUDA: 48 KB mỗi khối khi không opt-in, 64 đến 327 KB khi có, tùy compute capability. AMD: LDS theo từng sản phẩm, không tìm thấy
              mức tối đa mỗi workgroup; hãy truy vấn <code>sharedMemPerBlock</code>. Metal: 32 KB bộ nhớ threadgroup.
            </p>
          }
        >
          <p>
            CUDA: 48 KB per block without opt-in, 64 to 327 KB with it, by compute capability. AMD: LDS per product, no
            per-workgroup maximum found; query <code>sharedMemPerBlock</code>. Metal: 32 KB of threadgroup memory.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Sách nhắc thêm một chỗ ở nữa cho bản sao, không kèm mã: trên các GPU có thread block cluster, một bản sao mỗi
            cluster trong bộ nhớ chia sẻ phân tán (distributed shared memory), sau một rào chắn toàn cluster, chứa được nhiều bin hơn một khối.
          </p>
        }
      >
        <p>
          The book mentions one more home for a copy, without code: on GPUs with thread block clusters, one copy per
          cluster in distributed shared memory, after a cluster-wide barrier, which holds more bins than one block can.
        </p>
      </Translated>
      <Source section="9.4" pages="213" />
      <Translated
        vi={
          <p>
            Hướng dẫn lập trình của NVIDIA giờ dùng một histogram làm ví dụ cho bộ nhớ chia sẻ phân tán,{' '}
            <code>clusterHist_kernel</code>, cho compute capability 9.0 trở lên. HIP có cluster group trên các thiết bị báo{' '}
            <code>clusterLaunch</code>; Metal không có gì tương tự. Chúng tôi không xây nó ở đây vì cả hai trình giả lập đều không mô hình hóa
            cluster. Với một bộ từ vựng, những gì còn lại là các bản sao trong bộ nhớ toàn cục của Phiên bản 1 (số khối × bộ từ vựng × 4 byte,
            và một bước ghi qua mọi bin), riêng tư hóa một-bin-mỗi-luồng của mục 9.6, hoặc hoàn toàn không đếm lại. Lab 9.3
            trên <a href={href('ch09', 'labs')}>trang lab</a> đếm token id.
          </p>
        }
      >
        <p>
          NVIDIA's programming guide now uses a histogram as its distributed-shared-memory example,{' '}
          <code>clusterHist_kernel</code>, for compute capability 9.0 and later. HIP has cluster groups on devices that report{' '}
          <code>clusterLaunch</code>; Metal has nothing like it. We do not build it here because neither emulator models
          clusters. For a vocabulary, what is left is Version 1's copies in global memory (blocks × vocabulary × 4 bytes,
          and a commit over every bin), section 9.6's one-bin-per-thread privatisation, or not recounting at all. Lab 9.3
          on the <a href={href('ch09', 'labs')}>labs page</a> counts token ids.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch09-facts.md, A6; infra-facts.md, B4)', '2026-11-20 (notes/today/ch09-facts.md, mục A6; infra-facts.md, B4)')}>
        <Translated
          vi={
            <p>
              Ví dụ trong hướng dẫn xóa bộ nhớ chia sẻ của khối về không, gọi <code>cluster.sync()</code>, cập nhật qua{' '}
              <code>cluster.map_shared_rank</code> với <code>atomicAdd</code>, đồng bộ lại và ghi mọi bin. Hàm{' '}
              <code>map_shared_rank()</code> của HIP cần một thiết bị báo có hỗ trợ cluster nhiều khối.
            </p>
          }
        >
          <p>
            The guide's example zeroes the block's shared memory, calls <code>cluster.sync()</code>, updates through{' '}
            <code>cluster.map_shared_rank</code> with <code>atomicAdd</code>, syncs again and commits every bin. HIP's{' '}
            <code>map_shared_rank()</code> needs a device that reports multi-block cluster support.
          </p>
        </Translated>
      </Today>

      <h2><Translated vi="Bạn đã gặp điều này ở đâu">Where you have met this</Translated></h2>
      <Translated
        vi={
          <p>
            <code>torch.histc</code> trên một tensor CUDA chạy đúng mục này. <code>SummaryOps.cu</code> của PyTorch tính xem
            một bản sao cần bao nhiêu, <code>nbins × sizeof(counter)</code> cộng 8 byte bảo vệ, và nếu con số đó nhỏ hơn{' '}
            <code>sharedMemPerBlock</code> thì nó chạy một bản sao bộ nhớ chia sẻ mỗi khối: Hình 9.10 không có phép kiểm tra bằng không. Ngược lại,
            mọi luồng cộng vào một bản sao toàn cục: Hình 9.6. Trên MPS, cùng lựa chọn đó được đưa ra ở host, trong{' '}
            <code>operations/HistogramKernel.mm</code>: một bản sao threadgroup khi các bin vừa trong 8 KiB, ngược lại là một bản sao
            toàn cục. Đọc <code>SummaryOps.cu</code> là đọc mục này.
          </p>
        }
      >
        <p>
          <code>torch.histc</code> on a CUDA tensor runs this section. PyTorch's <code>SummaryOps.cu</code> works out what
          one copy needs, <code>nbins × sizeof(counter)</code> plus 8 guard bytes, and if that is below{' '}
          <code>sharedMemPerBlock</code> it runs a shared-memory copy per block: Fig. 9.10 without the zero test. Otherwise
          every thread adds to one global copy: Fig. 9.6. On MPS the same choice is made on the host, in{' '}
          <code>operations/HistogramKernel.mm</code>: a threadgroup copy when the bins fit in 8 KiB, otherwise one global
          copy. Reading <code>SummaryOps.cu</code> is reading this section.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              PyTorch không có trong sách. Đọc vào 2026-11-20: <code>aten/src/ATen/native/cuda/SummaryOps.cu</code> và các tệp{' '}
              <code>HistogramKernel</code> của MPS (notes/today/ch09-facts.md D1, D2).
            </p>
          }
        >
          <p>
            PyTorch is not in the book. Read on 2026-11-20: <code>aten/src/ATen/native/cuda/SummaryOps.cu</code> and the MPS{' '}
            <code>HistogramKernel</code> files (notes/today/ch09-facts.md D1, D2).
          </p>
        </Translated>
      </Beyond>

      <h2><Translated vi="Đo nó">Measure it</Translated></h2>
      <Translated
        vi={
          <p>
            Lab 9.2 đo thời gian kernel cơ bản và kernel bộ nhớ chia sẻ trên cùng các ảnh. Ảnh hằng là trường hợp tệ nhất
            cho kernel cơ bản: mọi phép toán nguyên tử của lưới xếp hàng trên một bộ đếm.
          </p>
        }
      >
        <p>
          Lab 9.2 times the basic kernel and the shared-memory kernel on the same images. The constant image is the worst
          case for the basic kernel: every atomic of the grid queues on one counter.
        </p>
      </Translated>
      <Predict
        id="ch09/9.4/p-shared-speedup"
        prompt={
          <Translated
            vi={
              <>
                Trên GPU <em>của bạn</em>, với mọi điểm ảnh như nhau, bạn kỳ vọng kernel bộ nhớ chia sẻ của lab 9.2 chạy nhanh hơn
                kernel cơ bản bao nhiêu lần, ở hệ số gộp việc 16? Hãy chốt một con số trước khi chạy lab.
              </>
            }
          >
            <>
              On <em>your</em> GPU, with every pixel the same, how many times faster than the basic kernel do you expect lab
              9.2's shared-memory kernel to run, at a coarsening factor of 16? Commit a number before running the lab.
            </>
          </Translated>
        }
        unit={tr('× faster', '× nhanh hơn')}
      >
        {(v) => <MeasuredSharedSpeedup predicted={v} />}
      </Predict>
    </>
  );
}
