import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import {
  BLOCK_SIZES,
  KERNELS,
  KERNEL_LABEL,
  blockCosts,
  isPow2,
  ones,
  twoStage,
  type Costs,
  type Kernel,
} from '../../sim/shuffle';
import { functionSource } from '../../sim/toolchain';
import { widthChoices } from '../../sim/warps';
import { MIN_W, VIEW_W } from './ShuffleScene';
import { useLabStatus } from './status';

import { NoteGroup } from './NoteGroup';
const LAB = 'ch10/lab1_sum';
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', row: 'global', partialSums_s: 'shared',
  partialSum: 'reg', s: 'reg', t: 'reg', warps: 'reg', lane: 'reg', sg: 'reg', groups: 'reg', width: 'reg',
};

function useWidths() {
  const { reports } = useReports();
  return useMemo(
    () => widthChoices(reports as Reports).filter((c) => isPow2(c.width) && c.width >= 32 && c.width <= 64),
    [reports],
  );
}

// ---------------------------------------------------------------- the listing
function TwoStageListing() {
  const status = useLabStatus(LAB);
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const k = !missing(cu) ? functionSource(cu!, 'TwoStageWarpLevelSumReductionKernel') : null;
  const m = !missing(metal) ? functionSource(metal!, 'two_stage_warp_sum') : null;
  if (!k)
    return (
      <p className="muted">
        <Translated vi={<>Kernel được nạp từ <code>labs/{LAB}/solution/kernel.cu</code> khi lab 10.1 có trên máy này.</>}>The kernel is loaded from <code>labs/{LAB}/solution/kernel.cu</code> once lab 10.1 is on this machine.</Translated>
      </p>
    );
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: k.text, tiers: TIERS, status: status.cu },
        ...(m ? [{ label: 'Metal', code: m.text, tiers: TIERS, status: status.metal }] : []),
      ]}
      caption={
        <>
          <Translated
            vi={
              <>
                <code>TwoStageWarpLevelSumReductionKernel</code> từ lời giải tham chiếu của lab 10.1 (
                <code>labs/{LAB}/solution/kernel.cu</code>, dòng {k.firstLine} đến {k.lastLine}), cùng bản tương ứng cho Metal. Nó
                gọi <code>warp_reduce</code> khả chuyển của mục 10.7, nên cùng file đó được viết để biên dịch cho HIP; điều đó là
                unverified, vì chưa có gì ở đây được build với ROCm.
              </>
            }
          >
            <code>TwoStageWarpLevelSumReductionKernel</code> from lab 10.1's reference solution (
            <code>labs/{LAB}/solution/kernel.cu</code>, lines {k.firstLine} to {k.lastLine}), with its Metal counterpart. It
            calls the portable <code>warp_reduce</code> of section 10.7, so the same file is written to compile for HIP;
            that is unverified, since nothing here has been built with ROCm.
          </Translated>
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- explorer: count the costs
const KIND_FILL: Record<string, string> = { load: 'var(--global)', shared: 'var(--shared)', shuffle: 'var(--reg)' };
const KIND_WASH: Record<string, string> = { load: 'var(--global-wash)', shared: 'var(--shared-wash)', shuffle: 'var(--reg-wash)' };
const SHORT: Record<Kernel, string> = {
  get shared() { return tr('Fig. 10.10', 'Hình 10.10'); },
  get hybrid() { return tr('Fig. 10.14', 'Hình 10.14'); },
  get twostage() { return tr('Fig. 10.16', 'Hình 10.16'); },
};

function Timeline({ costs }: { costs: Costs[] }) {
  const LABEL = 92;
  const most = Math.max(...costs.map((c) => c.steps.length));
  const cw = Math.min(46, (VIEW_W - LABEL - 12) / most);
  const BAR = 50;
  const ROW = BAR + 30;
  const H = costs.length * ROW + 24;
  const warps = costs[0].blockDim / costs[0].width;
  return (
    <svg viewBox={`0 0 ${VIEW_W} ${H}`} role="img" aria-label={tr('Steps of each kernel, one after another, with warps busy and barriers', 'Các bước của mỗi kernel, nối tiếp nhau, với số warp bận và các rào chắn')}
      style={{ width: '100%', maxWidth: VIEW_W, minWidth: MIN_W, display: 'block' }}>
      <text x={LABEL} y={12} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>mỗi bước một cột, theo thứ tự; chiều cao: số warp bận (trên {warps}); nhạt: lane bận mà vô ích</>}>one column per step, in order; height: warps busy (of {warps}); pale: lanes busy for nothing</Translated></text>
      {costs.map((c, r) => {
        const y0 = 22 + r * ROW;
        return (
          <g key={c.kernel} data-kernel={c.kernel} data-barriers={c.barriers}>
            <text x={LABEL - 8} y={y0 + BAR / 2 + 4} fontSize="11" textAnchor="end">{SHORT[c.kernel]}</text>
            <line x1={LABEL} y1={y0 + BAR} x2={LABEL + most * cw} y2={y0 + BAR} stroke="var(--rule)" />
            {c.steps.map((s, i) => {
              const h = Math.max(2, (s.warps / warps) * BAR);
              const fullH = h;
              const usefulH = (s.useful / s.slots) * fullH;
              const x = LABEL + i * cw;
              return (
                <g key={i} data-step={i} data-warps={s.warps} data-slots={s.slots} data-useful={s.useful} data-barrier={s.barrierBefore ? 1 : 0}>
                  <title>{tr(`${s.label}: ${s.warps} warp${s.warps === 1 ? '' : 's'} busy, ${s.useful} of ${s.slots} lane slots useful`, `${s.label}: ${s.warps} warp bận, ${s.useful} trên ${s.slots} khe lane có ích`)}</title>
                  <rect x={x + 2} y={y0 + BAR - fullH} width={cw - 4} height={fullH} fill={KIND_WASH[s.what]} stroke={KIND_FILL[s.what]} strokeWidth="0.8" />
                  <rect x={x + 2} y={y0 + BAR - usefulH} width={cw - 4} height={usefulH} fill={KIND_FILL[s.what]} />
                  {s.barrierBefore ? <line x1={x + 0.5} y1={y0 - 4} x2={x + 0.5} y2={y0 + BAR + 4} stroke="var(--ink)" strokeWidth="2" /> : null}
                </g>
              );
            })}
            <text x={LABEL} y={y0 + BAR + 16} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
              <Translated
                vi={
                  <>
                    {c.barriers} rào chắn (các vạch dọc), {c.steps.length} bước
                  </>
                }
              >
                {c.barriers} barrier{c.barriers === 1 ? '' : 's'} (the vertical lines), {c.steps.length} steps
              </Translated>
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function CostExplorer() {
  const widths = useWidths();
  const [B, setB] = useState(1024);
  const [key, setKey] = useState('book');
  const chosen = widths.find((c) => c.key === key) ?? widths[0];
  const W = chosen.width;
  const costs = useMemo(() => KERNELS.map((k) => blockCosts(k, B, W)), [B, W]);
  const [s, h, t] = costs;
  const rows: { label: string; k: string; get: (c: Costs) => string | number }[] = [
    { label: tr('__syncthreads() executed', 'số lần thực thi __syncthreads()'), k: 'barriers', get: (c) => c.barriers },
    { label: tr('shared-memory writes', 'số lần ghi bộ nhớ chia sẻ'), k: 'writes', get: (c) => c.sharedWrites },
    { label: tr('shared-memory reads', 'số lần đọc bộ nhớ chia sẻ'), k: 'reads', get: (c) => c.sharedReads },
    { label: tr('shuffle steps, one after another', 'số bước shuffle, nối tiếp nhau'), k: 'rounds', get: (c) => c.shuffleRounds },
    { label: tr('warp-wide shuffles issued in the block', 'số shuffle toàn warp phát ra trong khối'), k: 'shuffles', get: (c) => c.warpShuffles },
    { label: tr('steps, one after another', 'số bước, nối tiếp nhau'), k: 'steps', get: (c) => c.steps.length },
    { label: tr('lane slots spent (busy warps × width, per step)', 'số khe lane đã dùng (warp bận × độ rộng, mỗi bước)'), k: 'slots', get: (c) => c.slots },
    { label: tr('useful additions', 'số phép cộng có ích'), k: 'useful', get: (c) => c.useful },
    { label: tr('lane slots spent on nothing', 'số khe lane dùng vô ích'), k: 'wasted', get: (c) => c.wasted },
  ];
  return (
    <Bench
      title={tr('Hybrid or two-stage: count the costs', 'Lai hay hai giai đoạn: đếm chi phí')}
      task={tr('At 1,024 threads and width 32: how many barriers does each kernel execute? Then try 256 threads, and width 64.', 'Với 1,024 luồng và độ rộng 32: mỗi kernel thực thi bao nhiêu rào chắn? Rồi thử 256 luồng, và độ rộng 64.')}
      foot={
        <>
          <Translated
            vi={
              <>
                Một khối rút gọn 2 × kích thước khối giá trị, cả ba kernel đặt cạnh nhau. Một khe lane là một lane của một warp bận
                trong một bước, thước đo tài nguyên thực thi của sách (p. 231). Mô hình chỉ đếm; nó không đo thời gian.
              </>
            }
          >
            One block reducing 2 × block size values, all three kernels side by side. A lane slot is one lane of one busy warp
            in one step, the book's measure of execution resources (p. 231). The model counts; it does not time.
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Kích thước khối">Block size</Translated>
          <select value={B} aria-label={tr('block size', 'kích thước khối')} onChange={(e) => setB(+e.target.value)}>
            {BLOCK_SIZES.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={chosen.key} aria-label={tr('warp width', 'độ rộng warp')} onChange={(e) => setKey(e.target.value)}>
            {widths.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.8rem' }} data-costs>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}><Translated vi="mỗi khối">per block</Translated></th>
            {costs.map((c) => <th key={c.kernel} className="num">{KERNEL_LABEL[c.kernel]}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.k} data-row={r.k}>
              <td style={{ paddingLeft: '0.6rem' }}>{r.label}</td>
              {costs.map((c) => <td key={c.kernel} className="num" data-kernel={c.kernel}>{r.get(c)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      <Timeline costs={costs} />
      <p className="narrate" role="status" style={{ margin: '0.7rem 0 0' }} data-costs-says>
        <Translated
          vi={
            <>
              Mọi kernel đều thực hiện cùng {s.useful} phép cộng có ích; chúng khác nhau ở những gì bao quanh. So với kernel lai,
              kernel hai giai đoạn tiết kiệm {h.barriers - t.barriers} trên {h.barriers} rào chắn và{' '}
              {h.sharedWrites + h.sharedReads - t.sharedWrites - t.sharedReads} lần truy cập bộ nhớ chia sẻ, và tốn thêm{' '}
              {t.wasted - h.wasted} khe lane cho những kết quả không ai đọc. Lập luận của sách cho sự đánh đổi này: khi dữ liệu đã
              được nạp, một phép rút gọn bị giới hạn bởi độ trễ, nên những lần chờ dài nó loại bỏ quan trọng hơn những lane rảnh nó
              thêm vào. Hiệu ứng nào thắng trên thiết bị của bạn là chuyện đo đạc, ở cuối trang này.
            </>
          }
        >
          Every kernel performs the same {s.useful} useful additions; they differ in what surrounds them. Against the hybrid
          kernel, the two-stage kernel saves {h.barriers - t.barriers} of {h.barriers} barriers and{' '}
          {h.sharedWrites + h.sharedReads - t.sharedWrites - t.sharedReads} shared-memory accesses, and spends{' '}
          {t.wasted - h.wasted} more lane slots on results nobody reads. The book's argument for the trade: once the data is
          loaded, a reduction is latency-bound, so the long waits it removes matter more than the idle lanes it adds. Which
          effect wins on your device is a measurement, at the bottom of this page.
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: read past the array
function PastArrayExplorer() {
  const widths = useWidths();
  const [B, setB] = useState(256);
  const [key, setKey] = useState('book');
  const [guarded, setGuarded] = useState(false);
  const chosen = widths.find((c) => c.key === key) ?? widths[0];
  const W = chosen.width;
  const r = useMemo(() => twoStage(ones(2 * B), B, W, guarded), [B, W, guarded]);
  const perRow = 32;
  const n = Math.max(r.slots.length, W);
  const rowsN = Math.ceil(n / perRow);
  const cw = (VIEW_W - 16) / perRow;
  const CH = 22;
  const GAPY = 34;
  const BLOCK = 12 + CH + GAPY + CH + 18;
  const H = rowsN * BLOCK + 6;
  const x = (k: number) => 8 + (k % perRow) * cw;
  const y = (k: number) => Math.floor(k / perRow) * BLOCK + 12;
  const past = r.slots.filter((s) => s.state === 'past').length;
  const garbage = r.slots.filter((s) => s.state === 'garbage').length;
  const reads = r.lanes.filter((l) => l.read != null);
  const badReads = reads.filter((l) => r.slots[l.read!].state !== 'written').length;
  return (
    <Bench
      title={tr('Read past the array', 'Đọc vượt quá mảng')}
      task={tr('As printed, at 256 threads: which slots does warp 0 read? Then 1,024 threads at width 64. Then guard the read.', 'Như bản in, với 256 luồng: warp 0 đọc những ô nào? Rồi 1,024 luồng với độ rộng 64. Rồi thêm điều kiện bảo vệ cho lần đọc.')}
      foot={
        <>
          <Translated
            vi={
              <>
                Trên: <code>partialSums_s</code>, mỗi warp một ô, được lane 0 của warp đó ghi ở giai đoạn 1 (mọi đầu vào đều là 1, nên
                tổng của một warp là 2 × độ rộng). Dưới: giá trị mà mỗi lane của warp 0 mang vào giai đoạn 2. Một ô nét đứt nằm quá
                cuối mảng.
              </>
            }
          >
            Top: <code>partialSums_s</code>, one slot per warp, written by that warp's lane 0 in stage 1 (all inputs are 1, so a
            warp's sum is 2 × width). Bottom: what each lane of warp 0 starts stage 2 with. A dashed slot lies past the end of
            the array.
          </Translated>
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Kích thước khối">Block size</Translated>
          <select value={B} aria-label={tr('block size', 'kích thước khối')} onChange={(e) => setB(+e.target.value)}>
            {BLOCK_SIZES.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={chosen.key} aria-label={tr('warp width', 'độ rộng warp')} onChange={(e) => setKey(e.target.value)}>
            {widths.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label={tr('Stage-2 read', 'Lần đọc ở giai đoạn 2')}>
          <button aria-pressed={!guarded} onClick={() => setGuarded(false)}><Translated vi="như bản in">as printed</Translated></button>
          <button aria-pressed={guarded} onClick={() => setGuarded(true)}><Translated vi="có bảo vệ, đệm bằng 0">guarded, padded with 0</Translated></button>
        </span>
      </div>
      <pre className="mono" style={{ margin: '0 0 0.6rem', whiteSpace: 'pre-wrap' }} data-read-code>
        {guarded
          ? `__shared__ float partialSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];   // ${r.arraySize} entries at this block size\nfloat partialSum = laneIdx() < warps ? partialSums_s[laneIdx()] : 0.0f;`
          : `__shared__ float partialSums_s[BLOCK_DIM/WARP_SIZE];   // ${B}/${W} = ${r.arraySize} entries\nfloat partialSum = partialSums_s[t];`}
      </pre>
      <svg viewBox={`0 0 ${VIEW_W} ${H}`} role="img" aria-label={tr('Per-warp sums in shared memory and what warp 0 reads', 'Tổng của từng warp trong bộ nhớ chia sẻ và những gì warp 0 đọc')} style={{ width: '100%', maxWidth: VIEW_W, minWidth: MIN_W, display: 'block' }}>
        {r.lanes.map((l) => {
          if (l.read == null) return null;
          const k = l.read;
          const ok = r.slots[k].state === 'written';
          return (
            <line key={'a' + l.lane} x1={x(k) + cw / 2} y1={y(k) + CH} x2={x(l.lane) + cw / 2} y2={y(l.lane) + CH + GAPY}
              stroke={ok ? 'var(--shared)' : 'var(--fail)'} strokeWidth="1" strokeDasharray={ok ? undefined : '3 2'} opacity="0.8" />
          );
        })}
        {range(n).map((k) => {
          const s = r.slots[k];
          if (!s) return null;
          const fill = s.state === 'written' ? 'var(--shared)' : s.state === 'garbage' ? 'var(--fail-wash)' : 'none';
          const stroke = s.state === 'written' ? 'var(--shared)' : 'var(--fail)';
          return (
            <g key={'s' + k} data-slot={k} data-state={s.state}>
              {k % 8 === 0 ? <text x={x(k) + cw / 2} y={y(k) - 3} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{k}</text> : null}
              <rect x={x(k) + 0.6} y={y(k)} width={cw - 1.2} height={CH} rx="2" fill={fill} stroke={stroke} strokeDasharray={s.state === 'past' ? '3 2' : undefined} />
              <text x={x(k) + cw / 2} y={y(k) + 14.5} fontSize="9" textAnchor="middle"
                style={{ fill: s.state === 'written' ? 'var(--page)' : 'var(--fail)' }} fontWeight={s.state === 'written' ? 400 : 700}>
                {s.value ?? '?'}
              </text>
            </g>
          );
        })}
        {r.arraySize < n && r.arraySize % perRow !== 0 ? (
          <line x1={x(r.arraySize) - 0.5} y1={y(r.arraySize) - 8} x2={x(r.arraySize) - 0.5} y2={y(r.arraySize) + CH + 6} stroke="var(--ink)" strokeWidth="2" />
        ) : null}
        {r.lanes.map((l) => {
          const yy = y(l.lane) + CH + GAPY;
          const bad = l.value == null;
          return (
            <g key={'l' + l.lane} data-lane={l.lane} data-value={l.value == null ? '?' : l.value}>
              <rect x={x(l.lane) + 0.6} y={yy} width={cw - 1.2} height={CH} rx="2"
                fill={bad ? 'var(--fail-wash)' : l.read == null ? 'var(--cell)' : 'var(--reg-wash)'} stroke={bad ? 'var(--fail)' : l.read == null ? 'var(--rule)' : 'var(--reg)'} />
              <text x={x(l.lane) + cw / 2} y={yy + 14.5} fontSize="9" textAnchor="middle" style={{ fill: bad ? 'var(--fail)' : l.read == null ? 'var(--ink-3)' : 'var(--ink)' }}>
                {l.value == null ? '?' : l.value}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="narrate" role="status" style={{ margin: '0.6rem 0 0' }} data-past-says>
        <Translated
          vi={
            <>
              {r.warps} warp rộng {W} ghi {r.warps} tổng từng phần vào một mảng {r.arraySize} phần tử.{' '}
              {guarded
                ? `Lane 0 đến ${r.warps - 1} đọc chúng; ${W - r.warps} lane còn lại của warp 0 bắt đầu với 0, phần tử đơn vị của +. ${garbage ? `${garbage} ô không dùng không bao giờ được đọc. ` : ''}`
                : badReads
                  ? `Cả ${W} lane của warp 0 đều đọc một ô: ${past} lần đọc trong số đó rơi quá cuối mảng, nơi không có gì được ghi, và tổng là không xác định. `
                  : `Cả ${W} lane của warp 0 đều đọc một ô, và mọi ô đều đã được ghi: đây là hình dạng duy nhất mà bản in khớp. `}
            </>
          }
        >
          {r.warps} warp{r.warps === 1 ? '' : 's'} of {W} write {r.warps} partial sum{r.warps === 1 ? '' : 's'} into an array of {r.arraySize}.{' '}
          {guarded
            ? `Lanes 0 to ${r.warps - 1} read them; the other ${W - r.warps} lanes of warp 0 start with 0, the identity of +. ${garbage ? `The ${garbage} unused slots are never read. ` : ''}`
            : badReads
              ? `All ${W} lanes of warp 0 read a slot: ${past} of those reads land past the end of the array, where nothing was written, and the sum is undefined. `
              : `All ${W} lanes of warp 0 read a slot, and every slot was written: this is the one shape the printed listing fits. `}
        </Translated>
        <span className={'chip ' + (r.correct ? 'ok' : 'fail')} data-result={r.correct ? 'ok' : 'fail'}>
          {r.result == null ? tr('undefined', 'không xác định') : r.result}
        </span>{' '}
        <span className="muted"><Translated vi={<>(đáp án đúng {r.expected})</>}>(right answer {r.expected})</Translated></span>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the measurement
function measured(reports: Reports) {
  return realBackends(reports).flatMap((b) => {
    const rows = benchRows(reports[b], LAB).filter((r) => r.name === 'rows');
    const blocks = [...new Set(rows.map((r) => num(r.block)).filter((v): v is number => v != null))];
    const block = blocks.includes(256) ? 256 : blocks[0];
    const pick = (k: string) => num(rows.find((r) => r.kernel === k && num(r.block) === block)?.seconds);
    const shared = pick('shared');
    const warp = pick('warp');
    return shared && warp && block
      ? [{ b, device: String(reports[b].device.name ?? b), block, shared, warp, speedup: shared / warp }]
      : [];
  });
}

function MeasuredSpeedup({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = measured(reports);
  if (!found.length)
    return (
      <p data-measured="none">
        <Translated
          vi={
            <>
              Chưa có số đo nào trên máy này. Chạy <code>python3 labs/pmpp.py run 10.1 --solution --bench</code> trên một máy có
              GPU, rồi tải lại trang: ô này sẽ so dự đoán của bạn với kết quả.
            </>
          }
        >
          No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 10.1 --solution --bench</code> on a machine
          with a GPU, then reload the page: this box will compare your prediction with the result.
        </Translated>
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-measured={f.b}>
          <Translated
            vi={
              <>
                Trên <b>{f.device}</b> ({f.b}), các hàng {2 * f.block} giá trị được rút gọn bởi các khối {f.block}: kernel bộ nhớ chia
                sẻ mất {fmt(f.shared * 1e3)} ms và kernel hai giai đoạn {fmt(f.warp * 1e3)} ms, nên kernel hai giai đoạn chạy nhanh
                gấp <b>{fmt(f.speedup)}×</b>.{' '}
                {predicted > f.speedup * 1.5
                  ? 'Bạn đã kỳ vọng nhiều hơn đáng kể. Bỏ rào chắn chỉ giúp được đúng bằng cái giá của chúng trên thiết bị này, và cả hai kernel đọc đúng cùng dữ liệu từ bộ nhớ toàn cục.'
                  : predicted < f.speedup / 1.5
                    ? 'Bạn đã kỳ vọng ít hơn đáng kể.'
                    : 'Kết quả gần với dự đoán của bạn.'}
              </>
            }
          >
            On <b>{f.device}</b> ({f.b}), rows of {2 * f.block} values reduced by blocks of {f.block}: the shared-memory kernel
            took {fmt(f.shared * 1e3)} ms and the two-stage kernel {fmt(f.warp * 1e3)} ms, so the two-stage kernel ran{' '}
            <b>{fmt(f.speedup)}×</b> as fast.{' '}
            {predicted > f.speedup * 1.5
              ? 'You expected considerably more. Removing barriers helps only as much as they cost on this device, and both kernels read exactly the same data from global memory.'
              : predicted < f.speedup / 1.5
                ? 'You expected considerably less.'
                : 'That is close to what you predicted.'}
          </Translated>
        </p>
      ))}
      <p className="muted"><Translated vi="Đây là số đo từ máy của bạn, không phải giá trị kỳ vọng. Một thiết bị khác có thể chênh lệch rất nhiều theo cả hai chiều.">These are measurements from your machine, not expected values. Another device can differ by a wide margin in either direction.</Translated></p>
    </>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS108: Record<string, Question> = {
  'ch10/10.8/q-barriers': {
    kind: 'numeric',
    id: 'ch10/10.8/q-barriers',
    prompt: (
      <Translated
        vi={
          <>
            Một khối 1,024 luồng với warp rộng 32. Kernel lai của Hình 10.14 thực thi{' '}
            <code>__syncthreads()</code> bao nhiêu lần?
          </>
        }
      >
        <>
          A block of 1,024 threads in 32-wide warps. How many times does the hybrid kernel of Fig. 10.14 execute{' '}
          <code>__syncthreads()</code>?
        </>
      </Translated>
    ),
    answer: 6,
    explain: (
      <Translated
        vi={
          <p>
            Năm lần trong vòng lặp, mỗi bước nhảy một lần (512, 256, 128, 64, 32), và một lần sau nó (dòng 11). Hình 10.10
            thực thi 10 lần, mỗi bước nhảy từ 512 xuống 1 một lần; kernel hai giai đoạn thực thi 1 lần. Tổng quát, kernel lai
            thực thi log2(block) − log2(width) + 1 lần.
          </p>
        }
      >
        <p>
          Five in the loop, one per stride (512, 256, 128, 64, 32), and one after it (line 11). Fig. 10.10 executes 10, one
          per stride from 512 down to 1; the two-stage kernel executes 1. In general the hybrid kernel executes
          log2(block) − log2(width) + 1.
        </p>
      </Translated>
    ),
    source: 'book Figs. 10.10, 10.14, 10.16 (pp. 237, 240, 243); model: src/sim/shuffle.ts',
    vi: { source: 'Hình 10.10, 10.14, 10.16 của sách (pp. 237, 240, 243); mô hình: src/sim/shuffle.ts' },
  },
  'ch10/10.8/q-latency': {
    kind: 'choice',
    id: 'ch10/10.8/q-latency',
    prompt: (
      <Translated vi={<>Vì sao một kernel có nhiều lane tính ra kết quả không ai dùng hơn lại có thể nhanh hơn một kernel có ít lane như vậy hơn?</>}>
        <>Why can a kernel in which more lanes compute results nobody uses be faster than one with fewer such lanes?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Sau lần nạp, nó bị giới hạn bởi độ trễ: rào chắn và các chuyến đi tới bộ nhớ chia sẻ là những lần chờ dài, còn lane rảnh tốn rất ít.">After the load it is latency-bound: barriers and shared-memory trips are long waits, and idle lanes cost little.</Translated>,
        correct: true,
      },
      { label: <Translated vi="Lane lãng phí là miễn phí: phần cứng phát hiện những phép cộng có kết quả không bao giờ được đọc và bỏ qua chúng.">Wasted lanes are free: the hardware detects additions whose results are never read and skips them.</Translated>, why: <Translated vi="Không gì bỏ qua chúng; mọi lane của warp đều thực thi mọi shuffle và phép cộng.">Nothing skips them; every lane of the warp executes every shuffle and add.</Translated> },
      { label: <Translated vi="Nhiều lane làm việc hơn thì tăng occupancy, nên nhiều khối hơn vừa trên mỗi SM và che độ trễ bộ nhớ cho nhau.">More lanes doing work raise occupancy, so more blocks fit on each SM and hide each other's memory latency.</Translated>, why: <Translated vi="Occupancy phụ thuộc vào tài nguyên mà một khối giữ chỗ, không phải số lane làm việc có ích.">Occupancy depends on resources a block reserves, not on how many lanes do useful work.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lập luận của sách (pp. 243–244): các chu kỳ tiết kiệm được nhờ cho lane nghỉ phần lớn sẽ thành chu kỳ đình trệ.
            Kernel hai giai đoạn đổi một vài phép toán có độ trễ dài (rào chắn, truy cập bộ nhớ chia sẻ) lấy nhiều phép rẻ
            (shuffle và phép cộng ở những lane có kết quả không bao giờ được đọc). Trong một kernel phần lớn thời gian là chờ,
            bỏ bớt các lần chờ mới là điều quan trọng.
          </p>
        }
      >
        <p>
          The book's argument (pp. 243–244): the cycles saved by idling lanes would mostly become stall cycles. The
          two-stage kernel trades a few long-latency operations (barriers, shared-memory
          accesses) for many cheap ones (shuffles and adds in lanes whose results are never read). In a kernel that is mostly
          waiting, removing waits is what counts.
        </p>
      </Translated>
    ),
    source: 'book §10.8, pp. 243–244',
    vi: { source: 'sách §10.8, pp. 243–244' },
  },
  'ch10/10.8/q-guard-256': {
    kind: 'choice',
    id: 'ch10/10.8/q-guard-256',
    prompt: (
      <Translated
        vi={
          <>
            Kernel hai giai đoạn đúng như bản in (<code>partialSums_s[BLOCK_DIM/WARP_SIZE]</code>, mọi lane của warp 0 đọc{' '}
            <code>partialSums_s[t]</code>), khởi chạy với 256 luồng mỗi khối, warp rộng 32. Ở giai đoạn 2, warp 0 đọc gì?
          </>
        }
      >
        <>
          The two-stage kernel exactly as printed (<code>partialSums_s[BLOCK_DIM/WARP_SIZE]</code>, every lane of warp 0 reads{' '}
          <code>partialSums_s[t]</code>), launched with 256 threads per block of 32-wide warps. What does warp 0 read in stage 2?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="8 tổng của từng warp, rồi 24 lần đọc vượt quá cuối mảng: tổng là không xác định.">8 per-warp sums, then 24 reads past the end of the array: the sum is undefined.</Translated>, correct: true },
      { label: <Translated vi="8 tổng của từng warp và 24 số không, vì bộ nhớ chia sẻ của một kernel ban đầu được đặt về không.">8 per-warp sums and 24 zeros, because a kernel's shared memory starts out zeroed.</Translated>, why: <Translated vi="Bộ nhớ chia sẻ không được khởi tạo, và dù sao các lần đọc này cũng vượt quá cuối mảng. Số không ở đó sẽ che giấu lỗi, không sửa nó.">Shared memory is not initialised, and these reads are past the end of the array anyway. Zeros there would hide the bug, not fix it.</Translated> },
      { label: <Translated vi="32 tổng của từng warp: mảng có một phần tử cho mỗi lane của warp 0, nên mọi khối đều có 32.">32 per-warp sums: the array has one entry per lane of warp 0, so 32 in every block.</Translated>, why: <Translated vi="Nó có BLOCK_DIM / WARP_SIZE = 256 / 32 = 8.">It has BLOCK_DIM / WARP_SIZE = 256 / 32 = 8.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            8 warp ghi 8 tổng vào một mảng 8 phần tử; lane 8 đến 31 đọc vượt quá nó. Điều kiện bảo vệ chỉ là một phép so sánh:
            chỉ đọc khi <code>lane &lt; blockDim.x / warpSize</code>, và cộng 0 trong trường hợp còn lại. Bản in chỉ đúng khi số
            warp đúng bằng số lane, một khối 1,024 luồng với warp rộng 32 (chú thích 5 nghĩ đến trường hợp đó). Lab 10.1 định
            kích thước mảng cho khối lớn nhất và warp hẹp nhất, và chỉ đọc khi <code>laneIdx() &lt; warps</code>.
          </p>
        }
      >
        <p>
          8 warps write 8 sums into an array of 8; lanes 8 to 31 read past it. The guard is one comparison: read only
          where <code>lane &lt; blockDim.x / warpSize</code>, and add 0 otherwise. The listing is right only when there are exactly
          as many warps as lanes, a block of 1,024 threads of 32-wide warps (footnote 5 has that case in mind). Lab 10.1 sizes
          the array for the largest block and the narrowest warp, and reads only <code>laneIdx() &lt; warps</code>.
        </p>
      </Translated>
    ),
    source: 'book Fig. 10.16, p. 243 (notes/ch10.md, "Unclear in the PDF"); ch10-facts B2, D1, D4',
    vi: { source: 'Hình 10.16 của sách, p. 243 (notes/ch10.md, "Unclear in the PDF"); ch10-facts B2, D1, D4' },
  },
  'ch10/10.8/q-w64-warps': {
    kind: 'numeric',
    id: 'ch10/10.8/q-w64-warps',
    prompt: (
      <Translated vi={<>Một khối 1,024 luồng trên phần cứng có warp rộng 64. Giai đoạn 1 tạo ra bao nhiêu tổng từng phần của từng warp?</>}>
        <>A block of 1,024 threads on hardware with 64-wide warps. How many per-warp partial sums does stage 1 produce?</>
      </Translated>
    ),
    answer: 16,
    explain: (
      <Translated
        vi={
          <p>
            1,024 / 64 = 16 warp, nên có 16 tổng, trong khi warp 0 có 64 lane. Đọc không có bảo vệ, 48 lane đọc vượt quá chúng.
            Trên phần cứng warp rộng 64, bản in sai ở mọi kích thước khối.
          </p>
        }
      >
        <p>
          1,024 / 64 = 16 warps, so 16 sums, while warp 0 has 64 lanes. Read unguarded, 48 lanes read past them. On 64-wide
          hardware the printed listing is wrong at every block size.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch10-facts.md B2', // i18n-ignore: citation only
  },
  'ch10/10.8/q-metal-lanes': {
    kind: 'choice',
    id: 'ch10/10.8/q-metal-lanes',
    prompt: (
      <Translated
        vi={
          <>
            Một softmax trên Metal giao cho mỗi hàng một threadgroup, đủ lớn để trải trên nhiều SIMD-group. Mỗi luồng cộng các số
            hạng <code>exp(x − max)</code> của nó vào <code>e</code>, rồi gọi <code>float s = simd_sum(e);</code>. Điều gì phải
            xảy ra trước khi luồng 100, vốn không nằm trong SIMD-group đầu tiên, được chia các số hạng của nó cho tổng của hàng?
          </>
        }
      >
        <>
          A Metal softmax gives each row one threadgroup, large enough to span several SIMD-groups. Each thread adds its{' '}
          <code>exp(x − max)</code> terms into <code>e</code>, then calls <code>float s = simd_sum(e);</code>. What has to
          happen before thread 100, which is not in the first SIMD-group, may divide its terms by the row's sum?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Không gì cả: simd_sum trao kết quả cho mọi luồng, nên s của luồng 100 đã là tổng của hàng.">Nothing: simd_sum hands its result to every thread, so thread 100's s is already the row's sum.</Translated>,
        why: <Translated vi="Cho mọi luồng đang hoạt động trong SIMD-group của nó. s của luồng 100 chỉ là tổng các số hạng của SIMD-group của chính nó.">To every active thread of its SIMD-group. Thread 100's s is the sum of its own SIMD-group's terms only.</Translated>,
      },
      {
        label: <Translated vi="Lane 0 của mỗi SIMD-group lưu s của nó; một rào chắn; một SIMD-group cộng các giá trị đó và lưu tổng; một rào chắn; tất cả đọc nó.">Each SIMD-group's lane 0 stores its s; a barrier; one SIMD-group sums those and stores the total; a barrier; all read it.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Luồng 100 lấy s của lane 0 bằng simd_shuffle, vì simd_sum để tổng của hàng ở lane 0 của threadgroup.">Thread 100 fetches lane 0's s with simd_shuffle, since simd_sum leaves the row's sum in lane 0 of the threadgroup.</Translated>,
        why: <Translated vi="simd_sum đã trao kết quả cho mọi lane đang hoạt động của nhóm. Phần còn thiếu là các nhóm khác.">simd_sum already gives its result to every active lane of the group. The missing part is the other groups.</Translated>,
      },
      {
        label: <Translated vi="Một threadgroup_barrier(mem_flags::mem_threadgroup) sau simd_sum, để mọi SIMD-group đã xong trước khi bất kỳ luồng nào chia.">One threadgroup_barrier(mem_flags::mem_threadgroup) after the simd_sum, so that every SIMD-group has finished before any thread divides.</Translated>,
        why: <Translated vi="Một rào chắn sắp thứ tự các truy cập bộ nhớ; nó không kết hợp tổng của các nhóm, vốn vẫn nằm trong các thanh ghi riêng.">A barrier orders memory accesses; it does not combine the groups' sums, which are still in separate registers.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đặc tả của Apple nói <code>simd_sum</code> phát kết quả "tới mọi luồng đang hoạt động trong SIMD-group",
            không phải threadgroup. Kết hợp các nhóm là giai đoạn hai của Hình 10.16, và vì <code>simd_sum</code> thứ hai của Metal{' '}
            chỉ để tổng của hàng trong SIMD-group đã chạy nó, các nhóm khác đọc lại nó từ bộ nhớ threadgroup sau thêm một rào
            chắn. <code>softmax_single_row</code> của MLX làm đúng như vậy, qua{' '}
            <code>local_max[simd_group_id]</code> rồi <code>local_max[0]</code> cho max, và tương tự cho sum.
          </p>
        }
      >
        <p>
          Apple's specification says <code>simd_sum</code> broadcasts its result "to all active threads in the SIMD-group",
          not the threadgroup. Combining the groups is Fig. 10.16's second stage, and because Metal's second{' '}
          <code>simd_sum</code> leaves the row sum only in the SIMD-group that ran it, the others read it back from
          threadgroup memory after one more barrier. MLX's <code>softmax_single_row</code> does exactly this, through{' '}
          <code>local_max[simd_group_id]</code> and then <code>local_max[0]</code> for the max, and the same for the sum.
        </p>
      </Translated>
    ),
    source: 'MSL spec 6.10.2 (notes/today/ch10-facts.md C1); MLX softmax.h (ch10-facts D3)', // i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- the section
export function S108() {
  return (
    <>
      <Translated
        vi={
          <p>
            Kernel lai của mục 10.7 vẫn trả giá cho bộ nhớ chia sẻ và một rào chắn ở mỗi vòng lặp cho đến khi chỉ còn một warp:
            sáu rào chắn trong một khối 1,024 luồng trước khi <code>warp_reduce</code> tiếp quản. Bước tiếp theo là dùng phép rút
            gọn warp ngay từ đầu.
          </p>
        }
      >
        <p>
          The hybrid kernel of section 10.7 still pays for shared memory and a barrier in every iteration until one warp is
          left: six barriers in a block of 1,024 threads before <code>warp_reduce</code> takes over. The next step is to use
          warp reduction from the very start.
        </p>
      </Translated>
      <Source section="10.8" pages="241" />

      <h2><Translated vi="Hai giai đoạn, một rào chắn">Two stages, one barrier</Translated></h2>
      <Translated
        vi={
          <p>
            Ở giai đoạn 1, mỗi luồng cộng hai phần tử đầu vào của nó vào một thanh ghi, và mỗi warp rút gọn 32 giá trị của chính
            nó bằng <code>warp_reduce</code>. Mọi warp làm việc cùng lúc, và không warp nào chạm vào bộ nhớ chia sẻ. Sau đó lane
            0 của mỗi warp ghi tổng của warp mình vào <code>partialSums_s[warpIdx()]</code>, và khối gặp nhau tại{' '}
            <code>__syncthreads()</code> duy nhất của nó. Ở giai đoạn 2, warp 0 nạp các tổng của từng warp, mỗi lane một tổng, và
            chạy <code>warp_reduce</code> thêm một lần; luồng 0 ghi kết quả.
          </p>
        }
      >
        <p>
          In stage 1 every thread adds its two input elements into a register, and every warp reduces its own 32 values with{' '}
          <code>warp_reduce</code>. All warps work at once, and none of them touches shared memory. Lane 0 of each warp then
          writes its warp's sum to <code>partialSums_s[warpIdx()]</code>, and the block meets at its only{' '}
          <code>__syncthreads()</code>. In stage 2, warp 0 loads the per-warp sums, one per lane, and runs{' '}
          <code>warp_reduce</code> once more; thread 0 writes the result.
        </p>
      </Translated>
      <Source pages="241–243" figs={tr('Figs. 10.15, 10.16', 'Hình 10.15, 10.16')} />
      <Translated
        vi={
          <p>
            Vì sao một warp là đủ cho giai đoạn 2: một khối có tối đa 1,024 luồng và một warp có 32 lane, nên có tối đa 32 tổng
            từng phần, mỗi lane của warp 0 một tổng. Giống mọi kernel cho đến giờ, kernel này vẫn chỉ rút gọn 2,048 giá trị của
            một khối; mục 10.9 gỡ giới hạn đó.
          </p>
        }
      >
        <p>
          Why one warp is enough for stage 2: a block has at most 1,024 threads and a warp 32 lanes, so there are at most 32
          partial sums, one per lane of warp 0. Like every kernel so far, this one still reduces a single block's 2,048 values;
          section 10.9 lifts that limit.
        </p>
      </Translated>
      <Source pages="242–243" />
      <TwoStageListing />
      <Translated
        vi={
          <p>
            So với Hình 10.16, phiên bản của lab lấy hàng mà mỗi khối rút gọn từ <code>blockIdx.x</code> (mỗi hàng một khối, để
            có thể benchmark trên nhiều hàng), dùng <code>warpSize</code> và <code>warp_reduce</code> khả chuyển của mục 10.7,
            định kích thước <code>partialSums_s</code> cho khối lớn nhất và warp hẹp nhất, và chỉ cho số lane bằng số warp đọc
            nó. Thay đổi cuối đó sửa một lỗi trong bản in, thứ mà trình khám phá thứ hai bên dưới mổ xẻ. Trên Metal, lane, chỉ
            số SIMD-group và số nhóm được truyền vào dưới dạng attribute, vì Apple để việc chia threadgroup thành các SIMD-group
            cho phần hiện thực quyết định.
          </p>
        }
      >
        <p>
          Compared with Fig. 10.16, the lab's version takes the row each block reduces from <code>blockIdx.x</code> (one block
          per row, so it can be benchmarked on many rows), uses <code>warpSize</code> and the portable{' '}
          <code>warp_reduce</code> of section 10.7, sizes <code>partialSums_s</code> for the largest block and the narrowest
          warp, and lets only as many lanes read it as there are warps. That last change fixes a bug in the listing, which
          the second explorer below takes apart. On Metal the lane, SIMD-group index and group count arrive as attributes,
          because Apple leaves the split of a threadgroup into SIMD-groups to the implementation.
        </p>
      </Translated>

      <h2><Translated vi="Nó tiết kiệm gì, và tốn gì">What it saves, and what it spends</Translated></h2>
      <Translated
        vi={
          <p>
            Sách thẳng thắn về sự đánh đổi. Kernel hai giai đoạn chỉ cần bộ nhớ chia sẻ và một rào chắn một lần, ở ranh giới giữa
            hai giai đoạn. Nhưng nếu kernel lai cho cả warp nghỉ khi cây co lại, thì ở đây mọi warp đều bận suốt giai đoạn 1, và ở
            mỗi bước shuffle phần lớn lane của nó tính một thứ không bao giờ đến được lane 0. Sách gọi đây là "more control
            divergence". Nói chặt chẽ, <code>warp_reduce</code> hoàn toàn không có nhánh; thứ tăng lên là số lane làm việc không ai
            đọc, và trình khám phá đếm đúng điều đó.
          </p>
        }
      >
        <p>
          The book is candid about the trade. The two-stage kernel needs shared memory and a barrier only once, at the boundary
          between the stages. But where the hybrid kernel retired whole warps as the tree shrank, here every warp stays busy
          through stage 1, and in each shuffle step most of its lanes compute something that never reaches lane 0. The book
          calls this "more control divergence". Strictly, <code>warp_reduce</code> has no branch at all; what grows is the
          number of lanes doing work nobody reads, and the explorer counts exactly that.
        </p>
      </Translated>
      <NoteGroup>
        <Source pages="243" />
        <Simplification>
          <Translated
            vi={
              <p>
                Mô hình đếm số phép toán mỗi bước cho một khối: rào chắn, số lần đọc và ghi bộ nhớ chia sẻ, shuffle, và khe lane. Nó
                không mô hình hóa thời gian, độ trễ, bộ nhớ đệm hay lập lịch. Độ trễ là lập luận của sách, không phải thứ mô hình
                tính ra.
              </p>
            }
          >
            <p>
              The model counts operations per step for one block: barriers, shared-memory reads and writes, shuffles, and lane
              slots. It does not model time, latency, caches or scheduling. Latency is the book's argument, not something the
              model computes.
            </p>
          </Translated>
        </Simplification>
      </NoteGroup>
      <CostExplorer />
      <Ask q={questionsS108['ch10/10.8/q-barriers']} label={tr('Work it out', 'Tự tính ra')} />
      <Translated
        vi={
          <p>
            Vì sao tốn thêm lane để tiết kiệm rào chắn? Khi dữ liệu đã được nạp, phần còn lại của một phép rút gọn ngắn và phần
            lớn là chờ: nó <b>bị giới hạn bởi độ trễ</b>. Một rào chắn và một chuyến đi khứ hồi tới bộ nhớ chia sẻ là những lần
            chờ dài. Một lane rảnh trong một warp đằng nào cũng phát lệnh thì gần như không tốn gì, và nếu bỏ nó đi, sách lập
            luận, các chu kỳ tiết kiệm được "will likely be replaced by stall cycles". Vậy hãy bỏ những lần chờ dài, và chấp
            nhận các lane rảnh.
          </p>
        }
      >
        <p>
          Why spend more lanes to save barriers? Once the data has been loaded, the rest of a reduction is short and mostly
          waiting: it is <b>latency-bound</b>. A barrier and a shared-memory round trip are long waits. An idle lane in a warp
          that issues the instruction anyway costs almost nothing, and if it were removed, the book argues, the cycles saved
          "will likely be replaced by stall cycles". Remove the long waits, then, and accept the idle lanes.
        </p>
      </Translated>
      <Source pages="243–244" />
      <Ask q={questionsS108['ch10/10.8/q-latency']} />

      <h2><Translated vi="Một bản in chỉ khớp một kích thước khối">A listing that fits one block size</Translated></h2>
      <Translated
        vi={
          <p>
            Hình 10.16 khai báo <code>partialSums_s[BLOCK_DIM/WARP_SIZE]</code> rồi cho mọi lane của warp 0 đọc{' '}
            <code>partialSums_s[t]</code>. Cả hai đều đúng với 1,024 luồng trong warp rộng 32: 32 warp, 32 phần tử, 32 lane.
            Với bất kỳ khối nhỏ hơn nào, mảng có ít phần tử hơn số lane của warp 0, và các lane thừa đọc vượt quá cuối nó. Trên
            phần cứng warp rộng 64, một khối 1,024 luồng chỉ có 16 warp, nên lần đọc vượt biên ở mọi kích thước khối.
          </p>
        }
      >
        <p>
          Fig. 10.16 declares <code>partialSums_s[BLOCK_DIM/WARP_SIZE]</code> and then lets every lane of warp 0 read{' '}
          <code>partialSums_s[t]</code>. Both are right for 1,024 threads in 32-wide warps: 32 warps, 32 entries, 32 lanes.
          For any smaller block the array has fewer entries than warp 0 has lanes, and the extra lanes read past its end. On
          64-wide hardware a block of 1,024 threads has only 16 warps, so the read overruns at every block size.
        </p>
      </Translated>
      <NoteGroup>
        <Source pages="243" figs={tr('Fig. 10.16', 'Hình 10.16')} />
        <Today checked="2026-11-20 (ch10-facts D1, D3, D4)">{/* i18n-ignore: date and citation */}
          <Translated
            vi={
              <p>
                <code>block_reduce</code> của llama.cpp, các phép rút gọn Metal của MLX và <code>BlockReduceSum</code> của PyTorch
                đều bảo vệ lần đọc đó và đệm bằng phần tử đơn vị.
              </p>
            }
          >
            <p>
              llama.cpp's <code>block_reduce</code>, MLX's Metal reductions and PyTorch's <code>BlockReduceSum</code> all
              guard that read and pad with the identity.
            </p>
          </Translated>
        </Today>
      </NoteGroup>
      <Translated
        vi={
          <p>
            Bản sửa tốn một phép so sánh: định kích thước mảng cho khối lớn nhất và warp hẹp nhất mà backend nào đó ghi trong
            tài liệu (32 lane), và cho lane <i>t</i> đọc ô <i>t</i> chỉ khi warp <i>t</i> tồn tại, nếu không thì cộng 0. So sánh
            hai phiên bản bên dưới với 256 luồng, và với 1,024 luồng trong warp rộng 64.
          </p>
        }
      >
        <p>
          The fix costs one comparison: size the array for the largest block and the narrowest warp any backend documents
          (32 lanes), and let lane <i>t</i> read slot <i>t</i> only if warp <i>t</i> exists, adding 0 otherwise. Compare the
          two versions below at 256 threads, and at 1,024 threads in 64-wide warps.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một ô chưa bao giờ được ghi, hoặc nằm quá mảng, được vẽ là không xác định. Trình giả lập của khóa học lấp bộ nhớ chia
              sẻ bằng byte 0xCD, nên lỗi lộ ra; trên GPU, bộ nhớ có thể tình cờ chứa số không và che giấu nó.
            </p>
          }
        >
          <p>
            A slot that was never written, or that lies past the array, is drawn as undefined. The course's emulator fills
            shared memory with the byte 0xCD, so the bug shows; on a GPU the memory may happen to hold zeros and hide it.
          </p>
        </Translated>
      </Simplification>
      <PastArrayExplorer />
      <Ask q={questionsS108['ch10/10.8/q-guard-256']} />
      <Ask q={questionsS108['ch10/10.8/q-w64-warps']} label={tr('Work it out', 'Tự tính ra')} />

      <h2><Translated vi="Cùng hàm đó trong các thư viện thật">The same function in real libraries</Translated></h2>
      <Translated
        vi={
          <p>
            Bạn sẽ thấy kernel này bên trong các kernel softmax của những thư viện bạn dùng. <code>soft_max_f32</code> của{' '}
            llama.cpp gọi một <code>block_reduce</code> có cấu trúc này hai lần, một lần cho giá trị lớn nhất của hàng và một lần
            cho tổng các hàm mũ, và đặt một rào chắn giữa hai lần gọi vì chúng dùng chung buffer.{' '}
            <code>BlockReduceSum</code> của PyTorch bắt đầu bằng một rào chắn vì cùng lý do, để có thể được gọi hai lần liên tiếp.
            Trên Metal, <code>softmax_single_row</code> của MLX và <code>kernel_op_sum_f32</code> của llama.cpp làm tương tự với{' '}
            <code>simd_max</code> và <code>simd_sum</code>, vốn trả kết quả về mọi lane của SIMD-group.
          </p>
        }
      >
        <p>
          You will find this kernel inside the softmax kernels of the libraries you use. llama.cpp's{' '}
          <code>soft_max_f32</code> calls a <code>block_reduce</code> with this structure twice, once for the row's maximum and
          once for the sum of exponentials, and puts a barrier between the two calls because they share the buffer. PyTorch's{' '}
          <code>BlockReduceSum</code> starts with a barrier for the same reason, so it can be called twice in a row. On Metal,
          MLX's <code>softmax_single_row</code> and llama.cpp's <code>kernel_op_sum_f32</code> do the same with{' '}
          <code>simd_max</code> and <code>simd_sum</code>, which return the result to every lane of the SIMD-group.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Mã nguồn được đọc ngày 2026-11-20 (notes/today/ch10-facts.md C1, D1 đến D4). Sách không nhắc đến các thư viện này.
              Lab 10.3 xây một softmax và một argmax trên logits; xem <a href={href('ch10', 'labs')}>trang lab</a>.
            </p>
          }
        >
          <p>
            Source read on 2026-11-20 (notes/today/ch10-facts.md C1, D1 to D4). The book does not mention these libraries.
            Lab 10.3 builds a softmax and an argmax over logits; see the <a href={href('ch10', 'labs')}>labs page</a>.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS108['ch10/10.8/q-metal-lanes']} />

      <h2><Translated vi="Đo nó">Measure it</Translated></h2>
      <Translated
        vi={
          <p>
            Lab 10.1 đo thời gian kernel bộ nhớ chia sẻ của Hình 10.10 và kernel hai giai đoạn trên cùng dữ liệu: nhiều hàng 512
            giá trị, mỗi hàng một khối 256 luồng, cách một softmax trên các hàng ngắn sẽ chạy. Hãy chốt một con số trước.
          </p>
        }
      >
        <p>
          Lab 10.1 times the shared-memory kernel of Fig. 10.10 and the two-stage kernel on the same data: many rows of 512
          values, one block of 256 threads per row, the way a softmax over short rows would run. Commit to a number first.
        </p>
      </Translated>
      <Predict
        id="ch10/10.8/p-twostage"
        prompt={
          <Translated
            vi={
              <>
                Trên thiết bị <em>của bạn</em>, kernel hai giai đoạn sẽ chạy nhanh gấp bao nhiêu lần kernel bộ nhớ chia sẻ (Hình
                10.10) trên các hàng 512 giá trị?
              </>
            }
          >
            <>
              On <em>your</em> device, how many times as fast as the shared-memory kernel (Fig. 10.10) will the two-stage kernel
              run on rows of 512 values?
            </>
          </Translated>
        }
        unit={tr('× as fast', '× nhanh hơn')}
      >
        {(v) => <MeasuredSpeedup predicted={v} />}
      </Predict>
    </>
  );
}
