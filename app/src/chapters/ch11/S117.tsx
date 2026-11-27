import { useEffect, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Bench, Beyond, NoteGroup, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  bankMap,
  conflictDegree,
  sharedAccesses,
  worstDegree,
  type BankSide,
  type CoarseKernel,
  type Phase,
} from '../../sim/coarsescan';
import { functionSource } from '../../sim/toolchain';

const STATUS = {
  get printed() {
    return tr(
      'front-end-checked; emulator-checked (N a multiple of the block segment); unverified on CUDA hardware; as printed',
      'front-end-checked; emulator-checked (N là bội số của đoạn khối); unverified trên phần cứng CUDA; như bản in',
    );
  },
} as const;
const FILES: Record<CoarseKernel, string> = {
  'fig11.12': 'ch11/snippets/s116_coarsened_as_printed.cu',
  'fig11.13': 'ch11/snippets/s117_regtiled_as_printed.cu',
};
const LAB = 'ch11/lab2_coarsened';
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', buffer_s: 'shared', threadSums: 'shared',
  buffer_r: 'reg', threadSum: 'reg', prevPartialSum: 'reg', blockSegment: 'reg', threadSegment: 'reg', c: 'reg',
};
const range = (n: number, from = 0) => Array.from({ length: n }, (_, i) => from + i);
const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2).replace(/0$/, ''));


function useNarrow(px = 600): boolean {
  const query = `(max-width: ${px}px)`; // i18n-ignore: CSS media query
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setNarrow(m.matches);
    on();
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, [query]);
  return narrow;
}

// ---------------------------------------------------------------- explorer: what registers save
const PHASES: Phase[] = ['load', 'scan', 'total', 'hand-off', 'add', 'store'];
const KERNEL_LABEL: Record<CoarseKernel, string> = {
  get 'fig11.12'() {
    return tr('Fig. 11.12 (run in shared memory)', 'Fig. 11.12 (dãy trong bộ nhớ chia sẻ)');
  },
  get 'fig11.13'() {
    return tr('Fig. 11.13 (run in registers)', 'Fig. 11.13 (dãy trong thanh ghi)');
  },
};

function AccessCounter() {
  const [F, setF] = useState(4);
  const [kernel, setKernel] = useState<CoarseKernel>('fig11.13');
  const [who, setWho] = useState<'other' | 'first'>('other');
  const src = useLabFile(FILES[kernel]);
  const narrow = useNarrow();
  const a = { 'fig11.12': sharedAccesses('fig11.12', F, who), 'fig11.13': sharedAccesses('fig11.13', F, who) };
  const fn = !missing(src) ? functionSource(src!, 'scan_kernel') : null;
  const book = fn ? fn.text.split('\n') : [];
  const lineOf = new Map(a[kernel].lines.map((l) => [l.line, l]));
  const phaseSum = (k: CoarseKernel, ph: Phase) =>
    a[k].lines.filter((l) => l.phase === ph && l.array === 'buffer_s').reduce((n, l) => n + l.sharedRead + l.sharedWrite, 0);
  const regs = (k: CoarseKernel, ph: Phase) => a[k].lines.filter((l) => l.phase === ph).reduce((n, l) => n + l.regRead + l.regWrite, 0);
  const note = (l: (typeof a)[CoarseKernel]['lines'][number]) => {
    const parts: string[] = [];
    if (l.sharedRead) parts.push(tr(`${l.sharedRead} shared read${l.sharedRead === 1 ? '' : 's'}`, `${l.sharedRead} lần đọc shared`));
    if (l.sharedWrite) parts.push(tr(`${l.sharedWrite} shared write${l.sharedWrite === 1 ? '' : 's'}`, `${l.sharedWrite} lần ghi shared`));
    if (l.regRead || l.regWrite) parts.push(tr(`${l.regRead + l.regWrite} register`, `${l.regRead + l.regWrite} thanh ghi`));
    if (l.globalRead) parts.push(tr(`${l.globalRead} global read${l.globalRead === 1 ? '' : 's'}`, `${l.globalRead} lần đọc global`));
    if (l.globalWrite) parts.push(tr(`${l.globalWrite} global write${l.globalWrite === 1 ? '' : 's'}`, `${l.globalWrite} lần ghi global`));
    return l.times === 0 ? tr('thread 0 skips it', 'luồng 0 bỏ qua dòng này') : `×${l.times}: ${parts.join(', ')}`;
  };

  return (
    <Bench
      title={tr('What registers save', 'Thanh ghi tiết kiệm được gì')}
      task={tr("Count one thread's shared-memory accesses in each kernel at factor 4, then move the factor. Which counts grow with it, and which do not?", 'Đếm số lượt truy cập bộ nhớ chia sẻ của một luồng trong mỗi kernel ở hệ số 4, rồi đổi hệ số. Con số nào tăng theo nó, con số nào không?')}
      foot={
        <Translated
          vi={
            <>
              Các con số tính theo luồng, theo mỗi lần thực thi một dòng, nhân với số lần luồng chạy dòng đó. "Shared" ở đây nghĩa
              là chính dãy trong <code>buffer_s</code>; phần chuyển giao qua <code>threadSums</code> (mỗi luồng một lần ghi và tối đa
              một lần đọc) và các lượt truy cập bên trong <code>blockScan</code> được liệt kê riêng. Truy cập global ở cả hai là như
              nhau.
            </>
          }
        >
          <>
            Counts are per thread, per execution of each line, times how often the thread runs it. "Shared" here means the run
            itself in <code>buffer_s</code>; the hand-off through <code>threadSums</code> (one write and at most one read per
            thread) and the accesses inside <code>blockScan</code> are listed apart. Global accesses are the same in both.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          COARSE_FACTOR: <b>{F}</b>
          <input type="range" min={1} max={16} value={F} aria-label="COARSE_FACTOR" onChange={(e) => setF(+e.target.value)} />
        </label>
        <span className="seg" role="group" aria-label={tr('Kernel', 'Kernel')}>
          {(['fig11.12', 'fig11.13'] as CoarseKernel[]).map((k) => (
            <button key={k} aria-pressed={kernel === k} onClick={() => setKernel(k)}>{KERNEL_LABEL[k]}</button>
          ))}
        </span>
        <label className="field">
          <Translated vi="Luồng">Thread</Translated>
          <select value={who} aria-label={tr('thread', 'luồng')} onChange={(e) => setWho(e.target.value as 'other' | 'first')}>
            <option value="other">{tr('any thread but thread 0', 'luồng bất kỳ trừ luồng 0')}</option>
            <option value="first">{tr('thread 0', 'luồng 0')}</option>
          </select>
        </label>
      </div>
      <div style={{ overflowX: 'auto', marginBottom: '0.8rem' }}>
        <table className="data" style={{ margin: 0, fontSize: narrow ? '0.78rem' : undefined }} data-compare>
          <thead>
            <tr>
              <th><Translated vi="pha">phase</Translated></th>
              <th className="num">{narrow ? tr('11.12 shared', '11.12 shared') : tr('Fig. 11.12: shared', 'Fig. 11.12: shared')}</th>
              <th className="num">{narrow ? tr('11.13 shared', '11.13 shared') : tr('Fig. 11.13: shared', 'Fig. 11.13: shared')}</th>
              <th className="num">{narrow ? tr('11.13 regs', '11.13 thanh ghi') : tr('Fig. 11.13: registers', 'Fig. 11.13: thanh ghi')}</th>
            </tr>
          </thead>
          <tbody>
            {PHASES.filter((p) => p !== 'hand-off').map((p) => (
              <tr key={p} data-phase={p}>
                <td>{p}</td>
                <td className="num" data-k="1112">{phaseSum('fig11.12', p)}</td>
                <td className="num" data-k="1113">{phaseSum('fig11.13', p)}</td>
                <td className="num t-reg">{regs('fig11.13', p)}</td>
              </tr>
            ))}
            <tr data-phase="per-thread">
              <td><b><Translated vi="mỗi luồng, buffer_s">per thread, buffer_s</Translated></b></td>
              <td className="num" data-k="1112"><b>{a['fig11.12'].bufferS.total}</b></td>
              <td className="num" data-k="1113"><b>{a['fig11.13'].bufferS.total}</b></td>
              <td className="num t-reg">{a['fig11.13'].registers.read + a['fig11.13'].registers.write}</td>
            </tr>
            <tr data-phase="per-element">
              <td><b><Translated vi="mỗi phần tử">per element</Translated></b></td>
              <td className="num" data-k="1112"><b>{fmt(a['fig11.12'].perElement)}</b></td>
              <td className="num" data-k="1113"><b>{fmt(a['fig11.13'].perElement)}</b></td>
              <td className="num muted">–</td>
            </tr>
            <tr>
              <td className="muted"><Translated vi="chuyển giao, threadSums">hand-off, threadSums</Translated></td>
              <td className="num muted">{a['fig11.12'].threadSums.write + a['fig11.12'].threadSums.read}</td>
              <td className="num muted">{a['fig11.13'].threadSums.write + a['fig11.13'].threadSums.read}</td>
              <td className="num muted">–</td>
            </tr>
            <tr>
              <td className="muted"><Translated vi="đọc + ghi global">global reads + writes</Translated></td>
              <td className="num muted">{a['fig11.12'].global.read + a['fig11.12'].global.write}</td>
              <td className="num muted">{a['fig11.13'].global.read + a['fig11.13'].global.write}</td>
              <td className="num muted">–</td>
            </tr>
          </tbody>
        </table>
      </div>
      {fn ? (
        <pre className="mono" data-annotated={kernel}
          style={{ margin: 0, padding: '0.5rem 0', fontSize: narrow ? '0.68rem' : '0.78rem', lineHeight: 1.55, overflowX: 'auto', background: 'var(--page)', border: '1px solid var(--rule)', borderRadius: 3 }}>
          {book.map((text, i) => {
            const n = i + 1;
            const l = lineOf.get(n);
            return (
              <span key={n} data-bookline={n} style={{ display: 'block', whiteSpace: 'pre', padding: '0 0.5rem', background: l ? (l.array === 'threadSums' ? 'transparent' : 'var(--shared-wash)') : undefined }}>
                <span style={{ display: 'inline-block', width: '2.2em', color: 'var(--ink-3)', userSelect: 'none' }}>{String(n).padStart(2, '0')}</span>
                {text}
                {l ? <span className="t-shared" style={{ marginLeft: '1.2em', fontWeight: 600 }} data-note={n}>{'  // ' + note(l)}</span> : null}
              </span>
            );
          })}
        </pre>
      ) : null}
      <p className="narrate" role="status" style={{ margin: '0.7rem 0 0' }} data-says>
        <Translated
          vi={
            <>
              Ở hệ số {F}, {who === 'first' ? 'luồng 0' : 'một luồng'} truy cập dãy của nó trong bộ nhớ chia sẻ{' '}
              {a['fig11.12'].bufferS.total} lần ở Fig. 11.12 và {a['fig11.13'].bufferS.total} lần ở Fig. 11.13:{' '}
              {fmt(a['fig11.12'].perElement)} và {fmt(a['fig11.13'].perElement)} lần mỗi phần tử. Ở Fig. 11.13, mỗi phần tử được
              ghi vào bộ nhớ chia sẻ một lần khi vào, đọc một lần vào thanh ghi, ghi một lần sau phép cộng và đọc một lần khi ra,
              bất kể hệ số; {a['fig11.13'].registers.read + a['fig11.13'].registers.write} lượt truy cập được chuyển đi đã sang thanh
              ghi.
            </>
          }
        >
          At factor {F}, {who === 'first' ? 'thread 0' : 'a thread'} makes {a['fig11.12'].bufferS.total} accesses to its run in
          shared memory in Fig. 11.12 and {a['fig11.13'].bufferS.total} in Fig. 11.13: {fmt(a['fig11.12'].perElement)} and{' '}
          {fmt(a['fig11.13'].perElement)} per element. In Fig. 11.13 every element is written to shared memory once on the way in,
          read once into a register, written once after the add and read once on the way out, whatever the factor; the{' '}
          {a['fig11.13'].registers.read + a['fig11.13'].registers.write} accesses that moved went to registers.
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: bank map
type Banks = 32 | 64;
const BANK_OPTIONS: { banks: Banks; label: string }[] = [
  {
    banks: 32,
    get label() {
      return tr('32 banks of 4 bytes (NVIDIA; AMD CDNA 1 to 3)', '32 bank 4 byte (NVIDIA; AMD CDNA 1 đến 3)');
    },
  },
  {
    banks: 64,
    get label() {
      return tr('64 banks of 4 bytes (AMD CDNA 4, RDNA 2 to 4)', '64 bank 4 byte (AMD CDNA 4, RDNA 2 đến 4)');
    },
  },
];

function BankMap() {
  const [F, setF] = useState(4);
  const [banks, setBanks] = useState<Banks>(32);
  const [pad, setPad] = useState(false);
  const [c, setC] = useState(0);
  const [side, setSide] = useState<BankSide>('scan');
  const narrow = useNarrow();
  useEffect(() => setC((x) => Math.min(x, F - 1)), [F]);
  const cc = Math.min(c, F - 1);
  const m = bankMap(F, banks, pad, { c: cc, side, blockDim: 256 });
  const worst = side === 'scan' ? worstDegree(F, banks, pad) : m.degree;
  const perRow = narrow ? 16 : 32;
  const rows = banks / perRow;
  const VW = narrow ? 344 : 704;
  const bw = (VW - 4) / perRow;
  const CH = 15;
  const maxDeg = Math.max(...m.perBank, 1);
  const colH = maxDeg * (CH + 1) + 16;
  const H = rows * (colH + 8) + 4;
  // lanes per bank, in lane order
  const lanesIn = (b: number) => m.lanes.filter((l) => l.bank === b);
  const busiest = m.perBank.indexOf(m.degree);
  return (
    <Bench
      title={tr('Bank map', 'Bản đồ bank')}
      task={tr('At factor 8, find the lanes that share bank 0. Then try an odd factor, then padding, then padding with an odd factor.', 'Ở hệ số 8, tìm các lane dùng chung bank 0. Rồi thử một hệ số lẻ, rồi padding, rồi padding với hệ số lẻ.')}
      foot={
        <Translated
          vi={
            <>
              Một lượt truy cập của 32 lane: mỗi cột là một bank, mỗi ô là một word riêng biệt rơi vào bank đó, ghi nhãn lane yêu
              cầu nó. Cột có nhiều hơn một ô là một xung đột bank (bank conflict). Padding chèn một word không dùng sau mỗi{' '}
              {banks} word, nên chỉ số <i>i</i> rơi vào <i>i</i> + <i>i</i>/{banks} (chia nguyên). Apple không công bố bank nào cho
              bộ nhớ threadgroup, nên không có lựa chọn Metal.
            </>
          }
        >
          <>
            One access of 32 lanes: each column is a bank, each box a distinct word that lands in it, labelled with the lane
            that asks for it. A column with more than one box is a conflict. Padding inserts one unused word after every{' '}
            {banks} words, so index <i>i</i> lands at <i>i</i> + <i>i</i>/{banks} (integer division). Apple documents no banks
            for threadgroup memory, so Metal is not offered.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          COARSE_FACTOR: <b>{F}</b>
          <input type="range" min={1} max={16} value={F} aria-label="COARSE_FACTOR" onChange={(e) => setF(+e.target.value)} />
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          <Translated vi="Bank">Banks</Translated>
          <select value={banks} aria-label={tr('banks', 'bank')} onChange={(e) => setBanks(Number(e.target.value) as Banks)} style={{ maxWidth: '100%' }}>
            {BANK_OPTIONS.map((o) => <option key={o.banks} value={o.banks}>{o.label}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={pad} onChange={(e) => setPad(e.target.checked)} /> <Translated vi={<>chèn một word mỗi {banks}</>}>pad one word every {banks}</Translated></label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Access', 'Truy cập')}>
          <button aria-pressed={side === 'scan'} onClick={() => setSide('scan')}><Translated vi="phía scan: buffer_s[t·F + c]">scan side: buffer_s[t·F + c]</Translated></button>
          <button aria-pressed={side === 'load'} onClick={() => setSide('load')}><Translated vi="phía nạp: buffer_s[c·BLOCK_DIM + t]">load side: buffer_s[c·BLOCK_DIM + t]</Translated></button>
        </span>
        <span className="row" style={{ gap: '0.4rem' }}>
          <button onClick={() => setC(Math.max(0, cc - 1))} disabled={cc === 0}>c − 1</button>
          <button className="primary" onClick={() => setC(Math.min(F - 1, cc + 1))} disabled={cc >= F - 1}>c + 1</button>
          <span className="muted" aria-live="polite">c = {cc}</span>
        </span>
      </div>
      <svg viewBox={`0 0 ${VW} ${H}`} role="img" aria-label={tr("Which bank each lane's word lands in", 'Word của mỗi lane rơi vào bank nào')} style={{ width: '100%', maxWidth: VW, display: 'block' }} data-degree={m.degree}>
        {range(banks).map((b) => {
          const r = Math.floor(b / perRow);
          const x = 2 + (b % perRow) * bw;
          const y0 = r * (colH + 8) + 12;
          const ls = lanesIn(b);
          const conflict = ls.length > 1;
          return (
            <g key={b} data-bank={b} data-count={m.perBank[b]}>
              {b % 4 === 0 ? <text x={x + bw / 2} y={y0 - 3} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{b}</text> : null}
              <rect x={x + 0.5} y={y0} width={bw - 1} height={colH - 12} fill="none" stroke="var(--rule-soft)" />
              {ls.map((l, k) => (
                <g key={l.lane}>
                  <rect x={x + 1.5} y={y0 + 1 + k * (CH + 1)} width={bw - 3} height={CH} rx="1.5"
                    fill={conflict ? 'var(--fail-wash)' : 'var(--shared-wash)'} stroke={conflict ? 'var(--fail)' : 'var(--shared)'} />
                  <text x={x + bw / 2} y={y0 + 1 + k * (CH + 1) + 11} fontSize={bw < 14 ? 7 : 8.5} textAnchor="middle" style={{ fill: 'var(--ink)' }}>{l.lane}</text>
                </g>
              ))}
            </g>
          );
        })}
      </svg>
      <p className="narrate" role="status" style={{ margin: '0.6rem 0 0' }} data-bank-says data-worst={worst}>
        {m.degree === 1
          ? tr(
              `Every lane's word is in a bank of its own: no conflict${side === 'scan' && worst > 1 ? `, in this iteration. Another iteration c does conflict (${worst}-way): step through them.` : '.'}`,
              `Word của mỗi lane nằm trong bank riêng: không xung đột${side === 'scan' && worst > 1 ? `, ở lần lặp này. Một lần lặp c khác có xung đột (${worst} chiều): hãy đi qua từng lần.` : '.'}`,
            )
          : tr(
              `${m.degree}-way conflict: bank ${busiest} serves lanes ${lanesIn(busiest).map((l) => l.lane).join(', ')}, ${m.degree} different words.`,
              `Xung đột ${m.degree} chiều: bank ${busiest} phục vụ các lane ${lanesIn(busiest).map((l) => l.lane).join(', ')}, ${m.degree} word khác nhau.`,
            ) +
            (side === 'scan' && !pad && banks === 32 ? ` gcd(${F}, 32) = ${conflictDegree(F, 32)}.` : '') +
            (pad ? tr(' Padding moved the conflict rather than removing it: this factor is not a power of two.', ' Padding dời xung đột đi chứ không xóa nó: hệ số này không phải lũy thừa của hai.') : '')}
        {banks === 64 ? tr(' AMD serves at most 16 lanes of a wavefront per cycle, so on AMD this count is not a cycle count.', ' AMD phục vụ tối đa 16 lane của một wavefront mỗi chu kỳ, nên trên AMD con số này không phải số chu kỳ.') : tr(' NVIDIA serialises the conflicting accesses.', ' NVIDIA tuần tự hóa các truy cập xung đột.')}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- code
function RegTiledListing() {
  const printed = useLabFile(FILES['fig11.13']);
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs.
  const { reports } = useReports();
  const labStatus = { cu: solutionTabStatus(reports, 'ch11/lab2_coarsened', 'kernel.cu'), metal: solutionTabStatus(reports, 'ch11/lab2_coarsened', 'kernel.metal') };
  if (missing(printed)) return null;
  const k = functionSource(printed!, 'scan_kernel');
  if (!k) return null;
  const labCu = !missing(cu) ? functionSource(cu!, 'RegisterTiledScanKernel') : null;
  const labMetal = !missing(metal) ? functionSource(metal!, 'register_tiled_scan') : null;
  const tabs = [
    { label: tr('Fig. 11.13 as printed', 'Fig. 11.13 như bản in'), code: k.text, tiers: TIERS, status: STATUS.printed, highlight: [12, 13, 14, 15, 16, 19, 26, 27, 28, 29] },
    ...(labCu ? [{ label: tr('lab 11.2 (CUDA and HIP)', 'lab 11.2 (CUDA và HIP)'), code: labCu.text, tiers: TIERS, status: labStatus.cu }] : []),
    ...(labMetal ? [{ label: tr('lab 11.2 (Metal)', 'lab 11.2 (Metal)'), code: labMetal.text, tiers: TIERS, status: labStatus.metal }] : []),
  ];
  return (
    <Code
      tabs={tabs}
      caption={
        <>
          <Translated vi={<>Fig. 11.13 với số dòng của sách (<code>labs/{FILES['fig11.13']}</code>); được đánh dấu là các dòng khác với Fig. 11.12. Giống Fig. 11.12, nó không hề đọc <code>N</code>.</>}>
            Fig. 11.13 with the book's line numbers (<code>labs/{FILES['fig11.13']}</code>); marked, the lines that differ from
            Fig. 11.12. Like Fig. 11.12 it never reads <code>N</code>.
          </Translated>{' '}
          {labCu
            ? <Translated vi={<>Các tab lab là lời giải tham chiếu (solution) của lab 11.2 (<code>labs/{LAB}/solution/</code>). Kernel Metal không có pragma unroll: hệ số của nó là một macro được đặt khi build pipeline, nên mọi vòng lặp có số lần lặp hằng.</>}><>The lab tabs are lab 11.2's reference solution (<code>labs/{LAB}/solution/</code>). The Metal kernel has no unroll pragma: its factor is a macro set when the pipeline is built, so every loop has a constant trip count.</></Translated>
            : <Translated vi={<>Lời giải tham chiếu của lab 11.2 sẽ hiện ở đây khi nó có trên máy này.</>}><>Lab 11.2's reference solution will appear here once it is on this machine.</></Translated>}
        </>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
export const questionsS117: Record<string, Question> = {
  'ch11/11.7/q-shared': {
    kind: 'numeric',
    id: 'ch11/11.7/q-shared',
    prompt: (
      <Translated
        vi={
          <>
            Tính trên mỗi phần tử của dãy một luồng, Fig. 11.13 truy cập <code>buffer_s</code> bao nhiêu lần (cả đọc lẫn ghi)?
          </>
        }
      >
        <>
          Per element of a thread's run, how many accesses to <code>buffer_s</code> does Fig. 11.13 make (reads and writes
          together)?
        </>
      </Translated>
    ),
    answer: 4,
    explain: (
      <Translated
        vi={
          <p>
            Một lần ghi khi nạp hợp nhất (dòng 06), một lần đọc vào <code>buffer_r</code> (dòng 13 và 16), một lần ghi lại sau phép
            cộng (dòng 29), một lần đọc khi ghi ra hợp nhất (dòng 35). Không phụ thuộc vào hệ số. Fig. 11.12 làm 7 − 2/F lần mỗi
            phần tử với mọi luồng trừ luồng 0: 6.5 ở hệ số 4, vì scan của nó đọc hai phần tử và ghi một mỗi bước, và phép cộng của
            nó đọc và ghi lại từng phần tử.
          </p>
        }
      >
        <p>
          One write on the coalesced load (line 06), one read into <code>buffer_r</code> (lines 13 and 16), one write back after
          the add (line 29), one read on the coalesced store (line 35). It does not depend on the factor. Fig. 11.12 makes 7 −
          2/F per element for any thread but thread 0: 6.5 at factor 4, because its scan reads two elements and writes one per
          step and its add reads and writes each element again.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.7, pp. 271–272 (Figs. 11.12, 11.13); model: src/sim/coarsescan.ts, tied to the listings by its test',
    vi: { source: 'ch. 11 §11.7, pp. 271–272 (Figs. 11.12, 11.13); mô hình: src/sim/coarsescan.ts, gắn với mã nguồn qua test của nó' },
  },
  'ch11/11.7/q-bank': {
    kind: 'numeric',
    id: 'ch11/11.7/q-bank',
    prompt: (
      <Translated
        vi={
          <>
            Hệ số 8, 32 bank 4 byte, không padding. Một warp 32 lane truy cập <code>buffer_s[t*8 + c]</code>. Bao nhiêu lane rơi
            vào cùng một bank?
          </>
        }
      >
        <>
          Factor 8, 32 banks of 4 bytes, no padding. One warp of 32 lanes accesses <code>buffer_s[t*8 + c]</code>. How many
          lanes land in the same bank?
        </>
      </Translated>
    ),
    answer: 8,
    unit: 'lanes',
    vi: { unit: 'lane', source: 'bài tập 4, p. 286; infra-facts A10; mô hình: src/sim/coarsescan.ts' },
    explain: (
      <Translated
        vi={
          <p>
            Word của lane <i>t</i> là 8<i>t</i> + <i>c</i>, nên bank của nó là (8<i>t</i> + <i>c</i>) mod 32: lane <i>t</i> và{' '}
            <i>t</i> + 4 cách nhau 32 word và dùng chung một bank. 32 lane trải trên 4 bank, mỗi bank 8, toàn word khác nhau: xung
            đột 8 chiều, được phục vụ lần lượt. Padding một word mỗi 32 dời lane <i>t</i> đi <i>t</i>/4 (chia nguyên) và đặt mỗi
            lane vào bank riêng.
          </p>
        }
      >
        <p>
          Lane <i>t</i>'s word is 8<i>t</i> + <i>c</i>, so its bank is (8<i>t</i> + <i>c</i>) mod 32: lanes <i>t</i> and{' '}
          <i>t</i> + 4 differ by 32 words and share a bank. 32 lanes spread over 4 banks, 8 per bank, all different words: an
          8-way conflict, served one after another. Padding one word every 32 shifts lane <i>t</i> by <i>t</i>/4 (integer
          division) and puts every lane in its own bank.
        </p>
      </Translated>
    ),
    source: 'exercise 4, p. 286; infra-facts A10; model: src/sim/coarsescan.ts',
  },
  'ch11/11.7/q-odd': {
    kind: 'choice',
    id: 'ch11/11.7/q-odd',
    prompt: (
      <Translated vi={<>Không padding, những hệ số gộp việc nào làm truy cập phía scan <code>buffer_s[t*F + c]</code> không xung đột trên 32 bank?</>}>
        <>Without padding, which coarsening factors make the scan-side access <code>buffer_s[t*F + c]</code> conflict-free on 32 banks?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Các hệ số lẻ.">The odd ones.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Các lũy thừa của hai.">The powers of two.</Translated>,
        why: <Translated vi="Đó là trường hợp tệ nhất: bậc xung đột là gcd(F, 32), bằng chính F với các lũy thừa của hai đến 32.">Those are the worst: the degree is gcd(F, 32), which is F itself for powers of two up to 32.</Translated>,
      },
      {
        label: <Translated vi="Chỉ F = 1.">Only F = 1.</Translated>,
        why: <Translated vi="Thử F = 3: lane t và t' chỉ dùng chung bank nếu 3(t − t') là bội số của 32.">{'Try F = 3: lanes t and t\' share a bank only if 3(t − t\') is a multiple of 32.'}</Translated>,
      },
      {
        label: <Translated vi="Không hệ số nào: truy cập có bước nhảy luôn xung đột.">None: a strided access always conflicts.</Translated>,
        why: <Translated vi="Một bước nhảy không có ước chung với 32 sẽ đi qua cả 32 bank.">A stride that shares no factor with 32 visits all 32 banks.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lane <i>t</i> và <i>t'</i> va nhau khi F·(<i>t</i> − <i>t'</i>) là bội số của 32. Với F lẻ, điều đó cần{' '}
            <i>t</i> − <i>t'</i> là bội số của 32, không thể xảy ra trong một warp. Tổng quát, bậc xung đột là gcd(F, 32).
          </p>
        }
      >
        <p>
          Lanes <i>t</i> and <i>t'</i> collide when F·(<i>t</i> − <i>t'</i>) is a multiple of 32. For odd F that needs{' '}
          <i>t</i> − <i>t'</i> to be a multiple of 32, impossible inside one warp. In general the degree is gcd(F, 32).
        </p>
      </Translated>
    ),
    source: 'our arithmetic from infra-facts A10; model: src/sim/coarsescan.ts',
    vi: { source: 'phép tính của chúng tôi từ infra-facts A10; mô hình: src/sim/coarsescan.ts' },
  },
  'ch11/11.7/q-pad-odd': {
    kind: 'choice',
    id: 'ch11/11.7/q-pad-odd',
    prompt: (
      <Translated vi={<>Bạn padding <code>buffer_s</code> một word mỗi 32 để sửa hệ số 8, rồi đổi hệ số thành 3 và giữ padding. Điều gì xảy ra trên 32 bank?</>}>
        <>You pad <code>buffer_s</code> by one word every 32 to fix factor 8, then change the factor to 3 and keep the padding. What happens on 32 banks?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Vẫn không xung đột: một word padding trên 32 không thể tạo ra xung đột.">Still conflict-free: one word of padding in 32 cannot create a conflict.</Translated>,
        why: <Translated vi="Hãy đi qua từng c ở hệ số 3 khi bật padding.">Step through c at factor 3 with padding on.</Translated>,
      },
      {
        label: <Translated vi="Một số lần lặp phát sinh xung đột 2 chiều mà trước đó không có.">Some iterations gain a 2-way conflict they did not have.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Xung đột 8 chiều, như ở hệ số 8 trước khi padding.">An 8-way conflict, as at factor 8 before the padding.</Translated>,
        why: <Translated vi="Hệ số 3 hoàn toàn không xung đột khi không padding.">Factor 3 had no conflict at all without padding.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Ở hệ số 3, 32 lane trải trên 96 word. Padding dời các lane qua word 32 đi một và các lane qua word 64 đi hai, và một số
            giờ rơi vào bank mà lane khác đã dùng. Padding sửa được các hệ số là lũy thừa của hai; với mọi hệ số khác đến 16, nó gây
            ra xung đột 2 chiều ở đâu đó. Hãy chọn padding theo hệ số bạn dùng, hoặc đo cả hai.
          </p>
        }
      >
        <p>
          At factor 3 the 32 lanes span 96 words. Padding shifts the lanes past word 32 by one and those past word 64 by two,
          and some of them now land on a bank another lane already uses. Padding fixes power-of-two factors; for any other
          factor up to 16 it introduces a 2-way conflict somewhere. Choose the padding for the factor you use, or measure both.
        </p>
      </Translated>
    ),
    source: 'our arithmetic; model: src/sim/coarsescan.ts (its test lists every factor up to 16)',
    vi: { source: 'phép tính của chúng tôi; mô hình: src/sim/coarsescan.ts (test của nó liệt kê mọi hệ số đến 16)' },
  },
  'ch11/11.7/q-why-shared': {
    kind: 'choice',
    id: 'ch11/11.7/q-why-shared',
    prompt: (
      <Translated vi={<>Ở Fig. 11.13 không có gì trong bộ nhớ chia sẻ được tái sử dụng: mỗi giá trị được ghi một lần và đọc một lần. Vậy sao vẫn phải đi qua bộ nhớ chia sẻ?</>}>
        <>In Fig. 11.13 nothing in shared memory is reused: each value is written once and read once. Why does it still go through shared memory at all?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Để giữ cả hai lần chuyển global được hợp nhất trong khi mỗi luồng scan dãy của chính nó.">To keep both global transfers coalesced while each thread scans its own run.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="Vì blockScan cần cả đoạn trong bộ nhớ chia sẻ để kết hợp các dãy.">Because blockScan needs the whole segment in shared memory to combine the runs.</Translated>,
        why: <Translated vi="blockScan nhận mỗi luồng một giá trị, trong thanh ghi.">blockScan takes one value per thread, in a register.</Translated>,
      },
      {
        label: <Translated vi="Vì một mảng cục bộ không thể được nạp thẳng từ bộ nhớ toàn cục.">Because a local array cannot be filled straight from global memory.</Translated>,
        why: (
          <Translated vi="Có thể; nạp mỗi dãy thẳng vào buffer_r vẫn chạy, nhưng các địa chỉ của warp sẽ cách nhau F.">
            {'It can; loading each run straight into buffer_r would work, but the warp\'s addresses would be F apart.'}
          </Translated>
        ),
      },
      {
        label: <Translated vi="Đó là phần thừa từ Fig. 11.12 và có thể bỏ mà không tốn gì.">It is a leftover from Fig. 11.12 and could be removed at no cost.</Translated>,
        why: <Translated vi="Bỏ nó sẽ mang lại mẫu truy cập không hợp nhất của p. 269.">Removing it brings back the uncoalesced pattern of p. 269.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bộ nhớ chia sẻ ở đây không phải một ô (tile): không có gì trong đó được tái sử dụng. Nó là nơi dữ liệu chuyển từ
            chunk-major (thứ tự mà warp đọc bộ nhớ toàn cục liền nhau) sang thread-major (thứ tự mà một luồng scan dãy của mình),
            tức kỹ thuật corner turning của chương 6.
          </p>
        }
      >
        <p>
          Shared memory here is not a tile: nothing in it is reused. It is where the data turns from chunk-major (the order in
          which a warp reads global memory contiguously) to thread-major (the order in which a thread scans its run), the corner
          turning of chapter 6.
        </p>
      </Translated>
    ),
    source: 'ch. 11 §11.7, p. 272',
  },
  'ch11/11.7/q-metal-unroll': {
    kind: 'choice',
    id: 'ch11/11.7/q-metal-unroll',
    prompt: (
      <Translated vi={<>Thứ tương đương <code>#pragma unroll</code> của Fig. 11.13 được tài liệu Metal ghi nhận là gì?</>}>
        <>What is the documented Metal counterpart of <code>#pragma unroll</code> in Fig. 11.13?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Không có trong đặc tả MSL: giữ số lần lặp và chỉ số là hằng, và đo.">None in the MSL spec: keep trip counts and indices constant, and measure.</Translated>,
        correct: true,
      },
      {
        label: <Translated vi="#pragma unroll, được đặc tả MSL ghi nhận với cùng cách viết như CUDA.">#pragma unroll, which the MSL specification documents with the same spelling as CUDA.</Translated>,
        why: <Translated vi="Trình biên dịch cục bộ chấp nhận nó, nhưng đặc tả MSL không ghi nhận pragma vòng lặp nào.">The local compiler accepts it, but the MSL specification documents no loop pragma.</Translated>,
      },
      {
        label: <Translated vi="[[unroll]] trên vòng lặp.">[[unroll]] on the loop.</Translated>,
        why: <Translated vi="Không có thuộc tính nào như vậy trong đặc tả.">No such attribute appears in the specification.</Translated>,
      },
      {
        label: <Translated vi="Không cần gì: Metal đưa mọi mảng cục bộ vào thanh ghi.">Nothing is needed: Metal promotes every local array to registers.</Translated>,
        why: <Translated vi="Apple cảnh báo rằng đánh chỉ số động vào mảng phạm vi luồng có thể đẩy chúng ra bộ nhớ thiết bị.">Apple warns that dynamic indexing into thread-scoped arrays can push them to device memory.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Đặc tả MSL không có "unroll" và không có pragma vòng lặp. MLX và llama.cpp dùng{' '}
            <code>_Pragma("clang loop unroll(full)")</code> của Clang, được trình biên dịch chấp nhận và ghi lại, nhưng Apple không
            ghi nhận rằng back end GPU tuân theo nó. Điều Apple có ghi nhận là lý do cần quan tâm: mảng phạm vi luồng bị đánh chỉ
            số động có thể nằm ở bộ nhớ thiết bị. Kernel Metal của lab 11.2 biến hệ số thành một macro, đặt khi build pipeline, nên
            cả hai vòng lặp có số lần lặp hằng và trình biên dịch tự quyết.
          </p>
        }
      >
        <p>
          The MSL specification has no "unroll" and no loop pragma. MLX and llama.cpp use Clang's{' '}
          <code>_Pragma("clang loop unroll(full)")</code>, which the compiler accepts and records, but Apple does not document
          that the GPU back end honours it. What Apple does document is the reason to care: thread-scoped arrays indexed
          dynamically can end up in device memory. Lab 11.2's Metal kernel makes the factor a macro, set when the pipeline is
          built, so both loops have constant trip counts and the compiler decides.
        </p>
      </Translated>
    ),
    source: 'infra-facts A7, B7, table; ch06-facts C4',
    vi: { source: 'infra-facts A7, B7, bảng; ch06-facts C4' },
  },
};

// ---------------------------------------------------------------- the section
export function S117() {
  return (
    <>
      <Translated
        vi={
          <p>
            Hãy nhìn lại Fig. 11.12 từ góc nhìn của một luồng. Dãy của nó là riêng tư: không luồng nào khác đọc hay ghi nó giữa lần
            nạp và lần ghi ra. Thế nhưng dãy nằm trong bộ nhớ chia sẻ suốt thời gian đó, và luồng đọc, ghi nó ở đó trong lúc scan,
            lại lần nữa để lấy tổng, và lần nữa trong phép cộng. Mỗi lượt truy cập ấy đều trả độ trễ của bộ nhớ chia sẻ cho dữ liệu
            mà không ai khác cần.
          </p>
        }
      >
        <p>
          Look again at Fig. 11.12 from a thread's point of view. Its run is private: no other thread reads or writes it
          between the load and the store. Yet the run lives in shared memory the whole time, and the thread reads and writes
          it there in the scan, again for its total, and again in the add. Every one of those accesses pays shared memory's
          latency for data nobody else needs.
        </p>
      </Translated>
      <Source section="11.7" pages="270–271" />

      <h2><Translated vi="Giữ dãy trong thanh ghi">Keep the run in registers</Translated></h2>
      <Translated
        vi={
          <p>
            Fig. 11.13 chép dãy vào một mảng cục bộ, <code>buffer_r</code>, ngay lần đầu đọc nó, và làm phần việc còn lại ở đó.
            Dòng 13 đọc phần tử đầu; dòng 16 đọc từng phần tử kế tiếp từ bộ nhớ chia sẻ và cộng tổng chạy đang giữ trong{' '}
            <code>buffer_r</code>. Tổng cho <code>blockScan</code> lấy từ một thanh ghi (dòng 19). Phép cộng và lần ghi lại là một
            câu lệnh (dòng 29): <code>buffer_r[c] + prevPartialSum</code> đi thẳng vào <code>buffer_s</code>. Luồng 0 giờ cộng 0
            thay vì bỏ qua vòng lặp (dòng 26), giúp mọi luồng đi cùng một nhánh. Đây là phân ô thanh ghi: giữ dữ liệu mà một luồng
            tái sử dụng trong bộ nhớ nhanh nhất của nó.
          </p>
        }
      >
        <p>
          Fig. 11.13 copies the run into a local array, <code>buffer_r</code>, the first time it reads it, and does the rest of
          the work there. Line 13 reads the first element; line 16 reads each next element from shared memory and adds the
          running sum held in <code>buffer_r</code>. The total for <code>blockScan</code> comes from a register (line 19). The
          add and the write back are one statement (line 29): <code>buffer_r[c] + prevPartialSum</code> goes straight to{' '}
          <code>buffer_s</code>. Thread 0 now adds 0 instead of skipping the loop (line 26), which keeps every thread on the same
          path. This is register tiling: keeping data a thread reuses in its fastest memory.
        </p>
      </Translated>
      <Source pages="271" figs="Fig. 11.13" />
      <RegTiledListing />
      <Translated
        vi={
          <p>
            Một mảng cục bộ chỉ nằm trong thanh ghi nếu mọi chỉ số vào nó đều biết lúc biên dịch. Cả hai vòng lặp trên{' '}
            <code>buffer_r</code> đều có <code>#pragma unroll</code> (dòng 14 và 27): khi được unroll, <code>c</code> là hằng
            trong mỗi bản sao của thân vòng lặp, như ở chương 6 (<a href={href('ch06', '6.6')}>mục 6.6</a>). Sách nói thêm rằng
            trình biên dịch nhiều khả năng sẽ tự unroll các vòng lặp này; pragma chỉ nêu rõ ý định.
          </p>
        }
      >
        <p>
          A local array lives in registers only if every index into it is known at compile time. Both loops over{' '}
          <code>buffer_r</code> have <code>#pragma unroll</code> (lines 14 and 27): unrolled, <code>c</code> is a constant in
          each copy of the body, as in chapter 6 (<a href={href('ch06', '6.6')}>section 6.6</a>). The book adds that the
          compiler would most likely unroll these loops anyway; the pragma states the intent.
        </p>
      </Translated>
      <NoteGroup>
        <Source pages="271–272" />
        <Today checked={tr('2026-11-20 (infra-facts A7, B7 and table; ch06-facts C4)', '2026-11-20 (infra-facts A7, B7 và bảng; ch06-facts C4)')}>
          <Translated
            vi={
              <p>
                NVIDIA: "The compiler unrolls small loops with a known trip count by default" (mặc định trình biên dịch unroll các vòng
                lặp nhỏ có số lần lặp biết trước). HIP: cùng cách viết. MSL không ghi nhận pragma unroll hay điều khiển inline nào; hãy
                giữ số lần lặp là hằng và đo.
              </p>
            }
          >
            <p>
              NVIDIA: "The compiler unrolls small loops with a known trip count by default". HIP: same spelling. MSL documents
              no unroll pragma and no inline control; keep the trip count constant and measure.
            </p>
          </Translated>
        </Today>
      </NoteGroup>
      <Translated
        vi={
          <p>
            Hãy đếm những gì đã thay đổi. Trước khi kéo thanh trượt: những pha nào của Fig. 11.12 chạm vào bộ nhớ chia sẻ nhiều hơn
            một lần mỗi phần tử, và những dòng nào của Fig. 11.13 thay thế chúng?
          </p>
        }
      >
        <p>
          Count what changed. Before you move the slider: which phases of Fig. 11.12 touch shared memory more than once per
          element, and which lines of Fig. 11.13 replace them?
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Bộ đếm đếm các lượt truy cập trong mã nguồn, từng câu lệnh, nhân với số lần lặp. Nó không biết trình biên dịch giữ gì
              trong thanh ghi hay gộp gì, và không con số nào ở đây là thời gian.
            </p>
          }
        >
          <p>
            The counter counts accesses in the source, statement by statement, times the loop's trip count. It does not know
            what the compiler keeps in registers or merges, and nothing here is a time.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questionsS117['ch11/11.7/q-shared']} label={tr('Work it out first', 'Tự tính trước')} />
      <AccessCounter />
      <Ask q={questionsS117['ch11/11.7/q-metal-unroll']} />

      <h2><Translated vi="Vì sao bộ nhớ chia sẻ vẫn ở lại">Why shared memory stays</Translated></h2>
      <Translated
        vi={
          <p>
            Nếu dãy nằm trong thanh ghi, sao không nạp nó thẳng từ bộ nhớ toàn cục vào <code>buffer_r</code>? Vì khi đó luồng{' '}
            <i>t</i> sẽ đọc các phần tử <i>t</i>·F + <i>c</i>, cách nhau F trong warp: mẫu không hợp nhất của{' '}
            <a href={href('ch11', '11.6')}>mục 11.6</a>. Bộ nhớ chia sẻ ở lại chỉ để chuyển thứ tự chunk-major, trong đó warp đọc
            bộ nhớ toàn cục liền nhau, sang thứ tự thread-major, trong đó một luồng scan dãy của mình. Dữ liệu của nó không được
            tái sử dụng, nên nó không còn là một ô. Sách gọi đây là corner turning của chương 6 (<a href={href('ch06', '6.1')}>mục
            6.1</a>).
          </p>
        }
      >
        <p>
          If the run lives in registers, why not load it from global memory straight into <code>buffer_r</code>? Because then
          thread <i>t</i> would read elements <i>t</i>·F + <i>c</i>, F apart across the warp: the uncoalesced pattern of{' '}
          <a href={href('ch11', '11.6')}>section 11.6</a>. Shared memory stays only to turn chunk-major order, in which the warp
          reads global memory contiguously, into thread-major order, in which a thread scans its run. Its data is not reused, so
          it is no longer a tile. The book calls this the corner turning of chapter 6 (<a href={href('ch06', '6.1')}>section
          6.1</a>).
        </p>
      </Translated>
      <Source pages="272" />
      <Ask q={questionsS117['ch11/11.7/q-why-shared']} />

      <h2><Translated vi="Điều sách để lại cho bài tập 4">What the book leaves to exercise 4</Translated></h2>
      <Translated
        vi={
          <p>
            Còn hai tối ưu hóa, và sách để cả hai cho bài tập 4. Thứ nhất, các lần chuyển giữa bộ nhớ toàn cục và bộ nhớ chia sẻ
            (dòng 05 đến 08 và 33 đến 36) chuyển mỗi luồng một số float mỗi lần lặp; chúng có thể dùng lệnh nạp và ghi vector.
            Thứ hai, các truy cập có bước nhảy vào <code>buffer_s</code> (sách nêu dòng 13 và 29; dòng 16 cũng đọc theo cách đó)
            có thể gây xung đột bank ở bộ nhớ chia sẻ, mà padding sẽ loại bỏ, như ở chương 6.
          </p>
        }
      >
        <p>
          Two optimisations remain, and the book leaves both as exercise 4. First, the transfers between global and shared memory
          (lines 05 to 08 and 33 to 36) move one float per thread per iteration; they could use vector loads and stores. Second,
          the strided accesses to <code>buffer_s</code> (the book names lines 13 and 29; line 16 reads the same way) can cause
          shared-memory bank conflicts, which padding removes, as in chapter 6.
        </p>
      </Translated>
      <Source pages="272, 286" />
      <Translated
        vi={
          <p>
            Về nạp vector, hãy nhớ lời cảnh báo của chương 6: ép kiểu <code>float*</code> sang <code>float4*</code>, kỹ thuật của
            sách ở đó, không được nhà cung cấp nào ghi nhận. Trên Metal, cách được ghi nhận là khai báo buffer{' '}
            <code>device float4*</code>.
          </p>
        }
      >
        <p>
          On vector loads, keep chapter 6's caution: casting a <code>float*</code> to a <code>float4*</code>, the book's
          technique there, is documented by no vendor. On Metal the documented way is to declare the buffer{' '}
          <code>device float4*</code>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (infra-facts A6, B6, C7; ch06-facts C3)', '2026-11-20 (infra-facts A6, B6 và C7; ch06-facts C3)')}>
        <Translated
          vi={
            <p>
              <code>float4</code> dài 16 byte với căn lề 16 byte trên cả ba. Phép ép kiểu con trỏ không được ghi nhận ở đâu cả, dù
              MLX và llama.cpp dùng nó trong MSL.
            </p>
          }
        >
          <p>
            <code>float4</code> is 16 bytes with 16-byte alignment on all three. The pointer cast is undocumented everywhere,
            though MLX and llama.cpp use it in MSL.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Về bank: một warp chạm vào <code>buffer_s[t*F + c]</code> đặt word của lane <i>t</i> vào bank (<i>t</i>·F + <i>c</i>)
            mod (số bank). NVIDIA ghi nhận 32 bank 4 byte trên mọi thiết bị; AMD 32 (CDNA 1 đến 3) hoặc 64 (CDNA 4, RDNA 2 đến 4);
            Apple không ghi nhận gì về bank trong bộ nhớ threadgroup. Trước khi mở bản đồ: ở hệ số 8, bao nhiêu lane dùng chung một
            bank?
          </p>
        }
      >
        <p>
          On banks: a warp touching <code>buffer_s[t*F + c]</code> puts lane <i>t</i>'s word in bank (<i>t</i>·F + <i>c</i>) mod
          (bank count). NVIDIA documents 32 banks of 4 bytes on every device; AMD 32 (CDNA 1 to 3) or 64 (CDNA 4, RDNA 2 to 4);
          Apple documents nothing about banks in threadgroup memory. Before you open the map: at factor 8, how many lanes share a
          bank?
        </p>
      </Translated>
      <NoteGroup>
      <Beyond>
        <Translated
          vi={
            <p>
              Số bank lấy từ infra-facts A10, B10 và C13. Chi phí của xung đột khác nhau: NVIDIA tuần tự hóa nó; AMD đằng nào cũng
              chỉ phục vụ tối đa 16 lane của một wavefront mỗi chu kỳ, nên con số tính cho NVIDIA không phải của AMD. Trên Metal,
              hãy đo.
            </p>
          }
        >
          <p>
            Bank counts from infra-facts A10, B10 and C13. The cost of a conflict differs: NVIDIA serialises it; AMD serves at
            most 16 lanes of a wavefront per cycle in any case, so a count computed for NVIDIA is not AMD's. On Metal, measure.
          </p>
        </Translated>
      </Beyond>
      <Simplification>
        <Translated
          vi={
            <p>
              Một lượt truy cập của 32 lane, word 4 byte, khối 256 luồng ở phía nạp. Số đếm của một bank là số word khác nhau rơi vào
              nó; trong lượt truy cập này không có word nào trùng nhau.
            </p>
          }
        >
          <p>
            One access by 32 lanes, 4-byte words, a block of 256 threads on the load side. A bank's count is the number of
            different words that land in it; identical words do not occur in this access.
          </p>
        </Translated>
      </Simplification>
      </NoteGroup>
      <Ask q={questionsS117['ch11/11.7/q-bank']} label={tr('Work it out first', 'Tự tính trước')} />
      <BankMap />
      <Ask q={questionsS117['ch11/11.7/q-odd']} />
      <Ask q={questionsS117['ch11/11.7/q-pad-odd']} />

      <h2><Translated vi="Nơi bạn gặp điều này: scan của MLX">Where you meet this: MLX's scan</Translated></h2>
      <Translated
        vi={
          <p>
            <code>contiguous_scan</code> của MLX trên Metal là phân ô thanh ghi trong một kernel đang được phát hành. Một
            threadgroup scan một hàng; mỗi luồng đọc <code>N_READS</code> giá trị liền nhau (4 với kiểu 4 byte) và scan chúng trong
            thanh ghi, các SIMD group scan tổng của các luồng bằng <code>simd_prefix_exclusive_sum</code>, và tổng của các SIMD
            group đi qua bộ nhớ threadgroup để được scan thêm lần nữa, tức thiết kế của mục này và của 11.4 gộp lại. Một vòng lặp
            trên hàng mang tổng chạy sang khúc kế tiếp, nên một hàng dài bao nhiêu cũng không cần giao tiếp giữa các threadgroup.
          </p>
        }
      >
        <p>
          MLX's Metal <code>contiguous_scan</code> is register tiling in a shipping kernel. One threadgroup scans a row; each
          thread reads <code>N_READS</code> contiguous values (4 for 4-byte types) and scans them in registers, the SIMD-groups
          scan the thread totals with <code>simd_prefix_exclusive_sum</code>, and the SIMD-group totals go through threadgroup
          memory to be scanned once more, the design of this section and of 11.4 together. A loop over the row carries the
          running total into the next chunk, so a row of any length needs no communication between threadgroups.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              <code>mlx/backend/metal/kernels/scan.h</code> và <code>scan.cpp</code> của MLX, đọc ngày 2026-11-20 (ch11-facts D3).
            </p>
          }
        >
          <p>
            MLX's <code>mlx/backend/metal/kernels/scan.h</code> and <code>scan.cpp</code> as read on 2026-11-20 (ch11-facts D3).
          </p>
        </Translated>
      </Beyond>
    </>
  );
}
