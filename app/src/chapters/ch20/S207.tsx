import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { fmt } from '../../lib/reports';
import { href } from '../../lib/router';
import { aiGQA, aiMLA, gqaTerms, kvHeadOf } from '../../sim/kvmath';
import { peakLive, simulate, type PagedConfig } from '../../sim/paged';
import { KvExplorer } from './KvExplorer';
import { ContiguousPool, PagedBlocksScene, PagedPool } from './PagedBlocksScene';

// ---------------------------------------------------------------- who shares a K/V head

type HeadVariant = 'MHA' | 'GQA' | 'MQA' | 'MLA';
const D_HEAD = 128;
const N_AI = 4096;
const divisorsBetween = (n: number) => Array.from({ length: n }, (_, i) => i + 1).filter((k) => n % k === 0 && k > 1 && k < n);

function HeadsBench() {
  const [hq, setHq] = useState(6);
  const [variant, setVariant] = useState<HeadVariant>('GQA');
  const [gqaPick, setGqaPick] = useState(3);
  const [sel, setSel] = useState<number | null>(4);
  const groups = divisorsBetween(hq);
  const gqaHkv = groups.includes(gqaPick) ? gqaPick : groups[Math.floor((groups.length - 1) / 2)] ?? null;
  const v: HeadVariant = variant === 'GQA' && gqaHkv == null ? 'MHA' : variant;
  const hkv = v === 'MHA' ? hq : v === 'MQA' || v === 'MLA' ? 1 : gqaHkv!;
  const { gq, G } = gqaTerms(hq, hkv);
  const values = v === 'MLA' ? 4 * D_HEAD + D_HEAD / 2 : 2 * hkv * D_HEAD;
  const ai = v === 'MLA' ? aiMLA(N_AI, hq) : aiGQA(N_AI, hq, gq);
  const selected = sel != null && sel < hq ? sel : null;
  const target = selected != null ? kvHeadOf(selected, hq, hkv) : null;

  const W = 720;
  const H = 250;
  const qw = Math.min(56, (W - 40) / hq - 4);
  const qx = (h: number) => 20 + h * ((W - 40) / hq) + ((W - 40) / hq - qw) / 2;
  const kvSlot = (W - 40) / hkv;
  const kw = Math.min(70, kvSlot - 8);
  const kx = (k: number) => 20 + k * kvSlot + (kvSlot - kw) / 2;
  const yQ = 46;
  const yK = 160;
  return (
    <Bench
      title={tr('Who shares a K/V head', 'Ai dùng chung một head K/V')}
      task={
        <Translated vi="Nhấn vào một head truy vấn để xem nó đọc head K/V nào. Rồi đếm những gì được cache.">
          Click a query head to see which K/V head it reads. Then count what is cached.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Các ô có viền là những gì được cache trong lúc suy luận (các ô đỏ của Fig. 20.19). Các head rộng {D_HEAD} và
              cường độ tính toán tính ở N = {N_AI.toLocaleString('en-US')} token đã cache. Latent của MLA dùng tỉ lệ của
              DeepSeek-V2, d_c = 4d và d_rope = d/2 (facts E5). Mô hình: <code>src/sim/kvmath.ts</code>.
            </>
          }
        >
          <>
            The outlined boxes are what is cached during inference (the red boxes of Fig. 20.19). Heads are {D_HEAD} wide
            and the intensity is at N = {N_AI.toLocaleString('en-US')} cached tokens. MLA’s latent uses DeepSeek-V2’s
            proportions, d_c = 4d and d_rope = d/2 (facts E5). Model: <code>src/sim/kvmath.ts</code>.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi={<>Số head truy vấn h_q = {hq}</>}>Query heads h_q = {hq}</Translated>
          <input type="range" min="1" max="64" value={hq} aria-label={tr('Query heads', 'Số head truy vấn')} onChange={(e) => { setHq(Number(e.target.value)); setSel(null); }} />
        </label>
        <span className="seg" role="group" aria-label={tr('Variant', 'Biến thể')}>
          {(['MHA', 'GQA', 'MQA', 'MLA'] as HeadVariant[]).map((x) => (
            <button key={x} aria-pressed={v === x} disabled={x === 'GQA' && gqaHkv == null} onClick={() => setVariant(x)}>{x}</button>
          ))}
        </span>
        {v === 'GQA' ? (
          <label className="field">
            <Translated vi="Số head K/V">K/V heads</Translated>
            <select value={hkv} onChange={(e) => setGqaPick(Number(e.target.value))} aria-label={tr('K/V heads', 'Số head K/V')}>
              {groups.map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
          </label>
        ) : null}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Query heads and the K/V heads they read', 'Các head truy vấn và các head K/V mà chúng đọc')} style={{ width: '100%', maxWidth: W, display: 'block' }}
        data-hq={hq} data-hkv={hkv} data-variant={v}>
        <text x={20} y={yQ - 10} fontSize="11.5" fontWeight="600">
          <Translated vi="các head truy vấn">query heads</Translated>
        </text>
        {Array.from({ length: hq }, (_, h) => {
          const k = kvHeadOf(h, hq, hkv);
          const on = selected === h;
          const x2 = v === 'MLA' ? W / 2 : kx(k) + kw / 2;
          return (
            <line key={'l' + h} x1={qx(h) + qw / 2} y1={yQ + 26} x2={x2} y2={yK - 4}
              stroke={on ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={on ? 2.4 : 0.9} strokeDasharray={on ? undefined : '3 3'} />
          );
        })}
        {Array.from({ length: hq }, (_, h) => (
          <g key={h} onClick={() => setSel(h)} style={{ cursor: 'pointer' }} data-qhead={h}>
            <rect x={qx(h)} y={yQ} width={qw} height={26} rx="2" fill={selected === h ? 'var(--reg)' : 'var(--reg-wash)'} stroke="var(--reg)" />
            {qw > 14 ? <text x={qx(h) + qw / 2} y={yQ + 17} fontSize={Math.min(11, qw / 2)} textAnchor="middle" style={{ fill: selected === h ? 'var(--page)' : 'var(--ink)' }}>{h}</text> : null}
          </g>
        ))}
        {v === 'MLA' ? (
          <g>
            <rect x={W / 2 - 90} y={yK} width={180} height={34} rx="2" fill="var(--global-wash)" stroke="var(--global)" strokeWidth="2.4" />
            <text x={W / 2} y={yK + 21} fontSize="11.5" textAnchor="middle">
              <Translated vi={<>một latent, {4 * D_HEAD} + {D_HEAD / 2} giá trị</>}>one latent, {4 * D_HEAD} + {D_HEAD / 2} values</Translated>
            </text>
            <text x={W / 2} y={yK + 54} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
              <Translated vi="đóng vai trò cả key lẫn value; K và V không bao giờ được tạo ra">
                serves as key and value; K and V are never formed
              </Translated>
            </text>
          </g>
        ) : (
          <g>
            <rect x={14} y={yK - 8} width={W - 28} height={66} rx="3" fill="none" stroke="var(--global)" strokeWidth="2.4" />
            <text x={W - 20} y={yK + 72} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--global)' }}>
              <Translated vi="được cache">cached</Translated>
            </text>
            {Array.from({ length: hkv }, (_, k) => (
              <g key={k} data-kvhead={k}>
                <rect x={kx(k)} y={yK} width={kw / 2 - 1} height={42} fill={target === k ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
                <rect x={kx(k) + kw / 2 + 1} y={yK} width={kw / 2 - 1} height={42} fill={target === k ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
                {kw > 22 ? (
                  <>
                    <text x={kx(k) + kw / 4} y={yK + 26} fontSize="10" textAnchor="middle" style={{ fill: target === k ? 'var(--page)' : 'var(--ink)' }}>K</text>
                    <text x={kx(k) + (3 * kw) / 4} y={yK + 26} fontSize="10" textAnchor="middle" style={{ fill: target === k ? 'var(--page)' : 'var(--ink)' }}>V</text>
                  </>
                ) : null}
              </g>
            ))}
            <text x={20} y={yK + 72} fontSize="11.5" fontWeight="600">
              <Translated vi="các head K/V">K/V heads</Translated>
            </text>
          </g>
        )}
      </svg>
      <table className="data" style={{ background: 'var(--page)', margin: '0.6rem 0 0' }} data-heads="">
        <tbody>
          <tr>
            <td>
              <Translated vi={<>Head truy vấn {selected ?? '(nhấn chọn một head)'} đọc</>}>
                Query head {selected ?? '(click one)'} reads
              </Translated>
            </td>
            <td className="num" data-reads={target ?? ''}>
              {selected == null ? '–' : v === 'MLA' ? tr('the latent', 'latent') : tr(`K/V head ${target}`, `head K/V ${target}`)}
            </td>
            <td className="muted">
              {v === 'MLA' ? tr('every head reads the same latent', 'mọi head đọc cùng một latent') : `⌊h ÷ (h_q / h_kv)⌋ = ⌊${selected ?? 'h'} ÷ ${fmt(hq / hkv)}⌋`}
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="g_q, số head truy vấn trên mỗi head K/V">g_q, query heads per K/V head</Translated>
            </td>
            <td className="num" data-gq={v === 'MLA' ? '' : gq}>{v === 'MLA' ? '–' : fmt(gq)}</td>
            <td className="muted">
              <Translated vi="thứ Pt. 20.12 và 20.13 dùng (định nghĩa của [13])">what Eqs. 20.12 and 20.13 use ([13]’s definition)</Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="G, số nhóm">G, number of groups</Translated>
            </td>
            <td className="num" data-groups={v === 'MLA' ? '' : G}>{v === 'MLA' ? '–' : G}</td>
            <td className="muted">
              <Translated vi="thứ bài báo GQA gọi là GQA-G; bằng h_kv">what the GQA paper calls GQA-G; equals h_kv</Translated>
            </td>
          </tr>
          <tr>
            <td>
              <Translated vi="Lượng cache mỗi token mỗi lớp">Cached per token per layer</Translated>
            </td>
            <td className="num" data-values={values}>
              <Translated vi={<>{values.toLocaleString('en-US')} giá trị</>}>{values.toLocaleString('en-US')} values</Translated>
            </td>
            <td className="muted">{v === 'MLA' ? 'd_c + d_rope' : `2 × ${hkv} × ${D_HEAD}`}{/* i18n-ignore: formula */}</td>
          </tr>
          <tr>
            <td>
              <Translated vi="Cường độ tính toán của decode (số phép nhân-cộng mỗi phần tử)">Decode intensity (multiply-adds per element)</Translated>
            </td>
            <td className="num" data-ai={ai.toFixed(3)}>{fmt(ai, 4)}</td>
            <td className="muted">{v === 'MLA' ? tr('2N h_q / (2h_q + N), about 2h_q', '2N h_q / (2h_q + N), khoảng 2h_q') : tr('Eq. 20.13, about g_q', 'Pt. 20.13, khoảng g_q')}</td>
          </tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- contiguous or paged

const POOLS = [512, 1024, 2048, 4096];
const BLOCKS = [16, 32, 64, 128, 256];
const MAXSEQ = [128, 256, 512, 1024];

function PagedStepper() {
  const [cfg, setCfg] = useState<PagedConfig>({ pool: 1024, block: 16, maxSeq: 256, requests: 16, seed: 1 });
  const run = useMemo(() => simulate(cfg), [cfg]);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const last = run.frames.length - 1;
  const at = Math.min(t, last);
  const f = run.frames[at];
  const peak = useMemo(() => peakLive(run), [run]);
  useEffect(() => { setT(0); setPlaying(false); }, [cfg]);
  useEffect(() => {
    if (!playing) return;
    if (at >= last) { setPlaying(false); return; }
    const id = window.setTimeout(() => setT((x) => x + 1), 120);
    return () => window.clearTimeout(id);
  }, [playing, at, last]);
  const set = (k: keyof PagedConfig, v: number) => setCfg((c) => ({ ...c, [k]: v }));
  const nextEvent = () => {
    for (let i = at + 1; i <= last; i++)
      if (run.frames[i].paged.events.length || run.frames[i].contiguous.events.length) return setT(i);
    setT(last);
  };
  const cRows = run.slots;
  const pRows = Math.ceil(run.blocks / Math.min(16, run.blocks));
  const cH = Math.min(230, cRows * 22) + (run.cfg.pool > cRows * run.cfg.maxSeq ? 14 : 0);
  const pH = Math.min(240, pRows * 26);
  const counters = (s: typeof f.paged) => (
    <table className="data" style={{ background: 'var(--page)', margin: '0.5rem 0 0' }}>
      <tbody>
        <tr>
          <td><Translated vi="đã đặt trước">reserved</Translated></td>
          <td className="num" data-reserved={s.reserved}>{s.reserved}</td>
          <td><Translated vi="đã dùng">used</Translated></td>
          <td className="num" data-used={s.used}>{s.used}</td>
          <td><Translated vi="lãng phí">wasted</Translated></td>
          <td className="num" data-wasted={s.wasted}>{s.wasted}</td>
        </tr>
        <tr>
          <td><Translated vi="đang chạy">running</Translated></td>
          <td className="num" data-running={s.live.length}>{s.live.length}</td>
          <td><Translated vi="đang chờ">waiting</Translated></td>
          <td className="num" data-waiting={s.waiting.length}>{s.waiting.length}</td>
          <td><Translated vi="đã xong">done</Translated></td>
          <td className="num" data-done={s.done}>{s.done}</td>
        </tr>
        <tr>
          <td><Translated vi="đã nhận">admitted</Translated></td>
          <td className="num" data-admitted={s.admitted}>{s.admitted}</td>
          <td colSpan={2}><Translated vi="đã đẩy ra">swapped out</Translated></td>
          <td className="num" colSpan={2} data-evictions={s.evictions}>{s.evictions}</td>
        </tr>
      </tbody>
    </table>
  );
  return (
    <Bench
      title={tr('Contiguous or paged', 'Liên tục hay phân trang')}
      task={
        <Translated vi="Chạy cùng một dòng yêu cầu qua cả hai vùng nhớ. Vùng nào giữ được nhiều yêu cầu đang chạy hơn, và vì sao?">
          Run the same stream of requests through both pools. Which one keeps more requests running, and why?
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Kích thước tính bằng token của cache một lớp. Mỗi yêu cầu tăng một token mỗi bước đến một độ dài cuối mà không
              bộ cấp phát nào biết trước. Đặc: token đã lưu; nhạt: đã đặt trước nhưng trống; nét đứt: trống. Mô hình:{' '}
              <code>src/sim/paged.ts</code>.
            </>
          }
        >
          <>
            Sizes are in tokens of one layer’s cache. Each request grows one token per step to a final length neither
            allocator knows in advance. Solid: tokens stored; washed: reserved but empty; dashed: free. Model:{' '}
            <code>src/sim/paged.ts</code>.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Vùng nhớ (token)">Pool (tokens)</Translated>
          <select value={cfg.pool} onChange={(e) => set('pool', Number(e.target.value))} aria-label={tr('Pool size', 'Kích thước vùng nhớ')}>
            {POOLS.map((x) => <option key={x} value={x}>{x.toLocaleString('en-US')}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Khối (token)">Block (tokens)</Translated>
          <select value={cfg.block} onChange={(e) => set('block', Number(e.target.value))} aria-label={tr('Block size', 'Kích thước khối')}>
            {BLOCKS.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="field">
          {'max_seq'}{/* i18n-ignore: identifier */}
          <select value={cfg.maxSeq} onChange={(e) => set('maxSeq', Number(e.target.value))} aria-label={tr('Maximum sequence length', 'Độ dài chuỗi tối đa')}>
            {MAXSEQ.map((x) => <option key={x} value={x}>{x.toLocaleString('en-US')}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi={<>Số yêu cầu = {cfg.requests}</>}>Requests = {cfg.requests}</Translated>
          <input type="range" min="4" max="48" value={cfg.requests} aria-label={tr('Requests', 'Số yêu cầu')} onChange={(e) => set('requests', Number(e.target.value))} />
        </label>
        <button onClick={() => set('seed', cfg.seed + 1)}>
          <Translated vi="Dòng khác">Another stream</Translated>
        </button>
      </div>
      {run.cfg.pool !== cfg.pool ? <p className="muted" style={{ marginTop: 0 }}>
          <Translated vi={<>Vùng nhớ được nâng lên {run.cfg.pool} token để vừa một yêu cầu max_seq.</>}>
            The pool was raised to {run.cfg.pool} tokens so that one max_seq request fits.
          </Translated>
        </p> : null}
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setT(0); }} disabled={at === 0}>
          <Translated vi="Đặt lại">Reset</Translated>
        </button>
        <button onClick={() => { setPlaying(false); setT(at - 1); }} disabled={at === 0}>
          <Translated vi="Lùi">Back</Translated>
        </button>
        <button className="primary" onClick={() => { setPlaying(false); setT(at + 1); }} disabled={at >= last}>
          <Translated vi="Bước">Step</Translated>
        </button>
        <button onClick={() => { setPlaying(false); nextEvent(); }} disabled={at >= last}>
          <Translated vi="Sự kiện tiếp theo">Next event</Translated>
        </button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= last}>{playing ? tr('Pause', 'Tạm dừng') : tr('Run', 'Chạy')}</button>
        <input type="range" min="0" max={last} value={at} aria-label={tr('Step', 'Bước')} style={{ flex: '1 1 10rem' }} onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }} />
        <span className="muted" aria-live="polite" data-step={f.step}>
          <Translated vi={<>bước {f.step} trên {last}</>}>step {f.step} of {last}</Translated>
        </span>
      </div>
      <div className="cols">
        <div data-pool="contiguous">
          <div className="mx-label t-global">
            <Translated vi={<>Liên tục: {run.slots} ô, mỗi ô {run.cfg.maxSeq} token</>}>
              Contiguous: {run.slots} slots of {run.cfg.maxSeq} tokens
            </Translated>
          </div>
          <svg viewBox={`0 0 360 ${cH}`} style={{ width: '100%', maxWidth: 360, display: 'block' }} role="img" aria-label={tr('Contiguous pool', 'Vùng nhớ liên tục')}>
            <ContiguousPool run={run} side={f.contiguous} x={2} y={2} w={356} h={Math.min(230, cRows * 22)} />
          </svg>
          {counters(f.contiguous)}
        </div>
        <div data-pool="paged">
          <div className="mx-label t-global">
            <Translated vi={<>Phân trang: {run.blocks} khối, mỗi khối {run.cfg.block} token, {f.paged.free.length} khối trống</>}>
              Paged: {run.blocks} blocks of {run.cfg.block} tokens, {f.paged.free.length} free
            </Translated>
          </div>
          <svg viewBox={`0 0 360 ${pH}`} style={{ width: '100%', maxWidth: 360, display: 'block' }} role="img" aria-label={tr('Paged pool', 'Vùng nhớ phân trang')}>
            <PagedPool run={run} side={f.paged} x={2} y={2} w={356} h={pH - 4} />
          </svg>
          {counters(f.paged)}
          <p className="muted" style={{ fontSize: '0.8rem', margin: '0.4rem 0 0' }} data-tables="">
            <Translated
              vi={
                <>
                  Bảng khối:{' '}
                  {f.paged.live.length ? f.paged.live.slice(0, 10).map((r) => `r${r.id} → [${r.blocks!.join(', ')}]`).join('; ') : 'chưa có'}
                  {f.paged.live.length > 10 ? `; … thêm ${f.paged.live.length - 10}` : ''}
                </>
              }
            >
              Block tables: {f.paged.live.length ? f.paged.live.slice(0, 10).map((r) => `r${r.id} → [${r.blocks!.join(', ')}]`).join('; ') : 'none yet'}
              {f.paged.live.length > 10 ? `; … ${f.paged.live.length - 10} more` : ''}
            </Translated>
          </p>
        </div>
      </div>
      <div className="narrate" role="status" style={{ marginTop: '0.8rem' }} data-events="">
        {f.step === 0 ? (
          tr('Nothing has arrived yet. Press Step, Next event or Run.', 'Chưa có gì đến. Nhấn Bước, Sự kiện tiếp theo hoặc Chạy.')
        ) : (
          <Translated
            vi={
              <>
                {/* The simulator's events read "r<id> admitted|finished|swapped out|swapped back in"; the vi branch rewrites the verb. */}
                Bước {f.step}. Phân trang:{' '}
                {f.paged.events
                  .map((e) =>
                    e
                      .replace(/ swapped back in$/, ' được đưa vào lại')
                      .replace(/ swapped out$/, ' bị đẩy ra')
                      .replace(/ admitted$/, ' được nhận')
                      .replace(/ finished$/, ' đã xong'),
                  )
                  .join(', ') || 'không có gì mới'}
                . Liên tục:{' '}
                {f.contiguous.events
                  .map((e) => e.replace(/ admitted$/, ' được nhận').replace(/ finished$/, ' đã xong'))
                  .join(', ') || 'không có gì mới'}
                .
              </>
            }
          >
            <>Step {f.step}. Paged: {f.paged.events.join(', ') || 'nothing new'}. Contiguous: {f.contiguous.events.join(', ') || 'nothing new'}.</>
          </Translated>
        )}
        {at >= last ? (
          <Translated
            vi={
              <>
                {' '}Cả {cfg.requests} yêu cầu đã xong: liên tục sau{' '}
                <b data-finished-contiguous={run.finishedAt.contiguous ?? ''}>{run.finishedAt.contiguous}</b> bước, phân trang
                sau <b data-finished-paged={run.finishedAt.paged ?? ''}>{run.finishedAt.paged}</b>. Tối đa {peak.contiguous} so
                với {peak.paged} chạy cùng lúc
                {f.paged.evictions ? `; vùng nhớ phân trang đã cạn ${f.paged.evictions} lần và đẩy một yêu cầu ra` : ''}.
              </>
            }
          >
            <>
              {' '}All {cfg.requests} requests done: contiguous after <b data-finished-contiguous={run.finishedAt.contiguous ?? ''}>{run.finishedAt.contiguous}</b> steps,
              paged after <b data-finished-paged={run.finishedAt.paged ?? ''}>{run.finishedAt.paged}</b>. At most {peak.contiguous} against {peak.paged} running at once
              {f.paged.evictions ? `; the paged pool ran out ${f.paged.evictions} ${f.paged.evictions === 1 ? 'time' : 'times'} and swapped a request out` : ''}.
            </>
          </Translated>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS207: Record<string, Question> = {
  'ch20/20.7/q-gq': {
    kind: 'numeric',
    id: 'ch20/20.7/q-gq',
    prompt: (
      <Translated vi={<>Mistral 7B có 32 head truy vấn và 8 head K/V. g_q bằng bao nhiêu theo cách Pt. 20.12 và 20.13 dùng nó?</>}>
        <>Mistral 7B has 32 query heads and 8 K/V heads. What is g_q as Eqs. 20.12 and 20.13 use it?</>
      </Translated>
    ),
    answer: 4,
    explain: (
      <Translated
        vi={
          <p>
            Trong các công thức, số head K/V là h_q / g_q, nên g_q = 32 / 8 = 4 head truy vấn trên mỗi head K/V. Bài báo GQA
            sẽ gọi mô hình này là GQA-8: G = 8 nhóm. Cùng một mô hình, hai con số; hãy nói rõ bạn muốn nói con số nào.
          </p>
        }
      >
        <p>
          In the formulas the number of K/V heads is h_q / g_q, so g_q = 32 / 8 = 4 query heads per K/V head. The GQA paper
          would call this model GQA-8: G = 8 groups. Same model, two numbers; say which one you mean.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md E3, E2', // i18n-ignore: citation
  },
  'ch20/20.7/q-gqa-ai': {
    kind: 'numeric',
    id: 'ch20/20.7/q-gqa-ai',
    prompt: (
      <Translated vi={<>Tính Pt. 20.13 ở N = 4,096 token đã cache, h_q = 32 và g_q = 4.</>}>
        <>Evaluate Eq. 20.13 at N = 4,096 cached tokens, h_q = 32 and g_q = 4.</>
      </Translated>
    ),
    answer: 3.996,
    tol: 0.01,
    explain: (
      <Translated
        vi={
          <p>
            N h_q / (h_q + (h_q/g_q) N) = 131,072 / (32 + 8 × 4,096) = 131,072 / 32,800 = 3.996, khoảng g_q: mỗi phần tử K/V
            được nạp giờ gặp bốn truy vấn.
          </p>
        }
      >
        <p>N h_q / (h_q + (h_q/g_q) N) = 131,072 / (32 + 8 × 4,096) = 131,072 / 32,800 = 3.996, about g_q: each loaded K/V element now meets four queries.</p>
      </Translated>
    ),
    source: 'ch. 20 §20.7, p. 509, Eq. 20.13; notes/today/ch20-facts.md F1', // i18n-ignore: citation
  },
  'ch20/20.7/q-head-map': {
    kind: 'numeric',
    id: 'ch20/20.7/q-head-map',
    prompt: (
      <Translated vi={<>Một mô hình có 6 head truy vấn và 2 head K/V (ví dụ của chính FlashAttention). Head truy vấn 4 đọc head K/V nào?</>}>
        <>A model has 6 query heads and 2 K/V heads (FlashAttention’s own example). Which K/V head does query head 4 read?</>
      </Translated>
    ),
    answer: 1,
    explain: (
      <Translated
        vi={
          <p>
            Mỗi head K/V phục vụ 6 / 2 = 3 head truy vấn liên tiếp: 0, 1, 2 đọc head K/V 0 và 3, 4, 5 đọc head K/V 1. Trong
            mã, kv_head = h / 3 làm tròn xuống, mà llama.cpp viết bằng gqa_ratio của nó.
          </p>
        }
      >
        <p>Each K/V head serves 6 / 2 = 3 consecutive query heads: 0, 1, 2 read K/V head 0 and 3, 4, 5 read K/V head 1. In code, kv_head = h / 3 rounded down, which llama.cpp writes with its gqa_ratio.</p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md A8, D1', // i18n-ignore: citation
  },
  'ch20/20.7/q-mla-bytes': {
    kind: 'numeric',
    id: 'ch20/20.7/q-mla-bytes',
    prompt: (
      <Translated vi={<>DeepSeek-V2 cache d_c + d_h^R = 512 + 64 giá trị mỗi token mỗi lớp, trong 60 lớp, ở 16 bit. Bao nhiêu byte mỗi token?</>}>
        <>DeepSeek-V2 caches d_c + d_h^R = 512 + 64 values per token per layer, in 60 layers, at 16 bits. How many bytes per token?</>
      </Translated>
    ),
    answer: 69120,
    unit: 'bytes',
    explain: (
      <Translated
        vi={
          <p>
            576 × 60 × 2 = 69,120 byte. Eq. 20.14 của sách với latent độ rộng d = 128 cho 60 × 128 × 2 = 15,360, ít hơn 4.5
            lần. Bản MHA tương đương, 128 head cỡ 128, sẽ là 3.75 MiB: nhiều hơn 57 lần.
          </p>
        }
      >
        <p>576 × 60 × 2 = 69,120 bytes. The book’s Eq. 20.14 with a latent of width d = 128 gives 60 × 128 × 2 = 15,360, 4.5 times less. The MHA equivalent, 128 heads of 128, would be 3.75 MiB: 57 times more.</p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md E5, F1', // i18n-ignore: citation
    vi: { unit: 'byte' },
  },
  'ch20/20.7/q-mla-ai': {
    kind: 'choice',
    id: 'ch20/20.7/q-mla-ai',
    prompt: (
      <Translated vi={<>Vì sao cường độ tính toán decode của MLA là khoảng 2 h_q, gấp đôi của MQA?</>}>
        <>Why is MLA’s decode intensity about 2 h_q, twice MQA’s?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi phần tử latent được nạp đóng vai trò cả key lẫn value.">
            Each loaded latent element serves as both key and value.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Vì latent có kích thước bằng một nửa một head attention.">
            Because the latent is half the size of an attention head.
          </Translated>
        ),
        why: (
          <Translated vi="Latent của DeepSeek-V2 rộng 4.5 head, và kích thước không quyết định tỉ số; mức dùng lại mới quyết định.">
            DeepSeek-V2’s latent is 4.5 heads wide, and size does not set the ratio; reuse does.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Vì MLA giải nén K và V trên chip ngay trước khi dùng.">
            Because MLA decompresses K and V on chip just before use.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là cách sách mô tả, và nó sẽ không cho ra con số này: K và V được giải nén sẽ mỗi thứ được đọc một lần, như trong MQA.">
            That is the book’s account, and it would not give this figure: decompressed K and V would each be read once, as in
            MQA.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Vì nó chỉ cache K và tính lại V từ đó.">Because it caches only K and recomputes V from it.</Translated>
        ),
        why: (
          <Translated vi="Nó cache một latent đóng cả hai vai.">It caches one latent that plays both parts.</Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khi suy luận, các phép chiếu lên (up-projection) được hấp thụ vào W_Q và W_O, nên attention chạy trực tiếp trên
            một latent duy nhất và K và V không bao giờ được tạo ra. MQA nạp một phần tử K và một phần tử V, mỗi phần tử được
            h_q truy vấn dùng: khoảng h_q phép toán mỗi phần tử. MLA nạp một phần tử latent và dùng nó cho cả điểm số lẫn tổng
            có trọng số, bởi cả h_q head: khoảng 2 h_q. Nguồn được trích dẫn nói latent &ldquo;serves both the key and the
            value&rdquo;.
          </p>
        }
      >
        <p>
          At inference the up-projections are absorbed into W_Q and W_O, so attention runs directly on the single latent
          and K and V are never formed. MQA loads a K element and a V element, each used by h_q queries: about h_q
          operations per element. MLA loads one latent element and uses it for the scores and for the weighted sum, by all h_q heads: about 2 h_q. The cited source
          says the latent “serves both the key and the value”.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md E4, E5', // i18n-ignore: citation
  },
  'ch20/20.7/q-paged': {
    kind: 'choice',
    id: 'ch20/20.7/q-paged',
    prompt: (
      <Translated vi={<>Trong lúc decode bình thường, các khối KV của PagedAttention nằm ở đâu, và bảng khối mang lại điều gì?</>}>
        <>During normal decoding, where are PagedAttention’s KV blocks, and what does the block table buy?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Ở bất cứ đâu trong bộ nhớ GPU; bộ nhớ được lấy từng khối một.">
            Anywhere in GPU memory; memory is taken one block at a time.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Trong bộ nhớ host, được sao chép vào GPU khi cần.">
            In host memory, copied into the GPU when they are needed.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là cách sách mô tả. Bài báo chỉ dùng bộ nhớ host khi hết khối GPU và phải đẩy một chuỗi ra.">
            That is the book’s account. The paper uses host memory only when it runs out of GPU blocks and must evict a
            sequence.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Trong một buffer liên tục cho mỗi chuỗi, có kích thước theo độ dài tối đa của nó.">
            In one contiguous buffer per sequence, sized for its maximum length.
          </Translated>
        ),
        why: <Translated vi="Đó là thứ mà phân trang thay thế.">That is what paging replaces.</Translated>,
      },
      {
        label: (
          <Translated vi="Trên đĩa, với bảng khối là chỉ mục vào tệp.">On disk, with the block table as an index into the file.</Translated>
        ),
        why: (
          <Translated vi="Không có gì trong thiết kế đi xuống đĩa.">Nothing in the design goes to disk.</Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bảng ánh xạ các khối logic của mỗi chuỗi sang khối vật lý, nên phần lãng phí tối đa là một khối được lấp một
            phần cho mỗi chuỗi. Bài báo vLLM: PagedAttention &ldquo;allows storing continuous keys and values in
            non-contiguous memory space&rdquo;, từng khối một, và chỉ đẩy các khối của chuỗi bị loại sang bộ nhớ CPU khi
            hết khối GPU trống.
          </p>
        }
      >
        <p>
          The table maps each sequence&rsquo;s logical blocks to physical ones, so the waste is at most one partly filled
          block per sequence. The vLLM paper: PagedAttention &ldquo;allows storing continuous keys and values in
          non-contiguous memory space&rdquo;, block by block, and swaps evicted sequences&rsquo; blocks to CPU memory only
          when free GPU blocks run out.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch20-facts.md E6; book p. 509',
    vi: { source: 'notes/today/ch20-facts.md E6; sách p. 509' },
  },
  'ch20/20.7/q-moe': {
    kind: 'choice',
    id: 'ch20/20.7/q-moe',
    prompt: (
      <Translated vi={<>Mixtral có 47B tham số nhưng chỉ dùng 13B trong số đó cho mỗi token. Mixture of experts cắt giảm chi phí nào, và chi phí nào không?</>}>
        <>Mixtral has 47B parameters but uses 13B of them per token. Which cost does a mixture of experts cut, and which not?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Tính toán mỗi token giảm; bộ nhớ trọng số thì không.">Compute per token falls; weight memory does not.</Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Cả tính toán lẫn bộ nhớ đều giảm.">Both the compute and the memory fall.</Translated>,
        why: (
          <Translated vi="Bất kỳ token nào cũng có thể được định tuyến tới bất kỳ expert nào, nên tất cả đều phải được nạp.">
            Any token may be routed to any expert, so all of them must be loaded.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Chỉ bộ nhớ trọng số giảm.">Only the weight memory falls.</Translated>,
        why: <Translated vi="Ngược lại mới đúng.">It is the other way round.</Translated>,
      },
      {
        label: (
          <Translated vi="Không cái nào: bộ định tuyến tốn bằng phần nó tiết kiệm.">Neither: the router costs as much as it saves.</Translated>
        ),
        why: (
          <Translated vi="Mỗi token chạy 2 trong 8 expert, nên tính toán của nó có giảm.">
            Each token runs 2 of the 8 experts, so its compute does fall.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi lớp có 8 expert feed-forward và một bộ định tuyến chọn 2 cho mỗi token, nên tính toán giảm. Mọi expert vẫn
            phải thường trú, nên toàn bộ trọng số nằm trong bộ nhớ, và KV cache không đổi: attention của Mixtral là của
            Mistral 7B.
          </p>
        }
      >
        <p>Each layer has 8 feed-forward experts and a router picks 2 per token, so the compute falls. Every expert must still be resident, so the weights all live in memory, and the KV cache is unchanged: Mixtral’s attention is Mistral 7B’s.</p>
      </Translated>
    ),
    source: 'ch. 20 §20.7, p. 510; notes/today/ch20-facts.md E7', // i18n-ignore: citation
  },
};

// ---------------------------------------------------------------- the page

export function S207() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 20.6 truy vấn đề bộ nhớ về một lựa chọn thiết kế. Trong multi-head attention, mỗi head có ma trận trọng số
            riêng, nên mỗi head tính, và phải cache, K và V của riêng nó. Mỗi biến thể trong mục này cache ít hơn, và, như
            một hệ quả phụ, làm nhiều phép tính hơn trên mỗi byte nó đọc.
          </p>
        }
      >
        <p>
          Section 20.6 traced the memory problem to one design choice. In multi-head attention every head has its own weight
          matrices, so every head computes, and must cache, its own K and V. Each variant in this section caches less, and,
          as a side effect, does more arithmetic per byte it reads.
        </p>
      </Translated>
      <Source section="20.7" pages="508" />

      <h2>
        <Translated vi="Multi-query attention">Multi-query attention</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Multi-query attention (MQA) cho mọi head truy vấn dùng chung một head K và một head V duy nhất. Cache co lại theo
            số head, còn b × N × 2 × l × d × p (Eq. 20.10).
          </p>
        }
      >
        <p>
          Multi-query attention (MQA) lets all the query heads share a single K head and a single V head. The cache shrinks by
          the number of heads, to b × N × 2 × l × d × p (Eq. 20.10).
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts E7)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Sách không trích dẫn nguồn của MQA: Shazeer 2019, với phần tóm tắt nêu đúng vấn đề này, &ldquo;the
              memory-bandwidth cost of repeatedly loading the large ‘keys’ and ‘values’ tensors&rdquo;.
            </p>
          }
        >
          <p>The book does not cite MQA’s source: Shazeer 2019, whose abstract names this very problem, “the memory-bandwidth cost of repeatedly loading the large ‘keys’ and ‘values’ tensors”.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Và mỗi phần tử K hoặc V được nạp giờ gặp cả h_q truy vấn, nên cường độ tính toán decode trở thành N h_q / (h_q +
            N), khoảng h_q (Eq. 20.11). Số byte giảm còn phép tính giữ nguyên: đó là toàn bộ mánh.
          </p>
        }
      >
        <p>
          And each loaded K or V element now meets all h_q queries, so the decode intensity becomes N h_q / (h_q + N), about
          h_q (Eq. 20.11). The bytes went down and the arithmetic stayed: that is the whole trick.
        </p>
      </Translated>
      <Source section="20.7" pages="508" figs={tr('Eqs. 20.10–20.11', 'Pt. 20.10–20.11')} />

      <h2>
        <Translated vi="Grouped-query attention, và g_q có nghĩa là gì">Grouped-query attention, and what g_q means</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Grouped-query attention (GQA) nằm giữa hai cái: các head truy vấn được chia thành nhóm, và mỗi nhóm dùng chung một
            head K và một head V. Sách cho cache là b × N × 2 × l × (h_q / g_q) × d × p (Eq. 20.12) và cường độ tính toán
            khoảng g_q (Eq. 20.13).
          </p>
        }
      >
        <p>
          Grouped-query attention (GQA) sits between the two: query heads are divided into groups, and each group shares one K
          head and one V head. The book gives the cache as b × N × 2 × l × (h_q / g_q) × d × p (Eq. 20.12) and the
          intensity as about g_q (Eq. 20.13).
        </p>
      </Translated>
      <Source section="20.7" pages="508–509" figs={tr('Eqs. 20.12–20.13, Fig. 20.19', 'Pt. 20.12–20.13, Hình 20.19')} />
      <Translated
        vi={
          <p>
            Hãy đọc công thức trước phần lời. h_q / g_q là số head K/V, nên g_q phải là số head truy vấn trên mỗi head K/V:
            g_q = 1 là MHA và g_q = h_q là MQA. Phần lời của sách gọi g_q là &ldquo;a few groups of query heads&rdquo;, tức
            là G của bài báo GQA, số nhóm, bằng số head K/V. Với Mistral 7B đó là 4 và 8. Khóa học này dùng g_q = h_q / h_kv
            xuyên suốt.
          </p>
        }
      >
        <p>
          Read the formula before the prose. h_q / g_q is the number of K/V heads, so g_q must be the number of query heads
          per K/V head: g_q = 1 is MHA and g_q = h_q is MQA. The book’s prose calls g_q “a few groups of query heads”, which
          is the GQA paper’s G, the number of groups, equal to the number of K/V heads. For Mistral 7B those are 4 and 8. This
          course uses g_q = h_q / h_kv throughout.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts E3)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              [13], nguồn của sách cho các công thức: &ldquo;We define the group size g_q as the number of query heads per
              distinct KV head.&rdquo; Bài báo GQA: &ldquo;GQA-G refers to grouped-query with G groups.&rdquo;
            </p>
          }
        >
          <p>[13], the book’s source for the formulas: “We define the group size g_q as the number of query heads per distinct KV head.” The GQA paper: “GQA-G refers to grouped-query with G groups.”</p>
        </Translated>
      </Today>
      <HeadsBench />
      <Translated
        vi={
          <p>
            Cách ánh xạ đó, các head truy vấn liên tiếp dùng chung một head K/V, là thứ các kernel decode bạn sẽ đọc cài đặt:
            head truy vấn h đọc head K/V h / g_q, và K và V không bị sao chép cho mỗi head truy vấn để các hình dạng khớp
            nhau.
          </p>
        }
      >
        <p>
          That mapping, consecutive query heads sharing one K/V head, is what the decode kernels you will read implement:
          query head h reads K/V head h / g_q, and K and V are not copied once per query head to make the shapes match.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts D1, A8, D4)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              llama.cpp tính <code>gqa_ratio = Q-&gt;ne[2] / K-&gt;ne[2]</code> và xử lý nhiều head Q cho mỗi head K/V được
              nạp. <code>flash_attn_with_kvcache</code> của FlashAttention: với 6 head Q và 2 head K/V, các head Q 0, 1, 2
              đọc head K/V 0. MLX yêu cầu k và v không được lát sẵn cho khớp với q.
            </p>
          }
        >
          <p>llama.cpp computes <code>gqa_ratio = Q-&gt;ne[2] / K-&gt;ne[2]</code> and processes several Q heads per loaded K/V head. FlashAttention’s <code>flash_attn_with_kvcache</code>: with 6 Q heads and 2 K/V heads, Q heads 0, 1, 2 read K/V head 0. MLX asks that k and v not be pre-tiled to match q.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Một kernel GQA tốt nạp mỗi hàng K/V một lần và dùng nó cho cả g_q head truy vấn của nhóm, đó là điều lab 20.5 yêu
            cầu bạn viết và đo.
          </p>
        }
      >
        <p>
          A good GQA kernel loads each K/V row once and uses it for all g_q query heads of its group, which is what lab 20.5
          asks you to write and measure.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Lab 20.5 là của khóa học này; xem <a href={href('ch20', 'labs')}>trang lab</a>.</p>}>
          <p>Lab 20.5 is ours; see the <a href={href('ch20', 'labs')}>labs page</a>.</p>
        </Translated>
      </Beyond>
      <Ask q={questionsS207['ch20/20.7/q-gq']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS207['ch20/20.7/q-head-map']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS207['ch20/20.7/q-gqa-ai']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            MHA, GQA và MQA đều hợp với cách phân ô flash attention của <a href={href('ch20', '20.5')}>mục 20.5</a>. Chúng
            đánh đổi tính toán, bộ nhớ và độ chính xác theo cách khác nhau; cấu trúc kernel vẫn như cũ.
          </p>
        }
      >
        <p>
          MHA, GQA and MQA all fit the flash-attention tiling of <a href={href('ch20', '20.5')}>section 20.5</a>. They trade
          computation, memory and accuracy differently; the kernel structure stays the same.
        </p>
      </Translated>
      <Source section="20.7" pages="509" />

      <h2>
        <Translated vi="Phân trang cache">Paging the cache</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ngay cả với ít head K/V hơn, kích thước batch và độ dài ngữ cảnh vẫn bị giới hạn bởi bộ nhớ, và sách thêm một
            nguyên nhân thứ hai: bộ nhớ cache thường được đặt trước cho đầu vào lớn nhất có thể, nên phần lớn nằm phân mảnh và
            không được dùng. PagedAttention, lấy cảm hứng từ phân trang trong hệ điều hành, chia cache thành các khối kích
            thước bằng nhau mà, theo lời sách, được nạp vào bộ nhớ GPU từ bộ nhớ hệ thống lớn hơn khi cần.
          </p>
        }
      >
        <p>
          Even with fewer K/V heads, batch size and context length are limited by memory, and the book adds a second cause:
          cache memory is usually reserved for the largest possible input, so most of it sits fragmented and unused.
          PagedAttention, inspired by paging in operating systems, splits the cache into equal-size blocks which, in the book’s
          words, are loaded into GPU memory from the larger system memory when needed.
        </p>
      </Translated>
      <Source section="20.7" pages="509" />
      <Translated
        vi={
          <p>
            Thiết kế của bài báo là về chỗ các khối nằm trong bộ nhớ GPU: bất cứ đâu. Mỗi chuỗi có một bảng khối ánh xạ các
            khối logic của nó sang khối vật lý, nên một chuỗi lấy bộ nhớ từng khối một khi nó lớn lên, và lãng phí tối đa một
            khối được lấp một phần. Bộ nhớ host chỉ xuất hiện như phương án dự phòng của vLLM khi nó hết khối và phải đẩy một
            chuỗi ra.
          </p>
        }
      >
        <p>
          The paper’s design is about where the blocks sit in GPU memory: anywhere. Each sequence has a block table that maps
          its logical blocks to physical ones, so a sequence takes memory one block at a time as it grows, and wastes at most
          one partly filled block. Host memory appears only as vLLM’s fallback when it runs out of blocks and has to evict a
          sequence.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts E6, A8, D3)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Trang thiết kế PagedAttention của chính vLLM giờ được đánh dấu là tài liệu lịch sử: việc phân trang nằm bên
              trong các backend attention của nó, thông qua bảng khối. FlashAttention nhận một <code>block_table</code> (các
              trang là bội của 256 token). Vòng các ô token của llama.cpp không có bảng khối: nó không được phân trang.
            </p>
          }
        >
          <p>vLLM’s own PagedAttention design page is now marked historical: paging lives inside its attention backends, through block tables. FlashAttention takes a <code>block_table</code> (pages a multiple of 256 tokens). llama.cpp’s ring of token slots has no block table: it is not paged.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Hãy chạy cùng một dòng yêu cầu qua một vùng nhớ liên tục, đặt trước max_seq cho mỗi yêu cầu như bản tham chiếu
            Llama làm, và một vùng nhớ phân trang. Trước khi nhấn Chạy, hãy dự đoán bên nào sẽ có yêu cầu phải chờ trong khi
            bộ nhớ còn trống.
          </p>
        }
      >
        <p>
          Run the same stream of requests through a contiguous pool, reserved at max_seq per request as the Llama reference
          does, and a paged one. Before you press Run, predict which side will have requests waiting while memory sits empty.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một lớp, kích thước tính bằng token. Độ dài cuối được rút ngẫu nhiên lệch về phía ngắn (giả định của chúng tôi).
              Việc nhận yêu cầu vào vùng phân trang theo thứ tự đến trước phục vụ trước. Khi danh sách trống cạn, một yêu cầu
              bị đẩy ra, như vLLM loại chuỗi khi hết khối; chọn yêu cầu nào (ở đây là yêu cầu mới nhất) là lựa chọn của chúng
              tôi, và việc sao chép không tốn gì. Không chia sẻ tiền tố.
            </p>
          }
        >
          <p>One layer, sizes in tokens. Final lengths are drawn short-skewed (our assumption). Paged admission is first come, first served. When the free list is empty a request is swapped out, as vLLM evicts sequences when it runs out of blocks; which one (here the newest) is our choice, and the copy costs nothing. No prefix sharing.</p>
        </Translated>
      </Simplification>
      <PagedStepper />
      <PagedBlocksScene />
      <Ask q={questionsS207['ch20/20.7/q-paged']} />

      <h2>
        <Translated vi="Multi-head latent attention">Multi-head latent attention</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách mô tả MLA là một phép nén hạng thấp kết hợp: K và V của mỗi token được nén thành một vectơ latent, thứ được
            cache thay vào đó (Eq. 20.14, b × N × l × d × p), và hai ma trận phép chiếu lên giải nén K và V ngay trước khi
            một head dùng chúng. Cường độ tính toán của nó khoảng 2 h_q (Eq. 20.15).
          </p>
        }
      >
        <p>
          The book describes MLA as a low-rank joint compression: each token’s K and V are compressed into one latent vector,
          which is cached instead (Eq. 20.14, b × N × l × d × p), and two up-projection matrices decompress K and V just before
          a head uses them. Its intensity is about 2 h_q (Eq. 20.15).
        </p>
      </Translated>
      <Source section="20.7" pages="509–510" figs={tr('Eqs. 20.14–20.15', 'Pt. 20.14–20.15')} />
      <Translated
        vi={
          <p>
            Bài báo của DeepSeek-V2 khác với cách mô tả đó ở ba điểm quan trọng với một kernel. Latent là một cho mỗi token
            mỗi lớp, dùng chung bởi mọi head, không phải một xuyên qua các lớp. Nó rộng d_c + d_h^R = 512 + 64 = 4.5 lần độ
            rộng head, không phải d, nên Eq. 20.14 đánh giá thấp cache của DeepSeek-V2 4.5 lần. Và khi suy luận, các phép
            chiếu lên được hấp thụ vào W_Q và W_O, nên K và V không bao giờ được tạo ra: attention chạy trực tiếp trên latent,
            vốn đóng vai trò cả key lẫn value. Việc dùng hai lần đó là lý do cường độ tính toán khoảng 2 h_q, gấp đôi MQA.
            Mẫu số của Eq. 20.15 lẽ ra phải là 2 h_q + N, không phải h_q + N; giới hạn thì như nhau.
          </p>
        }
      >
        <p>
          DeepSeek-V2’s paper differs from that account in three ways that matter to a kernel. The latent is one per token per
          layer, shared by all heads, not one across layers. It is d_c + d_h^R = 512 + 64 = 4.5 head widths, not d, so
          Eq. 20.14 understates DeepSeek-V2’s cache 4.5 times. And at inference the up-projections are absorbed into W_Q and
          W_O, so K and V are never formed: attention runs directly on the latent, which serves as both key and value. That
          double use is why the intensity is about 2 h_q, twice MQA’s. Eq. 20.15’s denominator should read 2 h_q + N, not
          h_q + N; the limit is the same.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts E5, E4, D3)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              DeepSeek-V2: &ldquo;W^UK can be absorbed into W^Q, and W^UV can be absorbed into W^O, we even do not need to
              compute keys and values out for attention.&rdquo; llama.cpp không cấp phát V cache nào cả cho các mô hình MLA.
            </p>
          }
        >
          <p>DeepSeek-V2: “W^UK can be absorbed into W^Q, and W^UV can be absorbed into W^O, we even do not need to compute keys and values out for attention.” llama.cpp allocates no V cache at all for MLA models.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Hãy đặt bốn biến thể cạnh nhau cho một mô hình. DeepSeek-V2 là mô hình dùng MLA; hãy thử các mô hình khác để xem
            mỗi biến thể sẽ làm gì với chúng.
          </p>
        }
      >
        <p>Put the four side by side for one model. DeepSeek-V2 is the one that uses MLA; try the others to see what each variant would do to them.</p>
      </Translated>
      <KvExplorer mode="compare" />
      <Ask q={questionsS207['ch20/20.7/q-mla-bytes']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS207['ch20/20.7/q-mla-ai']} />

      <h2>
        <Translated vi="Mixture of experts">Mixture of experts</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kỹ thuật cuối cùng độc lập với cache. Một mixture of experts cho mỗi lớp nhiều mạng con feed-forward chuyên biệt
            và một cổng chọn vài mạng trong số đó cho mỗi token, nên số tham số có thể rất lớn trong khi phép tính trên mỗi
            token vẫn nhỏ.
          </p>
        }
      >
        <p>
          The last technique is orthogonal to the cache. A mixture of experts gives each layer several specialised
          feed-forward sub-networks and a gate that picks a few of them for each token, so the parameter count can be huge
          while the arithmetic per token stays small.
        </p>
      </Translated>
      <Source section="20.7" pages="510" />
      <Translated
        vi={
          <p>
            Thứ nó không thu nhỏ là bộ nhớ: bất kỳ token nào cũng có thể được định tuyến tới bất kỳ expert nào, nên tất cả đều
            phải được nạp. Nó cũng không chạm tới KV cache.
          </p>
        }
      >
        <p>
          What it does not shrink is the memory: any token may be routed to any expert, so all of them must be loaded. Nor
          does it touch the KV cache.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch20-facts E7)">{/* i18n-ignore: date and fact references */}
        <Translated
          vi={
            <p>
              Mixtral: kiến trúc của Mistral 7B với 8 expert feed-forward mỗi lớp, chọn 2 cho mỗi token; &ldquo;each token has
              access to 47B parameters, but only uses 13B active parameters during inference.&rdquo;
            </p>
          }
        >
          <p>Mixtral: Mistral 7B’s architecture with 8 feed-forward experts per layer, 2 chosen per token; “each token has access to 47B parameters, but only uses 13B active parameters during inference.”</p>
        </Translated>
      </Today>
      <Ask q={questionsS207['ch20/20.7/q-moe']} />
    </>
  );
}
