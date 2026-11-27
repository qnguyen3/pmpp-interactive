import { useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  freeLds,
  freePitches,
  sharedBytes,
  stripRead,
  stripWorst,
  wStore,
  wStoreWorst,
  wrongOffsetReads,
  type Accessed,
  type Banks,
  type Worst,
} from '../../sim/gemmbanks';
import { THREADS, type Layout, type Width } from '../../sim/gemmremap';

// ---------------------------------------------------------------- the bank explorer

type Mode = 'A' | 'B' | 'W-scalar' | 'W-float4';
// Getters: the labels are read in the selected language when the buttons render.
const MODE_LABEL: Record<Mode, string> = {
  get A() {
    return tr('A strip', 'dải A');
  },
  get B() {
    return tr('B strip', 'dải B');
  },
  get 'W-scalar'() {
    return tr('W tile, scalar loads', 'ô W, lệnh nạp vô hướng');
  },
  get 'W-float4'() {
    return tr('W tile, float4 loads', 'ô W, lệnh nạp float4');
  },
};
const MODE_TITLE: Record<Mode, string> = {
  get A() {
    return tr('one register of the A strip, read from A_s', 'một thanh ghi của dải A, đọc từ A_s');
  },
  get B() {
    return tr('one register of the B strip, read from B_s', 'một thanh ghi của dải B, đọc từ B_s');
  },
  get 'W-scalar'() {
    return tr("the project's W tile stored transposed into B_s, loaded with scalar loads", 'ô W của dự án được lưu chuyển vị vào B_s, nạp bằng lệnh nạp vô hướng');
  },
  get 'W-float4'() {
    return tr("the project's W tile stored transposed into B_s, loaded with float4 loads", 'ô W của dự án được lưu chuyển vị vào B_s, nạp bằng lệnh nạp float4');
  },
};

interface St {
  mode: Mode;
  layout: Layout;
  ld: number;
  pitch: number;
  banks: Banks;
  W: Width;
  warp: number;
  i: number;
  p: number;
  bank: number | null;
  showFree: boolean;
}
const START: St = { mode: 'A', layout: 'remap', ld: 8, pitch: 128, banks: 32, W: 32, warp: 0, i: 0, p: 0, bank: null, showFree: false };

/** "0–3, 8–11": a sorted list of lanes as ranges. */
function ranges(xs: number[]): string {
  const out: string[] = [];
  for (let k = 0; k < xs.length; k++) {
    let j = k;
    while (j + 1 < xs.length && xs[j + 1] === xs[j] + 1) j++;
    out.push(j > k ? `${xs[k]}–${xs[j]}` : String(xs[k]));
    k = j;
  }
  return out.join(', ');
}

const regLabel = (s: St, p: number) => {
  if (s.mode === 'W-scalar') return tr(`sub-tile s = ${p}`, `ô con s = ${p}`);
  if (s.mode === 'W-float4') return tr(`value j = ${p}`, `giá trị j = ${p}`);
  const name = s.mode === 'A' ? 'a_r' : 'b_r';
  if (s.layout === 'coarse')
    return tr(`${name}[${p}]: ${s.mode === 'A' ? 'row tRow' : 'column tCol'} + ${p}`, `${name}[${p}]: ${s.mode === 'A' ? 'hàng tRow' : 'cột tCol'} + ${p}`);
  return tr(
    `${name}[${p}]: quadrant ${s.mode === 'A' ? 'row' : 'column'} ${p >> 2}, ${s.mode === 'A' ? 'row' : 'column'} ${p % 4}`,
    `${name}[${p}]: ${s.mode === 'A' ? 'hàng' : 'cột'} góc phần tư ${p >> 2}, ${s.mode === 'A' ? 'hàng' : 'cột'} ${p % 4}`,
  );
};

const nWay = (d: number) => (d <= 1 ? tr('no conflict', 'không xung đột') : tr(`${d}-way conflict`, `xung đột ${d} chiều`));

function compute(s: St): { acc: Accessed; worst: Worst } {
  if (s.mode === 'W-scalar' || s.mode === 'W-float4') {
    const mapping = s.mode === 'W-float4' ? 'float4' : 'scalar';
    return {
      acc: wStore({ mapping, pitch: s.pitch, banks: s.banks, W: s.W, warp: s.warp, step: s.p }),
      worst: wStoreWorst({ mapping, pitch: s.pitch, banks: s.banks, W: s.W }),
    };
  }
  const strip = { layout: s.layout, strip: s.mode, ld: s.ld, banks: s.banks, W: s.W } as const;
  return { acc: stripRead({ ...strip, warp: s.warp, i: s.i, p: s.p }), worst: stripWorst(strip) };
}

function Histogram({ acc, banks, selected, onPick }: { acc: Accessed; banks: number; selected: number; onPick: (b: number) => void }) {
  const L = 22;
  const cw = banks > 32 ? 8.5 : 14;
  const W = L + banks * cw + 6;
  const top = Math.max(1, ...acc.perBank.map((b) => b.words.length));
  const most = Math.max(4, top); // a fixed minimum height keeps the panel from jumping
  const unit = Math.min(16, 132 / most);
  const base = 6 + most * unit;
  const every = banks > 32 ? 8 : 2;
  return (
    <>
      <p className="ui muted" style={{ margin: '0 0 0.3rem', fontSize: '0.78rem' }}>
        <Translated vi="Số word khác nhau mỗi bank, mỗi word một hộp. Bấm vào một bank để xem các lane của nó.">
          Distinct words per bank, one box each. Click a bank to see its lanes.
        </Translated>
      </p>
      <svg viewBox={`0 0 ${W} ${base + 30}`} role="img" aria-label={tr('Distinct words per bank', 'Số word khác nhau mỗi bank')} style={{ width: '100%', maxWidth: W * 1.35, display: 'block' }}
        data-degree={acc.degree} data-banks={banks}>
        <text x={L - 5} y={base + 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>0</text>
        <text x={L - 5} y={base - top * unit + 9} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{top}</text>
        {acc.perBank.map((b) => {
          const x = L + b.bank * cw;
          const k = b.words.length;
          return (
            <g key={b.bank} data-bank={b.bank} data-words={k} data-lanes={b.lanes.length} onClick={() => onPick(b.bank)} style={{ cursor: 'pointer' }}>
              <rect x={x} y={0} width={cw} height={base + 16} fill="transparent" />
              {b.words.map((w, j) => (
                <rect key={w} x={x + 1} y={base - (j + 1) * unit + 1} width={Math.max(1, cw - 2)} height={unit - 2} rx="1.5"
                  fill={k > 1 ? 'var(--fail)' : 'var(--ok)'} />
              ))}
              <line x1={x + 1} x2={x + cw - 1} y1={base + 1} y2={base + 1} stroke={k ? 'var(--ink-2)' : 'var(--rule)'} />
              {b.bank === selected ? (
                <rect x={x} y={base - top * unit - 2} width={cw} height={top * unit + 6} fill="none" stroke="var(--ink)" strokeWidth="1.5" />
              ) : null}
              {b.bank % every === 0 ? (
                <text x={x + cw / 2} y={base + 15} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{b.bank}</text>
              ) : null}
            </g>
          );
        })}
        <text x={L} y={base + 28} fontSize="10" style={{ fill: 'var(--ink-3)' }}>{tr('bank', 'bank')}</text>
      </svg>
    </>
  );
}

function BankExplorer() {
  const [s, setS] = useState<St>(START);
  const set = (o: Partial<St>) => setS((x) => ({ ...x, bank: null, ...o }));
  const isW = s.mode === 'W-scalar' || s.mode === 'W-float4';
  const steps = isW ? 4 : 8;
  const warps = THREADS / s.W;
  const st: St = { ...s, warp: Math.min(s.warp, warps - 1), p: Math.min(s.p, steps - 1) };
  const { acc, worst } = compute(st);
  const busiest = acc.perBank.reduce((a, b) => (b.words.length > a.words.length ? b : a), acc.perBank[0]);
  const sel = acc.perBank[st.bank ?? busiest.bank];
  const distinct = new Set(acc.words.map((w) => w.word)).size;
  const free = st.showFree
    ? isW
      ? freePitches(st.mode === 'W-float4' ? 'float4' : 'scalar', st.banks, st.W)
      : st.mode === 'A'
        ? freeLds(st.layout, st.banks, st.W)
        : null
    : null;
  const arr = st.mode === 'A' ? 'A_s' : 'B_s';
  return (
    <Bench
      title={tr('Banks for one strip read', 'Các bank cho một lần đọc dải')}
      task={tr(
        "Find every leading dimension that removes the A-strip conflict, then check B in both layouts, then the project's W tile.",
        'Tìm mọi chiều đầu (leading dimension) loại bỏ được xung đột của dải A, rồi kiểm tra B ở cả hai cách bố trí, rồi đến ô W của dự án.',
      )}
      foot={
        <Translated
          vi={
            <>
              bank = chỉ số word 32 bit mod số bank. Đỏ: một bank phải trả về nhiều hơn một word khác nhau, lần lượt từng cái. Các
              lane đọc cùng một word thì dùng chung nó (một lần phát quảng bá, broadcast). 64 bank dùng cùng quy tắc với số bank của
              AMD: một mô hình, không phải chi phí AMD công bố.
            </>
          }
        >
          <>
            bank = 32-bit word index mod the number of banks. Red: a bank that must deliver more than one distinct word, one
            after another. Lanes reading the same word share it (a broadcast). 64 banks uses the same rule on AMD's bank count:
            a model, not AMD's documented cost.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <span className="seg" role="group" aria-label={tr('Access', 'Kiểu truy cập')}>
          {(Object.keys(MODE_LABEL) as Mode[]).map((m) => (
            <button key={m} aria-pressed={m === s.mode} onClick={() => set({ mode: m })} title={MODE_TITLE[m]}>{MODE_LABEL[m]}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        {!isW ? (
          <label className="field">
            <Translated vi="Cách bố trí luồng">Thread layout</Translated>
            <select value={s.layout} onChange={(e) => set({ layout: e.target.value as Layout })}>
              <option value="coarse">{tr('§15.3/§15.4: one 8 × 8 per thread', '§15.3/§15.4: mỗi luồng một ô 8 × 8')}</option>
              <option value="remap">{tr('§15.5: four 4 × 4 per thread', '§15.5: mỗi luồng bốn ô 4 × 4')}</option>
            </select>
          </label>
        ) : null}
        {s.mode === 'A' ? (
          <label className="field">
            <Translated vi={<>Chiều đầu của A_s = {s.ld}</>}>A_s leading dimension = {s.ld}</Translated>
            <input type="range" min={8} max={17} value={s.ld} aria-label={tr('A_s leading dimension', 'chiều đầu của A_s')} onChange={(e) => set({ ld: Number(e.target.value) })} />
          </label>
        ) : null}
        {s.mode === 'B' ? <span className="muted">
            <Translated vi="Các hàng của B_s vẫn là 128 float (p. 363).">B_s rows stay 128 floats (p. 363).</Translated>
          </span> : null}
        {isW ? (
          <label className="field">
            <Translated vi={<>Pitch của B_s = {s.pitch} float</>}>B_s pitch = {s.pitch} floats</Translated>
            <input type="range" min={128} max={148} value={s.pitch} aria-label={tr('B_s pitch', 'pitch của B_s')} onChange={(e) => set({ pitch: Number(e.target.value) })} />
          </label>
        ) : null}
        <label className="field">
          <Translated vi="Số bank">Banks</Translated>
          <select value={s.banks} onChange={(e) => set({ banks: Number(e.target.value) as Banks })}>
            <option value={32}>32 (NVIDIA; AMD CDNA 1–3){/* i18n-ignore: product names and numbers */}</option>
            <option value={64}>{tr('64 (AMD CDNA 4, RDNA 2–4): a model', '64 (AMD CDNA 4, RDNA 2–4): một mô hình')}</option>
          </select>
        </label>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={s.W} onChange={(e) => set({ W: Number(e.target.value) as Width })}>
            <option value={32}>{tr('32 lanes', '32 lane')}</option>
            <option value={64}>{tr('64 lanes (AMD CDNA)', '64 lane (AMD CDNA)')}</option>
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Warp">Warp</Translated>
          <select value={st.warp} onChange={(e) => set({ warp: Number(e.target.value) })}>
            {Array.from({ length: warps }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        {!isW ? (
          <label className="field">
            <Translated vi="Bước i">Step i</Translated>
            <select value={st.i} onChange={(e) => set({ i: Number(e.target.value) })}>
              {Array.from({ length: 8 }, (_, k) => <option key={k} value={k}>{k}</option>)}
            </select>
          </label>
        ) : null}
        <label className="field">
          {isW ? tr('Store', 'Lần ghi') : tr('Register', 'Thanh ghi')}
          <select value={st.p} onChange={(e) => set({ p: Number(e.target.value) })}>
            {Array.from({ length: steps }, (_, k) => <option key={k} value={k}>{regLabel(st, k)}</option>)}
          </select>
        </label>
        <button className="quiet" onClick={() => set({ ...START })}>
          {tr('Fig. 15.11(a)', 'Hình 15.11(a)')}
        </button>
        <button className="quiet" onClick={() => set({ ...START, ld: 9 })}>
          {tr('Fig. 15.11(b)', 'Hình 15.11(b)')}
        </button>
      </div>

      <Histogram acc={acc} banks={st.banks} selected={sel.bank} onPick={(b) => setS((x) => ({ ...x, bank: b }))} />

      <p className="ui" style={{ margin: '0.5rem 0 0.3rem' }} data-this={acc.degree}>
        <Translated
          vi={
            <>
              Lệnh này: <b className={acc.degree > 1 ? 't-fail' : 't-ok'}>{nWay(acc.degree)}</b>. {st.W} lane đọc{' '}
              {distinct} word khác nhau; bank bận nhất, {busiest.bank}, phải trả về {busiest.words.length}.
            </>
          }
        >
          This instruction: <b className={acc.degree > 1 ? 't-fail' : 't-ok'}>{nWay(acc.degree)}</b>. {st.W} lanes read{' '}
          {distinct} distinct words; the busiest bank, {busiest.bank}, must deliver {busiest.words.length}.
        </Translated>
      </p>
      <p className="ui" style={{ margin: '0 0 0.3rem' }} data-worst={worst.degree}>
        <Translated
          vi={
            <>
              Tệ nhất trên mọi truy cập loại này (cả {warps} warp{isW ? ', cả 4 lần ghi' : ', i = 0 đến 7, cả 8 thanh ghi'}):{' '}
              <b className={worst.degree > 1 ? 't-fail' : 't-ok'}>{nWay(worst.degree)}</b>.
            </>
          }
        >
          Worst over every access of this kind (all {warps} warps{isW ? ', all 4 stores' : ', i = 0 to 7, all 8 registers'}):{' '}
          <b className={worst.degree > 1 ? 't-fail' : 't-ok'}>{nWay(worst.degree)}</b>.
        </Translated>
      </p>
      <p className="ui muted" style={{ margin: '0 0 0.3rem' }} data-selected={sel.bank}>
        <Translated vi={<>Bank {sel.bank}:</>}>Bank {sel.bank}:</Translated>{' '}
        {sel.words.length === 0
          ? tr('no lane.', 'không có lane nào.')
          : sel.words.map((w, k) => {
              const ls = acc.words.filter((x) => x.word === w);
              const at = isW ? `B_s[${ls[0].row}][${ls[0].col}]` : `${arr}[${ls[0].row}][${ls[0].col}]`;
              return (
                <span key={w}>
                  <Translated vi={<>{k ? '; ' : ''}word {w} = {at}, lane {ranges(ls.map((x) => x.lane))}</>}>
                    {k ? '; ' : ''}word {w} = {at}, lane{ls.length > 1 ? 's' : ''} {ranges(ls.map((x) => x.lane))}
                  </Translated>
                </span>
              );
            })}
        .
      </p>
      {free ? (
        <p className="ui" style={{ margin: 0 }} data-free={free.join(',')}>
          <Translated
            vi={
              <>
                {isW ? 'Các pitch không xung đột từ 128 đến 148' : 'Các chiều đầu không xung đột từ 8 đến 17'} trong thiết lập này:{' '}
                <b>{free.length ? free.join(', ') : 'không có'}</b>.
              </>
            }
          >
            {isW ? 'Conflict-free pitches from 128 to 148' : 'Conflict-free leading dimensions from 8 to 17'} in this setting:{' '}
            <b>{free.length ? free.join(', ') : 'none'}</b>.
          </Translated>
        </p>
      ) : s.mode !== 'B' ? (
        <button className="quiet" onClick={() => setS((x) => ({ ...x, showFree: true }))}>
          <Translated vi={<>Liệt kê mọi {isW ? 'pitch' : 'chiều đầu'} không xung đột cho thiết lập này</>}>
            List every conflict-free {isW ? 'pitch' : 'leading dimension'} for this setting
          </Translated>
        </button>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the erratum hunt

// Lines 18 to 36 of labs/ch15/snippets/s156_padded_offset.cu (front-end-checked);
// S156.test.ts fails if this copy drifts from the file.
const PADDED_FIRST_LINE = 18;
const PADDED = `__global__ void mm_tiled_padded(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
    unsigned int bRow = blockIdx.y*bM;
    unsigned int bCol = blockIdx.x*bN;
    unsigned int tRow = (threadIdx.x/(bN/tN))*tM;
    unsigned int tCol = (threadIdx.x%(bN/tN))*tN;
    float C_r[tM][tN] = {};
    for (unsigned int tile = 0; tile < (K + bK - 1)/bK; ++tile) {
        __shared__ float A_s[bM*(bK + 1)];
        __shared__ float B_s[bK*bN];
        loadTile(&A[bRow*K + tile*bK], K, M - bRow, K - tile*bK, &A_s[0], bK + 1, bM, bK);
        loadTile(&B[tile*bK*N + bCol], N, K - tile*bK, N - bCol, &B_s[0], bN, bK, bN);
        __syncthreads();
        mm(tM, tN, bK, &A_s[tRow*bK], bK + 1, &B_s[tCol], bN, C_r);
        __syncthreads();
    }
    unsigned int maxRow = (bRow + tRow < M) ? (M - (bRow + tRow)) : 0;
    unsigned int maxCol = (bCol + tCol < N) ? (N - (bCol + tCol)) : 0;
    writeTile(&C[(bRow + tRow)*N + bCol + tCol], N, maxRow, maxCol, C_r, tM, tN);
}`;
/** File line numbers the hunt refers to. */
const LINE_DECL = 25;
const LINE_LOAD_A = 27;
const LINE_MM = 30;
const TIERS: Record<string, Tier> = {
  A: 'global', B: 'global', C: 'global', A_s: 'shared', B_s: 'shared', C_r: 'reg',
  bRow: 'reg', bCol: 'reg', tRow: 'reg', tCol: 'reg', tile: 'reg', maxRow: 'reg', maxCol: 'reg',
};

function MisReads() {
  const [ty, setTy] = useState(1);
  const reads = wrongOffsetReads(ty);
  const tRow = ty * 8;
  const lo = Math.min(...reads.map((m) => m.row), tRow);
  const hi = Math.max(...reads.map((m) => m.row), tRow + 7);
  const C = 22;
  const X0 = 64;
  const right = reads.filter((m) => m.right).length;
  const pad = reads.filter((m) => m.padding).length;
  const read = new Map(reads.map((m) => [m.row * 9 + m.col, m]));
  return (
    <Bench
      title={tr('What a thread row reads with the old offset', 'Một hàng luồng đọc gì với offset cũ')}
      task={tr('Pick a thread row. Compare the cells it should read with the cells it does read.', 'Chọn một hàng luồng. So sánh các ô nó lẽ ra phải đọc với các ô nó thực sự đọc.')}
      foot={
        <Translated
          vi={
            <>
              <code>A_s</code> có phần đệm, 9 float mỗi hàng; cột 8 là phần đệm, thứ mà <code>loadTile</code> không bao giờ ghi. Viền:
              các hàng mà hàng luồng ty sở hữu (tRow đến tRow + 7, cột 0 đến 7). Tô màu: 64 word nó đọc khi{' '}
              <code>mm</code> nhận <code>&amp;A_s[tRow*bK]</code> với chiều đầu bK + 1.
            </>
          }
        >
          <>
            The padded <code>A_s</code>, 9 floats per row; column 8 is the padding, which <code>loadTile</code> never writes.
            Outlined: the rows thread row ty owns (tRow to tRow + 7, columns 0 to 7). Filled: the 64 words it reads when{' '}
            <code>mm</code> gets <code>&amp;A_s[tRow*bK]</code> with leading dimension bK + 1.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Hàng luồng ty (tRow = 8·ty)">Thread row ty (tRow = 8·ty)</Translated>
          <select value={ty} onChange={(e) => setTy(Number(e.target.value))}>
            {Array.from({ length: 16 }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
      </div>
      <p className="narrate" data-misread={`${reads[0].row},${reads[0].col}`} data-right={right} data-padding={pad} style={{ marginBottom: '0.7rem' }}>
        {ty === 0 ? (
          <Translated vi={<>Hàng luồng 0 có tRow = 0, nên tRow·bK và tRow·(bK + 1) đều bằng 0: các luồng này đọc đúng word. Lỗi này ẩn khỏi mọi phép kiểm thử chỉ kiểm tra 8 hàng đầu của mỗi khối.</>}>
            <>Thread row 0 has tRow = 0, so tRow·bK and tRow·(bK + 1) are both 0: these threads read the right words. The bug hides from any test that only checks the first 8 rows of each block.</>
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                Lẽ ra nó phải bắt đầu ở word tRow·9 = {tRow * 9}, hàng {tRow}, cột 0. Nó bắt đầu ở word tRow·8 = {reads[0].word}:
                hàng {reads[0].row}, cột {reads[0].col}, sớm {tRow * 9 - reads[0].word} word. Đúng: {right} trên 64 lần đọc; từ cột
                đệm: {pad}.
              </>
            }
          >
            <>
              It should start at word tRow·9 = {tRow * 9}, row {tRow}, column 0. It starts at word tRow·8 = {reads[0].word}:
              row {reads[0].row}, column {reads[0].col}, {tRow * 9 - reads[0].word} words early. Right: {right} of 64 reads;
              from the padding column: {pad}.
            </>
          </Translated>
        )}
      </p>
      <svg viewBox={`0 0 ${X0 + 9 * C + 120} ${(hi - lo + 1) * C + 26}`} role="img" aria-label={tr('Padded A_s rows and the words read', 'Các hàng A_s có phần đệm và các word được đọc')}
        style={{ width: '100%', maxWidth: 420, display: 'block' }}>
        {Array.from({ length: 9 }, (_, c) => (
          <text key={c} x={X0 + c * C + C / 2} y={12} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{c === 8 ? tr('pad', 'đệm') : c}</text>
        ))}
        {Array.from({ length: hi - lo + 1 }, (_, k) => lo + k).map((r) => (
          <g key={r}>
            <text x={X0 - 6} y={18 + (r - lo) * C + C / 2 + 4} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>hàng {r}</>}>row {r}</Translated></text>
            {Array.from({ length: 9 }, (_, c) => {
              const m = read.get(r * 9 + c);
              const own = r >= tRow && r < tRow + 8 && c < 8;
              return (
                <rect key={c} x={X0 + c * C + 1} y={18 + (r - lo) * C + 1} width={C - 2} height={C - 2} rx="2"
                  data-cell={`${r},${c}`} data-read={m ? (m.right ? 'right' : 'wrong') : ''}
                  fill={m ? (m.right ? 'var(--ok)' : 'var(--fail)') : c === 8 ? 'var(--bench-grid)' : 'var(--cell)'}
                  stroke={own ? 'var(--ok)' : 'var(--rule)'} strokeWidth={own ? 2 : 1} />
              );
            })}
          </g>
        ))}
        <text x={X0 + 9 * C + 10} y={30} fontSize="10" style={{ fill: 'var(--ok)' }}>{tr('outlined: should read', 'viền: lẽ ra phải đọc')}</text>
        <text x={X0 + 9 * C + 10} y={46} fontSize="10" style={{ fill: 'var(--fail)' }}>{tr('red: wrong word read', 'đỏ: đọc sai word')}</text>
        <text x={X0 + 9 * C + 10} y={62} fontSize="10" style={{ fill: 'var(--ok)' }}>{tr('green: right word', 'xanh: đúng word')}</text>
      </svg>
    </Bench>
  );
}

function ErratumHunt() {
  const { progress } = useProgress();
  const [done, setDone] = useState(false);
  const answered = done || !!progress.answers['ch15/15.6/q-offset'];
  return (
    <>
      <Code
        tabs={[
          {
            label: tr('Fig. 15.3 with the change of pp. 362–363', 'Hình 15.3 với thay đổi ở pp. 362–363'),
            code: PADDED,
            status: tr('front-end-checked; one line deliberately left as printed', 'front-end-checked; một dòng được cố ý để nguyên như bản in'),
            tiers: TIERS,
          },
        ]}
        firstLine={PADDED_FIRST_LINE}
        highlight={answered ? [LINE_MM] : [LINE_DECL, LINE_LOAD_A, LINE_MM]}
        caption={
          <Translated
            vi={
              <>
                Từ <code>labs/ch15/snippets/s156_padded_offset.cu</code>. Được tô sáng
                {answered ? ': dòng vẫn còn phải thay đổi.' : <>: ba dòng nhắc đến kích thước hoặc chiều đầu của A_s.</>}{' '}
                Đừng chép vòng lặp này. Lời giải của lab 15.4 đặt tên cho chiều đầu có phần đệm, <code>LDA_S</code>, và dùng
                nó trong mọi chỉ số vào <code>A_s</code>.
              </>
            }
          >
            <>
              From <code>labs/ch15/snippets/s156_padded_offset.cu</code>. Highlighted
              {answered ? ': the line that still has to change.' : <>: the three lines that mention A_s's size or leading dimension.</>}{' '}
              Do not copy this loop. Lab 15.4's solution gives the padded leading dimension a name, <code>LDA_S</code>, and
              uses it in every index into <code>A_s</code>.
            </>
          </Translated>
        }
      />
      <Ask q={questionsS156['ch15/15.6/q-offset']} label={tr('Erratum hunt: what else must change?', 'Săn errata: còn gì phải thay đổi?')} onResult={() => setDone(true)} />
      {answered ? <MisReads /> : null}
    </>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS156: Record<string, Question> = {
  'ch15/15.6/q-why-odd': {
    kind: 'choice',
    id: 'ch15/15.6/q-why-odd',
    prompt: (
      <Translated vi={<>Trong cách bố trí §15.5, vì sao mọi chiều đầu lẻ đều loại bỏ xung đột của dải A trên 32 bank?</>}>
        <>In the §15.5 layout, why does any odd leading dimension remove the A-strip conflict on 32 banks?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Một số lẻ không có ước chung với 32, nên các hàng liên tiếp rơi vào các bank khác nhau."}>{'An odd number has no common factor with 32, so consecutive rows fall in different banks.'}</Translated>,
        why: <Translated vi={"Đúng với các hàng liên tiếp, nhưng các hàng lane đọc các hàng cách nhau 4, không phải 1. Lập luận phải xoay quanh 4·ld."}>{'True of consecutive rows, but the lane rows read rows 4 apart, not 1 apart. The argument has to be about 4·ld.'}</Translated>,
      },
      {
        label: <Translated vi={"Các hàng lane đọc các hàng cách nhau 4; với ld lẻ, 4g·ld mod 32 nhận tám giá trị khác nhau với g = 0 đến 7. Một ld chẵn ghép chúng thành từng cặp."}>{'Lane rows read rows 4 apart; for odd ld, 4g·ld mod 32 takes eight distinct values for g = 0 to 7. An even ld pairs them up.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Chiều đầu lẻ đặt các word của 32 lane vào 32 bank khác nhau, mỗi lane một bank."}>{'Odd leading dimensions put the 32 lanes\' words in 32 different banks, one per lane.'}</Translated>,
        why: <Translated vi={"32 lane chỉ đọc 8 word khác nhau (bốn lane mỗi word): chỉ cần 8 word đó nằm ở các bank khác nhau là đủ."}>{'The 32 lanes read only 8 distinct words (four lanes per word): it is enough that those 8 are in different banks.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hàng lane g đọc hàng 4g + (offset cố định), nên word của nó là 4g·ld + (cố định). Bank giữa các hàng lane chỉ khác nhau
            qua 4g·ld mod 32. Với ld lẻ, 4·ld là 4 lần một số lẻ: khi g chạy từ 0 đến 7, 4g·ld mod 32 chạm tám bội số khác nhau của
            4. Với ld chẵn, 4·ld là bội số của 8, và 8 bội số của 8 chỉ vừa trong 4 bank: hai hàng lane dùng chung mỗi bank, ít nhất
            2 chiều. Nút "liệt kê mọi chiều đầu không xung đột" của trình khám phá xác nhận điều đó.
          </p>
        }
      >
        <p>
          Lane row g reads row 4g + (fixed offset), so its word is 4g·ld + (fixed). The banks differ between lane rows only
          through 4g·ld mod 32. For odd ld, 4·ld is 4 times an odd number: as g runs over 0 to 7, 4g·ld mod 32 hits eight
          different multiples of 4. For even ld, 4·ld is a multiple of 8, and 8 multiples of 8 fit in only 4 banks: two
          lane rows share each, at least 2-way. The explorer's "list every conflict-free leading dimension" confirms it.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.6, pp. 361–362; the general rule is ours (notes [inferred], ch15-facts E1)',
    vi: { source: "ch. 15 §15.6, pp. 361–362; quy tắc tổng quát là của khóa học này (notes [suy ra], ch15-facts E1)" },
  },
  'ch15/15.6/q-64banks': {
    kind: 'choice',
    id: 'ch15/15.6/q-64banks',
    prompt: (
      <Translated vi={<>Trên mô hình 64 bank, chiều đầu nào để lại xung đột trong lần đọc dải A của §15.5: 10 hay 12?</>}>
        <>On the 64-bank model, which leading dimension leaves a conflict in the §15.5 A-strip read: 10 or 12?</>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"10 vẫn là 2 chiều, như trên 32 bank; 12 không xung đột."}>{'10 is still 2-way, as on 32 banks; 12 is free.'}</Translated>,
        why: <Translated vi={"Điều đó gần với bức tranh 32 bank hơn, nơi 10 là 2 chiều và 12 là 4 chiều."}>{'That is closer to the 32-bank picture, where 10 is 2-way and 12 is 4-way.'}</Translated>,
      },
      {
        label: <Translated vi={"12 là 2 chiều; 10 không xung đột."}>{'12 is 2-way; 10 is free.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Cả hai đều không xung đột: 64 bank là đủ cho mọi chiều đầu chẵn."}>{'Both are free: 64 banks are enough for any even leading dimension.'}</Translated>,
        why: <Translated vi={"Thử 12 trong trình khám phá với 64 bank."}>{'Try 12 in the explorer with 64 banks.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với 64 bank, word của các hàng lane khác nhau theo 4g·ld mod 64. Với ld = 10 đó là 40g mod 64: 0, 40, 16, 56, 32, 8, 48,
            24, tám bank khác nhau. Với ld = 12 đó là 48g mod 64: 0, 48, 32, 16, rồi lại bốn giá trị đó: 2 chiều. Trên 32 bank cả
            hai đều xung đột: 10 là 2 chiều và 12 là 4 chiều. Quy tắc trên 64 bank là "ld không chia hết cho 4". AMD không công bố
            cách một truy cập 64 lane được kiểm tra, nên đây là một mô hình.
          </p>
        }
      >
        <p>
          With 64 banks the lane rows' words differ by 4g·ld mod 64. For ld = 10 that is 40g mod 64: 0, 40, 16, 56, 32, 8, 48,
          24, eight different banks. For ld = 12 it is 48g mod 64: 0, 48, 32, 16, then the same four again: 2-way. On 32
          banks both conflict: 10 is 2-way and 12 is 4-way. The rule on
          64 banks is "ld not divisible by 4". AMD does not document how a 64-lane access is checked, so this is a model.
        </p>
      </Translated>
    ),
    source: 'ours (notes/today/ch15-facts.md E1; bank counts and the caveat from B2)',
    vi: { source: "của khóa học này (notes/today/ch15-facts.md E1; số bank và lưu ý từ B2)" },
  },
  'ch15/15.6/q-early-pad': {
    kind: 'choice',
    id: 'ch15/15.6/q-early-pad',
    prompt: (
      <Translated
        vi={
          <>
            Bạn đệm <code>A_s</code> trong kernel phân ô thanh ghi của §15.4, trước khi làm phép ánh xạ lại của §15.5. Bạn kỳ vọng
            điều gì trên một GPU NVIDIA?
          </>
        }
      >
        <>
          You pad <code>A_s</code> in the register-tiled kernel of §15.4, before doing the §15.5 remap. What do you expect on
          an NVIDIA GPU?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Cùng mức lợi như sau khi ánh xạ lại: xung đột 8 chiều trên dải A biến mất."}>{'The same gain as after the remap: the 8-way conflict on the A strip disappears.'}</Translated>,
        why: <Translated vi={"Xung đột 8 chiều thuộc về cách bố trí đã ánh xạ lại. Trước khi ánh xạ lại, lần đọc dải A là 2 chiều."}>{'The 8-way conflict belongs to the remapped layout. Before the remap the A-strip read was 2-way.'}</Translated>,
      },
      {
        label: <Translated vi={"Ít thôi: lần đọc dải A ở đó chỉ là 2 chiều, và lần đọc dải B 4 chiều không bị đụng tới."}>{'Little: the A-strip read there was only 2-way, and the 4-way B-strip read is untouched.'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Chậm đi: đệm A_s làm lệch các lần nạp toàn cục lấp đầy nó, nên chúng không còn được hợp nhất."}>{'A slowdown: padding A_s misaligns the global loads that fill it, so they no longer coalesce.'}</Translated>,
        why: <Translated vi={"Phần đệm nằm trong bộ nhớ chia sẻ; các lần nạp toàn cục không đổi."}>{'The padding is in shared memory; global loads are unchanged.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong cách bố trí §15.3/§15.4, một warp là hai hàng 16 luồng với ô 8 × 8. Lần đọc dải A của nó chạm 2 hàng cách nhau 8: 2
            chiều ở ld 8, được loại bỏ bằng phần đệm. Lần đọc dải B của nó chạm 16 word khác nhau cách nhau 8 float, rơi vào 4 bank:
            4 chiều, và đệm A không giúp gì cho nó. "B không cần thay đổi" chỉ đúng sau khi ánh xạ lại. Đó là lý do các lab đệm sau
            lab 15.3, không phải trước.
          </p>
        }
      >
        <p>
          In the §15.3/§15.4 layout a warp is two rows of 16 threads with 8 × 8 tiles. Its A-strip read touches 2 rows 8
          apart: 2-way at ld 8, removed by padding. Its B-strip read touches 16 distinct words 8 floats apart, which land in 4
          banks: 4-way, and padding A does nothing for it. "B needs no change" holds only after the remap. That is why the labs
          pad after lab 15.3, not before.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch15-facts.md E1 (our arithmetic with NVIDIA’s rule); book p. 363',
    vi: { source: "notes/today/ch15-facts.md E1 (phép tính của khóa học này với quy tắc của NVIDIA); sách p. 363" },
  },
  'ch15/15.6/q-offset': {
    kind: 'choice',
    id: 'ch15/15.6/q-offset',
    prompt: (
      <Translated
        vi={
          <>
            Đoạn mã ở trên thực hiện đúng hai chỉnh sửa của sách: <code>A_s[bM*(bK + 1)]</code>, và bK + 1 làm chiều đầu của A_s
            trong <code>loadTile</code> và <code>mm</code>. Còn gì phải thay đổi để kernel đúng?
          </>
        }
      >
        <>
          The listing above makes exactly the book's two edits: <code>A_s[bM*(bK + 1)]</code>, and bK + 1 as A_s's leading
          dimension in <code>loadTile</code> and <code>mm</code>. What else must change for the kernel to be correct?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"Dòng 27: đối số cuối của lời gọi loadTile cho A, bK, phải thành bK + 1."}>{'Line 27: the last argument of the A loadTile call, bK, must become bK + 1.'}</Translated>,
        why: <Translated vi={"Đối số đó là chiều rộng của ô, vẫn là 8 float. Chỉ chiều đầu của nó thay đổi, và đối số đó đã ghi bK + 1."}>{'That argument is the tile’s width, still 8 floats. Only its leading dimension changed, and that argument already says bK + 1.'}</Translated>,
      },
      {
        label: <Translated vi={"Dòng 30: offset bắt đầu &A_s[tRow*bK] phải thành &A_s[tRow*(bK + 1)]."}>{'Line 30: the start offset &A_s[tRow*bK] must become &A_s[tRow*(bK + 1)].'}</Translated>,
        correct: true,
      },
      {
        label: <Translated vi={"Dòng 26: B_s cũng cần một cột đệm."}>{'Line 26: B_s needs a padding column too.'}</Translated>,
        why: <Translated vi={"B_s không bị chỉnh sửa đụng tới, nên không có gì về nó trở nên sai. B có xung đột hay không là chuyện tốc độ, và trong cách bố trí này đệm một hàng cũng chẳng sửa được."}>{'B_s is untouched by the edit, so nothing about it became wrong. Whether B conflicts is a speed question, and in this layout padding a row would not fix it anyway.'}</Translated>,
      },
      {
        label: <Translated vi={"Không gì cả: hai chỉnh sửa là đủ."}>{'Nothing: the two edits are complete.'}</Translated>,
        why: <Translated vi={"Khi đó mọi hàng luồng trừ hàng đầu tiên đều đọc sai hàng của A_s; xem bên dưới."}>{'Then every thread row except the first reads the wrong rows of A_s; see below.'}</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>mm</code> nhận một con trỏ tới phần tử đầu tiên của các hàng của luồng, và điểm bắt đầu của hàng tRow phụ thuộc
            vào chiều đầu: tRow·9 khi có phần đệm, không phải tRow·8. Để nguyên như bản in, hàng luồng ty bắt đầu sớm 8·ty word.
            Hàng 0 không bị ảnh hưởng, đó là lý do một phép kiểm thử nhỏ có thể qua. Sách thay đổi khai báo và các đối số chiều đầu
            mà không nhắc đến dòng này.
          </p>
        }
      >
        <p>
          <code>mm</code> receives a pointer to the first element of the thread's rows, and the start of row tRow depends on
          the leading dimension: tRow·9 with the padding, not tRow·8. Left as printed, thread row ty starts 8·ty words early.
          Row 0 is unaffected, which is why a small test can pass. The book changes the declaration and the leading-dimension
          arguments and does not mention this line.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.6, pp. 362–363; notes/ch15.md, "Unclear in the PDF" item 4',
    vi: { source: "ch. 15 §15.6, pp. 362–363; notes/ch15.md, \"Unclear in the PDF\" mục 4" },
  },
  'ch15/15.6/q-bytes': {
    kind: 'numeric',
    id: 'ch15/15.6/q-bytes',
    prompt: (
      <Translated
        vi={
          <>
            <code>A_s</code> có phần đệm (128 hàng, chiều đầu 9) và <code>B_s</code> không đệm (8 × 128) cùng chiếm bao nhiêu byte
            bộ nhớ chia sẻ, với float 4 byte?
          </>
        }
      >
        <>
          How many bytes of shared memory do the padded <code>A_s</code> (128 rows, leading dimension 9) and the unpadded{' '}
          <code>B_s</code> (8 × 128) take together, with 4-byte floats?
        </>
      </Translated>
    ),
    answer: sharedBytes(9).total,
    unit: 'bytes',
    vi: { unit: 'byte', source: 'ch. 15 §15.6 và §15.7, p. 363; phép cộng là của khóa học này (notes [suy ra], ch15-facts E1)' },
    explain: (
      <Translated
        vi={
          <p>
            A_s: 128 × 9 × 4 = {sharedBytes(9).A} byte, tăng từ {sharedBytes(8).A}. B_s: 8 × 128 × 4 = {sharedBytes(9).B}. Cộng lại
            {sharedBytes(9).total}: rất nhỏ so với mức một khối có thể dùng, đó là lý do §15.7 thấy bộ nhớ chia sẻ không phải giới
            hạn của kernel này.
          </p>
        }
      >
        <p>
          A_s: 128 × 9 × 4 = {sharedBytes(9).A} bytes, up from {sharedBytes(8).A}. B_s: 8 × 128 × 4 = {sharedBytes(9).B}.
          Together {sharedBytes(9).total}: tiny next to what a block may use, which is why §15.7 finds shared memory is not
          this kernel's limit.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.6 and §15.7, p. 363; the sum is ours (notes [inferred], ch15-facts E1)',
  },
  'ch15/15.6/q-w-pitch': {
    kind: 'choice',
    id: 'ch15/15.6/q-w-pitch',
    prompt: (
      <Translated
        vi={
          <>
            Trong dự án, ô W được lưu chuyển vị vào <code>B_s</code>: với lệnh nạp vô hướng, các lane của một warp giữ 4 hàng × 8
            cột của W và ghi chúng xuống 8 hàng của <code>B_s</code>. Pitch nào của B_s làm lần ghi đó không xung đột trên 32 bank:
            128, 129 hay 132 float?
          </>
        }
      >
        <>
          In the project, the W tile is stored transposed into <code>B_s</code>: with scalar loads one warp's lanes hold 4
          rows × 8 columns of W and store them down 8 rows of <code>B_s</code>. Which B_s pitch makes that store
          conflict-free on 32 banks: 128, 129 or 132 floats?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi={"128, không đệm."}>{'128, with no padding.'}</Translated>,
        why: <Translated vi={"k·128 mod 32 = 0 với mọi k: 8 hàng bắt đầu ở cùng một bank, nên mỗi bank trong 4 bank chứa 8 word: 8 chiều."}>{'k·128 mod 32 = 0 for every k: the 8 rows start in the same bank, so each of 4 banks holds 8 words: 8-way.'}</Translated>,
      },
      {
        label: <Translated vi={"129, một cột đệm, như với A_s."}>{'129, one padding column, as for A_s.'}</Translated>,
        why: <Translated vi={"k·129 + n mod 32 = k + n: 32 word rơi vào các bank 0 đến 10, tối đa 4 word mỗi bank."}>{'k·129 + n mod 32 = k + n: the 32 words fall in banks 0 to 10, up to 4 per bank.'}</Translated>,
      },
      {
        label: <Translated vi={"132."}>{'132.'}</Translated>,
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lane (n, k) ghi word k·pitch + n, n = 0 đến 3, k = 0 đến 7. Với pitch 132, k·132 mod 32 = 4k, nên các bank là 4k + n:
            cả 32 đều khác nhau. Một cột đệm là cách sửa đúng khi các lane đi xuống một cột từng hàng một; ở đây 4 lane nằm cạnh nhau
            trong mỗi hàng, nên mỗi hàng phải dịch đi 4 bank. 132 float là 128 cộng 16 byte, phần đệm mà MLX dùng cho các ô
            threadgroup của nó. Chế độ ô W của trình khám phá cũng cho thấy cách ánh xạ float4.
          </p>
        }
      >
        <p>
          Lane (n, k) writes word k·pitch + n, n = 0 to 3, k = 0 to 7. With pitch 132, k·132 mod 32 = 4k, so the banks are 4k
          + n: all 32 different. One padding column is the right fix when lanes go down a column one row at a time; here 4
          lanes sit side by side in each row, so each row has to move by 4 banks. 132 floats is 128 plus 16 bytes, the padding
          MLX gives its threadgroup tiles. The explorer's W-tile mode shows the float4 mapping too.
        </p>
      </Translated>
    ),
    source: 'ours (the project layout of lab 6.4; MLX’s 16-byte padding from ch15-facts C2 and ch06-facts D3)',
    vi: { source: "của khóa học này (cách bố trí dự án của lab 6.4; phần đệm 16 byte của MLX từ ch15-facts C2 và ch06-facts D3)" },
  },
};

// ---------------------------------------------------------------- the page

export function S156() {
  const first = freeLds('remap', 32, 32)[0];
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 15.5 đã sửa các lần ghi. Các lần đọc từ bộ nhớ chia sẻ (shared memory) cũng có vấn đề, và đó là vấn đề mà phép ánh xạ
            lại làm tệ hơn. Một luồng sở hữu ô đầu ra lớn đọc bộ nhớ chia sẻ với một bước nhảy (stride), và các lần đọc có bước nhảy
            có thể xung đột ở các bank (chương 6).
          </p>
        }
      >
        <p>
          Section 15.5 fixed the stores. The reads from shared memory have a problem too, and it is one the remap made worse.
          A thread that owns a large output tile reads shared memory with a stride, and strided reads can conflict in the
          banks (chapter 6).
        </p>
      </Translated>
      <Source section="15.6" pages="361" />
      <Translated
        vi={
          <p>
            Lấy cách bố trí §15.5 và dải A. Ở mỗi bước i, một luồng nạp một giá trị từ mỗi hàng mà các ô con của nó phủ. Bốn lane của
            một hàng lane sở hữu cùng các hàng, nên chúng đọc cùng một word: một lần phát quảng bá (broadcast), không phải xung đột.
            Tám hàng lane đọc các hàng cách nhau 4. Với các hàng của <code>A_s</code> dài 8 float, hàng 4g bắt đầu ở word 32g, và 32g
            mod 32 bằng 0: mọi hàng lane rơi vào bank 0. Tám word khác nhau trong một bank là xung đột bank (bank conflict) 8 chiều,
            và nó xảy ra ở mọi lần truy cập dải.
          </p>
        }
      >
        <p>
          Take the §15.5 layout and the A strip. In each step i, a thread loads one value from each of the rows its
          sub-tiles cover. The four lanes of a lane row own the same rows, so they read the same word: a broadcast, not a
          conflict. The eight lane rows read rows 4 apart. With <code>A_s</code>'s rows 8 floats long, row 4g starts at word
          32g, and 32g mod 32 is 0: every lane row lands in bank 0. Eight distinct words in one bank is an 8-way conflict, and
          it happens on every strip access.
        </p>
      </Translated>
      <Source section="15.6" pages="361–362" figs={tr('Fig. 15.11(a)', 'Hình 15.11(a)')} />

      <h2>
        <Translated vi="Các bank cho một lần đọc dải">Banks for one strip read</Translated>
      </h2>
      <Predict
        id="ch15/15.6/p-first-free"
        prompt={
          <Translated vi={<>Chiều đầu nhỏ nhất lớn hơn 8 cho A_s mà không để lại xung đột dải A trên 32 bank, trong cách bố trí §15.5, là bao nhiêu?</>}>
            <>What is the smallest leading dimension above 8 for A_s that leaves no A-strip conflict on 32 banks, in the §15.5 layout?</>
          </Translated>
        }
      >
        {(v) => (
          <Translated
            vi={
              <p>
                Đó là <b>{first}</b>
                {v === first ? ', như bạn đã nói' : ''}: một cột đệm, cách sửa của sách. Hãy kiểm tra trong trình khám phá, rồi tìm
                các giá trị khác: nút liệt kê cho bạn tất cả sau khi bạn đã đoán.
              </p>
            }
          >
            <p>
              It is <b>{first}</b>
              {v === first ? ', as you said' : ''}: one padding column, the book's fix. Check it in the explorer, then find the
              others: the list button gives them all once you have a guess.
            </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá tính các bank của một lệnh đọc dải của một warp. Nó bắt đầu ở trường hợp của sách, Hình 15.11(a). Hãy thay
            đổi chiều đầu, rồi chuyển cách bố trí và kiểu truy cập.
          </p>
        }
      >
        <p>
          The explorer computes the banks of one strip-read instruction of one warp. It starts on the book's case,
          Fig. 15.11(a). Move the leading dimension, then switch the layout and the access.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Quy tắc của NVIDIA (ch15-facts A2): bank = chỉ số word 32 bit mod 32; các word khác nhau trong một bank được phục vụ lần
              lượt; các lane trên cùng một word dùng chung nó. 64 bank là số bank của AMD, dùng với cùng quy tắc: AMD không công bố cách
              một truy cập 64 lane được kiểm tra (B2). Mỗi thanh ghi dải là một lần nạp 32 bit. Không đo thời gian.
            </p>
          }
        >
          <p>
            NVIDIA's rule (ch15-facts A2): bank = 32-bit word index mod 32; distinct words in one bank are served one after
            another; lanes on the same word share it. 64 banks is AMD's count, used with the same rule: AMD does not document
            how a 64-lane access is checked (B2). Each strip register is one 32-bit load. No timing.
          </p>
        </Translated>
      </Simplification>
      <BankExplorer />

      <h2>
        <Translated vi="Một cột đệm">One padding column</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cho <code>A_s</code> thêm một cột không ai dùng. Ô vẫn rộng 8 float, nhưng chiều đầu của nó là 9. Hàng 4g giờ bắt đầu ở
            word 36g, và 36g mod 32 là 4g: tám hàng lane rơi vào các bank 0, 4, 8, ..., 28. "Các phần tử thứ hai" của sách, ở các
            bank 1, 5, ..., 29, là cột 1 của cùng các hàng đó, tức bước i kế tiếp.
          </p>
        }
      >
        <p>
          Give <code>A_s</code> one extra column that nothing uses. The tile is still 8 floats wide, but its leading
          dimension is 9. Row 4g now starts at word 36g, and 36g mod 32 is 4g: the eight lane rows fall in banks 0, 4, 8, ...,
          28. The book's "second elements", in banks 1, 5, ..., 29, are column 1 of the same rows, the next step i.
        </p>
      </Translated>
      <Source section="15.6" pages="362" figs={tr('Fig. 15.11(b)', 'Hình 15.11(b)')} />
      <Translated
        vi={
          <p>
            Sách mời bạn kiểm tra mọi truy cập. Một dòng là đủ. Hàng lane g, thanh ghi (qr, r) và bước i đọc word (4g + 32·qr + r)·9 +
            i, mà mod 32 là 4g + 9r + i: số hạng góc phần tư biến mất, vì 32·9 là bội số của 32. Với r và i cố định, đó là tám bank
            khác nhau khi g chạy từ 0 đến 7. Với 8 thay cho 9, cùng biểu thức đó là 8r + i, cùng một bank với mọi g.
          </p>
        }
      >
        <p>
          The book invites you to check every access. One line does it. Lane row g, register (qr, r) and step i read the word
          (4g + 32·qr + r)·9 + i, which mod 32 is 4g + 9r + i: the quadrant term drops out, because 32·9 is a multiple of 32.
          For fixed r and i that is eight different banks as g runs from 0 to 7. With 8 in place of 9 the same expression is
          8r + i, the same bank for every g.
        </p>
      </Translated>
      <Beyond title={tr('Our arithmetic', 'Phép tính của khóa học này')}>
        <Translated vi={<p>Phép kiểm tra tổng quát là của khóa học này (notes, §15.6 [suy ra]); "các phần tử thứ hai" mà sách muốn nói là cách hiểu của chúng tôi (notes, Unclear 6).</p>}>
          <p>The general check is ours (notes, §15.6 [inferred]); which "second elements" the book means is our reading (notes, Unclear 6).</p>
        </Translated>
      </Beyond>
      <Ask q={questionsS156['ch15/15.6/q-why-odd']} />
      <Translated
        vi={
          <p>
            Vậy quy tắc, trên 32 bank, là mọi chiều đầu lẻ. Trên mô hình 64 bank, đó là mọi chiều đầu không chia hết cho 4. Hãy đặt
            trình khám phá ở 64 bank và thử 10 và 12.
          </p>
        }
      >
        <p>
          So the rule, on 32 banks, is any odd leading dimension. On a 64-bank model it is any leading dimension not divisible
          by 4. Set the explorer to 64 banks and try 10 and 12.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Cả hai quy tắc là của khóa học này (ch15-facts E1), tính bằng quy tắc của NVIDIA trên mỗi số bank.</p>}>
          <p>Both rules are ours (ch15-facts E1), computed with NVIDIA's rule on each bank count.</p>
        </Translated>
      </Beyond>
      <Ask q={questionsS156['ch15/15.6/q-64banks']} />

      <h2>
        <Translated vi='"B không cần thay đổi", sau khi ánh xạ lại'>"B needs no change", after the remap</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách nói <code>B_s</code> không cần gì: lần đọc dải B của một warp chạm các phần tử liên tiếp của một hàng, nằm ở các bank
            khác nhau. Trong cách bố trí §15.5 điều đó đúng, và bức tranh còn tốt hơn những gì được nói: các lane của một cột lane
            đọc cùng một word, nên warp đọc 4 word khác nhau, mỗi word được phát quảng bá tới 8 lane.
          </p>
        }
      >
        <p>
          The book says <code>B_s</code> needs nothing: one warp's B-strip read touches consecutive elements of a row, which
          sit in different banks. In the §15.5 layout that holds, and the picture is even better than stated: the lanes of a
          lane column read the same word, so the warp reads 4 distinct words, each broadcast to 8 lanes.
        </p>
      </Translated>
      <Source section="15.6" pages="363" />
      <Translated
        vi={
          <p>
            Trong cách bố trí §15.3/§15.4 điều đó không đúng. Ở đó warp là hai hàng 16 luồng với ô 8 × 8. Lần đọc dải B của nó chạm 16
            word khác nhau cách nhau 8 float, rơi vào 4 bank: xung đột 4 chiều. Lần đọc dải A của nó chỉ 2 chiều. Đệm A trong kernel
            đó loại bỏ một xung đột 2 chiều và để lại một xung đột 4 chiều, đó là lý do các lab đệm sau khi ánh xạ lại, không phải
            trước. Hãy chuyển cách bố trí của trình khám phá để thấy cả hai.
          </p>
        }
      >
        <p>
          In the §15.3/§15.4 layout it does not hold. There the warp is two rows of 16 threads with 8 × 8 tiles. Its B-strip
          read touches 16 distinct words 8 floats apart, which land in 4 banks: a 4-way conflict. Its A-strip read was only
          2-way. Padding A in that kernel removes a 2-way conflict and leaves a 4-way one, which is why the labs pad after the
          remap, not before. Switch the explorer's layout to see both.
        </p>
      </Translated>
      <Beyond title={tr('Our arithmetic', 'Phép tính của khóa học này')}>
        <Translated vi={<p>Từ ch15-facts E1, với quy tắc 32 bank của NVIDIA: "B không cần thay đổi" (p. 363) chỉ đúng với cách bố trí §15.5.</p>}>
          <p>From ch15-facts E1, with NVIDIA's 32-bank rule: "B needs no change" (p. 363) holds for the §15.5 layout only.</p>
        </Translated>
      </Beyond>
      <Ask q={questionsS156['ch15/15.6/q-early-pad']} />

      <h2>
        <Translated vi="Thay đổi trong mã, và dòng mà sách bỏ sót">The code change, and the line the book leaves out</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Thay đổi của sách gồm hai chỉnh sửa: khai báo <code>A_s</code> với bM·(bK + 1) float, và truyền bK + 1 thay vì bK làm chiều
            đầu của <code>A_s</code> cho <code>loadTile</code> và <code>mm</code>. Bên dưới là kernel của Hình 15.3 với đúng các chỉnh
            sửa đó. Còn một dòng nữa phải thay đổi trước khi nó tính ra đáp án đúng. Hãy tìm nó.
          </p>
        }
      >
        <p>
          The book's change is two edits: declare <code>A_s</code> with bM·(bK + 1) floats, and pass bK + 1 instead of bK as{' '}
          <code>A_s</code>'s leading dimension to <code>loadTile</code> and <code>mm</code>. Below is Fig. 15.3's kernel with
          exactly those edits. One more line must change before it computes the right answer. Find it.
        </p>
      </Translated>
      <Source section="15.6" pages="362–363" figs={tr('Fig. 15.3', 'Hình 15.3')} />
      <ErratumHunt />

      <h2>
        <Translated vi="Phần đệm tốn gì">What the padding costs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            <code>A_s</code> tăng từ 1,024 lên 1,152 float: {sharedBytes(9).A.toLocaleString('en-US')} byte, và{' '}
            {sharedBytes(9).total.toLocaleString('en-US')} khi tính cả <code>B_s</code>. Như vậy là nhỏ; §15.7 giải thích vì sao bộ
            nhớ chia sẻ không phải thứ giới hạn kernel này. Chi phí kia tinh tế hơn. Một hàng có đệm dài 36 byte, nên phần lớn các
            hàng không bắt đầu ở ranh giới 16 byte và không thể nhận một lệnh ghi float4. Lab 15.4 nạp A bằng float4 từ bộ nhớ toàn
            cục và ghi bốn giá trị vào <code>A_s</code> thành bốn lệnh ghi vô hướng.
          </p>
        }
      >
        <p>
          <code>A_s</code> grows from 1,024 to 1,152 floats: {sharedBytes(9).A.toLocaleString('en-US')} bytes, and{' '}
          {sharedBytes(9).total.toLocaleString('en-US')} with <code>B_s</code>. That is small; §15.7 explains why shared
          memory is not what limits this kernel. The other cost is subtler. A padded row is 36 bytes long, so most rows do not
          start on a 16-byte boundary and cannot take a float4 store. Lab 15.4 loads A with float4s from global memory and
          writes the four values into <code>A_s</code> as four scalars.
        </p>
      </Translated>
      <Beyond title={tr('Our arithmetic', 'Phép tính của khóa học này')}>
        <Translated vi={<p>Số byte là của khóa học này (notes, §15.6 [suy ra]); các lệnh ghi vô hướng vào A_s theo bản phác thảo Bài tập 1 trong notes.</p>}>
          <p>The byte counts are ours (notes, §15.6 [inferred]); the scalar stores into A_s follow the notes' Exercise 1 sketch.</p>
        </Translated>
      </Beyond>
      <Ask q={questionsS156['ch15/15.6/q-bytes']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Bank trên các GPU khác">Banks on other GPUs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            NVIDIA công bố 32 bank rộng 4 byte. AMD công bố 32 bank trên CDNA 1 đến 3 và 64 trên CDNA 4 và RDNA 2 đến 4, cũng 4 byte,
            và một LDS phục vụ 16 lane mỗi chu kỳ; AMD không nói các lane của một wavefront rộng 64 được kiểm tra với nhau thế nào,
            nên số liệu 64 bank của trình khám phá là một mô hình, không phải chi phí của AMD. Apple hoàn toàn không công bố bank nào
            cho bộ nhớ threadgroup. Trên thực tế MLX đệm các ô threadgroup của nó thêm 16 byte, nên trên Metal phần đệm là thứ bạn đo,
            không phải thứ bạn có thể dự đoán.
          </p>
        }
      >
        <p>
          NVIDIA documents 32 banks of 4 bytes. AMD documents 32 banks on CDNA 1 to 3 and 64 on CDNA 4 and RDNA 2 to 4, also
          4 bytes, and an LDS that serves 16 lanes per cycle; it does not say how the lanes of a 64-wide wavefront are checked
          against each other, so the explorer's 64-bank counts are a model, not AMD's cost. Apple documents no banks for
          threadgroup memory at all. MLX pads its threadgroup tiles by 16 bytes in practice, so on Metal padding is something
          you measure, not something you can predict.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch15-facts.md A2, B2, C2; ch06-facts.md D3)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              NVIDIA: "Any 32-bit memory read or write request made of 32 addresses that fall in 32 distinct memory banks can therefore
              be serviced simultaneously" (mọi yêu cầu đọc hoặc ghi 32 bit gồm 32 địa chỉ rơi vào 32 bank khác nhau đều có thể được
              phục vụ đồng thời). AMD: "A single warp can achieve up to 64 bytes per cycle throughput (16 lanes per cycle)." (một warp
              có thể đạt thông lượng tới 64 byte mỗi chu kỳ, 16 lane mỗi chu kỳ). Metal: không có "bank" trong đặc tả MSL 4.1; MLX:{' '}
              <code>tgp_padding_a = 16 / sizeof(T)</code>.
            </p>
          }
        >
          <p>
            NVIDIA: "Any 32-bit memory read or write request made of 32 addresses that fall in 32 distinct memory banks can
            therefore be serviced simultaneously". AMD: "A single warp can achieve up to 64 bytes per cycle throughput (16
            lanes per cycle)." Metal: no "bank" in the MSL 4.1 specification; MLX: <code>tgp_padding_a = 16 / sizeof(T)</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Mô hình còn một giới hạn nữa: nó coi mỗi thanh ghi dải là một lần nạp 32 bit. NVIDIA không công bố quy tắc bank nào cho
            truy cập bộ nhớ chia sẻ 64 hoặc 128 bit, và trình biên dịch có thể gộp các lần nạp cạnh nhau thành lần nạp rộng hơn. Các
            lab đọc dải dưới dạng giá trị 32 bit để mô hình áp dụng được, nhưng chỉ một profiler mới cho thấy phần cứng đã làm gì:{' '}
            <code>l1tex__data_bank_conflicts_pipe_lsu.sum</code> trong Nsight Compute, "Bank Conflict Rate" của LDS trong ROCm Compute
            Profiler, và không có gì trên Metal.
          </p>
        }
      >
        <p>
          The model has one more limit: it treats each strip register as a 32-bit load. NVIDIA documents no bank rule for
          64- or 128-bit shared-memory accesses, and a compiler may merge neighbouring loads into wider ones. The labs read
          strips as 32-bit values so the model applies, but only a profiler shows what the hardware did:{' '}
          <code>l1tex__data_bank_conflicts_pipe_lsu.sum</code> in Nsight Compute, the LDS "Bank Conflict Rate" in the ROCm
          Compute Profiler, and nothing on Metal.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch15-facts.md A2; ch06-facts.md A12, B4)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Đã tìm mà không thấy: một quy tắc trong Programming Guide hiện tại của NVIDIA về cách truy cập bộ nhớ chia sẻ 64 bit hoặc
              128 bit ánh xạ vào các bank. Một lab đọc dải dưới dạng float4 phải tự đo xung đột của nó.
            </p>
          }
        >
          <p>
            Searched and not found: a rule in NVIDIA's current Programming Guide for how 64-bit or 128-bit shared-memory
            accesses map to banks. A lab that reads strips as float4 must measure its conflicts.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Trong dự án: ô W">In the project: the W tile</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lớp tuyến tính của dự án tính Y = X·Wᵀ với W được lưu dạng (out, in), như cách <code>nn.Linear</code> và ggml lưu nó. Vì
            vậy ô B là một ô chuyển vị của W. Lab 15.6 nạp một ô 128 × 8 của W dọc theo các hàng của nó, cách này được hợp nhất, và
            ghi mỗi giá trị vào <code>B_s[k][n]</code>. Với lệnh nạp vô hướng, một warp giữ 4 hàng × 8 cột của W, và lần ghi của nó
            chạy xuống 8 hàng của <code>B_s</code>. Với pitch 128 float, 8 hàng đó bắt đầu ở cùng một bank. Hãy dùng các chế độ ô W
            của trình khám phá để tìm một pitch không xung đột, cho lệnh nạp vô hướng và cho float4.
          </p>
        }
      >
        <p>
          The project's linear layer computes Y = X·Wᵀ with W stored (out, in), the way <code>nn.Linear</code> and ggml store
          it. So the B tile is a transposed tile of W. Lab 15.6 loads a 128 × 8 tile of W along its rows, which coalesces,
          and stores each value into <code>B_s[k][n]</code>. With scalar loads one warp holds 4 rows × 8 columns of W, and
          its store runs down 8 rows of <code>B_s</code>. With a pitch of 128 floats those 8 rows start in the same bank.
          Use the explorer's W-tile modes to find a pitch that is free, for scalar and for float4 loads.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Của khóa học này: cách bố trí dự án là của lab 6.4 (<a href={href('ch06', 'labs')}>lab chương 6</a>), theo PyTorch và
              ggml (ch03-facts D2, D6); phép tính dùng quy tắc của NVIDIA. Phần đệm 16 byte của MLX lấy từ ch06-facts D3.
            </p>
          }
        >
          <p>
            Ours: the project layout is lab 6.4's (<a href={href('ch06', 'labs')}>chapter 6 labs</a>), after PyTorch's and
            ggml's (ch03-facts D2, D6); the arithmetic uses NVIDIA's rule. MLX's 16-byte padding is from ch06-facts D3.
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS156['ch15/15.6/q-w-pitch']} />
      <Translated
        vi={
          <p>
            Lời giải tham chiếu của lab 15.6 dùng pitch đó, <code>LDB_S = bN + 4</code>, và giữ ô A ở bK + 1. Lab 15.4 thêm phần đệm
            và lệnh nạp float4 vào kernel đã ánh xạ lại của bạn; lab 15.6 xây lớp tuyến tính trên đó. Cả hai đều ở{' '}
            <a href={href('ch15', 'labs')}>trang lab</a>. Tiếp theo, §15.7 hỏi các ô lớn tốn bao nhiêu thanh ghi:{' '}
            <a href={href('ch15', '15.7')}>cân nhắc về occupancy</a>.
          </p>
        }
      >
        <p>
          Lab 15.6's reference solution uses that pitch, <code>LDB_S = bN + 4</code>, and keeps the A tile at bK + 1. Lab
          15.4 adds the padding and float4 loads to your remapped kernel; lab 15.6 builds the linear layer on top. Both
          are on the <a href={href('ch15', 'labs')}>labs page</a>. Next, §15.7 asks what the large tiles cost in registers:{' '}
          <a href={href('ch15', '15.7')}>occupancy considerations</a>.
        </p>
      </Translated>
    </>
  );
}
