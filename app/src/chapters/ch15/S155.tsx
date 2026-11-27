import { useState, type MouseEvent, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BM,
  BN,
  DEFAULT_STORE,
  PATTERNS,
  PATTERN_ORDER,
  coalesced,
  coarseThread,
  instructionsPerThread,
  ownerOf,
  remapGeometry,
  remapThread,
  rowStrips,
  storeColumns,
  storeInstruction,
  warpTileTotal,
  type Layout,
  type StorePattern,
  type Width,
} from '../../sim/gemmremap';
import { RemapScene } from './RemapScene';

const thousands = (v: number) => v.toLocaleString('en-US');
const pct = (v: number) => `${Math.round(v * 1000) / 10} %`;

// ---------------------------------------------------------------- the store visualiser

const N_CHOICES: { N: number; label: string }[] = [
  {
    N: 1024,
    get label() {
      return tr('1024 (a multiple of 8)', '1024 (bội số của 8)');
    },
  },
  {
    N: 1020,
    get label() {
      return tr('1020 (a multiple of 4, not of 8)', '1020 (bội số của 4, không phải của 8)');
    },
  },
  {
    N: 1001,
    get label() {
      return tr('1001 (odd)', '1001 (lẻ)');
    },
  },
];
const SEGMENTS: { bytes: number; label: string }[] = [
  {
    bytes: 32,
    get label() {
      return tr('32-byte sectors (NVIDIA)', 'sector 32 byte (NVIDIA)');
    },
  },
  {
    bytes: 64,
    get label() {
      return tr('64-byte lines (AMD)', 'dòng 64 byte (AMD)');
    },
  },
  {
    bytes: 128,
    get label() {
      return tr('128-byte lines (AMD)', 'dòng 128 byte (AMD)');
    },
  },
];
const UNIT: Record<number, string> = {
  get 32() {
    return tr('sectors', 'sector');
  },
  get 64() {
    return tr('lines', 'dòng');
  },
  get 128() {
    return tr('lines', 'dòng');
  },
};
const SHORT: Record<StorePattern, string> = {
  get scalar8x8() {
    return tr('8 × 8, scalar (§15.3)', '8 × 8, vô hướng (§15.3)');
  },
  get 'float4-8x8'() {
    return tr('8 × 8, float4 (Exercise 2)', '8 × 8, float4 (Bài tập 2)');
  },
  get remap4x4() {
    return tr('four 4 × 4, float4 (§15.5)', 'bốn ô 4 × 4, float4 (§15.5)');
  },
};

type Verdict = 'yes' | 'no';
/** Value cells of the thread table may wrap on a phone (the table style keeps cells on one line). */
const WRAP = { whiteSpace: 'normal' as const };

function StoreVisualiser() {
  const [pattern, setPattern] = useState<StorePattern>('scalar8x8');
  const [W, setW] = useState<Width>(32);
  const [N, setN] = useState(1024);
  const [seg, setSeg] = useState(32);
  const [warp, setWarp] = useState(0);
  const [row, setRow] = useState(0);
  const [col, setCol] = useState(0);
  const [said, setSaid] = useState<Partial<Record<StorePattern, Verdict>>>({});

  const cols = storeColumns(pattern, N);
  const c = cols.includes(col) ? col : col - (col % 4);
  const warps = 256 / W;
  const w = Math.min(warp, warps - 1);
  const res = storeInstruction({ pattern, W, N, warp: w, row, col: c, segmentBytes: seg });
  const strips = rowStrips(res, seg);
  const total = warpTileTotal(pattern, W, N, w, seg);
  const verdict = said[pattern];
  const truth: Verdict = coalesced(pattern) ? 'yes' : 'no';

  // Drawing: one strip per matrix row, labelled above it, at one scale for every pattern
  // so the patterns can be compared; the picture is never drawn larger than life.
  const span = Math.max(...strips.map((s) => (s.seg1 - s.seg0) * seg));
  const pxB = Math.min(1.25, 420 / Math.max(span, 1));
  const X0 = 6;
  const pitch = 40;
  const VW = Math.max(320, Math.ceil(X0 * 2 + span * pxB));
  const H = 8 + strips.length * pitch;
  return (
    <Bench
      title={tr('One warp, one store instruction', 'Một warp, một lệnh ghi')}
      task={
        <Translated vi="Với mỗi mẫu, hãy quyết định lệnh ghi có được hợp nhất (coalesced) hay không trước khi các con số hiện ra. Sau đó đổi độ rộng, N và kích thước đoạn.">
          For each pattern, decide whether the store is coalesced before the counts appear. Then change the width, N and the segment size.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Mỗi dải là đoạn của C mà một hàng ma trận đóng góp, từ đoạn {seg} byte đầu tiên mà lệnh chạm tới đến đoạn cuối
              cùng. Đậm: các byte một lane ghi. Nhạt: các byte hệ thống bộ nhớ di chuyển nhưng không ai yêu cầu. C bắt đầu ở một
              biên 256 byte; khối là khối (0, 0).
            </>
          }
        >
          <>
            Each strip is the stretch of C one matrix row contributes, from the first {seg}-byte segment the instruction touches
            to the last. Dark: bytes a lane writes. Pale: bytes the memory system moves but nobody asked for. C starts on a
            256-byte boundary; the block is block (0, 0).
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Store pattern', 'Mẫu ghi')}>
          {PATTERN_ORDER.map((p) => (
            <button
              key={p} aria-pressed={p === pattern} onClick={() => setPattern(p)}
              title={tr(
                `${PATTERNS[p].label} (${PATTERNS[p].book})`,
                {
                  scalar8x8: 'ô 8 × 8, ghi vô hướng (§15.3, Fig. 15.7)',
                  'float4-8x8': 'ô 8 × 8, ghi float4 (Bài tập 2)',
                  remap4x4: 'bốn ô con 4 × 4, ghi float4 (§15.5, Bài tập 3)',
                }[p],
              )}
            >
              {SHORT[p]}
            </button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={W} onChange={(e) => setW(Number(e.target.value) as Width)}>
            <option value={32}>{tr('32 lanes', '32 lane')}</option>
            <option value={64}>{tr('64 lanes (AMD CDNA)', '64 lane (AMD CDNA)')}</option>
          </select>
        </label>
        <label className="field">
          <Translated vi="N, độ rộng của C">N, the width of C</Translated>
          <select value={N} onChange={(e) => setN(Number(e.target.value))}>
            {N_CHOICES.map((o) => <option key={o.N} value={o.N}>{o.label}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Đoạn">Segment</Translated>
          <select value={seg} onChange={(e) => setSeg(Number(e.target.value))}>
            {SEGMENTS.map((o) => <option key={o.bytes} value={o.bytes}>{o.label}</option>)}
          </select>
        </label>
        <label className="field">
          {tr('Warp', 'Warp')}
          <select value={w} onChange={(e) => setWarp(Number(e.target.value))}>
            {Array.from({ length: warps }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Lệnh: hàng của ô">Instruction: tile row</Translated>
          <select value={row} onChange={(e) => setRow(Number(e.target.value))}>
            {Array.from({ length: 8 }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="các cột của ô">tile columns</Translated>
          <select value={c} onChange={(e) => setCol(Number(e.target.value))}>
            {cols.map((k) => <option key={k} value={k}>{res.vector ? `${k}–${k + 3}` : k}</option>)}
          </select>
        </label>
      </div>

      {res.fallback ? (
        <Translated
          vi={
            <p className="narrate" style={{ marginBottom: '0.8rem' }}>
              N = {N} không phải bội số của 4, nên phần lớn các hàng của C không bắt đầu ở biên 16 byte và một lệnh ghi
              float4 sẽ bị lệch căn chỉnh. Kernel đi theo nhánh vô hướng: một float mỗi lane, {instructionsPerThread(pattern, N)}{' '}
              lệnh ghi mỗi luồng.
            </p>
          }
        >
          <p className="narrate" style={{ marginBottom: '0.8rem' }}>
            N = {N} is not a multiple of 4, so most rows of C do not start on a 16-byte boundary and a float4 store would be
            misaligned. The kernel takes its scalar path: one float per lane, {instructionsPerThread(pattern, N)} store
            instructions per thread.
          </p>
        </Translated>
      ) : null}

      <svg viewBox={`0 0 ${VW} ${H}`} role="img" aria-label={tr('Bytes of C written by one store instruction', 'Các byte của C được ghi bởi một lệnh ghi')}
        data-transactions={res.transactions} data-efficiency={res.efficiency} data-rows={strips.length} data-vector={res.vector ? 'yes' : 'no'}
        style={{ width: '100%', maxWidth: VW * 1.15, display: 'block', marginBottom: '0.6rem' }}>
        {strips.map((s, k) => {
          const y = 4 + k * pitch;
          return (
            <g key={s.row} data-strip={s.row}>
              <text x={X0} y={y + 10} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
                <Translated
                  vi={
                    <>
                      hàng {s.row} <tspan style={{ fill: 'var(--ink-3)' }}>từ byte {thousands(s.seg0 * seg)}</tspan>
                    </>
                  }
                >
                  row {s.row} <tspan style={{ fill: 'var(--ink-3)' }}>from byte {thousands(s.seg0 * seg)}</tspan>
                </Translated>
              </text>
              {s.used.map((u, j) => (
                <rect key={j} x={X0 + j * seg * pxB} y={y + 15} width={seg * pxB - 1} height={18}
                  fill="var(--global-wash)" stroke="var(--global)" strokeWidth="0.7" />
              ))}
              {s.pieces.map((pc) => {
                const x = X0 + (pc.lo - s.seg0 * seg) * pxB;
                const wd = (pc.hi - pc.lo) * pxB;
                return (
                  <g key={pc.lane}>
                    <rect x={x} y={y + 17} width={Math.max(1, wd - 0.6)} height={14} fill="var(--global)">
                      <title>
                        <Translated vi={<>lane {pc.lane}: byte {thousands(pc.lo)} đến {thousands(pc.hi - 1)}</>}>
                          lane {pc.lane}: bytes {thousands(pc.lo)} to {thousands(pc.hi - 1)}
                        </Translated>
                      </title>
                    </rect>
                    {wd >= 15 ? (
                      <text x={x + wd / 2} y={y + 27.5} fontSize="8" textAnchor="middle" style={{ fill: 'var(--page)' }}>{pc.lane}</text>
                    ) : null}
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>

      {verdict == null ? (
        <div className="row" data-classify={pattern}>
          <span>
            <Translated vi="Lệnh ghi này có được hợp nhất không, với 32 lane, N = 1024 và sector 32 byte?">
              Is this store coalesced, with 32 lanes, N = 1024 and 32-byte sectors?
            </Translated>
          </span>
          <button onClick={() => setSaid({ ...said, [pattern]: 'yes' })}>
            <Translated vi="Được hợp nhất">Coalesced</Translated>
          </button>
          <button onClick={() => setSaid({ ...said, [pattern]: 'no' })}>
            <Translated vi="Không được hợp nhất">Not coalesced</Translated>
          </button>
        </div>
      ) : (
        <div data-counts={pattern}>
          <p style={{ margin: '0 0 0.5rem' }}>
            {verdict === truth ? (
              <span className="chip ok"><Translated vi="Đúng.">Right.</Translated></span>
            ) : (
              <span className="chip fail"><Translated vi="Chưa đúng hẳn.">Not quite.</Translated></span>
            )}{' '}
            <Translated
              vi={
                <>
                  Ở thiết lập mặc định, mẫu này{' '}
                  {truth === 'yes'
                    ? 'được hợp nhất: mọi byte được di chuyển đều là byte được yêu cầu'
                    : 'không được hợp nhất: phần lớn mỗi đoạn bị di chuyển vô ích'}
                  .
                </>
              }
            >
              In the default setting this pattern is {truth === 'yes' ? 'coalesced: every byte moved is a byte asked for' : 'not coalesced: most of each segment is moved for nothing'}.
            </Translated>
          </p>
          <p className="ui" style={{ margin: 0 }} data-readout="">
            <Translated
              vi={
                <>
                  Lệnh này: chạm tới <b>{res.transactions} {UNIT[seg]}</b>, {thousands(res.bytesNeeded)} trên{' '}
                  {thousands(res.bytesFetched)} byte có ích: <b>{pct(res.efficiency)}</b>. {res.runs} đoạn byte liên tiếp
                  riêng biệt, trên {res.rows.length} hàng ma trận.
                </>
              }
            >
              This instruction: <b>{res.transactions} {UNIT[seg]}</b> touched, {thousands(res.bytesNeeded)} of{' '}
              {thousands(res.bytesFetched)} bytes useful: <b>{pct(res.efficiency)}</b>. {res.runs} separate{' '}
              {res.runs === 1 ? 'run' : 'runs'} of contiguous bytes, on {res.rows.length} matrix rows.
            </Translated>
          </p>
          <p className="ui muted" style={{ margin: '0.3rem 0 0' }} data-total="">
            <Translated
              vi={
                <>
                  Cả ô của warp: {total.instructions} lệnh ghi mỗi luồng, tổng cộng {thousands(total.transactions)}{' '}
                  {UNIT[seg]}; {thousands(total.bytes)} byte của nó vừa trong {thousands(total.minimum)}.
                </>
              }
            >
              The warp's whole tile: {total.instructions} store instructions per thread, {thousands(total.transactions)}{' '}
              {UNIT[seg]} in all; its {thousands(total.bytes)} bytes fit in {thousands(total.minimum)}.
            </Translated>
          </p>
        </div>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- the lane picker

function LanePicker() {
  const [W, setW] = useState<Width>(32);
  const [warp, setWarp] = useState(6);
  const [lane, setLane] = useState(5);
  const [layout, setLayout] = useState<Layout>('remap');
  const [hover, setHover] = useState<{ row: number; col: number } | null>(null);

  const g = remapGeometry(W);
  const wp = Math.min(warp, g.warps - 1);
  const ln = Math.min(lane, W - 1);
  const t = wp * W + ln;
  const rt = remapThread(t, W);
  const ct = coarseThread(t);
  const S = 4; // pixels per element
  const O = 24;
  const X = (c: number) => O + c * S;
  const Y = (r: number) => O + r * S;
  const at = (e: MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const scale = (O * 2 + BN * S) / box.width;
    const col = Math.floor(((e.clientX - box.left) * scale - O) / S);
    const row = Math.floor(((e.clientY - box.top) * scale - O) / S);
    return row >= 0 && row < BM && col >= 0 && col < BN ? { row, col } : null;
  };
  const owner = hover ? ownerOf(layout, W, hover.row, hover.col) : null;
  const laneRowMates = Array.from({ length: g.LC }, (_, k) => Math.floor(ln / g.LC) * g.LC + k);

  let shapes: ReactNode;
  if (layout === 'remap') {
    shapes = (
      <>
        {Array.from({ length: g.warps }, (_, k) => {
          const r = remapThread(k * W, W);
          return (
            <rect key={'w' + k} x={X(r.wCol)} y={Y(r.wRow)} width={g.warpCols * S} height={g.warpRows * S}
              fill={k === wp ? 'var(--bench-grid)' : 'none'} stroke="var(--ink-2)" strokeWidth={k === wp ? 2 : 1} />
          );
        })}
        {[0, 1].flatMap((qr) => [0, 1].map((qc) => (
          <rect key={`q${qr}${qc}`} x={X(rt.wCol + qc * g.QC)} y={Y(rt.wRow + qr * g.QR)} width={g.QC * S} height={g.QR * S}
            fill="none" stroke="var(--ink-3)" strokeDasharray="4 3" />
        )))}
        {laneRowMates.filter((m) => m !== ln).flatMap((m) =>
          remapThread(wp * W + m, W).subTiles.map((s) => (
            <rect key={`m${m}-${s.qr}${s.qc}`} x={X(s.col)} y={Y(s.row)} width={4 * S} height={4 * S} fill="var(--reg-wash)" stroke="var(--reg)" strokeWidth="0.6" />
          )),
        )}
        {rt.subTiles.map((s) => (
          <rect key={`s${s.qr}${s.qc}`} data-subtile={`${s.row},${s.col}`} x={X(s.col)} y={Y(s.row)} width={4 * S} height={4 * S} fill="var(--reg)" />
        ))}
      </>
    );
  } else {
    shapes = (
      <>
        {Array.from({ length: 15 }, (_, k) => (k + 1) * 8).map((v) => (
          <g key={v}>
            <line x1={X(0)} x2={X(BN)} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
            <line y1={Y(0)} y2={Y(BM)} x1={X(v)} x2={X(v)} stroke="var(--rule-soft)" />
          </g>
        ))}
        {Array.from({ length: W }, (_, l) => {
          const c = coarseThread(wp * W + l);
          return <rect key={l} x={X(c.tCol)} y={Y(c.tRow)} width={8 * S} height={8 * S} fill="var(--bench-grid)" stroke="var(--ink-3)" strokeWidth="0.5" />;
        })}
        <rect data-subtile={`${ct.tRow},${ct.tCol}`} x={X(ct.tCol)} y={Y(ct.tRow)} width={8 * S} height={8 * S} fill="var(--reg)" />
      </>
    );
  }

  return (
    <Bench
      title={tr('Lane picker', 'Bộ chọn lane')}
      task={
        <Translated vi="Chọn một warp và một lane rồi tìm bốn ô vuông của nó. Rê chuột lên bất kỳ phần tử nào để xem ai sở hữu nó; bấm vào để chọn lane đó.">
          Pick a warp and a lane and find its four squares. Hover over any element to see who owns it; click it to pick that lane.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Ô khối 128 × 128. Đậm: ô của lane. Nhạt: các lane khác trong hàng lane của nó, có lệnh ghi nằm cạnh lệnh ghi của
              nó. Nét đứt: các góc phần tư của ô warp. Ở khung nhìn §15.3, các mảng xám là phần còn lại của warp.
            </>
          }
        >
          <>
            The 128 × 128 block tile. Dark: the lane's tile. Pale: the other lanes of its lane row, whose stores sit beside its
            own. Dashed: the quadrants of its warp's tile. In the §15.3 view the grey patches are the rest of the warp.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span className="seg" role="group" aria-label={tr('Mapping', 'Ánh xạ')}>
          <button aria-pressed={layout === 'remap'} onClick={() => setLayout('remap')}>
            <Translated vi="§15.5: bốn ô 4 × 4">§15.5: four 4 × 4</Translated>
          </button>
          <button aria-pressed={layout === 'coarse'} onClick={() => setLayout('coarse')}>
            <Translated vi="§15.3: một ô 8 × 8">§15.3: one 8 × 8</Translated>
          </button>
        </span>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={W} onChange={(e) => setW(Number(e.target.value) as Width)}>
            <option value={32}>{tr('32 lanes', '32 lane')}</option>
            <option value={64}>{tr('64 lanes (AMD CDNA)', '64 lane (AMD CDNA)')}</option>
          </select>
        </label>
        <label className="field">
          {tr('Warp', 'Warp')}
          <select value={wp} onChange={(e) => setWarp(Number(e.target.value))}>
            {Array.from({ length: g.warps }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <label className="field">
          {tr('Lane', 'Lane')}
          <select value={ln} onChange={(e) => setLane(Number(e.target.value))}>
            {Array.from({ length: W }, (_, k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 5fr)', alignItems: 'start' }}>
        <svg viewBox={`0 0 ${O * 2 + BN * S} ${O * 2 + BM * S}`} role="img" aria-label={tr("The block tile with the chosen lane's elements", 'Ô khối với các phần tử của lane đã chọn')}
          data-layout={layout} data-thread={t}
          style={{ width: '100%', maxWidth: 540, display: 'block', cursor: 'crosshair' }}
          onMouseMove={(e) => setHover(at(e))}
          onMouseLeave={() => setHover(null)}
          onClick={(e) => {
            const h = at(e);
            if (!h) return;
            const o = ownerOf(layout, W, h.row, h.col);
            setWarp(o.warp);
            setLane(o.lane);
          }}>
          <rect x={X(0)} y={Y(0)} width={BN * S} height={BM * S} fill="var(--cell)" stroke="var(--rule)" />
          {shapes}
          {hover ? <rect x={X(hover.col) - 1} y={Y(hover.row) - 1} width={S + 2} height={S + 2} fill="none" stroke="var(--ink)" strokeWidth="1.5" /> : null}
          {[0, 32, 64, 96, 127].map((v) => (
            <g key={v}>
              <text x={X(v) + (v === 127 ? S : 0)} y={O - 4} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{v === 127 ? 128 : v}</text>
              <text x={O - 3} y={Y(v) + (v === 127 ? S : 0) + 3} fontSize="9" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{v === 127 ? 128 : v}</text>
            </g>
          ))}
        </svg>
        <div>
          <p className="narrate" role="status" data-hover={owner ? `${owner.t}` : ''} style={{ marginTop: 0 }}>
            {owner && hover ? (
              <Translated
                vi={
                  <>
                    Phần tử (hàng {hover.row}, cột {hover.col}) thuộc về luồng {owner.t}: warp {owner.warp}, lane {owner.lane}.
                    Nó là phần tử ({owner.r}, {owner.c}) của ô logic 8 × 8 của luồng đó
                    {layout === 'remap' ? <>, trong ô con của nó ở góc phần tư ({owner.qr}, {owner.qc})</> : null}.
                  </>
                }
              >
                <>
                  Element (row {hover.row}, column {hover.col}) belongs to thread {owner.t}: warp {owner.warp}, lane {owner.lane}.
                  It is element ({owner.r}, {owner.c}) of that thread's logical 8 × 8 tile
                  {layout === 'remap' ? <>, in its sub-tile of quadrant ({owner.qr}, {owner.qc})</> : null}.
                </>
              </Translated>
            ) : (
              <Translated vi={<>Rê chuột lên ô để xem ai sở hữu một phần tử.</>}>
                <>Hover over the tile to see who owns an element.</>
              </Translated>
            )}
          </p>
          <table className="threads" aria-label={tr('The chosen thread', 'Luồng đã chọn')} data-table={layout}>
            <tbody>
              <tr>
                <td><Translated vi="luồng">thread</Translated></td>
                <td className="mono" style={WRAP}>t = {wp}·{W} + {ln} = {t}</td>
              </tr>
              {layout === 'remap' ? (
                <>
                  <tr>
                    <td><Translated vi="ô warp">warp tile</Translated></td>
                    <td className="mono" style={WRAP}>
                      <Translated vi={<>hàng {rt.wRow}–{rt.wRow + g.warpRows - 1}, cột {rt.wCol}–{rt.wCol + g.warpCols - 1}</>}>
                        rows {rt.wRow}–{rt.wRow + g.warpRows - 1}, columns {rt.wCol}–{rt.wCol + g.warpCols - 1}
                      </Translated>
                    </td>
                  </tr>
                  <tr>
                    <td><Translated vi="hàng lane, cột lane">lane row, column</Translated></td>
                    <td className="mono" style={WRAP}>
                      <Translated vi={<>{rt.lRow / 4}, {rt.lCol / 4} (trên 8 × {g.LC})</>}>
                        {rt.lRow / 4}, {rt.lCol / 4} (of 8 × {g.LC})
                      </Translated>
                    </td>
                  </tr>
                  {rt.subTiles.map((s) => (
                    <tr key={`${s.qr}${s.qc}`} data-row-subtile={`${s.row},${s.col}`}>
                      <td>
                        <Translated vi={<>góc phần tư ({s.qr}, {s.qc})</>}>
                          quadrant ({s.qr}, {s.qc})
                        </Translated>
                      </td>
                      <td className="mono t-reg" style={WRAP}>
                        <Translated vi={<>hàng {s.row}–{s.row + 3}, cột {s.col}–{s.col + 3}</>}>
                          rows {s.row}–{s.row + 3}, columns {s.col}–{s.col + 3}
                        </Translated>
                      </td>
                    </tr>
                  ))}
                  <tr>
                    <td><Translated vi="hàng lane của nó ghi">its lane row stores</Translated></td>
                    <td className="mono" style={WRAP}>
                      <Translated vi={<>{g.LC} lane × 16 B = {g.laneRowBytes} byte liên tiếp</>}>
                        {g.LC} lanes × 16 B = {g.laneRowBytes} contiguous bytes
                      </Translated>
                    </td>
                  </tr>
                </>
              ) : (
                <>
                  <tr><td>ty, tx{/* i18n-ignore: code variables */}</td><td className="mono" style={WRAP}>{ct.ty}, {ct.tx}</td></tr>
                  <tr data-row-subtile={`${ct.tRow},${ct.tCol}`}>
                    <td><Translated vi="mảng 8 × 8 của nó">its 8 × 8 patch</Translated></td>
                    <td className="mono t-reg" style={WRAP}>
                      <Translated vi={<>hàng {ct.tRow}–{ct.tRow + 7}, cột {ct.tCol}–{ct.tCol + 7}</>}>
                        rows {ct.tRow}–{ct.tRow + 7}, columns {ct.tCol}–{ct.tCol + 7}
                      </Translated>
                    </td>
                  </tr>
                  <tr>
                    <td><Translated vi="lane kế bên trong warp">its neighbour in the warp</Translated></td>
                    <td className="mono" style={WRAP}>
                      <Translated vi="8 float (32 byte) về bên phải">8 floats (32 bytes) to the right</Translated>
                    </td>
                  </tr>
                </>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS155: Record<string, Question> = {
  'ch15/15.5/q-float4-alone': {
    kind: 'choice',
    id: 'ch15/15.5/q-float4-alone',
    prompt: (
      <Translated
        vi={
          <>
            Bạn làm Bài tập 2: <code>writeTile</code> ghi mỗi hàng của ô 8 × 8 thành hai float4 thay vì tám float, và không
            gì khác thay đổi. Một lệnh ghi của một warp 32 lane giờ tốn bao nhiêu, với N là bội số của 8 và sector 32 byte?
          </>
        }
      >
        <>
          You do Exercise 2: <code>writeTile</code> stores each row of the 8 × 8 tile as two float4s instead of eight floats,
          and nothing else changes. What does one store instruction of a 32-lane warp cost now, with N a multiple of 8 and
          32-byte sectors?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="8 sector: một phần tư của 32 sector của lệnh ghi vô hướng, vì mỗi lane giờ ghi bốn float một lúc.">
            {'8 sectors: a quarter of the scalar store’s 32, since each lane now stores four floats at once.'}
          </Translated>
        ),
        why: (
          <Translated vi="Ít lệnh hơn không có nghĩa là ít sector hơn cho mỗi lệnh. 16 byte của mỗi lane vẫn nằm riêng lẻ, cách lane kế bên 32 byte, nên một lệnh vẫn chạm tới 32 sector.">
            {'Fewer instructions do not mean fewer sectors per instruction. Each lane’s 16 bytes still sit alone, 32 bytes from its neighbour’s, so one instruction still touches 32.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="16 sector, mọi byte đều được ghi, như một lệnh ghi 512 byte được hợp nhất hoàn toàn.">
            {'16 sectors, every byte written, as in a fully coalesced 512-byte store.'}
          </Translated>
        ),
        why: (
          <Translated vi="Đó là con số sau khi ánh xạ lại ở §15.5, khi các mảnh của 4 lane tạo thành các đoạn 64 byte.">
            {'That is the count after the remap of §15.5, where the pieces of 4 lanes form 64-byte runs.'}
          </Translated>
        ),
      },
      {
        label: <Translated vi="32 sector, mỗi sector được ghi một nửa.">{'32 sectors, each half written.'}</Translated>,
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một float4 phủ các cột 0–3 của mảng của một lane; mảng của lane kế bên bắt đầu xa hơn 8 float. Vì vậy 16 byte của
            mỗi lane nằm riêng lẻ trong một sector 32 byte, và lệnh chạm tới 32 sector ở mức 50 %. Sách nói thẳng: một lệnh
            ghi vector rộng bằng nửa một ô 8 × 8, nên lệnh ghi của các lane kế bên vẫn cách nhau bốn phần tử. Ít lệnh hơn,
            vẫn phân tán như cũ.
          </p>
        }
      >
        <p>
          A float4 covers columns 0–3 of a lane's patch; the neighbouring lane's patch starts 8 floats further on. So each
          lane's 16 bytes sit alone in a 32-byte sector, and the instruction touches 32 sectors at 50 %. The book says it
          plainly: a vector store is half the width of an 8 × 8 tile, so neighbours' stores are still four elements apart.
          Fewer instructions, same scattering.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.5, p. 360; sector rule from notes/today/ch06-facts.md A2',
    vi: { source: 'ch. 15 §15.5, p. 360; quy tắc sector lấy từ notes/today/ch06-facts.md A2' },
  },
  'ch15/15.5/q-remap-sectors': {
    kind: 'numeric',
    id: 'ch15/15.5/q-remap-sectors',
    prompt: (
      <Translated
        vi={
          <>
            Sau khi ánh xạ lại, một lệnh ghi float4 của một warp 32 lane chạm tới bao nhiêu sector 32 byte, với N là bội số
            của 8?
          </>
        }
      >
        <>
          After the remap, how many 32-byte sectors does one float4 store instruction of a 32-lane warp touch, with N a
          multiple of 8?
        </>
      </Translated>
    ),
    answer: 16,
    unit: 'sectors',
    explain: (
      <Translated
        vi={
          <p>
            32 lane × 16 byte = 512 byte. 4 lane của mỗi hàng lane ghi 64 byte liên tiếp, bắt đầu ở biên sector khi N là bội
            số của 8: 2 sector mỗi hàng lane, 8 hàng lane, 16 sector, và mọi byte trong đó đều được ghi. Đó cũng là mức tối
            thiểu mà 512 byte bất kỳ có thể chiếm.
          </p>
        }
      >
        <p>
          32 lanes × 16 bytes = 512 bytes. The 4 lanes of each lane row write 64 contiguous bytes, which start on a sector
          boundary when N is a multiple of 8: 2 sectors per lane row, 8 lane rows, 16 sectors, and every byte of them is
          written. That is also the least any 512 bytes can occupy.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.5, pp. 360–361; our count with NVIDIA’s 32-byte sectors (ch06-facts A2)',
    vi: {
      unit: 'sector',
      source: 'ch. 15 §15.5, pp. 360–361; phép đếm của chúng tôi với sector 32 byte của NVIDIA (ch06-facts A2)',
    },
  },
  'ch15/15.5/q-registers': {
    kind: 'choice',
    id: 'ch15/15.5/q-registers',
    prompt: (
      <Translated vi={<>Việc ánh xạ lại có thay đổi số thanh ghi một luồng cần cho ô và các dải của nó không?</>}>
        <>Does the remap change how many registers a thread needs for its tile and strips?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: bốn ô con cần bốn bộ dải, nên 4 × (4 + 4) = 32 giá trị dải mỗi bước.">
            {'Yes: four sub-tiles need four sets of strips, so 4 × (4 + 4) = 32 strip values per step.'}
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi mảnh A phục vụ hai ô con (hai ô cùng hàng góc phần tư), và mỗi mảnh B cũng phục vụ hai.">
            {'Each A piece serves two sub-tiles (the two in the same quadrant row), and each B piece two as well.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không: 64 bộ tích lũy, và mỗi bước hai mảnh A 4 giá trị cùng hai mảnh B 4 giá trị, 16 giá trị, như trước.">
            {'No: 64 accumulators, and per step two 4-value A pieces and two 4-value B pieces, 16 values, as before.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Ít hơn: một ô con 4 × 4 chỉ cần 4 + 4 giá trị dải.">
            {'Fewer: a 4 × 4 sub-tile needs only 4 + 4 strip values.'}
          </Translated>
        ),
        why: (
          <Translated vi="Với mỗi ô con thì đúng, nhưng luồng cập nhật cả bốn ô con trong mỗi bước.">
            {'Per sub-tile, yes, but the thread updates all four sub-tiles in each step.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Luồng vẫn sở hữu 64 đầu ra, giờ là bốn ô vuông 4 × 4. Ở bước i nó cần cột i của A cho 8 hàng của nó (hai mảnh 4,
            mỗi hàng góc phần tư một mảnh) và hàng i của B cho 8 cột của nó (hai mảnh 4): cùng 8 + 8 giá trị như ô 8 × 8 của
            §15.4. Ngân sách thanh ghi của §15.7 không đổi.
          </p>
        }
      >
        <p>
          The thread still owns 64 outputs, now as four 4 × 4 squares. In step i it needs column i of A for its 8 rows (two
          pieces of 4, one per quadrant row) and row i of B for its 8 columns (two pieces of 4): the same 8 + 8 values as the
          8 × 8 tile of §15.4. The register budget of §15.7 does not change.
        </p>
      </Translated>
    ),
    source: 'ch. 15 §15.5, pp. 360–361, Fig. 15.10; the count is ours (notes [inferred])',
    vi: { source: 'ch. 15 §15.5, pp. 360–361, Fig. 15.10; phép đếm là của chúng tôi (notes [inferred])' },
  },
  'ch15/15.5/q-wave64': {
    kind: 'choice',
    id: 'ch15/15.5/q-wave64',
    prompt: (
      <Translated
        vi={
          <>
            Một khối 256 luồng trên GPU CDNA là 4 wavefront 64 lane. Cách bố trí nào giữ được bốn ô con 4 × 4 mỗi luồng và
            các lệnh ghi float4 liên tiếp?
          </>
        }
      >
        <>
          A block of 256 threads on a CDNA GPU is 4 wavefronts of 64. Which layout keeps four 4 × 4 sub-tiles per thread and
          contiguous float4 stores?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Giữ nguyên cách bố trí của sách: các ô warp 2 × 4 kích thước 64 × 32.">
            {'The book’s layout unchanged: 2 × 4 warp tiles of 64 × 32.'}
          </Translated>
        ),
        why: (
          <Translated vi="Chỉ có 4 wavefront, không phải 8, và một wavefront có 64 lane cần sắp xếp, không phải 32.">
            {'There are only 4 wavefronts, not 8, and a wavefront has 64 lanes to place, not 32.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các ô wavefront 2 × 2 kích thước 64 × 64; lane 8 × 8; góc phần tư 32 × 32; một hàng lane 8 lane ghi 128 byte liên tiếp.">
            {'2 × 2 wavefront tiles of 64 × 64; lanes 8 × 8; quadrants 32 × 32; a lane row of 8 lanes stores 128 contiguous bytes.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Các ô wavefront 1 × 4 kích thước 128 × 32; lane 16 × 4; góc phần tư 64 × 16.">
            {'1 × 4 wavefront tiles of 128 × 32; lanes 16 × 4; quadrants 64 × 16.'}
          </Translated>
        ),
        why: (
          <Translated vi="Nó phủ được ô, nhưng mỗi hàng lane vẫn chỉ ghi 64 byte, và lane cao 16 hàng: các lane dư đi vào hàng, không vào các đoạn dài hơn.">
            {'It covers the tile, but each lane row still writes only 64 bytes, and the lanes are 16 rows tall: the extra lanes go into rows, not into longer runs.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Giữ 8 hàng lane và bốn ô vuông 4 × 4 mỗi lane; 32 lane dư đi vào các hàng lane, giờ rộng 8. Một góc phần tư khi đó
            là 32 hàng nhân 8 × 4 = 32 cột, một ô wavefront 64 × 64, và bốn ô như vậy lấp đầy ô khối theo dạng 2 × 2. Mỗi hàng
            lane ghi 8 × 16 = 128 byte liên tiếp mỗi lệnh. Đây là cách suy ra của chúng tôi, không phải của AMD.
          </p>
        }
      >
        <p>
          Keep 8 lane rows and four 4 × 4 squares per lane; the extra 32 lanes go into the lane rows, which become 8 wide.
          A quadrant is then 32 rows by 8 × 4 = 32 columns, a wavefront tile 64 × 64, and four of them fill the block tile as
          2 × 2. Each lane row writes 8 × 16 = 128 contiguous bytes per instruction. This is our derivation, not AMD's.
        </p>
      </Translated>
    ),
    source: 'our derivation (notes/today/ch15-facts.md B1); wavefront widths from ch05-facts B5',
    vi: { source: 'cách suy ra của chúng tôi (notes/today/ch15-facts.md B1); độ rộng wavefront lấy từ ch05-facts B5' },
  },
  'ch15/15.5/q-width-wrong': {
    kind: 'choice',
    id: 'ch15/15.5/q-width-wrong',
    prompt: (
      <Translated vi={<>Kernel đã ánh xạ lại được biên dịch cho độ rộng 32 và chạy trên GPU có wavefront 64 lane. Điều gì xảy ra?</>}>
        <>The remapped kernel is compiled for a width of 32 and runs on a GPU with 64-lane wavefronts. What happens?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Kết quả sai: mỗi wavefront 64 lane chứa hai warp 32 lane, nên từng cặp luồng tính và ghi cùng các phần tử.">
            {'Wrong results: each 64-lane wavefront holds two 32-lane warps, so pairs of threads compute and write the same elements.'}
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi phần tử vẫn có đúng một chủ sở hữu: ánh xạ là một song ánh trên threadIdx.x bất kể nó được biên dịch với hằng số nào.">
            {'Every element still has exactly one owner: the mapping is a bijection over threadIdx.x whatever constant it was compiled with.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Kết quả đúng; chỉ có mẫu ghi không phải mẫu mà cách bố trí được thiết kế cho.">
            {'Correct results; only the store pattern is not the one the layout was designed for.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó không biên dịch được: hằng số độ rộng của kernel phải khớp với của thiết bị.">
            {'It does not compile: the kernel\'s width constant must match the device\'s.'}
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì trong kernel hỏi phần cứng về độ rộng của nó; nó dùng hằng số được cho.">
            {'Nothing in the kernel asks the hardware for its width; it uses the constant it was given.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Ánh xạ thuần là phép tính chỉ số: warp và lane được suy ra từ <code>threadIdx.x</code> với hằng số đã biên
              dịch, và mọi phần tử của ô vẫn có một chủ sở hữu. Không gì trong đó phụ thuộc vào việc luồng nào thực sự chạy
              cùng nhau. Thứ phụ thuộc vào điều đó là sự hợp nhất truy cập: một wavefront 64 lane giờ chứa hai "warp" 32 lane
              mà cách bố trí giả định, nên một lệnh ghi viết hai nhóm đoạn riêng biệt.
            </p>
          }
        >
          <p>
            The mapping is pure index arithmetic: warp and lane are derived from <code>threadIdx.x</code> with the compiled
            constant, and every element of the tile still gets one owner. Nothing in it depends on which threads really
            execute together. What does depend on that is the coalescing: a 64-lane wavefront now holds two of the 32-lane
            "warps" the layout assumed, so one store instruction writes two separate groups of runs.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Ở đây sai lệch độ rộng làm mất tốc độ, không làm mất tính đúng. Điều đó không cho phép viết một hằng 32 cứng:
              hãy chọn hằng số theo thiết bị, như bên dưới, để cách bố trí khớp với phần cứng nó chạy trên.
            </p>
          }
        >
          <p>
            A width mismatch here costs speed, not correctness. That is no licence to write a literal 32: choose the constant
            from the device, as below, so the layout matches the hardware it runs on.
          </p>
        </Translated>
      </>
    ),
    source: 'our analysis; widths from notes/today/ch15-facts.md B1 and C1',
    vi: { source: 'phân tích của chúng tôi; độ rộng lấy từ notes/today/ch15-facts.md B1 và C1' },
  },
  'ch15/15.5/q-align': {
    kind: 'choice',
    id: 'ch15/15.5/q-align',
    prompt: (
      <Translated vi={<>C có N = 1001 cột. Kernel có được ghi mỗi hàng ô con bằng một float4 không?</>}>
        <>C has N = 1001 columns. May the kernel store each sub-tile row with one float4?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: một lệnh ghi float4 chỉ cần bốn float liên tiếp, và mọi hàng ô con đều có bốn.">
            {'Yes: a float4 store needs only four consecutive floats, and every sub-tile row has four.'}
          </Translated>
        ),
        why: (
          <Translated vi="Nó cần chúng ở một địa chỉ là bội số của 16 byte.">
            {'It needs them at an address that is a multiple of 16 bytes.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không nói chung: hàng r bắt đầu ở byte 4004·r, chỉ căn chỉnh 16 byte khi r là bội số của 4, nên kernel ghi vô hướng trừ khi N % 4 == 0.">
            {'Not in general: row r starts at byte 4004·r, 16-byte aligned only for r a multiple of 4, so the kernel stores scalars unless N % 4 == 0.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ với cột khối cuối cùng, nơi các hàng ô con chạm tới mép của C.">
            {'Only for the last block column, where the sub-tile rows reach the edge of C.'}
          </Translated>
        ),
        why: (
          <Translated vi="Căn chỉnh phụ thuộc vào hàng, không phụ thuộc vào cột khối nào.">
            {'Alignment depends on the row, not on which block column.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            4 × 1001 = 4004 byte mỗi hàng, và 4004 mod 16 = 4, nên hàng r bắt đầu cách biên 16 byte 4r byte: chỉ căn chỉnh với
            r = 0, 4, 8, và cứ thế. Một lệnh ghi float4 tới địa chỉ lệch căn chỉnh là không được phép. Quy tắc đơn giản mà các
            lab dùng: đi theo nhánh vector khi N là bội số của 4, và ghi vô hướng có kiểm tra biên trong trường hợp còn lại.
            Một kernel đầy đủ cũng sẽ kiểm tra mép phải của C, nơi còn lại ít hơn 4 cột.
          </p>
        }
      >
        <p>
          4 × 1001 = 4004 bytes per row, and 4004 mod 16 = 4, so row r starts 4r bytes past a 16-byte boundary: aligned only
          for r = 0, 4, 8, and so on. A float4 store to a misaligned address is not allowed. The simple rule the labs use:
          take the vector path when N is a multiple of 4, and bounds-checked scalar stores otherwise. A full kernel would
          also check the right edge of C, where fewer than 4 columns remain.
        </p>
      </Translated>
    ),
    source: 'ours, from notes/today/ch06-facts.md A11 and ch15-facts.md E1 (alignment note)',
    vi: { source: 'của chúng tôi, từ notes/today/ch06-facts.md A11 và ch15-facts.md E1 (ghi chú về căn chỉnh)' },
  },
  'ch15/15.5/q-epilogue': {
    kind: 'choice',
    id: 'ch15/15.5/q-epilogue',
    prompt: (
      <Translated
        vi={
          <>
            CUTLASS giữ ánh xạ luồng phù hợp với phép tính số học và làm cho các lệnh ghi được hợp nhất trong một phần kết
            (epilogue) riêng, thông qua bộ nhớ chia sẻ. Điều đó tốn gì mà cách ánh xạ lại của sách không tốn?
          </>
        }
      >
        <>
          CUTLASS keeps the thread mapping that suits the arithmetic and makes its stores coalesced in a separate epilogue,
          through shared memory. What does that cost that the book's remap does not?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không tốn thêm gì: cùng các lệnh ghi đó xảy ra sau, và rào chắn cuối của vòng lặp đã sắp thứ tự chúng, nên không cần rào chắn mới.">
            {'Nothing extra: the same stores happen later, and the loop\'s last barrier already orders them, so no new barrier is needed.'}
          </Translated>
        ),
        why: (
          <Translated vi="Kết quả phải đi thêm một vòng qua bộ nhớ chia sẻ, và các luồng phải chờ ở một rào chắn mới trước khi đọc lại đầu ra của nhau.">
            {'The results take an extra trip through shared memory, and the threads must wait at a new barrier before reading back each other\'s outputs.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một vòng qua bộ nhớ chia sẻ, một rào chắn và không gian bộ nhớ chia sẻ, để ánh xạ có thể chỉ phục vụ phép tính số học.">
            {'A trip through shared memory, a barrier and shared-memory space, so the mapping can suit the arithmetic alone.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Thêm thanh ghi mỗi luồng, để giữ một bản sao thứ hai của các bộ tích lũy theo cách bố trí của lệnh ghi.">
            {'More registers per thread, to hold a second copy of the accumulators in the store\'s layout.'}
          </Translated>
        ),
        why: (
          <Translated vi="Các bộ tích lũy vẫn như cũ; phần kết chuyển chúng qua bộ nhớ chia sẻ, không qua thêm thanh ghi.">
            {'The accumulators are the same; the epilogue moves them through shared memory, not through more registers.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong phần kết, mỗi luồng ghi đầu ra vào bộ nhớ chia sẻ, khối đồng bộ hóa, và các luồng đọc lại chúng theo mẫu phù
            hợp với bộ nhớ toàn cục. Điều đó tốn lưu lượng bộ nhớ chia sẻ, một rào chắn và không gian, nhưng nó tách rời hai
            lựa chọn: ánh xạ cho phép nhân và mẫu cho lệnh ghi. Cách ánh xạ lại gắn chúng với nhau. Nó làm vậy được vì bốn ô
            vuông 4 × 4 cần cùng thanh ghi và cùng các cập nhật hạng 1 như một ô 8 × 8; cái giá của nó lộ ra ở các lần đọc bộ
            nhớ chia sẻ thay vào đó, điều mà §15.6 xử lý.
          </p>
        }
      >
        <p>
          In the epilogue each thread writes its outputs to shared memory, the block synchronises, and the threads read them
          back in a pattern that suits global memory. That costs shared-memory traffic, a barrier and space, but it decouples
          two choices: the mapping for the multiply and the pattern for the store. The remap couples them. It gets away with
          that because four 4 × 4 squares need the same registers and the same rank-1 updates as one 8 × 8; its price shows
          up in the shared-memory reads instead, which §15.6 deals with.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch15-facts.md A12 (CUTLASS “Efficient GEMM in CUDA”, section “Epilogue”)',
    vi: { source: 'notes/today/ch15-facts.md A12 (CUTLASS “Efficient GEMM in CUDA”, mục “Epilogue”)' },
  },
};

// ---------------------------------------------------------------- the page

export function S155() {
  const scalar = storeInstruction({ ...DEFAULT_STORE, pattern: 'scalar8x8' });
  const g32 = remapGeometry(32);
  const g64 = remapGeometry(64);
  const w6 = remapThread(6 * 32, 32);
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 15.3 và 15.4 cho mỗi luồng một ô 8 × 8 của C trong thanh ghi (register). Điều đó tốt cho tái sử dụng: mỗi giá
            trị một luồng nạp cấp cho tám tích. Cái giá xuất hiện một lần, ở cuối cùng, khi ô quay về bộ nhớ toàn cục (global
            memory). Một lệnh ghi viết cùng một phần tử trong ô của mọi lane, và các lane kế nhau sở hữu các ô kế nhau, nên địa
            chỉ của chúng cách nhau 8 float.
          </p>
        }
      >
        <p>
          Sections 15.3 and 15.4 gave every thread an 8 × 8 tile of C in registers. That is good for reuse: each value a
          thread loads feeds eight products. The cost appears once, at the very end, when the tile goes back to global
          memory. One store instruction writes the same element of every lane's tile, and neighbouring lanes own
          neighbouring tiles, so their addresses are 8 floats apart.
        </p>
      </Translated>
      <Source section="15.5" pages="360" />
      <Predict
        id="ch15/15.5/p-scalar-sectors"
        prompt={
          <Translated
            vi={
              <>
                Một lệnh ghi vô hướng của một warp theo cách bố trí §15.3, N là bội số của 8. Nó chạm tới bao nhiêu sector 32
                byte?
              </>
            }
          >
            <>
              One scalar store instruction of one warp in the §15.3 layout, N a multiple of 8. How many 32-byte sectors
              does it touch?
            </>
          </Translated>
        }
        unit={tr('sectors', 'sector')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                Nó chạm tới <b>{scalar.transactions}</b>
                {v === scalar.transactions ? ', đúng như bạn nói' : ''}. Một warp là 32 luồng liên tiếp: hai hàng 16 ô trong
                ô khối. Mỗi lane ghi 4 byte, và lane kế bên ghi xa hơn 32 byte, nên mỗi lane rơi vào một sector của riêng nó:{' '}
                {scalar.transactions} sector để chuyển {scalar.bytesNeeded} byte, {pct(scalar.efficiency)} lượng được di
                chuyển. Trình trực quan hóa bên dưới vẽ điều đó.
              </p>
            }
          >
            <p>
              It touches <b>{scalar.transactions}</b>
              {v === scalar.transactions ? ', as you said' : ''}. A warp is 32 consecutive threads: two rows of 16 tiles in the
              block tile. Each lane writes 4 bytes, and its neighbour writes 32 bytes further on, so every lane lands in a
              sector of its own: {scalar.transactions} sectors to deliver {scalar.bytesNeeded} bytes, {pct(scalar.efficiency)}{' '}
              of what moves. The visualiser below draws it.
            </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Cách sửa hiển nhiên là Bài tập 2 của sách: ghi bốn float một lúc bằng một float4. Nó giảm số lệnh ghi đi bốn lần.
            Nhưng một float4 phủ nửa hàng của ô 8 × 8, nên lệnh ghi của lane kế bên vẫn bắt đầu cách chỗ lệnh này kết thúc bốn
            float.
          </p>
        }
      >
        <p>
          The obvious repair is the book's Exercise 2: store four floats at once with a float4. It cuts the number of store
          instructions by four. But a float4 covers half a row of an 8 × 8 tile, so the neighbouring lane's store still starts
          four floats after this one ends.
        </p>
      </Translated>
      <Source section="15.5" pages="360" />

      <h2>
        <Translated vi="Một warp, một lệnh ghi">One warp, one store instruction</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trình trực quan hóa cho thấy, với một lệnh ghi của một warp, những byte nào của C mà mỗi lane ghi: một dải cho mỗi
            hàng ma trận mà warp chạm tới, cắt theo các đoạn của hệ thống bộ nhớ. Hãy nói mỗi mẫu có được hợp nhất (coalesced)
            hay không trước khi các con số hiện ra. Sau đó thử mẫu thứ ba, mà phần còn lại của trang này giải thích, và đổi độ
            rộng, N và kích thước đoạn.
          </p>
        }
      >
        <p>
          The visualiser shows, for one store instruction of one warp, which bytes of C each lane writes: one strip per
          matrix row the warp touches, cut into the memory system's segments. Say whether each pattern is coalesced before
          its counts appear. Then try the third pattern, which the rest of this page explains, and change the width, N and
          the segment size.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi lần một lệnh, đếm theo sector 32 byte mà NVIDIA công bố (ch06-facts A2). AMD công bố dòng cache 64 hoặc 128
              byte (ch06-facts B1), có sẵn làm lựa chọn; Apple không công bố quy tắc nào (ch06-facts C1). Một bộ nhớ đệm
              (cache) gộp các sector được ghi một phần từ các lệnh khác nhau không được mô hình hóa, thời gian cũng vậy.
            </p>
          }
        >
          <p>
            One instruction at a time, counted in NVIDIA's documented 32-byte sectors (ch06-facts A2). AMD documents 64- or
            128-byte cache lines (ch06-facts B1), offered as options; Apple documents no rule (ch06-facts C1). A cache that merges partly written
            sectors from different instructions is not modelled, and neither is time.
          </p>
        </Translated>
      </Simplification>
      <StoreVisualiser />
      <Ask q={questionsS155['ch15/15.5/q-float4-alone']} />

      <h2>
        <Translated vi="Bốn ô con 4 × 4 mỗi luồng">Four 4 × 4 sub-tiles per thread</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cách sửa của sách giữ 64 đầu ra của mỗi luồng nhưng đổi đó là 64 đầu ra nào. Nó thêm một tầng vào thứ bậc ô: warp.
            Ô khối 128 × 128 được chia cho 8 warp của khối thành 2 hàng 4 <b>ô cấp warp</b>, mỗi ô 64 hàng nhân 32 cột. Mỗi ô
            warp được cắt thành bốn góc phần tư 32 × 16. 32 lane của warp được xếp 8 × 4, và mỗi lane lấy ô vuông 4 × 4 ở vị
            trí của nó trong <i>mọi</i> góc phần tư. Bốn ô vuông đó là ô logic 8 × 8 của luồng; các ô vuông là các ô con vật lý
            của nó.
          </p>
        }
      >
        <p>
          The book's fix keeps each thread's 64 outputs but changes which 64. It adds a level to the tile hierarchy: the warp.
          The 128 × 128 block tile is divided among the block's 8 warps as 2 rows of 4 <b>warp-level tiles</b>, each 64 rows by
          32 columns. Each warp tile is cut into four quadrants of 32 × 16. The warp's 32 lanes are arranged 8 × 4, and each
          lane takes the 4 × 4 square at its position in <i>every</i> quadrant. Those four squares are the thread's logical
          8 × 8 tile; the squares are its physical sub-tiles.
        </p>
      </Translated>
      <Source section="15.5" pages="360–361" figs="Fig. 15.10" />
      <Translated
        vi={
          <p>
            Giờ một lệnh ghi viết một hàng của một ô vuông mỗi lane: 4 float, 16 byte, một float4. 4 lane của một hàng lane sở
            hữu các ô vuông kế nhau, nên lệnh ghi của chúng nối đuôi nhau thành 64 byte liên tiếp. 8 hàng lane cách nhau 4
            hàng ma trận. Sách gọi kết quả là được hợp nhất hoàn toàn. Sách không nhắc rằng một lệnh vẫn phủ 8 hàng riêng
            biệt; tính theo sector thì điều đó không tốn gì, miễn là mỗi đoạn 64 byte bắt đầu ở biên sector.
          </p>
        }
      >
        <p>
          Now a store instruction writes one row of one square per lane: 4 floats, 16 bytes, one float4. The 4 lanes of a
          lane row own neighbouring squares, so their stores sit end to end as 64 contiguous bytes. The 8 lane rows are 4
          matrix rows apart. The book calls the result fully coalesced. It does not mention that one instruction still covers
          8 separate rows; in sectors that costs nothing, as long as each 64-byte run starts on a sector boundary.
        </p>
      </Translated>
      <Beyond title={tr('Our reading', 'Cách đọc của chúng tôi')}>
        <Translated vi={<p>Việc tách theo hàng là của chúng tôi (notes, §15.5 [inferred]); sách dừng ở "fully coalesced".</p>}>
          <p>The per-row split is ours (notes, §15.5 [inferred]); the book stops at "fully coalesced".</p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trong mã, mọi tọa độ đến từ chỉ số luồng qua phép chia và phép lấy dư. Với W là độ rộng warp, LC = W / 8 lane mỗi
            hàng lane, và t là chỉ số luồng:
          </p>
        }
      >
        <p>
          In code, every coordinate comes from the thread index by division and remainder. With W the warp width, LC = W / 8
          lanes per lane row, and t the thread index:
        </p>
      </Translated>
      <ul>
        <li><code>warp = t / W</code>, <code>lane = t % W</code></li>
        <Translated vi={<li><code>lRow = 4*(lane / LC)</code>, <code>lCol = 4*(lane % LC)</code>: ô vuông của lane bên trong một góc phần tư</li>}>
          <li><code>lRow = 4*(lane / LC)</code>, <code>lCol = 4*(lane % LC)</code>: the lane's square inside a quadrant</li>
        </Translated>
        <Translated vi={<li><code>QR = 32</code>, <code>QC = 4*LC</code>: góc phần tư; <code>WC = 128 / (2*QC)</code> ô warp mỗi hàng</li>}>
          <li><code>QR = 32</code>, <code>QC = 4*LC</code>: the quadrant; <code>WC = 128 / (2*QC)</code> warp tiles per row</li>
        </Translated>
        <Translated vi={<li><code>wRow = 64*(warp / WC)</code>, <code>wCol = 2*QC*(warp % WC)</code>: ô warp</li>}>
          <li><code>wRow = 64*(warp / WC)</code>, <code>wCol = 2*QC*(warp % WC)</code>: the warp tile</li>
        </Translated>
        <Translated vi={<li>ô con (qr, qc) bắt đầu ở <code>(wRow + qr*QR + lRow, wCol + qc*QC + lCol)</code></li>}>
          <li>sub-tile (qr, qc) starts at <code>(wRow + qr*QR + lRow, wCol + qc*QC + lCol)</code></li>
        </Translated>
      </ul>
      <Translated
        vi={
          <p>
            Với W = 32, các công thức này cho ra số của sách: LC = {g32.LC}, QC = {g32.QC}, WC = {g32.WC}, và warp 6, warp mà
            Fig. 15.10 tô đậm, bắt đầu ở hàng {w6.wRow}, cột {w6.wCol}.
          </p>
        }
      >
        <p>
          With W = 32 these give the book's numbers: LC = {g32.LC}, QC = {g32.QC}, WC = {g32.WC}, and warp 6, the warp
          Fig. 15.10 shades, starts at row {w6.wRow}, column {w6.wCol}.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách đưa cách bố trí dưới dạng hình và để phần mã cho Bài tập 3. Các công thức này là của chúng tôi, viết để W
              cũng có thể là 64; lab 15.3 và mô hình đằng sau các hình của trang này dùng chúng.
            </p>
          }
        >
          <p>
            The book gives the layout as a figure and leaves the code to its Exercise 3. These formulas are ours, written so
            W can be 64 too; lab 15.3 and the model behind this page's pictures use them.
          </p>
        </Translated>
      </Beyond>
      <LanePicker />
      <Translated
        vi={
          <p>
            Hoạt hình sắp các bước theo thứ tự: lệnh ghi tệ, việc phân chia lại từng tầng một, cùng lệnh ghi đó được phát lại,
            và điều gì thay đổi với 64 lane.
          </p>
        }
      >
        <p>
          The animation puts the steps in order: the bad store, the re-partition one level at a time, the same store
          instruction replayed, and what changes with 64 lanes.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Vẽ từ cùng mô hình với bộ chọn lane và trình trực quan hóa: phép tính chỉ số, và sector 32 byte của NVIDIA cho
              mỗi lần một lệnh ghi. Không có cache, không có thời gian.
            </p>
          }
        >
          <p>
            Drawn from the same model as the lane picker and the visualiser: index arithmetic, and NVIDIA's 32-byte sectors
            for one store instruction at a time. No caches, no timing.
          </p>
        </Translated>
      </Simplification>
      <RemapScene />
      <Ask q={questionsS155['ch15/15.5/q-remap-sectors']} label={tr('Work it out', 'Tính ra')} />

      <h2>
        <Translated vi="Thanh ghi không thay đổi">Registers do not change</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Việc ánh xạ lại thay đổi luồng sở hữu những phần tử nào, không thay đổi bao nhiêu. 64 bộ tích lũy vẫn là 64 thanh
            ghi. Ở mỗi bước của vòng lặp trong, luồng cần một mảnh 4 giá trị của dải A cho mỗi hàng góc phần tư trong hai hàng
            của nó và một mảnh 4 giá trị của dải B cho mỗi cột góc phần tư trong hai cột của nó: 16 giá trị, cùng 8 + 8 như
            trước. Điều thay đổi là mã: <code>mm</code> và <code>writeTile</code> phải duyệt qua bốn ô con, và{' '}
            <code>writeTile</code> phát một float4 cho mỗi hàng ô con. Sách không đưa mã cho phần này. Nó là Bài tập 3, và
            nó là <a href={href('ch15', 'labs')}>lab 15.3</a>.
          </p>
        }
      >
        <p>
          The remap changes which elements a thread owns, not how many. The 64 accumulators are still 64 registers. In each
          step of the inner loop the thread needs a piece of 4 values of the A strip for each of its two quadrant rows and a
          piece of 4 values of the B strip for each of its two quadrant columns: 16 values, the same 8 + 8 as before. What
          does change is the code: <code>mm</code> and <code>writeTile</code> must walk over the four sub-tiles, and{' '}
          <code>writeTile</code> issues one float4 per sub-tile row. The book shows no code for this. It is Exercise 3, and
          it is <a href={href('ch15', 'labs')}>lab 15.3</a>.
        </p>
      </Translated>
      <Source section="15.5" pages="361" figs={tr('Fig. 15.10, right panel', 'Fig. 15.10, khung bên phải')} />
      <Ask q={questionsS155['ch15/15.5/q-registers']} />

      <h2>
        <Translated vi="Khi một warp không phải 32 lane">When a warp is not 32 lanes</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mọi con số đến giờ đều giả định 32 lane, như sách. Dưới HIP, wavefront của AMD là 64 lane trên CDNA (gfx9) và 32
            trên RDNA (gfx10 trở về sau). Trên Metal, độ rộng SIMD group là bất cứ giá trị nào compute pipeline báo cáo, không
            có giá trị được công bố.
          </p>
        }
      >
        <p>
          Every number so far assumed 32 lanes, as the book does. Under HIP, AMD wavefronts are 64 lanes on CDNA (gfx9) and
          32 on RDNA (gfx10 and later). On Metal the SIMD-group width is whatever the compute pipeline reports, with no
          documented value.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 (notes/today/ch15-facts.md B1, C1, D1; ch05-facts.md B5, C3)',
          '2026-11-20 (notes/today/ch15-facts.md, mục B1, C1, D1; ch05-facts.md, mục B5, C3)',
        )}
      >
        <Translated
          vi={
            <p>
              AMD: "Since warpSize can differ between devices, it can not be assumed to be a compile-time constant on the
              host" (vì warpSize có thể khác nhau giữa các thiết bị, không thể giả định nó là hằng số lúc biên dịch trên host).
              Apple: độ rộng được "returned by the threadExecutionWidth of your compute pipeline state object".
            </p>
          }
        >
          <p>
            AMD: "Since warpSize can differ between devices, it can not be assumed to be a compile-time constant on the
            host." Apple: the width is "returned by the threadExecutionWidth of your compute pipeline state object".
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Cách bố trí vẫn áp dụng được. Với 256 luồng, một khối CDNA là 4 wavefront. Giữ 8 hàng lane và bốn ô vuông 4 × 4
            mỗi lane: lane trở thành 8 × {g64.LC}, góc phần tư {g64.QR} × {g64.QC}, và các ô wavefront {g64.warpRows} ×{' '}
            {g64.warpCols}, xếp {g64.WR} × {g64.WC}. Một hàng lane giờ là {g64.LC} lane và ghi {g64.laneRowBytes} byte liên
            tiếp. Chuyển bộ chọn lane và trình trực quan hóa sang 64 lane để thấy điều đó.
          </p>
        }
      >
        <p>
          The layout carries over. With 256 threads a CDNA block is 4 wavefronts. Keep 8 lane rows and four 4 × 4 squares per
          lane: the lanes become 8 × {g64.LC}, the quadrants {g64.QR} × {g64.QC}, and the wavefront tiles {g64.warpRows} ×{' '}
          {g64.warpCols}, arranged {g64.WR} × {g64.WC}. A lane row is now {g64.LC} lanes and writes {g64.laneRowBytes}{' '}
          contiguous bytes. Switch the lane picker and the visualiser to 64 lanes to see it.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Cách bố trí 64 lane là suy ra của chúng tôi (ch15-facts B1). AMD công bố độ rộng, không công bố cách bố trí này.</p>}>
          <p>The 64-lane layout is our derivation (ch15-facts B1). AMD documents the width, not this layout.</p>
        </Translated>
      </Beyond>
      <Ask q={questionsS155['ch15/15.5/q-wave64']} />
      <Translated
        vi={
          <p>
            Cách bố trí là phép tính trên các hằng số: độ rộng góc phần tư, số ô warp mỗi hàng. Vì vậy độ rộng phải là hằng
            số lúc biên dịch, chọn cho khớp với thiết bị. AMD công bố hai cách: truy vấn kích thước warp trên host và tạo một
            instance của template kernel cho nó, hoặc, khi cần một hằng số phía device, định nghĩa nó theo target, 64 khi{' '}
            <code>__GFX8__</code> hoặc <code>__GFX9__</code> được định nghĩa và 32 trong trường hợp còn lại. Bản build HIP
            của llama.cpp làm cách thứ hai. Lab 15.3 làm cách thứ nhất: kernel là một template theo độ rộng, và host khởi chạy
            instance cho độ rộng mà thiết bị báo cáo. Trên Metal, đọc <code>threadExecutionWidth</code> từ pipeline trên host
            và biên dịch kernel cho nó, đó là điều host Metal của lab 15.3 làm; MLX và llama.cpp đơn giản giả định 32.
          </p>
        }
      >
        <p>
          The layout is arithmetic on constants: the quadrant width, the warp tiles per row. So the width has to be a
          compile-time constant, chosen to match the device. AMD documents two ways: query the warp size on the host and
          instantiate a kernel template for it, or, where a device-side constant is needed, define it from the target, 64
          when <code>__GFX8__</code> or <code>__GFX9__</code> is defined and 32 otherwise. llama.cpp's HIP build does the
          second. Lab 15.3 does the first: the kernel is a template over the width, and the host launches the instance for
          the width the device reports. On Metal, read <code>threadExecutionWidth</code> from the pipeline on the host and
          compile the kernel for it, which is what lab 15.3's Metal host does; MLX and llama.cpp simply assume 32.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch15-facts.md B1, C1, D1)', '2026-11-20 (notes/today/ch15-facts.md, mục B1, C1, D1)')}>
        <Translated
          vi={
            <p>
              Cách AMD khuyến nghị là "to determine the warp size of the GPU on host side and setup the kernel accordingly"
              (xác định kích thước warp của GPU ở phía host và thiết lập kernel tương ứng), hoặc một <code>#define</code> dưới{' '}
              <code>#if defined(__GFX8__) || defined(__GFX9__)</code>. llama.cpp:{' '}
              <code>ggml_cuda_get_physical_warp_size()</code> trả về 64 dưới phép kiểm tra đó, nếu không thì 32. MLX:{' '}
              <code>tgp_size = WM * WN * 32</code>.
            </p>
          }
        >
          <p>
            AMD's recommended approach is "to determine the warp size of the GPU on host side and setup the kernel
            accordingly", or a <code>#define</code> under <code>#if defined(__GFX8__) || defined(__GFX9__)</code>. llama.cpp:{' '}
            <code>ggml_cuda_get_physical_warp_size()</code> returns 64 under that test, else 32. MLX: <code>tgp_size = WM * WN * 32</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Dù backend nào, đừng bao giờ viết hằng 32 cứng cho độ rộng. Một cách bố trí xây từ một hằng số có tên có thể được
            dựng lại cho độ rộng khác; một hằng cứng giấu giả định cho đến khi mẫu ghi lặng lẽ xuống cấp.
          </p>
        }
      >
        <p>
          Whatever the backend, never write a literal 32 for the width. A layout built from a named constant can be rebuilt
          for another width; a literal hides the assumption until the store pattern quietly degrades.
        </p>
      </Translated>
      <Ask q={questionsS155['ch15/15.5/q-width-wrong']} />
      <Translated
        vi={
          <p>
            Câu trả lời đó đáng nhớ khi làm lab. Ánh xạ đúng với bất kỳ độ rộng nào nó được biên dịch, vì nó thuần là phép
            tính chỉ số: mỗi phần tử có một chủ sở hữu. Chỉ có sự hợp nhất truy cập phụ thuộc vào việc độ rộng đã biên dịch
            khớp với phần cứng. Đó là lý do lab 15.3 kiểm tra cả hai độ rộng so với tham chiếu CPU trên mọi backend, và lần
            khởi chạy của nó chọn độ rộng mà thiết bị của bạn báo cáo.
          </p>
        }
      >
        <p>
          That answer is worth keeping in mind for the labs. The mapping is correct for whatever width it was compiled with,
          because it is pure index arithmetic: each element has one owner. Only the coalescing depends on the compiled width
          matching the hardware. That is why lab 15.3 tests both widths against the CPU reference on every backend, and its
          launch picks the width your device reports.
        </p>
      </Translated>

      <h2>
        <Translated vi="Căn chỉnh">Alignment</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một lệnh ghi float4 viết 16 byte và cần một địa chỉ căn chỉnh 16 byte; sách không nhắc điều này. Hàng r của C bắt
            đầu sau phần tử đầu tiên của C 4·N·r byte, và trên CUDA phần tử đó được căn chỉnh: <code>cudaMalloc</code> trả về
            bộ nhớ căn chỉnh ít nhất 256 byte. Vì vậy khi N là bội số của 4, mọi hàng bắt đầu ở biên 16 byte và mọi hàng ô con
            có thể là một float4. Khi không phải, phần lớn các hàng thì không. Kernel quyết định một lần: nhánh vector khi{' '}
            <code>N % 4 == 0</code>, ghi vô hướng có kiểm tra biên trong trường hợp còn lại. Cùng tệp <code>.cu</code> trên
            HIP lấy C từ <code>hipMalloc</code>, mà tài liệu bộ nhớ của AMD không nêu căn chỉnh nào (ch15-facts B8), nên ở đó
            quy tắc là unverified.
          </p>
        }
      >
        <p>
          A float4 store writes 16 bytes and needs a 16-byte-aligned address; the book does not mention it. Row r of C starts
          4·N·r bytes after C's first element, and on CUDA that element is aligned: <code>cudaMalloc</code> returns memory
          aligned to at least 256 bytes. So when N is a multiple of 4, every row starts on a 16-byte boundary and every
          sub-tile row can be one float4. When it is not, most rows do not. The kernel decides once: the vector path when{' '}
          <code>N % 4 == 0</code>, bounds-checked scalar stores otherwise. The same <code>.cu</code> on HIP gets C from{' '}
          <code>hipMalloc</code>, for which AMD's memory reference states no alignment (ch15-facts B8), so there the rule
          is unverified.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Của chúng tôi, từ ch06-facts A3 (<code>cudaMalloc</code>: "at least 256 bytes"), A11 (NVIDIA: "These
              instructions require aligned data") và ghi chú về căn chỉnh của ch15-facts E1.
            </p>
          }
        >
          <p>
            Ours, from ch06-facts A3 (<code>cudaMalloc</code>: "at least 256 bytes"), A11 (NVIDIA: "These instructions
            require aligned data") and ch15-facts E1's alignment note.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Bản thân lệnh ghi là thành ngữ chương 6 của sách: ép kiểu <code>float*</code> sang <code>float4*</code> rồi gán.
            Không tài liệu tham chiếu nào của nhà cung cấp công bố phép ép kiểu đó; blog dành cho nhà phát triển của NVIDIA
            có trình bày nó, với quy tắc căn chỉnh ở trên. Metal có một cách thay thế được công bố, và lab 15.3 dùng nó: host
            gắn C lần thứ hai dưới dạng buffer <code>device float4*</code>, và kernel ghi trọn các phần tử float4 qua đó.
          </p>
        }
      >
        <p>
          The store itself is the book's chapter 6 idiom: cast the <code>float*</code> to a <code>float4*</code> and assign.
          No vendor reference documents that cast; NVIDIA's developer blog shows it, with the alignment rule above. Metal has a
          documented alternative, and lab 15.3 uses it: the host binds C a second time as a <code>device float4*</code>{' '}
          buffer, and the kernel stores whole float4 elements through it.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 (notes/today/ch06-facts.md A11, C3; infra-facts.md A6, C7)',
          '2026-11-20 (notes/today/ch06-facts.md, mục A11, C3; infra-facts.md, mục A6, C7)',
        )}
      >
        <Translated
          vi={
            <p>
              Blog của NVIDIA: các kiểu vector "enable the compiler to generate these wider memory operations through simple
              pointer casting with reinterpret_cast". Cách của Apple là một buffer khai báo <code>device float4*</code> (căn
              chỉnh 16 byte).
            </p>
          }
        >
          <p>
            NVIDIA's blog: vector types "enable the compiler to generate these wider memory operations through simple pointer
            casting with reinterpret_cast". Apple's route is a buffer declared <code>device float4*</code> (16-byte aligned).
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Đặt N thành 1020 trong trình trực quan hóa và duyệt qua các hàng của ô: các hàng lẻ bắt đầu lệch 16 byte vào trong
            một sector, nên mỗi đoạn 64 byte trải trên ba sector thay vì hai. Bội số của 4 là đủ để đúng; bội số của 8 giữ mọi
            đoạn ở biên sector. Sau đó đặt N thành 1001: các lệnh ghi lùi về vô hướng.
          </p>
        }
      >
        <p>
          Set N to 1020 in the visualiser and step through the tile rows: odd rows start 16 bytes into a sector, so each
          64-byte run spans three sectors instead of two. A multiple of 4 is enough to be correct; a multiple of 8 keeps every
          run on a sector boundary. Then set N to 1001: the stores fall back to scalars.
        </p>
      </Translated>
      <Ask q={questionsS155['ch15/15.5/q-align']} />

      <h2>
        <Translated vi="Cách khác: sửa các lệnh ghi về sau">The other way: fix the stores afterwards</Translated>
      </h2>
      <Translated
        vi={
          <p>
            CUTLASS, thư viện GEMM mã nguồn mở của NVIDIA, giải cùng vấn đề theo chiều ngược lại. Nó giữ ánh xạ luồng nào tốt
            nhất cho phép tính số học và thêm một <b>phần kết (epilogue)</b>: các luồng ghi kết quả vào bộ nhớ chia sẻ (shared
            memory), đồng bộ hóa, và đọc lại theo một mẫu được chọn cho các lệnh ghi toàn cục được hợp nhất. Kernel ma trận
            Metal của llama.cpp làm điều tương tự với các ô ở mép ma trận: nó đặt tạm chúng trong bộ nhớ threadgroup và chép
            ra bằng lệnh ghi float4, trong khi các ô đầy đủ đi thẳng vào bộ nhớ device. Khi bạn đọc một GEMM dùng trong sản
            xuất, hãy tìm một trong hai cách.
          </p>
        }
      >
        <p>
          CUTLASS, NVIDIA's open-source GEMM library, solves the same problem the other way round. It keeps whichever thread
          mapping is best for the arithmetic and adds an <b>epilogue</b>: threads write their results to shared memory,
          synchronise, and read them back in a pattern chosen for coalesced global stores. llama.cpp's Metal matrix kernel does
          the same for tiles at the edge of the matrix: it stages them in threadgroup memory and copies them out with float4
          stores, while full tiles go straight to device memory. When you read a production GEMM, look for one of the two.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch15-facts.md A12, D2)', '2026-11-20 (notes/today/ch15-facts.md, mục A12, D2)')}>
        <Translated
          vi={
            <p>
              CUTLASS: "The epilogue is a separate phase in which threads exchange data through shared memory then
              cooperatively access global memory using efficient striped access patterns." llama.cpp,{' '}
              <code>mul_mm.metal</code>: "if no bounds checks on the output are needed, we can directly write to device
              memory" (nếu không cần kiểm tra biên ở đầu ra, ta có thể ghi thẳng vào bộ nhớ device).
            </p>
          }
        >
          <p>
            CUTLASS: "The epilogue is a separate phase in which threads exchange data through shared memory then cooperatively
            access global memory using efficient striped access patterns." llama.cpp, <code>mul_mm.metal</code>: "if no bounds
            checks on the output are needed, we can directly write to device memory".
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS155['ch15/15.5/q-epilogue']} />

      <h2>
        <Translated vi="Việc ánh xạ lại tác động gì đến bộ nhớ chia sẻ">What the remap does to shared memory</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Thay đổi phần tử nào một luồng sở hữu cũng thay đổi những hàng nào của <code>A_s</code> và cột nào của{' '}
            <code>B_s</code> nó đọc ở mỗi bước. Ở cách bố trí §15.3, lần đọc dải A của một warp là xung đột bank (bank
            conflict) 2 chiều và lần đọc dải B là 4 chiều. Sau khi ánh xạ lại, lần đọc dải A là 8 chiều, và xung đột của B biến
            mất. Mục 15.6 loại bỏ xung đột 8 chiều bằng một cột đệm duy nhất:{' '}
            <a href={href('ch15', '15.6')}>loại bỏ xung đột bank</a>.
          </p>
        }
      >
        <p>
          Changing which elements a thread owns also changes which rows of <code>A_s</code> and columns of{' '}
          <code>B_s</code> it reads in each step. In the §15.3 layout a warp's A-strip read was a 2-way bank conflict and its
          B-strip read 4-way. After the remap the A-strip read is 8-way, and B's conflict is gone. Section 15.6 removes the
          8-way conflict with a single padding column: <a href={href('ch15', '15.6')}>eliminating bank conflicts</a>.
        </p>
      </Translated>
      <Beyond title={tr('Our arithmetic', 'Phép tính của chúng tôi')}>
        <Translated vi={<p>Các con số trước và sau lấy từ ch15-facts E1, với quy tắc 32 bank của NVIDIA; sách chỉ phân tích cách bố trí đã ánh xạ lại.</p>}>
          <p>The before-and-after counts are from ch15-facts E1, with NVIDIA's 32-bank rule; the book analyses only the remapped layout.</p>
        </Translated>
      </Beyond>
    </>
  );
}
