import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, num, realBackends, useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  AMD_CDNA_L2, BUS_PRESETS, DEMAND, FIG_6_9, banksNeeded, busBandwidth, busesFor, bytesInFlight, channelTimeline, deal,
  elementsToFill, phaseChannels, utilisationBound, type Bus, type Mapping,
} from '../../sim/dram';
import { ChannelDrawing, DramBanksScene } from './DramBanksScene';

const pct = (fraction: number) => Math.round(fraction * 1000) / 10;
const round2 = (v: number) => Math.round(v * 100) / 100;
const gb = (v: number) => round2(v).toLocaleString('en-US', { maximumFractionDigits: 2 });
const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

// ---------------------------------------------------------------- Fig. 6.7, redrawn
function ChannelsFigure() {
  const chW = 156;
  const gap = 20;
  const x0 = 26;
  const bankW = 30;
  return (
    <figure className="wide" style={{ maxWidth: '58rem', margin: '0.4rem 0 1.6rem' }}>
      <svg viewBox="0 0 720 214" role="img" aria-label={tr('A processor with four channels; each channel is a memory controller with its own bus, and each bus serves four banks', 'Một bộ xử lý với bốn kênh; mỗi kênh là một bộ điều khiển bộ nhớ với bus riêng, và mỗi bus phục vụ bốn bank')} style={{ width: '100%', maxWidth: 720, display: 'block', fontFamily: 'var(--sans)' }}>
        <rect x={10} y={146} width={700} height={60} rx={4} fill="var(--page)" stroke="var(--ink)" />
        <text x={20} y={200} fontSize="11.5" fontWeight="600"><Translated vi="bộ xử lý">processor</Translated></text>
        {[0, 1, 2, 3].map((c) => {
          const cx = x0 + c * (chW + gap);
          return (
            <g key={c}>
              <rect x={cx} y={154} width={chW} height={28} rx={3} fill="var(--global-wash)" stroke="var(--global)" />
              <text x={cx + chW / 2} y={172} fontSize="11.5" textAnchor="middle"><Translated vi={<>kênh {c}: bộ điều khiển</>}><>channel {c}: controller</></Translated></text>
              <line x1={cx + chW / 2} x2={cx + chW / 2} y1={154} y2={118} stroke="var(--global)" strokeWidth="5" />
              <text x={cx + chW / 2 + 8} y={140} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="bus">bus</Translated></text>
              <line x1={cx + 12} x2={cx + chW - 12} y1={118} y2={118} stroke="var(--global)" strokeWidth="3" />
              {[0, 1, 2, 3].map((b) => {
                const bx = cx + 12 + b * ((chW - 24 - bankW) / 3);
                return (
                  <g key={b}>
                    <line x1={bx + bankW / 2} x2={bx + bankW / 2} y1={118} y2={100} stroke="var(--global)" strokeWidth="1.5" />
                    <rect x={bx} y={40} width={bankW} height={60} rx={2} fill="var(--cell)" stroke="var(--global)" />
                    <text x={bx + bankW / 2} y={74} fontSize="10" textAnchor="middle">b{b}</text>
                  </g>
                );
              })}
            </g>
          );
        })}
        <text x={x0} y={28} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi="bank: mảng ô nhớ, bộ khuếch đại cảm biến, giao diện burst">banks: cell array, sense amplifiers, burst interface</Translated></text>
      </svg>
      <figcaption className="caption muted" style={{ fontFamily: 'var(--sans)', fontSize: '0.8rem' }}>
        <Translated vi="Cấu trúc của Hình 6.7 trong sách, vẽ lại: bốn kênh, mỗi kênh là một bộ điều khiển bộ nhớ với bus riêng, bốn bank trên mỗi bus. Thiết bị thật có nhiều hơn ở cả hai.">
          The structure of the book's Fig. 6.7, redrawn: four channels, each a memory controller with its own bus, four
          banks on each bus. Real devices have more of both.
        </Translated>
      </figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------- bus calculator
const PRESET_KEYS = ['ddr5', 'hbm3e'] as const;
const sameBus = (a: Bus, b: Bus) => a.widthBits === b.widthBits && a.clockGHz === b.clockGHz && a.ddr === b.ddr && a.buses === b.buses;

function BusCalculator() {
  const [bus, setBus] = useState<Bus>({ widthBits: 64, clockGHz: 3.2, ddr: true, buses: 1 });
  const [target, setTarget] = useState<number>(DEMAND.cpu);
  const per = busBandwidth(bus.widthBits, bus.clockGHz, bus.ddr);
  const total = busBandwidth(bus.widthBits, bus.clockGHz, bus.ddr, bus.buses);
  const need = busesFor(target, per);
  const isHbm = sameBus(bus, BUS_PRESETS.hbm3e);
  const field = (label: string, key: 'widthBits' | 'clockGHz' | 'buses', step: number, min: number) => (
    <label className="field">
      {label}
      <input
        type="number"
        step={step}
        min={min}
        value={bus[key]}
        aria-label={label}
        onChange={(e) => setBus({ ...bus, [key]: Math.max(min, Number(e.target.value) || min) })}
      />
    </label>
  );
  return (
    <Bench
      title={tr('Bus bandwidth', 'Băng thông bus')}
      task={tr(
        "Find how many DDR5 buses a CPU and a GPU need. Then load the book's HBM3E inputs.",
        'Tìm xem một CPU và một GPU cần bao nhiêu bus DDR5. Rồi nạp các đầu vào HBM3E của sách.',
      )}
      foot={tr(
        "Width × clock, × 2 for double data rate (the book's formula, p. 133; NVIDIA documents the same one). GB here is 10⁹ bytes. Peak figures: a bus delivers this only when it is never idle, which is the rest of this section.",
        'Độ rộng × xung nhịp, × 2 với tốc độ dữ liệu kép (công thức của sách, p. 133; NVIDIA công bố cùng công thức). GB ở đây là 10⁹ byte. Số liệu đỉnh: một bus chỉ đạt mức này khi nó không bao giờ rảnh, và đó là phần còn lại của mục này.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }} role="group" aria-label={tr('Presets', 'Thiết lập sẵn')}>
        <span className="muted">
          <Translated vi="Thiết lập sẵn">Presets</Translated>
        </span>
        {PRESET_KEYS.map((k) => (
          <button key={k} className="quiet" data-preset={k} aria-pressed={sameBus(bus, BUS_PRESETS[k])}
            onClick={() => { const p = BUS_PRESETS[k]; setBus({ widthBits: p.widthBits, clockGHz: p.clockGHz, ddr: p.ddr, buses: p.buses }); }}>
            <Translated vi={{ ddr5: 'DDR5-6400 (ví dụ của sách)', hbm3e: 'một stack HBM3E (đầu vào của sách)' }[k]}>
              {BUS_PRESETS[k].label}
            </Translated>
          </button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '1rem', alignItems: 'flex-end' }}>
        {field(tr('Bus width (bits)', 'Độ rộng bus (bit)'), 'widthBits', 8, 1)}
        {field(tr('Bus clock (GHz)', 'Xung nhịp bus (GHz)'), 'clockGHz', 0.1, 0.1)}
        <label className="field">
          <Translated vi="Số lần truyền mỗi xung nhịp">Transfers per clock</Translated>
          <span className="seg">
            <button aria-pressed={!bus.ddr} onClick={() => setBus({ ...bus, ddr: false })}>
              <Translated vi="1 (đơn)">1 (single)</Translated>
            </button>
            <button aria-pressed={bus.ddr} onClick={() => setBus({ ...bus, ddr: true })}>
              <Translated vi="2 (tốc độ dữ liệu kép)">2 (double data rate)</Translated>
            </button>
          </span>
        </label>
        {field(tr('Number of buses', 'Số bus'), 'buses', 1, 1)}
      </div>
      <div className="cols">
        <div className="panel" aria-live="polite">
          <p style={{ margin: '0 0 0.4rem' }}>
            <Translated
              vi={
                <>
                  Một bus: {bus.widthBits / 8} byte × {bus.ddr ? '2 × ' : ''}
                  {bus.clockGHz} GHz = <b><output data-out="per" data-value={round2(per)}>{gb(per)}</output> GB/s</b>
                </>
              }
            >
              <>
                One bus: {bus.widthBits / 8} bytes × {bus.ddr ? '2 × ' : ''}
                {bus.clockGHz} GHz = <b><output data-out="per" data-value={round2(per)}>{gb(per)}</output> GB/s</b>
              </>
            </Translated>
          </p>
          <p style={{ fontSize: '1.6rem', fontWeight: 600, margin: '0.3rem 0' }}>
            <output data-out="total" data-value={round2(total)}>{gb(total)}</output> GB/s
          </p>
          <p className="muted" style={{ margin: 0 }}>
            <Translated vi={<>từ {bus.buses} bus</>}>
              <>from {plural(bus.buses, 'bus', 'buses')}</>
            </Translated>
          </p>
          {isHbm ? (
            <p style={{ margin: '0.7rem 0 0' }} data-out="erratum">
              <Translated
                vi={
                  <>
                    <span className="chip fail">sách in 1 TB/s</span> cho các đầu vào này (p. 133), và "4.9 GHz/s" cho xung
                    nhịp. Chính công thức của sách cho ra {gb(total)} GB/s, khoảng 1.25 TB/s, và đó là một stack HBM.
                  </>
                }
              >
                <>
                  <span className="chip fail">the book prints 1 TB/s</span> for these inputs (p. 133), and "4.9 GHz/s" for the
                  clock. Its own formula gives {gb(total)} GB/s, about 1.25 TB/s, and that is one HBM stack.
                </>
              </Translated>
            </p>
          ) : null}
        </div>
        <div className="panel">
          <label className="field">
            <Translated vi="Băng thông bạn cần (GB/s)">Bandwidth you need (GB/s)</Translated>
            <input type="number" min={1} value={target} aria-label={tr('Bandwidth you need', 'Băng thông bạn cần')} onChange={(e) => setTarget(Math.max(1, Number(e.target.value) || 1))} />
          </label>
          <div className="row" style={{ margin: '0.5rem 0' }}>
            <button className="quiet" data-target="cpu" aria-pressed={target === DEMAND.cpu} onClick={() => setTarget(DEMAND.cpu)}>
              <Translated vi="200: một CPU (p. 133)">200: a CPU (p. 133)</Translated>
            </button>
            <button className="quiet" data-target="gpu" aria-pressed={target === DEMAND.gpu} onClick={() => setTarget(DEMAND.gpu)}>
              <Translated vi="1000: một GPU (p. 133)">1000: a GPU (p. 133)</Translated>
            </button>
          </div>
          <p style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  {target} / {gb(per)} = {fmt(target / per)}, nên cần{' '}
                  <b><output data-out="need">{need}</output> bus</b> loại này.
                </>
              }
            >
              <>
                {target} / {gb(per)} = {fmt(target / per)}, so{' '}
                <b><output data-out="need">{need}</output> {need === 1 ? 'bus' : 'buses'}</b> of this kind.
              </>
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- channel and banks explorer
function ChannelExplorer() {
  const [R, setR] = useState(20);
  const [banks, setBanks] = useState(1);
  const [share, setShare] = useState(0);
  const [seed, setSeed] = useState(1);
  const [chan, setChan] = useState(16);
  const tl = useMemo(() => channelTimeline({ R, banks, conflictShare: share / 100, seed }), [R, banks, share, seed]);
  const span = Math.max(43, Math.min(124, 3 * (R + 1) + 1));
  const bound = utilisationBound(R, banks);
  const need = banksNeeded(R);
  const conflicts = tl.accesses.filter((a) => a.conflict).length;
  const book = R === 20 && banks === 1 && share === 0 && chan === 16;
  const slider = (name: string, value: number, set: (v: number) => void, min: number, max: number, step = 1, unit = '') => (
    <label className="field">
      {name} = {value}
      {unit}
      <input type="range" min={min} max={max} step={step} value={value} aria-label={name} style={{ width: '11rem' }} onChange={(e) => set(+e.target.value)} />
    </label>
  );
  return (
    <Bench
      title={tr('One channel, its banks, in time', 'Một kênh, các bank của nó, theo thời gian')}
      task={tr(
        "Start at the book's case. Add banks until the bus is full; then send some requests to a busy bank.",
        'Bắt đầu từ trường hợp của sách. Thêm bank cho tới khi bus đầy; rồi gửi một số yêu cầu tới một bank đang bận.',
      )}
      foot={tr(
        'Time runs left to right in units of one burst. Each request waits R units in its bank, then sends one burst. The percentage is measured once the start-up (the first latency, which nothing hides) has passed.',
        'Thời gian chạy từ trái sang phải theo đơn vị một burst. Mỗi yêu cầu chờ R đơn vị trong bank của nó, rồi gửi một burst. Tỉ lệ phần trăm được đo sau khi giai đoạn khởi động (độ trễ đầu tiên, không gì che được) đã qua.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        {slider(tr('Latency-to-burst ratio R', 'Tỉ lệ độ trễ trên burst R'), R, setR, 1, 40)}
        {slider(tr('Banks', 'Số bank'), banks, setBanks, 1, 48)}
        {slider(tr('Requests sent to a busy bank', 'Yêu cầu gửi tới một bank đang bận'), share, setShare, 0, 50, 5, '%')}
        <label className="field">
          <Translated vi="Bus của kênh (GB/s)">Channel bus (GB/s)</Translated>
          <input type="number" min={1} value={chan} aria-label={tr('Channel bus bandwidth', 'Băng thông bus của kênh')} onChange={(e) => setChan(Math.max(1, Number(e.target.value) || 1))} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '1rem' }}>
        <button className="quiet" data-preset="book" aria-pressed={book} onClick={() => { setR(20); setBanks(1); setShare(0); setChan(16); }}>
          <Translated vi="Trường hợp của sách: R = 20, một bank, 16 GB/s">The book's case: R = 20, one bank, 16 GB/s</Translated>
        </button>
        <button className="quiet" data-preset="full" aria-pressed={banks === need && share === 0} onClick={() => { setBanks(need); setShare(0); }}>
          <Translated vi="R + 1 bank">R + 1 banks</Translated>
        </button>
        {share > 0 ? (
          <button className="quiet" onClick={() => setSeed(seed + 1)}>
            <Translated vi="Rút các xung đột khác">Draw other conflicts</Translated>
          </button>
        ) : null}
      </div>
      <div className="panel" style={{ marginBottom: '0.9rem', overflowX: 'auto' }}>
        <ChannelDrawing minWidth={540} tl={tl} span={span} label={tr(`Timeline of ${banks} banks and the bus over ${span} burst times`, `Dòng thời gian của ${banks} bank và bus qua ${span} thời gian burst`)} data={{ util: pct(tl.utilisation) }} />
      </div>
      <div className="cols">
        <div className="panel" aria-live="polite">
          <p className="muted" style={{ margin: 0 }}>
            <Translated vi="Bus bận">Bus busy</Translated>
          </p>
          <p style={{ fontSize: '2rem', fontWeight: 600, margin: '0 0 0.2rem', lineHeight: 1.1 }}>
            <output data-out="util" data-value={pct(tl.utilisation)}>{pct(tl.utilisation)}%</output>
          </p>
          <p style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  cung cấp <b><output data-out="gbs" data-value={round2(tl.utilisation * chan)}>{gb(tl.utilisation * chan)}</output> GB/s</b> trên{' '}
                  {chan} của bus
                </>
              }
            >
              <>
                delivers <b><output data-out="gbs" data-value={round2(tl.utilisation * chan)}>{gb(tl.utilisation * chan)}</output> GB/s</b> of the bus's {chan}
              </>
            </Translated>
          </p>
        </div>
        <div className="panel">
          <p style={{ margin: '0 0 0.6rem' }}>
            <Translated
              vi={
                <>
                  Cận của sách, min(1, banks / (R + 1)) = min(1, {banks} / {R + 1}) ={' '}
                  <b><output data-out="bound" data-value={pct(bound)}>{pct(bound)}%</output></b>
                </>
              }
            >
              <>
                The book's bound, min(1, banks / (R + 1)) = min(1, {banks} / {R + 1}) ={' '}
                <b><output data-out="bound" data-value={pct(bound)}>{pct(bound)}%</output></b>
              </>
            </Translated>
          </p>
          <p style={{ margin: '0 0 0.6rem' }}>
            <Translated vi={<>Số bank lấp đầy bus: R + 1 = <b><output data-out="need">{need}</output></b></>}>
              <>Banks that fill the bus: R + 1 = <b><output data-out="need">{need}</output></b></>
            </Translated>
          </p>
          <p style={{ margin: 0 }} data-out="verdict">
            {conflicts > 0 ? (
              <Translated
                vi={
                  <>
                    <b>{conflicts} xung đột</b> trong {tl.accesses.length} yêu cầu. Mỗi yêu cầu đã phải chờ một bank vẫn còn
                    bận, mọi thứ phía sau nó cũng chờ theo, và bus ngồi không trong lúc đó: thấp hơn cận{' '}
                    {pct(bound - tl.utilisation)} điểm phần trăm.
                  </>
                }
              >
                <>
                  <b>{plural(conflicts, 'conflict')}</b> in {tl.accesses.length} requests. Each one waited for a bank that
                  was still busy, everything behind it waited too, and the bus sat idle meanwhile: {pct(bound - tl.utilisation)}{' '}
                  points below the bound.
                </>
              </Translated>
            ) : banks < need ? (
              <Translated
                vi={
                  <>
                    <b>Thiếu {need - banks} bank.</b> Bus rảnh {pct(1 - tl.utilisation)}% thời gian, chờ một bank nào đó hết
                    độ trễ.
                  </>
                }
              >
                <>
                  <b>{plural(need - banks, 'bank')} short.</b> The bus is idle {pct(1 - tl.utilisation)}% of the time,
                  waiting for some bank to finish its latency.
                </>
              </Translated>
            ) : banks === need ? (
              <Translated
                vi={
                  <>
                    <b>Vừa đủ.</b> Khi một burst kết thúc, độ trễ của bank kế tiếp cũng vừa xong.
                  </>
                }
              >
                <>
                  <b>Exactly enough.</b> As one burst ends, the next bank's latency is over.
                </>
              </Translated>
            ) : (
              <Translated
                vi={
                  <>
                    <b>Nhiều hơn mức cần {banks - need} bank.</b> Không có lợi gì trong mô hình này. Trong một kênh thật, các
                    bank thừa làm giảm khả năng hai yêu cầu đến cùng lúc muốn cùng một bank; ở đây khả năng đó là thanh trượt
                    xung đột.
                  </>
                }
              >
                <>
                  <b>{plural(banks - need, 'bank')} more than needed.</b> No gain in this model. In a real channel the
                  extra banks make it less likely that two requests arriving together want the same bank; here that
                  likelihood is the conflict slider.
                </>
              </Translated>
            )}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the dealer
const elems = (cell: number[]): string => {
  const runs: [number, number][] = [];
  for (const i of cell) {
    const last = runs[runs.length - 1];
    if (last && last[1] === i - 1) last[1] = i;
    else runs.push([i, i]);
  }
  const show = runs.slice(0, 3).map(([a, b]) => (a === b ? `M[${a}]` : `M[${a}..${b}]`));
  return runs.length > 3 ? tr(`${show.join(', ')} and ${runs.length - 3} more runs`, `${show.join(', ')} và ${runs.length - 3} dải nữa`) : show.join(', ');
};
const sameMap = (a: Mapping, b: Mapping) => a.elemBytes === b.elemBytes && a.burstBytes === b.burstBytes && a.channels === b.channels && a.banks === b.banks;

function Dealer() {
  const [m, setM] = useState<Mapping>({ ...FIG_6_9 });
  const [n, setN] = useState(16);
  const cells = useMemo(() => deal(n, m), [n, m]);
  const fill = elementsToFill(m);
  const usedChannels = cells.filter((ch) => ch.some((b) => b.length > 0)).length;
  const usedCells = cells.flat().filter((c) => c.length > 0).length;
  const seg = (label: string, key: keyof Mapping, values: number[], unit = '') => (
    <label className="field">
      {label}
      <span className="seg">
        {values.map((v) => (
          <button key={v} aria-pressed={m[key] === v} onClick={() => setM({ ...m, [key]: v })}>
            {v}
            {unit}
          </button>
        ))}
      </span>
    </label>
  );
  return (
    <Bench
      title={tr('Dealing an array to channels and banks', 'Chia một mảng cho các kênh và bank')}
      task={tr(
        "Reproduce Fig. 6.9. Then load AMD's interleave and find how big an array must be before it uses every channel.",
        'Tái tạo Hình 6.9. Rồi nạp cách xen kẽ của AMD và tìm xem một mảng phải lớn đến đâu mới dùng hết mọi kênh.',
      )}
      foot={tr(
        "Consecutive bursts go to consecutive channels; after the last channel the next round moves to the next bank (the book's rule, p. 135). The array starts at channel 0, bank 0.",
        'Các burst liên tiếp đi tới các kênh liên tiếp; sau kênh cuối cùng, vòng kế tiếp chuyển sang bank kế tiếp (quy tắc của sách, p. 135). Mảng bắt đầu ở kênh 0, bank 0.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }} role="group" aria-label={tr('Presets', 'Thiết lập sẵn')}>
        <span className="muted">
          <Translated vi="Thiết lập sẵn">Presets</Translated>
        </span>
        <button className="quiet" data-preset="fig69" aria-pressed={sameMap(m, FIG_6_9)} onClick={() => { setM({ ...FIG_6_9 }); setN(16); }}>
          <Translated vi="Hình 6.9: burst 8 byte, 4 kênh, 2 bank">Fig. 6.9: 8-byte bursts, 4 channels, 2 banks</Translated>
        </button>
        <button className="quiet" data-preset="amd" aria-pressed={sameMap(m, AMD_CDNA_L2)} onClick={() => setM({ ...AMD_CDNA_L2 })}>
          <Translated vi="AMD CDNA: xen kẽ 256 byte trên 32 kênh L2">AMD CDNA: 256-byte interleave across 32 L2 channels</Translated>
        </button>
      </div>
      <div className="row" style={{ marginBottom: '1rem', alignItems: 'flex-end' }}>
        {seg(tr('Element size', 'Kích thước phần tử'), 'elemBytes', [2, 4, 8], ' B')}
        <label className="field">
          <Translated vi="Số byte mỗi lượt kênh (burst)">Bytes per channel turn (burst)</Translated>
          <select value={m.burstBytes} aria-label={tr('Bytes per channel turn', 'Số byte mỗi lượt kênh')} onChange={(e) => setM({ ...m, burstBytes: +e.target.value })}>
            {[8, 16, 32, 64, 128, 256].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Số kênh">Channels</Translated>
          <select value={m.channels} aria-label={tr('Channels', 'Số kênh')} onChange={(e) => setM({ ...m, channels: +e.target.value })}>
            {[1, 2, 4, 8, 16, 32].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        {seg(tr('Banks per channel', 'Số bank mỗi kênh'), 'banks', [1, 2, 4])}
        <label className="field">
          <Translated vi="Độ dài mảng (phần tử)">Array length (elements)</Translated>
          <select value={n} aria-label={tr('Array length', 'Độ dài mảng')} onChange={(e) => setN(+e.target.value)}>
            {[8, 16, 32, 64, 256, 1024, 2048, 4096].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
      </div>
      <div style={{ display: 'grid', gap: '0.5rem', gridTemplateColumns: `repeat(auto-fill, minmax(${m.channels > 8 ? 7.5 : 10.5}rem, 1fr))`, marginBottom: '0.9rem' }}>
        {cells.map((ch, c) => (
          <div key={c} className="panel" data-channel={c} style={{ padding: '0.45rem 0.6rem', borderColor: ch.some((b) => b.length) ? 'var(--global)' : 'var(--rule)' }}>
            <div className="mx-label">
              <Translated vi={<>kênh {c}</>}>
                <>channel {c}</>
              </Translated>
            </div>
            {ch.map((cell, b) => (
              <div key={b} data-cell={`${c},${b}`} data-count={cell.length} style={{ fontSize: '0.74rem', color: cell.length ? 'var(--ink)' : 'var(--ink-3)' }}>
                {m.banks > 1 ? <span className="muted">bank {b}: </span> : null}{/* i18n-ignore: "bank" is kept in Vietnamese */}
                {cell.length ? elems(cell) : tr('nothing', 'không có gì')}
              </div>
            ))}
          </div>
        ))}
      </div>
      <p style={{ margin: 0 }} aria-live="polite">
        <Translated
          vi={
            <>
              {n} phần tử {m.elemBytes} byte ({n * m.elemBytes} byte) chạm vào{' '}
              <b><output data-out="channels">{usedChannels}</output> trên {m.channels} kênh</b>
              {m.banks > 1 ? <> và {usedCells} trên {m.channels * m.banks} cặp kênh-bank</> : null}. Mọi kênh nhận được một
              lượt trọn vẹn từ <output data-out="fill-channels">{fill.channels}</output> phần tử ({fill.channels * m.elemBytes}{' '}
              byte){m.banks > 1 ? <>, mọi bank của mọi kênh từ {fill.banks}</> : null}.
            </>
          }
        >
          <>
            {n} elements of {m.elemBytes} bytes ({n * m.elemBytes} bytes) touch{' '}
            <b><output data-out="channels">{usedChannels}</output> of {m.channels} channels</b>
            {m.banks > 1 ? <> and {usedCells} of {m.channels * m.banks} channel-bank pairs</> : null}. Every channel gets a whole
            turn from <output data-out="fill-channels">{fill.channels}</output> elements ({fill.channels * m.elemBytes} bytes)
            {m.banks > 1 ? <>, every bank of every channel from {fill.banks}</> : null}.
          </>
        </Translated>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the toy multiplication, phase by phase
function ToyReplay() {
  const { progress } = useProgress();
  const unlocked = progress.answers['ch06/6.2/p-8x8'] != null;
  const [width, setWidth] = useState(4);
  const [ph, setPh] = useState(0);
  const phases = useMemo(() => phaseChannels(width, 2, FIG_6_9), [width]);
  const cur = phases[Math.min(ph, phases.length - 1)];
  const loaded = new Set(cur.blocks.flatMap((b) => b.elements));
  const tiles = width / 2;
  const rows = Array.from({ length: tiles }, (_, r) => cur.blocks.filter((b) => b.row === r));
  const chOf = (i: number) => Math.floor((i * FIG_6_9.elemBytes) / FIG_6_9.burstBytes) % FIG_6_9.channels;
  return (
    <Bench
      title={tr('The tiled multiplication, phase by phase', 'Phép nhân phân ô, từng pha một')}
      task={
        unlocked
          ? tr(
              'Step through the phases at both sizes and count the channels each phase keeps busy.',
              'Đi qua từng pha ở cả hai kích thước và đếm số kênh mỗi pha giữ bận.',
            )
          : tr(
              'Step through the 4 × 4 case. The 8 × 8 case opens once you commit the prediction below.',
              'Đi qua từng bước trường hợp 4 × 4. Trường hợp 8 × 8 mở ra khi bạn chốt dự đoán bên dưới.',
            )
      }
      foot={tr(
        "Figs. 6.10 and 6.11's setting: row-major M, 2 × 2 tiles, every block running at once, Fig. 6.9's mapping (2-element bursts, 4 channels, 2 banks). Only M's loads are traced, as in the book. The subscript in each cell is its channel.",
        'Bối cảnh của Hình 6.10 và 6.11: M thứ tự hàng, ô 2 × 2, mọi khối chạy cùng lúc, ánh xạ của Hình 6.9 (burst 2 phần tử, 4 kênh, 2 bank). Chỉ các phép nạp M được theo dõi, như trong sách. Chỉ số dưới trong mỗi ô là kênh của nó.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Ma trận">Matrix</Translated>
          <span className="seg">
            <button aria-pressed={width === 4} onClick={() => { setWidth(4); setPh(0); }}>4 × 4</button>
            <button aria-pressed={width === 8} disabled={!unlocked} title={unlocked ? undefined : tr('Commit your prediction below first', 'Hãy chốt dự đoán bên dưới trước')} onClick={() => { setWidth(8); setPh(0); }}>8 × 8</button>
          </span>
        </label>
        <label className="field">
          <Translated vi="Pha">Phase</Translated>
          <span className="seg">
            {phases.map((p) => (
              <button key={p.ph} aria-pressed={p.ph === cur.ph} onClick={() => setPh(p.ph)}>
                <Translated vi={<>pha {p.ph}</>}>
                  <>phase {p.ph}</>
                </Translated>
              </button>
            ))}
          </span>
        </label>
      </div>
      <div className="cols">
        <div>
          <div className="mx-label">
            <Translated vi={<>M ({width} × {width}, thứ tự hàng; các phép nạp của pha này được tô)</>}>
              <>M ({width} × {width}, row-major; this phase's loads filled)</>
            </Translated>
          </div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${width}, auto)` }} data-out="grid">
            {Array.from({ length: width * width }, (_, i) => (
              <span key={i} className={loaded.has(i) ? 'hot' : ''} data-i={i}>
                {i}
                <sub>c{chOf(i)}</sub>
              </span>
            ))}
          </div>
        </div>
        <div className="panel" aria-live="polite">
          <p style={{ margin: '0 0 0.5rem' }}>
            <Translated
              vi={
                <>
                  <b>Pha {cur.ph}</b>: các kênh bận{' '}
                  <output data-out="phase-channels" data-value={cur.channels.join(',')}>{cur.channels.join(' và ')}</output>
                  {' '}({cur.channels.length} trên {FIG_6_9.channels})
                </>
              }
            >
              <>
                <b>Phase {cur.ph}</b>: channels busy{' '}
                <output data-out="phase-channels" data-value={cur.channels.join(',')}>{cur.channels.join(' and ')}</output>
                {' '}({cur.channels.length} of {FIG_6_9.channels})
              </>
            </Translated>
          </p>
          <div className="row" style={{ marginBottom: '0.6rem', gap: '0.3rem' }}>
            {Array.from({ length: FIG_6_9.channels }, (_, c) => (
              <span key={c} className={'chip' + (cur.channels.includes(c) ? ' global' : '')} data-lit={cur.channels.includes(c) ? 'yes' : 'no'}>
                <Translated vi={<>kênh {c}</>}>
                  <>channel {c}</>
                </Translated>
              </span>
            ))}
          </div>
          <div style={{ overflowX: 'auto' }}>
          <table className="threads">
            <tbody>
              {rows.map((bs) => (
                <tr key={bs[0].row}>
                  <td>
                    {bs.length === 2
                      ? tr(`Blocks ${bs[0].row},0 and ${bs[0].row},1`, `Khối ${bs[0].row},0 và ${bs[0].row},1`)
                      : tr(`Blocks ${bs[0].row},0 to ${bs[0].row},${bs.length - 1}`, `Khối ${bs[0].row},0 đến ${bs[0].row},${bs.length - 1}`)}
                  </td>
                  <td>{bs[0].elements.map((e) => `M[${e}]`).join(', ')}</td>
                  <td>
                    <Translated vi={<>kênh {bs[0].channels.join(', ')}</>}>
                      <>ch. {bs[0].channels.join(', ')}</>
                    </Translated>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
          <p className="muted" style={{ margin: '0.5rem 0 0' }}>
            <Translated vi="Các khối trong cùng một hàng ô nạp cùng các phần tử M; tên khối của sách ghi hàng ô trước.">
              Blocks in the same row of tiles load the same M elements; the book's block names give the tile row first.
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- bytes in flight (ours)
const BOOK_BW = {
  key: 'book',
  get label() {
    return tr("The book's example, H100 SXM: 3.35 TB/s (p. 94)", 'Ví dụ của sách, H100 SXM: 3.35 TB/s (p. 94)');
  },
  gbs: 3350,
};

function BytesInFlight() {
  const { reports } = useReports();
  const measured = useMemo(
    () =>
      realBackends(reports).flatMap((b) => {
        const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
        const v = num(r?.bandwidth_gb_per_s);
        return v
          ? [
              {
                key: b,
                get label() {
                  return tr(
                    `Yours: ${String(reports[b].device.name ?? b)} (${b}), lab 5.1: ${fmt(v)} GB/s`,
                    `Của bạn: ${String(reports[b].device.name ?? b)} (${b}), lab 5.1: ${fmt(v)} GB/s`,
                  );
                },
                gbs: v,
              },
            ]
          : [];
      }),
    [reports],
  );
  const options = [...measured, BOOK_BW];
  const [key, setKey] = useState<string | null>(null);
  const [custom, setCustom] = useState(500);
  const [lat, setLat] = useState(50);
  const [bytes, setBytes] = useState(4);
  const [warp, setWarp] = useState(32);
  const chosen = key ?? options[0].key;
  const bw = chosen === 'custom' ? custom : (options.find((o) => o.key === chosen) ?? BOOK_BW).gbs;
  const f = bytesInFlight(bw, lat, bytes, warp);
  const wider = bytesInFlight(bw, lat, 16, warp);
  return (
    <Bench
      title={tr("Bytes in flight (ours, not the book's)", 'Số byte đang bay (của chúng tôi, không phải của sách)')}
      task={tr(
        'Pick a bandwidth and a latency, then see how many loads must be outstanding at every moment. Switch to 16-byte loads.',
        'Chọn một băng thông và một độ trễ, rồi xem cần bao nhiêu phép nạp đang chờ ở mọi thời điểm. Chuyển sang phép nạp 16 byte.',
      )}
      foot={tr(
        "Little's law: bytes outstanding = bandwidth × latency. One GB/s is one byte per nanosecond. The latency is an input: no vendor documents one in nanoseconds; NVIDIA describes global-memory latency as hundreds of clock cycles (facts A1).",
        'Định luật Little: số byte đang chờ = băng thông × độ trễ. Một GB/s là một byte mỗi nano giây. Độ trễ là một đầu vào: không nhà cung cấp nào công bố nó theo nano giây; NVIDIA mô tả độ trễ bộ nhớ toàn cục là hàng trăm chu kỳ xung nhịp (facts A1).',
      )}
    >
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Băng thông cần duy trì">Bandwidth to sustain</Translated>
          <select value={chosen} aria-label={tr('Bandwidth to sustain', 'Băng thông cần duy trì')} onChange={(e) => setKey(e.target.value)}>
            {options.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
            <option value="custom">{tr('Custom', 'Tùy chỉnh')}</option>
          </select>
        </label>
        {chosen === 'custom' ? (
          <label className="field">
            GB/s
            <input type="number" min={1} value={custom} aria-label={tr('Custom bandwidth', 'Băng thông tùy chỉnh')} onChange={(e) => setCustom(Math.max(1, Number(e.target.value) || 1))} />
          </label>
        ) : null}
        <label className="field">
          <Translated vi={<>Độ trễ = {lat} ns</>}>
            <>Latency = {lat} ns</>
          </Translated>
          <input type="range" min={10} max={1000} step={10} value={lat} aria-label={tr('Latency in nanoseconds', 'Độ trễ tính bằng nano giây')} style={{ width: '11rem' }} onChange={(e) => setLat(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Số byte mỗi phép nạp (mỗi luồng)">Bytes per load (per thread)</Translated>
          <span className="seg">
            {[4, 8, 16].map((b) => (
              <button key={b} aria-pressed={bytes === b} onClick={() => setBytes(b)}>
                {b === 4 ? '4 (float)' : b === 8 ? '8 (float2)' : '16 (float4)'}{/* i18n-ignore: sizes and type names */}
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <span className="seg">
            {[32, 64].map((w) => (
              <button key={w} aria-pressed={warp === w} onClick={() => setWarp(w)}>{w}</button>
            ))}
          </span>
        </label>
      </div>
      <div className="cols">
        <div className="panel" aria-live="polite">
          <p style={{ margin: 0 }}>
            {fmt(bw)} GB/s × {lat} ns ={/* i18n-ignore: formula with units */}
          </p>
          <p style={{ fontSize: '1.6rem', fontWeight: 600, margin: '0.2rem 0' }}>
            <Translated
              vi={
                <>
                  <output data-out="bytes" data-value={Math.round(f.bytes)}>{fmt(Math.round(f.bytes))}</output> byte
                </>
              }
            >
              <>
                <output data-out="bytes" data-value={Math.round(f.bytes)}>{fmt(Math.round(f.bytes))}</output> bytes
              </>
            </Translated>
          </p>
          <p className="muted" style={{ margin: 0 }}>
            <Translated vi={<>khoảng {fmt(f.bytes / 1024)} KiB đang trên đường đi, mọi lúc</>}>
              <>about {fmt(f.bytes / 1024)} KiB on their way, all the time</>
            </Translated>
          </p>
        </div>
        <div className="panel">
          <p style={{ margin: '0 0 0.5rem' }}>
            <Translated
              vi={
                <>
                  Tức là <b><output data-out="loads" data-value={Math.round(f.loads)}>{fmt(Math.round(f.loads))}</output></b> phép
                  nạp {bytes} byte đang chờ, hay{' '}
                  <b><output data-out="warps" data-value={Math.round(f.warpLoads)}>{fmt(Math.round(f.warpLoads))}</output></b>{' '}
                  lệnh nạp toàn warp gồm {warp} lane.
                </>
              }
            >
              <>
                That is <b><output data-out="loads" data-value={Math.round(f.loads)}>{fmt(Math.round(f.loads))}</output></b> loads of {bytes} bytes
                outstanding, or <b><output data-out="warps" data-value={Math.round(f.warpLoads)}>{fmt(Math.round(f.warpLoads))}</output></b>{' '}
                warp-wide load instructions of {warp} lanes.
              </>
            </Translated>
          </p>
          <p style={{ margin: 0 }} className="muted">
            {bytes === 16
              ? tr(
                  'With 16 bytes per lane each instruction carries four times what a float load carries.',
                  'Với 16 byte mỗi lane, mỗi lệnh chở gấp bốn lần một phép nạp float.',
                )
              : tr(
                  `With 16-byte loads it would take ${fmt(Math.round(wider.warpLoads))} warp-wide instructions instead.`,
                  `Với phép nạp 16 byte thì chỉ cần ${fmt(Math.round(wider.warpLoads))} lệnh toàn warp.`,
                )}{' '}
            <Translated vi="Trên Metal, độ rộng SIMD group lấy từ pipeline, không phải từ một hằng số.">
              On Metal the SIMD-group width comes from the pipeline, not from a constant.
            </Translated>
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS62: Record<string, Question> = {
  'ch06/6.2/q-buses': {
    kind: 'numeric',
    id: 'ch06/6.2/q-buses',
    prompt: (
      <Translated vi={<>Cần bao nhiêu bus DDR5-6400 64 bit (mỗi bus 51.2 GB/s) để cung cấp 1000 GB/s?</>}>
        <>How many 64-bit DDR5-6400 buses (51.2 GB/s each) does it take to supply 1000 GB/s?</>
      </Translated>
    ),
    answer: 20,
    explain: (
      <Translated
        vi={
          <p>
            1000 / 51.2 = 19.5, và không có nửa bus: 20, đúng con số của sách. Một CPU ở 200 GB/s cần 3.9, tức bốn. Hai mươi
            bus dây dẫn chạy ngang một bo mạch là vấn đề mà HBM giải quyết.
          </p>
        }
      >
        <p>
          1000 / 51.2 = 19.5, and half a bus is not on offer: 20, the book's own count. A CPU at 200 GB/s needs 3.9, so
          four. Twenty buses of wires across a board is the problem HBM solves.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.2, p. 133',
  },
  'ch06/6.2/q-hbm': {
    kind: 'numeric',
    id: 'ch06/6.2/q-hbm',
    prompt: (
      <Translated
        vi={
          <>
            Ví dụ HBM3E của sách: xung nhịp bus 4.9 GHz, tốc độ dữ liệu kép, và 16 bus 64 bit. Dùng công thức ở cùng trang,
            các đầu vào này cho băng thông bao nhiêu?
          </>
        }
      >
        <>
          The book's HBM3E example: a 4.9 GHz bus clock, double data rate, and 16 buses of 64 bits. Using the formula on
          the same page, what bandwidth do these inputs give?
        </>
      </Translated>
    ),
    answer: 1254.4,
    tol: 0.01,
    unit: 'GB/s',
    explain: (
      <Translated
        vi={
          <p>
            8 byte × 2 × 4.9 GHz × 16 = 1254.4 GB/s, khoảng 1.25 TB/s. Sách in "1 TB/s" (và "4.9 GHz/s" cho xung nhịp). Đó
            cũng là con số của một stack HBM: các hãng bộ nhớ ghi 1.18 đến hơn 1.23 TB/s cho mỗi stack HBM3E, và một GPU mang
            nhiều stack (facts A8).
          </p>
        }
      >
        <p>
          8 bytes × 2 × 4.9 GHz × 16 = 1254.4 GB/s, about 1.25 TB/s. The book prints "1 TB/s" (and "4.9 GHz/s" for the
          clock). It is also the figure for one HBM stack: memory makers list 1.18 to more than 1.23 TB/s per HBM3E stack,
          and a GPU carries several stacks (facts A8).
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.2, p. 133; erratum (notes item 3, facts A8 and E1)',
    vi: { source: 'ch. 6 §6.2, p. 133; lỗi in (notes mục 3, facts A8 và E1)' },
  },
  'ch06/6.2/q-util': {
    kind: 'numeric',
    id: 'ch06/6.2/q-util',
    prompt: (
      <Translated
        vi={
          <>
            Bus của một kênh có thể chuyển 32 GB/s. Mỗi truy cập mất 15 đơn vị độ trễ trong bank của nó, rồi 1 đơn vị burst
            trên bus. Kênh có hai bank và luôn có yêu cầu đang chờ. Nó cung cấp được bao nhiêu?
          </>
        }
      >
        <>
          A channel's bus can move 32 GB/s. Each access spends 15 units of latency in its bank, then 1 unit of burst on the
          bus. The channel has two banks and requests are always waiting. How much does it deliver?
        </>
      </Translated>
    ),
    answer: 4,
    tol: 0.01,
    unit: 'GB/s',
    explain: (
      <Translated
        vi={
          <p>
            Một bank giữ bus bận 1 đơn vị trong mỗi 15 + 1 = 16. Hai bank chồng độ trễ lên nhau và gửi hai burst trong mỗi 16
            đơn vị: 2/16 = 12.5 %, và 12.5 % của 32 GB/s là 4 GB/s. Lấp đầy bus sẽ cần R + 1 = 16 bank.
          </p>
        }
      >
        <p>
          One bank keeps the bus busy 1 unit in every 15 + 1 = 16. Two banks overlap their latencies and send two bursts
          in every 16 units: 2/16 = 12.5 %, and 12.5 % of 32 GB/s is 4 GB/s. Filling the bus would take R + 1 = 16 banks.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.2, p. 134 (the R + 1 reasoning)',
    vi: { source: 'ch. 6 §6.2, p. 134 (lập luận R + 1)' },
  },
  'ch06/6.2/q-banks': {
    kind: 'numeric',
    id: 'ch06/6.2/q-banks',
    prompt: (
      <Translated vi={<>Tỉ lệ độ trễ trên burst của một kênh là 30. Số bank nhỏ nhất giữ bus của nó luôn bận là bao nhiêu?</>}>
        <>A channel's latency-to-burst ratio is 30. What is the smallest number of banks that keeps its bus busy all the time?</>
      </Translated>
    ),
    answer: 31,
    explain: (
      <Translated
        vi={
          <p>
            R + 1 = 31: trong khi một bank gửi burst của nó, 30 bank còn lại mỗi bank đang ở đâu đó trong độ trễ 30 đơn vị
            của mình. Kênh thật có nhiều hơn, vì hai yêu cầu cần cùng một bank cùng lúc không thể chồng lên nhau (một xung đột
            bank), và vì mảng ô nhớ của một bank không thể lớn lên mà độ trễ không tăng theo, nên riêng dung lượng đã đòi hỏi
            nhiều bank.
          </p>
        }
      >
        <p>
          R + 1 = 31: while one bank sends its burst, the other 30 are each somewhere in their 30-unit latency. Real
          channels have more, because two requests that need the same bank at once cannot overlap (a bank conflict), and
          because a bank's cell array cannot grow without its latency growing, so capacity alone needs many banks.
        </p>
      </Translated>
    ),
    source: 'ch. 6 §6.2, pp. 134–135',
  },
  'ch06/6.2/q-onechannel': {
    kind: 'choice',
    id: 'ch06/6.2/q-onechannel',
    prompt: (
      <Translated
        vi={
          <>
            Một GPU CDNA xen kẽ địa chỉ trên 32 kênh L2, mỗi lần 256 byte. Nhiều warp, mỗi warp đọc một phần tử dọc cột 0
            của một ma trận float thứ tự hàng rộng 2048 (luồng r đọc hàng r). Những lần đọc này dùng bao nhiêu trong 32 kênh
            L2?
          </>
        }
      >
        <>
          A CDNA GPU interleaves addresses across 32 L2 channels, 256 bytes at a time. Many warps each read one element
          down column 0 of a row-major float matrix 2048 wide (thread r reads row r). How many of the 32 L2 channels do
          these reads use?
        </>
      </Translated>
    ),
    options: [
      { label: <Translated vi="Một, cùng một kênh cho mọi hàng.">One, the same one for every row.</Translated>, correct: true },
      {
        label: (
          <Translated vi="Cả 32: xen kẽ trải mọi mảng ra mọi kênh.">All 32: interleaving spreads every array over every channel.</Translated>
        ),
        why: (
          <Translated vi="Nó trải các địa chỉ liên tiếp. Những địa chỉ này cách nhau 8192 byte.">
            It spreads consecutive addresses. These are 8192 bytes apart.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Tám.">Eight.</Translated>,
        why: (
          <Translated vi="Hãy tính khoảng cách giữa hai lần đọc liên tiếp theo byte, rồi chia cho 256.">
            Work out the distance between two consecutive reads in bytes, and divide by 256.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Còn tùy độ rộng warp.">It depends on the warp width.</Translated>,
        why: (
          <Translated vi="Kênh chỉ phụ thuộc vào địa chỉ; warp nào phát lần đọc không làm thay đổi điều đó.">
            The channel depends only on the address; which warp issues the read does not change it.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi hàng là 2048 × 4 = 8192 byte = 32 × 256. Phần tử (r, 0) bắt đầu sau 32r khúc 256 byte tính từ đầu ma trận,
            nên nó luôn rơi vào cùng một kênh: mọi lần đọc này xếp hàng trên một kênh trong khi 31 kênh ngồi không. Dùng
            trình chia mảng với thiết lập sẵn của AMD để thấy mẫu 2048 phần tử. Trải các truy cập đồng thời ra các kênh là một
            nửa của sự cộng sinh ở p. 137; NVIDIA và Apple không công bố ánh xạ của họ, nên ở đó bạn chỉ có thể đo.
          </p>
        }
      >
        <p>
          Each row is 2048 × 4 = 8192 bytes = 32 × 256. Element (r, 0) starts 32r chunks of 256 bytes into the matrix, so
          it always falls in the same channel: every one of these reads queues on one channel while 31 sit idle. Use the
          dealer with AMD's preset to see the 2048-element pattern. Spreading simultaneous accesses over channels is half
          of the symbiosis on p. 137; NVIDIA and Apple do not document their mapping, so there you can only measure.
        </p>
      </Translated>
    ),
    source: "our arithmetic on AMD's documented interleave (facts A9); ch. 6 §6.2, p. 137",
    vi: { source: 'phép tính của chúng tôi trên cách xen kẽ được công bố của AMD (facts A9); ch. 6 §6.2, p. 137' },
  },
  'ch06/6.2/q-occupancy': {
    kind: 'choice',
    id: 'ch06/6.2/q-occupancy',
    prompt: (
      <Translated
        vi={
          <>
            Các phép nạp của một kernel được hợp nhất hoàn hảo, nhưng nó chỉ đạt một nửa băng thông bạn đã đo cho thiết bị
            của mình bằng một phép sao chép đơn giản. Occupancy (mức chiếm dụng) của nó là 25 %. Lý do khả dĩ nhất là gì?
          </>
        }
      >
        <>
          A kernel's loads are perfectly coalesced, but it reaches only half the bandwidth you measured for your device
          with a simple copy. Its occupancy is 25 %. What is the most likely reason?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Quá ít yêu cầu đang bay để giữ mọi kênh và bank bận rộn.">
            Too few requests in flight to keep every channel and bank busy.
          </Translated>
        ),
        correct: true,
      },
      {
        label: <Translated vi="Rốt cuộc các truy cập của nó hẳn là không được hợp nhất.">Its accesses must be uncoalesced after all.</Translated>,
        why: (
          <Translated vi="Hợp nhất truy cập quyết định mỗi yêu cầu chở bao nhiêu, không phải có bao nhiêu yêu cầu đang chờ. Giả thiết nói mỗi yêu cầu đều đầy.">
            Coalescing decides how much each request carries, not how many are outstanding. The premise says each request is full.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Độ trễ DRAM quá dài để bất kỳ kernel nào đạt được băng thông đó.">
            DRAM latency is too long for any kernel to reach that bandwidth.
          </Translated>
        ),
        why: (
          <Translated vi="Phép sao chép đơn giản của bạn đã đạt được nó, với cùng độ trễ. Đủ nhiều yêu cầu đang bay sẽ che được độ trễ.">
            Your simple copy reached it, against the same latency. Enough requests in flight hide it.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Bộ nhớ đệm đang bị thrashing.">The caches are thrashing.</Translated>,
        why: (
          <Translated vi="Có thể xảy ra trong một kernel tái sử dụng dữ liệu, nhưng không có gì ở đây chỉ tới việc tái sử dụng; một phần tư số warp chỉ tới việc có quá ít yêu cầu đang bay.">
            Possible in a kernel that reuses data, but nothing here points to reuse; a quarter of the warps points to too little in flight.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với một phần tư số warp có thể cư trú, kernel có ít phép nạp đang chờ hơn nhiều so với khả năng, và quá ít để giữ
            bận các bank của mọi kênh: bus rảnh trong khi các bank chờ. Cách sửa: thêm warp (occupancy), hoặc thêm byte vào
            mỗi phép nạp (mục 6.3). Hướng dẫn phân loại của NVIDIA gọi điều này là "bytes-in-flight starvation" (đói byte
            đang bay).
          </p>
        }
      >
        <p>
          With a quarter of the possible warps resident, the kernel has far fewer loads outstanding than it could, and
          too few to keep every channel's banks busy: the bus idles while banks wait. The fixes: more warps (occupancy),
          or more bytes in each load (section 6.3). NVIDIA's triage guide calls this "bytes-in-flight starvation".
        </p>
      </Translated>
    ),
    source: "ch. 6 §6.2, pp. 135–136; NVIDIA's Compute Triage Guide (facts A10)",
    vi: { source: 'ch. 6 §6.2, pp. 135–136; Compute Triage Guide của NVIDIA (facts A10)' },
  },
};

// ---------------------------------------------------------------- the 8x8 reveal
function EightByEight({ committed }: { committed: number }) {
  const eight = phaseChannels(8, 2, FIG_6_9);
  const per = eight[0].channels.length;
  return (
    <>
      <p data-out="p-8x8">
        <Translated
          vi={
            <>
              <b>{per}.</b> {committed === per ? 'Đúng như bạn đã đoán.' : 'Bản theo dõi cho thấy vì sao.'} Phần tử 8r + c của
              M nằm trong burst 4r + ⌊c/2⌋. Pha ph đọc các cột 2ph và 2ph + 1, nên mọi phép nạp M của pha đều nằm trong burst
              4r + ph, tức kênh ph: kênh {eight.map((p) => p.channels[0]).join(', rồi ')}. Cả bốn kênh đều được dùng qua toàn
              bộ lần chạy, nhưng từng kênh một, kém phân tán hơn mức trường hợp 4 × 4 đạt được.
            </>
          }
        >
          <>
            <b>{per}.</b> {committed === per ? 'That is what you had.' : 'The trace shows why.'} Element 8r + c of M sits in
            burst 4r + ⌊c/2⌋. Phase ph reads columns 2ph and 2ph + 1, so every M load of the phase is in burst 4r + ph, which
            is channel ph: channel {eight.map((p) => p.channels[0]).join(', then ')}. All four channels are used over the run,
            but one at a time, which is less spread than the 4 × 4 case manages.
          </>
        </Translated>
      </p>
      <p className="muted">
        <Translated vi={'Câu "use all four channels" (dùng cả bốn kênh) của sách (p. 137) đúng khi tính qua các pha, không phải cùng lúc. Phép suy ra này là của chúng tôi (notes mục 8). Lựa chọn 8 × 8 giờ đã mở trong trình khám phá ở trên.'}>
          The book's "use all four channels" (p. 137) holds across phases, not at once. This derivation is ours (notes item
          8). The 8 × 8 option is now open in the explorer above.
        </Translated>
      </p>
    </>
  );
}

// ---------------------------------------------------------------- page
function Muted({ children }: { children: ReactNode }) {
  return <span className="muted">{children}</span>;
}

export function S62() {
  return (
    <>
      <Translated
        vi={
          <p>
            Mục 6.1 làm cho mỗi yêu cầu hiệu quả: các địa chỉ kề nhau của một warp trở thành một burst. Vẫn chưa đủ. Sách
            đặt một lần truy cập DRAM ở mức hàng chục nano giây (p. 124) còn bản thân burst thì tốn rất ít thời gian; một bộ
            xử lý chờ từng burst một chỉ nhận được một phần nhỏ những gì bộ nhớ có thể cung cấp. Các hệ thống DRAM thêm hai
            mức song song nữa bên trên burst: kênh (channel) và bank. Mục này nói về những gì chúng cần từ kernel của bạn.
          </p>
        }
      >
        <p>
          Section 6.1 made each request efficient: a warp's neighbouring addresses become one burst. That is still not
          enough. The book puts a DRAM access at tens of nanoseconds (p. 124) and the burst itself at very little time;
          a processor that waits for one burst at a time gets a small fraction of what the memory could supply. DRAM
          systems add two more levels of parallelism on top of bursts: channels and banks. This section is about what
          they need from your kernel.
        </p>
      </Translated>
      <Source section="6.2" pages="132–133" figs={tr('Fig. 6.7', 'Hình 6.7')} />

      <h2>
        <Translated vi="Kênh và bank">Channels and banks</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một kênh là một bộ điều khiển bộ nhớ với bus riêng, và mỗi bus có vài bank phía sau. Một bank chứa một mảng ô
            DRAM, các bộ khuếch đại cảm biến đọc chúng, và giao diện đưa burst lên bus. Một bộ xử lý có một hoặc nhiều kênh.
          </p>
        }
      >
        <p>
          A channel is a memory controller with its own bus, and each bus has several banks behind it. A bank holds an
          array of DRAM cells, the sense amplifiers that read them, and the interface that puts bursts on the bus. A
          processor has one or more channels.
        </p>
      </Translated>
      <ChannelsFigure />
      <Translated
        vi={
          <p>
            Hai thứ khác nhau giới hạn một kênh. Bus của nó chỉ chuyển được một số byte nhất định mỗi giây, và mỗi bank cần
            một thời gian dài trước khi có gì để gửi. Thứ nhất là phép tính số học; thứ hai được giải quyết bằng cách có nhiều
            bank.
          </p>
        }
      >
        <p>
          Two different things limit a channel. Its bus can move only so many bytes per second, and each bank needs a long
          time before it has anything to send. The first is arithmetic; the second is solved by having many banks.
        </p>
      </Translated>

      <h2>
        <Translated vi="Một bus chở được bao nhiêu">What one bus can carry</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một bus chuyển số byte bằng độ rộng của nó trên mỗi cạnh xung nhịp nó dùng: độ rộng × xung nhịp, gấp đôi với tốc
            độ dữ liệu kép (DDR), loại truyền trên cả cạnh lên lẫn cạnh xuống. Ví dụ của sách là một bus DDR5-6400 64 bit chạy
            ở 3.2 GHz: 8 byte × 2 × 3.2 GHz = 51.2 GB/s. Sách cũng đặt nhu cầu của một CPU hiện đại ở mức 200 GB/s trở lên và
            của một GPU ở hơn 1000 GB/s. Hãy tính số bus trước khi chạm vào máy tính.
          </p>
        }
      >
        <p>
          A bus moves its width in bytes on every clock edge it uses: width × clock, twice that for double data rate (DDR),
          which transfers on both the rising and the falling edge. The book's example is a 64-bit DDR5-6400 bus clocked at
          3.2 GHz: 8 bytes × 2 × 3.2 GHz = 51.2 GB/s. The book also puts a modern CPU's appetite at 200 GB/s or more and a
          GPU's at more than 1000 GB/s. Work out the bus counts before you touch the calculator.
        </p>
      </Translated>
      <Source section="6.2" pages="133" />
      <Ask q={questionsS62['ch06/6.2/q-buses']} />
      <Ask q={questionsS62['ch06/6.2/q-hbm']} />
      <BusCalculator />
      <Translated
        vi={
          <p>
            Bốn bus cho CPU và hai mươi cho GPU. Hai mươi bus 64 bit chạy ngang một bo mạch tiêu tốn quá nhiều điện năng, đó
            là lý do GPU chuyển sang High Bandwidth Memory (HBM): DRAM được đóng gói cùng bộ xử lý, các kết nối ngắn, và cả
            xung nhịp lẫn số bus đều tăng.
          </p>
        }
      >
        <p>
          Four buses for the CPU and twenty for the GPU. Twenty 64-bit buses driven across a circuit board draw too much
          power, which is why GPUs moved to High Bandwidth Memory (HBM): the DRAM is packaged with the processor, the
          connections are short, and both the clock and the number of buses go up.
        </p>
      </Translated>
      <Source section="6.2" pages="133" />
      <Translated
        vi={
          <p>
            Ví dụ HBM3E của sách có hai chỗ sơ suất. Nó ghi xung nhịp bus là "4.9 GHz/s", lẽ ra phải là GHz. Và nó nói kết quả
            là "1 TB/s", nhưng chính công thức của sách trên chính đầu vào của sách cho ra 1254.4 GB/s, khoảng 1.25 TB/s, như
            máy tính cho thấy. Đó là cho một stack HBM, và một GPU mang nhiều stack. Câu "one to eight channels" (một đến tám
            kênh) mỗi bộ xử lý của sách cũng đếm thiếu với GPU: chính thiết bị ví dụ của sách, H100 SXM5, có 10 bộ điều khiển
            bộ nhớ và 5 stack HBM3.
          </p>
        }
      >
        <p>
          The book's HBM3E example has two slips. It gives the bus clock as "4.9 GHz/s", which should be GHz. And it says
          the result is "1 TB/s", but its own formula on its own inputs gives 1254.4 GB/s, about 1.25 TB/s, as the
          calculator shows. That is for one HBM stack, and a GPU carries several. The book's "one to eight channels" per
          processor also undercounts GPUs: its own example device, the H100 SXM5, has 10 memory controllers and 5 HBM3
          stacks.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch06-facts.md A7, A8; ch04-facts.md A4)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              NVIDIA công bố cùng công thức độ rộng × xung nhịp × 2; AMD gọi 8 đến 16 bộ điều khiển bộ nhớ là điển hình. Các
              hãng bộ nhớ ghi 1.18 đến hơn 1.23 TB/s cho mỗi stack HBM3E. Chưa kiểm tra: hai kênh con 32 bit của DDR5.
            </p>
          }
        >
          <p>
            NVIDIA documents the same width × clock × 2 formula; AMD calls 8 to 16 memory controllers typical. Memory
            makers list 1.18 to more than 1.23 TB/s per HBM3E stack. Not checked: DDR5's two 32-bit subchannels.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Vì sao một bank là không đủ">Why one bank is not enough</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bên trong một kênh, mỗi truy cập vào một bank có hai phần: một khoảng chờ dài trong khi bank đọc mảng ô nhớ của
            nó, rồi một burst ngắn trên bus. Bus chỉ được dùng trong lúc có burst. Nếu khoảng chờ dài gấp R lần burst, một bank
            giữ bus bận nhiều nhất một đơn vị trong mỗi R + 1. Minh họa của sách lấy R = 20: 1/21, tức 4.8 % thời gian, nên
            một kênh có bus chuyển được 16 GB/s chỉ cung cấp nhiều nhất 0.76 GB/s.
          </p>
        }
      >
        <p>
          Inside a channel, every access to a bank has two parts: a long wait while the bank reads its cell array, then a
          short burst on the bus. The bus is in use only during the burst. If the wait is R times as long as the burst,
          one bank keeps the bus busy at most one unit in every R + 1. The book's illustration takes R = 20: 1/21, or 4.8 %
          of the time, so a channel whose bus could move 16 GB/s delivers at most 0.76 GB/s.
        </p>
      </Translated>
      <Source section="6.2" pages="133–134" figs={tr('Fig. 6.8', 'Hình 6.8')} />
      <DramBanksScene />
      <Translated
        vi={
          <p>
            Cách chữa là bắt đầu một truy cập ở bank thứ hai trong khi bank thứ nhất vẫn đang chờ. Độ trễ của chúng chồng lên
            nhau và các burst đi ra nối tiếp nhau: hai bank, gấp đôi mức sử dụng. Mỗi bank thêm vào cộng thêm cùng một lượng
            cho tới khi R + 1 bank lấp đầy bus. Hãy kiểm tra điều đó trong trình khám phá, rồi cho một số yêu cầu rơi vào một
            bank vẫn còn bận.
          </p>
        }
      >
        <p>
          The cure is to start an access in a second bank while the first is still waiting. Their latencies overlap and
          their bursts go out back to back: two banks, twice the utilisation. Each further bank adds the same amount until
          R + 1 of them fill the bus. Check that in the explorer, then make some requests land on a bank that is still
          busy.
        </p>
      </Translated>
      <Source section="6.2" pages="134" />
      <Translated
        vi={
          <p>
            Trình khám phá bên dưới là mô hình một kênh của sách, chạy tiến theo thời gian. Hoạt cảnh có thuyết minh ở trên
            được vẽ từ cùng mô hình đó.
          </p>
        }
      >
        <p>
          The explorer below is the book's model of one channel, run forward in time. The narrated animation above was
          drawn from the same model.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <>
              <p>
                Mọi truy cập có cùng độ trễ và cùng burst; bộ điều khiển bắt đầu các yêu cầu theo thứ tự đến, nhiều nhất một
                yêu cầu mỗi thời gian burst. DRAM thật còn làm tươi (refresh) các ô nhớ, và xử lý khác đi một truy cập vào một
                hàng đã mở sẵn; cả hai điều đó đều không có ở đây.
              </p>
              <p>
                R = 20 là minh họa của sách, không phải một tỉ lệ đo được. Số bank mỗi kênh HBM3E chưa kiểm tra được (facts
                A9).
              </p>
            </>
          }
        >
          <>
            <p>
              Every access has the same latency and the same burst; the controller starts requests in arrival order, at most
              one per burst time. Real DRAM also refreshes its cells, and treats an access to an already open row
              differently; neither is here.
            </p>
            <p>R = 20 is the book's illustration, not a measured ratio. Banks per HBM3E channel could not be checked (facts A9).</p>
          </>
        </Translated>
      </Simplification>
      <ChannelExplorer />
      <Translated
        vi={
          <p>
            Kênh thật có nhiều hơn R + 1 bank, vì hai lý do sách nêu ra. Yêu cầu không đến theo một vòng xoay gọn gàng: hai
            yêu cầu muốn cùng một bank cùng lúc không thể chồng lên nhau, đó là một <b>xung đột bank</b> (bank conflict), và
            nhiều bank hơn làm điều đó ít xảy ra hơn. Và mảng ô nhớ của một bank không thể lớn lên mà độ trễ không tăng theo,
            nên riêng dung lượng đã đòi hỏi nhiều bank.
          </p>
        }
      >
        <p>
          Real channels have more than R + 1 banks, for two reasons the book gives. Requests do not arrive in a tidy
          rotation: two that want the same bank at the same time cannot overlap, which is a <b>bank conflict</b>, and more
          banks make one less likely. And a bank's cell array cannot grow without its latency growing, so capacity alone
          calls for many banks.
        </p>
      </Translated>
      <Source section="6.2" pages="134–135" />
      <Ask q={questionsS62['ch06/6.2/q-util']} />
      <Ask q={questionsS62['ch06/6.2/q-banks']} />

      <h2>
        <Translated vi="Trải địa chỉ ra các kênh và bank">Spreading addresses over channels and banks</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bank và kênh chỉ có ích nếu các yêu cầu đến cùng lúc đi tới những bank và kênh khác nhau. Phần cứng sắp xếp điều
            đó bằng cách chia không gian địa chỉ như chia bài: các burst liên tiếp đi tới các kênh liên tiếp, và khi mọi kênh
            đã nhận một burst, vòng kế tiếp đi tới bank kế tiếp. Trong ví dụ minh họa của sách (burst gồm hai phần tử 4 byte,
            bốn kênh, hai bank), M[0] và M[1] rơi vào kênh 0, bank 0; M[2] và M[3] vào kênh 1; M[8] và M[9] quay lại kênh 0,
            bank 1. Ngay cả một mảng 16 phần tử cũng được trải ra mọi kênh và bank.
          </p>
        }
      >
        <p>
          Banks and channels help only if requests that arrive together go to different ones. The hardware arranges that
          by dealing the address space out like cards: consecutive bursts go to consecutive channels, and once every
          channel has had one, the next round goes to the next bank. In the book's toy (bursts of two 4-byte elements, four
          channels, two banks) M[0] and M[1] land in channel 0, bank 0; M[2] and M[3] in channel 1; M[8] and M[9] back in
          channel 0, bank 1. Even a 16-element array is spread over every channel and bank.
        </p>
      </Translated>
      <Source section="6.2" pages="135" figs={tr('Fig. 6.9', 'Hình 6.9')} />
      <Dealer />
      <Translated
        vi={
          <p>
            Một nhà cung cấp công bố phiên bản của họ. Nạp thiết lập sẵn của AMD và để ý nó thô đến mức nào: 64 float dùng
            chung một kênh trước khi kênh kế tiếp nhận được gì, nên một mảng nhỏ nằm trong một kênh và phải tới 8192 byte mới
            cho cả 32 kênh mỗi kênh một lượt.
          </p>
        }
      >
        <p>
          One vendor documents its version. Load AMD's preset and notice how coarse it is: 64 floats share a channel before
          the next one gets any, so a small array sits in one channel and it takes 8192 bytes to give all 32 channels a
          turn.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch06-facts.md A9, B3)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              AMD: L2 của GPU CDNA có "32 ... channels at 256-byte interleaving" (32 kênh xen kẽ 256 byte), địa chỉ được xen
              kẽ trên chúng để cân bằng tải. Đây là các kênh L2; AMD không công bố ánh xạ DRAM bên dưới chúng. NVIDIA và Apple
              hoàn toàn không công bố ánh xạ địa chỉ nào.
            </p>
          }
        >
          <p>
            AMD: the L2 of CDNA GPUs has "32 ... channels at 256-byte interleaving", addresses interleaved across them for
            load balancing. These are L2 channels; AMD does not document the DRAM mapping beneath them. NVIDIA and Apple
            publish no address mapping at all.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Xen kẽ trải các địa chỉ liên tiếp. Các truy cập mà các luồng của bạn thực hiện cùng một lúc có liên tiếp hay không
            là một câu hỏi khác, và đó là câu hỏi của bạn.
          </p>
        }
      >
        <p>
          Interleaving spreads consecutive addresses. Whether the accesses your threads make at the same moment are
          consecutive is a different question, and it is yours.
        </p>
      </Translated>
      <Ask q={questionsS62['ch06/6.2/q-onechannel']} />

      <h2>
        <Translated vi="Luồng và DRAM cần nhau">Threads and DRAM need each other</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ghép các mảnh lại với nhau. Mỗi bank làm việc với một truy cập mỗi lúc, mỗi kênh cần khoảng R + 1 truy cập đang
            diễn ra để giữ bus đầy, và một GPU có nhiều kênh. Bộ nhớ chỉ đạt băng thông danh định khi có rất nhiều yêu cầu
            đang được xử lý cùng một lúc, trải ra các kênh và bank. Yêu cầu đến từ các luồng, nên đây là lý do thứ hai để muốn
            occupancy (mức chiếm dụng) cao. Chương 4 dùng các warp cư trú để giữ các đơn vị thực thi bận trong khi một warp
            chờ; ở đây chính những warp đó giữ DRAM bận, vì mỗi warp có các phép nạp đang chờ. Và mỗi truy cập vẫn phải được
            hợp nhất, nếu không một yêu cầu chở ít hơn một burst đầy.
          </p>
        }
      >
        <p>
          Put the pieces together. Each bank works on one access at a time, each channel needs about R + 1 of them going to
          keep its bus full, and a GPU has many channels. The memory reaches its rated bandwidth only when a great many
          requests are under way at the same moment, spread over the channels and banks. Requests come from threads, so
          this is the second reason to want high occupancy. Chapter 4 used resident warps to keep the execution units busy
          while one warp waited; here the same warps keep the DRAM busy, because each one has loads outstanding. And each
          access must still be coalesced, or a request carries less than a full burst.
        </p>
      </Translated>
      <Source section="6.2" pages="135–136" />
      <Translated
        vi={
          <p>
            Sách theo dõi điều này qua phép nhân phân ô của chương 5 ở kích thước minh họa: ma trận 4 × 4, ô 2 × 2, bốn khối
            chạy cùng lúc, và ánh xạ của Hình 6.9. Ở pha 0, các khối 0,0 và 0,1 đều nạp M[0], M[1], M[4] và M[5], còn các khối
            1,0 và 1,1 nạp M[8], M[9], M[12] và M[13]. Hãy đi qua từng bước.
          </p>
        }
      >
        <p>
          The book traces this through chapter 5's tiled multiplication at toy size: 4 × 4 matrices, 2 × 2 tiles, four
          blocks running at once, and Fig. 6.9's mapping. In phase 0, blocks 0,0 and 0,1 both load M[0], M[1], M[4] and
          M[5], and blocks 1,0 and 1,1 load M[8], M[9], M[12] and M[13]. Step through it.
        </p>
      </Translated>
      <Source section="6.2" pages="136–137" figs={tr('Figs. 6.10 and 6.11', 'Hình 6.10 và 6.11')} />
      <ToyReplay />
      <Translated
        vi={
          <p>
            Pha 0 giữ bận các kênh 0 và 2, pha 1 các kênh 1 và 3. Sách sau đó để lại cho bạn một bài kiểm tra: rằng ma trận
            8 × 8 với cùng ô 2 × 2 "use all four channels" (dùng cả bốn kênh).
          </p>
        }
      >
        <p>
          Phase 0 keeps channels 0 and 2 busy, phase 1 channels 1 and 3. The book then leaves you a check: that 8 × 8
          matrices with the same 2 × 2 tiles "use all four channels".
        </p>
      </Translated>
      <Source section="6.2" pages="137" />
      <Predict
        id="ch06/6.2/p-8x8"
        prompt={
          <Translated
            vi={
              <>
                Cùng ánh xạ (burst 2 phần tử, 4 kênh, 2 bank), ma trận 8 × 8, ô 2 × 2, cả mười sáu khối chạy cùng lúc. Trong
                một pha, các phép nạp M của tất cả các khối chạm vào bao nhiêu kênh?
              </>
            }
          >
            <>
              Same mapping (2-element bursts, 4 channels, 2 banks), 8 × 8 matrices, 2 × 2 tiles, all sixteen blocks running
              at once. In one phase, how many channels do the M loads of all the blocks hit?
            </>
          </Translated>
        }
        unit={tr('channels', 'kênh')}
      >
        {(v) => <EightByEight committed={v} />}
      </Predict>
      <Translated
        vi={
          <p>
            Hai khối nạp cùng các phần tử M vào khoảng cùng một lúc không nhất thiết đều phải đi tới DRAM: nếu thời điểm của
            chúng đủ gần, một bộ nhớ đệm có thể gộp chúng thành một truy cập. Sách nói bộ nhớ đệm của GPU chủ yếu được thiết kế
            để kết hợp những truy cập như vậy.
          </p>
        }
      >
        <p>
          Two blocks loading the same M elements at about the same time need not both go to DRAM: if their timing is close
          enough, a cache can merge them into one access. The book says GPU caches are mainly designed to combine such
          accesses.
        </p>
      </Translated>
      <Source section="6.2" pages="136–137" />
      <Translated
        vi={
          <p>
            Chúng làm được điều đó thường xuyên đến đâu trên thiết bị của bạn là một phép đo, và các nhà cung cấp mô tả bộ nhớ
            đệm của họ hẹp hơn sách.
          </p>
        }
      >
        <p>
          How often they manage it on your device is a measurement, and the vendors describe their caches more narrowly
          than the book does.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch06-facts.md A4)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              Không trang nào của nhà cung cấp mà chúng tôi đọc nói bộ nhớ đệm được "mainly designed" (chủ yếu được thiết kế)
              cho việc này. NVIDIA công bố dòng 128 byte gồm bốn sector 32 byte và cảnh báo một dòng có thể bị loại trước khi
              được dùng lại; AMD công bố hit-on-miss, một yêu cầu cho một dòng đang được nạp được tính là trúng (hit).
            </p>
          }
        >
          <p>
            No vendor page we read says caches are "mainly designed" for this. NVIDIA documents 128-byte lines of four
            32-byte sectors and warns a line may be evicted before reuse; AMD documents hit-on-miss, a request for a line
            already being fetched counting as a hit.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Lời cảnh báo kết thúc của sách là điều quan trọng trong thực tế: nếu mọi luồng đang chạy tại một thời điểm truy cập
            dữ liệu trong cùng một kênh, thông lượng giảm mạnh, dù có bao nhiêu luồng đi nữa.
          </p>
        }
      >
        <p>
          The book's closing warning is the one that matters in practice: if all the threads running at one moment access
          data in the same channel, throughput drops sharply, however many of them there are.
        </p>
      </Translated>
      <Source section="6.2" pages="137" />
      <Translated
        vi={
          <p>
            Thất bại ngược lại, quá ít yêu cầu cùng lúc, có một cái tên trong hướng dẫn profiler của NVIDIA:
            "bytes-in-flight starvation" (đói byte đang bay), một kernel "is not presenting enough outstanding loads to the
            bus" (không đưa đủ phép nạp đang chờ lên bus). Cách khắc phục của hướng dẫn là thêm song song mức lệnh, nạp vector
            (mục 6.3), hoặc sao chép bất đồng bộ. Best Practices Guide của NVIDIA thêm đối trọng mà sách để dành cho chương 15:
            với đủ song song mức lệnh, một kernel có thể che độ trễ ở occupancy thấp.
          </p>
        }
      >
        <p>
          The opposite failure, too few requests at once, has a name in NVIDIA's profiler guide: "bytes-in-flight starvation", a kernel that "is not
          presenting enough outstanding loads to the bus". The guide's remedies are more instruction-level parallelism,
          vectorised loads (section 6.3), or asynchronous copies. NVIDIA's Best Practices Guide adds the counterweight the
          book saves for chapter 15: with enough instruction-level parallelism, a kernel can cover latency at low
          occupancy.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch06-facts.md A10, C5; ch04-facts.md B10)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              Hai trích dẫn của NVIDIA lấy từ Nsight Compute Compute Triage Guide và Best Practices Guide. AMD: các kernel bị
              giới hạn bởi bộ nhớ được lợi nhiều hơn từ occupancy cao. Tài liệu Xcode của Apple định nghĩa occupancy và hiển
              thị một bộ đếm cho nó, không có API nào để truy vấn.
            </p>
          }
        >
          <p>
            The two NVIDIA quotes are from the Nsight Compute Compute Triage Guide and the Best Practices Guide. AMD:
            memory-bound kernels benefit more from high occupancy. Apple's Xcode documentation defines occupancy and shows a
            counter for it, with no API to query it.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS62['ch06/6.2/q-occupancy']} />

      <h2>
        <Translated vi="Cần bao nhiêu byte đang bay">How many bytes must be in flight</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách dừng ở "đủ luồng". Một quy tắc từ lý thuyết hàng đợi, định luật Little, cho nó một con số: để duy trì băng
            thông B trước độ trễ L, phải có B × L byte đang trên đường đi ở mọi thời điểm. Ở 1 GB/s, đó là một byte cho mỗi
            nano giây độ trễ. Decode batch 1, vốn truyền mọi trọng số một lần mỗi token với rất ít phép tính trên mỗi trọng số
            (mục 5.1), sống hay chết theo con số này.
          </p>
        }
      >
        <p>
          The book stops at "enough threads". A rule from queueing theory, Little's law, puts a number on it: to sustain a
          bandwidth B against a latency L, B × L bytes must be on their way at every moment. At 1 GB/s that is one byte per
          nanosecond of latency. Batch-1 decoding, which streams every weight once per token with little arithmetic on each
          (section 5.1), lives or dies by this number.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Định luật Little và trình khám phá này là của chúng tôi, không phải của sách. Độ trễ mặc định, 50 ns, là một giá
              trị minh họa trong khoảng "tens of nanoseconds" (hàng chục nano giây) của sách (p. 124), vốn chỉ là riêng DRAM;
              NVIDIA mô tả độ trễ của một lần nạp bộ nhớ toàn cục là "hundreds of clock cycles" (hàng trăm chu kỳ xung nhịp)
              (facts A1).
            </p>
          }
        >
          <p>
            Little's law and this explorer are ours, not the book's. The default latency, 50 ns, is an illustrative value in
            the book's "tens of nanoseconds" (p. 124), which is the DRAM alone; NVIDIA describes a global load's latency as
            "hundreds of clock cycles" (facts A1).
          </p>
        </Translated>
      </Beyond>
      <BytesInFlight />
      <Translated
        vi={
          <p>
            Hai thứ làm thay đổi câu trả lời, và chúng là hai đòn bẩy của mục này và mục kế tiếp. Nhiều warp cư trú hơn đưa
            nhiều phép nạp đang bay hơn cùng lúc; phép nạp rộng hơn đưa nhiều byte hơn vào mỗi phép nạp. Mục 6.3 nói về đòn bẩy
            thứ hai.
          </p>
        }
      >
        <p>
          Two things move the answer, and they are the two levers of this section and the next. More resident warps put
          more loads in flight at once; wider loads put more bytes in each one. Section 6.3 takes the second.
        </p>
      </Translated>

      <h2>
        <Translated vi="Trên Apple silicon">On Apple silicon</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hình ảnh của sách là một GPU rời với DRAM hoặc HBM riêng. GPU của Apple dùng chung bộ nhớ hệ thống với CPU, nên cả
            hai cùng rút từ một hệ thống bộ nhớ, và Apple cảnh báo rằng lưu lượng GPU lớn có thể làm chậm các truy cập bộ nhớ
            của chính CPU. Apple in một băng thông bộ nhớ cho mỗi cấu hình sản phẩm; bài học không dùng nó, vì đó là một mức
            đỉnh dùng chung, không phải thứ kernel của bạn nhận được. Con số của bạn là con số lab 5.1 đã đo{' '}
            <Muted>(<a href={href('ch05', 'labs')}>các lab chương 5</a>)</Muted>.
          </p>
        }
      >
        <p>
          The book's picture is a discrete GPU with its own DRAM or HBM. An Apple GPU shares system memory with the CPU, so
          both draw on one memory system, and Apple warns that heavy GPU traffic can slow the CPU's own memory accesses. Apple prints a memory bandwidth for each product configuration; the lesson does not use it, because it
          is a shared peak, not what your kernels get. Your number is the one lab 5.1 measured{' '}
          <Muted>(<a href={href('ch05', 'labs')}>chapter 5 labs</a>)</Muted>.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch06-facts.md C6; ch05-facts.md C6; infra-facts.md C13)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              "Apple GPUs have a unified memory model in which the CPU and the GPU share system memory." (GPU của Apple có mô
              hình unified memory, trong đó CPU và GPU dùng chung bộ nhớ hệ thống.) Tài liệu dành cho nhà phát triển của Apple
              không đưa ra số liệu băng thông nào; các bộ đếm Xcode của họ đo lượng đọc và ghi của GPU theo GB/s khi ứng dụng
              của bạn chạy.
            </p>
          }
        >
          <p>
            "Apple GPUs have a unified memory model in which the CPU and the GPU share system memory." Apple's developer
            documentation gives no bandwidth figures; its Xcode counters measure the GPU's reads and writes in GB/s while
            your app runs.
          </p>
        </Translated>
      </Today>
    </>
  );
}
