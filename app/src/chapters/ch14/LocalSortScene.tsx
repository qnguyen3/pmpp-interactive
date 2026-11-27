import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { PROBLEMS, sampled, sampledDirect } from '../../sim/radixcost';
import {
  BOOK_KEYS,
  bin,
  directPass,
  localPass,
  summarise,
  uniformKeys,
  warpStores,
  type WarpStore,
} from '../../sim/radixlocal';

// "Sort locally, then write in runs" (section 14.6, Figs. 14.5, 14.8 and 14.9): the
// direct scatter on the book's keys and on one 32-lane warp of random keys, then the
// two phases of the local design. Every key, address, count, run and sector drawn is
// computed by src/sim/radixlocal.ts and src/sim/radixcost.ts. Six beats, one narration
// clip each (content/ch14/narration/local-sort-N.txt).

const DIRECT = directPass(BOOK_KEYS, 0, 1, 4);
const LOCAL = localPass(BOOK_KEYS, 0, 1, 4);
const BLOCKS = LOCAL.blocks;

// One warp of random keys, NVIDIA's 32-byte sectors: the first warp whose cost is the average one.
const WKEYS = uniformKeys(4096, 1);
const warpAt = (r: number) => warpStores(directPass(WKEYS, 0, r, 256), 32, 4, 32);
const W1 = warpAt(1);
const PICK = W1.findIndex((w) => w.units.length === Math.round(summarise(W1).meanUnits));
const WARP: Record<number, WarpStore> = { 1: W1[PICK], 2: warpAt(2)[PICK], 8: warpAt(8)[PICK] };

// Beat 6: averages over a whole sort of a sample (blocks of 256 threads, random 32-bit keys).
const UNIFORM = PROBLEMS[0];
const BARS = [1, 2, 4, 8].map((r) => ({
  r,
  direct: sampledDirect(UNIFORM, 32, r, 256).meanUnits,
  local: sampled({ dist: 'uniform', keyBits: 32, rowLength: UNIFORM.rowLength, r, threads: 256, c: 1 }).meanUnits,
}));

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const one = (v: number) => (Math.round(v * 10) / 10).toFixed(1);

// Geometry: sixteen key cells.
const X0 = 40;
const CW = 40;
const CH = 28;
const cx = (i: number) => X0 + i * CW + CW / 2;

function KeyRow({ y, keys, tier, label, hot, show = 16, tag, below }: {
  y: number; keys: readonly number[]; tier: 'global' | 'shared'; label: string; hot?: Set<number>; show?: number; tag?: string; below?: boolean;
}) {
  return (
    <g data-row={tag}>
      <text x={X0} y={below ? y + CH + 14 : y - 7} fontSize="11" style={{ fill: `var(--${tier})` }} fontWeight="600">{label}</text>
      {keys.map((k, i) => (
        <g key={i} data-cell={i} data-key={i < show ? bin(k, 4) : ''}>
          <rect x={X0 + i * CW + 1} y={y} width={CW - 2} height={CH} rx="2"
            fill={i < show ? (hot?.has(i) ? `var(--${tier})` : `var(--${tier}-wash)`) : 'var(--cell)'}
            stroke={i < show ? `var(--${tier})` : 'var(--rule)'} />
          {i < show ? (
            <text x={cx(i)} y={y + 18} fontSize="11.5" textAnchor="middle" style={{ fill: hot?.has(i) ? 'var(--page)' : 'var(--ink)' }}>
              {bin(k, 4).slice(0, 3)}
              <tspan fontWeight="700" textDecoration="underline">{bin(k, 4).slice(3)}</tspan>
            </text>
          ) : null}
        </g>
      ))}
    </g>
  );
}

function Separators({ y1, y2, labelY = y2 + 13 }: { y1: number; y2: number; labelY?: number }) {
  return (
    <g>
      {[1, 2, 3].map((b) => (
        <line key={b} x1={X0 + b * 4 * CW} x2={X0 + b * 4 * CW} y1={y1} y2={y2} stroke="var(--ink-3)" strokeDasharray="3 4" />
      ))}
      {[0, 1, 2, 3].map((b) => (
        <text key={b} x={X0 + b * 4 * CW + 2 * CW} y={labelY} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
          <Translated vi={<>khối {b}</>}>block {b}</Translated>
        </text>
      ))}
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, strong, opacity = 1 }: { x1: number; y1: number; x2: number; y2: number; strong?: boolean; opacity?: number }) {
  return (
    <line x1={x1} y1={y1} x2={x2} y2={y2} opacity={opacity}
      stroke={strong ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={strong ? 2 : 1} markerEnd={strong ? 'url(#ls-head-strong)' : 'url(#ls-head)'} />
  );
}

function Frame({ children, headline, note }: { children: ReactNode; headline: ReactNode; note?: ReactNode }) {
  return (
    <svg viewBox="0 0 720 360" role="img" aria-label={tr('Animation of a radix pass with a local sort', 'Hoạt hình một lượt radix với phép sắp xếp cục bộ')} style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <defs>
        <marker id="ls-head" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">{/* i18n-ignore: SVG attribute value */}
          <path d="M0,0 L8,4 L0,8 z" fill="var(--ink-3)" />
        </marker>
        <marker id="ls-head-strong" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">{/* i18n-ignore: SVG attribute value */}
          <path d="M0,0 L8,4 L0,8 z" fill="var(--ink)" />
        </marker>
      </defs>
      {children}
      <text x={X0} y={334} fontSize="13" fontWeight="600" data-headline>{headline}</text>
      {note ? <text x={X0} y={352} fontSize="11" style={{ fill: 'var(--ink-2)' }} data-note>{note}</text> : null}
    </svg>
  );
}

// ---------------------------------------------------------------- the warp view (beats 1 and 2)
const UW = 7; // one 4-byte key in a 32-byte sector
function WarpView({ r, w }: { r: number; w: WarpStore }) {
  const per = 8;
  const units = w.units;
  const perRow = 10;
  const written = new Set(w.addrs);
  const cy = 104 + Math.ceil(units.length / perRow) * 34 + (units.length <= 10 ? 26 : 12);
  return (
    <g data-warp data-r={r} data-runs={w.runs} data-sectors={units.length} data-min={w.minUnits}>
      <text x={X0} y={30} fontSize="11" fontWeight="600" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi={<>một warp: 32 lane, mỗi lane ghi một khóa 4 byte; con số là chữ số {r} bit của lane</>}>
          one warp: 32 lanes, each storing one 4-byte key; the number is the lane's {r}-bit digit
        </Translated>
      </text>
      {w.digits.map((d, l) => (
        <g key={l}>
          <rect x={X0 + l * 20} y={38} width={18} height={22} rx="2" fill="var(--cell)" stroke="var(--rule)" />
          <text x={X0 + l * 20 + 9} y={53} fontSize={d > 99 ? 7.5 : 10} textAnchor="middle">{d}</text>
        </g>
      ))}
      <text x={X0} y={94} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>
        <Translated vi="các sector 32 byte của đầu ra mà lần ghi của nó chạm tới (tô đầy: một khóa warp này ghi)">
          the 32-byte sectors of the output its store touches (filled: a key this warp writes)
        </Translated>
      </text>
      {units.map((u, k) => {
        const x = X0 + (k % perRow) * (per * UW + 10);
        const y = 104 + Math.floor(k / perRow) * 34;
        const gap = k > 0 && u !== units[k - 1] + 1;
        return (
          <g key={u} data-sector={u}>
            {gap && k % perRow !== 0 ? <text x={x - 6} y={y + 12} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>…</text> : null}
            {Array.from({ length: per }, (_, e) => {
              const a = u * per + e;
              return <rect key={e} x={x + e * UW} y={y} width={UW - 1} height={16}
                fill={written.has(a) ? 'var(--global)' : 'var(--cell)'} stroke="var(--global)" strokeWidth="0.6" />;
            })}
            {units.length <= 10 ? <text x={x} y={y + 28} fontSize="9" style={{ fill: 'var(--ink-3)' }}><Translated vi={<>sector {u}</>}>sector {u}</Translated></text> : null}
          </g>
        );
      })}
      <g data-counter="runs">
        <text x={X0} y={cy} fontSize="12"><Translated vi="số dải địa chỉ liên tiếp">runs of consecutive addresses</Translated></text>
        <text x={300} y={cy} fontSize="13" fontWeight="700" data-value>{w.runs}</text>
      </g>
      <g data-counter="sectors">
        <text x={X0} y={cy + 20} fontSize="12"><Translated vi="số sector bị chạm">sectors touched</Translated></text>
        <text x={300} y={cy + 20} fontSize="13" fontWeight="700" data-value>{units.length}</text>
      </g>
      <g data-counter="minimum">
        <text x={X0} y={cy + 40} fontSize="12"><Translated vi="một lần ghi được hợp nhất gồm 32 khóa">a coalesced store of 32 keys</Translated></text>
        <text x={300} y={cy + 40} fontSize="13" fontWeight="700" data-value>{w.minUnits}</text>
      </g>
    </g>
  );
}

// ---------------------------------------------------------------- beats
const Y_IN = 40;
const Y_LOCAL = 166;
const Y_OUT = 262;
const Y_FROM = Y_IN + CH + 20; // arrows leave the top row below its block labels

function directBook(p: number) {
  const shown = Math.ceil(ramp(p, 0.04, 0.4) * 16);
  const outHot = new Set(DIRECT.writes.slice(0, 4).map((w) => w.addr));
  return (
    <g data-view="direct-book">
      <KeyRow y={Y_IN} keys={BOOK_KEYS} tier="global" label={tr('input, bit 0 underlined (global memory)', 'đầu vào, bit 0 được gạch chân (bộ nhớ toàn cục)')} hot={new Set([0, 1, 2, 3])} tag="input" />{/* i18n-ignore: data-row key */}
      <KeyRow y={Y_OUT} keys={DIRECT.out} tier="global" label={tr('output after the pass (global memory)', 'đầu ra sau lượt (bộ nhớ toàn cục)')} hot={outHot}
        show={16} tag="output" below />{/* i18n-ignore: data-row key */}
      {DIRECT.writes.slice(0, shown).map((w) => (
        <Arrow key={w.slot} x1={cx(w.slot)} y1={Y_FROM} x2={cx(w.addr)} y2={Y_OUT - 2} strong={w.block === 0} opacity={w.block === 0 ? 1 : 0.55} />
      ))}
      <Separators y1={Y_IN - 2} y2={Y_IN + CH + 4} />
    </g>
  );
}

const beats: Beat[] = [
  {
    clip: 'local-sort-1',
    get label() {
      return tr('Direct scatter', 'Scatter trực tiếp');
    },
    draw: (p) =>
      p < 0.5 ? (
        <Frame headline={tr(`Fig. 14.5: block 0 writes positions ${DIRECT.writes.slice(0, 4).map((w) => w.addr).join(', ')}`, `Hình 14.5: khối 0 ghi vào các vị trí ${DIRECT.writes.slice(0, 4).map((w) => w.addr).join(', ')}`)}
          note={tr('zero, one, zero, one: neighbouring threads, distant addresses', 'không, một, không, một: các luồng kề nhau, các địa chỉ xa nhau')}>
          {directBook(p / 0.5)}
        </Frame>
      ) : (
        <Frame headline={tr(`1-bit digit: ${WARP[1].runs} runs, ${WARP[1].units.length} sectors, where a coalesced store needs ${WARP[1].minUnits}`, `chữ số 1 bit: ${WARP[1].runs} dải, ${WARP[1].units.length} sector, trong khi một lần ghi được hợp nhất cần ${WARP[1].minUnits}`)}
          note={tr('lane order does not matter: the sectors the addresses touch do (ch06-facts A2)', 'thứ tự lane không quan trọng: các sector mà địa chỉ chạm tới mới quan trọng (ch06-facts A2)')}>
          <WarpView r={1} w={WARP[1]} />
        </Frame>
      ),
  },
  {
    clip: 'local-sort-2',
    get label() {
      return tr('Wider digits', 'Chữ số rộng hơn');
    },
    draw: (p) => {
      const r = p < 0.45 ? 2 : 8;
      const w = WARP[r];
      return (
        <Frame headline={tr(`${r}-bit digit, ${2 ** r} buckets: ${w.runs} runs, ${w.units.length} sectors`, `chữ số ${r} bit, ${2 ** r} bucket: ${w.runs} dải, ${w.units.length} sector`)}
          note={tr('the same 32 keys; with 8 bits most lanes have a bucket, and a run, to themselves', 'cùng 32 khóa đó; với 8 bit, phần lớn lane có riêng một bucket, và một dải')}>
          <WarpView r={r} w={w} />
        </Frame>
      );
    },
  },
  {
    clip: 'local-sort-3',
    get label() {
      return tr('Sort locally', 'Sắp xếp cục bộ');
    },
    draw: (p) => {
      const nb = Math.min(4, Math.floor(ramp(p, 0.05, 0.75) * 4.999));
      const sortedKeys = BLOCKS.flatMap((b) => b.sorted);
      return (
        <Frame
          headline={tr('phase one: each block sorts its keys by bit 0 in shared memory', 'pha một: mỗi khối sắp các khóa của nó theo bit 0 trong bộ nhớ chia sẻ')}
          note={tr('Fig. 14.8: keys move a few places and never leave their block', 'Hình 14.8: các khóa dịch đi vài chỗ và không bao giờ rời khối của chúng')}
        >
          <g data-view="local-sort" data-blocks-done={nb}>
            <KeyRow y={Y_IN} keys={BOOK_KEYS} tier="global" label={tr('input (global memory)', 'đầu vào (bộ nhớ toàn cục)')} tag="input" />{/* i18n-ignore: data-row key */}
            <KeyRow y={Y_LOCAL} keys={sortedKeys} tier="shared" label={tr("each block's shared memory, sorted by bit 0", 'bộ nhớ chia sẻ của mỗi khối, sắp theo bit 0')} show={nb * 4} tag="local" below />{/* i18n-ignore: data-row key */}
            {BLOCKS.slice(0, nb).flatMap((b) =>
              b.from.map((src, j) => <Arrow key={`${b.block}-${j}`} x1={cx(src)} y1={Y_FROM} x2={cx(b.first + j)} y2={Y_LOCAL - 2} strong />),
            )}
            <Separators y1={Y_IN - 2} y2={Y_LOCAL + CH + 4} labelY={Y_IN + CH + 14} />
            <rect x={X0} y={Y_OUT} width={16 * CW} height={CH} rx="3" fill="none" stroke="var(--rule)" strokeDasharray="4 4" />
            <text x={X0 + 8 * CW} y={Y_OUT + 18} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
              <Translated vi="đầu ra toàn cục: chưa ghi gì">global output: nothing written yet</Translated>
            </text>
          </g>
        </Frame>
      );
    },
  },
  {
    clip: 'local-sort-4',
    get label() {
      return tr('The table', 'Bảng');
    },
    draw: (p) => {
      const braces = ramp(p, 0.02, 0.25);
      const fill = Math.ceil(ramp(p, 0.25, 0.55) * 8);
      const scan = Math.ceil(ramp(p, 0.6, 0.92) * 8);
      const TW = 64;
      const tx = (k: number) => X0 + 40 + k * TW + (k >= 4 ? 24 : 0);
      const cell = (k: number, y: number, v: number | null, tier: 'global' | 'shared', on: boolean) => (
        <g key={`${y}-${k}`}>
          <rect x={tx(k)} y={y} width={TW - 6} height={28} rx="2" fill={on ? `var(--${tier}-wash)` : 'var(--cell)'} stroke={on ? `var(--${tier})` : 'var(--rule)'} />
          {on && v != null ? <text x={tx(k) + (TW - 6) / 2} y={y + 19} fontSize="13" textAnchor="middle" fontWeight="600">{v}</text> : null}
          <text x={tx(k) + (TW - 6) / 2} y={y - 5} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>
            <Translated vi={<>{k < 4 ? 'bit 0' : 'bit 1'} b{k % 4}</>}>
              {k < 4 ? 'zeros' : 'ones'} b{k % 4}
            </Translated>
          </text>
        </g>
      );
      return (
        <Frame
          headline={tr("every block's zero count, then every block's one count; one exclusive scan", 'số bit 0 của mọi khối, rồi số bit 1 của mọi khối; một scan loại trừ')}
          note={tr('Fig. 14.9: the scanned table is where each local bucket starts in the output', 'Hình 14.9: bảng đã scan cho biết mỗi bucket cục bộ bắt đầu ở đâu trong đầu ra')}
        >
          <g data-view="table" data-table={LOCAL.table.join(' ')} data-scanned={scan >= 8 ? LOCAL.scanned.join(' ') : ''}>
            <KeyRow y={18} keys={BLOCKS.flatMap((b) => b.sorted)} tier="shared" label="" tag="local" />{/* i18n-ignore: data-row key */}
            {BLOCKS.flatMap((b) =>
              b.counts.map((c, d) => {
                if (!c) return null;
                const x1 = X0 + (b.first + b.offsets[d]) * CW + 4;
                const x2 = x1 + c * CW - 8;
                return (
                  <g key={`${b.block}-${d}`} opacity={braces}>
                    <path d={`M${x1},54 v6 H${x2} v-6`} fill="none" stroke="var(--shared)" strokeWidth="1.4" />
                    <text x={(x1 + x2) / 2} y={73} fontSize="11" textAnchor="middle" style={{ fill: 'var(--shared)' }}>{c}</text>
                  </g>
                );
              }),
            )}
            <text x={X0} y={110} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>
              <Translated vi="bảng, trong bộ nhớ toàn cục">the table, in global memory</Translated>
            </text>
            {LOCAL.table.map((v, k) => cell(k, 132, v, 'global', k < fill))}
            <text x={X0} y={200} fontSize="12" style={{ fill: 'var(--ink-2)' }} opacity={ramp(p, 0.55, 0.62)}>
              <Translated vi="↓ scan loại trừ">↓ exclusive scan</Translated>
            </text>
            {LOCAL.scanned.map((v, k) => cell(k, 228, v, 'global', k < scan))}
            <text x={tx(0)} y={276} fontSize="11" opacity={scan >= 4 ? 1 : 0} style={{ fill: 'var(--ink-2)' }}>
              <Translated vi="nơi bucket bit 0 của mỗi khối bắt đầu">where each block's zero bucket starts</Translated>
            </text>
            <text x={tx(4)} y={276} fontSize="11" opacity={scan >= 8 ? 1 : 0} style={{ fill: 'var(--ink-2)' }}>
              <Translated vi="nơi bucket bit 1 của mỗi khối bắt đầu">where each block's one bucket starts</Translated>
            </text>
          </g>
        </Frame>
      );
    },
  },
  {
    clip: 'local-sort-5',
    get label() {
      return tr('Write in runs', 'Ghi theo dải');
    },
    draw: (p) => {
      const shown = Math.ceil(ramp(p, 0.05, 0.8) * 16);
      const done = LOCAL.writes.slice(0, shown);
      const outShow = new Set(done.map((w) => w.addr));
      const outKeys = LOCAL.out;
      const b2 = LOCAL.writes.filter((w) => w.block === 2);
      return (
        <Frame headline={tr(`block 2: thread 0 writes position ${b2[0].addr}; threads 1 to 3 write ${b2.slice(1).map((w) => w.addr).join(', ')}`, `khối 2: luồng 0 ghi vị trí ${b2[0].addr}; các luồng 1 đến 3 ghi ${b2.slice(1).map((w) => w.addr).join(', ')}`)}
          note={tr('phase two: within a local bucket, neighbouring threads write neighbouring addresses', 'pha hai: trong một bucket cục bộ, các luồng kề nhau ghi vào các địa chỉ kề nhau')}>
          <g data-view="write" data-block2={b2.map((w) => w.addr).join(' ')}>
            <KeyRow y={Y_IN} keys={BLOCKS.flatMap((b) => b.sorted)} tier="shared" label={tr('shared memory, sorted locally', 'bộ nhớ chia sẻ, đã sắp cục bộ')} hot={new Set([8, 9, 10, 11])} tag="local" />{/* i18n-ignore: data-row key */}
            <g data-row="output">
              <text x={X0} y={Y_OUT + CH + 14} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>
                <Translated vi="đầu ra (bộ nhớ toàn cục)">output (global memory)</Translated>
              </text>
              {outKeys.map((k, i) => (
                <g key={i} data-cell={i} data-key={outShow.has(i) ? bin(k, 4) : ''}>
                  <rect x={X0 + i * CW + 1} y={Y_OUT} width={CW - 2} height={CH} rx="2"
                    fill={outShow.has(i) ? 'var(--global-wash)' : 'var(--cell)'} stroke={outShow.has(i) ? 'var(--global)' : 'var(--rule)'} />
                  {outShow.has(i) ? <text x={cx(i)} y={Y_OUT + 18} fontSize="11.5" textAnchor="middle">{bin(k, 4)}</text> : null}
                </g>
              ))}
            </g>
            {done.map((w) => (
              <Arrow key={`${w.block}-${w.slot}`} x1={cx(w.block * 4 + w.slot)} y1={Y_FROM} x2={cx(w.addr)} y2={Y_OUT - 2}
                strong={w.block === 2} opacity={w.block === 2 ? 1 : 0.55} />
            ))}
            <Separators y1={Y_IN - 2} y2={Y_IN + CH + 4} />
          </g>
        </Frame>
      );
    },
  },
  {
    clip: 'local-sort-6',
    get label() {
      return tr('What it buys', 'Nó đem lại gì');
    },
    draw: (p) => {
      const g = ramp(p, 0.05, 0.5);
      const scale = 15; // pixels per sector
      return (
        <Frame
          headline={tr('sectors per warp store, averaged over a whole sort', 'số sector mỗi lần ghi của warp, lấy trung bình trên cả phép sắp xếp')}
          note={tr("blocks of 256 threads, one random 32-bit key per thread, 32-lane warps, NVIDIA's 32-byte sectors", 'khối 256 luồng, một khóa 32 bit ngẫu nhiên mỗi luồng, warp 32 lane, sector 32 byte của NVIDIA')}
        >
          <g data-view="bars">
            {BARS.map((b, i) => {
              const y = 40 + i * 66;
              return (
                <g key={b.r} data-bars={b.r} data-direct={one(b.direct)} data-local={one(b.local)}>
                  <text x={X0} y={y + 22} fontSize="12" fontWeight="600"><Translated vi={<>chữ số {b.r} bit</>}>{b.r}-bit digit</Translated></text>
                  <text x={X0 + 132} y={y + 13} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="trực tiếp">direct</Translated></text>
                  <rect x={X0 + 140} y={y + 2} width={b.direct * scale * g} height={14} fill="var(--cell)" stroke="var(--global)" />
                  <text x={X0 + 146 + b.direct * scale * g} y={y + 13} fontSize="11">{one(b.direct * g)}</text>
                  <text x={X0 + 132} y={y + 35} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="sắp cục bộ">local sort</Translated></text>
                  <rect x={X0 + 140} y={y + 24} width={b.local * scale * g} height={14} fill="var(--global-wash)" stroke="var(--global)" />
                  <text x={X0 + 146 + b.local * scale * g} y={y + 35} fontSize="11">{one(b.local * g)}</text>
                </g>
              );
            })}
            <line x1={X0 + 140 + 4 * scale} x2={X0 + 140 + 4 * scale} y1={34} y2={300} stroke="var(--ink)" strokeDasharray="4 3" />
            <text x={X0 + 140 + 4 * scale + 4} y={312} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
              <Translated vi="4: một lần ghi được hợp nhất">4: one coalesced store</Translated>
            </text>
          </g>
        </Frame>
      );
    },
  },
];

export function LocalSortScene() {
  return <Scene chapter="ch14" title={tr('Sort locally, then write in runs', 'Sắp cục bộ, rồi ghi theo dải')} beats={beats} />;
}
