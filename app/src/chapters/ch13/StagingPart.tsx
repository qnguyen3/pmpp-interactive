import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BOOK_EXAMPLE,
  PRESETS,
  bookTiled,
  exactTiled,
  iterationsFor,
  outputsPerBlock,
  sortedRandom,
  type Formula,
  type TiledRun,
} from '../../sim/mergetile';
import { TileWasteScene } from './TileWasteScene';

// Section 13.3's part on staging (book §13.6, with §13.7 in one paragraph): why the
// book's tiled kernel loads a full tile of each input, what that wastes, the exact-load
// design the libraries ship (Exercise 3), and lab 13.1's version of it. Rendered inside
// S133 between "What it costs" and "Where you will meet it". Every number about the
// tiled kernel comes from src/sim/mergetile.ts.

// ---------------------------------------------------------------- the book's example, from the model
const BE = BOOK_EXAMPLE;
const BE_TOTAL = BE.m + BE.n;
const BE_PER_BLOCK = outputsPerBlock(BE_TOTAL, BE.blocks, 'corrected'); // 4,000
const BE_ITERS = iterationsFor(BE_PER_BLOCK, BE.tile, 'corrected'); // 4
const BE_ITERS_PRINTED = iterationsFor(BE_PER_BLOCK, BE.tile, 'printed'); // 3
const BE_MISSED_PER_BLOCK = BE_PER_BLOCK - BE_ITERS_PRINTED * BE.tile; // 928
const BE_SHARED_BYTES = 2 * BE.tile * 4; // 8,192
const n = (v: number) => v.toLocaleString('en-US');

export const questionsStaging: Record<string, Question> = {
  'ch13/13.3/q-stage-why-two': {
    kind: 'choice',
    id: 'ch13/13.3/q-stage-why-two',
    prompt: (
      <Translated
        vi={
          <>
            Trong mỗi lần lặp, kernel phân ô của sách ghi <code>tile_size</code> đầu ra, vậy mà trước đó nó nạp{' '}
            <code>tile_size</code> phần tử của A <i>và</i> <code>tile_size</code> phần tử của B vào bộ nhớ chia sẻ (shared
            memory). Vì sao lại cả hai?
          </>
        }
      >
        In each iteration the book's tiled kernel writes <code>tile_size</code> outputs, yet it first loads{' '}
        <code>tile_size</code> elements of A <i>and</i> <code>tile_size</code> elements of B into shared memory. Why
        both?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Hợp nhất truy cập bộ nhớ (coalescing) cần hai vòng nạp chép cùng một số phần tử.">
            Coalescing needs the two load loops to copy the same number of elements.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi vòng nạp tự nó đã được hợp nhất: các luồng liên tiếp chép các phần tử liên tiếp, bất kể số lượng. Lab 13.1 nạp những khoảng có độ dài khác nhau mà vẫn được hợp nhất.">
            Each load loop is coalesced on its own: consecutive threads copy consecutive elements, whatever the count.
            Lab 13.1 loads ranges of different lengths and still coalesces.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cách chia A/B chỉ được tìm ra sau các lần nạp; mọi đầu ra có thể đến từ cùng một đầu vào.">
            The A/B split is found only after the loads; all outputs may come from the same input.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Những gì lần lặp này không dùng sẽ ở lại bộ nhớ chia sẻ và nuôi lần lặp tiếp theo.">
            Whatever this iteration does not use stays in shared memory and feeds the next iteration.
          </Translated>
        ),
        why: (
          <Translated vi="Không phải trong kernel này: lần lặp tiếp theo lại nạp từ A_curr + A_consumed vào A_S[0] và ghi đè nó. Giữ lại nửa đó chính là điều kernel buffer vòng của §13.7 bổ sung.">
            Not in this kernel: the next iteration loads from A_curr + A_consumed into A_S[0] again and overwrites it.
            Keeping that half is what the circular-buffer kernel of §13.7 adds.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một nửa số luồng của khối trộn từ A và nửa còn lại từ B.">
            Half of the block's threads merge from A and the other half from B.
          </Translated>
        ),
        why: (
          <Translated vi="Mọi luồng đều trộn từ cả hai ô, lấy bất cứ thứ gì phần đầu ra của riêng nó cần: 3 của A và 5 của B cho một luồng, 6 và 2 cho luồng khác (p. 318).">
            Every thread merges from both tiles, taking whatever its own share of the outputs needs: 3 of A and 5 of B
            for one thread, 6 and 2 for another (p. 318).
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Khối biết nó sẽ ghi bao nhiêu đầu ra tiếp theo, nhưng không biết chúng đến từ đâu: đó là điều các phép tìm
            co-rank theo từng luồng trên các ô tìm ra, và chúng chạy sau các lần nạp. Trong trường hợp xấu nhất, mọi đầu ra
            của lần lặp đến từ cùng một đầu vào, nên mỗi ô phải chứa <code>tile_size</code> phần tử. Trong mỗi lần lặp đầy
            đủ, đúng một nửa trong <code>2 × tile_size</code> phần tử đã nạp đi vào đầu ra (p. 318); phần còn lại được nạp
            lại ở lần sau.
          </p>
        }
      >
        <p>
          The block knows how many outputs it will write next, not where they come from: that is what the per-thread
          co-rank searches on the tiles find out, and they run after the loads. Worst case, all of the iteration's outputs
          come from the same input, so each tile has to hold <code>tile_size</code> elements. In every full iteration exactly
          half of the <code>2 × tile_size</code> loaded elements go into outputs (p. 318); the rest is loaded again next
          time.
        </p>
      </Translated>
    ),
    source: 'ch. 13 §13.6, pp. 315–316 and 318',
    vi: { source: 'ch. 13 §13.6, pp. 315–316 và 318' },
  },
  'ch13/13.3/q-stage-exact': {
    kind: 'choice',
    id: 'ch13/13.3/q-stage-exact',
    prompt: (
      <Translated
        vi={
          <>
            CUB, rocPRIM và MLX nạp đúng những phần tử mỗi ô sẽ dùng, nên không gì bị nạp hai lần. Thay vào đó, chúng trả
            giá bằng gì?
          </>
        }
      >
        CUB, rocPRIM and MLX load exactly the elements each tile will use, so nothing is loaded twice. What do they pay
        for it instead?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Gấp đôi bộ nhớ chia sẻ mỗi khối, để chứa trường hợp xấu nhất.">
            Twice the shared memory per block, to hold the worst case.
          </Translated>
        ),
        why: (
          <Translated vi="Nếu có thì còn ít hơn: giờ một ô chứa đúng đầu vào của các đầu ra của nó, tổng cộng tile_size phần tử, trong khi sách chứa tile_size của mỗi mảng.">
            Less, if anything: a tile now holds exactly its outputs' inputs, tile_size elements in all, where the book
            holds tile_size of each array.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một buffer vòng để giữ các phần tử còn thừa của mỗi ô.">
            A circular buffer to keep each tile's leftover elements.
          </Translated>
        ),
        why: (
          <Translated vi="Với nạp chính xác thì không có phần thừa nào để giữ. Buffer vòng là câu trả lời khác của sách cho cùng sự lãng phí đó.">
            With exact loads there are no leftovers to keep. The circular buffer is the book's other answer to the same
            waste.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các lần nạp không được hợp nhất, vì hai khoảng có độ dài phụ thuộc vào dữ liệu.">
            Uncoalesced loads, because the two ranges have data-dependent lengths.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi khoảng vẫn được các luồng liên tiếp chép vào các ô nhớ liên tiếp. Chỉ có cận của vòng lặp phụ thuộc vào dữ liệu.">
            Each range is still copied by consecutive threads into consecutive slots. Only the loop bounds depend on the
            data.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Thêm một phép tìm kiếm nhị phân trên bộ nhớ toàn cục cho mỗi ô, chạy trước các lần nạp.">
            An extra binary search on global memory per tile, run before the loads.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách nêu sự đánh đổi này ở p. 316 và để nó thành Bài tập 3: tìm co-rank của điểm cuối ô tiếp theo trước, với
            cái giá là thêm một phép tìm kiếm nhị phân, rồi nạp đúng những gì sẽ được dùng. Số lần đọc của một phép tìm tăng
            theo log2 của độ dài đầu vào, trong khi sự lãng phí nó loại bỏ tăng theo kích thước ô, đó là lý do mọi phép trộn
            phân ô mà khóa học này đã đọc đều chọn đánh đổi này. Các phép tìm ở cấp khối độc lập với nhau, nên CUB và rocPRIM
            làm tất cả cùng lúc, trong một kernel phân chia riêng chạy trước phép trộn.
          </p>
        }
      >
        <p>
          The book names this trade on p. 316 and leaves it as Exercise 3: find the co-rank of the end of the next tile
          first, at the cost of an extra binary search, then load exactly what will be used. The reads of one search grow
          with log2 of the input length, while the waste it removes grows with the tile, which is why every tiled merge read
          for this course makes the trade. The block-level searches are independent of each other, so CUB and rocPRIM do
          them all at once, in a separate partition kernel that runs before the merge.
        </p>
      </Translated>
    ),
    source: 'ch. 13 §13.6, p. 316, and Exercise 3, p. 328; notes/today/ch13-facts.md A7, B2, D1',
    vi: { source: 'ch. 13 §13.6, p. 316, và Bài tập 3, p. 328; notes/today/ch13-facts.md A7, B2, D1' },
  },
};

// ---------------------------------------------------------------- the loaded-versus-used explorer
type DataId = 'uneven' | 'allFromA' | 'shortTileEarly' | 'random' | 'book';
type Scheme = 'book' | 'exact';
const DATA: { id: DataId; label: string }[] = [
  {
    id: 'uneven',
    get label() {
      return tr(PRESETS.uneven.label, 'xen kẽ không đều');
    },
  },
  {
    id: 'allFromA',
    get label() {
      return tr(PRESETS.allFromA.label, 'trường hợp xấu nhất: cả ô từ A');
    },
  },
  {
    id: 'shortTileEarly',
    get label() {
      return tr(PRESETS.shortTileEarly.label, 'A hết sớm');
    },
  },
  {
    id: 'random',
    get label() {
      return tr('random, m = n = 24', 'ngẫu nhiên, m = n = 24');
    },
  },
  {
    id: 'book',
    get label() {
      return tr("the book's example", 'ví dụ của sách');
    },
  },
];
const THREADS = 2;

/** Random arrays for the explorer: 24 keys each in [0, 40), from a seed. */
const randomPair = (seed: number) => ({ A: sortedRandom(24, 40, 2 * seed + 1), B: sortedRandom(24, 40, 2 * seed + 2) });

// The book's example, computed once when first asked for: random 30-bit keys, threads = 1 (the
// per-thread splits do not change any total).
let bookCache: { runs: Record<Formula, TiledRun>; exact: ReturnType<typeof exactTiled> } | null = null;
function bookExample() {
  if (!bookCache) {
    const A = sortedRandom(BE.m, 1 << 30, 1);
    const B = sortedRandom(BE.n, 1 << 30, 2);
    bookCache = {
      runs: { printed: bookTiled(A, B, BE.blocks, BE.tile, 'printed'), corrected: bookTiled(A, B, BE.blocks, BE.tile, 'corrected') },
      exact: exactTiled(A, B, BE.tile),
    };
  }
  return bookCache;
}

const ratio = (a: number, b: number) => (b ? (a / b).toFixed(2) : '–');

function Explorer() {
  const [data, setData] = useState<DataId>('uneven');
  const [seed, setSeed] = useState(1);
  const [scheme, setScheme] = useState<Scheme>('book');
  const [formula, setFormula] = useState<Formula>('corrected');
  const [tile, setTile] = useState(8);

  const arrays = useMemo(() => (data === 'random' ? randomPair(seed) : data === 'book' ? null : { A: PRESETS[data].A, B: PRESETS[data].B }), [data, seed]);
  const run = useMemo(() => (arrays ? bookTiled(arrays.A, arrays.B, 1, tile, formula, THREADS) : null), [arrays, tile, formula]);
  const exact = useMemo(() => (arrays ? exactTiled(arrays.A, arrays.B, tile) : null), [arrays, tile]);
  const big = data === 'book' ? bookExample() : null;

  const pick = (id: DataId) => {
    if (id === 'random' && data === 'random') setSeed(seed + 1);
    setData(id);
  };

  // Totals for the readout and the data-* attributes.
  const tot = big
    ? scheme === 'book'
      ? { ...big.runs[formula].totals, extra: 0 }
      : { ...big.exact.totals, missed: 0, extra: big.exact.totals.extraSearchReads }
    : scheme === 'book'
      ? { ...run!.totals, extra: 0 }
      : { ...exact!.totals, missed: 0, extra: exact!.totals.extraSearchReads };
  const stale = scheme === 'book' && run ? run.blocks[0].iterations.reduce((s, it) => s + it.staleSlots, 0) : 0;
  const m = arrays ? arrays.A.length : BE.m;
  const nn = arrays ? arrays.B.length : BE.n;

  return (
    <Bench
      title={tr('Loaded versus used', 'Đã nạp so với đã dùng')}
      task={
        <Translated
          vi={
            <>
              Trước khi chọn "ngẫu nhiên", hãy dự đoán tỉ lệ đã nạp ÷ đã dùng cho cách làm của sách. Rồi tìm dữ liệu làm sự
              lãng phí tệ nhất.
            </>
          }
        >
          <>Before you pick "random", predict loaded ÷ used for the book's scheme. Then find the data that makes the waste worst.</>
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Chạy kernel phân ô của sách (Figs. 13.11 to 13.13) và cách nạp chính xác từ src/sim/mergetile.ts,{' '}
              {big
                ? `trên ${BE.blocks} khối, mỗi khối được mô hình hóa như một luồng (cách chia giữa các luồng không đổi các tổng).`
                : `trên một khối gồm ${THREADS} luồng.`}{' '}
              Các ô nhớ của một ô: xanh lá khi một đầu ra đã dùng phần tử đó, nét đứt đỏ khi nó được nạp mà không dùng, dấu
              hỏi đỏ khi nó không được nạp ở lần lặp này dù dòng 48 vẫn đếm nó.
            </>
          }
        >
          <>
            Runs the book's tiled kernel (Figs. 13.11 to 13.13) and the exact-load scheme from src/sim/mergetile.ts,{' '}
            {big
              ? `on ${BE.blocks} blocks, each modelled as one thread (the split between threads does not change the totals).`
              : `on one block of ${THREADS} threads.`}{' '}
            A tile's slots: green where an output used the element, red dashes where it was loaded and not used, a red
            question mark where it was not loaded this iteration although line 48 counts it.
          </>
        </Translated>
      }
    >
      <div
        data-explorer="tile"
        data-data={data}
        data-scheme={scheme}
        data-formula={formula}
        data-tile={big ? BE.tile : tile}
        data-loaded={tot.loaded}
        data-used={tot.used}
        data-minimum={tot.minimum}
        data-missed={tot.missed}
        data-extra={tot.extra}
        data-stale={stale}
        data-arrays={arrays ? `${arrays.A.join(',')}|${arrays.B.join(',')}` : 'book'}
      >
        <div className="row" style={{ marginBottom: '0.6rem' }}>
          <span className="muted"><Translated vi="Dữ liệu:">Data:</Translated></span>
          {DATA.map((d) => (
            <button key={d.id} className="quiet" aria-pressed={data === d.id} onClick={() => pick(d.id)}>
              {d.id === 'random' && data === 'random' ? tr('new random', 'ngẫu nhiên mới') : d.label}
            </button>
          ))}
        </div>
        <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
          <span className="seg" role="group" aria-label={tr('Scheme', 'Cách làm')}>
            <button aria-pressed={scheme === 'book'} onClick={() => setScheme('book')}><Translated vi="sách: một ô đầy cho mỗi mảng">book: a full tile of each</Translated></button>
            <button aria-pressed={scheme === 'exact'} onClick={() => setScheme('exact')}><Translated vi="chính xác: tìm trước">exact: search first</Translated></button>
          </span>
          <span className="seg" role="group" aria-label={tr('Line 21', 'Dòng 21')}>
            <button aria-pressed={formula === 'corrected'} onClick={() => setFormula('corrected')} disabled={scheme === 'exact'}>
              <Translated vi="dòng 21 đã sửa">line 21 corrected</Translated>
            </button>
            <button aria-pressed={formula === 'printed'} onClick={() => setFormula('printed')} disabled={scheme === 'exact'}>
              <Translated vi="như bản in">as printed</Translated>
            </button>
          </span>
          <label className="field">{/* i18n-ignore: kernel parameter name tile_size */}
            tile_size
            <select value={big ? BE.tile : tile} disabled={!!big} aria-label={tr('tile size', 'kích thước ô')} onChange={(e) => setTile(Number(e.target.value))}>
              {big ? <option value={BE.tile}>{n(BE.tile)}</option> : [4, 8, 16].map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </label>
        </div>

        <div className="narrate" role="status" style={{ marginBottom: '0.8rem' }} data-readout>
          <Translated
            vi={
              <>
                m = {n(m)}, n = {n(nn)}{big ? `, ${BE.blocks} khối, mỗi khối ${n(BE_PER_BLOCK)} đầu ra` : ', một khối'}. Đã
                nạp <b>{n(tot.loaded)}</b> phần tử vào bộ nhớ chia sẻ cho <b>{n(tot.used)}</b> đầu ra; mọi phép trộn phải
                đọc ít nhất m + n = {n(tot.minimum)}. Đã nạp ÷ đã dùng: <b>{ratio(tot.loaded, tot.used)}</b>.
              </>
            }
          >
            m = {n(m)}, n = {n(nn)}{big ? `, ${BE.blocks} blocks of ${n(BE_PER_BLOCK)} outputs` : ', one block'}. Loaded{' '}
            <b>{n(tot.loaded)}</b> elements into shared memory for <b>{n(tot.used)}</b> outputs; every merge must read at
            least m + n = {n(tot.minimum)}. Loaded ÷ used: <b>{ratio(tot.loaded, tot.used)}</b>.
          </Translated>
          {scheme === 'exact' ? (
            <Translated vi={<> Các phép tìm ranh giới ô đọc thêm {n(tot.extra)} phần tử của bộ nhớ toàn cục.</>}>
              <> The searches for the tile boundaries read {n(tot.extra)} elements of global memory on top.</>
            </Translated>
          ) : null}
          {tot.missed ? (
            <Translated
              vi={
                <>
                  {' '}
                  <b className="t-fail">{n(tot.missed)} đầu ra không bao giờ được ghi</b>: dòng 21 như bản in dừng sớm một lần
                  lặp.
                </>
              }
            >
              <> <b className="t-fail">{n(tot.missed)} outputs were never written</b>: line 21 as printed stops one iteration short.</>
            </Translated>
          ) : null}
          {stale ? (
            <Translated
              vi={
                <>
                  {' '}Có rủi ro: {stale} ô nhớ mà phép tìm của dòng 48 có thể đọc dù không được nạp trong lần lặp của nó,
                  đếm theo <code>A_consumed</code> đã sửa. Một khi dòng 48 như bản in đọc nhầm một ô, các lần lặp sau đó
                  không còn chạy như hình vẽ.
                </>
              }
            >
              <>
                {' '}At risk: {stale} slot{stale === 1 ? '' : 's'} that line 48's search could read although{' '}
                {stale === 1 ? 'it was' : 'they were'} not loaded in {stale === 1 ? 'its' : 'their'} iteration, counted under the
                corrected <code>A_consumed</code>. Once the printed line 48 misreads one, the iterations after it no longer
                run as drawn.
              </>
            </Translated>
          ) : null}
        </div>

        {big ? <BookTable /> : scheme === 'book' ? <BookRows run={run!} tile={tile} /> : <ExactRows run={exact!} tile={tile} />}
      </div>
    </Bench>
  );
}

const CELLS_W = 720;

function BookRows({ run, tile }: { run: TiledRun; tile: number }) {
  const its = run.blocks[0].iterations;
  const cw = Math.min(24, Math.floor((CELLS_W - 330) / (2 * tile)));
  const ax = 76;
  const bxs = ax + tile * cw + 18;
  const tx = bxs + tile * cw + 14;
  const rh = 30;
  const H = 22 + its.length * rh + 6;
  return (
    <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${CELLS_W} ${H}`} role="img" aria-label={tr("Each iteration's tiles: loaded, used and unused slots", 'Các ô của mỗi lần lặp: ô nhớ đã nạp, đã dùng và không dùng')} style={{ width: '100%', minWidth: 520, display: 'block' }}>
        <text x={ax} y={13} fontSize="11" fontWeight="650" style={{ fill: 'var(--shared)' }}>A_S</text>
        <text x={bxs} y={13} fontSize="11" fontWeight="650" style={{ fill: 'var(--shared)' }}>B_S</text>
        {its.map((it, r) => {
          const y = 22 + r * rh;
          const last = r === its.length - 1;
          const slot = (which: 'A' | 'B', s: number) => {
            const loaded = which === 'A' ? it.aLoaded : it.bLoaded;
            const used = which === 'A' ? it.aUsed : it.bUsed;
            return s < used ? 'used' : s < loaded ? 'unused' : last ? 'empty' : 'stale';
          };
          const cell = (which: 'A' | 'B', s: number, x0: number) => {
            const st = slot(which, s);
            return (
              <g key={which + s} data-slot={st}>
                <rect x={x0 + s * cw} y={y} width={cw - 2} height={rh - 8} rx="2"
                  fill={st === 'used' ? 'var(--ok-wash)' : st === 'unused' ? 'var(--shared-wash)' : 'none'}
                  stroke={st === 'used' ? 'var(--ok)' : st === 'empty' ? 'var(--rule)' : 'var(--fail)'}
                  strokeDasharray={st === 'unused' ? '3 2' : st === 'stale' ? '1 2' : st === 'empty' ? '2 3' : undefined}
                  strokeWidth={st === 'used' || st === 'unused' ? 1.5 : 1} />
                {st === 'stale' ? (
                  <text x={x0 + s * cw + (cw - 2) / 2} y={y + (rh - 8) / 2 + 4} fontSize="10" textAnchor="middle" style={{ fill: 'var(--fail)' }}>?</text>
                ) : null}
              </g>
            );
          };
          return (
            <g key={r} data-iteration={r} data-aloaded={it.aLoaded} data-bloaded={it.bLoaded} data-aused={it.aUsed} data-bused={it.bUsed} data-stale={it.staleSlots}>
              <text x={8} y={y + 15} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>lần lặp {r}</>}>iteration {r}</Translated></text>
              {Array.from({ length: tile }, (_, s) => cell('A', s, ax))}
              {Array.from({ length: tile }, (_, s) => cell('B', s, bxs))}
              <text x={tx} y={y + 10} fontSize="10.5" style={{ fill: 'var(--ink)' }}>
                <Translated vi={<>đã dùng {it.aUsed} + {it.bUsed} trong {it.aLoaded + it.bLoaded} đã nạp</>}>
                  {it.aUsed} + {it.bUsed} used of {it.aLoaded + it.bLoaded} loaded
                </Translated>
              </text>
              <text x={tx} y={y + 22} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>
                {it.threads.map((t, q) => `t${q}: ${t.a}A ${t.b}B`).join(', ')}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function ExactRows({ run, tile }: { run: ReturnType<typeof exactTiled>; tile: number }) {
  const cw = Math.min(24, Math.floor((CELLS_W - 330) / (2 * tile)));
  const ax = 76;
  const tx = ax + 2 * tile * cw + 32;
  const rh = 30;
  const H = 22 + run.tiles.length * rh + 6;
  return (
    <div style={{ overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${CELLS_W} ${H}`} role="img" aria-label={tr('Each tile: exactly the elements its outputs use', 'Mỗi ô: đúng những phần tử mà các đầu ra của nó dùng')} style={{ width: '100%', minWidth: 520, display: 'block' }}>
        <text x={ax} y={13} fontSize="11" fontWeight="650" style={{ fill: 'var(--shared)' }}><Translated vi="ô: phần của A, rồi phần của B">tile: A's part, then B's</Translated></text>
        {run.tiles.map((t, r) => {
          const y = 22 + r * rh;
          const a = t.aRange[1] - t.aRange[0];
          const b = t.bRange[1] - t.bRange[0];
          return (
            <g key={r} data-tile-row={r} data-a={a} data-b={b} data-extra={t.extraSearchReads}>
              <text x={8} y={y + 15} fontSize="11" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>ô {r}</>}>tile {r}</Translated></text>
              {Array.from({ length: a + b }, (_, s) => (
                <rect key={s} x={ax + s * cw} y={y} width={cw - 2} height={rh - 8} rx="2" fill="var(--ok-wash)" stroke="var(--ok)" strokeWidth="1.5"
                  data-slot="used" />
              ))}
              {a > 0 && b > 0 ? <line x1={ax + a * cw - 1} y1={y - 3} x2={ax + a * cw - 1} y2={y + rh - 5} stroke="var(--ink)" strokeWidth="2" /> : null}
              <text x={tx} y={y + 10} fontSize="10.5" style={{ fill: 'var(--ink)' }}>
                <Translated vi={<>{a} của A + {b} của B, dùng hết</>}>
                  {a} of A + {b} of B, all used
                </Translated>
              </text>
              <text x={tx} y={y + 22} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>
                <Translated vi={<>tìm điểm cuối của nó: {t.extraSearchReads} lần đọc</>}>
                  search for its end: {t.extraSearchReads} reads
                </Translated>
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function BookTable() {
  const big = bookExample();
  const rows: [string, string, number, number, number, number][] = [
    [tr('book, line 21 as printed', 'sách, dòng 21 như bản in'), tr(`${BE_ITERS_PRINTED} per block`, `${BE_ITERS_PRINTED} mỗi khối`), big.runs.printed.totals.loaded, big.runs.printed.totals.used, big.runs.printed.totals.missed, 0],
    [tr('book, line 21 corrected', 'sách, dòng 21 đã sửa'), tr(`${BE_ITERS} per block`, `${BE_ITERS} mỗi khối`), big.runs.corrected.totals.loaded, big.runs.corrected.totals.used, 0, 0],
    [tr('exact: search first', 'chính xác: tìm trước'), tr(`${big.exact.tiles.length} tiles`, `${big.exact.tiles.length} ô`), big.exact.totals.loaded, big.exact.totals.used, 0, big.exact.totals.extraSearchReads],
  ];
  return (
    <div style={{ overflowX: 'auto' }}>
      <table className="data" style={{ margin: 0 }} data-table="book-example">
        <thead>
          <tr>
            <th>
              <Translated vi="Cách làm">Scheme</Translated>
            </th>
            <th>
              <Translated vi="Số lần lặp">Iterations</Translated>
            </th>
            <th className="num">
              <Translated vi="Đã nạp">Loaded</Translated>
            </th>
            <th className="num">
              <Translated vi="Đã dùng">Used</Translated>
            </th>
            <th className="num">
              <Translated vi="Không được ghi">Never written</Translated>
            </th>
            <th className="num">
              <Translated vi="Lần đọc thêm để tìm">Extra search reads</Translated>
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r[0]}>
              <td>{r[0]}</td>
              <td>{r[1]}</td>
              <td className="num">{n(r[2])}</td>
              <td className="num">{n(r[3])}</td>
              <td className="num">{r[4] ? n(r[4]) : '0'}</td>
              <td className="num">{r[5] ? n(r[5]) : '–'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted" style={{ margin: '0.5rem 0 0', fontSize: '0.85rem' }}>
        <Translated vi="Khóa ngẫu nhiên. Cách làm của sách đã sửa nạp ít hơn hai lần số đầu ra vì lần lặp cuối của một khối chỉ nạp phần còn lại của các lát của nó.">
          Random keys. The corrected book scheme loads less than twice the outputs because a block's last iteration loads
          only what is left of its slices.
        </Translated>
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- lab 13.1's version
/** The file from the line matching `start` up to (not including) the line matching `end`. */
function partOf(file: string | null, start: RegExp, end: RegExp): string | null {
  if (file == null) return null;
  const lines = file.split('\n');
  const a = lines.findIndex((l) => start.test(l));
  if (a < 0) return file;
  const b = lines.findIndex((l, i) => i > a && end.test(l));
  return lines.slice(a, b < 0 ? undefined : b).join('\n').replace(/\s+$/, '');
}

const LAB = 'ch13/lab1_merge';
export const LAB_TIERS: Record<string, Tier> = {
  A: 'global', B: 'global', C: 'global', bounds: 'global',
  tile: 'shared', A_S: 'shared', B_S: 'shared', As: 'shared', Bs: 'shared',
  b: 'reg', k: 'reg', i: 'reg', tileSize: 'reg', numTiles: 'reg', C_curr: 'reg', C_next: 'reg', A_curr: 'reg', A_next: 'reg',
  B_curr: 'reg', B_next: 'reg', A_length: 'reg', B_length: 'reg', C_length: 'reg', c_curr: 'reg', c_next: 'reg',
  a_curr: 'reg', a_next: 'reg', b_curr: 'reg', b_next: 'reg',
};

function LabTiled() {
  const { reports } = useReports();
  const cu = partOf(useLabFile(`${LAB}/solution/kernel.cu`), /^\/\/ -{20,} Part \(b\)/, /^\/\/ -{20,} Provided/);
  const metal = partOf(useLabFile(`${LAB}/solution/kernel.metal`), /^\/\/ Part \(b\), step 1/, /^\/\/ -{20,} Provided/);
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu ?? `// loading labs/${LAB}/solution/kernel.cu`, tiers: LAB_TIERS, status: solutionTabStatus(reports, LAB, 'kernel.cu') },
        { label: 'Metal', code: metal ?? `// loading labs/${LAB}/solution/kernel.metal`, tiers: LAB_TIERS, status: solutionTabStatus(reports, LAB, 'kernel.metal') },
      ]}
      caption={
        <Translated
          vi={
            <>
              Phần (b) của lời giải tham chiếu (solution) của lab 13.1, nạp từ <code>labs/{LAB}/solution/</code>: một kernel
              phân chia, rồi mỗi khối một ô, nạp chính xác. <code>co_rank</code> và <code>merge_sequential</code> là Figs.
              13.5 và 13.2 đã sửa, trên các cặp khóa–giá trị. Hãy tự viết nó trong <a href={href('ch13', 'labs')}>lab</a>{' '}
              trước khi xem bản này.
            </>
          }
        >
          <>
            Part (b) of lab 13.1's reference solution, loaded from <code>labs/{LAB}/solution/</code>: a partition kernel,
            then one tile per block, loaded exactly. <code>co_rank</code> and <code>merge_sequential</code> are the corrected
            Figs. 13.5 and 13.2 on key–value pairs. Write it yourself in the <a href={href('ch13', 'labs')}>lab</a> before
            studying this one.
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- the part
export function StagingPart() {
  const q = questionsStaging;
  return (
    <>
      <h2>
        <Translated vi="Đưa đầu vào vào bộ nhớ chia sẻ">Stage the inputs in shared memory</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các truy cập của kernel cơ bản phụ thuộc vào dữ liệu, nên không cách ánh xạ lại luồng nào xếp chúng thẳng hàng.
            Chiến lược thứ ba của chương 6 hợp với trường hợp này: chép dữ liệu vào bộ nhớ chia sẻ (shared memory) bằng các
            lần nạp được hợp nhất (coalesced), rồi thực hiện các truy cập không đều ở đó, nơi hợp nhất không quan trọng. Điều
            làm việc này khả thi là các lát đầu vào của những luồng kề nhau cũng kề nhau trong bộ nhớ. Tất cả luồng của một
            khối cùng nhau đọc một đoạn liền của A và một đoạn liền của B, và mỗi ranh giới khối chỉ cần một co-rank để tìm
            ra các đoạn đó.
          </p>
        }
      >
        <p>
          The basic kernel's accesses depend on the data, so no remapping of threads lines them up. Chapter 6's third
          strategy fits this case: copy the data into shared memory with coalesced loads, then make the irregular accesses
          there, where coalescing does not matter. What makes it possible is that neighbouring threads' input slices are
          neighbours in memory. All the threads of a block together read one contiguous piece of A and one of B, and one
          co-rank per block boundary finds those pieces.
        </p>
      </Translated>
      <Source section="13.6" pages="314–315" figs="Fig. 13.10" />
      <Translated
        vi={
          <p>
            Kernel phân ô của sách (Figs. 13.11 to 13.13) có ba phần. Luồng 0 của mỗi khối tính hai co-rank của khối trên
            bộ nhớ toàn cục (global memory) và trao chúng cho các luồng khác qua bộ nhớ chia sẻ, giữa hai rào chắn (barrier).
            Sau đó khối lặp. Mỗi lần lặp chép <code>tile_size</code> phần tử của A và <code>tile_size</code> phần tử của B
            vào bộ nhớ chia sẻ, các luồng liên tiếp trên các phần tử liên tiếp. Sau một rào chắn, mọi luồng chạy{' '}
            <code>co_rank</code> hai lần trên hai ô để tìm phần của riêng nó trong <code>tile_size</code> đầu ra của lần
            lặp, rồi trộn từ bộ nhớ chia sẻ vào C. Một rào chắn cuối giữ cho các lần nạp của lần lặp tiếp theo không chạm vào
            những ô vẫn đang được đọc.
          </p>
        }
      >
        <p>
          The book's tiled kernel (Figs. 13.11 to 13.13) has three parts. Thread 0 of each block computes the block's two
          co-ranks on global memory and hands them to the other threads through shared memory, between two barriers. Then
          the block loops. Each iteration copies <code>tile_size</code> elements of A and <code>tile_size</code> of B into
          shared memory, consecutive threads on consecutive elements. After a barrier, every thread runs <code>co_rank</code>{' '}
          twice on the two tiles to find its own share of the iteration's <code>tile_size</code> outputs, and merges from
          shared memory into C. A last barrier keeps the next iteration's loads off tiles that are still being read.
        </p>
      </Translated>
      <Source section="13.6" pages="316–318" figs="Figs. 13.11–13.13" />
      <Translated
        vi={
          <p>
            Ví dụ của sách cho kích thước: m = {n(BE.m)}, n = {n(BE.n)}, {BE.blocks} khối, mỗi khối {BE.threads} luồng và{' '}
            <code>tile_size</code> = {n(BE.tile)}. Mỗi khối ghi {n(BE_PER_BLOCK)} đầu ra trong {BE_ITERS} lần lặp, mỗi luồng{' '}
            {BE.tile / BE.threads} đầu ra mỗi lần lặp, và hai ô chiếm ({n(BE.tile)} + {n(BE.tile)}) × 4 ={' '}
            {n(BE_SHARED_BYTES)} byte bộ nhớ chia sẻ. Đó là các con số trong ví dụ của sách, không phải của thiết bị nào.
            Fig. 13.12 dòng 21 đếm số lần lặp bằng cùng phép <code>ceil</code> của một thương số nguyên như dòng 03 của kernel
            cơ bản, nên theo bản in mỗi khối chạy {BE_ITERS_PRINTED} lần lặp, không phải {BE_ITERS}, và{' '}
            {n(BE_MISSED_PER_BLOCK)} đầu ra của nó không bao giờ được ghi. Trình khám phá bên dưới có cùng công tắc "như bản
            in".
          </p>
        }
      >
        <p>
          The book's example sizes it: m = {n(BE.m)}, n = {n(BE.n)}, {BE.blocks} blocks of {BE.threads} threads and{' '}
          <code>tile_size</code> = {n(BE.tile)}. Each block writes {n(BE_PER_BLOCK)} outputs in {BE_ITERS} iterations, each
          thread {BE.tile / BE.threads} outputs per iteration, and the two tiles take ({n(BE.tile)} + {n(BE.tile)}) × 4 ={' '}
          {n(BE_SHARED_BYTES)} bytes of shared memory. Those are the book's example figures, not any device's. Fig. 13.12
          line 21 counts the iterations with the same <code>ceil</code> of an integer quotient as the basic kernel's line
          03, so as printed each block runs {BE_ITERS_PRINTED} iterations, not {BE_ITERS}, and{' '}
          {n(BE_MISSED_PER_BLOCK)} of its outputs are never written. The explorer below has the same "as printed" switch.
        </p>
      </Translated>
      <Source section="13.6" pages="316–317" />

      <Translated
        vi={
          <p>
            Giờ đến phần khiến phép trộn khó phân ô hơn mọi thứ trước nó. Trước khi đọc tiếp, hãy trả lời câu này.
          </p>
        }
      >
        <p>
          Now the part that makes merge harder to tile than anything before it. Before you read on, answer this one.
        </p>
      </Translated>
      <Ask q={q['ch13/13.3/q-stage-why-two']} label={tr('Before you read on', 'Trước khi đọc tiếp')} />
      <Translated
        vi={
          <p>
            Khối không thể biết <code>tile_size</code> đầu ra tiếp theo chia giữa A và B thế nào cho đến khi nó đã tìm
            trên các ô, và các phép tìm đến sau các lần nạp. Trong trường hợp xấu nhất, tất cả chúng đến từ cùng một đầu
            vào, nên khối nạp một ô đầy của mỗi mảng, <code>2 × tile_size</code> phần tử, để chắc chắn có{' '}
            <code>tile_size</code> đầu ra. Những phần tử nó không dùng được nạp lại ở lần lặp tiếp theo, và kết luận của sách
            là điều này "lãng phí một nửa băng thông bộ nhớ".
          </p>
        }
      >
        <p>
          The block cannot know how the next <code>tile_size</code> outputs split between A and B until it has searched the
          tiles, and the searches come after the loads. In the worst case every one of them comes from the same input, so
          the block loads a full tile of each, <code>2 × tile_size</code> elements, to be sure of <code>tile_size</code>{' '}
          outputs. The elements it does not use are loaded again in the next iteration, and the book's verdict is that this
          "wastes half of the memory bandwidth".
        </p>
      </Translated>
      <Source section="13.6" pages="315–316, 318, 320" />
      <Translated
        vi={<p>Hoạt hình cho thấy một khối với ô 8 phần tử qua hai lần lặp, rồi trường hợp xấu nhất, rồi cách sửa.</p>}
      >
        <p>The animation shows one block with a tile of 8 through two iterations, then the worst case, then the fix.</p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một khối gồm hai luồng, ô tám phần tử, và đếm phần tử thay vì byte. Hình vẽ được tính bởi
              src/sim/mergetile.ts; các luồng trông như di chuyển đồng bộ, điều đó không quan trọng ở đây.
            </p>
          }
        >
          <p>
            One block of two threads, eight-element tiles, and counts of elements rather than bytes. The drawings are
            computed by src/sim/mergetile.ts; the threads look as if they move in step, which does not matter here.
          </p>
        </Translated>
      </Simplification>
      <TileWasteScene />

      <Translated
        vi={
          <p>
            Trình khám phá chạy cùng mô hình trên toàn bộ phép trộn. Mỗi hàng là một lần lặp của một khối: nó đã nạp gì vào{' '}
            <code>A_S</code> và <code>B_S</code>, và các đầu ra của nó đã dùng những ô nhớ nào trong số đó. Ba cấu hình dựng
            sẵn được tạo để cho thấy một điều gì đó: một cách xen kẽ không đều, một phép trộn mà cả ô đến từ A, và một phép
            trộn mà A hết sau ô đầu tiên. "Ví dụ của sách" chạy các kích thước ở trên với khóa ngẫu nhiên.
          </p>
        }
      >
        <p>
          The explorer runs the same model on whole merges. Each row is one iteration of one block: what it loaded into{' '}
          <code>A_S</code> and <code>B_S</code>, and which of those slots its outputs used. The three presets are built to
          show something: an uneven interleaving, a merge where whole tiles come from A, and one where A runs out after
          the first tile. "The book's example" runs the sizes above on random keys.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Đếm phần tử, không đếm byte, không có thời gian, warp hay xung đột bank (bank conflict). Nó tăng{' '}
              <code>A_consumed</code> theo đúng số các phép trộn thực sự đã dùng, điều mà dòng 48 không phải lúc nào cũng
              tính đúng (bên dưới). Các dấu hỏi được đếm theo <code>A_consumed</code> đã sửa đó.
            </p>
          }
        >
          <p>
            Counts elements, not bytes, with no timing, warps or bank conflicts. It advances <code>A_consumed</code> by
            what the merges actually used, which line 48 does not always compute (below). The question marks are counted
            under that corrected <code>A_consumed</code>.
          </p>
        </Translated>
      </Simplification>
      <Explorer />
      <Translated
        vi={
          <p>
            Dòng 48 của Fig. 13.13 tăng <code>A_consumed</code> thêm{' '}
            <code>co_rank(tile_size, A_S, tile_size, B_S, tile_size)</code>, như thể cả hai ô đều đầy. Sách cảnh báo rằng
            số đếm có thể sai sau lần lặp cuối, nơi nó không còn được dùng (p. 320). Bản giả lập đoạn mã của chính chúng tôi
            tìm ra một trường hợp thứ hai: một lần lặp không phải lần cuối nhưng ô của nó chỉ được nạp một phần, vì một đầu
            vào đã hết. Khi đó phép tìm có thể đọc những ô nhớ vẫn còn giữ ô trước. Chọn "A hết sớm" và các ô nhớ có rủi ro
            được đánh dấu bằng dấu hỏi. Một phép tìm kiếm nhị phân chỉ đọc vài ô trong số đó: ở đó, lần lặp 1 chỉ nạp A[8]
            và A[9], dòng 48 đọc <code>A_S[7]</code>, vẫn còn giữ A[7], và đếm 8 phần tử của A trong khi các phép trộn dùng
            2. Dòng 50 của kernel buffer vòng đếm với độ dài <code>min(tile_size, ...)</code>, và dạng đó là đúng.
          </p>
        }
      >
        <p>
          Line 48 of Fig. 13.13 moves <code>A_consumed</code> on by{' '}
          <code>co_rank(tile_size, A_S, tile_size, B_S, tile_size)</code>, as if both tiles were full. The book warns that
          the count can be wrong after the last iteration, where it is no longer used (p. 320). Our own emulation of the
          listing found a second case: an iteration that is not the last but whose tile is only partly loaded, because one
          input ran out. The search can then read slots still holding the previous tile. Pick "A runs out early" and the
          slots at risk are marked with a question mark. A binary search reads only a few of them: there, iteration 1
          loads just A[8] and A[9], line 48 reads <code>A_S[7]</code>, which still holds A[7], and counts 8 elements of A
          where the merges used 2. The circular-buffer kernel's line 50 counts with <code>min(tile_size, ...)</code>{' '}
          lengths, and that form is correct.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Trường hợp lỗi thứ hai của dòng 48 là của chúng tôi, từ một bản giả lập tuần tự Figs. 13.11 to 13.13
              (notes/ch13.md, "Unclear in the PDF", mục 18). Sách không nhắc đến nó.
            </p>
          }
        >
          <p>
            The second failure case of line 48 is ours, from a sequential emulation of Figs. 13.11 to 13.13 (notes/ch13.md,
            "Unclear in the PDF", item 18). The book does not mention it.
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Nạp đúng những gì một ô dùng">Load exactly what a tile uses</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách nêu lối thoát kia ở p. 316 và để nó thành Bài tập 3. Tìm trước: tìm co-rank của điểm cuối các đầu ra của ô
            tiếp theo trên bộ nhớ toàn cục, rồi nạp đúng những phần tử A và B mà các đầu ra đó cần. Nó tốn thêm một phép tìm
            kiếm nhị phân mỗi ô, và mọi phần tử đã nạp đều được dùng. Chọn "chính xác: tìm trước" trong trình khám phá và so
            sánh các số đếm.
          </p>
        }
      >
        <p>
          The book names the other way out on p. 316 and leaves it as Exercise 3. Search first: find the co-rank of the end
          of the next tile's outputs on global memory, then load exactly the A and B elements those outputs need. It costs
          one more binary search per tile, and every loaded element is used. Choose "exact: search first" in the explorer
          and compare the counts.
        </p>
      </Translated>
      <Source section="13.6" pages="316, 328" />
      <Ask q={q['ch13/13.3/q-stage-exact']} />
      <Translated
        vi={
          <p>
            Mọi phép trộn phân ô mà khóa học này đã đọc (CUB, rocPRIM, MLX, PyTorch MPS) đều đi đường đó. CUB và rocPRIM
            chạy một kernel phân chia lưu co-rank của mọi ranh giới ô; rồi mỗi khối nạp đúng hai khoảng của nó vào bộ nhớ
            chia sẻ, tìm theo từng luồng ở đó và trộn. Merge sort trên Metal của MLX được xây theo cùng cách. Phép trộn MPS
            của PyTorch bỏ qua kernel phân chia: mọi luồng của một khối tự tính cách chia của khối, trong thanh ghi
            (register), và chú thích trong mã nói lý do, "same result, no barriers". Đó là một bước cố ý rời khỏi cách phát
            tán từ luồng 0 của sách. Phép trộn Metal của llama.cpp, gặp ở dưới, hoàn toàn không đưa vào bộ nhớ chia sẻ: nó
            trộn thẳng từ bộ nhớ thiết bị, như kernel cơ bản.
          </p>
        }
      >
        <p>
          Every tiled merge read for this course (CUB, rocPRIM, MLX, PyTorch MPS) takes that route. CUB and rocPRIM run a
          partition kernel that stores the co-rank of every tile boundary; then each block loads exactly its two ranges
          into shared memory, searches per thread there and merges. MLX's Metal merge sort is built the same way.
          PyTorch's MPS merge skips the partition kernel: every thread of a block computes the block's split itself, in
          registers, and its comment says why, "same result, no barriers". That is a deliberate step away from the book's
          thread-0 broadcast. llama.cpp's Metal merge, met below, does not stage at all: it merges straight from device
          memory, like the basic kernel.
        </p>
      </Translated>
      <Today
        checked={tr(
          "2026-11-20 in the projects' source (notes/today/ch13-facts.md A7, B2, D1, D2)",
          '2026-11-20 trong mã nguồn của các dự án (notes/today/ch13-facts.md A7, B2, D1, D2)',
        )}
      >
        <Translated
          vi={
            <p>
              CUB: <code>device_partition_merge_path_kernel</code> và <code>agent_merge.cuh</code>. rocPRIM:{' '}
              <code>partition_kernel_impl</code>. MLX: <code>mb_block_partition</code>, <code>mb_block_merge</code>.
              PyTorch: <code>mb_merge_body</code> trong <code>SortMerge.h</code>.
            </p>
          }
        >
          <p>
            CUB: <code>device_partition_merge_path_kernel</code> and <code>agent_merge.cuh</code>. rocPRIM:{' '}
            <code>partition_kernel_impl</code>. MLX: <code>mb_block_partition</code>, <code>mb_block_merge</code>.
            PyTorch: <code>mb_merge_body</code> in <code>SortMerge.h</code>.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Phần (b) của lab 13.1 là thiết kế này, mỗi khối một ô. Đây là nửa phân ô trong lời giải tham chiếu (solution) của
            nó; tab Metal cho thấy cùng các kernel với một ô <code>threadgroup</code>, và hàm trợ giúp co-rank ở đó là một
            template trên kiểu con trỏ, vì các không gian địa chỉ của Metal tách rời nhau.
          </p>
        }
      >
        <p>
          Lab 13.1's part (b) is this design, one tile per block. Here is its reference solution's tiled half; the Metal tab
          shows the same kernels with a <code>threadgroup</code> tile, and the co-rank helper there is a template on the
          pointer type, because Metal's address spaces are disjoint.
        </p>
      </Translated>
      <LabTiled />

      <h2>
        <Translated vi="Buffer vòng, tóm tắt">The circular buffer, in brief</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Câu trả lời của chính sách, §13.7, giữ vòng lặp phân ô và thôi vứt dữ liệu đi. Các phần tử chưa dùng của mỗi ô ở
            lại trong bộ nhớ chia sẻ, chỉ những ô nhớ đã tiêu thụ được nạp lại, và vị trí quay vòng qua cuối buffer bằng{' '}
            <code>%</code>. Để giữ việc quay vòng ra khỏi mã cấp luồng, mọi vị trí được lưu như một độ lệch tính từ đầu ô,
            nên một ô luôn trông như một dải thẳng, và chỉ hai hàm, <code>co_rank_circular</code> và{' '}
            <code>merge_sequential_circular</code>, biến độ lệch thành chỉ số đã quay vòng. Sách rút ra bài học trong Fig.
            13.17: giấu một sự phức tạp sau giao diện duy nhất chạm vào dữ liệu, và mã dùng nó gần như không thay đổi.
          </p>
        }
      >
        <p>
          The book's own answer, §13.7, keeps the tiled loop and stops throwing data away. Each tile's unused elements stay
          in shared memory, only the consumed slots are refilled, and positions wrap around the end of the buffer with{' '}
          <code>%</code>. To keep the wrap out of the thread-level code, every position is kept as an offset from the tile's
          start, so a tile always looks like one straight run, and only two functions, <code>co_rank_circular</code> and{' '}
          <code>merge_sequential_circular</code>, turn offsets into wrapped indices. The book draws the lesson in Fig. 13.17:
          hide a complication behind the one interface that touches the data, and the code that uses it barely changes.
        </p>
      </Translated>
      <Source section="13.7" pages="321–327" figs="Figs. 13.15–13.20" />
      <Translated
        vi={
          <p>
            Khóa học này không xây phần nào của nó. Không thư viện hay dự án nào được đọc cho khóa học vào 2026-11-20 dùng buffer vòng cho phép trộn: bốn cái có phân ô đều nạp chính xác, nên không còn gì thừa để giữ, và phép trộn Metal của
            llama.cpp hoàn toàn không đưa vào bộ nhớ chia sẻ. Các đoạn mã cũng có vài chỗ nhầm, được liệt kê ở{' '}
            <a href={href('ch13', 'wrap')}>trang tóm tắt</a>.
          </p>
        }
      >
        <p>
          This course builds none of it. No library or project read for it on 2026-11-20 uses a circular buffer for a
          merge: the four tiled ones load exactly, which leaves nothing over to keep, and llama.cpp's Metal merge does not
          stage at all. The listings also carry several slips, which the{' '}
          <a href={href('ch13', 'wrap')}>summary page</a> lists.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch13-facts.md A7, B2, D1, D2, D3)" /* i18n-ignore: date and citation */>
        <Translated
          vi={
            <p>
              CUB, rocPRIM, MLX và PyTorch MPS đều tính cách chia ô trước và nạp chính xác; không cái nào giữ phần thừa.{' '}
              <code>kernel_argsort_merge_f32_i32</code> của llama.cpp đọc trực tiếp bộ nhớ thiết bị.
            </p>
          }
        >
          <p>
            CUB, rocPRIM, MLX and PyTorch MPS all compute the tile splits first and load exactly; none keeps leftovers.
            llama.cpp's <code>kernel_argsort_merge_f32_i32</code> reads device memory directly.
          </p>
        </Translated>
      </Today>
    </>
  );
}
