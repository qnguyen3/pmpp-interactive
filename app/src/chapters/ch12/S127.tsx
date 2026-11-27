import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { useReports } from '../../lib/reports';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BOOK_PUZZLE,
  FIG_KEEP,
  KIND_LABEL,
  dependencePath,
  orderingTable,
  orderingVerdict,
  overlap,
  type Direction,
  type PathNode,
  type PuzzleConfig,
  type Who,
  type Write,
} from '../../sim/inplace';
import { ScanChainScene } from './ScanChainScene';

// Section 12.7, p. 300. Vendor facts: notes/today/ch12-facts.md A8, A9, A10, B5, B6, C4 and
// ch11-facts.md C5. Every number the explorer and the puzzle show comes from sim/inplace.ts.

const LAB5 = 'ch12/lab5_inplace/solution/kernel.cu';
// Lines the listing marks, found by their text so that edits to the lab file cannot shift them.
const LAB5_MARKS = [
  'counter.fetch_add(1',
  '// every read of this block',
  'memory_order_acquire) == 0',
  'memory_order_release)',
  '// the whole block writes only after',
];
const marked = (code: string | null) =>
  code ? code.split('\n').flatMap((l, i) => (LAB5_MARKS.some((m) => l.includes(m)) ? [i + 1] : [])) : [];

export const questionsS127: Record<string, Question> = {
  'ch12/12.7/q-erratum': {
    kind: 'choice',
    id: 'ch12/12.7/q-erratum',
    prompt: (
      <Translated
        vi={
          <>
            Trang 300 nói luồng 1 phải đọc xong k0 trước khi luồng 3 ghi k3. Trong ví dụ của Hình 12.10 (k1, k3, k4, k6, k8,
            k10, k11 và k14 được giữ), khóa nào phải được đọc trước khi k3 được ghi vào ô mới của nó?
          </>
        }
      >
        <>
          Page 300 says that thread 1 must finish reading k0 before thread 3 writes k3. In Fig. 12.10's example (k1, k3,
          k4, k6, k8, k10, k11 and k14 kept), which key must have been read before k3 is written into its new slot?
        </>
      </Translated>
    ),
    options: [
      {
        label: 'k0',
        why: (
          <Translated vi="Ô 0 nhận k1, không phải k3. k0 là khóa bị k1 ghi đè.">
            Slot 0 receives k1, not k3. k0 is the key k1 overwrites.
          </Translated>
        ),
      },
      { label: 'k1', correct: true },
      { label: 'k2', why: <Translated vi="Ô 2 nhận k4.">Slot 2 receives k4.</Translated> },
      {
        label: (
          <Translated vi="Chính k3">
            k3 itself
          </Translated>
        ),
        why: (
          <Translated vi="k3 được đọc bởi chính luồng ghi nó, luồng này dù sao cũng đọc trước khi ghi.">
            k3 is read by the very thread that writes it, which reads before it writes in any case.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tính đến k3 có hai khóa được giữ, k1 rồi k3, nên k3 là khóa được giữ thứ hai và rơi vào ô 1. Ô 1 đang chứa k1. Chữ
            "k0" trên trang là một lỗi viết nhầm; chính câu đó nói k3 chiếm chỗ của k1.
          </p>
        }
      >
        <p>
          Two keys are kept before k3 is reached, k1 and then k3, so k3 is the second kept key and lands in slot 1. Slot
          1 held k1. The page's "k0" is a slip; its own sentence says k3 takes k1's place.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.7, p. 300; Fig. 12.10; notes "Unclear in the PDF"',
    vi: { source: 'ch. 12 §12.7, p. 300; Hình 12.10; ghi chú "Unclear in the PDF"' },
  },
  'ch12/12.7/q-crossblock': {
    kind: 'numeric',
    id: 'ch12/12.7/q-crossblock',
    prompt: (
      <Translated
        vi={
          <>
            Các khóa của Hình 12.10 (k1, k3, k4, k6, k8, k10, k11 và k14 được giữ trong số k0 đến k15) được lọc tại chỗ bởi hai
            khối, mỗi khối tám khóa. Bao nhiêu lần ghi của khối 1 rơi vào các ô mà khối 0 đọc?
          </>
        }
      >
        <>
          Fig. 12.10's keys (k1, k3, k4, k6, k8, k10, k11 and k14 kept out of k0 to k15) are filtered in place by two
          blocks of eight keys each. How many of block 1's writes land on slots that block 0 reads?
        </>
      </Translated>
    ),
    answer: 4,
    unit: 'writes',
    vi: { unit: 'lần ghi', source: 'ch. 12 §12.7, p. 300; Hình 12.10' },
    explain: (
      <Translated
        vi={
          <p>
            Khối 0 giữ bốn khóa, nên các khóa được giữ của khối 1 là k8, k10, k11 và k14 đi tới các ô 4, 5, 6 và 7. Cả bốn ô đều
            nằm trong phạm vi của khối 0, từ 0 đến 7, nên mọi lần ghi của khối 1 đều phải chờ đến khi khối 0 đã đọc.
          </p>
        }
      >
        <p>
          Block 0 keeps four keys, so block 1's kept keys k8, k10, k11 and k14 go to slots 4, 5, 6 and 7. All four slots
          are inside block 0's range, 0 to 7, so every one of block 1's writes must wait until block 0 has read.
        </p>
      </Translated>
    ),
    source: 'ch. 12 §12.7, p. 300; Fig. 12.10',
  },
  'ch12/12.7/q-relaxed': {
    kind: 'choice',
    id: 'ch12/12.7/q-relaxed',
    prompt: (
      <Translated
        vi={
          <>
            Một đồng nghiệp chép lựa chọn mà CUB dùng cho các cờ look-back khi làm không tại chỗ, tức các lần load và store
            relaxed, rồi gọi kernel tại chỗ. Điều gì có thể hỏng?
          </>
        }
      >
        <>
          A colleague copies the out-of-place choice CUB makes for its look-back flags, relaxed loads and stores, and
          calls the kernel in place. What can go wrong?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Chỉ là nó chậm hơn: các lần load relaxed thấy cờ muộn hơn, nên các khối quay chờ lâu hơn.">
            Only that it is slower: relaxed loads see the flag later, so blocks spin longer.
          </Translated>
        ),
        why: (
          <Translated vi="Relaxed là lựa chọn nhanh hơn. Vấn đề là tính đúng đắn.">
            Relaxed is the faster choice. The problem is correctness.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không gì cả: khối j vẫn quay chờ đến khi thấy cờ, và khối i chỉ đặt cờ sau khi mọi luồng của nó đã đọc khóa của mình.">
            Nothing: block j still spins until it sees the flag, and block i sets the flag only after all its threads have read their keys.
          </Translated>
        ),
        why: (
          <Translated vi="Việc chờ sắp thứ tự cho cờ. Không có release và acquire, nó không sắp thứ tự cho các truy cập bộ nhớ khác quanh nó.">
            Waiting orders the flag. Without release and acquire it does not order the other memory accesses around it.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Thấy một cờ relaxed không sắp thứ tự cho gì khác: khối j có thể ghi đè một khóa mà khối i chưa đọc, một race condition (tranh chấp dữ liệu).">
            Seeing a relaxed flag orders nothing else: block j may overwrite a key block i has not read yet, a data race.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Release và acquire là thứ biến "tôi đã thấy cờ của bạn" thành "mọi việc bạn làm trước khi đặt cờ đã xong". CUB chuyển
            look-back của nó sang release và acquire đúng khi một phép select có thể làm tại chỗ. Chính phép scan một lượt của
            sách (Hình 11.17) đã dùng chúng.
          </p>
        }
      >
        <p>
          Release and acquire are what turn "I saw your flag" into "everything you did before setting it is done". CUB
          switches its look-back to release and acquire exactly when a selection may be in place. The book's own
          single-pass scan (Fig. 11.17) already uses them.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch12-facts.md A9; ch. 11 Fig. 11.17, pp. 278–279',
    vi: { source: 'notes/today/ch12-facts.md, mục A9; ch. 11 Hình 11.17, pp. 278–279' },
  },
  'ch12/12.7/q-blockidx': {
    kind: 'choice',
    id: 'ch12/12.7/q-blockidx',
    prompt: (
      <Translated vi={<>Vì sao một khối chờ khối đứng trước phải lấy số thứ tự của nó từ một bộ đếm atomic thay vì từ <code>blockIdx.x</code>?</>}>
        <>Why must a block that waits on its predecessor take its number from an atomic counter rather than from <code>blockIdx.x</code>?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Đọc một bộ đếm atomic rẻ hơn đọc blockIdx.x trong một lưới lớn hơn số khối thiết bị chứa được cùng lúc.">
            Reading an atomic counter is cheaper than reading blockIdx.x in a grid larger than the device holds at once.
          </Translated>
        ),
        why: (
          <Translated vi="Đọc blockIdx không tốn gì. Bộ đếm tốn một phép atomic. Mấu chốt là thứ tự.">
            blockIdx costs nothing to read. The counter costs an atomic. The point is the order.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="blockIdx.x không duy nhất trong một lưới lớn: đợt khối thứ hai dùng lại chỉ số của các khối đã xong.">
            blockIdx.x is not unique in a large grid: a second wave of blocks reuses the indices of finished ones.
          </Translated>
        ),
        why: (
          <Translated vi="Nó là duy nhất. Vấn đề là khi nào mỗi khối bắt đầu, không phải nó là khối nào.">
            It is unique. The problem is when each block starts, not which block it is.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các phép atomic phạm vi device chỉ hoạt động với một chỉ số động.">
            Device-scope atomics only work with a dynamic index.
          </Translated>
        ),
        why: (
          <Translated vi="Phạm vi và chỉ số khối không liên quan gì đến nhau.">
            Scopes and the block index are unrelated.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Với blockIdx, khối b có thể quay chờ một khối b − 1 chưa được lập lịch, mà CUDA chỉ hứa tiến triển cho các khối đã bắt đầu.">
            With blockIdx, block b may spin on a block b − 1 that has not been scheduled, and CUDA promises progress only to started blocks.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách giới thiệu bộ đếm ở chương 11 để tránh deadlock. Mô hình thực thi của NVIDIA giờ nêu rõ quy tắc bên dưới: một khi
            một luồng đã tiến triển, các luồng trong khối của nó (cluster của nó) sẽ tiếp tục tiến triển; một khối chưa bắt đầu
            thì không được hứa gì. Một khối chỉ lấy số từ bộ đếm khi nó đã chạy, nên con số mà nó chờ luôn thuộc về một khối đã
            bắt đầu.
          </p>
        }
      >
        <p>
          The book introduces the counter in chapter 11 to prevent deadlock. NVIDIA's execution model now states the
          rule underneath it: once a thread has made progress, the threads of its block (its cluster) will keep making
          progress; a block that has not started is promised nothing. A block takes its number from the counter only once
          it is running, so the number it waits on always belongs to a block that has already started.
        </p>
      </Translated>
    ),
    source: 'ch. 11 p. 280; notes/today/ch12-facts.md A10 (CUDA programming guide section 5.9)',
    vi: { source: 'ch. 11 p. 280; notes/today/ch12-facts.md, mục A10 (hướng dẫn lập trình CUDA, mục 5.9)' },
  },
  'ch12/12.7/q-metal': {
    kind: 'choice',
    id: 'ch12/12.7/q-metal',
    prompt: (
      <Translated vi={<>Thiết kế bộ lọc ổn định tại chỗ nào chỉ dựa trên hành vi mà Apple có ghi nhận trong tài liệu?</>}>
        <>Which in-place stable filter design rests only on behaviour Apple documents?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một dispatch duy nhất với look-back tách rời giữa các threadgroup, đánh số từ một bộ đếm atomic như trên CUDA.">
            A single dispatch with a decoupled look-back between threadgroups, numbered from an atomic counter as on CUDA.
          </Translated>
        ),
        why: (
          <Translated vi="Apple không ghi nhận đảm bảo tiến triển nào giữa các threadgroup, và một kỹ sư Apple đã gọi kiểu chờ nối chuỗi là hành vi không xác định.">
            Apple documents no forward progress between threadgroups, and an Apple engineer called the chained wait undefined behaviour.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một threadgroup đi qua phạm vi của nó theo thứ tự, chỉ dùng rào chắn threadgroup; một threadgroup cho mỗi hàng của batch.">
            One threadgroup walks its range in order, with threadgroup barriers only; one per row of a batch.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nhiều dispatch (đếm, scan các số đếm, rải), vì mỗi dispatch kết thúc trước khi dispatch sau bắt đầu.">
            Several dispatches (count, scan the counts, scatter), since each dispatch ends before the next begins.
          </Translated>
        ),
        why: (
          <Translated vi="Khi không tại chỗ, đó là thiết kế an toàn. Khi tại chỗ, dispatch rải vẫn để một threadgroup sau ghi đè các khóa mà threadgroup trước chưa đọc.">
            Out of place that is the safe design. In place, the scatter dispatch still lets a later threadgroup overwrite keys an earlier one has not read yet.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Bên trong một threadgroup, rào chắn sắp thứ tự mọi thứ: đọc một đoạn, rào chắn, scan nó, ghi nó tại offset đang chạy,
            rào chắn, đoạn kế tiếp. Các khóa chỉ chuyển về ô thấp hơn, nên các lần ghi của một đoạn rơi vào các ô đã được đọc.
            Giữa các threadgroup, Metal không ghi nhận gì có thể sắp thứ tự cho chúng.
          </p>
        }
      >
        <p>
          Inside one threadgroup, barriers order everything: read a chunk, barrier, scan it, write it at the running
          offset, barrier, next chunk. Keys only move to lower slots, so a chunk's writes land on slots already read.
          Across threadgroups Metal documents nothing that would order them.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch12-facts.md C4; ch11-facts C5; the chunk argument is ours',
    vi: { source: 'notes/today/ch12-facts.md, mục C4; ch11-facts C5; lập luận theo đoạn là của khóa học này' },
  },
  'ch12/12.7/q-ex2': {
    kind: 'choice',
    id: 'ch12/12.7/q-ex2',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 2 dựng bộ lọc ổn định thành một kernel trên nền một phép scan một lượt, và hỏi nó được lợi gì so với ba kernel
            của bài tập 1. Điều nào sau đây <b>không</b> phải là ưu điểm của kernel duy nhất?
          </>
        }
      >
        <>
          Exercise 2 builds the stable filter as one kernel on top of a single-pass scan, and asks what it gains over
          exercise 1's three kernels. Which of these is <b>not</b> an advantage of the single kernel?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi khóa được đọc một lần và mỗi khóa được giữ được ghi một lần; các cờ và offset không bao giờ phải đi một vòng qua bộ nhớ toàn cục.">
            Each key is read once and each kept key written once; the flags and offsets never make a round trip through global memory.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là một ưu điểm, và là ưu điểm lớn nhất với một kernel bị giới hạn bởi bộ nhớ.">
            That is an advantage, and the largest one for a memory-bound kernel.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không có lần khởi chạy kernel thừa nào.">
            There are no extra kernel launches.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là một ưu điểm.">
            That is an advantage.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó chạy trên mọi backend mà không cần thêm giả định nào.">
            It runs on every backend without further assumptions.
          </Translated>
        ),
        correct: true,
        why: (
          <Translated vi="Đúng: nó cần các khối chờ khối trước được hứa tiến triển, điều mà chỉ CUDA ghi nhận.">
            Right: it needs blocks that wait on earlier blocks to be promised progress, which only CUDA documents.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Thứ tự giữa các khối của nó cho phép dùng tại chỗ.">
            Its order between blocks makes it usable in place.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là một ưu điểm: toàn bộ lập luận của mục 12.7.">
            That is an advantage: section 12.7's whole argument.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Phiên bản ba kernel không cần sắp thứ tự giữa các khối chút nào: mỗi kernel kết thúc trước khi kernel sau bắt đầu.
            Đó là lý do nó là mức cơ sở chạy được mọi nơi, và vì sao phiên bản một lượt là phần mở rộng tùy chọn của lab 12.5.
          </p>
        }
      >
        <p>
          The three-kernel version needs no ordering between blocks at all: each kernel finishes before the next one
          starts. That is why it is the portable baseline, and why the single-pass version is lab 12.5's optional extra.
        </p>
      </Translated>
    ),
    source: 'ch. 12 exercise 2, p. 302 (the list of advantages is our derivation, notes); ch12-facts A10, B6, C4',
    vi: { source: 'ch. 12 bài tập 2, p. 302 (danh sách ưu điểm do khóa học này suy ra, trong notes); ch12-facts A10, B6, C4' },
  },
};

// ---------------------------------------------------------------- the overlap explorer

const nodeName = (n: PathNode) => `${n.op}(${n.block})`;

function defaultKeep(n: number) {
  return n === 16 ? [...FIG_KEEP] : [...FIG_KEEP, ...FIG_KEEP];
}

function ChainColumns({ blocks, chain, path }: { blocks: number; chain: Direction; path: PathNode[] | null }) {
  const CW = 132;
  const width = 24 + blocks * CW;
  const on = (op: PathNode['op'], b: number) => !!path?.some((n) => n.op === op && n.block === b);
  const edgeOn = (a: PathNode, b: PathNode) => {
    if (!path) return false;
    for (let i = 0; i + 1 < path.length; i++)
      if (path[i].op === a.op && path[i].block === a.block && path[i + 1].op === b.op && path[i + 1].block === b.block) return true;
    return false;
  };
  const box = (op: PathNode['op'], b: number, y: number, label: string) => {
    const lit = on(op, b);
    return (
      <g key={op + b} data-node={`${op}${b}`} data-lit={lit ? '1' : '0'}>
        <rect x={16 + b * CW} y={y} width={104} height={26} rx="3" fill={lit ? 'var(--ink)' : 'var(--cell)'} stroke="var(--ink-2)" />
        <text x={16 + b * CW + 52} y={y + 17} fontSize="11.5" textAnchor="middle" style={{ fill: lit ? 'var(--page)' : 'var(--ink)' }}>{label}</text>
      </g>
    );
  };
  const vline = (b: number, y1: number, y2: number, lit: boolean, key: string) => (
    <line key={key} x1={16 + b * CW + 52} x2={16 + b * CW + 52} y1={y1} y2={y2} stroke={lit ? 'var(--ink)' : 'var(--rule)'} strokeWidth={lit ? 2.4 : 1.4} markerEnd="url(#ip-head)" />
  );
  return (
    <svg viewBox={`0 0 ${width} 168`} style={{ width: '100%', maxWidth: width, minWidth: Math.min(width, 300), display: 'block' }} role="img" aria-label={tr('Dependence diagram, one column per block', 'Sơ đồ phụ thuộc, mỗi khối một cột')} data-part="chain">
      <defs>
        <marker id="ip-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute */}
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink-2)' }} />
        </marker>
      </defs>
      {Array.from({ length: blocks }, (_, b) => (
        <g key={b}>
          <text x={16 + b * CW + 52} y={14} fontSize="11.5" fontWeight="600" textAnchor="middle"><Translated vi={<>khối {b}</>}>block {b}</Translated></text>
          {box('read', b, 22, tr(`read input ${b}`, `đọc đầu vào ${b}`))}
          {box('scan', b, 72, tr(`scan ${b}`, `scan ${b}`))}
          {box('write', b, 122, tr(`write output ${b}`, `ghi đầu ra ${b}`))}
          {vline(b, 48, 70, edgeOn({ op: 'read', block: b }, { op: 'scan', block: b }), 'rs' + b)}
          {vline(b, 98, 120, edgeOn({ op: 'scan', block: b }, { op: 'write', block: b }), 'sw' + b)}
          {path && path.length === 3 && path[1].op === 'barrier' && path[0].block === b ? (
            <text x={16 + b * CW + 112} y={90} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}><Translated vi="rào chắn">barrier</Translated></text>
          ) : null}
        </g>
      ))}
      {Array.from({ length: blocks - 1 }, (_, b) => {
        const from = chain === 'remove' ? b : b + 1;
        const to = chain === 'remove' ? b + 1 : b;
        const lit = edgeOn({ op: 'scan', block: from }, { op: 'scan', block: to });
        const x1 = chain === 'remove' ? 16 + b * CW + 104 : 16 + (b + 1) * CW;
        const x2 = chain === 'remove' ? 16 + (b + 1) * CW - 2 : 16 + b * CW + 106;
        return <line key={'h' + b} data-edge={`scan${from}-scan${to}`} x1={x1} x2={x2} y1={85} y2={85} stroke={lit ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={lit ? 2.4 : 1.4} markerEnd="url(#ip-head)" />;
      })}
    </svg>
  );
}

function OverlapExplorer() {
  const [n, setN] = useState<16 | 32>(16);
  const [blocks, setBlocks] = useState(2);
  const [keep, setKeep] = useState<boolean[]>(() => defaultKeep(16));
  const [dir, setDir] = useState<Direction>('remove');
  const [chain, setChain] = useState<Direction>('remove');
  const [sel, setSel] = useState<number | null>(null);
  const [pair, setPair] = useState<[number, number]>([0, 1]);

  const perBlock = Math.ceil(n / blocks);
  const ov = useMemo(() => overlap(keep, perBlock, dir), [keep, perBlock, dir]);
  const nb = ov.blocks;
  const selected = sel == null ? undefined : ov.writes.find((w) => w.key === sel);
  const [ri, wj] = [Math.min(pair[0], nb - 1), Math.min(pair[1], nb - 1)];
  const path = dependencePath(ri, wj, chain);

  const reset = (nn: 16 | 32, bb: number) => {
    setN(nn);
    setBlocks(bb);
    setKeep(defaultKeep(nn));
    setSel(null);
    setPair([0, 1]);
  };
  const choose = (w: Write) => {
    setSel(w.key);
    if (w.reader != null && w.kind !== 'own') setPair([w.reader, w.writer]);
  };

  // Geometry of the array drawing.
  const CW = 24;
  const X0 = 62;
  const YB = 34;
  const YA = 126;
  const width = X0 + n * CW + 8;
  const cx = (s: number) => X0 + s * CW + (CW - 2) / 2;
  const cell = (y: number, s: number, v: number | null, opts: { kept?: boolean; victim?: boolean; hot?: boolean; onClick?: () => void }) => (
    <g key={`${y}-${s}`} onClick={opts.onClick} style={{ cursor: opts.onClick ? 'pointer' : undefined }} data-cell={`${y === YB ? 'before' : 'after'}-${s}`}>
      <rect x={X0 + s * CW} y={y} width={CW - 2} height={24} rx="2"
        fill={opts.hot ? 'var(--global)' : opts.kept ? 'var(--global-wash)' : 'var(--cell)'}
        stroke={opts.victim ? 'var(--fail)' : opts.kept || opts.hot ? 'var(--global)' : 'var(--rule)'} strokeWidth={opts.victim ? 2.4 : 1} />
      <text x={cx(s)} y={y + 16} fontSize="9.5" textAnchor="middle" style={{ fill: opts.hot ? 'var(--page)' : v == null ? 'var(--ink-3)' : 'var(--ink)' }}>
        {v == null ? '·' : `k${v}`}
      </text>
    </g>
  );

  const pathText = path
    ? path.map(nodeName).join(' → ')
    : tr(
        `no path: the chain runs ${chain === 'remove' ? 'from block 0 upwards' : 'from the last block downwards'}, so nothing orders block ${ri}'s read before block ${wj}'s write`,
        `không có đường đi: chuỗi chạy ${chain === 'remove' ? 'từ khối 0 đi lên' : 'từ khối cuối đi xuống'}, nên không gì xếp lần đọc của khối ${ri} trước lần ghi của khối ${wj}`,
      );

  return (
    <Bench
      title={tr('In-place overlap explorer', 'Trình khám phá chồng lấn tại chỗ')}
      task={tr(
        'Click a red arrow: whose key does it overwrite, and what orders the two? Then switch to inserting.',
        'Bấm vào một mũi tên đỏ: nó ghi đè khóa của ai, và điều gì sắp thứ tự cho hai việc đó? Sau đó chuyển sang chèn.',
      )}
      foot={
        <Translated
          vi={
            <>
              Hàng trên: mảng trước; hàng dưới: cùng vùng nhớ đó sau. Các khóa tô đậm được giữ. Ở chế độ loại bỏ, bấm vào một
              khóa ở hàng trên để giữ hoặc bỏ nó. Bấm vào một mũi tên, hoặc một khóa trong bảng, để chọn lần ghi đó.
            </>
          }
        >
          <>
            Top row: the array before; bottom row: the same memory after. Shaded keys are kept. In remove mode, click a key
            in the top row to keep or drop it. Click an arrow, or a key in the table, to select that write.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Số khóa">Keys</Translated>
          <select value={n} onChange={(e) => reset(+e.target.value as 16 | 32, blocks)}>
            <option value={16}>{tr("16 (the book's figures)", '16 (các hình của sách)')}</option>
            <option value={32}>32</option>
          </select>
        </label>
        <label className="field">
          <Translated vi="Số khối">Blocks</Translated>
          <select value={blocks} onChange={(e) => { setBlocks(+e.target.value); setSel(null); setPair([0, 1]); }}>
            {[2, 3, 4].map((b) => <option key={b} value={b}>{b} {tr('blocks of', 'khối, mỗi khối')} {Math.ceil(n / b)} {tr('keys', 'khóa')}</option>)}
          </select>
        </label>
        <span className="seg" role="group" aria-label={tr('Movement', 'Chiều di chuyển')}>
          <button aria-pressed={dir === 'remove'} onClick={() => { setDir('remove'); setSel(null); }}><Translated vi="loại bỏ (lọc)">remove (filter)</Translated></button>
          <button aria-pressed={dir === 'insert'} onClick={() => { setDir('insert'); setSel(null); }}><Translated vi="chèn (chiều ngược lại, §12.8)">insert (the reverse, §12.8)</Translated></button>
        </span>
        <button className="quiet" onClick={() => reset(n, blocks)}><Translated vi="Mẫu giữ của sách">Book's keep pattern</Translated></button>
      </div>
      <p className="muted" style={{ margin: '0 0 0.6rem' }}>
        <Translated
          vi={
            <>
              {nb} khối, mỗi khối {perBlock} khóa{perBlock % 4 === 0 ? `; với 4 luồng mỗi khối như sách, hệ số gộp việc là ${perBlock / 4}` : ''}.
            </>
          }
        >
          {nb} blocks of {perBlock} keys{perBlock % 4 === 0 ? `; with the book's 4 threads per block, a coarsening factor of ${perBlock / 4}` : ''}.
        </Translated>
        {dir === 'insert' ? tr(' Insert mode starts from the compacted list and puts every kept key back where it came from.', ' Chế độ chèn bắt đầu từ danh sách đã nén gọn và đưa mọi khóa được giữ về đúng chỗ cũ của nó.') : ''}
      </p>

      {/* Never drawn below its design size: on a narrow screen the drawing scrolls in its own frame. */}
      <div style={{ overflowX: 'auto' }} data-part="overlap-frame">
      <svg viewBox={`0 0 ${width} 164`} role="img" aria-label={tr('Array before and after the in-place move', 'Mảng trước và sau lần di chuyển tại chỗ')} data-part="overlap"
        style={{ width: '100%', maxWidth: width, minWidth: width, display: 'block' }}>
        <defs>
          {(['ink', 'fail', 'ink-3'] as const).map((c) => (
            <marker key={c} id={`ov-${c}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">{/* i18n-ignore: SVG attribute */}
              <path d="M0,0 L10,5 L0,10 z" style={{ fill: `var(--${c})` }} />
            </marker>
          ))}
        </defs>
        <text x={4} y={YB + 16} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="trước">before</Translated></text>
        <text x={4} y={YA + 16} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}><Translated vi="sau">after</Translated></text>
        {Array.from({ length: nb }, (_, b) => {
          const a = b * perBlock;
          const z = Math.min(n, (b + 1) * perBlock) - 1;
          const lit = selected && (selected.reader === b || selected.writer === b);
          return (
            <g key={b}>
              <line x1={X0 + a * CW + 1} x2={X0 + z * CW + CW - 3} y1={YB - 6} y2={YB - 6} stroke={lit ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={lit ? 2 : 1} />
              <text x={(X0 + a * CW + X0 + z * CW + CW) / 2} y={YB - 10} fontSize="10.5" textAnchor="middle" fontWeight={lit ? 700 : 400}><Translated vi={<>khối {b}</>}>block {b}</Translated></text>
            </g>
          );
        })}
        {ov.before.map((v, s) =>
          cell(YB, s, v, {
            kept: dir === 'remove' ? keep[s] : v != null,
            victim: !!selected && selected.victim != null && selected.to === s,
            hot: !!selected && selected.from === s,
            onClick: dir === 'remove' ? () => { setKeep(keep.map((k, i) => (i === s ? !k : k))); setSel(null); } : undefined,
          }),
        )}
        {ov.after.map((v, s) => cell(YA, s, v, { kept: v != null, hot: !!selected && selected.to === s }))}
        {ov.writes.map((w) => {
          const cross = w.kind === 'earlier' || w.kind === 'later';
          const isSel = selected?.key === w.key;
          const color = cross ? 'fail' : w.kind === 'same' ? 'ink' : 'ink-3';
          return (
            <g key={w.key} onClick={() => choose(w)} style={{ cursor: 'pointer' }} data-write={w.key} data-kind={w.kind}>
              <line x1={cx(w.from)} y1={YB + 25} x2={cx(w.to)} y2={YA - 3} stroke="transparent" strokeWidth={9} />
              <line x1={cx(w.from)} y1={YB + 25} x2={cx(w.to)} y2={YA - 3}
                stroke={`var(--${color})`} strokeWidth={isSel ? 3 : cross ? 1.8 : 1.2}
                strokeDasharray={w.kind === 'own' ? '3 3' : w.kind === 'fresh' ? '1 3' : undefined}
                markerEnd={`url(#ov-${color})`} />
            </g>
          );
        })}
      </svg>
      </div>

      <div className={'narrate' + (selected && (selected.kind === 'earlier' || selected.kind === 'later') ? ' bad' : '')} role="status" style={{ margin: '0.8rem 0' }} data-part="selected">
        {selected ? (
          <Translated
            vi={
              <>
                <b>k{selected.key}</b> chuyển từ ô {selected.from} sang ô {selected.to}
                {selected.victim == null
                  ? ', nơi không có khóa nào: không có gì phải bảo vệ.'
                  : selected.kind === 'own'
                    ? ': chính ô của nó.'
                    : <>, đè lên <b>k{selected.victim}</b>, khóa mà khối {selected.reader} đọc. </>}
                {selected.kind === 'same' ? 'Cùng khối: một rào chắn giữa "mọi luồng đọc" và "mọi luồng ghi" sắp thứ tự cho nó.' : null}
                {selected.kind === 'earlier' ? `Một khối trước: khối ${selected.writer} chỉ được ghi sau khi khối ${selected.reader} đã đọc. Chuỗi scan phải sắp thứ tự cho việc này.` : null}
                {selected.kind === 'later' ? `Một khối sau: khối ${selected.writer} chỉ được ghi sau khi khối ${selected.reader} đã đọc, nên chuỗi phải chạy từ khối cuối về khối đầu.` : null}
              </>
            }
          >
            <>
              <b>k{selected.key}</b> moves from slot {selected.from} to slot {selected.to}
              {selected.victim == null
                ? ', where no key lived: nothing to protect.'
                : selected.kind === 'own'
                  ? ': its own slot.'
                  : <>, over <b>k{selected.victim}</b>, which block {selected.reader} reads. </>}
              {selected.kind === 'same' ? 'Same block: a barrier between "all threads read" and "all threads write" orders it.' : null}
              {selected.kind === 'earlier' ? `An earlier block: block ${selected.writer} may write only after block ${selected.reader} has read. The chain of scans has to order it.` : null}
              {selected.kind === 'later' ? `A later block: block ${selected.writer} may write only after block ${selected.reader} has read, so the chain has to run from the last block towards the first.` : null}
            </>
          </Translated>
        ) : (
          <Translated vi={<>Chưa chọn gì. Các mũi tên đỏ đi vào phạm vi của một khối khác.</>}>
            <>Nothing selected. Red arrows cross into another block's range.</>
          </Translated>
        )}
      </div>

      <div className="row" style={{ marginBottom: '0.6rem' }} data-part="counts">
        {(['own', 'same', 'earlier', 'later', 'fresh'] as const)
          .filter((k) => ov.counts[k] > 0 || k === 'same' || k === (dir === 'remove' ? 'earlier' : 'later'))
          .map((k) => (
            <span key={k} className={'chip' + (k === 'earlier' || k === 'later' ? (ov.counts[k] ? ' fail' : '') : '')} data-count={k} data-value={ov.counts[k]}>
              {KIND_LABEL[k]}: {ov.counts[k]}
            </span>
          ))}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)' }}>
        <div style={{ overflowX: 'auto' }}>
          <table className="threads" aria-label={tr('Every write of the move', 'Mọi lần ghi của lần di chuyển')} data-part="writes">
            <thead>
              <tr><th><Translated vi="khóa">key</Translated></th><th><Translated vi="ô">slot</Translated></th><th><Translated vi="ghi đè">overwrites</Translated></th><th><Translated vi="được đọc bởi">read by</Translated></th><th><Translated vi="cần">needs</Translated></th></tr>
            </thead>
            <tbody>
              {ov.writes.map((w) => (
                <tr key={w.key} className={selected?.key === w.key ? 'cur' : ''} data-write-row={w.key}>
                  <td><button className="quiet" onClick={() => choose(w)} style={{ padding: '0.05rem 0.4rem' }}>k{w.key}</button></td>
                  <td>{w.from} → {w.to}</td>
                  <td>{w.victim == null ? '—' : w.kind === 'own' ? tr('itself', 'chính nó') : `k${w.victim}`}</td>
                  <td>{w.reader == null ? '—' : tr(`block ${w.reader}`, `khối ${w.reader}`)}</td>
                  <td className={w.kind === 'earlier' || w.kind === 'later' ? 't-fail' : ''}>{KIND_LABEL[w.kind]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <div className="row" style={{ marginBottom: '0.5rem' }}>
            <label className="field">
              <Translated vi="Khối đọc">Block that reads</Translated>
              <select value={ri} onChange={(e) => setPair([+e.target.value, wj])} aria-label={tr('reader block', 'khối đọc')}>
                {Array.from({ length: nb }, (_, b) => <option key={b} value={b}>{b}</option>)}
              </select>
            </label>
            <label className="field">
              <Translated vi="Khối ghi">Block that writes</Translated>
              <select value={wj} onChange={(e) => setPair([ri, +e.target.value])} aria-label={tr('writer block', 'khối ghi')}>
                {Array.from({ length: nb }, (_, b) => <option key={b} value={b}>{b}</option>)}
              </select>
            </label>
            <span className="seg" role="group" aria-label={tr('Chain direction', 'Chiều của chuỗi')}>
              <button aria-pressed={chain === 'remove'} onClick={() => setChain('remove')}><Translated vi="chuỗi: đầu → cuối">chain: first → last</Translated></button>
              <button aria-pressed={chain === 'insert'} onClick={() => setChain('insert')}><Translated vi="chuỗi: cuối → đầu">chain: last → first</Translated></button>
            </span>
          </div>
          <ChainColumns blocks={nb} chain={chain} path={path} />
          <p className={path ? '' : 't-fail'} style={{ margin: '0.5rem 0 0', fontFamily: 'var(--sans)', fontSize: '0.85rem' }} data-part="path">
            {path ? <Translated vi={<>Có thứ tự: {pathText}</>}><>Ordered: {pathText}</></Translated> : <Translated vi={<>Không có thứ tự: {pathText}.</>}><>Not ordered: {pathText}.</></Translated>}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the ordering puzzle

function OrderingPuzzle() {
  const [cfg, setCfg] = useState<PuzzleConfig>(BOOK_PUZZLE);
  const [reader, setReader] = useState<Who>('other');
  const [writer, setWriter] = useState<Who>('other');
  const v = orderingVerdict(cfg, reader, writer);
  const table = orderingTable(cfg);
  const set = (p: Partial<PuzzleConfig>) => setCfg({ ...cfg, ...p });
  const whoLabel = (w: Who, block: 'i' | 'j') => (w === 'leader' ? tr(`block ${block}'s leader`, `luồng dẫn đầu của khối ${block}`) : tr(`another thread of block ${block}`, `một luồng khác của khối ${block}`));
  return (
    <Bench
      title={tr('Ordering puzzle', 'Câu đố về thứ tự')}
      task={tr(
        "Can block j's write to slot s land before block i's read of slot s? Break the book's kernel one switch at a time.",
        'Lần ghi của khối j vào ô s có thể xảy ra trước lần đọc ô s của khối i không? Phá kernel của sách từng công tắc một.',
      )}
      foot={tr(
        "Block i comes before block j. The leader is the one thread per block that touches the flag (the block's last thread in the book's Fig. 11.17).",
        'Khối i đứng trước khối j. Luồng dẫn đầu là luồng duy nhất mỗi khối chạm vào cờ (luồng cuối của khối trong Hình 11.17 của sách).',
      )}
    >
      <div className="cols" style={{ marginBottom: '0.8rem' }}>
        <div className="panel" data-part="block-i">
          <div className="mx-label"><Translated vi="Khối i (trước)">Block i (earlier)</Translated></div>
          <ol style={{ margin: 0, paddingLeft: '1.2rem', lineHeight: 2 }}>
            <li><Translated vi="mọi luồng đọc khóa của mình">every thread reads its keys</Translated></li>
            <li>
              <label className="check"><input type="checkbox" checked={cfg.barrierI} onChange={(e) => set({ barrierI: e.target.checked })} /> <Translated vi="rào chắn">barrier</Translated></label>
            </li>
            <li>
              <Translated vi="luồng dẫn đầu store cờ:">the leader stores the flag:</Translated>{' '}
              <span className="seg" role="group" aria-label={tr('Store order', 'Thứ tự store')}>
                <button aria-pressed={cfg.publish === 'relaxed'} onClick={() => set({ publish: 'relaxed' })}>relaxed{/* i18n-ignore: C++ memory order name */}</button>
                <button aria-pressed={cfg.publish === 'release'} onClick={() => set({ publish: 'release' })}>release{/* i18n-ignore: C++ memory order name */}</button>
              </span>
            </li>
          </ol>
        </div>
        <div className="panel" data-part="block-j">
          <div className="mx-label"><Translated vi="Khối j (sau)">Block j (later)</Translated></div>
          <ol style={{ margin: 0, paddingLeft: '1.2rem', lineHeight: 2 }}>
            <li>
              <Translated vi="luồng dẫn đầu load cờ cho đến khi nó được đặt:">the leader loads the flag until it is set:</Translated>{' '}
              <span className="seg" role="group" aria-label={tr('Load order', 'Thứ tự load')}>
                <button aria-pressed={cfg.observe === 'relaxed'} onClick={() => set({ observe: 'relaxed' })}>relaxed{/* i18n-ignore: C++ memory order name */}</button>
                <button aria-pressed={cfg.observe === 'acquire'} onClick={() => set({ observe: 'acquire' })}>acquire{/* i18n-ignore: C++ memory order name */}</button>
              </span>
            </li>
            <li>
              <label className="check"><input type="checkbox" checked={cfg.barrierJ} onChange={(e) => set({ barrierJ: e.target.checked })} /> <Translated vi="rào chắn">barrier</Translated></label>
            </li>
            <li><Translated vi="mọi luồng ghi các khóa được giữ của mình">every thread writes its kept keys</Translated></li>
          </ol>
        </div>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span><Translated vi="Ô s được đọc bởi">Slot s is read by</Translated></span>
        <span className="seg" role="group" aria-label={tr('Reader', 'Bên đọc')}>
          <button aria-pressed={reader === 'leader'} onClick={() => setReader('leader')}><Translated vi="luồng dẫn đầu của khối i">block i's leader</Translated></button>
          <button aria-pressed={reader === 'other'} onClick={() => setReader('other')}><Translated vi="một luồng khác">another thread</Translated></button>
        </span>
        <span><Translated vi="và được ghi bởi">and written by</Translated></span>
        <span className="seg" role="group" aria-label={tr('Writer', 'Bên ghi')}>
          <button aria-pressed={writer === 'leader'} onClick={() => setWriter('leader')}><Translated vi="luồng dẫn đầu của khối j">block j's leader</Translated></button>
          <button aria-pressed={writer === 'other'} onClick={() => setWriter('other')}><Translated vi="một luồng khác">another thread</Translated></button>
        </span>
        <button className="quiet" onClick={() => setCfg(BOOK_PUZZLE)}><Translated vi="Kernel của sách">The book's kernel</Translated></button>
      </div>
      <div className={'narrate' + (v.ordered ? '' : ' bad')} role="status" data-part="verdict" data-ordered={v.ordered ? '1' : '0'}>
        {v.ordered ? (
          <b className="t-ok">
            <Translated vi={<>Có thứ tự: lần đọc ô s bởi {whoLabel(reader, 'i')} xảy ra trước lần ghi bởi {whoLabel(writer, 'j')}.</>}>
              Ordered: the read of slot s by {whoLabel(reader, 'i')} happens before the write by {whoLabel(writer, 'j')}.
            </Translated>
          </b>
        ) : (
          <b className="t-fail">
            <Translated vi="Không có thứ tự: một race condition (tranh chấp dữ liệu). Lần ghi có thể xảy ra trước lần đọc.">
              Not ordered: a data race. The write may land before the read.
            </Translated>
          </b>
        )}
        <ol style={{ margin: '0.4rem 0 0', paddingLeft: '1.2rem' }} data-part="links">
          {v.links.map((l, k) => (
            <li key={k} className={l.ok ? '' : 't-fail'} data-ok={l.ok ? '1' : '0'}>
              {l.from} → {l.to}: <i>{l.why}</i>
            </li>
          ))}
        </ol>
      </div>
      <table className="threads" style={{ marginTop: '0.8rem', maxWidth: '34rem' }} aria-label={tr('Every reader and writer', 'Mọi bên đọc và bên ghi')} data-part="table">
        <thead>
          <tr><th></th><th><Translated vi="luồng dẫn đầu của j ghi">j's leader writes</Translated></th><th><Translated vi="một luồng khác ghi">another thread writes</Translated></th></tr>
        </thead>
        <tbody>
          {(['leader', 'other'] as Who[]).map((r) => (
            <tr key={r}>
              <td>{r === 'leader' ? tr("i's leader reads", 'luồng dẫn đầu của i đọc') : tr('another thread reads', 'một luồng khác đọc')}</td>
              {(['leader', 'other'] as Who[]).map((w) => {
                const t = table.find((x) => x.reader === r && x.writer === w)!;
                return (
                  <td key={w} data-cell={`${r}-${w}`} data-ordered={t.verdict.ordered ? '1' : '0'}>
                    {t.verdict.ordered ? <span className="chip ok"><Translated vi="có thứ tự">ordered</Translated></span> : <span className="chip fail"><Translated vi="tranh chấp dữ liệu">data race</Translated></span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page

export function S127() {
  const q = questionsS127;
  const lab5 = useLabFile(LAB5);
  // The labs page's words (LabKit's solutionTabStatus), from this machine's runs. Lab 12.5 has no Metal files.
  const { reports } = useReports();
  const lab5Status = solutionTabStatus(reports, 'ch12/lab5_inplace', 'kernel.cu');
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi bộ lọc cho đến giờ đều ghi vào một mảng thứ hai. Việc đó cần chỗ cho cả một bản sao thứ hai của đầu vào, và đầu
            vào có thể lớn so với bộ nhớ bạn có: ví dụ của sách là nén gọn một heap, nơi một bản sao thứ hai nghĩa là một heap
            thứ hai. Một bộ lọc <b>tại chỗ</b> (in-place) ghi các phần tử còn lại vào chính mảng nó đang đọc.
          </p>
        }
      >
        <p>
          Every filter so far wrote into a second array. That needs room for a whole second copy of the input, and the
          input may be large compared with the memory you have: the book's example is compacting a heap, where a second
          copy would mean a second heap. An <b>in-place</b> filter writes the survivors into the array it is reading.
        </p>
      </Translated>
      <Source section="12.7" pages="300" />
      <Translated
        vi={
          <p>
            Một quy tắc làm nó đúng: không bao giờ ghi một khóa vào một ô trước khi khóa từng nằm ở đó đã được đọc. Dồn các khóa
            về phía đầu nghĩa là một khóa có thể rơi lên trên một khóa vẫn đang chờ được đọc, có khi là khóa chính nó cũng phải
            di chuyển.
          </p>
        }
      >
        <p>
          One rule makes it correct: never write a key into a slot before the key that used to live there has been read.
          Moving keys towards the front means a key can land on top of a key that is still waiting to be read, maybe one
          that has to move itself.
        </p>
      </Translated>

      <h2><Translated vi="Bên trong một khối">Inside one block</Translated></h2>
      <Translated
        vi={
          <p>
            Lấy ví dụ có gộp việc cho luồng của sách, Hình 12.10: mười sáu khóa, hai khối bốn luồng, tám khóa mỗi khối, với k1,
            k3, k4, k6, k8, k10, k11 và k14 được giữ. k3 là khóa được giữ thứ hai, nên nó rơi vào ô 1, nơi k1 từng nằm. Cả hai
            khóa đều thuộc khối 0, nên cách sửa là cách bạn đã dùng từ chương 4 (
            <a href={href('ch04', '4.3')}>mục 4.3</a>): một rào chắn giữa "mọi luồng của khối đã đọc khóa của mình" và "bất kỳ
            luồng nào ghi".
          </p>
        }
      >
        <p>
          Take the book's coarsened example, Fig. 12.10: sixteen keys, two blocks of four threads, eight keys per block,
          with k1, k3, k4, k6, k8, k10, k11 and k14 kept. k3 is the second kept key, so it lands in slot 1, where k1 lived.
          Both keys belong to block 0, so the fix is the one you have used since chapter 4 (
          <a href={href('ch04', '4.3')}>section 4.3</a>): a barrier between "all threads of the block have read their
          keys" and "any thread writes".
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              <b>Trang sách ghi k0.</b> Trang 300 để luồng 1 đọc xong k0 trước khi luồng 3 ghi k3. Theo chính logic của trang đó,
              khóa đó là k1. Và trong Hình 12.10 bốn luồng xử lý tám khóa, nên "luồng 1" và "luồng 3" không ánh xạ một-một lên các
              khóa.
            </p>
          }
        >
          <p>
            <b>The page says k0.</b> Page 300 has thread 1 finish reading k0 before thread 3 writes k3. By the page's own
            logic the key is k1. And in Fig. 12.10 four threads handle eight keys, so "thread 1" and "thread 3" do not
            map one to one onto keys.
          </p>
        </Translated>
      </Aside>
      <Ask q={q['ch12/12.7/q-erratum']} label={tr('Read the figure', 'Đọc hình')} />

      <h2><Translated vi="Giữa các khối">Across blocks</Translated></h2>
      <Translated
        vi={
          <p>
            k8 rơi vào ô 4, nơi k4 từng nằm, và k4 thuộc khối kia. Một rào chắn không giúp được ở đây: nó chỉ sắp thứ tự các
            luồng của một khối. Yêu cầu tổng quát là không khối nào ghi trước khi mọi khối đứng trước đã đọc. (Các khóa chỉ dồn
            về phía đầu, nên một khối không bao giờ ghi vào phạm vi của một khối sau.)
          </p>
        }
      >
        <p>
          k8 lands in slot 4, where k4 lived, and k4 belongs to the other block. A barrier cannot help here: it orders
          the threads of one block only. The general requirement is that no block writes before every earlier block has
          read. (Keys only move towards the front, so a block never writes into a later block's range.)
        </p>
      </Translated>
      <Source section="12.7" pages="300" figs={tr('Fig. 12.10', 'Hình 12.10')} />
      <Ask q={q['ch12/12.7/q-crossblock']} label={tr('Count them', 'Đếm chúng')} />
      <Translated
        vi={
          <p>
            Giờ đến nhận xét của sách. Phép scan một lượt của chương 11 vốn đã sắp thứ tự các khối: khối <i>j</i> không thể xong
            phép scan của nó cho đến khi khối <i>j</i> − 1 đã công bố, điều mà khối đó chỉ làm sau phép scan của chính nó, vốn
            đến sau các lần đọc của nó. Vậy với mọi cặp khối <i>i</i> &lt; <i>j</i> có một đường đi read(<i>i</i>) →
            scan(<i>i</i>) → … → scan(<i>j</i>) → write(<i>j</i>). Mọi lần đọc trước đều đến trước mọi lần ghi sau, và theo sách,
            kernel không tại chỗ chạy được tại chỗ mà không cần thay đổi.
          </p>
        }
      >
        <p>
          Now the book's observation. The single-pass scan of chapter 11 already orders the blocks: block <i>j</i> cannot
          finish its scan until block <i>j</i> − 1 has published, which it does only after its own scan, which comes after
          its reads. So for any blocks <i>i</i> &lt; <i>j</i> there is a path read(<i>i</i>) → scan(<i>i</i>) → … →
          scan(<i>j</i>) → write(<i>j</i>). Every earlier read comes before every later write, and, says the book, the
          out-of-place kernel works in place unchanged.
        </p>
      </Translated>
      <Source section="12.7" pages="300–301" figs={tr('Fig. 12.11', 'Hình 12.11')} />
      <Translated
        vi={
          <p>
            Hãy thử bên dưới. Tìm các mũi tên đỏ, chọn một mũi tên, và kiểm tra rằng chuỗi scan sắp thứ tự cho hai khối của nó.
            Sau đó đổi số khối và các khóa được giữ: có mũi tên đỏ nào chỉ vào một khối sau không? Cuối cùng chuyển sang chèn,
            chiều di chuyển ngược lại từ mục 12.8, và xem bạn cần chuỗi nào.
          </p>
        }
      >
        <p>
          Try it below. Find the red arrows, select one, and check that the chain of scans orders its two blocks. Then
          change the number of blocks and the kept keys: does a red arrow ever point into a later block? Finally switch
          to inserting, which is the reverse movement from section 12.8, and see which chain you need.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Không có thời gian: trình khám phá chỉ trả lời khối nào đọc ô mà mỗi lần ghi rơi vào, và một chuỗi scan chạy theo
              một chiều có nối hai khối đó không. Các luồng trong một khối không được vẽ; giả định có một rào chắn giữa các lần
              đọc và các lần ghi của một khối.
            </p>
          }
        >
          <p>
            No timing: the explorer only answers which block reads the slot each write lands on, and whether a chain of
            scans running one way connects the two. Threads within a block are not drawn; a barrier is assumed between
            a block's reads and its writes.
          </p>
        </Translated>
      </Simplification>
      <OverlapExplorer />

      <h2><Translated vi="Điều sách để ngầm hiểu">What the book leaves implicit</Translated></h2>
      <Translated
        vi={
          <p>
            "Phép scan sắp thứ tự các khối" đúng với <i>cờ</i> của phép scan. Nó có sắp thứ tự cả các lần đọc và ghi khóa thông
            thường quanh cờ hay không thì phụ thuộc vào cách cờ được ghi và đọc. Để các lần đọc của khối <i>i</i> xảy ra trước
            các lần ghi của khối <i>j</i>, ba điều phải đúng: mọi luồng của khối <i>i</i> đã đọc xong trước khi luồng dẫn đầu
            của nó store cờ (một rào chắn); lần store là một <b>release</b>; và luồng dẫn đầu của khối <i>j</i> load nó bằng một{' '}
            <b>acquire</b>, theo sau là một rào chắn trước khi khối <i>j</i> ghi.
          </p>
        }
      >
        <p>
          "The scan orders the blocks" is true of the scan's <i>flag</i>. Whether it also orders the plain reads and
          writes of keys around the flag depends on how the flag is written and read. For block <i>i</i>'s reads to
          happen before block <i>j</i>'s writes, three things must hold: every thread of block <i>i</i> has finished
          reading before its leader stores the flag (a barrier); the store is a <b>release</b>; and block <i>j</i>'s
          leader loads it with an <b>acquire</b>, followed by a barrier before block <i>j</i> writes.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md, A9)', '2026-11-20 (notes/today/ch12-facts.md, mục A9)')}>
        <Translated
          vi={
            <p>
              Kernel select của CUB chuyển look-back sang release và acquire đúng khi phép select có thể làm tại chỗ, và dùng thứ
              tự relaxed trong các trường hợp khác. Chú thích trong mã nguồn của nó nêu lý do bằng đúng các thuật ngữ trên: mọi
              phần tử của một ô (tile) phải được load trước khi báo hiệu cho một khối sau.
            </p>
          }
        >
          <p>
            CUB's select kernel switches its look-back to release and acquire exactly when the selection may be in place,
            and uses relaxed ordering otherwise. Its source comment gives the reason in the terms above: all of a tile's
            items must be loaded before a later block is signalled.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Chính phép scan một lượt của sách thỏa cả ba điều kiện: Hình 11.17 đặt cờ bằng một release và chờ bằng một acquire,
            và Hình 11.18 có một rào chắn sau các lần load và một rào chắn sau bước liên khối. Vậy khẳng định của sách đúng với
            kernel của sách. Nó không đúng với mọi phép scan một lượt.
          </p>
        }
      >
        <p>
          The book's own single-pass scan meets all three conditions: Fig. 11.17 sets the flag with a release and waits
          with an acquire, and Fig. 11.18 has a barrier after the loads and one after the inter-block step. So the book's
          claim holds for the book's kernel. It does not hold for every single-pass scan.
        </p>
      </Translated>
      <Source section="11.9" pages="278–281" figs={tr('Figs. 11.17, 11.18', 'Hình 11.17, 11.18')} />
      <Translated
        vi={
          <p>
            Câu đố đặt hai khối cạnh nhau với bốn công tắc đặt như trong kernel của sách. Hãy phá nó từng công tắc một, và với mỗi
            thiết lập, hỏi xem lần đọc của những luồng nào vẫn an toàn.
          </p>
        }
      >
        <p>
          The puzzle puts the two blocks side by side with the four switches set as in the book's kernel. Break it one
          switch at a time, and for each setting ask which threads' reads are still safe.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Các quy tắc happens-before của C++ áp dụng cho hai khối, không phải một mô hình phần cứng. Phạm vi bị bỏ qua (một
              cờ thật có phạm vi device). Một rào chắn xếp mọi thứ trước nó trong khối lên trước mọi thứ sau nó, đúng như sách
              dùng ở p. 300.
            </p>
          }
        >
          <p>
            The C++ happens-before rules applied to two blocks, not a model of hardware. Scopes are ignored (a real
            flag is device scope). A barrier orders everything before it in the block before everything after it, as
            the book uses it on p. 300.
          </p>
        </Translated>
      </Simplification>
      <OrderingPuzzle />
      <Translated
        vi={
          <p>
            Một hệ quả cho việc kiểm thử. <code>emu</code> không thể tìm ra lỗi này: nó chạy các khối lần lượt, nên không khối nào
            ghi trong khi khối khác còn đang đọc. <code>race</code> có thể báo nó, như một truy cập không có thứ tự vào{' '}
            <code>inclusive[]</code> hoặc vào các khóa, đó là thí nghiệm tùy chọn của lab 12.5. Nhưng một lần chạy sạch không
            chứng minh gì, và một phiên bản relaxed có thể vượt qua mọi lần chạy trên thiết bị thật mà vẫn sai. Tính đúng đắn ở
            đây đến từ lập luận, không đến từ một bài kiểm thử.
          </p>
        }
      >
        <p>
          One consequence for testing. <code>emu</code> cannot find this bug: it runs blocks one after another, so no
          block ever writes while another is still reading. <code>race</code> can report it, as an unordered access to{' '}
          <code>inclusive[]</code> or to the keys, which is lab 12.5's optional experiment. But a clean run proves
          nothing, and a relaxed version can pass every run on a real device and still be wrong. Correctness here comes
          from the argument, not from a test.
        </p>
      </Translated>
      <Ask q={q['ch12/12.7/q-relaxed']} />

      <h2><Translated vi="Một lần chờ có thể không bao giờ kết thúc">A wait that may never end</Translated></h2>
      <Translated
        vi={
          <p>
            Có một điều kiện ẩn thứ hai. Một khối quay chờ cờ của khối trước giả định rằng khối trước rồi sẽ chạy. Chương 11 làm
            điều đó an toàn bằng cách cho mỗi khối một số từ một bộ đếm atomic khi nó bắt đầu, thay vì dùng{' '}
            <code>blockIdx.x</code>: khi đó một số nhỏ hơn luôn thuộc về một khối đã bắt đầu.
          </p>
        }
      >
        <p>
          There is a second hidden condition. A block that spins on an earlier block's flag assumes the earlier block
          will eventually run. Chapter 11 made that safe by giving each block a number from an atomic counter as it
          starts, instead of using <code>blockIdx.x</code>: then a lower number always belongs to a block that has
          already started.
        </p>
      </Translated>
      <Source section="11.9" pages="280" />
      <Translated
        vi={
          <p>
            CUDA giờ nêu rõ quy tắc bên dưới mẹo đó. Một khi một luồng đã bắt đầu, các luồng trong khối của nó được hứa tiến
            triển; một khối chưa bắt đầu thì không được hứa gì. Vậy một khối chỉ chờ các số nhỏ hơn từ bộ đếm thì chỉ chờ các khối
            đã bắt đầu, và những khối đó sẽ xong.
          </p>
        }
      >
        <p>
          CUDA now states the rule underneath that trick. Once a thread has started, the threads of its block are promised
          progress; a block that has not started is promised nothing. So a block that waits only on lower numbers from
          the counter waits only on blocks that have started, and those will finish.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md, A10)', '2026-11-20 (notes/today/ch12-facts.md, mục A10)')}>
        <Translated
          vi={
            <p>
              Quy tắc này là mô hình thực thi của CUDA, mục 5.9 của hướng dẫn lập trình, mà sách không trích dẫn (mô hình bộ nhớ
              là mục 5.8). Nói chặt chẽ, lời hứa bao trùm cluster của luồng đã bắt đầu, cluster này chứa khối của nó.
            </p>
          }
        >
          <p>
            The rule is CUDA's execution model, programming guide section 5.9, which the book does not cite (the memory
            model is 5.8). Strictly, the promise covers the started thread's cluster, which contains its block.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Hai backend còn lại nói ít hơn. Cả AMD lẫn Apple đều không ghi nhận lời hứa như vậy, nên trên HIP và Metal cùng kiểu
            chờ đó dựa vào cách phần cứng hiện nay tình cờ lập lịch các khối. Chính hướng dẫn của AMD nói GPU của họ bắt đầu các
            khối theo thứ tự tăng dần nhưng API không bảo đảm điều đó; đặc tả của Apple không nói gì về các threadgroup trong
            quan hệ với nhau.
          </p>
        }
      >
        <p>
          The other two backends say less. Neither AMD nor Apple documents such a promise, so on HIP and Metal the same
          wait rests on how today's hardware happens to schedule blocks. AMD's own tutorial says its GPUs start blocks in
          increasing order but the API does not guarantee it; Apple's specification says nothing about threadgroups
          relative to each other.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md, B5, B6, C4; ch11-facts C5)', '2026-11-20 (notes/today/ch12-facts.md, mục B5, B6, C4; ch11-facts C5)')}>
        <Translated
          vi={
            <p>
              Số thứ tự khối có sắp xếp của rocPRIM mặc định bị tắt. Một kỹ sư Apple đã gọi kiểu chờ nối chuỗi giữa các
              threadgroup là hành vi không xác định.
            </p>
          }
        >
          <p>
            rocPRIM's ordered block number is off by default. An Apple engineer called a chained wait between
            threadgroups undefined behaviour.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Hoạt hình chạy bộ lọc tại chỗ hai lần trên cùng một GPU, với cùng thứ tự bắt đầu: một lần với số lấy từ bộ đếm, một
            lần với số lấy từ <code>blockIdx.x</code>.
          </p>
        }
      >
        <p>
          The animation runs the in-place filter twice on the same GPU, with the same start order: once with numbers
          from a counter, once with numbers from <code>blockIdx.x</code>.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              GPU trong hoạt hình chứa hai khối cùng lúc và di chuyển chúng từng bước một; GPU thật chứa nhiều hơn nhiều, nhưng
              lập luận vẫn như vậy bất cứ khi nào các khối đang chờ chiếm hết mọi chỗ mà khối còn thiếu có thể bắt đầu.
            </p>
          }
        >
          <p>
            The animation's GPU holds two blocks at once and moves them one step at a time; a real one holds many more,
            but the argument is the same whenever the waiting blocks fill every place the missing one could start in.
          </p>
        </Translated>
      </Simplification>
      <ScanChainScene />
      <Ask q={q['ch12/12.7/q-blockidx']} />

      <h2><Translated vi="Các thiết kế dựa trên hành vi có tài liệu">Designs that rest on documented behaviour</Translated></h2>
      <Translated
        vi={
          <p>
            Gộp lại, mỗi backend nhận thiết kế tại chỗ mà tài liệu của nhà sản xuất hỗ trợ, và không hơn.
          </p>
        }
      >
        <p>
          Put together, each backend gets the in-place design its vendor documents support for, and nothing more.
        </p>
      </Translated>
      <table className="data" data-part="designs">
        <thead>
          <tr><th><Translated vi="Backend">Backend</Translated></th><th><Translated vi="Bộ lọc ổn định tại chỗ">In-place stable filter</Translated></th><th><Translated vi="Dựa trên">Rests on</Translated></th></tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <td>
              <Translated vi="một kernel: scan một lượt, số lấy từ bộ đếm, release và acquire, rào chắn ở cả hai phía (lab 12.5)">
                one kernel: single-pass scan, numbers from a counter, release and acquire, barriers on both sides (lab 12.5)
              </Translated>
            </td>
            <td><Translated vi="quy tắc tiến triển của mô hình thực thi (A10)">the execution model's progress rule (A10)</Translated></td>
          </tr>
          <tr>
            <td>HIP</td>
            <td>
              <Translated
                vi={
                  <>
                    an toàn theo tài liệu: một khối đi qua phạm vi của nó theo thứ tự, chỉ dùng rào chắn khối; một khối cho mỗi
                    hàng của batch. Cũng dựng được: tệp <code>.cu</code> của lab 12.5, unverified
                  </>
                }
              >
                documented-safe: one block walks its range in order, with block barriers only; one block per row of a
                batch. Also builds: lab 12.5's <code>.cu</code> file, unverified
              </Translated>
            </td>
            <td>
              <Translated
                vi={
                  <>
                    rào chắn bên trong một khối. Tệp một lượt dựa vào cách lập lịch khối mà AMD không ghi nhận (B6), và vẫn là
                    unverified cho đến khi một báo cáo từ máy ROCm cho thấy nó vượt qua
                  </>
                }
              >
                barriers inside a block. The single-pass file rests on block scheduling AMD does not document (B6), and
                stays unverified until a report from a ROCm machine shows it passing
              </Translated>
            </td>
          </tr>
          <tr>
            <td>Metal</td>
            <td>
              <Translated vi="một threadgroup đi qua phạm vi của nó theo thứ tự, chỉ dùng rào chắn threadgroup; một threadgroup cho mỗi hàng của batch">
                one threadgroup walks its range in order, with threadgroup barriers only; one threadgroup per row of a batch
              </Translated>
            </td>
            <td>
              <Translated vi="rào chắn bên trong một threadgroup; không có gì giữa các threadgroup (C4)">
                barriers inside a threadgroup; nothing between threadgroups (C4)
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Sao không dùng nhiều dispatch trên HIP hay Metal, như một bộ lọc không tại chỗ sẽ dùng? Đếm, scan các số đếm và rải,
            từng bước đều an toàn, nhưng bước rải đọc và ghi cùng một mảng, và không gì ngăn một khối sau ghi vào phạm vi của
            khối trước trước khi khối đó đọc xong. Một khối duy nhất né được câu hỏi này: nó đọc một đoạn, đồng bộ hóa, ghi các
            khóa được giữ của đoạn tại offset đang chạy, đồng bộ hóa, rồi đi tiếp. Các khóa chỉ dồn về phía đầu, nên mỗi đoạn ghi
            lên các ô đã được đọc. Đây cũng là cấu trúc lab 12.4 dùng cho mỗi hàng logits. (Lập luận này là của khóa học này,
            dựng trên các sự kiện có tài liệu trong bảng.)
          </p>
        }
      >
        <p>
          Why not several dispatches on HIP or Metal, as an out-of-place filter would use? Count, scan the counts and
          scatter are each safe on their own, but the scatter reads and writes the same array, and nothing stops a later
          block from writing into an earlier one's range before that one has read it. One block avoids the question: it
          reads a chunk, synchronises, writes the chunk's kept keys at its running offset, synchronises, and moves on.
          Keys only move towards the front, so each chunk writes onto slots already read. The structure is the one lab
          12.4 uses for each row of logits. (This argument is ours, built on the documented facts in the table.)
        </p>
      </Translated>
      <Ask q={q['ch12/12.7/q-metal']} />
      <Translated
        vi={
          <p>
            Đây là phiên bản CUDA, lời giải tham chiếu của lab 12.5. Hãy tự viết nó trong lab trước khi đọc; đó là bài tập 2 của
            chương cộng thêm biến thể tại chỗ.
          </p>
        }
      >
        <p>
          Here is the CUDA version, lab 12.5's reference solution. Write it yourself in the lab before reading it; it is
          exercise 2 of the chapter plus the in-place twist.
        </p>
      </Translated>
      <Code
        tabs={[{ label: tr('CUDA (and HIP)', 'CUDA (và HIP)'), code: lab5 ?? '// loading labs/' + LAB5, status: lab5Status, highlight: marked(lab5) }]}
        caption={
          <Translated
            vi={
              <>
                <code>stableFilterSinglePassKernel</code>, tải từ <code>labs/{LAB5}</code>. Được đánh dấu: số lấy từ bộ đếm, rào
                chắn sau các lần đọc, acquire, release, và rào chắn trước các lần ghi. Ở đây luồng 0 là luồng dẫn đầu, trong khi
                Hình 11.17 của sách dùng luồng cuối của khối. Bản dựng HIP biên dịch cùng tệp này. Xem{' '}
                <a href={href('ch12', 'labs')}>trang bài thực hành</a> để biết cái gì đã chạy ở đâu.
              </>
            }
          >
            <>
              <code>stableFilterSinglePassKernel</code>, loaded from <code>labs/{LAB5}</code>. Marked: the number from
              the counter, the barrier after the reads, the acquire, the release, and the barrier before the writes.
              Here thread 0 is the leader, where the book's Fig. 11.17 uses the block's last thread. The HIP build
              compiles the same file. See the <a href={href('ch12', 'labs')}>labs page</a> for what has been run where.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Trên CUDA, các thư viện làm việc này giúp bạn. Mục 12.9 chỉ tới Thrust và CUB; <code>DeviceSelect::If</code> của
            CUB có một overload nén gọn tại chỗ và giữ thứ tự ban đầu, và nó dựa vào việc chuyển sang release và acquire nói ở
            trên.
          </p>
        }
      >
        <p>
          Libraries do this for you on CUDA. Section 12.9 points to Thrust and CUB; CUB's <code>DeviceSelect::If</code>{' '}
          has an overload that compacts in place and keeps the original order, and it rests on the release and acquire
          switch above.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 (notes/today/ch12-facts.md, A8, B5)', '2026-11-20 (notes/today/ch12-facts.md, mục A8, B5)')}>
        <Translated
          vi={
            <p>
              Overload tại chỗ của hipCUB đưa cho rocPRIM một con trỏ làm cả đầu vào lẫn đầu ra; tài liệu của rocPRIM không nói
              gì về aliasing: unverified.
            </p>
          }
        >
          <p>
            hipCUB's in-place overload hands rocPRIM one pointer as input and output; rocPRIM's documentation says
            nothing about aliasing: unverified.
          </p>
        </Translated>
      </Today>
      <Ask q={q['ch12/12.7/q-ex2']} label={tr('Exercise 2, p. 302', 'Bài tập 2, p. 302')} />
      <Translated
        vi={
          <p>
            Cùng mối nguy đó xuất hiện ở bất cứ đâu một máy chủ suy luận nén gọn một buffer mà nó không đủ sức sao chép. Bỏ các
            chuỗi đã xong khỏi các buffer theo từng ô, để các chuỗi còn chạy nằm cạnh nhau, là một bộ lọc ổn định tại chỗ, với
            cùng quy tắc và cùng cách sửa.
          </p>
        }
      >
        <p>
          The same hazard turns up wherever an inference server compacts a buffer it cannot afford to copy. Dropping the
          sequences that finished from per-slot buffers, so that the live ones sit together, is an in-place stable
          filter, with the same rule and the same fix.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>Ví dụ máy chủ suy luận là của khóa học này; notes liệt kê việc bỏ các chuỗi đã xong là một công việc nén gọn.</p>}>
          <p>The inference-server example is ours; the notes list dropping finished sequences as a compaction job.</p>
        </Translated>
      </Beyond>
    </>
  );
}
