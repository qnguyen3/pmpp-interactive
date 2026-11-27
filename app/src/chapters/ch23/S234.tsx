import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  DEFAULTS, DEFECTS, VERSIONS, changedEdges, defectsFor, edges, simulate,
  type Defect, type Durations, type EdgeId, type Issue, type Mechanism, type Priority, type Version,
} from '../../sim/haloloop';
import { LoopScene, Timeline, VIEW_W, r2, timelineHeight } from './LoopScene';

// ---------------------------------------------------------------- code shown
// Copies of labs/ch23/snippets/overlap_loop.cu (the marked regions, dedented) and
// labs/ch23/snippets/nccl_exchange.cu; S234.snippets.test.ts fails if they drift and runs
// the CUDA front-end check on both files.
const SETUP = `int low, high;                                    // least, then greatest priority
cudaDeviceGetStreamPriorityRange(&low, &high);    // a lower number is a higher priority
cudaStream_t topStream, bottomStream, internalStream;
cudaStreamCreateWithPriority(&internalStream, cudaStreamDefault, low);
cudaStreamCreateWithPriority(&topStream, cudaStreamDefault, high);
cudaStreamCreateWithPriority(&bottomStream, cudaStreamDefault, high);

cudaEvent_t resetL2, computeTop, computeBottom;   // only ever waited on: no timing
cudaEventCreateWithFlags(&resetL2, cudaEventDisableTiming);
cudaEventCreateWithFlags(&computeTop, cudaEventDisableTiming);
cudaEventCreateWithFlags(&computeBottom, cudaEventDisableTiming);

float* l2norm_h;                                  // pinned, so line 21 stays asynchronous
cudaMallocHost(&l2norm_h, sizeof(float));
float l2norm = 1.0f;                              // anything above tol`;

const LOOP = `for(unsigned int iter = 0; l2norm > tol && iter < iter_max; ++iter) {

    // zero the norm, then make both boundary streams wait for that
    cudaMemsetAsync(l2norm_d, 0, sizeof(float), internalStream);
    cudaEventRecord(resetL2, internalStream);
    cudaStreamWaitEvent(topStream, resetL2, 0);
    cudaStreamWaitEvent(bottomStream, resetL2, 0);

    // one kernel, three launches: row 1, row ny - 2, rows 2 to ny - 3
    launch_jacobi_kernel(output, input, l2norm_d, 3, nx, topStream);
    cudaEventRecord(computeTop, topStream);
    launch_jacobi_kernel(output + (ny - 3)*nx, input + (ny - 3)*nx, l2norm_d, 3,
                         nx, bottomStream);
    cudaEventRecord(computeBottom, bottomStream);
    launch_jacobi_kernel(output + nx, input + nx, l2norm_d, ny - 2, nx,
                         internalStream);

    // the copy of the norm waits for all three kernels
    cudaStreamWaitEvent(internalStream, computeTop, 0);
    cudaStreamWaitEvent(internalStream, computeBottom, 0);
    cudaMemcpyAsync(l2norm_h, l2norm_d, sizeof(float), cudaMemcpyDeviceToHost,
                    internalStream);

    // halo exchange: the host waits for each boundary row, then sends it
    const int top = rank > 0 ? rank - 1 : (numRanks - 1);
    const int bottom = (rank + 1) % numRanks;
    cudaStreamSynchronize(topStream);
    MPI_Sendrecv(output + nx, nx, MPI_FLOAT, top, 0,
                 output + (ny - 1)*nx, nx, MPI_FLOAT, bottom, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cudaStreamSynchronize(bottomStream);
    MPI_Sendrecv(output + (ny - 2) * nx, nx, MPI_FLOAT, bottom, 0,
                 output, nx, MPI_FLOAT, top, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // the host needs the norm: wait for the copy, then reduce across ranks
    cudaStreamSynchronize(internalStream);
    MPI_Allreduce(l2norm_h, &l2norm, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    l2norm = std::sqrt(l2norm);

    std::swap(output, input);

}`;

const NCCL_EXCHANGE = `ncclGroupStart();
ncclSend(output + nx,          nx, ncclFloat, top,    nccl_comm, topStream);
ncclRecv(output + (ny - 1)*nx, nx, ncclFloat, bottom, nccl_comm, topStream);
ncclGroupEnd();
cudaEventRecord(exchangeTop, topStream);
ncclGroupStart();
ncclSend(output + (ny - 2)*nx, nx, ncclFloat, bottom, nccl_comm, bottomStream);
ncclRecv(output,               nx, ncclFloat, top,    nccl_comm, bottomStream);
ncclGroupEnd();
cudaEventRecord(exchangeBottom, bottomStream);

// the host's one wait: it needs the norm
cudaStreamSynchronize(internalStream);
MPI_Allreduce(l2norm_h, &l2norm, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
l2norm = std::sqrt(l2norm);

cudaStreamWaitEvent(internalStream, exchangeTop, 0);
cudaStreamWaitEvent(internalStream, exchangeBottom, 0);`;

const CHECKED_LOOP = 'front-end-checked (labs/ch23/snippets/overlap_loop.cu)';
const CHECKED_NCCL = 'front-end-checked (labs/ch23/snippets/nccl_exchange.cu)';

/** "05, 06" or "28–34" to line numbers. */
function linesOf(spec: string): number[] {
  const out: number[] = [];
  for (const part of spec.split(',')) {
    const m = /(\d+)(?:–(\d+))?/.exec(part.trim());
    if (!m) continue;
    const a = Number(m[1]);
    const b = m[2] ? Number(m[2]) : a;
    for (let k = a; k <= b; k++) out.push(k);
  }
  return out;
}


// ---------------------------------------------------------------- the dependency puzzle
const MECH: Record<Mechanism, string> = {
  get stream() { return tr('same stream', 'cùng stream'); },
  get event() { return tr('event + stream wait', 'event + lần chờ của stream'); },
  get host() { return tr('host waits for a stream', 'host chờ một stream'); },
  get blocking() { return tr('a blocking call returns', 'một lời gọi chặn trả về'); },
  get barrier() { return tr('a barrier on the stream', 'một rào chắn trên stream'); },
  get kernel() { return tr('inside the kernel', 'bên trong kernel'); },
};
const PICKS: Mechanism[] = ['stream', 'event', 'host', 'blocking'];

// The two operations of each ordering in Fig. 23.14, for the feedback.
interface Ends {
  a: string;
  b: string;
  same: boolean;
  /** The second operation is a host call, not stream work. */
  hostSide: boolean;
  done: ReactNode;
  /** Where to look when the pick was a host wait but the code uses an event. */
  look?: string;
}
const ENDS: Record<EdgeId, Ends> = {
  'reset-top': {
    get a() { return tr('the memset (line 04, internalStream)', 'memset (dòng 04, internalStream)'); }, get b() { return tr('the top kernel (line 10, topStream)', 'kernel trên (dòng 10, topStream)'); }, same: false, hostSide: false, get look() { return tr('05 and 06', '05 và 06'); },
    done: <Translated vi={<>Dòng 05 ghi <code>resetL2</code> vào <code>internalStream</code> ngay sau memset, và dòng 06 bắt <code>topStream</code> chờ nó. Không luồng host nào phải chờ.</>}><>Line 05 records <code>resetL2</code> in <code>internalStream</code> right after the memset, and line 06 makes <code>topStream</code> wait for it. No host thread waits.</></Translated>,
  },
  'reset-bottom': {
    get a() { return tr('the memset (line 04, internalStream)', 'memset (dòng 04, internalStream)'); }, get b() { return tr('the bottom kernel (line 12, bottomStream)', 'kernel dưới (dòng 12, bottomStream)'); }, same: false, hostSide: false, get look() { return tr('05 and 07', '05 và 07'); },
    done: <Translated vi={<>Cùng event đó, <code>resetL2</code>, được <code>bottomStream</code> chờ ở dòng 07.</>}><>The same event, <code>resetL2</code>, waited for by <code>bottomStream</code> at line 07.</></Translated>,
  },
  'reset-internal': {
    get a() { return tr('the memset (line 04)', 'memset (dòng 04)'); }, get b() { return tr('the internal kernel (line 15)', 'kernel bên trong (dòng 15)'); }, same: true, hostSide: false,
    done: <Translated vi={<>Cả hai đều ở trong <code>internalStream</code>, memset đi trước (dòng 04, rồi dòng 15): thứ tự stream là đủ.</>}><>Both are in <code>internalStream</code>, the memset first (line 04, then line 15): stream order is enough.</></Translated>,
  },
  'top-copy': {
    get a() { return tr('the top kernel (line 10, topStream)', 'kernel trên (dòng 10, topStream)'); }, get b() { return tr('the copy (line 21, internalStream)', 'lần sao chép (dòng 21, internalStream)'); }, same: false, hostSide: false, get look() { return tr('11 and 19', '11 và 19'); },
    done: <Translated vi={<><code>computeTop</code> được ghi sau kernel trên (dòng 11); <code>internalStream</code> chờ nó ở dòng 19, trước lần sao chép.</>}><><code>computeTop</code> is recorded after the top kernel (line 11); <code>internalStream</code> waits for it at line 19, before the copy.</></Translated>,
  },
  'bottom-copy': {
    get a() { return tr('the bottom kernel (line 12, bottomStream)', 'kernel dưới (dòng 12, bottomStream)'); }, get b() { return tr('the copy (line 21, internalStream)', 'lần sao chép (dòng 21, internalStream)'); }, same: false, hostSide: false, get look() { return tr('14 and 20', '14 và 20'); },
    done: <Translated vi={<><code>computeBottom</code>, được ghi ở dòng 14 và được chờ ở dòng 20.</>}><><code>computeBottom</code>, recorded at line 14 and waited for at line 20.</></Translated>,
  },
  'internal-copy': {
    get a() { return tr('the internal kernel (line 15)', 'kernel bên trong (dòng 15)'); }, get b() { return tr('the copy (line 21)', 'lần sao chép (dòng 21)'); }, same: true, hostSide: false,
    done: <Translated vi={<>Cùng stream: kernel bên trong đứng trước lần sao chép trong <code>internalStream</code>.</>}><>Same stream: the internal kernel is ahead of the copy in <code>internalStream</code>.</></Translated>,
  },
  'top-exchange': {
    get a() { return tr('the top kernel (line 10, topStream)', 'kernel trên (dòng 10, topStream)'); }, get b() { return tr('MPI_Sendrecv (line 28)', 'MPI_Sendrecv (dòng 28)'); }, same: false, hostSide: true,
    done: <Translated vi={<><code>MPI_Sendrecv</code> là một lời gọi phía host, nên chỉ host mới giữ nó lại được: dòng 27 chờ <code>topStream</code>.</>}><><code>MPI_Sendrecv</code> is a host call, so only the host can hold it back: line 27 waits for <code>topStream</code>.</></Translated>,
  },
  'bottom-exchange': {
    get a() { return tr('the bottom kernel (line 12, bottomStream)', 'kernel dưới (dòng 12, bottomStream)'); }, get b() { return tr('MPI_Sendrecv (line 32)', 'MPI_Sendrecv (dòng 32)'); }, same: false, hostSide: true,
    done: <Translated vi={<>Dòng 31 làm điều tương tự cho <code>bottomStream</code>, khi phép trao đổi phía trên đã trả về.</>}><>Line 31 does the same for <code>bottomStream</code>, once the top exchange has returned.</></Translated>,
  },
  'copy-allreduce': {
    get a() { return tr('the copy (line 21, internalStream)', 'lần sao chép (dòng 21, internalStream)'); }, get b() { return tr('MPI_Allreduce (line 38)', 'MPI_Allreduce (dòng 38)'); }, same: false, hostSide: true,
    done: <Translated vi={<><code>MPI_Allreduce</code> đọc <code>l2norm_h</code> trên host, nên host chờ <code>internalStream</code> ở dòng 37.</>}><><code>MPI_Allreduce</code> reads <code>l2norm_h</code> on the host, so the host waits for <code>internalStream</code> at line 37.</></Translated>,
  },
  'exchange-next': {
    get a() { return tr('the two MPI_Sendrecv calls (lines 28 and 32)', 'hai lời gọi MPI_Sendrecv (dòng 28 và 32)'); }, get b() { return tr("the next iteration's launches", 'các lần khởi chạy của vòng lặp tiếp theo'); }, same: false, hostSide: true,
    done: <Translated vi={<><code>MPI_Sendrecv</code> chỉ trả về khi phép trao đổi của nó hoàn tất (dòng 28–34), nên khi host xếp hàng vòng lặp tiếp theo thì các vùng biên đã tới (p. 573).</>}><><code>MPI_Sendrecv</code> returns only when its exchange is complete (lines 28–34), so by the time the host queues the next iteration the halos are in (p. 573).</></Translated>,
  },
};

function hint(id: EdgeId, pick: Mechanism, right: Mechanism): ReactNode {
  const e = ENDS[id];
  if (pick === 'stream')
    return e.hostSide && id !== 'exchange-next'
      ? <Translated vi={<>{e.b.replace(/ \(.*/, '')} không nằm trong stream nào: nó là một lời gọi phía host. Thứ tự stream không giữ nó lại được.</>}><>{e.b.replace(/ \(.*/, '')} is not in any stream: it is a host call. Stream order cannot hold it back.</></Translated>
      : id === 'exchange-next'
        ? <Translated vi={<>Không bên nào là công việc của stream: các phép trao đổi và các lần khởi chạy đều là lời gọi phía host.</>}><>Neither side is stream work: the exchanges and the launches are both host calls.</></Translated>
        : <Translated vi={<>{e.a} và {e.b} nằm ở hai stream khác nhau. Thứ tự stream chỉ sắp thứ tự công việc bên trong một stream.</>}><>{e.a} and {e.b} are in different streams. Stream order only orders work inside one stream.</></Translated>;
  if (pick === 'event')
    return e.same
      ? <Translated vi={<>Cả hai đã ở trong <code>internalStream</code>, theo đúng thứ tự này. Một event không thêm được gì.</>}><>Both are already in <code>internalStream</code>, in this order. An event would add nothing.</></Translated>
      : e.hostSide
        ? <Translated vi={<>Một event có thể bắt một stream chờ, nhưng {e.b.replace(/ \(.*/, '')} là một lời gọi phía host, không phải công việc của stream.</>}><>An event can make a stream wait, but {e.b.replace(/ \(.*/, '')} is a host call, not stream work.</></Translated>
        : <Translated vi={<>Đúng hướng cho hai stream, vậy hãy xem lại đáp án kia: có thật là một event không? Ở đây thì không.</>}><>Right idea for two streams, so check the other answer: is it really an event? It is not here.</></Translated>;
  if (pick === 'host')
    return e.same
      ? <Translated vi={<>Ở đây không có gì trên host phải chờ, và cũng không cần: cả hai đều ở trong <code>internalStream</code>.</>}><>Nothing on the host waits here, and nothing needs to: both are in <code>internalStream</code>.</></Translated>
      : right === 'event'
        ? <Translated vi={<>Một <code>cudaStreamSynchronize</code> cũng được, nhưng nó sẽ chặn host. Fig. 23.14 làm một cách rẻ hơn: hãy xem các dòng {e.look}.</>}><>A <code>cudaStreamSynchronize</code> would also work, but it would stop the host. Fig. 23.14 does something cheaper: look at lines {e.look}.</></Translated>
        : <Translated vi={<>Không có lời gọi đồng bộ hóa nào giữa dòng 34 và các lần khởi chạy của vòng lặp tiếp theo. Thứ bảo đảm điều đó đã nằm sẵn trong các dòng 28–34.</>}><>There is no synchronisation call between line 34 and the next iteration&rsquo;s launches. What guarantees it is already in lines 28–34.</></Translated>;
  // blocking
  if (id === 'top-exchange' || id === 'bottom-exchange')
    return <Translated vi={<><code>MPI_Sendrecv</code> có chặn, nhưng chỉ sau khi nó đã bắt đầu gửi bất cứ thứ gì đang nằm trong hàng. Phải có thứ gì đó chặn host lại trước khi nó gọi hàm này.</>}><><code>MPI_Sendrecv</code> does block, but only after it has started sending whatever is in the row. Something must stop the host before it calls it.</></Translated>;
  if (id === 'copy-allreduce')
    return <Translated vi={<><code>MPI_Allreduce</code> chặn cho đến khi phép rút gọn xong, nhưng nó sẽ đọc <code>l2norm_h</code> trước khi lần sao chép tới nơi. Host phải chờ trước.</>}><><code>MPI_Allreduce</code> blocks until the reduction is done, but it would read <code>l2norm_h</code> before the copy lands. The host has to wait first.</></Translated>;
  return <Translated vi={<>Lời gọi nào sẽ chặn? Các lần khởi chạy kernel, <code>cudaMemsetAsync</code>, <code>cudaMemcpyAsync</code> và <code>cudaEventRecord</code> đều trả về ngay.</>}><>Which call would block? Kernel launches, <code>cudaMemsetAsync</code>, <code>cudaMemcpyAsync</code> and <code>cudaEventRecord</code> all return at once.</></Translated>;
}

interface Row {
  pick: Mechanism | '';
  checked: Mechanism | null;
}
const blank = (): Record<EdgeId, Row> =>
  Object.fromEntries(edges('streams').map((e) => [e.id, { pick: '', checked: null }])) as Record<EdgeId, Row>;

// The one comparison with the libraries' loops the page keeps: a table, from the model's edges.
const LIBS: Version[] = ['nccl', 'nvshmem'];

function CompareTable() {
  const book = edges('streams');
  const lib = Object.fromEntries(LIBS.map((v) => [v, edges(v)])) as Record<Version, ReturnType<typeof edges>>;
  const changed = Object.fromEntries(LIBS.map((v) => [v, changedEdges('streams', v)])) as Record<Version, EdgeId[]>;
  const where = (lines: string) => (
    <span className="muted"> ({/^\d/.test(lines) ? (linesOf(lines).length > 1 ? tr('lines ', 'dòng ') : tr('line ', 'dòng ')) : ''}{lines})</span>
  );
  return (
    <div style={{ overflowX: 'auto', marginTop: '0.8rem' }}>
      <table className="data" data-compare-table style={{ margin: 0 }}>
        <thead>
          <tr>
            <th><Translated vi="Thứ tự cần giữ">Ordering</Translated></th>
            <th>{VERSIONS.streams.fig}</th>
            {LIBS.map((v) => (
              <th key={v} data-compare-count={changed[v].length} data-version={v}>
                <Translated vi={<>{VERSIONS[v].label} ({VERSIONS[v].fig}): {changed[v].length} thay đổi</>}>
                  {VERSIONS[v].label} ({VERSIONS[v].fig}): {changed[v].length} changed
                </Translated>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {book.map((e) => (
            <tr key={e.id} data-compare-row={e.id}>
              <td style={{ whiteSpace: 'normal' }}>{tr(e.first, (({
                'reset-top': 'chuẩn được đặt về 0', 'reset-bottom': 'chuẩn được đặt về 0', 'reset-internal': 'chuẩn được đặt về 0',
                'top-copy': 'kernel trên đã cộng phần của nó', 'bottom-copy': 'kernel dưới đã cộng phần của nó', 'internal-copy': 'kernel bên trong đã cộng phần của nó',
                'top-exchange': 'kernel trên đã tính xong hàng 1', 'bottom-exchange': 'kernel dưới đã tính xong hàng ny − 2',
                'copy-allreduce': 'lần sao chép của chuẩn đã tới l2norm_h', 'exchange-next': 'cả hai phép trao đổi đã giao xong vùng biên',
              }) as Record<string, string>)[e.id] ?? e.first)} <span className="muted"><Translated vi="trước khi">before</Translated></span> {tr(e.then, (({
                'reset-top': 'kernel trên cộng vào nó', 'reset-bottom': 'kernel dưới cộng vào nó', 'reset-internal': 'kernel bên trong cộng vào nó',
                'top-copy': 'chuẩn được sao chép về host', 'bottom-copy': 'chuẩn được sao chép về host', 'internal-copy': 'chuẩn được sao chép về host',
                'top-exchange': 'phép trao đổi phía trên gửi nó đi', 'bottom-exchange': 'phép trao đổi phía dưới gửi nó đi',
                'copy-allreduce': 'MPI_Allreduce đọc nó', 'exchange-next': 'các kernel của vòng lặp tiếp theo đọc chúng',
              }) as Record<string, string>)[e.id] ?? e.then)}</td>
              <td style={{ whiteSpace: 'normal' }}>{MECH[e.mechanism]}</td>
              {LIBS.map((v) => {
                const o = lib[v].find((x) => x.id === e.id)!;
                const isChanged = changed[v].includes(e.id);
                return (
                  <td key={v} style={{ whiteSpace: 'normal' }} data-version={v} data-mechanism={o.mechanism} data-changed={isChanged ? 'true' : 'false'}>
                    {isChanged ? <b>{MECH[o.mechanism]}</b> : MECH[o.mechanism]}
                    {where(o.lines)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DependencyPuzzle() {
  const book = edges('streams');
  const [rows, setRows] = useState<Record<EdgeId, Row>>(blank);
  const [focus, setFocus] = useState<EdgeId | null>(null);
  const solved = book.filter((e) => rows[e.id].checked === e.mechanism);
  const all = solved.length === book.length;
  const hl = focus ? linesOf(book.find((e) => e.id === focus)!.lines) : [];
  return (
    <Bench
      title={tr('What enforces each ordering?', 'Thứ gì bảo đảm mỗi thứ tự?')}
      task={tr(
        'Ten orderings one iteration needs. For each, pick what makes it hold in Fig. 23.14.',
        'Mười thứ tự mà một vòng lặp cần. Với mỗi thứ tự, chọn thứ làm nó được giữ trong Fig. 23.14.',
      )}
      foot={tr(
        "The orderings are the ones the book names on pp. 559–563 and 573. The line numbers are the book's.",
        'Các thứ tự là những thứ tự sách nêu ở pp. 559–563 và 573. Số dòng là của sách.',
      )}
    >
      <div data-puzzle>
        <Code
          tabs={[
            { label: tr('Fig. 23.14, corrected', 'Fig. 23.14, đã sửa'), code: LOOP, status: CHECKED_LOOP, highlight: hl },
            { label: tr('setup (pp. 558–562)', 'thiết lập (pp. 558–562)'), code: SETUP, status: CHECKED_LOOP },
          ]}
          caption={
            <Translated
              vi={
                <>
                  Vòng lặp của sách với chính số dòng của nó. Dòng 19 được in thừa một dấu đóng ngoặc trên trang sách; bản này
                  dùng dạng mà Fig. 23.20 in. Các chú thích là của khóa học này. Tab thiết lập gom những gì văn bản trình bày rải
                  rác, với cả ba event được tạo không có đo thời gian.
                </>
              }
            >
              <>
                The book&rsquo;s loop with its own line numbers. Line 19 is printed with an extra closing parenthesis on the
                page; this copy has the one Fig. 23.20 prints. The comments are ours. The setup tab gathers what the text
                shows in pieces, with all three events created without timing.
              </>
            </Translated>
          }
        />
        <div className="panel">
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: '0.2rem' }}>
            <b><Translated vi="Mười thứ tự">Ten orderings</Translated></b>
            <span className="muted" data-puzzle-score>
              <Translated vi={<>đã đặt {solved.length} trên {book.length}</>}>
                {solved.length} of {book.length} placed
              </Translated>
            </span>
          </div>
          {book.map((e, n) => {
            const s = rows[e.id];
            const done = s.checked === e.mechanism;
            return (
              <div key={e.id} data-edge={e.id} data-solved={done ? 'true' : 'false'}
                style={{ padding: '0.6rem 0', borderTop: n ? '1px solid var(--rule-soft)' : undefined }}>
                <div className="row" style={{ alignItems: 'flex-end' }}>
                  <p style={{ margin: 0, flex: '1 1 16rem', minWidth: '12rem' }}>
                    {tr(e.first, (({
                      'reset-top': 'chuẩn được đặt về 0', 'reset-bottom': 'chuẩn được đặt về 0', 'reset-internal': 'chuẩn được đặt về 0',
                      'top-copy': 'kernel trên đã cộng phần của nó', 'bottom-copy': 'kernel dưới đã cộng phần của nó', 'internal-copy': 'kernel bên trong đã cộng phần của nó',
                      'top-exchange': 'kernel trên đã tính xong hàng 1', 'bottom-exchange': 'kernel dưới đã tính xong hàng ny − 2',
                      'copy-allreduce': 'lần sao chép của chuẩn đã tới l2norm_h', 'exchange-next': 'cả hai phép trao đổi đã giao xong vùng biên',
                    }) as Record<string, string>)[e.id] ?? e.first)} <span className="muted"><Translated vi="trước khi">before</Translated></span> {tr(e.then, (({
                      'reset-top': 'kernel trên cộng vào nó', 'reset-bottom': 'kernel dưới cộng vào nó', 'reset-internal': 'kernel bên trong cộng vào nó',
                      'top-copy': 'chuẩn được sao chép về host', 'bottom-copy': 'chuẩn được sao chép về host', 'internal-copy': 'chuẩn được sao chép về host',
                      'top-exchange': 'phép trao đổi phía trên gửi nó đi', 'bottom-exchange': 'phép trao đổi phía dưới gửi nó đi',
                      'copy-allreduce': 'MPI_Allreduce đọc nó', 'exchange-next': 'các kernel của vòng lặp tiếp theo đọc chúng',
                    }) as Record<string, string>)[e.id] ?? e.then)}
                  </p>
                  <label className="field">
                    <Translated vi="Được bảo đảm bởi">Enforced by</Translated>
                    <select aria-label={tr(`Mechanism for ${e.id}`, `Cơ chế cho ${e.id}`)} value={s.pick} disabled={done}
                      onChange={(ev) => setRows((r) => ({ ...r, [e.id]: { pick: ev.target.value as Mechanism | '', checked: null } }))}>
                      <option value="">{tr('choose', 'chọn')}</option>
                      {PICKS.map((m) => (
                        <option key={m} value={m}>{MECH[m]}</option>
                      ))}
                    </select>
                  </label>
                  <div style={{ minWidth: '4.6rem' }}>
                    {done ? (
                      <span className="chip ok"><Translated vi="đã đặt">placed</Translated></span>
                    ) : (
                      <button className="primary" aria-label={tr(`Check ${e.id}`, `Kiểm tra ${e.id}`)} disabled={s.pick === ''}
                        onClick={() => {
                          setRows((r) => ({ ...r, [e.id]: { ...r[e.id], checked: s.pick as Mechanism } }));
                          setFocus(e.id);
                        }}>
                        <Translated vi="Kiểm tra">Check</Translated>
                      </button>
                    )}
                  </div>
                </div>
                {s.checked ? (
                  <p data-feedback={e.id} role="status" style={{ margin: '0.4rem 0 0' }}>
                    {done ? (
                      <><span className="chip ok">{linesOf(e.lines).length > 1 ? tr('lines', 'dòng') : tr('line', 'dòng')} {e.lines}</span> {ENDS[e.id].done}</>
                    ) : (
                      <><span className="chip fail"><Translated vi="không phải">not that</Translated></span> {hint(e.id, s.checked, e.mechanism)}</>
                    )}
                  </p>
                ) : null}
              </div>
            );
          })}
        </div>
        <div className="row" style={{ marginTop: '0.8rem', justifyContent: 'space-between' }}>
          <span className="ui muted">
            {all ? null : tr('Place all ten to see the same orderings in the NCCL and NVSHMEM loops.', 'Đặt đủ cả mười để xem cùng các thứ tự đó trong vòng lặp NCCL và NVSHMEM.')}
          </span>
          <button className="quiet" onClick={() => { setRows(blank()); setFocus(null); }}>
            <Translated vi="Làm lại từ đầu">Start again</Translated>
          </button>
        </div>
        {all ? (
          <div data-puzzle-done>
            <Translated
              vi={
                <p className="panel" style={{ margin: '0.8rem 0 0' }}>
                  <b>Đã đặt đủ mười.</b> Đếm các cơ chế: bốn loại, và loại cuối cùng là host chờ dưới một hình thức khác. Bảng
                  đặt cùng mười thứ tự đó vào vòng lặp NCCL và NVSHMEM của sách; những thứ tự thay đổi được in đậm. Trong vòng
                  lặp NCCL, các phép trao đổi được xếp hàng trong các stream biên, nên thứ tự stream và việc chờ event thay cho
                  việc host chờ. Trong vòng lặp NVSHMEM, một kernel tự đặt (put) các hàng, và một rào chắn xếp hàng trong stream
                  thay cho phép trao đổi chặn.
                </p>
              }
            >
              <p className="panel" style={{ margin: '0.8rem 0 0' }}>
                <b>All ten placed.</b> Count the mechanisms: four kinds, and the last one is the host waiting in disguise.
                The table puts the same ten orderings in the book&rsquo;s NCCL and NVSHMEM loops; the changed ones are in
                bold. In NCCL&rsquo;s the exchanges are queued in the boundary streams, so stream order and event waits
                replace the host&rsquo;s. In NVSHMEM&rsquo;s one kernel puts the rows itself, and a barrier queued in the
                stream replaces the blocking exchange.
              </p>
            </Translated>
            <CompareTable />
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- who waits
// The explorer models the book's two MPI loops only; NCCL and NVSHMEM are the comparison table's.
const ALL_VERSIONS: Version[] = ['mpi', 'streams'];
const SLIDERS: { key: keyof Durations; label: string; min: number; max: number; step: number }[] = [
  { key: 'internal', get label() { return tr('internal kernel', 'kernel bên trong'); }, min: 2, max: 12, step: 0.5 },
  { key: 'boundary', get label() { return tr('boundary kernels', 'các kernel biên'); }, min: 0.5, max: 4, step: 0.25 },
  { key: 'exTop', get label() { return tr('top exchange', 'trao đổi phía trên'); }, min: 0.5, max: 12, step: 0.5 },
  { key: 'exBottom', get label() { return tr('bottom exchange', 'trao đổi phía dưới'); }, min: 0.5, max: 12, step: 0.5 },
];

// Where to look when a removed ordering did not fire: the model's own findings.
const QUIET: Record<Exclude<Defect, 'none'>, Partial<Record<Version, ReactNode>>> = {
  'reset-waits': {
    streams: <Translated vi={<>Trong vòng lặp này, theo mô hình, phép đặt lại luôn thắng cuộc đua: dòng 37 vừa làm rỗng <code>internalStream</code>, và memset là thứ đầu tiên trong đó. Lỗi vẫn còn đó: không có gì trong mã bắt phép đặt lại đi trước, và một host hay GPU lập lịch khác đi có thể thua cuộc đua.</>}><>In this loop the reset always wins the race in the model: line 37 has just emptied <code>internalStream</code>, and the memset is the first thing in it. The bug is still there: nothing in the code makes the reset come first, and a host or a GPU that schedules differently can lose it.</></Translated>,
  },
  'compute-waits': {
    streams: <Translated vi={<>Hãy cho một kernel biên chạy lâu hơn kernel bên trong, hoặc phát lệnh kernel bên trong trước với độ ưu tiên bằng nhau.</>}><>Make a boundary kernel outlast the internal one, or issue the internal kernel first with equal priority.</></Translated>,
  },
  'sync-top': {},
  'exchange-waits': {},
  barrier: {},
  pageable: {},
};

function Seg<T extends string>({ label, value, options, onChange, disabled }: {
  label: string; value: T; options: [T, string][]; onChange: (v: T) => void; disabled?: boolean;
}) {
  return (
    <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
      <span>{label}</span>
      <span className="seg" role="group" aria-label={label}>
        {options.map(([v, text]) => (
          <button key={v} aria-pressed={value === v} disabled={disabled} onClick={() => onChange(v)}>{text}</button>
        ))}
      </span>
    </div>
  );
}

function WhoWaits() {
  const [version, setVersion] = useState<Version>('streams');
  const [d, setD] = useState<Durations>(DEFAULTS);
  const [priority, setPriority] = useState<Priority>('higher');
  const [issue, setIssue] = useState<Issue>('boundary-first');
  const [defect, setDefect] = useState<Defect>('none');
  const one = version === 'mpi' || version === 'nvshmem';
  const locale = useLocale();
  const r = useMemo(() => simulate({ ...d, version, priority, issue, defect }), [d, version, priority, issue, defect, locale]);
  const others = useMemo(() => ALL_VERSIONS.map((v) => simulate({ ...d, version: v, priority, issue })), [d, priority, issue]);
  const t1 = Math.max(4, Math.ceil(r.end + 0.2));
  const svgH = 18 + timelineHeight(r.lanes.length, true) + 4;
  const pick = (v: Version) => {
    setVersion(v);
    if (!defectsFor(v).includes(defect)) setDefect('none');
  };
  const quiet = defect !== 'none' && r.violations.length === 0 ? QUIET[defect][version] : null;
  return (
    <Bench
      title={tr('Who waits', 'Ai chờ')}
      task={tr(
        'Pick a version and drag the durations. Then remove one ordering and find when it breaks.',
        'Chọn một phiên bản và kéo các khoảng thời gian. Rồi bỏ một thứ tự và tìm lúc nó hỏng.',
      )}
      foot={tr(
        "Times are unitless and come from this page's model (src/sim/haloloop.ts), not from any device. Two iterations are drawn.",
        'Thời gian không có đơn vị và đến từ mô hình của trang này (src/sim/haloloop.ts), không từ thiết bị nào. Hai vòng lặp được vẽ.',
      )}
    >
      <div data-who-waits data-version={version}>
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <Seg label={tr('Version', 'Phiên bản')} value={version} onChange={pick}
            options={ALL_VERSIONS.map((v) => [v, tr(`${VERSIONS[v].label} (${VERSIONS[v].fig})`, `${({ mpi: 'MPI, không chồng lấp', streams: 'MPI với stream' } as Record<string, string>)[v] ?? VERSIONS[v].label} (${VERSIONS[v].fig})`)] as [Version, string])} />
        </div>
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          {SLIDERS.map((s) => (
            <label key={s.key} className="field" style={{ minWidth: '9.5rem' }}>
              <span>
                {s.label}: <b data-dur={s.key}>{d[s.key]}</b>
              </span>
              <input type="range" min={s.min} max={s.max} step={s.step} value={d[s.key]} aria-label={s.label}
                onChange={(e) => setD((x) => ({ ...x, [s.key]: Number(e.target.value) }))} />
            </label>
          ))}
        </div>
        <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
          <Seg label={tr("Boundary streams' priority", 'Độ ưu tiên của các stream biên')} value={priority} onChange={setPriority} disabled={one}
            options={[['equal', tr('equal', 'bằng nhau')], ['higher', tr('higher (the book)', 'cao hơn (sách)')]]} />
          <Seg label={tr('Launch order', 'Thứ tự khởi chạy')} value={issue} onChange={setIssue} disabled={one}
            options={[['boundary-first', tr('boundary kernels first (the book)', 'kernel biên trước (sách)')], ['internal-first', tr('internal kernel first', 'kernel bên trong trước')]]} />
          <label className="field" style={{ maxWidth: '100%' }}>
            <Translated vi="Bỏ một thứ tự">Remove one ordering</Translated>
            <select aria-label={tr('Remove one ordering', 'Bỏ một thứ tự')} value={defect} onChange={(e) => setDefect(e.target.value as Defect)} style={{ maxWidth: '100%' }}>
              {defectsFor(version).map((x) => (
                <option key={x} value={x}>
                  {x === 'none' ? (version === 'mpi' ? tr('none (nothing to remove here)', 'không (ở đây không có gì để bỏ)') : tr("none: the book's code", 'không: mã của sách')) : tr(`${DEFECTS[x].label} (${DEFECTS[x].where})`, `${({
                    'reset-waits': 'bỏ hai lần chờ resetL2',
                    'compute-waits': 'bỏ các lần chờ computeTop và computeBottom',
                    'sync-top': 'bỏ lần đồng bộ host trước phép trao đổi phía trên',
                    'exchange-waits': 'bỏ các lần chờ exchangeTop và exchangeBottom',
                    barrier: 'bỏ nvshmemx_barrier_all_on_stream',
                    pageable: 'cấp phát l2norm_h bằng malloc (pageable)',
                  } as Record<string, string>)[x] ?? DEFECTS[x].label} (${DEFECTS[x].where.replace(/lines/, 'dòng').replace(/line/, 'dòng')})`)}
                </option>
              ))}
            </select>
          </label>
          <button className="quiet" onClick={() => { setD(DEFAULTS); setPriority('higher'); setIssue('boundary-first'); setDefect('none'); }}>
            <Translated vi="Về mặc định">Back to the defaults</Translated>
          </button>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <svg viewBox={`0 0 ${VIEW_W} ${svgH}`} role="img" style={{ width: '100%', minWidth: 700, display: 'block' }}
            aria-label={tr(`Timeline of two iterations of ${VERSIONS[version].label}`, `Dòng thời gian hai vòng lặp của ${VERSIONS[version].label}`)} data-axis={t1}>
            <Timeline r={r} y={18} t1={t1} slots gap violations />
          </svg>
        </div>
        <div className="legend ui" style={{ margin: '0.5rem 0 0.8rem', fontSize: '0.8rem' }}>
          <span><i style={{ background: 'var(--ink-2)' }} /><Translated vi="kernel">kernel</Translated></span>
          <span><i style={{ border: '1px dashed var(--ink-3)' }} /><Translated vi="đã xếp hàng, đang chờ chỗ cho khối">queued, waiting for block slots</Translated></span>
          <span><i style={{ border: '1px dotted var(--ink-3)' }} /><Translated vi="stream bị giữ bởi một lần chờ hoặc một rào chắn">stream held by a wait or a barrier</Translated></span>
          <span><i style={{ background: 'repeating-linear-gradient(45deg, var(--ink-3) 0 1.5px, var(--page) 1.5px 4px)', border: '1px solid var(--ink-2)' }} /><Translated vi="host bị chặn trong một lời gọi">host blocked in a call</Translated></span>
          <span><i style={{ background: 'var(--global-wash)', border: '1px solid var(--global)' }} /><Translated vi="dữ liệu đang di chuyển: sao chép hoặc trao đổi">data on the move: copy or exchange</Translated></span>
          <span><i style={{ background: 'var(--fail)' }} /><Translated vi="một thứ tự bị phá vỡ">an ordering broken</Translated></span>
        </div>
        <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(13rem, 1fr))' }}>
          <div className="panel" data-readouts>
            <table className="data" style={{ margin: 0 }}>
              <tbody>
                <tr>
                  <td><Translated vi="thời gian mỗi vòng lặp">time per iteration</Translated></td>
                  <td className="num" data-readout="iter" data-value={r2(r.iterTime)}><b>{r2(r.iterTime)}</b></td>
                </tr>
                <tr>
                  <td><Translated vi="host bị chặn mỗi vòng lặp">host blocked per iteration</Translated></td>
                  <td className="num" data-readout="blocked" data-value={r2(r.hostBlocked)}>{r2(r.hostBlocked)}</td>
                </tr>
                <tr>
                  <td><Translated vi="hàng dưới đã sẵn sàng, chưa được gửi">bottom rows ready, not yet sent</Translated></td>
                  <td className="num" data-readout="gap" data-value={r2(r.bottomGap)}>{r2(r.bottomGap)}</td>
                </tr>
                <tr>
                  <td><Translated vi="phép trao đổi được giấu">exchanges hidden</Translated></td>
                  <td className="num" data-readout="hidden" data-value={r.hidden ? 'yes' : 'no'}>
                    {r.hidden ? <span className="chip ok"><Translated vi="có">yes</Translated></span> : <span className="chip fail"><Translated vi="không">no</Translated></span>}
                  </td>
                </tr>
              </tbody>
            </table>
            <p className="ui muted" style={{ margin: '0.6rem 0 0', fontSize: '0.8rem' }} data-compare-times>
              <Translated vi="Thời gian mỗi vòng lặp ở các khoảng thời gian này:">Time per iteration at these durations:</Translated>{' '}
              {others.map((o, k) => (
                <span key={o.input.version} data-version-time={o.input.version} data-value={r2(o.iterTime)}
                  style={o.input.version === version ? { color: 'var(--ink)', fontWeight: 600 } : undefined}>
                  {k ? ' · ' : ''}{tr(VERSIONS[o.input.version].label, ({ mpi: 'MPI, không chồng lấp', streams: 'MPI với stream' } as Record<string, string>)[o.input.version] ?? VERSIONS[o.input.version].label)} {r2(o.iterTime)}
                </span>
              ))}
              <Translated vi=". Vòng lặp 2 là vòng được đo.">. Iteration 2 is the one measured.</Translated>
            </p>
          </div>
          <div className="panel" data-problems={r.violations.length}
            style={r.violations.length ? { borderColor: 'var(--fail)', background: 'var(--fail-wash)' } : undefined} aria-live="polite">
            {r.violations.length ? (
              <>
                <p style={{ margin: '0 0 0.4rem' }}>
                  <b className="t-fail">
                    {r.violations.some((v) => v.kind !== 'stall') ? tr(`${r.violations.filter((v) => v.kind !== 'stall').length} broken ordering${r.violations.filter((v) => v.kind !== 'stall').length > 1 ? 's' : ''}`, `${r.violations.filter((v) => v.kind !== 'stall').length} thứ tự bị phá vỡ`) : tr('No wrong result, but the host stalls', 'Không có kết quả sai, nhưng host bị kẹt')}
                  </b>
                </p>
                <ul style={{ margin: 0, paddingLeft: '1.1rem' }}>
                  {r.violations.map((v, k) => (
                    <li key={k} data-violation={v.kind}>{tr(v.text, ((s: string) => {
                      const P: Record<string, string> = { top: 'trên', bottom: 'dưới', internal: 'bên trong' };
                      return s
                        .replace(/^Iteration (\d+): a block of the (\w+) (rows|kernel) adds its squared residues to the norm at ([\d.]+), but the memset zeroes it only at ([\d.]+)\. That part is lost: the norm comes out too small\.$/,
                          (_m, i: string, p: string, w: string, a: string, b: string) => `Vòng lặp ${i}: một khối của ${w === 'rows' ? 'các hàng' : 'kernel'} ${P[p] ?? p} cộng bình phương phần dư vào chuẩn ở ${a}, nhưng memset chỉ đặt nó về 0 ở ${b}. Phần đó bị mất: chuẩn ra nhỏ hơn thực tế.`)
                        .replace(/^Iteration (\d+): the norm is copied to the host at ([\d.]+), but the (\w+) kernel adds its last part only at ([\d.]+)\. The host reduces a norm that is missing it\.$/,
                          (_m, i: string, a: string, p: string, b: string) => `Vòng lặp ${i}: chuẩn được sao chép về host ở ${a}, nhưng kernel ${P[p] ?? p} chỉ cộng phần cuối của nó ở ${b}. Host rút gọn một chuẩn bị thiếu phần đó.`)
                        .replace(/^Iteration (\d+): the (\w+) exchange starts sending at ([\d.]+), but the \w+ kernel finishes the row only at ([\d.]+)\. The neighbour receives a row that is not yet computed\.$/,
                          (_m, i: string, w: string, a: string, b: string) => `Vòng lặp ${i}: phép trao đổi phía ${P[w] ?? w} bắt đầu gửi ở ${a}, nhưng kernel ${P[w] ?? w} chỉ tính xong hàng đó ở ${b}. Lân cận nhận một hàng chưa được tính.`)
                        .replace(/^Iteration (\d+): the (\w+) (rows start|kernel starts) at ([\d.]+) and reads? the \w+ halo, which arrives only at ([\d.]+) \(with the (\w+) exchange of iteration (\d+)\)\. (They read|It reads) stale values\.$/,
                          (_m, i: string, w: string, k: string, a: string, b: string, w2: string, j: string) => `Vòng lặp ${i}: ${k === 'rows start' ? 'các hàng' : 'kernel'} ${P[w] ?? w} bắt đầu ở ${a} và đọc vùng biên ${P[w] ?? w}, vốn chỉ tới ở ${b} (cùng phép trao đổi phía ${P[w2] ?? w2} của vòng lặp ${j}). Nó đọc giá trị cũ.`)
                        .replace(/^Iteration (\d+): MPI_Allreduce starts at ([\d.]+), before the copy of the norm ends at ([\d.]+)\.$/,
                          (_m, i: string, a: string, b: string) => `Vòng lặp ${i}: MPI_Allreduce bắt đầu ở ${a}, trước khi lần sao chép của chuẩn kết thúc ở ${b}.`)
                        .replace(/^Iteration (\d+): cudaMemcpyAsync into pageable memory is synchronous: the host waits in it from ([\d.]+) to ([\d.]+), and nothing after it can be issued until then\.$/,
                          (_m, i: string, a: string, b: string) => `Vòng lặp ${i}: cudaMemcpyAsync vào bộ nhớ phân trang là đồng bộ: host chờ trong đó từ ${a} đến ${b}, và không gì sau nó được phát lệnh cho đến lúc đó.`);
                    })(v.text))}</li>
                  ))}
                </ul>
              </>
            ) : (
              <>
                <p style={{ margin: 0 }} data-no-problems>
                  <b><Translated vi="Không thứ tự nào bị phá vỡ ở các khoảng thời gian này.">No ordering is broken at these durations.</Translated></b>
                </p>
                {quiet ? <p style={{ margin: '0.4rem 0 0' }} className="muted" data-quiet>{defect !== 'none' ? <Translated vi={<>Lỗi vẫn ở đó; nó chỉ không phát tác. </>}><>The defect is there; it did not fire. </></Translated> : null}{quiet}</p> : null}
              </>
            )}
          </div>
        </div>
      </div>
    </Bench>
  );
}

// The longest exchange each loop still hides at the defaults, both exchanges equal: from the
// model, for the prediction. In Fig. 23.14 the two run one after the other, so the slack after
// two short exchanges is shared between them; in the NCCL loop each starts when its kernel ends.
const SHORT = 0.5;
const STREAMS_SHORT = simulate({ ...DEFAULTS, exTop: SHORT, exBottom: SHORT, version: 'streams' }).iters[0];
export const WINDOW_STREAMS = SHORT + (STREAMS_SHORT.internal.end - STREAMS_SHORT.exBottom.end) / 2;
const NCCL_SHORT = simulate({ ...DEFAULTS, exTop: SHORT, exBottom: SHORT, version: 'nccl' }).iters[0];
export const WINDOW_NCCL = NCCL_SHORT.internal.end - Math.max(NCCL_SHORT.top.end, NCCL_SHORT.bottom.end);
const STREAMS_DEF = simulate({ ...DEFAULTS, version: 'streams' });

// ---------------------------------------------------------------- questions
export const questionsS234: Record<string, Question> = {
  'ch23/23.4/q-threerows': {
    kind: 'numeric',
    id: 'ch23/23.4/q-threerows',
    prompt: (
      <Translated
        vi={
          <>
            <code>ny = 10</code>. Lần khởi chạy dưới của Fig. 23.14 nhận <code>output + (ny − 3)*nx</code> và{' '}
            <code>input + (ny − 3)*nx</code> cùng số hàng là 3. Tính hàng vùng biên trên là hàng 0, nó ghi hàng nào của mảng
            cục bộ?
          </>
        }
      >
        <code>ny = 10</code>. The bottom launch of Fig. 23.14 receives <code>output + (ny − 3)*nx</code> and{' '}
        <code>input + (ny − 3)*nx</code> and a row count of 3. Counting the top halo as row 0, which row of the local
        array does it write?
      </Translated>
    ),
    answer: 8,
    explain: (
      <Translated
        vi={
          <p>
            Kernel của Fig. 23.1 đánh số hàng từ 1 và dừng trước hàng <code>ny − 1</code> của phần nó được giao. Được giao 3
            hàng, nó chỉ nhận hàng 1 trong ba hàng: hàng giữa. Hàng 0 của lần khởi chạy là hàng 7 của mảng (
            <code>ny − 3</code>), nên nó ghi hàng 8, tức <code>ny − 2</code>, hàng rìa dưới. Lập luận tương tự cho hàng 1 với
            lần khởi chạy trên và các hàng 2 đến 7 với lần khởi chạy bên trong: mỗi hàng bên trong đúng một lần.
          </p>
        }
      >
        <p>
          Fig. 23.1&rsquo;s kernel numbers its rows from 1 and stops before row <code>ny − 1</code> of what it was given.
          Given 3 rows, that admits only row 1 of the three: the middle one. The launch&rsquo;s row 0 is the array&rsquo;s
          row 7 (<code>ny − 3</code>), so it writes row 8, which is <code>ny − 2</code>, the bottom boundary row. The same
          reasoning gives row 1 for the top launch and rows 2 to 7 for the internal one: every interior row once.
        </p>
      </Translated>
    ),
    source: 'Fig. 23.14 lines 12–13 (p. 557) with the guard of Fig. 23.1 (p. 542); the derivation is ours (notes/ch23.md)',
    vi: { source: 'Fig. 23.14 dòng 12–13 (p. 557) với điều kiện chặn của Fig. 23.1 (p. 542); phép suy ra là của khóa học này (notes/ch23.md)' },
  },
  'ch23/23.4/q-prio': {
    kind: 'choice',
    id: 'ch23/23.4/q-prio',
    prompt: (
      <Translated
        vi={
          <>
            Kernel bên trong được khởi chạy trước và các khối của nó lấp đầy GPU. Hai kernel biên sau đó được khởi chạy vào
            các stream có độ ưu tiên cao hơn. Khi nào chúng bắt đầu?
          </>
        }
      >
        The internal kernel was launched first and its blocks fill the GPU. The two boundary kernels are then launched
        into streams of higher priority. When do they start?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Khi các khối của kernel bên trong xong dần: độ ưu tiên không chiếm quyền (preempt) gì cả; nó chỉ cho các kernel đang chờ lấy trước những chỗ vừa được giải phóng.">
            As blocks of the internal kernel finish: a priority preempts nothing; it gives pending kernels the freed slots first.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Ngay lập tức: bộ lập lịch tạm dừng một số khối đang chạy của kernel bên trong để nhường chỗ cho các kernel ưu tiên cao hơn.">
            At once: the scheduler pauses some of the internal kernel’s running blocks to make room for the higher-priority kernels.
          </Translated>
        ),
        why: <Translated vi="NVIDIA: độ ưu tiên “do not preempt already-running work” (A2).">NVIDIA: priorities “do not preempt already-running work” (A2).</Translated>,
      },
      {
        label: (
          <Translated vi="Chỉ khi kernel bên trong đã xong hoàn toàn, vì độ ưu tiên chỉ sắp thứ tự các kernel được khởi chạy cùng một lúc.">
            Only once the internal kernel has finished entirely, because a priority only orders kernels launched at the same moment.
          </Translated>
        ),
        why: <Translated vi="Đó là điều độ ưu tiên bằng nhau cho ra trong trình khám phá. Với độ ưu tiên cao hơn, chúng lấy những chỗ đầu tiên được giải phóng.">That is what equal priority gives in the explorer. With a higher priority they take the first slots that free up.</Translated>,
      },
      {
        label: (
          <Translated vi="Ngay lập tức, trên một phần GPU dành riêng cho các stream ưu tiên cao mà các khối của kernel bên trong không chiếm được.">
            At once, on part of the GPU set aside for high-priority streams that the internal kernel’s blocks cannot occupy.
          </Translated>
        ),
        why: <Translated vi="Không có phần cứng dành riêng nào như vậy trong tài liệu. Độ ưu tiên quyết định ai được lấy một chỗ vừa trống, không hơn.">No such reserved hardware is documented. A priority decides who gets a slot that frees, nothing more.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Độ ưu tiên của stream là một gợi ý về công việc đang chờ. Khi các khối của kernel bên trong đã chiếm mọi chỗ, không
            có gì bị lấy khỏi chúng. Thứ độ ưu tiên cao hơn mua được là những chỗ tiếp theo: khi mỗi khối bên trong xong, một
            khối biên đang chờ vào thế chỗ thay vì một khối bên trong khác. Trong trình khám phá, khi kernel bên trong được phát
            lệnh trước, đó là khác biệt giữa bắt đầu ở lần đầu tiên một khối kết thúc và bắt đầu sau khi khối cuối cùng của
            kernel bên trong đã được điều phối.
          </p>
        }
      >
        <p>
          A stream priority is a hint about pending work. Once the internal kernel&rsquo;s blocks occupy every slot,
          nothing is taken from them. What the higher priority buys is the next slots: as each internal block finishes, a
          waiting boundary block takes its place instead of another internal block. In the explorer, with the internal
          kernel issued first, that is the difference between starting at the first block retirement and starting after
          the internal kernel&rsquo;s last block has been dispatched.
        </p>
      </Translated>
    ),
    source: 'p. 558; CUDA 13.5 runtime reference and programming guide, checked 2026-11-20 (notes/today/ch23-facts.md, A2); the slot picture is this page’s model',
    vi: { source: 'p. 558; tài liệu tham khảo runtime và hướng dẫn lập trình CUDA 13.5, kiểm tra 2026-11-20 (notes/today/ch23-facts.md, A2); hình ảnh các chỗ là mô hình của trang này' },
  },
  'ch23/23.4/q-hip-prio': {
    kind: 'choice',
    id: 'ch23/23.4/q-hip-prio',
    prompt: (
      <Translated
        vi={
          <>
            Trên ROCm 10.2, mã của bạn đọc <code>streamPrioritiesSupported</code> (1) và dải ưu tiên từ −1 đến 1, rồi tạo hai
            stream biên ở mức −1. Chúng có được ưu tiên hơn stream bên trong không?
          </>
        }
      >
        On ROCm 10.2 your code reads <code>streamPrioritiesSupported</code> (1) and a priority range of −1 to 1, then
        creates the two boundary streams at −1. Are they preferred over the internal stream?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không: một công tắc runtime, bật theo mặc định, coi mọi stream là độ ưu tiên bình thường, bất kể thuộc tính và dải giá trị nói gì.">
            No: a runtime switch, on by default, treats every stream as normal priority, whatever the property and the range say.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có: −1 là độ ưu tiên cao nhất trong dải đó, và thiết bị vừa nói rằng nó hỗ trợ độ ưu tiên stream.">
            Yes: −1 is the greatest priority in that range, and the device has just said that it supports stream priorities.
          </Translated>
        ),
        why: <Translated vi="Đó là điều các báo cáo nói. Công tắc runtime ghi đè chúng (B2).">That is what the reports say. The runtime switch overrides them (B2).</Translated>,
      },
      {
        label: (
          <Translated vi="Có, nhưng chỉ cho các lần sao chép: AMD áp dụng độ ưu tiên stream cho các lần truyền bộ nhớ trong một stream thay vì cho các lần khởi chạy kernel.">
            Yes, but only their copies: AMD applies stream priorities to the memory transfers in a stream rather than to its kernel launches.
          </Translated>
        ),
        why: <Translated vi="Không nguồn nào của AMD nói vậy, và công tắc khiến mọi stream có độ ưu tiên bình thường.">Nothing in AMD’s sources says so, and the switch makes every stream normal priority.</Translated>,
      },
      {
        label: (
          <Translated vi="Không: quy ước của HIP ngược với CUDA, nên −1 là độ ưu tiên thấp nhất trong dải nó báo.">
            No: HIP’s convention is the reverse of CUDA’s, so −1 is the least priority in the range it reports.
          </Translated>
        ),
        why: <Translated vi="HIP giữ quy ước của CUDA: “Lower numbers represent higher priorities.” (B2)">HIP keeps CUDA’s convention: “Lower numbers represent higher priorities.” (B2)</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Header của AMD giữ tên và quy ước của CUDA, đồng thời cảnh báo rằng độ ưu tiên stream bị tắt. Cờ runtime{' '}
            <code>DEBUG_HIP_IGNORE_STREAM_PRIORITY</code> mặc định là true và ép mọi hàng đợi về độ ưu tiên bình thường, trong
            khi <code>hipGetDeviceProperties</code> vẫn báo <code>streamPrioritiesSupported = 1</code> và dải giá trị trả về là
            −1 đến 1. Trên AMD, đừng xây một sự chồng lấp phụ thuộc vào độ ưu tiên, và đừng cố đo nó: hãy phát lệnh các kernel
            biên trước.
          </p>
        }
      >
        <p>
          AMD&rsquo;s header keeps CUDA&rsquo;s names and convention, and also warns that stream priority is disabled. The
          runtime flag <code>DEBUG_HIP_IGNORE_STREAM_PRIORITY</code> defaults to true and forces every queue to normal
          priority, while <code>hipGetDeviceProperties</code> still reports <code>streamPrioritiesSupported = 1</code> and
          the range comes back as −1 to 1. On AMD, do not build an overlap that depends on priorities, and do not try to
          measure one: issue the boundary kernels first.
        </p>
      </Translated>
    ),
    source: 'not in the book; ROCm 10.2 sources, checked 2026-11-20 (notes/today/ch23-facts.md, B2)',
    vi: { source: 'không có trong sách; mã nguồn ROCm 10.2, kiểm tra 2026-11-20 (notes/today/ch23-facts.md, B2)' },
  },
  'ch23/23.4/q-memset': {
    kind: 'choice',
    id: 'ch23/23.4/q-memset',
    prompt: (
      <Translated
        vi={
          <>
            Sách thay <code>cudaMemset</code> bằng <code>cudaMemsetAsync</code> trong <code>internalStream</code>, nói rằng{' '}
            <code>cudaMemset</code> là lời gọi chặn. Một lời gọi <code>cudaMemset(l2norm_d, 0, sizeof(float))</code> thông
            thường thực sự sẽ làm gì ở dòng 04 của vòng lặp này?
          </>
        }
      >
        The book replaces <code>cudaMemset</code> with <code>cudaMemsetAsync</code> in <code>internalStream</code>,
        saying <code>cudaMemset</code> is blocking. What would a plain <code>cudaMemset(l2norm_d, 0, sizeof(float))</code>{' '}
        actually do at line 04 of this loop?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Trả về ngay, nhưng xếp phép đặt lại vào default stream kiểu cũ (legacy), stream này hợp nhất với mọi stream chặn.">
            Return at once, but queue the reset in the legacy default stream, which joins every blocking stream.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chặn host cho đến khi phép đặt lại xong trên device, đúng là lý do sách đưa ra để thay nó.">
            Block the host until the reset has finished on the device, which is exactly the reason the book gives for replacing it.
          </Translated>
        ),
        why: <Translated vi="NVIDIA ghi trong tài liệu rằng cudaMemset trên bộ nhớ device là bất đồng bộ đối với host (A3).">NVIDIA documents cudaMemset on device memory as asynchronous with respect to the host (A3).</Translated>,
      },
      {
        label: (
          <Translated vi="Vẫn chạy trong internalStream, vì runtime đặt memset vào stream dùng cùng con trỏ gần nhất.">
            Run in internalStream anyway, since the runtime puts a memset into the stream that last used the same pointer.
          </Translated>
        ),
        why: <Translated vi="Không có quy tắc nào như vậy: các dạng đồng bộ được phát lệnh qua default stream (A3).">There is no such rule: the synchronous forms are issued through the default stream (A3).</Translated>,
      },
      {
        label: (
          <Translated vi="Báo lỗi, vì l2norm_d nằm trong bộ nhớ host được ghim, thứ chỉ dạng Async mới ghi được.">
            Fail with an error, because l2norm_d lives in pinned host memory, which only the Async form is able to write.
          </Translated>
        ),
        why: <Translated vi="l2norm_d là bộ nhớ device. Buffer được ghim là l2norm_h.">l2norm_d is device memory. The pinned buffer is l2norm_h.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Cách sửa của sách vẫn đúng, vì một lý do khác. Một <code>cudaMemset</code> thông thường không bắt host chờ, nhưng
            nó đi vào default stream kiểu cũ, và các stream tạo bằng <code>cudaStreamDefault</code> là stream chặn: legacy
            stream trước tiên chờ tất cả chúng rồi bắt tất cả chúng chờ nó. Mỗi vòng lặp sẽ bắt đầu bằng một lần hợp nhất trên
            toàn device. <code>cudaMemsetAsync</code> trong <code>internalStream</code> sắp thứ tự phép đặt lại với riêng stream
            đó, và event <code>resetL2</code> mang thứ tự đó sang hai stream còn lại.
          </p>
        }
      >
        <p>
          The book&rsquo;s fix stands, for another reason. A plain <code>cudaMemset</code> does not make the host wait,
          but it goes into the legacy default stream, and streams created with <code>cudaStreamDefault</code> are blocking
          streams: the legacy stream first waits for all of them and then makes all of them wait for it. Every iteration
          would begin with a device-wide join. <code>cudaMemsetAsync</code> in <code>internalStream</code> orders the reset
          with that one stream, and the <code>resetL2</code> event carries the order to the other two.
        </p>
      </Translated>
    ),
    source: 'p. 559; CUDA 13.5 runtime reference, checked 2026-11-20 (notes/today/ch23-facts.md, A3, A9)',
    vi: { source: 'p. 559; tài liệu tham khảo runtime CUDA 13.5, kiểm tra 2026-11-20 (notes/today/ch23-facts.md, A3, A9)' },
  },
  'ch23/23.4/q-mechanism': {
    kind: 'choice',
    id: 'ch23/23.4/q-mechanism',
    prompt: (
      <Translated
        vi={
          <>
            Trong Fig. 23.14, lần sao chép của chuẩn, xếp hàng trong <code>internalStream</code>, phải chờ kernel trên trong{' '}
            <code>topStream</code>. Thứ gì bảo đảm điều đó, và vì sao thứ tự các lời gọi của host là không đủ?
          </>
        }
      >
        In Fig. 23.14 the copy of the norm, queued in <code>internalStream</code>, must wait for the top kernel in{' '}
        <code>topStream</code>. What enforces it, and why is the order of the host&rsquo;s calls not enough?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một event được ghi trong topStream sau kernel và được chờ trong internalStream: thứ tự lời gọi chỉ được giữ bên trong một stream.">
            An event recorded in topStream after the kernel and waited for in internalStream: call order only holds within a stream.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không cần gì thêm: host khởi chạy kernel trên trước, và runtime giữ thứ tự lời gọi giữa các stream.">
            Nothing extra: the host launches the top kernel first, and the runtime keeps call order across streams.
          </Translated>
        ),
        why: <Translated vi="Runtime giữ thứ tự lời gọi bên trong một stream. Giữa các stream, công việc có thể chạy theo bất kỳ thứ tự nào (p. 556).">The runtime keeps call order within a stream. Across streams, work may run in any order (p. 556).</Translated>,
      },
      {
        label: (
          <Translated vi="Bản thân cudaMemcpyAsync chờ mọi công việc trước đó trên device, ở mọi stream, trước khi bắt đầu sao chép chuẩn.">
            cudaMemcpyAsync itself waits for all earlier work on the device, in every stream, before it starts copying the norm.
          </Translated>
        ),
        why: <Translated vi="Đó là hành vi của default stream kiểu cũ (A3), không phải của một lần sao chép xếp hàng trong một stream có tên.">That is the legacy default stream’s behaviour (A3), not that of a copy queued in a named stream.</Translated>,
      },
      {
        label: (
          <Translated vi="Các stream biên có độ ưu tiên cao hơn, nên bộ lập lịch cho kernel của chúng xong trước khi bắt đầu lần sao chép.">
            The boundary streams have the higher priority, so the scheduler finishes their kernels before it starts the copy.
          </Translated>
        ),
        why: <Translated vi="Độ ưu tiên là một gợi ý cho các kernel đang chờ. Nó không sắp thứ tự gì, và hoàn toàn không áp dụng cho lần sao chép (A2).">A priority is a hint for pending kernels. It orders nothing, and it does not apply to copies at all (A2).</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Thứ tự stream là thứ tự duy nhất runtime tự giữ, và nó dừng ở ranh giới của stream. Để bắt{' '}
            <code>internalStream</code> chờ một điểm trong <code>topStream</code>, Fig. 23.14 ghi <code>computeTop</code> sau
            kernel trên (dòng 11) và xếp một lần chờ nó trong <code>internalStream</code> (dòng 19). Một{' '}
            <code>cudaStreamSynchronize(topStream)</code> trên host cũng được, nhưng host sẽ ngồi không cho đến khi kernel trên
            xong.
          </p>
        }
      >
        <p>
          Stream order is the only order the runtime keeps by itself, and it stops at the stream&rsquo;s edge. To make{' '}
          <code>internalStream</code> wait for a point in <code>topStream</code>, Fig. 23.14 records{' '}
          <code>computeTop</code> after the top kernel (line 11) and queues a wait for it in <code>internalStream</code>{' '}
          (line 19). A <code>cudaStreamSynchronize(topStream)</code> on the host would also work, but the host would sit
          idle until the top kernel finished.
        </p>
      </Translated>
    ),
    source: 'pp. 559–560; CUDA 13.5 runtime reference (notes/today/ch23-facts.md, A5)',
    vi: { source: 'pp. 559–560; tài liệu tham khảo runtime CUDA 13.5 (notes/today/ch23-facts.md, A5)' },
  },
  'ch23/23.4/q-metal-events': {
    kind: 'choice',
    id: 'ch23/23.4/q-metal-events',
    prompt: (
      <Translated
        vi={
          <>
            Bạn chuyển Fig. 23.14 sang Metal. Dòng 19 và 20 bắt lần sao chép chờ hai nguồn tạo, mỗi pass biên một nguồn. Một{' '}
            <code>MTLEvent</code> mà cả hai pass biên cùng báo hiệu có làm được việc đó không?
          </>
        }
      >
        You port Fig. 23.14 to Metal. Lines 19 and 20 make the copy wait for two producers, one per boundary pass. Can
        one <code>MTLEvent</code> that both boundary passes signal do that job?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không: một event giữ một giá trị chỉ tăng, nên hai nguồn tạo không thể gộp tín hiệu; hãy dùng mỗi nguồn một event.">
            No: an event holds one value that only grows, so two producers cannot combine signals; use one event each.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có: Metal đếm số tín hiệu một event nhận được, như một semaphore, nên lần sao chép có thể chờ đến khi số đếm đạt hai.">
            Yes: Metal counts the signals an event receives, like a semaphore, so the copy can wait until the count reaches two.
          </Translated>
        ),
        why: <Translated vi="Apple nói điều ngược lại: tín hiệu của một event là “unlike a semaphore” (C2).">Apple says the opposite: an event’s signal is “unlike a semaphore” (C2).</Translated>,
      },
      {
        label: (
          <Translated vi="Có, nhưng chỉ với MTLSharedEvent, loại thêm bộ đếm tín hiệu mà MTLEvent thường không có.">
            Yes, but only an MTLSharedEvent, which adds the signal counter that a plain MTLEvent does not have.
          </Translated>
        ),
        why: <Translated vi="MTLSharedEvent thêm việc chờ từ CPU, từ thiết bị khác và tiến trình khác, không phải việc đếm (C2).">MTLSharedEvent adds waits from the CPU, other devices and other processes, not counting (C2).</Translated>,
      },
      {
        label: (
          <Translated vi="Không: một command buffer không thể chờ công việc ở một hàng đợi Metal khác, nên host phải chờ cả hai pass.">
            No: a command buffer cannot wait for work in another Metal queue, so the host must wait for both passes.
          </Translated>
        ),
        why: <Translated vi="Nó có thể: một command buffer có thể chờ một giá trị event mà một buffer ở hàng đợi khác báo hiệu (C2).">It can: a command buffer can wait for an event value that a buffer in another queue signals (C2).</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một event Metal là một giá trị 64-bit chỉ tăng, và một lần chờ được thông qua khi giá trị đạt đến con số nó chờ.
            Nếu hai pass cùng báo hiệu một event, bên chờ không thể biết một hay cả hai đã báo. Lời khuyên của Apple là mỗi khối
            công việc tạo một event; hai giá trị được báo hiệu theo thứ tự đã biết cũng được. Một event CUDA lại khác nữa, là
            một ảnh chụp của một stream, nên Fig. 23.14 cũng cần hai event, vì một lý do khác: các nguồn tạo nằm ở hai stream.
          </p>
        }
      >
        <p>
          A Metal event is a 64-bit value that only grows, and a wait passes once the value reaches the number it waits
          for. If two passes signal the same event, the waiter cannot tell whether one or both have done so. Apple&rsquo;s
          advice is one event per producing workload; two values signalled in a known order also work. A CUDA event is different again, a snapshot of one stream, so Fig.
          23.14 also needs two of them, for a different reason: the producers are in two streams.
        </p>
      </Translated>
    ),
    source: 'not in the book; Apple Metal documentation, checked 2026-11-20 (notes/today/ch23-facts.md, C2, A5)',
    vi: { source: 'không có trong sách; tài liệu Metal của Apple, kiểm tra 2026-11-20 (notes/today/ch23-facts.md, C2, A5)' },
  },
  'ch23/23.4/q-drop-reset': {
    kind: 'choice',
    id: 'ch23/23.4/q-drop-reset',
    prompt: (
      <Translated
        vi={
          <>
            Ai đó xóa hai lời gọi <code>cudaStreamWaitEvent</code> trên <code>resetL2</code> (dòng 06–07). Điều gì có thể
            sai, và vì sao một test trên trình giả lập (emulator) của khóa học vẫn qua?
          </>
        }
      >
        Someone deletes the two <code>cudaStreamWaitEvent</code> calls on <code>resetL2</code> (lines 06–07). What can go
        wrong, and why does a test on the course&rsquo;s emulator still pass?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một kernel biên có thể cộng vào chuẩn trước khi memset đặt nó về 0; trình giả lập chạy các stream từng lời gọi một.">
            A boundary kernel may add to the norm before the memset zeroes it; the emulator runs streams one call at a time.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chương trình bị deadlock, vì topStream giờ chờ một event không bao giờ được ghi; trình giả lập không mô hình hóa việc chờ.">
            The program deadlocks, because topStream now waits for an event that is never recorded; the emulator does not model waits.
          </Translated>
        ),
        why: <Translated vi="Bỏ một lần chờ không thể tạo ra một lần chờ. Trình giả lập có bản thay thế cho event (labs/README.md).">Removing a wait cannot create one. The emulator does have stand-ins for events (labs/README.md).</Translated>,
      },
      {
        label: (
          <Translated vi="Không có gì: memset và cả hai kernel biên nằm trong cùng một stream, nên chúng vẫn đúng thứ tự khi không có các lần chờ.">
            Nothing: the memset and both boundary kernels are in the same stream, so they stay in order without the waits.
          </Translated>
        ),
        why: <Translated vi="Memset nằm trong internalStream (dòng 04); các kernel biên nằm trong topStream và bottomStream.">The memset is in internalStream (line 04); the boundary kernels are in topStream and bottomStream.</Translated>,
      },
      {
        label: (
          <Translated vi="Các vùng biên bị gửi hai lần mỗi vòng lặp; trình giả lập không chạy MPI, nên nó không thấy được phép trao đổi thứ hai.">
            The halos are sent twice per iteration; the emulator does not run MPI, so it cannot see the second exchange.
          </Translated>
        ),
        why: <Translated vi="Các phép trao đổi là lời gọi phía host ở dòng 28 đến 34. Các lần chờ đặt lại không liên quan gì đến chúng.">The exchanges are host calls at lines 28 to 34. The reset waits have nothing to do with them.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Không có các lần chờ, không gì sắp thứ tự phép đặt lại trong <code>internalStream</code> so với các kernel ở hai
            stream kia. Nếu một khối biên xong và thực hiện <code>atomicAdd</code> trước khi memset chạy, memset xóa mất phần
            đóng góp đó, chuẩn ra quá nhỏ, và vòng lặp có thể dừng trước khi hội tụ. Việc nó có xảy ra hay không là chuyện thời
            điểm. Trong trình khám phá nó không bao giờ phát tác, vì host vừa làm rỗng <code>internalStream</code> và memset là
            thứ đầu tiên trong đó; trong vòng lặp NCCL của sách, nơi stream đó có thể vẫn đang chờ một phép trao đổi, cùng mô
            hình đó khiến nó phát tác. Trình giả lập chạy các stream từng lời gọi một, nên race condition này hoàn toàn không
            thể xảy ra ở đó.
          </p>
        }
      >
        <p>
          Without the waits nothing orders the reset in <code>internalStream</code> against the kernels in the other two
          streams. If a boundary block finishes and does its <code>atomicAdd</code> before the memset runs, the memset wipes
          that contribution, the norm comes out too small, and the loop can stop before it has converged. Whether it
          happens is a matter of timing. In the explorer it never fires, because the host has just emptied{' '}
          <code>internalStream</code> and the memset is the first thing in it; in the book&rsquo;s NCCL loop, where that
          stream can still be waiting on an exchange, the same model makes it fire. The emulator runs streams one call at
          a time, so the race cannot occur there at all.
        </p>
      </Translated>
    ),
    source: 'pp. 559–560; labs/README.md (emulator streams are synchronous); this page’s model',
    vi: { source: 'pp. 559–560; labs/README.md (stream của trình giả lập là đồng bộ); mô hình của trang này' },
  },
  'ch23/23.4/q-tp': {
    kind: 'choice',
    id: 'ch23/23.4/q-tp',
    prompt: (
      <Translated
        vi={
          <>
            Một máy chủ suy luận chia trọng số của mỗi lớp cho 4 GPU (song song tensor) thay vì giao cho mỗi GPU một phần tư số
            lớp (song song pipeline). Cách nào đặt một collective lên đường găng của mọi lớp, và điều đó đòi hỏi gì ở phần cứng?
          </>
        }
      >
        An inference server splits each layer&rsquo;s weights across 4 GPUs (tensor parallelism) instead of giving each
        GPU a quarter of the layers (pipeline parallelism). Which one puts a collective on every layer&rsquo;s critical
        path, and what does that ask of the hardware?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Song song tensor: mọi lớp kết thúc bằng một all-reduce mà mọi GPU phải chờ, nên nó cần đường kết nối nhanh nhất.">
            Tensor parallelism: every layer ends in an all-reduce all GPUs wait for, so it wants the fastest link.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Song song pipeline: ở mọi lớp, mỗi giai đoạn phải chờ tất cả các giai đoạn khác, nên nó cần đường kết nối nhanh nhất.">
            Pipeline parallelism: at every layer each stage has to wait for all the other stages, so it needs the fastest link.
          </Translated>
        ),
        why: <Translated vi="Một giai đoạn gửi activation một lần, ở ranh giới của nó, sang giai đoạn tiếp theo. Nó không chờ all-reduce nào.">A stage sends activations once, at its boundary, to the next stage. It waits for no all-reduce.</Translated>,
      },
      {
        label: (
          <Translated vi="Không cách nào: với NCCL cả hai đều chồng lấp hoàn toàn giao tiếp với tính toán, nên tốc độ đường kết nối gần như không quan trọng.">
            Neither: with NCCL both kinds overlap the communication fully with computation, so the link speed hardly matters.
          </Translated>
        ),
        why: <Translated vi="Kết quả của all-reduce là đầu vào của lớp tiếp theo. Không còn gì độc lập để chồng lấp với nó.">The all-reduce’s result is the next layer’s input. Nothing independent is left to overlap it with.</Translated>,
      },
      {
        label: (
          <Translated vi="Song song tensor, nhưng chỉ trong prefill: khi decode, all-reduce được bỏ qua vì một token là quá nhỏ.">
            Tensor parallelism, but only during prefill: in decoding the all-reduce is skipped because one token is too small.
          </Translated>
        ),
        why: <Translated vi="Đầu ra từng phần của mọi lớp phải được cộng lại cho mọi token, kể cả khi decode.">Every layer’s partial outputs must be summed for every token, decoding included.</Translated>,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong song song tensor, mỗi GPU tính một phần của mọi lớp, và các phần phải được cộng lại trước khi lớp tiếp theo có
            thể bắt đầu. Đó là phép trao đổi vùng biên của chương này mà không có các hàng bên trong để giấu sau. Song song
            pipeline gửi activation một lần mỗi ranh giới giai đoạn và có thể giấu lần truyền đó sau các micro-batch khác, với
            cái giá là một bong bóng trong lúc pipeline được nạp và rút.
          </p>
        }
      >
        <p>
          In tensor parallelism each GPU computes part of every layer, and the parts must be summed before the next layer
          can start. That is this chapter&rsquo;s halo exchange without the internal rows to hide behind. Pipeline
          parallelism sends activations once per stage boundary and can hide that transfer behind other micro-batches, at
          the cost of a bubble while the pipeline fills and drains.
        </p>
      </Translated>
    ),
    source: 'not in the book; ours, on llama.cpp and vLLM sources and docs read 2026-11-20 (notes/today/ch23-facts.md, D2, D5, D7)',
    vi: { source: 'không có trong sách; của khóa học này, dựa trên mã nguồn và tài liệu của llama.cpp và vLLM đọc ngày 2026-11-20 (notes/today/ch23-facts.md, D2, D5, D7)' },
  },
};

// ---------------------------------------------------------------- the page
export function S234() {
  const it = STREAMS_DEF.iters[0];
  return (
    <>
      <Translated
        vi={
          <p>
            Phần <a href={href('ch23', '23.3')}>23.3</a> đã cho bạn các bộ phận: stream, lần sao chép bất đồng bộ, bộ nhớ ghim
            (pinned memory). Trang này lắp chúng thành vòng lặp của sách và đặt câu hỏi mà mọi chương trình có nhiều hơn một
            stream phải trả lời: khi một thao tác phải chờ một thao tác khác, ai là người chờ? Host, một stream, hay không ai cả.
            Câu trả lời cũng là thứ phân biệt ba thư viện của chương.
          </p>
        }
      >
        <p>
          Section <a href={href('ch23', '23.3')}>23.3</a> gave you the parts: streams, asynchronous copies, pinned memory.
          This page assembles them into the book&rsquo;s loop and asks the question every program with more than one
          stream has to answer: when one operation must wait for another, who does the waiting? The host, a stream, or
          nobody. The answer is also what separates the chapter&rsquo;s three libraries.
        </p>
      </Translated>
      <Source section="23.3" pages="556–563" figs="Figs. 23.14, 23.15" />

      <h2>
        <Translated vi="Một kernel, ba lần khởi chạy">One kernel, three launches</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Fig. 23.14 giữ kernel của Fig. 23.1 và khởi chạy nó ba lần mỗi vòng lặp, mỗi lần trên một phần các hàng và vào
            stream riêng: hàng rìa trên vào <code>topStream</code>, hàng rìa dưới vào <code>bottomStream</code>, các hàng bên
            trong vào <code>internalStream</code>. Mỗi lần khởi chạy nhận một con trỏ vào cùng các mảng và một số hàng, và điều
            kiện chặn của kernel lo phần còn lại. Bạn đã gặp phép tính này trong trình khám phá ở{' '}
            <a href={href('ch23', 'start')}>trang đầu tiên</a>; đây là nó dưới dạng một bài kiểm tra.
          </p>
        }
      >
        <p>
          Fig. 23.14 keeps Fig. 23.1&rsquo;s kernel and launches it three times per iteration, each time on a slice of the
          rows and into its own stream: the top boundary row into <code>topStream</code>, the bottom one into{' '}
          <code>bottomStream</code>, the internal rows into <code>internalStream</code>. Each launch gets a pointer into the
          same arrays and a row count, and the kernel&rsquo;s guard does the rest. You met the arithmetic in the explorer
          on the <a href={href('ch23', 'start')}>first page</a>; here it is as a check.
        </p>
      </Translated>
      <Source section="23.3" pages="557–558" figs="Fig. 23.14, lines 10–16" />
      <Ask q={questionsS234['ch23/23.4/q-threerows']} />
      <Translated
        vi={
          <p>
            Vậy ba lần khởi chạy ghi hàng 1, hàng <code>ny − 2</code> và các hàng 2 đến <code>ny − 3</code>: mỗi hàng bên trong
            đúng một lần. Sách nói mỗi lần khởi chạy cần những hàng nào làm đầu vào; còn mỗi lần ghi những hàng nào là phép suy
            ra của chúng tôi từ điều kiện chặn.
          </p>
        }
      >
        <p>
          So the three launches write row 1, row <code>ny − 2</code> and rows 2 to <code>ny − 3</code>: every interior row
          exactly once. The book says which rows each launch needs as input; which rows each one writes is our derivation
          from the guard.
        </p>
      </Translated>

      <h2>
        <Translated vi="Độ ưu tiên: một gợi ý, không phải một mệnh lệnh">Priorities: a hint, not an order</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Các kernel ở các stream khác nhau có thể chạy theo bất kỳ thứ tự nào. Sách kỳ vọng các kernel biên chạy trước, vì
            chúng được khởi chạy trước, và lưu ý rằng không gì bảo đảm điều đó. Nếu kernel bên trong bắt đầu trước và các khối
            của nó chiếm cả GPU, các kernel biên phải chờ, các phép trao đổi chờ chúng, và sự chồng lấp biến mất. Cách chữa của
            sách là cho hai stream biên độ ưu tiên cao hơn. Mã thiết lập bên cạnh câu đố bên dưới tạo các stream theo cách đó.
          </p>
        }
      >
        <p>
          Kernels in different streams may run in any order. The book expects the boundary kernels to run first, since
          they are launched first, and notes that nothing guarantees it. If the internal kernel starts first and its blocks
          occupy the whole GPU, the boundary kernels wait, the exchanges wait for them, and the overlap is gone. The
          book&rsquo;s remedy is to give the two boundary streams a higher priority. The setup code beside the puzzle below
          creates the streams that way.
        </p>
      </Translated>
      <Source section="23.3" pages="558" />
      <Translated
        vi={
          <p>
            Sách để ngỏ hai điều. Thứ nhất là chiều. Trên CUDA, số nhỏ hơn nghĩa là độ ưu tiên cao hơn, và mặc định là 0.{' '}
            <code>cudaDeviceGetStreamPriorityRange(&amp;low, &amp;high)</code> trả về độ ưu tiên thấp nhất trước và cao nhất
            sau, nên <code>low, high</code> của sách là đúng như được in. Bản thân dải giá trị thuộc về thiết bị; hãy đọc nó,
            đừng giả định.
          </p>
        }
      >
        <p>
          Two things the book leaves open. The first is the direction. On CUDA a lower number means a higher priority, and
          the default is 0. <code>cudaDeviceGetStreamPriorityRange(&amp;low, &amp;high)</code> returns the least priority
          first and the greatest second, so the book&rsquo;s <code>low, high</code> is right as printed. The range itself
          belongs to the device; read it, do not assume it.
        </p>
      </Translated>
      <Today checked="2026-11-20, CUDA 13.5 (ch23-facts A1)">{/* i18n-ignore: date and citation */}
        <Translated vi={<p>&ldquo;Lower numbers imply greater priorities.&rdquo; Cả hai đều là 0 khi không có độ ưu tiên.</p>}>
          <p>&ldquo;Lower numbers imply greater priorities.&rdquo; Both are 0 without priorities.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Thứ hai là độ ưu tiên làm gì. NVIDIA gọi nó là một gợi ý: nó áp dụng cho các kernel còn đang chờ bắt đầu, không
            chiếm quyền công việc đang chạy, không bảo đảm thứ tự nào, và không ảnh hưởng gì đến lần sao chép. Trong tình huống nguy
            hiểm sách mô tả, khi kernel bên trong đã lấp đầy GPU, độ ưu tiên cao hơn không thể tạo chỗ. Nó chỉ cho các kernel
            biên quyền lấy trước những chỗ được giải phóng khi các khối của kernel bên trong xong. Vậy hãy giữ cả hai biện pháp
            của sách, thứ tự khởi chạy và độ ưu tiên, rồi đo: không biện pháp nào là một sự bảo đảm.
          </p>
        }
      >
        <p>
          The second is what a priority does. NVIDIA calls it a hint: it applies to kernels still waiting to start, it
          does not preempt work that is already running, it guarantees no order, and it has no effect on copies. In the
          hazard the book describes, where the internal kernel already fills the GPU, a higher priority cannot make room.
          It only gives the boundary kernels first claim on the slots that free up as the internal kernel&rsquo;s blocks
          finish. So keep both of the book&rsquo;s measures, the launch order and the priority, and then measure: neither
          is a guarantee.
        </p>
      </Translated>
      <Today checked="2026-11-20, CUDA 13.5 (ch23-facts A2)">{/* i18n-ignore: date and citation */}
        <Translated vi={<p>Độ ưu tiên &ldquo;do not preempt already-running work&rdquo; và không áp dụng cho lần sao chép.</p>}>
          <p>Priorities &ldquo;do not preempt already-running work&rdquo; and do not apply to copies.</p>
        </Translated>
      </Today>
      <Ask q={questionsS234['ch23/23.4/q-prio']} />
      <Translated
        vi={
          <p>
            Trên AMD các lời gọi giữ tên (<code>hipDeviceGetStreamPriorityRange</code>,{' '}
            <code>hipStreamCreateWithPriority</code>) và quy ước, nhưng ROCm 10.2 mặc định bỏ qua độ ưu tiên: một công tắc
            runtime coi mọi stream là độ ưu tiên bình thường, trong khi thiết bị vẫn báo rằng nó hỗ trợ chúng. Metal hoàn toàn
            không có độ ưu tiên cho hàng đợi tính toán; độ ưu tiên duy nhất trong các header của nó thuộc về hàng đợi nạp dữ
            liệu tệp vào tài nguyên.
          </p>
        }
      >
        <p>
          On AMD the calls keep their names (<code>hipDeviceGetStreamPriorityRange</code>,{' '}
          <code>hipStreamCreateWithPriority</code>) and the convention, but ROCm 10.2 ignores priorities by default: a
          runtime switch treats every stream as normal priority, while the device still reports that it supports them.
          Metal has no priority for compute queues at all; the only priority in its headers belongs to the queue that
          loads file data into resources.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against ROCm 10.2 sources and the macOS 27 SDK headers (notes/today/ch23-facts.md, B2, C3)', '2026-11-20, đối chiếu với mã nguồn ROCm 10.2 và header của macOS 27 SDK (notes/today/ch23-facts.md, B2, C3)')}>
        <Translated
          vi={
            <p>
              HIP: <code>DEBUG_HIP_IGNORE_STREAM_PRIORITY</code> mặc định là true. Metal: <code>MTLIOCommandQueue</code> có độ
              ưu tiên; <code>MTLCommandQueue</code> thì không.
            </p>
          }
        >
          <p>
            HIP: <code>DEBUG_HIP_IGNORE_STREAM_PRIORITY</code> defaults to true. Metal: <code>MTLIOCommandQueue</code> has
            a priority; <code>MTLCommandQueue</code> has none.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS234['ch23/23.4/q-hip-prio']} />

      <h2>
        <Translated vi="Một chuẩn, ba stream: event">One norm, three streams: events</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Ba kernel dùng chung một thứ: chuẩn. Mỗi kernel cộng bình phương phần dư của các khối của nó vào{' '}
            <code>l2norm_d</code> bằng một phép toán nguyên tử (atomic). Điều đó tạo ra hai thứ tự cắt ngang các stream. Phép
            đặt lại phải xảy ra trước khi bất kỳ kernel nào trong ba kernel cộng vào chuẩn, và cả ba phải xong trước khi chuẩn
            được sao chép về host. Nhưng phép đặt lại và lần sao chép mỗi thứ chỉ có thể nằm trong một stream.
          </p>
        }
      >
        <p>
          The three kernels share one thing: the norm. Each adds its blocks&rsquo; squared residues to{' '}
          <code>l2norm_d</code> with an atomic. That creates two orderings that cross streams. The reset must happen before
          any of the three kernels adds to the norm, and all three must have finished before the norm is copied to the
          host. But the reset and the copy can each sit in only one stream.
        </p>
      </Translated>
      <Source section="23.3" pages="559" />
      <Translated
        vi={
          <p>
            Thứ tự stream không diễn đạt được điều đó, và một lần đồng bộ hóa trên host chỉ làm được với cái giá là chặn host.
            Câu trả lời của CUDA là event: một mốc bạn ghi trong một stream mà một stream khác có thể được bảo chờ.{' '}
            <code>cudaEventRecord(e, s)</code> đặt nó; <code>cudaStreamWaitEvent(s2, e, 0)</code> bắt mọi thứ xếp hàng trong{' '}
            <code>s2</code> sau đó chờ đến khi <code>s</code> đạt tới mốc. Không luồng host nào phải chờ. Tạo event với{' '}
            <code>cudaEventDisableTiming</code> nói rằng nó sẽ không bao giờ đo thời gian gì, điều làm nó rẻ hơn cho đúng
            mục đích này.
          </p>
        }
      >
        <p>
          Stream order cannot express that, and a host synchronisation could only at the price of stopping the host.
          CUDA&rsquo;s answer is the event: a marker you record in one stream that another stream can be told to wait for.{' '}
          <code>cudaEventRecord(e, s)</code> places it; <code>cudaStreamWaitEvent(s2, e, 0)</code> makes everything queued
          in <code>s2</code> afterwards wait until <code>s</code> reaches the marker. No host thread waits. Creating the
          event with <code>cudaEventDisableTiming</code> says it will never time anything, which makes it cheaper for
          exactly this use.
        </p>
      </Translated>
      <Source section="23.3" pages="559–560" />
      <Translated
        vi={
          <p>
            Có một chi tiết quan trọng bên trong vòng lặp. Một event không phải là một bộ đếm. Ghi nó là chụp lại những gì
            stream của nó đang chứa lúc đó, và ghi lại lần nữa thì thay ảnh chụp đó. Một lần chờ dùng bất cứ thứ gì event đang
            giữ khi lần chờ được xếp hàng, và những lần ghi sau không thay đổi nó. Đó là lý do Fig. 23.14 có thể ghi cùng ba
            event trong mọi vòng lặp: các lần chờ của mỗi vòng lặp tham chiếu đến các lần ghi của chính vòng lặp đó.
          </p>
        }
      >
        <p>
          One detail matters inside a loop. An event is not a counter. Recording it captures what its stream holds at
          that moment, and recording it again replaces the capture. A wait uses whatever the event held when the wait was
          queued, and later records do not change it. That is why Fig. 23.14 can record the same three events in every
          iteration: each iteration&rsquo;s waits refer to that iteration&rsquo;s records.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the CUDA 13.5 runtime reference (notes/today/ch23-facts.md, A5)', '2026-11-20, đối chiếu với tài liệu tham khảo runtime CUDA 13.5 (notes/today/ch23-facts.md, A5)')}>
        <Translated
          vi={
            <p>
              Các lần chờ &ldquo;use the most recent captured state at the time of the API call, and are not affected by
              later calls to cudaEventRecord()&rdquo;. Sách gọi <code>cudaEventDisableTiming</code> là một hằng số lúc biên
              dịch; thực ra nó là một giá trị cờ. HIP có cùng tên và cờ.
            </p>
          }
        >
          <p>
            Waits &ldquo;use the most recent captured state at the time of the API call, and are not affected by later
            calls to cudaEventRecord()&rdquo;. The book calls <code>cudaEventDisableTiming</code> a compile-time constant;
            it is a flag value. HIP has the same names and flags.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Sách cũng đổi phép đặt lại từ <code>cudaMemset</code> sang <code>cudaMemsetAsync</code> trong{' '}
            <code>internalStream</code>, nói rằng cái đầu là lời gọi chặn. Lý do đó sai: NVIDIA ghi trong tài liệu rằng{' '}
            <code>cudaMemset</code> trên bộ nhớ device là bất đồng bộ đối với host. Thay đổi vẫn đúng vì một lý do khác. Một{' '}
            <code>cudaMemset</code> thông thường đi vào default stream kiểu cũ, stream này chờ mọi stream chặn và bắt tất cả
            chúng chờ nó, và các stream của sách là stream chặn. Phép đặt lại sẽ hợp nhất toàn bộ device mỗi vòng lặp một lần.
          </p>
        }
      >
        <p>
          The book also turns the reset from <code>cudaMemset</code> into <code>cudaMemsetAsync</code> in{' '}
          <code>internalStream</code>, saying the first is blocking. That reason is wrong: NVIDIA documents{' '}
          <code>cudaMemset</code> on device memory as asynchronous with respect to the host. The change is right for
          another reason. A plain <code>cudaMemset</code> goes into the legacy default stream, which waits for every
          blocking stream and makes them all wait for it, and the book&rsquo;s streams are blocking streams. The reset
          would join the whole device once per iteration.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the CUDA 13.5 runtime reference (notes/today/ch23-facts.md, A3, B4)', '2026-11-20, đối chiếu với tài liệu tham khảo runtime CUDA 13.5 (notes/today/ch23-facts.md, A3, B4)')}>
        <Translated
          vi={
            <p>
              &ldquo;The cudaMemset functions are asynchronous with respect to the host except when the target memory is
              pinned host memory.&rdquo; AMD không ghi trong tài liệu liệu <code>hipMemset</code> thông thường có bắt host
              chờ hay không.
            </p>
          }
        >
          <p>
            &ldquo;The cudaMemset functions are asynchronous with respect to the host except when the target memory is
            pinned host memory.&rdquo; AMD does not document whether plain <code>hipMemset</code> makes the host wait.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS234['ch23/23.4/q-memset']} />
      <Translated
        vi={
          <p>
            Giờ hãy tháo rời vòng lặp. Một vòng lặp cần mười thứ tự dưới đây. Với mỗi thứ tự, chọn thứ bảo đảm nó trong Fig.
            23.14: hai thao tác nằm cùng một stream, một event cùng một lần chờ của stream, host chờ một stream, hay một lời
            gọi chỉ trả về khi công việc của nó xong. Phản hồi sẽ dẫn dòng mã. Khi đã đặt đủ cả mười, một bảng cho thấy cùng
            các thứ tự đó trong vòng lặp NCCL và NVSHMEM của sách.
          </p>
        }
      >
        <p>
          Now take the loop apart. One iteration needs the ten orderings below. For each, pick what enforces it in Fig.
          23.14: the two operations being in the same stream, an event and a stream wait, the host waiting for a stream,
          or a call that returns only when its work is done. The feedback cites the line. Once all ten are placed, a table
          shows the same orderings in the book&rsquo;s NCCL and NVSHMEM loops.
        </p>
      </Translated>
      <DependencyPuzzle />
      <Translated
        vi={
          <p>
            Hãy xem các câu trả lời có điểm gì chung. Trong Fig. 23.14 công việc được sắp thứ tự ở một trong ba nơi: bên trong
            một stream, giữa các stream qua một event, hoặc trên host, và một lời gọi chặn chính là host chờ dưới một cái tên
            khác. Vòng lặp NCCL và NVSHMEM chuyển các thứ tự ra khỏi host: vào thứ tự stream, vào việc chờ event, vào một rào
            chắn xếp hàng trong stream, hoặc vào chính thứ tự thao tác bên trong kernel. Đó là thứ giải phóng host và loại bỏ
            thời gian rảnh mà bạn sắp thấy.
          </p>
        }
      >
        <p>
          Look at what the answers have in common. In Fig. 23.14 work is ordered in one of three places: inside one
          stream, between streams through an event, or on the host, and a blocking call is the host waiting by another
          name. The NCCL and NVSHMEM loops move orderings off the host: into stream order, into event waits, into a
          barrier queued in a stream, or into the kernel&rsquo;s own order of operations. That is what frees the host and
          removes the idle time you are about to see.
        </p>
      </Translated>
      <Ask q={questionsS234['ch23/23.4/q-mechanism']} />
      <Translated
        vi={
          <p>
            Event của Metal hoạt động khác, và khác biệt lộ ra đúng ở khuôn mẫu này. Một <code>MTLEvent</code> là một giá trị
            64-bit chỉ tăng. Một command buffer báo hiệu một giá trị và một command buffer khác chờ một giá trị, giữa các pass,
            không bao giờ bên trong một pass. Nhiều nguồn tạo không thể gộp tín hiệu vào một event, nên dòng 19 và 20 trở thành
            hai event, hoặc hai giá trị được báo hiệu theo thứ tự đã biết. Các lần chờ của Metal cũng có thể hết hạn, điều mà
            CUDA không có, và <code>MTLSharedEvent</code> thêm việc chờ từ CPU.
          </p>
        }
      >
        <p>
          Metal&rsquo;s events work differently, and the difference shows in exactly this pattern. An{' '}
          <code>MTLEvent</code> is a 64-bit value that only increases. A command buffer signals a value and another waits
          for a value, between passes, never inside one. Several producers cannot combine their signals into one event,
          so lines 19 and 20 become two events, or two values signalled in a known order. Metal&rsquo;s waits can also
          time out, which CUDA&rsquo;s cannot, and <code>MTLSharedEvent</code> adds waits from the CPU.
        </p>
      </Translated>
      <Today checked={tr("2026-11-20 against Apple's Metal documentation and the macOS 27 SDK headers (notes/today/ch23-facts.md, C2)", '2026-11-20, đối chiếu với tài liệu Metal của Apple và header của macOS 27 SDK (notes/today/ch23-facts.md, C2)')}>
        <Translated
          vi={
            <p>
              Apple viết: &ldquo;Multiple producing workloads can&rsquo;t combine their signals with one event to unblock any
              dependent workloads.&rdquo; (nhiều khối công việc tạo tín hiệu không thể gộp tín hiệu của chúng vào một event để mở khóa các khối công việc phụ thuộc.)
            </p>
          }
        >
          <p>
            &ldquo;Multiple producing workloads can&rsquo;t combine their signals with one event to unblock any dependent
            workloads.&rdquo;
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS234['ch23/23.4/q-metal-events']} />

      <h2>
        <Translated vi="Sao chép sớm, chờ một lần">Copy early, wait once</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Dòng 21 xếp hàng lần sao chép của chuẩn về host ngay sau các lần chờ, trước phép trao đổi vùng biên. Vì{' '}
            <code>cudaMemcpyAsync</code> trả về ngay, lần sao chép có thể chạy trong khi các phép trao đổi đang diễn ra, ngay khi
            kernel bên trong xong. Host không được đọc <code>l2norm_h</code> trước khi lần sao chép tới nơi, và đó là việc của dòng
            37: một lần chờ trên <code>internalStream</code>, ngay trước <code>MPI_Allreduce</code>.
          </p>
        }
      >
        <p>
          Line 21 queues the copy of the norm to the host straight after the waits, before the halo exchange. Because{' '}
          <code>cudaMemcpyAsync</code> returns at once, the copy can run while the exchanges are in flight, as soon as the
          internal kernel is done. The host must not read <code>l2norm_h</code> before the copy has landed, and that is
          line 37&rsquo;s job: one wait on <code>internalStream</code>, just before <code>MPI_Allreduce</code>.
        </p>
      </Translated>
      <Source section="23.3" pages="560–561" />
      <Translated
        vi={
          <p>
            Điều đó chỉ hoạt động nếu <code>l2norm_h</code> được ghim. Phần <a href={href('ch23', '23.3')}>23.3</a> đã cho
            thấy vì sao: từ bộ nhớ phân trang, runtime trung chuyển lần sao chép qua một buffer ghim của riêng nó, với host tham gia,
            và lời gọi thôi bất đồng bộ. Trong vòng lặp này, điều đó sẽ chặn host ở dòng 21 cho đến khi kernel bên trong xong,
            và cả hai phép trao đổi chỉ bắt đầu sau đó. Trình khám phá bên dưới cho bạn thử điều đó.
          </p>
        }
      >
        <p>
          It only works if <code>l2norm_h</code> is pinned. Section <a href={href('ch23', '23.3')}>23.3</a> showed why: from
          pageable memory the runtime stages the copy through a pinned buffer of its own, with the host taking part, and
          the call stops being asynchronous. In this loop that would stop the host at line 21 until the internal kernel
          had finished, and both exchanges would start only after it. The explorer below lets you try that.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the CUDA 13.5 programming guide (notes/today/ch23-facts.md, A6)', '2026-11-20, đối chiếu với hướng dẫn lập trình CUDA 13.5 (notes/today/ch23-facts.md, A6)')}>
        <Translated
          vi={
            <p>
              Không phải lỗi: <code>cudaMemcpyAsync</code> &ldquo;will function correctly&rdquo; với bộ nhớ phân trang,
              &ldquo;but it will revert to a synchronous behavior which will not overlap with other work.&rdquo;
            </p>
          }
        >
          <p>
            Not an error: <code>cudaMemcpyAsync</code> &ldquo;will function correctly&rdquo; with pageable memory, &ldquo;but
            it will revert to a synchronous behavior which will not overlap with other work.&rdquo;
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Một vòng lặp theo thời gian">One iteration in time</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Fig. 23.15 vẽ, cho một vòng lặp, lúc host đưa từng thao tác vào và lúc một stream chạy nó. Cả hai đều là chuyện thời
            gian, nên ở đây chúng được làm thành một hoạt cảnh. Mọi thanh đều đến từ cùng mô hình với trình khám phá phía sau,
            với các khoảng thời gian mà trình khám phá bắt đầu: kernel bên trong {DEFAULTS.internal} đơn vị, các kernel biên{' '}
            {DEFAULTS.boundary}, các phép trao đổi {DEFAULTS.exTop}.
          </p>
        }
      >
        <p>
          Fig. 23.15 draws, for one iteration, when the host inserts each operation and when a stream runs it. Both are
          about time, so here they are as an animation. Every bar comes from the same model as the explorer after it, at
          the durations the explorer starts from: an internal kernel of {DEFAULTS.internal} units, boundary kernels of{' '}
          {DEFAULTS.boundary}, exchanges of {DEFAULTS.exTop}.
        </p>
      </Translated>
      <Source section="23.3" pages="561–562" figs="Fig. 23.15" />
      <Simplification>
        <Translated
          vi={
            <p>
              Mô hình của trình khám phá bên dưới, với ghi chú liệt kê những gì nó bỏ qua: một rank, một GPU có 8 chỗ cho khối,
              không chiếm quyền, mỗi phép trao đổi trên đường kết nối riêng, thời gian không có đơn vị. Trong các thanh NCCL và
              NVSHMEM, kernel riêng của NCCL không chiếm chỗ khối nào, và mỗi lần put của NVSHMEM đi ngay khi hàng của nó được
              tính xong.
            </p>
          }
        >
          <p>
            The model of the explorer below, whose note lists what it leaves out: one rank, a GPU of 8 block slots, no
            preemption, each exchange on its own link, unitless times. In the NCCL and NVSHMEM bars, NCCL&rsquo;s own
            kernels take no block slots, and each NVSHMEM put travels as soon as its row is computed.
          </p>
        </Translated>
      </Simplification>
      <LoopScene />
      <Translated
        vi={
          <p>
            Các hàng dưới đã sẵn sàng ở {r2(it.bottom.end)} và rời đi ở {r2(it.exBottom.start)}. Không có gì trên GPU gây ra điều
            đó: host đang ở bên trong <code>MPI_Sendrecv</code> phía trên, và một lời gọi chặn không thể bị ngắt để bắt đầu một
            lời gọi khác. Sách nêu việc giảm chi phí này là một mục tiêu thiết kế của NCCL và NVSHMEM.
          </p>
        }
      >
        <p>
          The bottom rows were ready at {r2(it.bottom.end)} and left at {r2(it.exBottom.start)}. Nothing on the GPU caused
          that: the host was inside the top <code>MPI_Sendrecv</code>, and a blocking call cannot be interrupted to start
          another. The book names reducing this overhead as a design goal of NCCL and NVSHMEM.
        </p>
      </Translated>
      <Source section="23.3" pages="562" />

      <h2>
        <Translated vi="Ai chờ">Who waits</Translated>
      </h2>
      <Translated vi={<p>Giờ hãy tự điều khiển cùng mô hình đó. Trước khi chạm vào nó, hãy chốt một con số.</p>}>
        <p>
          Now drive the same model yourself. Before you touch it, commit to a number.
        </p>
      </Translated>
      <Predict
        id="ch23/23.4/p-window"
        prompt={
          <Translated
            vi={
              <>
                Trong vòng lặp có stream (Fig. 23.14), với kernel bên trong {DEFAULTS.internal}, các kernel biên{' '}
                {DEFAULTS.boundary} và hai phép trao đổi dài bằng nhau: mỗi phép trao đổi có thể dài bao nhiêu mà vẫn giấu được
                sau kernel bên trong?
              </>
            }
          >
            <>
              In the loop with streams (Fig. 23.14), with an internal kernel of {DEFAULTS.internal}, boundary kernels of{' '}
              {DEFAULTS.boundary} and two exchanges of the same length: how long can each exchange be and still hide behind
              the internal kernel?
            </>
          </Translated>
        }
        unit={tr('units', 'đơn vị')}
      >
        {(c) => (
          <Translated
            vi={
              <p style={{ margin: 0 }} data-window={r2(WINDOW_STREAMS)} data-window-nccl={r2(WINDOW_NCCL)}>
                Mô hình cho <b>{r2(WINDOW_STREAMS)}</b>{Math.abs(c - WINDOW_STREAMS) < 0.5 ? ', gần với con số của bạn' : ''}.
                Phép trao đổi phía trên bắt đầu khi host thấy kernel trên kết thúc, khoảng {r2(STREAMS_SHORT.exTop.start)}, và
                phép phía dưới chỉ bắt đầu khi phép phía trên đã trả về. Cả hai phải vừa trước khi kernel bên trong, vốn lúc đầu
                dùng chung GPU với các kernel biên, kết thúc ở {r2(STREAMS_SHORT.internal.end)}. Kiểm tra: đặt cả hai phép trao
                đổi bằng 2.5, rồi 3, và theo dõi &ldquo;phép trao đổi được giấu&rdquo;. Trong MPI không chồng lấp (Fig. 23.7)
                không phép trao đổi nào được giấu dù dài bao nhiêu. Trong vòng lặp NCCL, nơi hai phép chạy song song, cùng mô
                hình cho {r2(WINDOW_NCCL)}: dài khoảng gấp đôi.
              </p>
            }
          >
            <p style={{ margin: 0 }} data-window={r2(WINDOW_STREAMS)} data-window-nccl={r2(WINDOW_NCCL)}>
              The model says <b>{r2(WINDOW_STREAMS)}</b>{Math.abs(c - WINDOW_STREAMS) < 0.5 ? ', close to your number' : ''}.
              The top exchange starts once the host has seen the top kernel end, at about {r2(STREAMS_SHORT.exTop.start)}, and
              the bottom one only when the top one has returned. Both must fit before the internal kernel, which shared the
              GPU with the boundary kernels at first, ends at {r2(STREAMS_SHORT.internal.end)}. Check it: set both exchanges
              to 2.5, then 3, and watch &ldquo;exchanges hidden&rdquo;. In MPI with no overlap (Fig. 23.7) no exchange hides
              at any length. In the NCCL loop, where the two run side by side, the same model gives {r2(WINDOW_NCCL)}: about
              twice as long.
            </p>
          </Translated>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Rồi thử ba điều. Chuyển giữa hai vòng lặp MPI và theo dõi làn của phép trao đổi phía dưới. Cho phép trao đổi phía
            trên ngắn và phía dưới dài, rồi ngược lại, và xem khi nào các hàng dưới phải chờ. Chọn &ldquo;kernel bên trong
            trước&rdquo; và chuyển độ ưu tiên giữa bằng nhau và cao hơn: khi nào các kernel biên bắt đầu? Cuối cùng, bỏ từng thứ
            tự một và tìm các khoảng thời gian mà ở đó nó làm hỏng điều gì đó.
          </p>
        }
      >
        <p>
          Then try three things. Switch between the two MPI loops and watch the bottom exchange&rsquo;s lane. Make the top
          exchange short and the bottom one long, then the other way round, and see when the bottom rows wait. Choose
          &ldquo;internal kernel first&rdquo; and switch the priority between equal and higher: when do the boundary kernels
          start? Finally, remove one ordering at a time and find durations at which it breaks something.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một rank mà các lân cận hành xử y hệt nó. Một GPU có 8 chỗ cho khối; một kernel gồm các khối dài bằng nhau và
              không gì bị chiếm quyền. Mỗi phép trao đổi có đường kết nối riêng. Các lời gọi MPI giữ host cho đến khi phép trao
              đổi của chúng kết thúc. Một lần sao chép từ bộ nhớ phân trang là hoàn toàn đồng bộ. Mọi lời gọi API tốn cùng một khoảng thời gian
              nhỏ. Kernel riêng của NCCL không chiếm chỗ khối nào. Một lần put của NVSHMEM đi ngay khi các khối của hàng nó xong,
              và rào chắn trên stream kết thúc sau một khoảng nhỏ cố định kể từ khi lần put cuối tới nơi. Thời gian không có
              đơn vị.
            </p>
          }
        >
          <p>
            One rank whose neighbours behave exactly like it. A GPU of 8 block slots; a kernel is equal-length blocks and
            nothing is preempted. Each exchange has its own link. MPI calls hold the host until their exchange ends. A
            pageable copy is fully synchronous. Every API call costs the same small time. NCCL&rsquo;s own kernels take no
            block slots. An NVSHMEM put travels as soon as the blocks of its row finish, and the barrier on the stream
            ends a fixed small time after the last put lands. Times are unitless.
          </p>
        </Translated>
      </Simplification>
      <WhoWaits />
      <Translated
        vi={
          <p>
            Ba điều cần rút ra. Trong vòng lặp có stream, phép trao đổi phía dưới chờ phép trao đổi chặn phía trên ngay cả khi
            kernel dưới đã xong từ lâu, đó là chi phí của p. 562. Các phép trao đổi cộng lại dài hơn kernel bên trong thì không
            thể giấu được, đó là điều kiện sách đặt ra ở p. 554: đủ hàng bên trong. Và khi kernel bên trong được phát lệnh trước,
            độ ưu tiên cao hơn chỉ cho các kernel biên những chỗ đầu tiên được giải phóng, nên các phép trao đổi bắt đầu muộn;
            với độ ưu tiên bằng nhau, các kernel biên chờ những khối cuối của kernel bên trong, và các phép trao đổi hoàn toàn
            không còn chồng lấp với nó.
          </p>
        }
      >
        <p>
          Three things to take from it. In the loop with streams the bottom exchange waits for the blocking top exchange
          even when the bottom kernel finished long before, which is the overhead of p. 562. Exchanges that together
          outlast the internal kernel cannot be hidden, which is the condition the book sets on p. 554: enough internal
          rows. And with the internal kernel issued first, a higher priority only gets the boundary kernels the first slots
          that free up, so the exchanges start late; with equal priority the boundary kernels wait for the internal
          kernel&rsquo;s last blocks, and the exchanges no longer overlap it at all.
        </p>
      </Translated>
      <Source section="23.3" pages="554, 558, 562" />
      <Translated
        vi={
          <p>
            Cuộc săn lỗi có bài học thứ hai. Phần lớn các thứ tự bị bỏ không làm hỏng gì ở các khoảng thời gian mặc định, và
            các lần chờ đặt lại không làm hỏng gì ở bất kỳ khoảng thời gian nào trong mô hình này: chúng là race condition giữa
            các stream, do thời điểm quyết định, không phải lỗi chắc chắn xảy ra. Các trình giả lập của khóa học chạy mọi stream
            đồng bộ, theo thứ tự lời gọi, nên không race condition nào trong số này có thể xảy ra ở đó. Một test qua trên trình
            giả lập không nói gì về chúng; chỉ có lập luận về các thứ tự mới nói được.
          </p>
        }
      >
        <p>
          The bug hunt has a second lesson. Most removed orderings break nothing at the default durations, and the reset
          waits break nothing at any duration in this model: they are races between streams, decided by timing, not
          certain failures. The course&rsquo;s emulators run every stream synchronously, in call order, so none of these races can
          happen there. A test that passes on the emulator says nothing about them; only reasoning about the orderings
          does.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Mô hình và các phát hiện của nó là của khóa học này. Bản thay thế stream của các trình giả lập là đồng bộ
              (labs/README.md, &ldquo;What is and is not modelled&rdquo;).
            </p>
          }
        >
          <p>
            The model and its findings are ours. The emulators&rsquo; stream stand-ins are synchronous (labs/README.md,
            &ldquo;What is and is not modelled&rdquo;).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS234['ch23/23.4/q-drop-reset']} />

      <h2>
        <Translated vi="Ba thư viện, ba nơi để chờ">Three libraries, three places to wait</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách dùng vòng lặp này để giới thiệu ba mô hình lập trình. Mỗi mô hình có một đoạn ở đây: nó dùng để làm gì, đọc nó
            ở đâu trong sách, và điều gì đã thay đổi từ đó.
          </p>
        }
      >
        <p>
          The book uses this loop to introduce three programming models. Each gets a paragraph here: what it is for, where
          to read it in the book, and what has changed since.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            <b>MPI</b> dành cho các tiến trình có bộ nhớ riêng trao đổi thông điệp với nhau. Bạn khởi động cùng một chương trình
            nhiều lần bằng <code>mpiexec</code> (tên trong chuẩn; <code>mpirun</code> là một từ đồng nghĩa phổ biến), và mỗi
            tiến trình biết rank của nó và số rank từ một communicator, <code>MPI_COMM_WORLD</code> cho tất cả. Phép trao đổi
            vùng biên dùng <code>MPI_Sendrecv</code>, gửi cho một lân cận, nhận từ lân cận kia và chỉ trả về khi phép trao đổi
            hoàn tất; chuẩn dùng <code>MPI_Allreduce</code>, cộng một số float từ mọi rank và đưa kết quả cho mỗi rank. Truyền
            con trỏ device cho MPI, như các vòng lặp này làm, cần một MPI hỗ trợ CUDA (CUDA-aware), và đó là thuộc tính của
            cách MPI của bạn được build.
          </p>
        }
      >
        <p>
          <b>MPI</b> is for processes with separate memories that exchange messages. You start the same program several
          times with <code>mpiexec</code> (the standard&rsquo;s name; <code>mpirun</code> is a common synonym), and each
          process learns its rank and the number of ranks from a communicator, <code>MPI_COMM_WORLD</code> for all of them.
          The halo exchange uses <code>MPI_Sendrecv</code>, which sends to one neighbour, receives from the other and
          returns only when the exchange is complete; the norm uses <code>MPI_Allreduce</code>, which sums one float from
          every rank and gives each rank the result. Passing device pointers to MPI, as these loops do, needs a CUDA-aware
          MPI, and that is a property of how your MPI was built.
        </p>
      </Translated>
      <Source section="23.2" pages="545–553" figs="Figs. 23.4–23.11" />
      <Translated
        vi={
          <p>
            MPICH tự bật hỗ trợ GPU nếu nó tìm thấy CUDA hoặc HIP khi được cấu hình; Open MPI phải được cấu hình với CUDA và
            dùng một UCX được build kèm CUDA; dòng GPU của MVAPICH giờ gọi là MVAPICH-Plus. Hãy kiểm tra bản build của bạn; đừng
            giả định.
          </p>
        }
      >
        <p>
          MPICH turns GPU support on by itself if it finds CUDA or HIP when it is configured; Open MPI has to be configured
          with CUDA and use a UCX that was built with it; MVAPICH&rsquo;s GPU line is now called MVAPICH-Plus. Check your
          build; do not assume it.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the MPI Forum, MPICH, Open MPI and MVAPICH documentation (notes/today/ch23-facts.md, A10, A11, B7)', '2026-11-20, đối chiếu với tài liệu của MPI Forum, MPICH, Open MPI và MVAPICH (notes/today/ch23-facts.md, A10, A11, B7)')}>
        <Translated vi={<p>MPI 5.0 (tháng 6 năm 2025) là chuẩn hiện hành. Trên AMD, Open MPI hỗ trợ ROCm đi qua UCX.</p>}>
          <p>
            MPI 5.0 (June 2025) is the current standard. On AMD, ROCm-aware Open MPI goes through UCX.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            <b>NCCL</b> cung cấp các primitive giao tiếp chạy trên GPU và được xếp hàng trong stream, nên host thôi làm người
            trung gian. Một lần gửi và một lần nhận chạy như một thao tác gộp khi bạn đặt chúng giữa{' '}
            <code>ncclGroupStart</code> và <code>ncclGroupEnd</code>. Hãy để ý <code>ncclGroupEnd</code> hứa gì: các thao tác
            đã được xếp hàng, chưa hoàn tất. Đó là lý do dòng 31 và 36 bên dưới ghi event, và dòng 43–44 bắt stream bên trong
            chờ chúng trước vòng lặp tiếp theo.
          </p>
        }
      >
        <p>
          <b>NCCL</b> provides communication primitives that run on the GPU and are queued in streams, so the host stops
          being the go-between. A send and a receive run as one fused operation when you put them between{' '}
          <code>ncclGroupStart</code> and <code>ncclGroupEnd</code>. Note what <code>ncclGroupEnd</code> promises: the
          operations are queued, not complete. That is why lines 31 and 36 below record events, and lines 43–44 make the
          internal stream wait for them before the next iteration.
        </p>
      </Translated>
      <Source section="23.4" pages="563–568" figs="Figs. 23.16–23.21" />
      <Code
        tabs={[{ label: tr('Fig. 23.20, lines 27–44', 'Fig. 23.20, dòng 27–44'), code: NCCL_EXCHANGE, status: CHECKED_NCCL }]}
        firstLine={27}
        caption={
          <Translated
            vi={
              <>
                Phần trao đổi và phần cuối thân vòng lặp NCCL, với số dòng của sách; các dòng 1–26 là của Fig. 23.14. Chú thích ở
                dòng 38 là của khóa học này. Phần thiết lập nó giả định, <code>cudaSetDevice</code> trước{' '}
                <code>ncclCommInitRank</code>, nằm ở đầu tệp.
              </>
            }
          >
            <>
              The NCCL loop&rsquo;s exchange and the end of its body, with the book&rsquo;s line numbers; lines 1–26 are
              those of Fig. 23.14. The comment on line 38 is ours. The setup it assumes, <code>cudaSetDevice</code> before{' '}
              <code>ncclCommInitRank</code>, is in the file&rsquo;s header.
            </>
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Hai lời cảnh báo từ chính tài liệu của NCCL. <code>ncclCommInitRank</code> cần device của mỗi rank được đặt trước,
            điều Fig. 23.17 không trình bày. Và dùng NCCL cùng một MPI hỗ trợ CUDA để chuyển dữ liệu giữa cùng các GPU vào cùng
            lúc có thể gây deadlock. Trên AMD, RCCL giữ tên hàm của NCCL; header của nó là <code>&lt;rccl/rccl.h&gt;</code>.
          </p>
        }
      >
        <p>
          Two cautions from NCCL&rsquo;s own documentation. <code>ncclCommInitRank</code> needs each rank&rsquo;s device to
          be set first, which Fig. 23.17 does not show. And using NCCL and a CUDA-aware MPI to move data between the same
          GPUs at the same time can deadlock. On AMD, RCCL keeps NCCL&rsquo;s function names; its header is{' '}
          <code>&lt;rccl/rccl.h&gt;</code>.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the NCCL 2.32 documentation and ROCm 10.2 sources (notes/today/ch23-facts.md, A11, A12, B5)', '2026-11-20, đối chiếu với tài liệu NCCL 2.32 và mã nguồn ROCm 10.2 (notes/today/ch23-facts.md, A11, A12, B5)')}>
        <Translated
          vi={
            <p>
              <code>ncclGroupEnd</code>: các primitive &ldquo;have been enqueued to the provided streams, but are not
              necessarily complete.&rdquo;
            </p>
          }
        >
          <p>
            <code>ncclGroupEnd</code>: the primitives &ldquo;have been enqueued to the provided streams, but are not
            necessarily complete.&rdquo;
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            <b>NVSHMEM</b> là một chiều (one-sided): một luồng GPU có thể đặt (put) một giá trị vào bộ nhớ GPU của tiến trình
            khác, hoặc lấy (get) một giá trị, mà tiến trình đó không phải làm gì. Địa chỉ được định danh qua một symmetric heap:
            mọi tiến trình cấp phát cùng các đối tượng ở cùng các offset, nên một luồng truyền địa chỉ bản sao của chính nó và
            thư viện tìm ra bản ở xa. Trong Fig. 23.23, một luồng tính một giá trị rìa trên sẽ đặt nó thẳng vào vùng biên dưới
            của lân cận phía trên. Điều đó cho sự chồng lấp mà không phải tách kernel, và cho phép gộp vào một kernel những bước
            cần giao tiếp giữa chúng. Cái giá là rào chắn trên stream: một lần put có thể trả về trước khi dữ liệu của nó được
            giao.
          </p>
        }
      >
        <p>
          <b>NVSHMEM</b> is one-sided: a GPU thread can put a value into another process&rsquo;s GPU memory, or get one,
          without that process doing anything. Addresses are named through a symmetric heap: every process allocates the
          same objects at the same offsets, so a thread passes the address of its own copy and the library finds the
          remote one. In Fig. 23.23 a thread that computes a top boundary value puts it straight into the bottom halo of
          the top neighbour. That gives the overlap without splitting the kernel, and it lets steps that need
          communication between them be fused into one kernel. The price is the barrier on the stream: a put may return
          before its data is delivered.
        </p>
      </Translated>
      <Source section="23.5" pages="568–574" figs="Figs. 23.22–23.25" />
      <Translated
        vi={
          <p>
            Sách nói thêm rằng mỗi điểm lưới một lần put hợp với các GPU nối bằng NVLink, và đường kết nối chậm hơn thì cần ít
            lần put hơn nhưng lớn hơn, như <code>nvshmemx_float_put_block</code>, nơi cả một khối thực hiện một lần put. Không
            trang NVSHMEM nào chúng tôi tìm thấy nói như vậy: hãy coi đó là lời khuyên của sách, unverified.
          </p>
        }
      >
        <p>
          The book adds that one put per grid point suits GPUs joined by NVLink, and that slower links want fewer, larger
          puts, such as <code>nvshmemx_float_put_block</code>, where a whole block makes one put. No NVSHMEM page we found
          says so: treat it as the book&rsquo;s advice, unverified.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against the NVSHMEM documentation and GitHub, ROCm 10.2 and PyTorch 2.15 (notes/today/ch23-facts.md, A14, A15, B6, D5)', '2026-11-20, đối chiếu với tài liệu và GitHub của NVSHMEM, ROCm 10.2 và PyTorch 2.15 (notes/today/ch23-facts.md, A14, A15, B6, D5)')}>
        <Translated
          vi={
            <p>
              NVSHMEM là mã nguồn mở (Apache-2.0) từ tháng 8 năm 2025. Đối tác bên AMD là rocSHMEM, với tên{' '}
              <code>rocshmem_</code> và các biến thể <code>_wg</code>/<code>_wave</code>. Symmetric memory của PyTorch chạy
              trên NVSHMEM, rocSHMEM hoặc cửa sổ NCCL.
            </p>
          }
        >
          <p>
            NVSHMEM is open source (Apache-2.0) since August 2025. AMD&rsquo;s counterpart is rocSHMEM, with{' '}
            <code>rocshmem_</code> names and <code>_wg</code>/<code>_wave</code> variants. PyTorch&rsquo;s symmetric memory
            runs over NVSHMEM, rocSHMEM or NCCL windows.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Phần tóm tắt của sách (§23.6, pp. 574–575) vạch ranh giới giữa giao tiếp do host khởi tạo và giao tiếp do device
            khởi tạo. Do host khởi tạo, như NCCL ở đây, hợp với dữ liệu hạt thô (coarse-grained) và có thể được xác định độc lập
            với phần tính toán mà nó chồng lấp, như một hàng rìa được tính bởi kernel riêng. Do device khởi tạo, như NVSHMEM,
            hợp với dữ liệu hạt mịn (fine-grained) được xác định như một phần của phần tính toán đó. Sách nói thêm rằng các API
            đang hội tụ. Chúng đã hội tụ xa hơn sách nói: bên cạnh các lời gọi hai chiều, NCCL giờ đăng ký các cửa sổ đối
            xứng, có device API, và có <code>ncclPutSignal</code> và <code>ncclWaitSignal</code> một chiều, nên câu &ldquo;may
            add one-sided communication soon&rdquo; đã lỗi thời.
          </p>
        }
      >
        <p>
          The book&rsquo;s summary (§23.6, pp. 574–575) draws the line between host-initiated and device-initiated
          communication.
          Host-initiated, as with NCCL here, suits data that is coarse-grained and can be determined independently of the
          computation it overlaps, like a boundary row computed by its own kernel. Device-initiated, as with NVSHMEM, suits
          fine-grained data determined as part of that computation. The book adds that the APIs are converging. They have
          converged further than it says: besides its two-sided calls, NCCL now registers symmetric windows, has a device
          API, and has one-sided <code>ncclPutSignal</code> and <code>ncclWaitSignal</code>, so &ldquo;may add one-sided
          communication soon&rdquo; is out of date.
        </p>
      </Translated>
      <Today checked="2026-11-20, NCCL 2.32 (ch23-facts A13)">{/* i18n-ignore: date and citation */}
        <Translated vi={<p>Cửa sổ từ 2.27, device API từ 2.28, <code>ncclPutSignal</code> từ 2.29.</p>}>
          <p>Windows since 2.27, the device API 2.28, <code>ncclPutSignal</code> 2.29.</p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Metal không có thứ nào trong ba thứ đó. Trên Mac, công việc trải trên nhiều thiết bị là một tính năng của framework
            trải trên nhiều máy, không phải một API của Metal: chẳng hạn lớp phân tán của MLX chạy trên MPI, một vòng TCP, hoặc
            RDMA qua Thunderbolt.
          </p>
        }
      >
        <p>
          Metal has none of the three. On a Mac, work across several devices is a framework feature that spans machines,
          not a Metal API: MLX&rsquo;s distributed layer, for one, runs over MPI, a TCP ring, or RDMA over Thunderbolt.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch23-facts C6; ch24-facts C4)">{/* i18n-ignore: date and citation */}
        <Translated vi={<p>Metal không ghi trong tài liệu API collective hay API một chiều nào.</p>}>
          <p>Metal documents no collective or one-sided API.</p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Điều này nằm ở đâu trong một máy chủ mô hình">Where this sits in a model server</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Song song tensor đặt một all-reduce ở cuối mỗi lớp: mỗi GPU giữ một lát của mọi ma trận trọng số, tính một đầu ra
            từng phần, và các phần phải được cộng lại trước khi lớp tiếp theo có thể bắt đầu. Đó là vai trò mà phép trao đổi
            vùng biên đóng trong chương này, mà không có các hàng bên trong để giấu sau: không còn gì độc lập để tính trong khi
            tổng đang di chuyển. <code>--split-mode tensor</code> của llama.cpp dùng khuôn mẫu nhóm của sách cho việc đó,{' '}
            <code>ncclAllReduce</code> giữa <code>ncclGroupStart</code> và <code>ncclGroupEnd</code> trên stream của mỗi device,
            qua NCCL hoặc, khi build cho HIP, RCCL. Đó cũng là lý do vLLM giữ song song tensor bên trong một node, nơi đường kết
            nối nhanh nhất, và trải qua các node bằng song song pipeline, mà các lần truyền có thể giấu sau công việc khác như
            pipeline ở trang <a href={href('ch23', '23.3')}>23.3</a>.
          </p>
        }
      >
        <p>
          Tensor parallelism puts an all-reduce at the end of every layer: each GPU holds a slice of every weight matrix,
          computes a partial output, and the partials must be summed before the next layer can start. That is the role
          the halo exchange plays in this chapter, without the internal rows to hide behind: nothing independent is left
          to compute while the sum travels. llama.cpp&rsquo;s <code>--split-mode tensor</code> uses the book&rsquo;s group
          pattern for it, <code>ncclAllReduce</code> between <code>ncclGroupStart</code> and <code>ncclGroupEnd</code> on
          each device&rsquo;s stream, through NCCL or, built for HIP, RCCL. It is also why vLLM keeps tensor parallelism
          inside a node, where the links are fastest, and spans nodes with pipeline parallelism, whose transfers can hide
          behind other work like the pipeline on page <a href={href('ch23', '23.3')}>23.3</a>.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Việc đối chiếu này là của khóa học này. Các sự kiện bên dưới: llama.cpp <code>master</code> và tài liệu vLLM, đọc
              ngày 2026-11-20 (notes/today/ch23-facts.md, D2, D7).
            </p>
          }
        >
          <p>
            The mapping is ours. The facts under it: llama.cpp <code>master</code> and the vLLM docs, read 2026-11-20
            (notes/today/ch23-facts.md, D2, D7).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS234['ch23/23.4/q-tp']} />

      <h2>
        <Translated vi="Đọc các backend thật">Reading real backends</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn cũng sẽ gặp các cơ chế của trang này trên một GPU. Backend CUDA của llama.cpp tạo các stream non-blocking và các
            event với <code>cudaEventDisableTiming</code>. Sau một lần sao chép giữa hai device, nó ghi một event trong stream nguồn
            và bắt stream đích chờ nó: khuôn mẫu của Fig. 23.14, trải qua các device. Với <code>GGML_CUDA_GRAPH_OPT=1</code>, và
            khi dùng CUDA graph, nó có thể chạy các phép chiếu Q, K và V trên các stream riêng, rẽ nhánh bằng một event và hợp
            lại bằng các event khác. <code>stream.wait_stream(other)</code> của PyTorch là ghi-rồi-chờ trong một lời gọi. MLX
            trên Metal cho mỗi stream một command queue riêng và sắp thứ tự các stream bằng các giá trị{' '}
            <code>MTLSharedEvent</code>.
          </p>
        }
      >
        <p>
          You will meet this page&rsquo;s mechanisms on a single GPU too. llama.cpp&rsquo;s CUDA backend creates its
          streams non-blocking and its events with <code>cudaEventDisableTiming</code>. After a copy between two devices
          it records an event in the source stream and makes the destination stream wait for it: Fig. 23.14&rsquo;s
          pattern, across devices. With <code>GGML_CUDA_GRAPH_OPT=1</code>, and CUDA graphs in use, it can run the Q, K
          and V projections on separate streams, forking with one event and joining with others. PyTorch&rsquo;s{' '}
          <code>stream.wait_stream(other)</code> is record-then-wait in one call. MLX on Metal gives each stream its own
          command queue and orders streams with <code>MTLSharedEvent</code> values.
        </p>
      </Translated>
      <Today checked={tr('2026-11-20 against llama.cpp master, PyTorch main and MLX main (notes/today/ch23-facts.md, D3, D4, D6)', '2026-11-20, đối chiếu với llama.cpp master, PyTorch main và MLX main (notes/today/ch23-facts.md, D3, D4, D6)')}>
        <Translated
          vi={
            <p>
              llama.cpp: <code>ggml-cuda.cu</code> và <code>common.cuh</code>. PyTorch: <code>torch/cuda/streams.py</code>. MLX:{' '}
              <code>mlx/backend/metal/device.cpp</code> và <code>event.cpp</code>. Số dòng có thể xê dịch.
            </p>
          }
        >
          <p>
            llama.cpp: <code>ggml-cuda.cu</code> and <code>common.cuh</code>. PyTorch: <code>torch/cuda/streams.py</code>.
            MLX: <code>mlx/backend/metal/device.cpp</code> and <code>event.cpp</code>. Line numbers drift.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            <a href={href('ch23', 'labs')}>Lab</a> đưa nửa một-GPU của nội dung này lên chính thiết bị của bạn: giấu các lần sao chép sau một kernel bằng stream và bộ nhớ ghim, và đo xem bao nhiêu thời gian sao chép biến mất.
          </p>
        }
      >
        <p>
          The <a href={href('ch23', 'labs')}>lab</a> takes the single-GPU half of this to your own device: hide copies
          behind a kernel with streams and pinned memory, and measure how much of the copy time disappears.
        </p>
      </Translated>
    </>
  );
}
