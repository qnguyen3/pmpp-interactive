import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr as trText, useLocale } from '../../lib/i18n';
import { benchRows, fmt, num, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  FLOAT_BYTES,
  boundaryTraffic,
  chainOps,
  chainTimes,
  program,
  type Memory,
  type Strategy,
  type Traffic,
} from '../../sim/offload';
import { RoundTripScene } from './RoundTripScene';

const SEQUENTIAL = 'ch02/snippets/vecadd_sequential.cu';
const STARTER = 'ch02/lab1_vecadd/starter/kernel.cu';
const LAB3 = 'ch02/lab3_elementwise';

/** The host function of lab 2.1's kernel.cu, with the line of the file it starts on. */
export function hostFunction(src: string): { code: string; firstLine: number } | null {
  const lines = src.replace(/\n$/, '').split('\n');
  const at = lines.findIndex((l) => /^int vecAdd\(/.test(l));
  return at < 0 ? null : { code: lines.slice(at).join('\n'), firstLine: at + 1 };
}

// The corrected declaration line of Fig. 2.5, as the tested stub has it (sim/offload.test.ts ties it to the file).
const DECLARATION = program('cuda').head[2].trim();

// ---------------------------------------------------------------- interactive A: boundary traffic
const STRATEGY_LABEL: Record<Strategy, string> = {
  get transparent() {
    return trText('every operation is a transparent offload', 'mỗi phép toán là một lần chuyển việc trong suốt');
  },
  get resident() {
    return trText('data stays on the device between operations', 'dữ liệu ở lại trên device giữa các phép toán');
  },
};
const MEMORY_LABEL: Record<Memory, string> = {
  get separate() {
    return trText('separate device memory (CUDA, HIP)', 'bộ nhớ device riêng (CUDA, HIP)');
  },
  get shared() {
    return trText(
      'one shared memory (Metal on Apple silicon, data created in shared buffers)',
      'một bộ nhớ dùng chung (Metal trên Apple silicon, dữ liệu được tạo trong các buffer chia sẻ)',
    );
  },
};

export function bytesText(b: number): string {
  if (b === 0) return trText('0 bytes', '0 byte');
  const units = [trText('bytes', 'byte'), 'KiB', 'MiB', 'GiB', 'TiB'];
  let v = b;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${fmt(v)} ${units[u]}`;
}
const count = (v: number) => v.toLocaleString('en-US');

// The picture is drawn narrow on purpose, so that its labels stay legible in a
// half-width column and on a phone.
const PW = 380;
const ROW = 54; // one operation: up to three arrows
const TOP = 46;
const HOST_W = 104;
const DEV_W = 132;
const SLOT = 15;

/** A host box, a device box, and one arrow per tensor-sized transfer. */
function TrafficPicture({ t, memory }: { t: Traffic; memory: Memory }) {
  const k = t.ops.length;
  const h = TOP + k * ROW + 12;
  const hostX = 1;
  const devX = PW - 1 - DEV_W;
  const x0 = hostX + HOST_W + 4; // where an arrow meets host memory
  const x1 = devX - 4; // where it meets device memory
  const style = { width: '100%', maxWidth: 460, display: 'block' } as const;
  const ops = t.ops.map((name, i) => (
    <g key={i} data-op={name}>
      <rect x={devX + 8} y={TOP + i * ROW + 6} width={DEV_W - 16} height={ROW - 12} rx="3" fill="var(--global-wash)" stroke="var(--global)" />
      <text x={devX + DEV_W / 2} y={TOP + i * ROW + ROW / 2 + 4} fontSize="12" textAnchor="middle">{name}</text>
    </g>
  ));
  if (memory === 'shared')
    return (
      <svg viewBox={`0 0 ${PW} ${h}`} role="img" aria-label={trText('One shared memory: no transfers', 'Một bộ nhớ dùng chung: không có lần truyền nào')} data-picture="shared" style={style}>
        <rect x={hostX} y={10} width={PW - 2} height={h - 14} rx="4" fill="var(--page)" stroke="var(--rule)" strokeWidth="1.4" />
        <text x={hostX + 12} y={30} fontSize="12.5" fontWeight="600">
          {trText('One memory, used by host code and kernels', 'Một bộ nhớ, dùng bởi mã host và các kernel')}
        </text>
        <text x={hostX + 12} y={TOP + 24} fontSize="12.5">{trText('Nothing crosses:', 'Không gì đi qua:')}</text>
        <text x={hostX + 12} y={TOP + 41} fontSize="12.5">{trText('there is no boundary.', 'không có ranh giới nào.')}</text>
        <text x={hostX + 12} y={TOP + 66} fontSize="12" style={{ fill: 'var(--ink-2)' }}>
          {trText('The kernels still read', 'Các kernel vẫn đọc')}
        </text>
        <text x={hostX + 12} y={TOP + 82} fontSize="12" style={{ fill: 'var(--ink-2)' }}>
          {trText('and write every byte.', 'và ghi mọi byte.')}
        </text>
        {ops}
      </svg>
    );
  const slot = new Map<number, number>(); // how many arrows this operation's row already holds
  return (
    <svg viewBox={`0 0 ${PW} ${h}`} role="img" aria-label={trText('Transfers between host memory and device memory', 'Các lần truyền giữa bộ nhớ host và bộ nhớ device')} data-picture="separate" style={style}>
      <rect x={hostX} y={10} width={HOST_W} height={h - 14} rx="4" fill="var(--page)" stroke="var(--rule)" strokeWidth="1.4" />
      <text x={hostX + 10} y={30} fontSize="12.5" fontWeight="600">{trText('Host memory', 'Bộ nhớ host')}</text>
      <rect x={devX} y={10} width={DEV_W} height={h - 14} rx="4" fill="var(--page)" stroke="var(--rule)" strokeWidth="1.4" />
      <text x={devX + 10} y={30} fontSize="12.5" fontWeight="600">{trText('Device memory', 'Bộ nhớ device')}</text>
      <line x1={(x0 + x1) / 2} y1={14} x2={(x0 + x1) / 2} y2={h - 6} stroke="var(--ink-3)" strokeDasharray="4 4" />
      {ops}
      {t.transfers.map((tr, j) => {
        const s = slot.get(tr.op) ?? 0;
        slot.set(tr.op, s + 1);
        const y = TOP + tr.op * ROW + ROW / 2 - SLOT + s * SLOT;
        const into = tr.dir === 'in';
        const tip = into ? x1 : x0;
        const d = into ? -7 : 7;
        return (
          <g key={j} data-transfer={tr.dir}>
            <line x1={x0} y1={y} x2={x1} y2={y} stroke="var(--ink-2)" strokeWidth="1.4" />
            <polygon points={`${tip},${y} ${tip + d},${y - 3.5} ${tip + d},${y + 3.5}`} fill="var(--ink-2)" />
            <text x={hostX + HOST_W - 8} y={y + 4} fontSize="12" textAnchor="end" style={into ? undefined : { fontWeight: 650 }}>
              {into ? tr.what : trText(`${tr.what} back`, `${tr.what} về`)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ChainMeasured() {
  const { reports, loading } = useReports();
  const found = Object.keys(reports)
    .filter((b) => b !== 'emu' && b !== 'race')
    .flatMap((b) =>
      benchRows(reports[b], LAB3)
        .filter((r) => r.name === 'chain')
        .flatMap((r) => {
          const t = chainTimes(r);
          const rows = num(r.rows);
          const d = num(r.d);
          const shape =
            rows && d
              ? trText(`${count(rows)} rows of ${count(d)} floats`, `${count(rows)} hàng, mỗi hàng ${count(d)} số float`)
              : num(r.n)
                ? `n = ${count(num(r.n)!)}`
                : '';
          // Lab 2.3 says in words what one round trip costs on this backend (its 'copies' column).
          const trip = typeof r.copies === 'string' ? r.copies : typeof r.roundtrip === 'string' ? r.roundtrip : '';
          return t ? [{ b, device: String(reports[b].device.name ?? b), shape, trip, ...t }] : [];
        }),
    );
  if (loading) return null;
  if (!found.length)
    return (
      <Translated
        vi={
          <p className="ui" data-out="chain-measured" data-found="0" style={{ margin: '0.9rem 0 0' }}>
            Chưa có số đo nào trên máy này, nên không có thời gian nào được hiển thị: trang này không mô hình hóa thời gian. Chạy{' '}<code>python3 labs/pmpp.py run 2.3 --solution --bench</code> rồi tải lại trang này: lần chạy ghi lại thời gian của nó, và thời gian cho chuỗi giữ dữ liệu trên device cùng ba lượt đi về sẽ xuất hiện ở đây.
          </p>
        }
      >
        <p className="ui" data-out="chain-measured" data-found="0" style={{ margin: '0.9rem 0 0' }}>
          No measurement on this machine yet, so no time is shown: this page does not model one. Run{' '}
          <code>python3 labs/pmpp.py run 2.3 --solution --bench</code> and reload this page: the run records its
          times, and those for the resident chain and for three round trips appear here.
        </p>
      </Translated>
    );
  return (
    <div data-out="chain-measured" data-found={found.length} style={{ margin: '0.9rem 0 0' }}>
      {found.map((f, i) => (
        <Translated
          key={i}
          vi={
            <p className="ui" style={{ margin: '0 0 0.3rem' }}>
              Lab 2.3 trên <b>{f.device}</b> ({f.b}){f.shape ? `, ${f.shape}` : ''}: chuỗi ba phép toán của nó mất{' '}
              <b>{fmt(f.resident * 1e3)} ms</b> khi là một lời gọi giữ dữ liệu ở nơi các kernel chạy và{' '}
              <b>{fmt(f.roundTrips * 1e3)} ms</b> khi là ba lượt đi về{f.trip ? ` (mỗi lượt đi về: ${f.trip})` : ''}.
              {f.b === 'metal'
                ? ' Trên Metal không gì đi qua ranh giới: thứ mà các lượt đi về phải trả là việc sao chép mảng host vào các buffer chia sẻ rồi sao chép ngược lại, điều mà một giao diện nhận mảng host buộc phải làm.'
                : ''}
            </p>
          }
        >
          <p className="ui" key={i} style={{ margin: '0 0 0.3rem' }}>
            Lab 2.3 on <b>{f.device}</b> ({f.b}){f.shape ? `, ${f.shape}` : ''}: its chain of three operations took{' '}
            <b>{fmt(f.resident * 1e3)} ms</b> as one call that keeps the data where the kernels run and{' '}
            <b>{fmt(f.roundTrips * 1e3)} ms</b> as three round trips{f.trip ? ` (each round trip: ${f.trip})` : ''}.
            {f.b === 'metal'
              ? ' On Metal nothing crosses a boundary: what the round trips pay for is copying host arrays into shared buffers and back, which an interface that takes host arrays forces.'
              : ''}
          </p>
        </Translated>
      ))}
      <p className="muted" style={{ margin: 0 }}>
        <Translated vi="Đo trên máy của bạn, không phải dự đoán. Một thiết bị khác có thể chênh lệch nhiều theo cả hai hướng.">
          Measured on your machine, not predicted. Another device can differ widely in either direction.
        </Translated>
      </p>
    </div>
  );
}

function BoundaryTraffic() {
  const [k, setK] = useState(3);
  const [exp, setExp] = useState(20);
  const [strategy, setStrategy] = useState<Strategy>('transparent');
  const [memory, setMemory] = useState<Memory>('separate');
  const n = 2 ** exp;
  const locale = useLocale();
  const both = useMemo(
    () => ({
      transparent: boundaryTraffic(k, 'transparent', memory, n),
      resident: boundaryTraffic(k, 'resident', memory, n),
    }),
    [k, memory, n, locale],
  );
  const t = both[strategy];
  const ops = chainOps(k);
  const cols: Strategy[] = ['transparent', 'resident'];
  const row = (label: string, key: string, f: (x: Traffic) => string) => (
    <tr>
      <td style={{ paddingLeft: '0.6rem' }}>{label}</td>
      {cols.map((s) => (
        <td key={s} data-cell={`${key}-${s}`} style={{ fontWeight: s === strategy ? 650 : 400, fontVariantNumeric: 'tabular-nums' }}>
          {f(both[s])}
        </td>
      ))}
    </tr>
  );
  return (
    <Bench
      title={trText('Boundary traffic', 'Lưu lượng qua ranh giới')}
      task={trText(
        'Lengthen the chain and watch the two columns of the table. Then switch to one shared memory.',
        'Kéo dài chuỗi và quan sát hai cột của bảng. Sau đó chuyển sang một bộ nhớ dùng chung.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mỗi phép toán kết hợp kết quả đang tính <code>x</code> với một tensor mới gồm n số float (mỗi số {FLOAT_BYTES}{' '}byte). Một mũi tên là một lần truyền cỡ một tensor. Các con số đếm là chính xác; không có thời gian nào trong hình này.
            </>
          }
        >
          <>
            Each operation combines the running result <code>x</code> with one fresh tensor of n floats ({FLOAT_BYTES}{' '}
            bytes each). An arrow is one tensor-sized transfer. The counts are exact; there are no times in this picture.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi={<span>Số phép toán trong chuỗi: {k}</span>}>
            <span>Operations in the chain: {k}</span>
          </Translated>
          <input type="range" min="1" max="8" value={k} aria-label={trText('Operations in the chain', 'Số phép toán trong chuỗi')} onChange={(e) => setK(+e.target.value)} />
        </label>
        <label className="field">
          <Translated
            vi={
              <span>
                Số phần tử mỗi tensor: n = 2<sup>{exp}</sup> ({bytesText(n * FLOAT_BYTES)})
              </span>
            }
          >
            <span>Elements per tensor: n = 2<sup>{exp}</sup> ({bytesText(n * FLOAT_BYTES)})</span>
          </Translated>
          <input type="range" min="10" max="28" value={exp} aria-label={trText('Elements per tensor, as a power of two', 'Số phần tử mỗi tensor, dưới dạng lũy thừa của hai')} onChange={(e) => setExp(+e.target.value)} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.5rem' }}>
        <span className="seg" role="group" aria-label={trText('Strategy', 'Chiến lược')} style={{ flexWrap: 'wrap' }}>
          {cols.map((s) => (
            <button key={s} aria-pressed={strategy === s} onClick={() => setStrategy(s)}>{STRATEGY_LABEL[s]}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="seg" role="group" aria-label={trText('Memory', 'Bộ nhớ')} style={{ flexWrap: 'wrap' }}>
          {(['separate', 'shared'] as Memory[]).map((m) => (
            <button key={m} aria-pressed={memory === m} onClick={() => setMemory(m)}>{MEMORY_LABEL[m]}</button>
          ))}
        </span>
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div className="panel" style={{ background: 'var(--bench)' }}>
          <TrafficPicture t={t} memory={memory} />
        </div>
        <div>
          <p className="ui" data-out="traffic-summary" style={{ margin: '0 0 0.6rem' }}>
            {ops.map((o) => o.name).join(', ')}:{' '}
            <Translated
              vi={
                <>
                  <b>
                    {t.transfers.length} lần truyền cỡ một tensor, {bytesText(t.bytesAcross)}
                  </b>{' '}
                  qua ranh giới.
                </>
              }
            >
              <b>
                {t.transfers.length} tensor-sized {t.transfers.length === 1 ? 'transfer' : 'transfers'}, {bytesText(t.bytesAcross)}
              </b>{' '}
              across the boundary.
            </Translated>
          </p>
          <table className="data" style={{ background: 'var(--page)', margin: 0, width: '100%' }}>
            <thead>
              <tr>
                <th style={{ paddingLeft: '0.6rem' }}>
                  <Translated vi="Với chuỗi này">For this chain</Translated>
                </th>
                <th>
                  <Translated vi="mỗi phép toán một lượt đi về">each operation a round trip</Translated>
                </th>
                <th>
                  <Translated vi="dữ liệu giữ trên device">data kept on the device</Translated>
                </th>
              </tr>
            </thead>
            <tbody>
              {row(trText('tensor-sized transfers in', 'lần truyền vào cỡ một tensor'), 'in', (x) => String(x.transfersIn))}
              {row(trText('tensor-sized transfers out', 'lần truyền ra cỡ một tensor'), 'out', (x) => String(x.transfersOut))}
              {row(trText('bytes across the boundary', 'số byte qua ranh giới'), 'across', (x) => bytesText(x.bytesAcross))}
              {row(trText('bytes the kernels read and write', 'số byte các kernel đọc và ghi'), 'kernel', (x) => bytesText(x.kernelBytes))}
              {row(trText('bytes across per byte the kernels touch', 'số byte qua ranh giới trên mỗi byte các kernel chạm tới'), 'ratio', (x) => fmt(x.ratio))}
            </tbody>
          </table>
          <ChainMeasured />
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const FOUR = {
  transparent: boundaryTraffic(4, 'transparent', 'separate', 1).transfers.length,
  resident: boundaryTraffic(4, 'resident', 'separate', 1).transfers.length,
};

export const questionsS23: Record<string, Question> = {
  'ch02/2.3/q-output-only': {
    kind: 'choice',
    id: 'ch02/2.3/q-output-only',
    prompt: (
      <Translated
        vi={
          <>
            Với <code>C = A + B</code>, hàm stub cấp phát ba mảng trên device. Mảng nào phải được điền bằng một phép sao chép trước lần khởi chạy?
          </>
        }
      >
        For <code>C = A + B</code>, the stub allocates three device arrays. Which of them must be filled by a copy
        before the launch?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả ba.">
            All three.
          </Translated>
        ),
        why: (
          <Translated vi="Kết quả vẫn đúng. Bạn đã trả cho thêm một lần truyền n số float, và kernel ghi đè lên từng số đó.">
            The result is still right. You have paid for one more transfer of n floats, and the kernel overwrites every one of them.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={
              <>
                Chỉ <code>A_d</code> và <code>B_d</code>.
              </>
            }
          >
            <>
              <code>A_d</code> and <code>B_d</code> only.
            </>
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated
            vi={
              <>
                Không mảng nào: kernel đọc <code>A_h</code> và <code>B_h</code> từ bộ nhớ host.
              </>
            }
          >
            <>
              None: the kernel reads <code>A_h</code> and <code>B_h</code> from host memory.
            </>
          </Translated>
        ),
        why: (
          <Translated vi="Kernel được trao các địa chỉ trên device. Trong chương này kernel hoàn toàn không chạm vào bộ nhớ host.">
            The kernel is handed device addresses. In this chapter kernels do not touch host memory at all.
          </Translated>
        ),
      },
      {
        label: (
          <Translated
            vi={
              <>
                Chỉ <code>C_d</code>.
              </>
            }
          >
            <>
              <code>C_d</code> only.
            </>
          </Translated>
        ),
        why: (
          <Translated vi="C_d là mảng duy nhất mà nội dung cũ không ai đọc.">
            C_d is the one array whose old contents nobody reads.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>C</code> được ghi và không bao giờ được đọc, nên nó cần được cấp phát chứ không cần nội dung cũ. Đoạn mã đầy đủ hơn của sách làm đúng như vậy: ba lần cấp phát, hai lần sao chép vào, một lần sao chép ra. Quy tắc này áp dụng cho mọi hàm stub bạn sẽ viết: sao chép vào những gì kernel đọc, sao chép ra những gì bên gọi cần, và cấp phát mọi thứ kernel chạm tới.
          </p>
        }
      >
        <p>
          <code>C</code> is written and never read, so it needs an allocation and not its old contents. The book's more
          complete listing does exactly this: three allocations, two copies in, one copy out. The rule carries to every
          stub you will write: copy in what the kernel reads, copy out what the caller needs, and allocate everything
          the kernel touches.
        </p>
      </Translated>
    ),
    source: 'ch. 2 §2.4, p. 31, Fig. 2.8',
  },
  'ch02/2.3/q-pays': {
    kind: 'choice',
    id: 'ch02/2.3/q-pays',
    prompt: (
      <Translated
        vi={
          <>
            Bạn được chuyển trong suốt đúng một trong hai hàm host sang device (sao chép vào, tính, sao chép ra): cộng hai vectơ gồm n·n số float, hoặc nhân hai ma trận float n × n. Hàm nào có khả năng bù lại chi phí sao chép hơn khi n tăng?
          </>
        }
      >
        You may offload exactly one of two host functions transparently (copy in, compute, copy out): adding two
        vectors of n·n floats, or multiplying two n × n matrices of floats. Which is more likely to repay its copies as
        n grows?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Phép cộng: nó là kernel đơn giản hơn, nên device làm xong sớm hơn.">
            The addition: it is the simpler kernel, so the device finishes it sooner.
          </Translated>
        ),
        why: (
          <Translated vi="Xong sớm hơn chính là vấn đề. Nó làm một phép cộng cho mỗi 12 byte nó di chuyển, với mọi n.">
            Finishing sooner is the trouble. It does one addition for every 12 bytes it moves, at every n.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Phép nhân ma trận: lượng phép tính của nó tăng nhanh hơn số byte nó phải di chuyển.">
            The matrix product: its arithmetic grows faster than the bytes it has to move.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không phân biệt được: cả hai di chuyển cùng số byte.">
            Neither can be told apart: both move the same number of bytes.
          </Translated>
        ),
        why: (
          <Translated vi="Chúng đúng là di chuyển cùng số byte, và chính vì thế hàm làm nhiều phép tính hơn trên mỗi byte di chuyển là lựa chọn tốt hơn.">
            They do move the same bytes, which is exactly why the one that does more arithmetic per byte moved is the better bet.
          </Translated>
        ),
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <p>
              Cả hai lời gọi sao chép vào hai đầu vào gồm n² số float và sao chép ra một kết quả: 12n² byte qua ranh giới. Phép cộng làm n² phép cộng cho lượng đó, một phép cho mỗi 12 byte, và tỉ lệ đó không bao giờ cải thiện. Phép nhân làm khoảng n³ phép nhân và n³ phép cộng, nên lượng phép tính thu được trên mỗi byte di chuyển tăng tỉ lệ với n.
            </p>
          }
        >
          <p>
            Both calls copy two inputs of n² floats in and one result out: 12n² bytes across the boundary. The addition
            does n² additions for that, one per 12 bytes, and the ratio never improves. The product does about n³
            multiplications and n³ additions, so the arithmetic bought per byte moved grows in proportion to n.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Đây là lập luận dựa trên số đếm, không phải một phép đo. Đếm số phép toán trên mỗi byte là phương pháp của{' '}<a href={href('ch05', '5.1')}>chương 5, mục 5.1</a>, ở đó áp dụng cho số byte một kernel nạp và ở đây cho số byte một hàm stub sao chép.
            </p>
          }
        >
          <p>
            This is reasoning about counts, not a measurement. Counting operations per byte is the method of{' '}
            <a href={href('ch05', '5.1')}>chapter 5, section 5.1</a>, applied there to the bytes a kernel fetches and
            here to the bytes a stub copies.
          </p>
        </Translated>
      </>
    ),
    source: 'our reasoning from ch. 2 §2.3, p. 28, with the counting method of ch. 5 §5.1',
    vi: { source: 'lập luận của khóa học này từ ch. 2 §2.3, p. 28, với phương pháp đếm của ch. 5 §5.1' },
  },
  'ch02/2.3/q-interface': {
    kind: 'choice',
    id: 'ch02/2.3/q-interface',
    prompt: (
      <Translated
        vi={
          <>
            Trên Apple silicon bạn giữ chữ ký hàm{' '}<code>vecAdd(const float* A_h, const float* B_h, float* C_h, int n)</code>, và bên gọi truyền vào các mảng heap thông thường. Hàm stub CUDA sao chép dữ liệu ba lần. Một bản Metal với các buffer lưu trữ chia sẻ có thể tránh được bao nhiêu trong ba lần sao chép đó?
          </>
        }
      >
        On Apple silicon you keep the signature{' '}
        <code>vecAdd(const float* A_h, const float* B_h, float* C_h, int n)</code>, and the caller passes ordinary heap
        arrays. The CUDA stub copies data three times. How many of those three copies can a Metal version with
        shared-storage buffers avoid?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả ba: với lưu trữ chia sẻ, GPU dùng nguyên các mảng của bên gọi.">
            All three: with shared storage the GPU uses the caller's arrays as they are.
          </Translated>
        ),
        why: (
          <Translated vi="Lưu trữ chia sẻ cho phép CPU và GPU dùng cùng những byte đó. Nó không đặt các mảng heap của bên gọi vào trong một buffer.">
            Shared storage lets the CPU and the GPU use the same bytes. It does not put the caller's heap arrays inside a buffer.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không lần nào, trừ khi các mảng được căn theo trang và được bọc lại mà không sao chép, hoặc bên gọi được sửa để tạo dữ liệu của nó trong các buffer.">
            None, unless the arrays are page-aligned and wrapped without copying, or the caller is changed to create its data in buffers.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Chỉ lần sao chép về: buffer kết quả có thể đọc tại chỗ.">
            Only the copy back: the result buffer can be read in place.
          </Translated>
        ),
        why: (
          <Translated vi="Nó có thể đọc tại chỗ, nhưng chữ ký hàm này hứa trả kết quả trong C_h, là một vùng cấp phát khác. Đưa kết quả đến đó là một lần sao chép.">
            It can be read in place, but this signature promises the result in C_h, which is a different allocation. Getting it there is a copy.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các đối số mảng của một kernel Metal được gắn với buffer. <code>newBufferWithBytes:length:options:</code> tạo một buffer bằng cách sao chép mảng của bên gọi; <code>newBufferWithBytesNoCopy:</code> bọc vùng nhớ có sẵn, và chỉ khi con trỏ và độ dài được căn theo trang. Bộ nhớ dùng chung loại bỏ nhu cầu sao chép. Nó không loại bỏ những lần sao chép mà một giao diện bắt buộc.
          </p>
        }
      >
        <p>
          A Metal kernel's array arguments are bound to buffers. <code>newBufferWithBytes:length:options:</code> makes one by copying the
          caller's array; <code>newBufferWithBytesNoCopy:</code> wraps existing memory, and only when the pointer and the
          length are page-aligned. Shared memory removes the need for a copy. It does not remove the copies an interface
          forces.
        </p>
      </Translated>
    ),
    source: 'not in the book; Apple documentation as recorded in notes/today/ch02-facts.md, C2 and C3 (read 2026-11-20)',
    vi: { source: 'không có trong sách; tài liệu của Apple như ghi trong notes/today/ch02-facts.md, C2 và C3 (đọc ngày 2026-11-20)' },
  },
};

// ---------------------------------------------------------------- the section
export function S23() {
  const sequential = useLabFile(SEQUENTIAL);
  const starterFile = useLabFile(STARTER);
  const starter = starterFile ? hostFunction(starterFile) : null;
  return (
    <>
      <Translated
        vi={
          <p>
            Chương trình đầu tiên của sách cộng hai vectơ: <code>C[i] = A[i] + B[i]</code> với mọi <code>i</code>. Sách gọi nó là hello world của lập trình song song dữ liệu, và theo phép kiểm tra của{' '}<a href={href('ch02', '2.1')}>mục 2.1</a> nó là trường hợp dễ nhất có thể: mỗi đầu ra đọc một phần tử của mỗi đầu vào và không đọc gì mà một lần lặp khác ghi. Đây là nó khi chưa có GPU nào tham gia.
          </p>
        }
      >
        <p>
          The book's first program adds two vectors: <code>C[i] = A[i] + B[i]</code> for every <code>i</code>. It calls
          this the hello world of data-parallel programming, and by the test of{' '}
          <a href={href('ch02', '2.1')}>section 2.1</a> it is the easiest case there is: each output reads one element of
          each input and nothing that another iteration writes. Here it is with no GPU involved.
        </p>
      </Translated>
      <Source section="2.3" pages="26" figs={trText('Fig. 2.4', 'Hình 2.4')} />
      <Code
        tabs={[{ label: trText('host only', 'chỉ host'), code: sequential ?? '// loading labs/' + SEQUENTIAL, status: 'front-end-checked' }]}
        caption={
          <Translated
            vi={
              <>
                <code>labs/{SEQUENTIAL}</code>: Fig. 2.4 của sách với <code>const</code> trên các đầu vào và dữ liệu thật thay cho các chú thích. Tệp này không có mã device, và nó vẫn là một chương trình CUDA C++: mọi chương trình C++ thông thường đều là một chương trình CUDA tình cờ chỉ có mã host (p. 24).
              </>
            }
          >
            <code>labs/{SEQUENTIAL}</code>: the book's Fig. 2.4 with <code>const</code> on the inputs and real data in
            place of its comments. There is no device code in this file, and it is still a CUDA C++ program: any
            ordinary C++ program is one that happens to have only host code (p. 24).
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Hậu tố trên các tên là một quy ước sách giới thiệu ở đây: <code>_h</code> đánh dấu dữ liệu nằm trong bộ nhớ host và <code>_d</code> dữ liệu nằm trên device. Từ giờ một hàm có thể giữ hai con trỏ đến cùng một vectơ ở hai bộ nhớ khác nhau, và hậu tố là thứ duy nhất cho bạn biết cái nào là cái nào. Khóa học này áp dụng nó ở mọi nơi. Chính các đoạn mã của sách thì không; {' '}<a href={href('ch02', 'wrap')}>trang tóm tắt</a> liệt kê những chỗ sách bỏ nó.
          </p>
        }
      >
        <p>
          The suffix on the names is a convention the book introduces here: <code>_h</code> marks data that lives in
          host memory and <code>_d</code> data that lives on the device. From now on a function can hold two pointers to
          the same vector in two different memories, and the suffix is the only thing that tells you which is which.
          This course applies it everywhere. The book's own listings do not; the{' '}
          <a href={href('ch02', 'wrap')}>summary page</a> lists where they drop it.
        </p>
      </Translated>
      <Source section="2.3" pages="26" />

      <h2>
        <Translated vi="Chuyển vòng lặp sang device: ba phần">Moving the loop to the device: three parts</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Để chạy vòng lặp trên GPU, <code>vecAdd</code> giữ nguyên tên và các đối số và thay đổi phần thân. Phần thân giờ có ba phần. Phần 1 cấp phát bộ nhớ trên device cho A, B và C và sao chép A và B vào đó. Phần 2 khởi chạy một kernel, nơi các phép cộng diễn ra. Phần 3 sao chép C về host và giải phóng ba mảng trên device. Đây là hàm bạn điền vào cho lab 2.1, đúng như lab giao cho bạn:
          </p>
        }
      >
        <p>
          To run the loop on a GPU, <code>vecAdd</code> keeps its name and its arguments and changes its body. The body
          now has three parts. Part 1 allocates memory on the device for A, B and C and copies A and B into it. Part 2
          launches a kernel, which is where the additions happen. Part 3 copies C back to the host and frees the three
          device arrays. This is the function you fill in for lab 2.1, as the lab hands it to you:
        </p>
      </Translated>
      <Source section="2.3" pages="27" figs={trText('Fig. 2.5', 'Hình 2.5')} />
      <Code
        tabs={[{ label: trText('lab 2.1 starter', 'mã khởi đầu (starter) của lab 2.1'), code: starter?.code ?? starterFile ?? '// loading labs/' + STARTER }]}
        firstLine={starter?.firstLine ?? 1}
        caption={
          <Translated
            vi={
              <>
                Hàm host trong <code>labs/{STARTER}</code>, tệp mà lab 2.1 bắt đầu (kernel phía trên nó được bỏ ra ở đây). Ba phần là của sách; lab thêm một giá trị trả về, một kích thước khối do bên gọi chọn, và một phép kiểm tra sau lần khởi chạy mà <a href={href('ch02', '2.6')}>mục 2.6</a> giải thích.
              </>
            }
          >
            The host function in <code>labs/{STARTER}</code>, the file lab 2.1 starts from (the kernel above it is left
            out here). The three parts are the book's; the lab adds a return value, a block size chosen by the caller,
            and a check after the launch that <a href={href('ch02', '2.6')}>section 2.6</a> explains.
          </Translated>
        }
      />
      <Translated
        vi={
          <p>
            Một dòng trong khung mã của sách cần cảnh báo trước khi bạn gõ nó. Dòng 03 của Fig. 2.5 được in là{' '}<code>float *d_A *d_B, *d_C;</code>. Thiếu một dấu phẩy, nên nó không biên dịch được, và nó dùng tiền tố{' '}<code>d_</code> trong khi phần chữ và mọi hình sau đó dùng hậu tố. Dòng mà khóa học dùng, và kiểm thử, là <code>{DECLARATION}</code>. Cuộc săn lỗi trong phần Kiểm tra lỗi ở{' '}<a href={href('ch02', '2.4')}>mục 2.4</a> cho bạn thấy lại dòng đã in, bên cạnh lỗi sơ suất trong đoạn mã kiểm tra lỗi của sách.
          </p>
        }
      >
        <p>
          One line of the book's skeleton needs a warning before you type it. Line 03 of Fig. 2.5 is printed as{' '}
          <code>float *d_A *d_B, *d_C;</code>. A comma is missing, so it does not compile, and it uses a{' '}
          <code>d_</code> prefix where the text and every later figure use the suffix. The line the course uses, and
          tests, is <code>{DECLARATION}</code>. The hunt under Error checking in{' '}
          <a href={href('ch02', '2.4')}>section 2.4</a> shows you the printed line again, beside the slip in the book's
          error-check snippet.
        </p>
      </Translated>
      <Source section="2.3" pages="27" figs={trText('Fig. 2.5, line 03', 'Hình 2.5, dòng 03')} />

      <h2>
        <Translated vi="Ba phần trên hai làn">Three parts on two lanes</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hình của sách cho ba phần là hai hộp và hai mũi tên: bộ nhớ host, bộ nhớ device, Phần 1 đi sang, Phần 3 quay về. Điều một hình tĩnh không thể hiện được là ai đang chờ ai. Có hai làn hoạt động, một luồng host và một device, cùng hai bộ nhớ, và ở hầu hết các thời điểm một làn đang rảnh hoặc bị chặn. Hoạt hình theo 8 con số nhỏ đi qua toàn bộ hàm, rồi chạy lại cùng các bước trên một máy mà ranh giới giữa hai bộ nhớ không tồn tại.
          </p>
        }
      >
        <p>
          The book's figure for the three parts is two boxes and two arrows: host memory, device memory, Part 1 across,
          Part 3 back. What a still picture cannot show is who is waiting for whom. There are two lanes of activity, a
          host thread and a device, and two memories, and at most moments one lane is idle or blocked. The animation
          follows 8 small numbers through the whole function, and then plays the same steps again on a machine where the
          boundary between the two memories does not exist.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mô hình đằng sau hoạt hình không có đồng hồ. Độ rộng các bước không phải là thời lượng, và kernel đã tạo ra bao nhiêu kết quả khi host đến lần sao chép về là lựa chọn của hình vẽ. Những gì chính xác: giá trị nào nằm ở đâu sau mỗi lời gọi, bao nhiêu byte đã đi qua, và luồng host chờ trong những lời gọi nào. Kết quả xuất hiện không theo thứ tự là có chủ ý, vì không gì hứa hẹn một thứ tự giữa các luồng.
            </p>
          }
        >
          <p>
            The model behind the animation has no clock. Step widths are not durations, and how many results the kernel
            has produced when the host reaches the copy back is the drawing's choice. What is exact: which values are
            where after each call, how many bytes have crossed, and in which calls the host thread waits. Results appear
            out of order on purpose, because nothing promises an order among threads.
          </p>
        </Translated>
      </Simplification>
      <RoundTripScene />
      <Translated
        vi={
          <p>
            Hai điều trong hoạt hình không có trên các trang sách. Chương này không bao giờ nói một lần khởi chạy có chờ kernel hay không, và các đoạn mã của nó không bao giờ chờ tường minh. Chúng chạy đúng là nhờ những gì lần sao chép về làm:{' '}<a href={href('ch02', '2.2')}>mục 2.2</a> cho thấy lần khởi chạy trả về ngay, và ở đây bạn thấy lần sao chép từ chối trả về cho đến khi kernel đi trước nó xong.
          </p>
        }
      >
        <p>
          Two things in the animation are not on the book's pages. The chapter never says whether a launch waits for
          the kernel, and its listings never wait explicitly. They work because of what the copy back does:{' '}
          <a href={href('ch02', '2.2')}>section 2.2</a> has the launch returning at once, and here you saw the copy
          refuse to return until the kernel ahead of it was done.
        </p>
      </Translated>
      <Today checked={trText('2026-11-20 (notes/today/ch02-facts.md, A4 and A5)', '2026-11-20 (notes/today/ch02-facts.md, A4 và A5)')}>
        <Translated
          vi={
            <p>
              Hướng dẫn của NVIDIA nói ba điều: một lần khởi chạy là bất đồng bộ đối với luồng host,{' '}<code>cudaMemcpy</code> không trả về cho đến khi việc sao chép hoàn tất, và các thao tác trong stream mặc định chạy theo thứ tự chúng được phát lệnh. Việc lần sao chép về do đó chờ kernel là cách khóa học này hiểu ba phát biểu đó gộp lại. Không có câu nào trong hướng dẫn nói điều đó.
            </p>
          }
        >
          <p>
            NVIDIA's guide says three things: a launch is asynchronous with respect to the host thread,{' '}
            <code>cudaMemcpy</code> does not return until the copy has completed, and operations in the default stream
            run in the order they were issued. That the copy back therefore waits for the kernel is our reading of those
            three statements together. No single sentence of the guide says it.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS23['ch02/2.3/q-output-only']} />

      <h2>
        <Translated vi="Một đại lý thuê ngoài, và cái giá nó tính">An outsourcing agent, and what it charges</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Nhìn từ <code>main</code>, không gì thay đổi. Nó gọi <code>vecAdd(A, B, C, N)</code> với các mảng host và nhận các tổng trong <code>C</code>. Hàm đã trở thành một đại lý gửi đầu vào đi, nhờ làm việc ở nơi khác, rồi gửi kết quả về, và bên gọi không thể nhận ra. Sách gọi đây là thuê ngoài trong suốt và cảnh báo về nó ngay trong cùng câu: tất cả việc sao chép qua lại đó có thể làm nó rất kém hiệu quả. Mã thực tế giữ các cấu trúc dữ liệu lớn trên device và gọi các hàm device trên chúng từ host. Bản trong suốt có mặt trong chương này vì nó là thứ đơn giản nhất mà vẫn hoàn chỉnh.
          </p>
        }
      >
        <p>
          Seen from <code>main</code>, nothing changed. It calls <code>vecAdd(A, B, C, N)</code> with host arrays and
          finds the sums in <code>C</code>. The function has become an agent that ships the inputs out, has the work done
          elsewhere, and ships the result home, and the caller cannot tell. The book calls this transparent outsourcing
          and warns about it in the same breath: all that copying back and forth can make it very inefficient. Real code
          keeps its large data structures on the device and calls device functions on them from the host. The
          transparent version is in this chapter because it is the simplest thing that is complete.
        </p>
      </Translated>
      <Source section="2.3" pages="28" />
      <Translated
        vi={
          <p>
            Hoạt hình đã gắn một con số cho một lời gọi: 12 byte được di chuyển cho mỗi phần tử được cộng. Chi phí quan trọng trong thực tế là chuyện xảy ra khi các lời gọi được nối thành chuỗi, vì đầu ra của một lời gọi trong suốt là đầu vào của lời gọi tiếp theo và nó đi về nhà rồi lại đi ra ở giữa.
          </p>
        }
      >
        <p>
          The animation put a number on one call: 12 bytes moved for every element added. The cost that matters in
          practice is what happens when calls are chained, because the output of one transparent call is the input of
          the next and it travels home and back out in between.
        </p>
      </Translated>
      <Predict
        id="ch02/2.3/p-crossings"
        prompt={
          <Translated
            vi={
              <>
                Bốn phép toán theo từng phần tử trong một chuỗi. Mỗi phép kết hợp kết quả đang tính với một tensor mới gồm n số float, và mỗi phép được viết như một lần chuyển việc trong suốt riêng, giống <code>vecAdd</code>. Tổng cộng có bao nhiêu lần truyền cỡ một tensor đi qua ranh giới giữa host và device?
              </>
            }
          >
            <>
              Four elementwise operations in a chain. Each combines the running result with one new tensor of n floats,
              and each is written as its own transparent offload, like <code>vecAdd</code>. In total, how many
              tensor-sized transfers cross the boundary between host and device?
            </>
          </Translated>
        }
        unit={trText('transfers', 'lần truyền')}
      >
        {(v) => (
          <>
            <Translated
              vi={
                <p>
                  Đáp án là {FOUR.transparent}: mỗi lời gọi sao chép vào hai tensor và sao chép ra một.{' '}
                  {v === FOUR.transparent
                    ? 'Đúng như bạn đã đếm.'
                    : v === FOUR.resident
                      ? 'Số bạn đếm là số chuỗi cần khi dữ liệu ở lại trên device, đó chính là điểm của bước tiếp theo.'
                      : 'Hãy đếm theo lời gọi, không theo tensor: kết quả đang tính đi ra rồi về nhà ở mỗi bước.'}
                </p>
              }
            >
              <p>
                It is {FOUR.transparent}: every call copies two tensors in and one out.{' '}
                {v === FOUR.transparent
                  ? 'That is what you counted.'
                  : v === FOUR.resident
                    ? 'Your count is what the chain needs when the data stays on the device, which is the point of the next step.'
                    : 'Count per call, not per tensor: the running result goes out and comes home at every step.'}
              </p>
            </Translated>
            <Translated
              vi={
                <p>
                  Giữ dữ liệu trên device giữa các phép toán thì con số là {FOUR.resident}: giá trị ban đầu và bốn toán hạng đi vào, một kết quả đi ra. Đặt trình khám phá bên dưới thành 4 phép toán và chuyển qua lại giữa hai cách.
                </p>
              }
            >
              <p>
                Keep the data on the device between the operations and it is {FOUR.resident}: the start value and the
                four operands in, one result out. Set the explorer below to 4 operations and switch between the two.
              </p>
            </Translated>
          </>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Trình khám phá đếm điều này cho chuỗi bất kỳ. Mặc định của nó là một chuỗi ba phép toán theo từng phần tử: cộng bias, nhân hệ số, cộng residual. Hãy tìm quy luật của mỗi cột trong bảng khi chuỗi dài ra, rồi tìm hàng giống nhau ở cả hai cột.
          </p>
        }
      >
        <p>
          The explorer counts this for any chain. Its default is a chain of three elementwise operations: add a bias,
          scale, add a residual. Find the rule for each column of the table as the chain grows, then find the row that
          is the same in both columns.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trình khám phá đếm số byte đi qua ranh giới, con số này chính xác. Nó không mô hình hóa một byte mất bao lâu để đi qua, bộ nhớ host khóa trang (page-locked), hay các bộ cấp phát lưu đệm bộ nhớ device. Thời gian duy nhất nó hiển thị là số đo của chính bạn từ lab 2.3.
            </p>
          }
        >
          <p>
            The explorer counts bytes that cross the boundary, which is exact. It does not model how long a byte takes to
            cross, page-locked host memory, or allocators that cache device memory. The only times it shows are your own
            measurements from lab 2.3.
          </p>
        </Translated>
      </Simplification>
      <BoundaryTraffic />
      <Translated
        vi={
          <p>
            Một chuỗi trong suốt di chuyển ba tensor cho mỗi phép toán. Một chuỗi giữ dữ liệu trên device di chuyển mỗi đầu vào một lần và kết quả một lần, k + 2 tensor cho k phép toán, và khoảng cách nới rộng với mỗi phép toán bạn thêm vào. Hàng không thay đổi là công việc của chính các kernel: chúng đọc và ghi cùng số byte theo cả hai cách. Giữ dữ liệu trên device không làm kernel nào nhanh hơn. Nó loại bỏ công việc vốn chẳng phải phép tính số học.
          </p>
        }
      >
        <p>
          A transparent chain moves three tensors per operation. A resident chain moves each input once and the result
          once, k + 2 tensors for k operations, and the gap widens with every operation you add. The row that does not
          move is the kernels' own work: they read and write the same bytes either way. Keeping data resident does not
          make any kernel faster. It removes work that was never arithmetic.
        </p>
      </Translated>

      <h2>
        <Translated vi="Điều này nằm ở đâu bên dưới một lời gọi framework">Where this sits under a framework call</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bạn đã viết mã tuân theo quy tắc này. Bạn gọi <code>.to(device)</code> trên một mô hình một lần và để trọng số ở đó. Bạn giữ activation trên device từ lớp này sang lớp tiếp theo, và cache key và value của một decoder ở lại đó từ token này sang token tiếp theo. Mỗi thói quen đó là cột bên phải của bảng. Và bản thân lời gọi là Phần 1 của mục này: trong backend CUDA của PyTorch, việc di chuyển một tensor giữa host và device kết thúc bằng một <code>cudaMemcpy</code> host-to-device hoặc device-to-host, là biến thể bất đồng bộ khi bạn yêu cầu <code>non_blocking</code> và là một lần sao chép rồi chờ khi bạn không yêu cầu.
          </p>
        }
      >
        <p>
          You already write code that obeys this rule. You call <code>.to(device)</code> on a model once and leave the
          weights there. You keep activations on the device from one layer to the next, and a decoder's cache of keys
          and values stays there from one token to the next. Each of those habits is the right-hand column of the table.
          And the call itself is this section's Part 1: in PyTorch's CUDA backend, moving a tensor between the host and
          the device ends in a host-to-device or device-to-host <code>cudaMemcpy</code>, the asynchronous variant when
          you ask for <code>non_blocking</code> and a copy followed by a wait when you do not.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              PyTorch không có trong sách. Phép sao chép nằm trong <code>aten/src/ATen/native/cuda/Copy.cu</code>, đọc ngày 2026-11-20 (notes/today/ch02-facts.md, D4). Việc trọng số và cache được giữ trên device vì lý do này là mô tả của khóa học này về thực hành phổ biến, không phải trích dẫn.
            </p>
          }
        >
          <p>
            PyTorch is not in the book. The copy is in <code>aten/src/ATen/native/cuda/Copy.cu</code>, read on
            2026-11-20 (notes/today/ch02-facts.md, D4). That weights and caches are kept resident for this reason is our
            account of common practice, not a quotation.
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Cùng hàm stub trên GPU của AMD và Apple">The same stub on AMD and Apple GPUs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trên AMD, hàm stub chính là hàm này với những cái tên khác. Tài liệu của AMD mô tả trình tự thao tác điển hình của một chương trình host HIP là: khởi tạo, cấp phát bộ nhớ trên host và device, truyền dữ liệu đầu vào sang device, khởi chạy kernel, đồng bộ hóa, xử lý lỗi, dọn dẹp. Các lời gọi là <code>hipMalloc</code>,{' '}<code>hipMemcpy</code> và <code>hipFree</code>, với cùng tham số theo cùng thứ tự.
          </p>
        }
      >
        <p>
          On AMD the stub is this one with other names. AMD's documentation describes the typical sequence of operations
          for a HIP host program as: initialise, allocate memory on the host and the device, transfer the input data to
          the device, launch the kernel, synchronise, handle errors, clean up. The calls are <code>hipMalloc</code>,{' '}
          <code>hipMemcpy</code> and <code>hipFree</code>, with the same parameters in the same order.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Các phát biểu về HIP lấy từ trang mô hình lập trình và header runtime của AMD (notes/today/ch02-facts.md, B1 và B3), đọc ngày 2026-11-20. Chưa lab nào ở đây chạy trên một device HIP.
            </p>
          }
        >
          <p>
            The book uses CUDA only. The HIP statements are from AMD's programming-model page and runtime header
            (notes/today/ch02-facts.md, B1 and B3), read on 2026-11-20. No lab here has run on a HIP device.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Trên Apple silicon, bức tranh thay đổi ở Phần 1 và Phần 3, như phần cuối của hoạt hình cho thấy. Một buffer được tạo với lưu trữ chia sẻ là một vùng nhớ mà cả CPU lẫn GPU đều dùng. Nếu dữ liệu của bạn được tạo trong một buffer như vậy, không gì phải đi qua đâu cả. Nhưng một hàm stub giữ <em>giao diện này</em>, mảng host thông thường vào và ra, vẫn phải đưa các mảng đó vào buffer: bằng một lần sao chép cho mỗi mảng, hoặc bằng cách bọc vùng nhớ tình cờ được căn theo trang. Vì vậy giao diện tự nhiên trên Metal là buffer, không phải mảng host.{' '}<a href={href('ch02', '2.4')}>Mục 2.4</a> đi qua cả hai phiên bản từng lời gọi một.
          </p>
        }
      >
        <p>
          On Apple silicon the picture changes at Part 1 and Part 3, as the last part of the animation showed. A buffer
          created with shared storage is one piece of memory that both the CPU and the GPU use. If your data is created
          in such a buffer, nothing crosses anything. But a stub that keeps <em>this</em> interface, ordinary host arrays
          in and out, still has to get those arrays into buffers: by one copy each, or by wrapping memory that happens to
          be page-aligned. So the natural interface on Metal is buffers, not host arrays.{' '}
          <a href={href('ch02', '2.4')}>Section 2.4</a> steps through both versions call by call.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Metal không có trong sách. Lưu trữ chia sẻ, <code>newBufferWithBytes</code> và{' '}<code>newBufferWithBytesNoCopy</code> đúng như Apple mô tả (notes/today/ch02-facts.md, C2 và C3; ch05-facts.md, C6).
            </p>
          }
        >
          <p>
            Metal is not in the book. Shared storage, <code>newBufferWithBytes</code> and{' '}
            <code>newBufferWithBytesNoCopy</code> are as Apple documents them (notes/today/ch02-facts.md, C2 and C3;
            ch05-facts.md, C6).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS23['ch02/2.3/q-interface']} />

      <h2>
        <Translated vi="Phép toán nào bù lại được một lượt đi về">Which operations repay a round trip</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đôi khi một lượt đi về là lựa chọn duy nhất: dữ liệu bắt đầu từ một tệp hoặc một tokenizer, và câu trả lời phải đến được CPU. Một kernel có đáng lượt đi đó hay không phụ thuộc vào lượng phép tính nó làm cho mỗi byte được di chuyển, và <a href={href('ch05', '5.1')}>chương 5</a> biến điều đó thành thói quen đếm số phép toán trên mỗi byte. Một câu hỏi dùng nó ngay bây giờ.
          </p>
        }
      >
        <p>
          Sometimes a round trip is the only option: the data starts in a file or a tokenizer, and the answer has to
          reach a CPU. Whether a kernel is worth that trip depends on how much arithmetic it does for each byte moved,
          and <a href={href('ch05', '5.1')}>chapter 5</a> turns that into a habit of counting operations per byte. One
          question uses it now.
        </p>
      </Translated>
      <Ask q={questionsS23['ch02/2.3/q-pays']} />
      <Translated
        vi={
          <p>
            Ba phần vẫn chỉ là các chú thích. <a href={href('ch02', '2.4')}>Mục 2.4</a> điền Phần 1 và Phần 3 bằng các lời gọi thật, và <a href={href('ch02', 'labs')}>lab 2.1</a> yêu cầu bạn viết cả ba.
          </p>
        }
      >
        <p>
          The three parts are still only comments. <a href={href('ch02', '2.4')}>Section 2.4</a> fills in Parts 1 and 3
          with real calls, and <a href={href('ch02', 'labs')}>lab 2.1</a> has you write all three.
        </p>
      </Translated>
    </>
  );
}
