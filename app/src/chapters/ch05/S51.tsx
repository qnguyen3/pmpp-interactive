import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';

// ---------------------------------------------------------------- hardware
interface Hardware {
  key: string;
  label: string;
  /** Peak arithmetic throughput, GFLOP/s. */
  peak: number;
  /** Peak global-memory bandwidth, GB/s. */
  bw: number;
  note: string;
}
// The book's worked example (p. 94). Used only as the book's example.
const BOOK: Hardware = {
  key: 'book',
  get label() {
    return tr("The book's example (Hopper H100)", 'Ví dụ của sách (Hopper H100)');
  },
  peak: 66900,
  bw: 3350,
  get note() {
    return tr(
      "Figures the book quotes on p. 94: 66.9 TFLOPS single precision, 3.35 TB/s.",
      'Số liệu sách nêu ở p. 94: 66.9 TFLOPS độ chính xác đơn, 3.35 TB/s.',
    );
  },
};

function measured(reports: Reports): Hardware[] {
  return realBackends(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const peak = num(r?.compute_gflops);
    const bw = num(r?.bandwidth_gb_per_s);
    if (!peak || !bw) return [];
    return [
      {
        key: b,
        // Getters: measured() runs inside a useMemo keyed on reports, so text is read when rendered, in the current language.
        get label() {
          return tr(
            `Yours: ${String(reports[b].device.name ?? b)} (${b})`,
            `Của bạn: ${String(reports[b].device.name ?? b)} (${b})`,
          );
        },
        peak,
        bw,
        get note() {
          return tr(
            `Measured by lab 5.1 on ${reports[b].generated.slice(0, 10)}: two simple probe kernels, not the vendor's peak figures.`,
            `Đo bằng lab 5.1 ngày ${reports[b].generated.slice(0, 10)}: hai kernel thăm dò đơn giản, không phải số liệu đỉnh của nhà sản xuất.`,
          );
        },
      },
    ];
  });
}

// ---------------------------------------------------------------- roofline plot
interface Mark {
  label: string;
  x: number; // FLOP/B
  y?: number; // achieved GFLOP/s, if measured
  tone: 'model' | 'measured';
}

function Roofline({ hw, marks }: { hw: Hardware; marks: Mark[] }) {
  const W = 700;
  const H = 380;
  const pad = { l: 64, r: 16, t: 14, b: 44 };
  const xMin = 0.03;
  const xMax = 3000;
  const yMax = hw.peak * 3;
  const yMin = hw.bw * xMin * 0.6;
  const X = (v: number) => pad.l + ((Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const knee = hw.peak / hw.bw;
  const roof = (x: number) => Math.min(hw.peak, hw.bw * x);
  const xt = [0.1, 1, 10, 100, 1000];
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e++) yt.push(10 ** e);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Roofline plot', 'Biểu đồ roofline')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {/* the two regimes */}
      <rect x={pad.l} y={pad.t} width={X(knee) - pad.l} height={H - pad.t - pad.b} fill="var(--global-wash)" />
      <text x={pad.l + 8} y={pad.t + 16} fontSize="11" style={{ fill: 'var(--global)' }}>{tr('memory-bound', 'bị giới hạn bởi bộ nhớ')}</text>
      <text x={X(knee) + 8} y={pad.t + 16} fontSize="11" style={{ fill: 'var(--reg)' }}>{tr('compute-bound', 'bị giới hạn bởi tính toán')}</text>
      {xt.map((v) => (
        <g key={'x' + v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle">{v}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{v >= 1000 ? v / 1000 + 'k' : v}</text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="11.5" textAnchor="middle">
        {tr('operations per byte of global-memory traffic (FLOP/B)', 'số phép toán trên mỗi byte lưu lượng bộ nhớ toàn cục (FLOP/B)')}
      </text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}>{tr('throughput (GFLOP/s)', 'thông lượng (GFLOP/s)')}</text>
      {/* the roof */}
      <polyline
        points={`${X(xMin)},${Y(hw.bw * xMin)} ${X(knee)},${Y(hw.peak)} ${X(xMax)},${Y(hw.peak)}`}
        fill="none" stroke="var(--ink)" strokeWidth="2.2" />
      <line x1={X(knee)} x2={X(knee)} y1={Y(hw.peak)} y2={H - pad.b} stroke="var(--ink)" strokeDasharray="3 3" />
      <text x={X(knee)} y={H - pad.b - 6} fontSize="11" textAnchor="middle">
        <Translated vi={<>ngưỡng {fmt(knee)}</>}>threshold {fmt(knee)}</Translated>
      </text>
      {marks.map((m, i) => {
        const cx = X(Math.min(Math.max(m.x, xMin), xMax));
        const top = roof(m.x);
        if (m.tone === 'model')
          return (
            <g key={i}>
              <line x1={cx} x2={cx} y1={Y(top)} y2={H - pad.b} stroke="var(--ink-3)" strokeWidth="1" />
              <circle cx={cx} cy={Y(top)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
              <text x={cx + 7} y={Y(top) + (i % 2 ? 16 : -8)} fontSize="11">{m.label}</text>
            </g>
          );
        const above = (m.y ?? 0) > top * 1.02;
        return (
          <g key={i}>
            <rect x={cx - 4.5} y={Y(m.y!) - 4.5} width="9" height="9" transform={`rotate(45 ${cx} ${Y(m.y!)})`}
              fill={above ? 'var(--fail)' : 'var(--ok)'} />
            <text x={cx + 9} y={Y(m.y!) + 4} fontSize="11" style={{ fill: above ? 'var(--fail)' : 'var(--ok)' }}>{m.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function RooflineBench() {
  const { reports } = useReports();
  const options = useMemo(() => [BOOK, ...measured(reports)], [reports]);
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState({ peak: 1000, bw: 100 });
  const [tile, setTile] = useState(16);
  const [n, setN] = useState(1024);
  const [showMeasured, setShowMeasured] = useState(true);
  const hw: Hardware =
    key === 'custom'
      ? { key: 'custom', label: tr('Custom', 'Tùy chỉnh'), ...custom, note: tr('Numbers you typed.', 'Các số bạn đã nhập.') }
      : (options.find((o) => o.key === key) ?? BOOK);
  const knee = hw.peak / hw.bw;

  const marks: Mark[] = [
    { label: tr('vector add', 'cộng vector'), x: 1 / 12, tone: 'model' },
    { label: tr('naive matmul', 'matmul ngây thơ'), x: 0.25, tone: 'model' },
    { label: tr(`tiled, T=${tile}`, `phân ô, T=${tile}`), x: 0.25 * tile, tone: 'model' },
    { label: tr(`ideal matmul, N=${n}`, `matmul lý tưởng, N=${n}`), x: n / 6, tone: 'model' },
  ];
  // Your own measurements, when this hardware is a device lab 5.1 ran on.
  const run = reports[hw.key];
  const naive = num(benchRows(run, 'ch05/lab1_roofline').find((x) => x.name === 'roofline')?.naive_gflops);
  const tiled = benchRows(run, 'ch05/lab2_tiled').filter((x) => x.name === 'tiled');
  if (showMeasured && naive) marks.push({ label: tr(`naive, measured ${fmt(naive)}`, `ngây thơ, đo được ${fmt(naive)}`), x: 0.25, y: naive, tone: 'measured' });
  if (showMeasured)
    for (const t of tiled) {
      const g = num(t.gflops);
      const T = num(t.tile);
      if (g && T === tile) marks.push({ label: tr(`T=${T}, measured ${fmt(g)}`, `T=${T}, đo được ${fmt(g)}`), x: 0.25 * T, y: g, tone: 'measured' });
    }
  const row = (label: string, x: number) => (
    <tr key={label}>
      <td>{label}</td>
      <td className="num">{fmt(x)}</td>
      <td>
        {x < knee ? (
          <span className="chip global">
            <Translated vi="bị giới hạn bởi bộ nhớ">memory-bound</Translated>
          </span>
        ) : (
          <span className="chip reg">
            <Translated vi="bị giới hạn bởi tính toán">compute-bound</Translated>
          </span>
        )}
      </td>
      <td className="num">{fmt(Math.min(hw.peak, hw.bw * x))}</td>
      <td className="num">{fmt((100 * Math.min(hw.peak, hw.bw * x)) / hw.peak)}%</td>
    </tr>
  );
  return (
    <Bench
      title={tr('Roofline', 'Roofline')}
      task={tr(
        'Pick the hardware, then move the tile size until the tiled kernel crosses the threshold.',
        'Chọn phần cứng, rồi thay đổi kích thước ô cho đến khi kernel phân ô vượt qua ngưỡng.',
      )}
      foot={
        <Translated
          vi={
            <>
              Cả hai trục đều theo thang log. Vòng tròn rỗng là trần theo mô hình (mức tốt nhất mà một kernel có tỉ lệ
              đó có thể đạt); hình thoi đặc là số đo từ các lab của bạn. {hw.note}
            </>
          }
        >
          <>
            Log scales on both axes. Open circles are ceilings from the model (the best a kernel with that ratio could
            do); filled diamonds are measurements from your labs. {hw.note}
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          <Translated vi="Phần cứng">Hardware</Translated>
          <select value={key} onChange={(e) => setKey(e.target.value)}>
            {options.map((o) => (
              <option key={o.key} value={o.key}>{o.label}</option>
            ))}
            <option value="custom">{tr('Custom numbers', 'Số tùy chỉnh')}</option>
          </select>
        </label>
        {key === 'custom' ? (
          <>
            <label className="field">
              <Translated vi="Thông lượng đỉnh (GFLOP/s)">Peak throughput (GFLOP/s)</Translated>
              <input type="number" min="1" value={custom.peak} onChange={(e) => setCustom({ ...custom, peak: Math.max(1, +e.target.value) })} />
            </label>
            <label className="field">
              <Translated vi="Băng thông bộ nhớ (GB/s)">Memory bandwidth (GB/s)</Translated>
              <input type="number" min="1" value={custom.bw} onChange={(e) => setCustom({ ...custom, bw: Math.max(1, +e.target.value) })} />
            </label>
          </>
        ) : null}
        <label className="field">
          <Translated vi={<>Độ rộng ô T = {tile}</>}>Tile width T = {tile}</Translated>
          <input type="range" min="1" max="128" value={tile} onChange={(e) => setTile(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi={<>Độ rộng ma trận N = {n}</>}>Matrix width N = {n}</Translated>
          <input type="range" min="64" max="8192" step="64" value={n} onChange={(e) => setN(+e.target.value)} />
        </label>
        {run ? (
          <label className="check">
            <input type="checkbox" checked={showMeasured} onChange={(e) => setShowMeasured(e.target.checked)} />
            <Translated vi="hiện số đo của tôi">show my measurements</Translated>
          </label>
        ) : null}
      </div>
      <div className="cols plot">
        <Roofline hw={hw} marks={marks} />
        <div className="panel">
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th>
                  <Translated vi="Kernel">Kernel</Translated>
                </th>
                <th className="num">FLOP/B</th>
                <th>
                  <Translated vi="Chế độ">Regime</Translated>
                </th>
                <th className="num">
                  <Translated vi="Trần, GFLOP/s">Ceiling, GFLOP/s</Translated>
                </th>
                <th className="num">
                  <Translated vi="so với đỉnh">of peak</Translated>
                </th>
              </tr>
            </thead>
            <tbody>{marks.filter((m) => m.tone === 'model').map((m) => row(m.label, m.x))}</tbody>
          </table>
          <p className="muted" style={{ margin: '0.7rem 0 0' }}>
            <Translated
              vi={
                <>
                  Đỉnh {fmt(hw.peak)} GFLOP/s, băng thông {fmt(hw.bw)} GB/s, ngưỡng {fmt(knee)} FLOP/B.
                </>
              }
            >
              Peak {fmt(hw.peak)} GFLOP/s, bandwidth {fmt(hw.bw)} GB/s, threshold {fmt(knee)} FLOP/B.
            </Translated>
          </p>
          {marks.some((m) => m.tone === 'measured' && (m.y ?? 0) > Math.min(hw.peak, hw.bw * m.x) * 1.02) ? (
            <p style={{ margin: '0.7rem 0 0' }}>
              <Translated
                vi={
                  <>
                    <span className="chip fail">trên mái</span> Một điểm đo nằm trên đường mà mô hình nói không gì
                    vượt qua được. Mô hình không sai về phần cứng; nó sai về kernel. Xem bên dưới.
                  </>
                }
              >
                <span className="chip fail">above the roof</span> A measured point sits above the line the model says
                nothing can cross. The model is not wrong about the hardware; it is wrong about the kernel. See below.
              </Translated>
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- linear layer
function LinearLayerBench() {
  const { reports } = useReports();
  const [batch, setBatch] = useState(1);
  const [bytes, setBytes] = useState(4);
  const k = 1024;
  const l = 4096;
  const ai = (j: number, b: number) => (2 * j * k * l) / (k * l * b + j * (k + l) * b);
  const backends = realBackends(reports);
  const rows = backends.flatMap((b) =>
    benchRows(reports[b], 'ch05/lab3_gemm').filter((r) => r.name === 'linear_layer').map((r) => ({ b, r })),
  );
  return (
    <Bench
      title={tr('A linear layer, by batch size', 'Một lớp tuyến tính, theo kích thước batch')}
      task={tr(
        "Find the batch size at which this layer stops being limited by memory on the book's example GPU (threshold 20 FLOP/B).",
        'Tìm kích thước batch mà tại đó lớp này thôi bị giới hạn bởi bộ nhớ trên GPU ví dụ của sách (ngưỡng 20 FLOP/B).',
      )}
      foot={tr(
        "Ideal traffic only: every weight, input and output touched exactly once. Layer shape from the book's own example in §20.6 (p. 505), which uses 16-bit values.",
        'Chỉ tính lưu lượng lý tưởng: mỗi trọng số, đầu vào và đầu ra được chạm đúng một lần. Kích thước lớp lấy từ chính ví dụ của sách ở §20.6 (p. 505), vốn dùng giá trị 16-bit.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi={<>Kích thước batch (số hàng đi qua lớp cùng lúc) = {batch}</>}>
            Batch size (rows through the layer at once) = {batch}
          </Translated>
          <input type="range" min="0" max="10" value={Math.log2(batch)} onChange={(e) => setBatch(2 ** +e.target.value)} />
        </label>
        <label className="field">
          <Translated vi="Số byte mỗi giá trị">Bytes per value</Translated>
          <span className="seg">
            {[4, 2, 1, 0.5].map((b) => (
              <button key={b} aria-pressed={bytes === b} onClick={() => setBytes(b)}>
                {b === 4 ? '4 (float32)' : b === 2 ? '2 (16-bit)' : b === 1 ? '1 (int8)' : '0.5 (4-bit)'}{/* i18n-ignore: numbers and type names */}
              </button>
            ))}
          </span>
        </label>
      </div>
      <div className="cols">
        <div className="panel">
          <Translated
            vi={
              <p style={{ margin: 0 }}>
                {k} đầu vào → {l} đầu ra. Số phép toán: 2 × {batch} × {k} × {l} = {fmt(2 * batch * k * l, 4)}. Số byte:{' '}
                {fmt(k * l * bytes, 4)} của trọng số + {fmt(batch * (k + l) * bytes, 4)} của activation.
              </p>
            }
          >
            <p style={{ margin: 0 }}>
              {k} inputs → {l} outputs. Operations: 2 × {batch} × {k} × {l} = {fmt(2 * batch * k * l, 4)}. Bytes:{' '}
              {fmt(k * l * bytes, 4)} of weights + {fmt(batch * (k + l) * bytes, 4)} of activations.
            </p>
          </Translated>
          <p style={{ fontSize: '1.5rem', margin: '0.6rem 0 0.2rem', fontWeight: 600 }}>{fmt(ai(batch, bytes))} FLOP/B</p>
          <p className="muted" style={{ margin: 0 }}>
            {ai(batch, bytes) < 20 ? (
              <span className="chip global">
                <Translated vi="bị giới hạn bởi bộ nhớ">memory-bound</Translated>
              </span>
            ) : (
              <span className="chip reg">
                <Translated vi="bị giới hạn bởi tính toán">compute-bound</Translated>
              </span>
            )}{' '}
            <Translated vi="so với ngưỡng 20 FLOP/B của sách.">against the book's 20 FLOP/B threshold.</Translated>
          </p>
        </div>
        <div className="panel">
          {rows.length ? (
            <>
              <Translated
                vi={
                  <p style={{ margin: '0 0 0.4rem' }}>
                    <b>Số đo lab 5.3 của bạn</b> (float32, kernel phân ô)
                  </p>
                }
              >
                <p style={{ margin: '0 0 0.4rem' }}>
                  <b>Your lab 5.3 measurements</b> (float32, tiled kernel)
                </p>
              </Translated>
              <table className="data" style={{ margin: 0 }}>
                <thead>
                  <tr>
                    <th>
                      <Translated vi="Backend">Backend</Translated>
                    </th>
                    <th className="num">
                      <Translated vi="Batch">Batch</Translated>
                    </th>
                    <th className="num">GFLOP/s</th>
                    <th className="num">
                      <Translated vi="Số hàng mỗi giây">Rows per second</Translated>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map(({ b, r }, i) => (
                    <tr key={i} style={num(r.batch) === batch ? { background: 'var(--rule-soft)' } : undefined}>
                      <td>{b}</td>
                      <td className="num">{String(r.batch)}</td>
                      <td className="num">{fmt(num(r.gflops) ?? NaN)}</td>
                      <td className="num">{fmt(num(r.rows_per_s) ?? NaN)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : (
            <Translated
              vi={
                <p style={{ margin: 0 }}>
                  Chưa có số đo. Lab 5.3 đo thời gian chính lớp này ở các kích thước batch từ 1 đến 1024 trên thiết bị của
                  bạn (<a href={href('ch05', 'labs')}>lab</a>).
                </p>
              }
            >
              <p style={{ margin: 0 }}>
                No measurements yet. Lab 5.3 times exactly this layer at batch sizes 1 to 1024 on your device (
                <a href={href('ch05', 'labs')}>labs</a>).
              </p>
            </Translated>
          )}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questions51: Record<string, Question> = {
  'ch05/5.1/q-regime': {
    kind: 'choice',
    id: 'ch05/5.1/q-regime',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel thực hiện 36 phép toán dấu phẩy động và bảy lần truy cập bộ nhớ toàn cục 4 byte cho mỗi luồng. Trên
            thiết bị A, mức đỉnh là 200 GFLOP/s và 100 GB/s; trên thiết bị B là 300 GFLOP/s và 250 GB/s. Nó bị giới hạn
            bởi bộ nhớ ở đâu?
          </>
        }
      >
        <>
          A kernel performs 36 floating-point operations and seven 4-byte global-memory accesses per thread. On device A
          the peaks are 200 GFLOP/s and 100 GB/s; on device B, 300 GFLOP/s and 250 GB/s. Where is it memory-bound?
        </>
      </Translated>
    ),
    options: [
      {
        label: <Translated vi="Trên cả hai.">On both.</Translated>,
        why: (
          <Translated vi="Hãy tính từng ngưỡng: thông lượng đỉnh chia cho băng thông đỉnh.">
            Compute each threshold: peak throughput divided by peak bandwidth.
          </Translated>
        ),
      },
      { label: <Translated vi="Chỉ trên A.">On A only.</Translated>, correct: true },
      {
        label: <Translated vi="Chỉ trên B.">On B only.</Translated>,
        why: (
          <Translated vi="B có nhiều băng thông hơn trên mỗi đơn vị tính toán, nên ngưỡng của nó thấp hơn chứ không cao hơn.">
            B has more bandwidth per unit of compute, so its threshold is lower, not higher.
          </Translated>
        ),
      },
      {
        label: <Translated vi="Không ở đâu cả.">On neither.</Translated>,
        why: (
          <Translated vi="A cần 2 FLOP/B để giữ các lõi luôn bận; kernel này chỉ cung cấp 1.29.">
            A needs 2 FLOP/B to keep its cores busy; this kernel supplies 1.29.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tỉ lệ của kernel là 36 / 28 = 1.29 FLOP/B. Ngưỡng của A là 200 / 100 = 2.0, nên trên A hệ thống bộ nhớ không
            cấp đủ dữ liệu cho các lõi: bị giới hạn bởi bộ nhớ. Ngưỡng của B là 300 / 250 = 1.2, nên trên B cùng kernel đó
            bị giới hạn bởi tính toán. Chế độ thuộc về cặp (kernel, thiết bị), không bao giờ chỉ thuộc về riêng kernel.
          </p>
        }
      >
        <p>
          The kernel's ratio is 36 / 28 = 1.29 FLOP/B. A's threshold is 200 / 100 = 2.0, so on A the memory system cannot
          feed the cores: memory-bound. B's threshold is 300 / 250 = 1.2, so on B the same kernel is compute-bound. The
          regime belongs to the pair (kernel, device), never to the kernel alone.
        </p>
      </Translated>
    ),
    source: 'ch. 5 exercise 9, p. 119',
    vi: { source: 'ch. 5 bài tập 9, p. 119' },
  },
  'ch05/5.1/q-speedoflight': {
    kind: 'numeric',
    id: 'ch05/5.1/q-speedoflight',
    prompt: (
      <Translated
        vi={
          <>
            Một kernel cộng vector cộng hai vector, mỗi vector 250 triệu số float, trong 2.0 ms trên một thiết bị có băng
            thông bộ nhớ toàn cục đỉnh là 2000 GB/s. Con số đó bằng bao nhiêu phần trăm tốc độ ánh sáng (speed of light)?
          </>
        }
      >
        <>
          A vector-addition kernel adds two vectors of 250 million floats in 2.0 ms on a device whose global-memory
          bandwidth peaks at 2000 GB/s. What percentage of the speed of light is that?
        </>
      </Translated>
    ),
    answer: 75,
    tol: 0.02,
    unit: '%',
    explain: (
      <Translated
        vi={
          <p>
            Ba mảng, mỗi mảng 250 triệu số float 4 byte, là 3 GB được di chuyển. 3 GB / 2.0 ms = 1500 GB/s, và 1500 / 2000 =
            75%. Vậy kernel này trên thiết bị này chỉ còn tăng tốc được tối đa 1.33×, dù mã có khéo đến đâu. Giới hạn đó
            chính là mục đích của phép phân tích: nó cho bạn biết khi nào nên dừng.
          </p>
        }
      >
        <p>
          Three arrays of 250 million 4-byte floats is 3 GB moved. 3 GB / 2.0 ms = 1500 GB/s, and 1500 / 2000 = 75%. So
          at most a 1.33× speed-up is left for this kernel on this device, however clever the code. That bound is what
          the analysis is for: it tells you when to stop.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.1, p. 96',
  },
  'ch05/5.1/q-decode': {
    kind: 'choice',
    id: 'ch05/5.1/q-decode',
    prompt: (
      <Translated
        vi={
          <>
            Sinh một token ở kích thước batch 1 đẩy một hàng duy nhất qua mỗi ma trận trọng số (một phép nhân ma trận với
            vector). Giả sử bạn viết được kernel tốt nhất có thể cho việc này. Phát biểu nào đúng?
          </>
        }
      >
        <>
          Generating one token at batch size 1 pushes a single row through each weight matrix (a matrix-vector product).
          Suppose you write the best possible kernel for it. Which statement holds?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Phân ô với ô đủ lớn sẽ khiến nó bị giới hạn bởi tính toán, như với matmul ma trận vuông.">
            Tiling with a large enough tile makes it compute-bound, as it does for square matmul.
          </Translated>
        ),
        why: (
          <Translated vi="Phân ô loại bỏ những lần nạp lặp lại cùng một phần tử. Ở đây mỗi trọng số chỉ cần đúng một lần cho mỗi token; không có gì để dùng lại.">
            Tiling removes repeated fetches of the same element. Here each weight is needed exactly once per token; there
            is nothing to reuse.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó vẫn bị giới hạn bởi bộ nhớ: mỗi trọng số được dùng một lần, nên tỉ lệ không thể vượt quá khoảng 2 phép toán trên mỗi trọng số được nạp.">
            It stays memory-bound: each weight is used once, so the ratio cannot rise above about 2 operations per weight
            fetched.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó bị giới hạn bởi tính toán, vì ma trận lớn và bài toán lớn thì có cường độ cao.">
            It is compute-bound, because the matrix is large and large problems have high intensity.
          </Translated>
        ),
        why: (
          <Translated vi="Với matmul ma trận vuông, tỉ lệ lý tưởng tăng theo N vì mỗi phần tử tham gia N tích. Trong phép nhân ma trận với vector, mỗi trọng số chỉ tham gia một.">
            For square matmul the ideal ratio grows with N because each element takes part in N products. In a
            matrix-vector product each weight takes part in one.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Một phép nhân và một phép cộng cho mỗi trọng số, và ít nhất một lần nạp trọng số đó: 2 FLOP trên 4 byte = 0.5
              FLOP/B với float32, dù kernel được viết thế nào. Cách duy nhất để tăng là cho mỗi trọng số phục vụ nhiều hàng
              cùng lúc hơn (batching) hoặc làm mỗi trọng số nhỏ đi (lượng tử hóa). Cả hai đều hiện ra trực tiếp trong trình
              khám phá ở trên.
            </p>
          </>
        }
      >
        <>
          <p>
            One multiply and one add per weight, and at least one fetch of that weight: 2 FLOP per 4 bytes = 0.5 FLOP/B
            for float32, however the kernel is written. The only ways up are to make each weight serve more rows at once
            (batching) or to make each weight smaller (quantisation). Both show up directly in the explorer above.
          </p>
        </>
      </Translated>
    ),
    source: 'our analysis, using the §5.1 method; the book reaches the same conclusion for LLM decoding in §20.6, p. 505',
    vi: {
      source:
        'phân tích của khóa học này, theo phương pháp của §5.1; sách đi đến cùng kết luận cho việc decode LLM ở §20.6, p. 505',
    },
  },
};

export function S51() {
  return (
    <>
      <Translated
        vi={
          <p>
            Một GPU có hai giới hạn tốc độ quan trọng ở đây. Một là nó làm được bao nhiêu phép toán số học mỗi giây. Hai là nó
            chuyển được bao nhiêu byte vào hoặc ra khỏi một bộ nhớ nhất định mỗi giây. Kernel sẽ chạm vào giới hạn nào nó gặp
            trước.
          </p>
        }
      >
        <p>
          A GPU has two speed limits that matter here. One is how many arithmetic operations it can do per second. The
          other is how many bytes it can move in or out of a given memory per second. A kernel runs into whichever one it
          reaches first.
        </p>
      </Translated>
      <Source section="5.1" pages="93–94" />
      <Translated
        vi={
          <p>
            Nếu các lõi (core) luôn có sẵn toán hạng chờ, kernel <b>bị giới hạn bởi tính toán</b>: nó chạy nhanh đến mức các đơn
            vị số học cho phép. Nếu các lõi ngồi không chờ toán hạng tới, nó <b>bị giới hạn bởi bộ nhớ</b>: nó chạy nhanh đến mức
            bộ nhớ cấp được. Trường hợp nào xảy ra phụ thuộc vào một thuộc tính duy nhất của kernel: nó thực hiện bao nhiêu phép
            toán dấu phẩy động cho mỗi byte nó kéo từ bộ nhớ toàn cục (global memory). Sách gọi đây là tỉ lệ tính toán trên truy
            cập bộ nhớ toàn cục (compute-to-global-memory-access ratio) và đo nó bằng FLOP/B; bạn cũng sẽ gặp tên gọi cường độ
            tính toán (arithmetic intensity).
          </p>
        }
      >
        <p>
          If the cores always have operands waiting, the kernel is <b>compute-bound</b>: it goes as fast as the arithmetic
          units allow. If the cores sit idle waiting for operands to arrive, it is <b>memory-bound</b>: it goes as fast
          as memory can deliver. Which one applies depends on a single property of the kernel: how many floating-point
          operations it performs for each byte it pulls from global memory. The book calls this the
          compute-to-global-memory-access ratio and measures it in FLOP/B; you will also see it called arithmetic
          intensity.
        </p>
      </Translated>

      <h2>
        <Translated vi="Đếm thử với một kernel bạn đã biết">Count it for a kernel you know</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cộng vector, <code>C[i] = A[i] + B[i]</code>, trên số float 4 byte: mỗi luồng (thread) nạp hai giá trị, cộng chúng và
            ghi một giá trị.
          </p>
        }
      >
        <p>
          Vector addition, <code>C[i] = A[i] + B[i]</code>, on 4-byte floats: each thread loads two values, adds them and
          stores one.
        </p>
      </Translated>
      <Source section="5.1" pages="96" />
      <Predict
        id="ch05/5.1/p-vecadd"
        prompt={
          <Translated vi="Nó thực hiện bao nhiêu phép toán dấu phẩy động trên mỗi byte bộ nhớ toàn cục mà nó chạm tới?">
            How many floating-point operations does it perform per byte of global memory it touches?
          </Translated>
        }
        unit="FLOP/B"
      >
        {(v) => (
          <p>
            <Translated vi={<>Một phép cộng cho 12 byte (nạp 4, nạp 4, ghi 4): 1/12 ≈ <b>0.083 FLOP/B</b>.</>}>
              One addition for 12 bytes (4 loaded, 4 loaded, 4 stored): 1/12 ≈ <b>0.083 FLOP/B</b>.
            </Translated>{' '}
            {Math.abs(v - 1 / 12) < 0.01
              ? tr('That is what you had.', 'Đúng như bạn đã dự đoán.')
              : v > 0.2
                ? tr(
                    'If you were higher, check whether you counted the store: writing C[i] is traffic too.',
                    'Nếu bạn ra số cao hơn, hãy xem bạn có tính lần ghi không: ghi C[i] cũng là lưu lượng.',
                  )
                : tr(
                    'Count bytes, not accesses: each access moves 4 of them.',
                    'Hãy đếm byte, không phải số lần truy cập: mỗi lần truy cập di chuyển 4 byte.',
                  )}
          </p>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Giờ đến phía thiết bị. Chia thông lượng đỉnh cho băng thông đỉnh, bạn được tỉ lệ mà tại đó hai giới hạn gặp nhau. Một
            kernel ở trên <b>ngưỡng</b> đó cho các lõi nhiều việc trên mỗi byte hơn mức bộ nhớ có thể bỏ đói chúng; một kernel ở
            dưới ngưỡng để các lõi phải chờ. Với GPU mà sách dùng làm ví dụ xuyên suốt, ngưỡng là khoảng 20 FLOP/B. Cộng vector ở
            mức 0.083 còn thiếu tới hai trăm lần.
          </p>
        }
      >
        <p>
          Now the device side. Divide peak throughput by peak bandwidth and you get the ratio at which the two limits
          meet. A kernel above that <b>threshold</b> gives the cores more work per byte than memory can starve them of; a
          kernel below it leaves cores waiting. For the GPU the book uses as its running example, the threshold is about
          20 FLOP/B. Vector addition at 0.083 is two hundred times short of it.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against nvidia.com and the CUDA 13.5 docs (notes/today/ch05-facts.md, A8)',
          '2026-11-20, đối chiếu với nvidia.com và tài liệu CUDA 13.5 (notes/today/ch05-facts.md, A8)',
        )}
      >
        <Translated
          vi={
            <p>
              Thiết bị của sách là NVIDIA H100. Trang sản phẩm của NVIDIA xác nhận 3.35 TB/s cho phiên bản SXM và ghi thông lượng
              FP32 là 67 teraFLOPS; con số 66.9 của sách không xuất hiện trong nguồn nào của NVIDIA mà chúng tôi tìm thấy. Thiết bị
              của bạn có ngưỡng riêng: lab 5.1 đo nó.
            </p>
          }
        >
          <p>
            The book's device is an NVIDIA H100. NVIDIA's product page confirms 3.35 TB/s for the SXM variant and prints
            FP32 throughput as 67 teraFLOPS; the book's 66.9 appears in no NVIDIA source we found. Your device has its own
            threshold: lab 5.1 measures it.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Roofline">The roofline</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đặt tỉ lệ của kernel lên trục ngang và thông lượng lên trục dọc. Phần cứng vẽ ra một mái: một đoạn dốc nơi băng thông
            là giới hạn (thông lượng = băng thông × tỉ lệ) và một đỉnh phẳng nơi phép tính số học là giới hạn. Mọi kernel đều nằm
            dưới mái đó. Nằm dưới bao xa cho bạn biết nó được viết tốt đến đâu; phần nào của mái nằm ngay phía trên cho bạn biết
            loại cải tiến nào sẽ có ích.
          </p>
        }
      >
        <p>
          Put the kernel's ratio on the horizontal axis and throughput on the vertical. The hardware draws a roof: a
          slope where bandwidth is the limit (throughput = bandwidth × ratio) and a flat top where arithmetic is. Every
          kernel lives under it. How far under tells you how well it is written; which part of the roof is overhead tells
          you what kind of work would help.
        </p>
      </Translated>
      <Source section="5.1" pages="95–96" figs={tr('the Roofline Model sidebar', 'khung bên lề "Roofline Model"')} />
      <RooflineBench />
      <Translated
        vi={
          <p>
            Ba điều cần để ý. Một kernel bị giới hạn bởi bộ nhớ nằm sát đoạn dốc đã hiệu quả rồi, và cách duy nhất để nó nhanh hơn
            là tăng tỉ lệ của nó, dịch nó sang phải. Một kernel nằm xa dưới mái còn chỗ để tận dụng phần cứng tốt hơn ngay với tỉ
            lệ hiện tại. Và các mức đỉnh là giới hạn, không phải lời hứa: không có gì bảo đảm một kernel bị giới hạn bởi tính toán
            sẽ chạm tới phần phẳng.
          </p>
        }
      >
        <p>
          Three things to notice. A memory-bound kernel sitting near the slope is already efficient, and the only way to
          make it faster is to raise its ratio, moving it right. A kernel far below the roof has room to use the hardware
          better as it is. And the peaks are limits, not promises: nothing guarantees a compute-bound kernel reaches the
          flat part.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mái có hai đoạn thẳng vì nó chỉ có hai con số. Thiết bị thật có nhiều bộ nhớ với băng thông khác nhau, và mức đỉnh khác
              nhau cho các kiểu dữ liệu khác nhau. Phiên bản của sách, cũng như phiên bản này, chỉ dùng độ chính xác đơn và bộ nhớ
              toàn cục.
            </p>
          }
        >
          <p>
            The roof has two straight segments because it has two numbers. Real devices have several memories with
            different bandwidths, and different peaks for different data types. The book's version, like this one, uses
            single precision and global memory only.
          </p>
        </Translated>
        <Translated
          vi={
            <p>Hình phác của sách dùng trục tuyến tính; ở đây dùng trục log để 0.08 và 170 vừa trong một hình.</p>
          }
        >
          <p>The book's sketch uses linear axes; log axes are used here so that 0.08 and 170 fit in one picture.</p>
        </Translated>
      </Simplification>
      <Ask q={questions51['ch05/5.1/q-regime']} />

      <h2>
        <Translated vi="Tốc độ ánh sáng">Speed of light</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Với một kernel bị giới hạn bởi bộ nhớ, có một cách kiểm tra trực tiếp xem nó tốt đến đâu. Cộng số byte nó phải di
            chuyển, chia cho thời gian nó chạy, rồi so với băng thông đỉnh. Ví dụ của sách: cộng hai vector một tỉ số float di
            chuyển 12 GB; xong trong 4 ms nghĩa là 3 TB/s, tức 90% của 3.35 TB/s trên thiết bị ví dụ. Kernel chạy ở 90% tốc độ ánh
            sáng (speed of light), và không cách viết lại nào giành thêm được quá 10% còn lại.
          </p>
        }
      >
        <p>
          For a memory-bound kernel there is a direct check on how good it is. Add up the bytes it must move, divide by
          the time it took, and compare with the peak bandwidth. The book's example: adding two vectors of a billion
          floats moves 12 GB; finishing in 4 ms means 3 TB/s, which is 90% of the example device's 3.35 TB/s. The kernel
          runs at 90% of the speed of light, and no rewrite can gain more than the remaining 10%.
        </p>
      </Translated>
      <Source section="5.1" pages="96" />
      <Ask q={questions51['ch05/5.1/q-speedoflight']} />

      <h2>
        <Translated vi="Phép nhân ma trận: khoảng cách giữa điều có thể và điều bạn đã viết">
          Matrix multiplication: the gap between what is possible and what you wrote
        </Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đếm khối lượng công việc trong một phép nhân N×N mà không cần nhìn mã. Mỗi đầu ra trong N² đầu ra là một tích vô
            hướng: N phép nhân và N phép cộng. Tổng cộng là 2N³ phép toán. Nếu mỗi phần tử đầu vào được nạp đúng một lần và mỗi
            đầu ra được ghi một lần, lưu lượng sẽ là 3N² số float, tức 12N² byte. Vậy tỉ lệ tốt nhất có thể là 2N³ / 12N² = N/6
            FLOP/B. Với N = 1024, con số đó khoảng 170: nằm xa về bên phải mọi ngưỡng. Về bản chất, phép nhân ma trận là một phép
            tính có thể giữ các đơn vị số học luôn bận.
          </p>
        }
      >
        <p>
          Count the work in an N×N multiplication without looking at any code. Each of the N² outputs is a dot product: N
          multiplies and N adds. That is 2N³ operations. If every input element were fetched exactly once and every
          output stored once, the traffic would be 3N² floats, or 12N² bytes. The best possible ratio is therefore
          2N³ / 12N² = N/6 FLOP/B. At N = 1024 that is about 170: far to the right of any threshold. Matrix
          multiplication is, by nature, a computation the arithmetic units could be kept busy with.
        </p>
      </Translated>
      <Source section="5.1" pages="96–97" />
      <Predict
        id="ch05/5.1/p-naive"
        prompt={
          <Translated
            vi={
              <>
                Giờ đến kernel bạn thực sự có. Vòng lặp trong của nó là{' '}
                <code>Pvalue += M[row*Width+k] * N[k*Width+col]</code>. Tỉ lệ của nó là bao nhiêu, nếu chỉ tính vòng lặp?
              </>
            }
          >
            <>
              Now the kernel you actually have. Its inner loop is{' '}
              <code>Pvalue += M[row*Width+k] * N[k*Width+col]</code>. What is its ratio, counting only the loop?
            </>
          </Translated>
        }
        unit="FLOP/B"
      >
        {(v) => (
          <p>
            <Translated
              vi={
                <>
                  Mỗi lần lặp nạp một phần tử của M và một của N (8 byte) cho một phép nhân và một phép cộng:{' '}
                  <b>0.25 FLOP/B</b>, bất kể N là bao nhiêu.
                </>
              }
            >
              Each iteration loads one element of M and one of N (8 bytes) for one multiply and one add:{' '}
              <b>0.25 FLOP/B</b>, whatever N is.
            </Translated>{' '}
            {Math.abs(v - 0.25) < 0.02
              ? tr('Exactly.', 'Chính xác.')
              : tr(
                  'The matrix size cancels: more elements, proportionally more fetches.',
                  'Kích thước ma trận triệt tiêu: nhiều phần tử hơn thì số lần nạp cũng tăng theo tỉ lệ.',
                )}{' '}
            <Translated
              vi={
                <>
                  Tỉ lệ lý tưởng tăng theo N còn tỉ lệ này thì không, vì kernel nạp lại mỗi phần tử đầu vào cho từng luồng
                  cần đến nó. Trên thiết bị ví dụ của sách, điều đó giới hạn nó ở 3350 × 0.25 ≈ 840 GFLOP/s, nhỉnh hơn 1%
                  của mức đỉnh một chút.
                </>
              }
            >
              The ideal ratio grows with N and this one does not, because the kernel fetches each input element again for
              every thread that needs it. On the book's example device that caps it at 3350 × 0.25 ≈ 840 GFLOP/s, a little
              over 1% of the peak.
            </Translated>
          </p>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Quay lại biểu đồ roofline, chọn phần cứng của sách, và nhìn khoảng cách giữa "matmul ngây thơ" và "matmul lý tưởng".
            Khoảng cách đó là chủ đề của chương này và của chương 15. Phần còn lại của chương này thu hẹp một phần khoảng cách đó
            theo một hệ số do bạn chọn: độ rộng ô.
          </p>
        }
      >
        <p>
          Go back to the roofline, select the book's hardware, and look at the distance between "naive matmul" and "ideal
          matmul". That distance is the subject of this chapter and of chapter 15. The rest of this chapter closes part
          of it by a factor you choose: the tile width.
        </p>
      </Translated>

      <h3>
        <Translated vi="Khi một số đo nằm trên mái">When a measurement lands above the roof</Translated>
      </h3>
      <Translated
        vi={
          <p>
            Nếu bạn đã chạy lab 5.1, hãy chọn thiết bị của chính bạn trong trình khám phá. Trên máy dùng để xây dựng khóa học này,
            kernel ngây thơ đo được kết quả nằm hẳn <em>phía trên</em> mức trần mà mô hình đặt cho nó; trình khám phá cho thấy cao hơn bao
            nhiêu trên máy của bạn. Không có gì hỏng cả. Con số 0.25 tính mọi lần nạp vào bộ nhớ toàn cục, và sách nói ngay sau đó
            rằng tỉ lệ thật cao hơn vì một số lần nạp được phục vụ bởi bộ nhớ đệm (cache) trên chip. Cao hơn bao nhiêu tùy vào
            thiết bị và kích thước ma trận, và có thể là rất nhiều.
          </p>
        }
      >
        <p>
          If you have run lab 5.1, select your own device in the explorer. On the machine this course was built on, the
          naive kernel measured well <em>above</em> the ceiling the model gives it; your explorer shows by how much on
          yours. Nothing broke. The 0.25
          figure charges every load to global memory, and the book says in the same breath that the true ratio is higher
          because some of those loads are served by on-chip caches. How much higher depends on the device and the matrix
          size, and it can be a lot.
        </p>
      </Translated>
      <Source section="5.1" pages="97" />
      <Translated
        vi={
          <p>
            Hãy nắm chắc cả hai nửa của điều đó. Mô hình cho bạn biết một kernel <em>đòi hỏi</em> gì ở bộ nhớ. Chỉ phép đo mới cho
            bạn biết thiết bị thực sự phải nạp những gì. Khi chương này nói phân ô (tiling) cắt lưu lượng đi T lần, đó là một phát
            biểu về phần đòi hỏi; mức tăng tốc bạn đo được ở lab 5.2 sẽ nhỏ hơn ở bất cứ đâu bộ nhớ đệm đã làm sẵn một phần công
            việc.
          </p>
        }
      >
        <p>
          Hold on to both halves of that. The model tells you what a kernel <em>asks</em> of memory. Only a measurement
          tells you what the device actually had to fetch. When this chapter says tiling cuts traffic by a factor of T,
          that is a statement about the asking; the speed-up you measure in lab 5.2 will be smaller wherever caches were
          already doing part of the job.
        </p>
      </Translated>

      <h2>
        <Translated vi="Bên dưới một lớp tuyến tính">Underneath a linear layer</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Cùng cách đếm đó giải thích hai điều bạn đã biết từ phía framework: vì sao sinh token ở kích thước batch 1 chậm trên
            mỗi đơn vị tính toán, và vì sao lượng tử hóa làm nó nhanh hơn ngay cả khi không phép tính nào rẻ đi.
          </p>
        }
      >
        <p>
          The same counting explains two things you already know from the framework side: why generation at batch size 1
          is slow per unit of compute, and why quantisation speeds it up even when no arithmetic gets cheaper.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Phần này áp dụng phương pháp §5.1 của sách cho một phép nhân ma trận chữ nhật. Chính sách cũng làm vậy ở phần sau rất
              xa, cho suy luận LLM (§20.6, p. 505), với cùng kích thước lớp dùng ở đây.
            </p>
          }
        >
          <p>
            This applies the book's §5.1 method to a rectangular product. The book does it itself much later, for LLM
            inference (§20.6, p. 505), with the same layer shape used here.
          </p>
        </Translated>
      </Beyond>
      <LinearLayerBench />
      <Translated
        vi={
          <p>
            Ở kích thước batch 1 với giá trị 16-bit, trình khám phá cho 1.0 FLOP/B, đúng con số sách nêu; ở batch 512 nó cho 315,
            cũng là con số của sách. Trọng số chiếm phần lớn lưu lượng, và một trọng số nạp một lần có thể phục vụ mọi hàng của
            batch. Giảm một nửa số byte mỗi trọng số làm tỉ lệ tăng gấp đôi vì cùng lý do đó.
          </p>
        }
      >
        <p>
          At batch size 1 with 16-bit values the explorer gives 1.0 FLOP/B, the figure the book quotes; at batch 512 it
          gives 315, also the book's. Weights dominate the traffic, and a weight fetched once can serve every row of the
          batch. Halving the bytes per weight doubles the ratio for the same reason.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trình khám phá đếm lưu lượng lý tưởng: mỗi trọng số, đầu vào và đầu ra được di chuyển đúng một lần, không gì được lưu
              trong bộ nhớ đệm và không gì bị nạp hai lần. Ngưỡng 20 FLOP/B là của thiết bị ví dụ của sách. Lab 5.3 đo một kernel
              thật trên thiết bị của bạn.
            </p>
          }
        >
          <p>
            The explorer counts ideal traffic: every weight, input and output moved exactly once, nothing cached and
            nothing fetched twice. The 20 FLOP/B threshold is the book's example device. Lab 5.3 measures a real kernel
            on yours.
          </p>
        </Translated>
      </Simplification>
      <Ask q={questions51['ch05/5.1/q-decode']} />
    </>
  );
}
