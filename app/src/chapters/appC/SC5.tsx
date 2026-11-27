import { useState, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  MAX_PEERS_WITHOUT_NVSWITCH,
  allToAll,
  canEnable,
  deref,
  emptyGrants,
  enable,
  enableAll,
  haloExchange,
  outgoing,
  revoke,
  type Grants,
  type PeerSystem,
} from '../../sim/peeraccess';

/** A long identifier that may break anywhere, for the narrow margin notes. */
function Id({ children }: { children: ReactNode }) {
  return <code style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{children}</code>;
}

const COUNTS = [2, 4, 8, 9, 16];

// ---------------------------------------------------------------- the explorer
function PeerMatrix() {
  const [sys, setSys] = useState<PeerSystem>({ n: 4, nvswitch: false, iommu: false });
  const [g, setG] = useState<Grants>(() => emptyGrants(4));
  // 'neutral' for a grant the model leaves out because the rule is not documented.
  const [status, setStatus] = useState<{ tone: 'ok' | 'bad' | 'neutral'; text: string } | null>(null);
  const [reader, setReader] = useState(0);
  const [owner, setOwner] = useState(1);
  const [approach, setApproach] = useState<'copy' | 'peer'>('copy');

  const change = (next: PeerSystem) => {
    setSys(next);
    setG(emptyGrants(next.n));
    setStatus({ tone: 'ok', text: tr('The system changed, so every grant was cleared.', 'Hệ thống đã thay đổi, nên mọi quyền cấp đã bị xóa.') });
    if (reader >= next.n) setReader(0);
    if (owner >= next.n) setOwner(1);
  };
  const click = (i: number, j: number) => {
    if (i === j) return;
    if (g[i][j]) {
      setG(revoke(g, i, j));
      setStatus({ tone: 'ok', text: tr(`Taken back: GPU ${i} no longer loads from GPU ${j}.`, `Đã thu hồi: GPU ${i} không còn nạp từ GPU ${j}.`) });
      return;
    }
    const r = enable(sys, g, i, j);
    setG(r.grants);
    setStatus(
      r.result.ok
        ? {
            tone: 'ok',
            text: tr(
              `${r.result.call}: GPU ${i} may now load from GPU ${j}'s memory. GPU ${j} may not load from GPU ${i}'s.`,
              `${r.result.call}: GPU ${i} giờ được nạp từ bộ nhớ của GPU ${j}. GPU ${j} không được nạp từ bộ nhớ của GPU ${i}.`,
            ),
          }
        : { tone: r.result.reason === 'undocumented' ? 'neutral' : 'bad', text: r.result.text },
    );
  };
  const all = () => {
    const r = enableAll(sys);
    setG(r.grants);
    setStatus(
      r.unknown
        ? {
            tone: 'neutral',
            text: tr(
              `Tried every pair: ${r.granted} grants succeeded; for ${r.unknown} the rule on this system is not documented, so the model granted none of them.`,
              `Đã thử mọi cặp: ${r.granted} lần cấp quyền thành công; với ${r.unknown} cặp, quy tắc trên hệ thống này không có tài liệu, nên mô hình không cấp cặp nào trong số đó.`,
            ),
          }
        : r.refused
          ? {
              tone: 'bad',
              text: tr(
                `Tried every pair: ${r.granted} grants succeeded and ${r.refused} were refused.`,
                `Đã thử mọi cặp: ${r.granted} lần cấp quyền thành công và ${r.refused} lần bị từ chối.`,
              ),
            }
          : {
              tone: 'ok',
              text: tr(`Tried every pair: all ${r.granted} grants succeeded.`, `Đã thử mọi cặp: cả ${r.granted} lần cấp quyền đều thành công.`),
            },
    );
  };
  const d = deref(g, reader, owner);
  const whole = allToAll(sys);
  const halo = haloExchange(approach, g);
  const cell = sys.n > 9 ? 26 : 34;

  return (
    <Bench
      title={tr('Peer access matrix', 'Ma trận truy cập ngang hàng (peer)')}
      task={tr(
        'Make every GPU able to load from every other, then find the system where you cannot.',
        'Làm cho mọi GPU nạp được từ mọi GPU khác, rồi tìm hệ thống mà bạn không làm được điều đó.',
      )}
      foot={
        <Translated
          vi={
            <>
              Các quy tắc là của NVIDIA, từ hướng dẫn CUDA §3.4.2 và tài liệu tham chiếu runtime (appC-facts.md, A8): quyền
              cấp chỉ theo một chiều, tối đa {MAX_PEERS_WITHOUT_NVSWITCH} peer cho mỗi device khi không có NVSwitch, không
              có PCIe peer-to-peer trên Linux bare-metal khi bật IOMMU. Các bước halo là cách chúng tôi đọc phép so sánh
              của sách (p. 626).
            </>
          }
        >
          <>
            The rules are NVIDIA's, from the CUDA guide §3.4.2 and the runtime reference (appC-facts.md, A8): grants are one
            way, at most {MAX_PEERS_WITHOUT_NVSWITCH} peers per device without NVSwitch, no PCIe peer-to-peer on bare-metal
            Linux with the IOMMU on. The halo steps are our reading of the book's comparison (p. 626).
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="Số GPU">GPUs</Translated>
          <select value={sys.n} aria-label={tr('Number of GPUs', 'Số GPU')} onChange={(e) => change({ ...sys, n: Number(e.target.value) })}>
            {COUNTS.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={sys.nvswitch} onChange={(e) => change({ ...sys, nvswitch: e.target.checked })} /> NVSwitch{/* i18n-ignore: product name */}
        </label>
        <label className="check">
          <input type="checkbox" checked={sys.iommu} onChange={(e) => change({ ...sys, iommu: e.target.checked })} />{' '}
          <Translated vi="Linux bare-metal, bật IOMMU">
            bare-metal Linux,
            IOMMU on
          </Translated>
        </label>
        <button className="quiet" onClick={all}>
          <Translated vi="Bật mọi cặp">Enable every pair</Translated>
        </button>
        <button
          className="quiet"
          onClick={() => {
            setG(emptyGrants(sys.n));
            setStatus(null);
          }}
        >
          <Translated vi="Xóa">Clear</Translated>
        </button>
      </div>

      <div className="cols plot">
        <div className="panel" style={{ overflowX: 'auto' }}>
          <Translated
            vi={
              <p className="muted" style={{ margin: '0 0 0.5rem' }}>
                Hàng: GPU có kernel thực hiện nạp. Cột: GPU có bộ nhớ được nạp từ đó. Bấm vào một ô để gọi{' '}
                <code>cudaDeviceEnablePeerAccess</code> trên GPU của hàng; bấm lần nữa để thu hồi quyền.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0 0 0.5rem' }}>
              Row: the GPU whose kernels load. Column: the GPU whose memory they load from. Click a cell to call{' '}
              <code>cudaDeviceEnablePeerAccess</code> on the row's GPU; click again to take the grant back.
            </p>
          </Translated>
          <table style={{ borderCollapse: 'collapse', fontFamily: 'var(--sans)', fontSize: '0.74rem' }} data-part="matrix" data-n={sys.n}>
            <thead>
              <tr>
                <th style={{ padding: '0 0.4rem 0.2rem 0', textAlign: 'right', fontWeight: 500 }} className="muted">
                  <Translated vi="từ \ tới">from \ to</Translated>
                </th>
                {g.map((_, j) => (
                  <th key={j} style={{ width: cell, textAlign: 'center', fontWeight: 600 }}>
                    {j}
                  </th>
                ))}
                <th style={{ paddingLeft: '0.5rem', textAlign: 'left', fontWeight: 500 }} className="muted">
                  <Translated vi="số peer">peers</Translated>
                </th>
              </tr>
            </thead>
            <tbody>
              {g.map((row, i) => (
                <tr key={i}>
                  <th style={{ padding: '0 0.4rem 0 0', textAlign: 'right', fontWeight: 600 }}>{i}</th>
                  {row.map((on, j) => {
                    const self = i === j;
                    const res = self || on ? null : canEnable(sys, g, i, j);
                    const unknown = !!res && !res.ok && res.reason === 'undocumented';
                    const blocked = !!res && !res.ok && !unknown;
                    return (
                      <td key={j} style={{ padding: 1 }}>
                        <button
                          data-cell={`${i}-${j}`}
                          data-on={on ? 1 : 0}
                          disabled={self}
                          aria-label={
                            self
                              ? tr(`GPU ${i}, own memory`, `GPU ${i}, bộ nhớ của chính nó`)
                              : tr(
                                  `GPU ${i} loads from GPU ${j}: ${on ? 'enabled' : unknown ? 'not documented' : 'not enabled'}`,
                                  `GPU ${i} nạp từ GPU ${j}: ${on ? 'đã bật' : unknown ? 'không có tài liệu' : 'chưa bật'}`,
                                )
                          }
                          onClick={() => click(i, j)}
                          style={{
                            width: cell,
                            height: cell - 4,
                            padding: 0,
                            borderRadius: 2,
                            fontWeight: 600,
                            border: '1px solid ' + (on ? 'var(--ok)' : blocked ? 'var(--fail)' : 'var(--rule)'),
                            background: self ? 'var(--rule-soft)' : on ? 'var(--ok-wash)' : 'var(--cell)',
                            color: on ? 'var(--ok)' : blocked ? 'var(--fail)' : 'var(--ink-3)',
                            opacity: 1,
                          }}
                        >
                          {self ? tr('own', 'mình') : on ? '✓' : blocked ? '×' : unknown ? '?' : ''}
                        </button>
                      </td>
                    );
                  })}
                  <td style={{ paddingLeft: '0.5rem' }} data-peers={outgoing(g, i)}>
                    {outgoing(g, i)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p data-out="status" className={status?.tone === 'bad' ? 'narrate bad' : 'narrate'} style={{ margin: '0.7rem 0 0' }} aria-live="polite">
            {status
              ? status.text
              : tr('No grants yet. Every GPU can load from its own memory only.', 'Chưa có quyền nào. Mỗi GPU chỉ nạp được từ bộ nhớ của chính nó.')}
          </p>
          <p className="muted" style={{ margin: '0.5rem 0 0' }} data-out="all-to-all" data-possible={whole === 'possible' ? 1 : 0} data-all={whole}>
            <Translated vi={<>Để mọi GPU nạp được từ mọi GPU khác, mỗi GPU cần {sys.n - 1} peer:</>}>
              Every GPU loading from every other needs {sys.n - 1} peer{sys.n === 2 ? '' : 's'} each:
            </Translated>{' '}
            {whole === 'unknown'
              ? tr(
                  'not documented here. NVIDIA\'s IOMMU rule is about PCIe peer-to-peer, and our sources say nothing about NVSwitch with the IOMMU on.',
                  'ở đây không có tài liệu. Quy tắc IOMMU của NVIDIA nói về PCIe peer-to-peer, và các nguồn của chúng tôi không nói gì về NVSwitch khi bật IOMMU.',
                )
              : sys.iommu
                ? tr('impossible here, since no PCIe peer grant is allowed at all.', 'không thể ở đây, vì hoàn toàn không cho phép cấp quyền peer qua PCIe.')
                : whole === 'possible'
                  ? sys.nvswitch
                    ? tr('possible, since NVSwitch lifts the limit.', 'có thể, vì NVSwitch gỡ bỏ giới hạn.')
                    : tr(`possible, within the limit of ${MAX_PEERS_WITHOUT_NVSWITCH}.`, `có thể, trong giới hạn ${MAX_PEERS_WITHOUT_NVSWITCH}.`)
                  : tr(
                      `impossible without NVSwitch, which allows ${MAX_PEERS_WITHOUT_NVSWITCH}.`,
                      `không thể nếu không có NVSwitch, vốn chỉ cho phép ${MAX_PEERS_WITHOUT_NVSWITCH}.`,
                    )}
          </p>
        </div>

        <div>
          <div className="panel" style={{ marginBottom: '1rem' }}>
            <p style={{ margin: '0 0 0.5rem' }}>
              <b>
                <Translated vi="Một kernel đi theo một con trỏ">A kernel follows a pointer</Translated>
              </b>
            </p>
            <div className="row" style={{ marginBottom: '0.5rem' }}>
              <label className="field">
                <Translated vi="Kernel trên GPU">Kernel on GPU</Translated>
                <select value={reader} aria-label={tr('Kernel on GPU', 'Kernel trên GPU')} onChange={(e) => setReader(Number(e.target.value))}>
                  {g.map((_, i) => (
                    <option key={i} value={i}>
                      {i}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <Translated vi="đọc bộ nhớ của GPU">reads memory of GPU</Translated>
                <select value={owner} aria-label={tr('Memory of GPU', 'Bộ nhớ của GPU')} onChange={(e) => setOwner(Number(e.target.value))}>
                  {g.map((_, i) => (
                    <option key={i} value={i}>
                      {i}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <p data-out="deref" data-how={d.how} style={{ margin: 0 }}>
              <span className={'chip ' + (d.ok ? 'ok' : 'fail')}>{d.how === 'local' ? tr('local', 'cục bộ') : d.how === 'peer' ? tr('peer load', 'nạp peer') : tr('error', 'lỗi')}</span>{' '}
              {d.text}
            </p>
          </div>

          <div className="panel">
            <p style={{ margin: '0 0 0.5rem' }}>
              <b>
                <Translated vi="Một lần trao đổi halo, GPU 0 và GPU 1">One halo exchange, GPU 0 and GPU 1</Translated>
              </b>
            </p>
            <span className="seg" role="group" aria-label={tr('Halo exchange', 'Trao đổi halo')}>
              <button aria-pressed={approach === 'copy'} onClick={() => setApproach('copy')}>
                <Translated vi="sao chép do host điều khiển">host-driven copies</Translated>
              </button>
              <button aria-pressed={approach === 'peer'} onClick={() => setApproach('peer')}>
                <Translated vi="nạp peer">peer loads</Translated>
              </button>
            </span>
            <ol style={{ margin: '0.7rem 0 0.6rem', paddingLeft: '1.3rem' }} data-part="halo-steps">
              {halo.steps.map((s, k) => (
                <li key={k} data-kind={s.kind}>
                  {s.text}
                  {s.kind === 'order' ? <span className="muted">
                      <Translated vi=" (của chúng tôi: sách không bàn tới điều này)"> (ours: the book does not discuss it)</Translated>
                    </span> : null}
                </li>
              ))}
            </ol>
            <table className="data" style={{ margin: 0 }} data-out="halo" data-staging={halo.staging} data-waits={halo.hostWaits} data-copies={halo.hostCopies}>
              <tbody>
                <tr>
                  <td>
                    <Translated vi="Buffer trung gian cho các hàng được sao chép">Staging buffers for copied rows</Translated>
                  </td>
                  <td className="num">{halo.staging}</td>
                </tr>
                <tr>
                  <td>
                    <Translated vi="Số lần sao chép host phải thiết lập">Copies the host sets up</Translated>
                  </td>
                  <td className="num">{halo.hostCopies}</td>
                </tr>
                <tr>
                  <td>
                    <Translated vi="Số lần host chờ một GPU">Times the host waits for a GPU</Translated>
                  </td>
                  <td className="num">{halo.hostWaits}</td>
                </tr>
                <tr>
                  <td>
                    <Translated vi="Số điểm sắp thứ tự giữa các GPU">Ordering points between the GPUs</Translated>
                  </td>
                  <td className="num">{halo.ordering}</td>
                </tr>
              </tbody>
            </table>
            {approach === 'peer' ? (
              <p style={{ margin: '0.6rem 0 0' }} data-out="halo-ready" data-ready={halo.ready ? 1 : 0}>
                {halo.ready ? (
                  <>
                    <span className="chip ok">
                      <Translated vi="sẵn sàng">ready</Translated>
                    </span>{' '}
                    <Translated vi="GPU 0 và GPU 1 mỗi bên đều có quyền tới bên kia.">
                      GPU 0 and GPU 1 each have a grant to the other.
                    </Translated>
                  </>
                ) : (
                  <>
                    <span className="chip fail">
                      <Translated vi="chưa sẵn sàng">not ready</Translated>
                    </span>{' '}
                    <Translated
                      vi={
                        <>
                          Còn thiếu trong ma trận: {halo.missing.map(([a, b]) => `GPU ${a} → GPU ${b}`).join(' và ')}. Một lần
                          trao đổi halo cần quyền theo cả hai chiều.
                        </>
                      }
                    >
                      Missing in the matrix:{' '}
                      {halo.missing.map(([a, b]) => `GPU ${a} → GPU ${b}`).join(' and ')}. A halo exchange needs a grant in each
                      direction.
                    </Translated>
                  </>
                )}
              </p>
            ) : null}
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSC5: Record<string, Question> = {
  'appC/C.5/q-oneway': {
    kind: 'choice',
    id: 'appC/C.5/q-oneway',
    prompt: (
      <Translated
        vi={
          <>
            Với GPU 0 là device hiện hành, một chương trình gọi <code>cudaDeviceEnablePeerAccess(1, 0)</code>. Giờ những lần
            nạp nào chạy được?
          </>
        }
      >
        <>
          With GPU 0 as the current device, a program calls <code>cudaDeviceEnablePeerAccess(1, 0)</code>. Which loads
          now work?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Kernel trên GPU 0 được nạp từ GPU 1, và kernel trên GPU 1 được nạp từ GPU 0: truy cập peer có tính đối xứng.">
            Kernels on GPU 0 may load from GPU 1, and kernels on GPU 1 may load from GPU 0: peer access is symmetric.
          </Translated>
        ),
        why: (
          <Translated vi="Quyền chỉ theo một chiều. Chiều ngược lại cần một lời gọi riêng, thực hiện trên GPU 1.">
            The grant runs one way only. The reverse needs its own call, made on GPU 1.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Kernel của GPU 0 được nạp từ bộ nhớ của GPU 1; chiều ngược lại cần một lời gọi riêng với GPU 1 là device hiện hành.">
            GPU 0's kernels may load from GPU 1's memory; the reverse needs its own call with GPU 1 current.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Kernel trên GPU 1 được nạp từ bộ nhớ của GPU 0: đối số nêu tên GPU được cấp quyền.">
            Kernels on GPU 1 may load from GPU 0's memory: the argument names the GPU being given access.
          </Translated>
        ),
        why: (
          <Translated vi="Đối số nêu tên peer có bộ nhớ trở nên với tới được. Device được cấp quyền là device hiện hành, GPU 0.">
            The argument names the peer whose memory becomes reachable. The device being given access is the current one,
            GPU 0.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Giờ mọi GPU trong hệ thống đều được nạp từ mọi GPU khác.">
            Any GPU in the system may now load from any other.
          </Translated>
        ),
        why: (
          <Translated vi="Một lời gọi cấp một chiều giữa hai device, không hơn.">
            One call grants one direction between two devices, nothing more.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Tài liệu tham chiếu của NVIDIA nói quyền truy cập do lời gọi này cấp "is unidirectional" (chỉ theo một chiều). Giá
            trị con trỏ mà qua đó kernel với tới bộ nhớ của peer là như nhau trên cả hai device, nên không có gì khác trong
            mã phải thay đổi.
          </p>
        }
      >
        <p>
          NVIDIA's reference says the access granted by this call "is unidirectional". The pointer value through which a
          kernel reaches the peer's memory is the same on both devices, so nothing else about the code changes.
        </p>
      </Translated>
    ),
    source: 'CUDA runtime reference, checked 2026-11-20 (notes/today/appC-facts.md, A8)',
    vi: { source: 'Tài liệu tham chiếu CUDA runtime, kiểm tra ngày 2026-11-20 (notes/today/appC-facts.md, A8)' },
  },
  'appC/C.5/q-eight': {
    kind: 'choice',
    id: 'appC/C.5/q-eight',
    prompt: (
      <Translated
        vi={
          <>
            Một hệ thống có mười sáu GPU và không có NVSwitch. Mọi GPU phải nạp trực tiếp được từ mọi GPU khác.{' '}
            <code>cudaDeviceEnablePeerAccess</code> có thiết lập được điều đó không?
          </>
        }
      >
        <>
          A system has sixteen GPUs and no NVSwitch. Every GPU should be able to load directly from every other. Can{' '}
          <code>cudaDeviceEnablePeerAccess</code> set that up?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: mỗi cặp có thứ tự một lời gọi, tổng cộng 240 lời gọi, mỗi lời gọi thực hiện với GPU nạp là device hiện hành.">
            Yes: one call per ordered pair, 240 calls in all, each made with the loading GPU current.
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi GPU sẽ cần 15 peer, mà giới hạn khi không có NVSwitch là tám.">
            Each GPU would need 15 peers, and the limit without NVSwitch is eight.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có, nếu mỗi cặp được bật theo cả hai chiều, với hai lời gọi cho mỗi cặp.">
            Yes, if each pair is enabled in both directions, with two calls per pair.
          </Translated>
        ),
        why: (
          <Translated vi="Cần cả hai chiều để nạp hai chiều, nhưng điều đó không gỡ bỏ giới hạn tám peer cho mỗi device.">
            Both directions are needed for two-way loads, but that does not lift the limit of eight peers per device.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không. Mỗi GPU sẽ cần 15 peer, và khi không có NVSwitch, một device hỗ trợ tối đa tám. Hãy dùng NVSwitch, quyền cấp VMM hoặc sao chép.">
            No. Each GPU would need 15 peers, and without NVSwitch a device supports at most eight. Use NVSwitch, VMM grants
            or copies.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có trên Windows, nơi driver gỡ bỏ giới hạn peer; không được trên Linux.">
            Yes on Windows, where the driver lifts the peer limit; not on Linux.
          </Translated>
        ),
        why: (
          <Translated vi="Hướng dẫn nêu giới hạn này cho mọi hệ thống không có NVSwitch, bất kể hệ điều hành. Quy tắc nền tảng duy nhất mà nó ghi lại thuộc loại khác: không có PCIe peer-to-peer trên Linux bare-metal khi bật IOMMU.">
            The guide states the limit for every system without NVSwitch, whatever the operating system. The one platform
            rule it documents is of another kind: no PCIe peer-to-peer on bare-metal Linux with the IOMMU on.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hãy thử trong ma trận: đặt mười sáu GPU và bấm "Bật mọi cặp". Mỗi hàng dừng ở tám quyền cấp. Khi bật NVSwitch, cả
            240 đều thành công. NVIDIA cũng gợi ý các lời gọi quản lý bộ nhớ ảo của <a href={href('appC', 'C.9')}>§C.9</a>{' '}
            như một cách dễ mở rộng hơn để cấp cho peer quyền truy cập vào những vùng nhớ cụ thể.
          </p>
        }
      >
        <p>
          Try it in the matrix: set sixteen GPUs and press "Enable every pair". Each row stops at eight grants. With
          NVSwitch on, all 240 go through. NVIDIA also suggests the virtual memory management calls of{' '}
          <a href={href('appC', 'C.9')}>§C.9</a> as the more scalable way to grant peers access to particular memory.
        </p>
      </Translated>
    ),
    source: 'CUDA guide §3.4.2.2, checked 2026-11-20 (notes/today/appC-facts.md, A8)',
    vi: { source: 'Hướng dẫn CUDA §3.4.2.2, kiểm tra ngày 2026-11-20 (notes/today/appC-facts.md, A8)' },
  },
  'appC/C.5/q-removes': {
    kind: 'choice',
    id: 'appC/C.5/q-removes',
    prompt: (
      <Translated
        vi={
          <>
            So với các lần sao chép device-sang-device do host kích hoạt, việc nạp halo của hàng xóm trực tiếp trong kernel
            loại bỏ được gì?
          </>
        }
      >
        <>
          Compared with host-triggered device-to-device copies, what does loading the neighbour's halo directly in the
          kernel remove?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Lần sao chép trung chuyển, cùng việc thiết lập sao chép và đồng bộ hóa của host.">
            The staging copy and the host's copy set-up and synchronisation.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Mọi sự đồng bộ hóa giữa các GPU, vì mỗi kernel đọc các hàng của hàng xóm khi nó cần.">
            All synchronisation between the GPUs, since each kernel reads its neighbour's rows when it needs them.
          </Translated>
        ),
        why: (
          <Translated vi="Một GPU vẫn không được đọc các hàng của hàng xóm trước khi hàng xóm ghi xong chúng. Vẫn còn một mức sắp thứ tự nào đó giữa các device.">
            A GPU must still not read its neighbour's rows before the neighbour has written them. Some ordering between the
            devices remains.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Lưu lượng qua đường kết nối, vì kernel nạp halo thay vì sao chép nó.">
            The traffic over the interconnect, because the kernel loads the halo instead of copying it.
          </Translated>
        ),
        why: (
          <Translated vi="Halo vẫn đi qua đường truyền; mỗi lần nạp peer đều đi qua nó. Thứ biến mất là lần sao chép thừa và phần việc của host trong đó.">
            The halo still crosses the link; each peer load goes over it. What disappears is the extra copy and the host's
            part in it.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Sách liệt kê đúng những chi phí này của cách sao chép: bộ nhớ thêm để chứa dữ liệu được sao chép, và chi phí đồng
            bộ hóa host-device cùng thiết lập sao chép. Việc vẫn cần sắp thứ tự giữa các device là ghi chú của chúng tôi;
            sách không bàn tới điều đó.
          </p>
        }
      >
        <p>
          The book lists exactly these costs of the copy approach: extra memory to hold the copied data, and the overhead of
          host-device synchronisation and copy set-up. That ordering between the devices is still needed is our note; the
          book does not discuss it.
        </p>
      </Translated>
    ),
    source: 'App. C §C.5, p. 626',
    vi: { source: 'Phụ lục C §C.5, p. 626' },
  },
  'appC/C.5/q-metal': {
    kind: 'choice',
    id: 'appC/C.5/q-metal',
    prompt: (
      <Translated
        vi={
          <>
            Bạn đang chuyển một server suy luận CUDA chia các lớp của mô hình cho hai GPU: sau{' '}
            <code>cudaDeviceEnablePeerAccess</code>, kernel của GPU 1 đọc activation của GPU 0 qua một con trỏ peer. Đích
            đến là một máy Mac chạy Apple silicon. Đường truy cập peer sẽ ra sao?
          </>
        }
      >
        <>
          You are porting a CUDA inference server that splits a model's layers between two GPUs: after{' '}
          <code>cudaDeviceEnablePeerAccess</code>, GPU 1's kernels read GPU 0's activations through a peer pointer. The
          target is a Mac with Apple silicon. What becomes of the peer-access path?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi='Không có thứ tương ứng trong Metal: peer group đã bị deprecated, "Not applicable on Apple Silicon". Hãy thiết kế bản Metal mà không chia việc giữa các GPU.'>
            No Metal counterpart: peer groups are deprecated, "Not applicable on Apple Silicon". Design the Metal build
            without a GPU-to-GPU split.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Thay cudaDeviceEnablePeerAccess bằng lời gọi truy cập peer của Metal; kernel giữ nguyên.">
            Swap cudaDeviceEnablePeerAccess for Metal's peer-access call; the kernels stay the same.
          </Translated>
        ),
        why: (
          <Translated vi="Metal không có lời gọi như vậy trên Apple silicon. Ở đó không có gì cấp cho một GPU quyền truy cập buffer của GPU khác.">
            Metal has no such call on Apple silicon. Nothing grants one GPU access to another GPU's buffer there.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Gắn buffer của GPU kia qua newRemoteBufferViewForDevice, thứ cho một view của buffer của peer, rồi đọc view đó trong kernel.">
            Bind the other GPU's buffer through newRemoteBufferViewForDevice, which gives a view of a peer's buffer, and read
            that view in the kernel.
          </Translated>
        ),
        why: (
          <Translated vi="Phương thức đó tạo một view cho các lần sao chép blit giữa các GPU trong một peer group, trên Mac chạy Intel. Nó chưa bao giờ phục vụ việc nạp từ kernel, và nó đã bị deprecated.">
            That method made a view for blit copies between GPUs in a peer group, on Intel Macs. It never served kernel
            loads, and it is deprecated.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Giữ nguyên thiết kế: trên một bộ nhớ vật lý duy nhất, mọi buffer đều hiển thị với mọi GPU, nên con trỏ peer cứ thế chạy được.">
            Keep the design: on one physical memory every buffer is visible to every GPU, so the peer pointer just works.
          </Translated>
        ),
        why: (
          <Translated vi="Một bộ nhớ là CPU và GPU dùng chung DRAM. Nó không nói gì về GPU thứ hai, và Metal không ghi lại peer nào trên Apple silicon.">
            One memory is the CPU and the GPU sharing DRAM. It says nothing about a second GPU, and Metal documents no peers
            on Apple silicon.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>MTLDevice.peerGroupID</code> và <code>newRemoteBufferViewForDevice:</code> đều được đánh dấu "Not
            applicable on Apple Silicon" trong macOS 27.0 SDK, và remote view là một đường sao chép, không phải giải tham
            chiếu. Một thiết kế dựa trên nạp peer là thiết kế CUDA và HIP; trên Mac, bạn lên kế hoạch cho bản Metal không có
            nó thay vì dịch lời gọi.
          </p>
        }
      >
        <p>
          <code>MTLDevice.peerGroupID</code> and <code>newRemoteBufferViewForDevice:</code> are both marked "Not applicable on
          Apple Silicon" in the macOS 27.0 SDK, and the remote view was a copy path, not a dereference. A design built on peer
          loads is a CUDA and HIP design; on a Mac you plan the Metal build without it rather than translate the call.
        </p>
      </Translated>
    ),
    source: "Apple's documentation and the macOS 27.0 SDK headers, checked 2026-11-20 (notes/today/appC-facts.md, C4)",
    vi: { source: 'Tài liệu của Apple và các header của macOS 27.0 SDK, kiểm tra ngày 2026-11-20 (notes/today/appC-facts.md, C4)' },
  },
};

// ---------------------------------------------------------------- the section
export function SC5() {
  return (
    <>
      <Translated
        vi={
          <p>
            Một địa chỉ vật lý rộng cũng có chỗ cho nhiều hơn một GPU. Ý của sách ở đây là phần mềm hệ thống có thể đặt bộ
            nhớ của mọi GPU vào một không gian địa chỉ vật lý hợp nhất. Khi đó một kernel trên một GPU có thể nạp từ bộ nhớ
            của GPU khác, trên cùng bus PCIe hoặc miền NVLink, bằng cách giải tham chiếu một con trỏ trỏ vào đó.
          </p>
        }
      >
        <p>
          A wide physical address also has room for more than one GPU. The book's point here is that system software can
          place every GPU's memory in one unified physical address space. A kernel on one GPU can then load from another
          GPU's memory, on the same PCIe bus or NVLink domain, by dereferencing a pointer into it.
        </p>
      </Translated>
      <Source section="C.5" pages="626" />
      <Translated
        vi={
          <p>
            Điều đó thay thế một thứ vụng về. Lấy ví dụ trao đổi halo trong stencil của chương 23 (
            <a href={href('ch23', '23.4')}>mục 23.4</a> ở đây), nơi mỗi GPU sở hữu một nửa lưới và cần vài hàng biên của nửa
            kia trước bước tiếp theo. Không có truy cập trực tiếp, host phải kích hoạt các lần sao chép device-sang-device:
            chờ cả hai GPU, thiết lập một lần sao chép mỗi chiều vào một buffer dự phòng, rồi khởi chạy bước tiếp theo. Sách
            nêu tên các chi phí: bộ nhớ thêm để chứa các hàng được sao chép, và việc đồng bộ hóa cùng thiết lập sao chép của
            host. Có truy cập trực tiếp, mỗi kernel nhận con trỏ của hàng xóm làm đối số và tự nạp các hàng.
          </p>
        }
      >
        <p>
          That replaces something clumsy. Take the halo exchange of chapter 23's stencil (
          <a href={href('ch23', '23.4')}>section 23.4</a> here), where each GPU owns half a grid
          and needs a few boundary rows of the other half before its next step. Without direct access, the host has to
          trigger device-to-device copies: wait for both GPUs, set up a copy each way into a spare buffer, then launch the
          next step. The book names the costs: extra memory to hold the copied rows, and the host's synchronisation and
          copy set-up. With direct access, each kernel takes its neighbour's pointer as an argument and loads the rows
          itself.
        </p>
      </Translated>
      <Source section="C.5" pages="626" />
      <Translated
        vi={
          <p>
            Sách nói điều này trở nên khả thi từ Kepler. Chính lời kể của NVIDIA không ủng hộ mốc đó: thông báo CUDA 6 của
            họ nói định địa chỉ hợp nhất, có từ CUDA 4 năm 2011, cho phép mã GPU với tới con trỏ trên "a different GPU" (một
            GPU khác), và năm 2011 là trước khi Kepler ra mắt. Chúng tôi không tìm thấy trang nào của NVIDIA nêu tên kiến
            trúc đầu tiên có nạp peer, nên khóa học này không dạy mốc nào.
          </p>
        }
      >
        <p>
          The book says this became possible with Kepler. NVIDIA's own account does not support that boundary: its CUDA 6
          announcement says unified addressing, since CUDA 4 in 2011, let GPU code reach pointers on "a different GPU", and
          2011 is before Kepler's launch. We found no NVIDIA page that names the first architecture with peer loads, so
          this course teaches no boundary.
        </p>
      </Translated>
      <Today
        checked={tr(
          "2026-11-20 against NVIDIA's CUDA 6 Unified Memory post (appC-facts.md, A8)",
          'ngày 2026-11-20, đối chiếu với bài viết về Unified Memory CUDA 6 của NVIDIA (appC-facts.md, A8)',
        )}
      >
        <Translated
          vi={<p>"Trước Kepler" không được lời kể của NVIDIA ủng hộ. Kiến trúc đầu tiên có nạp peer: unverified.</p>}
        >
          <p>"Before Kepler" is not supported by NVIDIA's account. The first architecture with peer loads: unverified.</p>
        </Translated>
      </Today>
      <Ask q={questionsSC5['appC/C.5/q-removes']} />

      <h2>
        <Translated vi="Nó phải được bật lên">It has to be switched on</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đoạn văn của sách đọc như thể GPU nào cũng nạp được từ GPU nào. Trong CUDA ngày nay, chương trình phải xin trước.{' '}
            <code>cudaDeviceCanAccessPeer</code> cho biết hai device có thể là peer của nhau hay không. Sau đó, với device sẽ
            thực hiện nạp là device hiện hành, chương trình gọi <code>cudaDeviceEnablePeerAccess(peerDevice, 0)</code>. Sau
            đó, kernel trên device hiện hành có thể nạp qua một con trỏ trỏ vào bộ nhớ của peer, và con trỏ có cùng giá trị
            trên cả hai device. Không có quyền cấp, dữ liệu vẫn di chuyển giữa các GPU bằng sao chép, với{' '}
            <code>cudaMemcpyPeer</code>.
          </p>
        }
      >
        <p>
          The book's paragraph reads as if any GPU could load from any other. In CUDA today a program must ask first.{' '}
          <code>cudaDeviceCanAccessPeer</code> says whether two devices can be peers at all. Then, with the device that will
          do the loading as the current device, the program calls <code>cudaDeviceEnablePeerAccess(peerDevice, 0)</code>.
          After that, kernels on the current device can load through a pointer into the peer's memory, and the pointer has
          the same value on both devices. Without a grant, data still moves between GPUs by copying, with{' '}
          <code>cudaMemcpyPeer</code>.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the CUDA 13.5 guide §3.4.2 and runtime reference (appC-facts.md, A8)',
          'ngày 2026-11-20, đối chiếu với hướng dẫn CUDA 13.5 §3.4.2 và tài liệu tham chiếu runtime (appC-facts.md, A8)',
        )}
      >
        <Translated vi={<p>Quyền do một lời gọi cấp "is unidirectional" (chỉ theo một chiều).</p>}>
          <p>Access granted by one call "is unidirectional".</p>
        </Translated>
        <Translated
          vi={
            <p>
              Không có NVSwitch thì tối đa tám kết nối peer cho mỗi device. Không có PCIe peer-to-peer trên Linux bare-metal
              khi bật IOMMU.
            </p>
          }
        >
          <p>
            Without NVSwitch, at most eight peer connections per device. No PCIe peer-to-peer on bare-metal Linux with the
            IOMMU on.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSC5['appC/C.5/q-oneway']} />
      <Translated
        vi={
          <p>
            Đi kèm là hai giới hạn. Không có NVSwitch, một device hỗ trợ tối đa tám kết nối peer, nên một máy lớn không thể
            nối mọi thứ với mọi thứ. Và trên Linux bare-metal khi bật IOMMU, CUDA hoàn toàn không hỗ trợ PCIe peer-to-peer.
            Còn có một chi phí tăng theo mức sử dụng: bật truy cập peer làm mọi lần cấp phát sau đó trên peer tốn kém hơn, và
            NVIDIA chỉ tới các lời gọi bộ nhớ ảo của <a href={href('appC', 'C.9')}>§C.9</a>, vốn chỉ cấp quyền cho vùng nhớ
            cần nó, như phương án dễ mở rộng.
          </p>
        }
      >
        <p>
          Two limits come with it. Without NVSwitch, a device supports at most eight peer connections, so a large box cannot
          connect everything to everything. And on bare-metal Linux with the IOMMU switched on, CUDA does not support PCIe
          peer-to-peer at all. There is also a cost that grows with use: enabling peer access makes every later allocation
          on the peer more expensive, and NVIDIA points to the virtual memory calls of{' '}
          <a href={href('appC', 'C.9')}>§C.9</a>, which grant access only to the memory that needs it, as the scalable
          alternative.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trình khám phá áp dụng các quy tắc này. Cấp quyền từng ô một và để ý chiều; rồi yêu cầu mọi cặp trên chín GPU và
            trên mười sáu GPU. Bên cạnh, một kernel đi theo một con trỏ, và một lần trao đổi halo được trình bày theo cả hai
            cách, với các chi phí sách nêu được đếm ra.
          </p>
        }
      >
        <p>
          The explorer applies these rules. Grant access cell by cell and watch the direction; then ask for every pair on
          nine GPUs and on sixteen. Beside it, a kernel follows a pointer, and one halo exchange is laid out both ways, with
          the costs the book names counted.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Việc hai GPU có thể là peer hay không được coi là cho trước; trên máy thật, <Id>cudaDeviceCanAccessPeer</Id>{' '}
              quyết định, dựa trên topo kết nối.
            </p>
          }
        >
          <p>
            Whether two GPUs can be peers at all is taken as given; on a real machine <Id>cudaDeviceCanAccessPeer</Id>{' '}
            decides, from the topology.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              "Tám kết nối peer" được mô hình hóa thành tám quyền cấp đi ra; hướng dẫn không nói kết nối được đếm thế nào.
              Không có băng thông hay độ trễ.
            </p>
          }
        >
          <p>
            "Eight peer connections" is modelled as eight outgoing grants; the guide does not say how connections are
            counted. No bandwidth or latency.
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Khi không có NVSwitch, các GPU được coi là peer qua PCIe, đúng phạm vi quy tắc IOMMU bao quát. Khi có NVSwitch
              và bật IOMMU, các nguồn của chúng tôi không nói gì, và trình khám phá nói rõ như vậy thay vì đoán.
            </p>
          }
        >
          <p>
            Without NVSwitch the GPUs are taken to be PCIe peers, which is what the IOMMU rule covers. With NVSwitch and the
            IOMMU on, our sources say nothing, and the explorer says so instead of guessing.
          </p>
        </Translated>
      </Simplification>
      <PeerMatrix />
      <Ask q={questionsSC5['appC/C.5/q-eight']} />

      <h2>
        <Translated vi="Nơi bạn gặp nó">Where you meet it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chia một mô hình cho nhiều GPU nghĩa là mỗi GPU tính một phần của một lớp và cần các kết quả từng phần mà GPU khác
            giữ. Những kết quả đó đến bằng nạp peer hay bằng sao chép chính là lựa chọn của mục này. llama.cpp, chẳng hạn, để
            việc đó cho bạn: backend CUDA của nó có một biến môi trường bật truy cập peer giữa các GPU.
          </p>
        }
      >
        <p>
          Splitting one model across several GPUs means each GPU computes part of a layer and needs partial results that
          another GPU holds. Whether those arrive by peer loads or by copies is exactly this section's choice. llama.cpp,
          for one, leaves it to you: its CUDA backend has an environment variable that turns peer access between GPUs
          on.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Ghi chú build của llama.cpp: "The environment variable <Id>GGML_CUDA_P2P</Id> can be set to enable
              peer-to-peer access between multiple GPUs" (có thể đặt biến môi trường này để bật truy cập peer-to-peer giữa
              nhiều GPU).
            </p>
          }
        >
          <p>
            llama.cpp's build notes: "The environment variable <Id>GGML_CUDA_P2P</Id> can be set to enable peer-to-peer
            access between multiple GPUs".
          </p>
        </Translated>
        <Translated vi={<p className="muted">Kiểm tra ngày 2026-11-20 trên nhánh master của llama.cpp (appC-facts.md, D3).</p>}>
          <p className="muted">Checked 2026-11-20 against llama.cpp master (appC-facts.md, D3).</p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Trên GPU AMD và Apple">On AMD and Apple GPUs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            HIP có cùng mô hình dưới tên của HIP, <code>hipDeviceCanAccessPeer</code> và <code>hipDeviceEnablePeerAccess</code>.
            Không có quyền cấp, <code>hipMemcpy</code> giữa các GPU vẫn chạy, nhưng AMD nói nó chuyển dữ liệu qua bộ nhớ host.
            Metal không có gì tương đương trên Apple silicon: các lời gọi peer group của Mac chạy Intel đã bị deprecated ở đó.
          </p>
        }
      >
        <p>
          HIP has the same model under HIP names, <code>hipDeviceCanAccessPeer</code> and{' '}
          <code>hipDeviceEnablePeerAccess</code>. Without a grant, <code>hipMemcpy</code> between GPUs still works, but AMD
          says it stages the data through host memory. Metal has nothing comparable on Apple silicon: the peer-group calls
          of Intel-based Macs are deprecated there.
        </p>
      </Translated>
      <Beyond>
        <Translated vi={<p>HIP: kiểm tra với header ROCm 10.2 và trang đa thiết bị của AMD (appC-facts.md, B4).</p>}>
          <p>
            HIP: checked against the ROCm 10.2 header and AMD's multi-device page (appC-facts.md, B4).
          </p>
        </Translated>
        <Translated
          vi={
            <p>
              Metal: <Id>peerGroupID</Id> và <Id>newRemoteBufferViewForDevice:</Id> đã bị deprecated, "Not applicable on
              Apple Silicon" (C4).
            </p>
          }
        >
          <p>
            Metal: <Id>peerGroupID</Id> and <Id>newRemoteBufferViewForDevice:</Id> are deprecated, "Not applicable on Apple
            Silicon" (C4).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsSC5['appC/C.5/q-metal']} />
      <Translated
        vi={
          <p>
            Quay lại <a href={href('appC', 'start')}>bản đồ</a>: đây là bước được đánh dấu C.4–C.5. Các ô nó thay đổi là hàng
            của bộ nhớ <code>cudaMalloc</code> của một GPU peer, từ "không" thành "chỉ sau khi truy cập được bật".
          </p>
        }
      >
        <p>
          Back to <a href={href('appC', 'start')}>the map</a>: this is the step marked C.4–C.5. The cells it changes are the
          row for a peer GPU's <code>cudaMalloc</code> memory, from "no" to "only after access is enabled".
        </p>
      </Translated>
    </>
  );
}
