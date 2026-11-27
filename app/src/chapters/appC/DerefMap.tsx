import { useState, type CSSProperties } from 'react';
import { Bench } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  ERAS,
  ROWS,
  ROW_LABEL,
  SITES,
  SITE_LABEL,
  VERDICT_LABEL,
  changed,
  grid,
  tone,
  type EraId,
  type Platform,
  type RowId,
  type SiteId,
  type Verdict,
} from '../../sim/derefmap';

export type { EraId } from '../../sim/derefmap';

// "Who can dereference what, era by era": the map of appendix C. Every cell comes
// from app/src/sim/derefmap.ts; this file only draws it. Used on the start page
// and, compact, on the summary page.

/** Which section page goes deeper into an era. */
const ERA_PAGES: Record<EraId, string[]> = {
  copies: [],
  fermi: ['C.1'],
  zerocopy: ['C.2'],
  uva: ['C.3'],
  kepler: ['C.4', 'C.5'],
  'um-limited': ['C.6'],
  pascal: ['C.7', 'C.8'],
  vmm: ['C.9'],
  hmm: ['C.8'],
};

const PLATFORM_LABEL: Record<Platform, string> = {
  get cuda() {
    return tr('CUDA, by era', 'CUDA, theo từng thời kỳ');
  },
  get hip() {
    return tr('HIP today (ROCm 10.2)', 'HIP hiện nay (ROCm 10.2)');
  },
  get metal() {
    return tr('Metal today (Apple silicon)', 'Metal hiện nay (Apple silicon)');
  },
};

const chipStyle = (v: Verdict): CSSProperties => {
  const t = tone(v);
  const base: CSSProperties = { whiteSpace: 'normal', lineHeight: 1.25, textAlign: 'left' };
  if (t === 'neutral') return { ...base, border: '1px solid var(--rule)', color: 'var(--ink)', background: 'var(--page)' }; // i18n-ignore: CSS value
  if (t === 'absent') return { ...base, border: '1px dashed var(--rule)', color: 'var(--ink-3)', background: 'transparent' }; // i18n-ignore: CSS value
  return base;
};
const chipClass = (v: Verdict) => {
  const t = tone(v);
  return 'chip' + (t === 'ok' ? ' ok' : t === 'fail' ? ' fail' : '');
};

export function VerdictChip({ v }: { v: Verdict }) {
  return (
    <span className={chipClass(v)} style={chipStyle(v)} data-chip={v}>
      {VERDICT_LABEL[v]}
    </span>
  );
}

/** "p. 625; A3" read aloud: book pages say "book", bare items name the appendix C facts file. */
export function sourceText(src: string): string {
  return src
    .split(';')
    .map((part) => {
      const p = part.trim();
      if (/^(p\.|pp\.|ch\.)/.test(p)) return 'book ' + p;
      if (/^[A-D]\d/.test(p)) return 'appC-facts ' + p;
      return p;
    })
    .join('; ');
}

export function DerefMap({
  initialEra = 'copies',
  platform: initialPlatform = 'cuda',
  compact = false,
}: {
  initialEra?: EraId;
  platform?: 'cuda' | 'hip' | 'metal';
  compact?: boolean;
}) {
  const [platform, setPlatform] = useState<Platform>(initialPlatform);
  const [eraIdx, setEraIdx] = useState(Math.max(0, ERAS.findIndex((e) => e.id === initialEra)));
  const [xnack, setXnack] = useState(false);
  const [sel, setSel] = useState<{ row: RowId; site: SiteId }>({ row: 'malloc', site: 'arg' });
  const e = ERAS[eraIdx];
  const g = grid(platform, e.id, { xnack });
  const diff = platform === 'cuda' ? changed(e.id) : [];
  const isNew = (r: RowId, s: SiteId) => diff.some((d) => d.row === r && d.site === s);
  const c = g[sel.row][sel.site];

  return (
    <Bench
      title={tr('Who can dereference what, era by era', 'Ai giải tham chiếu được gì, qua từng thời kỳ')}
      task={compact ? undefined : tr('Step through the eras, switch platform, and click any cell.', 'Đi qua từng thời kỳ, đổi nền tảng, và bấm vào ô bất kỳ.')}
      foot={
        <Translated
          vi={
            <>
              Mỗi ô là một quy tắc có tài liệu, và chi phí là các hạng, không phải con số. Nguồn: "book p." là sách;
              A1 đến D5 là các mục của notes/today/appC-facts.md, kiểm tra ngày 2026-11-20. "Không có tài liệu" nghĩa là
              các nguồn của chúng tôi không nói tới, chứ không phải câu trả lời là không.
            </>
          }
        >
          <>
            One cell is one documented rule, and costs are classes, not numbers. Sources: "book p." is the book;
            A1 to D5 are items of notes/today/appC-facts.md, checked 2026-11-20. "Not documented" means our sources do not
            say, not that the answer is no.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Platform', 'Nền tảng')}>
          {(['cuda', 'hip', 'metal'] as Platform[]).map((p) => (
            <button key={p} aria-pressed={platform === p} data-platform={p} onClick={() => setPlatform(p)}>
              {PLATFORM_LABEL[p]}
            </button>
          ))}
        </span>
        {platform === 'hip' ? (
          <label className="check ui">
            <input type="checkbox" checked={xnack} onChange={(ev) => setXnack(ev.target.checked)} />
            <Translated vi="HSA_XNACK=1 với driver HMM (CDNA2 trở lên)">HSA_XNACK=1 with an HMM driver (CDNA2 or later)</Translated>
          </label>
        ) : null}
      </div>

      {platform === 'cuda' ? (
        <>
          <div className="row ui" style={{ gap: '0.4rem', marginBottom: '0.7rem', alignItems: 'stretch' }}>
            <button className="quiet" onClick={() => setEraIdx((i) => Math.max(0, i - 1))} disabled={eraIdx === 0}>
              <Translated vi="Trước">Previous</Translated>
            </button>
            <div role="group" aria-label={tr('Eras', 'Các thời kỳ')} style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem', flex: '1 1 20rem' }}>
              {ERAS.map((x, i) => (
                <button
                  key={x.id}
                  data-era={x.id}
                  aria-pressed={i === eraIdx}
                  className={i === eraIdx ? undefined : 'quiet'}
                  onClick={() => setEraIdx(i)}
                  title={x.title}
                  style={{ fontSize: '0.8rem', padding: '0.25rem 0.55rem' }}
                >
                  {tr(x.strip, x.strip === 'beyond the book' ? 'ngoài sách' : x.strip)}
                </button>
              ))}
            </div>
            <button className="quiet" onClick={() => setEraIdx((i) => Math.min(ERAS.length - 1, i + 1))} disabled={eraIdx === ERAS.length - 1}>
              <Translated vi="Tiếp">Next</Translated>
            </button>
          </div>
          <div className="panel" data-era-head={e.id} style={{ marginBottom: '0.8rem' }}>
            <p style={{ margin: 0 }}>
              <b>
                {e.strip === 'beyond the book' ? tr('Beyond the book', 'Ngoài sách') : e.strip}: {e.title}
              </b>{' '}
              <span className="muted">({e.vehicle})</span>
            </p>
            <p className="ui" style={{ margin: '0.3rem 0 0', fontSize: '0.85rem' }}>
              <span data-date="book">
                <Translated vi={<>Sách: {e.book ?? 'không có trong sách'}.</>}>
                  The book: {e.book ?? 'not in the book'}.
                </Translated>
              </span>{' '}
              {e.corrected ? (
                <span data-date="ours">
                  <Translated vi={<>Nguồn của chúng tôi: {e.corrected.date} ({e.corrected.item}).</>}>
                    Our sources: {e.corrected.date} ({e.corrected.item}).
                  </Translated>
                </span>
              ) : null}
            </p>
            {compact ? null : <p style={{ margin: '0.4rem 0 0' }}>{e.adds}</p>}
            {e.note && !compact ? (
              <p className="muted" style={{ margin: '0.3rem 0 0', fontSize: '0.92rem' }}>
                {e.note}
              </p>
            ) : null}
            {ERA_PAGES[e.id].length ? (
              <p className="ui" style={{ margin: '0.4rem 0 0', fontSize: '0.85rem' }}>
                <Translated vi="Tìm hiểu thêm về bước này:">More on this step:</Translated>{' '}
                {ERA_PAGES[e.id].map((s, i) => (
                  <span key={s}>
                    {i ? ', ' : ''}
                    <a href={href('appC', s)}>
                      <Translated vi={<>mục {s}</>}>section {s}</Translated>
                    </a>
                  </span>
                ))}
                .
              </p>
            ) : null}
          </div>
        </>
      ) : (
        <div className="panel" style={{ marginBottom: '0.8rem' }}>
          <p style={{ margin: 0 }}>
            {platform === 'hip' ? (
              <Translated
                vi={
                  <>
                    <b>HIP hiện nay</b> <span className="muted">(ROCm 10.2, tài liệu và header của AMD; appC-facts B1 đến B7)</span>. AMD
                    cũng có các thời kỳ như trong sách, nhưng được chọn theo từng GPU và từng tiến trình thay vì theo ngày: công tắc
                    XNACK ở trên chính là khác biệt giữa loại bị giới hạn và loại có xử lý lỗi trang (B5).
                  </>
                }
              >
                <>
                  <b>HIP today</b> <span className="muted">(ROCm 10.2, AMD's documentation and headers; appC-facts B1 to B7)</span>. AMD has the
                  book's eras too, chosen per GPU and per process instead of by date: the XNACK switch above is the
                  difference between the limited and the page-faulting kind (B5).
                </>
              </Translated>
            ) : (
              <Translated
                vi={
                  <>
                    <b>Metal hiện nay</b> <span className="muted">(Apple silicon, đặc tả MSL 4.1 và macOS 27 SDK; appC-facts C1 đến C6)</span>.
                    Một bộ nhớ vật lý duy nhất, và kernel nhận buffer bằng cách gắn (bind) chúng, không bao giờ dưới dạng con trỏ CPU thô (C3).
                    Các hàng là những thứ tương ứng gần nhất trong Metal với các hàng của CUDA.
                  </>
                }
              >
                <>
                  <b>Metal today</b> <span className="muted">(Apple silicon, the MSL 4.1 specification and the macOS 27 SDK; appC-facts C1 to C6)</span>.
                  One physical memory, and kernels receive buffers by binding them, never as raw CPU pointers (C3). The
                  rows are the nearest Metal counterparts of CUDA's.
                </>
              </Translated>
            )}
          </p>
        </div>
      )}

      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-derefmap style={{ margin: 0, fontSize: compact ? 'clamp(0.7rem, 2.4vw, 0.78rem)' : 'clamp(0.7rem, 2.4vw, 0.84rem)' }}>
          <thead>
            <tr>
              <th style={{ width: '28%' }}>
                <Translated vi="Con trỏ tới">Pointer to</Translated>
              </th>
              {SITES.map((s) => (
                <th key={s} style={{ width: '24%' }}>
                  {SITE_LABEL[platform][s]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r}>
                <th scope="row" style={{ fontWeight: 500, overflowWrap: 'anywhere' }}>
                  {ROW_LABEL[platform][r]}
                </th>
                {SITES.map((s) => {
                  const v = g[r][s].verdict;
                  const on = sel.row === r && sel.site === s;
                  const fresh = isNew(r, s);
                  return (
                    <td key={s} data-row={r} data-site={s} data-verdict={v} data-changed={fresh ? '1' : undefined} style={{ padding: '0.25rem 0.4rem 0.25rem 0' }}>
                      <button
                        className="quiet"
                        aria-pressed={on}
                        aria-label={`${ROW_LABEL[platform][r]}, ${SITE_LABEL[platform][s]}: ${VERDICT_LABEL[v]}`}
                        onClick={() => setSel({ row: r, site: s })}
                        style={{
                          display: 'block',
                          width: '100%',
                          textAlign: 'left',
                          padding: '0.25rem 0.35rem',
                          background: on ? 'var(--rule-soft)' : 'transparent',
                          color: 'var(--ink)',
                          borderColor: on ? 'var(--ink)' : fresh ? 'var(--ink-3)' : 'transparent',
                          borderStyle: fresh && !on ? 'dashed' : 'solid',
                          fontWeight: 450,
                          // The table's font, not the app's button size, so the map narrows with the screen.
                          fontSize: 'inherit',
                        }}
                      >
                        <VerdictChip v={v} />
                        {fresh ? (
                          <span className="ui muted" style={{ fontSize: '0.7rem', marginLeft: '0.35rem' }}>
                            <Translated vi="đã đổi">changed</Translated>
                          </span>
                        ) : null}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="narrate" data-detail style={{ marginTop: '0.8rem' }}>
        <p style={{ margin: 0 }}>
          <b>{ROW_LABEL[platform][sel.row]}</b>, {SITE_LABEL[platform][sel.site]}: <VerdictChip v={c.verdict} />
        </p>
        <p style={{ margin: '0.35rem 0 0' }} data-cost>
          {c.cost}
        </p>
        <p className="muted ui" style={{ margin: '0.3rem 0 0', fontSize: '0.8rem' }} data-source>
          <Translated vi={<>Nguồn: {sourceText(c.source)}.</>}>Source: {sourceText(c.source)}.</Translated>
        </p>
      </div>

      {platform === 'cuda' && !compact && eraIdx > 0 ? (
        <div style={{ marginTop: '0.8rem' }} data-changes>
          <p className="ui" style={{ margin: '0 0 0.3rem', fontSize: '0.85rem' }}>
            <b>
              <Translated vi="Bước này đã thay đổi gì">What this step changed</Translated>
            </b>{' '}
            {diff.length ? null : (
              <span className="muted">
                <Translated vi="trong các ô: không gì cả. Nó thay đổi địa chỉ, không thay đổi quyền truy cập.">
                  in the cells: nothing. It changes addresses, not access.
                </Translated>
              </span>
            )}
          </p>
          {diff.length ? (
            <ul className="ui" style={{ margin: 0, paddingLeft: '1.1rem', fontSize: '0.85rem' }}>
              {diff.map((d) => (
                <li key={d.row + d.site}>
                  <a
                    href="#"
                    onClick={(ev) => {
                      ev.preventDefault();
                      setSel({ row: d.row, site: d.site });
                    }}
                  >
                    {ROW_LABEL.cuda[d.row]}, {SITE_LABEL.cuda[d.site]}
                  </a>
                  <Translated vi={<>: từ "{VERDICT_LABEL[d.from]}" thành "{VERDICT_LABEL[d.to]}"</>}>
                    : from "{VERDICT_LABEL[d.from]}" to "{VERDICT_LABEL[d.to]}"
                  </Translated>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}
    </Bench>
  );
}
