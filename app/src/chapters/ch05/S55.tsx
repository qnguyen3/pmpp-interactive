import { useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source } from '../../components/notes';
import { Code, MATMUL_TIERS, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Translated, tr, useLocale } from '../../lib/i18n';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  CORRECT,
  gridSide,
  makeProblem,
  phases,
  runAll,
  runBlock,
  type Access,
  type Config,
  type Step,
  type ViolationKind,
} from '../../sim/tiled';

// ---------------------------------------------------------------- the five switches
type Opt = 'guardM' | 'guardN' | 'guardP' | 'zeroFillM' | 'zeroFillN';
type Opts = Record<Opt, boolean>;
const ALL_OFF: Opts = { guardM: false, guardN: false, guardP: false, zeroFillM: false, zeroFillN: false };
// Both barriers stay in place throughout: this section is about bounds, not about races.
const config = (width: number, tile: number, o: Opts): Config => ({ ...CORRECT, ...o, width, tile });
const isLoad = (s: Step) => s.op === 'loadM' || s.op === 'loadN';

const CHECKS: { key: Opt; label: string }[] = [
  {
    key: 'guardM',
    get label() {
      return tr('the load from M', 'lần nạp từ M');
    },
  },
  {
    key: 'guardN',
    get label() {
      return tr('the load from N', 'lần nạp từ N');
    },
  },
  {
    key: 'guardP',
    get label() {
      return tr('the store to P', 'lần ghi vào P');
    },
  },
];
const FILLS: { key: Opt; needs: Opt; label: string }[] = [
  {
    key: 'zeroFillM',
    needs: 'guardM',
    get label() {
      return tr('the Mds slot', 'vị trí trong Mds');
    },
  },
  {
    key: 'zeroFillN',
    needs: 'guardN',
    get label() {
      return tr('the Nds slot', 'vị trí trong Nds');
    },
  },
];

const KIND: Record<ViolationKind, { title: string; count: (n: number) => string }> = {
  'wrong-element': {
    get title() {
      return tr('Loads that fetched a wrong element', 'Các lần nạp lấy nhầm phần tử');
    },
    count: (n) => tr(`${n} load${n === 1 ? '' : 's'} fetched a wrong element`, `${n} lần nạp lấy nhầm phần tử`),
  },
  'outside-read': {
    get title() {
      return tr('Loads that read beyond the end of an array', 'Các lần nạp đọc vượt quá cuối mảng');
    },
    count: (n) => tr(`${n} load${n === 1 ? '' : 's'} read beyond the end of an array`, `${n} lần nạp đọc vượt quá cuối mảng`),
  },
  'bad-store': {
    get title() {
      return tr('Stores by threads that own no element of P', 'Các lần ghi bởi luồng không sở hữu phần tử nào của P');
    },
    count: (n) => tr(`${n} store${n === 1 ? '' : 's'} came from threads that own no element of P`, `${n} lần ghi đến từ các luồng không sở hữu phần tử nào của P`),
  },
  unfilled: {
    get title() {
      return tr('Tile slots used although nothing filled them', 'Các vị trí trong ô được dùng dù chưa có gì điền vào');
    },
    count: (n) => tr(`${n} time${n === 1 ? '' : 's'} a thread used a tile slot that nothing had filled`, `${n} lần một luồng dùng một vị trí trong ô mà chưa có gì điền vào`),
  },
  'read-before-write': {
    get title() {
      return tr('Tile slots read before they were loaded', 'Các vị trí trong ô bị đọc trước khi được nạp');
    },
    count: (n) => tr(`${n} read${n === 1 ? '' : 's'} of a slot before it was loaded`, `${n} lần đọc một vị trí trước khi nó được nạp`),
  },
  'overwritten-before-read': {
    get title() {
      return tr('Tile slots overwritten while still needed', 'Các vị trí trong ô bị ghi đè khi vẫn còn cần');
    },
    count: (n) => tr(`${n} read${n === 1 ? '' : 's'} of a slot that was already overwritten`, `${n} lần đọc một vị trí đã bị ghi đè`),
  },
};
const KIND_ORDER = Object.keys(KIND) as ViolationKind[];

type Status = 'correct' | 'lucky' | 'wrong';
const statusOf = (r: { correct: boolean; violations: unknown[] }): Status =>
  !r.correct ? 'wrong' : r.violations.length ? 'lucky' : 'correct';
const STATUS_MARK: Record<Status, string> = { correct: '✓', lucky: '!', wrong: '✗' };
const STATUS_WORD: Record<Status, string> = {
  get correct() {
    return tr('correct', 'đúng');
  },
  get lucky() {
    return tr('correct result, but with accesses that should not happen', 'kết quả đúng, nhưng có những truy cập lẽ ra không được xảy ra');
  },
  get wrong() {
    return tr('wrong result', 'kết quả sai');
  },
};

// ---------------------------------------------------------------- drawing
/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };

function cellBox(size: number, r: number, c: number, tile: number): CSSProperties {
  return {
    width: size,
    height: size,
    marginLeft: c > 0 && c % tile === 0 ? 5 : 0,
    marginTop: r > 0 && r % tile === 0 ? 5 : 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 2,
    border: '1px solid var(--rule)', // i18n-ignore: CSS value
    background: 'var(--cell)',
    color: 'var(--ink)',
    fontSize: size >= 36 ? '0.74rem' : '0.68rem',
    lineHeight: 1.12,
    fontVariantNumeric: 'tabular-nums',
  };
}
/** A position on the padded grid that is not part of the matrix. */
const ABSENT: CSSProperties = { border: '1px dashed var(--rule)', background: 'transparent', color: 'var(--ink-3)' }; // i18n-ignore: CSS value
const MARK: Record<Access | 'got', { style: CSSProperties; tag: string }> = {
  ok: { style: { border: '1px solid var(--global)', background: 'var(--global-wash)' }, tag: '' },
  skipped: {
    style: { border: '2px dashed var(--ink-3)', background: 'transparent', color: 'var(--ink-2)' },
    get tag() {
      return tr('skip', 'bỏ qua');
    },
  },
  wrap: {
    style: { border: '2px solid var(--fail)', background: 'var(--fail-wash)', color: 'var(--fail)' },
    get tag() {
      return tr('wraps', 'tràn dòng');
    },
  },
  outside: {
    style: { border: '2px solid var(--fail)', background: 'var(--fail-wash)', color: 'var(--fail)' },
    get tag() {
      return tr('out', 'ngoài');
    },
  },
  got: {
    style: { border: '2px solid var(--fail)', color: 'var(--fail)' },
    get tag() {
      return tr('got', 'lấy');
    },
  },
};

function Matrix({ name, n, note, children }: { name: string; n: number; note?: string; children: ReactNode }) {
  return (
    <figure style={{ margin: 0 }}>
      <figcaption style={{ marginBottom: 4 }}>
        <b className="t-global">{name}</b>
        {note ? <span className="muted"> {note}</span> : null}
      </figcaption>
      <div
        role="group"
        aria-label={tr(`matrix ${name}`, `ma trận ${name}`)}
        style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, max-content)`, gap: 2, justifyContent: 'start' }}
      >
        {children}
      </div>
    </figure>
  );
}
const cells = (n: number) => Array.from({ length: n * n }, (_, i) => ({ i, r: Math.floor(i / n), c: i % n }));

// ---------------------------------------------------------------- the explorer
const WIDTHS = [2, 3, 4, 5, 6, 7];
const TILES = [2, 3];
const SWEEP_WIDTHS = [1, 2, 3, 4, 5, 6, 7];
/** Messages listed per kind of violation before "and n more". */
const SHOWN = 2;

function BoundaryExplorer() {
  const [w, setW] = useState(3);
  const [t, setT] = useState(2);
  const [opts, setOpts] = useState<Opts>(ALL_OFF);
  const [pos, setPos] = useState({ bx: 0, by: 0, ph: 1 });

  const cfg = config(w, t, opts);
  const g = gridSide(cfg);
  const nPh = phases(cfg);
  const pad = g * t;
  const bx = Math.min(pos.bx, g - 1);
  const by = Math.min(pos.by, g - 1);
  const ph = Math.min(pos.ph, nPh - 1);

  const locale = useLocale();
  const { prob, res } = useMemo(() => {
    const prob = makeProblem(w);
    return { prob, res: runAll(config(w, t, opts), prob, 'lockstep') };
  }, [w, t, opts, locale]);
  const sweep = useMemo(
    () =>
      TILES.map((tile) => ({
        tile,
        runs: SWEEP_WIDTHS.map((width) => ({
          width,
          status: statusOf(runAll(config(width, tile, opts), makeProblem(width), 'lockstep')),
        })),
      })),
    [opts],
  );
  const status = statusOf(res);
  const trace = res.traces.find((x) => x.bx === bx && x.by === by)!;
  const loads = trace.steps.filter((s) => isLoad(s) && s.ph === ph);

  // What the selected block's loads in this phase touch, by position on the padded grid.
  const marks: Record<'M' | 'N', Map<string, { access?: Access; got?: boolean }>> = { M: new Map(), N: new Map() };
  const mark = (a: 'M' | 'N', r: number, c: number, m: { access?: Access; got?: boolean }) =>
    marks[a].set(`${r},${c}`, { ...marks[a].get(`${r},${c}`), ...m });
  for (const s of loads) {
    const gl = s.global!;
    const a = gl.array as 'M' | 'N';
    mark(a, gl.row, gl.col, { access: gl.access });
    if (gl.access === 'wrap') mark(a, Math.floor(gl.index / w), gl.index % w, { got: true });
  }

  const size = pad <= 4 ? 44 : pad <= 6 ? 38 : 31;
  const input = (a: 'M' | 'N') =>
    cells(pad).map(({ i, r, c }) => {
      const exists = r < w && c < w;
      const m = marks[a].get(`${r},${c}`);
      const look = m?.access ? MARK[m.access] : null;
      const tag = [look?.tag, m?.got ? MARK.got.tag : ''].filter(Boolean).join(' ');
      return (
        <div
          key={i}
          data-cell={`${a}-${r}-${c}`}
          data-exists={exists ? 1 : 0}
          data-access={m?.access ?? ''}
          data-got={m?.got ? 1 : 0}
          style={{
            ...cellBox(size, r, c, t),
            ...(exists ? null : ABSENT),
            ...look?.style,
            ...(m?.got ? MARK.got.style : null),
          }}
        >
          {exists || m ? (
            <span>
              {r},{c}
            </span>
          ) : null}
          {tag ? <b style={{ fontSize: '0.86em' }}>{tag}</b> : null}
        </div>
      );
    });

  const threads = Array.from({ length: t * t }, (_, i) => {
    const mine = trace.steps.filter((s) => s.thread === i);
    return {
      m: mine.find((s) => s.op === 'loadM' && s.ph === ph)!,
      n: mine.find((s) => s.op === 'loadN' && s.ph === ph)!,
      store: mine.find((s) => s.op === 'store')!,
    };
  });
  const loadCell = (s: Step) => {
    const gl = s.global!;
    const name = `${gl.array}[${gl.row}][${gl.col}]`;
    const got = `${gl.array}[${Math.floor(gl.index / w)}][${gl.index % w}]`;
    return (
      <td data-load={gl.array} data-access={gl.access}>
        <span className={gl.access === 'ok' ? 't-global' : undefined}>{name}</span>{' '}
        {gl.access === 'ok' ? (
          <span className="muted">
            <Translated vi="đã nạp">loaded</Translated>
          </span>
        ) : gl.access === 'skipped' ? (
          <span className="muted">
            {tr('skipped, ', 'bỏ qua, ')}
            {s.write?.unfilled ? tr('slot left as it was', 'vị trí giữ nguyên') : tr('slot set to 0', 'vị trí đặt về 0')}
          </span>
        ) : gl.access === 'wrap' ? (
          <span className="chip fail">
            <Translated vi={<>không tồn tại: đã lấy {got}</>}>does not exist: fetched {got}</Translated>
          </span>
        ) : (
          <span className="chip fail">
            <Translated vi="vượt quá cuối mảng">beyond the end of the array</Translated>
          </span>
        )}
      </td>
    );
  };

  const byKind = KIND_ORDER.map((kind) => ({ kind, list: res.violations.filter((v) => v.kind === kind) })).filter(
    (x) => x.list.length,
  );
  const summary = byKind.map((x) => KIND[x.kind].count(x.list.length)).join('; ');
  const nOn = (Object.keys(opts) as Opt[]).filter((k) => opts[k]).length;
  const outOfRange = loads.filter((s) => s.global!.access !== 'ok').length;

  const preset = (width: number, tile: number, at: typeof pos) => {
    setW(width);
    setT(tile);
    setPos(at);
  };

  return (
    <Bench
      title={tr('Boundary explorer', 'Trình khám phá biên')}
      task={tr(
        'Switch on the fewest options that make P correct for every width.',
        'Bật ít tùy chọn nhất mà vẫn làm P đúng với mọi độ rộng.',
      )}
      foot={tr(
        'Inputs are small integers and P is compared with a direct computation of M × N. All threads of a block run in step and both barriers are in place, so every failure here is a boundary failure.',
        'Đầu vào là các số nguyên nhỏ và P được so sánh với phép tính trực tiếp M × N. Mọi luồng của một khối chạy cùng nhịp và cả hai rào chắn (barrier) đều có mặt, nên mọi lỗi ở đây đều là lỗi biên.',
      )}
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi={<>Độ rộng W = {w}</>}>Width W = {w}</Translated>
          <span className="seg" role="group" aria-label={tr('Width W', 'Độ rộng W')}>
            {WIDTHS.map((x) => (
              <button key={x} aria-pressed={w === x} onClick={() => setW(x)}>
                {x}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi={<>Ô T = {t}</>}>Tile T = {t}</Translated>
          <span className="seg" role="group" aria-label={tr('Tile T', 'Ô T')}>
            {TILES.map((x) => (
              <button key={x} aria-pressed={t === x} onClick={() => setT(x)}>
                {x}
              </button>
            ))}
          </span>
        </span>
        <button className="quiet" data-preset="book" onClick={() => preset(3, 2, { bx: 0, by: 0, ph: 1 })}>
          <Translated vi="Ví dụ của sách (W=3, T=2)">The book's example (W=3, T=2)</Translated>
        </button>
        <button className="quiet" data-preset="narrow" onClick={() => preset(1, 2, { bx: 0, by: 0, ph: 0 })}>
          <Translated vi="Hẹp hơn một ô (W=1, T=2)">Narrower than a tile (W=1, T=2)</Translated>
        </button>
      </div>

      <div className="panel" style={{ marginBottom: '1rem' }}>
        <div className="row">
          <span role="group" aria-label={tr('Bounds checks', 'Kiểm tra biên')} className="row" style={{ gap: '0.4rem 1rem' }}>
            <b>
              <Translated vi="Kiểm tra biên của">Check the bounds of</Translated>
            </b>
            {CHECKS.map((o) => (
              <label className="check" key={o.key}>
                <input
                  type="checkbox"
                  data-opt={o.key}
                  checked={opts[o.key]}
                  onChange={(e) => setOpts({ ...opts, [o.key]: e.target.checked })}
                />
                {o.label}
              </label>
            ))}
          </span>
        </div>
        <div className="row" style={{ marginTop: '0.5rem' }}>
          <span role="group" aria-label={tr('Zero-fills', 'Ghi số 0')} className="row" style={{ gap: '0.4rem 1rem' }}>
            <b>
              <Translated vi="Khi một lần nạp bị bỏ qua, ghi 0 vào">When a load is skipped, write 0 into</Translated>
            </b>
            {FILLS.map((o) => (
              <label className="check" key={o.key} style={opts[o.needs] ? undefined : { color: 'var(--ink-3)' }}>
                <input
                  type="checkbox"
                  data-opt={o.key}
                  checked={opts[o.key]}
                  disabled={!opts[o.needs]}
                  onChange={(e) => setOpts({ ...opts, [o.key]: e.target.checked })}
                />
                {o.label}
              </label>
            ))}
            <span className="muted">
              <Translated vi="(cần phép kiểm tra tương ứng: một lần nạp không được kiểm tra thì không bao giờ bị bỏ qua)">
                (needs the matching check: an unchecked load is never skipped)
              </Translated>
            </span>
          </span>
        </div>
      </div>

      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Khối (x, y)">Block (x, y)</Translated>
          <span
            role="group"
            aria-label={tr('Block', 'Khối')}
            style={{ display: 'grid', gridTemplateColumns: `repeat(${g}, max-content)`, gap: 2 }}
          >
            {cells(g).map(({ i, r, c }) => (
              <button
                key={i}
                aria-label={tr(`block x ${c}, y ${r}`, `khối x ${c}, y ${r}`)}
                aria-pressed={bx === c && by === r}
                onClick={() => setPos({ bx: c, by: r, ph })}
                style={{ padding: '0.2rem 0.45rem', fontWeight: 450 }}
              >
                {c},{r}
              </button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          <Translated vi="Pha">Phase</Translated>
          <span className="seg" role="group" aria-label={tr('Phase', 'Pha')}>
            {Array.from({ length: nPh }, (_, p) => (
              <button key={p} aria-pressed={ph === p} onClick={() => setPos({ bx, by, ph: p })}>
                {p}
              </button>
            ))}
          </span>
        </span>
        <span data-readout="selection" style={{ paddingBottom: '0.36rem' }} aria-live="polite">
          <Translated
            vi={
              <>
                Khối (x {bx}, y {by}), pha {ph}:{' '}
                <b>
                  {outOfRange} trên {loads.length}
                </b>{' '}
                lần nạp nằm ngoài phạm vi.
              </>
            }
          >
            Block (x {bx}, y {by}), phase {ph}:{' '}
            <b>
              {outOfRange} of {loads.length}
            </b>{' '}
            loads are out of range.
          </Translated>
        </span>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.2rem', alignItems: 'flex-start' }}>
        <div style={{ overflowX: 'auto', maxWidth: '100%', paddingBottom: 2 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'max-content max-content', gap: 14 }}>
            <div />
            <Matrix name="N" n={pad}>
              {input('N')}
            </Matrix>
            <Matrix name="M" n={pad}>
              {input('M')}
            </Matrix>
            <Matrix name="P" n={pad} note={tr('(result of the whole run)', '(kết quả của cả lần chạy)')}>
              {cells(pad).map(({ i, r, c }) => {
                const exists = r < w && c < w;
                const li = r * w + c;
                const wrong = exists && res.wrong.includes(li);
                const mine = Math.floor(r / t) === by && Math.floor(c / t) === bx;
                return (
                  <div
                    key={i}
                    data-cell={`P-${r}-${c}`}
                    data-exists={exists ? 1 : 0}
                    data-wrong={wrong ? 1 : 0}
                    title={exists ? tr(`P[${r}][${c}]: expected ${prob.ref[li]}`, `P[${r}][${c}]: kỳ vọng ${prob.ref[li]}`) : undefined}
                    style={{
                      ...cellBox(size, r, c, t),
                      ...(exists ? null : ABSENT),
                      ...(wrong
                        ? { border: '1px solid var(--fail)', background: 'var(--fail-wash)', color: 'var(--fail)' } // i18n-ignore: CSS value
                        : null),
                      ...(mine ? { boxShadow: '0 0 0 2px var(--ink)' } : null),
                    }}
                  >
                    {exists ? <b>{Number.isNaN(res.P[li]) ? 'NaN' : res.P[li]}</b> : null}
                  </div>
                );
              })}
            </Matrix>
          </div>
          <p className="legend" style={{ margin: '0.8rem 0 0', maxWidth: 2 * pad * (size + 4) }}>
            <span>
              <i style={{ background: 'var(--global-wash)', border: '1px solid var(--global)' }} />
              <Translated vi="đã nạp">loaded</Translated>
            </span>
            <span>
              <i style={{ border: '2px dashed var(--ink-3)' }} />
              <Translated vi="bị một phép kiểm tra bỏ qua">skipped by a check</Translated>
            </span>
            <span>
              <i style={{ background: 'var(--fail-wash)', border: '2px solid var(--fail)' }} />
              <Translated vi="lấy nhầm, kết quả sai">wrong fetch, wrong result</Translated>
            </span>
            <span>
              <i style={{ border: '1px dashed var(--rule)' }} />
              <Translated vi="không thuộc ma trận">not part of the matrix</Translated>
            </span>
            <span>
              <i style={{ boxShadow: '0 0 0 2px var(--ink)' }} />
              <Translated vi="phần của P thuộc khối đang chọn">the selected block's part of P</Translated>
            </span>
          </p>
        </div>

        <div style={{ flex: '1 1 17rem', minWidth: '15rem', display: 'grid', gap: '0.8rem' }}>
          <div
            className="panel"
            data-verdict={status}
            aria-live="polite"
            style={{ borderColor: status === 'correct' ? 'var(--ok)' : 'var(--fail)' }}
          >
            {status === 'correct' ? (
              <p style={{ margin: 0 }}>
                <span className="chip ok">
                  <Translated vi="P đúng">P is correct</Translated>
                </span>{' '}
                <Translated vi="Mọi truy cập đều nằm trong mảng của nó.">Every access stayed inside its array.</Translated>
              </p>
            ) : status === 'lucky' ? (
              <p style={{ margin: 0 }}>
                <span className="chip fail">
                  <Translated vi="P đúng, nhưng không an toàn">P is correct, but not safely</Translated>
                </span>{' '}
                {summary}.
              </p>
            ) : (
              <p style={{ margin: 0 }}>
                <span className="chip fail">
                  <Translated vi={<>P sai ở {res.wrong.length} trên {w * w} phần tử</>}>
                    P is wrong in {res.wrong.length} of {w * w} elements
                  </Translated>
                </span>{' '}
                {summary}.
              </p>
            )}
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>
              <Translated
                vi={
                  <>
                    W = {w}, T = {t}: {g * g} khối, {nPh} pha. Tùy chọn đang bật: {nOn} trên 5.
                  </>
                }
              >
                W = {w}, T = {t}: {g * g} block{g * g === 1 ? '' : 's'}, {nPh} phase{nPh === 1 ? '' : 's'}. Options on:{' '}
                {nOn} of 5.
              </Translated>
            </p>
          </div>
          {byKind.map(({ kind, list }) => (
            <div className="panel" key={kind} data-kind={kind}>
              <p style={{ margin: '0 0 0.3rem' }}>
                <b>{KIND[kind].title}</b> <span className="muted">({list.length})</span>
              </p>
              <ul style={{ margin: 0, paddingLeft: '1.1em' }}>
                {list.slice(0, SHOWN).map((v, i) => (
                  <li key={i} style={{ marginBottom: '0.2em' }}>
                    <Translated vi={<>khối ({v.bx}, {v.by}): {v.message}</>}>
                      block ({v.bx}, {v.by}): {v.message}
                    </Translated>
                  </li>
                ))}
              </ul>
              {list.length > SHOWN ? (
                <p className="muted" style={{ margin: '0.3rem 0 0' }}>
                  <Translated vi={<>và {list.length - SHOWN} mục nữa</>}>and {list.length - SHOWN} more</Translated>
                </p>
              ) : null}
            </div>
          ))}
        </div>
      </div>

      <div className="panel" style={{ margin: '1rem 0', overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0 }}>
          <thead>
            <tr>
              <th>
                <Translated vi={<>Luồng của khối ({bx}, {by})</>}>
                  Thread of block ({bx}, {by})
                </Translated>
              </th>
              <th>
                <Translated vi="Phần tử của nó trong P">Its element of P</Translated>
              </th>
              <th>
                <Translated vi={<>Lần nạp M ở pha {ph}</>}>M load in phase {ph}</Translated>
              </th>
              <th>
                <Translated vi={<>Lần nạp N ở pha {ph}</>}>N load in phase {ph}</Translated>
              </th>
            </tr>
          </thead>
          <tbody>
            {threads.map((th) => {
              const st = th.store.global!;
              return (
                <tr key={th.m.thread} data-thread={`${th.m.tx}-${th.m.ty}`}>
                  <td className="nowrap">
                    tx {th.m.tx}, ty {th.m.ty}{/* i18n-ignore: thread index names */}
                  </td>
                  <td data-owns={st.row < w && st.col < w ? 1 : 0}>
                    {st.row < w && st.col < w ? (
                      <span className="t-global">
                        P[{st.row}][{st.col}]
                      </span>
                    ) : (
                      <span className="muted">
                        <Translated vi={<>không có: P[{st.row}][{st.col}] không tồn tại</>}>
                          none: P[{st.row}][{st.col}] does not exist
                        </Translated>
                      </span>
                    )}
                  </td>
                  {loadCell(th.m)}
                  {loadCell(th.n)}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="panel" data-readout="sweep">
        <p style={{ margin: '0 0 0.5rem' }}>
          <b>
            <Translated vi="Cùng các tùy chọn ở mọi độ rộng.">The same options at every width.</Translated>
          </b>{' '}
          <span className="muted">
            <Translated vi="Bấm vào một ô để xem.">Click one to look at it.</Translated>
          </span>
        </p>
        {sweep.map(({ tile, runs }) => (
          <div className="row" key={tile} style={{ gap: '0.3rem', marginBottom: '0.3rem' }}>
            <span style={{ minWidth: '3.2rem' }}>T = {tile}</span>
            {runs.map(({ width, status: s }) => (
              <button
                key={width}
                data-sweep={`${width}-${tile}`}
                data-status={s}
                aria-label={tr(`width ${width}, tile ${tile}: ${STATUS_WORD[s]}`, `độ rộng ${width}, ô ${tile}: ${STATUS_WORD[s]}`)}
                title={STATUS_WORD[s]}
                onClick={() => preset(width, tile, { bx, by, ph })}
                style={{
                  padding: '0.2rem 0.5rem',
                  fontWeight: 450,
                  border: `1px solid var(--${s === 'correct' ? 'ok' : 'fail'})`,
                  background: `var(--${s === 'correct' ? 'ok' : 'fail'}-wash)`,
                  color: 'var(--ink)',
                  boxShadow: width === w && tile === t ? '0 0 0 2px var(--ink)' : undefined,
                }}
              >
                W={width} {STATUS_MARK[s]}
              </button>
            ))}
          </div>
        ))}
        <p className="muted" style={{ margin: '0.4rem 0 0' }}>
          <Translated vi="✓ đúng. ✗ kết quả sai. ! kết quả đúng nhưng đạt được qua những truy cập lẽ ra không được xảy ra.">
            ✓ correct. ✗ wrong result. ! correct result reached through accesses that should not happen.
          </Translated>
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- rectangular matrices
type Dim = 'j' | 'k' | 'l';
const USES: { code: string; what: string; answer: Dim; nudge: string }[] = [
  {
    code: 'Row < Width',
    get what() {
      return tr('the bound on the row of the output', 'giới hạn trên hàng của đầu ra');
    },
    answer: 'j',
    get nudge() {
      return tr('Row counts rows of C, which are also the rows of A. How many are there?', 'Row đếm các hàng của C, cũng là các hàng của A. Có bao nhiêu hàng?');
    },
  },
  {
    code: 'ph*TILE_WIDTH + tx < Width',
    get what() {
      return tr('the column bound of the load from A', 'giới hạn cột của lần nạp từ A');
    },
    answer: 'k',
    get nudge() {
      return tr('This index runs along a row of A. How many columns does A have?', 'Chỉ số này chạy dọc một hàng của A. A có bao nhiêu cột?');
    },
  },
  {
    code: 'ph*TILE_WIDTH + ty < Width',
    get what() {
      return tr('the row bound of the load from B', 'giới hạn hàng của lần nạp từ B');
    },
    answer: 'k',
    get nudge() {
      return tr('This index runs down a column of B. How many rows does B have?', 'Chỉ số này chạy xuống một cột của B. B có bao nhiêu hàng?');
    },
  },
  {
    code: 'Col < Width',
    get what() {
      return tr('the bound on the column of the output', 'giới hạn trên cột của đầu ra');
    },
    answer: 'l',
    get nudge() {
      return tr('Col counts columns of C, which are also the columns of B.', 'Col đếm các cột của C, cũng là các cột của B.');
    },
  },
  {
    code: 'M[Row * Width + …]',
    get what() {
      return tr('the stride in the index into A', 'bước nhảy (stride) trong chỉ số vào A');
    },
    answer: 'k',
    get nudge() {
      return tr('Moving one row down a row-major array skips one whole row. How long is a row of A?', 'Đi xuống một hàng trong mảng thứ tự hàng (row-major) là bỏ qua trọn một hàng. Một hàng của A dài bao nhiêu?');
    },
  },
  {
    code: 'N[(…) * Width + Col]',
    get what() {
      return tr('the stride in the index into B', 'bước nhảy trong chỉ số vào B');
    },
    answer: 'l',
    get nudge() {
      return tr('How long is a row of B?', 'Một hàng của B dài bao nhiêu?');
    },
  },
  {
    code: 'P[Row * Width + Col]',
    get what() {
      return tr('the stride in the index into C', 'bước nhảy trong chỉ số vào C');
    },
    answer: 'l',
    get nudge() {
      return tr('How long is a row of C?', 'Một hàng của C dài bao nhiêu?');
    },
  },
  {
    code: '(Width + TILE_WIDTH - 1) / TILE_WIDTH',
    get what() {
      return tr('the number of phases', 'số pha');
    },
    answer: 'k',
    get nudge() {
      return tr('A phase handles one tile of the terms of a dot product. How many terms does each dot product have?', 'Một pha xử lý một ô các số hạng của một tích vô hướng. Mỗi tích vô hướng có bao nhiêu số hạng?');
    },
  },
  {
    code: 'dimGrid.x',
    get what() {
      return tr('how many blocks the grid needs across', 'lưới (grid) cần bao nhiêu khối theo chiều ngang');
    },
    answer: 'l',
    get nudge() {
      return tr('blockIdx.x feeds Col. Which dimension must the blocks cover in x?', 'blockIdx.x đi vào Col. Các khối phải phủ chiều nào theo x?');
    },
  },
  {
    code: 'dimGrid.y',
    get what() {
      return tr('how many blocks the grid needs down', 'lưới cần bao nhiêu khối theo chiều dọc');
    },
    answer: 'j',
    get nudge() {
      return tr('blockIdx.y feeds Row. Which dimension must the blocks cover in y?', 'blockIdx.y đi vào Row. Các khối phải phủ chiều nào theo y?');
    },
  },
];

function ReferenceKernel() {
  const file = useLabFile('ch05/lab3_gemm/solution/kernel.cu');
  const { reports } = useReports();
  // Mark every line of code (not comment) that uses one of the three dimensions.
  const highlight = (file ?? '')
    .split('\n')
    .flatMap((ln, i) => (/\b[jkl]\b/.test(ln.replace(/\/\/.*$/, '')) ? [i + 1] : []));
  return (
    <Code
      tabs={[
        {
          label: 'CUDA',
          code: file ?? '// loading labs/ch05/lab3_gemm/solution/kernel.cu',
          tiers: MATMUL_TIERS,
          status: solutionTabStatus(reports, 'ch05/lab3_gemm', 'kernel.cu'),
        },
      ]}
      highlight={highlight}
      caption={
        <Translated
          vi={
            <>
              Lời giải tham chiếu (solution) của lab 5.3, <code>labs/ch05/lab3_gemm/solution/kernel.cu</code>, với mọi
              dòng dùng j, k hoặc l được đánh dấu. Nó khác Hình 5.13 của sách (tr. 115) ở ba điểm: nó dành cho ma trận chữ
              nhật, nó đặt tên các ma trận là A, B và C, và nó làm tròn lên số pha bằng số học nguyên ở chỗ sách viết{' '}
              <code>ceil(Width/(float)TILE_WIDTH)</code>.
            </>
          }
        >
          <>
            The reference solution of lab 5.3, <code>labs/ch05/lab3_gemm/solution/kernel.cu</code>, with every line that
            uses j, k or l marked. It differs from the book's Fig. 5.13 (p. 115) in three ways: it is rectangular, it
            names the matrices A, B and C, and it rounds the phase count up in integer arithmetic where the book writes{' '}
            <code>ceil(Width/(float)TILE_WIDTH)</code>.
          </>
        </Translated>
      }
    />
  );
}

function RectangularExercise() {
  const [picks, setPicks] = useState<string[]>(() => USES.map(() => ''));
  // The answers as they stood at the last press of Check; a row shows feedback only while unchanged since then.
  const [checked, setChecked] = useState<string[] | null>(null);
  const [show, setShow] = useState(false);
  const right = USES.filter((u, i) => picks[i] === u.answer).length;
  const allRight = checked !== null && USES.every((u, i) => checked[i] === u.answer && picks[i] === u.answer);
  const check = () => {
    setChecked(picks);
    if (right === USES.length) setShow(true);
  };
  return (
    <>
      <Bench
        title={tr('Which dimension did Width mean?', 'Width đã mang nghĩa chiều nào?')}
        task={tr(
          'C[j×l] = A[j×k] × B[k×l]. For each use of Width in the square kernel, choose what it becomes.',
          'C[j×l] = A[j×k] × B[k×l]. Với mỗi chỗ dùng Width trong kernel vuông, hãy chọn nó trở thành gì.',
        )}
        foot={tr(
          'A is M, B is N and C is P of the square kernel. The loads use Row and Col as well as their own index, so the first and fourth rows each appear in two of the three tests.',
          'A là M, B là N và C là P của kernel vuông. Các lần nạp dùng Row và Col cùng với chỉ số riêng của chúng, nên hàng thứ nhất và thứ tư mỗi hàng xuất hiện trong hai trên ba phép kiểm tra.',
        )}
      >
        <div className="panel" style={{ overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th>
                  <Translated vi="Trong kernel vuông">In the square kernel</Translated>
                </th>
                <th style={{ width: '25%' }}>
                  <Translated vi="Nó là gì">What it is</Translated>
                </th>
                <th>
                  <Translated vi="Trở thành">Becomes</Translated>
                </th>
                <th style={{ width: '36%' }}>
                  <Translated vi="Phản hồi">Feedback</Translated>
                </th>
              </tr>
            </thead>
            <tbody>
              {USES.map((u, i) => {
                const fresh = checked !== null && checked[i] === picks[i];
                const ok = picks[i] === u.answer;
                return (
                  <tr key={i} data-use={i} data-result={fresh ? (picks[i] === '' ? 'empty' : ok ? 'right' : 'wrong') : ''}>
                    <td>
                      <code>{u.code}</code>
                    </td>
                    <td>{u.what}</td>
                    <td>
                      <select
                        aria-label={tr(`${u.what} becomes`, `${u.what} trở thành`)}
                        value={picks[i]}
                        onChange={(e) => setPicks(picks.map((p, n) => (n === i ? e.target.value : p)))}
                      >
                        <option value="">{tr('choose', 'chọn')}</option>
                        <option value="j">j</option>
                        <option value="k">k</option>
                        <option value="l">l</option>
                      </select>
                    </td>
                    <td>
                      {!fresh ? null : picks[i] === '' ? (
                        <span className="muted">
                          <Translated vi="Chưa trả lời.">Not answered yet.</Translated>
                        </span>
                      ) : ok ? (
                        <span className="chip ok">
                          <Translated vi="đúng">right</Translated>
                        </span>
                      ) : (
                        <>
                          <span className="chip fail">
                            <Translated vi={<>không phải {picks[i]}</>}>not {picks[i]}</Translated>
                          </span>{' '}
                          {u.nudge}
                        </>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="row" style={{ marginTop: '0.9rem' }}>
          <button className="primary" onClick={check}>
            <Translated vi="Kiểm tra">Check</Translated>
          </button>
          <button className="quiet" onClick={() => setShow(!show)} aria-expanded={show}>
            {show
              ? tr('Hide the reference kernel', 'Ẩn kernel tham chiếu')
              : tr('Show the reference kernel', 'Hiện kernel tham chiếu')}
          </button>
          <span data-readout="rect" aria-live="polite">
            {checked === null ? null : allRight ? (
              <span className="chip ok">
                <Translated vi={<>Đúng cả {USES.length}.</>}>All {USES.length} right.</Translated>
              </span>
            ) : (
              <>
                <Translated
                  vi={
                    <>
                      Đúng {USES.filter((u, i) => checked[i] === u.answer).length} trên {USES.length} ở lần kiểm tra gần
                      nhất.
                    </>
                  }
                >
                  {USES.filter((u, i) => checked[i] === u.answer).length} of {USES.length} right at the last check.
                </Translated>
              </>
            )}
          </span>
        </div>
      </Bench>
      {show ? <ReferenceKernel /> : null}
    </>
  );
}

// ---------------------------------------------------------------- questions
// The count in the numeric question is read off the simulator, not worked out by hand.
const BLOCK11 = runBlock(config(3, 2, ALL_OFF), makeProblem(3), 1, 1, 'lockstep').steps.filter(
  (s) => isLoad(s) && s.ph === 0,
);
const BLOCK11_BAD = BLOCK11.filter((s) => s.global!.access !== 'ok').length;

export const questionsS55: Record<string, Question> = {
  'ch05/5.5/q-early-return': {
    kind: 'choice',
    id: 'ch05/5.5/q-early-return',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Một đồng nghiệp dọn dẹp kernel đã kiểm tra biên bằng cách thêm câu lệnh đầu tiên{' '}
              <code>if (Row &gt;= Width || Col &gt;= Width) return;</code> với lý do một luồng như vậy chẳng có gì để ghi. Kernel
              có còn đúng với mọi độ rộng không?
            </>
          }
        >
          A colleague tidies the boundary-checked kernel by adding, as its first statement,{' '}
          <code>if (Row &gt;= Width || Col &gt;= Width) return;</code> on the grounds that such a thread has nothing to
          store. Is the kernel still correct for every width?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <Translated vi="Đúng. Một luồng không sở hữu phần tử nào của P không đóng góp gì vào kết quả, nên nó có thể rời đi.">
            Yes. A thread that owns no element of P contributes nothing to the result, so it may leave.
          </Translated>
        ),
        why: (
          <Translated vi="Nó đóng góp các lần nạp. Những vị trí nào của Mds và Nds chỉ có mình luồng đó điền vào?">
            It contributes loads. Which slots of Mds and Nds is that thread the only one to fill?
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không. Luồng vẫn phải điền các vị trí của nó trong Mds và Nds, những vị trí mà các luồng bên cạnh có đầu ra hợp lệ sẽ đọc.">
            No. The thread must still fill its slots of Mds and Nds, which neighbours with valid outputs read.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Đúng, miễn là ba phép kiểm tra biên và các lần ghi số 0 vẫn nằm trong vòng lặp pha.">
            Yes, as long as the three bounds tests and the zero-writes stay inside the phase loop.
          </Translated>
        ),
        why: (
          <Translated vi="Những thứ đó chỉ chạy trong các luồng đến được vòng lặp. Luồng đã return không bao giờ ghi vào vị trí của nó, kể cả số 0.">
            Those run only in threads that reach the loop. The returned thread never writes its slots at all, not even a zero.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không, nhưng chỉ vì luồng đã return giờ bỏ qua lần ghi vào P.">
            No, but only because the returned thread now skips its store to P.
          </Translated>
        ),
        why: (
          <Translated vi="Nó không có gì để ghi: đó là điều duy nhất mà việc return sớm làm đúng.">
            It has no store to make: that is the one thing the early return gets right.
          </Translated>
        ),
      },
    ],
    explain: (
      <>
        <Translated
          vi={
            <>
              <p>
                Trong khối (1,1) của ví dụ 3×3, luồng có tx = 1, ty = 0 có Col = 3 và không sở hữu gì. Ở pha 0, nó vẫn là luồng
                nạp M[2][1] vào <code>Mds[0][1]</code>, và chủ của P[2][2] nhân với vị trí đó. Return sớm thì vị trí đó giữ
                nguyên thứ đã có trong nó. Đầu ra của một luồng có hợp lệ không và từng lần nạp của nó có hợp lệ không là hai câu
                hỏi riêng, đó là lý do kernel có ba phép kiểm tra chứ không phải một.
              </p>
              <p>
                Còn vấn đề thứ hai: một luồng đã return sẽ không còn đến <code>__syncthreads()</code>, mà quy tắc của chương 4 là
                một rào chắn (barrier) phải được mọi luồng của khối thực thi.
              </p>
            </>
          }
        >
          <p>
            In block (1,1) of the 3×3 example, the thread with tx = 1, ty = 0 has Col = 3 and owns nothing. In phase 0
            it is still the thread that loads M[2][1] into <code>Mds[0][1]</code>, and the owner of P[2][2] multiplies by
            that slot. Return early and the slot keeps whatever was in it. Whether a thread's output is valid and whether
            each of its loads is valid are separate questions, which is why the kernel has three tests and not one.
          </p>
          <p>
            There is a second problem: a thread that has returned no longer arrives at <code>__syncthreads()</code>, and
            chapter 4's rule is that a barrier must be executed by all threads of the block.
          </p>
        </Translated>
      </>
    ),
    source: 'ch. 5 §5.5, pp. 113–114; the barrier rule is ch. 4 §4.3, p. 72',
    vi: { source: 'chương 5 §5.5, tr. 113–114; quy tắc về rào chắn ở chương 4 §4.3, tr. 72' },
  },
  'ch05/5.5/q-silent': {
    kind: 'choice',
    id: 'ch05/5.5/q-silent',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Hai ma trận 5×5, mỗi ma trận được lưu theo thứ tự hàng (row-major) trong một mảng 25 số float. Một kernel không
              kiểm tra biên đọc <code>M[1][6]</code> và <code>N[6][1]</code>, tính chỉ số phẳng như thường lệ. Chuyện gì xảy ra?
            </>
          }
        >
          Two 5×5 matrices are each stored row-major in an array of 25 floats. A kernel with no bounds checks reads{' '}
          <code>M[1][6]</code> and <code>N[6][1]</code>, computing the flat index as usual. What happens?
        </Translated>
      </>
    ),
    options: [
      {
        label: (
          <Translated vi="Cả hai lần đọc đều rơi ra ngoài mảng 25 phần tử của chúng, nên cả hai có thể gây lỗi hoặc trả về rác.">
            Both reads fall outside their 25-element arrays, so both may fault or return garbage.
          </Translated>
        ),
        why: (
          <Translated vi="Tính chỉ số phẳng của M[1][6]: 1·5 + 6 = 11, nằm trong mảng.">
            Work out the flat index of M[1][6]: 1·5 + 6 = 11, which is inside the array.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="M[1][6] lặng lẽ trả về M[2][1]. N[6][1] là chỉ số 31, vượt quá cuối mảng: rác hoặc lỗi.">
            M[1][6] quietly returns M[2][1]. N[6][1] is index 31, past the end: garbage or a fault.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="M[1][6] rơi ra ngoài mảng. N[6][1] lặng lẽ trả về một phần tử khác của N.">
            M[1][6] falls outside the array. N[6][1] quietly returns some other element of N.
          </Translated>
        ),
        why: (
          <Translated vi="Ngược lại mới đúng. Chỉ số cột quá lớn rơi vào hàng kế tiếp; chỉ số hàng quá lớn bị nhân với cả độ dài hàng.">
            The other way round. A column index that is too large lands in the next row; a row index that is too large is multiplied by the whole row length.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cả hai lặng lẽ trả về một phần tử khác của chính ma trận đó, xa hơn khoảng một hàng.">
            Both quietly return some other element of their own matrix, a row or so further on.
          </Translated>
        ),
        why: (
          <Translated vi="6·5 + 1 = 31, mà mảng kết thúc ở chỉ số 24.">
            6·5 + 1 = 31, and the array ends at index 24.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lưu theo thứ tự hàng đặt hàng r + 1 ngay sau hàng r. Vì vậy chạy quá cuối một hàng vài phần tử sẽ rơi vào hàng kế
            tiếp: một địa chỉ hợp lệ chứa nhầm phần tử, và không có lỗi nào. Chạy quá cuối một cột nghĩa là chỉ số hàng vượt
            quá hàng cuối, và mọi hàng như vậy đều nằm ngoài cuối mảng. Trường hợp lặng lẽ là trường hợp các bài kiểm thử của
            bạn phải bắt được, vì không gì khác sẽ bắt.
          </p>
        }
      >
        <p>
          Row-major storage puts row r + 1 directly after row r. Running a few elements past the end of a row therefore
          lands in the next row: a valid address holding the wrong element, and no error. Running past the end of a
          column means a row index beyond the last row, and every such row lies beyond the end of the array. The quiet
          case is the one your tests have to catch, because nothing else will.
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.5, p. 113',
    vi: { source: 'chương 5 §5.5, tr. 113' },
  },
  'ch05/5.5/q-count': {
    kind: 'numeric',
    id: 'ch05/5.5/q-count',
    prompt: (
      <>
        <Translated
          vi={
            <>
              Ma trận 3×3, ô 2×2, không kiểm tra biên. Trước hết xác định kernel phải chạy bao nhiêu pha. Sau đó: trong{' '}
              {BLOCK11.length} lần nạp mà khối (1,1) phát ra ở pha 0 (bốn từ M, bốn từ N), bao nhiêu lần nằm ngoài phạm vi?
            </>
          }
        >
          3×3 matrices, 2×2 tiles, no bounds checks. First settle how many phases the kernel must run. Then: of the{' '}
          {BLOCK11.length} loads that block (1,1) issues in phase 0 (four from M, four from N), how many are out of
          range?
        </Translated>
      </>
    ),
    answer: BLOCK11_BAD,
    unit: 'loads',
    explain: (
      <Translated
        vi={
          <p>
            Hai pha: 3 / 2 làm tròn lên. Khối (1,1) có Row = 2 + ty và Col = 2 + tx. Ở pha 0, các lần nạp từ M của nó là
            M[2 + ty][tx], và hai lần có ty = 1 xin hàng 3. Các lần nạp từ N là N[ty][2 + tx], và hai lần có tx = 1 xin cột 3.
            Vậy là {BLOCK11_BAD} trên {BLOCK11.length}, ở pha đầu tiên, không phải pha cuối. Hai lần từ M vượt quá cuối mảng;
            hai lần từ N tràn sang hàng sau và lấy N[1][0] và N[2][0].
          </p>
        }
      >
        <p>
          Two phases: 3 / 2 rounded up. Block (1,1) has Row = 2 + ty and Col = 2 + tx. In phase 0 its loads from M are
          M[2 + ty][tx], and the two with ty = 1 ask for row 3. Its loads from N are N[ty][2 + tx], and the two with tx
          = 1 ask for column 3. That is {BLOCK11_BAD} of {BLOCK11.length}, in the first phase, not the last. The two from
          M are beyond the end of the array; the two from N wrap around and fetch N[1][0] and N[2][0].
        </p>
      </Translated>
    ),
    source: 'ch. 5 §5.5, p. 113, Fig. 5.12',
    vi: { unit: 'lần nạp', source: 'chương 5 §5.5, tr. 113, Hình 5.12' },
  },
};

// ---------------------------------------------------------------- the section
export function S55() {
  return (
    <>
      <Translated
        vi={
          <p>
            Kernel phân ô của phần 5.4 dựa trên hai giả định: độ rộng ma trận là bội số của độ rộng ô, và các ma trận là ma
            trận vuông. Kích thước thực tế chẳng tuân theo điều nào. Các ma trận đứng sau một lớp tuyến tính là batch × hidden
            và hidden × vocabulary, và không con số nào trong đó được chọn để vừa với ô của bạn. Phần này bỏ cả hai giả định.
          </p>
        }
      >
        <p>
          The tiled kernel of section 5.4 rests on two assumptions: the width of the matrices is a multiple of the tile
          width, and the matrices are square. Real shapes honour neither. The matrices behind a linear layer are batch ×
          hidden and hidden × vocabulary, and none of those numbers was chosen to suit your tile. This section removes
          both assumptions.
        </p>
      </Translated>
      <Source section="5.5" pages="112" />

      <h2>
        <Translated vi="Điều gì hỏng ở biên">What goes wrong at the edge</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lấy ma trận 3×3 và ô 2×2. Giả sử trước hết bạn sửa theo cách hiển nhiên và làm tròn lên: 2×2 khối để phủ hết P, và
            hai pha để phủ cả ba số hạng của mỗi tích vô hướng. Khi đó ô cuối cùng theo mỗi chiều chỉ có một nửa. Hãy nhìn khối
            (0,0) ở pha 1. Ô M của nó lẽ ra là các cột 2 và 3 của các hàng 0 và 1, mà không có cột 3. Ô N của nó lẽ ra là các
            hàng 2 và 3 của các cột 0 và 1, mà không có hàng 3. Kernel vẫn nạp tất cả như thường.
          </p>
        }
      >
        <p>
          Take 3×3 matrices and 2×2 tiles. Suppose you make the obvious repair first and round up: 2×2 blocks so that
          all of P is covered, and two phases so that all three terms of each dot product are. The last tile in each
          direction is then only half there. Look at block (0,0) in phase 1. Its tile of M should be columns 2 and 3 of
          rows 0 and 1, and there is no column 3. Its tile of N should be rows 2 and 3 of columns 0 and 1, and there is
          no row 3. The kernel loads them all the same.
        </p>
      </Translated>
      <Source section="5.5" pages="112–113" figs={tr('Fig. 5.11', 'Hình 5.11')} />
      <Translated vi={<p>Hai lỗi trông giống nhau nhưng hành xử khác nhau. Lý do là cách bố trí theo thứ tự hàng.</p>}>
        <p>The two mistakes look alike and behave differently. The reason is the row-major layout.</p>
      </Translated>
      <Predict
        id="ch05/5.5/p-wrap"
        prompt={
          <Translated
            vi={
              <>
                M là ma trận 3×3, lưu theo thứ tự hàng trong một mảng phẳng 9 số float. Một luồng không kiểm tra biên với{' '}
                <code>Row = 0</code> tính <code>M[Row*Width + 3]</code>, định đọc M[0][3]. Nó đọc chỉ số nào của mảng?
              </>
            }
          >
            <>
              M is 3×3, stored row-major in a flat array of 9 floats. An unchecked thread with <code>Row = 0</code>{' '}
              evaluates <code>M[Row*Width + 3]</code>, meaning to read M[0][3]. Which index of the array does it read?
            </>
          </Translated>
        }
      >
        {(v) => (
          <p>
            <Translated
              vi={
                <>
                  Chỉ số 0·3 + 3 = <b>3</b>, và chỉ số 3 là nơi hàng 1 bắt đầu: luồng nhận được M[1][0].
                </>
              }
            >
              Index 0·3 + 3 = <b>3</b>, and index 3 is where row 1 begins: the thread gets M[1][0].
            </Translated>{' '}
            {v === 3
              ? tr('That is what you had.', 'Đúng như bạn đã trả lời.')
              : tr(
                  'The flat index is all the hardware sees; it has no idea a row ended.',
                  'Chỉ số phẳng là tất cả những gì phần cứng thấy; nó không hề biết một hàng đã kết thúc.',
                )}{' '}
            <Translated vi="Không có lỗi nào, cũng không có cảnh báo nào.">Nothing faults and nothing warns.</Translated>
          </p>
        )}
      </Predict>
      <Translated
        vi={
          <p>
            Vậy một lần đọc quá cuối một <b>hàng</b> là lặng lẽ. Địa chỉ nằm trong mảng; nó chỉ thuộc về hàng kế tiếp. Luồng
            nhận một số hợp lệ nhưng là nhầm phần tử, và kết quả bị hỏng một cách lặng lẽ. Một lần đọc quá cuối một <b>cột</b>{' '}
            thì khác. N[3][0] là chỉ số 3·3 + 0 = 9 trong một mảng có chỉ số cuối là 8. Đó là ngoài vùng cấp phát: thứ trả về
            là bất cứ gì nằm sau mảng, hoặc trên một số hệ thống chương trình bị dừng.
          </p>
        }
      >
        <p>
          So a read past the end of a <b>row</b> is silent. The address is inside the array; it simply belongs to the
          next row. The thread receives a valid number that is the wrong element, and the result is quietly corrupted. A
          read past the end of a <b>column</b> is different. N[3][0] is index 3·3 + 0 = 9 in an array whose last index is
          8. That is outside the allocation: what comes back is whatever lies beyond the array, or on some systems the
          program is stopped.
        </p>
      </Translated>
      <Source section="5.5" pages="113" />
      <Translated
        vi={
          <p>
            Dễ nghĩ rằng chỉ pha cuối bị ảnh hưởng. Không phải vậy. Khối (1,1) phủ các hàng 2 và 3 và các cột 2 và 3 của P. Ở
            pha 0, ô M của nó là các hàng 2 và 3 của các cột 0 và 1, mà không có hàng 3; ô N của nó là các hàng 0 và 1 của các
            cột 2 và 3, mà không có cột 3. Trong ví dụ này, mọi khối ở mép phải hoặc mép dưới của lưới (grid) đều có lần nạp
            ngoài phạm vi ở mọi pha.
          </p>
        }
      >
        <p>
          It is tempting to think only the last phase is affected. It is not. Block (1,1) covers rows 2 and 3 and columns
          2 and 3 of P. In phase 0 its tile of M is rows 2 and 3 of columns 0 and 1, and there is no row 3; its tile of N
          is rows 0 and 1 of columns 2 and 3, and there is no column 3. In this example every block on the right or bottom
          edge of the grid makes out-of-range loads in every phase.
        </p>
      </Translated>
      <Source section="5.5" pages="113" figs={tr('Fig. 5.12', 'Hình 5.12')} />

      <h2>
        <Translated vi="Vì sao một phép kiểm tra là không đủ">Why one test is not enough</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel ngây thơ có một phép kiểm tra duy nhất: luồng này có sở hữu một phần tử của P không? Luồng nào không qua thì
            không làm gì. Điều đó không áp dụng được ở đây. Trong khối (1,1), luồng có tx = 1, ty = 0 sẽ tính P[2][3], phần tử
            không tồn tại. Thế nhưng ở pha 0 nó là luồng nạp M[2][1] vào ô chia sẻ, và luồng bên cạnh ở tx = 0, chủ của
            P[2][2], cần giá trị đó. Cho luồng thứ nhất về sớm thì luồng thứ hai tính ra rác.
          </p>
        }
      >
        <p>
          The naive kernel had a single test: does this thread own an element of P? Threads that failed it did nothing.
          That does not carry over. In block (1,1) the thread with tx = 1, ty = 0 would compute P[2][3], which does not
          exist. Yet in phase 0 it is the thread that loads M[2][1] into the shared tile, and its neighbour at tx = 0,
          the owner of P[2][2], needs that value. Send the first thread home early and the second computes garbage.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Ở đây các luồng được gọi theo tx và ty. Sách viết thread(hàng, cột), tức là (ty, tx); các thông báo của trình khám
              phá viết (tx, ty), x trước, giống <code>threadIdx</code> và <code>blockIdx</code>.
            </p>
          }
        >
          <p>
            Threads are named by tx and ty here. The book writes thread(row, column), which is (ty, tx); the explorer's
            messages write (tx, ty), x first, like <code>threadIdx</code> and <code>blockIdx</code>.
          </p>
        </Translated>
      </Aside>
      <Translated
        vi={
          <p>
            Điều ngược lại cũng sai. Trong khối (0,0), luồng có tx = 1, ty = 0 sở hữu P[0][1], một phần tử hoàn toàn hợp lệ, và
            ở pha 1 nó với tới M[0][3]. Sở hữu một đầu ra hợp lệ không nói gì về việc một lần nạp cụ thể có hợp lệ hay không.
          </p>
        }
      >
        <p>
          The converse fails too. In block (0,0) the thread with tx = 1, ty = 0 owns P[0][1], a perfectly good element,
          and in phase 1 it reaches for M[0][3]. Owning a valid output says nothing about whether a particular load is
          valid.
        </p>
      </Translated>
      <Source section="5.5" pages="113–114" />
      <Translated vi={<p>Vì vậy mỗi truy cập có phép kiểm tra riêng, viết trên chính các chỉ số mà truy cập đó dùng:</p>}>
        <p>So each access gets its own test, written on the indices that access actually uses:</p>
      </Translated>
      <Translated
        vi={
          <ul>
            <li>
              lần nạp từ M: <code>Row &lt; Width &amp;&amp; ph*TILE_WIDTH + tx &lt; Width</code>
            </li>
            <li>
              lần nạp từ N: <code>ph*TILE_WIDTH + ty &lt; Width &amp;&amp; Col &lt; Width</code>
            </li>
            <li>
              lần ghi vào P: <code>Row &lt; Width &amp;&amp; Col &lt; Width</code>
            </li>
          </ul>
        }
      >
        <ul>
          <li>
            the load from M: <code>Row &lt; Width &amp;&amp; ph*TILE_WIDTH + tx &lt; Width</code>
          </li>
          <li>
            the load from N: <code>ph*TILE_WIDTH + ty &lt; Width &amp;&amp; Col &lt; Width</code>
          </li>
          <li>
            the store to P: <code>Row &lt; Width &amp;&amp; Col &lt; Width</code>
          </li>
        </ul>
      </Translated>
      <Translated
        vi={
          <p>
            Ba truy cập, ba điều kiện khác nhau. Một lần nạp bị bỏ qua vẫn phải để lại thứ gì đó hợp lý trong vị trí của nó, vì
            các luồng khác sắp nhân với nó, nên kernel ghi 0.0 vào đó: số 0 không thêm gì vào một tổng các tích. Và vòng lặp
            pha phải làm tròn lên, điều mà sách viết là <code>ceil(Width/(float)TILE_WIDTH)</code>.
          </p>
        }
      >
        <p>
          Three accesses, three different conditions. A skipped load must still leave something sensible in its slot,
          because other threads are about to multiply by it, so the kernel writes 0.0 there: a zero adds nothing to a sum
          of products. And the phase loop has to round up, which the book writes as{' '}
          <code>ceil(Width/(float)TILE_WIDTH)</code>.
        </p>
      </Translated>
      <Source section="5.5" pages="114–115" figs={tr('Fig. 5.13', 'Hình 5.13')} />

      <h2>
        <Translated vi="Làm hỏng, rồi sửa">Break it, then fix it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trình khám phá chạy kernel phân ô trên các ma trận nhỏ, với mỗi phép kiểm tra trong ba phép và mỗi lần ghi số 0
            trong hai lần là một công tắc riêng. Ban đầu mọi thứ đều tắt. Các ma trận được vẽ trên một lưới được độn lên cho
            tròn số ô, để bạn thấy những vị trí mà một khối với tới nhưng không thuộc ma trận. Chọn một khối và một pha để xem
            mỗi luồng của nó xin gì và nhận được gì. Kết quả của cả phép nhân được đối chiếu với phép tính trực tiếp, và dải ở
            dưới cùng lặp lại phép đối chiếu ở mọi độ rộng.
          </p>
        }
      >
        <p>
          The explorer runs the tiled kernel on small matrices, with each of the three tests and each of the two
          zero-writes as a separate switch. Everything starts off. The matrices are drawn on a grid padded up to whole
          tiles, so you can see the positions a block reaches for that are not part of the matrix. Pick a block and a
          phase to see what each of its threads asks for and what it gets. The result of the whole multiplication is
          checked against a direct computation, and the strip at the bottom repeats the check at every width.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Ở đây một lần đọc vượt ngoài mảng trả về NaN; thiết bị thật trả về bất cứ gì đang nằm đó, hoặc gây lỗi. Bộ nhớ chia
              sẻ chưa từng được ghi cũng giữ NaN, trường hợp tệ nhất. Một lần ghi vượt ngoài P được báo cáo nhưng không làm hỏng
              gì. Luôn chạy đủ khối và pha để phủ ma trận.
            </p>
          }
        >
          <p>
            A read beyond an array returns NaN here; a real device returns whatever is there, or faults. Shared memory
            never written also holds NaN, the worst case. A store beyond P is reported but damages nothing. Enough
            blocks and phases to cover the matrix are always run.
          </p>
        </Translated>
      </Simplification>
      <BoundaryExplorer />
      <Translated
        vi={
          <p>
            Khi mọi thứ tắt, ví dụ của sách hỏng theo cả hai cách cùng lúc: những lần nạp tràn sang nhầm phần tử và những lần
            nạp ra khỏi mảng. Chỉ bật ba phép kiểm tra thì không sửa được, vì một lần nạp bị bỏ qua để vị trí của nó giữ nguyên
            thứ đã có trước đó. Chỉ khi bật cả năm công tắc thì mọi độ rộng mới đúng và mọi truy cập mới sạch.
          </p>
        }
      >
        <p>
          With everything off, the book's example goes wrong in both ways at once: loads that wrap to a wrong element
          and loads that leave the array. Switching on the three tests alone does not repair it, because a skipped load
          leaves its slot holding whatever was there before. Only with all five switches on is every width correct and
          every access clean.
        </p>
      </Translated>
      <Ask q={questionsS55['ch05/5.5/q-early-return']} />

      <h2>
        <Translated vi="Một số 0 có đủ không?">Is one zero enough?</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hãy thử một điều mà sách không làm. Bật ba phép kiểm tra và chỉ một trong hai lần ghi số 0, chẳng hạn lần cho{' '}
            <code>Nds</code>, rồi bấm "Ví dụ của sách". P đúng. Trình khám phá vẫn phản đối rằng có những vị trí được dùng mà
            chưa có gì điền vào, và nó phản đối là đúng, nhưng đáp án thì đúng, và nó sẽ qua mọi bài kiểm thử so sánh P với kết
            quả tham chiếu.
          </p>
        }
      >
        <p>
          Try something the book does not. Switch on the three tests and only one of the two zero-writes, say the one
          for <code>Nds</code>, and press "The book's example". P is correct. The explorer still objects that slots were
          used which nothing had filled, and it is right to object, but the answer is right, and it would pass any test
          that compares P with a reference.
        </p>
      </Translated>
      <Beyond title={tr("Our finding, not the book's", 'Phát hiện của chúng tôi, không phải của sách')}>
        <Translated
          vi={
            <p>
              Sách ghi số 0 vào cả hai ô và không bàn đến việc bỏ một cái. Phần này đến từ trình mô phỏng và các lab của khóa học
              này.
            </p>
          }
        >
          <p>
            The book zeroes both tiles and does not discuss leaving one out. This part comes from the course's simulator
            and labs.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Đây là lý do nó chạy được. Một vị trí của <code>Mds</code> bị bỏ trống trong hai tình huống. Hoặc cột M của nó nằm
            ngoài phạm vi, và khi đó vị trí <code>Nds</code> mà nó được nhân cùng đến từ hàng của N có cùng chỉ số ngoài phạm vi
            đó, nên vị trí đó đã được ghi số 0 và tích là giá trị sót lại × 0. Hoặc hàng M của nó nằm ngoài phạm vi, và khi đó
            chỉ những luồng trên hàng đó của khối mới đọc nó, mà các luồng này không sở hữu phần tử nào của P và vứt bỏ tổng
            của chúng. Dù thế nào thì cũng không có gì sai đến được P, miễn là giá trị sót lại × 0 bằng 0.
          </p>
        }
      >
        <p>
          Here is why it works. A slot of <code>Mds</code> is left unfilled in two situations. Either its column of M is
          out of range, and then the <code>Nds</code> slot it gets multiplied with comes from the row of N with that same
          out-of-range index, so that slot was zeroed and the product is leftover × 0. Or its row of M is out of range,
          and then the only threads that read it are the ones in that row of the block, which own no element of P and
          throw their sums away. Either way nothing wrong reaches P, provided leftover × 0 is 0.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Điều đó đúng với mọi giá trị sót lại hữu hạn và sai với NaN và vô cực. NaN × 0 là NaN, và một NaN đầu độc cả một
            tổng. Giờ hãy bấm "Hẹp hơn một ô". Một ma trận 1×1 với ô 2×2 chỉ có một pha, nên vị trí bị bỏ trống chưa được ghi
            lần nào, và mô hình coi bộ nhớ chia sẻ chưa từng được ghi là NaN, trường hợp tệ nhất: P[0][0] ra NaN. Trong ví dụ của
            sách, các giá trị sót lại là những số bình thường từ pha 0, đó là lý do nó thoát. Điều tương tự xảy ra khi chỉ bật
            lần ghi số 0 còn lại, và ở W = 2 với T = 3.
          </p>
        }
      >
        <p>
          That holds for every finite leftover and fails for NaN and for infinity. NaN × 0 is NaN, and one NaN
          poisons a whole sum. Now press "Narrower than a tile". A 1×1 matrix with 2×2 tiles has a single phase, so the
          unfilled slot has not been written even once, and the model treats never-written shared memory as NaN, the
          worst case: P[0][0] comes out as NaN. In the book's example the leftovers were ordinary numbers from phase 0, which is why it got
          away with it. The same happens with the other zero-write alone, and at W = 2 with T = 3.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Trên thiết bị thật, bộ nhớ chia sẻ chưa được ghi giữ bất cứ gì đã có ở đó trước, thường là vô hại, nên một bài kiểm
            thử thật có thể qua. Rắc rối nằm ở đó: một kernel chỉ ghi số 0 vào một ô là sai, và kiểm thử hiếm khi cho bạn biết.
            Các kernel tham chiếu trong lab ghi số 0 vào cả hai.
          </p>
        }
      >
        <p>
          On a real device, shared memory that has not been written holds whatever was there before, which is often
          harmless, so a real test may pass. That is the trouble: a kernel that zeroes only one tile is wrong, and
          testing will rarely tell you. The reference kernels in the labs zero both.
        </p>
      </Translated>

      <h2>
        <Translated vi="Ma trận chữ nhật">Rectangular matrices</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel vuông dùng một con số, <code>Width</code>, cho ba thứ khác nhau. Cho A có <code>j</code> hàng và{' '}
            <code>k</code> cột, B có <code>k</code> hàng và <code>l</code> cột, để C = A × B có <code>j</code> hàng và{' '}
            <code>l</code> cột. Khi đó mỗi <code>Width</code> phải được thay bằng đúng thứ nó đại diện trong ba thứ: chiều cao
            của A và C, chiều rộng của A cũng là chiều cao của B, hoặc chiều rộng của B và C. Sách để việc thay thế này làm bài
            tập. Trang này cũng vậy.
          </p>
        }
      >
        <p>
          The square kernel uses one number, <code>Width</code>, for three different things. Let A have <code>j</code>{' '}
          rows and <code>k</code> columns and B have <code>k</code> rows and <code>l</code> columns, so that C = A × B has{' '}
          <code>j</code> rows and <code>l</code> columns. Then every <code>Width</code> must be replaced by whichever of
          the three it stood for: the height of A and C, the width of A which is also the height of B, or the width of B
          and C. The book leaves the replacement as an exercise. So does this page.
        </p>
      </Translated>
      <Source section="5.5" pages="114–115" />
      <RectangularExercise />
      <Translated
        vi={
          <p>
            Lab 5.3 yêu cầu bạn viết kernel này. Các bài kiểm thử của nó dùng kích thước chữ nhật với ô không trọn ở mọi chiều,
            xuống tới một phần tử duy nhất, hẹp hơn mọi ô. Đây là mảnh đầu tiên của dự án xuyên suốt khóa học: kernel đứng sau
            mọi lớp tuyến tính. Bạn sẽ tìm thấy nó trong <a href={href('ch05', 'labs')}>các bài thực hành</a>.
          </p>
        }
      >
        <p>
          Lab 5.3 has you write this kernel. Its tests use rectangular shapes with partial tiles in every direction,
          down to a single element, which is narrower than any tile. It is the first piece of the project that runs
          through the course: the kernel behind every linear layer. You will find it under{' '}
          <a href={href('ch05', 'labs')}>labs</a>.
        </p>
      </Translated>
      <Ask q={questionsS55['ch05/5.5/q-silent']} />
      <Ask q={questionsS55['ch05/5.5/q-count']} />
    </>
  );
}
