import { useState, type CSSProperties } from 'react';
import { Bench } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BACKENDS,
  BACKEND_NAME,
  OPS,
  PROJECTS,
  PROJECT_NAME,
  cellLink,
  op as opRow,
  whereLink,
  type Backend,
  type Cell,
  type OpId,
  type Project,
} from '../../sim/backendmap';
import { linesText, shortCommit } from '../../sim/x3sources';

// The op map: for one op and one project, where the op enters each backend, the host function,
// the kernel's file and function with a permalink at the pinned commit, the facts items, and a
// status tag. All of it is data in sim/backendmap.ts. Later sections embed it preselected:
// <OpMap op="softmax" project="mlx" />.

const CARD: CSSProperties = {
  border: '1px solid var(--rule)', // i18n-ignore: CSS value
  borderRadius: 4,
  background: 'var(--page)',
  padding: '0.7rem 0.8rem', // i18n-ignore: CSS value
  fontFamily: 'var(--sans)',
  fontSize: '0.84rem',
  lineHeight: 1.42,
  minWidth: 0,
};
const DT: CSSProperties = { color: 'var(--ink-2)', fontSize: '0.74rem', margin: '0.45rem 0 0.05rem', textTransform: 'lowercase' }; // i18n-ignore: CSS value
const DD: CSSProperties = { margin: 0, overflowWrap: 'anywhere' };
const TAG: CSSProperties = {
  display: 'inline-block',
  border: '1px solid var(--rule)', // i18n-ignore: CSS value
  borderRadius: 3,
  padding: '0 0.4em', // i18n-ignore: CSS value
  margin: '0 0.3rem 0.25rem 0', // i18n-ignore: CSS value
  fontSize: '0.74rem',
  whiteSpace: 'nowrap',
};

function Tags({ c }: { c: Cell }) {
  return (
    <div data-part="tags" data-tags={c.tags.join('|')}>
      {c.tags.map((t) => (
        <span key={t} style={{ ...TAG, fontStyle: t === 'not read in the facts pass' || t === 'no backend' ? 'italic' : 'normal', color: t === 'no backend' ? 'var(--ink-3)' : 'var(--ink)' }}>
          {tr(t, ({ 'own kernel': 'kernel riêng', 'same source as CUDA': 'cùng mã nguồn với CUDA', 'hipified at build time': 'được hipify lúc build', 'generated at build time': 'được sinh ra lúc build', 'generated at run time (mx.compile)': 'được sinh ra lúc chạy (mx.compile)', 'vendor library': 'thư viện của nhà cung cấp', 'third-party submodule': 'submodule của bên thứ ba', 'composite of other ops': 'tổ hợp từ các op khác', 'fused before dispatch': 'được gộp trước khi dispatch', 'found by name string': 'tìm theo chuỗi tên', 'not read in the facts pass': 'chưa đọc trong lượt kiểm tra dữ kiện', 'no backend': 'không có backend' } as Record<string, string>)[t] ?? t)}
        </span>
      ))}
    </div>
  );
}

function CellCard({ project, backend, c }: { project: Project; backend: Backend; c: Cell }) {
  const link = cellLink(project, c);
  const none = c.tags.includes('no backend');
  return (
    <section style={{ ...CARD, opacity: none ? 0.75 : 1 }} data-backend={backend} data-file={c.file ?? ''} aria-label={tr(`${BACKEND_NAME[backend]} cell`, `Ô ${BACKEND_NAME[backend]}`)}>
      <div className="row" style={{ justifyContent: 'space-between', gap: '0.3rem 0.8rem', alignItems: 'baseline', marginBottom: '0.3rem' }}>
        <b style={{ fontSize: '0.95rem' }}>{BACKEND_NAME[backend]}</b>
        <span className="muted" style={{ fontSize: '0.74rem' }}>{c.facts.map((f) => (f.includes(' ') ? f : 'x3 ' + f)).join(', ')}</span>
      </div>
      <Tags c={c} />
      {none ? (
        <p style={{ margin: '0.3rem 0 0' }} data-part="note">{c.note}</p>
      ) : (
        <dl style={{ margin: 0 }}>
          <dt style={DT}><Translated vi="Điểm vào">Entry point</Translated></dt>
          <dd style={DD} data-part="entry">{c.entry}</dd>
          {c.host ? (
            <>
              <dt style={DT}><Translated vi="Hàm host">Host function</Translated></dt>
              <dd style={DD} data-part="host"><code>{c.host}</code></dd>
            </>
          ) : null}
          <dt style={DT}><Translated vi="Kernel">Kernel</Translated></dt>
          <dd style={DD} data-part="kernel">
            {c.kernel ? <>{c.kernel}</> : <span className="muted"><Translated vi="không có">none</Translated></span>}
            {c.file ? (
              <>
                <br />
                {link ? (
                  <a href={link} target="_blank" rel="noreferrer" data-part="permalink">
                    <code>{c.file}</code>
                  </a>
                ) : (
                  <code>{c.file}</code>
                )}
                {c.lines != null ? <span className="muted">, {linesText(c.lines)}</span> : null}
                <span className="muted">
                  {c.readAt === 'master' ? tr(' (read at master in another chapter’s facts pass; no pinned lines)', ' (đọc ở nhánh master trong lượt kiểm tra dữ kiện của một chương khác; không có dòng được ghim)') : tr(` at ${shortCommit(project)}`, ` tại ${shortCommit(project)}`)}
                </span>
              </>
            ) : (
              <>
                <br />
                <span className="muted" data-part="nofile"><Translated vi="không có tệp để mở">no file to open</Translated></span>
              </>
            )}
          </dd>
          {c.also?.length ? (
            <>
              <dt style={DT}><Translated vi="Cũng có ở commit được ghim">Also at the pinned commit</Translated></dt>
              <dd style={DD}>
                {c.also.map((w, i) => (
                  <span key={w.label}>
                    {i ? '; ' : ''}
                    <a href={whereLink(project, w)} target="_blank" rel="noreferrer">{w.label}</a>
                  </span>
                ))}
              </dd>
            </>
          ) : null}
          {c.note ? <p style={{ margin: '0.5rem 0 0', color: 'var(--ink-2)' }} data-part="note">{c.note}</p> : null}
        </dl>
      )}
    </section>
  );
}

/** One plain sentence on what the three backends of this project have in common for this op. */
function hipLine(p: Project, cells: Record<Backend, Cell>): string {
  const hip = cells.hip;
  if (hip.tags.includes('no backend')) return tr('HIP: MLX has no AMD backend, so the column is empty.', 'HIP: MLX không có backend AMD, nên cột này trống.');
  if (hip.tags.includes('same source as CUDA')) return tr('HIP: the CUDA file itself, compiled as HIP.', 'HIP: chính tệp CUDA đó, biên dịch thành HIP.');
  if (hip.tags.includes('not read in the facts pass')) return tr('HIP: not read in the facts pass.', 'HIP: chưa được đọc trong lượt kiểm tra dữ kiện.');
  if (hip.tags.includes('vendor library')) return tr(`HIP: ${PROJECT_NAME[p]} calls a different library on AMD.`, `HIP: ${PROJECT_NAME[p]} gọi một thư viện khác trên AMD.`);
  if (hip.tags.includes('hipified at build time')) return tr('HIP: the CUDA source, translated by hipify at build time.', 'HIP: mã nguồn CUDA, được hipify dịch lúc build.');
  return tr('HIP: no operator of its own; ordinary tensor ops.', 'HIP: không có operator riêng; chỉ là các op tensor thông thường.');
}

export function OpMap({ op = 'rmsnorm', project = 'llama.cpp', title = tr('Op map', 'Bản đồ op') }: { op?: OpId; project?: Project; title?: string }) {
  const [o, setO] = useState<OpId>(op);
  const [p, setP] = useState<Project>(project);
  const row = opRow(o);
  const cells = row.cells[p];
  return (
    <Bench
      title={title}
      task={tr('Pick an op and a project. Find where it lands on CUDA, HIP and Metal.', 'Chọn một op và một dự án. Tìm xem nó rơi vào đâu trên CUDA, HIP và Metal.')}
      foot={
        <Translated
          vi={
            <>
              Mọi ô đều được đọc trong lượt kiểm tra dữ kiện ngày 2026-11-20 (notes/today/x3-facts.md và các mục được nêu). Liên kết
              trỏ tới commit được ghim, trừ chỗ một ô dựa vào mục của chương khác được đọc ở master. Không có gì ở đây được build
              hay chạy.
            </>
          }
        >
          <>
            Every cell was read in the facts pass of 2026-11-20 (notes/today/x3-facts.md and the items named). Links go to the
            pinned commit, except where a cell rests on another chapter’s item read at master. Nothing here was built or run.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)', fontFamily: 'var(--sans)', fontSize: '0.84rem' }}>
          <Translated vi="Op">Op</Translated>
          <span className="seg" role="group" aria-label={tr('Op', 'Op')} style={{ flexWrap: 'wrap' }}>
            {OPS.map((r) => (
              <button key={r.id} aria-pressed={o === r.id} onClick={() => setO(r.id)} data-op={r.id}>{r.label}</button>
            ))}
          </span>
        </span>
        <span className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)', fontFamily: 'var(--sans)', fontSize: '0.84rem' }}>
          <Translated vi="Dự án">Project</Translated>
          <span className="seg" role="group" aria-label={tr('Project', 'Dự án')}>
            {PROJECTS.map((x) => (
              <button key={x} aria-pressed={p === x} onClick={() => setP(x)} data-project={x}>{PROJECT_NAME[x]}</button>
            ))}
          </span>
        </span>
      </div>
      <p className="ui" style={{ margin: '0 0 0.6rem', fontSize: '0.86rem' }} data-readout="course">
        <b>{PROJECT_NAME[p]}, {row.label}.</b>{' '}
        <span className="muted"><Translated vi="Nơi bạn đã gặp kỹ thuật này:">Where you met the technique:</Translated></span>{' '}
        {row.course.map((l, i) => (
          <span key={l.num}>
            {i ? '; ' : ''}
            <a href={href(l.chapter, 'labs')}>
              <Translated vi={<>lab {l.num}</>}>lab {l.num}</Translated>
            </a>{' '}
            ({l.what}{l.planned ? tr(', in this module', ', trong học phần này') : ''})
          </span>
        ))}
        . <span className="muted"><Translated vi="Đọc trong mục">Read in section</Translated></span> <a href={href('x3', row.section)}>{row.section}</a>.
      </p>
      <div data-part="cells" data-op={o} data-project={p}
        style={{ display: 'grid', gap: '0.6rem', gridTemplateColumns: 'repeat(auto-fit, minmax(15.5rem, 1fr))' }}>
        {BACKENDS.map((b) => (
          <CellCard key={b} project={p} backend={b} c={cells[b]} />
        ))}
      </div>
      <p className="ui muted" style={{ margin: '0.6rem 0 0', fontSize: '0.84rem' }} data-readout="hip">{hipLine(p, cells)}</p>
    </Bench>
  );
}
