import type { ReactNode } from 'react';
import { Translated, tr } from '../../lib/i18n';
import { attribution, linesText, permalink, shortCommit, type Excerpt as ExcerptData } from '../../sim/x3sources';

// A short quotation from llama.cpp, MLX or PyTorch, as module X3 shows every one (x3-facts A4):
// at most about a dozen lines, verbatim from the facts pass, with the file, the lines, a
// permalink at the pinned commit, and the project's licence and copyright line beside it.
// A line that is exactly "..." marks lines left out; it is drawn as a gap, not as code.
// Each section keeps its excerpts as data and a unit test checks them against the facts file.
//
// Line numbers are printed only where they are known: a single-range excerpt whose length
// matches its range. Otherwise the gutter is left blank rather than guessed.

export function Excerpt({ x, caption, highlight }: { x: ExcerptData; caption?: ReactNode; highlight?: number[] }) {
  const lines = x.code.replace(/\n$/, '').split('\n');
  const span = x.lines == null ? null : typeof x.lines === 'number' ? [x.lines, x.lines] : x.lines;
  const numbered = !!span && !lines.some((l) => l.trim() === '...') && span[1] - span[0] + 1 === lines.length;
  const hl = new Set(highlight ?? []);
  const link = permalink(x.project, x.path, x.lines);
  const file = x.path.split('/').pop();
  return (
    <figure className="code wide x3-excerpt" style={{ maxWidth: '58rem' }} data-excerpt={x.id}>
      <div className="tabs">
        <span style={{ padding: '0.45rem 0.9rem', color: 'var(--ink)', fontWeight: 560 }}>{file}</span>
        <span className="status" style={{ marginTop: 0 }}>
          {x.path}
          {x.lines != null ? `, ${linesText(x.lines)}` : ''}
        </span>
      </div>
      <pre>
        <code>
          {lines.map((ln, i) =>
            ln.trim() === '...' ? (
              <span key={i} className="ln" data-n="" style={{ color: 'var(--ink-3)' }}>
                {tr('    … lines left out …\n', '    … các dòng được lược bỏ …\n')}
              </span>
            ) : (
              <span key={i} className={'ln' + (hl.has(i + 1) ? ' hl' : '')} data-n={numbered ? String(span![0] + i) : ''}>
                {ln}
                {'\n'}
              </span>
            ),
          )}
        </code>
      </pre>
      <figcaption className="caption">
        {caption ? <>{caption} </> : null}
        <span className="muted">
          <a href={link} target="_blank" rel="noreferrer" data-permalink>
            <Translated vi={<>{x.path.split('/').slice(-2).join('/')} tại {shortCommit(x.project)}</>}>
              {x.path.split('/').slice(-2).join('/')} at {shortCommit(x.project)}
            </Translated>
          </a>
          <Translated
            vi={
              <>
                . {attribution(x.project)}. Trích đúng như đã đọc trong lượt kiểm tra dữ kiện ({x.factsItem}); khoảng trắng có
                thể khác với tệp gốc.
              </>
            }
          >
            . {attribution(x.project)}. Quoted as read in the facts pass ({x.factsItem}); spacing may differ from the file.
          </Translated>
        </span>
      </figcaption>
    </figure>
  );
}
