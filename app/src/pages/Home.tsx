import { PLAN, useAllChapters } from '../chapters/registry';
import { href } from '../lib/router';
import { useProgress } from '../lib/progress';
import { dueCount } from '../lib/review';
import { useReports } from '../lib/reports';
import { localized, tr, useI18n } from '../lib/i18n';

export function Home() {
  const { progress } = useProgress();
  const { reports } = useReports();
  const { t } = useI18n();
  const chapters = useAllChapters();
  const due = dueCount(progress.review);
  const backends = Object.entries(reports);
  const built = new Map((chapters ?? []).filter((c) => c.sections.length).map((c) => [c.id, c]));
  return (
    <article className="page" data-ready={chapters ? '' : undefined}>
      <header className="section-head">
        <span className="num" />
        <h1>{t.home.title}</h1>
        <p className="lede">{t.home.lede}</p>
      </header>
      <div className="prose">
        <h2 style={{ marginTop: 0 }}>{t.home.whereYouAre}</h2>
        <ul>
          {PLAN.filter((r) => r.id && built.has(r.id)).map((r) => {
            const ch = built.get(r.id!)!;
            const done = ch.sections.filter((s) => progress.sections[`${ch.id}/${s.id}`]?.done).length;
            return (
              <li key={ch.id}>
                <a href={href(ch.id, ch.sections[0].id)}>
                  {t.common.chapterWord(ch.num)}
                  {ch.num}: {localized(ch.title, ch.vi?.title)}
                </a>{' '}
                <span className="ui muted">{t.home.sectionsDone(done, ch.sections.length)}</span>
              </li>
            );
          })}
          <li>
            <a href={href('review')}>{t.home.review}</a>{' '}
            <span className="ui muted">{due ? t.home.dueToday(due) : t.home.nothingDue}</span>
          </li>
        </ul>

        <h2>{t.home.devices}</h2>
        {backends.length ? (
          <table className="data wide" style={{ maxWidth: '52rem' }}>
            <thead>
              <tr>
                <th>{t.home.backend}</th>
                <th>{t.home.device}</th>
                <th>{t.home.lastRun}</th>
              </tr>
            </thead>
            <tbody>
              {backends.map(([name, run]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{String(run.device.name ?? t.home.unknownDevice)}</td>
                  <td>{run.generated.replace('T', ' ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>{t.home.noResults}</p>
        )}
        <p className="status-line">{t.home.noAssumption}</p>

        <h2>{t.home.course}</h2>
        <p>{t.home.courseLede}</p>
        <table className="data wide">
          <thead>
            <tr>
              <th>{t.home.cols.num}</th>
              <th>{t.home.cols.title}</th>
              <th>{t.home.cols.pages}</th>
              <th>{t.home.cols.depth}</th>
              <th>{t.home.cols.why}</th>
              <th>{t.home.cols.step}</th>
              <th>{t.home.cols.status}</th>
            </tr>
          </thead>
          <tbody>
            {PLAN.map((r) => {
              const ch = r.id ? built.get(r.id) : undefined;
              const title = localized(r.title, r.vi?.title);
              return (
                <tr key={r.num}>
                  <td>{r.num}</td>
                  <td>{ch ? <a href={href(ch.id, ch.sections[0].id)}>{title}</a> : title}</td>
                  {/* Our own modules have no book pages; their row says so in words. */}
                  <td className="nowrap">{r.ours ? tr(r.pages, 'không có trong sách') : r.pages}</td>
                  <td>{t.home.tier[r.tier]}</td>
                  <td>{localized(r.why, r.vi?.why)}</td>
                  <td>{r.milestone ? localized(r.milestone, r.vi?.milestone) : ''}</td>
                  <td className="nowrap">{ch ? t.home.ready : r.tier === 'Skipped' ? t.home.notPlanned : t.home.planned}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </article>
  );
}
