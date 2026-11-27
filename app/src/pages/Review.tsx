import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../components/Ask';
import { useProgress } from '../lib/progress';
import { INTERVAL_DAYS, isDue } from '../lib/review';
import { loadAllQuestions } from '../chapters/registry';
import { useI18n } from '../lib/i18n';

// Questions you have met come back here on a widening schedule.
export function Review() {
  const { progress, loaded } = useProgress();
  const [questions, setQuestions] = useState<Record<string, Question>>();
  useEffect(() => {
    loadAllQuestions().then(setQuestions);
  }, []);
  // Freeze the queue once everything is loaded, so answering does not reshuffle it.
  const ready = loaded && !!questions;
  const queue = useMemo(() => {
    if (!ready) return [];
    return Object.keys(progress.review)
      .filter((id) => questions![id] && isDue(progress.review[id]))
      .sort((a, b) => (progress.review[a].due < progress.review[b].due ? -1 : 1));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);
  const [pos, setPos] = useState(0);
  const [answered, setAnswered] = useState(false);
  const scheduled = Object.keys(progress.review).length;
  const q = ready && queue[pos] ? questions![queue[pos]] : undefined;
  const { t } = useI18n();

  return (
    <article className="page" data-ready={ready ? '' : undefined}>
      <header className="section-head">
        <span className="num" />
        <h1>{t.review.title}</h1>
        <p className="lede">{t.review.lede(INTERVAL_DAYS.join(', '))}</p>
      </header>
      <div className="prose">
        {!ready ? (
          <p className="muted">{t.common.loading}</p>
        ) : q ? (
          <>
            <p className="status-line">{t.review.position(pos + 1, queue.length)}</p>
            <Ask key={q.id} q={q} review label={t.review.label} onResult={() => setAnswered(true)} />
            <p>
              <button className="primary" disabled={!answered} onClick={() => { setPos(pos + 1); setAnswered(false); }}>
                {pos + 1 < queue.length ? t.review.next : t.review.finish}
              </button>
            </p>
          </>
        ) : (
          <p>
            {queue.length
              ? t.review.lastOne
              : scheduled
                ? t.review.nothingDue(scheduled)
                : t.review.nothingScheduled}
          </p>
        )}
      </div>
    </article>
  );
}
