import { useEffect, useState } from 'react';
import { href, useRoute } from './lib/router';
import { useProgress } from './lib/progress';
import { dueCount } from './lib/review';
import { useChapter } from './chapters/registry';
import type { Chapter, Section, SectionModule } from './chapters/types';
import { Home } from './pages/Home';
import { Review } from './pages/Review';
import { LANGUAGE_NAMES, LOCALES, localized, useI18n } from './lib/i18n';

function useTheme(): [string, () => void] {
  const [theme, setTheme] = useState<string>(() => {
    try {
      return localStorage.getItem('theme') ?? '';
    } catch {
      return '';
    }
  });
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme;
    else delete document.documentElement.dataset.theme;
    try {
      if (theme) localStorage.setItem('theme', theme);
      else localStorage.removeItem('theme');
    } catch {
      /* private window: the choice just does not persist */
    }
  }, [theme]);
  const dark = theme ? theme === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  return [dark ? 'dark' : 'light', () => setTheme(dark ? 'light' : 'dark')];
}

function Rail({ chapter, section }: { chapter?: Chapter | null; section?: string }) {
  const { progress, saveError } = useProgress();
  const [theme, toggle] = useTheme();
  const { t, locale, setLocale } = useI18n();
  const due = dueCount(progress.review);
  return (
    <aside className="rail">
      <a className="book" href={href()}>
        Programming Massively Parallel Processors
        <small>{t.rail.subtitle}</small>
      </a>
      {chapter ? (
        <>
          <h4>
            {t.common.chapterWord(chapter.num)}
            {chapter.num}: {localized(chapter.title, chapter.vi?.title)}
          </h4>
          <nav aria-label={t.rail.sections}>
            {chapter.sections.map((s) => (
              <a key={s.id} href={href(chapter.id, s.id)} aria-current={s.id === section ? 'page' : undefined}>
                <span className="num">{s.num}</span>
                <span>{localized(s.title, s.vi?.title)}</span>
                <span className="tick" aria-label={progress.sections[`${chapter.id}/${s.id}`]?.done ? t.rail.done : undefined}>
                  {progress.sections[`${chapter.id}/${s.id}`]?.done ? '✓' : ''}
                </span>
              </a>
            ))}
          </nav>
        </>
      ) : null}
      <h4>{t.rail.study}</h4>
      <nav aria-label={t.rail.study}>
        <a href={href()} aria-current={!chapter && section === 'home' ? 'page' : undefined}>
          <span className="num" />
          <span>{t.rail.allChapters}</span>
          <span />
        </a>
        <a href={href('review')} aria-current={section === 'review' ? 'page' : undefined}>
          <span className="num" />
          <span>{t.rail.review}</span>
          <span className="num">{due ? t.rail.due(due) : ''}</span>
        </a>
      </nav>
      <div className="foot">
        <button className="quiet" onClick={toggle}>
          {theme === 'dark' ? t.rail.lightTheme : t.rail.darkTheme}
        </button>
        <span className="seg" role="group" aria-label={t.lang.label} data-language-switch="">
          {LOCALES.map((l) => (
            <button key={l} lang={l} aria-pressed={l === locale} onClick={() => setLocale(l)}>
              {LANGUAGE_NAMES[l]}
            </button>
          ))}
        </span>
      </div>
      {saveError ? (
        <p className="t-fail" role="alert" style={{ marginTop: '1rem' }}>
          {saveError}
        </p>
      ) : null}
    </aside>
  );
}

function SectionPage({ chapter, section }: { chapter: Chapter; section: Section }) {
  const { progress, visit, setDone, loaded } = useProgress();
  const key = `${chapter.id}/${section.id}`;
  const [mod, setMod] = useState<SectionModule | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    section.load().then(
      (m) => live && setMod(m),
      (e) => live && setFailed(String(e?.message ?? e)),
    );
    return () => {
      live = false;
    };
  }, [section]);
  useEffect(() => {
    if (loaded) visit(key);
  }, [key, loaded, visit]);
  const i = chapter.sections.indexOf(section);
  const prev = chapter.sections[i - 1];
  const next = chapter.sections[i + 1];
  const done = !!progress.sections[key]?.done;
  const { t } = useI18n();
  return (
    // data-ready tells the browser checks that the section's own code has rendered.
    <article className="page" data-ready={mod ? '' : undefined}>
      <header className="section-head">
        <span className="num">{section.num}</span>
        <h1>{localized(section.title, section.vi?.title)}</h1>
        <p className="lede">{localized(section.lede, section.vi?.lede)}</p>
      </header>
      <div className="prose">
        {mod ? <mod.Component /> : failed ? <p className="t-fail">{t.section.failed}{failed}</p> : <p className="muted">{t.common.loading}</p>}
      </div>
      <footer className="row ui" style={{ marginTop: '3rem', justifyContent: 'space-between', maxWidth: 'var(--measure)' }}>
        <span>{prev ? <a href={href(chapter.id, prev.id)}>{t.section.back}{localized(prev.title, prev.vi?.title)}</a> : null}</span>
        <button className={done ? 'quiet' : 'primary'} onClick={() => setDone(key, !done)}>
          {done ? t.section.markedDone : t.section.markDone}
        </button>
        <span>{next ? <a href={href(chapter.id, next.id)}>{t.section.next}{localized(next.title, next.vi?.title)}</a> : null}</span>
      </footer>
    </article>
  );
}

export function App() {
  const route = useRoute();
  const isStudy = !route[0] || route[0] === 'review';
  const chapter = useChapter(isStudy ? undefined : route[0]);
  const section = chapter ? (chapter.sections.find((s) => s.id === route[1]) ?? chapter.sections[0]) : undefined;
  // Subscribing here re-renders the whole tree in place when the language changes, so text read
  // with tr() anywhere below updates without remounting (answers and interactive state stay).
  const { t } = useI18n();
  return (
    <div className="frame">
      <Rail chapter={chapter} section={section?.id ?? (route[0] === 'review' ? 'review' : 'home')} />
      <main className="main">
        {route[0] === 'review' ? (
          <Review />
        ) : isStudy || chapter === null ? (
          <Home />
        ) : chapter === undefined ? (
          <article className="page">
            <p className="muted">{t.common.loading}</p>
          </article>
        ) : section ? (
          <SectionPage key={chapter.id + '/' + section.id} chapter={chapter} section={section} />
        ) : (
          <article className="page" data-ready="">
            <header className="section-head">
              <span className="num">{chapter.num}</span>
              <h1>{localized(chapter.title, chapter.vi?.title)}</h1>
              <p className="lede">{t.section.notBuilt}</p>
            </header>
          </article>
        )}
      </main>
    </div>
  );
}
