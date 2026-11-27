// The reader's language and the plain-string helpers, with no React and no imports, so the
// simulators can use tr() and the Node browser checks (app/e2e) can still import a simulator file
// directly. English is the source and the default; outside a browser everything reads English.
// The React side (useLocale, Translated, the dictionaries) is in i18n.tsx.

export type Locale = 'en' | 'vi';
export const LOCALES: Locale[] = ['en', 'vi'];
export const LANGUAGE_NAMES: Record<Locale, string> = { en: 'English', vi: 'Tiếng Việt' };

const STORAGE_KEY = 'lang';
const listeners = new Set<() => void>();

function readStored(): Locale {
  try {
    return typeof window !== 'undefined' && window.localStorage.getItem(STORAGE_KEY) === 'vi' ? 'vi' : 'en';
  } catch {
    return 'en'; // storage blocked (private window, sandboxed frame): English
  }
}

// tr() runs for every translated string on every render, so the stored choice is read once and
// then kept here; setLocale and the storage event (another tab switching) keep it current.
let current: Locale | null = null;

export function getLocale(): Locale {
  if (current == null) current = readStored();
  return current;
}

function notify() {
  for (const l of listeners) l();
}

export function setLocale(locale: Locale): void {
  current = locale;
  try {
    window.localStorage.setItem(STORAGE_KEY, locale);
  } catch {
    /* private window: the choice just does not persist */
  }
  if (typeof document !== 'undefined') document.documentElement.lang = locale;
  notify();
}

function onStorage(e: StorageEvent) {
  if (e.key !== STORAGE_KEY && e.key !== null) return;
  const next = readStored();
  if (next === current) return;
  current = next;
  document.documentElement.lang = next;
  notify();
}

export function subscribe(listener: () => void): () => void {
  if (!listeners.size && typeof window !== 'undefined') window.addEventListener('storage', onStorage);
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
    if (!listeners.size && typeof window !== 'undefined') window.removeEventListener('storage', onStorage);
  };
}

/** The English or the Vietnamese, by the language at the moment of the call. Pure; no React needed. */
export function tr(english: string, vietnamese: string): string {
  return getLocale() === 'vi' ? vietnamese : english;
}

/** The Vietnamese when it is the language and a translation exists, else the English. */
export function localized(english: string, vietnamese?: string): string {
  return getLocale() === 'vi' && vietnamese ? vietnamese : english;
}
