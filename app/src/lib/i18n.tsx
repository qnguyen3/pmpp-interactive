import { useSyncExternalStore, type ReactNode } from 'react';
import { en, type Messages } from './i18n/en';
import { vi } from './i18n/vi';
import { getLocale, LANGUAGE_NAMES, LOCALES, localized, setLocale, subscribe, tr, type Locale } from './locale.ts';

export { getLocale, LANGUAGE_NAMES, LOCALES, localized, setLocale, tr, type Locale };

// The reader's language. English is the source of the course and the default; Vietnamese sits
// beside it in the same files (see docs/i18n/README.md). The choice lives in localStorage.lang
// and in <html lang>. Nothing in progress/progress.json depends on it, so switching re-renders
// in place and never saves progress.

export type { Messages };
const MESSAGES: Record<Locale, Messages> = { en, vi };
const serverLocale = (): Locale => 'en';

export function useLocale(): Locale {
  return useSyncExternalStore(subscribe, getLocale, serverLocale);
}

/** English children, or `vi` when Vietnamese is chosen. Re-renders when the language changes. */
export function Translated({ vi: vietnamese, children }: { vi: ReactNode; children: ReactNode }): ReactNode {
  return useLocale() === 'vi' ? vietnamese : children;
}

export function getMessages(locale: Locale): Messages {
  return MESSAGES[locale];
}

export function useI18n(): { locale: Locale; t: Messages; setLocale: (l: Locale) => void; toggleLocale: () => void } {
  const locale = useLocale();
  return {
    locale,
    t: MESSAGES[locale],
    setLocale,
    toggleLocale: () => setLocale(locale === 'vi' ? 'en' : 'vi'),
  };
}
