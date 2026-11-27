import { useEffect, useState } from 'react';

/**
 * True while the viewport is at most `px` wide: a phone held upright. Chapter 10's
 * drawings use it to fold a row of lanes in two instead of scrolling it sideways.
 * Chapter 10's own helper; a shared version would belong in src/lib.
 */
export function useNarrow(px = 600): boolean {
  const query = `(max-width: ${px}px)`; // i18n-ignore: CSS media query
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setNarrow(m.matches);
    on();
    m.addEventListener('change', on);
    return () => m.removeEventListener('change', on);
  }, [query]);
  return narrow;
}
