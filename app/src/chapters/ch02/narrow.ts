import { useEffect, useState } from 'react';

/**
 * True on a screen too narrow for a scene's wide layout. The chapter's scenes
 * switch to a second geometry below this width, so that their text stays
 * legible instead of being scaled down with the whole drawing.
 */
export function useNarrow(query = '(max-width: 44rem)'): boolean { // i18n-ignore: CSS media query
  const [narrow, setNarrow] = useState(() => typeof window !== 'undefined' && !!window.matchMedia?.(query).matches);
  useEffect(() => {
    const m = window.matchMedia?.(query);
    if (!m) return;
    const update = () => setNarrow(m.matches);
    update();
    m.addEventListener('change', update);
    return () => m.removeEventListener('change', update);
  }, [query]);
  return narrow;
}
