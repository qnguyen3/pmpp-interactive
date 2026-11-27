import { isValidElement } from 'react';
import { describe, expect, it } from 'vitest';
import { LANGUAGE_NAMES, LOCALES, getLocale, getMessages, localized, tr } from './i18n';

type Tree = Record<string, unknown>;
const isTree = (v: unknown): v is Tree => typeof v === 'object' && v !== null && !isValidElement(v);

/** Every leaf path with what kind of value it holds: string, function or element. */
function leaves(tree: Tree, prefix = ''): Map<string, unknown> {
  const out = new Map<string, unknown>();
  for (const [k, v] of Object.entries(tree)) {
    const path = prefix + k;
    if (isTree(v)) for (const [p, x] of leaves(v, path + '.')) out.set(p, x);
    else out.set(path, v);
  }
  return out;
}
const kind = (v: unknown) => (isValidElement(v) ? 'element' : typeof v);

describe('dictionaries', () => {
  const en = leaves(getMessages('en'));
  const vi = leaves(getMessages('vi'));

  it('have the same keys, each holding the same kind of value', () => {
    expect([...vi.keys()].sort()).toEqual([...en.keys()].sort());
    for (const [k, v] of en) expect([k, kind(vi.get(k))]).toEqual([k, kind(v)]);
  });

  it('have no empty Vietnamese string', () => {
    const empty = [...vi].filter(([, v]) => typeof v === 'string' && !v.trim()).map(([k]) => k);
    expect(empty).toEqual([]);
  });
});

describe('without a browser', () => {
  it('reads English', () => {
    expect(typeof window).toBe('undefined');
    expect(getLocale()).toBe('en');
    expect(tr('Review', 'Ôn tập')).toBe('Review');
    expect(localized('Review', 'Ôn tập')).toBe('Review');
    expect(localized('Review')).toBe('Review');
  });
});

describe('locales', () => {
  it('are English and Tiếng Việt', () => {
    expect(LOCALES).toEqual(['en', 'vi']);
    expect(LANGUAGE_NAMES).toEqual({ en: 'English', vi: 'Tiếng Việt' });
  });
});
