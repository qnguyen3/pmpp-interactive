# Localization (English and Tiếng Việt)

English is the source of the course and the default. Vietnamese sits beside it, in the same
files, so a change to an English passage shows its translation on the next line of the diff.

The reader picks the language with the **English / Tiếng Việt** switch at the foot of the rail.
The choice is kept in the browser (`localStorage.lang`) and sets `<html lang>`. Switching
re-renders the open section in place: the route, answers, predictions, progress and the state
of the interactives stay as they were. Nothing in `progress/progress.json` depends on the
language.

## Where things live

| What | Where | How |
|---|---|---|
| Shared UI strings (rail, home, review, questions, labs pages, scene player, margin notes) | `app/src/lib/i18n/en.ts`, `vi.ts` | `const { t } = useI18n()`; same keys in both, checked by `i18n.test.ts` |
| Lesson prose, questions, labels in interactives | the section file itself | `<Translated vi={...}>English</Translated>`, `tr('English', 'Tiếng Việt')` |
| Step text, labels and notes built by a simulator | `app/src/sim/<name>.ts` | `tr()` from `../lib/locale.ts` (no React, no imports, so the Node checks in `app/e2e` can still import the file); getters for module-level data |
| Chapter and section titles and ledes | `app/src/chapters/<ch>/index.tsx` | `vi: { title, lede }` beside the English fields |
| Course table on the home page | `app/src/chapters/registry.ts` (`PLAN`) | `vi: { title, why, milestone? }` |
| Question units and sources (plain strings) | the question record | `vi: { unit, source }` |
| Narration captions | `content/<ch>/narration/vi/<clip>.txt` | one file per English script |
| Lab instructions | `labs/<ch>/<lab>/README.vi.md` | beside `README.md` |
| Project README | `README.vi.md` | beside `README.md` |
| Terms | `docs/i18n/glossary.md` | one rendering per term |

The helpers are in `app/src/lib/i18n.tsx`; the plain-string part (`tr`, `localized`, the stored language) is in
`app/src/lib/locale.ts`, which `i18n.tsx` re-exports.

## Translating a section file

Keep every English line exactly as it is. Add the Vietnamese beside it.

**Blocks of prose.** Wrap the English element and give the translation as `vi`:

```tsx
<Translated
  vi={
    <p>
      Một luồng tính một phần tử của <code>P = M × N</code>.
    </p>
  }
>
  <p>
    One thread computes one element of <code>P = M × N</code>.
  </p>
</Translated>
```

The Vietnamese keeps the same markup: the same `<code>`, `<b>`, links (`href` unchanged), margin
note placement and components. Only the words change. A margin note (`<Source>`, `<Aside>`, ...)
stays right after the paragraph it belongs to; translate inside it, not around it.

**Short text inside markup** (a heading, a button, an option, an SVG `<text>`):

```tsx
<h2><Translated vi="Trước khi bắt đầu">Before you start</Translated></h2>
```

**Plain strings** (props typed `string`, `aria-label`, `title`, option text in `<select>`,
labels built in code, template strings): `tr(english, vietnamese)`, called while rendering.

```tsx
<Bench title={tr('Roofline', 'Roofline')} task={tr('Pick the hardware.', 'Chọn phần cứng.')}>
label: tr(`tiled, T=${tile}`, `phân ô, T=${tile}`)
```

`tr()` reads the language at the moment it runs. Call it in render code or in a function that
render code calls, never at module level and never to initialise `useState` or inside a
`useMemo` whose dependencies do not change with the language. A module-level constant that
holds reader text becomes a getter or a function:

```ts
const BOOK = {
  key: 'book',
  get label() {
    return tr("The book's example (Hopper H100)", 'Ví dụ của sách (Hopper H100)');
  },
};
```

If the file already has a local variable named `tr`, import the helper under another name:
`import { tr as trText } from '../../lib/i18n'`.

**Questions.** `prompt`, option `label` and `why`, and `explain` are React nodes: wrap them in
`<Translated>`. `unit` and `source` are plain strings: add `vi: { unit, source }` to the record
(a `source` that is only a citation can stay as it is). Never change an id, the option order,
which option is `correct`, `answer` or `tol`: stored progress points at them.

**Never translate:** code in `<code>`, `<pre>` and `<Code>` blocks, code strings, identifiers,
CSS, ids, data keys, file paths, URLs, `href`s, the verification vocabulary, numbers. Never
change simulation logic, a test, or a lab file to make a translation fit.

**Leaving English on purpose.** A line that must stay English (a proper name or a value the
audit mistakes for prose) gets an `i18n-ignore: <reason>` comment on the same line: `//` in
code, `{/* i18n-ignore: <reason> */}` inside JSX (put it directly after the text, with no space
before it, or the space renders). The audit counts these apart.

**Terms kept in English.** Where the glossary keeps a term (Roofline, Kernel, Batch, GEMM), an
identical translation is correct: `tr('Roofline', 'Roofline')`. The audit only rejects identical
translations longer than three words.

**Lab names.** A labs page's `defineLabs` entries take `vi: { name }` beside `name`.

**Simulator identities.** A sim string that is also an identity (a union member, a key, a value compared with
`===`) stays English in the sim; the page that shows it gives the label: `tr(x, VI_LABELS[x] ?? x)` with the
lookup inside the second argument. A memo that calls a sim function returning reader text needs
`const locale = useLocale()` in its dependencies; the audit treats that variable as not English content.

**Memoised values.** If an object holding reader text is built inside a `useMemo` that does not
depend on the language, give it getters (`get label() { return tr(...) }`) rather than removing
the memo: the text is then read when rendered, and the English tokens stay as they were.

## Narration

Each English script `content/<ch>/narration/<clip>.txt` has a Vietnamese script at
`content/<ch>/narration/vi/<clip>.txt`. Write it for the ear and for reading as a caption:
short sentences, natural spoken Vietnamese, no symbols read aloud, numbers as digits.
Keep `[short pause]` tags where the English has them. The English script, audio and
`manifest.json` are never touched; `tools/narration.mjs` ignores the `vi/` folder.

There is no Vietnamese audio yet. In Vietnamese, the scene player shows the Vietnamese script
as a caption and leaves the recorded English narration off, behind an explicit "âm thanh gốc
(tiếng Anh)" switch, so English speech is never presented as Vietnamese narration. Recording
Vietnamese audio would follow the narration rules in `CLAUDE.md` (the one voice, a model the
maintainer approves, files under `vi/`) and needs a manifest for that folder; it has not been
done.

## Labs

`README.vi.md` beside each lab's `README.md`, opening with a link back to the English. Translate
the prose; keep commands, code, file names, function names, test names and numbers as they are.
Lab code, starters, `work/`, solutions and tests are not translated: their comments are part of
recorded starter hashes (`labs/starter-history.json`).

## Checking

```sh
node tools/i18n-audit.mjs                  # coverage of every area, exit 1 on any gap
node tools/i18n-audit.mjs --chapter ch05   # one chapter
node tools/i18n-audit.mjs --details        # every finding
cd app && npx tsc --noEmit && npx vitest run
```

The audit parses each source file and counts every piece of English a reader can see. It fails
when a piece sits outside an English branch, when a translation is empty, identical to its
English or reads as English, when a narration script or lab README has no Vietnamese file, and
when a chapter file's English no longer matches the base ref token for token once the
translations are removed. That last check is what shows ids, answers, numbers and simulation
code were left alone.

Browser checks for the switch: `APP_URL=http://localhost:<port> node e2e/app-language.mjs`.

## Adding or changing a chapter

Write the English first, as before. Then add the Vietnamese in the same commit or a follow-up,
run the audit for that chapter, and keep `glossary.md` the one place where a term is decided.
When an English passage changes, change its `vi` beside it; the audit will not notice a stale
translation, so reviewers should read the pair.
