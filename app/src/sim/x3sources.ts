// Module X3 reads three real codebases. Everything it says about them was read in one facts
// pass (notes/today/x3-facts.md, 2026-11-20) at the commits below, so every link on the
// module is a permalink to that commit, never to master: the excerpts and line numbers stay
// valid when master moves. This file is the one place those commits are written down.
//
// Items cited from other chapters' facts files (ch10 D1, ch20 D4, x1 D7, ...) were read at
// master on the same day. They are linked by their facts item and their master URL, never
// with invented line numbers at the pinned commit (see FactsRef in backendmap.ts).

import { tr } from '../lib/locale.ts';

export const FACTS_FILE = 'notes/today/x3-facts.md';
export const READ_ON = '2026-11-20';

export type ProjectId =
  | 'llama.cpp'
  | 'mlx'
  | 'pytorch'
  | 'mlx-lm'
  | 'transformers'
  | 'safetensors'
  | 'ggml'
  | 'llama2.c';

export interface PinnedProject {
  id: ProjectId;
  /** Name as the lessons write it. */
  name: string;
  /** owner/name on GitHub. */
  repo: string;
  branch: string;
  /** The full commit hash read in the facts pass. */
  commit: string;
  /** Committed date of that commit, from the facts header; null where the header gives none. */
  date: string | null;
  /** Licence as the facts header gives it. */
  licence: string;
  /** The project's copyright line, verbatim from the facts file; null where the facts quote none. */
  copyright: string | null;
}

/** The snapshot table of x3-facts.md, row for row. */
export const PINNED: PinnedProject[] = [
  {
    id: 'llama.cpp', name: 'llama.cpp', repo: 'ggml-org/llama.cpp', branch: 'master',
    commit: '919794500475862f6768c353255469dc661cb734', date: '2026-11-20',
    licence: 'MIT', copyright: 'Copyright (c) 2023-2026 The ggml authors', // i18n-ignore: copyright notice, verbatim
  },
  {
    id: 'mlx', name: 'MLX', repo: 'ml-explore/mlx', branch: 'main',
    commit: 'bd2d1fb43e9819164f7d04098ddbb4fdb95fe64a', date: '2026-11-20',
    licence: 'MIT', copyright: 'Copyright © 2023 Apple Inc.', // i18n-ignore: copyright notice, verbatim
  },
  {
    id: 'pytorch', name: 'PyTorch', repo: 'pytorch/pytorch', branch: 'main',
    commit: 'a6ec1fa444f247690030fe73106156341c95fcc3', date: '2026-11-20',
    licence: 'BSD-style three clause', copyright: null, // i18n-ignore: licence as the facts header gives it, verbatim
  },
  {
    id: 'mlx-lm', name: 'mlx-lm', repo: 'ml-explore/mlx-lm', branch: 'main', // i18n-ignore: project name
    commit: '2d4df8d94a80c0f9f8794e54fafee08728b04fc6', date: '2026-11-19',
    licence: 'MIT', copyright: null,
  },
  {
    id: 'transformers', name: 'transformers', repo: 'huggingface/transformers', branch: 'main', // i18n-ignore: project name
    commit: '4d9ef623e3b0a1fb6c12891a907e0e50d41f0af8', date: '2026-11-20',
    licence: 'Apache-2.0', copyright: null,
  },
  {
    id: 'safetensors', name: 'safetensors', repo: 'huggingface/safetensors', branch: 'main', // i18n-ignore: project name
    commit: '11319d360db5170565db914601fbf375762bf8eb', date: '2026-11-20',
    licence: 'Apache-2.0', copyright: null,
  },
  {
    id: 'ggml', name: 'ggml (GGUF spec)', repo: 'ggml-org/ggml', branch: 'master', // i18n-ignore: project name
    commit: 'd9fe7bff3d2bf4d077e25c654486d02c321734aa', date: null,
    licence: 'MIT', copyright: null,
  },
  {
    id: 'llama2.c', name: 'llama2.c', repo: 'karpathy/llama2.c', branch: 'master',
    commit: '350e04fe35433e6d2941dce5a1f53308f87058eb', date: '2024-05-29',
    licence: 'MIT', copyright: null,
  },
];

/** The three projects the tour reads. */
export const TOUR: ProjectId[] = ['llama.cpp', 'mlx', 'pytorch'];

export function pinned(id: ProjectId): PinnedProject {
  const p = PINNED.find((x) => x.id === id);
  if (!p) throw new Error('no pinned project ' + id);
  return p;
}

/** The commit as the lessons print it: the first seven characters. */
export const shortCommit = (id: ProjectId) => pinned(id).commit.slice(0, 7);

export type Lines = number | [number, number];

/** Text for a line range: "line 154", "lines 116 to 127". */
export function linesText(lines: Lines | null | undefined): string {
  if (lines == null) return '';
  if (typeof lines === 'number') return tr(`line ${lines}`, `dòng ${lines}`);
  return lines[0] === lines[1] ? tr(`line ${lines[0]}`, `dòng ${lines[0]}`) : tr(`lines ${lines[0]} to ${lines[1]}`, `dòng ${lines[0]} đến ${lines[1]}`);
}

/**
 * A permalink to a file (or a directory, with `tree`) at the commit the facts pass read.
 * Line anchors only where the caller has line numbers from x3-facts.md.
 */
export function permalink(project: ProjectId, path: string, lines?: Lines | null, tree = false): string {
  const p = pinned(project);
  const clean = path.replace(/^\/+/, '');
  const base = `https://github.com/${p.repo}/${tree ? 'tree' : 'blob'}/${p.commit}/${clean}`;
  if (lines == null) return base;
  const [a, b] = typeof lines === 'number' ? [lines, lines] : lines;
  return a === b ? `${base}#L${a}` : `${base}#L${a}-L${b}`;
}

/** The licence and copyright line printed beside an excerpt (A4). */
export function attribution(project: ProjectId): string {
  const p = pinned(project);
  if (project === 'pytorch')
    return tr(
      'PyTorch, BSD-style three-clause licence (GitHub reports NOASSERTION); notice in its LICENSE file',
      'PyTorch, giấy phép kiểu BSD ba điều khoản (GitHub báo NOASSERTION); thông báo nằm trong tệp LICENSE của nó',
    );
  return p.copyright ? `${p.name}, ${p.licence}, ${p.copyright}` : `${p.name}, ${p.licence}`;
}

// ---------------------------------------------------------------- excerpts

/**
 * A short quotation from a project, shown with its file, lines, permalink and licence.
 * `code` is copied from x3-facts.md. A line that is exactly "..." marks lines left out;
 * each run of lines between such markers must appear in the facts file (whitespace aside).
 */
export interface Excerpt {
  id: string;
  project: ProjectId;
  path: string;
  lines: Lines | null;
  code: string;
  /** The x3-facts item the text is quoted in, e.g. "B1". */
  factsItem: string;
  /** Language for the reader: 'cuda', 'metal', 'cpp', 'python', 'yaml'. */
  lang?: string;
}

/** Collapse every run of whitespace to one space, so a reflowed quote still matches. */
export const normalise = (s: string) => s.replace(/\s+/g, ' ').trim();

/** The runs of code between "..." lines. */
export function excerptFragments(code: string): string[] {
  const out: string[] = [];
  let cur: string[] = [];
  for (const line of code.split('\n')) {
    if (line.trim() === '...') {
      if (cur.length) out.push(cur.join('\n'));
      cur = [];
    } else cur.push(line);
  }
  if (cur.length) out.push(cur.join('\n'));
  return out.filter((f) => f.trim().length > 0);
}

/**
 * True when every fragment of the excerpt occurs in the facts text, comparing with
 * whitespace collapsed (the facts file removes line breaks inside inline quotes) and with
 * the facts file's backslash-escaped quotes read as plain quotes.
 */
export function excerptInFacts(code: string, factsText: string): boolean {
  const raw = normalise(factsText);
  const plain = normalise(factsText.replace(/\\"/g, '"'));
  return excerptFragments(code).every((f) => raw.includes(normalise(f)) || plain.includes(normalise(f)));
}

/** Lines of code in an excerpt, elision markers excluded: A4's "quote short" rule is about a dozen. */
export const excerptLength = (code: string) => code.split('\n').filter((l) => l.trim() !== '...').length;
