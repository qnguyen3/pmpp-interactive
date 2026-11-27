// The book as a graph, for section 1.7: the "Reading path" explorer, the
// prediction about chapter 20, the edition lookup and the path questions.
//
// Where each thing comes from, and what the app says wherever this is shown:
//   - EDGES are the prerequisite arrows of the book's Fig. P.1 (p. xxiii) as
//     recorded in notes/ch01.md. They were read from an image of the figure.
//     One junction is ambiguous in the PDF: the line from chapter 9 to
//     chapters 17 and 21 passes directly under the chapter 14 box, so those two
//     arrows carry `uncertain`. The arrow from 9 to 10 is drawn dashed in the
//     figure; the notes do not record what the book means by that, so the
//     closure does not follow it unless asked to.
//   - The Preface presents the arrows as prerequisites, drawn to help
//     instructors skip or reorder chapters (p. xxv). Treating them as advice
//     that this course's order may overrule is our reading, not the book's.
//   - "Leads nowhere further" for chapter 14 holds only under our reading of
//     the ambiguous junction (the line starts at 9, not at 14).
//   - GOALS (which chapters a goal of this reader needs), COURSE_STAGES (the
//     order of PLAN.md section 9), COURSE_LINKS (prerequisites this course adds)
//     and the tiers are this course's, not the book's.
//   - FOURTH_TO_FIFTH pairs the chapters of the 4th edition (titles from the
//     publisher's page) with those of the 5th by title. The pairing is our
//     inference (notes/today/ch01-facts.md, A15 and its addendum).
//
// It imports only lib/locale.ts and uses no syntax that needs compiling away
// beyond types, so the browser checks can load this file directly.

import { tr } from '../lib/locale.ts';

export type Part = 'I' | 'II' | 'III';
export type Tier = 'Deep' | 'Standard' | 'Condensed';

export interface ChapterInfo {
  /** Chapter number in the 5th edition. */
  n: number;
  /** This course's chapter id, e.g. "ch07". */
  id: string;
  /** The title as this course's registry gives it. */
  title: string;
  /** A label short enough for a small box (ours). */
  short: string;
  /** The part of the book the chapter is in; chapters 1 and 24 are in none. */
  part: Part | null;
  /** Book pages. */
  pages: string;
  /** How deeply this course treats it (PLAN.md section 3). */
  tier: Tier;
}

const ch = (n: number, title: () => string, short: () => string, pages: string, tier: Tier): ChapterInfo => ({
  n,
  id: 'ch' + String(n).padStart(2, '0'),
  get title() {
    return title();
  },
  get short() {
    return short();
  },
  part: n >= 2 && n <= 6 ? 'I' : n >= 7 && n <= 15 ? 'II' : n >= 16 && n <= 23 ? 'III' : null,
  pages,
  tier,
});

export const CHAPTERS: ChapterInfo[] = [
  ch(1, () => tr('Introduction', 'Giới thiệu'), () => tr('Introduction', 'Giới thiệu'), '1–20', 'Condensed'),
  ch(2, () => tr('Heterogeneous data-parallel computing', 'Tính toán song song dữ liệu trên hệ thống không đồng nhất'), () => tr('Data-parallel computing', 'Song song dữ liệu'), '21–44', 'Deep'),
  ch(3, () => tr('Multidimensional grids and data', 'Lưới đa chiều và dữ liệu'), () => tr('Grids and data', 'Lưới và dữ liệu'), '45–66', 'Deep'),
  ch(4, () => tr('Compute architecture and scheduling', 'Kiến trúc tính toán và lập lịch'), () => tr('Architecture, scheduling', 'Kiến trúc, lập lịch'), '67–92', 'Deep'),
  ch(5, () => tr('Memory architecture and data locality', 'Kiến trúc bộ nhớ và tính cục bộ dữ liệu'), () => tr('Memory, data locality', 'Bộ nhớ, tính cục bộ'), '93–121', 'Deep'),
  ch(6, () => tr('Performance considerations', 'Các cân nhắc về hiệu năng'), () => tr('Performance checklist', 'Danh sách kiểm tra hiệu năng'), '123–155', 'Deep'),
  ch(7, () => tr('Convolution', 'Tích chập'), () => tr('Convolution', 'Tích chập'), '159–181', 'Standard'),
  ch(8, () => tr('Stencil computation', 'Tính toán stencil'), () => tr('Stencil', 'Stencil'), '183–199', 'Condensed'),
  ch(9, () => tr('Histogram', 'Histogram'), () => tr('Histogram', 'Histogram'), '201–220', 'Standard'),
  ch(10, () => tr('Reduction', 'Phép rút gọn (reduction)'), () => tr('Reduction', 'Phép rút gọn'), '221–250', 'Deep'),
  ch(11, () => tr('Scan', 'Scan (tổng tiền tố)'), () => tr('Scan', 'Scan'), '251–287', 'Deep'),
  ch(12, () => tr('Filter', 'Lọc (filter)'), () => tr('Filter', 'Lọc'), '289–302', 'Deep'),
  ch(13, () => tr('Merge', 'Trộn (merge)'), () => tr('Merge', 'Trộn'), '303–328', 'Condensed'),
  ch(14, () => tr('Sorting', 'Sắp xếp'), () => tr('Sorting', 'Sắp xếp'), '329–348', 'Standard'),
  ch(15, () => tr('Advanced optimizations for matrix multiplication', 'Tối ưu hóa nâng cao cho phép nhân ma trận'), () => tr('Advanced matmul', 'Nhân ma trận nâng cao'), '349–370', 'Deep'),
  ch(16, () => tr('Dynamic programming and wavefront parallelism', 'Quy hoạch động và song song theo wavefront'), () => tr('Wavefront parallelism', 'Song song theo wavefront'), '373–400', 'Condensed'),
  ch(17, () => tr('Sparse matrix computation', 'Tính toán ma trận thưa'), () => tr('Sparse matrices', 'Ma trận thưa'), '401–423', 'Condensed'),
  ch(18, () => tr('Graph traversal', 'Duyệt đồ thị'), () => tr('Graph traversal', 'Duyệt đồ thị'), '425–451', 'Condensed'),
  ch(19, () => tr('Convolutional neural networks', 'Mạng nơ-ron tích chập'), () => tr('Convolutional networks', 'Mạng tích chập'), '453–475', 'Standard'),
  ch(20, () => tr('Large language models', 'Mô hình ngôn ngữ lớn'), () => tr('Language models', 'Mô hình ngôn ngữ'), '477–512', 'Deep'),
  ch(21, () => tr('Electrostatic potential map', 'Bản đồ thế tĩnh điện'), () => tr('Electrostatic map', 'Bản đồ tĩnh điện'), '513–528', 'Condensed'),
  ch(22, () => tr('Algorithm selection and problem formulation', 'Lựa chọn thuật toán và phát biểu bài toán'), () => tr('Algorithm selection', 'Chọn thuật toán'), '529–540', 'Standard'),
  ch(23, () => tr('Multi-GPU programming', 'Lập trình đa GPU'), () => tr('Multi-GPU', 'Đa GPU'), '541–576', 'Condensed'),
  ch(24, () => tr('Conclusion and future outlook', 'Kết luận và triển vọng'), () => tr('Conclusion', 'Kết luận'), '577–580', 'Condensed'),
];

export const chapter = (n: number): ChapterInfo => {
  const c = CHAPTERS[n - 1];
  if (!c || c.n !== n) throw new Error('no chapter ' + n);
  return c;
};

// ---------------------------------------------------------------- the arrows
export interface Edge {
  from: number;
  to: number;
  /** Drawn dashed in Fig. P.1. */
  dashed?: boolean;
  /** Read from the ambiguous junction under the chapter 14 box. */
  uncertain?: boolean;
}

const e = (from: number, to: number, flag?: 'dashed' | 'uncertain'): Edge =>
  flag === 'dashed' ? { from, to, dashed: true } : flag === 'uncertain' ? { from, to, uncertain: true } : { from, to };

/** The arrows of Fig. P.1 exactly as recorded in notes/ch01.md. */
export const EDGES: Edge[] = [
  e(1, 2),
  e(2, 3),
  e(2, 4),
  e(3, 5),
  e(4, 5),
  e(5, 6),
  e(6, 7),
  e(6, 9),
  e(6, 10),
  e(6, 13),
  e(6, 15),
  e(9, 10, 'dashed'),
  e(7, 8),
  e(7, 19),
  e(9, 11),
  e(10, 11),
  e(11, 12),
  e(12, 14),
  e(13, 14),
  e(9, 17, 'uncertain'),
  e(9, 21, 'uncertain'),
  e(8, 16),
  e(8, 23),
  e(19, 20),
  e(17, 18),
  e(16, 22),
  e(18, 22),
  e(21, 22),
];

export interface ClosureOptions {
  /** Also follow the dashed arrow (9 to 10). Off by default. */
  dashed?: boolean;
}

const follows = (edge: Edge, o?: ClosureOptions) => !edge.dashed || !!o?.dashed;
const ascending = (a: number, b: number) => a - b;
const unique = (xs: number[]) => [...new Set(xs)].sort(ascending);

/** Chapters with an arrow straight into `n`. */
export function parents(n: number, o?: ClosureOptions): number[] {
  return unique(EDGES.filter((x) => x.to === n && follows(x, o)).map((x) => x.from));
}

/** Chapters `n` has an arrow straight into. */
export function successors(n: number, o?: ClosureOptions): number[] {
  return unique(EDGES.filter((x) => x.from === n && follows(x, o)).map((x) => x.to));
}

/**
 * Every chapter the arrows put before any of `targets`: the transitive closure
 * of their parents. Includes chapter 1; includes a target only when it is a
 * prerequisite of another target. Ascending.
 */
export function prerequisites(targets: number[], o?: ClosureOptions): number[] {
  const seen = new Set<number>();
  const todo = [...targets];
  while (todo.length) {
    for (const p of parents(todo.pop()!, o))
      if (!seen.has(p)) {
        seen.add(p);
        todo.push(p);
      }
  }
  return [...seen].sort(ascending);
}

/** `prerequisites` without chapter 1 (the reader of section 1.7 is in it) and without the targets themselves. */
export function chaptersBefore(targets: number[], o?: ClosureOptions): number[] {
  return prerequisites(targets, o).filter((n) => n !== 1 && !targets.includes(n));
}

/** Chapters that come before `targets` only if the dashed arrow is followed. */
export function optionalPrerequisites(targets: number[]): number[] {
  const firm = new Set(prerequisites(targets));
  return prerequisites(targets, { dashed: true }).filter((n) => !firm.has(n) && !targets.includes(n));
}

/** Every chapter that has any of `sources` among its prerequisites. Ascending. */
export function leadsTo(sources: number[], o?: ClosureOptions): number[] {
  const seen = new Set<number>();
  const todo = [...sources];
  while (todo.length) {
    for (const s of successors(todo.pop()!, o))
      if (!seen.has(s)) {
        seen.add(s);
        todo.push(s);
      }
  }
  return [...seen].sort(ascending);
}

/**
 * The targets and everything before them, in an order that never lists a
 * chapter ahead of one of its prerequisites. Among chapters that are ready at
 * the same time the lowest number goes first. Returns [] if the arrows
 * contain a cycle (they do not; the unit test checks).
 */
export function readingOrder(targets: number[], o?: ClosureOptions): number[] {
  const set = new Set<number>([...targets, ...prerequisites(targets, o)]);
  const waiting = new Map<number, number>();
  for (const n of set) waiting.set(n, parents(n, o).filter((p) => set.has(p)).length);
  const out: number[] = [];
  while (out.length < set.size) {
    const ready = [...waiting].filter(([, k]) => k === 0).map(([n]) => n);
    if (!ready.length) return [];
    const next = Math.min(...ready);
    out.push(next);
    waiting.delete(next);
    for (const s of successors(next, o)) if (waiting.has(s)) waiting.set(s, waiting.get(s)! - 1);
  }
  return out;
}

/** True when no chapter is, through the arrows, its own prerequisite. */
export function isAcyclic(): boolean {
  const all = CHAPTERS.map((c) => c.n);
  return readingOrder(all, { dashed: true }).length === all.length;
}

// ---------------------------------------------------------------- this course's additions
/** Pages this course has that are not boxes of Fig. P.1. */
export type ExtraId = 'appA' | 'appC' | 'x1' | 'x2' | 'x3';
export interface ExtraInfo {
  id: ExtraId;
  num: string;
  title: string;
  short: string;
  /** Book pages, or null when the material is not in the book at all. */
  pages: string | null;
  tier: Tier;
}
export const EXTRAS: ExtraInfo[] = [
  { id: 'appA', num: 'A', get title() { return tr('Numerical considerations', 'Các cân nhắc về số học'); }, get short() { return tr('Numerics', 'Số học'); }, pages: '583–600', tier: 'Standard' },
  { id: 'appC', num: 'C', get title() { return tr('CUDA memories, address spaces, and pointers', 'Bộ nhớ, không gian địa chỉ và con trỏ trong CUDA'); }, get short() { return tr('CUDA memory models', 'Mô hình bộ nhớ CUDA'); }, pages: '623–629', tier: 'Standard' },
  { id: 'x1', num: 'X1', get title() { return tr('Sampling kernels', 'Kernel lấy mẫu'); }, get short() { return tr('Sampling kernels', 'Kernel lấy mẫu'); }, pages: null, tier: 'Deep' },
  { id: 'x2', num: 'X2', get title() { return tr('Quantised kernels', 'Kernel lượng tử hóa'); }, get short() { return tr('Quantised kernels', 'Kernel lượng tử hóa'); }, pages: null, tier: 'Deep' },
  { id: 'x3', num: 'X3', get title() { return tr('Reading real backends', 'Đọc các backend thực tế'); }, get short() { return tr('Reading real backends', 'Đọc các backend thực tế'); }, pages: null, tier: 'Standard' },
];
export const extra = (id: ExtraId): ExtraInfo => EXTRAS.find((x) => x.id === id)!;

/** A chapter number or one of this course's extra pages. */
export type Item = number | ExtraId;

export type GoalId = 'matmul' | 'attention' | 'sampling' | 'quantised' | 'backends';
export interface Goal {
  id: GoalId;
  label: string;
  /** Book chapters the goal ends at. */
  chapters: number[];
  /** Appendices and course modules the goal also needs. */
  extras: ExtraId[];
}
/** Which chapters each of the reader's goals ends at. Ours, from PLAN.md sections 2 and 3. */
export const GOALS: Goal[] = [
  { id: 'matmul', get label() { return tr('Fast matmul', 'Nhân ma trận nhanh'); }, chapters: [15], extras: [] },
  { id: 'attention', get label() { return tr('Attention', 'Attention'); }, chapters: [20], extras: [] },
  { id: 'sampling', get label() { return tr('Sampling', 'Lấy mẫu'); }, chapters: [10, 11, 12, 14], extras: ['x1'] },
  { id: 'quantised', get label() { return tr('Quantised kernels', 'Kernel lượng tử hóa'); }, chapters: [15], extras: ['appA', 'x2'] },
  { id: 'backends', get label() { return tr('Reading real backends', 'Đọc các backend thực tế'); }, chapters: [], extras: ['x3'] },
];
export const goal = (id: GoalId): Goal => GOALS.find((g) => g.id === id)!;

export interface Stage {
  name: string;
  /** The items as PLAN.md section 9 lists them, in its order. */
  items: Item[];
  /** Items placed here by this model and not by PLAN.md section 9's list. */
  placedHere?: number[];
}
/**
 * PLAN.md section 9: the order in which this course takes the book, by what the
 * reader's goals need first. Two adjustments, both ours: section 9 is an order
 * of work that starts after the pilot (chapter 5) and leaves chapter 1 with the
 * condensed chapters at the end. As a reading order, chapter 1 comes first (the
 * reader is in it) and chapter 5 sits with the other foundations.
 */
export const COURSE_STAGES: Stage[] = [
  { get name() { return tr('Where you are', 'Bạn đang ở đây'); }, items: [1], placedHere: [1] },
  { get name() { return tr('Foundations', 'Nền tảng'); }, items: [2, 3, 4, 5, 6], placedHere: [5] },
  { get name() { return tr('Matrix multiplication', 'Phép nhân ma trận'); }, items: [15] },
  { get name() { return tr('Sampling', 'Lấy mẫu'); }, items: [10, 11, 12, 14, 9, 'x1'] },
  { get name() { return tr('Attention and batching', 'Attention và batching'); }, items: [20, 22] },
  { get name() { return tr('Quantisation', 'Lượng tử hóa'); }, items: ['appA', 'x2'] },
  { get name() { return tr('Convolution, CNNs and memory models', 'Tích chập, CNN và mô hình bộ nhớ'); }, items: [7, 19, 'appC'] },
  { get name() { return tr('Reading real backends', 'Đọc các backend thực tế'); }, items: ['x3'] },
  { get name() { return tr('Condensed chapters', 'Các chương rút gọn'); }, items: [8, 13, 16, 17, 18, 21, 23, 24] },
];

/**
 * The stages flattened into one reading order. Stages keep PLAN.md's order.
 * Inside a stage the chapters follow the book's arrows (lowest number first
 * among those that are ready), then the stage's appendices and modules. This
 * moves chapter 9 ahead of chapter 10, where PLAN.md lists it after 14.
 */
export function courseOrder(): Item[] {
  return COURSE_STAGES.flatMap((s) => {
    const chapters = s.items.filter((i): i is number => typeof i === 'number');
    const inStage = new Set(chapters);
    const ordered = readingOrder(chapters, { dashed: true }).filter((n) => inStage.has(n));
    return [...ordered, ...s.items.filter((i) => typeof i !== 'number')];
  });
}

/** 0-based position of an item in `courseOrder()`. */
export function coursePosition(item: Item): number {
  const at = courseOrder().indexOf(item);
  if (at < 0) throw new Error('not in the course order: ' + item);
  return at;
}

/** The name of the stage an item is in. */
export function stageOf(item: Item): string {
  return COURSE_STAGES.find((s) => s.items.includes(item))!.name;
}

/** Arrows of the chart whose tail this course reads after their head. */
export function courseDepartures(): Edge[] {
  const order = courseOrder();
  return EDGES.filter((x) => order.indexOf(x.from) > order.indexOf(x.to));
}

export interface CourseLink {
  from: Item;
  to: Item;
  /** Where the link comes from. */
  source: string;
}
/** Prerequisites this course adds to the book's arrows. Ours. */
export const COURSE_LINKS: CourseLink[] = [
  { from: 10, to: 20, get source() { return tr('notes/ch20.md, pp. 488 and 497–501; notes/ch10.md, §10.7', 'notes/ch20.md, pp. 488 và 497–501; notes/ch10.md, §10.7'); } },
  { from: 10, to: 'x1', get source() { return tr('PLAN.md section 3', 'PLAN.md mục 3'); } },
  { from: 11, to: 'x1', get source() { return tr('PLAN.md section 3', 'PLAN.md mục 3'); } },
  { from: 12, to: 'x1', get source() { return tr('PLAN.md section 3', 'PLAN.md mục 3'); } },
  { from: 14, to: 'x1', get source() { return tr('PLAN.md section 3', 'PLAN.md mục 3'); } },
  { from: 'appA', to: 'x2', get source() { return tr('PLAN.md section 3', 'PLAN.md mục 3'); } },
  { from: 15, to: 'x2', get source() { return tr('PLAN.md section 3', 'PLAN.md mục 3'); } },
];

export interface CoursePath {
  /** What to read, selection included, in this course's order. */
  order: Item[];
  /** Chapters in `order` that only this course asks for, not the book's arrows. */
  added: number[];
  /** Chapters the book's arrows put before the selection and this course reads after it. */
  later: number[];
}

/**
 * The same question as `readingOrder`, answered in this course's order. The
 * chapters needed are the book's prerequisites of the selection plus whatever
 * COURSE_LINKS adds. Those this course places after the last selected item
 * are returned in `later`; the rest are sorted by course position.
 *
 * A selection that needs no chapter at all (module X3, which neither the book
 * nor PLAN.md gives prerequisites) is placed by position alone: `order` is then
 * everything this course puts before it.
 */
export function coursePath(targets: number[], extras: ExtraId[] = []): CoursePath {
  const order = courseOrder();
  const pos = (i: Item) => order.indexOf(i);
  const selected = new Set<Item>([...targets, ...extras]);
  // Follow this course's own links back from the selection.
  const wanted = new Set<Item>(selected);
  for (let grew = true; grew; ) {
    grew = false;
    for (const l of COURSE_LINKS)
      if (wanted.has(l.to) && !wanted.has(l.from)) {
        wanted.add(l.from);
        grew = true;
      }
  }
  const wantedChapters = [...wanted].filter((i): i is number => typeof i === 'number');
  if (!wantedChapters.length) {
    const last = Math.max(...[...selected].map(pos));
    return { order: order.slice(0, last + 1), added: [], later: [] };
  }
  const byBook = new Set<number>([...targets, ...prerequisites(targets)]);
  const needed = new Set<number>([...wantedChapters, ...prerequisites(wantedChapters)]);
  const last = Math.max(...[...selected].map(pos));
  const later = [...needed].filter((n) => !selected.has(n) && pos(n) > last).sort(ascending);
  const kept: Item[] = [...needed].filter((n) => !later.includes(n));
  for (const i of wanted) if (typeof i !== 'number') kept.push(i);
  return {
    order: kept.sort((a, b) => pos(a) - pos(b)),
    added: [...needed].filter((n) => !byBook.has(n) && !later.includes(n)).sort(ascending),
    later,
  };
}

// ---------------------------------------------------------------- editions
/** Where the material of a 4th-edition chapter is in the 5th edition. */
export type Fate =
  | { kind: 'same' }
  | { kind: 'moved'; to: number }
  | { kind: 'removed'; note: string; foldedInto?: number; appendix?: string };

export interface FourthChapter {
  /** Chapter number in the 4th edition (2022). */
  n: number;
  /** Main title on the publisher's 4th-edition page (subtitles after a colon dropped). */
  title: string;
  fate: Fate;
}

const same: Fate = { kind: 'same' };
const moved = (to: number): Fate => ({ kind: 'moved', to });

export const FOURTH_TO_FIFTH: FourthChapter[] = [
  { n: 1, get title() { return tr('Introduction', 'Giới thiệu'); }, fate: same },
  { n: 2, get title() { return tr('Heterogeneous data parallel computing', 'Tính toán song song dữ liệu trên hệ thống không đồng nhất'); }, fate: same },
  { n: 3, get title() { return tr('Multidimensional grids and data', 'Lưới đa chiều và dữ liệu'); }, fate: same },
  { n: 4, get title() { return tr('Compute architecture and scheduling', 'Kiến trúc tính toán và lập lịch'); }, fate: same },
  { n: 5, get title() { return tr('Memory architecture and data locality', 'Kiến trúc bộ nhớ và tính cục bộ dữ liệu'); }, fate: same },
  { n: 6, get title() { return tr('Performance considerations', 'Các cân nhắc về hiệu năng'); }, fate: same },
  { n: 7, get title() { return tr('Convolution', 'Tích chập'); }, fate: same },
  { n: 8, get title() { return tr('Stencil', 'Stencil'); }, fate: same },
  { n: 9, get title() { return tr('Parallel histogram', 'Histogram song song'); }, fate: same },
  { n: 10, get title() { return tr('Reduction', 'Phép rút gọn'); }, fate: same },
  { n: 11, get title() { return tr('Prefix sum (scan)', 'Tổng tiền tố (scan)'); }, fate: same },
  { n: 12, get title() { return tr('Merge', 'Trộn (merge)'); }, fate: moved(13) },
  { n: 13, get title() { return tr('Sorting', 'Sắp xếp'); }, fate: moved(14) },
  { n: 14, get title() { return tr('Sparse matrix computation', 'Tính toán ma trận thưa'); }, fate: moved(17) },
  { n: 15, get title() { return tr('Graph traversal', 'Duyệt đồ thị'); }, fate: moved(18) },
  { n: 16, get title() { return tr('Deep learning', 'Học sâu'); }, fate: moved(19) },
  {
    n: 17,
    get title() {
      return tr('Iterative magnetic resonance imaging reconstruction', 'Tái tạo ảnh cộng hưởng từ theo phương pháp lặp');
    },
    fate: { kind: 'removed', get note() { return tr('Its scatter-versus-gather lesson duplicated the electrostatic potential map chapter, which took over some of its fundamentals.', 'Bài học scatter so với gather của nó trùng với chương bản đồ thế tĩnh điện, và chương đó đã tiếp nhận một phần kiến thức nền tảng của nó.'); }, foldedInto: 21 },
  },
  { n: 18, get title() { return tr('Electrostatic potential map', 'Bản đồ thế tĩnh điện'); }, fate: moved(21) },
  { n: 19, get title() { return tr('Parallel programming and computational thinking', 'Lập trình song song và tư duy tính toán'); }, fate: moved(22) },
  { n: 20, get title() { return tr('Programming a heterogeneous computing cluster', 'Lập trình một cụm máy tính không đồng nhất'); }, fate: moved(23) },
  {
    n: 21,
    get title() {
      return tr('CUDA dynamic parallelism', 'Song song động trong CUDA');
    },
    fate: { kind: 'removed', get note() { return tr('Dropped because the feature saw little adoption. The Preface points to the CUDA programming guide and to older editions.', 'Bị bỏ vì tính năng này ít được dùng. Lời nói đầu chỉ đến CUDA programming guide và các ấn bản cũ hơn.'); } },
  },
  {
    n: 22,
    get title() {
      return tr('Advanced practices and future evolution', 'Các kỹ thuật nâng cao và hướng phát triển tương lai');
    },
    fate: { kind: 'removed', get note() { return tr('Cooperative groups and thread block clusters are now used in the regular chapters. The history of the CUDA memory model moved to an appendix.', 'Cooperative groups và cụm khối luồng (thread block cluster) giờ được dùng trong các chương chính. Lịch sử của mô hình bộ nhớ CUDA đã chuyển sang một phụ lục.'); }, appendix: 'C' },
  },
  { n: 23, get title() { return tr('Conclusion and outlook', 'Kết luận và triển vọng'); }, fate: moved(24) },
];

/** 5th-edition chapters with no counterpart in the 4th. */
export const NEW_IN_FIFTH: number[] = [12, 15, 16, 20];

/** The 5th-edition chapter number for a 4th-edition one, or null when the chapter was removed. */
export function fifthFor(fourth: number): number | null {
  const c = FOURTH_TO_FIFTH.find((x) => x.n === fourth);
  if (!c) throw new Error('the 4th edition has chapters 1 to 23, not ' + fourth);
  return c.fate.kind === 'same' ? c.n : c.fate.kind === 'moved' ? c.fate.to : null;
}

/** The first 4th-edition chapter number that means something else in the 5th. */
export function firstRenumbered(): number {
  return FOURTH_TO_FIFTH.find((c) => c.fate.kind !== 'same')!.n;
}
