// The model behind chapter 24's steps map: which of the book's three steps each
// chapter belongs to (section 24.1, pp. 577–578), which chapters feed each of the
// reader's goals (PLAN.md sections 3 and 5), and the reader's own record per chapter.
//
// What it simplifies: a chapter either feeds a goal or does not. PLAN.md gives one
// reason per chapter and that is all the map shows; it says nothing about how much
// of a chapter a goal needs. Labs are counted from the run reports by the chapter
// prefix of their id, so a chapter whose labs have never been run shows none.
import { labStatus, type Reports } from '../lib/reports';
import type { PlanRow } from '../chapters/types';
import { tr } from '../lib/locale.ts';

/**
 * Backends that are a GPU. Both emulators are left out: `emu` and its race-detection
 * flavour `race` run on the CPU. (lib/reports' realBackends drops only `emu`.)
 */
export const gpuBackends = (r: Reports) => Object.keys(r).filter((b) => b !== 'emu' && b !== 'race');

export type Step = 1 | 2 | 3 | 'outside';

export const STEPS: { step: Step; name: string; chapters: string; what: string }[] = [
  {
    step: 1,
    get name() { return tr('Step one', 'Bước một'); },
    get chapters() { return tr('chapters 2 to 6', 'chương 2 đến 6'); },
    get what() { return tr('Kernels, thread-to-data mapping, the execution and memory model, the optimisation checklist.', 'Kernel, ánh xạ luồng tới dữ liệu, mô hình thực thi và mô hình bộ nhớ, danh sách kiểm tra tối ưu hóa.'); },
  },
  {
    step: 2,
    get name() { return tr('Step two', 'Bước hai'); },
    get chapters() { return tr('chapters 7 to 15', 'chương 7 đến 15'); },
    get what() { return tr('Parallel patterns, each one a vehicle for the step-one optimisations.', 'Các mẫu song song, mỗi mẫu là một phương tiện để áp dụng các tối ưu hóa của bước một.'); },
  },
  {
    step: 3,
    get name() { return tr('Step three', 'Bước ba'); },
    get chapters() { return tr('chapters 16 to 23', 'chương 16 đến 23'); },
    get what() { return tr('Advanced patterns and applications: alternative decompositions and their data structures.', 'Các mẫu và ứng dụng nâng cao: những cách phân rã khác và cấu trúc dữ liệu của chúng.'); },
  },
  {
    step: 'outside',
    get name() { return tr("Outside the book's steps", 'Ngoài các bước của sách'); },
    get chapters() { return tr('chapters 1 and 24, the appendices, our modules', 'chương 1 và 24, các phụ lục, các module của khóa học này'); },
    get what() { return tr('Framing, reference material, and the three modules this course adds.', 'Phần định hướng, tài liệu tham khảo và ba module mà khóa học này bổ sung.'); },
  },
];

/** The step the book assigns a chapter to on pp. 577–578; 'outside' for everything it does not place. */
export function stepOf(chapterId: string): Step {
  const m = /^ch(\d\d)$/.exec(chapterId);
  if (!m) return 'outside';
  const n = Number(m[1]);
  if (n >= 2 && n <= 6) return 1;
  if (n >= 7 && n <= 15) return 2;
  if (n >= 16 && n <= 23) return 3;
  return 'outside';
}

export type GoalId = 'matmul' | 'attention' | 'sampling' | 'quantised' | 'backends' | 'reading';
export interface Goal {
  id: GoalId;
  label: string;
  /** Chapter ids that feed it, from PLAN.md sections 3 and 5. */
  chapters: string[];
  /** True where every lab contributes, whatever its chapter. */
  everyLab?: boolean;
}
export const GOALS: Goal[] = [
  { id: 'matmul', get label() { return tr('matmul', 'matmul'); }, chapters: ['ch03', 'ch04', 'ch05', 'ch06', 'ch15', 'ch19'] },
  { id: 'attention', get label() { return tr('attention', 'attention'); }, chapters: ['ch10', 'ch20', 'ch22'] },
  { id: 'sampling', get label() { return tr('sampling', 'lấy mẫu'); }, chapters: ['ch09', 'ch10', 'ch11', 'ch12', 'ch14', 'x1'] },
  { id: 'quantised', get label() { return tr('quantised ops', 'phép toán lượng tử hóa'); }, chapters: ['appA', 'ch15', 'x2'] },
  { id: 'backends', get label() { return tr('other backends', 'các backend khác'); }, chapters: ['appC', 'x3'], everyLab: true },
  { id: 'reading', get label() { return tr('reading real code', 'đọc mã thật'); }, chapters: ['ch19', 'x3'] },
];
export const goal = (id: GoalId) => GOALS.find((g) => g.id === id)!;

/** The plan's rows that have a page in the app, grouped by step, in plan order. */
export function lanes(plan: PlanRow[]): { step: Step; rows: PlanRow[] }[] {
  return STEPS.map((s) => ({ step: s.step, rows: plan.filter((r) => r.id && stepOf(r.id) === s.step) }));
}

/** The distinct steps a goal draws on, in order. */
export function stepsOf(id: GoalId): Step[] {
  const set = new Set(goal(id).chapters.map(stepOf));
  return STEPS.map((s) => s.step).filter((s) => set.has(s));
}

export interface ChapterRecord {
  /** Sections of this chapter the reader has marked done. */
  sectionsDone: number;
  /** Labs of this chapter that appear in any run report. */
  labs: number;
  /** Of those, labs whose edited work/ copy passes on at least one real backend. */
  labsPassing: number;
}

/**
 * The reader's record per chapter. Labs are matched by the "chNN/" prefix of the
 * lab id, never by name. Neither emulator counts as a backend here, and an
 * untouched starter is "not started", not a pass or a failure.
 */
export function rollup(
  plan: PlanRow[],
  sections: Record<string, { done?: boolean }>,
  reports: Reports,
): Record<string, ChapterRecord> {
  const out: Record<string, ChapterRecord> = {};
  for (const r of plan) {
    if (!r.id) continue;
    const prefix = r.id + '/';
    const sectionsDone = Object.entries(sections).filter(([k, v]) => k.startsWith(prefix) && v.done).length;
    const seen = new Set<string>();
    const passing = new Set<string>();
    for (const run of Object.values(reports))
      for (const x of run.results) if (x.lab.startsWith(prefix)) seen.add(x.lab);
    for (const backend of gpuBackends(reports))
      for (const lab of seen) if (labStatus(reports[backend], lab, 'work') === 'pass') passing.add(lab);
    out[r.id] = { sectionsDone, labs: seen.size, labsPassing: passing.size };
  }
  return out;
}
