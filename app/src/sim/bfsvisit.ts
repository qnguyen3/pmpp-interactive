// "Two threads, one neighbour": a statement-level model of two or three threads of one BFS
// level reaching the same unvisited neighbours, for the stepper and bug hunt on page 18.3.
//
// From the book (notes/ch18.md): without frontiers a duplicate label is harmless in practice
// because the write is idempotent (pp. 433-434); with frontiers a duplicate is appended twice
// and expanded twice at the next level, so check-and-label must be one compare-and-swap
// (pp. 440-441, Fig. 18.14). The expected value is an lvalue (p. 441).
// From the facts (notes/today/ch18-facts.md): a failed compare-and-swap loads the current
// value into the expected variable (A6, and the same on HIP and Metal); a weak
// compare-and-swap may fail spuriously (A6); Metal has only the weak form (C4); libhipcxx
// lowers the strong form to the weak built-in (B4). Whether AMD or Apple GPUs ever fail a
// weak compare-and-swap spuriously is not documented (B4, C4), so here a spurious failure
// happens only where the reader, or a test, injects one.
// Ours: the eight visit forms, the two scenarios, and the outcome names. Every form's code
// is the text of a region of labs/ch18/snippets/s183_variants.cu, which
// app/src/chapters/ch18/S183.snippets.test.ts pins line for line.
//
// What this model simplifies:
//  - A step is one source statement and is indivisible. The ones that matter are called
//    out separately: the load of level[neighbor], the store to it, the compare-and-swap,
//    and the append (a fetch_add on the counter and a store to the slot it returned; the
//    counter is atomic, so the two can be one step without hiding anything). Taking the
//    next neighbour, and declaring e beside it, is one step that touches no shared data.
//  - No memory-order effects: every load sees the latest store (sequential consistency).
//  - Spurious failures happen only where injected.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

export const UNVISITED = 4294967295;

export type FormId = 'plain' | 'plain-append' | 'cas' | 'hoisted' | 'weak' | 'weak-loop' | 'weak-reset' | 'weak-correct';
type Pc = 'next' | 'check' | 'store' | 'cas' | 'reset' | 'test' | 'append' | 'done';
export interface Form {
  id: FormId;
  label: string;
  /** The region of labs/ch18/snippets/s183_variants.cu that holds this form. */
  region: string;
  /** The region's lines, exactly as in the file. */
  code: string[];
  /** 0-based line of each step, within code. */
  line: Partial<Record<Pc, number>>;
  weak: boolean;
  appends: boolean;
}

import { tr } from '../lib/locale.ts';

// The forms. Each is the body of a device function that expands one vertex of the previous
// level: walk its out-edges and visit every neighbour.
const HEAD = [
  '__device__ void NAME(CSRGraph g, unsigned int vertex, unsigned int* level,', // i18n-ignore: CUDA source shown as code
  '        unsigned int* currFrontier, unsigned int* numCurrFrontier, unsigned int currLevel) {', // i18n-ignore: CUDA source shown as code
];
const head = (name: string) => HEAD.map((l) => l.replace('NAME', name));
const FOR = '    for (unsigned int edge = g.srcPtrs[vertex]; edge < g.srcPtrs[vertex + 1]; ++edge) {'; // i18n-ignore: CUDA source shown as code
const NB = '        unsigned int neighbor = g.dst[edge];'; // i18n-ignore: CUDA source shown as code
const E = '        unsigned int e = UNVISITED;';
const APPEND = 'append(currFrontier, numCurrFrontier, neighbor);'; // i18n-ignore: CUDA source shown as code

export const FORMS: Record<FormId, Form> = {
  plain: {
    id: 'plain',
    get label() { return tr('plain check, then label (Fig. 18.6 lines 09–11)', 'kiểm tra thường, rồi gán nhãn (Fig. 18.6 dòng 09–11)'); },
    region: 'visit_plain',
    code: [...head('visitPlain'), FOR, NB, '        if (level[neighbor] == UNVISITED) {', '            level[neighbor] = currLevel;', '        }', '    }', '}'],
    line: { next: 3, check: 4, store: 5, done: 2 },
    weak: false,
    appends: false,
  },
  'plain-append': {
    id: 'plain-append',
    get label() { return tr('plain check and label, with a frontier append', 'kiểm tra và gán nhãn thường, có nối thêm vào frontier'); },
    region: 'visit_plain_append',
    code: [...head('visitPlainAppend'), FOR, NB, '        if (level[neighbor] == UNVISITED) {', '            level[neighbor] = currLevel;', '            ' + APPEND, '        }', '    }', '}'],
    line: { next: 3, check: 4, store: 5, append: 6, done: 2 },
    weak: false,
    appends: true,
  },
  cas: {
    id: 'cas',
    get label() { return tr('compare-and-swap, e fresh for each neighbour (Fig. 18.14)', 'compare-and-swap, e đặt mới cho mỗi đỉnh kề (Fig. 18.14)'); },
    region: 'visit_cas',
    code: [...head('visitCAS'), FOR, NB, E, '        if (casStrong(&level[neighbor], e, currLevel)) {', '            ' + APPEND, '        }', '    }', '}'],
    line: { next: 3, cas: 5, append: 6, done: 2 },
    weak: false,
    appends: true,
  },
  hoisted: {
    id: 'hoisted',
    get label() { return tr('compare-and-swap, e declared once above the loop', 'compare-and-swap, e khai báo một lần phía trên vòng lặp'); },
    region: 'visit_hoisted',
    code: [...head('visitHoisted'), '    unsigned int e = UNVISITED;', FOR, NB, '        if (casStrong(&level[neighbor], e, currLevel)) {', '            ' + APPEND, '        }', '    }', '}'],
    line: { next: 4, cas: 5, append: 6, done: 3 },
    weak: false,
    appends: true,
  },
  weak: {
    id: 'weak',
    get label() { return tr('weak compare-and-swap, no loop', 'compare-and-swap dạng weak, không có vòng lặp'); },
    region: 'visit_weak',
    code: [...head('visitWeak'), FOR, NB, E, '        if (casWeak(&level[neighbor], e, currLevel)) {', '            ' + APPEND, '        }', '    }', '}'],
    line: { next: 3, cas: 5, append: 6, done: 2 },
    weak: true,
    appends: true,
  },
  'weak-loop': {
    id: 'weak-loop',
    get label() { return tr('weak, while (!cas(e)) {}', 'dạng weak, while (!cas(e)) {}'); },
    region: 'visit_weak_loop',
    code: [...head('visitWeakLoop'), FOR, NB, E, '        while (!casWeak(&level[neighbor], e, currLevel)) {}', '        ' + APPEND, '    }', '}'],
    line: { next: 3, cas: 5, append: 6, done: 2 },
    weak: true,
    appends: true,
  },
  'weak-reset': {
    id: 'weak-reset',
    get label() { return tr('weak, while (!cas(e)) e = UNVISITED;', 'dạng weak, while (!cas(e)) e = UNVISITED;'); },
    region: 'visit_weak_reset',
    code: [...head('visitWeakReset'), FOR, NB, E, '        while (!casWeak(&level[neighbor], e, currLevel)) e = UNVISITED;', '        ' + APPEND, '    }', '}'],
    line: { next: 3, cas: 5, reset: 5, append: 6, done: 2 },
    weak: true,
    appends: true,
  },
  'weak-correct': {
    id: 'weak-correct',
    get label() { return tr('weak, while (!cas(e) && e == UNVISITED) {}', 'dạng weak, while (!cas(e) && e == UNVISITED) {}'); },
    region: 'visit_weak_correct',
    code: [
      ...head('visitWeakCorrect'), FOR, NB, E, '        bool won;',
      '        while (!(won = casWeak(&level[neighbor], e, currLevel)) && e == UNVISITED) {}',
      '        if (won) ' + APPEND, '    }', '}',
    ],
    line: { next: 3, cas: 6, test: 6, append: 7, done: 2 },
    weak: true,
    appends: true,
  },
};
export const FORM_IDS = Object.keys(FORMS) as FormId[];

// ---------------------------------------------------------------- scenarios
export interface Scenario {
  id: 'book' | 'trap';
  label: string;
  /** Edges of the small graph, for drawing. */
  edges: [number, number][];
  /** level[] before this level's launch. */
  level0: number[];
  currLevel: number;
  /** The threads of the launch: the vertex each expands and its out-neighbours in CSR order. */
  threads: { vertex: number; nbs: number[] }[];
  /** What a correct level labels. */
  expectNew: number[];
}
const U = UNVISITED;
export const SCENARIOS: Record<Scenario['id'], Scenario> = {
  book: {
    id: 'book',
    get label() { return tr("the book's graph, root 0, level 3: threads on 3, 4 and 6 all point to 8", 'đồ thị của sách, gốc 0, mức 3: các luồng trên 3, 4 và 6 đều trỏ tới 8'); },
    edges: [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6], [2, 7], [3, 4], [3, 8], [4, 5], [4, 8], [5, 6], [6, 8], [7, 0], [7, 6]],
    level0: [0, 1, 1, 2, 2, 2, 2, 2, U],
    currLevel: 3,
    threads: [{ vertex: 3, nbs: [4, 8] }, { vertex: 4, nbs: [5, 8] }, { vertex: 6, nbs: [8] }],
    expectNew: [8],
  },
  trap: {
    id: 'trap',
    get label() { return tr('0 → {1, 2}, 1 → {3}, 2 → {3, 4}, level 2: threads on 1 and 2', '0 → {1, 2}, 1 → {3}, 2 → {3, 4}, mức 2: các luồng trên 1 và 2'); },
    edges: [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4]],
    level0: [0, 1, 1, U, U],
    currLevel: 2,
    threads: [{ vertex: 1, nbs: [3] }, { vertex: 2, nbs: [3, 4] }],
    expectNew: [3, 4],
  },
};

// ---------------------------------------------------------------- state and steps
export interface ThreadState {
  pc: Pc;
  /** Index of the current neighbour in the thread's list. */
  k: number;
  nb: number | null;
  e: number;
  won: boolean | null;
  steps: number;
}
export interface State {
  level: number[];
  /** Stores to level[v], counted per vertex. */
  writes: number[];
  frontier: number[];
  counter: number;
  threads: ThreadState[];
  spurious: number;
}
export interface Move {
  thread: number;
  /** A weak compare-and-swap that fails although the values are equal. */
  spurious?: boolean;
}
export interface Event {
  thread: number;
  /** 0-based line in the form's code. */
  line: number;
  text: string;
}

export function initial(sc: Scenario): State {
  return {
    level: sc.level0.slice(),
    writes: sc.level0.map(() => 0),
    frontier: [],
    counter: 0,
    threads: sc.threads.map(() => ({ pc: 'next', k: 0, nb: null, e: UNVISITED, won: null, steps: 0 })),
    spurious: 0,
  };
}
const show = (v: number) => (v === UNVISITED ? 'UNVISITED' : String(v));

/** May this move be made? A spurious failure needs a weak compare-and-swap that would succeed. */
export function canMove(f: Form, st: State, mv: Move, maxSpurious = Infinity): boolean {
  const t = st.threads[mv.thread];
  if (!t || t.pc === 'done') return false;
  if (!mv.spurious) return true;
  return f.weak && t.pc === 'cas' && st.level[t.nb!] === t.e && st.spurious < maxSpurious;
}
export const isDone = (st: State) => st.threads.every((t) => t.pc === 'done');

/** Advance one thread by one statement. */
export function step(f: Form, sc: Scenario, st: State, mv: Move): { state: State; event: Event } {
  const s: State = {
    ...st,
    level: st.level.slice(),
    writes: st.writes.slice(),
    frontier: st.frontier.slice(),
    threads: st.threads.map((t) => ({ ...t })),
  };
  const t = s.threads[mv.thread];
  const me = sc.threads[mv.thread];
  const who = tr(`Thread on ${me.vertex}`, `Luồng trên ${me.vertex}`);
  const L = sc.currLevel;
  const nextNb = () => {
    t.k++;
    t.pc = 'next';
  };
  let text = '';
  const pc = t.pc;
  switch (pc) {
    case 'next':
      if (t.k >= me.nbs.length) {
        t.pc = 'done';
        t.nb = null;
        text = tr(`${who} has no more out-edges and returns.`, `${who} không còn cạnh ra nào và trả về.`);
        break;
      }
      t.nb = me.nbs[t.k];
      t.won = null;
      if (f.id === 'hoisted') {
        text = tr(`${who} takes neighbour ${t.nb}. Its e was set once, above the loop, and now holds ${show(t.e)}.`, `${who} lấy đỉnh kề ${t.nb}. Biến e của nó chỉ được đặt một lần, phía trên vòng lặp, và giờ giữ ${show(t.e)}.`);
      } else if (f.id === 'plain' || f.id === 'plain-append') {
        text = tr(`${who} takes neighbour ${t.nb}.`, `${who} lấy đỉnh kề ${t.nb}.`);
      } else {
        t.e = UNVISITED;
        text = tr(`${who} takes neighbour ${t.nb} and sets e = UNVISITED.`, `${who} lấy đỉnh kề ${t.nb} và đặt e = UNVISITED.`);
      }
      t.pc = f.id === 'plain' || f.id === 'plain-append' ? 'check' : 'cas';
      break;
    case 'check': {
      const v = s.level[t.nb!];
      if (v === UNVISITED) {
        t.pc = 'store';
        text = tr(`${who} loads level[${t.nb}] and sees UNVISITED.`, `${who} load level[${t.nb}] và thấy UNVISITED.`);
      } else {
        text = tr(`${who} loads level[${t.nb}] = ${v}: already visited, nothing to do.`, `${who} load level[${t.nb}] = ${v}: đã được thăm, không có gì để làm.`);
        nextNb();
      }
      break;
    }
    case 'store':
      s.level[t.nb!] = L;
      s.writes[t.nb!]++;
      text = s.writes[t.nb!] > 1
        ? tr(`${who} stores level[${t.nb}] = ${L}: write number ${s.writes[t.nb!]} to that slot in this level.`, `${who} store level[${t.nb}] = ${L}: lần ghi thứ ${s.writes[t.nb!]} vào ô đó trong mức này.`)
        : tr(`${who} stores level[${t.nb}] = ${L}.`, `${who} store level[${t.nb}] = ${L}.`);
      if (f.id === 'plain-append') t.pc = 'append';
      else nextNb();
      break;
    case 'cas': {
      const cur = s.level[t.nb!];
      const equal = cur === t.e;
      const kind = f.weak ? tr('weak compare-and-swap', 'compare-and-swap dạng weak') : 'compare-and-swap';
      if (equal && !mv.spurious) {
        s.level[t.nb!] = L;
        s.writes[t.nb!]++;
        t.won = true;
        text = tr(`${who}: ${kind} on level[${t.nb}] finds ${show(cur)}, equal to e, and swaps in ${L}. It wins.`, `${who}: ${kind} trên level[${t.nb}] thấy ${show(cur)}, bằng e, và hoán đổi vào ${L}. Nó thắng.`);
        if (cur !== UNVISITED) text = tr(`${who}: ${kind} on level[${t.nb}] finds ${show(cur)}, equal to e = ${show(t.e)}, and swaps in ${L}. It "wins" a vertex someone else had already labelled.`, `${who}: ${kind} trên level[${t.nb}] thấy ${show(cur)}, bằng e = ${show(t.e)}, và hoán đổi vào ${L}. Nó "thắng" một đỉnh mà luồng khác đã gán nhãn rồi.`);
      } else if (equal) {
        s.spurious++;
        t.won = false;
        text = tr(`${who}: the weak compare-and-swap on level[${t.nb}] fails spuriously. The values were equal; e stays ${show(t.e)}.`, `${who}: compare-and-swap dạng weak trên level[${t.nb}] thất bại giả (spurious). Hai giá trị bằng nhau; e vẫn là ${show(t.e)}.`);
      } else {
        t.e = cur;
        t.won = false;
        text = tr(`${who}: ${kind} on level[${t.nb}] finds ${show(cur)}, not e. It fails and loads ${show(cur)} into e.`, `${who}: ${kind} trên level[${t.nb}] thấy ${show(cur)}, không phải e. Nó thất bại và nạp ${show(cur)} vào e.`);
      }
      if (t.won) t.pc = 'append';
      else if (f.id === 'weak-loop') t.pc = 'cas';
      else if (f.id === 'weak-reset') t.pc = 'reset';
      else if (f.id === 'weak-correct') t.pc = 'test';
      else nextNb();
      break;
    }
    case 'reset':
      t.e = UNVISITED;
      t.pc = 'cas';
      text = tr(`${who} resets e = UNVISITED and tries again.`, `${who} đặt lại e = UNVISITED và thử lại.`);
      break;
    case 'test':
      if (t.e === UNVISITED) {
        t.pc = 'cas';
        text = tr(`${who}: e is still UNVISITED, so the failure was spurious. It tries again.`, `${who}: e vẫn là UNVISITED, nên lần thất bại là giả. Nó thử lại.`);
      } else {
        text = tr(`${who}: e = ${show(t.e)}, so another thread labelled ${t.nb}. It stops and does not append.`, `${who}: e = ${show(t.e)}, nên một luồng khác đã gán nhãn ${t.nb}. Nó dừng và không nối thêm.`);
        nextNb();
      }
      break;
    case 'append': {
      const slot = s.counter++;
      s.frontier[slot] = t.nb!;
      text = tr(`${who} appends ${t.nb}: fetch_add returns slot ${slot}, and currFrontier[${slot}] = ${t.nb}.`, `${who} nối thêm ${t.nb}: fetch_add trả về ô ${slot}, và currFrontier[${slot}] = ${t.nb}.`);
      nextNb();
      break;
    }
    case 'done':
      throw new Error('thread already finished'); // i18n-ignore: developer error
  }
  t.steps++;
  const line = (t.pc === 'done' ? f.line.done : f.line[pc]) ?? 0;
  return { state: s, event: { thread: mv.thread, line, text } };
}

// ---------------------------------------------------------------- outcomes
export type OutcomeKind = 'correct' | 'duplicate-label' | 'duplicate-entry' | 'missed' | 'wrong-level' | 'never-terminates';
export const OUTCOME_NAME: Record<OutcomeKind, string> = {
  get correct() { return tr('correct', 'đúng'); },
  get 'duplicate-label'() { return tr('duplicate label', 'gán nhãn trùng'); },
  get 'duplicate-entry'() { return tr('duplicate frontier entry', 'mục frontier trùng'); },
  get missed() { return tr('missed vertex', 'sót đỉnh'); },
  get 'wrong-level'() { return tr('wrong level', 'sai mức'); },
  get 'never-terminates'() { return tr('never terminates', 'không bao giờ kết thúc'); },
};
/** Worst first. */
export const OUTCOME_ORDER: OutcomeKind[] = ['never-terminates', 'wrong-level', 'missed', 'duplicate-entry', 'duplicate-label', 'correct'];
export interface Outcome {
  kind: OutcomeKind;
  /** Everything wrong with the final state, worst first. */
  flags: OutcomeKind[];
  detail: string;
}

/** Judge a finished level. */
export function classify(sc: Scenario, st: State): Outcome {
  const flags: OutcomeKind[] = [];
  const notes: string[] = [];
  const wrong = sc.level0.flatMap((L, v) => (L !== UNVISITED && st.level[v] !== L ? [v] : []));
  if (wrong.length) {
    flags.push('wrong-level');
    notes.push(wrong.map((v) => tr(`vertex ${v} was at level ${sc.level0[v]} and now reads ${show(st.level[v])}`, `đỉnh ${v} trước đó ở mức ${sc.level0[v]} và giờ đọc ra ${show(st.level[v])}`)).join('; '));
  }
  const missed = sc.expectNew.filter((v) => st.level[v] === UNVISITED);
  if (missed.length) {
    flags.push('missed');
    notes.push(tr(`vertex ${missed.join(' and ')} is still UNVISITED`, `đỉnh ${missed.join(' và ')} vẫn là UNVISITED`));
  }
  const seen = new Map<number, number>();
  for (const v of st.frontier) seen.set(v, (seen.get(v) ?? 0) + 1);
  const twice = [...seen].filter(([, c]) => c > 1).map(([v]) => v);
  if (twice.length) {
    flags.push('duplicate-entry');
    notes.push(tr(`vertex ${twice.join(' and ')} is in the frontier more than once`, `đỉnh ${twice.join(' và ')} có mặt trong frontier nhiều hơn một lần`));
  }
  const relabelled = st.writes.flatMap((w, v) => (w > 1 ? [v] : []));
  if (relabelled.length) {
    flags.push('duplicate-label');
    notes.push(tr(`level[${relabelled.join('] and level[')}] was written more than once`, `level[${relabelled.join('] và level[')}] bị ghi nhiều hơn một lần`));
  }
  const kind = flags[0] ?? 'correct';
  return { kind, flags, detail: notes.join('; ') || tr('every new vertex labelled once, nothing else touched', 'mỗi đỉnh mới được gán nhãn một lần, không gì khác bị động tới') };
}

/** Run the threads one after another, each to its end (the emulator's forward and reversed orders). */
export function runOrder(f: Form, sc: Scenario, order: 'forward' | 'reverse', bound = 60): { states: State[]; events: Event[]; outcome: Outcome } {
  const ids = sc.threads.map((_, i) => i);
  if (order === 'reverse') ids.reverse();
  let st = initial(sc);
  const states = [st];
  const events: Event[] = [];
  for (const id of ids)
    while (st.threads[id].pc !== 'done') {
      if (st.threads[id].steps >= bound) {
        return { states, events, outcome: { kind: 'never-terminates', flags: ['never-terminates'], detail: tr(`thread on ${sc.threads[id].vertex} is still looping after ${bound} steps`, `luồng trên ${sc.threads[id].vertex} vẫn đang lặp sau ${bound} bước`) } };
      }
      const r = step(f, sc, st, { thread: id });
      st = r.state;
      states.push(st);
      events.push(r.event);
    }
  return { states, events, outcome: classify(sc, st) };
}

/** Replay a schedule from the start. */
export function replay(f: Form, sc: Scenario, moves: Move[]): { states: State[]; events: Event[] } {
  let st = initial(sc);
  const states = [st];
  const events: Event[] = [];
  for (const mv of moves) {
    const r = step(f, sc, st, mv);
    st = r.state;
    states.push(st);
    events.push(r.event);
  }
  return { states, events };
}

const key = (st: State) =>
  st.level.join(',') + '|' + st.writes.join(',') + '|' + st.frontier.join(',') + '|' + st.counter + '|' + st.spurious + '|' +
  st.threads.map((t) => `${t.pc}:${t.k}:${t.e}:${t.won}`).join(';');

export interface Enumeration {
  /** Each outcome that some schedule reaches, with one schedule that reaches it. */
  outcomes: { kind: OutcomeKind; example: Move[]; detail: string }[];
  states: number;
  /** A state can repeat: some schedule never ends. */
  cyclic: boolean;
}

/**
 * Every interleaving of the threads, with up to `spurious` injected failures in total. The
 * reachable states are explored once each; a schedule that returns to a state it has already
 * been in on the same path can go round for ever, which is reported as "never terminates".
 */
export function enumerate(f: Form, sc: Scenario, opts: { spurious?: number; maxStates?: number } = {}): Enumeration {
  const maxSpurious = f.weak ? opts.spurious ?? 0 : 0;
  const maxStates = opts.maxStates ?? 200000;
  const s0 = initial(sc);
  const colour = new Map<string, 1 | 2>(); // 1: on the current path, 2: finished
  const found = new Map<OutcomeKind, { example: Move[]; detail: string }>();
  let cyclic = false;
  const path: Move[] = [];
  const moves = (st: State): Move[] => {
    const out: Move[] = [];
    st.threads.forEach((_, i) => {
      if (canMove(f, st, { thread: i })) out.push({ thread: i });
      if (canMove(f, st, { thread: i, spurious: true }, maxSpurious)) out.push({ thread: i, spurious: true });
    });
    return out;
  };
  // Iterative depth-first search: a stack of (state, its moves, next move index).
  const stack: { st: State; k: string; mv: Move[]; i: number }[] = [];
  const push = (st: State) => {
    const k = key(st);
    colour.set(k, 1);
    stack.push({ st, k, mv: moves(st), i: 0 });
    if (isDone(st)) {
      const o = classify(sc, st);
      if (!found.has(o.kind)) found.set(o.kind, { example: path.slice(), detail: o.detail });
    }
  };
  push(s0);
  while (stack.length) {
    if (colour.size > maxStates) throw new Error('too many states');
    const top = stack[stack.length - 1];
    if (top.i >= top.mv.length) {
      colour.set(top.k, 2);
      stack.pop();
      path.pop();
      continue;
    }
    const mv = top.mv[top.i++];
    const nxt = step(f, sc, top.st, mv).state;
    const k = key(nxt);
    const c = colour.get(k);
    if (c === 1) {
      cyclic = true;
      if (!found.has('never-terminates')) {
        const t = sc.threads[mv.thread].vertex;
        found.set('never-terminates', { example: [...path, mv], detail: tr(`the thread on ${t} comes back to a state it was already in, and can go round for ever`, `luồng trên ${t} quay lại một trạng thái nó đã ở trước đó, và có thể lặp vòng mãi mãi`) });
      }
      continue;
    }
    if (c === 2) continue;
    path.push(mv);
    push(nxt);
  }
  const outcomes = OUTCOME_ORDER.filter((k) => found.has(k)).map((kind) => ({ kind, ...found.get(kind)! }));
  return { outcomes, states: colour.size, cyclic };
}

// ---------------------------------------------------------------- listings
/**
 * The lines of `after` that are not in `before`, 1-based: what a corrected listing changed or
 * added. A longest-common-subsequence diff on whole lines, ignoring trailing spaces.
 */
export function changedLines(before: string, after: string): number[] {
  const a = before.split('\n').map((l) => l.trimEnd());
  const b = after.split('\n').map((l) => l.trimEnd());
  const n = a.length;
  const m = b.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--) dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const keep = new Set<number>();
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      keep.add(j);
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) i++;
    else j++;
  }
  return b.flatMap((_, k) => (keep.has(k) ? [] : [k + 1]));
}

/** The text between "// region: name" and "// endregion" in a source file. */
export function region(src: string, name: string): string | null {
  const m = new RegExp('//\\s*region:\\s*' + name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\s*\\n([\\s\\S]*?)\\n\\s*//\\s*endregion').exec(src);
  return m ? m[1] : null;
}
