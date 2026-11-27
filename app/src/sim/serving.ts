// A request scheduler over decode steps, for the bridge in chapter 22, section 22.5:
// static batching against iteration-level ("continuous") batching. Not in the book, which
// names neither policy in chapter 22; chapter 20 mentions in-flight batching (p. 506).
//
// Sources for the two policies (notes/today/ch22-facts.md):
//   D1  Orca's abstract: with request-level scheduling "requests that have finished earlier
//       than other requests in a batch cannot return to the client, while newly arrived
//       requests have to wait until the current batch completely finishes"; with
//       iteration-level scheduling the batch can change at every iteration. TensorRT-LLM's
//       docs: in-flight, continuous and iteration-level batching are one technique.
//   D2  vLLM's max_num_seqs, D3 llama-server's slots (-np), D4 mlx-lm's
//       --decode-concurrency: each server caps the number of requests in a step.
//
// The model (what it leaves out is stated wherever it is shown):
//   - Time advances in decode steps. A step adds one token to every request that is still
//     generating, and takes stepTime(active) seconds, a function of the number of active
//     requests only (the caller supplies it, from batching.ts).
//   - No prefill: a request starts decoding the step after it is admitted. No KV-cache
//     memory limit (chapter 20, p. 505, is why real servers have one). No speculation.
//   - Static: a batch is formed from whatever is waiting (up to `slots`, first come first
//     served) when no batch is running, and runs until its longest request finishes. A
//     finished request's slot then sits idle and costs nothing (the kindest assumption for
//     static batching), and the request is returned only when the batch ends.
//   - Continuous: at every step boundary finished requests leave (and are returned) and
//     waiting ones join, up to `slots`.
//   - Arrivals and output lengths are drawn from a seeded generator, so a run is repeatable.
//
// Erasable TypeScript only and no imports: the browser checks import this file directly.

import { tr } from '../lib/locale.ts';

export type Policy = 'static' | 'continuous';
export type Mix = 'short' | 'long';

export interface Request {
  id: number;
  /** Arrival time, seconds (or any unit the step times use). */
  arrival: number;
  /** Output tokens wanted. */
  length: number;
}

/** A request is "short" if it wants at most this many tokens. */
export const SHORT_MAX = 32;

/** mulberry32: a small, well-mixed, seeded 32-bit generator. */
export function rng(seed: number): () => number {
  let h = seed >>> 0;
  return () => {
    h = (h + 0x6d2b79f5) >>> 0;
    let t = Math.imul(h ^ (h >>> 15), h | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const uniformInt = (r: () => number, lo: number, hi: number) => lo + Math.floor(r() * (hi - lo + 1));

/** Output-length mixes (ours, for illustration; no server's statistics). */
export const MIXES: Record<Mix, { label: string; mean: number; draw: (r: () => number) => number }> = {
  short: {
    get label() {
      return tr('similar lengths (8 to 40 tokens)', 'độ dài gần nhau (8 đến 40 token)');
    },
    mean: 24,
    draw: (r) => uniformInt(r, 8, 40),
  },
  long: {
    get label() {
      return tr('long-tailed (4 in 5 short, 1 in 5 of 128 to 512 tokens)', 'đuôi dài (4 trên 5 ngắn, 1 trên 5 dài 128 đến 512 token)');
    },
    // 0.8 · 18 + 0.2 · 320
    mean: 78.4,
    draw: (r) => (r() < 0.8 ? uniformInt(r, 4, 32) : uniformInt(r, 128, 512)),
  },
};

/** `count` requests arriving at `rate` per second on average (exponential gaps), the first at time 0. */
export function makeRequests({ count, rate, mix, seed }: { count: number; rate: number; mix: Mix; seed: number }): Request[] {
  const r = rng(seed);
  const out: Request[] = [];
  let t = 0;
  for (let id = 0; id < count; id++) {
    if (id > 0) t += -Math.log(1 - r()) / rate;
    out.push({ id, arrival: t, length: MIXES[mix].draw(r) });
  }
  return out;
}

export interface Served extends Request {
  /** Slot index it occupied. */
  slot: number;
  /** Time and step index at which it was admitted to the batch. */
  admit: number;
  admitStep: number;
  /** Time its last token was produced, and the index of that step. */
  lastToken: number;
  lastStep: number;
  /** Time it was returned to the client, and the index of the first step after it left. */
  returned: number;
  returnStep: number;
}

export interface StepRecord {
  start: number;
  dt: number;
  /** Requests generating a token in this step. */
  active: number;
  /** Finished requests still holding a slot (static only). */
  held: number;
  /** Requests waiting outside the batch. */
  queued: number;
}

export interface ServeResult {
  policy: Policy;
  slots: number;
  served: Served[];
  steps: StepRecord[];
  tokens: number;
  /** From the first arrival to the last return. */
  elapsed: number;
  tokensPerSecond: number;
  /** Per request: (time of its last token − admission) / its tokens. Mean and worst over requests. */
  meanTimePerToken: number;
  worstTimePerToken: number;
  /** Mean of (returned − arrival) over the short requests (at most `shortMax` tokens); null if there are none. */
  meanFinishShort: number | null;
  /** Generating slot-steps over all slot-steps: at most 1. */
  utilisation: number;
  /** Mean number of requests generating per step. */
  meanBatch: number;
}

export interface ServeOptions {
  requests: Request[];
  slots: number;
  policy: Policy;
  /** Seconds one step takes with `active` requests generating. */
  stepTime: (active: number) => number;
  /** A request of at most this many tokens counts as short (default SHORT_MAX). */
  shortMax?: number;
}

export function serve({ requests, slots, policy, stepTime, shortMax = SHORT_MAX }: ServeOptions): ServeResult {
  if (slots < 1) throw new Error('slots must be at least 1');
  const served: Served[] = [...requests]
    .sort((a, b) => a.arrival - b.arrival || a.id - b.id)
    .map((r) => ({ ...r, slot: -1, admit: NaN, admitStep: -1, lastToken: NaN, lastStep: -1, returned: NaN, returnStep: -1 }));
  const made = new Map<number, number>(); // tokens made so far, by id
  let occupants: Served[] = []; // requests holding a slot
  let free = Array.from({ length: slots }, (_, k) => k); // free slot indices, ascending
  const release = (k: number) => {
    let i = free.length;
    while (i > 0 && free[i - 1] > k) i--;
    free.splice(i, 0, k);
  };
  const queue: Served[] = [];
  let head = 0; // queue[head] is the oldest waiting request
  const steps: StepRecord[] = [];
  let next = 0;
  let finished = 0;
  let t = served.length ? served[0].arrival : 0;
  const done = (r: Served) => (made.get(r.id) ?? 0) >= r.length;

  while (finished < served.length) {
    while (next < served.length && served[next].arrival <= t) queue.push(served[next++]);
    if (policy === 'static' && occupants.length && occupants.every(done)) {
      // A static batch ends only when every request in it is done; then all return together.
      for (const s of occupants) {
        s.returned = t;
        s.returnStep = steps.length;
        finished++;
      }
      occupants = [];
      free = Array.from({ length: slots }, (_, k) => k);
    }
    if (policy === 'continuous' || occupants.length === 0) {
      while (head < queue.length && free.length) {
        const r = queue[head++];
        r.slot = free.shift()!;
        r.admit = t;
        r.admitStep = steps.length;
        occupants.push(r);
      }
    }
    const active = occupants.filter((s) => !done(s));
    if (!active.length) {
      if (finished >= served.length) break;
      // Nothing to compute: wait for the next arrival.
      if (next < served.length) t = Math.max(t, served[next].arrival);
      continue;
    }
    const dt = stepTime(active.length);
    steps.push({ start: t, dt, active: active.length, held: occupants.length - active.length, queued: queue.length - head });
    t += dt;
    for (const r of active) {
      made.set(r.id, (made.get(r.id) ?? 0) + 1);
      if (done(r)) {
        r.lastToken = t;
        r.lastStep = steps.length - 1;
        if (policy === 'continuous') {
          r.returned = t;
          r.returnStep = steps.length;
          release(r.slot);
          finished++;
        }
      }
    }
    if (policy === 'continuous') occupants = occupants.filter((s) => !done(s));
  }

  const tokens = served.reduce((s, r) => s + r.length, 0);
  const first = served.length ? served[0].arrival : 0;
  const last = served.reduce((m, r) => Math.max(m, r.returned), first);
  const elapsed = last - first;
  const perToken = served.map((r) => (r.lastToken - r.admit) / r.length);
  const short = served.filter((r) => r.length <= shortMax);
  const activeSteps = steps.reduce((s, x) => s + x.active, 0);
  return {
    policy,
    slots,
    served,
    steps,
    tokens,
    elapsed,
    tokensPerSecond: elapsed > 0 ? tokens / elapsed : 0,
    meanTimePerToken: perToken.length ? perToken.reduce((s, x) => s + x, 0) / perToken.length : 0,
    worstTimePerToken: perToken.length ? Math.max(...perToken) : 0,
    meanFinishShort: short.length ? short.reduce((s, r) => s + (r.returned - r.arrival), 0) / short.length : null,
    utilisation: steps.length ? activeSteps / (slots * steps.length) : 0,
    meanBatch: steps.length ? activeSteps / steps.length : 0,
  };
}

export type SlotState = { id: number; state: 'gen' | 'done' } | null;

/** What each slot holds during step `k`: a request generating, a finished one still held, or nothing. */
export function slotsAt(result: ServeResult, k: number): SlotState[] {
  const out: SlotState[] = Array.from({ length: result.slots }, () => null);
  for (const r of result.served) {
    if (r.admitStep <= k && k < r.returnStep) out[r.slot] = { id: r.id, state: k <= r.lastStep ? 'gen' : 'done' };
  }
  return out;
}

/** Requests that have arrived by the start of step `k` and are not yet admitted. */
export function waitingAt(result: ServeResult, k: number): number[] {
  const t = k < result.steps.length ? result.steps[k].start : Infinity;
  return result.served.filter((r) => r.arrival <= t && (r.admitStep < 0 || r.admitStep > k)).map((r) => r.id);
}

// ---------------------------------------------------------------- the scene's six requests
/**
 * The narrated scene's example: six requests, arrival times in steps of one unit, four
 * slots. Ours, chosen so that static batching shows two short requests finishing early
 * and two newcomers waiting for the longest.
 */
export const SCENE_SLOTS = 4;
export const SCENE_REQUESTS: Request[] = [
  { id: 0, arrival: 0, length: 3 },
  { id: 1, arrival: 0, length: 8 },
  { id: 2, arrival: 0, length: 2 },
  { id: 3, arrival: 0, length: 6 },
  { id: 4, arrival: 1, length: 4 },
  { id: 5, arrival: 2, length: 3 },
];
export const SCENE_NAMES = ['A', 'B', 'C', 'D', 'E', 'F'];
/** In the scene, "short" means at most 4 tokens: A, C, E and F. */
export const SCENE_SHORT_MAX = 4;
