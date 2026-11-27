// Contiguous reservation against paged blocks for the KV cache, chapter 20 section 20.7
// (book p. 509, corrected by notes/today/ch20-facts.md E6).
//
// What the facts file says PagedAttention is (vLLM paper, section 4.1): the cache of each
// sequence is split into KV blocks holding a fixed number of tokens, stored in
// non-contiguous GPU memory and found through a per-sequence block table; host memory is
// used only when "vLLM exhausts free physical blocks for new tokens" and evicts sequences
// (section 4.5). The contiguous alternative is the Llama reference's: a cache preallocated
// to max_seq_len for every sequence (facts D3).
//
// The model, one layer, sizes in tokens:
//   - A seeded stream of requests. Request i arrives 0 to 2 steps after request i - 1, with
//     a prompt of 1 to max_seq/8 tokens and a final length in (prompt, max_seq], drawn
//     skewed short (our assumption: most requests stop well before the maximum). The final
//     length is hidden from both allocators.
//   - Each step, in order: arrivals join the back of the queue; every live request grows by
//     one token (one decode step); requests that reached their final length finish and give
//     their memory back; then waiting requests are admitted from the front of the queue
//     while they fit (strict first come, first served), and their prompt is written.
//   - Contiguous: the pool is cut into floor(pool / max_seq) fixed slots of max_seq tokens;
//     a request takes a whole slot on admission. Nothing grows, nothing is evicted.
//   - Paged: the pool is cut into blocks of B tokens on a free list. Admission needs
//     ceil(length / B) blocks; a request whose last block is full takes the block at the
//     front of the free list; freed blocks go back to the front. If the free list is empty,
//     a live request is swapped out, as vLLM evicts sequences when it runs out of blocks
//     (E6): its blocks are freed and it waits at the front of the queue with its tokens kept
//     in host memory. Which request is ours to choose (the facts file does not say which
//     vLLM picks): the most recently admitted, so the oldest always makes progress. The copy
//     itself costs nothing here.
//   - Not modelled: prefix sharing, copy time, the pool's own fragmentation (fixed slots and
//     fixed blocks have none), more than one layer, time per step.
//
// Erasable TypeScript only and no imports: the browser checks import this file directly.

export interface PagedConfig {
  /** Pool size in tokens. */
  pool: number;
  /** Block size in tokens. */
  block: number;
  /** Longest sequence a request may reach, in tokens. */
  maxSeq: number;
  /** Requests in the stream. */
  requests: number;
  seed: number;
}

export interface Request {
  id: number;
  arrive: number;
  prompt: number;
  final: number;
}

export interface LiveRequest {
  id: number;
  len: number;
  final: number;
  /** Contiguous: the slot it holds. */
  slot?: number;
  /** Paged: its block table, logical block i at physical block blocks[i]. */
  blocks?: number[];
}

export interface SideFrame {
  live: LiveRequest[];
  waiting: number[];
  /** Paged: free block ids, front first. Contiguous: free slot ids. */
  free: number[];
  done: number;
  admitted: number;
  reserved: number;
  used: number;
  wasted: number;
  evictions: number;
  /** What happened this step, e.g. "r3 admitted". */
  events: string[];
}

export interface Frame {
  step: number;
  contiguous: SideFrame;
  paged: SideFrame;
}

export interface PagedRun {
  cfg: PagedConfig;
  requests: Request[];
  frames: Frame[];
  slots: number;
  blocks: number;
  /** Frames after which each side had finished every request (null if the cap was hit). */
  finishedAt: { contiguous: number | null; paged: number | null };
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Clamp a configuration to one the simulator can run: every request must fit both pools. */
export function normalise(cfg: PagedConfig): PagedConfig {
  const block = Math.max(1, Math.floor(cfg.block));
  const maxSeq = Math.max(2, Math.floor(cfg.maxSeq));
  const needBlocks = Math.ceil(maxSeq / block);
  const pool = Math.max(Math.floor(cfg.pool), maxSeq, needBlocks * block);
  return { pool, block, maxSeq, requests: Math.max(1, Math.floor(cfg.requests)), seed: Math.floor(cfg.seed) };
}

export function makeRequests(cfg: PagedConfig): Request[] {
  const r = mulberry32(cfg.seed * 1103515245 + 12345);
  const out: Request[] = [];
  let t = 0;
  for (let i = 0; i < cfg.requests; i++) {
    if (i > 0) t += Math.floor(r() * 3);
    const prompt = 1 + Math.floor(r() * Math.max(1, Math.floor(cfg.maxSeq / 8)));
    const room = cfg.maxSeq - prompt;
    const u = r();
    const final = Math.min(cfg.maxSeq, prompt + 1 + Math.floor(u * u * room));
    out.push({ id: i, arrive: t, prompt, final });
  }
  return out;
}

const blocksFor = (len: number, B: number) => Math.ceil(len / B);

/** Physical token slot of token t of a request in the paged pool. */
export function physicalSlot(blocks: number[], t: number, B: number): number {
  return blocks[Math.floor(t / B)] * B + (t % B);
}

export function simulate(input: PagedConfig): PagedRun {
  const cfg = normalise(input);
  const reqs = makeRequests(cfg);
  const B = cfg.block;
  const slots = Math.floor(cfg.pool / cfg.maxSeq);
  const nBlocks = Math.floor(cfg.pool / B);

  // Contiguous side.
  const cLive: LiveRequest[] = [];
  const cWait: number[] = [];
  const cFree: number[] = Array.from({ length: slots }, (_, i) => i);
  const cAdmitted = new Set<number>();
  let cDone = 0;

  // Paged side.
  const pLive: LiveRequest[] = [];
  const pWait: { id: number; len: number }[] = [];
  const pFree: number[] = Array.from({ length: nBlocks }, (_, i) => i);
  const pAdmitted = new Set<number>();
  let pDone = 0;
  let evictions = 0;

  const snap = (side: 'c' | 'p', events: string[]): SideFrame => {
    const live = (side === 'c' ? cLive : pLive).map((r) => ({ ...r, blocks: r.blocks ? [...r.blocks] : undefined }));
    const used = live.reduce((s, r) => s + r.len, 0);
    const reserved = side === 'c' ? live.length * cfg.maxSeq : live.reduce((s, r) => s + r.blocks!.length * B, 0);
    return {
      live,
      waiting: side === 'c' ? [...cWait] : pWait.map((w) => w.id),
      free: side === 'c' ? [...cFree] : [...pFree],
      done: side === 'c' ? cDone : pDone,
      admitted: side === 'c' ? cAdmitted.size : pAdmitted.size,
      reserved,
      used,
      wasted: reserved - used,
      evictions: side === 'c' ? 0 : evictions,
      events,
    };
  };

  const frames: Frame[] = [{ step: 0, contiguous: snap('c', []), paged: snap('p', []) }];
  const finishedAt: PagedRun['finishedAt'] = { contiguous: null, paged: null };
  const cap = cfg.requests * cfg.maxSeq * 4 + 1000;

  for (let t = 0; t < cap && (cDone < reqs.length || pDone < reqs.length); t++) {
    const cEv: string[] = [];
    const pEv: string[] = [];
    // 1. Arrivals.
    for (const q of reqs)
      if (q.arrive === t) {
        cWait.push(q.id);
        pWait.push({ id: q.id, len: q.prompt });
      }

    // 2. Growth, oldest admission first.
    for (const r of cLive) r.len++;
    for (const r of [...pLive]) {
      if (!pLive.includes(r)) continue; // swapped out earlier in this step
      if (r.len % B === 0) {
        if (pFree.length === 0) {
          const victim = pLive[pLive.length - 1];
          pLive.splice(pLive.length - 1, 1);
          pFree.unshift(...victim.blocks!);
          pWait.unshift({ id: victim.id, len: victim.len });
          evictions++;
          pEv.push(`r${victim.id} swapped out`); // i18n-ignore: identifier also shown to the reader
          if (victim === r) continue;
        }
        r.blocks!.push(pFree.shift()!);
      }
      r.len++;
    }

    // 3. Finishes.
    for (let i = cLive.length - 1; i >= 0; i--)
      if (cLive[i].len >= cLive[i].final) {
        const [r] = cLive.splice(i, 1);
        cFree.push(r.slot!);
        cFree.sort((a, b) => a - b);
        cDone++;
        cEv.push(`r${r.id} finished`); // i18n-ignore: identifier also shown to the reader
      }
    for (let i = pLive.length - 1; i >= 0; i--)
      if (pLive[i].len >= pLive[i].final) {
        const [r] = pLive.splice(i, 1);
        pFree.unshift(...r.blocks!);
        pDone++;
        pEv.push(`r${r.id} finished`); // i18n-ignore: identifier also shown to the reader
      }

    // 4. Admissions, strict first come, first served.
    while (cWait.length && cFree.length) {
      const id = cWait.shift()!;
      const q = reqs[id];
      cLive.push({ id, len: q.prompt, final: q.final, slot: cFree.shift()! });
      cAdmitted.add(id);
      cEv.push(`r${id} admitted`); // i18n-ignore: identifier also shown to the reader
    }
    while (pWait.length && pFree.length >= blocksFor(pWait[0].len, B)) {
      const w = pWait.shift()!;
      const n = blocksFor(w.len, B);
      pLive.push({ id: w.id, len: w.len, final: reqs[w.id].final, blocks: pFree.splice(0, n) });
      pEv.push(pAdmitted.has(w.id) ? `r${w.id} swapped back in` : `r${w.id} admitted`); // i18n-ignore: identifier also shown to the reader
      pAdmitted.add(w.id);
    }

    frames.push({ step: t + 1, contiguous: snap('c', cEv), paged: snap('p', pEv) });
    if (finishedAt.contiguous == null && cDone === reqs.length) finishedAt.contiguous = t + 1;
    if (finishedAt.paged == null && pDone === reqs.length) finishedAt.paged = t + 1;
  }
  return { cfg, requests: reqs, frames, slots, blocks: nBlocks, finishedAt };
}

/** A block table whose blocks are not consecutive in the pool. */
export const isScattered = (blocks: number[]) => blocks.some((b, i) => i > 0 && b !== blocks[i - 1] + 1);

/**
 * The first frame in which, on the paged side, a request finishes and a request that was
 * already waiting in the frame before is admitted: freed blocks taken at once. -1 if none.
 */
export function firstReuse(run: PagedRun): number {
  return run.frames.findIndex((f, i) => {
    if (i === 0 || !f.paged.events.some((e) => e.endsWith(' finished'))) return false;
    const waited = run.frames[i - 1].paged.waiting;
    return f.paged.events.some((e) => e.endsWith(' admitted') && waited.includes(Number(e.slice(1).split(' ')[0])));
  });
}

/** The largest number of requests live at once on each side over a run. */
export function peakLive(run: PagedRun): { contiguous: number; paged: number } {
  let c = 0;
  let p = 0;
  for (const f of run.frames) {
    c = Math.max(c, f.contiguous.live.length);
    p = Math.max(p, f.paged.live.length);
  }
  return { contiguous: c, paged: p };
}
