// Chapter 6, section 6.2: the book's model of a DRAM system as channels and banks.
// It runs the bus calculator, the channel-and-banks timeline, the dealer (with its
// replay of the toy matrix multiplication) and the bytes-in-flight explorer of
// section 6.2, and every frame of the narrated animation "Why one bank is not enough".
//
// Sources for the rules (the lesson cites them where they are used):
//  - the book, section 6.2, p. 133: a bus moves width x clock bytes per second, twice
//    that with double data rate; the DDR5-6400 and HBM3E examples;
//  - the book, pp. 133 to 134 and Fig. 6.8: an access to a bank is a long cell-array
//    latency followed by a short burst on the channel's bus; with a latency-to-burst
//    ratio R, one bank keeps the bus busy at most 1/(R + 1) of the time and R + 1 banks
//    are needed to fill it; one bank serves one access at a time (a bank conflict);
//  - the book, p. 135 and Fig. 6.9: consecutive bursts are dealt across channels first,
//    then across banks;
//  - the book, pp. 136 to 137 and Figs. 6.10 and 6.11: which M elements each block of a
//    4 x 4 tiled multiplication loads in each phase;
//  - notes/today/ch06-facts.md A9 and B3: AMD documents 256-byte interleaving across 32
//    L2 channels on CDNA GPUs; NVIDIA and Apple publish no address mapping.
//
// The model of a channel (what it leaves out is said beside the explorer):
//  - time is counted in bursts: a burst takes 1 unit on the bus, the cell-array latency
//    before it takes R units, the same every time;
//  - the controller has a queue of requests that never runs dry and starts them in
//    arrival order, at most one per unit of time; a request starts when its bank is
//    free; no reordering, no refresh, no row-buffer hits, no read/write turnaround;
//  - a bank is busy from the start of an access until the end of its burst.
//
// Little's law (bytes in flight = bandwidth x latency) in bytesInFlight is ours, not
// the book's; the lesson labels it so.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- bus bandwidth
export interface Bus {
  /** Width of one bus in bits. */
  widthBits: number;
  /** Bus clock in GHz. */
  clockGHz: number;
  /** Double data rate: two transfers per clock, on the rising and the falling edge. */
  ddr: boolean;
  /** How many such buses work side by side. */
  buses: number;
}

/** Bandwidth in GB/s (10^9 bytes per second) of `buses` buses: width x clock (x 2 for DDR). */
export function busBandwidth(widthBits: number, clockGHz: number, ddr: boolean, buses = 1): number {
  return (widthBits / 8) * clockGHz * (ddr ? 2 : 1) * buses;
}

/** Buses of `perBusGBs` each needed to supply `targetGBs`, rounded up. */
export function busesFor(targetGBs: number, perBusGBs: number): number {
  if (!(perBusGBs > 0) || !(targetGBs > 0)) return 0;
  return Math.ceil(targetGBs / perBusGBs - 1e-9);
}

/** The book's two examples (p. 133). */
export const BUS_PRESETS: Record<'ddr5' | 'hbm3e', Bus & { label: string }> = {
  // 64-bit DDR5-6400: 3.2 GHz, double data rate, one bus: 51.2 GB/s.
  ddr5: {
    get label() {
      return tr("DDR5-6400 (the book's example)", 'DDR5-6400 (ví dụ của sách)');
    },
    widthBits: 64, clockGHz: 3.2, ddr: true, buses: 1,
  },
  // HBM3E as printed: a 4.9 GHz bus clock, 16 buses of 64 bits. The book says "1 TB/s";
  // its own formula gives 1254.4 GB/s (notes item 3, facts A8 and E1).
  hbm3e: {
    get label() {
      return tr("one HBM3E stack (the book's inputs)", 'một stack HBM3E (số liệu đầu vào của sách)');
    },
    widthBits: 64, clockGHz: 4.9, ddr: true, buses: 16,
  },
};

/** The book's two demand figures (p. 133): a modern CPU, a GPU. */
export const DEMAND = { cpu: 200, gpu: 1000 } as const;

// ---------------------------------------------------------------- one channel, its banks, in time
export interface ChannelConfig {
  /** Latency-to-burst ratio: cell-array latency in units of one burst time. */
  R: number;
  /** Banks on the channel. */
  banks: number;
  /** Requests to simulate (the queue never runs dry before they are all started). */
  requests?: number;
  /** Share of requests sent to the bank the previous request went to, which is still busy (0 to 1). */
  conflictShare?: number;
  /** Seed for the conflict draws, so a drawing is the same every time. */
  seed?: number;
  /** Request numbers forced to conflict (sent to the previous request's bank), whatever the share. */
  forced?: number[];
}

export interface Access {
  /** Request number, in arrival order. */
  k: number;
  bank: number;
  /** When the bank starts the access: the latency runs from here for R units. */
  start: number;
  /** When its burst starts on the bus; it lasts one unit. */
  burst: number;
  /** Sent to a bank that was still busy with the previous request. */
  conflict: boolean;
}

export interface Segment {
  kind: 'latency' | 'burst';
  start: number;
  end: number;
  /** Request number. */
  k: number;
}

export interface ChannelTimeline {
  config: Required<Omit<ChannelConfig, 'forced'>> & { forced: number[] };
  accesses: Access[];
  /** Per bank, its latency and burst segments in time order. */
  banks: Segment[][];
  /** The bus: one burst segment per access, in time order. */
  bus: Segment[];
  /** Fraction of time the bus carries a burst, measured over the steady part of the run. */
  utilisation: number;
  /** The time of the last burst's end. */
  end: number;
}

/** A small deterministic generator (mulberry32), so conflict draws repeat exactly. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const whole = (v: number, min: number) => (Number.isFinite(v) ? Math.max(min, Math.floor(v)) : min);

/** The book's bound: one bank fills 1/(R + 1) of the bus, and banks add up until the bus is full. */
export function utilisationBound(R: number, banks: number): number {
  return Math.min(1, Math.max(0, banks) / (Math.max(0, R) + 1));
}

/** Fewest banks that keep the bus busy all the time: R + 1 (p. 134). */
export const banksNeeded = (R: number) => Math.ceil(Math.max(0, R)) + 1;

export function channelTimeline(cfg: ChannelConfig): ChannelTimeline {
  const R = Math.max(0, cfg.R);
  const nb = whole(cfg.banks, 1);
  const n = whole(cfg.requests ?? Math.max(240, nb * 8), 2);
  const share = Math.max(0, Math.min(1, cfg.conflictShare ?? 0));
  const seed = cfg.seed ?? 1;
  const forced = cfg.forced ?? [];
  const draw = rng(seed);
  const bankFree = new Array<number>(nb).fill(0);
  const accesses: Access[] = [];
  let prevStart = -1;
  let prevBank = -1;
  for (let k = 0; k < n; k++) {
    // Draw for every request, so changing the share does not reshuffle which ones conflict.
    const roll = draw();
    const conflict = k > 0 && nb > 1 && (forced.includes(k) || roll < share);
    const bank = k === 0 ? 0 : conflict ? prevBank : (prevBank + 1) % nb;
    const start = Math.max(prevStart + 1, bankFree[bank]);
    const burst = start + R;
    bankFree[bank] = burst + 1;
    accesses.push({ k, bank, start, burst, conflict });
    prevStart = start;
    prevBank = bank;
  }
  const banks: Segment[][] = Array.from({ length: nb }, () => []);
  for (const a of accesses) {
    if (R > 0) banks[a.bank].push({ kind: 'latency', start: a.start, end: a.burst, k: a.k });
    banks[a.bank].push({ kind: 'burst', start: a.burst, end: a.burst + 1, k: a.k });
  }
  const bus: Segment[] = accesses.map((a) => ({ kind: 'burst', start: a.burst, end: a.burst + 1, k: a.k }));
  // Steady state: from the burst of request k0 to the burst of request k1, a whole number
  // of rounds of the banks apart, after one round of start-up. Exact without conflicts.
  const k0 = Math.min(nb, Math.floor(n / 4));
  const rounds = Math.max(1, Math.floor((n - 1 - k0) / nb));
  const k1 = Math.min(n - 1, k0 + rounds * nb);
  const span = accesses[k1].burst - accesses[k0].burst;
  const utilisation = span > 0 ? (k1 - k0) / span : 1;
  return {
    config: { R, banks: nb, requests: n, conflictShare: share, seed, forced },
    accesses,
    banks,
    bus,
    utilisation,
    end: accesses[n - 1].burst + 1,
  };
}

/** Busy fraction of the bus between two times, from the timeline's bursts. */
export function busBusy(tl: ChannelTimeline, from: number, to: number): number {
  if (to <= from) return 0;
  let busy = 0;
  for (const s of tl.bus) busy += Math.max(0, Math.min(to, s.end) - Math.max(from, s.start));
  return busy / (to - from);
}

/** Requests in flight at time t: started and not yet through their burst. */
export function inFlight(tl: ChannelTimeline, t: number): number {
  return tl.accesses.filter((a) => a.start <= t && t < a.burst + 1).length;
}

/** Idle stretches of the bus between `from` and `to`, as [start, end) pairs. */
export function busGaps(tl: ChannelTimeline, from: number, to: number): [number, number][] {
  const gaps: [number, number][] = [];
  let t = from;
  for (const s of tl.bus) {
    if (s.end <= from) continue;
    if (s.start >= to) break;
    if (s.start > t) gaps.push([t, s.start]);
    t = Math.max(t, s.end);
  }
  if (t < to) gaps.push([t, to]);
  return gaps;
}

// ---------------------------------------------------------------- dealing addresses to channels and banks
export interface Mapping {
  /** Bytes per array element. */
  elemBytes: number;
  /** Bytes dealt to one channel before moving to the next: the burst in the book's toy. */
  burstBytes: number;
  channels: number;
  banks: number;
}

export interface Place {
  burst: number;
  channel: number;
  bank: number;
}

/** Fig. 6.9's mapping (p. 135): bursts of two 4-byte elements, 4 channels, 2 banks. */
export const FIG_6_9: Mapping = { elemBytes: 4, burstBytes: 8, channels: 4, banks: 2 };

/**
 * AMD's documented interleave on CDNA GPUs (facts A9, B3): 256 bytes per L2 channel,
 * 32 channels. These are L2 channels; AMD does not document a bank level for them, so
 * the preset uses one bank. NVIDIA and Apple publish no mapping.
 */
export const AMD_CDNA_L2: Mapping = { elemBytes: 4, burstBytes: 256, channels: 32, banks: 1 };

/** Where element i of an array that starts at channel 0, bank 0 lands: channels first, then banks. */
export function interleave(i: number, elemBytes: number, burstBytes: number, channels: number, banks: number): Place {
  const burst = Math.floor((i * elemBytes) / burstBytes);
  const channel = burst % channels;
  const bank = Math.floor(burst / channels) % banks;
  return { burst, channel, bank };
}

export const place = (i: number, m: Mapping) => interleave(i, m.elemBytes, m.burstBytes, m.channels, m.banks);

/** Elements 0..n-1 grouped by where they land: cells[channel][bank] lists element indices in order. */
export function deal(n: number, m: Mapping): number[][][] {
  const cells: number[][][] = Array.from({ length: m.channels }, () => Array.from({ length: m.banks }, () => []));
  for (let i = 0; i < n; i++) {
    const p = place(i, m);
    cells[p.channel][p.bank].push(i);
  }
  return cells;
}

/**
 * Elements that give every channel one whole burst, and every bank of every channel one:
 * the smallest array that spreads over all of them evenly. Fig. 6.9: 8 and 16 (p. 135).
 */
export function elementsToFill(m: Mapping): { channels: number; banks: number } {
  const perBurst = m.burstBytes / m.elemBytes;
  return { channels: m.channels * perBurst, banks: m.channels * m.banks * perBurst };
}

/** Channels hit by `indices`, sorted, without repeats. */
export const channelsOf = (indices: number[], m: Mapping) =>
  [...new Set(indices.map((i) => place(i, m).channel))].sort((a, b) => a - b);

// ---------------------------------------------------------------- the toy matrix multiplication, phase by phase
export interface BlockLoads {
  /** The block's row and column among the output tiles: the book's "Block 0,1" is row 0, column 1. */
  row: number;
  col: number;
  /** Linear indices of the M elements the block loads in this phase. */
  elements: number[];
  channels: number[];
}

export interface Phase {
  ph: number;
  blocks: BlockLoads[];
  /** Every channel the phase's M loads hit, across all blocks. */
  channels: number[];
}

/**
 * Figs. 6.10 and 6.11: a width x width row-major M, tile x tile blocks, every block running
 * at once. In phase ph, the block in tile row by loads M rows by*tile.., columns ph*tile..
 */
export function phaseChannels(width: number, tile: number, m: Mapping = FIG_6_9): Phase[] {
  const tiles = Math.floor(width / tile);
  const phases: Phase[] = [];
  for (let ph = 0; ph < tiles; ph++) {
    const blocks: BlockLoads[] = [];
    for (let row = 0; row < tiles; row++)
      for (let col = 0; col < tiles; col++) {
        const elements: number[] = [];
        for (let r = 0; r < tile; r++) for (let c = 0; c < tile; c++) elements.push((row * tile + r) * width + ph * tile + c);
        blocks.push({ row, col, elements, channels: channelsOf(elements, m) });
      }
    phases.push({ ph, blocks, channels: channelsOf(blocks.flatMap((b) => b.elements), m) });
  }
  return phases;
}

/** Channels hit by reading one column of a row-major matrix (rows 0..rows-1). */
export function columnChannels(width: number, col: number, rows: number, m: Mapping): number[] {
  return channelsOf(
    Array.from({ length: rows }, (_, r) => r * width + col),
    m,
  );
}

// ---------------------------------------------------------------- bytes in flight (ours: Little's law)
export interface InFlight {
  /** Bytes that must be on their way at any moment: bandwidth x latency. */
  bytes: number;
  /** Loads of `bytesPerLoad` each that this takes. */
  loads: number;
  /** The same in warp-wide load instructions, one load per lane. */
  warpLoads: number;
}

/** Little's law: GB/s is bytes per nanosecond, so bytes outstanding = GB/s x ns. */
export function bytesInFlight(bandwidthGBs: number, latencyNs: number, bytesPerLoad: number, warpWidth: number): InFlight {
  const bytes = Math.max(0, bandwidthGBs) * Math.max(0, latencyNs);
  const loads = bytesPerLoad > 0 ? bytes / bytesPerLoad : 0;
  const warpLoads = warpWidth > 0 ? loads / warpWidth : 0;
  return { bytes, loads, warpLoads };
}
