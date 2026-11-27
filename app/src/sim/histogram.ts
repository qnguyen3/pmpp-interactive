// Image data and binning for section 9.1 ("What a histogram computes").
//
// The book's running example is an 8x8 grayscale image binned into intervals of 64
// values (Fig. 9.1, p. 202). The images here are our own, never the book's tree. The
// sequential loop this model mirrors is labs/ch09/snippets/histogram_sequential.cpp
// (Fig. 9.2, p. 203); histogram.test.ts compiles that file and checks that it counts
// exactly what histogram() counts.

import { tr } from '../lib/locale.ts';

/** Side of the explorer's square images. */
export const SIDE = 8;
/** Pixels per preset image. */
export const PIXELS = SIDE * SIDE;
/** 8-bit grayscale: 0 is black, 255 is white (p. 202). */
export const LEVELS = 256;

export type PresetId = 'sky' | 'night' | 'noise' | 'flat';

export interface Preset {
  id: PresetId;
  label: string;
  /** Row-major, SIDE x SIDE values in 0..255. */
  values: number[];
}

/** Interval widths the explorer offers, and the number of bins each gives. */
export const WIDTHS = [64, 32, 16, 1] as const;
export const binCount = (intervalWidth: number) => Math.ceil(LEVELS / intervalWidth);

/** The bin a value falls in: one integer division (p. 203). */
export function binOf(value: number, intervalWidth: number): number {
  if (!Number.isInteger(value) || value < 0 || value >= LEVELS) throw new Error(`pixel value ${value} is not in 0..255`);
  if (!Number.isInteger(intervalWidth) || intervalWidth < 1) throw new Error(`interval width ${intervalWidth}`);
  return Math.floor(value / intervalWidth);
}

/** The values [lo, hi] that land in bin b. */
export function binRange(b: number, intervalWidth: number): [number, number] {
  return [b * intervalWidth, Math.min(LEVELS - 1, (b + 1) * intervalWidth - 1)];
}

/**
 * Counts per bin. With intervalWidth 1 this is Fig. 9.2: one bin per value, bins
 * starting at zero (the listing assumes the caller zeroed them).
 */
export function histogram(values: readonly number[], intervalWidth: number): number[] {
  const bins = new Array<number>(binCount(intervalWidth)).fill(0);
  for (const v of values) bins[binOf(v, intervalWidth)] += 1;
  return bins;
}

/** Row-major position of (row, col) in an image `width` pixels wide (chapter 3). */
export function flatIndex(row: number, col: number, width: number): number {
  if (col < 0 || col >= width || row < 0) throw new Error(`(${row}, ${col}) is outside a ${width}-wide image`);
  return row * width + col;
}

/** The fullest bin: with one thread per pixel, its height is the number of threads that collide on it. */
export function hottest(counts: readonly number[]): { bin: number; count: number; share: number } {
  let bin = 0;
  for (let b = 1; b < counts.length; b++) if (counts[b] > counts[bin]) bin = b;
  const total = counts.reduce((s, c) => s + c, 0);
  return { bin, count: counts[bin], share: total ? counts[bin] / total : 0 };
}

// ------------------------------------------------------------------ deterministic noise
/** mulberry32: a small seeded generator, so every "random" image is the same on every run. */
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

// ------------------------------------------------------------------ the presets (ours)
// A bright sky over dark ground: a pure-white sky with a small grey cloud, a broken
// horizon, then darker rows of ground. Skewed on purpose: almost half the pixels are 255.
const SKY = [
  255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 240, 240, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 240, 240, 255,
  255, 255, 255, 255, 255, 255, 255, 255,
  150, 140, 255, 255, 160, 150, 140, 255,
  90, 80, 100, 120, 90, 70, 80, 100,
  50, 60, 40, 50, 70, 60, 40, 30,
  20, 30, 40, 20, 10, 30, 20, 40,
];
// A night scene: near-black sky with three stars and a moon, darker ground below.
const NIGHT = [
  10, 12, 255, 10, 8, 10, 230, 235,
  12, 10, 10, 14, 10, 12, 235, 230,
  10, 255, 12, 10, 10, 16, 12, 10,
  14, 12, 10, 12, 255, 12, 10, 12,
  20, 22, 18, 20, 24, 20, 22, 18,
  30, 28, 32, 30, 26, 30, 34, 30,
  40, 36, 44, 40, 38, 42, 40, 36,
  60, 56, 62, 58, 64, 60, 66, 58,
];

function noise(seed: number, n: number): number[] {
  const r = rng(seed);
  return Array.from({ length: n }, () => Math.floor(r() * LEVELS));
}

export const PRESETS: Preset[] = [
  {
    id: 'sky',
    get label() {
      return tr('bright sky over dark ground', 'trời sáng trên nền đất tối');
    },
    values: SKY,
  },
  {
    id: 'night',
    get label() {
      return tr('night scene', 'cảnh ban đêm');
    },
    values: NIGHT,
  },
  {
    id: 'noise',
    get label() {
      return tr('noise', 'nhiễu');
    },
    values: noise(9, PIXELS),
  },
  {
    id: 'flat',
    get label() {
      return tr('flat grey', 'xám đồng nhất');
    },
    values: new Array<number>(PIXELS).fill(128),
  },
];

export const preset = (id: PresetId): Preset => PRESETS.find((p) => p.id === id)!;

// ------------------------------------------------------------------ larger images
export type ImageKind = 'uniform' | 'bright' | 'constant';

/**
 * A seeded image of n pixels. 'uniform': every byte equally likely. 'constant': every
 * pixel 255. 'bright': pixels drawn from the sky preset's own distribution, so about
 * half of them are 255.
 */
export function generateImage(n: number, kind: ImageKind, seed = 1): number[] {
  if (kind === 'constant') return new Array<number>(n).fill(255);
  const r = rng(seed);
  if (kind === 'uniform') return Array.from({ length: n }, () => Math.floor(r() * LEVELS));
  return Array.from({ length: n }, () => SKY[Math.floor(r() * SKY.length)]);
}
