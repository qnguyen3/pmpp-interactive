// The model behind chapter 24's table audit, its "what limits what" classification
// and its threshold strip.
//
// What it models: the book's one table (Fig. 24.1, p. 578) exactly as printed, the
// same rows taken from NVIDIA's documents for one board on each side
// (notes/today/ch24-facts.md, group A), the arithmetic that connects the cells, and
// the threshold of section 5.1 (peak throughput over peak bandwidth) for each column.
//
// What it leaves out, and what the lesson says wherever this model is shown:
//   - Every figure is a published peak. Nothing here was measured, and a peak is a
//     limit, not what a kernel reaches.
//   - The book's 2007 column describes no single product: its bandwidth is one
//     board's and its capacity another's (facts A3, A4).
//   - The tensor-core figure is TF32 arithmetic counted with sparsity, and applies
//     only to matmul-shaped work (facts A9).
//   - Kernel ratios count the traffic a kernel asks for. Caches are ignored.
//   - The two linear-layer marks are 16-bit kernels set against FP32 and TF32 peaks.
//     A 16-bit kernel on tensor cores faces a higher 16-bit peak that the book's table
//     does not give, so read their regime as indicative; batch 1 is memory-bound on
//     any peak.
//   - The audit separates two questions: whether the table contradicts itself (a ratio
//     its two figures do not give, a unit that cannot be meant) and whether NVIDIA's
//     documents give a different figure or label. Only the first is "wrong" here.

import { tr } from '../lib/locale.ts';

// ---------------------------------------------------------------- arithmetic behind the cells

/** Peak single-precision throughput in GFLOPS: one multiply-add is two operations. */
export const peakFromCores = (madPerClock: number, clockGHz: number) => madPerClock * 2 * clockGHz;

/** Peak interface rate of a double-data-rate memory bus, in GB/s. */
export const memoryBandwidth = (busBits: number, clockMHz: number) => (busBits * 2 * clockMHz) / 8 / 1000;

/** Line rate (GT/s) and encoding per PCI Express generation, as in Linux's PCIE_SPEED2MBS_ENC (facts A5). */
export const PCIE: Record<number, { gt: number; enc: [number, number] }> = {
  1: { gt: 2.5, enc: [8, 10] },
  2: { gt: 5, enc: [8, 10] },
  3: { gt: 8, enc: [128, 130] },
  4: { gt: 16, enc: [128, 130] },
  5: { gt: 32, enc: [128, 130] },
  6: { gt: 64, enc: [1, 1] },
};

/** Payload rate of a PCI Express link in GB/s, one direction. */
export function pcieRate(generation: number, lanes: number): number {
  const g = PCIE[generation];
  if (!g) throw new Error('unknown PCI Express generation ' + generation);
  return (g.gt * g.enc[0] * lanes) / g.enc[1] / 8;
}

// ---------------------------------------------------------------- the table as printed

export type RowId = 'fp32' | 'tensor' | 'membw' | 'memcap' | 'pcie' | 'nvlink';

export interface PrintedRow {
  id: RowId;
  /** The book's row label. The two compute rows share one. */
  label: string;
  /** Cells exactly as printed. null: the cell belongs to the row above (the book prints it once). */
  g80: string | null;
  b200: string;
  ratio: string;
  /** What the printed cells state as numbers, in `unit`. */
  from: number | null;
  to: number;
  unit: string;
  printedRatio: number | null;
}

/**
 * Fig. 24.1 as printed on p. 578. The book prints the two compute figures in one
 * cell ("80 TFLOPS / 2200 TFLOPS (FP32 / Tensor Core FP32)", "232x / 638x"); they
 * are two rows here so each can be checked alone.
 */
export const BOOK: { source: string; columns: [string, string, string]; rows: PrintedRow[] } = {
  source: 'Fig. 24.1, p. 578',
  get columns(): [string, string, string] {
    return ['G80 (2007)', 'B200 (2025)', tr('B200/G80 Ratio', 'Tỉ lệ B200/G80')];
  },
  rows: [
    {
      id: 'fp32',
      get label() {
        return tr('Compute Throughput', 'Thông lượng tính toán');
      },
      g80: '0.345 TFLOPS (FP32)', // i18n-ignore: the book's printed cell, a number and unit
      b200: '80 TFLOPS (FP32)', // i18n-ignore: the book's printed cell, a number and unit
      ratio: '232x', from: 0.345, to: 80, unit: 'TFLOPS', printedRatio: 232,
    },
    {
      id: 'tensor',
      get label() {
        return tr('Compute Throughput', 'Thông lượng tính toán');
      },
      g80: null,
      b200: '2200 TFLOPS (Tensor Core FP32)', // i18n-ignore: the book's printed cell, a number and the label the audit discusses
      ratio: '638x', from: 0.345, to: 2200, unit: 'TFLOPS', printedRatio: 638,
    },
    {
      id: 'membw',
      get label() {
        return tr('Memory Bandwidth', 'Băng thông bộ nhớ');
      },
      g80: '86.4 GB/s', // i18n-ignore: the book's printed cell, a number and unit
      b200: '8,000 GB/s', // i18n-ignore: the book's printed cell, a number and unit
      ratio: '93x', from: 86.4, to: 8000, unit: 'GB/s', printedRatio: 93,
    },
    {
      id: 'memcap',
      get label() {
        return tr('Memory Capacity', 'Dung lượng bộ nhớ');
      },
      g80: '1.5GB', b200: '192 GB', // i18n-ignore: the book's printed cells, numbers and units
      ratio: '128x', from: 1.5, to: 192, unit: 'GB', printedRatio: 128,
    },
    {
      id: 'pcie',
      get label() {
        return tr('PCIe Bandwidth', 'Băng thông PCIe');
      },
      get g80() {
        return tr('8 GB/s (Gen2 x16, each way)', '8 GB/s (Gen2 x16, mỗi chiều)');
      },
      get b200() {
        return tr('64 GB/s (Gen5 x16, each way)', '64 GB/s (Gen5 x16, mỗi chiều)');
      },
      ratio: '8x', from: 8, to: 64, unit: 'GB/s', printedRatio: 8,
    },
    // As printed the unit is TB/s, so the number the cell states is 1,800,000 GB/s.
    {
      id: 'nvlink',
      get label() {
        return tr('NVLink Bandwidth', 'Băng thông NVLink');
      },
      g80: 'N/A',
      b200: '1,800 TB/s', // i18n-ignore: the book's printed cell, a number and unit
      ratio: '-', from: null, to: 1_800_000, unit: 'GB/s', printedRatio: null,
    },
  ],
};

const bookRow = (id: RowId) => BOOK.rows.find((r) => r.id === id)!;

// ---------------------------------------------------------------- the same rows from NVIDIA's documents

export interface Figure {
  value: number;
  unit: string;
  /** How the vendor's documents give it. */
  how: string;
  /** Item in notes/today/ch24-facts.md. */
  fact: string;
}

/** One board on each side: the G80-based Tesla C870 and a B200 on an HGX B200 board. */
export const VENDOR = {
  boards: { g80: 'Tesla C870', b200: 'HGX B200' }, // i18n-ignore: product names
  g80: {
    madPerClock: 128,
    coreClockGHz: 1.35,
    busBits: 384,
    memClockMHz: 800,
    pcieGen: 1 as const,
    fp32: { value: peakFromCores(128, 1.35) / 1000, unit: 'TFLOPS',
      get how() {
        return tr('128 multiply-adds per clock × 2 × 1.35 GHz; NVIDIA prints the factors, not the product', '128 phép nhân-cộng mỗi chu kỳ × 2 × 1.35 GHz; NVIDIA in các thừa số, không in tích');
      },
      fact: 'A2',
    } as Figure,
    membw: { value: memoryBandwidth(384, 800), unit: 'GB/s',
      get how() {
        return tr('384-bit bus × 2 transfers per clock × 800 MHz / 8; NVIDIA prints no GB/s figure for this board', 'bus 384-bit × 2 lần truyền mỗi chu kỳ × 800 MHz / 8; NVIDIA không in con số GB/s nào cho bo mạch này');
      },
      fact: 'A3',
    } as Figure,
    memcap: { value: 1.5, unit: 'GB',
      get how() {
        return tr('1536 MB of GDDR3 in the board specification', '1536 MB GDDR3 trong thông số kỹ thuật của bo mạch');
      },
      fact: 'A4',
    } as Figure,
  },
  b200: {
    pcieGen: 5 as const,
    fp32: { value: 75, unit: 'TFLOPS',
      get how() {
        return tr('printed per GPU for HGX B200; also 600 TFLOPS for a board of 8', 'in cho mỗi GPU của HGX B200; cũng in 600 TFLOPS cho một bo mạch 8 GPU');
      },
      fact: 'A8',
    } as Figure,
    tensorSparse: { value: 2200, unit: 'TFLOPS',
      get how() {
        return tr('2.2 petaFLOPS, labelled TF32 Tensor Core, with sparsity', '2.2 petaFLOPS, ghi là TF32 Tensor Core, có sparsity');
      },
      fact: 'A9',
    } as Figure,
    tensorDense: { value: 1100, unit: 'TFLOPS',
      get how() {
        return tr('1.1 petaFLOPS, TF32 Tensor Core, dense: half the figure with sparsity', '1.1 petaFLOPS, TF32 Tensor Core, dense: một nửa con số có sparsity');
      },
      fact: 'A9',
    } as Figure,
    membw: { value: 8000, unit: 'GB/s',
      get how() {
        return tr('"up to 8 TB/s" in two documents, 7.7 TB/s in a third', '"up to 8 TB/s" trong hai tài liệu, 7.7 TB/s trong tài liệu thứ ba');
      },
      fact: 'A10',
    } as Figure,
    memcap: { value: 180, unit: 'GB',
      get how() {
        return tr('1,440 GB for 8 GPUs on shipped boards; "up to 192 GB" appears in one technical brief', '1,440 GB cho 8 GPU trên các bo mạch đã xuất xưởng; "up to 192 GB" xuất hiện trong một bản tóm tắt kỹ thuật');
      },
      fact: 'A11',
    } as Figure,
    nvlinkBoth: { value: 1800, unit: 'GB/s',
      get how() {
        return tr('18 links, 1.8 TB/s in total for both directions', '18 liên kết, tổng cộng 1.8 TB/s cho cả hai chiều');
      },
      fact: 'A13',
    } as Figure,
    nvlinkEachWay: { value: 900, unit: 'GB/s',
      get how() {
        return tr('900 GB/s in each direction', '900 GB/s mỗi chiều');
      },
      fact: 'A13',
    } as Figure,
  },
  /** The 8800 GTX, whose 384-bit bus at 900 MHz gives the book's 86.4 GB/s (768 MB card; facts A3). */
  gtx: { busBits: 384, memClockMHz: 900, fact: 'A3' },
};

// ---------------------------------------------------------------- ratios under each reading

export type FigureSet = 'book' | 'vendor';
export interface Options {
  /** The tensor-core figure: as printed (with sparsity) or NVIDIA's dense figure. */
  tensor: 'sparse' | 'dense';
  /** PCI Express generation of the 2007 board. The book prints 2; the C870's specification says 1. */
  g80Pcie: 1 | 2;
  /** NVLink counted the way NVIDIA quotes it (both directions) or like the PCIe row (each way). */
  nvlink: 'both' | 'eachWay';
}
export const DEFAULTS: Record<FigureSet, Options> = {
  book: { tensor: 'sparse', g80Pcie: 2, nvlink: 'both' },
  vendor: { tensor: 'sparse', g80Pcie: 1, nvlink: 'both' },
};

export interface Pair {
  /** 2007 figure, or null where the row has none (NVLink). */
  from: number | null;
  to: number;
  unit: string;
}

/** The two figures of a row under a figure set and the reader's options. */
export function figures(row: RowId, set: FigureSet, o: Options = DEFAULTS[set]): Pair {
  const b = bookRow(row);
  const v = VENDOR;
  switch (row) {
    case 'fp32':
      return set === 'book' ? { from: b.from, to: b.to, unit: b.unit } : { from: v.g80.fp32.value, to: v.b200.fp32.value, unit: 'TFLOPS' };
    case 'tensor': {
      const to = o.tensor === 'dense' ? v.b200.tensorDense.value : set === 'book' ? b.to : v.b200.tensorSparse.value;
      return { from: set === 'book' ? b.from : v.g80.fp32.value, to, unit: 'TFLOPS' };
    }
    case 'membw':
      return set === 'book' ? { from: b.from, to: b.to, unit: b.unit } : { from: v.g80.membw.value, to: v.b200.membw.value, unit: 'GB/s' };
    case 'memcap':
      return set === 'book' ? { from: b.from, to: b.to, unit: b.unit } : { from: v.g80.memcap.value, to: v.b200.memcap.value, unit: 'GB' };
    case 'pcie':
      // The book's printed 64 is the Gen5 x16 rate before encoding overhead; the rule gives 63.0.
      return { from: pcieRate(o.g80Pcie, 16), to: set === 'book' ? b.to : pcieRate(v.b200.pcieGen, 16), unit: 'GB/s' };
    case 'nvlink':
      // The printed unit is corrected here (facts A13); auditPrinted() reports it as printed.
      return { from: null, to: o.nvlink === 'both' ? v.b200.nvlinkBoth.value : v.b200.nvlinkEachWay.value, unit: 'GB/s' };
  }
}

/** Growth factor of a row, or null where the 2007 column has no figure. */
export function ratio(row: RowId, set: FigureSet, o: Options = DEFAULTS[set]): number | null {
  const f = figures(row, set, o);
  return f.from == null ? null : f.to / f.from;
}

/** NVLink in one direction over the Gen5 x16 host link in one direction. */
export const nvlinkOverHostLink = () => VENDOR.b200.nvlinkEachWay.value / pcieRate(VENDOR.b200.pcieGen, 16);

export const fmtInt = (v: number) => Math.round(v).toLocaleString('en-US');
export const fmtX = (v: number) => fmtInt(v) + 'x';

// ---------------------------------------------------------------- auditing the printed cells

export interface RatioAudit {
  row: RowId;
  printed: number;
  /** The quotient of the two printed inputs. */
  recomputed: number;
  matches: boolean;
  /** "2200 / 0.345 = 6,377" */
  arithmetic: string;
  /** For a ratio that does not match: the 2025 input that would make it right. */
  wouldMatch?: number;
}

/** Recompute every printed ratio from the two printed cells beside it. */
export function auditPrinted(): RatioAudit[] {
  return BOOK.rows
    .filter((r) => r.printedRatio != null && r.from != null)
    .map((r) => {
      const recomputed = r.to / r.from!;
      const matches = Math.round(recomputed) === r.printedRatio;
      return {
        row: r.id,
        printed: r.printedRatio!,
        recomputed,
        matches,
        arithmetic: `${r.to} / ${r.from} = ${fmtInt(recomputed)}`,
        ...(matches ? {} : { wouldMatch: Math.round(r.printedRatio! * r.from!) }),
      };
    });
}

export type Column = 'g80' | 'b200' | 'ratio';
/**
 * What the audit says about a cell:
 *   wrong       the table contradicts itself here (a ratio its own figures do not give, or a unit that cannot be meant);
 *   differs     the table's arithmetic does not catch it, but NVIDIA's documents give another figure or label;
 *   consistent  the table's arithmetic holds and no facts item corrects the cell.
 */
export type CellKind = 'wrong' | 'differs' | 'consistent';
export interface CellVerdict {
  /** "tensor.ratio" */
  id: string;
  row: RowId;
  col: Column;
  text: string;
  kind: CellKind;
  /** Facts items (notes/today/ch24-facts.md) behind a correction, for kind 'differs'. */
  fact?: string;
  /** The check that settles it. */
  arithmetic: string;
  /** What the check shows about this cell. */
  verdict: string;
}

/**
 * Cells the table's own arithmetic cannot catch but NVIDIA's documents correct, with the
 * facts items that say so. Ratios are not listed: each follows from its two printed
 * figures, and the explorer recomputes them on NVIDIA's figures.
 */
export const CORRECTIONS: Record<string, { fact: string; text: string }> = {
  'fp32.b200': {
    fact: 'A8',
    get text() {
      return tr('80 matches NVIDIA\'s GB200 NVL72 rack figure divided by 72 (5760 / 72). For a B200 on an HGX B200 board NVIDIA prints 75 TFLOPS.', '80 khớp với con số cho cả rack GB200 NVL72 của NVIDIA chia cho 72 (5760 / 72). Với một B200 trên bo mạch HGX B200, NVIDIA in 75 TFLOPS.');
    },
  },
  'tensor.b200': {
    fact: 'A9',
    get text() {
      return tr('NVIDIA prints 2.2 petaFLOPS for this row, so the figure stands and the ratio is the error. The label does not: NVIDIA lists the figure as TF32 on Tensor Cores, counted with sparsity, and dense is half.', 'NVIDIA in 2.2 petaFLOPS cho hàng này, nên con số đứng vững và tỉ lệ mới là chỗ sai. Nhãn thì không: NVIDIA ghi con số này là TF32 trên Tensor Core, tính với sparsity, và bản dense chỉ bằng một nửa.');
    },
  },
  'membw.g80': {
    fact: 'A3, A4',
    get text() {
      return tr('NVIDIA\'s pages we read do not print 86.4 GB/s. It matches the 768 MB GeForce 8800 GTX\'s 384-bit bus at 900 MHz; the 1.5 GB board in the row below gives 76.8 GB/s by the same formula.', 'Các trang của NVIDIA mà chúng tôi đã đọc không in 86.4 GB/s. Con số này khớp với bus 384-bit ở 900 MHz của GeForce 8800 GTX 768 MB; bo mạch 1.5 GB ở hàng dưới cho 76.8 GB/s theo cùng công thức.');
    },
  },
  'memcap.b200': {
    fact: 'A11',
    get text() {
      return tr('192 GB matches the "up to" of one NVIDIA technical brief. Shipped HGX and DGX B200 boards have 180 GB per GPU.', '192 GB khớp với chữ "up to" trong một bản tóm tắt kỹ thuật của NVIDIA. Các bo mạch HGX và DGX B200 đã xuất xưởng có 180 GB mỗi GPU.');
    },
  },
  'pcie.g80': {
    fact: 'A6',
    get text() {
      return tr('NVIDIA\'s board specification for the 1.5 GB Tesla C870 says PCI Express Generation 1, 4 GB/s each way. On that figure the host link grew about 16x.', 'Thông số kỹ thuật bo mạch của NVIDIA cho Tesla C870 1.5 GB ghi PCI Express Generation 1, 4 GB/s mỗi chiều. Theo con số đó, đường truyền tới host đã tăng khoảng 16x.');
    },
  },
};

/** The input cells NVIDIA's documents bear out, with the items that say so. */
export const CONFIRMATIONS: Record<string, { fact: string; text: string }> = {
  'fp32.g80': {
    fact: 'A2',
    get text() {
      return tr('NVIDIA prints the factors, not the product: 128 multiply-adds per clock × 2 × 1.35 GHz = 345.6 GFLOPS.', 'NVIDIA in các thừa số, không in tích: 128 phép nhân-cộng mỗi chu kỳ × 2 × 1.35 GHz = 345.6 GFLOPS.');
    },
  },
  'membw.b200': {
    fact: 'A10',
    get text() {
      return tr('NVIDIA prints "up to 8 TB/s" in two documents, and 7.7 TB/s in a third.', 'NVIDIA in "up to 8 TB/s" trong hai tài liệu, và 7.7 TB/s trong tài liệu thứ ba.');
    },
  },
  'memcap.g80': {
    fact: 'A4',
    get text() {
      return tr('NVIDIA\'s board specification for the Tesla C870 gives 1536 MB.', 'Thông số kỹ thuật bo mạch của NVIDIA cho Tesla C870 ghi 1536 MB.');
    },
  },
  'pcie.b200': {
    fact: 'A5, A12',
    get text() {
      return tr('Generation 5 x16 is NVIDIA\'s; 64 GB/s is that link\'s rate before encoding overhead, 63.0 after.', 'Generation 5 x16 là theo NVIDIA; 64 GB/s là tốc độ của liên kết đó trước chi phí mã hóa, 63.0 sau mã hóa.');
    },
  },
};

/** Every cell a reader can tick, with the check that settles it. */
export function printedCells(): CellVerdict[] {
  const audits = Object.fromEntries(auditPrinted().map((a) => [a.row, a]));
  const out: CellVerdict[] = [];
  for (const r of BOOK.rows) {
    const cols: [Column, string | null][] = [['g80', r.g80], ['b200', r.b200], ['ratio', r.ratio]];
    for (const [col, text] of cols) {
      if (text == null || text === 'N/A' || text === '-') continue;
      const id = `${r.id}.${col}`;
      const a = audits[r.id];
      if (r.id === 'nvlink') {
        const times = r.to / bookRow('membw').to;
        out.push({
          id, row: r.id, col, text, kind: 'wrong',
          arithmetic: `${fmtInt(r.to)} GB/s / ${fmtInt(bookRow('membw').to)} GB/s = ${fmtInt(times)}`,
          verdict: tr(
            `Every other bandwidth in the table is in GB/s. Read as printed, this link would carry ${fmtInt(times)} times the device's own memory bandwidth. NVIDIA's figure is ${fmtInt(VENDOR.b200.nvlinkBoth.value)} GB/s.`,
            `Mọi băng thông khác trong bảng đều tính bằng GB/s. Đọc đúng như in, liên kết này sẽ mang ${fmtInt(times)} lần băng thông bộ nhớ của chính device. Con số của NVIDIA là ${fmtInt(VENDOR.b200.nvlinkBoth.value)} GB/s.`,
          ),
        });
        continue;
      }
      if (!a.matches && col === 'ratio') {
        // The one ratio its inputs contradict.
        out.push({
          id, row: r.id, col, text, kind: 'wrong', arithmetic: a.arithmetic,
          verdict: tr(
            `Off by ten. ${a.printed}x would need ${a.wouldMatch} ${r.unit} in the 2025 column, and no NVIDIA document we read prints that.`,
            `Lệch mười lần. ${a.printed}x sẽ cần ${a.wouldMatch} ${r.unit} ở cột 2025, và không tài liệu NVIDIA nào chúng tôi đã đọc in con số đó.`,
          ),
        });
        continue;
      }
      const c = CORRECTIONS[id];
      if (c) {
        const lead = a.matches
          ? tr("The table's arithmetic holds.", 'Phép tính của bảng đúng.')
          : tr(
              'This figure and the ratio beside it cannot both be right, and the table alone does not say which.',
              'Con số này và tỉ lệ bên cạnh không thể cùng đúng, và riêng bảng không cho biết cái nào đúng.',
            );
        out.push({ id, row: r.id, col, text, kind: 'differs', fact: c.fact, arithmetic: a.arithmetic, verdict: `${lead} ${c.text}` });
        continue;
      }
      const ok = CONFIRMATIONS[id];
      out.push({
        id, row: r.id, col, text, kind: 'consistent', arithmetic: a.arithmetic,
        ...(ok ? { fact: ok.fact } : {}),
        verdict: ok
          ? tr(`The table's arithmetic holds. ${ok.text}`, `Phép tính của bảng đúng. ${ok.text}`)
          : tr('The printed ratio follows from the two printed figures.', 'Tỉ lệ được in suy ra đúng từ hai con số được in.'),
      });
    }
  }
  return out;
}

// ---------------------------------------------------------------- how each figure is built

export interface Built {
  when: '2007' | '2025';
  text: string;
  fact: string;
}
const f1 = (v: number) => v.toFixed(1);
const pcieText = (gen: number) => {
  const g = PCIE[gen];
  return tr(
    `Generation ${gen}: ${g.gt} GT/s × ${g.enc[0]}/${g.enc[1]} encoding × 16 lanes / 8 = ${f1(pcieRate(gen, 16))} GB/s each way`,
    `Thế hệ ${gen}: ${g.gt} GT/s × mã hóa ${g.enc[0]}/${g.enc[1]} × 16 lane / 8 = ${f1(pcieRate(gen, 16))} GB/s mỗi chiều`,
  );
};

/** Where the two figures of a row come from under the chosen reading. */
export function built(row: RowId, set: FigureSet, o: Options = DEFAULTS[set]): Built[] {
  const g = VENDOR.g80;
  const b = VENDOR.b200;
  const peak = tr(
    `${g.madPerClock} multiply-adds per clock × 2 × ${g.coreClockGHz} GHz = ${f1(peakFromCores(g.madPerClock, g.coreClockGHz))} GFLOPS`,
    `${g.madPerClock} phép nhân-cộng mỗi chu kỳ × 2 × ${g.coreClockGHz} GHz = ${f1(peakFromCores(g.madPerClock, g.coreClockGHz))} GFLOPS`,
  );
  switch (row) {
    case 'fp32':
      return [
        { when: '2007', text: tr(`${peak}. NVIDIA prints the two factors, not the product, so this is a theoretical peak.`, `${peak}. NVIDIA in hai thừa số, không in tích, nên đây là đỉnh lý thuyết.`), fact: 'A2' },
        set === 'book'
          ? { when: '2025', text: tr('Matches NVIDIA\'s rack figure for GB200 NVL72, 5,760 TFLOPS for 72 GPUs: 5760 / 72 = 80 TFLOPS.', 'Khớp với con số cho cả rack GB200 NVL72 của NVIDIA, 5,760 TFLOPS cho 72 GPU: 5760 / 72 = 80 TFLOPS.'), fact: 'A8' }
          : { when: '2025', text: tr('Printed per GPU for HGX B200, and as 600 TFLOPS for a board of 8: 600 / 8 = 75 TFLOPS.', 'In cho mỗi GPU của HGX B200, và là 600 TFLOPS cho một bo mạch 8 GPU: 600 / 8 = 75 TFLOPS.'), fact: 'A8' },
      ];
    case 'tensor':
      return [
        { when: '2007', text: tr(`${peak}. The book divides by the same FP32 figure as in the row above.`, `${peak}. Sách chia cho cùng con số FP32 như ở hàng trên.`), fact: 'A2' },
        o.tensor === 'dense'
          ? { when: '2025', text: tr('NVIDIA: 1.1 petaFLOPS, TF32 on Tensor Cores, dense. Half the figure with sparsity.', 'NVIDIA: 1.1 petaFLOPS, TF32 trên Tensor Core, dense. Bằng một nửa con số có sparsity.'), fact: 'A9' }
          : { when: '2025', text: tr('NVIDIA: 2.2 petaFLOPS, TF32 on Tensor Cores, with sparsity. "Tensor Core FP32" is not a precision NVIDIA lists.', 'NVIDIA: 2.2 petaFLOPS, TF32 trên Tensor Core, có sparsity. "Tensor Core FP32" không phải là một độ chính xác mà NVIDIA liệt kê.'), fact: 'A9' },
      ];
    case 'membw': {
      const clock = set === 'book' ? VENDOR.gtx.memClockMHz : g.memClockMHz;
      const board = set === 'book' ? tr('the 768 MB GeForce 8800 GTX', 'GeForce 8800 GTX 768 MB') : tr(`the 1.5 GB ${VENDOR.boards.g80}`, `${VENDOR.boards.g80} 1.5 GB`);
      return [
        { when: '2007', text: tr(`${g.busBits} bits × 2 transfers per clock × ${clock} MHz / 8 = ${f1(memoryBandwidth(g.busBits, clock))} GB/s, the memory clock of ${board}.`, `${g.busBits} bit × 2 lần truyền mỗi chu kỳ × ${clock} MHz / 8 = ${f1(memoryBandwidth(g.busBits, clock))} GB/s, xung nhịp bộ nhớ của ${board}.`), fact: 'A3' },
        { when: '2025', text: tr('NVIDIA: "up to 8 TB/s" in two documents and 7.7 TB/s in a third.', 'NVIDIA: "up to 8 TB/s" trong hai tài liệu và 7.7 TB/s trong tài liệu thứ ba.'), fact: 'A10' },
      ];
    }
    case 'memcap':
      return [
        { when: '2007', text: tr(`1536 MB on the ${VENDOR.boards.g80}. The GeForce 8800 GTX, where the book's bandwidth figure comes from, had 768 MB.`, `1536 MB trên ${VENDOR.boards.g80}. GeForce 8800 GTX, nơi con số băng thông của sách bắt nguồn, có 768 MB.`), fact: 'A4' },
        set === 'book'
          ? { when: '2025', text: tr('"Up to 192 GB" in one NVIDIA technical brief.', '"Up to 192 GB" trong một bản tóm tắt kỹ thuật của NVIDIA.'), fact: 'A11' }
          : { when: '2025', text: tr('Shipped boards: 1,440 GB for 8 GPUs, 1440 / 8 = 180 GB.', 'Các bo mạch đã xuất xưởng: 1,440 GB cho 8 GPU, 1440 / 8 = 180 GB.'), fact: 'A11' },
      ];
    case 'pcie':
      return [
        {
          when: '2007',
          text: `${pcieText(o.g80Pcie)}. ${o.g80Pcie === 1 ? tr(`NVIDIA's board specification for the ${VENDOR.boards.g80} says Generation 1.`, `Thông số kỹ thuật bo mạch của NVIDIA cho ${VENDOR.boards.g80} ghi Generation 1.`) : tr('The book prints Gen2; no NVIDIA page found states the generation of the GeForce 8800 GTX.', 'Sách in Gen2; không tìm thấy trang NVIDIA nào nêu thế hệ PCIe của GeForce 8800 GTX.')}`,
          fact: 'A6',
        },
        { when: '2025', text: `${pcieText(b.pcieGen)}${set === 'book' ? tr('; the book prints 64, the rate before encoding overhead', '; sách in 64, tốc độ trước chi phí mã hóa') : ''}.`, fact: 'A12' },
      ];
    case 'nvlink':
      return [
        { when: '2007', text: tr('No counterpart: the row is N/A in the 2007 column.', 'Không có đối ứng: hàng này là N/A ở cột 2007.'), fact: 'A13' },
        {
          when: '2025',
          text: tr(
            `NVIDIA: ${b.nvlinkBoth.how}, ${b.nvlinkEachWay.how}. Each way that is ${f1(nvlinkOverHostLink())} times a Gen5 x16 host link (${fmtInt(b.nvlinkEachWay.value)} / ${f1(pcieRate(b.pcieGen, 16))}).`,
            `NVIDIA: ${b.nvlinkBoth.how}, ${b.nvlinkEachWay.how}. Mỗi chiều, con số đó gấp ${f1(nvlinkOverHostLink())} lần một đường truyền tới host Gen5 x16 (${fmtInt(b.nvlinkEachWay.value)} / ${f1(pcieRate(b.pcieGen, 16))}).`,
          ),
          fact: 'A13',
        },
      ];
  }
}

// ---------------------------------------------------------------- the threshold of section 5.1

/** FLOP/B at which peak throughput and peak bandwidth meet (ch. 5 section 5.1, p. 94). */
export const threshold = (peakGflops: number, bandwidthGBs: number) => peakGflops / bandwidthGBs;

const EPS = 1e-9;
/** A kernel at or above the threshold is compute-bound; below it, memory-bound. */
export const regime = (flopPerByte: number, thr: number): 'memory-bound' | 'compute-bound' =>
  flopPerByte >= thr * (1 - EPS) ? 'compute-bound' : 'memory-bound';

/** FLOP/B of the square-tiled matmul with one thread per output element (ch. 5, pp. 105 and 111). */
export const tiledRatio = (tile: number) => 0.25 * tile;

/** Smallest tile width T at which 0.25 × T reaches the threshold. */
export const tileFor = (thr: number) => Math.max(1, Math.ceil((thr * (1 - EPS)) / 0.25));

/** FLOP/B of an m × n output tile (ch. 15 section 15.2, p. 351): 0.5 · m · n / (m + n). */
export const outputTileRatio = (m: number, n: number) => (0.5 * m * n) / (m + n);

/** FLOP/B of a linear layer with every weight, input and output touched once (ch. 20 section 20.6, p. 505). */
export const linearLayerRatio = (batch: number, inputs: number, outputs: number, bytesPerValue: number) =>
  (2 * batch * inputs * outputs) / ((inputs * outputs + batch * (inputs + outputs)) * bytesPerValue);

export interface Device {
  key: string;
  label: string;
  /** Peak throughput, GFLOP/s. */
  peak: number;
  /** Peak memory bandwidth, GB/s. */
  bw: number;
  note: string;
}
const col = (id: RowId) => bookRow(id);
/**
 * The published figures the strip offers: the book's two columns, and NVIDIA's dense
 * TF32 figure as the one alternative to the printed tensor-core peak. Other devices,
 * the reader's own among them, are chapter 5's roofline (§5.1), not this chapter's.
 */
export const DEVICES: Device[] = [
  {
    key: 'book2007',
    get label() {
      return tr('The book\'s table, 2007 column', 'Bảng của sách, cột 2007');
    },
    peak: col('fp32').from! * 1000,
    bw: col('membw').from!,
    get note() {
      return tr('Fig. 24.1, p. 578: 0.345 TFLOPS over 86.4 GB/s.', 'Fig. 24.1, p. 578: 0.345 TFLOPS trên 86.4 GB/s.');
    },
  },
  {
    key: 'book2025',
    get label() {
      return tr('The book\'s table, 2025 column, FP32', 'Bảng của sách, cột 2025, FP32');
    },
    peak: col('fp32').to * 1000,
    bw: col('membw').to,
    get note() {
      return tr('Fig. 24.1, p. 578: 80 TFLOPS over 8,000 GB/s.', 'Fig. 24.1, p. 578: 80 TFLOPS trên 8,000 GB/s.');
    },
  },
  {
    key: 'book2025tc',
    get label() {
      return tr('The book\'s table, 2025 column, tensor-core figure', 'Bảng của sách, cột 2025, con số tensor core');
    },
    peak: col('tensor').to * 1000,
    bw: col('membw').to,
    get note() {
      return tr('Fig. 24.1, p. 578: 2200 TFLOPS over 8,000 GB/s. The compute figure is TF32 with sparsity, and applies only to matmul-shaped work.', 'Fig. 24.1, p. 578: 2200 TFLOPS trên 8,000 GB/s. Con số tính toán là TF32 có sparsity, và chỉ áp dụng cho công việc có dạng phép nhân ma trận.');
    },
  },
  {
    key: 'book2025dense',
    get label() {
      return tr('2025 column, dense TF32 from NVIDIA\'s documents', 'Cột 2025, TF32 dense theo tài liệu của NVIDIA');
    },
    peak: VENDOR.b200.tensorDense.value * 1000,
    bw: col('membw').to,
    get note() {
      return tr('NVIDIA\'s dense TF32 figure, 1.1 petaFLOPS, over the book\'s 8,000 GB/s. It applies only to matmul-shaped work.', 'Con số TF32 dense của NVIDIA, 1.1 petaFLOPS, trên 8,000 GB/s của sách. Nó chỉ áp dụng cho công việc có dạng phép nhân ma trận.');
    },
  },
];

export interface Mark {
  id: string;
  label: string;
  ratio: number;
  source: string;
  /** Whether a wider tile moves it. */
  tiled?: boolean;
}
/** Kernels from earlier chapters on the FLOP/B axis. Only the tiled matmul moves with T. */
export function marks(tile: number): Mark[] {
  return [
    { id: 'vecadd', label: tr('vector add', 'cộng vector'), ratio: 1 / 12, source: 'ch. 5 §5.1, p. 96' },
    { id: 'naive', label: tr('naive matmul', 'phép nhân ma trận ngây thơ'), ratio: 0.25, source: 'ch. 5 §5.1, p. 97' },
    { id: 'tiled', label: tr(`tiled matmul, T = ${tile}`, `phép nhân ma trận phân ô, T = ${tile}`), ratio: tiledRatio(tile), source: 'ch. 5 §5.3, p. 105 and §5.4, p. 111', tiled: true },
    { id: 'lin1', label: tr('linear layer, 16-bit, batch 1', 'lớp tuyến tính, 16-bit, batch 1'), ratio: linearLayerRatio(1, 1024, 4096, 2), source: 'ch. 20 §20.6, p. 505' },
    { id: 'out128', label: tr('128 × 128 output tile', 'ô đầu ra 128 × 128'), ratio: outputTileRatio(128, 128), source: 'ch. 15 §15.2, p. 351' },
    { id: 'lin512', label: tr('linear layer, 16-bit, batch 512', 'lớp tuyến tính, 16-bit, batch 512'), ratio: linearLayerRatio(512, 1024, 4096, 2), source: 'ch. 20 §20.6, p. 505' },
  ];
}

// ---------------------------------------------------------------- what limits what

export type LimitId = 'compute' | 'membw' | 'memcap' | 'hostlink' | 'gpulink';
/** The five rows of the table as limits, in the table's order. */
export const LIMIT_ROWS: { id: LimitId; name: string; bookRow: string }[] = [
  {
    id: 'compute',
    get name() {
      return tr('compute throughput', 'thông lượng tính toán');
    },
    get bookRow() {
      return tr('Compute Throughput', 'Thông lượng tính toán');
    },
  },
  {
    id: 'membw',
    get name() {
      return tr('memory bandwidth', 'băng thông bộ nhớ');
    },
    get bookRow() {
      return tr('Memory Bandwidth', 'Băng thông bộ nhớ');
    },
  },
  {
    id: 'memcap',
    get name() {
      return tr('memory capacity', 'dung lượng bộ nhớ');
    },
    get bookRow() {
      return tr('Memory Capacity', 'Dung lượng bộ nhớ');
    },
  },
  {
    id: 'hostlink',
    get name() {
      return tr('host link', 'đường truyền tới host');
    },
    get bookRow() {
      return tr('PCIe Bandwidth', 'Băng thông PCIe');
    },
  },
  {
    id: 'gpulink',
    get name() {
      return tr('GPU-to-GPU link', 'liên kết giữa các GPU');
    },
    get bookRow() {
      return tr('NVLink Bandwidth', 'Băng thông NVLink');
    },
  },
];

export interface LimitCard {
  /** Question id in the progress file. */
  qid: string;
  /** The earlier chapters the card draws on, as the prerequisite check labels it. */
  from: string;
  card: string;
  row: LimitId;
  why: string;
  source: string;
}
export const LIMITS: LimitCard[] = [
  {
    qid: 'ch24/start/q-limit-load',
    get from() {
      return tr('From chapter 2', 'Từ chương 2');
    },
    get card() {
      return tr('Copying a model\'s weights from host memory to device memory with cudaMemcpy.', 'Sao chép trọng số của một mô hình từ bộ nhớ host sang bộ nhớ device bằng cudaMemcpy.');
    },
    row: 'hostlink',
    get why() {
      return tr('A host-to-device copy has to cross the link between the two memories, and it cannot go faster than that link.', 'Một lần sao chép từ host sang device phải đi qua đường truyền giữa hai bộ nhớ, và không thể nhanh hơn đường truyền đó.');
    },
    get source() {
      return tr('ch. 2 §2.4, pp. 28–32, for the copy; that the link is PCIe is the table\'s own fourth row', 'ch. 2 §2.4, pp. 28–32, cho lần sao chép; việc đường truyền là PCIe là hàng thứ tư của chính bảng');
    },
  },
  {
    qid: 'ch24/start/q-limit-decode',
    get from() {
      return tr('From chapters 5 and 20', 'Từ chương 5 và 20');
    },
    get card() {
      return tr('Generating one token at batch 1 with a KV cache: every weight is read once for a matrix-vector product.', 'Sinh một token ở batch 1 với KV cache: mỗi trọng số được đọc một lần cho một phép nhân ma trận với vector.');
    },
    row: 'membw',
    get why() {
      return tr('Each weight is fetched once and used once, so the ratio stays near 1 FLOP/B and the kernel waits on memory, which is what the book says of generation.', 'Mỗi trọng số được nạp một lần và dùng một lần, nên tỉ lệ ở gần 1 FLOP/B và kernel phải chờ bộ nhớ, đúng như điều sách nói về quá trình sinh token.');
    },
    source: 'ch. 5 §5.1, pp. 93–97; ch. 20 §20.4, pp. 491–492',
  },
  {
    qid: 'ch24/start/q-limit-prefill',
    get from() {
      return tr('From chapter 20', 'Từ chương 20');
    },
    get card() {
      return tr('Prefill of a long prompt: one pass of large matrix-matrix products.', 'Prefill một prompt dài: một lượt các phép nhân ma trận với ma trận cỡ lớn.');
    },
    row: 'compute',
    get why() {
      return tr('Matrix-matrix products reuse every operand many times, so the ratio is high and the arithmetic units set the pace: the book calls prefill typically compute-bound.', 'Phép nhân ma trận với ma trận dùng lại mỗi toán hạng nhiều lần, nên tỉ lệ cao và các đơn vị tính toán quyết định tốc độ: sách gọi prefill thường là bị giới hạn bởi tính toán.');
    },
    source: 'ch. 20 §20.4, pp. 491–492',
  },
  {
    qid: 'ch24/start/q-limit-fit',
    get from() {
      return tr('From chapter 20', 'Từ chương 20');
    },
    get card() {
      return tr('Whether the weights plus the KV cache of every live sequence fit on the device.', 'Trọng số cộng với KV cache của mọi chuỗi đang chạy có vừa trên device hay không.');
    },
    row: 'memcap',
    get why() {
      return tr('Each sequence in a batch needs its own KV cache, so the batch you can serve is capped by bytes of device memory, not by any rate.', 'Mỗi chuỗi trong một batch cần KV cache riêng, nên batch bạn có thể phục vụ bị giới hạn bởi số byte bộ nhớ device, không phải bởi tốc độ nào.');
    },
    source: 'ch. 20 §20.6, pp. 505–506',
  },
  {
    qid: 'ch24/start/q-limit-allreduce',
    get from() {
      return tr('From chapter 23', 'Từ chương 23');
    },
    get card() {
      return tr('An all-reduce through NCCL among the GPUs of one node, which are joined to each other by NVLink.', 'Một all-reduce qua NCCL giữa các GPU của một node, vốn được nối với nhau bằng NVLink.');
    },
    row: 'gpulink',
    get why() {
      return tr('The collective moves each GPU\'s data to the others over what joins them, here NVLink. The book calls NCCL topology-aware and lists PCIe and NVLINK among the interconnects it uses (p. 564).', 'Phép toán tập thể chuyển dữ liệu của mỗi GPU sang các GPU khác qua thứ nối chúng với nhau, ở đây là NVLink. Sách gọi NCCL là nhận biết topology và liệt kê PCIe và NVLINK trong số các kết nối nó dùng (p. 564).');
    },
    get source() {
      return tr('ch. 23 §23.4, pp. 563–568, for NCCL; NCCL\'s documentation for all-reduce (notes/today/ch24-facts.md, A16)', 'ch. 23 §23.4, pp. 563–568, cho NCCL; tài liệu của NCCL về all-reduce (notes/today/ch24-facts.md, A16)');
    },
  },
];
