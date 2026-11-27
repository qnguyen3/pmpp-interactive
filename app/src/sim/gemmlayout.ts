// One product, three spellings (book §15.1, p. 350, plus vendor facts).
//
// Maps operations a framework performs onto the book's C = A·B (A is M x K, B is K x N,
// C is M x N, everything row-major in the book), onto a column-major BLAS call
// (cublasSgemm / hipblasSgemm: ch03-facts A7, B6; ch15-facts A10, B6) and onto
// MPSMatrixMultiplication, which is row-major and takes transpose flags (ch03-facts C7,
// ch15-facts C6). It also executes both calls on small buffers, reading memory the way
// each library is documented to read it, so a wrong argument shows as a wrong cell.
//
// The row-major trick (ch15-facts A10, our derivation of the standard idiom, as llama.cpp
// does it in ch03-facts D2): a row-major M x N array read column-major is the N x M
// transpose, so C = A·B is computed as Cᵀ = Bᵀ·Aᵀ with B passed first.

import { tr } from '../lib/locale.ts';

export type OpId = 'book' | 'linear' | 'xw' | 'qkt' | 'pv';
export type Role = 'A' | 'B' | 'C';
export type Trans = 'N' | 'T';

export interface Operation {
  id: OpId;
  /** Short name for a selector. */
  label: string;
  /** The operation as the framework writes it. */
  formula: string;
  /** What the book's M, N, K are called here. */
  dims: { M: string; N: string; K: string };
  /** What the book's A, B, C are called here. */
  names: Record<Role, string>;
  /** B is held as an N x K row-major array (the book's B stored column-major), not K x N. */
  bStoredNK: boolean;
  /** One sentence on where the operation comes from. */
  note: string;
}

export const OPERATIONS: readonly Operation[] = [
  {
    id: 'book',
    get label() {
      return tr("The book's C = A × B", 'C = A × B của sách');
    },
    formula: 'C = A × B',
    dims: { M: 'M', N: 'N', K: 'K' }, names: { A: 'A', B: 'B', C: 'C' }, bStoredNK: false,
    get note() {
      return tr('Both inputs row-major, as the chapter assumes (p. 350).', 'Cả hai đầu vào theo thứ tự hàng, như chương này giả định (p. 350).');
    },
  },
  {
    id: 'linear',
    get label() {
      return tr('nn.Linear, W stored (out, in)', 'nn.Linear, W lưu dạng (out, in)');
    },
    formula: 'y = x Wᵀ',
    dims: { M: 'batch', N: 'out', K: 'in' }, names: { A: 'x', B: 'W', C: 'y' }, bStoredNK: true,
    get note() {
      return tr("PyTorch's weight has shape (out_features, in_features); ggml stores it the same way (ch03-facts D2, D6).", 'Trọng số của PyTorch có hình dạng (out_features, in_features); ggml lưu nó theo cùng cách (ch03-facts D2, D6).');
    },
  },
  {
    id: 'xw',
    get label() {
      return tr('x @ W, W stored (in, out)', 'x @ W, W lưu dạng (in, out)');
    },
    formula: 'y = x W',
    dims: { M: 'batch', N: 'out', K: 'in' }, names: { A: 'x', B: 'W', C: 'y' }, bStoredNK: false,
    get note() {
      return tr('A weight kept with one input feature per row: exactly the book\'s layout.', 'Ma trận trọng số được lưu mỗi hàng một đặc trưng đầu vào: đúng cách bố trí của sách.');
    },
  },
  {
    id: 'qkt',
    get label() {
      return tr('Attention scores Q Kᵀ', 'Điểm attention Q Kᵀ');
    },
    formula: 'S = Q Kᵀ',
    dims: { M: 'n_q', N: 'n_kv', K: 'd' }, names: { A: 'Q', B: 'K', C: 'S' }, bStoredNK: true,
    get note() {
      return tr('Q and K both hold one token per row, d values long.', 'Q và K đều giữ mỗi hàng một token, dài d giá trị.');
    },
  },
  {
    id: 'pv',
    get label() {
      return tr('Attention output P V', 'Đầu ra attention P V');
    },
    formula: 'O = P V',
    dims: { M: 'n_q', N: 'd', K: 'n_kv' }, names: { A: 'P', B: 'V', C: 'O' }, bStoredNK: false,
    get note() {
      return tr('P has one row of weights per query; V one row per key.', 'P có mỗi truy vấn một hàng trọng số; V mỗi key một hàng.');
    },
  },
];

export const operation = (id: OpId): Operation => {
  const op = OPERATIONS.find((o) => o.id === id);
  if (!op) throw new Error('unknown operation ' + id);
  return op;
};

export interface Shape {
  M: number;
  N: number;
  K: number;
}

/** How an operand sits in memory: a row-major array of rows x cols, rows `ld` floats apart. */
export interface Stored {
  role: Role;
  rows: number;
  cols: number;
  ld: number;
}

export interface Mapping {
  op: Operation;
  shape: Shape;
  A: Stored;
  B: Stored;
  C: Stored;
}

export function mapOperation(id: OpId, shape: Shape): Mapping {
  const op = operation(id);
  const { M, N, K } = shape;
  for (const [v, w] of [[M, 'M'], [N, 'N'], [K, 'K']] as const)
    if (!(Number.isInteger(v) && v > 0)) throw new Error(`${w} must be a positive integer`);
  return {
    op,
    shape,
    A: { role: 'A', rows: M, cols: K, ld: K },
    B: op.bStoredNK ? { role: 'B', rows: N, cols: K, ld: K } : { role: 'B', rows: K, cols: N, ld: N },
    C: { role: 'C', rows: M, cols: N, ld: N },
  };
}

// ---------------------------------------------------------------- column-major BLAS
/** The arguments of cublasSgemm / hipblasSgemm after the handle (ch15-facts A10, B6). */
export interface BlasCall {
  transa: Trans;
  transb: Trans;
  m: number;
  n: number;
  k: number;
  /** Which of the book's operands is passed in the library's "A" position, and its leading dimension. */
  a: Role;
  lda: number;
  b: Role;
  ldb: number;
  ldc: number;
  /** The same arguments with the operation's own names in place of numbers. */
  symbolic: { m: string; n: string; k: string; a: string; lda: string; b: string; ldb: string; c: string; ldc: string };
}

/**
 * The call that computes the row-major product: Cᵀ (N x M, column-major, ld N) = Bᵀ·Aᵀ.
 * B goes first. A row-major K x N B read column-major with ld N is already Bᵀ (no
 * transpose); an N x K row-major B read with ld K is B itself, so it is transposed.
 */
export function blasCall(map: Mapping): BlasCall {
  const { op, shape } = map;
  const d = op.dims;
  const tB = op.bStoredNK;
  return {
    transa: tB ? 'T' : 'N',
    transb: 'N',
    m: shape.N,
    n: shape.M,
    k: shape.K,
    a: 'B',
    lda: map.B.ld,
    b: 'A',
    ldb: map.A.ld,
    ldc: map.C.ld,
    symbolic: {
      m: d.N, n: d.M, k: d.K,
      a: op.names.B, lda: tB ? d.K : d.N,
      b: op.names.A, ldb: d.K,
      c: op.names.C, ldc: d.N,
    },
  };
}

export type BlasEdit = 'swap' | 'flipTransa' | 'mnBook';
export const BLAS_EDITS: Record<BlasEdit, string> = {
  get swap() {
    return tr('swap the two input pointers', 'đổi chỗ hai con trỏ đầu vào');
  },
  get flipTransa() {
    return tr('flip transa', 'lật transa');
  },
  get mnBook() {
    return tr('pass m, n in the book\'s order (M, N)', 'truyền m, n theo thứ tự của sách (M, N)');
  },
};

/** The call with the reader's edits applied. Leading dimensions stay as they were. */
export function editBlas(call: BlasCall, edits: Partial<Record<BlasEdit, boolean>>): BlasCall {
  let c = { ...call, symbolic: { ...call.symbolic } };
  if (edits.swap) c = { ...c, a: c.b, b: c.a, symbolic: { ...c.symbolic, a: c.symbolic.b, b: c.symbolic.a } };
  if (edits.flipTransa) c = { ...c, transa: c.transa === 'N' ? 'T' : 'N' };
  if (edits.mnBook) c = { ...c, m: c.n, n: c.m, symbolic: { ...c.symbolic, m: c.symbolic.n, n: c.symbolic.m } };
  return c;
}

export type Buffers = Record<Role, number[]>;

export interface RunResult {
  /** The output buffer after the call. */
  C: number[];
  /** The call read past the end of an input buffer (those reads count as NaN). */
  readOutside: boolean;
  /** The call tried to write past the end of the output buffer (those writes are dropped). */
  writeOutside: boolean;
}

const read = (buf: number[], i: number, flag: { out: boolean }) => {
  if (i < 0 || i >= buf.length) {
    flag.out = true;
    return NaN;
  }
  return buf[i];
};

/**
 * Executes a column-major GEMM the way BLAS defines it: C = alpha·op(A)·op(B) + beta·C,
 * where the library's A (m x k after op) and B (k x n after op) are read column-major
 * from the buffers named in the call. alpha = 1, beta = 0. No argument validation.
 */
export function runBlas(call: BlasCall, bufs: Buffers): RunResult {
  const first = bufs[call.a];
  const second = bufs[call.b];
  const C = bufs.C.slice();
  const r = { out: false };
  let writeOutside = false;
  for (let j = 0; j < call.n; j++)
    for (let i = 0; i < call.m; i++) {
      let s = 0;
      for (let l = 0; l < call.k; l++) {
        const a = call.transa === 'N' ? read(first, i + l * call.lda, r) : read(first, l + i * call.lda, r);
        const b = call.transb === 'N' ? read(second, l + j * call.ldb, r) : read(second, j + l * call.ldb, r);
        s += a * b;
      }
      const at = i + j * call.ldc;
      if (at < 0 || at >= C.length) writeOutside = true;
      else C[at] = s;
    }
  return { C, readOutside: r.out, writeOutside };
}

// ---------------------------------------------------------------- MPS (row-major)
export interface MpsMatrix {
  role: Role;
  rows: number;
  columns: number;
  /** Bytes between rows; a multiple of the element size (ch15-facts C6). */
  rowBytes: number;
}
/** The initialiser's arguments and the three matrix descriptors (ch15-facts C6). */
export interface MpsCall {
  transposeLeft: boolean;
  transposeRight: boolean;
  resultRows: number;
  resultColumns: number;
  interiorColumns: number;
  left: MpsMatrix;
  right: MpsMatrix;
  result: MpsMatrix;
  symbolic: { resultRows: string; resultColumns: string; interiorColumns: string };
}

const desc = (s: Stored): MpsMatrix => ({ role: s.role, rows: s.rows, columns: s.cols, rowBytes: s.ld * 4 });

/** MPSMatrixMultiplication for the operation: left = A, right = B, transposed when stored N x K. */
export function mpsCall(map: Mapping): MpsCall {
  const d = map.op.dims;
  return {
    transposeLeft: false,
    transposeRight: map.op.bStoredNK,
    resultRows: map.shape.M,
    resultColumns: map.shape.N,
    interiorColumns: map.shape.K,
    left: desc(map.A),
    right: desc(map.B),
    result: desc(map.C),
    symbolic: { resultRows: d.M, resultColumns: d.N, interiorColumns: d.K },
  };
}

export const editMps = (call: MpsCall, flipRight: boolean): MpsCall =>
  flipRight ? { ...call, transposeRight: !call.transposeRight } : call;

/** Executes result = op(left)·op(right) with every matrix read row-major through its rowBytes. */
export function runMps(call: MpsCall, bufs: Buffers): RunResult {
  const L = bufs[call.left.role];
  const R = bufs[call.right.role];
  const ldl = call.left.rowBytes / 4;
  const ldr = call.right.rowBytes / 4;
  const ldo = call.result.rowBytes / 4;
  const C = bufs[call.result.role].slice();
  const r = { out: false };
  let writeOutside = false;
  for (let row = 0; row < call.resultRows; row++)
    for (let col = 0; col < call.resultColumns; col++) {
      let s = 0;
      for (let l = 0; l < call.interiorColumns; l++) {
        const a = call.transposeLeft ? read(L, l * ldl + row, r) : read(L, row * ldl + l, r);
        const b = call.transposeRight ? read(R, col * ldr + l, r) : read(R, l * ldr + col, r);
        s += a * b;
      }
      const at = row * ldo + col;
      if (at >= C.length) writeOutside = true;
      else C[at] = s;
    }
  return { C, readOutside: r.out, writeOutside };
}

// ---------------------------------------------------------------- data and the reference
/** Small integers, deterministic per role and position, so cells can be checked by eye. */
export function exampleBuffers(map: Mapping): Buffers {
  const fill = (s: Stored, mul: number, add: number) =>
    Array.from({ length: s.rows * s.ld }, (_, i) => ((i * mul + add) % 7) - 3);
  return { A: fill(map.A, 5, 2), B: fill(map.B, 3, 1), C: new Array<number>(map.C.rows * map.C.ld).fill(0) };
}

/** Seeded uniform values in [-1, 1), for the tests. */
export function randomBuffers(map: Mapping, seed: number): Buffers {
  let s = seed >>> 0 || 1;
  const rnd = () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 2 ** 32 - 0.5;
  };
  const fill = (st: Stored) => Array.from({ length: st.rows * st.ld }, () => 2 * rnd());
  return { A: fill(map.A), B: fill(map.B), C: new Array<number>(map.C.rows * map.C.ld).fill(0) };
}

/**
 * The row-major product as the framework defines each operation, written out per
 * operation from its own formula (not from the mapping above), so the tests compare two
 * independent derivations. Returns C as an M x N row-major array.
 */
export function reference(map: Mapping, bufs: Buffers): number[] {
  const { M, N, K } = map.shape;
  const { A, B } = bufs;
  const out = new Array<number>(M * N).fill(0);
  const at = (r: number, c: number, v: number) => (out[r * N + c] = v);
  switch (map.op.id) {
    case 'book': // C[i][j] = Σ_l A[i][l] · B[l][j], B is K x N
      for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) {
        let s = 0;
        for (let l = 0; l < K; l++) s += A[i * K + l] * B[l * N + j];
        at(i, j, s);
      }
      break;
    case 'linear': { // y[b][o] = Σ_i x[b][i] · W[o][i], W is out x in
      const [batch, outF, inF] = [M, N, K];
      for (let b = 0; b < batch; b++) for (let o = 0; o < outF; o++) {
        let s = 0;
        for (let i = 0; i < inF; i++) s += A[b * inF + i] * B[o * inF + i];
        at(b, o, s);
      }
      break;
    }
    case 'xw': { // y[b][o] = Σ_i x[b][i] · W[i][o], W is in x out
      const [batch, outF, inF] = [M, N, K];
      for (let b = 0; b < batch; b++) for (let o = 0; o < outF; o++) {
        let s = 0;
        for (let i = 0; i < inF; i++) s += A[b * inF + i] * B[i * outF + o];
        at(b, o, s);
      }
      break;
    }
    case 'qkt': { // S[q][t] = Σ_e Q[q][e] · K[t][e], K is n_kv x d
      const [nq, nkv, d] = [M, N, K];
      for (let q = 0; q < nq; q++) for (let t = 0; t < nkv; t++) {
        let s = 0;
        for (let e = 0; e < d; e++) s += A[q * d + e] * B[t * d + e];
        at(q, t, s);
      }
      break;
    }
    case 'pv': { // O[q][e] = Σ_t P[q][t] · V[t][e], V is n_kv x d
      const [nq, d, nkv] = [M, N, K];
      for (let q = 0; q < nq; q++) for (let e = 0; e < d; e++) {
        let s = 0;
        for (let t = 0; t < nkv; t++) s += A[q * nkv + t] * B[t * d + e];
        at(q, e, s);
      }
      break;
    }
  }
  return out;
}

/** Per cell of the M x N result: does the library's output equal the reference? */
export function agreement(map: Mapping, got: number[], ref: number[]): boolean[] {
  const { M, N } = map.shape;
  const ok: boolean[] = [];
  for (let r = 0; r < M; r++)
    for (let c = 0; c < N; c++) {
      const g = got[r * map.C.ld + c];
      const w = ref[r * N + c];
      ok.push(Number.isFinite(g) && Math.abs(g - w) <= 1e-9 * Math.max(1, Math.abs(w)));
    }
  return ok;
}
