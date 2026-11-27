import { Scene, type Beat } from '../../components/Scene';
import { TOY, makeProblem, runBlock, type KStep } from '../../sim/flashkernel';
import { FlashBlockView } from './FlashBlockView';
import { Translated, tr } from '../../lib/i18n';

// "One block's journey through the flash kernel" (section 20.5): block 1 (rows 4 to 7) of
// the toy kernel, lanes in lockstep, warps taking turns. Every frame is a snapshot of
// src/sim/flashkernel.ts's trace; the scripts are content/ch20/narration/flash-block-N.txt.
export const PROB = makeProblem(TOY.n, TOY.d);
export const TRACE = runBlock(TOY, PROB, 1, 'lockstep', 'together');
const S = TRACE.steps;
const first = (pred: (s: KStep) => boolean, from = 0) => {
  const i = S.findIndex((s, k) => k >= from && pred(s));
  return i < 0 ? S.length : i;
};

// Step ranges [a, b): the frame shows the state after steps a+1 ... b as p runs from 0 to 1.
const loadQEnd = first((s) => s.op !== 'load_Q');
const tile0Barrier = first((s) => s.op === 'release_syncthreads' && s.barrier === 'load');
const tile0P = first((s) => s.op === 'compute_P_and_update_D');
const tile0End = first((s) => s.op === 'release_syncthreads' && s.barrier === 'end');
const storeAt = first((s) => s.op === 'store_O');
export const RANGES: [number, number][] = [
  [0, loadQEnd],
  [loadQEnd, tile0Barrier + 1],
  [tile0Barrier + 1, tile0P],
  [tile0P, tile0End],
  [tile0End, storeAt],
  [storeAt, S.length],
];

const frame = (k: number, p: number) => {
  const [a, b] = RANGES[k];
  return Math.min(b, a + 1 + Math.floor(p * (b - a)));
};

const captions = [
  <Translated vi="khối 1 nạp các hàng 4 đến 7 của Q vào thanh ghi">block 1 loads rows 4 to 7 of Q into registers</Translated>,
  <Translated vi="ô 0 của K (chuyển vị) và V vào bộ nhớ chia sẻ, rồi một rào chắn khối">tile 0 of K (transposed) and V into shared memory, then a block barrier</Translated>,
  <Translated vi="điểm số từ Q đã shuffle, che mặt nạ theo chỉ số; giá trị lớn nhất cập nhật m và D">scores from shuffled Q, masked by index; the max updates m and D</Translated>,
  <Translated vi="P tại chỗ, một rào chắn warp, rồi O được co giãn lại và cộng thêm P·V">P in place, a warp barrier, then O rescaled and P·V added</Translated>,
  <Translated vi="ô 2 nằm trên đường chéo: mọi P đều bằng 0, sách vẫn tính nó">tile 2 lies above the diagonal: every P is 0, the book computes it anyway</Translated>,
  <Translated vi="chia cho D một lần; ghi O và L = m + log D">divide by D once; write O and L = m + log D</Translated>,
];

const beats: Beat[] = captions.map((cap, k) => ({
  clip: `flash-block-${k + 1}`,
  get label() {
    return tr(
      ['Load Q', 'Load a tile', 'Scores', 'P and O', 'Every tile', 'Store'][k],
      ['Nạp Q', 'Nạp một ô', 'Điểm số', 'P và O', 'Mọi ô', 'Ghi'][k],
    );
  },
  draw: (p: number) => <FlashBlockView trace={TRACE} prob={PROB} at={frame(k, p)} caption={cap} />,
}));

export function FlashBlockScene() {
  return <Scene chapter="ch20" title={tr("One block's journey through the flash kernel", 'Hành trình của một khối qua kernel flash')} beats={beats} />;
}
