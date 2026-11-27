import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  FIG103B,
  TIE_ROW,
  buildTree,
  builderTree,
  disagreeingSet,
  f32hex,
  f32short,
  mapTree,
  pairCombine,
  pairs,
  treeStats,
  ulpDistance,
  type Pair,
  type PairRule,
  type Tree,
  type TreeNode,
} from '../../sim/reducetree';
import { Translated, tr } from '../../lib/i18n';

// "A reduction tree folding" (section 10.2): the loop and the tree on the same eight
// values, their time profiles, the profile at 1024 values, rounding, and ties. Every
// number drawn comes from src/sim/reducetree.ts. Six beats, one narration clip each
// (content/ch10/narration/tree-fold-N.txt).

// ---------------------------------------------------------------- a tree, drawn
export interface TreeSvgProps {
  tree: Tree<string>;
  /** Centre of leaf 0. */
  x: number;
  /** Top of the leaves. */
  y: number;
  dx: number;
  dy: number;
  /** Draw levels 0..upto (default all). */
  upto?: number;
  /** A level drawn emphasised (the step in progress). */
  hot?: number;
  /** Box width of a leaf. */
  w?: number;
  h?: number;
  /** Longest text shown in a node before it is cut with an ellipsis. */
  maxChars?: number;
  /** Extra text under each leaf (an index, say). */
  under?: (k: number) => string;
  /** Marks some nodes (drawn with a heavier outline). */
  mark?: (nd: TreeNode<string>) => boolean;
  name?: string;
  /**
   * Place the leaves in the order the tree combines them (root.leaves) and each match
   * midway between its two operands, so a reordering tree is drawn as a clean bracket
   * whose leaf labels show the permutation. Otherwise the model's positions are used.
   */
  operandOrder?: boolean;
}

const cut = (s: string, n: number) => (s.length > n ? s.slice(0, n - 1) + '…' : s);

/** An SVG group drawing a reduction tree: leaves on top, one row per step, time flowing down. */
export function TreeSvg({ tree, x, y, dx, dy, upto, hot, w = 34, h = 22, maxChars = 9, under, mark, name, operandOrder }: TreeSvgProps) {
  const last = upto ?? tree.levels.length - 1;
  const pos = new Map<TreeNode<string>, number>();
  if (operandOrder) {
    tree.root.leaves.forEach((leaf, p) => pos.set(tree.levels[0][leaf], p));
    for (const level of tree.levels.slice(1)) for (const nd of level) pos.set(nd, (pos.get(nd.kids![0])! + pos.get(nd.kids![1])!) / 2);
  }
  const cx = (nd: TreeNode<string>) => x + (operandOrder ? pos.get(nd)! : nd.x) * dx;
  const cy = (nd: TreeNode<string>) => y + nd.level * dy;
  const width = (nd: TreeNode<string>) => Math.max(w, cut(nd.value, maxChars).length * 6.7 + 10);
  const shown = tree.levels.slice(0, last + 1).flat();
  return (
    <g data-part="tree" data-name={name} data-shape={tree.shape} data-upto={last}>
      {shown
        .filter((nd) => nd.kids)
        .flatMap((nd) =>
          nd.kids!.map((kid, k) => (
            <line key={`${nd.level}-${nd.x}-${k}`} x1={cx(kid)} y1={cy(kid) + h + (under && kid.leaf !== null ? 15 : 0)} x2={cx(nd)} y2={cy(nd)}
              stroke={nd.level === hot ? 'var(--ink)' : 'var(--ink-2)'} strokeWidth={nd.level === hot ? 1.6 : 1} />
          )),
        )}
      {shown.map((nd) => {
        const bw = width(nd);
        const isHot = nd.level === hot && nd.level > 0;
        const isMark = mark?.(nd);
        return (
          <g key={`${nd.level}-${nd.x}`} data-part="node" data-level={nd.level} data-value={nd.value}>
            <rect x={cx(nd) - bw / 2} y={cy(nd)} width={bw} height={h} rx={nd.leaf !== null ? 2 : h / 2}
              fill={isHot ? 'var(--ink)' : 'var(--cell)'} stroke={isMark ? 'var(--ink)' : 'var(--rule)'} strokeWidth={isMark ? 2.4 : 1} />
            <text x={cx(nd)} y={cy(nd) + h / 2 + 4} fontSize="12" textAnchor="middle" style={{ fill: isHot ? 'var(--page)' : 'var(--ink)' }}>
              {cut(nd.value, maxChars)}
            </text>
            {under && nd.leaf !== null ? (
              <text x={cx(nd)} y={cy(nd) + h + 12} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }} data-part="under">
                {under(nd.leaf)}
              </text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

// ---------------------------------------------------------------- the scene's data, all from the model
const LOOP = builderTree(FIG103B, 'sum', 'loop');
const TREE = builderTree(FIG103B, 'sum', 'neighbours');
const BIG = treeStats(1024);
export const SCENE_FLOATS = disagreeingSet(8, 'uniform', 1);
/** A bracket over the tie row: leaves show their value (the index is drawn under them), matches show "value #index". */
export function pairTree(row: number[], rule: PairRule, shape: 'loop' | 'neighbours' | 'halves'): Tree<string> {
  const t = mapTree(buildTree(pairs(row), pairCombine(rule), shape), (p: Pair) => `${f32short(p.v)} #${p.i}`);
  for (const leaf of t.levels[0]) leaf.value = f32short(row[leaf.leaf!]);
  return t;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a), 0, 1);
const W = 720;
const H = 340;
const LX = 44; // centre of leaf 0 in the eight-leaf drawings
const DX = 54;

function Stage({ children, headline }: { children: ReactNode; headline: string }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Animation of a reduction tree', 'Hoạt hình một cây rút gọn')} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      {children}
      <text x={16} y={H - 14} fontSize="13.5" fontWeight="600" data-part="headline">{headline}</text>
    </svg>
  );
}

/** Adders busy in each step, as stacks of unit squares: the area is the work, the width the span. */
function Profile({ x, y, ops, unit = 14, label, upto }: { x: number; y: number; ops: number[]; unit?: number; label: string; upto?: number }) {
  const shown = upto ?? ops.length;
  return (
    <g data-part="profile" data-name={label} data-ops={ops.slice(0, shown).join(' ')}>
      <text x={x} y={y - 6} fontSize="12" fontWeight="600">{label}</text>
      <line x1={x} y1={y + 4 * unit + 2} x2={x + Math.max(ops.length, 7) * (unit + 4)} y2={y + 4 * unit + 2} stroke="var(--rule)" />
      {ops.slice(0, shown).map((k, s) => (
        <g key={s}>
          {Array.from({ length: k }, (_, j) => (
            <rect key={j} x={x + s * (unit + 4)} y={y + (3 - j) * unit + 1} width={unit} height={unit - 2} rx="1.5" fill="var(--ink-2)" />
          ))}
          <text x={x + s * (unit + 4) + unit / 2} y={y + 4 * unit + 14} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{s + 1}</text>
        </g>
      ))}
    </g>
  );
}

// ---------------------------------------------------------------- beats
const beats: Beat[] = [
  {
    clip: 'tree-fold-1',
    get label() {
      return tr('One adder', 'Một bộ cộng');
    },
    draw: (p) => {
      const k = clamp(Math.floor(p * 8.6), 0, LOOP.span);
      const total = LOOP.levels[k][0].value;
      return (
        <Stage headline={k === 0 ? tr('the loop starts from element 0: running total 4', 'vòng lặp bắt đầu từ phần tử 0: tổng tích lũy 4') : tr(`the loop, step ${k} of ${LOOP.span}: running total ${total}`, `vòng lặp, bước ${k} trên ${LOOP.span}: tổng tích lũy ${total}`)}>
          <TreeSvg tree={LOOP} x={LX} y={18} dx={DX} dy={34} upto={k} hot={k} name="loop" /* i18n-ignore: data key */ />
          <g>
            <text x={480} y={44} fontSize="12" fontWeight="600"><Translated vi="một bộ cộng">one adder</Translated></text>
            <text x={480} y={64} fontSize="12" data-part="total"><Translated vi={<>tổng tích lũy: {total}</>}>running total: {total}</Translated></text>
            <Profile x={480} y={110} ops={LOOP.opsPerStep} upto={k} label={tr('adders busy, step by step', 'số bộ cộng bận, từng bước một')} />
          </g>
        </Stage>
      );
    },
  },
  {
    clip: 'tree-fold-2',
    get label() {
      return tr('Pairs', 'Từng cặp');
    },
    draw: (p) => {
      const k = clamp(Math.floor(p * 4.4), 0, TREE.span);
      const ops = TREE.opsPerStep[k - 1];
      return (
        <Stage headline={k === 0 ? tr('the same 8 values, folded in pairs', 'cùng 8 giá trị đó, gộp theo từng cặp') : tr(`the tree, step ${k}: ${ops} ${ops === 1 ? 'adder' : 'adders'} busy`, `cây, bước ${k}: ${ops} bộ cộng bận`)}>
          <TreeSvg tree={TREE} x={LX} y={30} dx={DX} dy={62} upto={k} hot={k} name="tree" /* i18n-ignore: data key */ />
          <Profile x={480} y={110} ops={TREE.opsPerStep} upto={k} label={tr('adders busy, step by step', 'số bộ cộng bận, từng bước một')} />
        </Stage>
      );
    },
  },
  {
    clip: 'tree-fold-3',
    get label() {
      return tr('Same work', 'Cùng khối lượng việc');
    },
    draw: (p) => {
      const general = p >= 0.55;
      return (
        <Stage headline={general ? tr('N − 1 operations either way; N − 1 steps against log₂ N', 'cách nào cũng N − 1 phép toán; N − 1 bước so với log₂ N') : tr('same work, shorter span', 'cùng khối lượng việc, span ngắn hơn')}>
          <Profile x={40} y={70} ops={LOOP.opsPerStep} unit={22} label={tr(`the loop: ${LOOP.work} operations in ${LOOP.span} steps`, `vòng lặp: ${LOOP.work} phép toán trong ${LOOP.span} bước`)} />
          <Profile x={400} y={70} ops={TREE.opsPerStep} unit={22} label={tr(`the tree: ${TREE.work} operations in ${TREE.span} steps`, `cây: ${TREE.work} phép toán trong ${TREE.span} bước`)} />
          <g opacity={ramp(p, 0.55, 0.7)}>
            <text x={40} y={222} fontSize="12.5" data-part="general"><Translated vi={<>khối lượng việc: N − 1 = {LOOP.work} cho cả hai</>}>work: N − 1 = {LOOP.work} for both</Translated></text>
            <text x={400} y={222} fontSize="12.5" data-part="general"><Translated vi={<>span: log₂ N = {TREE.span} cho cây, N − 1 = {LOOP.span} cho vòng lặp</>}>span: log₂ N = {TREE.span} for the tree, N − 1 = {LOOP.span} for the loop</Translated></text>
          </g>
          <text x={40} y={256} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="mỗi ô vuông là một bộ cộng bận trong một bước: diện tích là khối lượng việc, bề rộng là span">each square is one adder busy for one step: the area is the work, the width is the span</Translated></text>
        </Stage>
      );
    },
  },
  {
    clip: 'tree-fold-4',
    get label() {
      return tr('Scale it up', 'Phóng to quy mô');
    },
    draw: (p) => {
      const shown = clamp(Math.ceil(p * 13), 1, BIG.span);
      const top = 60;
      const base = 250;
      const scale = (base - top) / BIG.peak;
      const bw = 44;
      const avgY = base - BIG.average * scale;
      return (
        <Stage headline={p < 0.75 ? tr(`${BIG.n} values: ${BIG.work} operations in ${BIG.span} steps`, `${BIG.n} giá trị: ${BIG.work} phép toán trong ${BIG.span} bước`) : tr('hardware sized for the peak idles most of the time', 'phần cứng định cỡ theo đỉnh thì phần lớn thời gian ngồi không')}>
          <text x={40} y={34} fontSize="12" fontWeight="600"><Translated vi={<>số bộ cộng bận ở mỗi bước, {BIG.n} giá trị</>}>adders busy in each step, {BIG.n} values</Translated></text>
          <line x1={40} y1={base} x2={40 + BIG.span * (bw + 8)} y2={base} stroke="var(--rule)" />
          {BIG.opsPerStep.slice(0, shown).map((k, s) => (
            <g key={s} data-part="bar" data-ops={k}>
              <rect x={40 + s * (bw + 8)} y={base - Math.max(1, k * scale)} width={bw} height={Math.max(1, k * scale)} fill="var(--ink-2)" />
              <text x={40 + s * (bw + 8) + bw / 2} y={base - Math.max(1, k * scale) - 5} fontSize="11" textAnchor="middle">{k}</text>
              <text x={40 + s * (bw + 8) + bw / 2} y={base + 14} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>bước {s + 1}</>}>step {s + 1}</Translated></text>
            </g>
          ))}
          <g opacity={ramp(p, 0.4, 0.55)}>
            <line x1={36} y1={avgY} x2={40 + BIG.span * (bw + 8)} y2={avgY} stroke="var(--ink)" strokeDasharray="5 4" />
            <text x={40 + BIG.span * (bw + 8) - 4} y={avgY - 6} fontSize="12" textAnchor="end" data-part="average">
              <Translated vi={<>trung bình {BIG.average.toFixed(1)}, đỉnh {BIG.peak}</>}>
                average {BIG.average.toFixed(1)}, peak {BIG.peak}
              </Translated>
            </text>
          </g>
        </Stage>
      );
    },
  },
  {
    clip: 'tree-fold-5',
    get label() {
      return tr('Rounding', 'Làm tròn');
    },
    draw: (p) => {
      const floats = p >= 0.4;
      const F = SCENE_FLOATS;
      const hl = f32hex(F.loop);
      const ht = f32hex(F.tree);
      let same = 0;
      while (same < hl.length && hl[same] === ht[same]) same++;
      const hexLine = (h: string, y: number, label: string, v: number) => (
        <text x={40} y={y} fontSize="14" data-part="sum" data-name={label} data-hex={h}>
          {label}: {f32short(v)} = {h.slice(0, same)}
          <tspan fontWeight="700" textDecoration="underline">{h.slice(same)}</tspan>
        </text>
      );
      return (
        <Stage headline={floats ? tr(`the same 8 floats, regrouped: the sums are ${ulpDistance(F.loop, F.tree)} float32 steps apart`, `cùng 8 số float đó, nhóm lại: hai tổng cách nhau ${ulpDistance(F.loop, F.tree)} bước float32`) : tr('whole numbers: the loop and the tree agree', 'số nguyên: vòng lặp và cây cho cùng kết quả')}>
          {!floats ? (
            <g opacity={1 - ramp(p, 0.34, 0.4)}>
              <TreeSvg tree={TREE} x={LX} y={20} dx={DX} dy={46} name="integers" /* i18n-ignore: data key */ />
              <text x={480} y={80} fontSize="13" data-part="intsum"><Translated vi={<>vòng lặp: {LOOP.root.value}</>}>the loop: {LOOP.root.value}</Translated></text>
              <text x={480} y={100} fontSize="13" data-part="intsum"><Translated vi={<>cây: {TREE.root.value}</>}>the tree: {TREE.root.value}</Translated></text>
            </g>
          ) : (
            <g opacity={ramp(p, 0.4, 0.5)}>
              <text x={40} y={40} fontSize="12" fontWeight="600"><Translated vi={<>8 giá trị float32 (seed {F.seed})</>}>8 float32 values (seed {F.seed})</Translated></text>
              {F.values.map((v, k) => (
                <text key={k} x={40 + (k % 4) * 150} y={64 + Math.floor(k / 4) * 20} fontSize="12.5" data-part="float">{f32short(v)}</text>
              ))}
              {hexLine(hl, 140, tr('the loop', 'vòng lặp'), F.loop)}
              {hexLine(ht, 168, tr('the tree', 'cây'), F.tree)}
              <text x={40} y={206} fontSize="12" style={{ fill: 'var(--ink-2)' }}><Translated vi="float32 làm tròn sau mỗi phép cộng; cách nhóm quyết định cái gì bị làm tròn">float32 rounds after every addition; the grouping decides what gets rounded</Translated></text>
            </g>
          )}
        </Stage>
      );
    },
  },
  {
    clip: 'tree-fold-6',
    get label() {
      return tr('Ties', 'Hòa nhau');
    },
    draw: (p) => {
      const rule: PairRule = p < 0.5 ? 'left' : 'lower';
      const a = pairTree(TIE_ROW, rule, 'neighbours');
      const b = pairTree(TIE_ROW, rule, 'halves');
      const win = (t: Tree<string>) => t.root.value.split('#')[1];
      const isRoot = (t: Tree<string>) => (nd: TreeNode<string>) => nd === t.root;
      return (
        <Stage headline={rule === 'left' ? tr(`keep the left one on a tie: #${win(a)} here, #${win(b)} there`, `khi hòa giữ cái bên trái: #${win(a)} ở đây, #${win(b)} ở kia`) : tr(`lower index wins a tie: #${win(a)} in every bracket`, `khi hòa chỉ số thấp hơn thắng: #${win(a)} ở mọi nhánh đấu`)}>
          <text x={16} y={20} fontSize="12" fontWeight="600" data-part="rule">{rule === 'left' ? tr('rule: larger value wins; on a tie keep the left one', 'luật: giá trị lớn hơn thắng; khi hòa giữ cái bên trái') : tr('rule: larger value wins; on a tie the lower index wins', 'luật: giá trị lớn hơn thắng; khi hòa chỉ số thấp hơn thắng')}</text>
          <text x={16} y={40} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="Sơ đồ nhánh đấu của Hình 10.3 (láng giềng)">Fig. 10.3's bracket (neighbours)</Translated></text>
          <TreeSvg tree={a} x={34} y={50} dx={42} dy={62} w={36} under={(k) => `#${k}`} maxChars={8} mark={isRoot(a)} name="neighbours" /* i18n-ignore: data key */ />
          <text x={376} y={40} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}><Translated vi="sơ đồ nhánh đấu theo nửa: #0 gặp #4 trước">the halves bracket: #0 meets #4 first</Translated></text>
          <TreeSvg tree={b} x={394} y={50} dx={42} dy={62} w={36} under={(k) => `#${k}`} maxChars={8} mark={isRoot(b)} name="halves" /* i18n-ignore: data key */ operandOrder />
        </Stage>
      );
    },
  },
];

export function TreeFoldScene() {
  return <Scene chapter="ch10" title={tr('A reduction tree folding', 'Một cây rút gọn gộp dần')} beats={beats} />;
}
