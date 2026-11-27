import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BLOCK_SIZES,
  HANDOFFS,
  OP_PARAM,
  callText,
  hybrid,
  isPow2,
  ones,
  shuffle,
  smallValues,
  sourceOf,
  warpReduceTrace,
  type Handoff,
  type Op,
  type Outcome,
  type Trace,
} from '../../sim/shuffle';
import { functionSource } from '../../sim/toolchain';
import { widthChoices } from '../../sim/warps';
import { MIN_W, ShuffleScene, VIEW_W, WarpGrid, gridHeight } from './ShuffleScene';
import { useNarrow } from './useNarrow';
import { STATUS, useLabStatus } from './status';

import { NoteGroup } from './NoteGroup';
const LAB = 'ch10/lab1_sum';
const AS_PRINTED = 'ch10/snippets/s107_warp_reduce_as_printed.cu';
const HYBRID = 'ch10/snippets/s107_hybrid.cu';
const missing = (s: string | null) => !s || s.startsWith('// could not load');

/** Identifiers coloured by the memory they live in. */
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', input_s: 'shared',
  partialSum: 'reg', val: 'reg', t: 'reg', stride: 'reg', full: 'reg', lane: 'reg', width: 'reg',
};

/** Warp widths to offer: the 8-lane toy of the book's figures, then the book's 32, AMD's 64 and any device reported. */
function useWarpChoices(withToy: boolean) {
  const { reports } = useReports();
  return useMemo(() => {
    const real = widthChoices(reports as Reports).filter((c) => isPow2(c.width) && c.width >= 8 && c.width <= 64);
    const toy = {
      key: 'toy',
      get label() {
        return tr('8 lanes, the scale of Fig. 10.12', '8 lane, cỡ của Fig. 10.12');
      },
      width: 8,
      get note() {
        return tr(
          'A toy: the book draws warps of 4 threads in Fig. 10.12; no GPU has warps of 8.',
          'Một mô hình đồ chơi: sách vẽ warp 4 luồng ở Fig. 10.12; không GPU nào có warp 8 luồng.',
        );
      },
    };
    return [...(withToy ? [toy] : []), ...real.map((c) => ({ key: c.key, label: c.label, width: c.width, note: c.note }))];
  }, [reports, withToy]);
}

const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const count = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

// ---------------------------------------------------------------- explorer: one shuffle
type ClickMode = 'explain' | 'exit' | 'mask';
const OP_LABEL: Record<Op, string> = {
  down: 'down',
  up: 'up',
  xor: 'xor',
  get idx() {
    return tr('by source lane', 'theo lane nguồn');
  },
};

function explainLane(op: Op, lane: number, param: number, width: number, warp: number, r: { outcome: Outcome; value: number | null; src: number | null }, exited: boolean[], mask: boolean[]): string {
  if (r.outcome === 'absent')
    return exited[lane]
      ? tr(
          `Lane ${lane} has returned, so it does not call. A lane that asks for its value gets an undefined one.`,
          `Lane ${lane} đã thoát, nên nó không gọi. Một lane hỏi giá trị của nó sẽ nhận một giá trị không xác định.`,
        )
      : tr(
          `Lane ${lane} is not in the mask, so it takes no part in this call. A lane that asks for its value gets an undefined one.`,
          `Lane ${lane} không có trong mask, nên nó không tham gia lời gọi này. Một lane hỏi giá trị của nó sẽ nhận một giá trị không xác định.`,
        );
  const { target, inRange } = sourceOf(op, lane, param, width);
  const base = lane - (lane % width);
  const pos = lane % width;
  const whole = width === warp;
  const where = whole ? tr('the warp', 'warp') : tr(`its group, lanes ${base} to ${base + width - 1}`, `nhóm của nó, các lane ${base} đến ${base + width - 1}`);
  // With the whole warp as the group, positions are lane numbers and need no introduction.
  const intro = whole ? tr(`Lane ${lane}'s source would be lane ${op === 'down' ? pos + param : pos - param}`, `Nguồn của lane ${lane} sẽ là lane ${op === 'down' ? pos + param : pos - param}`)
    : tr(`Lane ${lane} is position ${pos} of ${where}. Its source would be position ${op === 'down' ? pos + param : pos - param}`, `Lane ${lane} là vị trí ${pos} của ${where}. Nguồn của nó sẽ là vị trí ${op === 'down' ? pos + param : pos - param}`);
  let s: string;
  switch (op) {
    case 'down':
      s = intro + (inRange ? (whole ? '.' : tr(`, lane ${target}.`, `, tức lane ${target}.`)) : tr(`, past the end of ${where}. The source lane is not wrapped around, so lane ${lane} keeps its own value, ${lane}.`, `, vượt quá cuối ${where}. Lane nguồn không được quay vòng, nên lane ${lane} giữ giá trị của chính nó, ${lane}.`));
      break;
    case 'up':
      s = intro + (inRange ? (whole ? '.' : tr(`, lane ${target}.`, `, tức lane ${target}.`)) : tr(`, before the start of ${where}, so lane ${lane} keeps its own value, ${lane}.`, `, trước điểm đầu của ${where}, nên lane ${lane} giữ giá trị của chính nó, ${lane}.`));
      break;
    case 'xor':
      s = tr(`${lane} xor ${param} is ${target}. `, `${lane} xor ${param} là ${target}. `) +
        (!inRange ? tr(`That lane is in a later group, which xor cannot read, so lane ${lane} gets its own value back.`, `Lane đó nằm ở một nhóm phía sau, nơi xor không đọc được, nên lane ${lane} nhận lại giá trị của chính nó.`)
          : target < base ? tr(`That lane is in an earlier group, which xor may read.`, `Lane đó nằm ở một nhóm phía trước, nơi xor được phép đọc.`) : tr(`It is in the same group.`, `Nó nằm trong cùng nhóm.`));
      break;
    case 'idx':
      s = param >= width
        ? tr(`srcLane ${param} is outside 0 to ${width - 1}, so it is taken modulo ${width}: position ${param % width} of ${where}, lane ${target}.`, `srcLane ${param} nằm ngoài khoảng 0 đến ${width - 1}, nên nó được lấy modulo ${width}: vị trí ${param % width} của ${where}, tức lane ${target}.`)
        : tr(`srcLane ${param} is position ${param} of ${where}: lane ${target}.`, `srcLane ${param} là vị trí ${param} của ${where}: lane ${target}.`);
      break;
  }
  if (r.outcome === 'undefined') s += tr(` But lane ${r.src} ${exited[r.src!] ? 'has returned' : 'is not in the mask'}, so the value lane ${lane} gets is undefined.`, ` Nhưng lane ${r.src} ${exited[r.src!] ? 'đã thoát' : 'không có trong mask'}, nên giá trị mà lane ${lane} nhận được là không xác định.`);
  else if (r.outcome === 'moved') s += tr(` It receives ${r.value}.`, ` Nó nhận ${r.value}.`);
  return s;
}

function ShuffleExplorer() {
  const choices = useWarpChoices(true);
  const [key, setKey] = useState('book');
  const chosen = choices.find((c) => c.key === key) ?? choices[1] ?? choices[0];
  const warp = chosen.width;
  const [op, setOp] = useState<Op>('down');
  const [param, setParam] = useState(16);
  const [width, setWidth] = useState(32);
  const [mode, setMode] = useState<ClickMode>('explain');
  const [exited, setExited] = useState<boolean[]>(() => new Array(32).fill(false));
  const [mask, setMask] = useState<boolean[]>(() => new Array(32).fill(true));
  const [focus, setFocus] = useState(20);

  // A new warp width starts clean.
  useEffect(() => {
    setExited(new Array(warp).fill(false));
    setMask(new Array(warp).fill(true));
    setWidth(warp);
    setParam((p) => Math.min(Math.max(p, 1), warp - 1));
    setFocus((f) => Math.min(f, warp - 1));
  }, [warp]);

  const ready = exited.length === warp && mask.length === warp;
  const values = range(warp);
  const res = ready ? shuffle(op, values, param, { width, mask, exited }) : [];
  const tally = (o: Outcome) => res.filter((r) => r.outcome === o).length;
  const pmin = op === 'idx' ? 0 : 1;

  // On a phone the warp folds into rows of 16 lanes, drawn on a narrower canvas so the
  // lanes stay large enough to read and click; no lane is off-screen.
  const narrow = useNarrow();
  const VW = narrow ? 400 : VIEW_W;
  const perRow = Math.min(warp, narrow ? 16 : 32);
  const blocks = warp / perRow;
  const cw = Math.min((VW - 16) / perRow, 48);
  const X0 = (VW - cw * perRow) / 2;
  const CH = 24;
  const GAPY = 46;
  const BLOCK_H = 14 + CH + GAPY + CH + 22;
  const H = blocks * BLOCK_H + 4;
  const pos = (lane: number) => ({ x: X0 + (lane % perRow) * cw, y: Math.floor(lane / perRow) * BLOCK_H + 14 });
  const font = cw < 22 ? 9 : 11;

  const clickLane = (lane: number) => {
    if (mode === 'explain') setFocus(lane);
    else if (mode === 'exit') setExited((e) => e.map((x, i) => (i === lane ? !x : x)));
    else setMask((m) => m.map((x, i) => (i === lane ? !x : x)));
  };
  const reset = () => {
    setExited(new Array(warp).fill(false));
    setMask(new Array(warp).fill(true));
  };
  const f = res[focus];

  return (
    <Bench
      title={tr('One shuffle across a warp', 'Một lệnh shuffle trên cả warp')}
      task={tr(
        'Set down, delta 16, and explain lane 20. Then width 8, delta 2: what do lanes 13 and 14 get?',
        'Đặt down, delta 16, và giải thích lane 20. Rồi width 8, delta 2: các lane 13 và 14 nhận được gì?',
      )}
      foot={
        <Translated
          vi={
            <>
              Mỗi lane bắt đầu với chính số hiệu lane của nó, nên giá trị một lane nhận được cho biết nó đến từ lane nào. Hàng
              trên: giá trị mỗi lane truyền vào. Hàng dưới: giá trị lời gọi trả về cho nó. Ô viền nét đứt đã giữ giá trị của
              chính nó; dấu <b className="t-fail">?</b> màu đỏ là không xác định.
            </>
          }
        >
          <>
            Each lane starts with its own lane number, so the value a lane receives names the lane it came from. Top row:
            what each lane passes in. Bottom row: what the call returns to it. A dashed box kept its own value; a red{' '}
            <b className="t-fail">?</b> is undefined.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Warp">Warp</Translated>
          <select value={chosen.key} onChange={(e) => setKey(e.target.value)} aria-label={tr('warp width', 'độ rộng warp')}>
            {choices.map((c) => <option key={c.key} value={c.key}>{tr(c.label, c.key === 'book' ? '32 của sách' : c.key === 'w64' ? '64, như HIP ghi cho gfx9' : c.label)}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Hàm">Function</Translated>
          <select value={op} aria-label={tr('shuffle function', 'hàm shuffle')} onChange={(e) => { const o = e.target.value as Op; setOp(o); if (o !== 'idx' && param < 1) setParam(1); }}>
            {(['down', 'up', 'xor', 'idx'] as Op[]).map((o) => <option key={o} value={o}>{OP_LABEL[o]}</option>)}
          </select>
        </label>
        <label className="field">
          {OP_PARAM[op]}: <b>{param}</b>
          <input type="range" min={pmin} max={warp - 1} value={param} aria-label={OP_PARAM[op]} onChange={(e) => setParam(+e.target.value)} />
        </label>
        <label className="field">
          <Translated vi="width">width</Translated>
          <select value={width} aria-label={tr('width argument', 'đối số width')} onChange={(e) => setWidth(+e.target.value)}>
            {range(Math.round(Math.log2(warp)) + 1).map((k) => 2 ** k).map((w) => <option key={w} value={w}>{w}{w === warp ? tr(' (default)', ' (mặc định)') : ''}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="ui muted">
          <Translated vi="Bấm vào một lane">Clicking a lane</Translated>
        </span>
        <span className="seg" role="group" aria-label={tr('What clicking a lane does', 'Việc bấm vào một lane sẽ làm gì')}>
          {([['explain', tr('explains it', 'giải thích nó')], ['exit', tr('makes it return', 'cho nó thoát')], ['mask', tr('takes it out of the mask', 'bỏ nó ra khỏi mask')]] as [ClickMode, string][]).map(([m, l]) => (
            <button key={m} aria-pressed={mode === m} onClick={() => setMode(m)}>{l}</button>
          ))}
        </span>
        <button className="quiet" onClick={reset} disabled={!exited.some(Boolean) && mask.every(Boolean)}>
          <Translated vi="Đưa mọi lane trở lại">All lanes back</Translated>
        </button>
      </div>
      <pre className="mono" style={{ margin: '0 0 0.6rem', whiteSpace: 'pre-wrap' }} data-call>{ready ? callText(op, param, width, mask) : ''}</pre>
      {ready ? (
        <svg viewBox={`0 0 ${VW} ${H}`} role="img" aria-label={tr('Lanes before and after the shuffle', 'Các lane trước và sau lệnh shuffle')} data-rows={blocks}
          style={{ width: '100%', maxWidth: VW, minWidth: narrow ? 0 : warp > 32 ? 640 : warp > 8 ? MIN_W : 0, display: 'block' }}>
          {/* groups of `width` lanes */}
          {width < warp
            ? range(warp / width).map((g) => {
                if (g % 2 === 0) return null;
                const first = g * width;
                const parts = [];
                for (let l = first; l < first + width; l += perRow - ((l % perRow))) {
                  const n = Math.min(first + width - l, perRow - (l % perRow));
                  const p = pos(l);
                  parts.push(<rect key={l} x={p.x} y={p.y - 2} width={n * cw} height={CH * 2 + GAPY + 4} fill="var(--rule-soft)" />);
                }
                return <g key={g}>{parts}</g>;
              })
            : null}
          {/* arrows */}
          {res.map((r) => {
            if (r.outcome === 'absent' || r.src == null) return null;
            const d = pos(r.lane);
            const xd = d.x + cw / 2;
            if (r.outcome === 'own') return <line key={'a' + r.lane} x1={xd} y1={d.y + CH + 2} x2={xd} y2={d.y + CH + GAPY - 2} stroke="var(--ink-3)" strokeDasharray="2 2" />;
            const s = pos(r.src);
            const hot = r.lane === focus;
            return (
              <line key={'a' + r.lane} x1={s.x + cw / 2} y1={s.y + CH + 1} x2={xd} y2={d.y + CH + GAPY - 1}
                stroke={r.outcome === 'undefined' ? 'var(--fail)' : 'var(--reg)'} strokeWidth={hot ? 2.4 : 1}
                strokeDasharray={r.outcome === 'undefined' ? '3 2' : undefined} opacity={hot ? 1 : 0.55} />
            );
          })}
          {range(warp).map((lane) => {
            const p = pos(lane);
            const r = res[lane];
            const gone = exited[lane];
            const off = !mask[lane];
            const hot = lane === focus;
            const show = lane % (perRow > 16 ? 4 : 1) === 0 || hot;
            const outFill = r.outcome === 'moved' ? 'var(--reg-wash)' : r.outcome === 'undefined' ? 'var(--fail-wash)' : 'none';
            const outStroke = r.outcome === 'moved' ? 'var(--reg)' : r.outcome === 'undefined' ? 'var(--fail)' : 'var(--ink-3)';
            return (
              <g key={lane} onClick={() => clickLane(lane)} style={{ cursor: 'pointer' }}>
                {show ? <text x={p.x + cw / 2} y={p.y - 4} fontSize="9" textAnchor="middle" style={{ fill: hot ? 'var(--ink)' : 'var(--ink-2)' }} fontWeight={hot ? 700 : 400}>{lane}</text> : null}
                <rect data-in-lane={lane} data-state={gone ? 'exited' : off ? 'unmasked' : 'calls'} x={p.x + 0.8} y={p.y} width={cw - 1.6} height={CH} rx="2"
                  fill={gone || off ? 'none' : 'var(--cell)'} stroke={hot ? 'var(--ink)' : gone || off ? 'var(--ink-3)' : 'var(--reg)'}
                  strokeWidth={hot ? 2.2 : 1} strokeDasharray={gone ? '2 2' : off ? '1 2' : undefined} />
                <text x={p.x + cw / 2} y={p.y + CH / 2 + font / 2.8} fontSize={font} textAnchor="middle" style={{ fill: gone || off ? 'var(--ink-3)' : 'var(--ink)' }}>
                  {gone ? '×' : lane}
                </text>
                <rect data-lane={lane} data-outcome={r.outcome} data-value={r.value == null ? (r.outcome === 'absent' ? '' : '?') : r.value}
                  x={p.x + 0.8} y={p.y + CH + GAPY} width={cw - 1.6} height={CH} rx="2" fill={outFill} stroke={hot ? 'var(--ink)' : outStroke}
                  strokeWidth={hot ? 2.2 : 1} strokeDasharray={r.outcome === 'own' || r.outcome === 'absent' ? '3 2' : undefined} />
                <text x={p.x + cw / 2} y={p.y + CH + GAPY + CH / 2 + font / 2.8} fontSize={font} textAnchor="middle"
                  style={{ fill: r.outcome === 'undefined' ? 'var(--fail)' : r.outcome === 'absent' ? 'var(--ink-3)' : 'var(--ink)' }} fontWeight={r.outcome === 'undefined' ? 700 : 400}>
                  {r.outcome === 'absent' ? '–' : r.value == null ? '?' : r.value}
                </text>
              </g>
            );
          })}
        </svg>
      ) : null}
      {ready ? (
        <>
          <p className="narrate" role="status" style={{ margin: '0.7rem 0 0.5rem' }} data-explain>
            {explainLane(op, focus, param, width, warp, f, exited, mask)}
          </p>
          <p className="muted" style={{ margin: 0 }} data-tally>
            <Translated
              vi={
                <>
                  {tally('moved')} lane nhận giá trị của lane khác, {tally('own')} lane giữ giá trị của chính mình,{' '}
                  {tally('undefined')} lane nhận một giá trị không xác định, {tally('absent')} lane không gọi.
                  {warp === 64 ? ' Trên HIP, mask phải là một số nguyên 64-bit, như được viết ở đây.' : ''}
                  {key === 'toy' ? ' ' + chosen.note : ''}
                </>
              }
            >
              {count(tally('moved'), 'lane')} received another lane's value, {count(tally('own'), 'lane')} kept {tally('own') === 1 ? 'its' : 'their'} own,{' '}
              {count(tally('undefined'), 'lane')} got an undefined value, {count(tally('absent'), 'lane')} did not call.
              {warp === 64 ? ' On HIP the mask must be a 64-bit integer, as written here.' : ''}
              {key === 'toy' ? ' ' + chosen.note : ''}
            </Translated>
          </p>
        </>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- stepper: warp_reduce
type Scenario = 'full' | 'wave64' | 'return' | 'zero';
const SCENARIOS: { key: Scenario; label: string }[] = [
  {
    key: 'full',
    get label() {
      return tr('a full 32-lane warp', 'một warp 32 lane đầy đủ');
    },
  },
  {
    key: 'wave64',
    get label() {
      return tr('the loop written for 32 lanes, on a 64-wide wavefront', 'vòng lặp viết cho 32 lane, trên một wavefront rộng 64');
    },
  },
  {
    key: 'return',
    get label() {
      return tr('lanes from n on returned early', 'các lane từ n trở đi thoát sớm');
    },
  },
  {
    key: 'zero',
    get label() {
      return tr('lanes from n on stay and contribute 0', 'các lane từ n trở đi ở lại và đóng góp 0');
    },
  },
];

export function stepperTrace(pattern: 'down' | 'xor', data: 'small' | 'ones', scenario: Scenario, n: number): Trace {
  const hw = scenario === 'wave64' ? 64 : 32;
  const v = data === 'ones' ? ones(hw) : smallValues(hw);
  if (scenario === 'wave64') return warpReduceTrace(v, { pattern, loopWidth: 32 });
  if (scenario === 'full') return warpReduceTrace(v, { pattern });
  return warpReduceTrace(v, { pattern, tail: { from: n, mode: scenario } });
}

function stepText(t: Trace, k: number): string {
  if (k === 0)
    return tr(
      `Every lane holds its starting value${t.live.some((x) => !x) ? `; lanes ${t.live.indexOf(false)} and up have returned` : ''}. ` +
        `The loop starts at stride ${t.loopWidth / 2}. Press Step.`,
      `Mỗi lane giữ giá trị ban đầu của nó${t.live.some((x) => !x) ? `; các lane từ ${t.live.indexOf(false)} trở lên đã thoát` : ''}. ` +
        `Vòng lặp bắt đầu ở bước nhảy ${t.loopWidth / 2}. Bấm Bước.`,
    );
  const st = t.steps[k - 1];
  const live = st.lanes.filter((l) => l.live);
  const n = (o: Outcome) => live.filter((l) => l.outcome === o).length;
  const useful = live.filter((l) => l.useful).length;
  let s = t.pattern === 'down'
    ? tr(
        `Stride ${st.stride}: every lane that is still running adds the value of the lane ${st.stride} above it. ${count(n('moved'), 'lane')} received another lane's value`,
        `Bước nhảy ${st.stride}: mỗi lane còn đang chạy cộng giá trị của lane nằm trên nó ${st.stride} vị trí. ${n('moved')} lane nhận giá trị của lane khác`,
      )
    : tr(
        `xor ${st.stride}: every lane that is still running trades values with the lane whose number differs from its own in the bit worth ${st.stride}. ${count(n('moved'), 'lane')} received a partner's value`,
        `xor ${st.stride}: mỗi lane còn đang chạy đổi giá trị với lane có số hiệu khác số của nó ở bit có giá trị ${st.stride}. ${n('moved')} lane nhận giá trị của đối tác`,
      );
  if (n('own')) s += tr(`; ${count(n('own'), 'lane')} had no lane ${st.stride} above ${n('own') === 1 ? 'it' : 'them'} and added ${n('own') === 1 ? 'its' : 'their'} own value to ${n('own') === 1 ? 'itself' : 'themselves'}`, `; ${n('own')} lane không có lane nào nằm trên ${st.stride} vị trí nên cộng giá trị của chính mình vào chính mình`);
  if (n('undefined')) s += tr(`; ${count(n('undefined'), 'lane')} read a lane that had returned and got an undefined value`, `; ${n('undefined')} lane đọc một lane đã thoát và nhận giá trị không xác định`);
  s += tr(
    `. ${useful} of these ${live.length} additions ${t.pattern === 'down' ? 'reach lane 0' : 'combine data every lane needs'}.`,
    `. ${useful} trên ${live.length} phép cộng này ${t.pattern === 'down' ? 'tới được lane 0' : 'kết hợp dữ liệu mà mọi lane cần'}.`,
  );
  return s;
}

function resultText(t: Trace, scenario: Scenario, n: number): { ok: boolean; text: string } {
  const lanes = t.hardwareWidth;
  if (scenario === 'wave64' && t.pattern === 'down')
    return { ok: false, text: tr(`Lane 0 holds ${t.final[0]} and lane 32 holds ${t.final[32]}. The sum is ${t.total}, and no lane holds it: the loop reduced each half of the wavefront separately.`, `Lane 0 giữ ${t.final[0]} và lane 32 giữ ${t.final[32]}. Tổng là ${t.total}, và không lane nào giữ nó: vòng lặp đã rút gọn riêng từng nửa của wavefront.`) };
  if (scenario === 'wave64')
    return { ok: false, text: tr(`Every lane holds the sum of its own half: ${t.final[0]} in lanes 0 to 31, ${t.final[32]} in lanes 32 to 63. The sum is ${t.total}.`, `Mỗi lane giữ tổng của nửa chứa nó: ${t.final[0]} ở các lane 0 đến 31, ${t.final[32]} ở các lane 32 đến 63. Tổng là ${t.total}.`) };
  if (t.pattern === 'down') {
    if (t.lane0 == null)
      return { ok: false, text: tr(`Lane 0's result is undefined. Lanes ${n} and up returned; lanes that read them passed the undefined value on, and it reached lane 0.`, `Kết quả của lane 0 là không xác định. Các lane từ ${n} trở lên đã thoát; những lane đọc chúng truyền tiếp giá trị không xác định, và nó tới được lane 0.`) };
    return { ok: t.lane0 === t.total, text: tr(`Lane 0 holds ${t.lane0}, the sum of ${scenario === 'zero' ? `lanes 0 to ${n - 1}` : `all ${lanes} values`}.`, `Lane 0 giữ ${t.lane0}, tổng của ${scenario === 'zero' ? `các lane 0 đến ${n - 1}` : `cả ${lanes} giá trị`}.`) };
  }
  const undef = t.final.filter((v, i) => t.live[i] && v == null).length;
  if (undef) return { ok: false, text: tr(`${count(undef, 'lane')} ended with an undefined value; ${t.holders.length} hold the sum.`, `${undef} lane kết thúc với một giá trị không xác định; ${t.holders.length} lane giữ tổng.`) };
  return { ok: t.holders.length === t.live.filter(Boolean).length, text: tr(`Every lane holds ${t.total}.`, `Mọi lane đều giữ ${t.total}.`) };
}

function ReduceStepper() {
  const [pattern, setPattern] = useState<'down' | 'xor'>('down');
  const [data, setData] = useState<'small' | 'ones'>('small');
  const [scenario, setScenario] = useState<Scenario>('full');
  const [n, setN] = useState(20);
  const [k, setK] = useState(0);
  const [cone, setCone] = useState(true);
  const t = useMemo(() => stepperTrace(pattern, data, scenario, n), [pattern, data, scenario, n]);
  const total = t.steps.length;
  const at = Math.min(k, total);
  useEffect(() => setK(0), [pattern, data, scenario, n]);
  const done = t.steps.slice(0, at);
  const ops = done.reduce((a, s) => a + s.lanes.filter((l) => l.live).length, 0);
  const useful = done.reduce((a, s) => a + s.lanes.filter((l) => l.useful).length, 0);
  const res = at === total ? resultText(t, scenario, n) : null;
  const tail = scenario === 'return' || scenario === 'zero';
  const H = gridHeight(total + 1, 20) + 6;
  const ring = at === total ? (scenario === 'wave64' && pattern === 'down' ? [0, 32] : pattern === 'down' ? [0] : []) : [];
  return (
    <Bench
      title={tr('warp_reduce, step by step', 'warp_reduce, từng bước một')}
      task={tr(
        'Step a full warp. Count the additions that reach lane 0. Then break it two ways and mend it.',
        'Chạy từng bước một warp đầy đủ. Đếm các phép cộng tới được lane 0. Rồi làm hỏng nó theo hai cách và sửa lại.',
      )}
      foot={
        <Translated
          vi={
            <>
              Các hàng chạy xuống theo thời gian: các giá trị ban đầu, rồi tổng bộ phận của mỗi lane sau mỗi bước. Khi bật hình
              nón, một ô tô đậm là một phép cộng có kết quả tới được lane 0 (với xor: phép mà mọi lane cần); ô nhạt là công việc
              không ai đọc. Các ô viền nét đứt là các lane đã thoát. Mũi tên cho thấy bước vừa thực hiện.
            </>
          }
        >
          <>
            Rows run down in time: the starting values, then each lane's partial sum after each step. With the cone on, a solid
            cell is an addition whose result reaches lane 0 (for xor: one every lane needs); a pale one is work nobody reads.
            Dashed cells are lanes that returned. Arrows show the step just taken.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label={tr('Pattern', 'Mẫu')}>
          <button aria-pressed={pattern === 'down'} onClick={() => setPattern('down')}>
            <Translated vi="down (Hình 10.13)">down (Fig. 10.13)</Translated>
          </button>
          <button aria-pressed={pattern === 'xor'} onClick={() => setPattern('xor')}>
            <Translated vi="xor hình cánh bướm">xor butterfly</Translated>
          </button>
        </span>
        <label className="field">
          <Translated vi="Trường hợp">Case</Translated>
          <select value={scenario} aria-label={tr('case', 'trường hợp')} onChange={(e) => setScenario(e.target.value as Scenario)}>
            {SCENARIOS.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
          </select>
        </label>
        {tail ? (
          <label className="field">
            n = <b>{n}</b>
            <input type="range" min={1} max={31} value={n} aria-label="n" onChange={(e) => setN(+e.target.value)} />
          </label>
        ) : null}
        <label className="field">
          <Translated vi="Giá trị">Values</Translated>
          <select value={data} aria-label={tr('values', 'giá trị')} onChange={(e) => setData(e.target.value as 'small' | 'ones')}>
            <option value="small">{tr('small integers', 'số nguyên nhỏ')}</option>
            <option value="ones">{tr('all ones', 'toàn số 1')}</option>
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={cone} onChange={(e) => setCone(e.target.checked)} />{' '}
          <Translated vi="hiện hình nón">show the cone</Translated>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => setK(0)} disabled={at === 0}>
          <Translated vi="Bắt đầu lại">Restart</Translated>
        </button>
        <button onClick={() => setK(at - 1)} disabled={at === 0}>
          <Translated vi="Lùi">Back</Translated>
        </button>
        <button className="primary" onClick={() => setK(at + 1)} disabled={at >= total}>
          <Translated vi="Bước">Step</Translated>
        </button>
        <button onClick={() => setK(total)} disabled={at >= total}>
          <Translated vi="Chạy tới cuối">Run to the end</Translated>
        </button>
        <span className="muted" aria-live="polite">
          <Translated vi={<>bước {at} trên {total}</>}>
            step {at} of {total}
          </Translated>
        </span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.7rem' }} data-narrate>{stepText(t, at)}</div>
      <svg viewBox={`0 0 ${VIEW_W} ${H}`} role="img" aria-label={tr('Partial sums of every lane after each step', 'Tổng bộ phận của mọi lane sau mỗi bước')}
        style={{ width: '100%', maxWidth: VIEW_W, minWidth: t.hardwareWidth > 32 ? 640 : MIN_W, display: 'block' }} data-stepper-grid>
        <WarpGrid trace={t} upto={at} cone={cone} ring={ring} gap={20} />
      </svg>
      <p style={{ margin: '0.7rem 0 0' }} data-counts>
        <span className="muted">
          <Translated vi="Số phép toán lane tới giờ: ">Lane-operations so far: </Translated>
        </span>
        <b>{ops}</b>
        <span className="muted">
          <Translated vi={<>; tới được {pattern === 'down' ? 'lane 0' : 'mọi lane'}: </>}>
            ; reaching {pattern === 'down' ? 'lane 0' : 'every lane'}: </Translated>
        </span>
        <b>{useful}</b>
        <span className="muted">
          <Translated vi="; lãng phí: ">; wasted: </Translated>
        </span>
        <b>{ops - useful}</b>.
        {res ? <> <span className={'chip ' + (res.ok ? 'ok' : 'fail')} data-result={res.ok ? 'ok' : 'fail'}>{res.ok ? tr('correct', 'đúng') : tr('wrong', 'sai')}</span> {res.text}</> : null}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- explorer: where to hand over
const HANDOFF_LABEL: Record<Handoff, string> = {
  get printed() {
    return tr('as printed: shared loop while stride ≥ warp width', 'như khi in: vòng lặp bộ nhớ chia sẻ khi stride ≥ độ rộng warp');
  },
  get early() {
    return tr('one level early: stride > warp width', 'sớm một tầng: stride > độ rộng warp');
  },
  get double() {
    return tr('shared loop down to 1, then warp_reduce too', 'vòng lặp bộ nhớ chia sẻ xuống tới 1, rồi thêm warp_reduce');
  },
  get early2() {
    return tr('one level early, each lane adds slots t and t + width (ours)', 'sớm một tầng, mỗi lane cộng các ô t và t + width (của khóa học này)');
  },
};

function HandoffExplorer() {
  const choices = useWarpChoices(false);
  const [B, setB] = useState(1024);
  const [key, setKey] = useState('book');
  const [variant, setVariant] = useState<Handoff>('printed');
  const [line11, setLine11] = useState(true);
  const chosen = choices.find((c) => c.key === key) ?? choices[0];
  const W = chosen.width;
  const h = useMemo(() => hybrid(ones(2 * B), B, W, variant, { postLoopBarrier: line11 }), [B, W, variant, line11]);
  const readSet = new Set(h.handoff.reads.flat());
  const slots = h.handoff.values.length;
  const perRow = 32;
  const rowsN = Math.ceil(slots / perRow);
  const cw = (VIEW_W - 16) / perRow;
  const RH = 40;
  const svgH = rowsN * RH + 8;
  const state = (k: number) => {
    const live = h.handoff.live[k];
    const read = readSet.has(k);
    return live && read ? 'used' : live ? 'dropped' : read ? 'again' : 'idle';
  };
  const dropped = range(slots).filter((k) => state(k) === 'dropped').length;
  const again = range(slots).filter((k) => state(k) === 'again').length;
  const race = h.races[0];
  return (
    <Bench
      title={tr('Where to hand over', 'Bàn giao ở đâu')}
      task={tr(
        'Find out what goes wrong when warp 0 takes over one level too early, or one level too late.',
        'Tìm xem điều gì sai khi warp 0 tiếp quản sớm hơn một tầng, hoặc muộn hơn một tầng.',
      )}
      foot={
        <Translated
          vi={
            <>
              Mọi phần tử đầu vào đều là 1, nên đáp án đúng là gấp đôi kích thước khối. Dải ô cho thấy <code>input_s</code> tại
              lúc bàn giao: các ô tô đậm vẫn giữ một tổng bộ phận và được warp 0 đọc; các ô đỏ hoặc vẫn giữ một tổng mà không
              lane nào đọc, hoặc đã được cộng vào một ô thấp hơn và bị đọc lại.
            </>
          }
        >
          <>
            Every input element is 1, so the right answer is twice the block size. The strip shows <code>input_s</code> at the
            hand-off: solid slots still hold a partial sum and are read by warp 0; red ones are either still holding a sum that
            no lane reads, or already added into a lower slot and read again.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          <Translated vi="Kích thước khối">Block size</Translated>
          <select value={B} aria-label={tr('block size', 'kích thước khối')} onChange={(e) => setB(+e.target.value)}>
            {BLOCK_SIZES.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <select value={chosen.key} aria-label={tr('warp width', 'độ rộng warp')} onChange={(e) => setKey(e.target.value)}>
            {choices.map((c) => <option key={c.key} value={c.key}>{tr(c.label, c.key === 'book' ? '32 của sách' : c.key === 'w64' ? '64, như HIP ghi cho gfx9' : c.label)}</option>)}
          </select>
        </label>
        <label className="field">
          <Translated vi="Bàn giao">Hand-off</Translated>
          <select value={variant} aria-label={tr('hand-off', 'bàn giao')} onChange={(e) => setVariant(e.target.value as Handoff)}>
            {HANDOFFS.map((v) => <option key={v} value={v}>{HANDOFF_LABEL[v]}</option>)}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={line11} onChange={(e) => setLine11(e.target.checked)} />{' '}
          <Translated vi="rào chắn sau vòng lặp (dòng 11)">barrier after the loop (line 11)</Translated>
        </label>
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.8rem' }} data-handoff-table>
        <tbody>
          <tr>
            <td style={{ paddingLeft: '0.6rem' }}>
              <Translated vi="số vòng lặp bộ nhớ chia sẻ (các bước nhảy)">shared-memory iterations (strides)</Translated>
            </td>
            <td className="num" data-k="strides">{h.strides.length ? h.strides.join(', ') : tr('none', 'không có')}</td>
          </tr>
          <tr>
            <td style={{ paddingLeft: '0.6rem' }}>
              <Translated vi="số lần __syncthreads() được thực thi">__syncthreads() executed</Translated>
            </td>
            <td className="num" data-k="barriers">{h.barriers}</td>
          </tr>
          <tr>
            <td style={{ paddingLeft: '0.6rem' }}>
              <Translated vi="số lần ghi / đọc bộ nhớ chia sẻ">shared-memory writes / reads</Translated>
            </td>
            <td className="num" data-k="shared">{h.sharedWrites} / {h.sharedReads}</td>
          </tr>
          <tr>
            <td style={{ paddingLeft: '0.6rem' }}>
              <Translated vi="số bước shuffle trong warp 0">shuffle steps in warp 0</Translated>
            </td>
            <td className="num" data-k="shuffles">{h.shuffleSteps}</td>
          </tr>
          <tr>
            <td style={{ paddingLeft: '0.6rem' }}>
              <Translated vi={<>kết quả (đáp án đúng {h.expected})</>}>result (right answer {h.expected})</Translated>
            </td>
            <td className="num" data-k="result">
              <span className={'chip ' + (h.correct ? 'ok' : 'fail')}>{h.result == null ? tr('undefined', 'không xác định') : h.result}</span>
            </td>
          </tr>
        </tbody>
      </table>
      <svg viewBox={`0 0 ${VIEW_W} ${svgH}`} role="img" aria-label={tr('Shared memory at the hand-off', 'Bộ nhớ chia sẻ tại lúc bàn giao')} style={{ width: '100%', maxWidth: VIEW_W, minWidth: MIN_W, display: 'block' }}>
        {range(slots).map((k) => {
          const x = 8 + (k % perRow) * cw;
          const y = Math.floor(k / perRow) * RH + 12;
          const s = state(k);
          const fill = s === 'used' ? 'var(--shared)' : s === 'idle' ? 'var(--cell)' : 'var(--fail-wash)';
          const stroke = s === 'used' ? 'var(--shared)' : s === 'idle' ? 'var(--rule)' : 'var(--fail)';
          return (
            <g key={k} data-slot={k} data-state={s}>
              {k % 8 === 0 ? <text x={x + cw / 2} y={y - 3} fontSize="8.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{k}</text> : null}
              <rect x={x + 0.6} y={y} width={cw - 1.2} height={22} rx="2" fill={fill} stroke={stroke} />
              <text x={x + cw / 2} y={y + 14.5} fontSize={cw < 22 ? 8 : 9.5} textAnchor="middle"
                style={{ fill: s === 'used' ? 'var(--page)' : s === 'idle' ? 'var(--ink-3)' : 'var(--fail)' }}>{h.handoff.values[k] ?? ''}</text>
            </g>
          );
        })}
      </svg>
      <p className="narrate" role="status" style={{ margin: '0.6rem 0 0' }} data-handoff-says>
        {variant === 'printed'
          ? tr(
              `The last shared iteration ${h.strides.length ? `(stride ${h.strides[h.strides.length - 1]})` : '(none ran: the block is one warp)'} leaves ${W} partial sums, exactly one for each lane of warp 0.`,
              `Vòng lặp bộ nhớ chia sẻ cuối cùng ${h.strides.length ? `(bước nhảy ${h.strides[h.strides.length - 1]})` : '(không vòng nào chạy: khối chỉ có một warp)'} để lại ${W} tổng bộ phận, đúng một tổng cho mỗi lane của warp 0.`,
            )
          : variant === 'early'
            ? dropped
              ? tr(
                  `${dropped} slots still hold partial sums that no lane reads: half the input is silently dropped. Nothing crashes and nothing warns.`,
                  `${dropped} ô vẫn giữ các tổng bộ phận mà không lane nào đọc: một nửa đầu vào bị bỏ đi trong im lặng. Không có gì sập và không có gì cảnh báo.`,
                )
              : tr(
                  `With a block of one warp the loop never runs, so stopping early changes nothing here.`,
                  `Với một khối chỉ có một warp, vòng lặp không bao giờ chạy, nên dừng sớm không thay đổi gì ở đây.`,
                )
            : variant === 'double'
              ? tr(
                  `Only slot 0 still holds a partial sum. Warp 0 also reads ${again} slots whose values are already inside slot 0, and counts them again.`,
                  `Chỉ ô 0 còn giữ một tổng bộ phận. Warp 0 còn đọc ${again} ô có giá trị đã nằm trong ô 0, và đếm chúng thêm lần nữa.`,
                )
              : h.pastEnd
                ? tr(
                    `This block has only one warp's worth of slots, so lanes read past the end of input_s.`,
                    `Khối này chỉ có số ô bằng một warp, nên các lane đọc vượt quá cuối input_s.`,
                  )
                : race
                  ? tr(
                      `Without line 11, thread ${race.thread} reads input_s[${race.slot}], which thread ${race.other} wrote in the same iteration with no barrier between: a data race, and an undefined result.`,
                      `Không có dòng 11, luồng ${race.thread} đọc input_s[${race.slot}], ô mà luồng ${race.other} đã ghi trong cùng vòng lặp mà không có rào chắn ở giữa: một race condition (tranh chấp dữ liệu), và một kết quả không xác định.`,
                    )
                  : tr(
                      `Each lane adds two slots that other threads wrote, so line 11 is what makes their writes visible. Untick it.`,
                      `Mỗi lane cộng hai ô mà các luồng khác đã ghi, nên dòng 11 chính là thứ làm cho các lần ghi đó hiển thị. Hãy bỏ chọn nó.`,
                    )}
        {variant !== 'early2' && !line11 && h.races.length === 0
          ? tr(
              ' Line 11 is off, and nothing changed: every lane of warp 0 reads only a slot it wrote itself.',
              ' Dòng 11 đã tắt, và không có gì thay đổi: mỗi lane của warp 0 chỉ đọc ô mà chính nó đã ghi.',
            )
          : ''}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- code
function WarpReduceListing() {
  const status = useLabStatus(LAB);
  const printed = useLabFile(AS_PRINTED);
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  const fn = !missing(cu) ? functionSource(cu!, 'warp_reduce') : null;
  const mask = !missing(cu) ? /#if defined\(__HIPCC__\)[\s\S]*?#endif/.exec(cu!)?.[0] ?? null : null;
  const mfn = !missing(metal) ? functionSource(metal!, 'warp_reduce') : null;
  if (missing(printed)) return null;
  const tabs = [
    { label: tr('as printed (CUDA only)', 'như khi in (chỉ CUDA)'), code: printed!, tiers: TIERS, status: STATUS.printed },
    ...(fn ? [{ label: tr('portable (lab 10.1)', 'khả chuyển (lab 10.1)'), code: (mask ? mask + '\n\n' : '') + fn.text, tiers: TIERS, status: status.cu }] : []),
    ...(mfn ? [{ label: tr('Metal (lab 10.1)', 'Metal (lab 10.1)'), code: mfn.text, tiers: TIERS, status: status.metal }] : []),
  ];
  return (
    <Code
      tabs={tabs}
      caption={
        <Translated
          vi={
            <>
              Tab đầu: Fig. 10.13 và các hàm trợ giúp của pp. 238–239 như khi in (<code>labs/{AS_PRINTED}</code>). Các tab còn
              lại là lời giải tham chiếu (solution) của lab 10.1 (<code>labs/{LAB}/solution/</code>): một kiểu mask 32 bit cho
              CUDA và 64 bit cho HIP, và một vòng lặp bắt đầu từ <code>warpSize / 2</code>; trên Metal, không có mask và độ rộng
              được truyền vào từ <code>[[threads_per_simdgroup]]</code>.
              {fn ? null : <> Các tệp của lab chưa có trên máy này, nên chỉ tab đầu được hiển thị.</>}
            </>
          }
        >
          <>
            First tab: Fig. 10.13 and the helpers of pp. 238–239 as printed (<code>labs/{AS_PRINTED}</code>). The others are
            lab 10.1's reference solution (<code>labs/{LAB}/solution/</code>): a mask type that is 32 bits for CUDA and 64 for
            HIP, and a loop that starts at <code>warpSize / 2</code>; on Metal, no mask and a width passed in from{' '}
            <code>[[threads_per_simdgroup]]</code>.
            {fn ? null : <> The lab's files are not on this machine yet, so only the first tab is shown.</>}
          </>
        </Translated>
      }
    />
  );
}

function HybridListing() {
  const src = useLabFile(HYBRID);
  const k = !missing(src) ? functionSource(src!, 'WarpLevelSumReductionKernel') : null;
  if (!k) return null;
  return (
    <Code
      tabs={[{ label: tr('Fig. 10.14, corrected', 'Fig. 10.14, đã sửa'), code: k.text, tiers: TIERS, status: STATUS.snippet }]}
      highlight={[5, 11, 12, 13, 14]}
      caption={
        <Translated
          vi={
            <>
              Kernel của sách với số dòng của nó, từ <code>labs/{HYBRID}</code>, nơi <code>warp_reduce</code> khả chuyển ở trên
              và một <code>warpIdx()</code> dựa trên <code>warpSize</code> nằm bên cạnh. Được đánh dấu: vòng lặp dừng ở độ rộng
              warp (dòng 5), rào chắn sau nó (dòng 11) và phần bàn giao (dòng 12 đến 14).
            </>
          }
        >
          <>
            The book's kernel with its line numbers, from <code>labs/{HYBRID}</code>, where the portable{' '}
            <code>warp_reduce</code> above and a <code>warpIdx()</code> on <code>warpSize</code> sit beside it. Marked: the
            loop that stops at the warp width (line 5), the barrier after it (line 11) and the hand-off (lines 12 to 14).
          </>
        </Translated>
      }
    />
  );
}

// ---------------------------------------------------------------- questions
export const questionsS107: Record<string, Question> = {
  'ch10/10.7/q-top-lanes': {
    kind: 'numeric',
    id: 'ch10/10.7/q-top-lanes',
    prompt: (
      <Translated
        vi={
          <>
            Một warp 32 lane, lane <i>i</i> giữ <i>i</i> trong <code>v</code>. Mọi lane chạy{' '}
            <code>v = __shfl_down_sync(0xffffffff, v, 16)</code>. Sau đó lane 20 giữ giá trị gì?
          </>
        }
      >
        <>
          A 32-lane warp, lane <i>i</i> holding <i>i</i> in <code>v</code>. Every lane runs{' '}
          <code>v = __shfl_down_sync(0xffffffff, v, 16)</code>. What does lane 20 hold afterwards?
        </>
      </Translated>
    ),
    answer: 20,
    explain: (
      <Translated
        vi={
          <p>
            Nguồn của lane 20 sẽ là lane 36, vượt quá cuối warp. Lane nguồn không được quay vòng, nên 16 lane phía trên giữ
            giá trị của chính chúng: 20. Đó là lý do, trong <code>warp_reduce</code>, các lane phía trên cộng giá trị của chính
            chúng vào chính chúng và tính những tổng không ai dùng.
          </p>
        }
      >
        <p>
          Lane 20's source would be lane 36, past the end of the warp. The source lane is not wrapped around, so the upper 16
          lanes keep their own value: 20. That is why, in <code>warp_reduce</code>, the top lanes add their own value to
          themselves and compute sums nobody uses.
        </p>
      </Translated>
    ),
    source: 'CUDA guide 5.4.6 (notes/today/ch10-facts.md A3); book Fig. 10.11, p. 238',
    vi: { source: 'CUDA guide 5.4.6 (notes/today/ch10-facts.md A3); sách Fig. 10.11, p. 238' },
  },
  'ch10/10.7/q-width8': {
    kind: 'numeric',
    id: 'ch10/10.7/q-width8',
    prompt: (
      <Translated
        vi={
          <>
            Cùng warp đó, lane <i>i</i> giữ <i>i</i>, nhưng giờ là <code>v = __shfl_down_sync(0xffffffff, v, 2, 8)</code>. Lane
            14 nhận được gì?
          </>
        }
      >
        <>
          Same warp, lane <i>i</i> holding <i>i</i>, but now <code>v = __shfl_down_sync(0xffffffff, v, 2, 8)</code>. What does
          lane 14 receive?
        </>
      </Translated>
    ),
    answer: 14,
    explain: (
      <Translated
        vi={
          <p>
            Width 8 chia warp thành bốn nhóm tám lane hoạt động như các warp riêng biệt. Lane 14 là vị trí 6 của nhóm 8 đến 15;
            nguồn của nó sẽ là vị trí 8, nằm ngoài nhóm, nên nó giữ giá trị của chính nó, 14. Lane 13 (vị trí 5) nhận lane 15.
          </p>
        }
      >
        <p>
          Width 8 splits the warp into four groups of eight that behave as separate warps. Lane 14 is position 6 of the group
          8 to 15; its source would be position 8, outside the group, so it keeps its own value, 14. Lane 13 (position 5)
          gets lane 15.
        </p>
      </Translated>
    ),
    source: 'book p. 241; CUDA guide 5.4.6 (ch10-facts A3)',
    vi: { source: 'sách p. 241; CUDA guide 5.4.6 (ch10-facts A3)' },
  },
  'ch10/10.7/q-syncwarp': {
    kind: 'choice',
    id: 'ch10/10.7/q-syncwarp',
    prompt: (
      <Translated
        vi={
          <>
            Một hướng dẫn cũ kết thúc phép tổng của khối chỉ bằng warp cuối. Vòng lặp bộ nhớ chia sẻ đã để lại 64 tổng bộ phận
            trong <code>input_s</code>, và rồi, không có rào chắn và không có lời gọi <code>_sync</code> nào:
            <pre className="mono" style={{ margin: '0.5rem 0', whiteSpace: 'pre-wrap' }}>
              {'if (t < 32) {\n  volatile float* s = input_s;\n  s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];\n  s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];\n}'}
            </pre>
            Trên một thiết bị có lập lịch luồng độc lập, tức mọi thiết bị trong các bảng hiện hành của NVIDIA, lane 0 có thể
            kết thúc với giá trị gì?
          </>
        }
      >
        <>
          An older tutorial finishes a block sum with the last warp alone. The shared-memory loop has left 64 partial sums
          in <code>input_s</code>, and then, with no barrier and no <code>_sync</code> call:
          <pre className="mono" style={{ margin: '0.5rem 0', whiteSpace: 'pre-wrap' }}>
            {'if (t < 32) {\n  volatile float* s = input_s;\n  s[t] += s[t + 32]; s[t] += s[t + 16]; s[t] += s[t + 8];\n  s[t] += s[t + 4];  s[t] += s[t + 2];  s[t] += s[t + 1];\n}'}
          </pre>
          On a device with independent thread scheduling, which is every device in NVIDIA's current tables, what can lane 0
          end up with?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một tổng sai: các lane có thể chạy trước, nên lane 0 có thể đọc s[16] trước khi lane 16 cộng s[48] vào nó, hoặc sau khi lane 16 đã cộng cả s[32].">
            A wrong sum: lanes may run ahead, so lane 0 can read s[16] before lane 16 adds s[48] into it, or after it has also added s[32].
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Tổng đúng: volatile buộc mọi lần đọc và ghi đi tới bộ nhớ chia sẻ theo thứ tự chương trình.">
            The right sum: volatile forces every read and write to go to shared memory in program order.
          </Translated>
        ),
        why: (
          <Translated vi="volatile giữ các giá trị khỏi nằm trong thanh ghi. Nó không sắp thứ tự gì giữa lane này với lane khác.">
            volatile keeps the values out of registers. It orders nothing between one lane and another.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Tổng đúng: một warp phát một lệnh cho tất cả các lane của nó.">
            The right sum: a warp issues one instruction for all its lanes.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là giả định mà đoạn mã được viết dựa trên. NVIDIA nói lập lịch luồng độc lập làm nó mất hiệu lực, và nêu tên chính mẫu này.">
            That is the assumption the code was written on. NVIDIA says independent thread scheduling makes it invalid, and names this very pattern.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không gì cả: không có rào chắn hay lời gọi _sync, warp bị deadlock ngay bước đầu.">
            Nothing: without a barrier or a _sync call, the warp deadlocks at its first step.
          </Translated>
        ),
        why: (
          <Translated vi="Không có gì trong đó phải chờ, nên không gì có thể treo. Các lần đọc chỉ đơn giản là tranh chấp với các lần ghi.">
            Nothing in it waits, so nothing can hang. The reads simply race the writes.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Hướng dẫn của NVIDIA nói các "synchronization-free intra-warp reductions" viết cho GPU trước compute capability 7.0
            phải được xem lại, và đồng bộ hóa bằng <code>__syncwarp()</code>. Vì <code>__syncwarp()</code> chỉ đảm bảo các lane
            gặp nhau tại lời gọi, đoạn mã này sẽ cần một lời gọi giữa các lần đọc và các lần ghi của mỗi bước, và một lời gọi
            nữa trước khi bước sau đọc (cách chúng tôi đọc quy tắc). Phép rút gọn bằng shuffle ở phần sau của mục này không cần
            gì như vậy: mỗi <code>__shfl_down_sync</code> nêu tên các lane của nó và trao đổi thanh ghi trong một lời gọi.
          </p>
        }
      >
        <p>
          NVIDIA's guide says "synchronization-free intra-warp reductions" written for GPUs before compute capability 7.0
          must be revisited, and synchronised with <code>__syncwarp()</code>. Since <code>__syncwarp()</code> guarantees only
          that the lanes meet at the call, this code would need one between each step's reads and its writes, and another
          before the next step reads (our reading of the rule). The shuffle reduction later in this section needs none of
          that: each <code>__shfl_down_sync</code> names its lanes and exchanges registers in one call.
        </p>
      </Translated>
    ),
    source: 'CUDA guide 3.2.2.1.1, 5.4.4.2 (notes/today/ch10-facts.md A2, A5); book p. 237',
    vi: { source: 'CUDA guide 3.2.2.1.1, 5.4.4.2 (notes/today/ch10-facts.md A2, A5); sách p. 237' },
  },
  'ch10/10.7/q-early-return': {
    kind: 'choice',
    id: 'ch10/10.7/q-early-return',
    prompt: (
      <Translated
        vi={
          <>
            Một phép rút gọn mức warp trên n = 50 phần tử, mỗi luồng một phần tử, hai warp. Mỗi luồng bắt đầu bằng{' '}
            <code>if (threadIdx.x &gt;= n) return;</code> rồi gọi <code>warp_reduce</code>. Điều gì sai, và cách sửa là gì?
          </>
        }
      >
        <>
          A warp-level reduction over n = 50 elements, one per thread, two warps. Each thread starts with{' '}
          <code>if (threadIdx.x &gt;= n) return;</code> and then calls <code>warp_reduce</code>. What is wrong, and what is the
          fix?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không gì cả: mask đầy đủ bỏ qua các luồng đã thoát, nên warp 1 cộng 18 giá trị còn sống của nó và lane 0 của nó giữ tổng.">
            Nothing: the full mask ignores exited threads, so warp 1 sums its 18 live values and its lane 0 holds their total.
          </Translated>
        ),
        why: (
          <Translated vi="Mask thì ổn với các lane đã thoát. Đọc giá trị của chúng thì không.">
            The mask is fine with exited lanes. Reading their values is not.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Các lane của warp 1 đọc những lane đã thoát, và các giá trị không xác định đó tới được lane 0. Hãy giữ mọi lane trong các lệnh shuffle, cho các lane không có phần tử giá trị 0.">
            Lanes of warp 1 read lanes that returned, and those undefined values reach lane 0. Keep every lane in the shuffles, giving lanes with no element 0.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Kernel bị deadlock, vì các lane đã thoát không bao giờ tới lệnh shuffle mà mask đầy đủ chờ.">
            The kernel deadlocks, because the lanes that returned never reach the shuffle the full mask waits for.
          </Translated>
        ),
        why: (
          <Translated vi="Các lane đã thoát không chặn một lời gọi _sync; chúng làm cho các giá trị đọc từ chúng trở nên không xác định.">
            Exited lanes do not block a _sync call; they make the values read from them undefined.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Dùng một mask nhỏ hơn chỉ nêu các lane còn sống, 18 bit thấp trong warp 1; khi đó mọi lần đọc đều xác định.">
            Use a smaller mask naming only the live lanes, the low 18 bits in warp 1; then every read is defined.
          </Translated>
        ),
        why: (
          <Translated vi="Một lane nằm ngoài mask cũng không tham gia, nên đọc nó cũng không xác định y như vậy.">
            A lane outside the mask is not taking part either, so reading it is just as undefined.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Với mask thì "exited threads are ignored", nhưng "if the target thread is inactive, the retrieved value is
            undefined". Trong warp 1, các luồng 50 đến 63 (lane 18 đến 31) đã thoát; ở bước nhảy 16, lane 2 đọc lane 18, các lane
            2 đến 15 đều nhận giá trị không xác định, và tới bước thứ hai đã có một giá trị như vậy tới lane 0. Mọi lane phải
            tham gia, giữ phần tử đơn vị của toán tử (0 cho phép tổng, giá trị nhỏ nhất cho max) khi nó không có dữ liệu. Hãy thử
            trong trình chạy từng bước: "các lane từ n trở đi thoát sớm", rồi "ở lại và đóng góp 0".
          </p>
        }
      >
        <p>
          "Exited threads are ignored" for the mask, but "if the target thread is inactive, the retrieved value is
          undefined". In warp 1, threads 50 to 63 (lanes 18 to 31) have returned; at stride 16 lane 2 reads lane 18, lanes 2
          to 15 all get undefined values, and by the second step one has reached lane 0. Every lane must take part, holding the identity of the
          operator (0 for a sum, the lowest value for a max) when it has no data. Try it in the stepper: "lanes from n on
          returned early", then "stay and contribute 0".
        </p>
      </Translated>
    ),
    source: 'CUDA guide 5.4.6.5 to 5.4.6.6 (ch10-facts A3); model: src/sim/shuffle.ts',
    vi: { source: 'CUDA guide 5.4.6.5 đến 5.4.6.6 (ch10-facts A3); mô hình: src/sim/shuffle.ts' },
  },
  'ch10/10.7/q-line11': {
    kind: 'choice',
    id: 'ch10/10.7/q-line11',
    prompt: (
      <Translated
        vi={
          <>
            Trong Fig. 10.14, <code>__syncthreads()</code> ở dòng 11, ngay sau vòng lặp, có cần cho một kết quả đúng không?
          </>
        }
      >
        <>
          In Fig. 10.14, is the <code>__syncthreads()</code> on line 11, right after the loop, needed for a correct result?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có: thiếu nó, warp 0 có thể nạp input_s trước khi vòng lặp cuối ghi vào.">
            Yes: without it warp 0 could load input_s before the last iteration has written it.
          </Translated>
        ),
        why: <Translated vi="Luồng nào đã ghi ô mà lane t nạp?">Which thread wrote the slot that lane t loads?</Translated>,
      },
      {
        label: (
          <Translated vi="Không: với cách đánh chỉ số này, mỗi lane của warp 0 chỉ nạp ô mà chính nó đã ghi ở vòng lặp cuối.">
            No: as indexed, each lane of warp 0 loads only the slot it wrote in the last iteration.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không: các lệnh shuffle trong warp_reduce đồng bộ hóa warp, và điều đó bao gồm cả bộ nhớ chia sẻ.">
            No: the shuffles in warp_reduce synchronise the warp, and that covers shared memory.
          </Translated>
        ),
        why: (
          <Translated vi='Các lệnh shuffle hoàn toàn không sắp thứ tự truy cập bộ nhớ ("these intrinsics do not provide any memory ordering").'>
            Shuffles do not order memory accesses at all ("these intrinsics do not provide any memory ordering").
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Có: cần một rào chắn khối trước khi bất kỳ warp nào shuffle các giá trị nó đã nạp từ bộ nhớ chia sẻ.">
            Yes: a block barrier is needed before any warp shuffles values it loaded from shared memory.
          </Translated>
        ),
        why: (
          <Translated vi="Các lệnh shuffle di chuyển thanh ghi; chúng không cần rào chắn khối.">
            Shuffles move registers; they do not need a block barrier.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <>
            <p>
              Vòng lặp cuối có bước nhảy 32 (độ rộng warp): các luồng 0 đến 31, đúng là warp 0, ghi <code>input_s[t]</code>, và ở
              dòng 13 luồng <code>t</code> nạp <code>input_s[t]</code>. Một luồng đọc chính lần ghi của nó thì không cần rào
              chắn. Nếu vòng lặp không bao giờ chạy (khối chỉ có một warp), mỗi luồng đọc thứ nó đã ghi ở dòng 04. Rào chắn bên
              trong vòng lặp mới là thứ bảo vệ các lần đọc ô của luồng khác.
            </p>
            <p>
              Nó sẽ cần thiết ngay khi phần bàn giao đọc những ô mà luồng khác đã ghi: chọn "sớm một tầng, mỗi lane cộng các ô
              t và t + width" trong trình khám phá và bỏ chọn dòng 11. Cả hai khẳng định đã được kiểm tra, bằng mô hình ở mọi
              kích thước khối và độ rộng nó cung cấp, và trên trình giả lập của khóa học dưới ba thứ tự luồng (
              <code>labs/ch10/snippets/check/s107_hybrid_test.cu</code>). Giữ rào chắn chỉ tốn một rào chắn và vô hại.
            </p>
          </>
        }
      >
        <>
          <p>
            The last iteration has stride 32 (the warp width): threads 0 to 31, exactly warp 0, write{' '}
            <code>input_s[t]</code>, and on line 13 thread <code>t</code> loads <code>input_s[t]</code>. A thread reading its
            own write needs no barrier. If the loop never runs (a block of one warp), each thread reads what it wrote on line
            04. The barrier inside the loop is the one that protects reads of other threads' slots.
          </p>
          <p>
            It would be needed the moment the hand-off read slots other threads wrote: choose "one level early, each lane adds
            slots t and t + width" in the explorer and untick line 11. Both claims were checked, by the model at every block
            size and width it offers, and on the course emulator under three thread orders (
            <code>labs/ch10/snippets/check/s107_hybrid_test.cu</code>). Keeping the barrier costs one barrier and is harmless.
          </p>
        </>
      </Translated>
    ),
    source: 'our analysis of Fig. 10.14 (p. 240), checked with src/sim/shuffle.ts and the emulator',
    vi: { source: 'phân tích của chúng tôi về Fig. 10.14 (p. 240), kiểm tra bằng src/sim/shuffle.ts và trình giả lập' },
  },
  'ch10/10.7/q-hip-mask': {
    kind: 'choice',
    id: 'ch10/10.7/q-hip-mask',
    prompt: (
      <Translated
        vi={
          <>
            Đoạn này biên dịch được cho CUDA:
            <pre className="mono" style={{ margin: '0.5rem 0', whiteSpace: 'pre-wrap' }}>
              {'__device__ float warp_sum(float v) {\n  for (unsigned s = warpSize / 2; s > 0; s /= 2)\n    v += __shfl_down_sync(0xffffffff, v, s);\n  return v;\n}'}
            </pre>
            Dựng cho HIP, nó hỏng ở đâu, và thay đổi nhỏ nhất nào giữ được một mã nguồn cho cả hai?
          </>
        }
      >
        <>
          This compiles for CUDA:
          <pre className="mono" style={{ margin: '0.5rem 0', whiteSpace: 'pre-wrap' }}>
            {'__device__ float warp_sum(float v) {\n  for (unsigned s = warpSize / 2; s > 0; s /= 2)\n    v += __shfl_down_sync(0xffffffff, v, s);\n  return v;\n}'}
          </pre>
          Built for HIP, where does it fail, and what is the smallest change that keeps one source for both?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Lúc biên dịch, trên mọi target AMD. Đặt kiểu mask theo từng hãng (64 bit dưới __HIPCC__) và truyền giá trị toàn bit 1 của kiểu đó.">
            At compile time, on every AMD target. Type the mask per vendor (64 bits under __HIPCC__) and pass the all-ones value of that type.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Lúc chạy trên các wavefront 64 lane, nơi các lane 32 đến 63 không có trong mask. Hãy viết 0xffffffffffffffff thay thế.">
            At run time on 64-lane wavefronts, where lanes 32 to 63 are not in the mask. Write 0xffffffffffffffff instead.
          </Translated>
        ),
        why: (
          <Translated vi="HIP không bao giờ đi tới bước chạy nó: kiểu của mask được kiểm tra lúc biên dịch. Và tham số mask của CUDA là unsigned, 32 bit.">
            HIP never gets as far as running it: the mask's type is checked at compile time. And CUDA's mask parameter is an unsigned, 32 bits.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ trên gfx10 trở về sau, vốn chạy wavefront rộng 32 nên cần một mask có độ rộng riêng.">
            Only on gfx10 and later, which run 32-wide wavefronts and so need a mask of their own width.
          </Translated>
        ),
        why: (
          <Translated vi="Phép kiểm tra là trên kiểu của mask, không phải giá trị, và nó áp dụng cả ở chế độ wave32.">
            The check is on the mask's type, not its value, and it applies in wave32 mode too.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không đâu cả: vòng lặp đã bắt đầu từ warpSize / 2, nên nó tự thích ứng với wavefront 64 lane.">
            Nowhere: the loop already starts at warpSize / 2, so it adapts to 64-lane wavefronts.
          </Translated>
        ),
        why: (
          <Translated vi="Vòng lặp thì khả chuyển; literal mask thì không.">The loop is portable; the mask literal is not.</Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mỗi hàm <code>_sync</code> của HIP có một <code>static_assert</code> rằng mask là số nguyên 64-bit ("Implicitly
            promoting a smaller integer is almost always an error"), trên mọi target AMD, kể cả wave32. Một kiểu chọn theo
            từng hãng giữ được một tệp: lab 10.1 gọi nó là <code>lane_mask_t</code>, cái tên mà ví dụ của chính AMD dùng, và
            truyền <code>~lane_mask_t(0)</code>, toàn bit 1; llama.cpp gọi nó là <code>ggml_cuda_thread_mask_t</code>. Vòng lặp
            này đã bắt đầu từ <code>warpSize / 2</code>; vòng lặp của sách bắt đầu từ 16, đó là câu hỏi tiếp theo.
          </p>
        }
      >
        <p>
          Each HIP <code>_sync</code> function has a <code>static_assert</code> that the mask is a 64-bit integer ("Implicitly
          promoting a smaller integer is almost always an error"), on every AMD target, wave32 included. A type chosen per
          vendor keeps one file: lab 10.1 calls it <code>lane_mask_t</code>, the name AMD's own example uses, and passes{' '}
          <code>~lane_mask_t(0)</code>, all ones; llama.cpp calls it <code>ggml_cuda_thread_mask_t</code>. This loop already starts at <code>warpSize / 2</code>; the book's
          starts at 16, which is the next question.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch10-facts.md B1, B2; infra-facts.md A3, B3', // i18n-ignore: citation only
  },
  'ch10/10.7/q-wave64': {
    kind: 'numeric',
    id: 'ch10/10.7/q-wave64',
    prompt: (
      <Translated
        vi={
          <>
            Giả sử mask đã được sửa nhưng vòng lặp của sách (các bước nhảy 16, 8, 4, 2, 1) chạy trên một wavefront AMD rộng 64,
            mọi lane giữ 1. Lane 0 trả về giá trị gì?
          </>
        }
      >
        <>
          Suppose the mask is fixed but the book's loop (strides 16, 8, 4, 2, 1) runs on a 64-wide AMD wavefront, every lane
          holding 1. What does lane 0 return?
        </>
      </Translated>
    ),
    answer: 32,
    explain: (
      <Translated
        vi={
          <p>
            Với các bước nhảy từ 16 trở xuống, tổng của lane 0 chỉ lấy từ các lane 0 đến 31. Lane 32 gom các lane 32 đến 63 theo
            cùng cách, nên mỗi lane giữ 32 và không lane nào giữ 64. Vòng lặp phải bắt đầu từ <code>warpSize / 2</code>; trường
            hợp "vòng lặp viết cho 32 lane, trên một wavefront rộng 64" của trình chạy từng bước cho thấy điều đó.
          </p>
        }
      >
        <p>
          With strides from 16 down, lane 0's sum only ever draws on lanes 0 to 31. Lane 32 collects lanes 32 to 63 in the
          same way, so each holds 32 and no lane holds 64. The loop must start at <code>warpSize / 2</code>; the stepper's
          case "the loop written for 32 lanes, on a 64-wide wavefront" shows it.
        </p>
      </Translated>
    ),
    source: 'ch10-facts B1 (our arithmetic from HIP\'s documented out-of-range rule); model: src/sim/shuffle.ts',
    vi: { source: 'ch10-facts B1 (phép tính của chúng tôi từ quy tắc ngoài phạm vi mà HIP ghi trong tài liệu); mô hình: src/sim/shuffle.ts' },
  },
  'ch10/10.7/q-xor': {
    kind: 'choice',
    id: 'ch10/10.7/q-xor',
    prompt: (
      <Translated
        vi={
          <>
            Một softmax trên một hàng attention 128 điểm số, một warp cho mỗi hàng. Mọi lane cần max của hàng để tính{' '}
            <code>exp(x - max)</code> cho các điểm số của nó. Phép rút gọn nào tránh được một chuyến qua bộ nhớ chia sẻ để phát
            tán giá trị đó?
          </>
        }
      >
        <>
          A softmax over an attention row of 128 scores, one warp per row. Every lane needs the row's maximum to compute{' '}
          <code>exp(x - max)</code> for its own scores. Which reduction avoids a trip through shared memory to broadcast it?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một xor hình cánh bướm (hoặc simd_max trên Metal): mọi lane kết thúc với giá trị max.">
            An xor butterfly (or simd_max on Metal): every lane ends with the maximum.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Fig. 10.13 với max thay cho +: lane 0 có max, và các lane khác có thể đọc nó.">
            Fig. 10.13 with max in place of +: lane 0 has the maximum, and the others can read it.
          </Translated>
        ),
        why: (
          <Translated vi="Đọc được, nhưng chỉ qua một lệnh shuffle nữa hoặc bộ nhớ chia sẻ: cây down chỉ để nó ở lane 0.">
            They can, but only through another shuffle or shared memory: the down tree leaves it in lane 0 alone.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Cách nào cũng được; cả hai đều để max ở mọi lane.">Either; both leave the maximum in every lane.</Translated>
        ),
        why: (
          <Translated vi="Chạy trình chạy từng bước với cả hai mẫu và nhìn hàng cuối.">
            Run the stepper with both patterns and look at the last row.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Trong hình cánh bướm, mọi lane trao đổi với một đối tác theo cả hai chiều, nên sau log2(32) = 5 bước mọi lane đều giữ
            kết quả, với cùng số lệnh shuffle. <code>warp_reduce_max</code> của llama.cpp chính là vòng lặp này với{' '}
            <code>fmaxf</code>; trên Metal, <code>simd_max</code> phát tán kết quả tới mọi lane đang hoạt động.
          </p>
        }
      >
        <p>
          In the butterfly every lane trades with a partner in both directions, so after log2(32) = 5 steps every lane holds
          the result, for the same number of shuffles. llama.cpp's <code>warp_reduce_max</code> is exactly this loop with{' '}
          <code>fmaxf</code>; on Metal, <code>simd_max</code> broadcasts the result to every active lane.
        </p>
      </Translated>
    ),
    source: 'ch10-facts D1, C1; CUDA guide 5.4.6.5 example 3',
    vi: { source: 'ch10-facts D1, C1; CUDA guide 5.4.6.5 ví dụ 3' },
  },
};

// ---------------------------------------------------------------- the section
export function S107() {
  return (
    <>
      <Translated
        vi={
          <p>
            Chạy kernel dùng bộ nhớ chia sẻ của <a href={href('ch10', '10.6')}>mục 10.6</a> trên một khối 1,024 luồng và theo
            dõi các vòng lặp cuối của nó. Từ bước nhảy 32 trở đi, tối đa một warp (luồng 0 đến 31) có việc, vậy mà mỗi vòng
            trong sáu vòng đó vẫn đi qua bộ nhớ chia sẻ và một <code>__syncthreads()</code> mà cả khối phải tới. Các luồng
            cùng một warp có cách hợp tác rẻ hơn. Các <b>primitive mức warp</b> của CUDA cho phép chúng trao đổi dữ liệu và
            đồng bộ hóa mà không cần bộ nhớ chia sẻ và không cần rào chắn khối.
          </p>
        }
      >
        <p>
          Run the shared-memory kernel of <a href={href('ch10', '10.6')}>section 10.6</a> on a block of 1,024 threads and
          follow its last iterations. From stride 32 on, at most one warp (threads 0 to 31) has work, yet each of those six
          iterations still goes through shared memory and a <code>__syncthreads()</code> that the whole block must reach.
          Threads of the same warp have a cheaper way to cooperate. CUDA's <b>warp-level primitives</b> let them exchange data and synchronise
          without shared memory and without a block barrier.
        </p>
      </Translated>
      <Source section="10.7" pages="237" />

      <h2>
        <Translated vi="Một rào chắn cho một warp">A barrier for one warp</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Đơn giản nhất trong số đó là <code>__syncwarp()</code>: mọi luồng của warp tới đó trước khi bất kỳ luồng nào đi
            tiếp, giống <code>__syncthreads()</code> cho một warp duy nhất. Sách nói thêm rằng nó làm cho các luồng của warp đã
            phân kỳ hội tụ lại. Tài liệu của NVIDIA hứa ít hơn: <code>__syncwarp(mask)</code> là một rào chắn đồng thời sắp thứ
            tự bộ nhớ cho các lane trong <code>mask</code>, và sự hội tụ được đảm bảo tại một lời gọi <code>_sync</code>, không
            phải sau nó. Vì vậy đừng dựa vào việc các lane luôn đi cùng nhịp; mỗi lần trao đổi giữa các lane cần lời gọi{' '}
            <code>_sync</code> riêng.
          </p>
        }
      >
        <p>
          The simplest of them is <code>__syncwarp()</code>: every thread of the warp reaches it before any goes on, like{' '}
          <code>__syncthreads()</code> for a single warp. The book adds that it makes threads of the warp that had diverged
          reconverge. NVIDIA's documentation promises less: <code>__syncwarp(mask)</code> is a barrier that also orders
          memory for the lanes in <code>mask</code>, and convergence is guaranteed at a <code>_sync</code> call, not after
          it. So do not rely on lanes staying in step; every exchange between lanes needs its own <code>_sync</code> call.
        </p>
      </Translated>
      <NoteGroup>
        <Source pages="237" />
        <Today checked="2026-11-20 (ch10-facts A2)">{/* i18n-ignore: date and citation only */}
          <Translated vi={<p>Mask "đảm bảo hội tụ đúng đắn trước khi phần cứng thực thi intrinsic".</p>}>
            <p>
              The mask "ensures proper convergence before the hardware executes the intrinsic".
            </p>
          </Translated>
        </Today>
      </NoteGroup>

      <h2>
        <Translated vi="Đọc thanh ghi của lane khác">Reading another lane's register</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một thanh ghi thuộc về một luồng. Cho tới giờ, trao một giá trị cho luồng khác cần ba bước: ghi nó vào bộ nhớ chia
            sẻ, chờ ở một rào chắn, để luồng kia đọc. Một <b>warp shuffle</b> làm điều đó trong một bước: các lane của một warp
            đọc trực tiếp thanh ghi của nhau.
          </p>
        }
      >
        <p>
          A register belongs to one thread. So far, handing a value to another thread has meant three moves: write it to
          shared memory, wait at a barrier, let the other thread read it. A <b>warp shuffle</b> does it in one: the lanes of a
          warp read each other's registers directly.
        </p>
      </Translated>
      <Source pages="237–238" />
      <Translated
        vi={
          <p>
            Sách dùng <code>T __shfl_down_sync(unsigned mask, T var, unsigned int delta)</code> (Fig. 10.11). Mọi lane được nêu
            trong <code>mask</code> gọi nó với <code>var</code> của riêng mình, và lane <i>i</i> nhận lại <code>var</code> của
            lane <i>i</i> + <code>delta</code>. Các lane được đánh số trong warp của chúng. Hai hàm trợ giúp của sách đặt tên
            cho các con số: <code>warpIdx()</code> là <code>threadIdx.x / WARP_SIZE</code>, warp mà luồng nằm trong, và{' '}
            <code>laneIdx()</code> là <code>threadIdx.x % WARP_SIZE</code>, vị trí của nó trong warp đó. Luồng 33 là lane 1 của
            warp 1.
          </p>
        }
      >
        <p>
          The book uses <code>T __shfl_down_sync(unsigned mask, T var, unsigned int delta)</code> (Fig. 10.11). Every lane
          named in <code>mask</code> calls it with its own <code>var</code>, and lane <i>i</i> gets back the <code>var</code>{' '}
          of lane <i>i</i> + <code>delta</code>. Lanes are numbered within their warp. The book's two helpers name the numbers:{' '}
          <code>warpIdx()</code> is <code>threadIdx.x / WARP_SIZE</code>, the warp a thread is in, and <code>laneIdx()</code>{' '}
          is <code>threadIdx.x % WARP_SIZE</code>, its place in that warp. Thread 33 is lane 1 of warp 1.
        </p>
      </Translated>
      <Source pages="238–239" figs={tr('Fig. 10.11', 'Hình 10.11')} />
      <Translated
        vi={
          <p>
            Fig. 10.11 chỉ bao trường hợp cả hai lane đều tham gia. Nó không nói lane 31 nhận gì khi hỏi lane 47, hay một lane
            nhận gì từ một lane láng giềng đã thoát. Hai câu trả lời đó quyết định một phép rút gọn đúng hay sai, nên hãy thử
            chúng trước khi đọc tiếp.
          </p>
        }
      >
        <p>
          Fig. 10.11 covers the case where both lanes take part. It does not say what lane 31 gets when it asks for lane 47,
          or what a lane gets from a neighbour that has already returned. Those two answers decide whether a reduction is
          right, so try them before reading on.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch10-facts A3; infra-facts A3)">{/* i18n-ignore: date and citation only */}
        <Translated
          vi={
            <p>
              Chữ ký hàm giờ có tham số thứ tư, <code>int width = warpSize</code>. Trình khám phá theo các quy tắc của NVIDIA
              cho cả hai trường hợp.
            </p>
          }
        >
          <p>
            The signature now has a fourth parameter, <code>int width = warpSize</code>. The explorer follows NVIDIA's
            rules for both cases.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Trong trình khám phá, lane <i>i</i> bắt đầu với giá trị <i>i</i>, nên thứ một lane nhận được cho biết nó đến từ lane
            nào. Bắt đầu với down và delta 16 và đọc lời giải thích cho lane 20. Rồi cho một lane thoát, hoặc bỏ một lane ra
            khỏi mask, và xem ai bị ảnh hưởng. Cuối cùng, đặt width bằng 8 và xem warp tách thành bốn nhóm.
          </p>
        }
      >
        <p>
          In the explorer, lane <i>i</i> starts with the value <i>i</i>, so what a lane receives names the lane it came from.
          Start with down and delta 16 and read the explanation for lane 20. Then make a lane return, or take one out of the
          mask, and see who is affected. Last, set the width to 8 and watch the warp split into four groups.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Một warp và một lời gọi, theo các quy tắc NVIDIA ghi trong tài liệu. HIP ghi điều tương tự cho down, up và xor, và
              không nói gì về một <code>srcLane</code> ngoài phạm vi; Metal không có mask và không có width. Không tính thời gian.
            </p>
          }
        >
          <p>
            One warp and one call, by the rules NVIDIA documents. HIP documents the same for down, up and xor, and says
            nothing about an out-of-range <code>srcLane</code>; Metal has no mask and no width. No timing.
          </p>
        </Translated>
      </Simplification>
      <ShuffleExplorer />
      <Ask q={questionsS107['ch10/10.7/q-top-lanes']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsS107['ch10/10.7/q-width8']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Một phép rút gọn warp trong năm bước">A warp reduction in five steps</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Fig. 10.13 dựng một phép tổng chỉ từ các lệnh shuffle. Mọi lane truyền vào giá trị của mình. Bước nhảy bắt đầu từ 16
            và chia đôi xuống 1, và ở mỗi bước, mọi lane cộng giá trị của lane nằm trên nó <code>stride</code> vị trí. Sau bước
            nhảy 16, các lane 0 đến 15 mỗi lane giữ tổng của hai giá trị; sau bước nhảy 8, các lane 0 đến 7 giữ tổng của bốn;
            sau bước nhảy 1, lane 0 giữ tổng của cả 32.
          </p>
        }
      >
        <p>
          Fig. 10.13 builds a sum from shuffles alone. Every lane passes in its value. The stride starts at 16 and halves to
          1, and at each step every lane adds the value of the lane <code>stride</code> above it. After stride 16, lanes 0 to
          15 each hold a sum of two values; after stride 8, lanes 0 to 7 hold sums of four; after stride 1, lane 0 holds the
          sum of all 32.
        </p>
      </Translated>
      <Source pages="239–240" figs={tr('Fig. 10.13', 'Hình 10.13')} />
      <Translated
        vi={
          <p>
            Hai lựa chọn trong đó là có chủ ý. Không có <code>if</code>: mọi lane chạy mọi lệnh shuffle và mọi phép cộng, vì một
            lane chỉ có thể đọc một lane láng giềng có tham gia. Và chỉ giá trị trả về của lane 0 là có nghĩa. Sách nói rõ điều
            đó; các lane khác trả về những tổng bị thiếu giá trị hoặc đếm hai lần.
          </p>
        }
      >
        <p>
          Two choices in it are deliberate. There is no <code>if</code>: every lane runs every shuffle and every addition,
          because a lane can only read a neighbour that takes part. And only lane 0's return value means anything. The book
          says so; the other lanes return sums with values missing or counted twice.
        </p>
      </Translated>
      <WarpReduceListing />
      <Translated
        vi={
          <p>
            Chạy từng bước nó ở bên dưới. Đếm xem bao nhiêu phép cộng tới được lane 0. Rồi chuyển sang xor hình cánh bướm, rồi
            làm hỏng nó: chạy vòng lặp 32 lane trên một wavefront rộng 64, và cho các lane quá cuối dữ liệu thoát sớm.
          </p>
        }
      >
        <p>
          Step through it below. Count how many of the additions reach lane 0. Then switch to the xor butterfly, and then
          break it: run the 32-lane loop on a 64-wide wavefront, and let lanes past the end of the data return early.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mỗi bước là một lệnh shuffle và một phép cộng cho mọi lane, thực thi cùng lúc. Mô hình không biết gì về thời gian
              hay cách phần cứng lập lịch warp; nó chính xác về giá trị mà mỗi lane giữ.
            </p>
          }
        >
          <p>
            Each step is one shuffle and one add for every lane, executed at once. The model knows nothing of timing or of how
            the hardware schedules the warp; it is exact about which value each lane holds.
          </p>
        </Translated>
      </Simplification>
      <ReduceStepper />
      <Translated
        vi={
          <p>
            Một warp đầy đủ tốn 160 phép toán lane (32 lane, 5 bước) để tạo ra một tổng chỉ cần 31 phép cộng. 129 phép còn lại
            là công việc thật mà kết quả không ai đọc. Mục 10.8 quyết định cố ý trả cái giá đó. Hoạt ảnh đi qua các bước đó
            thêm một lần, có vẽ hình nón.
          </p>
        }
      >
        <p>
          A full warp spends 160 lane-operations (32 lanes, 5 steps) to produce a sum that needs 31 additions. The other 129
          are real work whose results nobody reads. Section 10.8 decides to pay that price on purpose. The animation goes
          through the same steps once more, with the cone drawn in.
        </p>
      </Translated>
      <ShuffleScene />
      <Ask q={questionsS107['ch10/10.7/q-early-return']} />

      <h2>
        <Translated vi="Bộ nhớ chia sẻ trước, rồi một warp">Shared memory first, then one warp</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Kernel lai (Fig. 10.14) giữ cây dùng bộ nhớ chia sẻ, cùng các rào chắn của nó, chỉ khi còn hơn một warp có việc:
            vòng lặp chạy khi <code>stride &gt;= WARP_SIZE</code>. Vòng lặp cuối của nó, ở bước nhảy 32, để lại 32 tổng bộ phận
            trong <code>input_s[0]</code> đến <code>input_s[31]</code>. Warp 0 sau đó nạp chúng vào thanh ghi, mỗi lane một giá
            trị, và kết thúc bằng <code>warp_reduce</code>. Từ bước nhảy 16 trở đi không còn bộ nhớ chia sẻ và không còn rào
            chắn.
          </p>
        }
      >
        <p>
          The hybrid kernel (Fig. 10.14) keeps the shared-memory tree, with its barriers, only while more than one warp has
          work: the loop runs while <code>stride &gt;= WARP_SIZE</code>. Its last iteration, at stride 32, leaves 32 partial
          sums in <code>input_s[0]</code> to <code>input_s[31]</code>. Warp 0 then loads them into registers, one per lane,
          and finishes with <code>warp_reduce</code>. From stride 16 on there is no shared memory and no barrier.
        </p>
      </Translated>
      <NoteGroup>
        <Source pages="240–241" figs={tr('Figs. 10.12, 10.14', 'Hình 10.12, 10.14')} />
        <Aside>
          <Translated
            vi={
              <p>
                Fig. 10.12 vẽ một khối đồ chơi gồm 4 warp 4 luồng; một khối thật 1,024 luồng có 32 warp 32 luồng (p. 238). Phần
                văn bản dẫn lần ghi cuối là dòng 17 đến 19; trong hình nó là dòng 15 đến 17.
              </p>
            }
          >
            <p>
              Fig. 10.12 draws a toy block of 4 warps of 4 threads; a real block of 1,024 threads has 32 warps of 32 (p. 238).
              The text cites the final write as lines 17 to 19; in the figure it is lines 15 to 17.
            </p>
          </Translated>
        </Aside>
      </NoteGroup>
      <HybridListing />
      <Translated
        vi={
          <p>
            Bàn giao chính xác ở đâu là quyết định thiết kế duy nhất trong kernel này. Lựa chọn của sách là điểm mà một warp giữ
            mọi giá trị còn lại. Dời nó một tầng theo bất kỳ hướng nào và xem điều gì xảy ra với đáp án.
          </p>
        }
      >
        <p>
          Where exactly to hand over is the only design decision in this kernel. The book's choice is the point at which
          one warp holds every remaining value. Move it one level either way and see what happens to the answer.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Từng vòng lặp một, mọi luồng xong trước vòng kế tiếp. Một lần đọc ô mà luồng khác đã ghi, không có rào chắn ở
              giữa, được tính là một race. Các thứ tự luồng được để cho phép kiểm tra trên trình giả lập.
            </p>
          }
        >
          <p>
            One iteration at a time, all threads before the next. A read of a slot another thread wrote, with no barrier
            between, counts as a race. Thread orders are left to the emulator check.
          </p>
        </Translated>
      </Simplification>
      <HandoffExplorer />
      <Ask q={questionsS107['ch10/10.7/q-line11']} />

      <h2>
        <Translated vi="Các nhóm hẹp hơn, và phần còn lại của họ hàm">Narrower groups, and the rest of the family</Translated>
      </h2>
      <Translated
        vi={
          <p>
            <code>__shfl_down_sync</code> và các hàm họ hàng nhận một đối số cuối tùy chọn, độ rộng của nhóm lane hợp tác với
            nhau. Mặc định là cả warp. Với width 8, warp hoạt động như bốn nhóm tám lane độc lập, mỗi nhóm rút gọn giá trị của
            riêng mình: bốn hàng ngắn cho mỗi warp. Các lệnh shuffle khác là <code>__shfl_sync</code> (đọc một lane được nêu
            tên), <code>__shfl_up_sync</code> và <code>__shfl_xor_sync</code>. Ngoài shuffle, sách nêu ba họ hàm: hàm bỏ phiếu
            (lane nào thỏa một điều kiện; chương 12 dùng một hàm), hàm match (lane nào giữ cùng giá trị) và hàm reduce, rút gọn
            trên cả warp "for integer values".
          </p>
        }
      >
        <p>
          <code>__shfl_down_sync</code> and its relatives take an optional last argument, the width of the group of lanes
          that cooperate. It defaults to the whole warp. With width 8 the warp behaves as four independent groups of eight,
          each reducing its own values: four short rows per warp. The other shuffles are <code>__shfl_sync</code> (read a named
          lane), <code>__shfl_up_sync</code> and <code>__shfl_xor_sync</code>. Beyond shuffles the book names three families:
          vote functions (which lanes meet a condition; chapter 12 uses one), match functions (which lanes hold the same
          value) and reduce functions, which reduce across a warp "for integer values".
        </p>
      </Translated>
      <NoteGroup>
        <Source pages="241" />
        <Today checked="2026-11-20 (ch10-facts A4)">{/* i18n-ignore: date and citation only */}
          <Translated
            vi={
              <p>
                Họ hàm reduce cần compute capability 8.x và chỉ nhận số nguyên, nên một phép tổng warp <code>float</code> trên
                CUDA vẫn là cây shuffle.
              </p>
            }
          >
            <p>
              The reduce family needs compute capability 8.x and takes integers only, so a <code>float</code> warp sum on
              CUDA is still the shuffle tree.
            </p>
          </Translated>
        </Today>
      </NoteGroup>
      <Translated
        vi={
          <p>
            Có một mẫu bạn sẽ gặp trong mã cũ không thuộc họ nào trong số này. Nó kết thúc một phép rút gọn bằng warp cuối được
            trải vòng lặp trên bộ nhớ chia sẻ <code>volatile</code>, không rào chắn và không lời gọi <code>_sync</code>, tin
            rằng 32 lane luôn đi cùng nhịp. Chúng không còn phải như vậy nữa: trên mọi thiết bị trong các bảng hiện hành của
            NVIDIA, các lane có thể phân kỳ và hội tụ lại ở bất kỳ điểm nào. Đừng sao chép nó. Hãy tìm ra nó sai ở đâu.
          </p>
        }
      >
        <p>
          One pattern you will meet in older code belongs to none of these families. It finishes a reduction with the last
          warp unrolled over <code>volatile</code> shared memory, with no barrier and no <code>_sync</code> call, trusting
          the 32 lanes to stay in step. They no longer have to: on every device in NVIDIA's current tables, lanes can
          diverge and reconverge at any point. Do not copy it. Work out what it gets wrong.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch10-facts A5)">{/* i18n-ignore: date and citation only */}
        <Translated
          vi={
            <p>
              NVIDIA nêu tên "synchronization-free intra-warp reductions" là mã cần xem lại vì lập lịch luồng độc lập.
            </p>
          }
        >
          <p>
            NVIDIA names "synchronization-free intra-warp reductions" as code to revisit for independent thread scheduling.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS107['ch10/10.7/q-syncwarp']} />

      <h2>
        <Translated vi="Một mã nguồn, ba backend">One source, three backends</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Tới đây, quy tắc của khóa học rằng một tệp <code>.cu</code> biên dịch được cho cả CUDA và HIP không còn đúng. Nó đúng
            với luồng, khối, bộ nhớ chia sẻ, rào chắn, atomic legacy và các lời gọi runtime. Nó không đúng với các intrinsic warp{' '}
            <code>_sync</code>. <code>warp_reduce</code> của sách hỏng dưới HIP theo ba cách riêng biệt: literal mask 32-bit là
            lỗi biên dịch, vì HIP kiểm tra mọi mask phải là 64 bit; warp rộng 64 lane trên các thiết bị gfx9 của AMD, nên một
            vòng lặp bắt đầu từ 16 rút gọn riêng từng nửa; và <code>warpSize</code> của HIP không còn là hằng số lúc biên dịch kể
            từ ROCm 7.0, nên không thể dựa vào nó ở nơi cần một hằng số, chẳng hạn kích thước một mảng. Cách sửa nằm trong tab
            "khả chuyển (lab 10.1)" của đoạn mã <code>warp_reduce</code> ở trên: một kiểu mask chọn theo từng hãng, một vòng lặp
            từ <code>warpSize / 2</code>, và không có literal 32 nào.
          </p>
        }
      >
        <p>
          Here the course's rule that one <code>.cu</code> file compiles for CUDA and HIP stops being true. It holds for
          threads, blocks, shared memory, barriers, legacy atomics and runtime calls. It does not hold for the warp{' '}
          <code>_sync</code> intrinsics.
          The book's <code>warp_reduce</code> fails under HIP in three separate ways: the 32-bit mask literal is a compile
          error, because HIP checks that every mask is 64 bits; warps are 64 lanes wide on AMD's gfx9 devices, so a loop that
          starts at 16 reduces each half separately; and HIP's <code>warpSize</code> has not been a compile-time constant since
          ROCm 7.0, so it cannot be relied on where one is needed, such as the size of an array. The fix is in the "portable (lab 10.1)" tab of the <code>warp_reduce</code> listing above: a mask type chosen per vendor, a loop
          from <code>warpSize / 2</code>, and no literal 32.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Sách chỉ dùng CUDA. Các thông tin về HIP và Metal ở đây lấy từ tài liệu của AMD và Apple
              (notes/today/ch10-facts.md B1, B2, C1; infra-facts.md B3, C2, C4, C5). Chưa có gì được chạy trên phần cứng AMD.
            </p>
          }
        >
          <p>
            The book uses CUDA only. HIP and Metal facts here are from AMD's and Apple's documentation
            (notes/today/ch10-facts.md B1, B2, C1; infra-facts.md B3, C2, C4, C5). Nothing has run on AMD hardware.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Metal làm điều đó với ít thủ tục hơn. <code>simd_shuffle_down(data, delta)</code> không có mask và không có width.
            Một kernel nhận lane của nó và độ rộng SIMD group qua <code>[[thread_index_in_simdgroup]]</code> và{' '}
            <code>[[threads_per_simdgroup]]</code>. Và Metal có thứ mà CUDA thiếu cho float: <code>simd_sum</code> và{' '}
            <code>simd_max</code> rút gọn một <code>float</code> trên cả SIMD group trong một lời gọi (Apple7 trở về sau, tức mọi
            máy Mac dùng Apple silicon) và trao kết quả cho mọi lane đang hoạt động. Apple không ghi trong tài liệu thứ tự mà{' '}
            <code>simd_sum</code> cộng, nên kết quả của nó có thể khác một phép tổng CPU ở những bit cuối.
          </p>
        }
      >
        <p>
          Metal does it with less ceremony. <code>simd_shuffle_down(data, delta)</code> has no mask and no width. A kernel
          receives its lane and the SIMD-group width as <code>[[thread_index_in_simdgroup]]</code> and{' '}
          <code>[[threads_per_simdgroup]]</code>. And Metal has what CUDA lacks for floats: <code>simd_sum</code> and{' '}
          <code>simd_max</code> reduce a <code>float</code> across the SIMD-group in one call (Apple7 and later, so every Apple
          silicon Mac) and hand the result to every active lane. Apple does not document the order in which{' '}
          <code>simd_sum</code> adds, so its result may differ from a CPU sum in the last bits.
        </p>
      </Translated>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-backends>
        <thead>
          <tr>
            <th style={{ width: '9rem' }}><Translated vi="Nội dung">What</Translated></th>
            <th style={{ width: '27%' }}>CUDA</th>
            <th style={{ width: '27%' }}>HIP</th>
            <th>Metal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Độ rộng warp mà kernel có thể giả định">Warp width a kernel may assume</Translated></td>
            <td><Translated vi={<>32 trên mọi thiết bị được liệt kê; đọc <code>warpSize</code></>}><>32 on every listed device; read <code>warpSize</code></></Translated></td>
            <td><Translated vi="64 trên gfx9, 32 trên gfx10 trở về sau; không phải hằng số lúc biên dịch kể từ ROCm 7.0">64 on gfx9, 32 on gfx10 and later; not a compile-time constant since ROCm 7.0</Translated></td>
            <td><Translated vi={<>không có tài liệu; <code>[[threads_per_simdgroup]]</code> lúc chạy</>}><>undocumented; <code>[[threads_per_simdgroup]]</code> at run time</></Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Mask cả warp">Full-warp mask</Translated></td>
            <td><code>0xffffffff</code> (<code>unsigned</code>)</td>
            <td><Translated vi={<>phải là số nguyên 64-bit; literal 32-bit không qua được <code>static_assert</code></>}><>must be a 64-bit integer; the 32-bit literal fails a <code>static_assert</code></></Translated></td>
            <td><Translated vi="không có mask">no masks</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Shuffle down">Shuffle down</Translated></td>
            <td><Translated vi={<><code>__shfl_down_sync(mask, v, d, width)</code>; width mặc định là <code>warpSize</code></>}><><code>__shfl_down_sync(mask, v, d, width)</code>; width defaults to <code>warpSize</code></></Translated></td>
            <td><Translated vi={<>cùng tên, mask 64-bit; còn có <code>__shfl_down(v, d)</code></>}><>same name, 64-bit mask; also <code>__shfl_down(v, d)</code></></Translated></td>
            <td><Translated vi={<><code>simd_shuffle_down(v, d)</code>; chỉ có width qua <code>simd_shuffle_and_fill_down</code>, Apple8 trở về sau (không có trên M1)</>}><><code>simd_shuffle_down(v, d)</code>; a width only through <code>simd_shuffle_and_fill_down</code>, Apple8 and later (not M1)</></Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Lane có nguồn nằm quá cuối">A lane whose source is past the end</Translated></td>
            <td><Translated vi="giữ giá trị của chính nó">keeps its own value</Translated></td>
            <td><Translated vi="trả về giá trị của chính nó">returns its own value</Translated></td>
            <td><Translated vi="các lane phía trên không đổi">upper lanes unchanged</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Tổng hoặc max float dựng sẵn cho warp">Built-in float warp sum or max</Translated></td>
            <td><Translated vi={<>không có (<code>__reduce_*_sync</code> chỉ cho số nguyên)</>}><>none (<code>__reduce_*_sync</code> is integer only)</></Translated></td>
            <td><Translated vi="mặc định không có (số nguyên)">none by default (integer)</Translated></td>
            <td><Translated vi={<><code>simd_sum</code>, <code>simd_max</code>, Apple7 trở về sau</>}><><code>simd_sum</code>, <code>simd_max</code>, Apple7 and later</></Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Lane nào giữ kết quả">Which lanes hold the result</Translated></td>
            <td><Translated vi="lane 0 với shuffle down; tất cả với xor">lane 0 with shuffle down; all with xor</Translated></td>
            <td><Translated vi="như trên">same</Translated></td>
            <td><Translated vi="mọi lane đang hoạt động">all active lanes</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Rào chắn warp">Warp barrier</Translated></td>
            <td><code>__syncwarp()</code></td>
            <td><Translated vi={<><code>__syncwarp()</code>; dạng có mask cần 64 bit</>}><><code>__syncwarp()</code>; a masked form needs 64 bits</></Translated></td>
            <td><code>simdgroup_barrier(mem_flags::mem_threadgroup)</code></td>
          </tr>
        </tbody>
      </table>
      <Ask q={questionsS107['ch10/10.7/q-hip-mask']} />
      <Ask q={questionsS107['ch10/10.7/q-wave64']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Nơi bạn gặp điều này: các hàng attention ngắn">Where you meet this: short attention rows</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Attention tính một softmax trên mọi hàng điểm số, và các hàng có thể ngắn. Softmax persistent của PyTorch giao cho
            mỗi hàng một warp, và gói nhiều hàng vào một warp khi chúng đủ ngắn; chú thích của nó ghi rằng "only __shfl_
            instructions are required for reductions". Nhiều hàng độc lập trong một warp chính là trường hợp mà đối số width
            tồn tại để phục vụ. Tuy nhiên softmax cần max của hàng ở mọi lane, không chỉ ở lane 0, nên mã thực tế dùng hình cánh
            bướm: <code>warp_reduce_max</code> của llama.cpp là một vòng lặp xor quanh <code>fmaxf</code>, còn trên Metal{' '}
            <code>simd_max</code> tự phát tán. Mục 10.8 biến phép rút gọn warp thành một phép rút gọn trên cả khối, đúng dạng mà
            một softmax trên hàng dài hơn cần.
          </p>
        }
      >
        <p>
          Attention computes a softmax over every row of scores, and the rows can be short. PyTorch's persistent softmax
          gives each row one warp, and packs several rows into a warp when they are short enough; its comment notes that
          "only __shfl_ instructions are required for reductions". Several independent rows in one warp is the case the
          width argument exists for. Softmax needs the row's maximum in every lane, though, not only in lane 0, so production code
          uses the butterfly: llama.cpp's <code>warp_reduce_max</code> is an xor loop around <code>fmaxf</code>, and on Metal{' '}
          <code>simd_max</code> broadcasts by itself. Section 10.8 turns the warp reduction into a block-wide one, the shape a
          softmax over a longer row needs.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              <code>PersistentSoftmax.cuh</code> của PyTorch và <code>common.cuh</code> của llama.cpp như được đọc vào
              2026-11-20 (notes/today/ch10-facts.md D1, D4).
            </p>
          }
        >
          <p>
            PyTorch's <code>PersistentSoftmax.cuh</code> and llama.cpp's <code>common.cuh</code> as read on 2026-11-20
            (notes/today/ch10-facts.md D1, D4).
          </p>
        </Translated>
      </Beyond>
      <Ask q={questionsS107['ch10/10.7/q-xor']} />
    </>
  );
}
