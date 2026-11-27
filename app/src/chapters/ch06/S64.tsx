import { useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  BANK_PRESETS,
  EX5_STRIDES,
  gcdRule,
  halfCase,
  sameWordCase,
  strideCase,
  tileAccess,
  type BankPresetId,
  type BankResult,
} from '../../sim/banks';
import { TurnAndPadScene } from './TurnAndPadScene';

const LAB63 = 'ch06/lab3_banks';

/** Backends that are real GPUs (realBackends() keeps the emulator's race-detection flavour). */
const gpuBackends = (reports: Reports) =>
  realBackends(reports).filter((b) => b !== 'race' && !/emulator/i.test(String(reports[b].device?.name ?? '')));
function labRows(reports: Reports, backend: string, lab: string): BenchRow[] {
  const s = benchRows(reports[backend], lab, 'solution');
  return s.length ? s : benchRows(reports[backend], lab, 'work');
}

// ---------------------------------------------------------------- the bank explorer

type Pattern = 'stride' | 'tile' | 'same' | 'half';

interface BankState {
  preset: BankPresetId;
  pattern: Pattern;
  stride: number;
  T: number;
  pad: number;
  orientation: 'x-first' | 'y-first';
  warp: number;
  lane: number;
}
const START: BankState = { preset: 'nvidia', pattern: 'stride', stride: 1, T: 32, pad: 0, orientation: 'x-first', warp: 0, lane: 1 };

/** Rows of threads in a block T threads wide, keeping it within 1024 threads. */
const blockRows = (T: number) => Math.min(T, Math.floor(1024 / T));

function compute(s: BankState): { r: BankResult; banks: number; lanes: number; describe: (lane: number) => string } | null {
  const p = BANK_PRESETS[s.preset];
  if (p.banks == null || p.warpWidth == null) return null;
  const banks = p.banks;
  const lanes = p.warpWidth;
  if (s.pattern === 'stride') {
    const r = strideCase(s.stride, lanes, banks);
    return {
      r, banks, lanes,
      describe: (l) =>
        tr(
          `lane ${l}: a[${l}·${s.stride}] is word ${l * s.stride}, bank ${l * s.stride} mod ${banks} = ${r.lanes[l].bank}`,
          `lane ${l}: a[${l}·${s.stride}] là từ ${l * s.stride}, bank ${l * s.stride} mod ${banks} = ${r.lanes[l].bank}`,
        ),
    };
  }
  if (s.pattern === 'tile') {
    const first = s.warp * lanes;
    const r = tileAccess({ T: s.T, pitch: s.T + s.pad, orientation: s.orientation, ty0: Math.floor(first / s.T), tx0: first % s.T, warpWidth: lanes, banks });
    return {
      r, banks, lanes,
      describe: (l) => {
        const t = r.threads[l];
        return tr(
          `lane ${l} is thread (tx ${t.tx}, ty ${t.ty}): a[${t.row}][${t.col}] is word ${t.row}·${s.T + s.pad} + ${t.col} = ${t.word}, bank ${t.word} mod ${banks} = ${r.lanes[l].bank}`,
          `lane ${l} là luồng (tx ${t.tx}, ty ${t.ty}): a[${t.row}][${t.col}] là từ ${t.row}·${s.T + s.pad} + ${t.col} = ${t.word}, bank ${t.word} mod ${banks} = ${r.lanes[l].bank}`,
        );
      },
    };
  }
  if (s.pattern === 'same') {
    const r = sameWordCase(0, lanes, banks);
    return {
      r,
      banks,
      lanes,
      describe: (l) => tr(`lane ${l}: word 0, bank 0, the same word as every other lane`, `lane ${l}: từ 0, bank 0, cùng từ với mọi lane khác`),
    };
  }
  const r = halfCase(lanes, banks);
  return {
    r, banks, lanes,
    describe: (l) =>
      tr(
        `lane ${l}: a 2-byte element at byte ${l * 2}, inside word ${Math.floor(l / 2)}, bank ${r.lanes[l].bank}`,
        `lane ${l}: một phần tử 2 byte ở byte ${l * 2}, nằm trong từ ${Math.floor(l / 2)}, bank ${r.lanes[l].bank}`,
      ),
  };
}

function BankPicture({ r, banks, lane }: { r: BankResult; banks: number; lane: number }) {
  const W = 700;
  const cw = (W - 40) / banks;
  const rows = Math.max(1, r.passes);
  const dot = Math.max(4, Math.min(9, cw - 2, 260 / rows - 2));
  const H = 44 + rows * (dot + 2) + 20;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Banks and the lanes that land in each', 'Các bank và những lane rơi vào mỗi bank')} style={{ width: '100%', display: 'block' }}>
      <text x={20} y={12} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="các bank (mỗi bank một cột); mỗi từ khác nhau một chấm, xếp chồng theo thứ tự bank phục vụ chúng">
          banks (one column each); one dot per distinct word, stacked in the order the bank serves them
        </Translated>
      </text>
      {r.perBank.map((b) => (
        <g key={b.bank} data-bank={b.bank} data-lanes={b.lanes.length} data-words={b.words.length}>
          <rect x={20 + b.bank * cw} y={20} width={cw - 1} height={6} fill={b.lanes.length ? 'var(--shared)' : 'var(--rule)'} />
          {banks <= 32 || b.bank % 8 === 0 ? (
            <text x={20 + b.bank * cw + cw / 2} y={37} fontSize="8" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{b.bank}</text>
          ) : null}
          {b.words.map((w, i) => {
            const shared = r.lanes.filter((l) => l.word === w);
            const mine = shared.some((l) => l.lane === lane);
            return (
              <g key={w}>
                <rect x={20 + b.bank * cw + (cw - 1 - dot) / 2} y={44 + i * (dot + 2)} width={dot} height={dot} rx={dot / 2}
                  fill={b.words.length > 1 ? 'var(--fail)' : 'var(--shared)'} stroke={mine ? 'var(--ink)' : 'none'} strokeWidth={mine ? 2 : 0} />
                {shared.length > 1 && cw >= 14 ? (
                  <text x={20 + b.bank * cw + cw - 1} y={44 + i * (dot + 2) + dot - 1} fontSize="7" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>×{shared.length}</text>
                ) : null}
              </g>
            );
          })}
        </g>
      ))}
      <text x={20} y={H - 4} fontSize="10" style={{ fill: 'var(--ink-3)' }}>
        <Translated vi="Khoanh viền: lane bạn chọn. Đỏ: một bank phải giao nhiều hơn một từ. “×n”: n lane trên cùng một từ.">
          Ringed: the lane you picked. Red: a bank with more than one word to deliver. “×n”: n lanes on one word.
        </Translated>
      </text>
    </svg>
  );
}

function BankExplorer() {
  const [s, setS] = useState<BankState>(START);
  const set = (o: Partial<BankState>) => setS((x) => ({ ...x, ...o }));
  const preset = BANK_PRESETS[s.preset];
  const c = compute(s);
  const lanes = preset.warpWidth ?? 32;
  const lane = Math.min(s.lane, lanes - 1);
  const warpsInBlock = Math.max(1, Math.ceil((s.T * blockRows(s.T)) / lanes));
  return (
    <Bench
      title={tr('Bank explorer', 'Trình khám phá bank')}
      task={tr(
        'Find a stride with a 32-way conflict and one with none. Then break the corner-turned tile and fix it.',
        'Tìm một bước nhảy có xung đột 32 chiều và một bước nhảy không có xung đột nào. Rồi làm hỏng ô đã xoay góc và sửa nó.',
      )}
      foot={
        preset.banks == null ? (
          <Translated vi={<>Apple không công bố cấu trúc bank nào cho bộ nhớ threadgroup, nên không có gì để đếm. {preset.facts}.</>}>
            <>Apple documents no bank structure for threadgroup memory, so there is nothing to count. {preset.facts}.</>
          </Translated>
        ) : (
          <Translated
            vi={
              <>
                Quy tắc được công bố của NVIDIA: bank = chỉ số từ 4 byte mod số bank; số lượt = số từ khác nhau nhiều nhất mà
                một bank phải giao. Cùng một từ thì không xung đột. {preset.facts}.
              </>
            }
          >
            <>
              NVIDIA's documented rule: bank = 4-byte word index mod the number of banks; passes = the most distinct words
              any bank must deliver. Same word, no conflict. {preset.facts}.
            </>
          </Translated>
        )
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          GPU
          <select value={s.preset} onChange={(e) => set({ preset: e.target.value as BankPresetId, warp: 0 })}>
            {Object.values(BANK_PRESETS).map((p) => (
              <option key={p.id} value={p.id}>
                <Translated
                  vi={
                    {
                      nvidia: 'NVIDIA: 32 bank, 32 lane',
                      'amd-cdna': 'AMD CDNA 1–3: 32 bank, 64 lane',
                      'amd-cdna4': 'AMD CDNA 4: 64 bank, 64 lane',
                      'amd-rdna': 'AMD RDNA 2–4: 64 bank, 32 lane',
                      apple: 'Apple: không có tài liệu',
                    }[p.id]
                  }
                >
                  {p.label}
                </Translated>
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <Translated vi="Truy cập">Access</Translated>
          <select value={s.pattern} onChange={(e) => set({ pattern: e.target.value as Pattern, warp: 0 })}>
            <option value="stride">a[threadIdx.x * stride]</option>
            <option value="tile">{tr('a 2D tile, a[..][..]', 'một ô 2D, a[..][..]')}</option>
            <option value="same">{tr('every lane on one word', 'mọi lane trên một từ')}</option>
            <option value="half">{tr('2-byte elements, a[threadIdx.x]', 'phần tử 2 byte, a[threadIdx.x]')}</option>
          </select>
        </label>
        {s.pattern === 'stride' ? (
          <label className="field">
            stride = {s.stride}
            <input type="range" min={1} max={64} value={s.stride} aria-label={tr('stride', 'stride')} onChange={(e) => set({ stride: Number(e.target.value) })} />
          </label>
        ) : null}
        {s.pattern === 'tile' ? (
          <>
            <label className="field">
              <Translated vi="Độ rộng ô (và blockDim.x)">Tile width (and blockDim.x)</Translated>
              <select value={s.T} onChange={(e) => set({ T: Number(e.target.value), warp: 0 })}>
                {[16, 32, 64].map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
            <label className="field">
              <Translated vi={<>Số cột đệm = {s.pad}</>}>
                <>Padding columns = {s.pad}</>
              </Translated>
              <input type="range" min={0} max={4} value={s.pad} aria-label={tr('padding columns', 'số cột đệm')} onChange={(e) => set({ pad: Number(e.target.value) })} />
            </label>
            <label className="field">
              <Translated vi="Thứ tự chỉ số">Index order</Translated>
              <select value={s.orientation} onChange={(e) => set({ orientation: e.target.value as 'x-first' | 'y-first' })}>
                <option value="x-first">{tr('a[threadIdx.x][threadIdx.y] (the turned store)', 'a[threadIdx.x][threadIdx.y] (phép ghi đã xoay)')}</option>
                <option value="y-first">a[threadIdx.y][threadIdx.x]</option>
              </select>
            </label>
            <label className="field">
              <Translated vi="Warp của khối">Warp of the block</Translated>
              <select value={Math.min(s.warp, warpsInBlock - 1)} onChange={(e) => set({ warp: Number(e.target.value) })}>
                {Array.from({ length: warpsInBlock }, (_, i) => <option key={i} value={i}>{i}</option>)}
              </select>
            </label>
          </>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="muted">
          <Translated vi="Các bước nhảy của Bài tập 5:">Exercise 5 strides:</Translated>
        </span>
        <span className="seg" role="group" aria-label={tr('Exercise 5 strides', 'Các bước nhảy của Bài tập 5')}>
          {EX5_STRIDES.map((v) => (
            <button key={v} aria-pressed={s.pattern === 'stride' && s.stride === v} onClick={() => set({ pattern: 'stride', stride: v })}>{v}</button>
          ))}
        </span>
        <span className="muted">
          <Translated vi="Ô:">Tiles:</Translated>
        </span>
        <button className="quiet" onClick={() => set({ pattern: 'tile', T: 32, pad: 0, orientation: 'x-first', warp: 0 })}>
          <Translated vi="32 × 32, phép ghi đã xoay">32 × 32, turned store</Translated>
        </button>
        <button className="quiet" onClick={() => set({ pattern: 'tile', T: 32, pad: 1, orientation: 'x-first', warp: 0 })}>
          <Translated vi="đệm thành 33">padded to 33</Translated>
        </button>
        <button className="quiet" onClick={() => set({ pattern: 'tile', T: 16, pad: 0, orientation: 'x-first', warp: 0 })}>
          <Translated vi="16 × 16, phép ghi đã xoay">16 × 16, turned store</Translated>
        </button>
      </div>

      {c ? (
        <>
          <label className="field" style={{ marginBottom: '0.5rem' }}>
            <Translated vi={<>Lane = {lane}</>}>
              <>Lane = {lane}</>
            </Translated>
            <input type="range" min={0} max={lanes - 1} value={lane} aria-label={tr('lane', 'lane')} onChange={(e) => set({ lane: Number(e.target.value) })} />
          </label>
          <p className="narrate" style={{ margin: '0 0 0.7rem' }} data-lane-arith>{c.describe(lane)}.</p>
          <BankPicture r={c.r} banks={c.banks} lane={lane} />
          <div className="row" style={{ marginTop: '0.5rem' }} data-readout data-passes={c.r.passes} data-banks-used={c.r.banksUsed} data-max-lanes={c.r.maxLanesPerBank}>
            <span className={'chip ' + (c.r.passes === 1 ? 'ok' : 'fail')}>
              {c.r.passes === 1
                ? tr('one pass: no conflict', 'một lượt: không xung đột')
                : tr(`${c.r.passes} passes: ${c.r.passes}-way conflict`, `${c.r.passes} lượt: xung đột ${c.r.passes} chiều`)}
            </span>
            <span>
              <Translated vi={<><b>{c.r.banksUsed}</b> trên {c.banks} bank được dùng</>}>
                <><b>{c.r.banksUsed}</b> of {c.banks} banks used</>
              </Translated>
            </span>
            <span>
              <Translated vi={<>nhiều nhất <b>{c.r.maxLanesPerBank}</b> lane trên một bank</>}>
                <>at most <b>{c.r.maxLanesPerBank}</b> lane{c.r.maxLanesPerBank === 1 ? '' : 's'} on one bank</>
              </Translated>
            </span>
            {c.r.sharedWord ? (
              <span className="muted">
                <Translated
                  vi={
                    <>
                      một số lane dùng chung một từ:{' '}
                      {({
                    nvidia: 'một lần đọc một từ được phát (broadcast) tới mọi lane yêu cầu nó; không xung đột',
                    'amd-cdna': 'phát từ một địa chỉ được "handled efficiently without conflicts" (xử lý hiệu quả, không xung đột)',
                    'amd-cdna4': 'phát từ một địa chỉ được "handled efficiently without conflicts" (xử lý hiệu quả, không xung đột)',
                    'amd-rdna': 'phát từ một địa chỉ được "handled efficiently without conflicts" (xử lý hiệu quả, không xung đột)',
                    apple: 'Apple khuyên loại bỏ các truy cập vào cùng một vị trí từ nhiều luồng của một threadgroup',
                  } as Record<string, string>)[preset.id]}
                    </>
                  }
                >
                  <>some lanes share a word: {preset.sameWord}</>
                </Translated>
              </span>
            ) : null}
          </div>
          {preset.caveat ? (
            <p className="muted" style={{ margin: '0.5rem 0 0' }} data-caveat>
              <Translated
                vi={
                  ({
                    'amd-cdna': 'AMD phục vụ 16 lane mỗi chu kỳ, nên một truy cập 64 lane dù sao cũng mất vài chu kỳ; con số này là phép tính kiểu NVIDIA trên số bank của AMD, không phải chi phí của AMD.',
                    'amd-cdna4': 'AMD phục vụ 16 lane mỗi chu kỳ, nên một truy cập 64 lane dù sao cũng mất vài chu kỳ; con số này là phép tính kiểu NVIDIA trên số bank của AMD, không phải chi phí của AMD.',
                    'amd-rdna': 'AMD phục vụ 16 lane mỗi chu kỳ; con số này là phép tính kiểu NVIDIA trên số bank của AMD, không phải chi phí của AMD.',
                    apple: 'Apple không công bố cấu trúc bank nào cho bộ nhớ threadgroup; lab 6.3 đo xem máy Mac của bạn làm gì.',
                  } as Record<string, string>)[preset.id] ?? preset.caveat
                }
              >
                {preset.caveat}
              </Translated>
            </p>
          ) : null}
        </>
      ) : (
        <p className="narrate" data-readout data-passes="" data-not-documented>
          <Translated
            vi={
              <>
                <b>Không có tài liệu.</b>{' '}
                {({
                    'amd-cdna': 'AMD phục vụ 16 lane mỗi chu kỳ, nên một truy cập 64 lane dù sao cũng mất vài chu kỳ; con số này là phép tính kiểu NVIDIA trên số bank của AMD, không phải chi phí của AMD.',
                    'amd-cdna4': 'AMD phục vụ 16 lane mỗi chu kỳ, nên một truy cập 64 lane dù sao cũng mất vài chu kỳ; con số này là phép tính kiểu NVIDIA trên số bank của AMD, không phải chi phí của AMD.',
                    'amd-rdna': 'AMD phục vụ 16 lane mỗi chu kỳ; con số này là phép tính kiểu NVIDIA trên số bank của AMD, không phải chi phí của AMD.',
                    apple: 'Apple không công bố cấu trúc bank nào cho bộ nhớ threadgroup; lab 6.3 đo xem máy Mac của bạn làm gì.',
                  } as Record<string, string>)[preset.id] ?? preset.caveat}{' '}
                Về việc nhiều lane đọc một vị trí,{' '}
                {({
                    nvidia: 'một lần đọc một từ được phát (broadcast) tới mọi lane yêu cầu nó; không xung đột',
                    'amd-cdna': 'phát từ một địa chỉ được "handled efficiently without conflicts" (xử lý hiệu quả, không xung đột)',
                    'amd-cdna4': 'phát từ một địa chỉ được "handled efficiently without conflicts" (xử lý hiệu quả, không xung đột)',
                    'amd-rdna': 'phát từ một địa chỉ được "handled efficiently without conflicts" (xử lý hiệu quả, không xung đột)',
                    apple: 'Apple khuyên loại bỏ các truy cập vào cùng một vị trí từ nhiều luồng của một threadgroup',
                  } as Record<string, string>)[preset.id]}, nên việc phát miễn phí của NVIDIA không phải điều có thể trông cậy ở đây.
              </>
            }
          >
            <>
              <b>Not documented.</b> {preset.caveat} On many lanes reading one location, {preset.sameWord}, so NVIDIA's free
              broadcast is not something to count on here.
            </>
          </Translated>
        </p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- Exercise 5

function Exercise5Table() {
  const [guess, setGuess] = useState<Record<number, string>>({});
  const [shown, setShown] = useState(false);
  const rows = EX5_STRIDES.map((st) => ({ st, r: strideCase(st) }));
  const right = rows.filter(({ st, r }) => Number(guess[st]) === r.passes).length;
  return (
    <Bench
      title={tr('Exercise 5', 'Bài tập 5')}
      task={tr(
        'For each stride, how many lanes share each bank the warp uses? Fill in all seven, then reveal.',
        'Với mỗi bước nhảy, bao nhiêu lane dùng chung mỗi bank mà warp dùng? Điền cả bảy, rồi xem đáp án.',
      )}
      foot={
        <Translated
          vi={
            <>
              <code>__shared__ float a[1024];</code> và <code>a[threadIdx.x*stride] = ...;</code> bởi một khối gồm một warp 32
              lane, <code>a[0]</code> ở bank 0 (sách p. 155). 32 bank, mỗi bank 4 byte. Câu trả lời của bạn ở đây không được
              ghi lại.
            </>
          }
        >
          <><code>__shared__ float a[1024];</code> and <code>a[threadIdx.x*stride] = ...;</code> by a block of one 32-lane warp, <code>a[0]</code> in bank 0 (book p. 155). 32 banks of 4 bytes. Your entries here are not recorded.</>
        </Translated>
      }
    >
      <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
        <thead>
          <tr>
            <th style={{ paddingLeft: '0.6rem' }}>
              <Translated vi="Phần">Part</Translated>
            </th>
            <th>
              <Translated vi="bước nhảy">stride</Translated>
            </th>
            <th>
              <Translated vi="số lane mỗi bank được dùng">lanes per bank used</Translated>
            </th>
            {shown ? (
              <>
                <th>
                  <Translated vi="mô hình">model</Translated>
                </th>
                <th>
                  <Translated vi="các bank bị chạm">banks hit</Translated>
                </th>
              </>
            ) : null}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ st, r }, i) => (
            <tr key={st} data-stride={st} data-result={shown ? (Number(guess[st]) === r.passes ? 'right' : 'wrong') : ''}>
              <td style={{ paddingLeft: '0.6rem' }}>({'abcdefg'[i]})</td>
              <td>{st}</td>
              <td>
                <input type="number" min={1} max={32} value={guess[st] ?? ''} aria-label={tr(`stride ${st}`, `bước nhảy ${st}`)}
                  onChange={(e) => { setGuess({ ...guess, [st]: e.target.value }); setShown(false); }} />
              </td>
              {shown ? (
                <>
                  <td><span className={'chip ' + (Number(guess[st]) === r.passes ? 'ok' : 'fail')}>{r.passes === 1 ? tr('1 (no conflict)', '1 (không xung đột)') : tr(`${r.passes}-way`, `${r.passes} chiều`)}</span></td>
                  <td>{r.banksUsed === 32 ? tr('all 32', 'cả 32') : r.perBank.filter((b) => b.lanes.length).map((b) => b.bank).join(', ')}</td>
                </>
              ) : null}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" onClick={() => setShown(true)} disabled={rows.some(({ st }) => !guess[st])}>
          <Translated vi="Xem đáp án">Reveal</Translated>
        </button>
        {shown ? (
          <span data-score={right}>
            <Translated vi={<>Đúng {right} trên 7.</>}>
              <>{right} of 7 right.</>
            </Translated>
          </span>
        ) : (
          <span className="muted">
            <Translated vi={<>Đã điền {rows.filter(({ st }) => guess[st]).length} trên 7</>}>
              <>{rows.filter(({ st }) => guess[st]).length} of 7 filled in</>
            </Translated>
          </span>
        )}
      </div>
      {shown ? (
        <div style={{ marginTop: '0.8rem' }} data-gcd-rule>
          <Translated
            vi={
              <p style={{ margin: '0 0 0.4rem' }}>
                <b>Quy tắc đứng sau bảng.</b> Lane <i>l</i> rơi vào bank (<i>l</i> · stride) mod 32, lặp lại sau mỗi 32 /{' '}
                <i>g</i> lane, trong đó <i>g</i> là ước chung lớn nhất của bước nhảy và 32. Vậy warp dùng 32 / <i>g</i> bank
                với <i>g</i> lane trên mỗi bank, tất cả ở các từ khác nhau: một xung đột <i>g</i> chiều.
              </p>
            }
          >
            <p style={{ margin: '0 0 0.4rem' }}>
              <b>The rule behind the table.</b> Lane <i>l</i> lands in bank (<i>l</i> · stride) mod 32, which repeats every
              32 / <i>g</i> lanes, where <i>g</i> is the greatest common divisor of the stride and 32. So the warp uses
              32 / <i>g</i> banks with <i>g</i> lanes on each, all on different words: a <i>g</i>-way conflict.
            </p>
          </Translated>
          <p className="muted" style={{ margin: 0 }}>
            <Translated
              vi={
                <>
                  {EX5_STRIDES.map((st) => `gcd(${st}, 32) = ${gcdRule(st).g}`).join('; ')}. Mọi bước nhảy lẻ đều không xung
                  đột; mỗi thừa số 2 trong bước nhảy làm xung đột tăng gấp đôi.
                </>
              }
            >
              <>
                {EX5_STRIDES.map((st) => `gcd(${st}, 32) = ${gcdRule(st).g}`).join('; ')}. Any odd stride is conflict-free;
                every factor of 2 in the stride doubles the conflict.
              </>
            </Translated>
          </p>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- your device

function YourDevice64() {
  const { reports, loading } = useReports();
  const found = gpuBackends(reports)
    .map((b) => {
      const rows = labRows(reports, b, LAB63);
      return {
        b,
        device: String(reports[b].device.name ?? b),
        probe: rows.filter((r) => r.name === 'bank_probe' && num(r.stride) != null).sort((p, q) => Number(p.stride) - Number(q.stride)),
        transpose: rows.filter((r) => r.name === 'transpose'),
      };
    })
    .filter((x) => x.probe.length || x.transpose.length);
  const strides = [...new Set(found.flatMap((f) => f.probe.map((r) => Number(r.stride))))].sort((a, b) => a - b);
  const rel = (f: (typeof found)[number], st: number) => {
    const row = f.probe.find((r) => Number(r.stride) === st);
    const base = num(f.probe.find((r) => Number(r.stride) === 1)?.ns_per_access);
    return num(row?.relative_to_stride1) ?? (base && num(row?.ns_per_access) ? num(row?.ns_per_access)! / base : undefined);
  };
  return (
    <Bench
      title={tr('Your device: banks', 'Thiết bị của bạn: bank')}
      task={tr(
        "Does your device's time per access follow the model's pass count?",
        'Thời gian mỗi lần truy cập trên thiết bị của bạn có đi theo số lượt của mô hình không?',
      )}
      foot={tr(
        "The model column is NVIDIA's documented rule for a 32-lane warp. Measured columns are lab 6.3's time per shared-memory access relative to stride 1; they include loop overhead, so a 32-pass access need not take 32 times as long. On AMD and Apple they are measurements only: neither vendor's documents predict them.",
        'Cột mô hình là quy tắc được công bố của NVIDIA cho một warp 32 lane. Các cột đo được là thời gian mỗi lần truy cập bộ nhớ chia sẻ của lab 6.3 so với bước nhảy 1; chúng gồm cả chi phí vòng lặp, nên một truy cập 32 lượt không nhất thiết mất thời gian gấp 32 lần. Trên AMD và Apple chúng chỉ là phép đo: tài liệu của cả hai nhà cung cấp đều không dự đoán chúng.',
      )}
    >
      {loading ? null : !found.length ? (
        <p data-no-measurement>
          <Translated
            vi={
              <>
                Chưa có phép đo nào. Chạy <code>python3 labs/pmpp.py run 6.3 --solution --bench</code> trên một máy có GPU
                (hoặc <code>run 6.3 --bench</code> khi kernel của chính bạn đã vượt qua) rồi tải lại.
              </>
            }
          >
            <>
              No measurement yet. Run <code>python3 labs/pmpp.py run 6.3 --solution --bench</code> on a machine with a GPU
              (or <code>run 6.3 --bench</code> once your own kernel passes) and reload.
            </>
          </Translated>
        </p>
      ) : (
        <>
          {strides.length ? (
            <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-probe>
              <thead>
                <tr>
                  <th style={{ paddingLeft: '0.6rem' }}>
                    <Translated vi="bước nhảy">stride</Translated>
                  </th>
                  <th className="num">
                    <Translated vi="Mô hình NVIDIA: số lượt">NVIDIA model: passes</Translated>
                  </th>
                  {found.map((f) => (
                    <th key={f.b} className="num">
                      <Translated vi={<>{f.device} ({f.b}): thời gian ÷ bước nhảy 1</>}>
                        <>{f.device} ({f.b}): time ÷ stride 1</>
                      </Translated>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {strides.map((st) => (
                  <tr key={st} data-stride={st}>
                    <td style={{ paddingLeft: '0.6rem' }}>{st}</td>
                    <td className="num">{strideCase(st).passes}</td>
                    {found.map((f) => {
                      const v = rel(f, st);
                      return <td key={f.b} className="num">{v != null ? `${fmt(v)}×` : '–'}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
          {found.some((f) => f.transpose.length) ? (
            <table className="data" style={{ background: 'var(--page)', margin: '0.8rem 0 0' }} data-transpose>
              <thead>
                <tr>
                  <th style={{ paddingLeft: '0.6rem' }}>
                    <Translated vi="Thiết bị">Device</Translated>
                  </th>
                  <th className="num">
                    <Translated vi="ô">tile</Translated>
                  </th>
                  <th className="num">pitch T: GB/s</th>{/* i18n-ignore: "pitch" is kept in Vietnamese, with a symbol and unit */}
                  <th className="num">pitch T + 1: GB/s</th>{/* i18n-ignore: "pitch" is kept in Vietnamese, with a symbol and unit */}
                  <th className="num">
                    <Translated vi="có đệm ÷ không đệm">padded ÷ unpadded</Translated>
                  </th>
                </tr>
              </thead>
              <tbody>
                {found.flatMap((f) =>
                  [...new Set(f.transpose.map((r) => Number(r.tile)))].sort((a, b) => a - b).map((T) => {
                    const a = num(f.transpose.find((r) => Number(r.tile) === T && Number(r.pitch) === T)?.gb_per_s);
                    const b = num(f.transpose.find((r) => Number(r.tile) === T && Number(r.pitch) === T + 1)?.gb_per_s);
                    return (
                      <tr key={f.b + T}>
                        <td style={{ paddingLeft: '0.6rem' }}>{f.device} ({f.b})</td>
                        <td className="num">{T}</td>
                        <td className="num">{a != null ? fmt(a) : '–'}</td>
                        <td className="num">{b != null ? fmt(b) : '–'}</td>
                        <td className="num">{a && b ? `${fmt(b / a)}×` : '–'}</td>
                      </tr>
                    );
                  }),
                )}
              </tbody>
            </table>
          ) : (
            <p className="muted" style={{ margin: '0.8rem 0 0' }}>
              <Translated vi="Chưa có phép đo chuyển vị phân ô nào: nó xuất hiện khi phép chuyển vị của lab 6.3 đã được chạy benchmark.">
                No tiled-transpose measurement yet: it appears once lab 6.3's transpose has been benchmarked.
              </Translated>
            </p>
          )}
          <p className="muted" style={{ margin: '0.6rem 0 0' }} data-measurement-only={found.filter((f) => f.b !== 'cuda').map((f) => f.b).join(',')}>
            <Translated vi="Phép đo từ máy của bạn, không phải giá trị kỳ vọng.">Measurements from your machine, not expected values.</Translated>
            {found.some((f) => f.b !== 'cuda')
              ? tr(
                  ` No vendor model predicts the ${found.filter((f) => f.b !== 'cuda').map((f) => `${f.device} (${f.b})`).join(' or ')} column${found.filter((f) => f.b !== 'cuda').length === 1 ? '' : 's'}: those are measurements only.`,
                  ` Không mô hình nào của nhà cung cấp dự đoán cột ${found.filter((f) => f.b !== 'cuda').map((f) => `${f.device} (${f.b})`).join(' hoặc ')}: đó chỉ là phép đo.`,
                )
              : ''}
          </p>
        </>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS64: Record<string, Question> = {
  'ch06/6.4/q-ex5-24': {
    kind: 'numeric',
    id: 'ch06/6.4/q-ex5-24',
    prompt: (
      <Translated
        vi={
          <>
            Bài tập 5(c): <code>__shared__ float a[1024];</code> và một warp 32 lane ghi <code>a[threadIdx.x*24]</code>. Với
            32 bank, mỗi bank 4 byte, bao nhiêu lane dùng chung mỗi bank mà warp dùng?
          </>
        }
      >
        <>
          Exercise 5(c): <code>__shared__ float a[1024];</code> and one 32-lane warp writes <code>a[threadIdx.x*24]</code>.
          With 32 banks of 4 bytes, how many lanes share each bank the warp uses?
        </>
      </Translated>
    ),
    answer: 8,
    explain: (
      <Translated vi={<p>Lane l ở bank 24l mod 32: các bank 0, 24, 16, 8, rồi lặp lại vòng. gcd(24, 32) = 8, nên 4 bank với 8 lane mỗi bank: một xung đột 8 chiều.</p>}>
        <p>Lane l is in bank 24l mod 32: banks 0, 24, 16, 8, then round again. gcd(24, 32) = 8, so 4 banks with 8 lanes each: an 8-way conflict.</p>
      </Translated>
    ),
    source: 'book p. 155 (exercise 5), §6.4 pp. 139–140',
    vi: { source: 'sách p. 155 (bài tập 5), §6.4 pp. 139–140' },
  },
  'ch06/6.4/q-ex5-12': {
    kind: 'numeric',
    id: 'ch06/6.4/q-ex5-12',
    prompt: (
      <Translated vi={<>Bài tập 5(e): cùng warp đó ghi <code>a[threadIdx.x*12]</code>. Bao nhiêu lane dùng chung mỗi bank được dùng?</>}>
        <>Exercise 5(e): the same warp writes <code>a[threadIdx.x*12]</code>. How many lanes share each bank used?</>
      </Translated>
    ),
    answer: 4,
    explain: (
      <Translated vi={<p>gcd(12, 32) = 4: các bank lặp lại sau mỗi 8 lane, nên 8 bank (0, 4, 8, … 28) với 4 lane mỗi bank.</p>}>
        <p>gcd(12, 32) = 4: the banks repeat every 8 lanes, so 8 banks (0, 4, 8, … 28) with 4 lanes each.</p>
      </Translated>
    ),
    source: 'book p. 155 (exercise 5)',
    vi: { source: 'sách p. 155 (bài tập 5)' },
  },
  'ch06/6.4/q-ex5-31': {
    kind: 'numeric',
    id: 'ch06/6.4/q-ex5-31',
    prompt: (
      <Translated vi={<>Bài tập 5(b): <code>a[threadIdx.x*31]</code>. Bao nhiêu lane dùng chung mỗi bank được dùng?</>}>
        <>Exercise 5(b): <code>a[threadIdx.x*31]</code>. How many lanes share each bank used?</>
      </Translated>
    ),
    answer: 1,
    explain: (
      <Translated
        vi={
          <p>
            31 và 32 không có ước chung, nên 32 lane rơi vào 32 bank khác nhau: không xung đột. Cả bảng: 32 → 32, 31 → 1,
            24 → 8, 16 → 16, 12 → 4, 8 → 8, 7 → 1.
          </p>
        }
      >
        <p>
          31 and 32 share no factor, so the 32 lanes fall in 32 different banks: no conflict. The whole table: 32 → 32,
          31 → 1, 24 → 8, 16 → 16, 12 → 4, 8 → 8, 7 → 1.
        </p>
      </Translated>
    ),
    source: 'book p. 155 (exercise 5)',
    vi: { source: 'sách p. 155 (bài tập 5)' },
  },
  'ch06/6.4/q-16col': {
    kind: 'numeric',
    id: 'ch06/6.4/q-16col',
    prompt: (
      <Translated
        vi={
          <>
            Nếu mảng rộng 16 float, một warp có 32 lane với <code>threadIdx.x</code> = 0 đến 31 (một khối rộng 32 luồng) ghi{' '}
            <code>a[threadIdx.x][threadIdx.y]</code> sẽ đặt bao nhiêu lane lên mỗi bank nó dùng?
          </>
        }
      >
        <>
          Had the array been 16 floats wide, a warp whose 32 lanes have <code>threadIdx.x</code> = 0 to 31 (a block 32
          threads wide) writing <code>a[threadIdx.x][threadIdx.y]</code> puts how many lanes on each bank it uses?
        </>
      </Translated>
    ),
    answer: 16,
    explain: (
      <Translated
        vi={
          <p>
            Chỉ số là 16·tx + ty: tx chẵn rơi vào bank ty, tx lẻ vào bank 16 + ty. Hai bank, 16 lane mỗi bank, như sách nói.
            Nếu thay bằng một khối vuông 16 × 16, một warp là hai hàng luồng, và con số trở thành bốn bank với 8 lane mỗi bank:
            thiết lập sẵn 16 × 16 của trình khám phá.
          </p>
        }
      >
        <p>
          The index is 16·tx + ty: even tx land in bank ty, odd tx in bank 16 + ty. Two banks, 16 lanes each, as the book
          says. With a square 16 × 16 block instead, one warp is two rows of threads, and the count becomes four banks with
          8 lanes each: the explorer's 16 × 16 preset.
        </p>
      </Translated>
    ),
    source: 'book §6.4 p. 141; the 16 × 16 block case is our arithmetic with the same rule',
    vi: { source: 'sách §6.4 p. 141; trường hợp khối 16 × 16 là phép tính của chúng tôi với cùng quy tắc' },
  },
  'ch06/6.4/q-padcost': {
    kind: 'numeric',
    id: 'ch06/6.4/q-padcost',
    prompt: (
      <Translated vi={<>Đệm một ô <code>float</code> 32 × 32 thành 32 × 33 tốn thêm bao nhiêu byte bộ nhớ chia sẻ?</>}>
        <>How many extra bytes of shared memory does padding one 32 × 32 <code>float</code> tile to 32 × 33 cost?</>
      </Translated>
    ),
    answer: 128,
    unit: 'bytes',
    explain: (
      <Translated vi={<p>Thêm một cột 32 float, mỗi float 4 byte. Với hai ô như vậy mỗi khối, như trong matmul phân ô, là 256 byte mỗi khối.</p>}>
        <p>One extra column of 32 floats, 4 bytes each. With two such tiles per block, as in a tiled matmul, it is 256 bytes per block.</p>
      </Translated>
    ),
    source: 'book §6.4 pp. 140–141',
    vi: { unit: 'byte', source: 'sách §6.4 pp. 140–141' },
  },
  'ch06/6.4/q-shift': {
    kind: 'choice',
    id: 'ch06/6.4/q-shift',
    prompt: (
      <Translated
        vi={
          <>
            Sách viết rằng nhân <code>i</code> với 33 trở thành <code>i &gt;&gt; 5 + i</code>. Biểu thức C đó tính ra gì?
          </>
        }
      >
        <>
          The book writes that multiplying <code>i</code> by 33 becomes <code>i &gt;&gt; 5 + i</code>. What does that C
          expression compute?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi={<>33 × <code>i</code>, đúng như dự định: dịch 5, rồi cộng.</>}>
            <>33 × <code>i</code>, as intended: a shift by 5, then an add.</>
          </Translated>
        ),
        why: (
          <Translated vi="Hai chỗ sơ suất: >> chia cho một lũy thừa của hai, và + có độ ưu tiên cao hơn phép dịch.">
            {'Two slips: >> divides by a power of two, and + binds more tightly than a shift.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi={<><code>i &gt;&gt; (5 + i)</code>, một phép dịch phải 5 + i bit.</>}>
            <><code>i &gt;&gt; (5 + i)</code>, a right shift by 5 + i bits.</>
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi={<><code>i / 32 + i</code>: một phép dịch phải 5 bit, rồi cộng i.</>}>
            <><code>i / 32 + i</code>: a right shift by 5 bits, then an add of i.</>
          </Translated>
        ),
        why: (
          <Translated vi="Điều đó cần dấu ngoặc: trong C, + có độ ưu tiên cao hơn >>.">
            {'That would need parentheses: in C, + binds more tightly than >>.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Nhân với 32 là một phép dịch trái 5, <code>i &lt;&lt; 5</code>; sách in phép dịch theo chiều ngược lại, và biểu
            thức thứ hai còn cần dấu ngoặc. Dạng đúng ý là <code>(i &lt;&lt; 5) + i</code>. Luận điểm của sách vẫn đứng vững:
            32 cột tốn một phép dịch, 33 cột tốn một phép dịch và một phép cộng.
          </p>
        }
      >
        <p>
          Multiplying by 32 is a left shift by 5, <code>i &lt;&lt; 5</code>; the book prints the shift the wrong way round,
          and the second expression also needs parentheses. The intended form is <code>(i &lt;&lt; 5) + i</code>. The book's point stands: 32 columns cost one shift, 33 cost a
          shift and an add.
        </p>
      </Translated>
    ),
    source: 'book §6.4 pp. 140–141 (printed form); notes/ch06.md "Unclear in the PDF" item 1; facts E1',
    vi: { source: 'sách §6.4 pp. 140–141 (dạng được in); notes/ch06.md "Unclear in the PDF" mục 1; facts E1' },
  },
  'ch06/6.4/q-broadcast': {
    kind: 'choice',
    id: 'ch06/6.4/q-broadcast',
    prompt: (
      <Translated vi={<>Mọi lane của một warp đọc cùng một từ của bộ nhớ chia sẻ. Phát biểu nào đúng?</>}>
        <>Every lane of a warp reads the same word of shared memory. Which statement is right?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Xung đột 32 chiều trên mọi GPU: một bank phục vụ 32 lane lần lượt từng lane.">
            A 32-way conflict on every GPU: one bank serves the 32 lanes one at a time.
          </Translated>
        ),
        why: (
          <Translated vi="NVIDIA công bố cơ chế phát (broadcast) cho trường hợp này, và AMD nói nó được xử lý hiệu quả. Một xung đột cần các từ khác nhau trong cùng một bank.">
            NVIDIA documents a broadcast for this case, and AMD says it is handled efficiently. A conflict needs different words in one bank.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Miễn phí trên mọi GPU: từ đó được đọc một lần và trao cho mọi lane.">
            Free on every GPU: the one word is read once and handed to every lane.
          </Translated>
        ),
        why: (
          <Translated vi="Apple không công bố bank nào, và lời khuyên tinh chỉnh của họ là loại bỏ việc nhiều luồng đọc cùng một vị trí.">
            Apple documents no banks, and its tuning advice is to remove many-thread reads of one location.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Miễn phí trên NVIDIA và AMD; không biết có miễn phí trên Metal hay không.">
            Free on NVIDIA and AMD; not known to be free on Metal.
          </Translated>
        ),
        correct: true,
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một xung đột là các lane ở các từ khác nhau của cùng một bank. Các lane ở cùng một từ là một trường hợp khác, và
            các nhà cung cấp xử lý nó khác nhau: NVIDIA và AMD phát nó, còn Apple khuyên loại bỏ những truy cập như vậy khỏi
            các lần đọc threadgroup.
          </p>
        }
      >
        <p>
          A conflict is lanes on different words of one bank. Lanes on the same word are a different case, and vendors
          treat it differently: NVIDIA and AMD broadcast it, while Apple advises removing such accesses from threadgroup
          reads.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch06-facts.md A12, C2; infra-facts.md A10, B10', // i18n-ignore: file citation
  },
  'ch06/6.4/q-amd': {
    kind: 'choice',
    id: 'ch06/6.4/q-amd',
    prompt: (
      <Translated
        vi={
          <>
            Trên một GPU AMD với 32 bank và wavefront 64 lane, các lane <i>i</i> và <i>i</i> + 32 của một truy cập{' '}
            <code>float</code> bước nhảy đơn vị rơi vào cùng một bank. Bạn có nên chèn đệm không?
          </>
        }
      >
        <>
          On an AMD GPU with 32 banks and 64-lane wavefronts, lanes <i>i</i> and <i>i</i> + 32 of a unit-stride{' '}
          <code>float</code> access land in the same bank. Should you pad?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Có, đệm một cột, như với ô đã xoay góc ở mục 6.4.">
            Yes, pad by one column, as for the corner-turned tile of section 6.4.
          </Translated>
        ),
        why: (
          <Translated vi="Chèn đệm không thay đổi gì với bước nhảy đơn vị: hai lane vẫn cách nhau 32 từ.">
            Padding changes nothing for a unit stride: the two lanes are still 32 words apart.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không dựa trên cơ sở đó: AMD dù sao cũng phục vụ 16 lane mỗi chu kỳ, nên cách đếm xung đột của NVIDIA không áp dụng. Hãy đo, bằng tỉ lệ xung đột bank của LDS hoặc lab 6.3.">
            Not on that basis: AMD serves 16 lanes per cycle in any case, so NVIDIA's conflict count does not apply. Measure, with the LDS bank-conflict rate or lab 6.3.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Có, đệm các hàng thành 65 từ cho hợp với wavefront 64 lane.">
            Yes, pad rows to 65 words to suit the 64-lane wavefront.
          </Translated>
        ),
        why: (
          <Translated vi="Đó là đoán mò về một chi phí không ai công bố. Hãy đo trước.">
            That is a guess at a cost nobody documented. Measure first.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Con số 32 bank là của AMD; việc ghép cặp lane i và i + 32 là phép tính của chúng tôi từ đó. AMD công bố rằng truy
            cập LDS của một wavefront được phục vụ 16 lane mỗi chu kỳ và không công bố cách nó nhóm các lane khi xét xung
            đột, nên một con số tính theo cách của NVIDIA không phải chi phí của AMD.
          </p>
        }
      >
        <p>
          The 32-bank figure is AMD's; the pairing of lanes i and i + 32 is our arithmetic from it. AMD documents that a
          wavefront's LDS access is served 16 lanes per cycle and does not document how it groups lanes for conflicts, so
          a count computed the NVIDIA way is not AMD's cost.
        </p>
      </Translated>
    ),
    source: 'notes/today/infra-facts.md B10; ch06-facts.md B4', // i18n-ignore: file citation
  },
};

// ---------------------------------------------------------------- the section

export function S64() {
  const turned = tileAccess({ T: 32, pitch: 32, orientation: 'x-first' });
  const padded = tileAccess({ T: 32, pitch: 33, orientation: 'x-first' });
  return (
    <>
      <Translated
        vi={
          <p>
            Bộ nhớ chia sẻ (shared memory) là SRAM, không phải DRAM. Độ trễ của nó ngắn, nên nó không cần burst, và nó có thể
            được chia thành nhiều <b>bank</b> hẹp làm việc song song.
          </p>
        }
      >
        <p>
          Shared memory is SRAM, not DRAM. Its latency is short, so it has no use for bursts, and it can be split into many
          narrow <b>banks</b> that work in parallel.
        </p>
      </Translated>
      <Source section="6.4" pages="139" />
      <Translated
        vi={
          <p>
            Bố cục điển hình của sách là 32 bank, mỗi bank rộng 4 byte, với các từ 4 byte liên tiếp nằm ở các bank liên tiếp:
            từ 0 ở bank 0, từ 31 ở bank 31, từ 32 quay lại bank 0. Một warp đọc 32 float liên tiếp được 32 bank phục vụ cùng
            lúc.
          </p>
        }
      >
        <p>
          The book's typical layout is 32 banks, each 4 bytes wide, with consecutive 4-byte words in consecutive banks: word
          0 in bank 0, word 31 in bank 31, word 32 back in bank 0. A warp reading 32 consecutive floats is served by 32
          banks at once.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/infra-facts.md A10; ch06-facts.md A12)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              NVIDIA công bố 32 bank, mỗi bank 4 byte, cho mọi compute capability. Họ không đưa ra độ trễ theo chu kỳ, chỉ nói
              “much higher bandwidth and lower latency” (băng thông cao hơn nhiều và độ trễ thấp hơn) so với bộ nhớ toàn cục,
              nên câu “a few clock cycles” (vài chu kỳ xung nhịp) của sách là unverified.
            </p>
          }
        >
          <p>
            NVIDIA documents 32 banks of 4 bytes for every compute capability. It gives no latency in cycles, only “much
            higher bandwidth and lower latency” than global memory, so the book's “a few clock cycles” is unverified.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Một xung đột">A conflict</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một bank phục vụ một từ mỗi lúc. Khi các lane của một warp chạm vào những từ khác nhau nằm trong cùng một bank,
            phần cứng chia truy cập thành bao nhiêu lượt không xung đột mà nó cần và phục vụ chúng lần lượt.
          </p>
        }
      >
        <p>
          A bank serves one word at a time. When lanes of one warp touch different words that sit in the same bank, the
          hardware splits the access into as many conflict-free passes as it needs and serves them one after another.
        </p>
      </Translated>
      <Source section="6.4" pages="139–140" />
      <Translated
        vi={
          <p>
            Đó là một <b>xung đột bank</b> (bank conflict); k lane ở các từ khác nhau của cùng một bank là một xung đột k chiều.
          </p>
        }
      >
        <p>
          That is a <b>bank conflict</b>; k lanes on different words of one bank is a k-way conflict.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/infra-facts.md A10; ch06-facts.md A12)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              Hai trường hợp trông giống xung đột nhưng không phải, trên GPU NVIDIA: các lane đọc cùng một từ được phát
              (broadcast) từ đó, và các lane chạm vào các byte khác nhau của cùng một từ 32 bit không xung đột.
            </p>
          }
        >
          <p>
            Two cases that look like conflicts are not, on NVIDIA GPUs: lanes reading the same word get it broadcast, and
            lanes touching different bytes of the same 32-bit word do not conflict.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={<p>Bài tập 5 của sách là lối vào gọn gàng nhất. Hãy điền bảng trước khi chạm vào trình khám phá bên dưới nó.</p>}
      >
        <p>
          The book's exercise 5 is the cleanest way in. Fill in the table before you touch the explorer below it.
        </p>
      </Translated>
      <Exercise5Table />
      <Translated
        vi={
          <p>
            Giờ hãy kiểm tra bảng của bạn trong trình khám phá, rồi đi xa hơn: các bước nhảy bài tập không liệt kê, cả một ô,
            mọi lane trên một từ, và các phần tử 2 byte.
          </p>
        }
      >
        <p>
          Now check your table in the explorer, then go past it: strides the exercise does not list, a whole tile, every
          lane on one word, and 2-byte elements.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Quy tắc được công bố của NVIDIA, không gì hơn: không đo thời gian. Các thiết lập sẵn của AMD dùng lại phép tính
              đó với số bank của AMD nhưng không mô hình hóa cách phục vụ 16 lane mỗi chu kỳ của họ. Apple không công bố gì để
              mô hình hóa.
            </p>
          }
        >
          <p>
            NVIDIA's documented rule, nothing more: no timing. AMD's presets reuse that arithmetic with AMD's bank counts
            but not its 16-lanes-per-cycle service. Apple publishes nothing to model.
          </p>
        </Translated>
      </Simplification>
      <BankExplorer />
      <Ask q={questionsS64['ch06/6.4/q-ex5-24']} label={tr('Exercise 5, for review', 'Bài tập 5, để ôn tập')} />
      <Ask q={questionsS64['ch06/6.4/q-ex5-12']} label={tr('Exercise 5, for review', 'Bài tập 5, để ôn tập')} />
      <Ask q={questionsS64['ch06/6.4/q-ex5-31']} label={tr('Exercise 5, for review', 'Bài tập 5, để ôn tập')} />

      <h2>
        <Translated vi="Xung đột mà xoay góc gây ra">The conflict corner turning makes</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mục 6.1 đã sửa các lần đọc toàn cục của một ô thứ tự cột bằng cách xoay góc. Ô được khai báo{' '}
            <code>__shared__ float a[TILE_DIM][TILE_DIM]</code> với <code>TILE_DIM</code> = 32 và được ghi là{' '}
            <code>a[threadIdx.x][threadIdx.y] = ...</code>. Một warp của khối 32 × 32 có một <code>threadIdx.y</code> và các{' '}
            <code>threadIdx.x</code> liên tiếp, nên luồng 0 ghi <code>a[0][0]</code>, luồng 1 <code>a[1][0]</code>, luồng 2{' '}
            <code>a[2][0]</code>: một cột. Phần tử <code>a[i][j]</code> là từ i·32 + j, nên các lane chạm vào các từ 0, 32,
            64 và cứ thế, và lane nào cũng nằm ở bank {turned.lanes[0].bank}. Đó là một xung đột {turned.passes} chiều, với 31
            bank ngồi không.
          </p>
        }
      >
        <p>
          Section 6.1 fixed the global reads of a column-major tile by turning the corner. The tile is declared{' '}
          <code>__shared__ float a[TILE_DIM][TILE_DIM]</code> with <code>TILE_DIM</code> = 32 and written as{' '}
          <code>a[threadIdx.x][threadIdx.y] = ...</code>. A warp of a 32 × 32 block has one <code>threadIdx.y</code> and
          consecutive <code>threadIdx.x</code>, so thread 0 writes <code>a[0][0]</code>, thread 1 <code>a[1][0]</code>,
          thread 2 <code>a[2][0]</code>: a column. Element <code>a[i][j]</code> is word i·32 + j, so the lanes hit words
          0, 32, 64 and so on, and every one of them is in bank {turned.lanes[0].bank}. That is a {turned.passes}-way
          conflict, with 31 banks idle.
        </p>
      </Translated>
      <Source section="6.4" pages="139–140" />
      <Translated
        vi={
          <p>
            Cách sửa là một cột không dùng: khai báo <code>a[TILE_DIM][TILE_DIM + 1]</code> và giữ nguyên cách truy cập.
          </p>
        }
      >
        <p>
          The fix is one unused column: declare <code>a[TILE_DIM][TILE_DIM + 1]</code> and keep the same access.
        </p>
      </Translated>
      <Source pages="140" />
      <Translated
        vi={
          <p>
            Phần tử <code>a[i][j]</code> giờ là từ i·33 + j, luồng 0 rơi vào bank 0, luồng 1 vào bank 1, luồng 2 vào bank 2, và
            phép ghi mất {padded.passes} lượt.
          </p>
        }
      >
        <p>
          Element <code>a[i][j]</code> is now word i·33 + j, thread 0 lands in bank 0, thread 1 in bank 1, thread 2 in
          bank 2, and the store takes {padded.passes} pass.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A6, B4)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              Các hướng dẫn của NVIDIA và hướng dẫn hiệu năng của AMD đều in đúng cách sửa này: một mảng 32 × 32 được truy cập
              dọc theo một cột, rồi cùng mảng đó được khai báo 32 × 33.
            </p>
          }
        >
          <p>
            NVIDIA's guides and AMD's performance guidelines both print this exact fix: a 32 × 32 array accessed down a
            column, then the same array declared 32 × 33.
          </p>
        </Translated>
      </Today>
      <TurnAndPadScene />

      <h2>
        <Translated vi="Chèn đệm tốn gì">What padding costs</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai thứ. Bộ nhớ: cột thừa không chứa gì. Và phép tính chỉ số: với 32 cột, phép nhân với độ dài hàng là một phép
            dịch duy nhất, còn với 33 nó trở thành một phép dịch và một phép cộng. Càng nhiều cột đệm, càng nhiều việc.
          </p>
        }
      >
        <p>
          Two things. Memory: the extra column holds nothing. And index arithmetic: with 32 columns the multiplication by
          the row length is a single shift, and with 33 it becomes a shift and an add. More padding columns, more work.
        </p>
      </Translated>
      <Source section="6.4" pages="140–141" />
      <Ask q={questionsS64['ch06/6.4/q-padcost']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Sách in phép dịch theo chiều ngược lại. Nó viết <code>i &gt;&gt; 5</code> cho phép nhân với 32 và{' '}
            <code>i &gt;&gt; 5 + i</code> cho 33. Phép nhân là một phép dịch trái, <code>i &lt;&lt; 5</code> và{' '}
            <code>(i &lt;&lt; 5) + i</code>, và C còn đọc dạng thứ hai được in thành một thứ khác nữa.
          </p>
        }
      >
        <p>
          The book prints the shift the wrong way round. It writes <code>i &gt;&gt; 5</code> for multiplying by 32 and{' '}
          <code>i &gt;&gt; 5 + i</code> for 33. Multiplying is a left shift, <code>i &lt;&lt; 5</code> and{' '}
          <code>(i &lt;&lt; 5) + i</code>, and C reads the printed second form as something else again.
        </p>
      </Translated>
      <Aside>
        <Translated
          vi={
            <p>
              Một kiểu chèn đệm khác mang cùng tên. <code>cudaMallocPitch</code> và <code>hipMallocPitch</code> đệm mỗi hàng
              của một mảng 2D trong bộ nhớ toàn cục để mọi hàng bắt đầu ở vị trí căn lề cho việc hợp nhất truy cập. Chèn đệm
              của mục 6.4 phá vỡ một bước nhảy lũy thừa của hai trong bộ nhớ chia sẻ. (notes/today/ch06-facts.md A3, B2.)
            </p>
          }
        >
          <p>
            A different padding goes by the same name. <code>cudaMallocPitch</code> and <code>hipMallocPitch</code> pad each
            row of a 2D array in global memory so every row starts aligned for coalescing. Section 6.4's padding breaks a
            power-of-two stride in shared memory. (notes/today/ch06-facts.md A3, B2.)
          </p>
        </Translated>
      </Aside>
      <Ask q={questionsS64['ch06/6.4/q-shift']} />

      <h2>
        <Translated vi="Mảng hẹp hơn thì xung đột ít hơn">Narrower arrays conflict less</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Chèn đệm mua được bao nhiêu phụ thuộc vào mẫu truy cập và kích thước. Ví dụ của sách: nếu mảng rộng 16 float, các
            lane của warp sẽ cách nhau 16 từ, rơi vào hai bank với xung đột 16 chiều ở mỗi bank. Các lane gần nhau hơn thì xung
            đột ít hơn.
          </p>
        }
      >
        <p>
          How much padding buys depends on the pattern and the size. The book's example: had the array been 16 floats wide,
          the warp's lanes would be 16 words apart, falling on two banks with a 16-way conflict each. Lanes closer together
          conflict less.
        </p>
      </Translated>
      <Source pages="141" />
      <Translated
        vi={
          <p>
            Con số đó giả định 32 lane của warp đều có <code>threadIdx.x</code> khác nhau, như trong một khối rộng 32 luồng.
            Với một khối vuông 16 × 16, một warp là hai hàng 16 luồng, và cùng phép ghi đó rơi vào bốn bank với 8 lane mỗi bank
            (phép tính của chúng tôi, với cùng quy tắc). Hãy thử cả hai trong trình khám phá: bước nhảy 16, rồi thiết lập sẵn ô
            16 × 16.
          </p>
        }
      >
        <p>
          That count assumes the warp's 32 lanes all have different <code>threadIdx.x</code>, as in a block 32 threads wide.
          With a square 16 × 16 block, a warp is two rows of 16 threads, and the same store lands on four banks with 8 lanes
          each (our arithmetic, with the same rule). Try both in the explorer: stride 16, then the 16 × 16 tile preset.
        </p>
      </Translated>
      <Ask q={questionsS64['ch06/6.4/q-16col']} label={tr('Work it out', 'Tự tính')} />

      <h2>
        <Translated vi="Trên các backend khác">On other backends</Translated>
      </h2>
      <Translated
        vi={
          <p>
            AMD gọi bộ nhớ chia sẻ là local data share (LDS). Nó có 32 bank trên CDNA 1 đến 3 và 64 trên CDNA 4 và RDNA 2 đến
            4, mỗi bank rộng 4 byte, phục vụ một wavefront 16 lane mỗi chu kỳ, và xử lý việc phát từ một địa chỉ mà không xung
            đột. Chính hướng dẫn của AMD dùng cùng cách đệm [32][33]. Nhưng một con số tính cho 32 lane và 32 bank của NVIDIA
            không phải con số của AMD: chuyển trình khám phá sang một thiết lập sẵn của AMD và đọc lưu ý dưới kết quả.
          </p>
        }
      >
        <p>
          AMD calls shared memory the local data share (LDS). It has 32 banks on CDNA 1 to 3 and 64 on CDNA 4 and RDNA 2 to
          4, each 4 bytes wide, serves a wavefront 16 lanes per cycle, and handles a broadcast from one address without a
          conflict. AMD's own guide uses the same [32][33] padding. But a count computed for NVIDIA's 32 lanes and 32 banks
          is not AMD's count: switch the explorer to an AMD preset and read the caveat under the result.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/infra-facts.md B10; ch06-facts.md B4)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              Trang mô hình lập trình của AMD nói “usually 32 banks” (thường là 32 bank), mâu thuẫn với trang phần cứng của họ
              cho CDNA 4 và RDNA 2 trở về sau. Trang phần cứng là trang cụ thể hơn.
            </p>
          }
        >
          <p>
            AMD's programming-model page says “usually 32 banks”, which contradicts its hardware page for CDNA 4 and RDNA 2
            and later. The hardware page is the specific one.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS64['ch06/6.4/q-amd']} />
      <Translated
        vi={
          <p>
            Apple không công bố cấu trúc bank nào cho bộ nhớ threadgroup. Thứ họ công bố là lời khuyên: căn lề các vùng cấp
            phát threadgroup theo 16 byte, để các luồng kề nhau của một quad group chạm vào các phần tử kề nhau, và loại bỏ các
            truy cập vào một vị trí bởi nhiều luồng của một threadgroup. Điểm cuối cùng đó ngược với việc phát miễn phí của
            NVIDIA. Trên GPU Apple family 9, bộ nhớ threadgroup được cấp phát từ bộ nhớ đệm L1. Vì vậy trên máy Mac, lab 6.3
            đo; không gì dự đoán được.
          </p>
        }
      >
        <p>
          Apple documents no bank structure for threadgroup memory. What it does publish is advice: align threadgroup
          allocations to 16 bytes, have neighbouring threads of a quad group touch neighbouring elements, and remove
          accesses to one location by many threads of a threadgroup. That last point is the opposite of NVIDIA's free
          broadcast. On Apple family 9 GPUs, threadgroup memory is allocated out of the L1 cache. So on a Mac, lab 6.3
          measures; nothing predicts.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md C2; infra-facts.md C13)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              Từ tài liệu Xcode của Apple, “Reducing shader bottlenecks”, và bài nói kỹ thuật năm 2023 về profiling GPU M3 và
              A17 Pro.
            </p>
          }
        >
          <p>
            From Apple's Xcode documentation, “Reducing shader bottlenecks”, and the 2023 tech talk on profiling M3 and A17
            Pro GPUs.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsS64['ch06/6.4/q-broadcast']} />

      <h2>
        <Translated vi="Thấy nó trong profiler">Seeing it in a profiler</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trên GPU NVIDIA, Nsight Compute đếm xung đột trực tiếp: <code>l1tex__data_bank_conflicts_pipe_lsu.sum</code>, và
            một con số N chiều trung bình cho mỗi lệnh bộ nhớ chia sẻ, trong đó 1 chiều nghĩa là không xung đột. Trên GPU AMD
            Instinct, bảng LDS của ROCm Compute Profiler báo cáo tỉ lệ xung đột bank, phần chu kỳ LDS dành cho xung đột. Apple
            không công bố bộ đếm bank nào; các bộ đếm giới hạn tile memory của Xcode là bằng chứng gián tiếp.
          </p>
        }
      >
        <p>
          On NVIDIA GPUs, Nsight Compute counts conflicts directly: <code>l1tex__data_bank_conflicts_pipe_lsu.sum</code>,
          and an average N-way figure per shared-memory instruction, where 1-way means no conflict. On AMD Instinct GPUs,
          the ROCm Compute Profiler's LDS panel reports a bank conflict rate, the share of LDS cycles spent on conflicts.
          Apple documents no bank counter; Xcode's tile-memory limiter counters are the indirect evidence.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md A12, A17, B4, B9, C5)" /* i18n-ignore: date and facts-file reference */>
        <Translated
          vi={
            <p>
              Hướng dẫn phân loại của NVIDIA đánh dấu xung đột bank trên khoảng 10 % số wavefront bộ nhớ chia sẻ và gợi ý ánh
              xạ lại cách đánh chỉ số. ROCm Compute Profiler chỉ liệt kê GPU Instinct và APU Ryzen AI Max là đã được kiểm thử.
            </p>
          }
        >
          <p>
            NVIDIA's triage guide flags bank conflicts above about 10 % of shared-memory wavefronts and suggests remapping
            the indexing. ROCm Compute Profiler lists only Instinct GPUs and Ryzen AI Max APUs as tested.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Trong các backend bạn sẽ đọc">In the backends you will read</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Hai GEMM chúng tôi đã đọc đệm 16 byte thay vì một float, và các luồng của chúng di chuyển hơn 4 byte mỗi lần. GEMM
            Metal của MLX đệm mỗi ô threadgroup 16 byte và định hình ô theo cách toán hạng được lưu. Phép nhân ma trận lượng tử
            hóa của llama.cpp đệm mọi khối 16 byte và khẳng định (assert) rằng bước nhảy hàng, tính theo từ 32 bit, bằng một
            bội của 8 cộng 4 trên nhánh <code>mma</code> (lẻ trên nhánh <code>dp4a</code>). Cùng ý tưởng với cột thừa, cho các
            truy cập rộng hơn.
          </p>
        }
      >
        <p>
          The two GEMMs we read pad by 16 bytes rather than by one float, and their threads move more than 4 bytes at a
          time. MLX's Metal GEMM pads each threadgroup tile by 16 bytes and shapes the tile by how the operand is stored.
          llama.cpp's quantised matrix multiplication pads every block by 16 bytes and asserts that the row stride, counted
          in 32-bit words, is 4 more than a multiple of 8 on its <code>mma</code> path (odd on its <code>dp4a</code> path).
          Same idea as the extra column, for wider accesses.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Đọc từ <code>steel/gemm/gemm.h</code> của MLX (<code>tgp_padding_a = 16 / sizeof(T)</code>) và{' '}
              <code>mmq.cuh</code> của llama.cpp vào 2026-11-20 (notes/today/ch06-facts.md D1, D3). Vì sao bước nhảy mod 8 = 4
              hợp với truy cập 16 byte không được mô hình hóa ở đây.
            </p>
          }
        >
          <p>
            Read from MLX's <code>steel/gemm/gemm.h</code> (<code>tgp_padding_a = 16 / sizeof(T)</code>) and llama.cpp's{' '}
            <code>mmq.cuh</code> on 2026-11-20 (notes/today/ch06-facts.md D1, D3). Why stride mod 8 = 4 suits 16-byte
            accesses is not modelled here.
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Thiết bị của bạn">Your device</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Lab 6.3 có một phép thăm dò đọc bộ nhớ chia sẻ ở từng bước nhảy của bài tập 5 và một phép chuyển vị phân ô với độ
            dài hàng là đối số lúc chạy, nên cùng một kernel chạy cả không đệm lẫn có đệm. Xem{' '}
            <a href={href('ch06', 'labs')}>trang lab</a>.
          </p>
        }
      >
        <p>
          Lab 6.3 has a probe that reads shared memory at each stride of exercise 5 and a tiled transpose with the row
          length as a run-time argument, so the same kernel runs unpadded and padded. See the{' '}
          <a href={href('ch06', 'labs')}>labs page</a>.
        </p>
      </Translated>
      <YourDevice64 />
    </>
  );
}
