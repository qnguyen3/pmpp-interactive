import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Today } from '../../components/notes';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  E3,
  FIRST_ONE,
  JUMPS,
  KATS,
  M0,
  M1,
  MPS_ROUNDS,
  ROUNDS,
  W0,
  W1,
  avalanche,
  binadeProfile,
  curandUniform,
  differingInputs,
  f32bits,
  f32short,
  hex,
  hipUniformAMD,
  m5Input,
  m5Uniform,
  mpsInput,
  onesCount,
  parse64,
  philoxTrace,
  roundsUntilAllWordsDiffer,
  sameWords,
  sampleConversions,
} from '../../sim/philox';
import { PhiloxRoundsScene } from './PhiloxRoundsScene';

// Section X1.7: counter-based random numbers. Everything the interactives show is computed by
// src/sim/philox.ts (philoxTrace, avalanche, the three conversions and the binade analysis),
// pinned in philox.test.ts, which also runs labs/x1/snippets/philox.cu on the emulator and
// philox.metal on this Mac's GPU against the same numbers.

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const MONO: CSSProperties = { fontFamily: 'var(--sans)', fontVariationSettings: "'MONO' 1" }; // i18n-ignore: CSS
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);
const big = (v: bigint) => '0x' + v.toString(16);
const MAX = (1n << 64n) - 1n;

// ---------------------------------------------------------------- the stepper
const PRESETS: { id: string; label: string; seed: bigint; stream: bigint; step: bigint }[] = [
  {
    id: 'zero',
    get label() {
      return tr('all zero', 'toàn số 0');
    },
    seed: 0n,
    stream: 0n,
    step: 0n,
  },
  {
    id: 'ones',
    get label() {
      return tr('all ones', 'toàn số 1');
    },
    seed: MAX,
    stream: MAX,
    step: MAX,
  },
  // Random123's "digits of pi" vector, expressed through M5's mapping (counter words are step lo, step hi, stream lo, stream hi)
  {
    id: 'pi',
    get label() {
      return tr('digits of pi', 'các chữ số của pi');
    },
    seed: 0x299f31d0a4093822n,
    stream: 0x0370734413198a2en,
    step: 0x85a308d3243f6a88n,
  },
];

function WordCell({ v, diff, ok, id }: { v: number; diff?: boolean; ok?: boolean; id?: string }) {
  return (
    <span data-w={id} data-value={hex(v)} data-diff={diff ? 1 : 0}
      style={{
        ...MONO, display: 'inline-block', padding: '0 0.3rem', marginRight: 3, borderRadius: 3, fontSize: '0.78rem',
        border: `1px solid ${ok ? 'var(--ok)' : diff ? 'var(--ink)' : 'var(--rule)'}`, // i18n-ignore: CSS
        background: ok ? 'var(--ok-wash)' : diff ? 'var(--rule)' : 'var(--cell)', fontWeight: diff ? 700 : 400,
      }}>
      {hex(v)}
    </span>
  );
}

function PhiloxStepper() {
  const [seedT, setSeedT] = useState('0');
  const [streamT, setStreamT] = useState('0');
  const [stepT, setStepT] = useState('0');
  const [mps, setMps] = useState(false);
  const [at, setAt] = useState(0);
  const seed = parse64(seedT);
  const stream = parse64(streamT);
  const step = parse64(stepT);
  const valid = seed !== null && stream !== null && step !== null;
  const R = mps ? MPS_ROUNDS : ROUNDS;

  const run = useMemo(() => {
    if (!valid) return null;
    const s1 = (stream! + 1n) & MAX;
    const A = mps ? mpsInput(seed!, stream!) : m5Input(seed!, stream!, step!);
    const B = mps ? mpsInput(seed!, s1) : m5Input(seed!, s1, step!);
    const ta = philoxTrace(A.ctr, A.key, R);
    const tb = philoxTrace(B.ctr, B.key, R);
    const av = avalanche(A.ctr, B.ctr, A.key, R);
    return { A, B, ta, tb, av, s1 };
  }, [seed, stream, step, mps, valid, R]);

  const set = (f: (s: string) => void) => (s: string) => {
    f(s);
    setAt(0);
  };
  const preset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)!;
    setSeedT(big(p.seed));
    setStreamT(big(p.stream));
    setStepT(big(p.step));
    setAt(0);
  };
  const activePreset = valid ? PRESETS.find((p) => p.seed === seed && p.stream === stream && p.step === step)?.id : undefined;

  if (!run)
    return (
      <Bench
        title={tr('Philox round by round', 'Philox từng vòng một')}
        task={
          <Translated vi="Nhập một seed, một stream và một step, hoặc chọn một đáp án đã biết.">
            Enter a seed, a stream and a step, or pick a known answer.
          </Translated>
        }
      >
        <PhiloxInputs {...{ seedT, streamT, stepT, setSeedT: set(setSeedT), setStreamT: set(setStreamT), setStepT: set(setStepT), preset, activePreset, mps, setMps: (v: boolean) => { setMps(v); setAt(0); } }} />
        <Translated vi={<p className="t-fail" data-invalid="">Mỗi giá trị phải là một số nguyên từ 0 đến 2⁶⁴ − 1, dạng thập phân hoặc có 0x.</p>}>
          <p className="t-fail" data-invalid="">Each value must be a whole number from 0 to 2⁶⁴ − 1, in decimal or with 0x.</p>
        </Translated>
      </Bench>
    );

  const { A, ta, tb, av, s1 } = run;
  const cur = at > 0 ? ta[at - 1] : null;
  const done = at === R;
  const kat = !mps ? KATS.find((k) => sameWords(k.ctr, A.ctr) && sameWords(k.key, A.key)) : undefined;
  const e3 = mps ? E3.find((e) => e.seed === seed && e.index === stream) : undefined;
  const out = ta[R - 1].out;
  const katOk = done && kat ? sameWords(out, kat.expected) : false;
  const mpsOk = done && e3 ? sameWords(out, e3.mps) : false;

  const narrate = !cur
    ? tr(
        `Nothing has run yet. The counter is (${A.ctr.map(hex).join(', ')}) and the key (${A.key.map(hex).join(', ')}). ${mps ? 'In PyTorch\'s Metal layout as we read it, the key is (seed high, seed low) and the index sits in words z and w, high word first.' : 'In M5\'s layout the counter is (step low, step high, stream low, stream high) and the key is (seed low, seed high).'} Press Step.`,
        `Chưa có gì chạy. Bộ đếm là (${A.ctr.map(hex).join(', ')}) và khóa là (${A.key.map(hex).join(', ')}). ${mps ? 'Trong bố cục Metal của PyTorch theo cách chúng tôi đọc, khóa là (seed cao, seed thấp) và chỉ số nằm ở các word z và w, word cao trước.' : 'Trong bố cục của M5, bộ đếm là (step thấp, step cao, stream thấp, stream cao) và khóa là (seed thấp, seed cao).'} Bấm Bước.`,
      )
    : tr(
        `Round ${cur.round}${cur.round > 1 ? `, with the key bumped by (${hex(W0)}, ${hex(W1)}) to (${cur.key.map(hex).join(', ')})` : ', with the key as given'}: ${hex(M0)} × ${hex(cur.p0.a)} = ${hex(cur.p0.hi)} ${hex(cur.p0.lo)} and ${hex(M1)} × ${hex(cur.p1.a)} = ${hex(cur.p1.hi)} ${hex(cur.p1.lo)}. The high halves are XORed with y and w and the key; the low halves move into y and w. ${av.differing[cur.round - 1]} of the 4 words now differ from stream ${s1}'s.`,
        `Vòng ${cur.round}${cur.round > 1 ? `, với khóa được tăng thêm (${hex(W0)}, ${hex(W1)}) thành (${cur.key.map(hex).join(', ')})` : ', với khóa như đã cho'}: ${hex(M0)} × ${hex(cur.p0.a)} = ${hex(cur.p0.hi)} ${hex(cur.p0.lo)} và ${hex(M1)} × ${hex(cur.p1.a)} = ${hex(cur.p1.hi)} ${hex(cur.p1.lo)}. Các nửa cao được XOR với y, w và khóa; các nửa thấp chuyển vào y và w. Giờ ${av.differing[cur.round - 1]} trên 4 word khác với của stream ${s1}.`,
      );

  return (
    <Bench
      title={tr('Philox round by round', 'Philox từng vòng một')}
      task={
        <Translated vi="Đi từng bước qua vector toàn số 0, rồi đổi stream đi một và theo dõi cột bên phải.">
          Step through the all-zero vector, then change the stream by one and watch the right column.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Phép tính 32 bit chính xác. Trái: đầu vào của bạn. Phải: cùng seed và step với stream + 1; một word in đậm có
              viền dày hơn là khác với word tương ứng bên trái. Một word màu xanh khớp với vector đã công bố.
            </>
          }
        >
          <>
            Exact 32-bit arithmetic. Left: your inputs. Right: the same seed and step with stream + 1; a word in bold with a
            heavier border differs from the one beside it on the left. A green word matches the published vector.
          </>
        </Translated>
      }
    >
      <PhiloxInputs {...{ seedT, streamT, stepT, setSeedT: set(setSeedT), setStreamT: set(setStreamT), setStepT: set(setStepT), preset, activePreset, mps, setMps: (v: boolean) => { setMps(v); setAt(0); } }} />
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setAt(0)} disabled={at === 0}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        <button onClick={() => setAt(at - 1)} disabled={at === 0}><Translated vi="Lùi">Back</Translated></button>
        <button className="primary" onClick={() => setAt(at + 1)} disabled={done}><Translated vi="Bước">Step</Translated></button>
        <button onClick={() => setAt(R)} disabled={done}><Translated vi="Chạy đến cuối">Run to the end</Translated></button>
        <span className="muted" aria-live="polite" data-step={at}>
          <Translated vi={<>vòng {at} trên {R}</>}>round {at} of {R}</Translated>
        </span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>{narrate}</div>

      <div style={{ overflowX: 'auto' }}>
        <table className="threads" style={{ minWidth: '44rem' }} data-part="rounds">
          <thead>
            <tr>
              <th><Translated vi="vòng">round</Translated></th>
              <th><Translated vi="khóa">key</Translated></th>
              <th><Translated vi={<>stream {stream!.toString()}</>}>stream {stream!.toString()}</Translated></th>
              <th><Translated vi={<>stream {s1.toString()}</>}>stream {s1.toString()}</Translated></th>
              <th><Translated vi="khác">differ</Translated></th>
            </tr>
          </thead>
          <tbody>
            <tr data-round={0}>
              <td><Translated vi="vào">in</Translated></td>
              <td>{A.key.map((k, i) => <WordCell key={i} v={k} />)}</td>
              <td>{A.ctr.map((v, i) => <WordCell key={i} v={v} id={`in-a-${i}`} />)}</td>
              <td>{run.B.ctr.map((v, i) => <WordCell key={i} v={v} id={`in-b-${i}`} diff={v !== A.ctr[i]} />)}</td>
              <td>{A.ctr.filter((v, i) => v !== run.B.ctr[i]).length}</td>
            </tr>
            {ta.slice(0, at).map((r, k) => (
              <tr key={k} data-round={r.round} data-differing={av.differing[k]} className={k === at - 1 ? 'cur' : ''}>
                <td>{r.round}</td>
                <td>{r.key.map((v, i) => <WordCell key={i} v={v} />)}</td>
                <td>{r.out.map((v, i) => <WordCell key={i} v={v} id={`a-${i}`} ok={k === R - 1 && (katOk || mpsOk)} />)}</td>
                <td>{tb[k].out.map((v, i) => <WordCell key={i} v={v} id={`b-${i}`} diff={av.masks[k][i]} />)}</td>
                <td>{av.differing[k]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {cur ? (
        <div className="panel" style={{ marginTop: '0.8rem' }} data-part="detail">
          <div className="mx-label">
            <Translated vi={<>Chi tiết vòng {cur.round} của cột bên trái</>}>Round {cur.round} of the left column in detail</Translated>
          </div>
          <div style={{ ...MONO, fontSize: '0.8rem', lineHeight: 1.7 }}>
            <Translated vi={<div>{hex(M0)} × x = {hex(cur.p0.a)} → cao {hex(cur.p0.hi)}, thấp {hex(cur.p0.lo)}</div>}>
              <div>{hex(M0)} × x = {hex(cur.p0.a)} → high {hex(cur.p0.hi)}, low {hex(cur.p0.lo)}</div>
            </Translated>
            <Translated vi={<div>{hex(M1)} × z = {hex(cur.p1.a)} → cao {hex(cur.p1.hi)}, thấp {hex(cur.p1.lo)}</div>}>
              <div>{hex(M1)} × z = {hex(cur.p1.a)} → high {hex(cur.p1.hi)}, low {hex(cur.p1.lo)}</div>
            </Translated>
            <Translated vi={<div>x mới = {hex(cur.p1.hi)} ⊕ y {hex(cur.ctrIn[1])} ⊕ khóa {hex(cur.key[0])} = <b>{hex(cur.out[0])}</b>; y mới = thấp 1 = <b>{hex(cur.out[1])}</b></div>}>
              <div>new x = {hex(cur.p1.hi)} ⊕ y {hex(cur.ctrIn[1])} ⊕ key {hex(cur.key[0])} = <b>{hex(cur.out[0])}</b>; new y = low 1 = <b>{hex(cur.out[1])}</b></div>
            </Translated>
            <Translated vi={<div>z mới = {hex(cur.p0.hi)} ⊕ w {hex(cur.ctrIn[3])} ⊕ khóa {hex(cur.key[1])} = <b>{hex(cur.out[2])}</b>; w mới = thấp 0 = <b>{hex(cur.out[3])}</b></div>}>
              <div>new z = {hex(cur.p0.hi)} ⊕ w {hex(cur.ctrIn[3])} ⊕ key {hex(cur.key[1])} = <b>{hex(cur.out[2])}</b>; new w = low 0 = <b>{hex(cur.out[3])}</b></div>
            </Translated>
          </div>
        </div>
      ) : null}

      {done ? (
        <div style={{ marginTop: '0.8rem' }} data-part="result" data-kat={kat?.id ?? ''} data-kat-ok={katOk ? 1 : 0} data-mps-ok={mpsOk ? 1 : 0} data-all-by={av.allBy ?? ''}>
          {kat ? (
            <p style={{ margin: 0 }}>
              {katOk ? <b className="t-ok">✓ </b> : <b className="t-fail">✗ </b>}
              <Translated
                vi={
                  <>
                    Vector "{kat.label}" của Random123: {kat.expected.map(hex).join(' ')}.{' '}
                    {katOk ? 'Đúng chính xác các word này.' : 'Các word không khớp.'}
                  </>
                }
              >
                Random123's vector "{kat.label}": {kat.expected.map(hex).join(' ')}. {katOk ? 'Exactly these words.' : 'The words do not match.'}
              </Translated>
            </p>
          ) : mps ? (
            <p style={{ margin: 0 }}>
              {e3 ? (
                <>
                  {mpsOk ? <b className="t-ok">✓ </b> : <b className="t-fail">✗ </b>}
                  <Translated
                    vi={
                      <>
                        Bản chép của E3 cho Philox Metal của PyTorch với seed {e3.seed.toString()}, chỉ số{' '}
                        {e3.index.toString()}: {e3.mps.map(hex).join(' ')}. Philox4x32-10 theo bố cục của M5 cho{' '}
                        {e3.ten.map(hex).join(' ')} với cùng seed và chỉ số: một stream khác.
                      </>
                    }
                  >
                    E3's transcription of PyTorch's Metal Philox for seed {e3.seed.toString()}, index {e3.index.toString()}:{' '}
                    {e3.mps.map(hex).join(' ')}. Philox4x32-10 in M5's layout gives {e3.ten.map(hex).join(' ')} for the same seed
                    and index: a different stream.
                  </Translated>
                </>
              ) : (
                <Translated vi={<>Chín vòng theo thứ tự word của PyTorch. Seed 0 và 42 với stream 0 có các vector đã chép (E3) để so sánh.</>}>
                  <>Nine rounds in PyTorch's word order. Seeds 0 and 42 with stream 0 have transcribed vectors (E3) to compare with.</>
                </Translated>
              )}
            </p>
          ) : (
            <Translated
              vi={
                <p style={{ margin: 0 }} className="muted">
                  Không có vector đã công bố cho các đầu vào này; mỗi preset có một vector. Các word này là những gì mọi
                  Philox4x32-10 đúng phải tạo ra cho chúng.
                </p>
              }
            >
              <p style={{ margin: 0 }} className="muted">
                No published vector for these inputs; the presets have one each. These words are what any correct Philox4x32-10 must
                produce for them.
              </p>
            </Translated>
          )}
          {!mps ? (
            <p style={{ margin: '0.4rem 0 0' }} data-u={m5Uniform(out[0])}>
              <Translated
                vi={
                  <>
                    u của M5 cho hàng và step này: word 0 là {hex(out[0])}, 24 bit cao của nó là{' '}
                    {(out[0] >>> 8).toLocaleString('en-US')}, và u = {(out[0] >>> 8).toLocaleString('en-US')} × 2⁻²⁴ ={' '}
                    <b>{f32short(m5Uniform(out[0]))}</b>.
                  </>
                }
              >
                M5's u for this row and step: word 0 is {hex(out[0])}, its top 24 bits are {(out[0] >>> 8).toLocaleString('en-US')},
                and u = {(out[0] >>> 8).toLocaleString('en-US')} × 2⁻²⁴ = <b>{f32short(m5Uniform(out[0]))}</b>.
              </Translated>
            </p>
          ) : null}
          <p style={{ margin: '0.4rem 0 0' }}>
            <Translated
              vi={
                <>
                  Stream {stream!.toString()} và {s1.toString()}: cả bốn word khác nhau từ vòng{' '}
                  <b data-allby="">{av.allBy ?? 'không bao giờ'}</b> trở đi.
                </>
              }
            >
              Streams {stream!.toString()} and {s1.toString()}: all four words differ from round <b data-allby="">{av.allBy ?? 'never'}</b> on.
            </Translated>
          </p>
        </div>
      ) : null}
    </Bench>
  );
}

function PhiloxInputs(p: {
  seedT: string; streamT: string; stepT: string;
  setSeedT: (s: string) => void; setStreamT: (s: string) => void; setStepT: (s: string) => void;
  preset: (id: string) => void; activePreset?: string; mps: boolean; setMps: (v: boolean) => void;
}) {
  return (
    <>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="seed (64 bit)">seed (64-bit)</Translated>
          <input type="text" value={p.seedT} aria-label={tr('seed', 'seed')} onChange={(e) => p.setSeedT(e.target.value)} style={{ width: '12.5rem' }} />
        </label>
        <label className="field">
          <Translated vi="stream (request của hàng)">stream (the row's request)</Translated>
          <input type="text" value={p.streamT} aria-label={tr('stream', 'stream')} onChange={(e) => p.setStreamT(e.target.value)} style={{ width: '12.5rem' }} />
        </label>
        <label className="field">
          <Translated vi="step">step</Translated>
          <input type="text" value={p.stepT} aria-label={tr('step', 'step')} disabled={p.mps} onChange={(e) => p.setStepT(e.target.value)} style={{ width: '12.5rem' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          <Translated vi="Đáp án đã biết">Known answers</Translated>
          <span className="seg" role="group" aria-label={tr('Known answers', 'Đáp án đã biết')}>
            {PRESETS.map((x) => (
              <button key={x.id} aria-pressed={p.activePreset === x.id} onClick={() => p.preset(x.id)}>{x.label}</button>
            ))}
          </span>
        </span>
        <label className="check">
          <input type="checkbox" checked={p.mps} aria-label={tr('PyTorch MPS as we read its source', 'PyTorch MPS theo cách chúng tôi đọc mã nguồn')} onChange={(e) => p.setMps(e.target.checked)} />
          <Translated vi="PyTorch MPS theo cách chúng tôi đọc mã nguồn (9 vòng, word cao trước; chưa chạy)">
            PyTorch MPS as we read its source (9 rounds, high word first; not run)
          </Translated>
        </label>
      </div>
    </>
  );
}

// ---------------------------------------------------------------- bits to u
function BitsExplorer() {
  const [x, setX] = useState(FIRST_ONE);
  const [text, setText] = useState('0x' + FIRST_ONE.toString(16));
  const profile = useMemo(() => binadeProfile(), []);
  const sample = useMemo(() => sampleConversions(200000, 1), []);
  const setFrom = (s: string) => {
    setText(s);
    const v = parse64(s);
    if (v !== null && v <= 0xffffffffn) setX(Number(v));
  };
  const jump = (v: number) => {
    setX(v);
    setText(v < 4096 ? String(v) : '0x' + v.toString(16));
  };
  const c = curandUniform(x);
  const h = hipUniformAMD(x);
  const m = m5Uniform(x);
  const agree = c === h;
  const binade = x === 0 ? -1 : Math.floor(Math.log2(x));
  const rows: { id: string; label: string; v: number; note: string }[] = [
    { id: 'curand', label: 'curand_uniform (NVIDIA): x·2⁻³² + 2⁻³³', v: c, note: '(0, 1]' }, // i18n-ignore: formula
    { id: 'hip', label: tr('hiprand_uniform on AMD (rocRAND): 2⁻³² + x·2⁻³²', 'hiprand_uniform trên AMD (rocRAND): 2⁻³² + x·2⁻³²'), v: h, note: '(0, 1]' },
    { id: 'm5', label: 'M5: (x >> 8) × 2⁻²⁴', v: m, note: '[0, 1 − 2⁻²⁴]' },
  ];
  const differ = differingInputs();
  return (
    <Bench
      title={tr('Bits to u', 'Từ bit sang u')}
      task={
        <Translated vi="Nhảy tới 0, rồi 2²³ và 2²⁴ + 1: các nhà cung cấp bất đồng ở đâu? Rồi 0xFFFFFF80.">
          Jump to 0, then 2²³ and 2²⁴ + 1: where do the vendors disagree? Then 0xFFFFFF80.
        </Translated>
      }
      foot={
        <Translated vi={<>Mỗi phép chuyển đổi được làm tròn về float32, làm tròn về số chẵn gần nhất, một lần cho mỗi phép toán, như mọi backend làm với chính các phép toán này.</>}>
          <>Each conversion rounded to float32, round to nearest even, once per operation, as every backend does for these exact operations.</>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field">
          <Translated vi="x (32 bit, thập phân hoặc 0x)">x (32 bits, decimal or 0x)</Translated>
          <input type="text" value={text} aria-label="x" onChange={(e) => setFrom(e.target.value)} style={{ width: '10rem' }} />
        </label>
        <span style={FIELD}>
          <Translated vi="Nhảy tới">Jump to</Translated>
          <span className="seg" role="group" aria-label={tr('Jump to', 'Nhảy tới')} style={{ flexWrap: 'wrap' }}>
            {JUMPS.map((j) => (
              <button key={j.label} aria-pressed={x === j.x} onClick={() => jump(j.x)}>{j.label}</button>
            ))}
          </span>
        </span>
      </div>
      <input type="range" min={0} max={4294967295} step={1} value={x} aria-label={tr('x slider', 'thanh trượt x')} style={{ width: '100%', marginBottom: '0.6rem' }}
        onChange={(e) => jump(Number(e.target.value))} />
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '36rem' }} data-part="conversions" data-x={x}>
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}><Translated vi="Phép chuyển đổi">Conversion</Translated></th>
              <th className="num">u</th>
              <th><Translated vi="các bit float32">float32 bits</Translated></th>
              <th><Translated vi="miền giá trị">range</Translated></th>
              <th><Translated vi="1.0?">1.0?</Translated></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id} data-conv={r.id} data-u={r.v} data-bits={hex(f32bits(r.v))}>
                <td style={{ paddingLeft: '0.6rem' }}>{r.label}</td>
                <td className="num" style={MONO}>{f32short(r.v)}</td>
                <td style={MONO}>{hex(f32bits(r.v))}</td>
                <td>{r.note}</td>
                <td>{r.v === 1 ? <b className="t-fail"><Translated vi="đúng bằng 1.0">exactly 1.0</Translated></b> : tr('no', 'không')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.7rem 0 0' }} data-agree={agree ? 1 : 0}>
        <Translated
          vi={
            <>
              Với x = {x.toLocaleString('en-US')}: NVIDIA và AMD{' '}
              {agree ? <b className="t-ok">trùng nhau</b> : <b className="t-fail">cho các số float khác nhau</b>}
              {agree ? '' : ` (${f32short(c)} và ${f32short(h)})`}.
            </>
          }
        >
          For x = {x.toLocaleString('en-US')}: NVIDIA and AMD {agree ? <b className="t-ok">agree</b> : <b className="t-fail">give different floats</b>}
          {agree ? '' : ` (${f32short(c)} and ${f32short(h)})`}.
        </Translated>
      </p>

      <div className="panel" style={{ marginTop: '0.8rem' }} data-part="strip">
        <div className="mx-label">
          <Translated vi="Nơi các nhà cung cấp bất đồng, theo từng binade: x = 0, rồi x trong [2ᵏ, 2ᵏ⁺¹) với k = 0 đến 31">
            Where the vendors disagree, binade by binade: x = 0, then x in [2ᵏ, 2ᵏ⁺¹) for k = 0 to 31
          </Translated>
        </div>
        <div style={{ display: 'flex', gap: 2, alignItems: 'flex-end', overflowX: 'auto' }}>
          {profile.map((r) => {
            const share = r.differ / r.size;
            return (
              <div key={r.k} data-binade={r.k} data-share={share}
                title={tr(
                  `k = ${r.k}: ${r.differ.toLocaleString('en-US')} of ${r.size.toLocaleString('en-US')} differ`,
                  `k = ${r.k}: ${r.differ.toLocaleString('en-US')} trên ${r.size.toLocaleString('en-US')} khác nhau`,
                )}
                style={{ flex: '1 1 0', minWidth: 10, textAlign: 'center', fontSize: '0.6rem', color: 'var(--ink-2)' }}>
                <div style={{ height: 48, display: 'flex', alignItems: 'flex-end', border: `1px solid ${binade === r.k ? 'var(--ink)' : 'var(--rule)'}` }}>
                  <div style={{ width: '100%', height: `${share * 100}%`, background: 'var(--fail)' }} />
                </div>
                <div style={{ height: 10 }}>{r.onesCurand ? <b style={{ color: 'var(--fail)' }}>1.0</b> : ''}</div>
                {r.k >= 0 && r.k % 4 === 0 ? r.k : ' '}
              </div>
            );
          })}
        </div>
        <p style={{ margin: '0.5rem 0 0', fontSize: '0.85rem' }} data-differ={differ} data-ones={onesCount('curand')} data-ones-hip={onesCount('hip')}>
          <Translated
            vi={
              <>
                Chiều cao cột: phần của binade mà tại đó hai phép chuyển đổi của nhà cung cấp khác nhau. Mọi x dưới 2²³ đều
                khác; một nửa của [2²³, 2²⁴); một phần tư của [2²⁴, 2²⁵), nơi 2⁻³² mà AMD cộng thêm đúng bằng nửa bước float
                và trường hợp hòa được làm tròn về số chẵn; phía trên thì không có. Tổng cộng{' '}
                <b>{differ.toLocaleString('en-US')}</b> trên 2³² đầu vào khác nhau, một trên {2 ** 32 / differ}. Cả hai phép
                chuyển đổi trả về đúng 1.0 cho <b>{onesCount('curand')}</b> đầu vào cao nhất (x ≥ 0xFFFFFF80). Của M5 thì
                không bao giờ.
              </>
            }
          >
            Bar height: the share of the binade where the two vendor conversions differ. Every x below 2²³ differs; half of [2²³,
            2²⁴); a quarter of [2²⁴, 2²⁵), where AMD's added 2⁻³² is exactly half a float step and the tie rounds to even; none
            above. In all <b>{differ.toLocaleString('en-US')}</b> of 2³² inputs differ, one in {2 ** 32 / differ}. Both conversions
            return exactly 1.0 for the top <b>{onesCount('curand')}</b> inputs (x ≥ 0xFFFFFF80). M5's never does.
          </Translated>
        </p>
        <p className="muted" style={{ margin: '0.4rem 0 0', fontSize: '0.8rem' }} data-sample={sample.differ}>
          <Translated
            vi={
              <>
                Kiểm tra trên một mẫu có seed gồm {sample.n.toLocaleString('en-US')} word Philox:{' '}
                {sample.differ.toLocaleString('en-US')} khác nhau ({((sample.differ / sample.n) * 1e6).toFixed(0)} trên một
                triệu; phân tích nói {(1e6 / 256).toFixed(0)}).{' '}
              </>
            }
          >
            Check on a seeded sample of {sample.n.toLocaleString('en-US')} Philox words: {sample.differ.toLocaleString('en-US')} differ
            ({((sample.differ / sample.n) * 1e6).toFixed(0)} per million; the analysis says {(1e6 / 256).toFixed(0)}).{' '}
          </Translated>
          {sample.m5AtOne
            ? tr(`M5's u reached 1.0 ${sample.m5AtOne} times.`, `u của M5 chạm 1.0 ${sample.m5AtOne} lần.`)
            : tr("M5's u never reached 1.0.", 'u của M5 không bao giờ chạm 1.0.')}
        </p>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSX17: Record<string, Question> = {
  'x1/X1.7/q-stateless': {
    kind: 'choice',
    id: 'x1/X1.7/q-stateless',
    prompt: (
      <Translated vi={<>Vì sao một bộ sinh dựa trên bộ đếm hợp với một sampler theo batch hơn một <code>std::mt19937</code> dùng chung cho cả batch?</>}>
        <>Why does a counter-based generator suit a batched sampler better than one <code>std::mt19937</code> shared by the batch?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Mỗi (seed, stream, step) ánh xạ thẳng tới các bit của nó, nên mỗi hàng tự tính u của mình theo bất kỳ thứ tự nào">
            Each (seed, stream, step) maps straight to its bits, so each row computes its own u in any order
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Philox có chu kỳ dài hơn mt19937, nên một lần sinh dài ở batch lớn không bao giờ lặp lại một số">
            Philox has a longer period than mt19937, so a long generation at a large batch never repeats a number
          </Translated>
        ),
        why: (
          <Translated vi="Chu kỳ của mt19937 dài hơn rất nhiều so với bất kỳ lần sinh nào; chu kỳ không phải vấn đề, trạng thái tuần tự dùng chung mới là vấn đề.">
            mt19937's period is far longer than any generation; the period is not the problem, the shared sequential state is.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="mt19937 không chạy trên GPU, nên host sẽ phải rút mọi u và tải lên ở mỗi bước">
            mt19937 does not run on a GPU, so the host would have to draw every u and upload it at each step
          </Translated>
        ),
        why: (
          <Translated vi="Nó có thể được chuyển sang GPU; rắc rối là giá trị tiếp theo của nó phụ thuộc vào mọi giá trị trước đó, nên các hàng sẽ phải chờ nhau và chờ thứ tự batch.">
            It can be ported; the trouble is that its next value depends on every value before it, so rows would wait on each other and on the batch order.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Một bộ sinh tuần tự phát giá trị theo thứ tự: u của hàng 5 phụ thuộc vào số giá trị mà các hàng 0 đến 4 đã lấy ở
            bước này, điều thay đổi theo batch. Một bộ sinh dựa trên bộ đếm là một hàm thuần của (counter, key): không luồng
            nào phải chờ, và bất kỳ backend nào hiện thực cùng hàm đó đều nhận được cùng các bit.
          </p>
        }
      >
        <p>
          A sequential generator hands out values in order: row 5's u depends on how many values rows 0 to 4 took this step,
          which changes with the batch. A counter-based generator is a pure function of (counter, key): no thread waits, and
          any backend that implements the same function gets the same bits.
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md A4, D3 (checked 2026-11-20)',
    vi: { source: 'notes/today/x1-facts.md A4, D3 (kiểm tra 2026-11-20)' },
  },
  'x1/X1.7/q-mapping': {
    kind: 'choice',
    id: 'x1/X1.7/q-mapping',
    prompt: (
      <Translated vi={<>Theo cách chúng tôi đọc header của cuRAND, lời gọi Philox nào tạo ra <code>curand()</code> đầu tiên sau <code>curand_init(seed, stream, 4 × step, &s)</code>?</>}>
        <>By our reading of cuRAND's header, which Philox call produces the first <code>curand()</code> after <code>curand_init(seed, stream, 4 × step, &s)</code>?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Counter (step thấp, step cao, stream thấp, stream cao), key (seed thấp, seed cao), word đầu ra 0">
            Counter (step low, step high, stream low, stream high), key (seed low, seed high), output word 0
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Counter (stream thấp, stream cao, step thấp, step cao), key (seed thấp, seed cao), word đầu ra 0">
            Counter (stream low, stream high, step low, step high), key (seed low, seed high), output word 0
          </Translated>
        ),
        why: (
          <Translated vi="Subsequence đi vào các word z và w, offset / 4 đi vào x và y.">
            The subsequence goes into words z and w, the offset / 4 into x and y.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Counter (step thấp, step cao, stream thấp, stream cao), key (seed cao, seed thấp), word đầu ra 3">
            Counter (step low, step high, stream low, stream high), key (seed high, seed low), output word 3
          </Translated>
        ),
        why: (
          <Translated vi="Word thấp của khóa đứng trước, và trạng thái bắt đầu tại offset mod 4 = 0, nên là word 0.">
            The key's low word comes first, and the state starts at offset mod 4 = 0, so word 0.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>curand_init</code> đặt seed vào khóa (thấp, cao), cộng subsequence vào các word z và w của counter, cộng
            offset / 4 vào x và y, và bắt đầu tại word offset mod 4. Với offset 4 × step thì đó là word 0 của counter M5. Đây
            là suy luận của chúng tôi từ header, unverified trên thiết bị cho đến khi một báo cáo verify CUDA cho thấy phép
            kiểm tra đáp án đã biết của lab X1.3 vượt qua.
          </p>
        }
      >
        <p>
          <code>curand_init</code> puts the seed in the key (low, high), adds the subsequence to counter words z and w, adds
          offset / 4 to x and y, and starts at word offset mod 4. With offset 4 × step that is word 0 of M5's counter. This is
          our derivation from the header, unverified on a device until a CUDA verify report shows lab X1.3's known-answer test
          passing.
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md A3, E1 (our derivation; unverified on a device)',
    vi: { source: 'notes/today/x1-facts.md A3, E1 (suy luận của chúng tôi; unverified trên thiết bị)' },
  },
  'x1/X1.7/q-one': {
    kind: 'numeric',
    id: 'x1/X1.7/q-one',
    prompt: (
      <Translated vi={<>Với bao nhiêu trong số 2³² đầu vào có thể, <code>curand_uniform</code> trả về đúng 1.0?</>}>
        <>For how many of the 2³² possible inputs does <code>curand_uniform</code> return exactly 1.0?</>
      </Translated>
    ),
    answer: 128,
    unit: 'inputs',
    explain: (
      <Translated
        vi={
          <p>
            Mọi x từ 0xFFFFFF80 trở lên chuyển thành số float 2³² (float32 gần nhất), nên x·2⁻³² là 1.0 và cộng 2⁻³³ vẫn giữ
            nguyên: 2³² − 0xFFFFFF80 = 128 đầu vào. Phép chuyển đổi của AMD trả về 1.0 cho cùng 128 đầu vào đó.
          </p>
        }
      >
        <p>
          Every x from 0xFFFFFF80 up converts to the float 2³² (the nearest float32), so x·2⁻³² is 1.0 and adding 2⁻³³ leaves
          it there: 2³² − 0xFFFFFF80 = 128 inputs. AMD's conversion returns 1.0 for the same 128.
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md A2, E2; our arithmetic, src/sim/philox.ts',
    vi: { unit: 'đầu vào', source: 'notes/today/x1-facts.md A2, E2; tính toán của chúng tôi, src/sim/philox.ts' },
  },
  'x1/X1.7/q-differ': {
    kind: 'numeric',
    id: 'x1/X1.7/q-differ',
    prompt: (
      <Translated vi={<>Với cùng các bit thô, <code>curand_uniform</code> trên NVIDIA và <code>hiprand_uniform</code> trên AMD trả về số float khác nhau cho 1 trên bao nhiêu đầu vào?</>}>
        <>For identical raw bits, <code>curand_uniform</code> on NVIDIA and <code>hiprand_uniform</code> on AMD return different floats for 1 in how many inputs?</>
      </Translated>
    ),
    answer: 256,
    explain: (
      <Translated
        vi={
          <p>
            Đúng 2²⁴ trên 2³² đầu vào, theo phân tích binade của chúng tôi: mọi x dưới 2²³, các x chẵn trong [2²³, 2²⁴), và
            một phần tư của [2²⁴, 2²⁵) nơi 2⁻³² của AMD là nửa bước float và trường hợp hòa làm tròn về số chẵn. Lượt kiểm tra
            dữ kiện lấy mẫu được 3,903 trên một triệu, khớp với kết quả này.
          </p>
        }
      >
        <p>
          Exactly 2²⁴ of 2³² inputs, by our binade analysis: all x below 2²³, the even x in [2²³, 2²⁴), and a quarter of [2²⁴,
          2²⁵) where AMD's 2⁻³² is half a float step and ties round to even. The facts pass sampled 3,903 per million, which
          agrees.
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md B2, E2; our arithmetic, src/sim/philox.ts',
    vi: { source: 'notes/today/x1-facts.md B2, E2; tính toán của chúng tôi, src/sim/philox.ts' },
  },
  'x1/X1.7/q-metal': {
    kind: 'choice',
    id: 'x1/X1.7/q-metal',
    prompt: (
      <Translated vi={<>Metal Shading Language cho kernel thứ gì để sinh số ngẫu nhiên?</>}>
        <>What does the Metal Shading Language give a kernel for random numbers?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không gì cả: MSL 4.1 không có hàm random, nên M5 viết Philox bằng MSL với mulhi hoặc một tích ulong">
            Nothing: MSL 4.1 has no random function, so M5 writes Philox in MSL with mulhi or a ulong product
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một kiểu trạng thái Philox và một hàm uniform, giống device API của cuRAND">
            A Philox state type and a uniform function, like cuRAND's device API
          </Translated>
        ),
        why: (
          <Translated vi="Philox của Apple là MPSMatrixRandomPhilox, được encode từ host, và các op MPSGraph; không cái nào gọi được từ shader.">
            Apple's Philox is MPSMatrixRandomPhilox, encoded from the host, and MPSGraph ops; neither is callable from a shader.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một hàm noise dựa trên hash dành cho shader đồ họa, mà sampler có thể gọi một lần mỗi hàng để lấy u">
            A hash-based noise function meant for graphics shaders, which a sampler can call once per row for its u
          </Translated>
        ),
        why: (
          <Translated vi={'Đặc tả hoàn toàn không có hàm random, noise hay hash; "random" xuất hiện một lần, nói về iterator.'}>
            The spec has no random, noise or hash function at all; "random" appears once, about iterators.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Chữ "random" duy nhất trong đặc tả 4.1 là "random access iterator". Philox mà Apple ghi trong tài liệu là một
            kernel điền dữ liệu được encode từ host và các op MPSGraph mà số vòng, cách gieo seed và phép chuyển đổi không được
            ghi lại. MSL có <code>mulhi</code> và <code>ulong</code>, đó là tất cả những gì Philox cần.
          </p>
        }
      >
        <p>
          The 4.1 spec's only "random" is "random access iterator". Apple's documented Philox is a host-encoded fill kernel
          and MPSGraph ops whose rounds, seeding and conversion are not documented. MSL does have <code>mulhi</code> and{' '}
          <code>ulong</code>, which is all Philox needs.
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md C1, C2, C3, C4 (checked 2026-11-20)',
    vi: { source: 'notes/today/x1-facts.md C1, C2, C3, C4 (kiểm tra 2026-11-20)' },
  },
  'x1/X1.7/q-raw-bits': {
    kind: 'choice',
    id: 'x1/X1.7/q-raw-bits',
    prompt: (
      <Translated vi={<>Vì sao M5 lấy các word 32 bit thô và tự chuyển đổi thay vì gọi <code>curand_uniform</code>?</>}>
        <>Why does M5 take raw 32-bit words and convert them itself instead of calling <code>curand_uniform</code>?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="cuRAND không cố định số giá trị thô một lời gọi tiêu thụ, các phép chuyển đổi của các nhà cung cấp khác nhau, và Metal không có cái nào">
            cuRAND does not fix how many raw values a call consumes, the vendors' conversions differ, and Metal has none
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="curand_uniform chậm hơn một phép dịch bit và một phép nhân">
            curand_uniform is slower than a shift and a multiply
          </Translated>
        ),
        why: (
          <Translated vi="Tốc độ không phải lý do; sự nhất quán giữa các backend mới là lý do.">
            Speed is not the reason; agreement across backends is.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="curand_uniform trả về double trong (0, 1], và Metal không có double, nên một kernel Metal sẽ cần thêm một phép chuyển đổi">
            curand_uniform returns a double in (0, 1], and Metal has no double, so a Metal kernel would need another conversion
          </Translated>
        ),
        why: (
          <Translated vi="Nó trả về float trong (0, 1]. Vấn đề là float nào, và trên backend nào.">
            It returns a float in (0, 1]. The problem is which float, and on which backend.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            "The number of values consumed is not guaranteed to be fixed" (tài liệu của cuRAND). Công thức của NVIDIA và AMD
            khác nhau với 1 trên 256 đầu vào và cả hai đều có thể trả về 1.0. <code>(x &gt;&gt; 8) × 2⁻²⁴</code> là một phép
            dịch số nguyên, một phép chuyển đổi chính xác và một phép nhân với lũy thừa của hai: cùng một float trên mọi
            backend dưới bất kỳ chế độ toán nào.
          </p>
        }
      >
        <p>
          "The number of values consumed is not guaranteed to be fixed" (cuRAND's docs). NVIDIA's and AMD's formulas differ for
          1 in 256 inputs and both can return 1.0. <code>(x &gt;&gt; 8) × 2⁻²⁴</code> is an integer shift, an exact conversion
          and a power-of-two scale: the same float on every backend under any math mode.
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md A1, A2, B2, C1, E2', // i18n-ignore: citation
  },
  'x1/X1.7/q-slot': {
    kind: 'choice',
    id: 'x1/X1.7/q-slot',
    prompt: (
      <Translated
        vi={
          <>
            Dưới continuous batching, một request nằm ở hàng 3 ở một bước và hàng 1 ở bước sau. Nếu subsequence của bộ sinh là
            hàng trong batch, như trong kernel của FlashInfer, thì tính tái lập của request đó ra sao?
          </>
        }
      >
        Under continuous batching a request sits in row 3 at one step and row 1 at the next. If the generator's subsequence
        were the batch row, as in FlashInfer's kernel, what happens to that request's reproducibility?
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="u của nó đi theo vị trí của nó, nên cùng seed trong một tổ hợp batch khác cho các token khác">
            Its u follows its slot, so the same seed in another batch mix gives different tokens
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Không sao: chỉ riêng seed đã cố định stream, nên các token của request không phụ thuộc vào hàng của nó">
            Nothing: the seed alone fixes the stream, so the request's tokens do not depend on its row
          </Translated>
        ),
        why: (
          <Translated vi="Seed là khóa; hàng chọn subsequence, nên vị trí làm thay đổi các bit.">
            The seed is the key; the row picks the subsequence, so the slot changes the bits.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Chỉ token đầu tiên sau khi chuyển là thay đổi; từ bước sau trở đi, các số của nó khớp lại">
            Only the first token after the move changes; from the next step on, its numbers realign
          </Translated>
        ),
        why: (
          <Translated vi="Mỗi bước có vị trí khác đều rút từ một subsequence khác.">
            Every step whose slot differs draws from a different subsequence.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            FlashInfer gọi <code>curand_init(philox_seed, bx, philox_offset, &state)</code> với chỉ số khối làm subsequence.
            Điều đó tái lập được với một batch cố định. M5 biến stream và step thành đầu vào theo từng hàng, nên các bit của
            một request đi theo request, không theo vị trí (lựa chọn thiết kế của chúng tôi).
          </p>
        }
      >
        <p>
          FlashInfer calls <code>curand_init(philox_seed, bx, philox_offset, &state)</code> with the block index as the
          subsequence. That is reproducible for a fixed batch. M5 makes the stream and the step inputs per row, so a request's
          bits follow the request, not the slot (our design choice).
        </p>
      </Translated>
    ),
    source: 'notes/today/x1-facts.md A6; ours',
    vi: { source: 'notes/today/x1-facts.md A6; của khóa học này' },
  },
};

// ---------------------------------------------------------------- the section
export function SX17() {
  const cu = loaded(useLabFile('x1/snippets/philox.cu'));
  const metal = loaded(useLabFile('x1/snippets/philox.metal'));
  return (
    <>
      <Translated
        vi={
          <p>
            Giai đoạn cuối của sampler cần một số đều u cho mỗi hàng và mỗi bước. Mọi backend phải tạo ra cùng u từ cùng đầu
            vào, nếu không cùng một request sẽ rút ra các token khác nhau trên CUDA, HIP và Metal. Trang này dựng con số đó từ
            ba đầu vào cho mỗi hàng: một seed, một stream và một step.
          </p>
        }
      >
        <p>
          The last stage of the sampler needs one uniform number u per row and per step. Every backend must produce the same u
          from the same inputs, or the same request will draw different tokens on CUDA, HIP and Metal. This page builds that
          number from three inputs per row: a seed, a stream and a step.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Không có gì ở đây nằm trong sách. Philox của Random123 dùng giấy phép BSD và được trình bày kèm thông báo của
              nó. Các header của cuRAND theo giấy phép hạn chế của NVIDIA, nên ở đây chúng chỉ được mô tả, không bao giờ được
              trình bày (x1-facts, ghi chú về giấy phép).
            </p>
          }
        >
          <p>
            Nothing here is in the book. Random123's Philox is BSD-licensed and is shown with its notice. cuRAND's headers are
            under NVIDIA's restrictive licence, so they are described here, never shown (x1-facts, licence note).
          </p>
        </Translated>
      </Beyond>

      <h2>
        <Translated vi="Vì sao không dùng một bộ sinh cho cả batch">Why not one generator for the batch</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sampler trên host của llama.cpp rút từ một <code>std::mt19937</code>. Đó là một trạng thái tuần tự: mỗi giá trị
            phụ thuộc vào mọi giá trị trước nó. Một bộ sinh dùng chung cho cả batch khiến u của hàng 5 phụ thuộc vào số giá
            trị mà các hàng 0 đến 4 đã lấy, nên thứ tự batch rò rỉ vào mọi token, và không hàng nào rút được cho đến khi các
            hàng trước nó rút xong.
          </p>
        }
      >
        <p>
          llama.cpp's host sampler draws from a <code>std::mt19937</code>. That is a sequential state: each value depends on
          every value before it. One generator shared by a batch makes row 5's u depend on how many values rows 0 to 4 took, so
          the batch order leaks into every token, and no row can draw until the rows before it have.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Một bộ sinh dựa trên bộ đếm hoàn toàn không có trạng thái. Nó là một hàm: result = f(counter, key). Đặt step và
            stream của hàng vào counter và seed vào key, và mỗi luồng tính trực tiếp các số của hàng mình, theo bất kỳ thứ tự
            nào, trên bất kỳ thiết bị nào. Bạn đã dùng một bộ sinh như vậy: <code>torch.rand</code> trên một tensor CUDA khởi
            tạo Philox của cuRAND cho mỗi luồng bằng <code>curand_init(seed, idx, offset, &state)</code>.
          </p>
        }
      >
        <p>
          A counter-based generator has no state at all. It is a function: result = f(counter, key). Put the step and the
          row's stream in the counter and the seed in the key, and every thread computes its own row's numbers directly, in
          any order, on any device. You have used one already: <code>torch.rand</code> on a CUDA tensor initialises cuRAND's
          Philox per thread with <code>curand_init(seed, idx, offset, &state)</code>.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A3, A4, D3)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              llama.cpp <code>dist</code>: <code>std::mt19937 rng</code>. Random123 (trích nguyên văn): counter-based
              generators are "stateless functions"; "a unique (counter, key) tuple will always produce the same result."
            </p>
          }
        >
          <p>
            llama.cpp <code>dist</code>: <code>std::mt19937 rng</code>. Random123: counter-based generators are "stateless
            functions"; "a unique (counter, key) tuple will always produce the same result."
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSX17['x1/X1.7/q-stateless']} />

      <h2>
        <Translated vi="Philox4x32-10, từng vòng một">Philox4x32-10, round by round</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Trạng thái của Philox là một counter 128 bit gồm bốn word 32 bit và một key 64 bit gồm hai word. Một vòng nhân
            word 0 với 0xD2511F53 và word 2 với 0xCD9E8D57. Mỗi tích 64 bit tách thành một nửa cao và một nửa thấp. Các nửa
            cao được XOR với hai word còn lại và key, các nửa thấp chuyển vào chỗ trống, và kết quả là counter của vòng kế
            tiếp. Giữa các vòng, key tăng thêm 0x9E3779B9 và 0xBB67AE85. Mười vòng tạo thành Philox4x32-10. Mỗi vòng là hai
            phép nhân số nguyên và vài phép XOR, vì vậy nó đủ rẻ để chạy một lần cho mỗi hàng ở mỗi bước.
          </p>
        }
      >
        <p>
          Philox's state is a 128-bit counter of four 32-bit words and a 64-bit key of two. One round multiplies word 0 by
          0xD2511F53 and word 2 by 0xCD9E8D57. Each 64-bit product splits into a high and a low half. The high halves are XORed
          with the other two words and the key, the low halves move into the gaps, and the result is the next round's counter.
          Between rounds the key grows by 0x9E3779B9 and 0xBB67AE85. Ten rounds make Philox4x32-10. Each round is two integer
          multiplies and a few XORs, which is why it is cheap enough to run once per row per step.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A4)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              Từ <code>philox.h</code> của Random123: đầu ra của vòng là{' '}
              <code>{'{hi1^ctr[1]^key[0], lo1, hi0^ctr[3]^key[1], lo0}'}</code>; không tăng key trước vòng đầu tiên; mặc định
              10 vòng.
            </p>
          }
        >
          <p>
            From Random123's <code>philox.h</code>: the round's output is <code>{'{hi1^ctr[1]^key[0], lo1, hi0^ctr[3]^key[1], lo0}'}</code>;
            no key bump before the first round; 10 rounds by default.
          </p>
        </Translated>
      </Today>
      <Predict
        id="x1/X1.7/p-avalanche"
        prompt={
          <Translated
            vi={
              <>
                Seed 0 và step 0. Hàng 0 và 1 khác nhau ở một bit của counter (word stream). Sau bao nhiêu vòng thì cả bốn
                word đầu ra đều khác nhau?
              </>
            }
          >
            Seed 0 and step 0. Rows 0 and 1 differ in one bit of the counter (the stream word). After how many rounds do
            all four output words differ?
          </Translated>
        }
        unit={tr('rounds', 'vòng')}
      >
        {(v) => {
          const r = roundsUntilAllWordsDiffer(0n, 0n, 0n);
          const a = avalanche(m5Input(0n, 0n, 0n).ctr, m5Input(0n, 1n, 0n).ctr, [0, 0]);
          return (
            <p data-reveal={r ?? ''}>
              <b>{r}.</b>{' '}
              <Translated
                vi={
                  <>
                    Một word khác nhau sau vòng 1 và 2 (mỗi lần một word khác), rồi {a.differing[2]} sau vòng 3,{' '}
                    {a.differing[3]} sau vòng 4, và cả bốn sau vòng 5.{' '}
                    {v === r ? 'Đúng như bạn nói.' : v < (r ?? 0) ? 'Chậm hơn bạn nghĩ: mỗi vòng chỉ đẩy một khác biệt qua hai phép nhân.' : 'Nhanh hơn bạn nghĩ: các phép nhân lan một khác biệt ra cả word ở mỗi vòng.'}{' '}
                    Năm vòng còn lại là biên an toàn: Random123 cũng công bố các vector 7 vòng, và một Philox ít vòng hơn là một
                    stream khác.
                  </>
                }
              >
                One word differs after rounds 1 and 2 (a different one each time), then {a.differing[2]} after round
                3, {a.differing[3]} after round 4, and all four after round 5. {v === r ? 'Just as you said.' : v < (r ?? 0) ? 'Slower than you expected: each round only moves a difference through the two multiplies.' : 'Faster than you expected: the multiplies spread a difference into whole words each round.'}{' '}
                The other five rounds are margin: Random123 also publishes 7-round vectors, and a Philox with fewer rounds is a
                different stream.
              </Translated>
            </p>
          );
        }}
      </Predict>
      <Translated
        vi={
          <p>
            Trình chạy từng bước chạy các vòng trên đầu vào của bạn, với một hàng thứ hai bên cạnh có stream cao hơn một. Các
            preset là ba vector đã công bố của Random123.
          </p>
        }
      >
        <p>
          The stepper runs the rounds on your inputs, with a second row beside it whose stream is one higher. The presets are
          Random123's three published vectors.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Phép tính của trình chạy từng bước là phép tính số nguyên 32 bit chính xác, được kiểm tra với BigInt. Biến thể
              PyTorch MPS là bản chép của chúng tôi từ mã nguồn của nó (chín vòng, word cao trước), chưa chạy trên thiết bị.
            </p>
          }
        >
          <p>
            The stepper's arithmetic is exact 32-bit integer arithmetic, checked against BigInt. The PyTorch MPS variant is our
            transcription of its source (nine rounds, high word first), not run on a device.
          </p>
        </Translated>
      </Simplification>
      <PhiloxStepper />
      <Translated
        vi={
          <p>
            Ba điều cần tìm. Bất kỳ (seed, stream, step) nào cũng được tính trực tiếp: gõ một stream 10¹⁸ và không có gì khác
            phải chạy. Một thay đổi một bit trong stream lan tới cả bốn word trong vòng năm vòng. Và một "Philox" chín vòng
            là một stream khác. Hãy nhìn vòng 9 của vector toàn số 0: 7028b1c8 d6594c40 4f2051bb 76d6628e. Đó là những gì
            Philox Metal của PyTorch, theo cách chúng tôi đọc, trả về cho seed 0 (E3): nó dừng sớm một vòng. Bật cách đọc
            PyTorch MPS với seed 42 và so sánh các word của nó với các word mười vòng.
          </p>
        }
      >
        <p>
          Three things to find. Any (seed, stream, step) is computed directly: type a stream of 10¹⁸ and nothing else runs. A
          one-bit change in the stream reaches all four words within five rounds. And a "Philox" with nine rounds is a
          different stream. Look at round 9 of the all-zero vector: 7028b1c8 d6594c40 4f2051bb 76d6628e. That is what our
          reading of PyTorch's Metal Philox returns for seed 0 (E3): it stops one round early. Switch on the PyTorch MPS reading
          with seed 42 and compare its words with the ten-round ones.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Hoạt ảnh bên dưới cho thấy hàng 0 và 1 (stream 0 và 1) ở seed 0 và step 0, với đúng phép tính 32 bit chính xác
              của trình chạy từng bước. Không có thời gian.
            </p>
          }
        >
          <p>
            The animation below shows rows 0 and 1 (streams 0 and 1) at seed 0 and step 0, in the stepper's exact 32-bit
            arithmetic. No timing.
          </p>
        </Translated>
      </Simplification>
      <PhiloxRoundsScene />

      <h2>
        <Translated vi="Cách ánh xạ của M5">M5's mapping</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Mốc dự án M5 đặt counter thành (step thấp, step cao, stream thấp, stream cao) và key thành (seed thấp, seed cao),
            và biến word đầu ra 0 thành u. Seed, stream và step là đầu vào theo từng hàng. Stream là id của request, không
            phải hàng trong batch: dưới continuous batching một request di chuyển giữa các hàng từ bước này sang bước khác, và
            các số của nó không được di chuyển theo.
          </p>
        }
      >
        <p>
          Milestone M5 sets the counter to (step low, step high, stream low, stream high) and the key to (seed low, seed high),
          and turns output word 0 into u. Seed, stream and step are inputs per row. The stream is the request's id, not the
          batch row: under continuous batching a request moves between rows from one step to the next, and its numbers must not
          move with it.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Theo cách chúng tôi đọc header của cuRAND, word đó đúng là <code>curand()</code> đầu tiên sau{' '}
            <code>curand_init(seed, stream, 4 × step, &s)</code>. Header device rocRAND của AMD, theo cách chúng tôi đọc, khởi
            tạo cùng cách, trong khi bảng tương thích của AMD nói Philox của nó không tái tạo chuỗi của cuRAND.
          </p>
        }
      >
        <p>
          By our reading of cuRAND's header, that word is exactly the first <code>curand()</code> after{' '}
          <code>curand_init(seed, stream, 4 × step, &s)</code>. AMD's rocRAND device header initialises the same way by our
          reading, while AMD's compatibility table says its Philox does not reproduce cuRAND's sequence.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A3, B1, E1)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              cuRAND: key từ seed, subsequence vào các word z và w của counter, offset / 4 vào x và y, trạng thái tại offset mod
              4. Bảng của AMD: "Philox 32x4-10 | No" (host API, thứ tự cũ). Chưa được giải quyết và unverified trên mọi thiết
              bị.
            </p>
          }
        >
          <p>
            cuRAND: key from the seed, subsequence into counter words z and w, offset / 4 into x and y, state at offset mod 4.
            AMD's table: "Philox 32x4-10 | No" (host API, legacy ordering). Unresolved and unverified on any device.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSX17['x1/X1.7/q-mapping']} />
      <Ask q={questionsSX17['x1/X1.7/q-slot']} />

      <h2>
        <Translated vi="Mỗi backend cho bạn những gì">What each backend gives you</Translated>
      </h2>
      <Translated
        vi={
          <p>
            CUDA có device API của cuRAND: <code>curandStatePhilox4_32_10_t</code> và{' '}
            <code>curand_init(seed, subsequence, offset, &state)</code>. Các hàm phân phối của nó "may use any number of
            unsigned integer values" từ bộ sinh, nên một lần rút tái lập được phải bắt đầu từ các bit thô. HIP có hipRAND và
            rocRAND với cùng đối số, và HIPIFY ánh xạ các tên, nhưng phép chuyển đổi uniform khác nhau (bên dưới). Metal không
            có gì: ngôn ngữ shading không có hàm random.
          </p>
        }
      >
        <p>
          CUDA has cuRAND's device API: <code>curandStatePhilox4_32_10_t</code> and{' '}
          <code>curand_init(seed, subsequence, offset, &state)</code>. Its distribution functions "may use any number of
          unsigned integer values" from the generator, so a reproducible draw starts from raw bits. HIP has hipRAND and rocRAND
          with the same arguments, and HIPIFY maps the names, but the uniform conversion differs (below). Metal has nothing: the
          shading language has no random function.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A1, B1, B3, C1 to C4)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              MSL 4.1: không có hàm random, noise hay hash. <code>MPSMatrixRandomPhilox</code> là một kernel điền dữ liệu
              được encode từ host; các op Philox của MPSGraph không ghi lại số vòng, cách gieo seed hay phép chuyển đổi. MSL có{' '}
              <code>mulhi</code> và <code>ulong</code>; Random123 đã từng chạy dưới dạng MSL; PyTorch có Philox Metal riêng.
            </p>
          }
        >
          <p>
            MSL 4.1: no random, noise or hash function. <code>MPSMatrixRandomPhilox</code> is a host-encoded fill kernel;
            MPSGraph's Philox ops do not document rounds, seeding or conversion. MSL has <code>mulhi</code> and{' '}
            <code>ulong</code>; Random123 has run as MSL; PyTorch ships its own Metal Philox.
          </p>
        </Translated>
      </Today>
      <Translated
        vi={
          <p>
            Vì vậy M5 tự viết Philox, một lần trong file <code>.cu</code> và một lần trong file <code>.metal</code>. Nó tạo
            tích 64 bit bằng <code>unsigned long long</code> (<code>ulong</code> trên Metal) thay vì gọi{' '}
            <code>__umulhi</code>. Cả hai đều đúng, và trình giả lập (emulator) cùng stub front-end của khóa học chấp nhận cả
            hai; dạng tích không cần intrinsic và đọc giống hệt trong file <code>.metal</code> với <code>ulong</code>.
          </p>
        }
      >
        <p>
          So M5 writes Philox itself, once in the <code>.cu</code> file and once in the <code>.metal</code> file. It forms the
          64-bit product in <code>unsigned long long</code> (<code>ulong</code> on Metal) instead of calling{' '}
          <code>__umulhi</code>. Both are correct, and the course's emulator and front-end stub accept both; the product form
          needs no intrinsic and reads the same in the <code>.metal</code> file with <code>ulong</code>.
        </p>
      </Translated>
      <Ask q={questionsSX17['x1/X1.7/q-metal']} />

      <h2>
        <Translated vi="Đáp án đã biết là phép thử tính di động">Known answers are the portability test</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Random123 công bố các vector kiểm thử: với một counter và key cho trước, bốn word mà một Philox4x32-10 đúng phải tạo
            ra. Ba trong số đó là các preset của trình chạy từng bước. Một Philox in ra những word đó trên một backend là cùng
            một hàm ở đó, đến từng bit, nên lab X1.3 kiểm tra chúng trên mọi backend trước khi tin vào bất kỳ token nào.
          </p>
        }
      >
        <p>
          Random123 publishes test vectors: for a given counter and key, the four words a correct Philox4x32-10 must produce.
          Three of them are the stepper's presets. A Philox that prints those words on a backend is the same function there, bit
          for bit, so lab X1.3 checks them on every backend before it trusts a single token.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A4, E1)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              <code>tests/kat_vectors</code>: counter 0, key 0 cho <code>6627e8d5 e169c58d bc57ac4c 9b00dbd8</code>. Bản chép
              của chúng tôi tái tạo cả ba vector 10 vòng (E1); mô hình của trang này cũng vậy, và các snippet bên dưới cũng vậy.
            </p>
          }
        >
          <p>
            <code>tests/kat_vectors</code>: counter 0, key 0 gives <code>6627e8d5 e169c58d bc57ac4c 9b00dbd8</code>. Our
            transcription reproduces all three 10-round vectors (E1); so does this page's model, and so do the snippets below.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Từ bit sang u">From bits to u</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một word 32 bit ngẫu nhiên vẫn phải trở thành một float. Các nhà cung cấp bất đồng về cách làm.{' '}
            <code>curand_uniform</code> tính x·2⁻³² + 2⁻³³ bằng float, với miền (0, 1]. rocRAND, và do đó{' '}
            <code>hiprand_uniform</code> trên AMD, tính 2⁻³² + x·2⁻³². Cả hai đều bao gồm 1.0. M5 giữ 24 bit cao:{' '}
            <code>(x &gt;&gt; 8) × 2⁻²⁴</code>. Điều đó là chính xác, vì mọi giá trị dưới 2²⁴ đều là một float và hệ số nhân
            là lũy thừa của hai, nên nó giống hệt đến từng bit trên mọi backend dưới bất kỳ chế độ toán nào, với miền [0, 1 −
            2⁻²⁴].
          </p>
        }
      >
        <p>
          A word of 32 random bits still has to become a float. The vendors disagree on how. <code>curand_uniform</code>{' '}
          computes x·2⁻³² + 2⁻³³ in float, with range (0, 1]. rocRAND, and so <code>hiprand_uniform</code> on AMD, computes
          2⁻³² + x·2⁻³². Both include 1.0. M5 keeps the top 24 bits: <code>(x &gt;&gt; 8) × 2⁻²⁴</code>. That is exact,
          because every value below 2²⁴ is a float and the scale is a power of two, so it is bit-identical on every backend
          under any math mode, with range [0, 1 − 2⁻²⁴].
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md A2, B2, E2)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              cuRAND (trích nguyên văn): "1.0 is included and 0.0 is excluded." rocRAND:{' '}
              <code>ROCRAND_2POW32_INV + (v * ROCRAND_2POW32_INV)</code>. Lượt kiểm tra dữ kiện tìm thấy 3,903 khác biệt trong
              một triệu đầu vào.
            </p>
          }
        >
          <p>
            cuRAND: "1.0 is included and 0.0 is excluded." rocRAND: <code>ROCRAND_2POW32_INV + (v * ROCRAND_2POW32_INV)</code>.
            The facts pass found 3,903 differences in a million inputs.
          </p>
        </Translated>
      </Today>
      <BitsExplorer />
      <Ask q={questionsSX17['x1/X1.7/q-one']} label={tr('Work it out', 'Tự tính')} />
      <Ask q={questionsSX17['x1/X1.7/q-differ']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            u của M5 có thể đúng bằng 0, với 1 trên 2²⁴ word. Vì vậy bước rút ở{' '}
            <a href={href('x1', 'X1.8')}>trang tiếp theo</a> dùng "lớn hơn hẳn": một token có trọng số bằng 0 ở đầu hàng
            tuyệt đối không được chọn.
          </p>
        }
      >
        <p>
          M5's u can be exactly 0, for 1 in 2²⁴ words. That is why the draw on the{' '}
          <a href={href('x1', 'X1.8')}>next page</a> uses "strictly greater": a token of zero weight at the start of the row
          must never be picked.
        </p>
      </Translated>
      <Ask q={questionsSX17['x1/X1.7/q-raw-bits']} />

      <h2>
        <Translated vi="Các framework đang làm gì hôm nay">What the frameworks do today</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Không có hai framework nào dùng chung một stream. PyTorch dùng mt19937 trên CPU và Philox của cuRAND cho mỗi luồng
            trên CUDA; trên MPS nó chạy Philox Metal riêng, mà theo cách chúng tôi đọc <code>c10/metal/random.h</code> có 9
            vòng và đặt word cao trước. MLX dùng một Threefry-2x32 có thể tách, với 20 vòng theo cách chúng tôi đếm trong
            kernel Metal của nó. Chính tài liệu của PyTorch nói kết quả có thể khác nhau giữa các nền tảng và giữa CPU với
            GPU, kể cả với cùng seed.
          </p>
        }
      >
        <p>
          No two of them share a stream. PyTorch uses mt19937 on the CPU and cuRAND's Philox per thread on CUDA; on MPS it
          runs its own Metal Philox, which by our reading of <code>c10/metal/random.h</code> has 9 rounds and puts the high
          word first. MLX uses a splittable Threefry-2x32, with 20 rounds by our count of its Metal kernel. PyTorch's own documentation says results may differ
          across platforms and between CPU and GPU, even with identical seeds.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D5, D6, D7, E3)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              PyTorch MPS, bản chép của chúng tôi (chưa chạy): seed 0, chỉ số 0 cho <code>7028b1c8 …</code> trong khi
              Philox4x32-10 cho <code>6627e8d5 …</code>. MLX (trích nguyên văn): "a splittable version of Threefry, which is a
              counter-based PRNG." 20 vòng là số chúng tôi đếm được trong vòng lặp ở <code>random.metal</code> (D7), năm lượt
              mỗi lượt bốn vòng; MLX không ghi con số này trong tài liệu.
            </p>
          }
        >
          <p>
            PyTorch MPS, our transcription (not run): seed 0, index 0 gives <code>7028b1c8 …</code> where Philox4x32-10 gives{' '}
            <code>6627e8d5 …</code>. MLX: "a splittable version of Threefry, which is a counter-based PRNG." The 20 rounds are our count of the
            loop in <code>random.metal</code> (D7), five passes of four rounds; MLX does not document the count.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Mã nguồn">The code</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Vòng, lịch trình key, cách ánh xạ của M5 và phép chuyển đổi 24 bit, kèm thông báo của Random123. File CUDA là phép
            tính số nguyên thuần túy, không có phần nào khác biệt trên HIP (các intrinsic <code>_sync</code> của warp và mặt
            nạ lane của chúng, cooperative groups, header nửa độ chính xác), nên cùng một mã nguồn phục vụ HIP qua{' '}
            <code>pmpp/vendors/hip.h</code> và các trình giả lập; nó unverified trên phần cứng HIP. File Metal là cùng hàm đó
            bằng MSL với <code>mulhi</code> cho nửa cao và tích 32 bit tràn vòng cho nửa thấp. Kernel Metal của lab X1.3 dùng
            dạng còn lại, một tích <code>ulong</code>, nên bạn thấy cả hai; chúng vượt qua cùng các vector đáp án đã biết.
          </p>
        }
      >
        <p>
          The round, the key schedule, M5's mapping and the 24-bit conversion, with Random123's notice. The CUDA file is
          plain integer arithmetic, with none of the pieces that differ on HIP (warp <code>_sync</code> intrinsics and their
          lane masks, cooperative groups, half-precision headers), so the same source serves HIP through{' '}
          <code>pmpp/vendors/hip.h</code> and the emulators; it is unverified on HIP hardware. The Metal file is the same
          function in MSL with <code>mulhi</code> for the high
          half and the wrapping 32-bit product for the low half. Lab X1.3's Metal kernel uses the other form, a{' '}
          <code>ulong</code> product, so you see both; they pass the same known-answer vectors.
        </p>
      </Translated>
      {cu && metal ? (
        <Code
          tabs={[
            {
              label: tr('CUDA and HIP', 'CUDA và HIP'),
              code: cu,
              status: tr(
                'emulator-checked and front-end-checked; unverified on CUDA and HIP hardware',
                'emulator-checked và front-end-checked; unverified trên phần cứng CUDA và HIP',
              ),
            },
            { label: 'Metal', code: metal, status: 'verified on Metal' },
          ]}
          caption={
            <Translated
              vi={
                <>
                  <code>labs/x1/snippets/philox.cu</code> và <code>philox.metal</code>.{' '}
                  <code>app/src/sim/philox.test.ts</code> chạy cả hai, in ra các vector đáp án đã biết và 64 hàng u, rồi so
                  sánh chúng với mô hình của trang này.
                </>
              }
            >
              <>
                <code>labs/x1/snippets/philox.cu</code> and <code>philox.metal</code>. <code>app/src/sim/philox.test.ts</code> runs
                both, prints the known-answer vectors and 64 rows of u, and compares them with this page's model.
              </>
            </Translated>
          }
        />
      ) : (
        <p className="muted">
          <Translated vi={<>Các snippet (<code>labs/x1/snippets/philox.cu</code>, <code>philox.metal</code>) sẽ hiện ở đây khi đã được nạp.</>}>
            The snippets (<code>labs/x1/snippets/philox.cu</code>, <code>philox.metal</code>) appear here once loaded.
          </Translated>
        </p>
      )}
      <Translated
        vi={
          <p>
            Lab X1.3 ở <a href={href('x1', 'labs')}>trang lab</a> dựng phần này vào mốc dự án M5 và kiểm tra các đáp án đã biết
            trên mọi backend trước tiên.
          </p>
        }
      >
        <p>
          Lab X1.3 on the <a href={href('x1', 'labs')}>labs page</a> builds this into milestone M5 and checks the known answers
          on every backend first.
        </p>
      </Translated>
    </>
  );
}
