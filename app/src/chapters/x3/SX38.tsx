import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Bench, Beyond, Simplification, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { Translated, tr } from '../../lib/i18n';
import {
  CHECKPOINTS,
  HEADER_FIELDS,
  LLAMA2C_EPS,
  LLAMA2C_THETA,
  M7_HEAD_DIM,
  SMOL_CONFIG,
  SMOL_HEADER_LEN,
  STORIES_15M,
  archFromHeader,
  bf16ToF32,
  bf16ToF32Bits,
  checkSize,
  checklist,
  encodeHeader,
  fileBytes,
  fitOf,
  floatCount,
  hex,
  hex16,
  hex32,
  layout,
  lengthBytes,
  numel,
  safetensorsBytes,
  shapeOf,
  type Arch,
  type Llama2cHeader,
  type TensorSpec,
} from '../../sim/ckptformat';

// Section X3.8: real checkpoint formats. Every number shown is computed by
// src/sim/ckptformat.ts and pinned in ckptformat.test.ts against notes/today/x3-facts.md
// (D1 to D6, E1, E2; appA A2). The C++ check is labs/x3/snippets/ckpt_check.cpp, which the
// same test compiles and runs on files built from the model.

const MONO: CSSProperties = { fontFamily: 'var(--sans)', fontVariationSettings: "'MONO' 1" }; // i18n-ignore: CSS value
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const n0 = (x: number) => x.toLocaleString('en-US');
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);

// ---------------------------------------------------------------- decode a header (stepper)

function HeaderDecoder() {
  const h = STORIES_15M;
  const bytes = hex(encodeHeader(h));
  const [k, setK] = useState(0); // field being decoded; 7 = all done
  const [guess, setGuess] = useState('');
  const [shown, setShown] = useState<(number | null)[]>(() => HEADER_FIELDS.map(() => null));
  const [verdicts, setVerdicts] = useState<(boolean | null)[]>(() => HEADER_FIELDS.map(() => null));
  const done = k >= HEADER_FIELDS.length;
  const revealedHere = !done && shown[k] != null;

  const reveal = () => {
    const v = h[HEADER_FIELDS[k]];
    const g = guess.trim() === '' ? null : Number(guess);
    setShown((s) => s.map((x, i) => (i === k ? v : x)));
    setVerdicts((s) => s.map((x, i) => (i === k ? (g == null ? null : g === v) : x)));
  };
  const next = () => {
    setK(k + 1);
    setGuess('');
  };
  const restart = () => {
    setK(0);
    setGuess('');
    setShown(HEADER_FIELDS.map(() => null));
    setVerdicts(HEADER_FIELDS.map(() => null));
  };
  const right = verdicts.filter((v) => v === true).length;
  const sh = shapeOf(h);

  return (
    <Bench
      title={tr('Decode a header', 'Giải mã một header')}
      task={<Translated vi="Đọc mỗi int32 từ bốn byte của nó trước khi bạn hiện đáp án.">Read each int32 from its four bytes before you reveal it.</Translated>}
      foot={
        <Translated vi={<>28 byte đầu của stories15M.bin, dựng lại từ header mà lượt kiểm tra dữ kiện đã đọc từ file (x3-facts D2). Mỗi trường là một int32 little-endian: byte đầu tiên là byte thấp nhất.</>}>
          <>The first 28 bytes of stories15M.bin, rebuilt from the header the facts pass read from the file (x3-facts D2). Each field is a little-endian int32: the first byte is the lowest.</>
        </Translated>
      }
    >
      <div style={{ ...MONO, display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.9rem' }} data-part="dump">
        {HEADER_FIELDS.map((f, i) => {
          const cur = i === k;
          return (
            <div key={f} data-field={f} data-shown={shown[i] ?? ''}
              style={{ border: `1px solid ${cur ? 'var(--ink)' : 'var(--rule)'}`, borderRadius: 4, padding: '0.35rem 0.5rem', background: cur ? 'var(--page)' : 'transparent', minWidth: '7.4rem' }}>
              <div style={{ fontSize: 'var(--step--1)', color: 'var(--ink-3)' }}><Translated vi={<>byte {4 * i} đến {4 * i + 3}</>}>bytes {4 * i} to {4 * i + 3}</Translated></div>
              <div style={{ fontSize: '1.02rem', letterSpacing: '0.04em' }}>{bytes.slice(4 * i, 4 * i + 4).join(' ')}</div>
              <div style={{ fontSize: 'var(--step--1)', color: 'var(--ink-2)' }}>
                {f}
                {shown[i] != null ? <> = <b style={{ color: 'var(--ink)' }}>{n0(shown[i]!)}</b></> : null}
                {verdicts[i] === true ? <span className="t-ok"> ✓</span> : verdicts[i] === false ? <span className="t-fail"> ✗</span> : null}
              </div>
            </div>
          );
        })}
      </div>

      {!done ? (
        <>
          <div className="row" style={{ alignItems: 'flex-end', marginBottom: '0.7rem' }}>
            <label style={FIELD}>
              <span>
                <Translated vi={<>Trường {k + 1} trên 7, <code>{HEADER_FIELDS[k]}</code>: các byte <span style={MONO}>{bytes.slice(4 * k, 4 * k + 4).join(' ')}</span></>}>
                  Field {k + 1} of 7, <code>{HEADER_FIELDS[k]}</code>: bytes <span style={MONO}>{bytes.slice(4 * k, 4 * k + 4).join(' ')}</span>
                </Translated>
              </span>
              <input type="number" step="1" value={guess} aria-label={tr('your int32', 'int32 của bạn')} disabled={revealedHere}
                onChange={(e) => setGuess(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && !revealedHere && reveal()} />
            </label>
            {!revealedHere ? (
              <button className="primary" onClick={reveal}><Translated vi="Hiện đáp án">Reveal</Translated></button>
            ) : (
              <button className="primary" onClick={next}>{k + 1 < HEADER_FIELDS.length ? tr('Next field', 'Trường tiếp theo') : tr('Finish', 'Hoàn tất')}</button>
            )}
            {k > 0 || revealedHere ? <button onClick={restart}><Translated vi="Bắt đầu lại">Restart</Translated></button> : null}
          </div>
          <div className="narrate" role="status" data-part="decode-says">
            {revealedHere ? (
              <Translated
                vi={
                  <>
                    Đảo ngược lại, các byte đọc thành 0x{bytes.slice(4 * k, 4 * k + 4).reverse().join('')} = <b>{n0(shown[k]!)}</b>.{' '}
                    {verdicts[k] === true ? 'Bạn đã đúng.' : verdicts[k] === false ? `Bạn đã viết ${guess}.` : ''}
                    {k === 0 && verdicts[k] === false && Number(guess) === 0x20010000 ? ' Đó là cách đọc big-endian.' : ''}
                  </>
                }
              >
                <>
                  Reversed, the bytes read 0x{bytes.slice(4 * k, 4 * k + 4).reverse().join('')} = <b>{n0(shown[k]!)}</b>.{' '}
                  {verdicts[k] === true ? 'You had it.' : verdicts[k] === false ? `You wrote ${guess}.` : ''}
                  {k === 0 && verdicts[k] === false && Number(guess) === 0x20010000 ? ' That is the big-endian reading.' : ''}
                </>
              </Translated>
            ) : k === 0 ? (
              <Translated vi={<>Little-endian nghĩa là các byte được lưu từ byte thấp nhất trước. Đảo ngược chúng và đọc thành một số hex: 20 01 00 00 trở thành 0x00000120.</>}>
                <>Little-endian means the bytes are stored lowest first. Reverse them and read one hex number: 20 01 00 00 becomes 0x00000120.</>
              </Translated>
            ) : (
              <Translated vi={<>Cùng quy tắc: đảo ngược bốn byte, rồi đọc chúng thành một số thập lục phân.</>}>
                <>Same rule: reverse the four bytes, then read them as one hexadecimal number.</>
              </Translated>
            )}
          </div>
        </>
      ) : (
        <div className="narrate" role="status" data-part="decode-done">
          <Translated
            vi={
              <>
                Đã giải mã cả bảy{verdicts.some((v) => v != null) ? `, bạn đúng ${right} trên 7` : ''}. vocab_size dương, nên bộ phân loại chính là
                embedding. Kích thước head {h.dim} / {h.n_heads} = <b>{sh.headSize}</b>; {h.n_kv_heads} head KV, nên mỗi head query có head KV riêng.{' '}
              </>
            }
          >
            <>
              All seven decoded{verdicts.some((v) => v != null) ? `, ${right} of 7 by you` : ''}. vocab_size is positive, so the classifier is the
              embedding. Head size {h.dim} / {h.n_heads} = <b>{sh.headSize}</b>; {h.n_kv_heads} KV heads, so every query head has its own.{' '}
            </>
          </Translated>
          <button onClick={restart} style={{ marginLeft: '0.4rem' }}><Translated vi="Bắt đầu lại">Restart</Translated></button>
        </div>
      )}
    </Bench>
  );
}

function SizeReveal({ predicted }: { predicted: number }) {
  const h = STORIES_15M;
  const exp = fileBytes(h);
  return (
    <>
      <Translated
        vi={
          <p data-reveal={exp}>
            <b>{n0(exp)} byte</b>: {n0(floatCount(h))} float nhân 4, cộng header 28 byte. Hub ghi stories15M.bin là{' '}
            {n0(CHECKPOINTS.find((c) => c.id === 'stories15M')!.hubBytes!)} byte.{' '}
            {predicted === exp
              ? 'Chính xác.'
              : Math.abs(predicted - exp) === 4 * h.seq_len * (h.dim / h.n_heads)
                ? 'Bạn đã bỏ sót hai bảng RoPE: run.c bỏ qua chúng, nhưng chúng có trong file.'
                : Math.abs(predicted - exp) === 28
                  ? 'Lệch đúng bằng header.'
                  : Math.abs(predicted - exp) / exp < 0.02
                    ? 'Gần đúng: hãy so các số hạng của bạn với trình khám phá bên dưới.'
                    : 'Hãy so các số hạng của bạn với bảng khối trong trình khám phá bên dưới.'}
          </p>
        }
      >
        <p data-reveal={exp}>
          <b>{n0(exp)} bytes</b>: {n0(floatCount(h))} floats times 4, plus the 28-byte header. The Hub lists stories15M.bin at{' '}
          {n0(CHECKPOINTS.find((c) => c.id === 'stories15M')!.hubBytes!)} bytes.{' '}
          {predicted === exp
            ? 'Exactly right.'
            : Math.abs(predicted - exp) === 4 * h.seq_len * (h.dim / h.n_heads)
              ? 'You left out the two RoPE tables: run.c skips them, but they are in the file.'
              : Math.abs(predicted - exp) === 28
                ? 'Off by the header.'
                : Math.abs(predicted - exp) / exp < 0.02
                  ? 'Close: compare your terms with the explorer below.'
                  : 'Compare your terms with the block table in the explorer below.'}
        </p>
      </Translated>
      <Translated
        vi={
          <p className="muted" style={{ marginBottom: 0 }}>
            Khối lớn nhất, bỏ xa các khối khác, là embedding, {n0(h.vocab_size)} × {h.dim} = {n0(h.vocab_size * h.dim)} float, khoảng{' '}
            {Math.round((100 * h.vocab_size * h.dim) / floatCount(h))}% file.
          </p>
        }
      >
        <p className="muted" style={{ marginBottom: 0 }}>
          By far the largest block is the embedding, {n0(h.vocab_size)} × {h.dim} = {n0(h.vocab_size * h.dim)} floats, about{' '}
          {Math.round((100 * h.vocab_size * h.dim) / floatCount(h))}% of the file.
        </p>
      </Translated>
    </>
  );
}

// ---------------------------------------------------------------- checkpoint explorer

type Choice = string; // a checkpoint id or 'custom'

const SHAPE_ROWS = (t: TensorSpec[]) => {
  // Group the 272 tensors by kind: embedding, the nine per-layer tensors, final norm.
  const kinds = new Map<string, { kind: string; shape: number[]; count: number; each: number }>();
  for (const x of t) {
    const kind = x.name.replace(/^model\.layers\.\d+\./, 'model.layers.N.');
    const k = kinds.get(kind);
    if (k) k.count += 1;
    else kinds.set(kind, { kind, shape: x.shape, count: 1, each: numel(x.shape) });
  }
  return [...kinds.values()];
};

function HeaderInputs({ h, set }: { h: Record<string, string>; set: (f: string, v: string) => void }) {
  return (
    <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }} data-part="custom">
      {HEADER_FIELDS.map((f) => (
        <label key={f} style={FIELD}>
          {f}
          <input type="number" step="1" value={h[f]} aria-label={f} style={{ width: '6.6rem' }} onChange={(e) => set(f, e.target.value)} />
        </label>
      ))}
    </div>
  );
}

function FitRow({ arch }: { arch: Arch }) {
  const f = fitOf(arch);
  const hs = Number.isInteger(f.headSize) ? String(f.headSize) : f.headSize.toFixed(2);
  return (
    <p style={{ margin: '0.7rem 0' }} data-part="fit" data-head={f.headSize} data-m7={f.m7 ? 'yes' : 'no'} data-lanes={f.perLane ?? ''}>
      <Translated
        vi={
          <>
            Kích thước head <b>{hs}</b> ({arch.dim} / {arch.heads}); <b>{Number.isInteger(f.qPerKv) ? f.qPerKv : f.qPerKv.toFixed(2)}</b> head query
            cho mỗi head KV.{' '}
          </>
        }
      >
        <>
          Head size <b>{hs}</b> ({arch.dim} / {arch.heads}); <b>{Number.isInteger(f.qPerKv) ? f.qPerKv : f.qPerKv.toFixed(2)}</b> query head
          {f.qPerKv === 1 ? '' : 's'} per KV head.{' '}
        </>
      </Translated>
      {f.m7 ? <span className="chip ok"><Translated vi={<>khớp HEAD_DIM {M7_HEAD_DIM} của M7</>}>fits M7's HEAD_DIM {M7_HEAD_DIM}</Translated></span> : <span className="chip fail"><Translated vi={<>không phải HEAD_DIM {M7_HEAD_DIM} của M7</>}>not M7's HEAD_DIM {M7_HEAD_DIM}</Translated></span>}{' '}
      {f.perLane != null ? (
        <span className="chip ok"><Translated vi={<>32 lane của lab 20.4: {f.perLane} mỗi lane</>}>lab 20.4's 32 lanes: {f.perLane} per lane</Translated></span>
      ) : (
        <span className="chip fail"><Translated vi="không phải bội số của 32">not a multiple of 32</Translated></span>
      )}
    </p>
  );
}

function Checklist({ arch }: { arch: Arch }) {
  const items = checklist(arch);
  return (
    <div data-part="checklist" style={{ marginTop: '0.9rem' }}>
      <div className="mx-label" style={{ marginBottom: '0.3rem' }}><Translated vi="Trước khi các kernel của khóa học có thể chạy nó">Before the course's kernels can run it</Translated></div>
      <ul style={{ margin: 0, paddingLeft: '1.1rem' }}>
        {items.map((i) => (
          <li key={i.id} data-item={i.id} data-todo={i.todo ? 'yes' : 'no'} style={{ marginBottom: '0.3rem' }}>
            {i.todo ? <span className="chip fail"><Translated vi="cần làm">to do</Translated></span> : <span className="chip ok"><Translated vi="ổn">fine</Translated></span>} <b>{i.label}.</b>{' '}
            <span className="muted">{i.detail}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Explorer() {
  const [pick, setPick] = useState<Choice>('stories260K');
  const [custom, setCustom] = useState<Record<string, string>>(() =>
    Object.fromEntries(HEADER_FIELDS.map((f) => [f, String(STORIES_15M[f])])),
  );
  const [download, setDownload] = useState('');
  const ck = CHECKPOINTS.find((c) => c.id === pick);
  const customHeader: Llama2cHeader | null = useMemo(() => {
    if (HEADER_FIELDS.some((f) => custom[f].trim() === '' || !Number.isInteger(Number(custom[f])))) return null;
    return Object.fromEntries(HEADER_FIELDS.map((f) => [f, Number(custom[f])])) as unknown as Llama2cHeader;
  }, [custom]);
  const header = ck ? ck.header ?? null : customHeader;
  const problems = header ? shapeOf(header).problems : [tr('every field must be an integer', 'mọi trường phải là số nguyên')];
  const usable = !!header && problems.length === 0 && header.n_layers > 0 && header.hidden_dim > 0 && header.seq_len > 0 && header.vocab_size !== 0;
  const arch: Arch | null = ck ? ck.arch : usable ? archFromHeader(header!) : null;
  const dl = download.trim() === '' ? null : Number(download);

  return (
    <Bench
      title={tr('Checkpoint explorer', 'Trình khám phá checkpoint')}
      task={
        <Translated vi="Chọn một checkpoint, hoặc gõ một header. Checkpoint thật cỡ nhỏ nào mà kernel decode của M7 chạy được đúng như đã viết?">
          Pick a checkpoint, or type a header. Which small real checkpoint can M7's decode kernel run as written?
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Kích thước và hình dạng từ x3-facts D2 và D4; các công thức là của chúng tôi (E1). Hai mô hình M10 là các hình dạng mà lab X3.2 sinh ra
              (một unit test gắn chúng với file CMake của lab).
            </>
          }
        >
          <>
            Sizes and shapes from x3-facts D2 and D4; the formulas are ours (E1). The two M10 models are the shapes lab X3.2 generates
            (a unit test ties them to its CMake file).
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Checkpoint', 'Checkpoint')} style={{ flexWrap: 'wrap' }}>
          {CHECKPOINTS.map((c) => (
            <button key={c.id} aria-pressed={pick === c.id} onClick={() => { setPick(c.id); setDownload(''); }}>{c.label}</button>
          ))}
          <button aria-pressed={pick === 'custom'} onClick={() => { setPick('custom'); setDownload(''); }}><Translated vi="header của riêng bạn">your own header</Translated></button>
        </span>
      </div>
      {pick === 'custom' ? (
        <HeaderInputs h={custom} set={(f, v) => setCustom((c) => ({ ...c, [f]: v }))} />
      ) : null}

      {ck ? (
        <p className="muted" style={{ marginTop: 0 }} data-part="file">
          {ck.file} · {ck.licence} · {ck.format === 'llama2c' ? tr('llama2.c version 0 .bin', '.bin phiên bản 0 của llama2.c') : tr('safetensors plus config.json', 'safetensors kèm config.json')} · <Translated vi="dữ kiện">facts</Translated> {ck.facts}
        </p>
      ) : null}

      {pick === 'custom' && !usable ? (
        <p data-part="problem">
          <span className="chip fail"><Translated vi="không phải header dùng được">not a usable header</Translated></span>{' '}
          {problems.length ? problems.join('; ') : tr('n_layers, hidden_dim and seq_len must be positive and vocab_size non-zero', 'n_layers, hidden_dim và seq_len phải dương và vocab_size khác 0')}.
        </p>
      ) : null}

      {(ck?.format === 'llama2c' || (!ck && usable)) && header && usable ? (
        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)', alignItems: 'start' }}>
          <div style={{ overflowX: 'auto' }}>
            <table className="data" style={{ margin: 0, background: 'var(--page)', width: '100%' }} data-part="blocks">
              <thead>
                <tr><th style={{ paddingLeft: '0.6rem' }}><Translated vi="Khối, theo thứ tự trong file">Block, in file order</Translated></th><th className="num"><Translated vi="Số float">Floats</Translated></th></tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ paddingLeft: '0.6rem' }}><Translated vi="header ">header </Translated><span className="muted" style={{ fontSize: 'var(--step--1)' }}>7 × int32</span></td>{/* i18n-ignore: type notation */}
                  <td className="num muted"><Translated vi="28 byte">28 bytes</Translated></td>
                </tr>
                {layout(header).map((b) => (
                  <tr key={b.name} data-block={b.name}>
                    <td style={{ paddingLeft: '0.6rem' }}>
                      <span style={MONO}>{b.name}</span>{b.skipped ? <span className="muted"><Translated vi=" (bị bỏ qua)"> (skipped)</Translated></span> : null}{' '}
                      <span className="muted" style={{ fontSize: 'var(--step--1)', whiteSpace: 'nowrap' }}>{b.what}</span>
                    </td>
                    <td className="num">{n0(b.floats)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <p style={{ marginTop: 0 }} data-part="size" data-floats={floatCount(header)} data-bytes={fileBytes(header)}>
              <Translated vi={<>{n0(floatCount(header))} float × 4 + 28 = <b>{n0(fileBytes(header))} byte</b>.</>}>
                {n0(floatCount(header))} floats × 4 + 28 = <b>{n0(fileBytes(header))} bytes</b>.
              </Translated>
              {ck && ck.hubBytes != null ? (
                fileBytes(header) === ck.hubBytes ? (
                  <> <span className="chip ok"><Translated vi="đúng kích thước trên Hub, đến từng byte">the Hub's size, to the byte</Translated></span></>
                ) : (
                  <> <span className="chip fail"><Translated vi={<>Hub ghi {n0(ck.hubBytes)}</>}>the Hub says {n0(ck.hubBytes)}</Translated></span></>
                )
              ) : ck ? (
                <> <span className="muted"><Translated vi="make_model.py kiểm tra file nó ghi ra bằng cùng công thức đó.">make_model.py checks the file it writes against the same formula.</Translated></span></>
              ) : null}
            </p>
            <label style={FIELD}>
              <Translated vi="Kích thước file bạn tải về, tính bằng byte">Your download's size in bytes</Translated>
              <input type="number" step="1" value={download} aria-label={tr('download size', 'kích thước file tải về')} onChange={(e) => setDownload(e.target.value)} />
            </label>
            {dl != null && Number.isFinite(dl) ? (
              (() => {
                const c = checkSize(header, dl);
                return (
                  <p data-part="download" data-ok={c.ok ? 'yes' : 'no'} style={{ margin: '0.5rem 0 0' }}>
                    {c.ok ? <span className="chip ok"><Translated vi="khớp với header">matches the header</Translated></span> : <span className="chip fail"><Translated vi="không khớp">does not match</Translated></span>}{' '}
                    {c.hint ?? tr('Safe to load as a version 0 file.', 'An toàn để nạp như một file phiên bản 0.')}
                  </p>
                );
              })()
            ) : null}
            <FitRow arch={arch!} />
          </div>
        </div>
      ) : null}

      {ck?.format === 'safetensors' && ck.st ? (
        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)', alignItems: 'start' }}>
          <div style={{ overflowX: 'auto' }}>
            <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '24rem' }} data-part="tensors">
              <thead>
                <tr><th style={{ paddingLeft: '0.6rem' }}><Translated vi="Tensor (BF16)">Tensor (BF16)</Translated></th><th><Translated vi="Hình dạng">Shape</Translated></th><th className="num"><Translated vi="Số lượng">Count</Translated></th></tr>
              </thead>
              <tbody>
                {SHAPE_ROWS(ck.st.tensors).map((r) => (
                  <tr key={r.kind}>
                    <td style={{ paddingLeft: '0.6rem', ...MONO, fontSize: 'var(--step--1)' }}>{r.kind}</td>
                    <td className="muted" style={{ whiteSpace: 'nowrap' }}>[{r.shape.join(', ')}]</td>
                    <td className="num">{r.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            {(() => {
              const elems = ck.st.tensors.reduce((s, t) => s + numel(t.shape), 0);
              const bytes = safetensorsBytes(ck.st.headerLen, ck.st.tensors);
              return (
                <p style={{ marginTop: 0 }} data-part="size" data-bytes={bytes} data-tensors={ck.st.tensors.length}>
                  <Translated
                    vi={
                      <>
                        {ck.st.tensors.length} tensor, {n0(elems)} giá trị × 2 byte, cộng 8 + N = 8 + {n0(ck.st.headerLen)} byte header:{' '}
                        <b>{n0(bytes)} byte</b>.{' '}
                      </>
                    }
                  >
                    <>
                      {ck.st.tensors.length} tensors, {n0(elems)} values × 2 bytes, plus 8 + N = 8 + {n0(ck.st.headerLen)} header bytes:{' '}
                      <b>{n0(bytes)} bytes</b>.{' '}
                    </>
                  </Translated>
                  {bytes === ck.hubBytes ? <span className="chip ok"><Translated vi="đúng kích thước trên Hub, đến từng byte">the Hub's size, to the byte</Translated></span> : <span className="chip fail"><Translated vi={<>Hub ghi {n0(ck.hubBytes ?? 0)}</>}>the Hub says {n0(ck.hubBytes ?? 0)}</Translated></span>}
                </p>
              );
            })()}
            <p style={{ margin: '0 0 0.4rem' }} data-part="first-bytes">
              <Translated
                vi={
                  <>
                    Các byte đầu: <span style={MONO}>{lengthBytes(SMOL_HEADER_LEN).join(' ')} 7B</span>: N = {n0(SMOL_HEADER_LEN)}, rồi đến{' '}
                    <code>{'{'}</code> của header.
                  </>
                }
              >
                <>
                  First bytes: <span style={MONO}>{lengthBytes(SMOL_HEADER_LEN).join(' ')} 7B</span>: N = {n0(SMOL_HEADER_LEN)}, then the
                  header's <code>{'{'}</code>.
                </>
              </Translated>
            </p>
            <Translated
              vi={
                <p className="muted" style={{ margin: 0 }}>
                  config.json: hidden {SMOL_CONFIG.hidden_size}, intermediate {SMOL_CONFIG.intermediate_size}, {SMOL_CONFIG.num_hidden_layers} lớp,{' '}
                  {SMOL_CONFIG.num_attention_heads} head trên {SMOL_CONFIG.num_key_value_heads} head KV, vocab {n0(SMOL_CONFIG.vocab_size)}, rope_theta{' '}
                  {n0(SMOL_CONFIG.rope_theta)}, eps {SMOL_CONFIG.rms_norm_eps.toExponential(0)}, embedding dùng chung (tied).
                </p>
              }
            >
              <p className="muted" style={{ margin: 0 }}>
                config.json: hidden {SMOL_CONFIG.hidden_size}, intermediate {SMOL_CONFIG.intermediate_size}, {SMOL_CONFIG.num_hidden_layers} layers,{' '}
                {SMOL_CONFIG.num_attention_heads} heads over {SMOL_CONFIG.num_key_value_heads} KV heads, vocab {n0(SMOL_CONFIG.vocab_size)}, rope_theta{' '}
                {n0(SMOL_CONFIG.rope_theta)}, eps {SMOL_CONFIG.rms_norm_eps.toExponential(0)}, tied embeddings.
              </p>
            </Translated>
            <FitRow arch={ck.arch} />
          </div>
        </div>
      ) : null}

      {arch ? <Checklist arch={arch} /> : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

export const questionsSX38: Record<string, Question> = {
  'x3/X3.8/q-size-260k': {
    kind: 'numeric',
    id: 'x3/X3.8/q-size-260k',
    prompt: (
      <Translated
        vi={
          <>
            Header của stories260K là dim 64, hidden_dim 172, 5 lớp, 8 head, 4 head KV, vocab_size 512, seq_len 512, nên bộ phân loại được
            dùng chung. File phải có bao nhiêu byte?
          </>
        }
      >
        <>
          stories260K's header is dim 64, hidden_dim 172, 5 layers, 8 heads, 4 KV heads, vocab_size 512, seq_len 512, so the classifier is
          shared. How many bytes should the file have?
        </>
      </Translated>
    ),
    answer: 1056540,
    unit: 'bytes',
    explain: (
      <Translated
        vi={
          <p>
            Kích thước head 64 / 8 = 8 và kv_dim = 64 × 4 / 8 = 32. Số float: embedding 512 × 64 = 32,768; hai norm mỗi lớp, 2 × 5 × 64 = 640; wq
            và wo 2 × 5 × 64 × 64 = 40,960; wk và wv 2 × 5 × 32 × 64 = 20,480; w1, w2, w3 3 × 5 × 64 × 172 = 165,120; norm cuối 64; hai bảng
            RoPE 512 × 8 = 4,096. Tổng là 264,128 float, × 4 + 28 = 1,056,540 byte, đúng kích thước Hub ghi.
          </p>
        }
      >
        <p>
          Head size 64 / 8 = 8 and kv_dim = 64 × 4 / 8 = 32. Floats: embedding 512 × 64 = 32,768; two norms per layer, 2 × 5 × 64 = 640; wq
          and wo 2 × 5 × 64 × 64 = 40,960; wk and wv 2 × 5 × 32 × 64 = 20,480; w1, w2, w3 3 × 5 × 64 × 172 = 165,120; final norm 64; the two
          RoPE tables 512 × 8 = 4,096. That is 264,128 floats, × 4 + 28 = 1,056,540 bytes, the size the Hub lists.
        </p>
      </Translated>
    ),
    source: 'ours (x3-facts E1), from the header in D2',
    vi: { unit: 'byte', source: 'của chúng tôi (x3-facts E1), từ header trong D2' },
  },
  'x3/X3.8/q-negative-vocab': {
    kind: 'choice',
    id: 'x3/X3.8/q-negative-vocab',
    prompt: (
      <Translated vi={<>vocab_size trong một header llama2.c là số âm. Bộ nạp làm gì khác đi?</>}>
        <>A llama2.c header's vocab_size is negative. What does the loader do differently?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="LM head không phải là embedding: khối vocab × dim riêng của nó nằm sau các bảng RoPE bị bỏ qua.">The LM head is not the embedding: its own vocab × dim block follows the skipped RoPE tables.</Translated>, correct: true },
      { label: <Translated vi="File là phiên bản 1 trở lên, với một header khác, và run.c từ chối nạp nó.">The file is version 1 or later, with a different header, and run.c refuses to load it.</Translated>, why: <Translated vi="Các phiên bản sau là một header khác. Dấu là cờ riêng của run.c bên trong phiên bản 0.">Later versions are a different header. The sign is run.c’s own flag inside version 0.</Translated> },
      { label: <Translated vi="Các chuỗi token được lưu theo thứ tự ngược, nên bộ nạp phải đọc chúng từ cuối lên đầu.">The token strings are stored in reverse order, so the loader must read them back to front.</Translated>, why: <Translated vi="Không có gì trong run.c đảo ngược thứ gì cả. Dấu mang một bit thông tin.">Nothing in run.c reverses anything. The sign carries one bit of information.</Translated> },
      { label: <Translated vi="Các trọng số được lưu dạng float16 thay vì float32, nên bộ nạp chuyển đổi từng giá trị.">The weights are stored as float16 instead of float32, so the loader converts each one.</Translated>, why: <Translated vi="Phiên bản 0 là float32 từ đầu đến cuối; run.c ánh xạ trực tiếp các float.">Version 0 is float32 throughout; run.c maps the floats directly.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            run.c đọc <code>shared_weights = vocab_size &gt; 0</code> rồi dùng |vocab_size|. Khi trọng số dùng chung, LM head dùng lại
            token_embedding_table; khi không, <code>wcls</code> là vocab × dim float tiếp theo sau freq_cis_imag. Một bộ nạp quên dấu sẽ
            đọc ra một kích thước âm và thất bại; một bộ nạp lấy giá trị tuyệt đối nhưng bỏ qua cờ sẽ chạy sai bộ phân loại. Phép kiểm
            tra kích thước bắt được trường hợp thứ hai: file dài hơn vocab × dim × 4 byte so với dự đoán từ một header dùng chung.
          </p>
        }
      >
        <p>
          run.c reads <code>shared_weights = vocab_size &gt; 0</code> and then uses |vocab_size|. With shared weights the LM head reuses
          token_embedding_table; without them, <code>wcls</code> is the next vocab × dim floats after freq_cis_imag. A loader that forgets
          the sign reads a negative size and fails; one that takes the absolute value but ignores the flag runs the wrong classifier. The
          size check catches the second: the file is vocab × dim × 4 bytes longer than a shared header predicts.
        </p>
      </Translated>
    ),
    source: 'x3-facts D1 (run.c lines 142 to 150, memory_map_weights)',
    vi: { source: 'x3-facts D1 (run.c dòng 142 đến 150, memory_map_weights)' },
  },
  'x3/X3.8/q-st-n': {
    kind: 'numeric',
    id: 'x3/X3.8/q-st-n',
    prompt: (
      <Translated
        vi={
          <>
            Một file safetensors bắt đầu bằng <code>40 77 00 00 00 00 00 00 7B</code>. Mục header của một tensor ghi{' '}
            <code>"data_offsets": [0, 56623104]</code>. Dữ liệu của tensor đó bắt đầu ở offset byte nào trong file?
          </>
        }
      >
        <>
          A safetensors file starts <code>40 77 00 00 00 00 00 00 7B</code>. A tensor's header entry says{' '}
          <code>"data_offsets": [0, 56623104]</code>. At what byte offset in the file does that tensor's data start?
        </>
      </Translated>
    ),
    answer: 30536,
    explain: (
      <Translated
        vi={
          <p>
            8 byte đầu là N dưới dạng số nguyên 64 bit little-endian: 0x7740 = 30,528. Header chiếm các byte 8 đến 30,535, và buffer byte
            bắt đầu ngay sau đó. data_offsets tính tương đối so với buffer đó, không phải so với file, nên tensor bắt đầu ở 8 + 30,528 + 0 = 30,536.
            (Đây là các byte đầu của SmolLM2-135M; 56,623,104 byte là embedding của nó, 49,152 × 576 giá trị × 2.)
          </p>
        }
      >
        <p>
          The first 8 bytes are N as a little-endian 64-bit integer: 0x7740 = 30,528. The header occupies bytes 8 to 30,535, and the byte
          buffer starts after it. data_offsets are relative to that buffer, not to the file, so the tensor starts at 8 + 30,528 + 0 = 30,536.
          (These are SmolLM2-135M's first bytes; 56,623,104 bytes is its embedding, 49,152 × 576 values × 2.)
        </p>
      </Translated>
    ),
    source: 'x3-facts D3 (format), D4 (N and the embedding shape)',
    vi: { source: 'x3-facts D3 (định dạng), D4 (N và hình dạng embedding)' },
  },
  'x3/X3.8/q-bf16': {
    kind: 'choice',
    id: 'x3/X3.8/q-bf16',
    prompt: (
      <Translated vi={<>Một tensor BF16 lưu 16 bit 0x3F80. Bạn ghi ra mẫu bit float32 nào, và phép chuyển đổi có chính xác không?</>}>
        <>A BF16 tensor stores the 16 bits 0x3F80. What float32 bit pattern do you write, and is the conversion exact?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi={`${hex32(bf16ToF32Bits(0x3f80))}, tức là ${bf16ToF32(0x3f80)}.0; chính xác, vì bf16 là nửa trên của một float32.`}>{`${hex32(bf16ToF32Bits(0x3f80))}, which is ${bf16ToF32(0x3f80)}.0; exact, because bf16 is the top half of a float32.`}</Translated>, correct: true },
      { label: <Translated vi="0x00003F80: giữ nguyên 16 bit; chính xác.">0x00003F80: the 16 bits as they are; exact.</Translated>, why: <Translated vi="Như thế là đặt các bit vào nửa dưới: kết quả là một số subnormal rất nhỏ, không phải 1.0.">That puts the bits in the bottom half: the result is a tiny subnormal, not 1.0.</Translated> },
      { label: <Translated vi="0x3F800000, nhưng phải làm tròn, nên nó chỉ xấp xỉ 1.0.">0x3F800000, but it must be rounded, so it is only approximately 1.0.</Translated>, why: <Translated vi="Mở rộng chỉ thêm mười sáu bit mantissa bằng 0. Không mất gì, nên không có gì phải làm tròn.">Widening adds sixteen zero mantissa bits. Nothing is lost, so nothing is rounded.</Translated> },
      { label: <Translated vi="Nó phụ thuộc vào độ lệch số mũ, mà bf16 và float32 không dùng chung.">It depends on the exponent bias, which bf16 and float32 do not share.</Translated>, why: <Translated vi="Chúng dùng chung: bf16 có số mũ 8 bit của float32.">They share it: bf16 has float32’s 8-bit exponent.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            bfloat16 giữ bit dấu của float32, 8 bit số mũ và 7 bit mantissa cao nhất. Dịch 16 bit đã lưu sang trái 16 vị trí cho ra
            float32 có đúng các trường đó và phần đuôi bằng 0, đúng cùng một giá trị. Chiều ngược lại, float32 sang bf16, mới là chiều phải làm tròn.
          </p>
        }
      >
        <p>
          bfloat16 keeps float32's sign bit, its 8 exponent bits and the top 7 mantissa bits. Shifting the 16 stored bits left by 16 gives the
          float32 with those fields and a zero tail, the exact same value. The other direction, float32 to bf16, is the one that rounds.
        </p>
      </Translated>
    ),
    source: 'appA-facts A2 (layout); x3-facts D4 (every SmolLM2 tensor is BF16)',
    vi: { source: 'appA-facts A2 (bố cục); x3-facts D4 (mọi tensor của SmolLM2 đều là BF16)' },
  },
  'x3/X3.8/q-theta': {
    kind: 'choice',
    id: 'x3/X3.8/q-theta',
    prompt: (
      <Translated
        vi={
          <>
            Bạn chạy SmolLM2-135M từ BOS với cơ số RoPE 10,000 viết cứng của llama2.c thay vì 100,000 trong config. Token sinh ra đầu tiên
            khớp với tham chiếu. Cơ số sai có vô hại không?
          </>
        }
      >
        <>
          You run SmolLM2-135M from BOS with llama2.c's hard-coded RoPE base 10,000 instead of the config's 100,000. The first generated token
          matches the reference. Is the wrong base harmless?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Không. Ở vị trí 0 mọi góc đều bằng 0 bất kể cơ số, nên token đầu tiên không thể cho thấy lỗi; các token sau trôi dần.">
            No. At position 0 every angle is 0 whatever the base, so the first token cannot show it; later tokens drift.
          </Translated>
        ),
        correct: true,
      },
      { label: <Translated vi="Có: góc RoPE nhỏ với hầu hết các cặp, nên hệ số 10 ở cơ số hầu như không đổi điểm attention nào.">Yes: RoPE angles are small for most pairs, so a factor of 10 in the base barely changes any attention score.</Translated>, why: <Translated vi="Các cặp tần số thấp thay đổi rất nhiều: cơ số quyết định mọi cặp sau cặp đầu tiên quay nhanh đến đâu.">Low-frequency pairs change by a lot: the base sets how fast every pair after the first rotates.</Translated> },
      { label: <Translated vi="Có, miễn là epsilon đúng: RMSNorm hấp thụ phần chênh lệch.">Yes, as long as epsilon is right: RMSNorm absorbs the difference.</Translated>, why: <Translated vi="RMSNorm co giãn một vector; nó không thể đảo ngược một phép quay của Q so với K.">RMSNorm rescales a vector; it cannot undo a rotation of Q against K.</Translated> },
      { label: <Translated vi="Không, và token đầu tiên lẽ ra đã phải khác; khớp nghĩa là kernel không bao giờ đọc cơ số RoPE.">No, and the first token should already differ; a match means the kernel never reads the RoPE base.</Translated>, why: <Translated vi="Ở vị trí 0, pos × tần số bằng 0 với mọi cặp và mọi cơ số.">At position 0, pos × frequency is 0 for every pair and every base.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Góc của một cặp là vị trí × tần số, và tần số là base<sup>−2i/d</sup>. Ở vị trí 0 tích bằng 0 với mọi cơ số, nên Q và K hoàn
            toàn không bị quay và cơ số sai là vô hình. Từ vị trí 1 trở đi, mọi cặp trừ cặp đầu quay một góc khác, điểm attention thay
            đổi, và một lần chạy greedy trôi dần. Một bài kiểm thử chỉ xét token đầu tiên, hoặc chỉ prompt ngắn, sẽ bỏ lỡ điều này; logits
            teacher-forced trên một chuỗi dài thì không (phần <a href={href('x3', 'X3.9')}>X3.9</a>).
          </p>
        }
      >
        <p>
          The angle for a pair is position × frequency, and the frequency is base<sup>−2i/d</sup>. At position 0 the product is 0 for any
          base, so Q and K are not rotated at all and a wrong base is invisible. From position 1 on, every pair except the first rotates by
          a different angle, attention scores change, and a greedy run drifts. A test that checks only the first token, or only short
          prompts, misses this; teacher-forced logits over a long sequence do not (section <a href={href('x3', 'X3.9')}>X3.9</a>).
        </p>
      </Translated>
    ),
    source: 'ours; the constants from x3-facts D1 (run.c) and D4 (config.json)',
    vi: { source: 'của chúng tôi; các hằng số từ x3-facts D1 (run.c) và D4 (config.json)' },
  },
  'x3/X3.8/q-head64': {
    kind: 'choice',
    id: 'x3/X3.8/q-head64',
    prompt: (
      <Translated vi={<>Trong ba checkpoint nhỏ, checkpoint nào mà kernel decode của lab 20.4 có thể phục vụ chỉ bằng cách đổi HEAD_DIM?</>}>
        <>Which of the three small checkpoints could lab 20.4's decode kernel serve by changing only HEAD_DIM?</>
      </Translated>
    ),
    options: [
      { label: <Translated vi="SmolLM2-135M: kích thước head 64 = 2 × 32, nên 2 giá trị mỗi lane.">SmolLM2-135M: head size 64 = 2 × 32, so 2 values per lane.</Translated>, correct: true },
      { label: <Translated vi="stories15M: kích thước head 48, một bội số của 16.">stories15M: head size 48, a multiple of 16.</Translated>, why: <Translated vi="48 không phải bội số của 32. HEAD_DIM / 32 làm tròn xuống thành 1, nên mỗi lane nhận một chiều và 16 trong 48 chiều không bao giờ được đọc.">48 is not a multiple of 32. HEAD_DIM / 32 rounds down to 1, so each lane takes one dimension and 16 of the 48 are never read.</Translated> },
      { label: <Translated vi="stories260K: kích thước head 8, mỗi lane trong 8 lane một giá trị.">stories260K: head size 8, one value for each of 8 lanes.</Translated>, why: <Translated vi="Ít chiều hơn số lane: HEAD_DIM / 32 bằng 0.">Fewer dimensions than lanes: HEAD_DIM / 32 is 0.</Translated> },
      { label: <Translated vi="Cả ba: HEAD_DIM là hằng số lúc biên dịch, và giá trị nào cũng build được.">All three: HEAD_DIM is a compile-time constant, and any value builds.</Translated>, why: <Translated vi="Kernel còn chia head cho 32 lane, điều đó ràng buộc hằng số này.">The kernel also divides the head over 32 lanes, which constrains the constant.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Lab 20.4 giao cho mỗi lane HEAD_DIM / 32 chiều, xen kẽ (lane L giữ L, L + 32, ...). Điều đó đòi một bội số của 32. Chỉ 64 của
            SmolLM2 đạt, và 9 head query trên 3 head KV của nó khớp quy tắc GQA của kernel (h_q là bội số của h_kv). Đây là phép tính của
            chúng tôi từ mã nguồn của lab; chưa chạy trên thiết bị với checkpoint này.
          </p>
        }
      >
        <p>
          Lab 20.4 gives each lane HEAD_DIM / 32 dimensions, interleaved (lane L holds L, L + 32, ...). That needs a multiple of 32. Only
          SmolLM2's 64 qualifies, and its 9 query heads over 3 KV heads fit the kernel's GQA rule (h_q a multiple of h_kv). Our arithmetic from
          the lab's source; not run on a device with this checkpoint.
        </p>
      </Translated>
    ),
    source: 'x3-facts E2; labs/ch20/lab4_decode/solution/kernel.cu (D_PER_LANE, contract); unverified on a device',
    vi: { source: 'x3-facts E2; labs/ch20/lab4_decode/solution/kernel.cu (D_PER_LANE, hợp đồng); unverified trên thiết bị' },
  },
  'x3/X3.8/q-gguf': {
    kind: 'choice',
    id: 'x3/X3.8/q-gguf',
    prompt: (
      <Translated vi={<>llama.cpp nạp GGUF. Vì sao đường checkpoint thật của M10 không bắt đầu từ đó?</>}>
        <>llama.cpp loads GGUF. Why does M10's real-checkpoint path not start there?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="GGUF thêm metadata có kiểu, dữ liệu được căn lề và thường là các kiểu đã lượng tử hóa (chủ đề của M9); .bin và safetensors cần ít hơn nhiều.">
            GGUF adds typed metadata, aligned data and usually quantised types (M9’s subject); .bin and safetensors need far less.
          </Translated>
        ),
        correct: true,
      },
      { label: <Translated vi="GGUF không có tài liệu: mã nguồn bộ nạp của llama.cpp là mô tả duy nhất của bố cục.">GGUF is undocumented: llama.cpp’s loader source is the only description of the layout.</Translated>, why: <Translated vi="Nó có một đặc tả viết thành văn bản, ggml/docs/gguf.md, phiên bản 3.">It has a written spec, ggml/docs/gguf.md, version 3.</Translated> },
      { label: <Translated vi="File GGUF là big-endian, nên mọi giá trị đều cần đảo byte trên các máy host của khóa học.">GGUF files are big-endian, so every value would need a byte swap on the course’s hosts.</Translated>, why: <Translated vi="Đặc tả nói mô hình mặc định là little-endian.">The spec says models are little-endian by default.</Translated> },
      { label: <Translated vi="GGUF không thể chứa một mô hình kiến trúc Llama.">GGUF cannot hold a Llama-architecture model.</Translated>, why: <Translated vi="Đó chính là định dạng llama.cpp dùng để chạy các mô hình Llama.">It is the format llama.cpp runs Llama models from.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            GGUF không có gì sai; nó làm nhiều hơn những gì M10 cần. Một .bin phiên bản 0 là bảy số nguyên và float32. safetensors là một độ
            dài, JSON và các byte thứ tự hàng. GGUF thêm một phần key-value có kiểu, offset tensor được đệm theo một mức căn lề (32 trừ khi
            file ghi khác), và, trong phần lớn file được phát hành, các kiểu lượng tử hóa theo khối mà các kernel FP32 của M10 không đọc được
            nếu không có bước giải lượng tử hóa của M9.
          </p>
        }
      >
        <p>
          Nothing is wrong with GGUF; it does more than M10 needs. A version 0 .bin is seven integers and float32. safetensors is a length,
          JSON and row-major bytes. GGUF adds a typed key-value section, tensor offsets padded to an alignment (32 unless the file says
          otherwise), and, in most published files, block-quantised types that M10's FP32 kernels cannot read without M9's dequantisation.
        </p>
      </Translated>
    ),
    source: 'x3-facts D5 (ggml/docs/gguf.md)', // i18n-ignore: citation
  },
  'x3/X3.8/q-export-gqa': {
    kind: 'choice',
    id: 'x3/X3.8/q-export-gqa',
    prompt: (
      <Translated vi={<>Bạn chuyển SmolLM2-135M sang .bin bằng export.py của llama2.c. Điều gì hỏng?</>}>
        <>You convert SmolLM2-135M to a .bin with llama2.c's export.py. What goes wrong?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="permute_reverse reshape k_proj (192 × 576) như thể nó có 576 hàng, nên nó thất bại hoặc hoán vị sai; K cần số head KV.">
            permute_reverse reshapes k_proj (192 × 576) as if it had 576 rows, so it fails or mis-permutes; K needs the KV head count.
          </Translated>
        ),
        correct: true,
      },
      { label: <Translated vi="Không có gì: export.py đọc config.json nên xử lý được mọi checkpoint Llama, grouped-query hay không.">Nothing: export.py reads config.json and so handles any Llama checkpoint, grouped-query or not.</Translated>, why: <Translated vi="permute của nó mặc định dùng n_heads và dim, cả cho K.">Its permute uses n_heads and dim by default, also for K.</Translated> },
      { label: <Translated vi="Nó ghi nguyên các giá trị bf16 vào một file mà run.c đọc như float32.">It writes the bf16 values unchanged into a file that run.c reads as float32.</Translated>, why: <Translated vi="Đó không phải lỗi mà mã nguồn cho thấy; hình dạng của K mới là lỗi.">That is not the failure the source shows; the shape of K is.</Translated> },
      { label: <Translated vi="Nó bỏ mất embedding dùng chung, nên run.c không tìm thấy bộ phân loại nào trong file.">It drops the tied embedding, so run.c finds no classifier in the file.</Translated>, why: <Translated vi="Thiếu lm_head chính là trường hợp bộ phân loại dùng chung mà định dạng đã có sẵn.">A missing lm_head is the shared-classifier case the format already has.</Translated> },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>permute_reverse(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim)</code> nhìn w dưới dạng (n_heads, 2, dim / n_heads /
            2, dim). Với 3 head KV, k_proj có 3 × 64 = 192 hàng, không phải 576, nên view không khớp. Bộ chuyển đổi của llama.cpp chuyển sang
            n_head_kv cho K. Đây là cách chúng tôi đọc mã nguồn; lượt kiểm tra dữ kiện không chạy nó.
          </p>
        }
      >
        <p>
          <code>permute_reverse(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim)</code> views w as (n_heads, 2, dim / n_heads /
          2, dim). With 3 KV heads, k_proj has 3 × 64 = 192 rows, not 576, so the view does not match. llama.cpp's converter switches to
          n_head_kv for K. This is our reading of the source; the facts pass did not run it.
        </p>
      </Translated>
    ),
    source: 'x3-facts D6 (export.py lines 468 to 475), B6 (llama.cpp permute); by reading, not run',
    vi: { source: 'x3-facts D6 (export.py dòng 468 đến 475), B6 (permute của llama.cpp); bằng cách đọc, không chạy' },
  },
};

// ---------------------------------------------------------------- the section

export function SX38() {
  const cpp = loaded(useLabFile('x3/snippets/ckpt_check.cpp'));
  return (
    <>
      <Translated
        vi={
          <p>
            Mọi mô hình trong khóa học tới giờ đều là trọng số do bạn sinh ra. Một checkpoint thật là một file người khác viết: một định dạng
            bạn phải phân tích chính xác, và những quy ước hoàn toàn không có trong file. Trang này đọc tay hai định dạng, kiểm tra một file
            tải về trước khi tin nó, và liệt kê những gì đứng giữa một Llama của Hugging Face và các kernel của khóa học.
          </p>
        }
      >
        <p>
          Every model in this course so far has been weights you generated. A real checkpoint is a file someone else wrote: a format you
          must parse exactly, and conventions that are not in the file at all. This page reads two formats by hand, checks a download before
          trusting it, and lists what stands between a Hugging Face Llama and the course's kernels.
        </p>
      </Translated>
      <Beyond>
        <Translated
          vi={
            <p>
              Module X3 là của chúng tôi. Định dạng, file và kích thước lấy từ notes/today/x3-facts.md (D1 đến D6, E1, E2), đọc ở các commit
              được ghim ngày 2026-11-20.
            </p>
          }
        >
          <p>
            Module X3 is ours. Formats, files and sizes are from notes/today/x3-facts.md (D1 to D6, E1, E2), read at pinned commits on
            2026-11-20.
          </p>
        </Translated>
      </Beyond>
      <Translated
        vi={
          <p>
            Bạn vốn đã dùng bước này mỗi ngày. Một checkpoint Hugging Face là một <code>config.json</code> bên cạnh một{' '}
            <code>model.safetensors</code>; <code>from_pretrained</code> biến hai thứ đó thành tensor trước khi bất kỳ kernel nào chạy. Ở đây
            bạn tự làm bước đó, trước tiên với định dạng thật đơn giản nhất.
          </p>
        }
      >
        <p>
          You already use this step every day. A Hugging Face checkpoint is a <code>config.json</code> beside a{' '}
          <code>model.safetensors</code>; <code>from_pretrained</code> turns the two into tensors before any kernel runs. Here you do that
          step yourself, for the simplest real format first.
        </p>
      </Translated>

      <h2><Translated vi=".bin của llama2.c: bảy số nguyên, rồi các float">llama2.c's .bin: seven integers, then floats</Translated></h2>
      <Translated
        vi={
          <p>
            <code>run.c</code> của llama2.c là trọn một bước decoder Llama bằng C thuần, và định dạng file của nó nhỏ hết mức một định dạng có
            thể. 28 byte đầu là bảy int32: <code>dim</code>, <code>hidden_dim</code>, <code>n_layers</code>, <code>n_heads</code>,{' '}
            <code>n_kv_heads</code>, <code>vocab_size</code>, <code>seq_len</code>. Sau đó là các trọng số float32 theo thứ tự cố định: embedding
            token, các trọng số RMSNorm của attention, <code>wq</code>, <code>wk</code>, <code>wv</code>, <code>wo</code>, các norm của
            feed-forward, <code>w1</code>, <code>w2</code>, <code>w3</code>, norm cuối, hai bảng RoPE cũ mà run.c bỏ qua, và một bộ phân loại
            chỉ khi header nói vậy. Dấu của <code>vocab_size</code> nói điều đó: âm nghĩa là bộ phân loại là một ma trận riêng.
          </p>
        }
      >
        <p>
          llama2.c's <code>run.c</code> is a whole Llama decoder step in plain C, and its file format is as small as a format can be. The
          first 28 bytes are seven int32: <code>dim</code>, <code>hidden_dim</code>, <code>n_layers</code>, <code>n_heads</code>,{' '}
          <code>n_kv_heads</code>, <code>vocab_size</code>, <code>seq_len</code>. Then come float32 weights in a fixed order: the token
          embedding, the attention RMSNorm weights, <code>wq</code>, <code>wk</code>, <code>wv</code>, <code>wo</code>, the feed-forward
          norms, <code>w1</code>, <code>w2</code>, <code>w3</code>, the final norm, two old RoPE tables that run.c skips, and a classifier
          only if the header says so. The sign of <code>vocab_size</code> says so: negative means the classifier is its own matrix.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Phép nhân ma trận của nó là <code>W (d, n) @ x</code> với <code>w[i * n + j]</code>: thứ tự hàng (row-major), (out, in), bố cục của{' '}
            <code>nn.Linear</code> và cũng là quy ước của dự án này. GEMV của bạn đọc các trọng số này mà không cần chuyển vị.
          </p>
        }
      >
        <p>
          Its matmul is <code>W (d, n) @ x</code> with <code>w[i * n + j]</code>: row-major, (out, in), the <code>nn.Linear</code> layout and
          this project's convention. Your GEMV reads these weights without a transpose.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x3-facts.md D1)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              Không có trong header, được viết cứng trong run.c: epsilon của RMSNorm {LLAMA2C_EPS.toExponential(0)} và cơ số RoPE {n0(LLAMA2C_THETA)}. Greedy giữ cực đại
              đầu tiên; quá trình sinh bắt đầu từ BOS = 1. run.c chỉ đọc bố cục "version 0" này.
            </p>
          }
        >
          <p>
            Not in the header, hard-coded in run.c: RMSNorm epsilon {LLAMA2C_EPS.toExponential(0)} and RoPE base {n0(LLAMA2C_THETA)}. Greedy keeps the first
            maximum; generation starts from BOS = 1. run.c reads only this "version 0" layout.
          </p>
        </Translated>
      </Today>
      <HeaderDecoder />
      <Predict
        id="x3/X3.8/p-size-15m"
        prompt={
          <Translated vi={<>Từ bảy giá trị bạn vừa giải mã và thứ tự các khối ở trên, stories15M.bin dài bao nhiêu byte?</>}>
            <>From the seven values you just decoded and the block order above, how many bytes long is stories15M.bin?</>
          </Translated>
        }
        unit={tr('bytes', 'byte')}
      >
        {(v) => <SizeReveal predicted={v} />}
      </Predict>
      <Ask q={questionsSX38['x3/X3.8/q-negative-vocab']} />

      <h2><Translated vi="Kiểm tra file tải về bằng kích thước">Check a download by its size</Translated></h2>
      <Translated
        vi={
          <p>
            Có hai mô hình nhỏ ở định dạng này trên Hugging Face Hub theo giấy phép MIT, trong <code>karpathy/tinyllamas</code>:
            stories260K (1,056,540 byte: dim 64, 5 lớp, 8 head trên 4 head KV, bộ từ vựng 512 token) và stories15M (60,816,028 byte).
            Cả hai kích thước đều suy ra chính xác từ header. Đếm số float từng khối, nhân 4, cộng 28. Các phép chiếu K và V cao{' '}
            <code>kv_dim = dim × n_kv_heads / n_heads</code> hàng, và hai bảng RoPE bị bỏ qua mỗi bảng có{' '}
            <code>seq_len × head_size / 2</code> float.
          </p>
        }
      >
        <p>
          Two small models in this format are on the Hugging Face Hub under the MIT licence, in <code>karpathy/tinyllamas</code>:
          stories260K (1,056,540 bytes: dim 64, 5 layers, 8 heads over 4 KV heads, a 512-token vocabulary) and stories15M (60,816,028 bytes).
          Both sizes follow exactly from their headers. Count the floats block by block, multiply by 4, add 28. The K and V projections are{' '}
          <code>kv_dim = dim × n_kv_heads / n_heads</code> rows tall, and the two skipped RoPE tables are{' '}
          <code>seq_len × head_size / 2</code> floats each.
        </p>
      </Translated>
      <Ask q={questionsSX38['x3/X3.8/q-size-260k']} label={tr('Work it out', 'Tự tính')} />
      <Translated
        vi={
          <p>
            Vậy một bộ nạp có thể kiểm tra file trước khi ánh xạ dù chỉ một trọng số: một lần tải bị gián đoạn, một header đọc sai thứ tự
            byte, một cờ bộ phân loại bị lật hay một file theo bố cục khác đều thất bại ở một phép so sánh thay vì thành logits vô nghĩa. Đây
            là phép kiểm tra đó viết bằng C++.
          </p>
        }
      >
        <p>
          So a loader can check a file before it maps a single weight: an interrupted download, a header read with the wrong byte order, a
          flipped classifier flag or a file in another layout all fail in one comparison instead of as nonsense logits. Here is that check in
          C++.
        </p>
      </Translated>
      {cpp ? (
        <Code
          tabs={[{ label: tr('C++ (host)', 'C++ (host)'), code: cpp, status: tr('host code, compiled and run by app/src/sim/ckptformat.test.ts on files built from the model', 'mã host, được biên dịch và chạy bởi app/src/sim/ckptformat.test.ts trên các file dựng từ mô hình') }]}
          caption={
            <Translated
              vi={
                <>
                  labs/x3/snippets/ckpt_check.cpp. Bài kiểm thử ghi các file với header đã công bố ở đúng độ dài dự đoán (và một file thiếu 4,096
                  byte), chạy chương trình này trên từng file và so đầu ra của nó với mô hình của trình khám phá.
                </>
              }
            >
              <>
                labs/x3/snippets/ckpt_check.cpp. The test writes files with the published headers at their predicted length (and one 4,096
                bytes short), runs this program on each and compares its output with the explorer's model.
              </>
            </Translated>
          }
        />
      ) : null}

      <h2><Translated vi="safetensors: một độ dài, JSON, rồi các byte">safetensors: a length, JSON, then bytes</Translated></h2>
      <Translated
        vi={
          <p>
            Các checkpoint Hugging Face dùng safetensors. 8 byte đầu là N, một số nguyên 64 bit không dấu little-endian. N byte tiếp theo là
            một header JSON UTF-8 phải bắt đầu bằng <code>{'{'}</code>, ánh xạ tên mỗi tensor sang <code>dtype</code>,{' '}
            <code>shape</code> và <code>data_offsets</code> <code>[begin, end]</code> của nó. Mọi thứ sau đó là buffer byte: thứ tự hàng,
            little-endian. Một khóa <code>__metadata__</code> có thể chứa các chuỗi tùy ý.
          </p>
        }
      >
        <p>
          Hugging Face checkpoints use safetensors. The first 8 bytes are N, an unsigned little-endian 64-bit integer. The next N bytes are a
          UTF-8 JSON header that must start with <code>{'{'}</code>, mapping each tensor name to its <code>dtype</code>,{' '}
          <code>shape</code> and <code>data_offsets</code> <code>[begin, end]</code>. Everything after that is the byte buffer: row-major,
          little-endian. A key <code>__metadata__</code> may hold free-form strings.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Hai chi tiết làm vấp các bộ nạp. Offset tính tương đối so với đầu buffer byte, nên một tensor bắt đầu ở 8 + N + begin trong file.
            Và định dạng không kiểm tra giá trị: NaN và vô cùng có thể nằm trong file, nên hãy kiểm tra chúng sau khi chuyển đổi, không phải
            trước. Thư viện chuẩn của Python là đủ để đọc nó: một độ dài, <code>json.loads</code> và các phép cắt lát.
          </p>
        }
      >
        <p>
          Two details trip loaders. The offsets are relative to the start of the byte buffer, so a tensor starts at 8 + N + begin in the file.
          And the format does not check the values: NaN and infinity may be in the file, so check for them after converting, not before.
          Python's standard library is enough to read it: a length, <code>json.loads</code> and slices.
        </p>
      </Translated>
      <Ask q={questionsSX38['x3/X3.8/q-st-n']} label={tr('Work it out', 'Tự tính')} />

      <h2><Translated vi="Một Llama của Hugging Face: SmolLM2-135M">A Hugging Face Llama: SmolLM2-135M</Translated></h2>
      <Translated
        vi={
          <p>
            <code>HuggingFaceTB/SmolLM2-135M</code> (Apache-2.0) là một <code>LlamaForCausalLM</code>: hidden size 576, intermediate 1536, 30
            lớp, 9 head query trên 3 head KV, nên kích thước head là 64. <code>model.safetensors</code> của nó chứa 272 tensor, tất cả đều BF16.
            Không có <code>lm_head.weight</code>: các embedding được dùng chung (tied), cùng ý tưởng với bộ phân loại dùng chung của llama2.c.
            Kiến trúc là đúng bước mà run.c tính (RMSNorm, grouped-query attention với RoPE, SwiGLU), nhưng bốn điều phải thay đổi trước khi
            các kernel của bạn chạy được nó.
          </p>
        }
      >
        <p>
          <code>HuggingFaceTB/SmolLM2-135M</code> (Apache-2.0) is a <code>LlamaForCausalLM</code>: hidden size 576, intermediate 1536, 30
          layers, 9 query heads over 3 KV heads, so head size 64. Its <code>model.safetensors</code> holds 272 tensors, every one BF16. There
          is no <code>lm_head.weight</code>: the embeddings are tied, the same idea as llama2.c's shared classifier. The architecture is the
          step run.c computes (RMSNorm, grouped-query attention with RoPE, SwiGLU), but four things must change before your kernels can run
          it.
        </p>
      </Translated>
      <ol>
        <Translated
          vi={
            <li>
              <b>bf16 sang float32.</b> Đặt 16 bit đã lưu vào nửa trên của một word 32 bit. bfloat16 là nửa trên của một float32, nên việc
              này chính xác: {hex16(0x3f80)} trở thành {hex32(bf16ToF32Bits(0x3f80))}, tức là {bf16ToF32(0x3f80)}.
            </li>
          }
        >
          <li>
            <b>bf16 to float32.</b> Put the 16 stored bits in the top half of a 32-bit word. bfloat16 is the top half of a float32, so this is
            exact: {hex16(0x3f80)} becomes {hex32(bf16ToF32Bits(0x3f80))}, which is {bf16ToF32(0x3f80)}.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Cách ghép cặp RoPE.</b> Hugging Face quay phần tử i với phần tử i + head_size / 2; run.c quay các cặp liên tiếp (phần{' '}
              <a href={href('x3', 'X3.7')}>X3.7</a>). Hoặc dùng cách ghép cặp chia nửa, hoặc hoán vị các hàng của Q và K như bộ chuyển đổi của
              llama.cpp làm, với số head KV cho K.
            </li>
          }
        >
          <li>
            <b>The RoPE pairing.</b> Hugging Face rotates element i with element i + head_size / 2; run.c rotates consecutive pairs (section{' '}
            <a href={href('x3', 'X3.7')}>X3.7</a>). Either use the half-split pairing, or permute the rows of Q and K as llama.cpp's converter
            does, with the KV head count for K.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Grouped-query attention, 9 trên 3.</b> Head query h đọc head KV h / 3, và k_proj với v_proj là 192 × 576, không phải 576 × 576.
            </li>
          }
        >
          <li>
            <b>Grouped-query attention, 9 over 3.</b> Query head h reads KV head h / 3, and k_proj and v_proj are 192 × 576, not 576 × 576.
          </li>
        </Translated>
        <Translated
          vi={
            <li>
              <b>Các hằng số.</b> Cơ số RoPE 100,000 và epsilon 1e-5 lấy từ <code>config.json</code>, không phải 10,000 của llama2.c.
            </li>
          }
        >
          <li>
            <b>The constants.</b> RoPE base 100,000 and epsilon 1e-5 come from <code>config.json</code>, not from llama2.c's 10,000.
          </li>
        </Translated>
      </ol>
      <Today checked="2026-11-20 (notes/today/x3-facts.md D4, B6; appA-facts A2)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              config.json ghi <code>"rope_interleaved": false</code> và <code>"tie_word_embeddings": true</code>. transformers chuyển một{' '}
              <code>rope_theta</code> cũ ở cấp trên cùng vào <code>rope_parameters</code> khi nạp, với 10,000 là mặc định nếu không có giá trị nào.
            </p>
          }
        >
          <p>
            config.json says <code>"rope_interleaved": false</code> and <code>"tie_word_embeddings": true</code>. transformers moves an old
            top-level <code>rope_theta</code> into <code>rope_parameters</code> on load, with 10,000 as the default when none is given.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSX38['x3/X3.8/q-bf16']} />
      <Ask q={questionsSX38['x3/X3.8/q-theta']} />
      <Translated
        vi={
          <p>
            Trình khám phá đặt ba checkpoint cạnh nhau. Kích thước safetensors tính giống như của .bin: số giá trị × 2 byte, cộng 8 + N, và
            với N 30,528 của SmolLM2 nó khớp kích thước trên Hub đến từng byte. Rồi hãy nhìn vào kích thước head.
          </p>
        }
      >
        <p>
          The explorer puts the three checkpoints side by side. The safetensors size works the same way as the .bin's: values × 2 bytes, plus
          8 + N, and with SmolLM2's N of 30,528 it lands on the Hub's size to the byte. Then look at the head sizes.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Trình khám phá tính kích thước và hình dạng; nó không đọc file thật nào và không chạy mô hình nào. Nó biết các trường header và
              hình dạng tensor mà lượt kiểm tra dữ kiện đã ghi lại, và không biết gì về giá trị trọng số hay tokenizer: M10 so sánh token id,
              nên không cần đến chúng.
            </p>
          }
        >
          <p>
            The explorer computes sizes and shapes; it reads no real file and runs no model. It knows the header fields and tensor shapes the
            facts pass recorded, and nothing about the weights' values or the tokenizer: M10 compares token ids, so it needs none.
          </p>
        </Translated>
      </Simplification>
      <Explorer />
      <Ask q={questionsSX38['x3/X3.8/q-head64']} />

      <h2><Translated vi="Hai cái bẫy">Two traps</Translated></h2>
      <Translated
        vi={
          <p>
            <b>GGUF không phải chỗ để bắt đầu.</b> Đó là thứ llama.cpp nạp, và đặc tả của nó rõ ràng, nhưng nó kéo theo metadata có kiểu, dữ
            liệu tensor căn lề 32 byte trừ khi file ghi khác, và, trong phần lớn file được phát hành, các kiểu tensor đã lượng tử hóa. Đó là
            chủ đề của M9; M10 chạy FP32, và Q8_0 hay Q4_0 qua lớp của M9 chỉ cho các trọng số mà chính chương trình điều khiển (driver) của nó lượng tử hóa: không có
            bộ nạp GGUF nào được xây.
          </p>
        }
      >
        <p>
          <b>GGUF is not the place to start.</b> It is what llama.cpp loads, and its spec is clear, but it brings typed metadata, tensor data
          aligned to 32 bytes unless the file says otherwise, and, in most published files, quantised tensor types. Those are M9's subject;
          M10 runs FP32, and Q8_0 or Q4_0 through M9's layer only for weights its own driver quantised: no GGUF loader is built.
        </p>
      </Translated>
      <Today checked="2026-11-20 (notes/today/x3-facts.md D5)">{/* i18n-ignore: citation */}
        <Translated
          vi={
            <p>
              ggml/docs/gguf.md: magic <code>GGUF</code>, phiên bản 3, offset tensor tương đối so với phần dữ liệu và là bội số của{' '}
              <code>general.alignment</code>, 32 nếu không đặt; mặc định little-endian.
            </p>
          }
        >
          <p>
            ggml/docs/gguf.md: magic <code>GGUF</code>, version 3, tensor offsets relative to the data section and a multiple of{' '}
            <code>general.alignment</code>, 32 if unset; little-endian by default.
          </p>
        </Translated>
      </Today>
      <Ask q={questionsSX38['x3/X3.8/q-gguf']} />
      <Translated
        vi={
          <p>
            <b>Đường nhập Hugging Face của chính llama2.c không xử lý K grouped-query.</b> <code>export.py</code> của nó đảo ngược phép hoán vị
            RoPE của Hugging Face bằng <code>n_heads</code> và <code>dim</code> cho cả Q lẫn K. Phép chiếu K của một mô hình GQA nhỏ hơn thế, nên
            phép reshape không thể khớp. Bộ chuyển đổi của llama.cpp dùng số head KV cho K. Đây là kết quả đọc mã nguồn, không phải một lần chạy.
          </p>
        }
      >
        <p>
          <b>llama2.c's own Hugging Face import does not handle grouped-query K.</b> Its <code>export.py</code> undoes Hugging Face's RoPE
          permutation with <code>n_heads</code> and <code>dim</code> for both Q and K. A GQA model's K projection is smaller than that, so the
          reshape cannot match. llama.cpp's converter uses the KV head count for K. This is a reading of the source, not a run.
        </p>
      </Translated>
      <Ask q={questionsSX38['x3/X3.8/q-export-gqa']} />

      <h2><Translated vi="Điều này có ý nghĩa gì với M10">What this means for M10</Translated></h2>
      <Translated
        vi={
          <p>
            Mốc dự án M10 chạy một mô hình kiểu Llama từ đầu đến cuối trên các kernel của khóa học. Mô hình của nó là tổng hợp, được sinh trong
            repository, và ghi theo định dạng phiên bản 0 của llama2.c, nên cùng bộ nạp đọc được một file stories mà không cần thay đổi. Kích
            thước head của nó là 128 theo cấu tạo: bộ sinh của lab yêu cầu dim = heads × 128. Thứ chặn một file thật chính là điều đó: không
            checkpoint nhỏ nào trong ba cái có 128 của M7, nên kernel decode cần HEAD_DIM dưới dạng tham số, hoặc mỗi mô hình một bản build, và
            bố cục lane giới hạn những giá trị dùng được.
          </p>
        }
      >
        <p>
          Milestone M10 runs a Llama-style model end to end on the course's kernels. Its model is synthetic, generated in the repository, and
          written in llama2.c's version 0 format, so the same loader reads a stories file with no change. Its head size is 128 by
          construction: the lab's generator requires dim = heads × 128. What stops a real file is exactly that: none of the three small
          checkpoints has M7's 128, so the decode kernel needs HEAD_DIM as a parameter, or one build per model, and the lane layout limits
          which values work.
        </p>
      </Translated>
      <Translated
        vi={
          <p>
            Không có gì trong repository tải checkpoint về. README của lab X3.2, trên <a href={href('x3', 'labs')}>trang lab</a>, mô tả đường
            thủ công cho một file bạn tự tải. Tóm tắt: tải stories260K, kiểm tra kích thước bằng chương trình ở trên, trỏ bộ nạp vào nó, và
            so sánh token id với lần chạy greedy của run.c (<code>-t 0.0</code>). Đường đó unverified từ đầu đến cuối. Phần tiếp theo nói về
            việc làm cho phép so sánh đó có ý nghĩa.
          </p>
        }
      >
        <p>
          Nothing in the repository downloads a checkpoint. Lab X3.2's README, on the <a href={href('x3', 'labs')}>labs page</a>, describes
          the manual path for a file you download yourself. In outline: fetch stories260K, check its size with the program above, point the
          loader at it, and compare token ids with run.c's greedy run (<code>-t 0.0</code>). That path is unverified end to end. The next
          section is about making that comparison mean something.
        </p>
      </Translated>
    </>
  );
}
