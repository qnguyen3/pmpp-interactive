import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { KATS, M0, M1, W0, W1, avalanche, f32short, hex, m5Input, m5U, philoxTrace, sameWords, type Ctr } from '../../sim/philox';

// "Ten rounds of Philox" (section X1.7): two rows of a batch, streams 0 and 1, at seed 0 and
// step 0, through Philox4x32-10 (src/sim/philox.ts). Six beats, one narration clip each
// (content/x1/narration/philox-rounds-N.txt). Every word drawn comes from philoxTrace and
// avalanche, pinned in philox.test.ts.
const A = m5Input(0n, 0n, 0n);
const B = m5Input(0n, 1n, 0n);
const TA = philoxTrace(A.ctr, A.key);
const TB = philoxTrace(B.ctr, B.key);
const AV = avalanche(A.ctr, B.ctr, A.key);
const ZERO = KATS[0];
/** The round the 'One round' beat shows: round 4 of row 1, whose words are no longer mostly zero. */
const ROUND_SHOWN = 4;

const W = 720;
const H = 330;
const MONO = { fontVariationSettings: "'MONO' 1" } as const; // i18n-ignore: CSS font variation value
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

function Word({ x, y, v, label, diff = false, ok = false, id, dim = 1, w = 84 }: { x: number; y: number; v: number; label?: string; diff?: boolean; ok?: boolean; id?: string; dim?: number; w?: number }) {
  return (
    <g data-word={id} data-value={hex(v)} data-diff={diff ? 1 : 0} opacity={dim}>
      <rect x={x} y={y} width={w} height={24} rx="3" fill={ok ? 'var(--ok-wash)' : diff ? 'var(--rule)' : 'var(--cell)'}
        stroke={ok ? 'var(--ok)' : diff ? 'var(--ink)' : 'var(--rule)'} strokeWidth={diff ? 1.6 : 1} />
      <text x={x + w / 2} y={y + 16} fontSize="12.5" textAnchor="middle" fontWeight={diff ? 700 : 400} style={{ ...MONO, fill: 'var(--ink)' }}>{hex(v)}</text>
      {label ? <text x={x + w / 2} y={y - 5} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{label}</text> : null}
    </g>
  );
}

function Frame({ beat, headline, children }: { beat: number; headline: string; children: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={headline} data-beat={beat} style={{ display: 'block' }}>
      <text x={14} y={22} fontSize="13" fontWeight="600" data-headline="" style={{ fill: 'var(--ink)' }}>{headline}</text>
      {children}
    </svg>
  );
}

const CTR_LABELS = () => [tr('step lo', 'bước, thấp'), tr('step hi', 'bước, cao'), tr('stream lo', 'stream, thấp'), tr('stream hi', 'stream, cao')];
/** Two rows' four words side by side, the differing ones marked. */
function Rows({ a, b, y, title, mask, dim = 1 }: { a: Ctr; b: Ctr; y: number; title: string; mask?: boolean[]; dim?: number }) {
  return (
    <g opacity={dim}>
      <text x={14} y={y - 12} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{title}</text>
      {[[tr('row 0 (stream 0)', 'hàng 0 (stream 0)'), a, 'a'], [tr('row 1 (stream 1)', 'hàng 1 (stream 1)'), b, 'b']].map(([name, ws, tag], r) => (
        <g key={r}>
          <text x={14} y={y + 16 + r * 34} fontSize="11.5" style={{ fill: 'var(--ink)' }}>{name as string}</text>
          {(ws as Ctr).map((v, i) => (
            <Word key={i} id={`${tag}-${i}`} x={136 + i * 92} y={y + r * 34} v={v} diff={!!mask?.[i]} />
          ))}
        </g>
      ))}
    </g>
  );
}

function Bars({ upto, x, y }: { upto: number; x: number; y: number }) {
  return (
    <g data-bars={AV.differing.slice(0, upto).join(',')}>
      <text x={x} y={y - 18} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('words that differ after each round', 'số word khác nhau sau mỗi vòng')}</text>
      {AV.differing.map((d, r) => (
        <g key={r} opacity={r < upto ? 1 : 0.25}>
          <rect x={x + r * 26} y={y + (4 - (r < upto ? d : 0)) * 14} width={18} height={(r < upto ? d : 0) * 14} fill="var(--ink-2)" />
          <rect x={x + r * 26} y={y} width={18} height={56} fill="none" stroke="var(--rule)" />
          <text x={x + r * 26 + 9} y={y + 70} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{r + 1}</text>
          {r < upto ? <text x={x + r * 26 + 9} y={y - 1 + (4 - d) * 14 - 2} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{d}</text> : null}
        </g>
      ))}
    </g>
  );
}

const beats: Beat[] = [
  {
    clip: 'philox-rounds-1',
    get label() { return tr('Counter and key', 'Counter và key'); },
    draw: (p) => (
      <Frame beat={1} headline={tr('no state: the bits are Philox(counter, key), a function of where you are', 'không có trạng thái: các bit là Philox(counter, key), một hàm của vị trí bạn đang ở')}>
        <text x={14} y={52} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('counter: step and stream, four 32-bit words', 'counter: step và stream, bốn word 32-bit')}</text>
        {[[tr('row 0', 'hàng 0'), A], [tr('row 1', 'hàng 1'), B]].map(([name, inp], r) => (
          <g key={r} opacity={ramp(p, 0.05 + r * 0.25, 0.25 + r * 0.25)}>
            <text x={14} y={100 + r * 52} fontSize="11.5" style={{ fill: 'var(--ink)' }}><Translated vi={<>{name as string} (stream {r})</>}>{name as string} (stream {r})</Translated></text>
            {(inp as typeof A).ctr.map((v, i) => (
              <Word key={i} id={`${r ? 'b' : 'a'}-${i}`} x={136 + i * 92} y={84 + r * 52} v={v} label={r === 0 ? CTR_LABELS()[i] : undefined} diff={r === 1 && v !== A.ctr[i]} />
            ))}
          </g>
        ))}
        <g opacity={ramp(p, 0.55, 0.75)}>
          <text x={14} y={210} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('key: the seed, two 32-bit words (seed 0)', 'key: seed, hai word 32-bit (seed 0)')}</text>
          <Word id="k-0" x={136} y={234} v={A.key[0]} label={tr('seed lo', 'seed, thấp')} />
          <Word id="k-1" x={228} y={234} v={A.key[1]} label={tr('seed hi', 'seed, cao')} />
        </g>
        <text x={520} y={250} fontSize="12" opacity={ramp(p, 0.8, 0.95)} style={{ fill: 'var(--ink)' }}>{tr('step 0, seed 0 for both rows', 'step 0, seed 0 cho cả hai hàng')}</text>
      </Frame>
    ),
  },
  {
    clip: 'philox-rounds-2',
    get label() { return tr('One round', 'Một vòng'); },
    draw: (p) => {
      const r = TB[ROUND_SHOWN - 1];
      const s1 = ramp(p, 0.1, 0.3);
      const s2 = ramp(p, 0.3, 0.5);
      const s3 = ramp(p, 0.5, 0.75);
      const s4 = ramp(p, 0.75, 0.92);
      return (
        <Frame beat={2} headline={tr(`round ${r.round} of row 1: two products, split, XOR, swap`, `vòng ${r.round} của hàng 1: hai tích, tách, XOR, hoán đổi`)} >
          <text x={14} y={46} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr("words in (x, y, z, w) and this round's key", 'các word vào (x, y, z, w) và key của vòng này')}</text>
          {r.ctrIn.map((v, i) => <Word key={i} id={`in-${i}`} x={136 + i * 92} y={72} v={v} label={['x', 'y', 'z', 'w'][i]} />)}
          <Word id="key-0" x={520} y={72} v={r.key[0]} label={tr('key 0', 'key 0')} w={80} />
          <Word id="key-1" x={610} y={72} v={r.key[1]} label={tr('key 1', 'key 1')} w={80} />
          <g opacity={s1} data-products="">
            <text x={14} y={136} fontSize="12" style={{ ...MONO, fill: 'var(--ink)' }}>{hex(M0)} × x = {hex(r.p0.hi)} {hex(r.p0.lo)}</text>
            <text x={14} y={170} fontSize="12" style={{ ...MONO, fill: 'var(--ink)' }}>{hex(M1)} × z = {hex(r.p1.hi)} {hex(r.p1.lo)}</text>
          </g>
          <g opacity={s2}>
            <Word id="hi0" x={430} y={120} v={r.p0.hi} label={tr('high 0', 'cao 0')} w={80} />
            <Word id="lo0" x={520} y={120} v={r.p0.lo} label={tr('low 0', 'thấp 0')} w={80} />
            <Word id="hi1" x={430} y={154} v={r.p1.hi} w={80} />
            <Word id="lo1" x={520} y={154} v={r.p1.lo} w={80} />
            <text x={610} y={170} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{tr('high 1, low 1', 'cao 1, thấp 1')}</text>
          </g>
          <g opacity={s3}>
            <text x={14} y={210} fontSize="12" style={{ ...MONO, fill: 'var(--ink)' }}><Translated vi={<>x mới = cao 1 ⊕ y ⊕ key 0 = {hex(r.out[0])}</>}>new x = high 1 ⊕ y ⊕ key 0 = {hex(r.out[0])}</Translated></text>
            <text x={14} y={230} fontSize="12" style={{ ...MONO, fill: 'var(--ink)' }}><Translated vi={<>z mới = cao 0 ⊕ w ⊕ key 1 = {hex(r.out[2])}</>}>new z = high 0 ⊕ w ⊕ key 1 = {hex(r.out[2])}</Translated></text>
          </g>
          <g opacity={s4}>
            <text x={14} y={256} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('words out: the low halves move into y and w', 'các word ra: các nửa thấp chuyển vào y và w')}</text>
            {r.out.map((v, i) => <Word key={i} id={`out-${i}`} x={136 + i * 92} y={292} v={v} label={[tr('hi 1 ⊕ y ⊕ k0', 'cao 1 ⊕ y ⊕ k0'), tr('low 1', 'thấp 1'), tr('hi 0 ⊕ w ⊕ k1', 'cao 0 ⊕ w ⊕ k1'), tr('low 0', 'thấp 0')][i]} />)}
          </g>
        </Frame>
      );
    },
  },
  {
    clip: 'philox-rounds-3',
    get label() { return tr('Key bump', 'Tăng key'); },
    draw: (p) => {
      const k2 = TA[1].key;
      return (
        <Frame beat={3} headline={tr('between rounds the key grows; after round 1 the rows differ in one word', 'giữa các vòng key tăng lên; sau vòng 1 hai hàng khác nhau ở một word')}>
          <text x={14} y={58} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('key for round 1, then for round 2', 'key cho vòng 1, rồi cho vòng 2')}</text>
          <Word id="key1-0" x={136} y={66} v={TA[0].key[0]} w={80} />
          <Word id="key1-1" x={226} y={66} v={TA[0].key[1]} w={80} />
          <g opacity={ramp(p, 0.1, 0.35)}>
            <text x={320} y={82} fontSize="12" style={{ ...MONO, fill: 'var(--ink)' }}>+ ({hex(W0)}, {hex(W1)}) =</text>
            <Word id="key2-0" x={520} y={66} v={k2[0]} w={80} />
            <Word id="key2-1" x={610} y={66} v={k2[1]} w={80} />
          </g>
          <g opacity={ramp(p, 0.45, 0.7)} data-round={1} data-differing={AV.differing[0]}>
            <Rows a={TA[0].out} b={TB[0].out} y={150} title={tr('after round 1', 'sau vòng 1')} mask={p > 0.75 ? AV.masks[0] : undefined} />
          </g>
          <text x={136} y={248} fontSize="12" opacity={ramp(p, 0.8, 0.95)} style={{ fill: 'var(--ink)' }}>
            <Translated vi={<>{AV.differing[0]} trên 4 word khác nhau: word {AV.masks[0].indexOf(true)}</>}>
              {AV.differing[0]} of 4 words differ: word {AV.masks[0].indexOf(true)}
            </Translated>
          </text>
        </Frame>
      );
    },
  },
  {
    clip: 'philox-rounds-4',
    get label() { return tr('Spreading', 'Lan rộng'); },
    draw: (p) => {
      const r = Math.min(5, 2 + Math.floor(ramp(p, 0.05, 0.9) * 4)); // rounds 2 to 5
      return (
        <Frame beat={4} headline={tr(`after round ${r}: ${AV.differing[r - 1]} of 4 words differ`, `sau vòng ${r}: ${AV.differing[r - 1]} trên 4 word khác nhau`)}>
          <g data-round={r} data-differing={AV.differing[r - 1]}>
            <Rows a={TA[r - 1].out} b={TB[r - 1].out} y={68} title={tr(`after round ${r}`, `sau vòng ${r}`)} mask={AV.masks[r - 1]} />
          </g>
          <Bars upto={r} x={136} y={186} />
        </Frame>
      );
    },
  },
  {
    clip: 'philox-rounds-5',
    get label() { return tr('Round ten', 'Vòng mười'); },
    draw: (p) => {
      const out = TA[9].out;
      const match = sameWords(out, ZERO.expected);
      const top = out[0] >>> 8;
      return (
        <Frame beat={5} headline={tr("after round 10: row 0 is Random123's vector for a zero counter and key", 'sau vòng 10: hàng 0 chính là vector của Random123 cho counter và key bằng 0')}>
          <g data-round={10} data-kat-match={match ? 1 : 0}>
            <text x={14} y={70} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('row 0 after round 10', 'hàng 0 sau vòng 10')}</text>
            {out.map((v, i) => <Word key={i} id={`a-${i}`} x={136 + i * 92} y={78} v={v} ok={p > 0.35 && match} />)}
          </g>
          <g opacity={ramp(p, 0.2, 0.35)}>
            <text x={14} y={136} fontSize="11" style={{ fill: 'var(--ink-2)' }}>tests/kat_vectors, philox4x32 10, counter 0, key 0</text>{/* i18n-ignore: file path and test-vector identifiers */}
            {ZERO.expected.map((v, i) => <Word key={i} id={`kat-${i}`} x={136 + i * 92} y={144} v={v} />)}
            <text x={520} y={161} fontSize="13" fontWeight="700" opacity={ramp(p, 0.35, 0.45)} style={{ fill: 'var(--ok)' }}>{tr('✓ the same', '✓ giống hệt')}</text>
          </g>
          <g opacity={ramp(p, 0.55, 0.75)} data-u={m5U(0n, 0n, 0n)}>
            <text x={14} y={224} fontSize="12" style={{ ...MONO, fill: 'var(--ink)' }}>
              <Translated vi={<>word 0 = {hex(out[0])}; 24 bit cao = 0x{top.toString(16).padStart(6, '0')} = {top.toLocaleString('en-US')}</>}>
                word 0 = {hex(out[0])}; top 24 bits = 0x{top.toString(16).padStart(6, '0')} = {top.toLocaleString('en-US')}
              </Translated>
            </text>
            <text x={14} y={250} fontSize="12" style={{ ...MONO, fill: 'var(--ink)' }}>
              u = {top.toLocaleString('en-US')} × 2⁻²⁴ = {f32short(m5U(0n, 0n, 0n))}
            </text>
          </g>
        </Frame>
      );
    },
  },
  {
    clip: 'philox-rounds-6',
    get label() { return tr('Every row at once', 'Mọi hàng cùng lúc'); },
    draw: (p) => {
      const rows = Array.from({ length: 8 }, (_, s) => ({ s, w: philoxTrace(m5Input(0n, BigInt(s), 0n).ctr, A.key)[9].out[0], u: m5U(0n, BigInt(s), 0n) }));
      const on = ramp(p, 0.1, 0.3);
      return (
        <Frame beat={6} headline={tr('every row computes its own u at once; same inputs, same bits on every backend', 'mọi hàng cùng lúc tính u của riêng nó; cùng đầu vào, cùng bit trên mọi backend')}>
          <text x={14} y={56} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('seed 0, step 0; one thread per row, none waits for another', 'seed 0, step 0; một luồng cho mỗi hàng, không luồng nào chờ luồng khác')}</text>
          {rows.map((r) => (
            <g key={r.s} opacity={on} data-row={r.s} data-u={r.u}>
              <text x={14} y={84 + r.s * 26} fontSize="11.5" style={{ fill: 'var(--ink)' }}><Translated vi={<>hàng {r.s} (stream {r.s})</>}>row {r.s} (stream {r.s})</Translated></text>
              <Word x={136} y={68 + r.s * 26} v={r.w} />
              <text x={236} y={84 + r.s * 26} fontSize="12" style={{ ...MONO, fill: 'var(--ink)' }}>u = {f32short(r.u)}</text>
            </g>
          ))}
          <g opacity={ramp(p, 0.55, 0.75)}>
            {['CUDA', 'HIP', 'Metal', 'CPU'].map((b, i) => (
              <g key={b}>
                <rect x={430} y={90 + i * 40} width={120} height={28} rx="3" fill="var(--cell)" stroke="var(--rule)" />
                <text x={490} y={109 + i * 40} fontSize="12" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{b}</text>
              </g>
            ))}
            <text x={570} y={150} fontSize="11.5" style={{ fill: 'var(--ink)' }}>{tr('same seed, stream', 'cùng seed, stream')}</text>
            <text x={570} y={166} fontSize="11.5" style={{ fill: 'var(--ink)' }}>{tr('and step: same bits,', 'và step: cùng bit,')}</text>
            <text x={570} y={182} fontSize="11.5" style={{ fill: 'var(--ink)' }}>{tr('once each passes', 'một khi mỗi backend')}</text>
            <text x={570} y={198} fontSize="11.5" style={{ fill: 'var(--ink)' }}>{tr('the known answers', 'khớp các đáp án đã biết')}</text>
          </g>
        </Frame>
      );
    },
  },
];

export function PhiloxRoundsScene() {
  return <Scene chapter="x1" title={tr('Ten rounds of Philox', 'Mười vòng Philox')} beats={beats} />;
}
