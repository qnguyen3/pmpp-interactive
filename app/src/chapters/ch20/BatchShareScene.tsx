import { Scene, type Beat } from '../../components/Scene';
import { GiB, batchedDecodeAttentionAI, linearAI, presetByKey, tokensThatFit } from '../../sim/kvmath';
import { Translated, tr } from '../../lib/i18n';

// "Batching shares weights, not caches" (section 20.6, Fig. 20.17). One layer of a
// Llama 2 7B-shaped model (4,096 wide; book p. 506), 16-bit, each request with 1,024
// tokens of cache. The projection is X W_QKV (4,096 in, 3 x 4,096 out). Every number comes
// from src/sim/kvmath.ts. Five beats, one narration clip each
// (content/ch20/narration/batch-share-N.txt).

const D = 4096;
const P = 2;
const N = 1024;
export const WEIGHT_BYTES = D * 3 * D * P; // read once per step, whatever the batch
export const CACHE_BYTES = 2 * N * D * P; // per request, this layer
export const projAI = (b: number) => linearAI(b, D, 3 * D, P);
export const attnAI = (b: number) => batchedDecodeAttentionAI(b, N, P);
const LLAMA = presetByKey('llama2-7b')!.shape;
/** Tokens of Llama 2 7B cache (all layers, 16-bit) that one GiB holds: 2,048. */
export const TOKENS_PER_GIB = tokensThatFit(GiB, LLAMA, P);

const W = 720;
const H = 340;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const mib = (b: number) => (b / 2 ** 20).toLocaleString('en-US', { maximumFractionDigits: 1 });

function Stage({ b, headline, attn = 1, showAttn = true, memory = false }: { b: number; headline: string; attn?: number; showAttn?: boolean; memory?: boolean }) {
  const shownRows = Math.min(b, 8);
  const cols = Math.min(b, 16);
  const pa = projAI(b);
  const aa = attnAI(b);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Weights shared by a batch, caches read per request', 'Trọng số được cả batch dùng chung, cache được đọc theo từng yêu cầu')} style={{ width: '100%', maxWidth: W, display: 'block' }}
      data-batch={b} data-proj-ai={pa.toFixed(3)} data-attn-ai={aa.toFixed(3)}>
      <text x={12} y={24} fontSize="14.5" fontWeight="600" data-headline="">{headline}</text>
      <text x={W - 12} y={44} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
        <Translated vi="một lớp, rộng 4,096, 16-bit, 1,024 token trong cache mỗi yêu cầu">one layer, 4,096 wide, 16-bit, 1,024 cached tokens per request</Translated>
      </text>
      {/* projections: one weight block, b rows in and out */}
      <text x={20} y={70} fontSize="12" fontWeight="600"><Translated vi="các phép chiếu">projections</Translated></text>
      {Array.from({ length: shownRows }, (_, i) => (
        <rect key={i} x={20} y={84 + i * 9} width={56} height={6} fill="var(--reg)" />
      ))}
      {b > shownRows ? <text x={20} y={84 + shownRows * 9 + 8} fontSize="10" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>+{b - shownRows} hàng</>}>+{b - shownRows} rows</Translated></text> : null}
      <text x={86} y={120} fontSize="16">×</text>
      <rect x={104} y={80} width={92} height={92} fill="var(--global-wash)" stroke="var(--global)" />
      <text x={150} y={122} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--global)' }}>W_QKV</text>
      <text x={150} y={137} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}><Translated vi="đọc một lần">read once</Translated></text>
      <text x={104} y={190} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>{mib(WEIGHT_BYTES)} MiB mỗi bước</>}>{mib(WEIGHT_BYTES)} MiB per step</Translated></text>
      {/* attention: one cache per request */}
      {showAttn ? (
        <g opacity={attn}>
          <text x={250} y={70} fontSize="12" fontWeight="600"><Translated vi="attention: cache riêng của từng yêu cầu">attention: each request’s own cache</Translated></text>
          {Array.from({ length: cols }, (_, i) => (
            <g key={i}>
              <rect x={250 + i * 28} y={80} width={22} height={92} fill="var(--global)" opacity={0.85} />
              <text x={261 + i * 28} y={186} fontSize="9" textAnchor="middle">r{i + 1}</text>
            </g>
          ))}
          {b > cols ? <text x={250 + cols * 28} y={130} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>+{b - cols}</text> : null}
          <text x={250} y={204} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}><Translated vi={<>mỗi cache {mib(CACHE_BYTES)} MiB, tổng cộng {mib(CACHE_BYTES * b)} MiB</>}>{mib(CACHE_BYTES)} MiB each, {mib(CACHE_BYTES * b)} MiB in all</Translated></text>
        </g>
      ) : null}
      {/* counters */}
      <g>
        <text x={20} y={240} fontSize="11.5"><Translated vi="byte trọng số mỗi token">weight bytes per token</Translated></text>
        <text x={230} y={240} fontSize="12.5" textAnchor="end" fontWeight="650" data-weight-per-token={WEIGHT_BYTES / b}>{mib(WEIGHT_BYTES / b)} MiB</text>
        <text x={20} y={260} fontSize="11.5"><Translated vi="các phép chiếu">projections</Translated></text>
        <text x={230} y={260} fontSize="12.5" textAnchor="end" fontWeight="650">{pa.toFixed(2)} FLOP/B</text>
        {showAttn ? (
          <g opacity={attn}>
            <text x={260} y={240} fontSize="11.5"><Translated vi="byte cache mỗi token">cache bytes per token</Translated></text>
            <text x={480} y={240} fontSize="12.5" textAnchor="end" fontWeight="650">{mib(CACHE_BYTES)} MiB</text>
            <text x={260} y={260} fontSize="11.5"><Translated vi="attention">attention</Translated></text>
            <text x={480} y={260} fontSize="12.5" textAnchor="end" fontWeight="650">{aa.toFixed(2)} FLOP/B</text>
          </g>
        ) : null}
      </g>
      {memory ? (
        <g>
          <text x={20} y={292} fontSize="11.5"><Translated vi="bộ nhớ lớp này cần cho một bước">memory this layer needs for one step</Translated></text>
          {(() => {
            const total = WEIGHT_BYTES + CACHE_BYTES * 64;
            const s = 680 / total;
            return (
              <g>
                <rect x={20} y={300} width={WEIGHT_BYTES * s} height={16} fill="var(--global-wash)" stroke="var(--global)" />
                <rect x={20 + WEIGHT_BYTES * s} y={300} width={CACHE_BYTES * b * s} height={16} fill="var(--global)" />
                <text x={20} y={332} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
                  <Translated vi={<>trọng số {mib(WEIGHT_BYTES)} MiB + cache {mib(CACHE_BYTES * b)} MiB</>}>
                    weights {mib(WEIGHT_BYTES)} MiB + caches {mib(CACHE_BYTES * b)} MiB
                  </Translated>
                </text>
              </g>
            );
          })()}
        </g>
      ) : null}
    </svg>
  );
}

function Tradeoff({ p }: { p: number }) {
  const batches = [1, 2, 4, 8, 16];
  const shown = Math.max(1, Math.ceil(ramp(p, 0.15, 0.8) * batches.length));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={tr('Batch size times context length per gibibyte', 'Kích thước batch nhân độ dài ngữ cảnh trên mỗi gibibyte')} style={{ width: '100%', maxWidth: W, display: 'block' }}
      data-tokens-per-gib={TOKENS_PER_GIB}>
      <text x={12} y={24} fontSize="14.5" fontWeight="600" data-headline=""><Translated vi="Batch × ngữ cảnh bị giới hạn bởi bộ nhớ còn lại sau trọng số">Batch × context is capped by the memory left after the weights</Translated></text>
      <text x={20} y={58} fontSize="11.5">
        <Translated vi={<>Llama 2 7B, cả 32 lớp, 16-bit: 1 GiB bộ nhớ trống chứa được {TOKENS_PER_GIB.toLocaleString('en-US')} token cache</>}>
          Llama 2 7B, all 32 layers, 16-bit: 1 GiB of free memory holds {TOKENS_PER_GIB.toLocaleString('en-US')} tokens of cache
        </Translated>
      </text>
      {batches.slice(0, shown).map((b, i) => {
        const n = TOKENS_PER_GIB / b;
        const y = 84 + i * 48;
        return (
          <g key={b} data-b={b} data-n={n}>
            <text x={20} y={y + 22} fontSize="11.5"><Translated vi={<>{b} chuỗi</>}>{b} {b === 1 ? 'sequence' : 'sequences'}</Translated></text>
            {Array.from({ length: b }, (_, k) => (
              <rect key={k} x={140} y={y + 6 + (k * 30) / b} width={(n / TOKENS_PER_GIB) * 440} height={Math.max(1.5, 30 / b - 1)} fill="var(--global)" />
            ))}
            <text x={140 + (n / TOKENS_PER_GIB) * 440 + 8} y={y + 22} fontSize="11.5"><Translated vi={<>mỗi chuỗi {n.toLocaleString('en-US')} token</>}>{n.toLocaleString('en-US')} tokens each</Translated></text>
          </g>
        );
      })}
    </svg>
  );
}

const beats: Beat[] = [
  { clip: 'batch-share-1', get label() { return tr('One request', 'Một yêu cầu'); }, draw: () => <Stage b={1} headline={tr('One request: weights read once, its cache read once', 'Một yêu cầu: trọng số đọc một lần, cache của nó đọc một lần')} /> },
  {
    clip: 'batch-share-2',
    get label() {
      return tr('Four share the weights', 'Bốn yêu cầu dùng chung trọng số');
    },
    draw: (p) => <Stage b={p < 0.15 ? 1 : 4} headline={tr('Four requests: the same weight bytes now serve four rows', 'Bốn yêu cầu: cùng số byte trọng số giờ phục vụ bốn hàng')} showAttn={false} />,
  },
  {
    clip: 'batch-share-3',
    get label() {
      return tr('Four caches', 'Bốn cache');
    },
    draw: (p) => <Stage b={4} headline={tr('Attention: four caches, four times the bytes and the work', 'Attention: bốn cache, gấp bốn lần số byte và khối lượng việc')} attn={ramp(p, 0.05, 0.3)} />,
  },
  {
    clip: 'batch-share-4',
    get label() {
      return tr('Grow the batch', 'Tăng batch');
    },
    draw: (p) => {
      const b = Math.max(4, Math.round(4 * 16 ** ramp(p, 0.05, 0.85)));
      return <Stage b={b} headline={tr(`Batch ${b}: weight bytes per token fall, cache bytes per token do not`, `Batch ${b}: byte trọng số mỗi token giảm, byte cache mỗi token thì không`)} memory />;
    },
  },
  { clip: 'batch-share-5', get label() { return tr('The trade-off', 'Sự đánh đổi'); }, draw: (p) => <Tradeoff p={p} /> },
];

export function BatchShareScene() {
  return <Scene chapter="ch20" title={tr('Batching shares weights, not caches', 'Batching dùng chung trọng số, không dùng chung cache')} beats={beats} />;
}
