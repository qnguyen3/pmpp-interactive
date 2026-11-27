import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { Translated, tr } from '../../lib/i18n';
import { ENTRY_POINTS, METAL_NORM_NAMES, ONE_CALL, PROJECT_NAME, cell, fuseGraph, metalNormName } from '../../sim/backendmap';

// "One RMSNorm call, three codebases": the start of a Llama layer in llama.cpp's graph, the CUDA
// fusion pass consuming RMS_NORM and MUL before the switch, the fused kernel in norm.cu, the
// Metal switch and the name string, the host_name instantiation it matches, and the rule.
// Every label is drawn from sim/backendmap.ts (ONE_CALL, fuseGraph, metalNormName,
// METAL_NORM_NAMES, ENTRY_POINTS, the RMSNorm cells). Six beats, one narration clip each
// (content/x3/narration/one-call-N.txt).

const G = ONE_CALL.graph;
const FUSED = fuseGraph(G, true);
const PLAIN = fuseGraph(G, false);
const RMS = G.indexOf('RMS_NORM');
const METAL = metalNormName(G, RMS);
const HIP_TAGS = cell('rmsnorm', 'llama.cpp', 'hip').tags;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

const NW = 128, NH = 34, NX = 24, NY = 34, GAP = 18; // graph nodes
const nodeX = (i: number) => NX + i * (NW + GAP);

function Node({ i, show = 1, hot = false, gone = false }: { i: number; show?: number; hot?: boolean; gone?: boolean }) {
  return (
    <g opacity={show * (gone ? 0.25 : 1)}>
      <rect x={nodeX(i)} y={NY} width={NW} height={NH} rx="3" fill={hot ? 'var(--ink)' : 'var(--cell)'} stroke="var(--ink)" strokeWidth={hot ? 1.5 : 1} />
      <text x={nodeX(i) + NW / 2} y={NY + 21} fontSize="11.5" textAnchor="middle" style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>
        GGML_OP_{G[i]}
      </text>
      {i < G.length - 1 ? <line x1={nodeX(i) + NW} y1={NY + NH / 2} x2={nodeX(i + 1)} y2={NY + NH / 2} stroke="var(--ink-3)" /> : null}
    </g>
  );
}

function Box({ x, y, w, h = 30, text, sub, dim = false, strong = false, op = 1 }: {
  x: number; y: number; w: number; h?: number; text: ReactNode; sub?: ReactNode; dim?: boolean; strong?: boolean; op?: number;
}) {
  return (
    <g opacity={op}>
      <rect x={x} y={y} width={w} height={h} rx="3" fill="var(--page)" stroke={dim ? 'var(--ink-3)' : 'var(--ink)'} strokeWidth={strong ? 2 : 1}
        strokeDasharray={dim ? '4 3' : undefined} />
      <text x={x + 10} y={y + (sub ? 13 : h / 2 + 4)} fontSize="11.5" style={{ fill: dim ? 'var(--ink-3)' : 'var(--ink)' }}
        textDecoration={dim ? 'line-through' : undefined}>{text}</text>
      {sub ? <text x={x + 10} y={y + 26} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{sub}</text> : null}
    </g>
  );
}

function Frame({ children, headline }: { children: ReactNode; headline?: ReactNode }) {
  return (
    <svg viewBox="0 0 720 330" role="img" aria-label={tr("Animation of one RMSNorm call through llama.cpp's CUDA and Metal backends", 'Hoạt cảnh một lời gọi RMSNorm đi qua backend CUDA và Metal của llama.cpp')}
      style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      {children}
      {headline ? <text x={24} y={318} fontSize="13" fontWeight="600">{headline}</text> : null}
    </svg>
  );
}

const fusedCall = FUSED.find((d) => d.fused)!;
const plainRms = PLAIN.find((d) => d.nodes[0] === RMS)!;

const beats: Beat[] = [
  {
    clip: 'one-call-1',
    get label() { return tr('The graph', 'Đồ thị'); },
    draw: (p) => (
      <Frame headline={tr('a graph of op numbers; nothing has run yet', 'một đồ thị các số hiệu op; chưa có gì chạy cả')}>
        <text x={NX} y={22} fontSize="12" fontWeight="600">{tr('llama.cpp graph (start of a Llama layer)', 'Đồ thị llama.cpp (đầu một lớp Llama)')}</text>
        {G.map((_, i) => <Node key={i} i={i} show={ramp(p, i * 0.15, i * 0.15 + 0.15)} hot={p > 0.7 && (i === RMS || i === RMS + 1)} />)}
        <text x={nodeX(RMS)} y={NY + NH + 22} fontSize="11" opacity={ramp(p, 0.7, 0.85)} style={{ fill: 'var(--ink-2)' }}>
          {tr('the norm, then a multiply by its weight', 'phép chuẩn hóa, rồi nhân với trọng số của nó')}
        </text>
      </Frame>
    ),
  },
  {
    clip: 'one-call-2',
    get label() { return tr('Fusion pass', 'Lượt gộp op'); },
    draw: (p) => {
      const scan = Math.min(G.length - 1, Math.floor(ramp(p, 0, 0.45) * G.length));
      const matched = p >= 0.45;
      const merge = ramp(p, 0.5, 0.7);
      return (
        <Frame headline={matched ? tr(`${fusedCall.call} consumes two nodes; ${plainRms.call} is never reached`, `${fusedCall.call} tiêu thụ hai nút; ${plainRms.call} không bao giờ được chạy tới`) : tr('the CUDA backend scans the graph before its switch', 'backend CUDA quét đồ thị trước câu lệnh switch của nó')}>
          <text x={NX} y={22} fontSize="12" fontWeight="600">{tr('CUDA: fusion pass, then ggml_cuda_compute_forward', 'CUDA: lượt gộp op, rồi ggml_cuda_compute_forward')}</text>
          {G.map((_, i) => <Node key={i} i={i} hot={!matched && i === scan} gone={matched && fusedCall.nodes.includes(i) && merge >= 1} />)}
          {matched ? (
            <g opacity={merge}>
              <rect x={nodeX(RMS) - 4} y={NY - 4} width={2 * NW + GAP + 8} height={NH + 8} rx="4" fill="none" stroke="var(--ok)" strokeWidth="2" />
              <Box x={nodeX(RMS)} y={NY + NH + 22} w={2 * NW + GAP} text={fusedCall.call} sub={tr('ggml_cuda_can_fuse matched RMS_NORM, MUL', 'ggml_cuda_can_fuse khớp RMS_NORM, MUL')} strong />
            </g>
          ) : null}
          <Box x={NX} y={170} w={430} h={44} op={ramp(p, 0.55, 0.8)} dim={matched}
            text={`switch (dst->op), case GGML_OP_RMS_NORM: ${plainRms.call}`} sub={undefined} />
          <text x={NX + 440} y={196} fontSize="11" opacity={ramp(p, 0.75, 0.9)} style={{ fill: 'var(--ink-2)' }}>{tr('not reached while fusion is on', 'không được chạy tới khi gộp op đang bật')}</text>
          <text x={NX} y={250} fontSize="11" opacity={ramp(p, 0.85, 1)} style={{ fill: 'var(--ink-2)' }}>{tr('GGML_CUDA_DISABLE_FUSION turns the pass off', 'GGML_CUDA_DISABLE_FUSION tắt lượt này')}</text>
        </Frame>
      );
    },
  },
  {
    clip: 'one-call-3',
    label: 'norm.cu',
    draw: (p) => {
      const rows = 4;
      const lanes = 8;
      const stage1 = ramp(p, 0.15, 0.45);
      const stage2 = ramp(p, 0.45, 0.7);
      return (
        <Frame headline={p > 0.8 /* i18n-ignore: backend name prefix */ ? `HIP: ${HIP_TAGS.includes('same source as CUDA') ? tr('the same file, compiled as HIP', 'cùng tệp đó, biên dịch thành HIP') : ''}` : tr('one block per row, a two-stage sum of squares', 'mỗi hàng một khối, tổng bình phương hai giai đoạn')}>
          <Box x={NX} y={14} w={420} text={tr(`${ONE_CALL.cudaKernel.file}: ${ONE_CALL.cudaKernel.kernel}, lines ${ONE_CALL.cudaKernel.lines[0]} to ${ONE_CALL.cudaKernel.lines[1]}`, `${ONE_CALL.cudaKernel.file}: ${ONE_CALL.cudaKernel.kernel}, dòng ${ONE_CALL.cudaKernel.lines[0]} đến ${ONE_CALL.cudaKernel.lines[1]}`)} strong />
          {Array.from({ length: rows }, (_, r) => (
            <g key={r}>
              <text x={NX} y={78 + r * 50} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr(`block ${r}`, `khối ${r}`)}</text>
              {Array.from({ length: lanes * 2 }, (_, c) => (
                <rect key={c} x={NX + 52 + c * 14} y={66 + r * 50} width={12} height={14} rx="1" fill="var(--global-wash)" stroke="var(--global)" strokeWidth="0.6" />
              ))}
              <g opacity={stage1}>
                {Array.from({ length: 4 }, (_, k) => (
                  <rect key={k} x={NX + 300 + k * 18} y={66 + r * 50} width={14} height={14} rx="1" fill="var(--reg-wash)" stroke="var(--reg)" />
                ))}
              </g>
              <g opacity={stage2}>
                <rect x={NX + 390} y={66 + r * 50} width={14} height={14} rx="1" fill="var(--shared-wash)" stroke="var(--shared)" />
                <text x={NX + 412} y={77 + r * 50} fontSize="11">scale = rsqrt(mean + eps)</text>
              </g>
            </g>
          ))}
          <text x={NX + 52} y={58} fontSize="10.5" style={{ fill: 'var(--global)' }}>{tr('row in global memory', 'hàng trong bộ nhớ toàn cục')}</text>
          <text x={NX + 300} y={58} fontSize="10.5" opacity={stage1} style={{ fill: 'var(--reg)' }}>{tr('partial sums', 'tổng từng phần')}</text>
          <text x={NX + 390} y={58} fontSize="10.5" opacity={stage2} style={{ fill: 'var(--shared)' }}>{tr('shared', 'chia sẻ')}</text>
          <g opacity={ramp(p, 0.8, 0.95)}>
            <rect x={530} y={14} width={166} height={30} rx="3" fill="var(--ok-wash)" stroke="var(--ok)" />
            <text x={540} y={33} fontSize="11.5">{tr(HIP_TAGS[0], ({ 'own kernel': 'kernel riêng', 'same source as CUDA': 'cùng mã nguồn với CUDA', 'hipified at build time': 'được hipify lúc build', 'generated at build time': 'được sinh ra lúc build', 'generated at run time (mx.compile)': 'được sinh ra lúc chạy (mx.compile)', 'vendor library': 'thư viện của nhà cung cấp', 'third-party submodule': 'submodule của bên thứ ba', 'composite of other ops': 'tổ hợp từ các op khác', 'fused before dispatch': 'được gộp trước khi dispatch', 'found by name string': 'tìm theo chuỗi tên', 'not read in the facts pass': 'chưa đọc trong lượt kiểm tra dữ kiện', 'no backend': 'không có backend' } as Record<string, string>)[HIP_TAGS[0]] ?? HIP_TAGS[0])}</text>
          </g>
        </Frame>
      );
    },
  },
  {
    clip: 'one-call-4',
    get label() { return tr('Metal switch', 'switch của Metal'); },
    draw: (p) => {
      const typed = METAL.name.slice(0, Math.round(ramp(p, 0.55, 0.9) * METAL.name.length));
      return (
        <Frame headline={p > 0.9 ? tr(`n_fuse = ${METAL.consumed}; the pipeline is asked for by name`, `n_fuse = ${METAL.consumed}; pipeline được yêu cầu theo tên`) : tr('Metal: its own switch over the same op numbers', 'Metal: switch riêng của nó trên cùng các số hiệu op')}>
          <text x={NX} y={22} fontSize="12" fontWeight="600">
            <Translated vi={<>Metal: {ONE_CALL.metalSwitch.fn} (dòng {ONE_CALL.metalSwitch.line})</>}>
              Metal: {ONE_CALL.metalSwitch.fn} (line {ONE_CALL.metalSwitch.line})
            </Translated>
          </text>
          {G.map((_, i) => <Node key={i} i={i} hot={p > 0.3 && i >= RMS && i < RMS + METAL.consumed} />)}
          <Box x={NX} y={110} w={430} op={ramp(p, 0.1, 0.3)} text={tr(`case GGML_OP_RMS_NORM (line ${ONE_CALL.metalSwitch.caseLine}): ${ONE_CALL.metalSwitch.host}`, `case GGML_OP_RMS_NORM (dòng ${ONE_CALL.metalSwitch.caseLine}): ${ONE_CALL.metalSwitch.host}`)}
            sub={p > 0.35 ? tr(`consumed ${METAL.consumed} nodes: RMS_NORM, MUL`, `đã tiêu thụ ${METAL.consumed} nút: RMS_NORM, MUL`) : undefined} h={40} />
          <g opacity={ramp(p, 0.5, 0.6)}>
            <text x={NX} y={190} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{tr('pipeline name, as a string', 'tên pipeline, dưới dạng chuỗi')}</text>
            <rect x={NX} y={198} width={330} height={34} rx="3" fill="var(--cell)" stroke="var(--ink)" />
            <text x={NX + 12} y={220} fontSize="14" className="mono" style={{ fontFamily: 'ui-monospace, monospace' }}>"{typed}"</text>
          </g>
        </Frame>
      );
    },
  },
  {
    clip: 'one-call-5',
    label: 'host_name', // i18n-ignore: Metal attribute name
    draw: (p) => {
      const sweep = Math.min(METAL_NORM_NAMES.length - 1, Math.floor(ramp(p, 0.1, 0.55) * METAL_NORM_NAMES.length));
      const found = p >= 0.55;
      return (
        <Frame headline={found ? tr(`found by name: ${ONE_CALL.metalTemplate} with F = ${METAL.F}`, `tìm thấy theo tên: ${ONE_CALL.metalTemplate} với F = ${METAL.F}`) : tr('search the .metal files for the string', 'tìm chuỗi đó trong các tệp .metal')}>
          <text x={NX} y={22} fontSize="12" fontWeight="600">{tr('norm.metal, the float instantiations (lines 178 to 185)', 'norm.metal, các bản khởi tạo float (dòng 178 đến 185)')}</text>
          <rect x={NX} y={34} width={330} height={28} rx="3" fill="var(--cell)" stroke="var(--ink)" />
          <text x={NX + 12} y={53} fontSize="13" style={{ fontFamily: 'ui-monospace, monospace' }}>"{METAL.name}"</text>
          {METAL_NORM_NAMES.map((m, k) => {
            const hit = found && m.name === METAL.name;
            const looking = !found && k === sweep;
            return (
              <g key={m.name}>
                <rect x={NX} y={86 + k * 50} width={672} height={38} rx="3" fill={hit ? 'var(--ok-wash)' : 'var(--page)'}
                  stroke={hit ? 'var(--ok)' : looking ? 'var(--ink)' : 'var(--rule)'} strokeWidth={hit ? 2 : 1} />
                <text x={NX + 10} y={109 + k * 50} fontSize="11.5" style={{ fontFamily: 'ui-monospace, monospace' }}>{/* i18n-ignore: Metal source line */}
                  template [[host_name("{m.name}")]] … {ONE_CALL.metalTemplate}&lt;float, {m.F}&gt;;{/* i18n-ignore: Metal source line */}
                </text>
              </g>
            );
          })}
        </Frame>
      );
    },
  },
  {
    clip: 'one-call-6',
    get label() { return tr('The rule', 'Quy tắc'); },
    draw: (p) => (
      <Frame headline={tr('start at the dispatch, follow one name, on Metal follow the string', 'bắt đầu từ chỗ dispatch, lần theo một cái tên, trên Metal thì lần theo chuỗi')}>
        {ENTRY_POINTS.map((e, k) => (
          <g key={e.project} opacity={ramp(p, k * 0.2, k * 0.2 + 0.25)}>
            <rect x={NX + k * 228} y={20} width={216} height={250} rx="4" fill="var(--page)" stroke="var(--rule)" />
            <text x={NX + k * 228 + 12} y={44} fontSize="13" fontWeight="600">{PROJECT_NAME[e.project]}</text>
            <foreignObject x={NX + k * 228 + 8} y={52} width={200} height={214}>
              <div style={{ fontFamily: 'var(--sans)', fontSize: 11, lineHeight: 1.35, color: 'var(--ink)' }}>
                <p style={{ margin: '0 0 6px', color: 'var(--ink-2)' }}>{e.what}</p>
                <p style={{ margin: '0 0 6px' }}><b>CUDA</b> {e.cuda}</p>
                <p style={{ margin: '0 0 6px' }}><b>HIP</b> {e.hip}</p>
                <p style={{ margin: 0 }}><b>Metal</b> {e.metal}</p>
              </div>
            </foreignObject>
          </g>
        ))}
      </Frame>
    ),
  },
];

export function OneCallScene() {
  return <Scene chapter="x3" title={tr('One RMSNorm call, three codebases', 'Một lời gọi RMSNorm, ba codebase')} beats={beats} />;
}
