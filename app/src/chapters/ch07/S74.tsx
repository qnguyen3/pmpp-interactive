import { useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { solutionTabStatus } from '../../components/LabKit';
import { Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { fmt, realBackends, useReports, type Reports } from '../../lib/reports';
import { Translated, tr } from '../../lib/i18n';
import { href } from '../../lib/router';
import {
  DEMO_SEED, amdScalarEligible, distinctCount, expectedDistinct, filterTrace, nvidiaCost, replaySchedule, warpAddresses,
  type Pattern,
} from '../../sim/constcache';
import { ConstantBroadcastScene, WarpTable } from './ConstantBroadcastScene';

// Section 7.4: constant memory on CUDA, HIP and Metal, the constant cache's one-address
// rule, and where small read-only tables go in a real inference engine. The explorer
// and the scene run on sim/constcache.ts; the code is lab 7.1's reference solution.

const LAB71 = 'ch07/lab1_constconv';
const SOL = `${LAB71}/solution`;

// ---------------------------------------------------------------- lab 7.1's code, one part at a time
/** The lines after `// @part <name>` up to the next part marker; the whole file if the marker is missing. */
function part(file: string | null, name: string): string | null {
  if (file == null) return null;
  const lines = file.split('\n');
  const at = lines.findIndex((l) => l.trim() === `// @part ${name}`);
  if (at < 0) return file;
  const end = lines.findIndex((l, i) => i > at && /^\/\/ @part \w+$/.test(l.trim()));
  return lines
    .slice(at + 1, end < 0 ? lines.length : end)
    .join('\n')
    .replace(/\n+$/, '');
}

const linesWith = (text: string, word: string) =>
  text.split('\n').flatMap((l, i) => (l.includes(word) && !l.trim().startsWith('//') ? [i + 1] : []));

const REG: Record<string, Tier> = {
  Pvalue: 'reg', outRow: 'reg', outCol: 'reg', inRow: 'reg', inCol: 'reg', fRow: 'reg', fCol: 'reg',
};
// Each tab gets the colours that are true for it. CUDA passes kernel arguments through
// constant memory (ch07-facts A4), so the by-value struct f is coloured as constant.
const CU_CONST: Record<string, Tier> = { ...REG, F_c: 'const', N: 'global', P: 'global' };
const CU_BYVALUE: Record<string, Tier> = { ...REG, f: 'const', N: 'global', P: 'global' };
const MSL_CONST: Record<string, Tier> = { ...REG, F: 'const', N: 'global', P: 'global' };
const MSL_BYVALUE: Record<string, Tier> = { ...REG, f: 'const', N: 'global', P: 'global' };

function useStatuses() {
  const { reports, loading } = useReports();
  if (loading) return { cuda: tr('reading reports', 'đang đọc báo cáo'), metal: tr('reading reports', 'đang đọc báo cáo') };
  // The labs page's words (LabKit's solutionTabStatus), so a failed run reads as failed here too.
  return { cuda: solutionTabStatus(reports, LAB71, 'kernel.cu'), metal: solutionTabStatus(reports, LAB71, 'kernel.metal') };
}

function LabPart({ name, caption, cuTiers, mslTiers, cuMark, mslMark, mmMark }: {
  name: 'const' | 'byvalue';
  caption: ReactNode;
  cuTiers: Record<string, Tier>;
  mslTiers: Record<string, Tier>;
  cuMark: string;
  mslMark: string;
  mmMark: string;
}) {
  const cu = part(useLabFile(SOL + '/kernel.cu'), name);
  const metal = part(useLabFile(SOL + '/kernel.metal'), name);
  const mm = part(useLabFile(SOL + '/launch.mm'), name);
  const status = useStatuses();
  if (cu == null || metal == null || mm == null)
    return (
      <p className="muted">
        <Translated vi="Đang tải lời giải tham chiếu của lab 7.1.">Loading lab 7.1&rsquo;s reference solution.</Translated>
      </p>
    );
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu, tiers: cuTiers, status: status.cuda, highlight: linesWith(cu, cuMark) },
        { label: tr('Metal kernel', 'Kernel Metal'), code: metal, tiers: mslTiers, status: status.metal, highlight: linesWith(metal, mslMark) },
        { label: tr('Metal host', 'Mã host Metal'), code: mm, tiers: {}, status: status.metal, highlight: linesWith(mm, mmMark) },
      ]}
      caption={caption}
    />
  );
}

// ---------------------------------------------------------------- explorer: one address or many
const PATTERNS: { id: Pattern; label: string; code: string }[] = [
  {
    id: 'uniform',
    get label() {
      return tr('filter tap', 'tap của bộ lọc');
    },
    code: 'F[fRow][fCol]',
  },
  {
    id: 'lane',
    get label() {
      return tr('one entry per lane', 'mỗi lane một mục');
    },
    code: 'T[lane]',
  },
  {
    id: 'laneGroup',
    get label() {
      return tr('one per 8 lanes', 'mỗi 8 lane một mục');
    },
    code: 'T[lane / 8]',
  },
  {
    id: 'codebook',
    get label() {
      return tr('codebook', 'codebook');
    },
    code: 'codebook[code[lane]]',
  },
];
const CODEBOOK_SIZES = [
  {
    k: 4,
    get label() {
      return tr('4 (2-bit codes)', '4 (mã 2-bit)');
    },
  },
  {
    k: 16,
    get label() {
      return tr('16 (4-bit)', '16 (mã 4-bit)');
    },
  },
  {
    k: 256,
    get label() {
      return tr('256 (8-bit)', '256 (mã 8-bit)');
    },
  },
];

function OneAddressOrMany() {
  const [pattern, setPattern] = useState<Pattern>('uniform');
  const [warp, setWarp] = useState(32);
  const [k, setK] = useState(16);
  const [seed, setSeed] = useState(DEMO_SEED);
  const [tap, setTap] = useState(0);
  const taps = filterTrace(2);
  const t = taps[tap];
  const addresses = warpAddresses(pattern, { warpSize: warp, tableSize: k, groupSize: 8, seed, entry: t.address });
  const tableSize = pattern === 'uniform' ? taps.length : pattern === 'codebook' ? k : warp;
  const rounds = replaySchedule(addresses);
  const distinct = distinctCount(addresses);
  const cost = nvidiaCost(addresses);
  const amd = amdScalarEligible(addresses);
  const order = new Map(rounds.map((r, i) => [r.address, i + 1]));
  const code = pattern === 'uniform' ? `F[${t.fRow}][${t.fCol}]` : PATTERNS.find((x) => x.id === pattern)!.code;
  return (
    <Bench
      title={tr('One address or many', 'Một địa chỉ hay nhiều')}
      task={tr(
        "Pick a read pattern for one warp and count what each vendor's documented rule makes of it. Then reseed the codebook a few times.",
        'Chọn một kiểu đọc cho một warp và đếm xem quy tắc trong tài liệu của từng hãng biến nó thành gì. Rồi đổi seed cho codebook vài lần.',
      )}
      foot={
        <Translated
          vi={
            <>
              NVIDIA: một lần đọc bộ nhớ đệm hằng (constant cache) cho mỗi địa chỉ phân biệt trong warp (ch07-facts A3).
              AMD: bộ nhớ đệm vô hướng (scalar cache) chỉ phục vụ một lần đọc <code>__constant__</code> khi mọi lane đọc
              cùng một địa chỉ (B1). Apple không ghi tài liệu về constant cache nào, nên không có quy tắc Apple nào để áp
              dụng (C2). Các con số phía trên bảng là thứ tự mà hình này phục vụ các mục; không hãng nào ghi tài liệu về thứ
              tự.
            </>
          }
        >
          <>
            NVIDIA: one constant-cache read per distinct address in the warp (ch07-facts A3). AMD: the scalar cache serves
            a <code>__constant__</code> read only when every lane reads the same address (B1). Apple documents no constant
            cache, so there is no Apple rule to apply (C2). The numbers over the table are the order in which this
            picture serves the entries; no vendor documents an order.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          <Translated vi="Kiểu đọc">Read pattern</Translated>
          <span className="seg" data-control="pattern">
            {PATTERNS.map((x) => (
              <button key={x.id} aria-pressed={pattern === x.id} data-pattern={x.id} onClick={() => setPattern(x.id)}>
                {x.label}
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          <Translated vi="Độ rộng warp">Warp width</Translated>
          <span className="seg" data-control="warp">
            {[32, 64].map((w) => (
              <button key={w} aria-pressed={warp === w} onClick={() => setWarp(w)}>
                {w === 32 ? tr('32 (NVIDIA; AMD wave32)', '32 (NVIDIA; AMD ở chế độ wave32)') : tr('64 (AMD wave64)', '64 (AMD ở chế độ wave64)')}
              </button>
            ))}
          </span>
        </label>
      </div>
      {pattern === 'codebook' ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <label className="field">
            <Translated vi="Số mục của codebook">Codebook entries</Translated>
            <span className="seg" data-control="entries">
              {CODEBOOK_SIZES.map((c) => (
                <button key={c.k} aria-pressed={k === c.k} onClick={() => setK(c.k)}>{c.label}</button>
              ))}
            </span>
          </label>
          <button className="quiet" onClick={() => setSeed((s) => s + 1)} data-reseed>
            <Translated vi="Mã ngẫu nhiên mới">New random codes</Translated>
          </button>
          <span className="muted ui" data-seed={seed}><Translated vi={<>seed {seed}</>}>seed {seed}</Translated></span>
        </div>
      ) : null}
      {pattern === 'uniform' ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <button className="quiet" onClick={() => setTap((x) => (x + 1) % taps.length)} data-next-tap>
            <Translated vi="Tap tiếp theo">Next tap</Translated>
          </button>
          <span className="muted ui">
            <Translated vi={<>tap {tap + 1} trên {taps.length} trong một bộ lọc 5 × 5</>}>
              tap {tap + 1} of {taps.length} in a 5 × 5 filter
            </Translated>
          </span>
        </div>
      ) : null}
      <svg viewBox="0 0 720 196" role="img" aria-label={tr('Lanes of one warp and the table entries they read', 'Các lane của một warp và các mục trong bảng mà chúng đọc')} style={{ width: '100%', maxWidth: 900, minWidth: 520, display: 'block' }}>
        <WarpTable
          addresses={addresses}
          tableSize={tableSize}
          gap={100}
          lane={() => 'asking'}
          entry={(e) => (order.has(e) ? 'asked' : 'idle')}
          rounds={distinct > 1 ? order : undefined}
          entryLabel={(e) => (tableSize <= 32 ? String(e) : null)}
          laneLabel={tr(`one warp: ${warp} lanes, each reading ${code}`, `một warp: ${warp} lane, mỗi lane đọc ${code}`)}
          tableLabel={
            pattern === 'uniform' ? tr('F: a 5 × 5 filter, row by row', 'F: một bộ lọc 5 × 5, từng hàng một') : pattern === 'codebook' ? tr(`a ${k}-entry codebook`, `một codebook ${k} mục`) : tr(`T: one entry per lane (${warp})`, `T: mỗi lane một mục (${warp})`)
          }
        />
      </svg>
      <div className="cols" style={{ marginTop: '0.8rem', gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div className="panel" data-pattern-shown={pattern} data-distinct={distinct}>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr>
                <td><Translated vi="Số địa chỉ phân biệt">Distinct addresses</Translated></td>
                <td className="num" data-cell="distinct"><b>{distinct}</b></td>
              </tr>
              <tr>
                <td><Translated vi="NVIDIA: số lần đọc constant cache tuần tự">NVIDIA: serialised constant-cache reads</Translated></td>
                <td className="num" data-cell="nvidia">{cost}</td>
              </tr>
              <tr>
                <td><Translated vi="AMD: scalar cache có phục vụ được không?">AMD: can the scalar cache serve it?</Translated></td>
                <td className="num" data-cell="amd">
                  {amd ? <span className="chip ok"><Translated vi="có, đồng nhất">yes, uniform</Translated></span> : <span className="chip fail"><Translated vi="không, tuần tự">no, serialised</Translated></span>}
                </td>
              </tr>
              <tr>
                <td>
                  <Translated vi="Apple">Apple</Translated>
                </td>
                <td className="num" data-cell="apple">
                  <span className="muted"><Translated vi="không có tài liệu">not documented</Translated></span>
                </td>
              </tr>
              {pattern === 'codebook' ? (
                <tr>
                  <td>
                    <Translated vi={<>Kỳ vọng với mã ngẫu nhiên: {k}(1 − (1 − 1/{k})<sup>{warp}</sup>)</>}>
                      Expected for random codes: {k}(1 − (1 − 1/{k})<sup>{warp}</sup>)
                    </Translated>
                  </td>
                  <td className="num" data-cell="expected">{fmt(expectedDistinct(k, warp), 4)}</td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <p style={{ margin: 0 }} data-summary>
            {distinct === 1
              ? tr('One address: one read answers the whole warp. On NVIDIA this is the case that can be as fast as a register.', 'Một địa chỉ: một lần đọc trả lời cả warp. Trên NVIDIA, đây là trường hợp có thể nhanh ngang một thanh ghi.')
              : tr(`${distinct} addresses: under NVIDIA's rule the warp's read costs ${cost} constant-cache reads, one after another, ${fmt(cost)} times the uniform case.`, `${distinct} địa chỉ: theo quy tắc của NVIDIA, lần đọc của warp tốn ${cost} lần đọc constant cache, lần lượt từng lần, gấp ${fmt(cost)} lần trường hợp đồng nhất.`)}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- classification: where would you put it?
type Bin = 'arg' | 'const' | 'global' | 'shared';
const BINS: { id: Bin; label: string }[] = [
  {
    id: 'arg',
    get label() {
      return tr('Kernel argument by value', 'Đối số kernel truyền theo giá trị');
    },
  },
  {
    id: 'const',
    get label() {
      return tr('__constant__ table', 'Bảng __constant__');
    },
  },
  {
    id: 'global',
    get label() {
      return tr('Ordinary global memory', 'Bộ nhớ toàn cục thông thường');
    },
  },
  {
    id: 'shared',
    get label() {
      return tr('Staged into shared memory', 'Nạp trước vào bộ nhớ chia sẻ');
    },
  },
];
interface Item {
  id: string;
  what: ReactNode;
  accept: Bin[];
  why: Record<Bin, ReactNode>;
  src: string;
}
const ITEMS: Item[] = [
  {
    id: 'sampling',
    what: (
      <Translated vi="Tham số lấy mẫu cho một bước decode: temperature, top-k, top-p và một seed ngẫu nhiên (16 byte), được mọi luồng đọc.">
        Sampling parameters for one decode step: temperature, top-k, top-p and a random seed (16 bytes), read by every
        thread.
      </Translated>
    ),
    accept: ['arg'],
    why: {
      arg: (
        <Translated vi="Vài byte, giống nhau cho mọi luồng, mới ở mỗi bước. Truyền chúng cùng lần khởi chạy sẽ sao chép chúng giúp bạn, và dù sao CUDA cũng đặt đối số kernel trong bộ nhớ hằng.">
          A few bytes, the same for every thread, new at every step. Passing them with the launch copies them for you,
          and CUDA places kernel arguments in constant memory anyway.
        </Translated>
      ),
      const: (
        <Translated vi="Cách này chạy được, nhưng chúng đổi ở mỗi bước, nên bạn sẽ phải sao chép chúng vào symbol trước mỗi lần khởi chạy để có được thứ mà một đối số truyền theo giá trị đã cho sẵn, trong cùng bộ nhớ hằng trên CUDA.">
          It would work, but they change every step, so you would copy them to the symbol before every launch to get
          what a by-value argument already gives you, in the same constant memory on CUDA.
        </Translated>
      ),
      global: (
        <Translated vi="Cách này chạy được, nhưng tốn một buffer và một đường nạp cho vài giá trị mà mọi luồng đều cần. Giá trị nhỏ và đồng nhất chính là thứ đối số sinh ra để chở.">
          It works, but costs a buffer and a load path for a few values that every thread needs. Small uniform values
          are what arguments are for.
        </Translated>
      ),
      shared: (
        <Translated vi="Bộ nhớ chia sẻ thuộc về một khối và phải được chính các luồng của khối đó điền từ nơi khác trước. Ở đây không có gì được tái sử dụng mà một đối số chưa đưa sẵn tới mọi luồng.">
          Shared memory belongs to one block and is filled by its own threads from somewhere else first. Nothing here is
          reused that an argument does not already hand to every thread.
        </Translated>
      ),
    },
    src: 'A4, D4',
  },
  {
    id: 'filter',
    what: (
      <Translated vi="Một bộ lọc ảnh 5 × 5 (100 byte), được mọi luồng đọc dưới dạng F[fRow][fCol], theo cùng một thứ tự.">
        A 5 × 5 image filter (100 bytes), read as F[fRow][fCol] by every thread, in the same order.
      </Translated>
    ),
    accept: ['const', 'arg'],
    why: {
      arg: (
        <Translated vi="Phương án thay thế đầu tiên của sách (tr. 172): một struct truyền theo giá trị, mỗi lần khởi chạy một bộ lọc mới và không cần sao chép symbol. Trên Metal là một constant Filter& được điền bằng setBytes.">
          {"The book's first alternative (p. 172): a struct passed by value, a new filter per launch and no symbol copy. On Metal, a constant Filter& filled with setBytes."}
        </Translated>
      ),
      const: (
        <Translated vi="Câu trả lời chính của sách (tr. 169). Mọi lane đọc cùng một địa chỉ, trường hợp tốt nhất của constant cache, và scalar cache của AMD cũng phục vụ được.">
          {"The book's main answer (p. 169). Every lane reads the same address, the constant cache's best case, and AMD's scalar cache serves it too."}
        </Translated>
      ),
      global: (
        <Translated vi="Kết quả đúng, nhưng mọi luồng nạp mọi trọng số qua đường thông thường: 0.25 FLOP/B của kernel cơ bản, trước khi tính đến bộ nhớ đệm.">
          {"Correct results, but every thread loads every weight through the ordinary path: the basic kernel's 0.25 FLOP/B before caches."}
        </Translated>
      ),
      shared: (
        <Translated vi="Có thể, nhưng mỗi khối sẽ phải nạp cùng 25 trọng số trước, trong khi bộ nhớ hằng đã cho mọi warp một lần đọc một-địa-chỉ.">
          Possible, but each block would first load the same 25 weights, and constant memory already gives every warp a
          one-address read.
        </Translated>
      ),
    },
    src: 'A3, A4, B1, C1',
  },
  {
    id: 'codebook',
    what: (
      <Translated vi="Một codebook int4 16 mục (64 byte), được đọc dưới dạng codebook[code], trong đó mỗi luồng giải mã mã trọng số của riêng nó.">
        {"A 16-entry int4 codebook (64 bytes), read as codebook[code] where each thread decodes its own weight's code."}
      </Translated>
    ),
    accept: ['global', 'shared'],
    why: {
      arg: (
        <Translated vi="Nó vừa, nhưng mỗi luồng đánh chỉ số nó bằng mã riêng. Trên CUDA, đối số nằm trong bộ nhớ hằng, nên đây lại là trường hợp tệ nhất của constant cache.">
          {'It fits, but each thread indexes it with its own code. On CUDA an argument lives in constant memory, so this is the constant cache\'s worst case again.'}
        </Translated>
      ),
      const: (
        <Translated vi="Nó vừa dễ dàng, điều khiến nó hấp dẫn. Nhưng một warp hỏi khoảng 14 trong 16 mục cùng lúc: NVIDIA tuần tự hóa chúng và scalar cache của AMD không phục vụ một lần đọc không đồng nhất.">
          {'It fits easily, which makes it tempting. But a warp asks for about 14 of the 16 entries at once: NVIDIA serialises them and AMD\'s scalar cache does not serve a read that is not uniform.'}
        </Translated>
      ),
      global: (
        <Translated vi="Cách llama.cpp làm trên CUDA và HIP: các bảng static const __device__. Lần nạp thông thường không có quy tắc một-địa-chỉ; chúng nhanh đến đâu là tùy bộ nhớ đệm, và đó là chuyện phải đo.">
          What llama.cpp does on CUDA and HIP: static const __device__ tables. Ordinary loads have no one-address rule;
          how fast they are is up to the caches, which is a measurement.
        </Translated>
      ),
      shared: (
        <Translated vi="Cũng tốt: 64 byte được nạp trước một lần mỗi khối, rồi đọc trên chip. 16 mục nằm ở 16 bank khác nhau, nên các lane chọn cùng một mục dùng chung một word còn các lane chọn mục khác nhau thì không xung đột (quy tắc bank của chương 6; lập luận của khóa học này).">
          {'Also good: 64 bytes staged once per block, then read on chip. The 16 entries sit in 16 different banks, so lanes that pick the same entry share a word and lanes that pick different ones do not conflict (chapter 6\'s bank rule; our reasoning).'}
        </Translated>
      ),
    },
    src: 'A3, B1, D4; infra A10',
  },
  {
    id: 'rope',
    what: (
      <Translated vi="Một bảng rotary embedding gồm 8192 vị trí × 64 float (2 MiB).">
        A rotary-embedding table of 8192 positions × 64 floats (2 MiB).
      </Translated>
    ),
    accept: ['global'],
    why: {
      arg: (
        <Translated vi="2 MiB không thể là một đối số: giới hạn của CUDA là 32,764 byte và setBytes của Metal nhận ít hơn 4 KB.">
          {"2 MiB cannot be an argument: CUDA's limit is 32,764 bytes and Metal's setBytes takes less than 4 KB."}
        </Translated>
      ),
      const: (
        <Translated vi="2 MiB gấp 32 lần 64 KB của bộ nhớ hằng. Nó không vừa.">
          2 MiB is 32 times the 64 KB of constant memory. It does not fit.
        </Translated>
      ),
      global: (
        <Translated vi="Quá lớn cho mọi nơi khác, nên nó nằm trong bộ nhớ toàn cục và được đọc như mọi tensor khác.">
          Far too big for anything else, so it lives in global memory and is read like any other tensor.
        </Translated>
      ),
      shared: (
        <Translated vi="Một khối có thể nạp trước vài hàng nó cần về sau, nhưng bản thân bảng phải nằm trong bộ nhớ toàn cục trước, và nó lớn gấp nhiều lần bộ nhớ chia sẻ của bất kỳ khối nào.">
          {'A block could stage the few rows it needs later, but the table itself has to live in global memory first, and it is many times larger than any block\'s shared memory.'}
        </Translated>
      ),
    },
    src: 'A4, C1; ch05-facts A4',
  },
  {
    id: 'gamma',
    what: (
      <Translated vi="Trọng số LayerNorm cho kích thước ẩn 4096 (16 KB), luồng j đọc gamma[j].">
        LayerNorm weights for a hidden size of 4096 (16 KB), read as gamma[j] by thread j.
      </Translated>
    ),
    accept: ['global'],
    why: {
      arg: (
        <Translated vi="16 KB dưới giới hạn đối số 32,764 byte của CUDA nhưng vượt giới hạn setBytes của Metal, và mỗi luồng đọc một phần tử khác nhau: trong bộ nhớ hằng, đó là mỗi lane một lần đọc tuần tự.">
          {"16 KB is under CUDA's 32,764-byte argument limit but over Metal's setBytes limit, and each thread reads a different element: in constant memory that is one serialised read per lane."}
        </Translated>
      ),
      const: (
        <Translated vi="Nó vừa trong 64 KB, điều khiến nó hấp dẫn. Nhưng luồng j đọc gamma[j], nên một warp hỏi số địa chỉ bằng số lane của nó: trường hợp tệ nhất của constant cache.">
          {'It fits in 64 KB, which is what makes it tempting. But thread j reads gamma[j], so a warp asks for as many addresses as it has lanes: the constant cache\'s worst case.'}
        </Translated>
      ),
      global: (
        <Translated vi="Các luồng liên tiếp đọc các phần tử liên tiếp là kiểu truy cập được hợp nhất của chương 6, một trong những việc rẻ nhất mà bộ nhớ toàn cục làm.">
          Consecutive threads reading consecutive elements is the coalesced pattern of chapter 6, one of the cheapest
          things global memory does.
        </Translated>
      ),
      shared: (
        <Translated vi="Chỉ đáng khi một khối đọc cùng các trọng số nhiều lần, chẳng hạn qua nhiều hàng, và nó tốn một lần sao chép và một rào chắn. Dù cách nào, trọng số vẫn nằm trong bộ nhớ toàn cục.">
          Worth it only if a block reads the same weights many times, for instance across several rows, and it costs a
          copy and a barrier. The weights live in global memory either way.
        </Translated>
      ),
    },
    src: 'A3, A4, C1',
  },
  {
    id: 'shapes',
    what: (
      <Translated vi="Hình dạng và bước nhảy của tensor cho lần khởi chạy này: một tá số nguyên, giống nhau cho mọi luồng.">
        Tensor shapes and strides for this launch: a dozen ints, the same for every thread.
      </Translated>
    ),
    accept: ['arg'],
    why: {
      arg: (
        <Translated vi="Cách llama.cpp truyền chúng: một const struct theo giá trị trên CUDA, một tham chiếu constant struct được điền bằng setBytes trên Metal.">
          How llama.cpp passes them: a const struct by value on CUDA, a constant struct reference filled with setBytes on
          Metal.
        </Translated>
      ),
      const: (
        <Translated vi="Cách này chạy được, nhưng chúng thuộc về lần khởi chạy này; một đối số theo giá trị được sao chép cho mỗi lần khởi chạy giúp bạn và đã nằm sẵn trong bộ nhớ hằng trên CUDA.">
          It works, but they belong to this launch; a by-value argument is copied per launch for you and already lands in
          constant memory on CUDA.
        </Translated>
      ),
      global: (
        <Translated vi="Cách này chạy được, nhưng tốn một buffer và một đường nạp cho một tá giá trị mà mọi luồng cần. Đối số sinh ra cho việc này.">
          It works, but costs a buffer and a load path for a dozen values every thread needs. Arguments exist for this.
        </Translated>
      ),
      shared: (
        <Translated vi="Mọi luồng đều cần chúng, nên mọi khối sẽ phải điền chúng từ nơi khác trước. Một đối số đã tới được mọi luồng.">
          Every thread needs them, so every block would fill them from somewhere else first. An argument already reaches
          every thread.
        </Translated>
      ),
    },
    src: 'A4, D4',
  },
];
const WHERE_ID = 'ch07/7.4/c-where';

function WhereWouldYouPutIt() {
  const { progress, answer } = useProgress();
  const saved = progress.answers[WHERE_ID]?.value;
  const [place, setPlace] = useState<Record<string, Bin>>(() =>
    saved && typeof saved === 'object' ? { ...(saved as Record<string, Bin>) } : {},
  );
  const pick = (id: string, b: Bin) => {
    const next = { ...place, [id]: b };
    setPlace(next);
    const all = ITEMS.every((it) => next[it.id]);
    answer(WHERE_ID, next, all ? ITEMS.every((it) => it.accept.includes(next[it.id])) : undefined);
  };
  const placed = ITEMS.filter((it) => place[it.id]).length;
  const good = ITEMS.filter((it) => place[it.id] && it.accept.includes(place[it.id])).length;
  return (
    <Bench
      title={tr('Where would you put it?', 'Bạn sẽ đặt nó ở đâu?')}
      task={tr(
        'Six small things an inference engine reads in its kernels. Give each a home; each card answers at once.',
        'Sáu thứ nhỏ mà một engine suy luận đọc trong các kernel của nó. Hãy chọn chỗ cho từng thứ; mỗi thẻ trả lời ngay.',
      )}
      foot={
        <Translated
          vi={
            <>
              Các mục và lập luận là của khóa học này, xây từ các sự thật của hãng được nêu trên mỗi thẻ
              (notes/today/ch07-facts.md trừ khi ghi khác). Hai thẻ có hai chỗ tốt. Lựa chọn của bạn được lưu cùng tiến độ
              học.
            </>
          }
        >
          <>
            The items and the reasoning are ours, built from the vendor facts named on each card (notes/today/ch07-facts.md
            unless stated). Two cards have two good homes. Your placements are kept with your progress.
          </>
        </Translated>
      }
    >
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(19rem, 1fr))' }}>
        {ITEMS.map((it) => {
          const b = place[it.id];
          const ok = b ? it.accept.includes(b) : undefined;
          return (
            <div key={it.id} className="panel" data-item={it.id} data-verdict={b ? (ok ? 'right' : 'wrong') : 'open'}>
              <p style={{ margin: 0 }}>{it.what}</p>
              <div className="seg" style={{ display: 'flex', flexWrap: 'wrap', marginTop: '0.5rem' }}>
                {BINS.map((x) => (
                  <button key={x.id} aria-pressed={b === x.id} data-bin={x.id} onClick={() => pick(it.id, x.id)}>
                    {x.label}
                  </button>
                ))}
              </div>
              {b ? (
                <p style={{ margin: '0.5rem 0 0' }} data-why>
                  {ok ? <span className="t-ok"><b><Translated vi="Một chỗ tốt.">A good home.</Translated></b></span> : <span className="t-fail"><b><Translated vi="Không phải chỗ tốt nhất.">Not the best home.</Translated></b></span>}{' '}
                  {it.why[b]}{' '}
                  {!ok ? (
                    <span className="muted">
                      <Translated vi={<>Tốt hơn: {it.accept.map((a) => BINS.find((x) => x.id === a)!.label.toLowerCase()).join(' hoặc ')}.</>}>
                        Better: {it.accept.map((a) => BINS.find((x) => x.id === a)!.label.toLowerCase()).join(' or ')}.
                      </Translated>
                    </span>
                  ) : null}{' '}
                  <span className="muted">({it.src})</span>
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <p className="status-line" data-where-result style={{ margin: 0 }}>
          {placed === ITEMS.length ? tr(`All six placed, ${good} of 6 in a good home.`, `Đã đặt cả sáu, ${good} trên 6 ở chỗ tốt.`) : tr(`${placed} of 6 placed.`, `Đã đặt ${placed} trên 6.`)}
        </p>
        <button className="quiet" onClick={() => setPlace({})} disabled={!placed}>
          <Translated vi="Làm lại từ đầu">Start again</Translated>
        </button>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- what the reader's device reports
function YourConstantMemory() {
  const { reports, loading } = useReports();
  // GPU backends only: realBackends() leaves out 'emu' but not the 'race' emulator.
  const backends = realBackends(reports).filter((b) => b !== 'race');
  if (loading) return null;
  const has = (r: Reports[string], key: string) => {
    const v = Number(r.device[key]);
    return isFinite(v) && v > 0;
  };
  const val = (r: Reports[string], key: string) => (has(r, key) ? tr(`${fmt(Number(r.device[key]) / 1024)} KB`, `${fmt(Number(r.device[key]) / 1024)} KB`) : tr('not in your device report', 'không có trong báo cáo thiết bị của bạn'));
  const missing = backends.some((b) => b !== 'metal' && (!has(reports[b], 'totalConstMem') || !has(reports[b], 'l2CacheSize')));
  return (
    <p data-yours>
      <b><Translated vi="Thiết bị của bạn.">Your devices.</Translated></b>{' '}
      {backends.length
        ? backends.map((b, i) => (
            <span key={b}>
              {i ? '; ' : ''}
              {String(reports[b].device.name ?? b)} ({b}):{' '}
              {b === 'metal'
                ? tr('Apple documents neither a constant-memory size nor an L2 size', 'Apple không ghi tài liệu về kích thước bộ nhớ hằng lẫn kích thước L2')
                : b === 'hip'
                  ? tr(`totalConstMem ${val(reports[b], 'totalConstMem')} (on AMD the largest single allocation, not a constant-memory size), L2 ${val(reports[b], 'l2CacheSize')}`, `totalConstMem ${val(reports[b], 'totalConstMem')} (trên AMD là vùng cấp phát đơn lớn nhất, không phải kích thước bộ nhớ hằng), L2 ${val(reports[b], 'l2CacheSize')}`)
                  : tr(`constant memory ${val(reports[b], 'totalConstMem')}, L2 ${val(reports[b], 'l2CacheSize')}`, `bộ nhớ hằng ${val(reports[b], 'totalConstMem')}, L2 ${val(reports[b], 'l2CacheSize')}`)}
            </span>
          ))
        : tr('no device report on this machine yet (run python3 labs/pmpp.py doctor)', 'máy này chưa có báo cáo thiết bị (chạy python3 labs/pmpp.py doctor)')}
      .
      {missing ? (
        <Translated
          vi={
            <>
              {' '}Một chương trình của riêng bạn có thể đọc cả hai từ <code>cudaGetDeviceProperties</code> (<code>hipGetDeviceProperties</code>{' '}
              trên HIP, nơi <code>l2CacheSize</code> bằng 0 nghĩa là không được báo cáo), dưới dạng <code>totalConstMem</code> và{' '}
              <code>l2CacheSize</code>.
            </>
          }
        >
          <>
            {' '}A program of your own can read both from <code>cudaGetDeviceProperties</code> (<code>hipGetDeviceProperties</code>{' '}
            on HIP, where an <code>l2CacheSize</code> of 0 means not reported), as <code>totalConstMem</code> and{' '}
            <code>l2CacheSize</code>.
          </>
        </Translated>
      ) : null}
    </p>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS74: Record<string, Question> = {
  'ch07/7.4/q-extern': {
    kind: 'choice',
    id: 'ch07/7.4/q-extern',
    prompt: (
      <Translated
        vi={
          <>
            Kernel của bạn nằm trong <code>conv.cu</code>; mảng bộ lọc được định nghĩa trong <code>filters.cu</code> cạnh
            mã host điền nó. Như sách gợi ý ở tr. 169, <code>conv.cu</code> khai báo{' '}
            <code>extern __constant__ float F_c[225];</code>. Chuyện gì xảy ra với các cờ mặc định của nvcc?
          </>
        }
      >
        <>
          Your kernel is in <code>conv.cu</code>; the filter array is defined in <code>filters.cu</code> next to the host
          code that fills it. As the book suggests on p. 169, <code>conv.cu</code> declares{' '}
          <code>extern __constant__ float F_c[225];</code>. What happens with nvcc&rsquo;s default flags?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Nó build và chạy: trình liên kết giải quyết một symbol extern __constant__ giữa các tệp như mọi biến toàn cục.">
            It builds and runs: the linker resolves an extern __constant__ symbol across files like any global.
          </Translated>
        ),
        why: (
          <Translated vi="Không phải ở chế độ mặc định. Tài liệu hướng dẫn của NVIDIA nêu đúng trường hợp này là một lỗi trong chế độ biên dịch toàn chương trình.">
            {'Not in the default mode. NVIDIA\'s guide gives exactly this as an error in whole-program compilation.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó không build được: theo mặc định, mã device không thể tham chiếu tới một biến được định nghĩa trong tệp khác.">
            It does not build: by default, device code cannot refer to a variable defined in another file.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó build được, nhưng kernel đọc ra số không: mỗi tệp có F_c riêng, và chỉ một bản được điền.">
            It builds, but the kernel reads zeros: each file gets its own F_c, and only one copy is filled.
          </Translated>
        ),
        why: (
          <Translated vi="Một phỏng đoán hợp lý, nhưng kết quả được ghi trong tài liệu là một lỗi biên dịch, không phải một bản sao âm thầm.">
            A plausible guess, but the documented outcome is a compile error, not a silent copy.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó build được với hipcc, vốn mặc định liên kết mã device giữa các tệp, nhưng nvcc từ chối nó.">
            It builds with hipcc, whose default links device code across files, but nvcc rejects it.
          </Translated>
        ),
        why: (
          <Translated vi="Mặc định của HIP cũng vậy: mã device trong mỗi tệp là khép kín, và liên kết giữa các tệp cần -fgpu-rdc.">
            {'HIP\'s default is the same: device code in each file is self-contained, and linking across files needs -fgpu-rdc.'}
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            nvcc mặc định biên dịch ở chế độ toàn chương trình: mã device không thể tham chiếu tới bất cứ thứ gì trong tệp
            khác, và <code>extern</code> trên một biến <code>__constant__</code> là một lỗi. Biên dịch tách rời sửa được điều
            đó (<code>-rdc=true</code>; trên HIP là <code>-fgpu-rdc</code>), đổi lại build phức tạp hơn. Lab 7.1 né câu hỏi
            này: symbol, kernel và lời gọi <code>cudaMemcpyToSymbol</code> đều nằm trong <code>kernel.cu</code>, còn{' '}
            <code>main.cpp</code> chỉ chuyển giao một mảng host thông thường.
          </p>
        }
      >
        <p>
          nvcc compiles in whole-program mode by default: device code cannot refer to anything in another file, and{' '}
          <code>extern</code> on a <code>__constant__</code> variable is an error. Separate compilation fixes it (
          <code>-rdc=true</code>; on HIP, <code>-fgpu-rdc</code>), at a cost in build complexity. Lab 7.1 sidesteps the question: the symbol, the kernel and the{' '}
          <code>cudaMemcpyToSymbol</code> call are all in <code>kernel.cu</code>, and <code>main.cpp</code> hands
          over a plain host array.
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.4, p. 169; notes/today/ch07-facts.md A1, B3',
    vi: { source: 'ch. 7 §7.4, tr. 169; notes/today/ch07-facts.md A1, B3' },
  },
  'ch07/7.4/q-device-vs-constant': {
    kind: 'choice',
    id: 'ch07/7.4/q-device-vs-constant',
    prompt: (
      <Translated
        vi={
          <>
            Trong lab 7.1, đổi <code>__constant__ float F_c[...]</code> thành <code>__device__ float F_c[...]</code> và
            không đổi gì khác. Chuyện gì xảy ra?
          </>
        }
      >
        <>
          In lab 7.1, change <code>__constant__ float F_c[...]</code> to <code>__device__ float F_c[...]</code> and
          change nothing else. What happens?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Kết quả như cũ: F_c trở thành một biến toàn cục thông thường, được đọc qua các bộ nhớ đệm bình thường.">
            Same results: F_c becomes an ordinary global variable, read through the normal caches.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Nó không build được: cudaMemcpyToSymbol chỉ nhận symbol __constant__ làm đích.">
            It does not build: cudaMemcpyToSymbol accepts only __constant__ symbols as its destination.
          </Translated>
        ),
        why: (
          <Translated vi="Nó nhận mọi symbol trong bộ nhớ toàn cục hoặc bộ nhớ hằng; chính ví dụ của NVIDIA sao chép vào một biến __device__.">
            {'It accepts any symbol in global or constant memory; NVIDIA\'s own example copies to a __device__ variable.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Nó build được, nhưng lần sao chép âm thầm không làm gì, nên kernel đọc ra số không và mọi đầu ra đều là 0.">
            It builds, but the copy silently does nothing, so the kernel reads zeros and every output is 0.
          </Translated>
        ),
        why: (
          <Translated vi="Lần sao chép hoạt động y hệt với một symbol __device__.">
            The copy works the same for a __device__ symbol.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không có gì thay đổi: chính lời gọi sao chép, không phải qualifier, khiến mảng thành chỉ-đọc và được constant cache lưu.">
            Nothing changes: the copy call, not the qualifier, makes the array read-only and constant-cached.
          </Translated>
        ),
        why: (
          <Translated vi="Ngược lại mới đúng. Sách mô tả lần sao chép là để báo cho runtime biết dữ liệu sẽ không đổi, nhưng chính qualifier __constant__ mới đặt mảng vào bộ nhớ hằng và khiến nó chỉ-đọc trong kernel.">
            The other way round. The book describes the copy as telling the runtime the data will not change, but it is
            the __constant__ qualifier that places the array in constant memory and makes it read-only in kernels.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            <code>cudaMemcpyToSymbol</code> chỉ là cách host ghi một biến device theo tên. Biến nằm ở bộ nhớ nào, và kernel
            có được ghi nó hay không, là do qualifier của nó quyết định. Với <code>__device__</code>{' '}
            kernel vẫn đúng, mảng trở nên ghi được bởi kernel, và các lần đọc chỉ mất constant cache, chính là thứ lab 7.1
            đo.
          </p>
        }
      >
        <p>
          <code>cudaMemcpyToSymbol</code> is just how the host writes a device variable by name. Which memory the
          variable lives in, and whether kernels may write it, is decided by its qualifier. With <code>__device__</code>{' '}
          the kernel is correct, the array becomes writable by kernels, and the reads lose only the constant cache, which
          is the thing lab 7.1 measures.
        </p>
      </Translated>
    ),
    source: 'ch. 7 §7.4, p. 169; notes/today/ch07-facts.md A2',
    vi: { source: 'ch. 7 §7.4, tr. 169; notes/today/ch07-facts.md A2' },
  },
  'ch07/7.4/q-cache-fit': {
    kind: 'numeric',
    id: 'ch07/7.4/q-cache-fit',
    prompt: (
      <Translated
        vi={
          <>
            NVIDIA ghi trong tài liệu một working set của constant cache là 8 KB mỗi SM. Độ rộng bộ lọc lẻ m lớn nhất mà m
            × m trọng số float vừa trong đó là bao nhiêu?
          </>
        }
      >
        <>
          NVIDIA documents a constant-cache working set of 8 KB per SM. What is the largest odd filter width m whose m × m
          float weights fit in it?
        </>
      </Translated>
    ),
    answer: 45,
    tol: 0,
    explain: (
      <Translated
        vi={
          <p>
            8 KB là 2048 float. 45 × 45 = 2025 vừa; 47 × 47 = 2209 thì không. Bản thân bộ nhớ hằng, 64 KB, chứa được bộ lọc
            rộng tới 127, nhưng vừa không gian lưu trữ không giống với nằm lại trong bộ nhớ đệm. Bán kính 7 của sách (225
            float, 900 byte) nằm sâu bên trong cả hai.
          </p>
        }
      >
        <p>
          8 KB is 2048 floats. 45 × 45 = 2025 fits; 47 × 47 = 2209 does not. Constant memory itself, 64 KB, would hold
          filters up to 127 wide, but fitting the space is not the same as staying in the cache. The book&rsquo;s radius
          7 (225 floats, 900 bytes) is far inside both.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch07-facts.md A3; our arithmetic',
    vi: { source: 'notes/today/ch07-facts.md A3; phép tính của khóa học này' },
  },
  'ch07/7.4/q-codebook': {
    kind: 'choice',
    id: 'ch07/7.4/q-codebook',
    prompt: (
      <Translated
        vi={
          <>
            llama.cpp giữ các codebook lượng tử hóa của nó trên CUDA và HIP dưới dạng mảng <code>static const __device__</code>,
            trong bộ nhớ device thông thường, không phải trong <code>__constant__</code>. Một lý do hợp lý là gì?
          </>
        }
      >
        <>
          llama.cpp keeps its quantisation codebooks on CUDA and HIP as <code>static const __device__</code> arrays, in
          ordinary device memory, not in <code>__constant__</code>. What is a plausible reason?
        </>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Các lane của một warp đọc các mục khác nhau, mà constant cache của cả hai hãng phục vụ tốt nhất khi chỉ có một mục.">
            {'The lanes of a warp read different entries, and both vendors\' constant caches serve one entry best.'}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Các bảng quá lớn: gộp lại, các codebook vượt quá 64 KB mà bộ nhớ hằng chứa được.">
            The tables are too big: together the codebooks exceed the 64 KB that constant memory holds.
          </Translated>
        ),
        why: (
          <Translated vi="Nhiều bảng rất nhỏ: kvalues_iq4nl có 16 mục. Kích thước không phải lý do.">
            Several are tiny: kvalues_iq4nl has 16 entries. Size is not the reason.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="HIP không có qualifier __constant__, và llama.cpp biên dịch cùng mã nguồn cho CUDA và HIP.">
            HIP has no __constant__ qualifier, and llama.cpp compiles the same source for CUDA and HIP.
          </Translated>
        ),
        why: (
          <Translated vi="Nó có, với cùng cách viết, được điền bằng hipMemcpyToSymbol.">
            It has, with the same spelling, filled by hipMemcpyToSymbol.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một mảng __constant__ không thể được khởi tạo lúc biên dịch, nên mọi bảng đều cần một lần sao chép.">
            A __constant__ array cannot be initialised at compile time, so every table would need a copy.
          </Translated>
        ),
        why: (
          <Translated vi="Một mảng __constant__ có khởi tạo được phép trên CUDA và HIP.">
            An initialised __constant__ array is allowed on CUDA and HIP.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Mã nguồn không nói lý do; đây là cách đọc của khóa học này. Kiểu truy cập ngược hẳn với bộ lọc: chỉ số của bộ
            lọc giống nhau cho mọi lane, chỉ số của codebook khác nhau ở gần như mọi lane, và constant cache của cả hai hãng
            được xây cho trường hợp đầu. NVIDIA tuần tự hóa các địa chỉ phân biệt, còn scalar cache của AMD không phục vụ
            một lần đọc không đồng nhất. Trình khám phá ở trên cho thấy con số.
          </p>
        }
      >
        <p>
          The source does not say why; this is our reading of it. The pattern is the opposite of the filter&rsquo;s: the
          filter&rsquo;s index is the same for every lane, a codebook&rsquo;s is different for nearly every lane, and both
          vendors&rsquo; constant caches are built for the first. NVIDIA serialises the distinct addresses, and AMD&rsquo;s
          scalar cache does not serve a read that is not uniform. The explorer above shows the count.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch07-facts.md A3, B1, D4; infra-facts B5', // i18n-ignore: citation only
  },
  'ch07/7.4/q-metal': {
    kind: 'choice',
    id: 'ch07/7.4/q-metal',
    prompt: (
      <Translated
        vi={
          <>
            Trên Metal, thứ gì đóng vai <code>__constant__</code> cộng <code>cudaMemcpyToSymbol</code> cho một bộ lọc được
            chọn lúc chạy?
          </>
        }
      >
        <>On Metal, what plays the part of <code>__constant__</code> plus <code>cudaMemcpyToSymbol</code> for a filter chosen at run time?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Một buffer do host điền, gắn vào một đối số kernel khai báo là constant float*.">
            A buffer the host fills, bound to a kernel argument declared constant float*.
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Một mảng constant ở phạm vi chương trình, mà host ghi đè sau khi thư viện được biên dịch.">
            A program-scope constant array, which the host overwrites once the library is compiled.
          </Translated>
        ),
        why: (
          <Translated vi="Một biến constant ở phạm vi chương trình phải được khởi tạo trong mã nguồn; Metal không có sao chép symbol. Đó là bản tương ứng của việc viết cứng.">
            A program-scope constant variable must be initialised in the source; Metal has no symbol copy. That is the
            counterpart of hard-coding.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một mảng function constant, mà host đặt giá trị khi tạo pipeline.">
            An array of function constants, whose values the host sets as it creates the pipeline.
          </Translated>
        ),
        why: (
          <Translated vi="Function constant chỉ có thể là vô hướng hoặc vector; một mảng là lỗi biên dịch.">
            Function constants can only be scalars or vectors; an array is a compile error.
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Một mảng threadgroup, mà host điền qua encoder trước mỗi lần dispatch.">
            A threadgroup array, which the host fills through the encoder before each dispatch.
          </Translated>
        ),
        why: (
          <Translated vi="Bộ nhớ threadgroup thuộc về một threadgroup và được chính các luồng của nó ghi; host không thể điền nó.">
            Threadgroup memory belongs to one threadgroup and is written by its own threads; the host cannot fill it.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Các tệp Metal của lab 7.1 làm việc này và thêm một cách nữa: <code>conv_const</code> nhận buffer bộ lọc dưới
            dạng <code>constant float*</code>, và <code>conv_by_value</code> nhận một struct 904 byte dưới dạng{' '}
            <code>constant Filter&amp;</code>, được điền bằng <code>setBytes</code>, cách này chạy được chừng nào struct còn
            dưới 4 KB.
          </p>
        }
      >
        <p>
          Lab 7.1&rsquo;s Metal files do this and one more: <code>conv_const</code> takes the filter buffer as{' '}
          <code>constant float*</code>, and <code>conv_by_value</code> takes a 904-byte struct as{' '}
          <code>constant Filter&amp;</code>, filled with <code>setBytes</code>, which works while the struct is under 4 KB.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch07-facts.md C1, C3; ch05-facts C1', // i18n-ignore: citation only
  },
  'ch07/7.4/q-setbytes': {
    kind: 'choice',
    id: 'ch07/7.4/q-setbytes',
    prompt: (
      <Translated
        vi={
          <>
            Bạn muốn truyền một khối 32 × 32 trọng số float cộng một int (4,100 byte) dưới dạng struct theo giá trị. Ở đâu
            thì cách đó chạy được như đang viết?
          </>
        }
      >
        <>You want to pass a 32 × 32 block of float weights plus an int (4,100 bytes) as a struct by value. Where does that work as written?</>
      </Translated>
    ),
    options: [
      {
        label: (
          <Translated vi="Dưới dạng đối số kernel CUDA, nhưng không qua setBytes của Metal: trên Metal nó cần một buffer.">
            {"As a CUDA kernel argument, but not through Metal's setBytes: on Metal it needs a buffer."}
          </Translated>
        ),
        correct: true,
      },
      {
        label: (
          <Translated vi="Trên mọi backend: đối số được sao chép vào bộ nhớ hằng, vốn còn dư 64 KB.">
            On every backend: arguments are copied into constant memory, which has 64 KB to spare.
          </Translated>
        ),
        why: (
          <Translated vi="CUDA ghi trong tài liệu 32,764 byte cho đối số và setBytes của Metal dừng dưới 4 KB.">
            {'CUDA documents 32,764 bytes for arguments and Metal\'s setBytes stops below 4 KB.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Không ở đâu cả: mọi backend giới hạn đối số kernel ở 4 KB, như CUDA từng làm.">
            On none of them: every backend caps kernel arguments at 4 KB, as CUDA used to.
          </Translated>
        ),
        why: (
          <Translated vi="4,096 byte là giới hạn của CUDA trước 12.1. Giờ là 32,764 trên Volta trở về sau.">
            {'4,096 bytes was CUDA\'s limit before 12.1. It is 32,764 now on Volta and later.'}
          </Translated>
        ),
      },
      {
        label: (
          <Translated vi="Qua setBytes của Metal, vốn sao chép mọi kích thước, nhưng không phải dưới dạng đối số kernel CUDA.">
            {"Through Metal's setBytes, which copies any size, but not as a CUDA kernel argument."}
          </Translated>
        ),
        why: (
          <Translated vi="Ngược lại mới đúng: setBytes nhận ít hơn 4 KB, còn CUDA nhận tới 32,764 byte.">
            The other way round: setBytes takes less than 4 KB, and CUDA takes up to 32,764 bytes.
          </Translated>
        ),
      },
    ],
    explain: (
      <Translated
        vi={
          <p>
            Giới hạn đối số kernel của CUDA là 32,764 byte kể từ CUDA 12.1; <code>setBytes</code> của Metal nhận ít hơn 4
            KB; AMD không ghi giới hạn nào cho HIP. Đây là lý do struct của lab 7.1 dừng ở bán kính 7: 904 byte nằm dưới mọi
            giới hạn được ghi trong tài liệu, và cũng dưới con số 1024 byte thấy trong mã nguồn runtime của AMD mà không gắn
            được với các lần khởi chạy thông thường. Quá 4 KB, lựa chọn khả chuyển là một buffer.
          </p>
        }
      >
        <p>
          CUDA&rsquo;s limit on kernel arguments has been 32,764 bytes since CUDA 12.1; Metal&rsquo;s{' '}
          <code>setBytes</code> takes less than 4 KB; AMD documents no limit for HIP. This is why lab 7.1&rsquo;s struct
          stops at radius 7: 904 bytes is under every documented limit, and also under a 1024-byte figure seen in
          AMD&rsquo;s runtime source that could not be tied to ordinary launches. Past 4 KB the portable choice is a buffer.
        </p>
      </Translated>
    ),
    source: 'notes/today/ch07-facts.md A4, B2, C1', // i18n-ignore: citation only
  },
};

// ---------------------------------------------------------------- the page
export function S74() {
  const q = questionsS74;
  return (
    <>
      <Translated
        vi={
          <p>
            Bộ lọc có ba tính chất khiến nó đặc biệt. Nó nhỏ: sách nói phần lớn bộ lọc có bán kính 7 trở xuống, nên tối đa
            15 × 15 = 225 float, 900 byte. Nó không bao giờ thay đổi trong lúc kernel chạy. Và mọi luồng (thread) đọc toàn bộ
            nó, theo cùng một thứ tự, bắt đầu từ <code>F[0][0]</code>. Dữ liệu như vậy có thể được phục vụ bởi thứ gì đó rẻ
            hơn nhiều so với đường truy cập bộ nhớ đa dụng. (Sách cũng giới hạn bộ lọc 3D ở 7³ = 343 phần tử, vừa với một
            bộ lọc rộng 7, không phải bán kính 7, vốn sẽ là 15³: một trong hai phát biểu bị lỏng.)
          </p>
        }
      >
        <p>
          The filter has three properties that make it special. It is small: the book says most filters have a radius of
          7 or less, so at most 15 × 15 = 225 floats, 900 bytes. It never changes while the kernel runs. And every thread
          reads all of it, in the same order, starting from <code>F[0][0]</code>. Data like that can be served by
          something much cheaper than the general-purpose memory path. (The book also bounds a 3D filter by 7³ = 343
          elements, which fits a filter 7 wide, not a radius of 7, which would be 15³: one of the two statements is
          loose.)
        </p>
      </Translated>
      <Source section="7.4" pages="168" />

      <h2>
        <Translated vi="Bộ nhớ hằng">Constant memory</Translated>
      </h2>
      <Translated
        vi={
          <p>
            CUDA có một loại bộ nhớ được làm cho việc này. Bộ nhớ hằng (constant memory) hiển thị với mọi khối, như bộ nhớ
            toàn cục (global memory), nhưng kernel chỉ có thể đọc nó; host ghi nó trước lần khởi chạy. Nó nhỏ: 64 KB. Chương
            5 đã đặt nó trên bản đồ bộ nhớ (<a href={href('ch05', '5.2')}>mục 5.2</a>); chương này là nơi nó được dùng.
          </p>
        }
      >
        <p>
          CUDA has a memory type made for it. Constant memory is visible to every block, like global memory, but kernels
          can only read it; the host writes it before the launch. It is small: 64 KB. Chapter 5 placed it in the memory
          map (<a href={href('ch05', '5.2')}>section 5.2</a>); this chapter is where it gets used.
        </p>
      </Translated>
      <Today
        checked={tr(
          '2026-11-20 against the CUDA 13.5 guide and HIP 7.17 docs (ch05-facts A4; ch07-facts B1)',
          '2026-11-20, đối chiếu với hướng dẫn CUDA 13.5 và tài liệu HIP 7.17 (ch05-facts A4; ch07-facts B1)',
        )}
      >
        <Translated
          vi={
            <p>
              64 KB trên mọi compute capability mà NVIDIA liệt kê hiện nay, và một chương trình có thể hỏi bằng{' '}
              <code>totalConstMem</code>. HIP giữ <code>__constant__</code>, nhưng AMD không ghi kích thước. Theo mã nguồn
              runtime của AMD (CLR), <code>totalConstMem</code> báo vùng cấp phát đơn lớn nhất; không có trong tài liệu,
              chưa chạy ở đây.
            </p>
          }
        >
          <p>
            64 KB on every compute capability NVIDIA lists today, and a program can ask with <code>totalConstMem</code>.
            HIP keeps <code>__constant__</code>, but AMD documents no size. According to AMD&rsquo;s runtime source
            (CLR), <code>totalConstMem</code> reports the largest single allocation; not documented, not run here.
          </p>
        </Translated>
      </Today>

      <h2>
        <Translated vi="Khai báo và điền dữ liệu">Declaring it and filling it</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một mảng hằng là một biến ở phạm vi tệp, nằm ngoài mọi hàm, với qualifier <code>__constant__</code>. Host điền
            nó bằng <code>cudaMemcpyToSymbol(dest, src, bytes)</code>, gọi tên chính biến đó thay vì một con trỏ.
          </p>
        }
      >
        <p>
          A constant array is a variable at file scope, outside every function, with the <code>__constant__</code>{' '}
          qualifier. The host fills it with <code>cudaMemcpyToSymbol(dest, src, bytes)</code>, naming the variable
          itself rather than a pointer.
        </p>
      </Translated>
      <Source section="7.4" pages="169" figs={tr('Fig. 7.9', 'Hình 7.9')} />
      <Translated
        vi={
          <p>
            Kernel đọc nó như một mảng toàn cục thông thường, nên nó không còn là tham số kernel. Hãy giữ mảng, kernel và lần
            sao chép trong cùng một tệp <code>.cu</code>: sách gợi ý một khai báo <code>extern</code> khi chúng bị tách ra,
            điều mà chế độ mặc định của nvcc từ chối. Dưới đây là phiên bản của lab 7.1, cùng bản tương ứng trên Metal.
          </p>
        }
      >
        <p>
          Kernels read it as an ordinary global array, so it is no longer a kernel parameter. Keep the array, the kernel
          and the copy in one <code>.cu</code> file: the book suggests an <code>extern</code> declaration when they are
          split, which nvcc&rsquo;s default mode rejects. Below is lab 7.1&rsquo;s version, with its Metal counterpart.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch07-facts A1, A2, B3)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Biến device <code>extern</code> cần <code>-rdc=true</code> (HIP: <code>-fgpu-rdc</code>). Lần sao chép ghi được
              mọi symbol device; tính chỉ-đọc đến từ <code>__constant__</code>, không phải từ lần sao chép.
            </p>
          }
        >
          <p>
            <code>extern</code> device variables need <code>-rdc=true</code> (HIP: <code>-fgpu-rdc</code>). The copy
            writes any device symbol; read-only comes from <code>__constant__</code>, not from the copy.
          </p>
        </Translated>
      </Today>
      <LabPart
        name="const" /* i18n-ignore: lab part marker */
        cuTiers={CU_CONST}
        mslTiers={MSL_CONST}
        cuMark="F_c"
        mslMark="constant float*" /* i18n-ignore: lab part marker */
        mmMark="atIndex:1" /* i18n-ignore: lab part marker */
        caption={
          <Translated
            vi={
              <>
                Lời giải tham chiếu của lab 7.1, phần bộ nhớ hằng, tải từ <code>labs/{SOL}/</code>; trạng thái bên cạnh các
                tab được đọc từ các lần chạy đã ghi lại trên máy này. Chỗ khác với Hình 7.9:{' '}
                <code>F_c</code> là mảng phẳng và có kích thước cho bán kính 7, nên một bản build phục vụ mọi bán kính từ 0
                đến 7; <code>inRow</code> và <code>inCol</code> được khai báo <code>int</code>; có thêm lệnh chặn đầu ra.
                Metal không có sao chép symbol: bộ lọc là một buffer thông thường do host điền, và kernel khai báo nó là{' '}
                <code>constant float*</code>. Bản build HIP được thiết lập để nhận nguyên tệp CUDA (<code>cudaMemcpyToSymbol</code>{' '}
                ánh xạ sang <code>hipMemcpyToSymbol</code>); nó chưa được biên dịch hay chạy trên phần cứng AMD (unverified).
              </>
            }
          >
            <>
              Lab 7.1&rsquo;s reference solution, the constant-memory part, loaded from <code>labs/{SOL}/</code>; the
              status beside the tabs is read from this machine&rsquo;s recorded runs. Where it differs from Fig. 7.9:{' '}
              <code>F_c</code> is flat and sized for radius 7, so one build serves every radius from 0 to 7;{' '}
              <code>inRow</code> and <code>inCol</code> are declared <code>int</code>; an output guard is added. Metal has
              no symbol copy: the filter is an ordinary buffer the host fills, and the kernel declares it{' '}
              <code>constant float*</code>. The HIP build is set up to take the CUDA file unchanged (<code>cudaMemcpyToSymbol</code>{' '}
              maps to <code>hipMemcpyToSymbol</code>); it has not been compiled or run on AMD hardware (unverified).
            </>
          </Translated>
        }
      />
      <Ask q={q['ch07/7.4/q-extern']} />
      <Ask q={q['ch07/7.4/q-device-vs-constant']} />

      <h2>
        <Translated vi="Nó nằm ở đâu, và các bộ nhớ đệm ở giữa">Where it lives, and the caches in between</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bộ nhớ hằng không phải một con chip riêng. Nó nằm trong DRAM cùng bộ nhớ toàn cục; điểm khác là phần cứng biết
            nó không thể thay đổi trong lúc một kernel chạy, nên có thể lưu nó vào bộ nhớ đệm một cách mạnh tay. Để thấy điều
            đó mang lại gì, sách đi một vòng ngắn qua bộ nhớ đệm (cache): chúng trong suốt (bạn đọc biến gốc và phần cứng giữ
            dữ liệu gần đây ở gần), khác với bộ nhớ chia sẻ (shared memory), thứ bạn tự điền; và chúng có nhiều tầng, nhỏ và
            nhanh gần mỗi lõi, lớn hơn và dùng chung ở xa hơn.
          </p>
        }
      >
        <p>
          Constant memory is not a separate chip. It sits in DRAM with global memory; the difference is that the
          hardware knows it cannot change during a kernel, so it can be cached aggressively. To see what that buys, the
          book takes a short tour of caches: they are transparent (you read the original variable and the hardware keeps
          recent data close), unlike shared memory, which you fill yourself; and they come in levels, small and fast
          near each core, larger and shared further out.
        </p>
      </Translated>
      <Source section="7.4" pages="169–171" figs={tr('Fig. 7.10', 'Hình 7.10')} />
      <Translated vi={<p>Các con số của sách cho những tầng đó là chung chung. Đây là những gì từng hãng ghi trong tài liệu hiện nay.</p>}>
        <p>The book&rsquo;s figures for those levels are generic. Here is what each vendor documents today.</p>
      </Translated>
      <Today checked="2026-11-20 (ch07-facts A5, B5, C4; infra-facts C13)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Con số 128 KB của sách đã lỗi thời với NVIDIA. AMD đưa phần lớn kích thước theo từng mẫu GPU, nên bảng chỉ giữ
              con số chung và cấu trúc của AMD. Mô tả của Apple đến từ các bài nói ở hội nghị, không phải tài liệu tham chiếu.
            </p>
          }
        >
          <p>
            The book&rsquo;s 128 KB is out of date for NVIDIA. AMD gives most sizes per GPU model, so the table keeps
            AMD&rsquo;s generic figure and the structure only. Apple&rsquo;s descriptions come from conference talks, not reference documentation.
          </p>
        </Translated>
      </Today>
      <table className="data" data-table="caches">
        <thead>
          <tr>
            <th><Translated vi="Tầng">Level</Translated></th>
            <th><Translated vi="Sách (tr. 171)">Book (p. 171)</Translated></th>
            <th>NVIDIA</th>
            <th>AMD</th>
            <th><Translated vi="Apple">Apple</Translated></th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="L1, mỗi lõi">L1, per core</Translated></td>
            <td><Translated vi="tới 128 KB">up to 128 KB</Translated></td>
            <td>
              <Translated vi="một L1 hợp nhất với bộ nhớ chia sẻ, 96 đến 336 KB mỗi SM, tùy compute capability">
                one unified L1 and shared memory, 96 to 336 KB per SM, by compute capability
              </Translated>
            </td>
            <td>
              <Translated vi={<>L1 vector mỗi CU (16 KB &ldquo;điển hình&rdquo;, lớn hơn trên một số GPU), bộ nhớ chia sẻ tách riêng</>}>
                vector L1 per CU (16 KB &ldquo;typical&rdquo;, larger on some GPUs), shared memory separate
              </Translated>
            </td>
            <td>
              <Translated vi="L1 riêng cho buffer và texture; không có kích thước">separate L1s for buffers and textures; no sizes</Translated>
            </td>
          </tr>
          <tr>
            <td>L2</td>
            <td>
              <Translated vi="vài trăm KB đến hàng chục MB, hàng chục đến hàng trăm chu kỳ">
                a few hundred KB to tens of MB, tens to hundreds of cycles
              </Translated>
            </td>
            <td>
              <Translated vi={<>một L2 dùng chung cho mọi SM, kích thước trong <code>l2CacheSize</code>; không có số liệu độ trễ</>}>
                one L2 shared by all SMs, size in <code>l2CacheSize</code>; no latency figure
              </Translated>
            </td>
            <td>
              <Translated vi="dùng chung bởi các CU của một die; trên MI300X, mỗi chiplet (XCD) một L2">
                shared by the CUs of one die; on MI300X, one L2 per chiplet (XCD)
              </Translated>
            </td>
            <td>
              <Translated vi="một cache cấp cuối dùng chung cho mọi lõi GPU; không có kích thước">
                a last-level cache shared by all GPU cores; no size
              </Translated>
            </td>
          </tr>
          <tr>
            <td>L3</td>
            <td>
              <Translated vi="hàng trăm MB trên một số bộ xử lý">hundreds of MB on some processors</Translated>
            </td>
            <td>
              <Translated vi="không có trong tài liệu">none documented</Translated>
            </td>
            <td>
              <Translated vi="trên MI300X, một cache cấp cuối dùng chung cho toàn GPU">
                on MI300X, a last-level cache shared by the whole GPU
              </Translated>
            </td>
            <td>
              <Translated vi="có nhắc tới một cache cấp hệ thống; không có kích thước">
                a system-level cache is mentioned; no size
              </Translated>
            </td>
          </tr>
        </tbody>
      </table>
      <YourConstantMemory />

      <h2>
        <Translated vi="Constant cache">The constant cache</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Một bộ nhớ đệm không bao giờ phải nhận lệnh ghi thì rẻ hơn nhiều so với loại phải nhận, nên GPU có một bộ nhớ
            đệm chỉ-đọc nhỏ cho bộ nhớ hằng.
          </p>
        }
      >
        <p>
          A cache that never has to accept writes is much cheaper to build than one that does, so GPUs have a small
          read-only cache for constant memory.
        </p>
      </Translated>
      <Source section="7.4" pages="171" />
      <Translated
        vi={
          <p>
            Thứ khiến nó nhanh là một quy tắc về warp. Khi mọi lane đọc cùng một địa chỉ, một lần đọc trả lời tất cả. Khi
            các lane đọc các địa chỉ khác nhau, các lần đọc được phục vụ lần lượt, và chi phí tăng theo số địa chỉ phân biệt.
            Bộ lọc là trường hợp tốt nhất có thể: chỉ số của nó phụ thuộc vào vòng lặp, không phụ thuộc vào luồng, nên ở mỗi
            tap cả warp chỉ hỏi một địa chỉ.
          </p>
        }
      >
        <p>
          What makes it fast is a rule about the warp. When every lane reads the same address, one read answers all of
          them. When the lanes read different addresses, the reads are served one after another, and the cost grows with
          the number of distinct addresses. The filter is the best possible case: its index depends on the loop, not on
          the thread, so at every tap the whole warp asks for one address.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch07-facts A3, B1, C2)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              NVIDIA: tốc độ thanh ghi cho một địa chỉ, chi phí tuyến tính theo số địa chỉ phân biệt, working set 8 KB mỗi
              SM. AMD: scalar L1, chỉ cho các lần đọc đồng nhất trong wavefront. Apple: không có constant cache trong tài
              liệu.
            </p>
          }
        >
          <p>
            NVIDIA: register speed for one address, cost linear in distinct addresses, 8 KB working set per SM. AMD: the
            scalar L1, for wavefront-uniform reads only. Apple: no constant cache documented.
          </p>
        </Translated>
      </Today>
      <Predict
        id="ch07/7.4/p-codebook"
        prompt={tr(
          "Now the opposite case. A warp of 32 decodes 4-bit weights through a 16-entry table in constant memory, each lane with its own random code. Under NVIDIA's rule, how many serialised reads does one lookup cost on average?",
          'Giờ là trường hợp ngược lại. Một warp 32 lane giải mã các trọng số 4-bit qua một bảng 16 mục trong bộ nhớ hằng, mỗi lane có mã ngẫu nhiên riêng. Theo quy tắc của NVIDIA, trung bình một lần tra bảng tốn bao nhiêu lần đọc tuần tự?',
        )}
        unit={tr('reads', 'lần đọc')}
      >
        {(v) => (
          <Translated
            vi={
              <p>
                Khoảng <b>14</b>: 16 × (1 − (15/16)<sup>32</sup>) = 13.97. Mỗi mục bị cả 32 lane bỏ qua với xác suất
                (15/16)<sup>32</sup> ≈ 0.13, nên khoảng 2 trong 16 mục không ai hỏi và mỗi mục còn lại là một lần đọc.{' '}
                {v <= 2
                  ? 'Một con số nhỏ là chi phí của bộ lọc, vì ở đó mọi lane đồng ý với nhau. Ở đây chúng gần như không bao giờ đồng ý.'
                  : v >= 16
                    ? 'Không thể nhiều hơn 16: các lane dùng chung một mục thì dùng chung lần đọc của nó.'
                    : Math.abs(v - 14) <= 1
                      ? 'Gần đúng.'
                      : 'Hãy thử trình khám phá bên dưới với kiểu codebook và đổi seed vài lần.'}
              </p>
            }
          >
            <p>
              About <b>14</b>: 16 × (1 − (15/16)<sup>32</sup>) = 13.97. Each entry is missed by all 32 lanes with
              probability (15/16)<sup>32</sup> ≈ 0.13, so about 2 of the 16 go unasked and the rest are each one read.{' '}
              {v <= 2
                ? 'A small number is what the filter costs, because there every lane agrees. Here they almost never do.'
                : v >= 16
                  ? 'It cannot be more than 16: lanes that share an entry share its read.'
                  : Math.abs(v - 14) <= 1
                    ? 'Close.'
                    : 'Try the explorer below with the codebook pattern and reseed it a few times.'}
            </p>
          </Translated>
        )}
      </Predict>
      <OneAddressOrMany />
      <Translated
        vi={
          <p>
            Trình khám phá đếm những gì quy tắc của các hãng ngụ ý; nó không đo thời gian. Bộ lọc tốn một lần đọc mỗi tap
            bất kể độ rộng warp. Mỗi lane một mục tốn số lần đọc bằng số lane, và một lần tra codebook ngẫu nhiên tốn gần
            bằng số mục của codebook. Hoạt ảnh cho thấy cùng các con số đó theo thời gian.
          </p>
        }
      >
        <p>
          The explorer counts what the vendors&rsquo; rules imply; it does not time anything. The filter costs one read
          per tap whatever the warp width. One entry per lane costs as many reads as there are lanes, and a random
          codebook lookup nearly as many as the codebook has entries. The animation shows the same counts in time.
        </p>
      </Translated>
      <Simplification>
        <Translated
          vi={
            <p>
              Mọi lần đọc đều trúng bộ nhớ đệm: không có lần trượt nào so với working set 8 KB của NVIDIA, không có chuyến
              đi tới DRAM, không có độ trễ. Thứ tự các vòng chỉ là thứ tự vẽ. Không có gì được mô hình hóa cho Apple, vì
              Apple không ghi quy tắc nào.
            </p>
          }
        >
          <p>
            Every read hits the cache: no misses against NVIDIA&rsquo;s 8 KB working set, no trip to DRAM, no latency. The
            order of the rounds is only a drawing order. Nothing is modelled for Apple, because Apple documents no rule.
          </p>
        </Translated>
      </Simplification>
      <ConstantBroadcastScene />
      <Ask q={q['ch07/7.4/q-cache-fit']} />

      <h2>
        <Translated vi="Nó mang lại gì: 0.5 FLOP/B">What it buys: 0.5 FLOP/B</Translated>
      </h2>
      <Predict
        id="ch07/7.4/p-ai"
        prompt={tr(
          "Assume every read of F is served by the constant cache and costs no DRAM traffic. What is the kernel's FLOP/B now, counting only the loop?",
          'Giả sử mọi lần đọc F đều được constant cache phục vụ và không tốn lưu lượng DRAM. FLOP/B của kernel giờ là bao nhiêu, chỉ tính vòng lặp?',
        )}
        unit="FLOP/B"
      >
        {(v) => (
          <Translated
            vi={
              <p>
                Một lần nạp 4 byte của <code>N</code> mỗi tap cho một phép nhân và một phép cộng: 2/4 = <b>0.5 FLOP/B</b>,
                gấp đôi 0.25 của kernel cơ bản.{' '}
                {Math.abs(v - 0.5) < 0.02
                  ? 'Chính xác.'
                  : v < 0.3
                    ? 'Nếu bạn ra 0.25, bạn vẫn tính lần nạp bộ lọc vào bộ nhớ: giả định là constant cache trả lời nó.'
                    : 'Chỉ còn lần nạp đầu vào trong phép đếm, và nó vẫn là một lần nạp mỗi tap.'}{' '}
                Trên thực tế con số còn cao hơn nữa, vì các phần tử của <code>N</code> được các luồng lân cận dùng lại và
                đôi khi được tìm thấy trong L1 hoặc L2.
              </p>
            }
          >
            <p>
              One 4-byte load of <code>N</code> per tap for one multiply and one add: 2/4 = <b>0.5 FLOP/B</b>, twice the
              basic kernel&rsquo;s 0.25.{' '}
              {Math.abs(v - 0.5) < 0.02
                ? 'Exactly.'
                : v < 0.3
                  ? 'If you had 0.25, you still charged the filter load to memory: the assumption is that the constant cache answers it.'
                  : 'Only the input load is left in the count, and it is still one load per tap.'}{' '}
              In practice it is higher again, because elements of <code>N</code> are reused by neighbouring threads and
              are sometimes found in L1 or L2.
            </p>
          </Translated>
        )}
      </Predict>
      <Source section="7.4" pages="171" />
      <Translated
        vi={
          <p>
            Trên giấy, điều đó giảm một nửa lưu lượng, và đó cũng là mức tối đa bước này làm được: bộ lọc đã ra khỏi phép
            đếm, và cái còn lại là một lần nạp đầu vào mỗi tap. Nó có giảm một nửa thời gian hay không tùy vào việc các bộ
            nhớ đệm thông thường đã hấp thụ bao nhiêu lưu lượng của bộ lọc, điều chỉ một phép đo mới cho thấy: bảng của mục
            7.3 đặt cả hai kernel so với trần của chúng trên thiết bị của bạn khi lab 7.1 đã chạy. Các lần nạp đầu vào là
            việc của phân ô (tiling) bằng bộ nhớ chia sẻ ở <a href={href('ch07', '7.5')}>mục 7.5</a>.
          </p>
        }
      >
        <p>
          On paper that halves the traffic, and it is also the most this step can do: the filter is gone from the count,
          and what is left is one input load per tap. Whether it halves the time depends on how much of the filter
          traffic the ordinary caches were already absorbing, which only a measurement shows: section 7.3&rsquo;s panel
          sets both kernels against their ceilings on your device once lab 7.1 has run. The input loads are the job of
          shared-memory tiling in <a href={href('ch07', '7.5')}>section 7.5</a>.
        </p>
      </Translated>

      <h2>
        <Translated vi="Hai cách khác để giữ một bộ lọc hằng">Two other ways to hold a constant filter</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Sách đưa ra hai phương án thay thế. Phương án thứ nhất là truyền trọng số dưới dạng đối số kernel: đặt chúng vào
            một struct và truyền theo giá trị. Trình biên dịch CUDA cũng đặt đối số kernel trong bộ nhớ hằng, nên các lần đọc
            vẫn đi qua constant cache, và mỗi lần khởi chạy có thể mang bộ lọc riêng mà không cần sao chép symbol.
          </p>
        }
      >
        <p>
          The book offers two alternatives. The first is to pass the weights as kernel arguments: put them in a struct
          and pass it by value. The CUDA compiler places kernel arguments in constant memory too, so the reads still go
          through the constant cache, and each launch can bring its own filter with no symbol copy.
        </p>
      </Translated>
      <Source section="7.4" pages="172" />
      <Translated
        vi={
          <p>
            Phương án thứ hai là viết trọng số thẳng vào mã nguồn khi chúng đã biết lúc biên dịch, đổi lại kernel chỉ áp dụng
            được một bộ lọc. Trên Metal, struct theo giá trị trở thành một đối số <code>constant Filter&amp;</code>{' '}
            được điền bằng <code>setBytes</code>, còn bộ lọc viết cứng thành một mảng <code>constant</code> ở phạm vi chương
            trình, phải được khởi tạo trong mã nguồn. Một function constant có thể cố định bán kính nhưng không thể định kích
            thước cho một mảng <code>threadgroup</code>.
          </p>
        }
      >
        <p>
          The second is to write the weights into the source when they are known at compile time, at the cost of a kernel
          that only ever applies one filter. On Metal the by-value struct becomes a <code>constant Filter&amp;</code>{' '}
          argument filled with <code>setBytes</code>, and the hard-coded filter a program-scope <code>constant</code> array,
          which must be initialised in the source. A function constant can fix the radius but cannot size a{' '}
          <code>threadgroup</code> array.
        </p>
      </Translated>
      <Today checked="2026-11-20 (ch07-facts A4, B2, C1, C3)">{/* i18n-ignore: date and citation */}
        <Translated
          vi={
            <p>
              Đối số CUDA có thể tổng cộng tới 32,764 byte kể từ CUDA 12.1 (trước đó là 4,096); <code>__grid_constant__</code>{' '}
              yêu cầu một bản sao dùng chung. HIP không ghi cả hai, nên tệp <code>.cu</code> khả chuyển truyền một struct{' '}
              <code>const</code> thông thường. <code>setBytes</code> nhận dưới 4 KB.
            </p>
          }
        >
          <p>
            CUDA arguments may total 32,764 bytes since CUDA 12.1 (4,096 before); <code>__grid_constant__</code> asks for
            one shared copy. HIP documents neither, so the portable <code>.cu</code> passes a plain <code>const</code>{' '}
            struct. <code>setBytes</code> takes under 4 KB.
          </p>
        </Translated>
      </Today>
      <LabPart
        name="byvalue" /* i18n-ignore: lab part marker */
        cuTiers={CU_BYVALUE}
        mslTiers={MSL_BYVALUE}
        cuMark="Filter" /* i18n-ignore: lab part marker */
        mslMark="Filter" /* i18n-ignore: lab part marker */
        mmMark="setBytes:&f" /* i18n-ignore: lab part marker */
        caption={
          <Translated
            vi={
              <>
                Lời giải tham chiếu của lab 7.1, phần truyền theo giá trị. Struct có 904 byte, dưới mọi giới hạn được ghi
                trong tài liệu trên cả ba backend. Trên CUDA, struct <code>f</code> được tô màu bộ nhớ hằng vì đó là nơi
                trình biên dịch đặt đối số kernel; trên Metal, nó là một tham chiếu <code>constant</code> tới các byte được
                sao chép vào command. Lab 7.1 đo thời gian cả ba cách trên thiết bị của bạn.
              </>
            }
          >
            <>
              Lab 7.1&rsquo;s reference solution, the by-value part. The struct is 904 bytes, under every documented limit
              on all three backends. On CUDA the struct <code>f</code> is coloured as constant memory because that is where
              the compiler puts kernel arguments; on Metal it is a <code>constant</code> reference to bytes copied into
              the command. Lab 7.1 times all three ways on your device.
            </>
          </Translated>
        }
      />
      <Ask q={q['ch07/7.4/q-metal']} />
      <Ask q={q['ch07/7.4/q-setbytes']} />

      <h2>
        <Translated vi="Các backend thực tế làm gì với dữ liệu chỉ-đọc nhỏ">What real backends do with small read-only data</Translated>
      </h2>
      <Translated
        vi={
          <p>
            Bộ lọc là trường hợp sách giáo khoa. Một engine suy luận có nhiều thứ chỉ-đọc nhỏ, và chúng không phải đều thuộc
            về cùng một chỗ. Câu hỏi cần đặt cho từng thứ là câu hỏi mà mục này xoay quanh: nó có nhỏ không, và các lane của
            một warp có đọc cùng một phần tử của nó không?
          </p>
        }
      >
        <p>
          The filter is the textbook case. An inference engine has many small read-only things, and they do not all
          belong in the same place. The question to ask of each is the one this section turns on: is it small, and do
          the lanes of a warp read the same element of it?
        </p>
      </Translated>
      <Beyond title={tr('Not in the book: real backends', 'Không có trong sách: các backend thực tế')}>
        <Translated
          vi={
            <p>
              llama.cpp giữ các codebook lượng tử hóa dưới dạng mảng <code>static const __device__</code> trên CUDA và HIP,
              và dưới dạng mảng <code>constant</code> ở phạm vi chương trình trên Metal. Cả backend CUDA của ggml lẫn các
              kernel CUDA gốc của PyTorch đều hoàn toàn không dùng <code>__constant__</code>. Các struct đối số nhỏ được
              truyền theo giá trị trên CUDA và dưới dạng tham chiếu <code>constant</code> được điền bằng{' '}
              <code>setBytes</code> trên Metal (ch07-facts D4).
            </p>
          }
        >
          <p>
            llama.cpp keeps its quantisation codebooks as <code>static const __device__</code> arrays on CUDA and HIP,
            and as program-scope <code>constant</code> arrays on Metal. Neither ggml&rsquo;s CUDA backend nor
            PyTorch&rsquo;s native CUDA kernels use <code>__constant__</code> at all. Small argument structs go by value
            on CUDA and as a <code>constant</code> reference filled with <code>setBytes</code> on Metal (ch07-facts D4).
          </p>
        </Translated>
      </Beyond>
      <table className="data" data-table="backends">
        <thead>
          <tr>
            <th><Translated vi="Tính năng">Feature</Translated></th>
            <th>CUDA</th>
            <th>HIP</th>
            <th>Metal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><Translated vi="Bảng chỉ-đọc do host đặt">Read-only table set by the host</Translated></td>
            <td><code>__constant__</code> + <code>cudaMemcpyToSymbol</code></td>
            <td><Translated vi={<>tương tự, với <code>hipMemcpyToSymbol</code></>}>same, with <code>hipMemcpyToSymbol</code></Translated></td>
            <td><Translated vi={<>buffer gắn dưới dạng <code>constant T*</code>; không có sao chép symbol</>}>buffer bound as <code>constant T*</code>; no symbol copy</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Kích thước trong tài liệu">Documented size</Translated></td>
            <td><Translated vi="64 KB; working set của cache 8 KB mỗi SM">64 KB; 8 KB cache working set per SM</Translated></td>
            <td><Translated vi={<>không có; trong mã nguồn runtime của AMD (CLR), <code>totalConstMem</code> là vùng cấp phát lớn nhất (không có trong tài liệu, chưa chạy ở đây)</>}>none; in AMD&rsquo;s runtime source (CLR), <code>totalConstMem</code> is the largest allocation (not documented, not run here)</Translated></td>
            <td><Translated vi={<>không có cho constant buffer; <code>setBytes</code> dưới 4 KB</>}>none for constant buffers; <code>setBytes</code> under 4 KB</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Bộ nhớ đệm phía sau">Cache behind it</Translated></td>
            <td><Translated vi="constant cache, mỗi địa chỉ phân biệt một lần đọc">constant cache, one read per distinct address</Translated></td>
            <td><Translated vi="scalar L1, chỉ cho lần đọc đồng nhất">scalar L1, uniform reads only</Translated></td>
            <td><Translated vi="không có trong tài liệu">none documented</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Struct tham số nhỏ">Small struct of parameters</Translated></td>
            <td><Translated vi="theo giá trị, qua bộ nhớ hằng, tới 32,764 byte">by value, via constant memory, up to 32,764 bytes</Translated></td>
            <td><Translated vi="theo giá trị, lưu trong scalar L1; không có giới hạn trong tài liệu">by value, cached in the scalar L1; no documented limit</Translated></td>
            <td><Translated vi={<><code>constant T&amp;</code> được điền bằng <code>setBytes</code></>}><code>constant T&amp;</code> filled with <code>setBytes</code></Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Bán kính lúc biên dịch">Compile-time radius</Translated></td>
            <td><Translated vi={<><code>#define</code> hoặc tham số template</>}><code>#define</code> or template parameter</Translated></td>
            <td><Translated vi="tương tự">same</Translated></td>
            <td><Translated vi={<>macro, hoặc một function constant (không thể định kích thước cho mảng <code>threadgroup</code>)</>}>macro, or a function constant (which cannot size a <code>threadgroup</code> array)</Translated></td>
          </tr>
          <tr>
            <td><Translated vi="Symbol hằng dùng từ tệp khác">Constant symbol used from another file</Translated></td>
            <td><Translated vi={<>cần <code>-rdc=true</code></>}>needs <code>-rdc=true</code></Translated></td>
            <td><Translated vi={<>cần <code>-fgpu-rdc</code></>}>needs <code>-fgpu-rdc</code></Translated></td>
            <td><Translated vi="không áp dụng">not applicable</Translated></td>
          </tr>
        </tbody>
      </table>
      <Translated
        vi={
          <p>
            Giờ hãy tự đặt sáu thứ trong số đó. Sau đó lab 7.1 đo bộ lọc theo ba cách trên thiết bị của bạn (
            <a href={href('ch07', 'labs')}>lab</a>).
          </p>
        }
      >
        <p>
          Now place six of them yourself. Lab 7.1 then measures the filter three ways on your device (
          <a href={href('ch07', 'labs')}>labs</a>).
        </p>
      </Translated>
      <WhereWouldYouPutIt />
      <Ask q={q['ch07/7.4/q-codebook']} />
    </>
  );
}
