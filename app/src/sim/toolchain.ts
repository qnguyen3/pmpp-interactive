// Rule model for chapter 2, section 2.7 (and the qualifier table of section 2.5):
// which compilation sees each function of a small program, what each backend's
// toolchain does with it stage by stage, and who may call a function and who
// then executes it. It runs the "Follow one source file" stepper, the "Who may
// call it, who runs it" classification, the Metal call table and the section's
// questions.
//
// Every rule carries the id of the fact it comes from, so the lesson can cite it.
// Ids without a prefix are items of notes/today/ch02-facts.md; "ch05 B4" and the
// like are items of notes/today/ch05-facts.md; "Fig. 2.11" and page numbers are
// the book (notes/ch02.md).
//  - Fig. 2.11, pp. 35-36: the three qualifiers, where each function may be called
//    from, where it runs and which thread runs it; __host__ is the default;
//    __host__ __device__ together give two compiled versions.
//  - A7: the same table in NVIDIA's current guide, and the __CUDA_ARCH__ macro.
//  - A13, ch05 B4, C7: a kernel launched from device code (dynamic parallelism):
//    CUDA yes, HIP no, Metal none documented.
//  - A14: nvcc splits host from device code; device code becomes PTX, then a cubin
//    through ptxas; both travel in a fatbin; the device driver compiles PTX at run
//    time only when it has to.
//  - B4, B6: amdclang++ makes one host pass and one device pass per architecture
//    named with --offload-arch; the machine code is embedded; nothing like PTX is
//    shipped by default.
//  - C1, C4, C6, C7: on Metal the kernel is a separate file in another language;
//    a library is built at build time or at run time; creating the pipeline state
//    finishes compilation for the GPU at hand; there are no qualifiers.
//
// What this leaves out:
//  - Linking, and the host compiler's own stages.
//  - Caches of just-in-time results, and the first-call cost of initialising a runtime.
//  - Run-time compilation as an API (nvrtc, hiprtc).
//  - CUDA Tile kernels (__tile__, __tile_global__), which add two rows to NVIDIA's
//    qualifier table (A7) and are not used in this course.
//  - Whether a cubin built for one GPU architecture also runs on another. The
//    facts file has no statement on it, so the model does not answer it.
//  - The host-side launch stub a compiler generates for a kernel: a kernel is
//    counted as device code only.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

import { tr } from '../lib/locale.ts';

export type Backend = 'cuda' | 'hip' | 'metal';
export const BACKENDS: readonly Backend[] = ['cuda', 'hip', 'metal'];
export const BACKEND_NAME: Record<Backend, string> = { cuda: 'CUDA', hip: 'HIP', metal: 'Metal' };

// ---------------------------------------------------------------- functions and qualifiers
/** The qualifier sets a function can carry. 'none' and 'host' mean the same thing (Fig. 2.11). */
export type Qualifiers = 'none' | 'host' | 'device' | 'global' | 'host-device';
export const QUALIFIERS: readonly Qualifiers[] = ['none', 'host', 'device', 'global', 'host-device'];
export const QUALIFIER_TEXT: Record<Qualifiers, string> = {
  get none() {
    return tr('no qualifier', 'không có qualifier');
  },
  host: '__host__',
  device: '__device__',
  global: '__global__',
  'host-device': '__host__ __device__', // i18n-ignore: code (the qualifiers as written)
};
/** The side of the program a compiled function belongs to. */
export type Space = 'host' | 'device';

/** Where a function with these qualifiers is compiled for: the host, the device, or both. */
export function spacesOf(q: Qualifiers): Space[] {
  if (q === 'none' || q === 'host') return ['host'];
  if (q === 'host-device') return ['host', 'device'];
  return ['device'];
}

export interface Fn {
  name: string;
  quals: Qualifiers;
  does: string;
}
/** The vector addition program of the chapter, with one shared helper added. */
export const PROGRAM: readonly Fn[] = [
  {
    name: 'main', // i18n-ignore: function name in the lab source
    quals: 'none',
    get does() {
      return tr('fills the arrays and calls vecAdd', 'điền dữ liệu vào các mảng và gọi vecAdd');
    },
  },
  {
    name: 'vecAdd',
    quals: 'none',
    get does() {
      return tr('the host stub: allocates, copies, launches the kernel, copies back', 'hàm stub phía host: cấp phát, sao chép, khởi chạy kernel, sao chép ngược lại');
    },
  },
  {
    name: 'vecAddKernel',
    quals: 'global',
    get does() {
      return tr('the kernel: one addition per thread', 'kernel: một phép cộng cho mỗi luồng');
    },
  },
  {
    name: 'add1', // i18n-ignore: function name in the lab source
    quals: 'host-device',
    get does() {
      return tr('a one-line helper called by the kernel and by the CPU reference', 'một hàm trợ giúp dài một dòng, được kernel và bản tham chiếu CPU gọi');
    },
  },
];

export interface Placement {
  space: Space;
  /** The compiler, or the compiler pass, that sees the function. */
  compiler: string;
  /** On Metal: the file the function has to be written in. */
  file?: string;
  /** On Metal, for a function wanted on both sides: it is a second, hand-written copy. */
  writtenAgain?: boolean;
  facts: string[];
}

const COMPILER: Record<Backend, Record<Space, { compiler: string; facts: string[] }>> = {
  cuda: {
    host: {
      get compiler() {
        return tr('the host C++ compiler, called by nvcc', 'trình biên dịch C++ của host, do nvcc gọi');
      },
      facts: ['A14'],
    },
    device: {
      get compiler() {
        return tr('nvcc\'s GPU compiler', 'trình biên dịch GPU của nvcc');
      },
      facts: ['A14'],
    },
  },
  hip: {
    host: {
      get compiler() {
        return tr('amdclang++, host pass', 'amdclang++, lượt biên dịch cho host');
      },
      facts: ['B4', 'B6'],
    },
    device: {
      get compiler() {
        return tr('amdclang++, one device pass per architecture', 'amdclang++, một lượt biên dịch cho device trên mỗi kiến trúc');
      },
      facts: ['B4', 'B6'],
    },
  },
  metal: {
    host: {
      get compiler() {
        return tr('the host compiler', 'trình biên dịch của host');
      },
      facts: ['C1'],
    },
    device: {
      get compiler() {
        return tr('the Metal compiler', 'trình biên dịch Metal');
      },
      facts: ['C1', 'C6'],
    },
  },
};
export const METAL_FILE: Record<Space, string> = { host: 'launch.mm', device: 'kernel.metal' };

/**
 * Which compilation(s) see a function. On CUDA and HIP the qualifiers decide, and
 * __host__ __device__ gives two. Metal has no qualifiers (C7): the same intent is
 * expressed by which file the function is written in, and a function wanted on
 * both sides is written twice.
 */
export function placement(fn: Fn, backend: Backend): Placement[] {
  const spaces = spacesOf(fn.quals);
  return spaces.map((space, k): Placement => {
    const c = COMPILER[backend][space];
    if (backend !== 'metal') return { space, compiler: c.compiler, facts: fn.quals === 'host-device' ? [...c.facts, 'A7'] : c.facts };
    return {
      space,
      compiler: c.compiler,
      file: METAL_FILE[space],
      ...(spaces.length > 1 && k > 0 ? { writtenAgain: true } : {}),
      facts: spaces.length > 1 ? [...c.facts, 'C7'] : c.facts,
    };
  });
}

// ---------------------------------------------------------------- stages
export type When = 'build' | 'run';
export interface Chip {
  fn: string;
  /** Which compiled version of the function this is, or null while it is still source. */
  copy: Space | null;
}
export interface Box {
  id: string;
  /** Boxes with the same group are drawn inside one frame with this title. */
  group?: string;
  title: string;
  /** What the contents are at this point: source, PTX, machine code for an architecture. */
  form: string;
  chips: Chip[];
  /** 'fail': this cannot run on the device at hand. 'unused': present but not what runs. */
  state?: 'ok' | 'fail' | 'unused';
}
export interface Stage {
  id: string;
  when: When;
  /** The tool or call that does this stage. */
  tool: string;
  title: string;
  text: string;
  boxes: Box[];
  facts: string[];
}
export interface Options {
  /** CUDA: the fatbin holds a cubin that suits the GPU the program runs on. Default true. */
  cubinFits?: boolean;
  /** HIP: the architecture names given to --offload-arch. Default none. */
  archs?: string[];
  /** HIP: the architecture of the GPU the program runs on. */
  deviceArch?: string;
  /** Metal: when the library is built. Default 'run', which is what the course's labs do. */
  metalLibrary?: 'build' | 'run';
}

const chipsFor = (program: readonly Fn[], space: Space, compiled: boolean): Chip[] =>
  program.filter((f) => spacesOf(f.quals).includes(space)).map((f) => ({ fn: f.name, copy: compiled ? space : null }));
const sourceChips = (program: readonly Fn[]): Chip[] => program.map((f) => ({ fn: f.name, copy: null }));
/** Trimmed, non-empty, no repeats, in the order typed. */
export const cleanArchs = (archs: string[] = []): string[] => [...new Set(archs.map((a) => a.trim()).filter(Boolean))];

function cudaStages(o: Options, program: readonly Fn[]): Stage[] {
  const host = chipsFor(program, 'host', true);
  const dev = chipsFor(program, 'device', true);
  const fits = o.cubinFits !== false;
  const exe = tr('the executable', 'tệp thực thi');
  return [
    {
      id: 'source', when: 'build', tool: tr('you', 'bạn'), title: tr('One source file', 'Một tệp mã nguồn'),
      text: tr('One .cu file holds host code and device code side by side. Only the qualifiers say which is which.', 'Một tệp .cu chứa mã host và mã device nằm cạnh nhau. Chỉ các qualifier cho biết phần nào là phần nào.'),
      boxes: [{ id: 'cu', title: 'vecadd.cu', form: tr('CUDA C++ source', 'mã nguồn CUDA C++'), chips: sourceChips(program) }],
      facts: ['A14'],
    },
    {
      id: 'split', when: 'build', tool: 'nvcc', title: tr('nvcc splits the file', 'nvcc tách tệp'),
      text: tr('nvcc separates device code from host code and sends each to its own compiler. A function marked __host__ __device__ goes both ways, so from here on there are two of it.', 'nvcc tách mã device khỏi mã host và gửi mỗi phần đến trình biên dịch riêng của nó. Một hàm được đánh dấu __host__ __device__ đi theo cả hai đường, nên từ đây trở đi có hai bản của nó.'),
      boxes: [
        { id: 'host-src', title: tr('host code', 'mã host'), form: tr('plain C++', 'C++ thông thường'), chips: chipsFor(program, 'host', true) },
        { id: 'dev-src', title: tr('device code', 'mã device'), form: tr('C++ for the GPU compiler', 'C++ cho trình biên dịch GPU'), chips: chipsFor(program, 'device', true) },
      ],
      facts: ['A14', 'A7'],
    },
    {
      id: 'compile', when: 'build', tool: tr('host compiler and GPU compiler', 'trình biên dịch host và trình biên dịch GPU'), title: tr('Host object and PTX', 'Tệp object của host và PTX'),
      text: tr('The host compiler turns the host code into an ordinary object file. The GPU compiler turns the device code into PTX: assembly for a virtual GPU instruction set, not yet for any real one.', 'Trình biên dịch host biến mã host thành một tệp object thông thường. Trình biên dịch GPU biến mã device thành PTX: hợp ngữ cho một tập lệnh GPU ảo, chưa dành cho tập lệnh thật nào.'),
      boxes: [
        { id: 'host-obj', title: tr('host object', 'tệp object của host'), form: tr('CPU machine code', 'mã máy CPU'), chips: host },
        { id: 'ptx', title: 'PTX', form: tr('virtual GPU instruction set', 'tập lệnh GPU ảo'), chips: dev },
      ],
      facts: ['A14'],
    },
    {
      id: 'cubin', when: 'build', tool: 'ptxas', title: tr('ptxas makes a cubin', 'ptxas tạo một cubin'),
      text: tr('ptxas compiles the PTX to a cubin: machine code for a real GPU instruction set. By default the target is the earliest architecture the toolkit supports. The PTX is kept.', 'ptxas biên dịch PTX thành một cubin: mã máy cho một tập lệnh GPU thật. Mặc định, đích là kiến trúc sớm nhất mà toolkit hỗ trợ. PTX vẫn được giữ lại.'),
      boxes: [
        { id: 'host-obj', title: tr('host object', 'tệp object của host'), form: tr('CPU machine code', 'mã máy CPU'), chips: host },
        { id: 'ptx', title: 'PTX', form: tr('virtual GPU instruction set', 'tập lệnh GPU ảo'), chips: dev },
        { id: 'cubin', title: tr('cubin', 'cubin'), form: tr('GPU machine code for one architecture', 'mã máy GPU cho một kiến trúc'), chips: dev },
      ],
      facts: ['A14'],
    },
    {
      id: 'fatbin', when: 'build', tool: 'nvcc', title: tr('A fatbin inside the executable', 'Một fatbin bên trong tệp thực thi'),
      text: tr('PTX and cubin are packed together in a fatbin, and the fatbin travels inside the executable beside the host machine code. This is what you ship.', 'PTX và cubin được đóng gói cùng nhau trong một fatbin, và fatbin đi bên trong tệp thực thi, cạnh mã máy của host. Đây là thứ bạn phát hành.'),
      boxes: [
        { id: 'host-code', group: exe, title: tr('host code', 'mã host'), form: tr('CPU machine code', 'mã máy CPU'), chips: host },
        { id: 'ptx', group: exe, title: tr('fatbin: PTX', 'fatbin: PTX'), form: tr('virtual GPU instruction set', 'tập lệnh GPU ảo'), chips: dev },
        { id: 'cubin', group: exe, title: tr('fatbin: cubin', 'fatbin: cubin'), form: tr('GPU machine code for one architecture', 'mã máy GPU cho một kiến trúc'), chips: dev },
      ],
      facts: ['A14'],
    },
    {
      id: 'run', when: 'run', tool: fits ? tr('the CUDA runtime', 'CUDA runtime') : tr('the device driver', 'driver của thiết bị'), title: fits ? tr('Run time: the cubin is loaded', 'Lúc chạy: cubin được nạp') : tr('Run time: the driver compiles the PTX', 'Lúc chạy: driver biên dịch PTX'),
      text: fits
        ? tr('The host code runs as a normal process. The fatbin holds a cubin that suits this GPU, so it is loaded as it is and nothing is compiled.', 'Mã host chạy như một tiến trình bình thường. Fatbin chứa một cubin phù hợp với GPU này, nên nó được nạp nguyên trạng và không có gì được biên dịch.')
        : tr('The host code runs as a normal process. No cubin in the fatbin suits this GPU, so the device driver compiles the PTX for it, just in time, and keeps a copy of the result. This is how a binary built today can run on a GPU that did not exist at build time.', 'Mã host chạy như một tiến trình bình thường. Không cubin nào trong fatbin phù hợp với GPU này, nên driver của thiết bị biên dịch PTX cho nó, ngay lúc cần (just in time), và giữ lại một bản sao của kết quả. Đây là cách một tệp nhị phân build hôm nay có thể chạy trên một GPU chưa tồn tại lúc build.'),
      boxes: [
        { id: 'host-code', title: tr('host process', 'tiến trình host'), form: tr('CPU machine code, running', 'mã máy CPU, đang chạy'), chips: host, state: 'ok' },
        { id: 'ptx', title: tr('fatbin: PTX', 'fatbin: PTX'), form: fits ? tr('not needed this time', 'lần này không cần') : tr('compiled now by the driver', 'được driver biên dịch ngay lúc này'), chips: dev, state: fits ? 'unused' : 'ok' },
        fits
          ? { id: 'cubin', title: tr('fatbin: cubin', 'fatbin: cubin'), form: tr('loaded into device memory', 'được nạp vào bộ nhớ device'), chips: dev, state: 'ok' }
          : { id: 'jit', title: tr('driver output', 'đầu ra của driver'), form: tr('GPU machine code for this GPU', 'mã máy GPU cho GPU này'), chips: dev, state: 'ok' },
      ],
      facts: ['A14', 'A5'],
    },
  ];
}

function hipStages(o: Options, program: readonly Fn[]): Stage[] {
  const host = chipsFor(program, 'host', true);
  const dev = chipsFor(program, 'device', true);
  const archs = cleanArchs(o.archs);
  const device = (o.deviceArch ?? '').trim();
  const exe = tr('the executable', 'tệp thực thi');
  const archBoxes = (group?: string): Box[] =>
    archs.map((a) => ({ id: 'arch-' + a, ...(group ? { group } : {}), title: tr('device code: ' + a, 'mã device: ' + a), form: tr('GPU machine code for ' + a, 'mã máy GPU cho ' + a), chips: dev }));
  const listed = archs.includes(device);
  const none = archs.length === 0;
  return [
    {
      id: 'source', when: 'build', tool: tr('you', 'bạn'), title: tr('One source file', 'Một tệp mã nguồn'),
      text: tr('The same single file as on CUDA, with the same qualifiers. It includes hip/hip_runtime.h, which HIP requires for the built-in variables.', 'Vẫn một tệp duy nhất như trên CUDA, với cùng các qualifier. Nó include hip/hip_runtime.h, tệp mà HIP yêu cầu cho các biến dựng sẵn.'),
      boxes: [{ id: 'src', title: 'vecadd.hip', form: tr('HIP C++ source', 'mã nguồn HIP C++'), chips: sourceChips(program) }],
      facts: ['B4', 'ch05 B6'], // i18n-ignore: fact ids
    },
    {
      id: 'host-pass', when: 'build', tool: 'amdclang++', title: tr('Host pass', 'Lượt biên dịch cho host'),
      text: tr('amdclang++ makes one pass over the file for the host code. There is no second compiler: one Clang-based compiler does both halves.', 'amdclang++ chạy một lượt qua tệp cho mã host. Không có trình biên dịch thứ hai: một trình biên dịch dựa trên Clang làm cả hai nửa.'),
      boxes: [{ id: 'host-obj', title: tr('host object', 'tệp object của host'), form: tr('CPU machine code', 'mã máy CPU'), chips: host }],
      facts: ['B4', 'B6'],
    },
    {
      id: 'device-passes', when: 'build', tool: 'amdclang++ --offload-arch', title: none ? tr('Device passes: no architecture named yet', 'Các lượt biên dịch cho device: chưa nêu kiến trúc nào') : tr(`Device ${archs.length === 1 ? 'pass' : 'passes'}: ${archs.join(', ')}`, `Lượt biên dịch cho device: ${archs.join(', ')}`),
      text: none
        ? tr('The compiler makes one more pass over the device code for each GPU architecture named with --offload-arch. Type at least one name above to see it.', 'Trình biên dịch chạy thêm một lượt qua mã device cho mỗi kiến trúc GPU được nêu bằng --offload-arch. Hãy gõ ít nhất một tên ở trên để thấy điều đó.')
        : tr(
            `The compiler makes one more pass over the device code for each GPU architecture named with --offload-arch: here ${archs.length}. Each pass ends in machine code for that architecture. There is no stage like PTX in between.`,
            `Trình biên dịch chạy thêm một lượt qua mã device cho mỗi kiến trúc GPU được nêu bằng --offload-arch: ở đây là ${archs.length}. Mỗi lượt kết thúc bằng mã máy cho kiến trúc đó. Không có giai đoạn nào giống PTX ở giữa.`,
          ),
      boxes: [{ id: 'host-obj', title: tr('host object', 'tệp object của host'), form: tr('CPU machine code', 'mã máy CPU'), chips: host }, ...archBoxes()],
      facts: ['B4', 'B6'],
    },
    {
      id: 'embed', when: 'build', tool: 'amdclang++', title: tr('Device code embedded', 'Mã device được nhúng'),
      text: tr('The compiled device code is embedded into the host object file. What ships is machine code for the named architectures and nothing a driver could compile later.', 'Mã device đã biên dịch được nhúng vào tệp object của host. Thứ được phát hành là mã máy cho các kiến trúc đã nêu, và không có gì mà một driver có thể biên dịch sau này.'),
      boxes: [{ id: 'host-code', group: exe, title: tr('host code', 'mã host'), form: tr('CPU machine code', 'mã máy CPU'), chips: host }, ...archBoxes(exe)],
      facts: ['B6'],
    },
    {
      id: 'run', when: 'run', tool: tr('the HIP runtime', 'HIP runtime'),
      title: device === '' ? tr('Run time: which GPU?', 'Lúc chạy: GPU nào?') : listed ? tr(`Run time on ${device}: runs`, `Lúc chạy trên ${device}: chạy được`) : tr(`Run time on ${device}: not runnable`, `Lúc chạy trên ${device}: không chạy được`),
      text:
        device === ''
          ? tr('Type the architecture of the GPU the program runs on to see what happens there.', 'Hãy gõ kiến trúc của GPU mà chương trình chạy trên đó để xem điều gì xảy ra ở đó.')
          : listed
            ? tr(
                `The binary carries machine code for ${device}, so the kernel is loaded and runs. Nothing is compiled at run time.`,
                `Tệp nhị phân mang mã máy cho ${device}, nên kernel được nạp và chạy. Không có gì được biên dịch lúc chạy.`,
              )
            : tr(
                `The binary carries no machine code for ${device}. Machine code for one AMD GPU architecture runs only on the physical architectures it is compatible with, and there is no portable form to compile from: the fix is to rebuild with ${device} in the list.`,
                `Tệp nhị phân không mang mã máy nào cho ${device}. Mã máy cho một kiến trúc GPU AMD chỉ chạy trên những kiến trúc vật lý tương thích với nó, và không có dạng khả chuyển nào để biên dịch từ đó: cách sửa là build lại với ${device} trong danh sách.`,
              ),
      boxes: [
        { id: 'host-code', title: tr('host process', 'tiến trình host'), form: tr('CPU machine code, running', 'mã máy CPU, đang chạy'), chips: host, state: 'ok' },
        ...archs.map((a): Box => ({
          id: 'arch-' + a, title: tr('device code: ' + a, 'mã device: ' + a), form: a === device ? tr('loaded on this GPU', 'được nạp trên GPU này') : tr('not for this GPU', 'không dành cho GPU này'), chips: dev,
          state: a === device ? 'ok' : 'unused',
        })),
        ...(device !== '' && !listed ? [{ id: 'missing', title: tr('device code: ' + device, 'mã device: ' + device), form: tr('not in the binary', 'không có trong tệp nhị phân'), chips: [], state: 'fail' } as Box] : []),
      ],
      facts: ['B6'],
    },
  ];
}

function metalStages(o: Options, program: readonly Fn[]): Stage[] {
  const host = chipsFor(program, 'host', true);
  const dev = chipsFor(program, 'device', true);
  const atBuild = o.metalLibrary === 'build';
  return [
    {
      id: 'source', when: 'build', tool: tr('you', 'bạn'), title: tr('Two source files from the start', 'Hai tệp mã nguồn ngay từ đầu'),
      text: tr('There is nothing to split. Host code is in an ordinary C++ or Objective-C++ file. The kernel is in a .metal file, written in Metal Shading Language. A helper wanted on both sides has to be written in both files.', 'Không có gì để tách. Mã host nằm trong một tệp C++ hoặc Objective-C++ thông thường. Kernel nằm trong một tệp .metal, viết bằng Metal Shading Language. Một hàm trợ giúp cần ở cả hai phía phải được viết trong cả hai tệp.'),
      boxes: [
        { id: 'host-src', title: METAL_FILE.host, form: tr('Objective-C++ source', 'mã nguồn Objective-C++'), chips: chipsFor(program, 'host', true) },
        { id: 'dev-src', title: METAL_FILE.device, form: tr('Metal Shading Language source', 'mã nguồn Metal Shading Language'), chips: chipsFor(program, 'device', true) },
      ],
      facts: ['C1', 'C7'],
    },
    {
      id: 'library', when: atBuild ? 'build' : 'run', tool: atBuild ? 'xcrun metal' : 'newLibraryWithSource:options:error:',
      title: atBuild ? tr('Build time: a .metallib', 'Lúc build: một .metallib') : tr('Run time: a library from source', 'Lúc chạy: một thư viện từ mã nguồn'),
      text: atBuild
        ? tr('The metal compiler turns kernel.metal into an intermediate representation and links it into a .metallib, which the app loads when it runs. The host file is compiled by the host compiler as usual.', 'Trình biên dịch metal biến kernel.metal thành một biểu diễn trung gian và liên kết nó vào một .metallib, tệp mà ứng dụng nạp khi chạy. Tệp host được trình biên dịch host biên dịch như thường lệ.')
        : tr('The host file was compiled by the host compiler as usual and carries the kernel source as text. When the program runs, newLibraryWithSource compiles that text into a library. This is the route the labs in this course take.', 'Tệp host đã được trình biên dịch host biên dịch như thường lệ và mang mã nguồn kernel dưới dạng văn bản. Khi chương trình chạy, newLibraryWithSource biên dịch văn bản đó thành một thư viện. Đây là con đường mà các lab trong khóa học này đi theo.'),
      boxes: [
        { id: 'host-code', title: tr('host executable', 'tệp thực thi host'), form: tr('CPU machine code', 'mã máy CPU'), chips: host },
        { id: 'library', title: atBuild ? 'default.metallib' : tr('library object', 'đối tượng thư viện'), form: tr('intermediate representation, not for any one GPU', 'biểu diễn trung gian, không dành cho một GPU cụ thể nào'), chips: dev },
      ],
      facts: ['C6'],
    },
    {
      id: 'function', when: 'run', tool: 'newFunctionWithName:', title: tr('Run time: a function object', 'Lúc chạy: một đối tượng hàm'),
      text: tr('The app asks the library for the kernel by name and gets a function object. It stands for the MSL function and is not executable code yet.', 'Ứng dụng yêu cầu thư viện cung cấp kernel theo tên và nhận về một đối tượng hàm. Nó đại diện cho hàm MSL và chưa phải là mã thực thi được.'),
      boxes: [
        { id: 'host-code', title: tr('host process', 'tiến trình host'), form: tr('CPU machine code, running', 'mã máy CPU, đang chạy'), chips: host, state: 'ok' },
        { id: 'function', title: tr('function object', 'đối tượng hàm'), form: tr('a proxy for the kernel, not executable', 'một đại diện cho kernel, không thực thi được'), chips: dev.filter((c) => program.find((f) => f.name === c.fn)?.quals === 'global') },
      ],
      facts: ['C6', 'C5'],
    },
    {
      id: 'pipeline', when: 'run', tool: 'newComputePipelineStateWithFunction:error:', title: tr('Run time: the pipeline state', 'Lúc chạy: pipeline state'),
      text: tr('Creating the pipeline state finishes compiling the kernel for this specific GPU. It happens on the machine that runs the program, in both routes, and it takes a while: do it once, outside anything you time.', 'Việc tạo pipeline state hoàn tất biên dịch kernel cho đúng GPU này. Nó diễn ra trên máy chạy chương trình, ở cả hai con đường, và mất một lúc: hãy làm việc này một lần, bên ngoài mọi đoạn bạn đo thời gian.'),
      boxes: [
        { id: 'host-code', title: tr('host process', 'tiến trình host'), form: tr('CPU machine code, running', 'mã máy CPU, đang chạy'), chips: host, state: 'ok' },
        { id: 'pipeline', title: tr('pipeline state', 'pipeline state'), form: tr('GPU machine code for this GPU', 'mã máy GPU cho GPU này'), chips: dev, state: 'ok' },
      ],
      facts: ['C6', 'C4'],
    },
  ];
}

/** The documented stages of one backend's toolchain, with where each function of the program is after each. */
export function stages(backend: Backend, options: Options = {}, program: readonly Fn[] = PROGRAM): Stage[] {
  if (backend === 'cuda') return cudaStages(options, program);
  if (backend === 'hip') return hipStages(options, program);
  return metalStages(options, program);
}

/** Does any stage after shipping compile device code on the machine that runs the program? */
export function compilesOnUsersMachine(backend: Backend): 'always' | 'when-needed' | 'never' {
  if (backend === 'metal') return 'always';
  return backend === 'hip' ? 'never' : 'when-needed';
}

// ---------------------------------------------------------------- a GPU architecture that appears later
export interface Artefacts {
  /** CUDA: the shipped fatbin carries PTX (the default, A14). */
  ptx?: boolean;
  /** HIP: the architectures named at build time. */
  archs?: string[];
  /** HIP: the name of the architecture that appeared later. */
  newArch?: string;
}
export interface NewArchVerdict {
  /** Can the shipped binary's kernel be compiled for the new architecture without a rebuild? */
  runs: 'yes' | 'no' | 'not-documented';
  why: string;
  facts: string[];
}
/** The three documented answers to "a new GPU architecture appears after you ship". */
export function canRunOnNewArchitecture(backend: Backend, a: Artefacts = {}): NewArchVerdict {
  if (backend === 'cuda')
    return a.ptx === false
      ? { runs: 'not-documented', why: tr('Without PTX there is nothing for the driver to compile. Whether a cubin built for an older architecture runs is not covered by the facts checked for this course.', 'Không có PTX thì driver không có gì để biên dịch. Một cubin build cho kiến trúc cũ hơn có chạy được hay không thì không nằm trong các dữ kiện đã kiểm tra cho khóa học này.'), facts: ['A14'] }
      : { runs: 'yes', why: tr('The fatbin carries PTX, and the device driver can compile PTX just in time for a target that did not exist at build time.', 'Fatbin mang PTX, và driver của thiết bị có thể biên dịch PTX ngay lúc cần (just in time) cho một đích chưa tồn tại lúc build.'), facts: ['A14'] };
  if (backend === 'hip') {
    const named = cleanArchs(a.archs).includes((a.newArch ?? '').trim()) && (a.newArch ?? '').trim() !== '';
    return named
      ? { runs: 'yes', why: tr('The architecture was named at build time, so its machine code is in the binary.', 'Kiến trúc đã được nêu lúc build, nên mã máy của nó có trong tệp nhị phân.'), facts: ['B6'] }
      : { runs: 'no', why: tr('A HIP binary holds machine code for the architectures named with --offload-arch and no portable form a driver could compile later. A new architecture needs a rebuild.', 'Một tệp nhị phân HIP chứa mã máy cho các kiến trúc được nêu bằng --offload-arch và không có dạng khả chuyển nào mà một driver có thể biên dịch sau này. Một kiến trúc mới cần build lại.'), facts: ['B6'] };
  }
  return { runs: 'yes', why: tr('Compilation for the specific GPU always happens when the pipeline state is created, on the machine that runs the program.', 'Việc biên dịch cho GPU cụ thể luôn diễn ra khi pipeline state được tạo, trên máy chạy chương trình.'), facts: ['C6'] };
}

// ---------------------------------------------------------------- who may call, who runs (book Fig. 2.11)
export type Caller = 'host-fn' | 'kernel' | 'device-fn';
export const CALLERS: readonly Caller[] = ['host-fn', 'kernel', 'device-fn'];
export const CALLER_TEXT: Record<Caller, string> = {
  get 'host-fn'() {
    return tr('a host function', 'một hàm host');
  },
  get kernel() {
    return tr('a kernel', 'một kernel');
  },
  get 'device-fn'() {
    return tr('a device function', 'một hàm device');
  },
};
/** The four columns of the classification: no qualifier and __host__ are one column, as in the book's table. */
export const CALLEES: readonly Qualifiers[] = ['none', 'global', 'device', 'host-device'];
export const CALLEE_TEXT: Record<Qualifiers, string> = {
  get none() {
    return tr('no qualifier, or __host__', 'không có qualifier, hoặc __host__');
  },
  host: '__host__',
  global: '__global__',
  device: '__device__',
  'host-device': '__host__ __device__', // i18n-ignore: code (the qualifiers as written)
};

export type Executor = 'caller-host-thread' | 'caller-device-thread' | 'new-grid';
export const EXECUTORS: readonly Executor[] = ['caller-host-thread', 'caller-device-thread', 'new-grid'];
export const EXECUTOR_TEXT: Record<Executor, string> = {
  get 'caller-host-thread'() {
    return tr('the calling host thread', 'luồng host gọi hàm');
  },
  get 'caller-device-thread'() {
    return tr('the calling device thread', 'luồng device gọi hàm');
  },
  get 'new-grid'() {
    return tr('a new grid of device threads', 'một lưới luồng device mới');
  },
};
/** 'dynamic-parallelism': legal only where the backend lets device code launch a kernel. */
export type Legality = 'legal' | 'illegal' | 'dynamic-parallelism';

export interface DynamicParallelism {
  /** true, false, or null where the vendor documents nothing like it. */
  supported: boolean | null;
  text: string;
  fact: string;
}
export const DYNAMIC_PARALLELISM: Record<Backend, DynamicParallelism> = {
  cuda: {
    supported: true,
    get text() {
      return tr('supported: the current version is called CDP2', 'được hỗ trợ: phiên bản hiện tại gọi là CDP2');
    },
    fact: 'A13',
  },
  hip: {
    supported: false,
    get text() {
      return tr('not supported: a kernel cannot be called from the device', 'không được hỗ trợ: không thể gọi kernel từ device');
    },
    fact: 'ch05 B4', // i18n-ignore: fact id
  },
  metal: {
    supported: null,
    get text() {
      return tr('none documented: MSL has no dispatch call and kernels cannot call kernels', 'không có tài liệu nào: MSL không có lời gọi dispatch và kernel không thể gọi kernel');
    },
    fact: 'C7',
  },
};

const callerSpace = (c: Caller): Space => (c === 'host-fn' ? 'host' : 'device');

/** Who executes the callee when the call is allowed by the book's table; null when it is not. */
export function executedBy(caller: Caller, callee: Qualifiers): Executor | null {
  const from = callerSpace(caller);
  if (callee === 'global') return 'new-grid';
  if (!spacesOf(callee).includes(from)) return null;
  return from === 'host' ? 'caller-host-thread' : 'caller-device-thread';
}

export interface CallRule {
  caller: Caller;
  callee: Qualifiers;
  backend: Backend;
  /** False on Metal, where these qualifiers do not exist (C7): use metalCall() there. */
  applies: boolean;
  /** What the book's table says. */
  book: Legality;
  /** Whether the call compiles on this backend; null where the rule does not apply. */
  legal: boolean | null;
  /** Who executes the callee when the call is legal. */
  executedBy: Executor | null;
  /** The row of Fig. 2.11 the answer is read from. */
  row: string;
  why: string;
  facts: string[];
}

const ROW: Record<Qualifiers, string> = {
  get none() {
    return tr('__host__ (default): callable from the host, executed on the host by the caller host thread', '__host__ (mặc định): gọi được từ host, chạy trên host bởi luồng host gọi hàm');
  },
  get host() {
    return tr('__host__ (default): callable from the host, executed on the host by the caller host thread', '__host__ (mặc định): gọi được từ host, chạy trên host bởi luồng host gọi hàm');
  },
  get global() {
    return tr('__global__: callable from the host (or the device), executed on the device by a new grid of device threads', '__global__: gọi được từ host (hoặc từ device), chạy trên device bởi một lưới luồng device mới');
  },
  get device() {
    return tr('__device__: callable from the device, executed on the device by the caller device thread', '__device__: gọi được từ device, chạy trên device bởi luồng device gọi hàm');
  },
  get 'host-device'() {
    return tr('__host__ and __device__ together: one version compiled for each side (p. 36)', '__host__ và __device__ cùng nhau: mỗi phía có một phiên bản được biên dịch (p. 36)');
  },
};

/** One cell of the caller by callee table: book Fig. 2.11, resolved for one backend. */
export function legalCall(caller: Caller, callee: Qualifiers, backend: Backend = 'cuda'): CallRule {
  const from = callerSpace(caller);
  const who = executedBy(caller, callee);
  const base = { caller, callee, backend, row: ROW[callee] };
  const applies = backend !== 'metal';
  const finish = (book: Legality, why: string, facts: string[]): CallRule => {
    const legal = !applies ? null : book === 'legal' ? true : book === 'illegal' ? false : DYNAMIC_PARALLELISM[backend].supported === true;
    return {
      ...base,
      applies,
      book,
      legal,
      executedBy: book === 'illegal' ? null : who,
      why: applies ? why : tr('Metal has no such qualifiers: every function in a .metal file is GPU code, and host functions live in another file and another language.', 'Metal không có các qualifier như vậy: mọi hàm trong một tệp .metal là mã GPU, còn các hàm host nằm trong một tệp khác và một ngôn ngữ khác.'),
      facts: applies ? facts : ['C7'],
    };
  };
  if (callee === 'global') {
    if (from === 'host')
      return finish('legal', tr('Calling a kernel from host code is the launch: it starts a new grid of device threads, and every one of them runs the kernel.', 'Gọi một kernel từ mã host chính là khởi chạy: nó bắt đầu một lưới luồng device mới, và mọi luồng trong đó đều chạy kernel.'), ['Fig. 2.11', 'A7']);
    return finish(
      'dynamic-parallelism',
      tr('Device code launching a kernel is dynamic parallelism. Where the backend supports it the call starts a new grid; where it does not, the call is an error.', 'Mã device khởi chạy một kernel là dynamic parallelism. Ở backend hỗ trợ nó, lời gọi bắt đầu một lưới mới; ở backend không hỗ trợ, lời gọi là một lỗi.'),
      ['Fig. 2.11', 'A13', 'ch05 B4', 'C7'], // i18n-ignore: fact ids
    );
  }
  if (callee === 'none' || callee === 'host') {
    if (from === 'host') return finish('legal', tr('An ordinary C++ call: the calling host thread runs the function.', 'Một lời gọi C++ thông thường: luồng host gọi hàm chạy hàm đó.'), ['Fig. 2.11', 'A7']);
    return finish(
      'illegal',
      tr('No qualifier means __host__. A host function exists only as CPU code, so device code cannot call it. This is the usual first compile error: mark the helper __device__, or __host__ __device__ if the host needs it too.', 'Không có qualifier nghĩa là __host__. Một hàm host chỉ tồn tại dưới dạng mã CPU, nên mã device không thể gọi nó. Đây là lỗi biên dịch đầu tiên thường gặp: hãy đánh dấu hàm trợ giúp là __device__, hoặc __host__ __device__ nếu host cũng cần nó.'),
      ['Fig. 2.11', 'A7'],
    );
  }
  if (callee === 'device') {
    if (from === 'device') return finish('legal', tr('The calling device thread runs the function itself. No thread is created.', 'Luồng device gọi hàm tự chạy hàm đó. Không có luồng nào được tạo.'), ['Fig. 2.11', 'A7']);
    return finish('illegal', tr('A __device__ function exists only as GPU code. Host code cannot call it.', 'Một hàm __device__ chỉ tồn tại dưới dạng mã GPU. Mã host không thể gọi nó.'), ['Fig. 2.11', 'A7']);
  }
  return finish(
    'legal',
    from === 'host'
      ? tr('The compiler made two versions of this function. A host caller gets the host version, run by the calling host thread.', 'Trình biên dịch đã tạo hai phiên bản của hàm này. Bên gọi phía host nhận phiên bản host, do luồng host gọi hàm chạy.')
      : tr('The compiler made two versions of this function. A device caller gets the device version, run by the calling device thread.', 'Trình biên dịch đã tạo hai phiên bản của hàm này. Bên gọi phía device nhận phiên bản device, do luồng device gọi hàm chạy.'),
    ['p. 36', 'A7'],
  );
}

/** What a reader may answer for one cell: it does not compile, or who executes the callee. */
export type Answer = 'error' | Executor;
export const ANSWERS: readonly Answer[] = ['error', ...EXECUTORS];
// The spread copies EXECUTOR_TEXT's getters once, in the language at import time, so the executor
// entries here stay English. Nothing renders ANSWER_TEXT today; render EXECUTOR_TEXT[answer] instead.
export const ANSWER_TEXT: Record<Answer, string> = { get error() { return tr('compile error', 'lỗi biên dịch'); }, ...EXECUTOR_TEXT };

/**
 * Marks an answer against the book's table. The dynamic-parallelism cells have two
 * defensible answers (an error without the feature, a new grid with it): both are
 * 'depends', and anything else is wrong.
 */
export function grade(caller: Caller, callee: Qualifiers, answer: Answer): 'right' | 'wrong' | 'depends' {
  const rule = legalCall(caller, callee, 'cuda');
  if (rule.book === 'dynamic-parallelism') return answer === 'error' || answer === 'new-grid' ? 'depends' : 'wrong';
  if (rule.book === 'illegal') return answer === 'error' ? 'right' : 'wrong';
  return answer === rule.executedBy ? 'right' : 'wrong';
}

// ---------------------------------------------------------------- the same question on Metal
export type MetalFn = 'host' | 'kernel' | 'msl';
export const METAL_FNS: readonly MetalFn[] = ['host', 'kernel', 'msl'];
export const METAL_FN_TEXT: Record<MetalFn, string> = {
  get host() {
    return tr('a host function (in the .mm or .cpp file)', 'một hàm host (trong tệp .mm hoặc .cpp)');
  },
  get kernel() {
    return tr('a kernel function (kernel void, in the .metal file)', 'một hàm kernel (kernel void, trong tệp .metal)');
  },
  get msl() {
    return tr('a plain function in the .metal file', 'một hàm thường trong tệp .metal');
  },
};
export interface MetalCall {
  caller: MetalFn;
  callee: MetalFn;
  /** 'dispatch': not a call, but the host can run it through a pipeline and an encoder. */
  result: 'call' | 'dispatch' | 'no';
  executedBy: Executor | null;
  why: string;
  facts: string[];
}
/** Who can reach what on Metal, where the file a function is written in plays the part of the qualifier. */
export function metalCall(caller: MetalFn, callee: MetalFn): MetalCall {
  const cell = (result: MetalCall['result'], executor: Executor | null, why: string, facts: string[]): MetalCall => ({
    caller, callee, result, executedBy: executor, why, facts,
  });
  if (caller === 'host') {
    if (callee === 'host') return cell('call', 'caller-host-thread', tr('An ordinary call.', 'Một lời gọi thông thường.'), []);
    if (callee === 'kernel')
      return cell('dispatch', 'new-grid', tr('Not a function call: the host builds a pipeline state from the kernel and dispatches a grid of threads through an encoder.', 'Không phải lời gọi hàm: host tạo một pipeline state từ kernel và dispatch một lưới luồng qua một encoder.'), ['C1', 'C4']);
    return cell('no', null, tr('Only functions with an attribute such as kernel are visible to the app.', 'Chỉ những hàm có thuộc tính như kernel mới hiển thị với ứng dụng.'), ['C1', 'C7']);
  }
  if (callee === 'host') return cell('no', null, tr('MSL code is a separate file in a separate language. It cannot call into the host program, and no function is compiled for both sides.', 'Mã MSL là một tệp riêng bằng một ngôn ngữ riêng. Nó không thể gọi vào chương trình host, và không có hàm nào được biên dịch cho cả hai phía.'), ['C7']);
  if (callee === 'kernel') return cell('no', null, tr('Kernel functions cannot be called by other shader functions: it is a compile error. Nothing like dynamic parallelism is documented.', 'Hàm kernel không thể được các hàm shader khác gọi: đó là một lỗi biên dịch. Không có tài liệu nào về thứ gì giống dynamic parallelism.'), ['C1', 'C7']);
  return cell('call', 'caller-device-thread', tr('A function with no attribute plays the part of a __device__ helper: the calling GPU thread runs it.', 'Một hàm không có thuộc tính đóng vai trò của một hàm trợ giúp __device__: luồng GPU gọi hàm chạy nó.'), ['C7']);
}

// ---------------------------------------------------------------- one function out of a source file
export interface FunctionSource {
  /** The definition, with the comment lines directly above it. */
  text: string;
  /** 1-based line of the first line of `text` in the file. */
  firstLine: number;
  lastLine: number;
  /** Read from the definition's own qualifiers. 'none' in a .metal or .mm file, which have none. */
  quals: Qualifiers;
}

/** Removes comments and string literals from one line, so that braces inside them are not counted. */
function codeOnly(line: string, state: { block: boolean }): string {
  let out = '';
  let i = 0;
  while (i < line.length) {
    if (state.block) {
      const end = line.indexOf('*/', i);
      if (end < 0) return out;
      state.block = false;
      i = end + 2;
    } else if (line.startsWith('//', i)) return out;
    else if (line.startsWith('/*', i)) {
      state.block = true;
      i += 2;
    } else if (line[i] === '"' || line[i] === "'") {
      const q = line[i];
      i++;
      while (i < line.length && line[i] !== q) i += line[i] === '\\' ? 2 : 1;
      i++;
    } else out += line[i++];
  }
  return out;
}

/**
 * Finds the definition of a function in C-family source (CUDA C++, MSL or
 * Objective-C++) and returns its text. Lessons use it to show one function of a
 * lab file without copying it. Null when the file does not define the function.
 */
export function functionSource(source: string, name: string): FunctionSource | null {
  const lines = source.replace(/\r\n/g, '\n').split('\n');
  const state = { block: false };
  const code = lines.map((l) => codeOnly(l, state));
  const depthBefore: number[] = [];
  let depth = 0;
  for (const c of code) {
    depthBefore.push(depth);
    for (const ch of c) depth += ch === '{' ? 1 : ch === '}' ? -1 : 0;
  }
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const named = new RegExp('(^|[^\\w])' + escaped + '\\s*\\(');
  for (let k = 0; k < lines.length; k++) {
    if (depthBefore[k] !== 0 || !named.test(code[k])) continue;
    // From the name on, the first of ';' and '{' decides: a declaration or a definition.
    let open = -1;
    for (let j = k; j < lines.length; j++) {
      const text = j === k ? code[j].slice(code[j].search(named)) : code[j];
      const semi = text.indexOf(';');
      const brace = text.indexOf('{');
      if (brace >= 0 && (semi < 0 || brace < semi)) open = j;
      if (brace >= 0 || semi >= 0) break;
    }
    if (open < 0) continue;
    // The definition may start above the name: a qualifier or a template header on its own line.
    let first = k;
    while (first > 0 && depthBefore[first - 1] === 0 && code[first - 1].trim() !== '' && !/[;}]\s*$/.test(code[first - 1]) && !code[first - 1].trim().startsWith('#')) first--;
    const head = first;
    while (first > 0 && lines[first - 1].trim().startsWith('//')) first--;
    // The end: the line on which the braces opened at `open` are closed again.
    let d = 0;
    let last = open;
    for (let j = open; j < lines.length; j++) {
      for (const ch of code[j]) d += ch === '{' ? 1 : ch === '}' ? -1 : 0;
      if (d === 0) {
        last = j;
        break;
      }
    }
    const headText = code.slice(head, open + 1).join(' ');
    const isHost = /\b__host__\b/.test(headText);
    const isDevice = /\b__device__\b/.test(headText);
    const quals: Qualifiers = /\b__global__\b/.test(headText) ? 'global' : isHost && isDevice ? 'host-device' : isDevice ? 'device' : isHost ? 'host' : 'none';
    return { text: lines.slice(first, last + 1).join('\n'), firstLine: first + 1, lastLine: last + 1, quals };
  }
  return null;
}
