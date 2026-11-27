#!/usr/bin/env node
// Vietnamese translation audit. Deterministic: it reads the source, it does not trust anyone's
// word that a file is translated.
//
//   node tools/i18n-audit.mjs                 every chapter, shared UI, narration and lab README
//   node tools/i18n-audit.mjs --chapter ch05  one chapter (its section files, narration, labs)
//   node tools/i18n-audit.mjs --chapter sim   the simulator modules (app/src/sim)
//   node tools/i18n-audit.mjs --details       also list every finding, not only the counts
//   node tools/i18n-audit.mjs --per-file      one row per source file
//   node tools/i18n-audit.mjs --base <ref>    git ref the English must match (default: the
//                                             merge base with upstream/main, else origin/main)
//
// Exit status 1 when anything is untranslated, a translation looks wrong, or the English of a
// translated file differs from the base ref.
//
// What it checks, per area:
// - App source (src/chapters, src/components, src/pages, src/App.tsx). Each piece of English a
//   reader can see must sit inside an English branch: the children of <Translated vi={...}>, the
//   first argument of tr(en, vi) or localized(en, vi), or a string field with a matching key in a
//   sibling `vi: {...}` object. Text inside <code>, <pre> and code-like strings is not prose.
//   A line ending in the comment `i18n-ignore` is left in English on purpose (counted apart).
// - English preservation. With every translation removed (vi props and vi objects dropped,
//   <Translated> and tr() unwrapped, the i18n import dropped), each file must give exactly the
//   same sequence of words, literals, identifiers and operators as the base ref. This is what
//   shows the English text, ids, answers and simulation code were left alone.
// - The Vietnamese side: no empty translation, none identical to its English, none that reads as
//   English (common English words and no Vietnamese letters).
// - Narration: content/<ch>/narration/<clip>.txt has content/<ch>/narration/vi/<clip>.txt.
// - Labs: labs/**/README.md has a README.vi.md beside it.
//
// The parser is oxc, reached through rolldown, which Vite already installs; no new dependency.

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const APP = path.join(ROOT, 'app');
const require = createRequire(path.join(APP, 'package.json'));
let parseSync;
try {
  ({ parseSync } = require('rolldown/experimental'));
} catch {
  console.error('i18n-audit: run `npm ci` in app/ first (the parser comes with Vite).');
  process.exit(2);
}

// ------------------------------------------------------------------ arguments
const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const option = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};
const onlyChapter = option('--chapter');
const details = flag('--details');
const perFile = flag('--per-file');
const git = (...a) => execFileSync('git', a, { cwd: ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
function resolveBase() {
  const given = option('--base');
  if (given) return given;
  for (const ref of ['upstream/main', 'origin/main']) {
    try {
      return git('merge-base', 'HEAD', ref);
    } catch {
      /* ref absent in this clone: try the next */
    }
  }
  return null;
}
const BASE = resolveBase();

// ------------------------------------------------------------------ text heuristics
const VI_LETTERS = /[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]/i;
const EN_STOPWORDS = new Set(
  'the and of is to that with for this are it each from which when what you your into its not can'.split(' '),
);
// Strings that stay English everywhere: the project's fixed verification vocabulary, backend
// names, and units. Matched against the whole (trimmed) string.
const KEEP_ENGLISH = [
  /^(verified on Metal|emulator-checked|front-end-checked|unverified)\b/,
  /^Programming Massively Parallel Processors$/, // the book's title
  /^(CUDA|HIP|Metal|MSL|emu|race|metal|cuda|hip|C\+\+|PyTorch|MLX|llama\.cpp|ROCm|Xcode)$/,
  /^(GFLOP\/s|GB\/s|TFLOP\/s|FLOP\/B|ms|µs|us|ns|s|GB|MB|KB|B|%|×|x)$/,
  // a bare citation: "ch. 3 §3.4, p. 62", "§5.2, pp. 98–104", "Fig. 5.3"
  /^((ch|p|pp|Fig|Figs|Table|Eq)\.|§|[\d.,–\-: ]|[A-Z]\.?\d|\(|\))+$/,
  // a facts-file citation, alone or after a page: "appC-facts A2", "book p. 624; x2-facts B1, B3"
  /^(book|and|(ch|p|pp|Fig|Table)\.|§[\d.]+|[\w]+-facts|[A-Z]\d+[a-z]?|[\d.,;–\-: ]|\(|\))+$/,
];
const CODE_TAGS = new Set(['code', 'pre', 'kbd', 'samp', 'var', 'math']);
// Attributes and object keys whose value is never prose shown to a reader.
const NON_TEXT_KEYS = new Set([
  'className', 'style', 'id', 'key', 'kind', 'href', 'src', 'd', 'transform', 'points', 'viewBox', 'fill',
  'stroke', 'strokeDasharray', 'textAnchor', 'fontFamily', 'dominantBaseline', 'role', 'type', 'value',
  'clip', 'code', 'file', 'dir', 'num', 'pages', 'tone', 'tier', 'tiers', 'lang', 'path', 'variant', 'mode',
  'status', 'data-testid', 'htmlFor', 'target', 'rel', 'step', 'min', 'max', 'inputMode', 'pattern', 'width',
  'height', 'x', 'y', 'cx', 'cy', 'r', 'rx', 'ry', 'x1', 'x2', 'y1', 'y2', 'color', 'background', 'font',
  'chapter', 'section', 'lab', 'figs', 'backend', 'column', 'metric', 'unitKey', 'accent', 'shape', 'align',
  'aria-live', 'aria-hidden', 'aria-current', 'aria-pressed', 'preload', 'loading', 'autoComplete', 'spellCheck',
  'wrap', 'scope', 'fontWeight', 'fontStyle', 'textDecoration', 'cursor', 'display', 'mask', 'filter', 'marker',
  'markerEnd', 'markerStart', 'clipPath', 'gradientUnits', 'patternUnits', 'vectorEffect', 'paintOrder',
]);
// Object keys whose string value is shown to the reader even when it is a single word.
const USER_KEYS = new Set([
  'label', 'title', 'caption', 'task', 'heading', 'head', 'text', 'unit', 'note', 'why', 'lede', 'desc',
  'description', 'legend', 'foot', 'summary', 'tip', 'hint', 'placeholder', 'message', 'msg', 'explain', 'prompt',
  'name', 'short', 'long', 'what', 'detail', 'details', 'help', 'verdict', 'reason', 'say', 'line', 'caveat',
]);
// Calls whose string arguments are for the program or the developer, never the reader.
const NON_TEXT_CALLS = /^(useLabFile|fetch|require|benchRows|pickBenchRow|defineLabs|numCol|msCol|labDir|excerpt|fnBody|complain|frontendStatus|labStatus|matchMedia|querySelector|querySelectorAll|getItem|setItem|removeItem|getElementById|addEventListener|removeEventListener|split|join|replace|replaceAll|startsWith|endsWith|includes|indexOf|match|test|exec|padStart|padEnd|toFixed|localeCompare|getPropertyValue|setProperty|dispatchEvent|postMessage|has|get|set|delete|find|filter|some|every|RegExp|Symbol)$/;
const isWordy = (s) => /[A-Za-z]{2,}/.test(s);
function looksLikeCode(s) {
  const t = s.trim();
  if (!t) return true;
  if (/\n\s*\S/.test(t) && /[;{}]/.test(t)) return true; // a multi-line listing
  if (/^[\w$.]+\([^)]*\)$/.test(t)) return true; // f(x)
  if (/^(python3|node|npm|npx|cd|cmake|make|git|export|xcrun|nvcc|hipcc|clang\+\+|\.\/)(\s|$)/.test(t)) return true; // a command
  if (/\//.test(t) && !/\s/.test(t.replace(/\s*\{\}\s*/g, ''))) return true; // a path with placeholders
  if (/^[#./~]|^https?:|^[\w-]+\.(tsx?|mjs|json|cu|metal|mm|cpp|hpp|h|md|txt|py|mp3)$/.test(t)) return true;
  if (/^var\(--|^[\d.]+(px|rem|em|%)?$|^#[0-9a-f]{3,8}$/i.test(t)) return true;
  if (/^[a-z]+([A-Z][a-z0-9]*)+$/.test(t)) return true; // camelCase identifier
  if (/^[A-Za-z_][\w]*(\[[^\]]*\]|\.[A-Za-z_]\w*)+$/.test(t)) return true; // a[i], a.b.c
  if (/[=<>]=?|&&|\|\||\+\+|->/.test(t) && !/\s[a-z]{3,}\s[a-z]{3,}\s/.test(t)) return true; // an expression
  return false;
}
function isProse(s, { inUserSlot }) {
  const t = s.trim();
  if (!isWordy(t) || looksLikeCode(t)) return false;
  if (KEEP_ENGLISH.some((re) => re.test(t))) return false;
  if (/\s/.test(t)) return true; // two or more words
  return inUserSlot && /[a-z]{3,}/.test(t); // a one-word label in a slot the reader sees
}
function readsAsEnglish(s) {
  if (VI_LETTERS.test(s)) return false;
  const words = s.toLowerCase().match(/[a-z]+/g) ?? [];
  return words.filter((w) => EN_STOPWORDS.has(w)).length >= 2;
}
const norm = (s) => s.replace(/\s+/g, ' ').trim();

// ------------------------------------------------------------------ AST walking
const isNode = (v) => v && typeof v === 'object' && typeof v.type === 'string';
function children(node) {
  const out = [];
  for (const [k, v] of Object.entries(node)) {
    if (k === 'start' || k === 'end' || k === 'loc' || k === 'range') continue;
    if (Array.isArray(v)) {
      for (const x of v) if (isNode(x)) out.push([k, x]);
    } else if (isNode(v)) out.push([k, v]);
  }
  return out;
}
const jsxName = (n) => (n?.type === 'JSXIdentifier' ? n.name : n?.type === 'JSXMemberExpression' ? jsxName(n.property) : null);
const calleeName = (c) => (c?.type === 'Identifier' ? c.name : null);
const propKey = (p) =>
  p?.type === 'Property' || p?.type === 'ObjectProperty'
    ? p.key?.type === 'Identifier'
      ? p.key.name
      : typeof p.key?.value === 'string'
        ? p.key.value
        : null
    : null;
const isViProp = (p) => propKey(p) === 'vi' && (p.value?.type === 'ObjectExpression' || p.value?.type === 'Literal');
const TR_FUNCS = new Set(['tr', 'localized']);

function lineOf(src, offset) {
  let line = 1;
  for (let i = 0; i < offset && i < src.length; i++) if (src.charCodeAt(i) === 10) line++;
  return line;
}

/** Tracks the import aliases a file uses for tr / localized / Translated. */
// The helpers come from lib/i18n (React side) or lib/locale.ts (plain strings, used by the simulators).
const I18N_MODULE = /lib\/(i18n|locale\.ts)$/;

function i18nNames(program) {
  const names = { tr: new Set(), translated: new Set() };
  for (const st of program.body) {
    if (st.type !== 'ImportDeclaration' || !I18N_MODULE.test(st.source.value)) continue;
    for (const sp of st.specifiers) {
      const imported = sp.imported?.name ?? sp.imported?.value;
      if (TR_FUNCS.has(imported)) names.tr.add(sp.local.name);
      if (imported === 'Translated') names.translated.add(sp.local.name);
    }
  }
  return names;
}

function auditSource(file, src) {
  const res = { file, english: 0, covered: 0, ignored: 0, missing: [], viProblems: [] };
  const { program, errors } = parseSync(file, src, { lang: file.endsWith('.tsx') ? 'tsx' : 'ts' });
  if (errors.length) {
    res.parseError = errors[0].message;
    return res;
  }
  const names = i18nNames(program);
  const lines = src.split('\n');
  const ignoredLine = (node) => /i18n-ignore/.test(lines[lineOf(src, node.start) - 1] ?? '');

  const viCheck = (node, enText, viText) => {
    const v = norm(viText);
    if (!v) res.viProblems.push({ line: lineOf(src, node.start), problem: 'empty translation', text: norm(enText).slice(0, 80) });
    else if (enText != null && v === norm(enText) && v.split(' ').length > 3 && isProse(v, { inUserSlot: true }))
      res.viProblems.push({ line: lineOf(src, node.start), problem: 'translation identical to English', text: v.slice(0, 80) });
    else if (readsAsEnglish(v)) res.viProblems.push({ line: lineOf(src, node.start), problem: 'translation reads as English', text: v.slice(0, 80) });
  };
  const textOf = (node) => {
    const parts = [];
    const go = (n) => {
      if (n.type === 'JSXText') parts.push(n.value);
      else if (n.type === 'Literal' && typeof n.value === 'string') parts.push(n.value);
      else if (n.type === 'TemplateElement') parts.push(n.value.cooked ?? n.value.raw);
      for (const [, c] of children(n)) go(c);
    };
    go(node);
    return parts.join(' ');
  };

  const count = (node, text, ctx, inUserSlot) => {
    if (ctx.vi || ctx.code) return;
    if (!isProse(text, { inUserSlot })) return;
    res.english++;
    if (ctx.en) res.covered++;
    else if (ignoredLine(node)) res.ignored++;
    else res.missing.push({ line: lineOf(src, node.start), text: norm(text).slice(0, 90) });
  };

  const walk = (node, ctx, slot) => {
    switch (node.type) {
      case 'ImportDeclaration':
      case 'TSInterfaceDeclaration':
      case 'TSTypeAliasDeclaration':
      case 'TSTypeAnnotation':
      case 'TSAsExpression':
        if (node.type !== 'TSAsExpression') return;
        break;
      case 'JSXText':
        count(node, node.value, ctx, true);
        return;
      case 'Literal':
        if (typeof node.value === 'string') count(node, node.value, ctx, slot);
        return;
      case 'TemplateLiteral': {
        const joined = node.quasis.map((q) => q.value.cooked ?? q.value.raw).join(' {} ');
        if (node.expressions.length === 0 || isWordy(joined)) count(node, joined, ctx, slot);
        for (const e of node.expressions) walk(e, ctx, false);
        return;
      }
      case 'TaggedTemplateExpression':
        return; // tagged templates hold code (none are prose in this app)
      case 'CallExpression': {
        const name = calleeName(node.callee);
        if (name && names.tr.has(name) && node.arguments.length >= 2) {
          walk(node.arguments[0], { ...ctx, en: true }, true);
          walk(node.arguments[1], { ...ctx, vi: true }, true);
          const enArg = textOf(node.arguments[0]);
          if (!ctx.vi && norm(enArg)) viCheck(node, enArg, textOf(node.arguments[1]));
          return;
        }
        // console output, errors thrown at developers, and program-facing calls are not reader text
        if (node.callee?.type === 'MemberExpression' && node.callee.object?.name === 'console') return;
        const method = node.callee?.type === 'MemberExpression' ? node.callee.property?.name : name;
        if (method && NON_TEXT_CALLS.test(method)) {
          walk(node.callee, { ...ctx, code: true }, false);
          for (const a of node.arguments) walk(a, { ...ctx, code: true }, false);
          return;
        }
        break;
      }
      case 'NewExpression':
      case 'ThrowStatement':
        if (node.type === 'ThrowStatement' || /Error$/.test(calleeName(node.callee) ?? '')) return;
        break;
      case 'JSXElement': {
        const tag = jsxName(node.openingElement.name);
        if (tag && names.translated.has(tag)) {
          const viAttr = node.openingElement.attributes.find((a) => a.type === 'JSXAttribute' && a.name?.name === 'vi');
          if (viAttr) {
            walk(viAttr.value, { ...ctx, vi: true }, true);
            if (!ctx.vi) viCheck(node, node.children.map(textOf).join(' '), textOf(viAttr.value));
          } else res.viProblems.push({ line: lineOf(src, node.start), problem: '<Translated> without vi', text: '' });
          for (const c of node.children) walk(c, { ...ctx, en: true }, true);
          return;
        }
        const inner = { ...ctx, code: ctx.code || CODE_TAGS.has(tag) };
        for (const a of node.openingElement.attributes) {
          if (a.type === 'JSXSpreadAttribute') walk(a.argument, inner, false);
          else if (a.value) {
            const an = a.name?.name;
            const skip = typeof an === 'string' && (NON_TEXT_KEYS.has(an) || an.startsWith('data-') || an.startsWith('on'));
            walk(a.value, skip ? { ...inner, code: true } : inner, !skip);
          }
        }
        for (const c of node.children) walk(c, inner, true);
        return;
      }
      case 'ObjectExpression': {
        const viObj = node.properties.find(isViProp);
        const viKeys = new Set(
          viObj?.value?.type === 'ObjectExpression' ? viObj.value.properties.map(propKey).filter(Boolean) : [],
        );
        for (const p of node.properties) {
          if (p === viObj) {
            walk(p.value, { ...ctx, vi: true }, true);
            continue;
          }
          const k = propKey(p);
          if (k && viKeys.has(k)) {
            walk(p.value, { ...ctx, en: true }, true);
            const viProp = viObj.value.properties.find((q) => propKey(q) === k);
            if (!ctx.vi) viCheck(p, textOf(p.value), textOf(viProp.value));
            continue;
          }
          if (p.type === 'SpreadElement') walk(p.argument, ctx, false);
          else if (p.computed) walk(p.value, ctx, false);
          else if (k && NON_TEXT_KEYS.has(k)) walk(p.value, { ...ctx, code: true }, false);
          else walk(p.value, ctx, !!k && USER_KEYS.has(k));
        }
        return;
      }
      case 'Property':
        walk(node.value, ctx, true);
        return;
      case 'BinaryExpression':
        // comparisons against string constants are logic, not text: kind === 'choice'
        if (/^[!=]==?$/.test(node.operator)) return;
        break;
      case 'SwitchCase':
        for (const c of node.consequent) walk(c, ctx, false);
        return;
      case 'MemberExpression':
        if (node.computed && node.property.type === 'Literal') {
          walk(node.object, ctx, false);
          return; // obj['key']
        }
        break;
      default:
        break;
    }
    for (const [, c] of children(node)) walk(c, ctx, false);
  };
  walk(program, { en: false, vi: false, code: false }, false);
  return res;
}

// ------------------------------------------------------------------ English preservation
function englishTokens(file, src) {
  const { program, errors } = parseSync(file, src, { lang: file.endsWith('.tsx') ? 'tsx' : 'ts' });
  if (errors.length) return null;
  const names = i18nNames(program);
  // `const locale = useLocale()` added so a memo recomputes on a language switch is not English content:
  // drop that declaration and every later use of the variable (typically in a dependency array).
  const localeVars = new Set();
  const findLocaleVars = (node) => {
    if (node.type === 'VariableDeclarator' && node.id?.type === 'Identifier' && node.init?.type === 'CallExpression' && calleeName(node.init.callee) === 'useLocale')
      localeVars.add(node.id.name);
    for (const [, c] of children(node)) findLocaleVars(c);
  };
  findLocaleVars(program);
  const out = [];
  const push = (s) => {
    for (const w of String(s).split(/\s+/)) if (w) out.push(w);
  };
  const walk = (node) => {
    switch (node.type) {
      case 'ImportDeclaration':
        if (I18N_MODULE.test(node.source.value)) {
          // keep imports of things that are not the translation helpers
          const rest = node.specifiers.filter((sp) => {
            const imported = sp.imported?.name ?? sp.imported?.value;
            return !TR_FUNCS.has(imported) && imported !== 'Translated' && imported !== 'Locale' && imported !== 'useLocale';
          });
          if (!rest.length) return;
          push('import');
          for (const sp of rest) push(sp.local.name);
          push(node.source.value);
          return;
        }
        // a specifier added only for translation (QuestionVi) is not English content
        push('import');
        for (const sp of node.specifiers) if ((sp.imported?.name ?? '') !== 'QuestionVi') push(sp.local.name);
        push(node.source.value);
        return;
      case 'JSXText':
        push(node.value);
        return;
      case 'Literal':
        push(typeof node.value === 'string' ? node.value : String(node.raw ?? node.value));
        return;
      case 'TemplateElement':
        push(node.value.cooked ?? node.value.raw);
        return;
      case 'VariableDeclaration':
        if (node.declarations.every((d) => d.id?.type === 'Identifier' && localeVars.has(d.id.name))) return;
        break;
      case 'Identifier':
        if (localeVars.has(node.name)) return;
        push(node.name);
        return;
      case 'JSXIdentifier':
        push(node.name);
        return;
      case 'CallExpression': {
        const name = calleeName(node.callee);
        if (name && names.tr.has(name) && node.arguments.length >= 2) {
          walk(node.arguments[0]);
          return;
        }
        break;
      }
      case 'JSXElement': {
        const tag = jsxName(node.openingElement.name);
        if (tag && names.translated.has(tag)) {
          for (const c of node.children) walk(c);
          return;
        }
        break;
      }
      case 'JSXAttribute':
        if (node.name?.name === 'vi') return;
        break;
      case 'ObjectExpression':
        for (const p of node.properties) if (!isViProp(p)) walk(p);
        return;
      default:
        if (typeof node.operator === 'string') push(node.operator);
    }
    for (const [, c] of children(node)) walk(c);
  };
  walk(program);
  return out;
}
const baseCache = new Map();
function baseSource(rel) {
  if (!BASE) return null;
  if (!baseCache.has(rel)) {
    try {
      baseCache.set(rel, git('show', `${BASE}:${rel}`));
    } catch {
      baseCache.set(rel, null); // the file is new on this branch
    }
  }
  return baseCache.get(rel);
}
// Files whose code (not their English) had to change so simulator text is read in the selected
// language at render time instead of being frozen at import. Each was reviewed by hand: same
// English words, same ids and results. The audit lists them; it still fails on any other change.
const RESTRUCTURED = {
  'app/src/chapters/ch02/S27.tsx': 'model sentences read while rendering (thunks + a Why component)',
  'app/src/chapters/ch06/DoubleBufferScene.tsx': 'trace cache keyed by language',
  'app/src/chapters/ch12/ScanChainScene.tsx': 'scene runs rebuilt on a language switch',
  'app/src/chapters/ch23/LoopScene.tsx': 'scene runs are getters',
  'app/src/chapters/x3/LlamaStepScene.tsx': 'launch shape line re-read from the model at render time',
};
const restructuredSeen = [];

function preservation(rel, src) {
  const before = baseSource(rel);
  if (before == null || before === src) return null;
  const a = englishTokens(rel, before);
  const b = englishTokens(rel, src);
  if (!a || !b) return 'parse error';
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++)
    if (a[i] !== b[i]) return `differs at token ${i}: base "${a.slice(i, i + 8).join(' ')}" vs now "${b.slice(i, i + 8).join(' ')}"`;
  return a.length === b.length ? null : `length ${a.length} in base vs ${b.length} now`;
}

// ------------------------------------------------------------------ inventory
const walkFiles = (dir, pred) =>
  fs.existsSync(dir)
    ? fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
        const p = path.join(dir, e.name);
        return e.isDirectory() ? walkFiles(p, pred) : pred(p) ? [p] : [];
      })
    : [];
const rel = (p) => path.relative(ROOT, p).split(path.sep).join('/');
const isAppSource = (p) => /\.(tsx|ts)$/.test(p) && !/\.test\.tsx?$|\.d\.ts$/.test(p);

// `--chapter sim` audits the simulator modules alone.
const onlySim = onlyChapter === 'sim';
const chapterDirs = fs
  .readdirSync(path.join(APP, 'src/chapters'), { withFileTypes: true })
  .filter((e) => e.isDirectory() && (!onlyChapter || e.name === onlyChapter))
  .map((e) => e.name)
  .sort();
const appFiles = [
  ...chapterDirs.flatMap((c) => walkFiles(path.join(APP, 'src/chapters', c), isAppSource)),
  ...(onlySim ? walkFiles(path.join(APP, 'src/sim'), isAppSource) : []),
  ...(onlyChapter
    ? []
    : [
        ...walkFiles(path.join(APP, 'src/components'), isAppSource),
        ...walkFiles(path.join(APP, 'src/pages'), isAppSource),
        path.join(APP, 'src/App.tsx'),
        // Simulators build step text, labels and evidence lines that the chapters render as is.
        ...walkFiles(path.join(APP, 'src/sim'), isAppSource),
        path.join(APP, 'src/chapters/registry.ts'),
      ]),
];

const report = { app: [], narration: [], labs: [] };
for (const f of appFiles) {
  const src = fs.readFileSync(f, 'utf8');
  const r = auditSource(rel(f), src);
  // Shared UI moved its strings into the dictionaries on purpose; chapter and simulator files must keep their English as is.
  r.preservation = /^app\/src\/(chapters\/[^/]+|sim)\//.test(rel(f)) ? preservation(rel(f), src) : null;
  if (r.preservation && RESTRUCTURED[rel(f)]) {
    restructuredSeen.push(`${rel(f)}: ${RESTRUCTURED[rel(f)]}`);
    r.preservation = null;
  }
  report.app.push(r);
}

const narrationChapters = onlyChapter ? [onlyChapter] : chapterDirs;
for (const c of narrationChapters) {
  const dir = path.join(ROOT, 'content', c, 'narration');
  if (!fs.existsSync(dir)) continue;
  for (const f of fs.readdirSync(dir).filter((x) => x.endsWith('.txt')).sort()) {
    const en = fs.readFileSync(path.join(dir, f), 'utf8');
    const viPath = path.join(dir, 'vi', f);
    const vi = fs.existsSync(viPath) ? fs.readFileSync(viPath, 'utf8') : null;
    const problem =
      vi == null ? 'missing' : !norm(vi) ? 'empty' : norm(vi) === norm(en) ? 'identical to English' : readsAsEnglish(vi) ? 'reads as English' : null;
    report.narration.push({ file: rel(path.join(dir, f)), problem });
  }
}

const labRoots = onlyChapter ? [path.join(ROOT, 'labs', onlyChapter)] : [path.join(ROOT, 'labs')];
for (const root of labRoots) {
  const readmes = walkFiles(root, (p) => path.basename(p) === 'README.md' && !/\/(work|build[^/]*|node_modules)\//.test(p));
  for (const f of readmes.sort()) {
    const viPath = path.join(path.dirname(f), 'README.vi.md');
    const vi = fs.existsSync(viPath) ? fs.readFileSync(viPath, 'utf8') : null;
    const problem = vi == null ? 'missing' : !VI_LETTERS.test(vi) ? 'no Vietnamese text' : null;
    report.labs.push({ file: rel(f), problem });
  }
}

// ------------------------------------------------------------------ output
const pct = (a, b) => (b ? ((100 * a) / b).toFixed(1) + '%' : 'n/a');
let failed = false;
const group = (r) => {
  const m = r.file.match(/^app\/src\/chapters\/([^/]+)\//);
  if (m && m[1] !== 'registry.ts') return m[1];
  return r.file.startsWith('app/src/sim/') ? 'simulators' : 'shared UI';
};
const byGroup = new Map();
for (const r of report.app) {
  const g = group(r);
  const s = byGroup.get(g) ?? { files: 0, english: 0, covered: 0, ignored: 0, missing: 0, vi: 0, changed: 0, parse: 0 };
  s.files++;
  s.english += r.english;
  s.covered += r.covered;
  s.ignored += r.ignored;
  s.missing += r.missing.length;
  s.vi += r.viProblems.length;
  s.changed += r.preservation ? 1 : 0;
  s.parse += r.parseError ? 1 : 0;
  byGroup.set(g, s);
}
console.log(`English base: ${BASE ? BASE.slice(0, 12) : 'none found (preservation not checked)'}\n`);
console.log('App text (pieces of English a reader sees)');
console.log('area        files  english  translated  ignored  untranslated  vi-issues  english-changed');
const tot = { files: 0, english: 0, covered: 0, ignored: 0, missing: 0, vi: 0, changed: 0, parse: 0 };
for (const [g, s] of byGroup) {
  for (const k of Object.keys(tot)) tot[k] += s[k];
  console.log(
    `${g.padEnd(11)} ${String(s.files).padStart(5)}  ${String(s.english).padStart(7)}  ${pct(s.covered, s.english).padStart(10)}  ${String(s.ignored).padStart(7)}  ${String(s.missing).padStart(12)}  ${String(s.vi).padStart(9)}  ${String(s.changed + s.parse).padStart(15)}`,
  );
}
console.log(
  `${'total'.padEnd(11)} ${String(tot.files).padStart(5)}  ${String(tot.english).padStart(7)}  ${pct(tot.covered, tot.english).padStart(10)}  ${String(tot.ignored).padStart(7)}  ${String(tot.missing).padStart(12)}  ${String(tot.vi).padStart(9)}  ${String(tot.changed + tot.parse).padStart(15)}`,
);
if (tot.missing || tot.vi || tot.changed || tot.parse) failed = true;
if (perFile) {
  console.log('\nfile                                              english  translated  untranslated  vi-issues  english-changed');
  for (const r of report.app)
    console.log(
      `${r.file.replace(/^app\/src\//, '').padEnd(48)} ${String(r.english).padStart(8)}  ${pct(r.covered, r.english).padStart(10)}  ${String(r.missing.length).padStart(12)}  ${String(r.viProblems.length).padStart(9)}  ${(r.parseError ? 'parse error' : r.preservation ? 'YES' : '').padStart(15)}`,
    );
}

const narr = report.narration;
const narrOk = narr.filter((n) => !n.problem).length;
if (restructuredSeen.length) console.log(`\nReviewed code changes for render-time text (English words unchanged):\n${restructuredSeen.map((l) => '  ' + l).join('\n')}`);
console.log(`\nNarration scripts: ${narrOk} of ${narr.length} have a Vietnamese script (${pct(narrOk, narr.length)})`);
const labsOk = report.labs.filter((n) => !n.problem).length;
console.log(`Lab READMEs: ${labsOk} of ${report.labs.length} have README.vi.md (${pct(labsOk, report.labs.length)})`);
if (narrOk < narr.length || labsOk < report.labs.length) failed = true;

const showLimit = details ? Infinity : 15;
const problems = [];
for (const r of report.app) {
  if (r.parseError) problems.push(`${r.file}: parse error: ${r.parseError}`);
  if (r.preservation) problems.push(`${r.file}: English changed: ${r.preservation}`);
  for (const m of r.missing) problems.push(`${r.file}:${m.line}: untranslated: ${m.text}`);
  for (const v of r.viProblems) problems.push(`${r.file}:${v.line}: ${v.problem}: ${v.text}`);
}
for (const n of narr) if (n.problem) problems.push(`${n.file}: Vietnamese script ${n.problem}`);
for (const n of report.labs) if (n.problem) problems.push(`${n.file}: README.vi.md ${n.problem}`);
if (problems.length) {
  console.log(`\n${problems.length} finding(s)${details ? '' : ', first ' + Math.min(showLimit, problems.length) + ' (use --details for all)'}:`);
  for (const p of problems.slice(0, showLimit)) console.log('  ' + p);
}
console.log(failed ? '\nFAIL' : '\nOK');
process.exit(failed ? 1 : 0);
