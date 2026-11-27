// Browser check for the language switch (English / Tiếng Việt): English by default, the
// switch translates in place without losing the route, answers or progress, the choice
// survives a reload, questions keep their ids, options and correct answer, and narrated
// animations show the Vietnamese script as a caption with the English audio off.
// Usage: node e2e/app-language.mjs
import { open, goto, BASE } from './browser.mjs';

const seed = {
  version: 1,
  sections: { 'ch05/start': { visited: '2026-11-20T10:00:00Z', done: true } },
  answers: { 'ch05/start/q-rowmajor': { value: 6, correct: true, attempts: 1, at: '2026-11-20T10:00:00Z' } },
  review: {},
};
const { browser, page, errors, puts } = await open({ width: 1440, height: 1000, progress: seed });
const fail = [];
const check = (name, ok, detail = '') => {
  console.log((ok ? 'ok   ' : 'FAIL ') + name + (ok ? '' : '  ' + detail));
  if (!ok) fail.push(name);
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const text = (sel) => page.$eval(sel, (e) => e.textContent.trim());
const lang = () => page.evaluate(() => document.documentElement.lang);
const pick = async (name) => {
  await page.evaluate((name) => {
    const b = [...document.querySelectorAll('[data-language-switch] button')].find((x) => x.textContent.trim() === name);
    b.click();
  }, name);
  await pause(250);
};
const pressed = () =>
  page.$$eval('[data-language-switch] button', (bs) => bs.filter((b) => b.getAttribute('aria-pressed') === 'true').map((b) => b.textContent.trim()));
const questionShape = () =>
  page.$$eval('[data-question]', (qs) =>
    qs.map((q) => ({
      id: q.getAttribute('data-question'),
      kind: q.getAttribute('data-question-kind'),
      options: [...q.querySelectorAll('input[data-written-index]')].map((i) => i.getAttribute('data-written-index')).join(','),
    })),
  );

await page.evaluateOnNewDocument(() => {
  if (!sessionStorage.getItem('seeded')) {
    localStorage.removeItem('lang');
    localStorage.removeItem('captions');
    sessionStorage.setItem('seeded', '1');
  }
});

// ------------------------------------------------------------------ default and switch
await goto(page, 'ch05/start');
check('English is the default language', (await lang()) === 'en' && (await text('.section-head h1')) === 'Before you start');
check('the switch offers English and Tiếng Việt, English selected', JSON.stringify(await pressed()) === '["English"]');
const shapeEn = await questionShape();
await pause(600); // let any save from opening the page land first
const putsBefore = puts.length;
const answeredEn = await page.$eval('[data-question="ch05/start/q-rowmajor"] input[type=number]', (i) => i.value);

await pick('Tiếng Việt');
check('switching sets <html lang="vi">', (await lang()) === 'vi');
check('the section title is translated', (await text('.section-head h1')) === 'Trước khi bắt đầu', await text('.section-head h1'));
check('the rail is translated', (await text('.rail')).includes('Ôn tập') || (await text('.rail')).includes('Ôn'));
check('the lesson prose is translated', (await text('.prose')).includes('Khi bạn gọi'));
check('the route is unchanged', page.url().endsWith('#/ch05/start'), page.url());
const shapeVi = await questionShape();
check('questions keep their ids, kinds and options', JSON.stringify(shapeVi) === JSON.stringify(shapeEn), JSON.stringify([shapeEn, shapeVi]));
check(
  'a stored answer is still shown',
  (await page.$eval('[data-question="ch05/start/q-rowmajor"] input[type=number]', (i) => i.value)) === answeredEn && answeredEn === '6',
);
check('the stored answer is still judged correct', (await page.$('[data-question="ch05/start/q-rowmajor"] .verdict.right')) !== null);
check('the unit is translated', (await text('[data-question="ch05/start/q-rowmajor"]')).includes('phần tử'));
check('the section is still marked done', (await text('.page footer')).length > 0 && (await page.$('.rail [aria-current=page] .tick[aria-label]')) !== null);
await pause(600);
check('switching did not write progress', puts.length === putsBefore, `${putsBefore} -> ${puts.length}`);

// a choice question answered in Vietnamese is graded on the written option, same as in English
await page.evaluate(() => document.querySelector('[data-question="ch05/start/q-mapping"] input[data-written-index="1"]').click());
await page.evaluate(() => [...document.querySelectorAll('[data-question="ch05/start/q-mapping"] button')].find((b) => b.classList.contains('primary')).click());
await pause(800); // saves are debounced
check('the correct option is still correct in Vietnamese', (await page.$('[data-question="ch05/start/q-mapping"] .verdict.right')) !== null);
const saved = puts.at(-1);
check(
  'the answer is saved under the same id and written index',
  saved?.answers?.['ch05/start/q-mapping']?.value === 1 && saved.answers['ch05/start/q-mapping'].correct === true,
  JSON.stringify(saved?.answers?.['ch05/start/q-mapping']),
);

// ------------------------------------------------------------------ persistence and deep links
await page.reload({ waitUntil: 'domcontentloaded' });
await page.waitForSelector('.page[data-ready]');
await pause(300);
check('the choice survives a reload', (await lang()) === 'vi' && (await text('.section-head h1')) === 'Trước khi bắt đầu');
await goto(page, 'ch05/5.3');
check('a deep link opens in the chosen language', (await lang()) === 'vi' && page.url().endsWith('#/ch05/5.3'));

// ------------------------------------------------------------------ narration
const S = 'section.scene';
await page.waitForSelector(S);
const viScript = (await (await fetch(`${BASE}/content/ch05/narration/vi/why-tiling-1.txt`)).text()).replace(/\[[^\]]*\]\s*/g, '').trim();
const soundBox = () =>
  page.evaluate((sel) => [...document.querySelectorAll(sel + ' label.check')].map((l) => [l.textContent.trim(), l.querySelector('input').checked]), S);
const boxes = await soundBox();
check('in Vietnamese the English audio is an explicit option, off by default', boxes.some(([l, on]) => l.includes('tiếng Anh') && !on), JSON.stringify(boxes));
check('the Vietnamese script is shown as the caption', (await page.$(S + ' .caption')) !== null && (await text(S + ' .caption')) === viScript);
check('the footer says there is no Vietnamese audio', (await text(S + ' footer')).includes('chưa có bản thu tiếng Việt'));

// ------------------------------------------------------------------ home and back to English
await goto(page, '');
check('the course table is translated', (await text('.prose')).includes('Kiến trúc bộ nhớ và tính cục bộ dữ liệu'));
await pick('English');
check('switching back restores English', (await lang()) === 'en' && (await text('.prose')).includes('Memory architecture and data locality'));
await goto(page, 'ch05/5.3');
await page.waitForSelector(S);
const boxesEn = await soundBox();
check('in English the sound is on by default as before', boxesEn.some(([l, on]) => l === 'sound' && on), JSON.stringify(boxesEn));
check('in English captions are off by default as before', (await page.$(S + ' .caption')) === null);

check('no page errors', errors.length === 0, errors.join(' | '));
await browser.close();
process.exit(fail.length ? 1 : 0);
