import { useEffect, useState } from 'react';
import { Bench } from '../../components/notes';
import { laneElements, ownerOf, wave64Broadcast } from '../../sim/flashplan';
import { Translated, tr } from '../../lib/i18n';

// "Who holds Q element dd?": Fig. 20.10's interleaved layout and Fig. 20.12's broadcast,
// then the same code on a 64-wide AMD wavefront (facts B2). Model: src/sim/flashplan.ts,
// which takes HIP's shuffle rule from src/sim/shuffle.ts.

type Mode = 'nvidia' | 'wave64' | 'wave64w32';
const MODE_LABEL: Record<Mode, string> = {
  get nvidia() {
    return tr('32-lane warp (the book)', 'warp 32 lane (sách)');
  },
  get wave64() {
    return tr('AMD wave64, book’s code, default width', 'AMD wave64, mã của sách, độ rộng mặc định');
  },
  get wave64w32() {
    return tr('AMD wave64, width 32', 'AMD wave64, độ rộng 32');
  },
};
const D = 128;

export function FlashLanes() {
  const [dd, setDd] = useState(77);
  const [mode, setMode] = useState<Mode>('nvidia');
  const [pick, setPick] = useState(40);
  const [playing, setPlaying] = useState(false);
  const own = ownerOf(dd);

  useEffect(() => {
    if (!playing) return;
    const id = window.setTimeout(() => setDd((x) => (x + 1) % D), 260);
    return () => window.clearTimeout(id);
  }, [playing, dd]);

  const wave = mode === 'nvidia' ? null : wave64Broadcast(dd, mode === 'wave64' ? 64 : 32);
  const picked = wave?.[pick];
  const wrong = wave ? wave.filter((x) => !x.ok).length : 0;

  return (
    <Bench
      title={tr('Who holds Q element dd?', 'Ai giữ phần tử dd của Q?')}
      task={
        <Translated vi="Trước khi kéo: lane nào và ô thanh ghi nào giữ phần tử 77?">
          Before dragging: which lane and which register slot hold element 77?
        </Translated>
      }
      foot={
        mode === 'nvidia'
          ? tr(
              'One warp, one row of Q with d = 128. Each lane holds four elements in four registers, slot 0 to slot 3.',
              'Một warp, một hàng của Q với d = 128. Mỗi lane giữ bốn phần tử trong bốn thanh ghi, từ ô 0 đến ô 3.',
            )
          : tr(
              'One 64-wide wavefront running the book’s code with WARP_SIZE 32: hardware lanes 0–31 are the first logical warp (one row of Q), 32–63 the second (another row). The width rule is HIP’s own source (ch20 facts B2); nothing was run on AMD hardware.',
              'Một wavefront rộng 64 chạy mã của sách với WARP_SIZE 32: các lane phần cứng 0–31 là warp logic thứ nhất (một hàng của Q), 32–63 là warp thứ hai (một hàng khác). Quy tắc độ rộng lấy từ chính mã nguồn của HIP (ch20 facts B2); chưa có gì được chạy trên phần cứng AMD.',
            )
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <label className="field" style={{ minWidth: '14rem' }}>
          <Translated vi={<>Phần tử dd = {dd}</>}>Element dd = {dd}</Translated>
          <input type="range" min={0} max={D - 1} value={dd} aria-label={tr('Element dd', 'Phần tử dd')} onChange={(e) => setDd(+e.target.value)} />
        </label>
        <button onClick={() => setDd((x) => (x + D - 1) % D)}>dd − 1{/* i18n-ignore: index arithmetic */}</button>
        <button className="primary" onClick={() => setDd((x) => (x + 1) % D)}>dd + 1{/* i18n-ignore: index arithmetic */}</button>
        <button onClick={() => setPlaying(!playing)}>{playing ? tr('Pause', 'Tạm dừng') : tr('Play the inner loop', 'Chạy vòng lặp trong')}</button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label={tr('Machine', 'Máy')}>
          {(Object.keys(MODE_LABEL) as Mode[]).map((m) => (
            <button key={m} aria-pressed={mode === m} onClick={() => setMode(m)}>{MODE_LABEL[m]}</button>
          ))}
        </span>
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.8rem' }} data-owner>
        <Translated
          vi={
            <>
              Phần tử {dd} của Q nằm ở <b className="t-reg">lane {own.lane}</b>, ô thanh ghi <b className="t-reg">{own.slot}</b> ({dd} = {own.slot} · 32 + {own.lane}).
              {mode === 'nvidia' ? (
                <> Ở bước dd của tích vô hướng, mọi lane gọi <code>__shfl_sync(mask, Q_i[ii][{own.slot}], {own.lane})</code> và
                  nhận được nó: một lần phát (broadcast) từ lane {own.lane} tới cả 32 lane.</>
              ) : (
                <> Mọi lane đều xin lane nguồn {own.lane}. {mode === 'wave64'
                  ? <>Với <code>width = warpSize</code> mặc định của HIP = 64, cả 64 lane phần cứng đọc lane phần cứng {own.lane}: <b className="t-fail">{wrong} lane của warp logic thứ hai nhận giá trị của hàng kia.</b></>
                  : <>Với <code>width</code> 32, mỗi nửa đọc lane {own.lane} của riêng nó: các lane phần cứng 32–63 đọc lane {32 + own.lane}. Tất cả đều đúng.</>}
                </>
              )}
            </>
          }
        >
          Q element {dd} lives in <b className="t-reg">lane {own.lane}</b>, register slot <b className="t-reg">{own.slot}</b> ({dd} = {own.slot} · 32 + {own.lane}).
          {mode === 'nvidia' ? (
            <> At step dd of the inner product, every lane calls <code>__shfl_sync(mask, Q_i[ii][{own.slot}], {own.lane})</code> and
              receives it: a broadcast from lane {own.lane} to all 32.</>
          ) : (
            <> Every lane asks for source lane {own.lane}. {mode === 'wave64'
              ? <>With HIP's default <code>width = warpSize</code> = 64, all 64 hardware lanes read hardware lane {own.lane}: <b className="t-fail">the {wrong} lanes of the second logical warp get the other row's value.</b></>
              : <>With <code>width</code> 32, each half reads its own lane {own.lane}: hardware lanes 32–63 read lane {32 + own.lane}. All correct.</>}
            </>
          )}
        </Translated>
      </div>

      {mode === 'nvidia' ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(3.3rem, 1fr))', gap: 4 }} data-lanes="32">
          {Array.from({ length: 32 }, (_, L) => {
            const els = laneElements(L, D);
            const isOwner = L === own.lane;
            return (
              <div key={L} data-lane={L} data-owner={isOwner ? 'yes' : 'no'}
                style={{ border: `1px solid ${isOwner ? 'var(--reg)' : 'var(--rule)'}`, background: isOwner ? 'var(--reg-wash)' : 'var(--page)', borderRadius: 3, padding: '2px 3px', fontSize: '0.7rem' }}>
                <div style={{ fontWeight: 600 }}><Translated vi={<>lane {L}</>}>lane {L}</Translated></div>
                {els.map((e, s) => (
                  <div key={s} style={{ fontVariantNumeric: 'tabular-nums', background: e === dd ? 'var(--reg)' : undefined, color: e === dd ? 'var(--page)' : 'var(--ink-2)', borderRadius: 2, paddingLeft: 2 }}>
                    s{s}: {e}
                  </div>
                ))}
                <div style={{ color: 'var(--ink-3)' }}>← {own.lane}</div>
              </div>
            );
          })}
        </div>
      ) : (
        <>
          {[0, 1].map((half) => (
            <div key={half} style={{ marginBottom: '0.6rem' }}>
              <div className="mx-label">
                <Translated vi={<>warp logic {half}: các lane phần cứng {half * 32}–{half * 32 + 31} ({half ? 'hàng B' : 'hàng A'} của Q)</>}>
                  logical warp {half}: hardware lanes {half * 32}–{half * 32 + 31} ({half ? 'row B' : 'row A'} of Q)
                </Translated>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(2.6rem, 1fr))', gap: 3 }} data-half={half}>
                {wave!.slice(half * 32, half * 32 + 32).map((x) => (
                  <button key={x.hw} onClick={() => setPick(x.hw)} aria-pressed={pick === x.hw} data-hw={x.hw} data-ok={x.ok ? 'yes' : 'no'}
                    style={{ padding: '2px 0', fontSize: '0.68rem', fontWeight: 450, borderColor: x.ok ? 'var(--rule)' : 'var(--fail)', background: pick === x.hw ? 'var(--ink)' : x.ok ? 'var(--page)' : 'var(--fail-wash)', color: pick === x.hw ? 'var(--page)' : x.ok ? 'var(--ink)' : 'var(--fail)' }}
                    title={tr(`hardware lane ${x.hw} wants ${x.wants}, gets ${x.gets}`, `lane phần cứng ${x.hw} muốn ${x.wants}, nhận ${x.gets}`)}>
                    {x.hw}<br />←{x.gets}
                  </button>
                ))}
              </div>
            </div>
          ))}
          {picked ? (
            <Translated
              vi={
                <p data-picked style={{ margin: '0.4rem 0 0' }}>
                  Lane phần cứng {picked.hw} (lane {picked.hw % 32} của warp logic {picked.logicalWarp}) xin lane nguồn {own.lane} và
                  nhận lane phần cứng <b className={picked.ok ? 't-ok' : 't-fail'}>{picked.gets}</b>
                  {picked.ok ? ', đúng lane cần.' : `, vốn thuộc warp logic kia; nó cần ${picked.wants}.`}
                </p>
              }
            >
              <p data-picked style={{ margin: '0.4rem 0 0' }}>
                Hardware lane {picked.hw} (lane {picked.hw % 32} of logical warp {picked.logicalWarp}) asks for source lane {own.lane} and
                receives hardware lane <b className={picked.ok ? 't-ok' : 't-fail'}>{picked.gets}</b>
                {picked.ok ? ', the right one.' : `, which belongs to the other logical warp; it wanted ${picked.wants}.`}
              </p>
            </Translated>
          ) : null}
        </>
      )}
    </Bench>
  );
}
