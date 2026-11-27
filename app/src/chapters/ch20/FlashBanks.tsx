import { useMemo, useState } from 'react';
import { Bench } from '../../components/notes';
import { BANK_PRESETS, type BankPresetId } from '../../sim/banks';
import { ktStoreBanks } from '../../sim/flashplan';
import { Translated, tr } from '../../lib/i18n';

// "Writing K^T with and without padding": Fig. 20.11's transposed store, one jj, one warp of
// consecutive threads dd. Model: src/sim/flashplan.ts on top of src/sim/banks.ts.

const PRESETS: BankPresetId[] = ['nvidia', 'amd-cdna', 'amd-cdna4', 'amd-rdna', 'apple'];

export function FlashBanks() {
  const [bc, setBc] = useState(32);
  const [padded, setPadded] = useState(false);
  const [preset, setPreset] = useState<BankPresetId>('nvidia');
  const [jj, setJj] = useState(0);
  const pr = BANK_PRESETS[preset];
  const j = Math.min(jj, bc - 1);
  const r = useMemo(
    () => (pr.banks ? ktStoreBanks(bc, j, padded, pr.banks, pr.warpWidth ?? 32) : null),
    [bc, j, padded, pr.banks, pr.warpWidth],
  );
  const maxLanes = r ? Math.max(1, ...r.perBank.map((b) => b.lanes.length)) : 1;
  const log = pr.banks ? Math.round(Math.log2(pr.banks)) : 0;

  return (
    <Bench
      title={tr('Writing K^T with and without padding', 'Ghi K^T có và không có đệm')}
      task={
        <Translated vi="Dự đoán số lượt cho B_c = 32 khi không đệm, rồi khi có đệm.">
          Predict the number of passes for B_c = 32 without padding, then with it.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Luồng dd ghi <code>KT_j[{padded ? `addr(dd·B_c + jj)` : 'dd·B_c + jj'}]</code>
              {padded ? <>, với <code>addr(x) = x + (x &gt;&gt; {log})</code> cho {pr.banks} bank</> : null}. Bank = chỉ số word
              modulo số bank; một lượt là một bank trả về một word. {pr.caveat ?? ''} Nguồn: {pr.facts}.
            </>
          }
        >
          <>
            Thread dd writes <code>KT_j[{padded ? `addr(dd·B_c + jj)` : 'dd·B_c + jj'}]</code>
            {padded ? <>, with <code>addr(x) = x + (x &gt;&gt; {log})</code> for {pr.banks} banks</> : null}. Bank = word index modulo
            the bank count; a pass is one bank delivering one word. {pr.caveat ?? ''} Sources: {pr.facts}.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          B_c
          <span className="seg" role="group" aria-label="B_c">
            {[8, 16, 32, 64].map((v) => <button key={v} aria-pressed={bc === v} onClick={() => setBc(v)}>{v}</button>)}
          </span>
        </label>
        <label className="field">
          jj = {j}
          <input type="range" min={0} max={bc - 1} value={j} aria-label="jj" onChange={(e) => setJj(+e.target.value)} />
        </label>
        <label className="check"><input type="checkbox" checked={padded} onChange={(e) => setPadded(e.target.checked)} /> <Translated vi="đệm bằng addr()">padding with addr()</Translated></label>
        <label className="field">
          <Translated vi="Bank">Banks</Translated>
          <select value={preset} onChange={(e) => setPreset(e.target.value as BankPresetId)} aria-label={tr('Banks', 'Bank')}>
            {PRESETS.map((p) => <option key={p} value={p}>{BANK_PRESETS[p].label}</option>)}
          </select>
        </label>
      </div>
      {r ? (
        <>
          <div className="narrate" role="status" style={{ marginBottom: '0.8rem' }} data-banks-summary data-passes={r.passes}>
            <Translated
              vi={
                <>
                  {r.lanes.length} luồng ghi các word {r.index.slice(0, 4).join(', ')}, … (bước nhảy {r.index[1] - r.index[0]}): chúng chạm{' '}
                  <b>{r.banksUsed}</b> trong {pr.banks} bank, <b>{r.maxLanesPerBank}</b> ở bank bận nhất, nên lần ghi mất{' '}
                  <b className={r.passes > 1 ? 't-fail' : 't-ok'}>{r.passes} lượt</b>.
                </>
              }
            >
              {r.lanes.length} threads write words {r.index.slice(0, 4).join(', ')}, … (stride {r.index[1] - r.index[0]}): they touch{' '}
              <b>{r.banksUsed}</b> of {pr.banks} banks, <b>{r.maxLanesPerBank}</b> on the busiest, so the store takes{' '}
              <b className={r.passes > 1 ? 't-fail' : 't-ok'}>{r.passes} pass{r.passes > 1 ? 'es' : ''}</b>.
            </Translated>
          </div>
          <svg viewBox={`0 0 ${pr.banks! * 12 + 20} 142`} role="img" aria-label={tr('Threads per bank', 'Số luồng trên mỗi bank')} style={{ width: '100%', maxWidth: 780, display: 'block' }}>
            {r.perBank.map((b) => {
              const h = (b.lanes.length / maxLanes) * 92;
              return (
                <g key={b.bank} transform="translate(0 12)">
                  <rect x={10 + b.bank * 12} y={100 - h} width={10} height={h} fill={b.lanes.length > 1 ? 'var(--fail)' : 'var(--shared)'} />
                  {b.lanes.length ? <text x={15 + b.bank * 12} y={96 - h} fontSize="8" textAnchor="middle">{b.lanes.length}</text> : null}
                  {b.bank % 4 === 0 ? <text x={15 + b.bank * 12} y={114} fontSize="8" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{b.bank}</text> : null}
                </g>
              );
            })}
            <text x={10} y={139} fontSize="9" style={{ fill: 'var(--ink-2)' }}><Translated vi="bank">bank</Translated></text>
          </svg>
        </>
      ) : (
        <div className="narrate" data-banks-summary data-passes="none">
          {pr.caveat}{' '}
          <Translated vi="Vì vậy không có gì để vẽ: trên máy Mac, đệm là một lựa chọn cần đo, không phải một quy tắc cần áp dụng.">
            So there is nothing to draw: on a Mac the padding is a choice to measure, not a rule to apply.
          </Translated>
        </div>
      )}
    </Bench>
  );
}
