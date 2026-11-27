import { useMemo, useState } from 'react';
import { Bench } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { BOOK, fits, largestFittingBc, limits, plan, type PlanInput } from '../../sim/flashplan';
import { Translated, tr } from '../../lib/i18n';

// "Fit the tiles": the constants of Fig. 20.9 and what they cost, against the documented
// limits and the reader's own Metal limit. Model: src/sim/flashplan.ts.

const int = (v: number) => v.toLocaleString('en-US');

function Seg({ label, value, options, onChange, fmt = String }: {
  label: string; value: number; options: number[]; onChange: (v: number) => void; fmt?: (v: number) => string;
}) {
  return (
    <label className="field">
      {label}
      <span className="seg" role="group" aria-label={label}>
        {options.map((o) => (
          <button key={o} aria-pressed={value === o} onClick={() => onChange(o)}>{fmt(o)}</button>
        ))}
      </span>
    </label>
  );
}

/** The reader's Metal threadgroup-memory limit, if a report has it. */
export function useMetalLimit(): { bytes: number; device: string } | null {
  const { reports } = useReports();
  const d = reports.metal?.device;
  const bytes = Number(d?.maxThreadgroupMemoryLength);
  return bytes > 0 ? { bytes, device: String(d?.name ?? tr('your Mac', 'máy Mac của bạn')) } : null;
}

export function FlashFit() {
  const [cfg, setCfg] = useState<PlanInput>({ ...BOOK });
  const set = (k: keyof PlanInput) => (v: number) => setCfg((c) => ({ ...c, [k]: v }));
  const p = useMemo(() => plan(cfg), [cfg]);
  const metal = useMetalLimit();
  const lims = limits(metal?.bytes, metal ? tr(`${metal.device} (your report)`, `${metal.device} (báo cáo của bạn)`) : undefined);
  const isBook = (Object.keys(BOOK) as (keyof PlanInput)[]).every((k) => cfg[k] === BOOK[k]);
  const metalLimit = lims.find((l) => l.id === 'metal')!.bytes;
  const { bc: _bc, ...rest } = cfg;
  const bestBc = largestFittingBc(rest, [64, 32, 16, 8], metalLimit);
  const ok = p.problems.length === 0;

  return (
    <Bench
      title={tr('Fit the tiles', 'Cho các ô vừa bộ nhớ')}
      task={
        <Translated vi="Tìm 37,376 byte và 20 thanh ghi của sách, rồi một cấu hình vừa với Metal.">
          Find the book’s 37,376 bytes and 20 registers, then a configuration that fits Metal.
        </Translated>
      }
      foot={
        <Translated
          vi={
            <>
              Bộ nhớ chia sẻ tính ba mảng ô của Hình 20.9 (KT_j được đệm thêm một float cho mỗi số-bank phần tử, S_i,
              V_j), 4 byte mỗi float; thanh ghi tính O_i, Q_i, m_i và D_i cho mỗi luồng. Giới hạn: CUDA ch05 facts A3, AMD
              ch20 facts B3, Metal {metal ? 'từ báo cáo của bạn' : 'ch05 facts C1 (máy này chưa có báo cáo Metal)'}.
            </>
          }
        >
          <>
            Shared memory counts the three tile arrays of Fig. 20.9 (KT_j padded with one float per bank-count
            elements, S_i, V_j), 4 bytes per float; registers count O_i, Q_i, m_i and D_i per thread. Limits: CUDA ch05
            facts A3, AMD ch20 facts B3, Metal {metal ? 'from your report' : 'ch05 facts C1 (no Metal report on this machine yet)'}.
          </>
        </Translated>
      }
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <Seg label={tr('Head dimension d', 'Số chiều head d')} value={cfg.d} options={[64, 128, 256]} onChange={set('d')} />
        <Seg label="B_r" value={cfg.br} options={[8, 16, 32, 64]} onChange={set('br')} />
        <Seg label="B_c" value={cfg.bc} options={[8, 16, 32, 64]} onChange={set('bc')} />
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <Seg label={tr('Threads per block', 'Số luồng mỗi khối')} value={cfg.threads} options={[128, 256, 512, 1024]} onChange={set('threads')} />
        <Seg label={tr('Warp width', 'Độ rộng warp')} value={cfg.warp} options={[32, 64]} onChange={set('warp')} />
        <Seg label={tr('Padding for', 'Đệm cho')} value={cfg.banks} options={[32, 64, 0]} onChange={set('banks')} fmt={(v) => (v ? tr(`${v} banks`, `${v} bank`) : tr('off', 'tắt'))} />
        <button onClick={() => setCfg({ ...BOOK })} disabled={isBook}><Translated vi="Ví dụ của sách">The book’s example</Translated></button>
      </div>

      <div className="cols">
        <table className="data" style={{ margin: 0, background: 'var(--page)' }} data-plan>
          <tbody>
            <tr><td><Translated vi="N_WARPS = số luồng / độ rộng warp">N_WARPS = threads / warp width</Translated></td><td data-v="nWarps">{Number.isInteger(p.nWarps) ? p.nWarps : p.nWarps.toFixed(2)}</td></tr>
            <tr><td>B_r_warp = B_r / N_WARPS</td><td data-v="brWarp" className={Number.isInteger(p.brWarp) && p.brWarp >= 1 ? '' : 't-fail'}>{Number.isInteger(p.brWarp) ? p.brWarp : p.brWarp.toFixed(3)}</td></tr>
            <tr><td><Translated vi="d_size = d / độ rộng warp">d_size = d / warp width</Translated></td><td data-v="dSize">{p.dSize}</td></tr>
            <tr><td className="t-shared">KT_j (d × B_c{p.floats.ktPad ? tr(` + ${int(p.floats.ktPad)} padding`, ` + ${int(p.floats.ktPad)} đệm`) : ''}){/* i18n-ignore: array shape formula */}</td><td data-v="kt">{int(p.bytes.kt)} B</td></tr>
            <tr><td className="t-shared">S_i (B_r × B_c)</td><td data-v="s">{int(p.bytes.s)} B</td></tr>
            <tr><td className="t-shared">V_j (B_c × d)</td><td data-v="v">{int(p.bytes.v)} B</td></tr>
            <tr><td className="t-shared"><b><Translated vi="bộ nhớ chia sẻ mỗi khối">shared memory per block</Translated></b></td><td data-v="total"><b>{int(p.bytes.total)} B</b></td></tr>
            <tr>
              <td className="t-reg">
                <Translated vi={<>thanh ghi mỗi luồng (O {p.regs.o} + Q {p.regs.q} + m {p.regs.m} + D {p.regs.D})</>}>
                  registers per thread (O {p.regs.o} + Q {p.regs.q} + m {p.regs.m} + D {p.regs.D})
                </Translated>
              </td>
              <td data-v="regs">{ok ? tr(`${p.regs.total} floats`, `${p.regs.total} float`) : '—'}</td>
            </tr>
          </tbody>
        </table>
        <div>
          {p.problems.length ? (
            <div className="narrate bad" style={{ marginBottom: '0.8rem' }} data-problems>
              {p.problems.map((x) => <div key={x}>{x}.</div>)}
            </div>
          ) : null}
          {lims.map((l) => {
            const f = fits(p.bytes.total, l.bytes);
            const frac = Math.min(1, p.bytes.total / l.bytes);
            return (
              <div key={l.id} style={{ marginBottom: '0.7rem' }} data-limit={l.id} data-fits={f ? 'yes' : 'no'}>
                <div className="row" style={{ justifyContent: 'space-between', gap: '0.4rem' }}>
                  <span>{l.label}: {int(l.bytes)} B</span>
                  {f ? <span className="chip ok"><Translated vi="vừa">fits</Translated></span> : <span className="chip fail"><Translated vi="không vừa">does not fit</Translated></span>}
                </div>
                <div style={{ height: 10, background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 2, marginTop: 3 }}>
                  <div style={{ width: `${frac * 100}%`, height: '100%', background: f ? 'var(--shared)' : 'var(--fail)' }} />
                </div>
              </div>
            );
          })}
          <Translated
            vi={
              <p className="muted" style={{ margin: '0.4rem 0 0' }} data-best-bc>
                Với d = {cfg.d} và B_r = {cfg.br}, B_c lớn nhất trong 64, 32, 16 và 8 vừa với {int(metalLimit)} B là{' '}
                {bestBc ?? 'không giá trị nào'}.
              </p>
            }
          >
            <p className="muted" style={{ margin: '0.4rem 0 0' }} data-best-bc>
              At d = {cfg.d} and B_r = {cfg.br}, the largest B_c of 64, 32, 16 and 8 that fits {int(metalLimit)} B is{' '}
              {bestBc ?? 'none of them'}.
            </p>
          </Translated>
        </div>
      </div>
    </Bench>
  );
}
