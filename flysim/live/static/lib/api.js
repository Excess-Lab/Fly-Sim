// Server access, shared constants and small formatting helpers.

export async function api(path, opt) {
  const r = await fetch(path, opt);
  let body = null;
  try { body = await r.json(); } catch (e) { body = { error: `ответ сервера не JSON (${r.status})` }; }
  if (body && typeof body === 'object' && !Array.isArray(body)) body._status = r.status;
  return body;
}
export const post = (path, body) => api(path, { method: 'POST', body: JSON.stringify(body || {}) });

let schemaPromise = null;
/** /api/schema, loaded once: settings schema, defaults, apple norms, brain regions, protocols. */
export function loadSchema() {
  if (!schemaPromise) schemaPromise = api('/api/schema').catch(e => { schemaPromise = null; throw e; });
  return schemaPromise;
}
export let SCHEMA = null;
export function setSchema(s) { SCHEMA = s; }

export function b64(s, T) {
  const bin = atob(s), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new T(u.buffer);
}

export const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
export const sexColor = sex => cssVar(sex === 'male' ? '--male' : sex === 'female' ? '--female' : '--neutral');
export const sexVar = sex => sex === 'male' ? 'var(--male)' : sex === 'female' ? 'var(--female)' : 'var(--neutral)';

export const CAT_COLORS = { fight: '#dc2626', courtship: '#db2777', mating: '#a855f7', life: '#6b7280', feeding: '#16a34a',
  distress: '#ea580c', reward: '#22c55e', memory: '#2563eb', movement: '#9ca3af', social: '#0ea5e9', world: '#a16207', info: '#9ca3af' };
export const CAT_RU = { fight: 'драки', courtship: 'ухаживание', mating: 'спаривание и яйца', life: 'рождения и смерти',
  feeding: 'кормление', distress: 'дистресс (корреляты)', reward: 'награда (корреляты)', memory: 'память',
  movement: 'перемещения', social: 'контакты', world: 'мир и погода', info: 'прочее' };
export const MODES = { connectome_only: 'только коннектом', connectome_plus_approximations: 'коннектом + приближения',
  experimental_shortcuts: 'есть обходы' };

export const fmtClock = h => `д${Math.floor((h || 0) / 24)} ${((h || 0) % 24).toFixed(2)}ч`;
export function fmtHM(hour) {
  const m = Math.round(((hour % 24) + 24) % 24 * 60) % 1440;
  return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
}
export function fmtNum(v, step) {
  if (v === undefined || v === null || v === '') return '—';
  if (step && step < 1) return (+v).toFixed(String(step).split('.')[1]?.length || 1);
  return String(Math.round(+v));
}
export function fmtLux(lx) {
  if (lx === undefined || lx === null) return '—';
  if (lx < 0.1) return '<0.1 лк';
  if (lx < 10) return `${(+lx).toFixed(1)} лк`;
  if (lx < 1000) return `${Math.round(lx)} лк`;
  return `${(lx / 1000).toLocaleString('ru', { maximumFractionDigits: lx < 10000 ? 1 : 0 })} тыс. лк`;
}
export const fmtInt = v => (v ?? 0).toLocaleString('ru');

/** 0 (deep night) … 1 (bright day) on a log scale of illuminance. */
export function daylight(lux) {
  if (lux === undefined || lux === null) return 1;
  return Math.max(0, Math.min(1, (Math.log10(Math.max(lux, 0.001)) + 1) / 5));
}

export const go = h => { location.hash = h; };

export function toast(text, kind = '') {
  const d = document.createElement('div');
  d.textContent = text;
  d.setAttribute('role', 'status');
  d.style.cssText = `position:fixed;left:50%;bottom:24px;transform:translateX(-50%);max-width:calc(100vw - 32px);
    background:${kind === 'error' ? 'var(--danger)' : 'var(--text)'};color:var(--surface);padding:8px 14px;border-radius:8px;
    font:500 13px var(--font);box-shadow:var(--shadow-l);z-index:100;transition:opacity .2s`;
  document.body.appendChild(d);
  setTimeout(() => { d.style.opacity = '0'; setTimeout(() => d.remove(), 250); }, 1600);
}

export function copy(text) {
  const done = () => toast('скопировано');
  if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text, done));
  fallbackCopy(text, done);
}
function fallbackCopy(text, done) {
  const t = document.createElement('textarea');
  t.value = text; t.style.cssText = 'position:fixed;opacity:0';
  document.body.appendChild(t); t.select();
  try { document.execCommand('copy'); done(); } catch (e) { toast('не удалось скопировать', 'error'); }
  t.remove();
}

export function storageGet(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
export function storageSet(k, v) { try { if (v === null) localStorage.removeItem(k); else localStorage.setItem(k, v); } catch (e) { /* private mode */ } }

/** Region shares (percent) sorted, for a fly frame record. */
export function regionShares(f, skip = new Set()) {
  const reg = (f.regions || []).map((v, k) => skip.has(k) ? 0 : v);
  const tot = reg.reduce((a, b) => a + b, 0) || 1;
  return reg.map((v, k) => ({ k, name: SCHEMA.region_names[k], color: SCHEMA.region_colors[k], pct: 100 * v / tot }));
}

/** One-line text summary of a fly's brain in a frame (used for copy in the chronicle). */
export function brainSummary(n, f) {
  const reg = regionShares(f).sort((a, b) => b.pct - a.pct).slice(0, 4).map(r => `${r.name} ${Math.round(r.pct)}%`).join(', ');
  let top = '';
  if (f.flow) {
    const pairs = [];
    f.flow.forEach((row, a) => row.forEach((v, b) => { if (a !== b) pairs.push([v, a, b]); }));
    top = pairs.sort((x, y) => y[0] - x[0]).slice(0, 3)
      .map(([v, a, b]) => `${SCHEMA.region_names[a]}→${SCHEMA.region_names[b]} ${v}`).join('; ');
  }
  return `${n}: активных нейронов ${f.n_active}; преобладают: ${reg || '—'}; сильнейшие потоки: ${top || '—'}; ` +
    `PPL1 ${f.rates?.ppl1 ?? '—'} Гц, PAM ${f.rates?.pam ?? '—'} Гц; голод ${f.hunger}`;
}
