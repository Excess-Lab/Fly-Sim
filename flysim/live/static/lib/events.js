// Event row template + styles shared by the observe view, chronicle and passports.
import { html, css, nothing } from '../vendor/lit-core.min.js';
import { CAT_COLORS, CAT_RU, fmtClock, sexVar } from './api.js';

export const eventCSS = css`
  .evlist { overflow: auto; overscroll-behavior: contain; }
  .ev { display: block; width: 100%; text-align: left; border: 0; background: none; border-bottom: 1px solid var(--border);
    padding: var(--sp-2) var(--sp-4) var(--sp-2) calc(var(--sp-4) - 3px); border-left: 3px solid transparent; cursor: pointer;
    font-size: var(--fs-s); color: var(--text); }
  .ev:hover { background: var(--surface-2); }
  .ev.open { background: var(--surface-2); }
  .ev.static { cursor: default; }
  .ev.imp3 { border-left-color: var(--danger); }
  .ev.imp2 { border-left-color: var(--warn); }
  .ev .who { font-weight: 600; margin-right: 4px; }
  .ev .meta { color: var(--text-3); font-size: var(--fs-xs); display: flex; gap: 2px var(--sp-3); flex-wrap: wrap; margin-top: 2px;
    font-variant-numeric: tabular-nums; }
  .ev .ev-cat { display: inline-flex; align-items: center; gap: 5px; }
  .ev-empty { padding: var(--sp-4); color: var(--text-3); font-size: var(--fs-s); text-align: center; }
`;

export const catDot = c => html`<span class="dot" style="background:${CAT_COLORS[c] || '#999'}"></span>`;

export function eventRow(e, { open = false, onClick = null, extra = nothing } = {}) {
  return html`<div class="ev imp${e.imp || 1} ${open ? 'open' : ''} ${onClick ? '' : 'static'}" data-id=${e.id}
      role=${onClick ? 'button' : nothing} tabindex=${onClick ? 0 : nothing}
      @click=${onClick || nothing} @keydown=${onClick ? (k => { if (k.key === 'Enter' && k.target === k.currentTarget) onClick(k); }) : nothing}>
    <div><span class="who" style="color:${sexVar(e.sex)}">${e.fly}</span>${e.text}</div>
    <div class="meta"><span class="ev-cat">${catDot(e.cat)}${CAT_RU[e.cat] || e.cat}</span><span>${fmtClock(e.clock_h)}</span>
      <span>сессия ${e.session ?? '—'} · ${(+e.t || 0).toFixed(1)} с</span><span>кадр ${e.frame}</span>
      <span title="важность">${'★'.repeat(e.imp || 1)}</span></div>${extra}</div>`;
}
