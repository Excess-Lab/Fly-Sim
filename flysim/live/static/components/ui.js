// Small form and layout primitives: switch, segmented control, slider+number, dialog, confirm.
import { LitElement, html, css, nothing } from '../vendor/lit-core.min.js';
import { shared } from '../styles.js';
import { fmtNum } from '../lib/api.js';
import { icons } from '../lib/icons.js';

const emit = (el, value) => el.dispatchEvent(new CustomEvent('value-changed', { detail: { value }, bubbles: true, composed: true }));

const fieldCSS = css`
  :host { display: block; padding: var(--sp-3) 0; border-bottom: 1px solid var(--border); }
  :host(:last-child) { border-bottom: 0; }
  :host([disabled]) { opacity: .45; pointer-events: none; }
  .top { display: flex; justify-content: space-between; align-items: center; gap: var(--sp-3); min-height: 28px; }
  .label { font-weight: 500; min-width: 0; }
  .range { color: var(--text-3); font-size: var(--fs-xs); font-weight: 400; margin-left: 6px; white-space: nowrap; }
  .norm { color: var(--text-2); font-size: var(--fs-xs); margin-top: 6px; line-height: 1.45; }
  .qb { all: unset; display: inline-grid; place-items: center; width: 17px; height: 17px; margin-left: 6px; border-radius: 50%;
    border: 1px solid var(--border-strong); color: var(--text-3); font-size: 11px; font-weight: 600; cursor: pointer;
    vertical-align: 1px; }
  .qb:hover, .qb[aria-expanded="true"] { color: var(--accent); border-color: var(--accent); }
  .help { background: var(--surface-2); border-radius: var(--r-s); padding: var(--sp-2) var(--sp-3); margin-top: 6px;
    font-size: var(--fs-xs); line-height: 1.5; color: var(--text-2); }
`;

// "?" button + expandable explanation (what the setting is, how the model uses it)
const helpBtn = el => el.help ? html`<button class="qb" aria-label="Что это?" title="Что это и как работает"
  aria-expanded=${el._hopen ? 'true' : 'false'} @click=${e => { e.stopPropagation(); el._hopen = !el._hopen; }}>?</button>` : nothing;
const helpText = el => el.help && el._hopen ? html`<div class="help">${el.help}</div>` : nothing;

/** Boolean switch with label and optional hint. */
export class FsSwitch extends LitElement {
  static properties = { label: {}, norm: {}, help: {}, _hopen: { state: true }, checked: { type: Boolean }, disabled: { type: Boolean, reflect: true } };
  static styles = [shared, fieldCSS, css`
    button { all: unset; position: relative; width: 38px; height: 22px; border-radius: 999px; background: var(--surface-3);
      cursor: pointer; flex: none; transition: background .15s; border: 1px solid var(--border-strong); }
    button::after { content: ""; position: absolute; top: 2px; left: 2px; width: 16px; height: 16px; border-radius: 50%;
      background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,.25); transition: transform .15s; }
    button[aria-checked="true"] { background: var(--accent); border-color: var(--accent); }
    button[aria-checked="true"]::after { transform: translateX(16px); }
    button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    .top { cursor: pointer; }
  `];
  render() {
    return html`<div class="top" @click=${this._toggle}>
        <span class="label">${this.label}${helpBtn(this)}</span>
        <button role="switch" aria-checked=${this.checked ? 'true' : 'false'} aria-label=${this.label}
          @click=${e => { e.stopPropagation(); this._toggle(); }}></button>
      </div>${this.norm ? html`<div class="norm">${this.norm}</div>` : nothing}${helpText(this)}`;
  }
  _toggle() { if (this.disabled) return; this.checked = !this.checked; emit(this, this.checked); }
}
customElements.define('fs-switch', FsSwitch);

/** Segmented control (radio group). options: [[value, shortLabel, title?], …]. */
export class FsSegmented extends LitElement {
  static properties = { options: { type: Array }, value: {}, size: {} };
  static styles = [shared, css`
    :host { display: inline-flex; max-width: 100%; }
    .seg { display: inline-flex; flex-wrap: wrap; gap: 2px; padding: 2px; background: var(--surface-2);
      border: 1px solid var(--border); border-radius: var(--r-s); max-width: 100%; }
    button { all: unset; box-sizing: border-box; padding: 0 var(--sp-3); height: 26px; line-height: 26px; border-radius: 4px;
      cursor: pointer; font-size: var(--fs-s); color: var(--text-2); white-space: nowrap; text-align: center; }
    :host([size="sm"]) button { height: 22px; line-height: 22px; padding: 0 var(--sp-2); font-size: var(--fs-xs); }
    button:hover { color: var(--text); }
    button[aria-checked="true"] { background: var(--surface); color: var(--text); font-weight: 600; box-shadow: var(--shadow); }
    button:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
  `];
  constructor() { super(); this.options = []; }
  render() {
    return html`<div class="seg" role="radiogroup">${this.options.map(([v, l, t]) => html`
      <button role="radio" aria-checked=${String(v) === String(this.value) ? 'true' : 'false'} title=${t || l}
        tabindex=${String(v) === String(this.value) ? 0 : -1}
        @click=${() => this._set(v)} @keydown=${this._key}>${l}</button>`)}</div>`;
  }
  _set(v) { this.value = v; emit(this, v); }
  _key(e) {
    const i = this.options.findIndex(o => String(o[0]) === String(this.value));
    const d = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    const n = (i + d + this.options.length) % this.options.length;
    this._set(this.options[n][0]);
    this.updateComplete.then(() => this.renderRoot.querySelectorAll('button')[n]?.focus());
  }
}
customElements.define('fs-segmented', FsSegmented);

/** Labelled field wrapper for a segmented choice or a native select (when there are many options). */
export class FsChoice extends LitElement {
  static properties = { label: {}, norm: {}, help: {}, _hopen: { state: true }, options: { type: Object }, value: {}, disabled: { type: Boolean, reflect: true } };
  static styles = [shared, fieldCSS, css`
    .top { flex-wrap: wrap; }
    .full { color: var(--text-2); font-size: var(--fs-xs); margin-top: 6px; }
  `];
  render() {
    const entries = Object.entries(this.options || {});
    const short = l => String(l).split(/[:(]/)[0].trim();
    const seg = entries.length <= 4 && entries.every(([, l]) => short(l).length <= 22);
    const cur = this.options?.[this.value];
    return html`<div class="top"><span class="label">${this.label}${helpBtn(this)}</span>
      ${seg ? html`<fs-segmented .options=${entries.map(([v, l]) => [v, short(l), l])} .value=${this.value}></fs-segmented>`
        : html`<select class="select" aria-label=${this.label} @change=${e => emit(this, e.target.value)}>
            ${entries.map(([v, l]) => html`<option value=${v} ?selected=${v === this.value}>${l}</option>`)}</select>`}</div>
      ${seg && cur && short(cur) !== cur ? html`<div class="full">${cur}</div>` : nothing}
      ${this.norm ? html`<div class="norm">${this.norm}</div>` : nothing}${helpText(this)}`;
  }
}
customElements.define('fs-choice', FsChoice);

/** Numeric field: label, value + unit readout, slider paired with a number input, range and norm hint. */
export class FsRange extends LitElement {
  static properties = { label: {}, unit: {}, norm: {}, help: {}, _hopen: { state: true }, min: { type: Number }, max: { type: Number }, step: { type: Number },
    value: { type: Number }, disabled: { type: Boolean, reflect: true }, display: { attribute: false } };
  static styles = [shared, fieldCSS, css`
    .ctl { display: flex; align-items: center; gap: var(--sp-3); margin-top: var(--sp-2); }
    input[type=range] { flex: 1; min-width: 0; accent-color: var(--accent); margin: 0; height: 22px; }
    .numbox { display: inline-flex; align-items: center; border: 1px solid var(--border-strong); border-radius: var(--r-s);
      background: var(--surface); height: 30px; flex: none; }
    .numbox:focus-within { outline: 2px solid var(--accent); outline-offset: 1px; }
    input[type=number] { width: 76px; border: 0; background: transparent; padding: 0 4px 0 8px; height: 28px; text-align: right;
      font-variant-numeric: tabular-nums; -moz-appearance: textfield; outline: none; }
    input[type=number]::-webkit-inner-spin-button { opacity: .5; }
    .unit { color: var(--text-3); font-size: var(--fs-xs); padding-right: 8px; white-space: nowrap; }
    .shown { color: var(--text-2); font-size: var(--fs-xs); font-variant-numeric: tabular-nums; }
  `];
  render() {
    const s = this.step || 1;
    return html`<div class="top"><span class="label">${this.label}${helpBtn(this)}<span class="range">${fmtNum(this.min, s)}–${fmtNum(this.max, s)}${this.unit ? ' ' + this.unit : ''}</span></span>
        ${this.display ? html`<span class="shown">${this.display(this.value)}</span>` : nothing}</div>
      <div class="ctl">
        <input type="range" min=${this.min} max=${this.max} step=${s} .value=${String(this.value)} aria-label=${this.label}
          @input=${e => this._set(+e.target.value)}>
        <span class="numbox"><input type="number" min=${this.min} max=${this.max} step=${s} .value=${fmtNum(this.value, s)}
          aria-label=${`${this.label}, ${this.unit || ''}`} @change=${this._num}>${this.unit ? html`<span class="unit">${this.unit}</span>` : nothing}</span>
      </div>${this.norm ? html`<div class="norm">${this.norm}</div>` : nothing}${helpText(this)}`;
  }
  _num(e) {
    let v = parseFloat(String(e.target.value).replace(',', '.'));
    if (!isFinite(v)) v = this.value;
    v = Math.min(this.max, Math.max(this.min, v));
    this._set(v); e.target.value = fmtNum(v, this.step);
  }
  _set(v) { this.value = v; emit(this, v); }
}
customElements.define('fs-range', FsRange);

/** Modal dialog built on <dialog>: heading, body slot, footer slot. Fires 'close'. */
export class FsDialog extends LitElement {
  static properties = { heading: {}, open: { type: Boolean }, wide: { type: Boolean, reflect: true }, _foot: { state: true } };
  static styles = [shared, css`
    dialog { border: 1px solid var(--border); border-radius: var(--r-l); background: var(--surface); color: var(--text);
      padding: 0; width: min(560px, calc(100vw - 24px)); max-height: calc(100vh - 48px); box-shadow: var(--shadow-l); }
    :host([wide]) dialog { width: min(880px, calc(100vw - 24px)); }
    dialog::backdrop { background: rgba(10, 10, 14, .45); }
    .h { display: flex; align-items: center; justify-content: space-between; gap: var(--sp-2);
      padding: var(--sp-4) var(--sp-4) var(--sp-3); position: sticky; top: 0; background: var(--surface); z-index: 1;
      border-bottom: 1px solid var(--border); }
    .b { padding: var(--sp-4); }
    .f { display: flex; gap: var(--sp-2); justify-content: flex-end; flex-wrap: wrap; padding: var(--sp-3) var(--sp-4);
      border-top: 1px solid var(--border); position: sticky; bottom: 0; background: var(--surface); }
  `];
  render() {
    return html`<dialog @close=${this._closed} @click=${e => { if (e.target.tagName === 'DIALOG') this.close(); }}>
      <div class="h"><h3>${this.heading}</h3><slot name="actions"></slot>
        <button class="btn ghost icon" aria-label="Закрыть" @click=${() => this.close()}>${icons.x}</button></div>
      <div class="b"><slot></slot></div>
      <div class="f" ?hidden=${!this._foot}><slot name="footer"
        @slotchange=${e => { this._foot = e.target.assignedElements().length > 0; }}></slot></div></dialog>`;
  }
  updated(ch) {
    const d = this.renderRoot.querySelector('dialog');
    if (ch.has('open')) { if (this.open && !d.open) d.showModal(); if (!this.open && d.open) d.close(); }
  }
  firstUpdated() { if (this.open) this.renderRoot.querySelector('dialog').showModal(); }
  close() { this.open = false; }
  _closed() { this.open = false; this.dispatchEvent(new CustomEvent('close')); }
}
customElements.define('fs-dialog', FsDialog);

/** Promise-based confirm dialog. */
export function confirmDialog({ title, text, ok = 'Удалить', danger = true }) {
  return new Promise(resolve => {
    const d = document.createElement('fs-dialog');
    d.heading = title;
    let result = false;
    const p = document.createElement('p');
    p.textContent = text;
    p.style.cssText = 'margin:0;color:var(--text-2);line-height:1.5';
    const mk = (label, cls, val) => {
      const b = document.createElement('button');
      b.textContent = label; b.className = cls; b.slot = 'footer';
      b.style.cssText = `height:32px;padding:0 14px;border-radius:6px;font:500 13px var(--font);cursor:pointer;
        border:1px solid ${val ? (danger ? 'var(--danger)' : 'var(--accent)') : 'var(--border-strong)'};
        background:${val ? (danger ? 'var(--danger)' : 'var(--accent)') : 'var(--surface)'};color:${val ? '#fff' : 'var(--text)'}`;
      b.onclick = () => { result = val; d.close(); };
      return b;
    };
    d.append(p, mk('Отмена', 'cancel', false), mk(ok, 'ok', true));
    d.addEventListener('close', () => { d.remove(); resolve(result); });
    document.body.appendChild(d);
    d.open = true;
  });
}
