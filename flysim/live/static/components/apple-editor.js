// Apple list editor (new-run form) and single-apple editor (add-apple dialog), with a top-down placement map.
import { LitElement, html, svg, css, nothing } from '../vendor/lit-core.min.js';
import { shared } from '../styles.js';
import { icons } from '../lib/icons.js';
import './ui.js';

export const newApple = () => ({ age_days: 0, cut: true, radius_mm: 30, x_mm: 0, y_mm: 0 });

class FsAppleEditor extends LitElement {
  static properties = { apples: { type: Array }, half: { type: Number }, single: { type: Boolean }, norm: {} };
  static styles = [shared, css`
    :host { display: block; }
    .wrap { display: grid; gap: var(--sp-3); }
    .map { display: flex; gap: var(--sp-3); align-items: center; }
    .map svg { width: 112px; height: 112px; flex: none; border-radius: var(--r-s); background: var(--surface-2); border: 1px solid var(--border); }
    .apple { border: 1px solid var(--border); border-radius: var(--r-s); padding: var(--sp-1) var(--sp-3); background: var(--surface); }
    .ah { display: flex; justify-content: space-between; align-items: center; padding-top: var(--sp-2); }
    .ah b { display: inline-flex; gap: var(--sp-2); align-items: center; }
    .sw { width: 12px; height: 12px; border-radius: 50%; background: #b45309; }
    .norm { color: var(--text-2); font-size: var(--fs-xs); line-height: 1.45; }
  `];
  constructor() { super(); this.apples = []; this.half = 150; }

  _emit(list) { this.dispatchEvent(new CustomEvent('value-changed', { detail: { value: list }, bubbles: true, composed: true })); }
  _set(i, k, v) { const list = this.apples.map((a, j) => j === i ? { ...a, [k]: v } : a); this.apples = list; this._emit(list); }
  _on(i, k) { return e => { e.stopPropagation(); this._set(i, k, e.detail.value); }; }
  _del(i) { const list = this.apples.filter((_, j) => j !== i); this.apples = list; this._emit(list); }
  _add() { const list = [...this.apples, newApple()]; this.apples = list; this._emit(list); }

  _map() {
    const H = this.half || 150, S = 100 / (2 * H);
    return svg`<svg viewBox="0 0 100 100" role="img" aria-label="Схема расположения яблок (вид сверху)">
      <line x1="50" y1="0" x2="50" y2="100" stroke="var(--border-strong)" stroke-width="0.4" stroke-dasharray="2 2"/>
      <line x1="0" y1="50" x2="100" y2="50" stroke="var(--border-strong)" stroke-width="0.4" stroke-dasharray="2 2"/>
      ${this.apples.map((a, i) => svg`<circle cx=${50 + a.x_mm * S} cy=${50 - a.y_mm * S} r=${Math.max(1.5, a.radius_mm * S)}
        fill="#c2410c" fill-opacity="0.75" stroke="#7c2d12" stroke-width="0.5"/>
        <text x=${50 + a.x_mm * S} y=${50 - a.y_mm * S + 1.8} font-size="5" text-anchor="middle" fill="#fff">${i + 1}</text>`)}</svg>`;
  }

  render() {
    const h = Math.round(this.half || 150);
    return html`<div class="wrap">
      ${this.single ? nothing : html`<div class="map">${this._map()}
        <div class="norm">${this.norm}<br><span class="faint">Схема — вид сверху, сторона ${(2 * h / 1000).toFixed(2)} м.</span></div></div>`}
      ${this.apples.map((a, i) => html`<div class=${this.single ? '' : 'apple'}>
        ${this.single ? nothing : html`<div class="ah"><b><span class="sw"></span>Яблоко ${i + 1}</b>
          <button class="btn sm ghost danger" @click=${() => this._del(i)} aria-label="Удалить яблоко ${i + 1}">${icons.trash}Удалить</button></div>`}
        <fs-range label="Возраст" unit="дн" min="0" max="30" step="0.5" .value=${+a.age_days}
          @value-changed=${this._on(i, 'age_days')}></fs-range>
        <fs-switch label="Надрезанное" .checked=${!!a.cut} norm="надрез ускоряет брожение и плесень"
          @value-changed=${this._on(i, 'cut')}></fs-switch>
        <fs-range label="Радиус" unit="мм" min="10" max="40" step="1" .value=${+a.radius_mm}
          @value-changed=${this._on(i, 'radius_mm')}></fs-range>
        <fs-range label="Положение X" unit="мм" min=${-h} max=${h} step="5" .value=${+a.x_mm}
          .display=${v => `${(v / 1000).toFixed(2)} м`} @value-changed=${this._on(i, 'x_mm')}></fs-range>
        <fs-range label="Положение Y" unit="мм" min=${-h} max=${h} step="5" .value=${+a.y_mm}
          .display=${v => `${(v / 1000).toFixed(2)} м`} @value-changed=${this._on(i, 'y_mm')}></fs-range>
      </div>`)}
      ${this.single ? nothing : html`<div><button class="btn sm" @click=${this._add}>${icons.plus}Добавить яблоко</button></div>`}
    </div>`;
  }
}
customElements.define('fs-apple-editor', FsAppleEditor);
