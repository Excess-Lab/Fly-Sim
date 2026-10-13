// One fly: whole-brain activity (rotatable), region shares, internal state, body schema.
import { LitElement, html, css, nothing } from '../vendor/lit-core.min.js';
import { shared } from '../styles.js';
import { MODES, sexVar, fmtInt, regionShares } from '../lib/api.js';
import { brainOf, drawBrain, attachOrbit } from '../lib/draw.js';
import { icons } from '../lib/icons.js';

const LEG = { lf: 'Л перед.', lm: 'Л сред.', lh: 'Л задн.', rf: 'П перед.', rm: 'П сред.', rh: 'П задн.' };
const LEG_ST = { intact: 'цела', paralyzed: 'парализ.', amputated: 'нет' };
const WHERE = { apple: 'яблоко', floor: 'пол', air: 'воздух' };

export function regionBar(f) {
  if (!f?.regions) return nothing;
  return html`<div class="stackbar" role="img" aria-label="Доли активности областей мозга">
    ${regionShares(f).map(r => html`<div title="${r.name}: ${Math.round(r.pct)}%" style="width:${r.pct}%;background:${r.color}"></div>`)}</div>`;
}

class FsFlyCard extends LitElement {
  static properties = { name: {}, fly: { attribute: false }, runId: {}, _ready: { state: true }, _err: { state: true } };
  static styles = [shared, css`
    :host { display: block; }
    .card-h h3 { font-size: var(--fs-m); }
    .name { font-weight: 700; }
    .ds { color: var(--text-3); font-size: var(--fs-xs); }
    .brain { position: relative; background: var(--brain-bg); }
    .brain canvas { aspect-ratio: 16 / 10; cursor: grab; touch-action: pan-y; }
    .brain .tip { position: absolute; right: 8px; bottom: 6px; font-size: 10.5px; color: #8b8b92; pointer-events: none; }
    .sec { padding: var(--sp-3) var(--sp-4); border-top: 1px solid var(--border); }
    .sec h4 { font-size: var(--fs-xs); color: var(--text-3); font-weight: 600; text-transform: uppercase; letter-spacing: .05em;
      margin-bottom: var(--sp-2); }
    .body { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--sp-2) var(--sp-3); font-size: var(--fs-xs); color: var(--text-2); }
    .body b { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }
    .body .bad b { color: var(--danger); }
    .regions-top { display: flex; gap: var(--sp-1) var(--sp-3); flex-wrap: wrap; margin-top: var(--sp-2); font-size: var(--fs-xs); color: var(--text-2); }
    .regions-top span { display: inline-flex; gap: 5px; align-items: center; }
    @media (max-width: 420px) { .body { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  `];

  constructor() { super(); this.view = { az: 0, el: 0.15, zoom: 1 }; this._redrawTheme = () => this._draw(); }
  connectedCallback() { super.connectedCallback(); addEventListener('fs-theme', this._redrawTheme); }
  disconnectedCallback() { super.disconnectedCallback(); removeEventListener('fs-theme', this._redrawTheme); this._detach?.(); }

  willUpdate(ch) {
    if (ch.has('fly') && this.fly && !this._loading) {
      this._loading = true;
      brainOf(this.fly.ds || this.fly.sex).then(() => { this._ready = true; }, () => { this._err = true; this._loading = false; });
    }
  }
  firstUpdated() {
    const cv = this.renderRoot.querySelector('canvas');
    this._detach = attachOrbit(cv, (dx, dy, wh) => {
      this.view.az += dx * .01; this.view.el = Math.max(-1.4, Math.min(1.4, this.view.el + dy * .01));
      if (wh) this.view.zoom *= wh > 0 ? .9 : 1.1;
    }, () => this._draw());
  }
  updated() { this._draw(); }
  _draw() { const cv = this.renderRoot.querySelector('canvas'); if (cv && this.fly) drawBrain(cv, this.view, this.fly); }

  _passport() {
    this.dispatchEvent(new CustomEvent('open-passport', { detail: { name: this.name }, bubbles: true, composed: true }));
  }

  _stats(f) {
    const R = f.rates || {}, S = (k, v, t) => html`<div class="stat" title=${t || nothing}>${k}<b>${v}</b></div>`;
    return [
      S('Активных нейронов', fmtInt(f.n_active)),
      S('Где', `${f.mode === 'fly' ? 'летит' : 'идёт'} · ${WHERE[f.surface] || f.surface || '—'}`),
      S('Голод', (+f.hunger || 0).toFixed(2)), S('Зоб, мкл', (+f.crop_ul || 0).toFixed(3)),
      S('Здоровье', (+f.health || 0).toFixed(2)), S('Ест (MN9)', `${f.proboscis ? 'да' : 'нет'} · ${R.per ?? 0}`),
      S('Награда PAM', `${R.pam ?? 0} Гц`), S('Наказание PPL1', `${R.ppl1 ?? 0} Гц`),
      S('Песня pIP10', `${R.song ?? 0} Гц`), S('Возбуждение P1', `${R.arousal_P1 ?? 0} Гц`),
      S('Готовность DNp37', `${R.receptive ?? 0} Гц`), S('Яйцеклад oviDN', `${R.ovi ?? 0} Гц`),
      S('Спарена', f.mated ? 'да' : 'нет'), S('Память KC→MBON', f.mbon_w_min ?? '—', 'самый ослабленный синапс; 1 = исходно'),
      S('Температура тела', `${f.temp ?? '—'} °C`), S('Возраст', `${f.age_d ?? '—'} дн`),
    ];
  }

  _body(sc) {
    if (!sc) return nothing;
    const side = (o, ok) => `${o.left === ok ? 'Л' : '—'} ${o.right === ok ? 'П' : '—'}`;
    return html`<div class="sec"><h4>Схема тела</h4><div class="body">
      ${Object.entries(sc.legs || {}).map(([l, o]) => html`<div class=${o.state !== 'intact' ? 'bad' : ''}>${LEG[l] || l}:
        <b>${LEG_ST[o.state] || o.state}</b> · опора ${Math.round((o.contact || 0) * 100)}%</div>`)}
      <div>крылья: <b>${Math.round((sc.wings?.left || 0) * 100)}% / ${Math.round((sc.wings?.right || 0) * 100)}%</b></div>
      <div>усики: <b>${side(sc.antenna || {}, 'intact')}</b></div>
      <div>глаза: <b>${side(sc.eye || {}, 'open')}</b></div></div></div>`;
  }

  render() {
    const f = this.fly || {};
    const top = f.regions ? regionShares(f).sort((a, b) => b.pct - a.pct).slice(0, 3) : [];
    return html`<article class="card">
      <div class="card-h">
        <h3><span class="dot" style="background:${sexVar(f.sex)}"></span><span class="name">${this.name}</span>
          ${f.mode_mech ? html`<span class="badge ${f.mode_mech === 'experimental_shortcuts' ? 'warn' : ''}"
            title="режим механизмов">${MODES[f.mode_mech] || f.mode_mech}</span>` : nothing}</h3>
        <button class="btn sm" @click=${this._passport}>${icons.id}Паспорт</button>
      </div>
      <div class="brain"><canvas aria-label="Активность мозга ${this.name}; перетаскивайте, чтобы вращать"></canvas>
        <span class="tip">${this._err ? 'не удалось загрузить мозг' : 'тяните — вращать, колесо — масштаб'}</span></div>
      <div class="sec">${regionBar(f)}
        <div class="regions-top">${top.map(r => html`<span><span class="dot" style="background:${r.color}"></span>${r.name} ${Math.round(r.pct)}%</span>`)}
          <span class="ds">${f.dataset || ''}</span></div></div>
      <div class="sec"><h4>Внутреннее состояние</h4><div class="stats">${this._stats(f)}</div></div>
      ${this._body(f.schema)}
    </article>`;
  }
}
customElements.define('fs-fly-card', FsFlyCard);
