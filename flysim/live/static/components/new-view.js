// New observation: free simulation (form generated from /api/schema) or a prepared experiment.
import { LitElement, html, css, nothing } from '../vendor/lit-core.min.js';
import { shared } from '../styles.js';
import { SCHEMA, post, go, toast, fmtHM } from '../lib/api.js';
import { icons } from '../lib/icons.js';
import './ui.js';
import './apple-editor.js';

const GROUP_TITLES = {
  environment: ['Воздух', 'газы, температура, объём и вентиляция'],
  world: ['Мир', 'размер, климат, сезоны, почва, ветер'],
  apples: ['Яблоки', 'сколько, какие и где лежат'],
  flies: ['Мухи', 'численность и состояние на старте'],
  injuries: ['Травмы и схема тела', 'задаёт экспериментатор'],
  hot_zone: ['Неприятный стимул', 'горячая зона на полу'],
  science: ['Контроли', 'научные режимы и обходы'],
  run: ['Расчёт и лимиты', 'подробные отрезки, перемотка, остановка'],
};
// fields that only matter for one climate
const CLIMATE_ONLY = { wind_speed: 'container', wind_dir_deg: 'container', start_hour: 'garden', garden_wind: 'garden' };
const DISPLAY = { start_hour: v => fmtHM(v) };

// form state survives navigation inside the page
let CFG = null, MODE = 'free', TITLE = '', EXP = null;
const clone = o => JSON.parse(JSON.stringify(o));

class FsNew extends LitElement {
  static properties = { cfg: { state: true }, mode: { state: true }, exp: { state: true }, starting: { state: true } };
  static styles = [shared, css`
    .modes { margin-bottom: var(--sp-5); display: flex; gap: var(--sp-3); flex-wrap: wrap; align-items: center; }
    .modes .hint { color: var(--text-3); font-size: var(--fs-s); }
    .cols { columns: 380px; column-gap: var(--sp-3); }
    .cols > .card { break-inside: avoid; margin-bottom: var(--sp-3); display: inline-block; width: 100%; }
    .card-b { padding-top: 0; padding-bottom: 0; }
    .card-h .hint { margin-left: auto; }
    .head-actions { display: flex; gap: var(--sp-2); flex-wrap: wrap; align-items: center; }
    .head-actions .input { width: 240px; }
    .callout { background: var(--surface-2); border-radius: var(--r-s); padding: var(--sp-3); font-size: var(--fs-s); line-height: 1.5; }
    .callout b { display: block; color: var(--text-2); font-weight: 500; font-size: var(--fs-xs); text-transform: uppercase;
      letter-spacing: .04em; margin-bottom: 2px; }
    .exp .card-b { display: grid; gap: var(--sp-4); padding: var(--sp-4); }
    .exp label.fl { display: grid; gap: var(--sp-2); font-weight: 500; }
    .note { color: var(--text-3); font-size: var(--fs-xs); }
    .hidden-note { font-size: var(--fs-xs); color: var(--text-3); padding: var(--sp-2) 0 var(--sp-3); }
    @media (max-width: 560px) { .head-actions { width: 100%; } .head-actions .input { flex: 1; width: auto; } }
  `];

  constructor() {
    super();
    if (!CFG) CFG = clone(SCHEMA.defaults);
    const protos = Object.keys(SCHEMA.protocols || {});
    if (!EXP) EXP = { protocol: protos[0], group: SCHEMA.protocols?.[protos[0]]?.groups?.[0], seed: 100 };
    this.cfg = CFG; this.mode = MODE; this.exp = EXP; this.starting = false;
  }

  _set(g, k, v) {
    this.cfg = CFG = { ...this.cfg, [g]: { ...this.cfg[g], [k]: v } };
  }
  _setExp(patch) {
    this.exp = EXP = { ...this.exp, ...patch };
    if (patch.protocol) this.exp = EXP = { ...EXP, group: SCHEMA.protocols[patch.protocol].groups[0] };
  }
  _mode(m) { this.mode = MODE = m; }
  _reset() { this.cfg = CFG = clone(SCHEMA.defaults); toast('настройки сброшены'); }

  async _start() {
    this.starting = true;
    let r;
    try {
      r = this.mode === 'exp'
        ? await post('/api/experiment', { protocol: this.exp.protocol, group: this.exp.group, seed: +this.exp.seed })
        : await post('/api/start', { title: TITLE, ...this.cfg });
    } catch (e) { r = { error: 'нет связи с сервером' }; }
    this.starting = false;
    if (r && r.run_id) go('#/observe/' + r.run_id);
    else toast(r?.error || 'не удалось запустить', 'error');
  }

  _field(g, k, s) {
    const v = this.cfg[g][k];
    const on = e => this._set(g, k, e.detail.value);
    const off = g === 'hot_zone' && k !== 'enabled' && !this.cfg.hot_zone.enabled;
    if (s.type === 'apples') {
      return html`<fs-apple-editor .apples=${v} .half=${this.cfg.world.size_m * 500} .norm=${s.norm}
        style="padding:var(--sp-3) 0" @value-changed=${on}></fs-apple-editor>`;
    }
    if (s.type === 'select') return html`<fs-choice .label=${s.label} .options=${s.options} .value=${v} .norm=${s.norm || ''}
      ?disabled=${off} @value-changed=${on}></fs-choice>`;
    if (s.type === 'bool') return html`<fs-switch .label=${s.label} .checked=${!!v} .norm=${s.norm || ''} ?disabled=${off}
      @value-changed=${on}></fs-switch>`;
    return html`<fs-range .label=${s.label} .unit=${s.unit || ''} .min=${s.min} .max=${s.max} .step=${s.step} .value=${+v}
      .norm=${s.norm ? (s.norm.startsWith('только') ? s.norm : 'Норма: ' + s.norm) : ''} .display=${DISPLAY[k] || null}
      ?disabled=${off} @value-changed=${on}></fs-range>`;
  }

  _group(g, items) {
    const [title, hint] = GROUP_TITLES[g] || [g, ''];
    const climate = this.cfg.world?.climate || 'container';
    const entries = Object.entries(items).filter(([k]) => g !== 'world' || !CLIMATE_ONLY[k] || CLIMATE_ONLY[k] === climate);
    const hidden = Object.keys(items).length - entries.length;
    return html`<section class="card"><div class="card-h"><h3>${title}</h3><span class="hint">${hint}</span></div>
      <div class="card-b">${entries.map(([k, s]) => this._field(g, k, s))}
        ${hidden ? html`<div class="hidden-note">${climate === 'garden'
          ? 'В саду ветер задаёт погода — постоянный ветер контейнера скрыт.'
          : 'Время суток и сила ветра в саду скрыты: они нужны только для климата «сад».'}</div>` : nothing}
      </div></section>`;
  }

  _experiment() {
    const P = SCHEMA.protocols || {}, p = P[this.exp.protocol];
    if (!p) return html`<div class="empty">Нет готовых протоколов.</div>`;
    return html`<section class="card exp" style="max-width:760px"><div class="card-h"><h3>Готовый эксперимент</h3>
        <span class="hint">протокол с гипотезой, группами и контролями</span></div>
      <div class="card-b">
        <label class="fl">Протокол
          <select class="select" @change=${e => this._setExp({ protocol: e.target.value })}>
            ${Object.keys(P).map(k => html`<option value=${k} ?selected=${k === this.exp.protocol}>${k}</option>`)}</select></label>
        <div class="callout"><b>Гипотеза</b>${p.hypothesis}</div>
        <div class="callout"><b>С чем сравнивать</b>${p.reference}</div>
        <div class="fl" style="display:grid;gap:8px;font-weight:500">Группа
          <fs-segmented .options=${p.groups.map(g => [g, g])} .value=${this.exp.group}
            @value-changed=${e => this._setExp({ group: e.detail.value })}></fs-segmented></div>
        <label class="fl">Seed
          <input class="input num" type="number" style="width:140px" .value=${String(this.exp.seed)}
            @change=${e => this._setExp({ seed: +e.target.value || 0 })}>
          <span class="note">Повторы — разные seed; для статистики — серия (scripts/series.py).</span></label>
      </div></section>`;
  }

  render() {
    return html`
      <div class="page-head"><div><h1>Новое наблюдение</h1>
        <p class="sub">Настройте условия и запустите. Поведением управляет симуляция коннектома; приближения помечены в паспорте мухи.</p></div>
        <div class="head-actions">
          ${this.mode === 'free' ? html`<input class="input" placeholder="Название (необязательно)" aria-label="Название"
              .value=${TITLE} @input=${e => { TITLE = e.target.value; }}>
            <button class="btn" @click=${this._reset}>Сбросить</button>` : nothing}
          <button class="btn primary" ?disabled=${this.starting} @click=${this._start}>
            ${this.starting ? 'Загрузка мозгов…' : html`${icons.play}Запустить`}</button>
        </div></div>
      <div class="modes">
        <fs-segmented .options=${[['free', 'Свободная симуляция'], ['exp', 'Готовый эксперимент']]} .value=${this.mode}
          @value-changed=${e => this._mode(e.detail.value)}></fs-segmented>
        <span class="hint">${this.mode === 'free' ? 'мир, яблоки и мухи по вашим условиям' : 'воспроизводимый протокол из научной литературы'}</span>
      </div>
      ${this.mode === 'exp' ? this._experiment()
        : html`<div class="cols">${Object.entries(SCHEMA.schema).map(([g, items]) => this._group(g, items))}</div>`}`;
  }
}
customElements.define('fs-new', FsNew);
