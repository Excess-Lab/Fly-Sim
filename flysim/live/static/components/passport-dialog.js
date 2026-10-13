// Fly passport: identity, origin, nervous system coverage, mechanisms mode, physiology, life events.
import { LitElement, html, css, nothing } from '../vendor/lit-core.min.js';
import { shared } from '../styles.js';
import { api, copy, MODES, CAT_RU, sexVar } from '../lib/api.js';
import { icons } from '../lib/icons.js';
import { eventCSS, eventRow } from '../lib/events.js';
import './ui.js';

class FsPassport extends LitElement {
  static properties = { runId: {}, name: {}, p: { state: true }, err: { state: true } };
  static styles = [shared, eventCSS, css`
    h4 { margin: var(--sp-5) 0 var(--sp-2); font-size: var(--fs-s); color: var(--text-2); text-transform: uppercase; letter-spacing: .05em; }
    ul { margin: 0; padding-left: 18px; font-size: var(--fs-s); color: var(--text-2); }
    .evlist { max-height: 320px; border: 1px solid var(--border); border-radius: var(--r-s); }
  `];
  connectedCallback() {
    super.connectedCallback();
    api(`/api/run/${encodeURIComponent(this.runId)}/fly/${encodeURIComponent(this.name)}`)
      .then(p => { this.p = p; }, () => { this.err = 'нет связи с сервером'; });
  }
  _close() { this.remove(); }
  render() {
    const p = this.p;
    return html`<fs-dialog open wide .heading=${`Паспорт мухи ${this.name}`} @close=${this._close}>
      ${p ? html`<button slot="actions" class="btn sm" style="margin-left:auto" @click=${() => copy(JSON.stringify(p, null, 1))}>
        ${icons.copy}Копировать</button>` : nothing}
      ${this.err ? html`<div class="err">${this.err}</div>` : !p ? html`<div class="muted">Загрузка…</div>` : this._body(p)}
    </fs-dialog>`;
  }
  _body(p) {
    const cov = p.connectome || {};
    const rows = [['ID', p.uid], ['Имя', p.name], ['Пол', p.sex === 'male' ? 'самец' : p.sex === 'female' ? 'самка' : '—'],
      ['Статус', p.status], ['Происхождение', p.origin], ['Родители', p.parents ? `${p.parents.mother} × ${p.parents.father}` : '—'],
      ['Родилась (мир, ч)', p.born_world_h], ['Возраст, дн', p.age_d], ['Физиологический возраст, дн', p.bio_age_d],
      ['Нервная система', cov.dataset],
      ['Нейронов / связей', cov.neurons ? `${cov.neurons.toLocaleString('ru')} / ${cov.connections.toLocaleString('ru')}` : '—'],
      ['Режим механизмов', MODES[p.mode] || p.mode], ['Обходы', (p.shortcuts || []).join('; ') || 'нет'], ['Seed', p.seed],
      ['Обучение', p.plasticity === undefined ? '—' : p.plasticity ? 'вкл' : 'выкл'], ['Спарена', p.mated ? `да (${p.mate})` : 'нет'],
      ['Голод / энергия / здоровье', p.hunger !== undefined ? `${p.hunger} / ${p.energy} / ${p.health}` : '—'],
      ['Индивидуальная выносливость', p.tolerances ? Object.entries(p.tolerances).map(([k, v]) => `${k} ${v}`).join(', ') : '—'],
      ['События', Object.entries(p.event_counts || {}).map(([k, v]) => `${CAT_RU[k] || k}: ${v}`).join(', ') || '—']];
    return html`
      ${p.status === 'архив' ? html`<p class="muted small" style="margin-bottom:12px">Прогон сохранён на диске: доступны только события жизни.</p>` : nothing}
      <table class="kv">${rows.map(([k, v]) => html`<tr><th>${k}</th><td style=${k === 'Имя' ? `color:${sexVar(p.sex)};font-weight:600` : ''}>${v ?? '—'}</td></tr>`)}</table>
      ${cov.groups ? html`<h4>Состав нервной системы</h4><table class="kv">${Object.entries(cov.groups).map(([k, v]) =>
        html`<tr><th>${k}</th><td class="num">${typeof v === 'number' ? v.toLocaleString('ru') : v}</td></tr>`)}</table>` : nothing}
      ${cov.missing ? html`<h4>Чего нет в нервной системе</h4><ul>${cov.missing.map(x => html`<li>${x}</li>`)}</ul>` : nothing}
      <h4>Жизнь (важные события)</h4>
      <div class="evlist">${(p.life || []).slice().reverse().map(e => eventRow(e))}
        ${(p.life || []).length ? nothing : html`<div class="ev-empty">нет</div>`}</div>`;
  }
}
customElements.define('fs-passport', FsPassport);

export function openPassport(runId, name) {
  const el = document.createElement('fs-passport');
  el.runId = runId; el.name = name;
  document.body.appendChild(el);
  return el;
}
