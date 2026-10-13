// Home: running / in-memory observations and saved runs on disk, with delete.
import { LitElement, html, css, nothing } from '../vendor/lit-core.min.js';
import { shared } from '../styles.js';
import { api, post, fmtClock, toast } from '../lib/api.js';
import { icons } from '../lib/icons.js';
import { confirmDialog } from './ui.js';

export function statusChip(s) {
  if (s.error) return html`<span class="chip error">ошибка</span>`;
  if (s.running && s.paused) return html`<span class="chip paused">пауза</span>`;
  if (s.running) return html`<span class="chip running">идёт</span>`;
  return html`<span class="chip finished">завершено</span>`;
}

class FsHome extends LitElement {
  static properties = { sims: { state: true }, saved: { state: true }, loading: { state: true }, offline: { state: true }, busy: { state: true }, data: { state: true } };
  static styles = [shared, css`
    .sim .card-b { display: grid; gap: var(--sp-3); }
    .sim h3 { font-size: var(--fs-m); overflow-wrap: anywhere; }
    .status { color: var(--text-2); font-size: var(--fs-s); }
    .pills { display: flex; gap: var(--sp-1); flex-wrap: wrap; }
    .actions { display: flex; gap: var(--sp-2); flex-wrap: wrap; align-items: center; }
    .list { padding: 0; }
    .item { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: var(--sp-2) var(--sp-4); align-items: center;
      padding: var(--sp-3) var(--sp-4); border-bottom: 1px solid var(--border); }
    .item:last-child { border-bottom: 0; }
    .item .t { font-weight: 500; overflow-wrap: anywhere; }
    .item .m { color: var(--text-3); font-size: var(--fs-xs); font-family: var(--mono); margin-top: 2px; }
    @media (max-width: 560px) { .item { grid-template-columns: 1fr; } }
  `];

  constructor() { super(); this.sims = []; this.saved = []; this.loading = true; this.busy = {}; }
  connectedCallback() { super.connectedCallback(); this._load(); }
  disconnectedCallback() { super.disconnectedCallback(); clearTimeout(this._t); }

  async _load() {
    clearTimeout(this._t);
    try {
      const d = await api('/api/sims');
      this.sims = d.sims || []; this.saved = d.saved || []; this.offline = false;
      this.data = await api('/api/status').catch(() => null);
    } catch (e) { this.offline = true; }
    this.loading = false;
    if (this.isConnected) this._t = setTimeout(() => this._load(), 5000);
  }

  async _stop(id) {
    this.busy = { ...this.busy, [id]: true };
    await post(`/api/sim/${id}/stop`).catch(() => null);
    toast('наблюдение останавливается');
    setTimeout(() => this._load(), 600);
  }

  async _delete(id, title) {
    const ok = await confirmDialog({ title: 'Удалить наблюдение?',
      text: `«${title}» (${id}) и все его сохранённые файлы — кадры, события, логи — будут удалены без возможности восстановления.` });
    if (!ok) return;
    this.busy = { ...this.busy, [id]: true };
    let r;
    try { r = await post(`/api/run/${encodeURIComponent(id)}/delete`); } catch (e) { r = { ok: false, error: 'нет связи с сервером' }; }
    this.busy = { ...this.busy, [id]: false };
    if (r && r.ok) toast('удалено');
    else toast(r?.error || 'не удалось удалить', 'error');
    this._load();
  }

  _simCard(s) {
    const id = s.run_id;
    return html`<article class="card sim">
      <div class="card-h"><h3>${s.title}</h3>${statusChip(s)}</div>
      <div class="card-b">
        <div class="status">${s.status}</div>
        <div class="pills">
          <span class="pill">${s.kind === 'experiment' ? 'эксперимент' : 'свободная симуляция'}</span>
          <span class="pill">мух <b>${s.flies}</b></span>
          <span class="pill">мир <b>${fmtClock(s.clock_h)}</b></span>
          <span class="pill">событий <b>${s.events}</b></span>
        </div>
        <div class="actions">
          <a class="btn primary sm" href="#/observe/${id}">Открыть</a>
          <a class="btn sm" href="#/chronicle/${id}">Хроника</a>
          <a class="btn sm ghost" href="#/observe/${id}" target="_blank" rel="noopener" title="Открыть в новой вкладке">${icons.external}Вкладка</a>
          <span class="spacer"></span>
          ${s.running
            ? html`<button class="btn sm" ?disabled=${this.busy[id]} @click=${() => this._stop(id)}>${icons.stop}Стоп</button>`
            : html`<button class="btn sm danger" ?disabled=${this.busy[id]} @click=${() => this._delete(id, s.title)}
                title="Удалить наблюдение">${icons.trash}Удалить</button>`}
        </div>
      </div></article>`;
  }

  _savedRow(r) {
    const id = r.run_id;
    return html`<div class="item">
      <div><div class="t">${r.title}</div><div class="m">${id} · ${r.size_mb} МБ</div></div>
      <div class="actions">
        <a class="btn sm" href="#/observe/${id}/0">Повтор</a>
        <a class="btn sm" href="#/chronicle/${id}">Хроника</a>
        <a class="btn sm ghost icon" href="#/observe/${id}/0" target="_blank" rel="noopener" title="Повтор в новой вкладке"
          aria-label="Повтор в новой вкладке">${icons.external}</a>
        <button class="btn sm ghost icon danger" ?disabled=${this.busy[id]} title="Удалить" aria-label="Удалить ${r.title}"
          @click=${() => this._delete(id, r.title)}>${icons.trash}</button>
      </div></div>`;
  }

  render() {
    return html`
      <div class="page-head"><div><h1>Наблюдения</h1>
        <p class="sub">Можно вести несколько наблюдений одновременно и открывать каждое в своей вкладке.</p></div>
        <a class="btn primary" href="#/new">${icons.plus}Новое наблюдение</a></div>
      ${this.offline ? html`<div class="err" style="margin-bottom:16px">Нет связи с сервером — список не обновляется.</div>` : nothing}
      ${this.data && !this.data.ready ? html`<div class="err" style="margin-bottom:16px">
        Данные коннектомов загружаются (первый запуск, ~1,2 ГБ): осталось файлов — ${this.data.missing.length}.
        Запуск наблюдений станет доступен после загрузки.
        ${this.data.log?.length ? html`<div class="small" style="font-family:var(--mono);margin-top:6px">${this.data.log.join(' · ')}</div>` : nothing}
      </div>` : nothing}
      <div class="section-title"><h2>Идут и недавние</h2><span class="count">${this.sims.length || ''}</span></div>
      ${this.loading ? html`<div class="muted small">Загрузка…</div>`
        : this.sims.length ? html`<div class="grid">${this.sims.map(s => this._simCard(s))}</div>`
        : html`<div class="empty">Сейчас ничего не идёт. <a href="#/new">Начните новое наблюдение</a>.</div>`}
      <div class="section-title"><h2>Сохранённые прогоны</h2><span class="count">${this.saved.length || ''}</span></div>
      ${this.saved.length ? html`<div class="card list">${this.saved.map(r => this._savedRow(r))}</div>`
        : this.loading ? nothing : html`<div class="empty">Сохранённых прогонов нет.</div>`}`;
  }
}
customElements.define('fs-home', FsHome);
