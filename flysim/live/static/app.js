// Fly-Sim web UI shell: top bar, theme, hash router.
//   #/ or #/home                 observations (running + saved)
//   #/new                        new free simulation / prepared experiment
//   #/observe/<id>               live observation        (alias #/run/<id>)
//   #/observe/<id>/<frame>       replay from a frame     (alias #/run/<id>/replay/<frame>)
//   #/chronicle/<id>             chronicle "god panel"   (alias #/run/<id>/chronicle)
//   #/passport/<id>/<fly>        chronicle with that fly's passport open
import { LitElement, html, css, nothing } from './vendor/lit-core.min.js';
import { shared, installTokens } from './styles.js';
import { loadSchema, setSchema, storageGet, storageSet } from './lib/api.js';
import { icons } from './lib/icons.js';
import './components/ui.js';
import './components/home-view.js';
import './components/new-view.js';
import './components/observe-view.js';
import './components/chronicle-view.js';

installTokens();

const THEMES = ['auto', 'light', 'dark'];
const THEME_RU = { auto: 'как в системе', light: 'светлая', dark: 'тёмная' };
function applyTheme(t) {
  if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
  window.dispatchEvent(new Event('fs-theme'));
}

export function parseRoute(hash) {
  const parts = (hash || '').replace(/^#\/?/, '').split('/').map(decodeURIComponent);
  let [page, id, a, b] = parts;
  if (page === 'run' && id) {
    if (a === 'chronicle') return { page: 'chronicle', id };
    if (a === 'replay') return { page: 'observe', id, frame: +b || 0 };
    return { page: 'observe', id };
  }
  if (page === 'observe' && id) return { page, id, frame: a === undefined || a === '' ? undefined : +a || 0 };
  if (page === 'chronicle' && id) return { page, id };
  if (page === 'passport' && id) return { page: 'chronicle', id, passport: a };
  if (page === 'new') return { page };
  return { page: 'home' };
}

class FsApp extends LitElement {
  static properties = { route: { state: true }, theme: { state: true }, error: { state: true }, ready: { state: true } };
  static styles = [shared, css`
    :host { display: block; min-height: 100vh; }
    header { position: sticky; top: 0; z-index: 10; background: color-mix(in srgb, var(--bg) 88%, transparent);
      backdrop-filter: saturate(1.4) blur(10px); -webkit-backdrop-filter: saturate(1.4) blur(10px); border-bottom: 1px solid var(--border); }
    .bar { max-width: 1440px; margin: 0 auto; display: flex; align-items: center; gap: var(--sp-2) var(--sp-4);
      padding: var(--sp-2) var(--sp-4); min-height: 52px; flex-wrap: wrap; }
    .brand { display: inline-flex; align-items: center; gap: var(--sp-2); font-weight: 700; font-size: var(--fs-m);
      text-decoration: none; letter-spacing: -0.01em; }
    .logo { width: 22px; height: 22px; border-radius: 6px; background: var(--accent); display: grid; place-items: center; color: var(--accent-ink); }
    .logo svg { width: 14px; height: 14px; }
    nav { display: flex; gap: 2px; flex-wrap: wrap; }
    nav a { text-decoration: none; padding: 6px var(--sp-3); border-radius: var(--r-s); color: var(--text-2); font-weight: 500;
      font-size: var(--fs-s); white-space: nowrap; }
    nav a:hover { color: var(--text); background: var(--surface-2); }
    nav a[aria-current="page"] { color: var(--text); background: var(--surface-3); }
    nav.run { padding-left: var(--sp-3); border-left: 1px solid var(--border); }
    .runid { color: var(--text-3); font-size: var(--fs-xs); font-family: var(--mono); padding: 6px 4px; }
    main { max-width: 1440px; margin: 0 auto; padding: var(--sp-5) var(--sp-4) var(--sp-7); }
    @media (max-width: 560px) { main { padding-top: var(--sp-4); } nav.run { border-left: 0; padding-left: 0; } .runid { display: none; } }
  `];

  constructor() {
    super();
    this.theme = storageGet('flysim-theme') || 'auto';
    applyTheme(this.theme);
    this.route = parseRoute(location.hash);
    this._view = null;
    this._viewKey = '';
    this._onHash = () => { this.route = parseRoute(location.hash); window.scrollTo(0, 0); };
  }
  connectedCallback() {
    super.connectedCallback();
    addEventListener('hashchange', this._onHash);
    this._boot();
  }
  disconnectedCallback() { super.disconnectedCallback(); removeEventListener('hashchange', this._onHash); }

  async _boot() {
    try { setSchema(await loadSchema()); this.ready = true; this.error = null; }
    catch (e) { this.error = true; setTimeout(() => this._boot(), 3000); }
  }

  _cycleTheme() {
    this.theme = THEMES[(THEMES.indexOf(this.theme) + 1) % THEMES.length];
    storageSet('flysim-theme', this.theme === 'auto' ? null : this.theme);
    applyTheme(this.theme);
  }

  /** One view element per route key, so a new route always starts with fresh state and timers. */
  _viewFor(r) {
    const key = JSON.stringify(r);
    if (key === this._viewKey && this._view) return this._view;
    const tag = { home: 'fs-home', new: 'fs-new', observe: 'fs-observe', chronicle: 'fs-chronicle' }[r.page] || 'fs-home';
    const el = document.createElement(tag);
    if (r.id) el.runId = r.id;
    if (r.frame !== undefined) el.frame = r.frame;
    if (r.passport) el.passport = r.passport;
    this._view = el; this._viewKey = key;
    return el;
  }

  render() {
    const r = this.route;
    const cur = p => r.page === p ? 'page' : nothing;
    const replay = r.page === 'observe' && r.frame !== undefined;
    return html`<header><div class="bar">
        <a class="brand" href="#/"><span class="logo">${icons.air}</span>Fly-Sim</a>
        <nav aria-label="Разделы">
          <a href="#/" aria-current=${cur('home')}>Наблюдения</a>
          <a href="#/new" aria-current=${cur('new')}>Новое</a>
        </nav>
        ${r.id ? html`<nav class="run" aria-label="Прогон">
          <a href="#/observe/${r.id}" aria-current=${r.page === 'observe' && !replay ? 'page' : nothing}>Наблюдение</a>
          <a href="#/observe/${r.id}/0" aria-current=${replay ? 'page' : nothing}>Повтор</a>
          <a href="#/chronicle/${r.id}" aria-current=${cur('chronicle')}>Хроника</a>
          <span class="runid">${r.id}</span></nav>` : nothing}
        <span class="spacer"></span>
        <button class="btn ghost icon" @click=${this._cycleTheme} title="Тема: ${THEME_RU[this.theme]}"
          aria-label="Тема: ${THEME_RU[this.theme]}">${this.theme === 'dark' ? icons.moon : this.theme === 'light' ? icons.sun : icons.auto}</button>
      </div></header>
      <main>${this.error ? html`<div class="err">Нет связи с сервером Fly-Sim. Запустите «docker compose up» или
          «python -m flysim.live» и обновите страницу (http://localhost:8800). Пробую снова…</div>`
        : this.ready ? this._viewFor(r) : html`<div class="muted">Загрузка…</div>`}</main>`;
  }
}
customElements.define('fs-app', FsApp);
