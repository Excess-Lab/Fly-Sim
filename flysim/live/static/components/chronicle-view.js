// Chronicle "god panel": event timeline, filters, event details with the brain at that moment,
// region dominance over time per fly, passports.
import { LitElement, html, css, nothing } from '../vendor/lit-core.min.js';
import { shared } from '../styles.js';
import { api, copy, SCHEMA, CAT_COLORS, CAT_RU, fmtClock, sexVar, brainSummary, regionShares } from '../lib/api.js';
import { drawTimeline, drawRegions } from '../lib/draw.js';
import { icons } from '../lib/icons.js';
import { eventCSS, eventRow, catDot } from '../lib/events.js';
import { openPassport } from './passport-dialog.js';
import { regionBar } from './fly-card.js';
import './ui.js';

const PAGE = 400;

class FsChronicle extends LitElement {
  static properties = {
    runId: {}, passport: {},
    info: { state: true }, events: { state: true }, err: { state: true },
    imp: { state: true }, cat: { state: true }, fly: { state: true },
    openId: { state: true }, openFrame: { state: true }, moment: { state: true },
    rFly: { state: true }, noSens: { state: true }, limit: { state: true },
  };
  static styles = [shared, eventCSS, css`
    .cats { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: var(--sp-2); margin-bottom: var(--sp-3); }
    .cat { all: unset; box-sizing: border-box; cursor: pointer; background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--r-s); padding: var(--sp-2) var(--sp-3); display: grid; gap: 2px; }
    .cat:hover { border-color: var(--border-strong); }
    .cat:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
    .cat[aria-pressed="true"] { border-color: var(--text); box-shadow: inset 0 0 0 1px var(--text); }
    .cat .n { font-size: var(--fs-l); font-weight: 600; font-variant-numeric: tabular-nums; }
    .cat .l { font-size: var(--fs-xs); color: var(--text-2); display: flex; align-items: center; gap: 6px; }
    .cat.zero { opacity: .55; }
    @media (max-width: 560px) {
      .cats { display: flex; overflow-x: auto; scrollbar-width: thin; padding-bottom: 4px; }
      .cat { flex: none; width: 128px; }
    }
    #timeline { height: 132px; cursor: crosshair; }
    #regions { height: 190px; }
    .layout { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: var(--sp-3); align-items: start; margin-top: var(--sp-3); }
    @media (max-width: 980px) { .layout { grid-template-columns: minmax(0, 1fr); } }
    .filters { display: flex; gap: var(--sp-2); flex-wrap: wrap; align-items: center; }
    .evlist { max-height: 640px; }
    .detail { margin-top: var(--sp-2); background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-s);
      padding: var(--sp-3); cursor: default; }
    .detail pre { white-space: pre-wrap; margin: var(--sp-2) 0 0; max-height: 280px; overflow: auto; font: 11.5px/1.5 var(--mono);
      color: var(--text-2); }
    .more { padding: var(--sp-3); text-align: center; }
    .fly-m + .fly-m { border-top: 1px solid var(--border); margin-top: var(--sp-3); padding-top: var(--sp-3); }
    .fly-m .hdr { display: flex; gap: var(--sp-2); align-items: baseline; flex-wrap: wrap; margin-bottom: var(--sp-2); }
    .flow { display: grid; gap: 1px; font-size: 10px; margin-top: var(--sp-2); }
    .flow div { padding: 2px; text-align: center; border-radius: 2px; overflow: hidden; white-space: nowrap; min-height: 16px; }
    .flow .lab { background: var(--surface-2); color: var(--text-2); }
    .row-ctl { display: flex; gap: var(--sp-2); align-items: center; flex-wrap: wrap; }
    .chk { display: inline-flex; gap: 6px; align-items: center; font-size: var(--fs-s); color: var(--text-2); cursor: pointer; }
    .chk input { accent-color: var(--accent); }
    .pp { display: flex; flex-wrap: wrap; gap: var(--sp-2); }
    .pp .btn { border-left-width: 3px; }
  `];

  constructor() {
    super();
    this.imp = '2'; this.cat = ''; this.fly = ''; this.events = []; this.noSens = true; this.limit = PAGE;
    this._onTheme = () => this._redraw();
    this._onResize = () => this._redraw();
  }
  connectedCallback() {
    super.connectedCallback();
    this._alive = true;
    addEventListener('fs-theme', this._onTheme); addEventListener('resize', this._onResize);
    this._loadInfo().then(() => { if (this.passport) openPassport(this.runId, this.passport); });
  }
  disconnectedCallback() {
    super.disconnectedCallback(); this._alive = false; clearTimeout(this._t);
    removeEventListener('fs-theme', this._onTheme); removeEventListener('resize', this._onResize);
  }

  get _rid() { return encodeURIComponent(this.runId); }

  async _loadInfo() {
    let info;
    try { info = await api(`/api/run/${this._rid}/info`); } catch (e) { this.err = 'нет связи с сервером'; return; }
    if (info.error) { this.err = info.error; return; }
    this.info = info;
    const flies = Object.keys(info.flies || {});
    if (!this.rFly || !flies.includes(this.rFly)) this.rFly = flies[0] || '';
    await this._loadEvents();
    this._series = null;
    this._loadSeries();
    clearTimeout(this._t);
    if (info.live && this._alive) this._t = setTimeout(() => { if (this.openId == null) this._loadInfo(); else this._retry(); }, 20000);
  }
  _retry() { clearTimeout(this._t); this._t = setTimeout(() => { if (this.openId == null) this._loadInfo(); else this._retry(); }, 20000); }

  async _loadEvents() {
    const p = new URLSearchParams({ min_imp: this.imp, cat: this.cat, fly: this.fly });
    const r = await api(`/api/run/${this._rid}/events?${p}`);
    this.events = r.events || [];
    this.limit = PAGE;
  }
  async _loadSeries() {
    const r = await api(`/api/run/${this._rid}/series`).catch(() => null);
    this._series = r?.series || [];
    this.requestUpdate();
  }

  _setFilter(patch) { Object.assign(this, patch); this.openId = null; this.moment = null; this._loadEvents(); }

  updated() { this._redraw(); }
  _redraw() {
    const tl = this.renderRoot.querySelector('#timeline');
    if (tl && this.info) this._pick = drawTimeline(tl, this.events, this.info.clock_h || [0, 0]);
    const rc = this.renderRoot.querySelector('#regions');
    if (rc && this._series && this.rFly) drawRegions(rc, this._series, this.rFly, this.events, this.noSens);
  }

  _timelineClick(ev) {
    if (!this._pick) return;
    const r = ev.currentTarget.getBoundingClientRect();
    const e = this._pick(ev.clientX - r.left);
    if (!e) return;
    const pos = this.events.length - 1 - this.events.indexOf(e);
    if (pos >= this.limit) this.limit = pos + 50;
    this._open(e, true);
  }

  async _open(e, scroll = false) {
    if (this.openId === e.id) { this.openId = null; return; }
    this.openId = e.id; this.openFrame = null;
    if (scroll) this.updateComplete.then(() => this.renderRoot.querySelector(`.ev[data-id="${e.id}"]`)?.scrollIntoView({ block: 'center', behavior: 'smooth' }));
    const fr = await api(`/api/run/${this._rid}/frame?i=${e.frame}`).catch(() => null);
    if (this.openId !== e.id) return;
    this.openFrame = fr;
    this.moment = { e, fr };
  }

  _brainText(fr) {
    return fr && fr.flies ? Object.entries(fr.flies).map(([n, f]) => brainSummary(n, f)).join('\n') : '(кадр недоступен)';
  }

  _detail(e) {
    const fr = this.openFrame, bt = this._brainText(fr);
    const text = `${e.fly}: ${e.text}\n${fmtClock(e.clock_h)}, кадр ${e.frame}\n\nМозг в этот момент:\n${bt}\n\n${JSON.stringify(e, null, 1)}`;
    return html`<div class="detail" @click=${ev => ev.stopPropagation()} @keydown=${ev => ev.stopPropagation()}>
      <div class="row">
        <button class="btn sm" @click=${() => copy(text)}>${icons.copy}Копировать</button>
        <a class="btn sm primary" href="#/observe/${this.runId}/${e.frame}">Перейти к моменту ${icons.fwd}</a>
        <button class="btn sm ghost" @click=${() => { this.openId = null; }}>Свернуть</button></div>
      <pre>${e.text}\n\nМозг в этот момент:\n${fr === null ? 'загрузка…' : bt}\n\n${JSON.stringify(e, null, 1)}</pre></div>`;
  }

  _moment() {
    if (!this.moment) return html`<p class="muted small">Выберите событие — здесь появится, что было в мозге каждой мухи:
      какие области преобладали и куда шёл поток сигнала между ними.</p>`;
    const { fr } = this.moment;
    if (!fr || !fr.flies) return html`<p class="muted small">кадр недоступен</p>`;
    const names = SCHEMA.region_names, keys = Object.keys(names);
    return Object.entries(fr.flies).map(([n, f]) => {
      const fl = f.flow || [];
      let mx = 1; fl.forEach(r => r.forEach(v => { mx = Math.max(mx, v); }));
      const top = f.regions ? regionShares(f).sort((a, b) => b.pct - a.pct).slice(0, 3) : [];
      return html`<div class="fly-m">
        <div class="hdr"><b style="color:${sexVar(f.sex)}">${n}</b>
          <span class="muted small">${f.mode}/${f.surface} · PPL1 ${f.rates?.ppl1 ?? '—'} Гц · PAM ${f.rates?.pam ?? '—'} Гц</span></div>
        ${regionBar(f)}
        <div class="legend" style="margin-top:6px">${top.map(r => html`<span><span class="dot" style="background:${r.color}"></span>${r.name} ${Math.round(r.pct)}%</span>`)}</div>
        ${fl.length ? html`<div class="faint xs" style="margin-top:8px">поток сигнала между областями (строка → столбец): чем насыщеннее, тем сильнее</div>
        <div class="flow" style="grid-template-columns:repeat(${keys.length + 1}, minmax(0, 1fr))"><div></div>
          ${keys.map(k => html`<div class="lab" title=${names[k]}>${names[k].slice(0, 5)}</div>`)}
          ${fl.map((row, a) => html`<div class="lab" title=${names[a] || ''}>${(names[a] || '').slice(0, 5)}</div>
            ${row.map((v, b) => html`<div title="${names[a]} → ${names[b]}: ${v}"
              style="background:color-mix(in srgb, var(--accent) ${Math.round(100 * v / mx)}%, transparent)"></div>`)}`)}</div>` : nothing}
      </div>`;
    });
  }

  render() {
    if (this.err) return html`<div class="err">${this.err}</div>`;
    const info = this.info;
    if (!info) return html`<div class="muted">Загрузка хроники…</div>`;
    const flies = Object.entries(info.flies || {});
    const shown = this.events.slice().reverse().slice(0, this.limit);
    return html`
      <div class="page-head"><div>
          <div class="row"><h1>Хроника — панель бога</h1>
            ${info.live ? html`<span class="chip running">идёт</span>` : html`<span class="chip finished">завершено</span>`}</div>
          <p class="sub">${info.live ? 'Хроника обновляется каждые 20 с' : 'Прогон завершён'} · кадров ${info.n_frames} ·
            мир ${fmtClock(info.clock_h[0])} → ${fmtClock(info.clock_h[1])}</p></div>
        <div class="row">
          <button class="btn" @click=${() => copy(this.events.map(e => `[${fmtClock(e.clock_h)} | кадр ${e.frame}] ${'!'.repeat(e.imp)} ${e.fly}: ${e.text}`).join('\n'))}>
            ${icons.copy}Копировать видимые</button>
          <a class="btn" href="#/observe/${this.runId}${info.live ? '' : '/0'}">К наблюдению</a></div></div>

      <div class="cats" role="group" aria-label="Категории событий">
        ${Object.entries(CAT_RU).map(([k, v]) => html`<button class="cat ${info.counts[k] ? '' : 'zero'}" aria-pressed=${this.cat === k ? 'true' : 'false'}
          @click=${() => this._setFilter(this.cat === k ? { cat: '' } : { cat: k, imp: '1' })}>
          <span class="n" style="color:${CAT_COLORS[k]}">${info.counts[k] || 0}</span><span class="l">${catDot(k)}${v}</span></button>`)}
      </div>

      <section class="card"><div class="card-h"><h3>Временная шкала</h3>
          <span class="hint">точка — событие · цвет — категория · размер — важность · клик — открыть</span></div>
        <div class="card-b"><canvas id="timeline" @click=${this._timelineClick} aria-label="Временная шкала событий"></canvas></div></section>

      <div class="layout">
        <section class="card"><div class="card-h"><h3>События <span class="faint small num">${this.events.length}</span></h3>
            <div class="filters">
              <fs-segmented size="sm" .options=${[['3', 'главные'], ['2', 'важные'], ['1', 'все']]} .value=${this.imp}
                @value-changed=${e => this._setFilter({ imp: e.detail.value })}></fs-segmented>
              <select class="select" aria-label="Категория" @change=${e => this._setFilter({ cat: e.target.value })}>
                <option value="" ?selected=${!this.cat}>все категории</option>
                ${Object.entries(CAT_RU).map(([k, v]) => html`<option value=${k} ?selected=${this.cat === k}>${v}</option>`)}</select>
              <select class="select" aria-label="Муха" @change=${e => this._setFilter({ fly: e.target.value })}>
                <option value="" ?selected=${!this.fly}>все мухи</option>
                ${flies.map(([n]) => html`<option value=${n} ?selected=${this.fly === n}>${n}</option>`)}
                <option value="мир" ?selected=${this.fly === 'мир'}>мир</option></select>
            </div></div>
          <div class="evlist">
            ${shown.map(e => eventRow(e, { open: this.openId === e.id, onClick: () => this._open(e),
              extra: this.openId === e.id ? this._detail(e) : nothing }))}
            ${this.events.length ? nothing : html`<div class="ev-empty">нет событий по фильтру</div>`}
            ${this.events.length > this.limit ? html`<div class="more"><button class="btn sm" @click=${() => { this.limit += PAGE; }}>
              Показать ещё (${this.events.length - this.limit})</button></div>` : nothing}
          </div></section>

        <div class="stack">
          <section class="card"><div class="card-h"><h3>Момент</h3>
              <span class="hint">${this.moment ? `${fmtClock(this.moment.e.clock_h)} · кадр ${this.moment.e.frame}` : 'выберите событие'}</span></div>
            <div class="card-b">${this._moment()}</div></section>
          <section class="card"><div class="card-h"><h3>Области мозга во времени</h3>
              <div class="row-ctl">
                <label class="chk"><input type="checkbox" .checked=${this.noSens} @change=${e => { this.noSens = e.target.checked; }}>без сенсорных</label>
                ${flies.length ? html`<select class="select" aria-label="Муха" @change=${e => { this.rFly = e.target.value; }}>
                  ${flies.map(([n]) => html`<option value=${n} ?selected=${this.rFly === n}>${n}</option>`)}</select>` : nothing}</div></div>
            <div class="card-b">${flies.length ? html`<canvas id="regions" aria-label="Преобладание областей мозга во времени"></canvas>`
              : html`<p class="muted small">в прогоне нет мух с событиями</p>`}
              ${this._series === null && flies.length ? html`<p class="faint xs">загрузка ряда…</p>` : nothing}
              <div class="legend" style="margin-top:8px">${Object.entries(SCHEMA.region_names).map(([k, v]) =>
                html`<span><span class="dot" style="background:${SCHEMA.region_colors[k]}"></span>${v}</span>`)}
                <span class="faint">| вертикальные линии — главные события</span></div></div></section>
          <section class="card"><div class="card-h"><h3>Паспорта мух</h3></div>
            <div class="card-b pp">${flies.length ? flies.map(([n, s]) => html`<button class="btn sm" style="border-left-color:${sexVar(s)}"
              @click=${() => openPassport(this.runId, n)}>${icons.id}${n}</button>`) : html`<span class="muted small">нет</span>`}</div></section>
        </div>
      </div>`;
  }
}
customElements.define('fs-chronicle', FsChronicle);
