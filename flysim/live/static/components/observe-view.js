// Observation: live (polling) or replay of any frame. Arena, environment, apples, events, one card per fly.
import { LitElement, html, css, nothing } from '../vendor/lit-core.min.js';
import { shared } from '../styles.js';
import { api, post, go, toast, SCHEMA, fmtClock, sexVar } from '../lib/api.js';
import { icons } from '../lib/icons.js';
import { eventCSS, eventRow } from '../lib/events.js';
import { newApple } from './apple-editor.js';
import { openPassport } from './passport-dialog.js';
import { statusChip } from './home-view.js';
import { confirmDialog } from './ui.js';
import './arena.js';
import './env-strip.js';
import './fly-card.js';

const APPLE_MAX = { sugar: 13, ethanol: 6, acetic: 6, yeast: 1, aab: 1, mold: 1, patulin: 1 };
const TRAIL = 400;

class FsObserve extends LitElement {
  static properties = {
    runId: {}, frame: { type: Number },
    state: { state: true }, cur: { state: true }, events: { state: true }, tick: { state: true },
    nFrames: { state: true }, idx: { state: true }, playing: { state: true }, paused: { state: true },
    addApple: { state: true }, offline: { state: true },
  };
  static styles = [shared, eventCSS, css`
    .layout { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr); gap: var(--sp-3); align-items: start; }
    @media (max-width: 980px) { .layout { grid-template-columns: minmax(0, 1fr); } }
    .title-row { display: flex; gap: var(--sp-3); align-items: center; flex-wrap: wrap; }
    .readouts { display: flex; gap: var(--sp-1); flex-wrap: wrap; }
    .toolbar { display: flex; gap: var(--sp-2); flex-wrap: wrap; align-items: center; margin-bottom: var(--sp-3); }
    .toolbar .sep { width: 1px; height: 20px; background: var(--border); margin: 0 var(--sp-1); }
    fs-env-strip { margin-bottom: var(--sp-3); }
    .replay { display: flex; gap: var(--sp-2); align-items: center; flex-wrap: wrap; padding: var(--sp-3) var(--sp-4); margin-bottom: var(--sp-3); }
    .replay input[type=range] { flex: 1; min-width: 180px; accent-color: var(--accent); }
    .replay .pos { font-size: var(--fs-xs); color: var(--text-2); font-variant-numeric: tabular-nums; min-width: 96px; text-align: right; }
    .arena-card { overflow: hidden; }
    .card-f .legend { row-gap: var(--sp-2); }
    .apple + .apple { border-top: 1px solid var(--border); padding-top: var(--sp-3); margin-top: var(--sp-3); }
    .apple .ah { display: flex; gap: var(--sp-2); align-items: baseline; margin-bottom: var(--sp-2); }
    .evlist { max-height: 300px; }
    .brains { display: grid; gap: var(--sp-3); }
    .regions .card-b { padding-top: var(--sp-2); padding-bottom: var(--sp-2); }
    .err { margin-bottom: var(--sp-3); }
  `];

  constructor() {
    super();
    this.state = null; this.cur = null; this.events = []; this.trails = {}; this.tick = 0;
    this.next = 0; this.idx = 0; this.nFrames = 0; this.playing = false; this.paused = false;
    this._cards = new Map();
    this._alive = false;
  }
  get replay() { return this.frame !== undefined && this.frame !== null && !Number.isNaN(this.frame); }

  connectedCallback() {
    super.connectedCallback();
    this._alive = true;
    this._init();
    this._onPassport = e => openPassport(this.runId, e.detail.name);
    this.addEventListener('open-passport', this._onPassport);
  }
  disconnectedCallback() {
    super.disconnectedCallback();
    this._alive = false; this.playing = false;
    clearTimeout(this._poll); clearTimeout(this._play);
    this.removeEventListener('open-passport', this._onPassport);
  }

  async _init() {
    let st = { kind: 'saved', title: this.runId, running: false, status: 'архив' };
    try {
      // only in-memory observations have /state; saved runs are replayed from disk
      const list = await api('/api/sims');
      if ((list.sims || []).some(s => s.run_id === this.runId)) {
        const s = await api(`/api/sim/${encodeURIComponent(this.runId)}/state`);
        if (s && s.title) st = s;
      }
    } catch (e) { /* offline: treat as saved */ }
    this.state = st;
    if (typeof st.paused === 'boolean') this.paused = st.paused;
    if (this.replay) return this._initReplay();
    if (st.kind === 'saved') { go(`#/observe/${this.runId}/0`); return; }
    this._pollLoop();
  }

  // ---------- live ----------
  async _pollLoop() {
    if (!this._alive) return;
    try {
      const now = Date.now();
      if (!this._stAt || now - this._stAt > 1500) {
        this._stAt = now;
        const st = await api(`/api/sim/${encodeURIComponent(this.runId)}/state`);
        if (st.error && !st.title) { go(`#/observe/${this.runId}/0`); return; }
        this.state = st;
        if (typeof st.paused === 'boolean') this.paused = st.paused;
      }
      const r = await api(`/api/sim/${encodeURIComponent(this.runId)}/frames?since=${this.next}`);
      if (r.frames && r.frames.length) {
        for (const fr of r.frames.slice(0, -1)) this._trail(fr);
        const last = r.frames[r.frames.length - 1];
        this.next = last.i + 1;
        this._show(last, last.events);
      }
      this.offline = false;
    } catch (e) { this.offline = true; }
    if (this._alive) this._poll = setTimeout(() => this._pollLoop(), 400);
  }

  _trail(fr) {
    for (const [n, f] of Object.entries(fr.flies || {})) {
      const t = this.trails[n] || (this.trails[n] = []);
      t.push(f.pos); if (t.length > TRAIL) t.shift();
    }
  }

  _show(fr, events) {
    this._trail(fr);
    for (const n of Object.keys(this.trails)) if (!fr.flies?.[n]) delete this.trails[n];
    this.cur = fr;
    if (events) this.events = events;
    this.tick++;
  }

  // ---------- replay ----------
  async _initReplay() {
    const rid = encodeURIComponent(this.runId);
    const info = await api(`/api/run/${rid}/info`);
    if (info.error) { this.state = { ...this.state, error: info.error, status: info.error }; return; }
    this.nFrames = info.n_frames || 0;
    this._allEvents = (await api(`/api/run/${rid}/events?min_imp=2`)).events || [];
    this._load(this.frame || 0);
  }
  async _load(i) {
    if (!this.nFrames) return;
    i = Math.max(0, Math.min(this.nFrames - 1, Math.round(i)));
    if (i !== this.idx + 1) this.trails = {};          // jumped: start the trails afresh
    this.idx = i;
    const want = i;
    const fr = await api(`/api/run/${encodeURIComponent(this.runId)}/frame?i=${i}`);
    if (want !== this.idx || !this._alive) return;
    if (fr && fr.flies) this._show(fr, this._allEvents.filter(e => e.frame <= i));
  }
  _togglePlay() {
    this.playing = !this.playing;
    const step = async () => {
      if (!this.playing || !this._alive) return;
      if (this.idx >= this.nFrames - 1) { this.playing = false; return; }
      await this._load(this.idx + 1);
      this._play = setTimeout(step, 100);
    };
    step();
  }

  // ---------- controls ----------
  async _addFly(sex) {
    await post(`/api/sim/${this.runId}/add_fly`, { sex });
    toast(sex === 'male' ? 'самец будет добавлен' : 'самка будет добавлена');
  }
  async _pause() { const r = await post(`/api/sim/${this.runId}/pause`); this.paused = !!r.paused; }
  async _stop() {
    const ok = await confirmDialog({ title: 'Остановить наблюдение?', ok: 'Остановить', danger: false,
      text: 'Мозг перестанет считаться, а мухи доживут свою жизнь: мир проматывается по суткам, мухи едят по правилам '
        + 'перемотки, стареют, голодают, когда кончится еда, и каждая умирает своей причиной (старость, голод, холод, '
        + 'аноксия, токсин). Новые мухи не появляются. Смерти записываются в хронику. Повторное нажатие «Стоп» '
        + 'обрывает доживание сразу.' });
    if (!ok) return;
    await post(`/api/sim/${this.runId}/stop`); toast('мухи доживают свою жизнь'); this._stAt = 0;
  }
  async _submitApple() {
    const a = this.addApple; this.addApple = null;
    await post(`/api/sim/${this.runId}/add_apple`, a);
    toast('яблоко будет положено');
  }

  _card(n, f) {
    let el = this._cards.get(n);
    if (!el) { el = document.createElement('fs-fly-card'); el.name = n; el.runId = this.runId; this._cards.set(n, el); }
    el.fly = f;
    return el;
  }

  _toolbar() {
    const st = this.state || {}, rid = this.runId;
    const live = st.running && !st.error && !this.replay;
    const logs = st.kind === 'free' ? html`<a class="btn ${live ? '' : 'primary'}" href="/api/sim/${rid}/logs/all.zip">${icons.download}Скачать логи</a>` : nothing;
    if (live && st.kind === 'free') {
      return html`<div class="toolbar">
        <button class="btn" @click=${() => this._addFly('male')}>${icons.plus}Самец</button>
        <button class="btn" @click=${() => this._addFly('female')}>${icons.plus}Самка</button>
        <button class="btn" @click=${() => { this.addApple = { ...newApple(), x_mm: 0, y_mm: 0 }; }}>${icons.plus}Яблоко</button>
        <span class="sep"></span>
        <button class="btn" @click=${this._pause}>${this.paused ? html`${icons.play}Продолжить` : html`${icons.pause}Пауза`}</button>
        <button class="btn" @click=${this._stop}>${icons.stop}Стоп</button>
        <span class="spacer"></span>
        <a class="btn ghost" href="#/observe/${rid}/0">Повтор с начала</a>${logs}</div>`;
    }
    if (logs !== nothing || (!this.replay && st.kind !== 'saved')) {
      return html`<div class="toolbar">${!this.replay ? html`<a class="btn" href="#/observe/${rid}/0">Повтор с начала</a>` : nothing}${logs}</div>`;
    }
    return nothing;
  }

  _replayBar() {
    if (!this.replay) return nothing;
    const n = this.nFrames;
    return html`<div class="card replay" role="group" aria-label="Повтор">
      <button class="btn sm icon" title="Назад на 10 кадров" aria-label="Назад на 10 кадров" @click=${() => this._load(this.idx - 10)}>${icons.back10}</button>
      <button class="btn sm icon" title="Кадр назад" aria-label="Кадр назад" @click=${() => this._load(this.idx - 1)}>${icons.back}</button>
      <button class="btn sm icon ${this.playing ? 'on' : ''}" title=${this.playing ? 'Пауза' : 'Воспроизвести'}
        aria-label=${this.playing ? 'Пауза' : 'Воспроизвести'} @click=${this._togglePlay}>${this.playing ? icons.pause : icons.play}</button>
      <button class="btn sm icon" title="Кадр вперёд" aria-label="Кадр вперёд" @click=${() => this._load(this.idx + 1)}>${icons.fwd}</button>
      <button class="btn sm icon" title="Вперёд на 10 кадров" aria-label="Вперёд на 10 кадров" @click=${() => this._load(this.idx + 10)}>${icons.fwd10}</button>
      <input type="range" min="0" max=${Math.max(0, n - 1)} .value=${String(this.idx)} aria-label="Кадр"
        @input=${e => this._load(+e.target.value)}>
      <span class="pos">${n ? `кадр ${this.idx} / ${n - 1}` : 'загрузка…'}</span>
      ${this.state?.running ? html`<a class="btn sm primary" href="#/observe/${this.runId}">к живому</a>` : nothing}
    </div>`;
  }

  _apples(fr) {
    const N = SCHEMA.apple_norms || {};
    const list = fr.apples || (fr.apple ? [fr.apple] : []);
    if (!list.length) return html`<div class="muted small">яблок нет</div>`;
    return list.map((ap, i) => html`<div class="apple">
      <div class="ah"><b>Яблоко ${i + 1}</b><span class="muted small">${ap.cut ? 'надрезанное' : 'целое'}</span></div>
      <div class="stats">${Object.entries(N).map(([k, [lab, unit, norm]]) => {
        const v = +((ap.state || {})[k] || 0);
        return html`<div class="stat">${lab}<b>${v.toFixed(2)} <span class="faint" style="font-weight:400;font-size:11px">${unit}</span></b>
          <div class="meter"><i style="width:${Math.min(100, 100 * v / (APPLE_MAX[k] || 1))}%"></i></div><div class="norm">${norm}</div></div>`;
      })}</div></div>`);
  }

  _legend(fr) {
    const b = fr.brood || {}, pl = fr.plants;
    return html`<div class="legend">
      ${Object.entries(fr.flies || {}).map(([n, f]) => html`<span><span class="dot" style="background:${sexVar(f.sex)}"></span>${n}</span>`)}
      <span class="faint">яиц ${b.egg ?? 0} · личинок ${b.larva ?? 0} · куколок ${b.pupa ?? 0}${pl ? ` · сеянцев ${pl.seedlings} (${pl.tallest_cm} см)` : ''}</span>
      <span class="faint">мир ${(2 * (fr.arena_mm || 55) / 1000).toFixed(2)} м</span></div>`;
  }

  render() {
    const st = this.state || {}, fr = this.cur, rid = this.runId;
    const chip = this.replay ? html`<span class="chip info">повтор</span>`
      : statusChip({ running: st.running, error: st.error, paused: this.paused });
    const flies = fr ? Object.entries(fr.flies || {}) : [];
    for (const n of [...this._cards.keys()]) if (!fr?.flies?.[n]) this._cards.delete(n);
    return html`
      <div class="page-head"><div>
          <div class="title-row"><h1>${st.title || 'Наблюдение'}</h1>${this.state ? chip : nothing}</div>
          <p class="sub">${this.offline ? 'нет связи с сервером' : st.status || ''}</p></div>
        <div class="readouts">
          <span class="pill">мир <b>${fr ? fmtClock(fr.clock_h) : '—'}</b></span>
          <span class="pill">сессия <b>${fr ? `${(+fr.t || 0).toFixed(1)} с` : '—'}</b></span>
          <span class="pill">кадр <b>${fr ? fr.i : '—'}</b></span></div></div>
      ${typeof st.error === 'string' ? html`<div class="err">${st.error}</div>` : nothing}
      ${this._toolbar()}
      ${this._replayBar()}
      <fs-env-strip .frame=${fr}></fs-env-strip>
      <div class="layout">
        <div class="stack">
          <section class="card arena-card">
            <div class="card-h"><h3>Арена</h3><span class="hint">${fr?.env?.weather ? 'сад' : 'контейнер'}</span></div>
            <fs-arena .frame=${fr} .trails=${this.trails} .tick=${this.tick}></fs-arena>
            ${fr ? html`<div class="card-f">${this._legend(fr)}</div>` : nothing}
          </section>
          <section class="card"><div class="card-h"><h3>Яблоки</h3><span class="hint">значение · норма</span></div>
            <div class="card-b">${fr ? this._apples(fr) : html`<div class="muted small">ожидание первого кадра…</div>`}</div></section>
          <section class="card"><div class="card-h"><h3>Последние события</h3>
              <a class="btn sm" href="#/chronicle/${rid}">Вся хроника ${icons.fwd}</a></div>
            <div class="evlist">${this.events.length
              ? this.events.slice().reverse().slice(0, 60).map(e => eventRow(e, { onClick: () => go(`#/chronicle/${rid}`) }))
              : html`<div class="ev-empty">событий пока нет</div>`}</div></section>
        </div>
        <div class="stack">
          <section class="card regions"><div class="card-h"><h3>Мозг</h3><span class="hint">цвет точки — область</span></div>
            <div class="card-b"><div class="legend">${Object.entries(SCHEMA.region_names).map(([k, v]) =>
              html`<span><span class="dot" style="background:${SCHEMA.region_colors[k]}"></span>${v}</span>`)}</div></div></section>
          ${fr && !flies.length ? html`<div class="empty">В мире нет мух с мозгом.</div>` : nothing}
          ${!fr ? html`<div class="empty">${st.running ? 'Загрузка мозгов и первого кадра… это может занять около минуты на муху.' : 'Загрузка…'}</div>` : nothing}
          <div class="brains">${flies.map(([n, f]) => this._card(n, f))}</div>
        </div>
      </div>
      ${this.addApple ? html`<fs-dialog open heading="Положить яблоко" @close=${() => { this.addApple = null; }}>
        <fs-apple-editor single .apples=${[this.addApple]} .half=${fr?.arena_mm || 150}
          @value-changed=${e => { this.addApple = e.detail.value[0]; }}></fs-apple-editor>
        <button slot="footer" class="btn" @click=${() => { this.addApple = null; }}>Отмена</button>
        <button slot="footer" class="btn primary" @click=${this._submitApple}>Положить</button>
      </fs-dialog>` : nothing}`;
  }
}
customElements.define('fs-observe', FsObserve);
