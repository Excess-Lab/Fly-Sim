// Arena canvas: apples (with mould), flies and trails, zones, brood, night dimming and wind arrow.
import { LitElement, html, css } from '../vendor/lit-core.min.js';
import { shared } from '../styles.js';
import { drawArena, attachOrbit } from '../lib/draw.js';

class FsArena extends LitElement {
  static properties = { frame: { attribute: false }, trails: { attribute: false }, tick: { type: Number } };
  static styles = [shared, css`
    :host { display: block; position: relative; }
    canvas { aspect-ratio: 4 / 3; cursor: grab; touch-action: pan-y; }
    .tip { position: absolute; left: 10px; bottom: 8px; font-size: 10.5px; color: var(--text-3); pointer-events: none; }
    .reset { position: absolute; left: 8px; top: 8px; }
  `];
  constructor() {
    super();
    this.view = { az: -0.6, el: 0.85, zoom: 1 };
    this.trails = {};
    this._onTheme = () => this._draw();
    this._onResize = () => this._draw();
  }
  connectedCallback() { super.connectedCallback(); addEventListener('fs-theme', this._onTheme); addEventListener('resize', this._onResize); }
  disconnectedCallback() {
    super.disconnectedCallback(); removeEventListener('fs-theme', this._onTheme); removeEventListener('resize', this._onResize); this._detach?.();
  }
  firstUpdated() {
    this._detach = attachOrbit(this.renderRoot.querySelector('canvas'), (dx, dy, wh) => {
      this.view.az += dx * .01; this.view.el = Math.min(1.5, Math.max(.1, this.view.el + dy * .01));
      if (wh) this.view.zoom *= wh > 0 ? .9 : 1.1;
    }, () => this._draw());
  }
  updated() { this._draw(); }
  _draw() { const cv = this.renderRoot.querySelector('canvas'); if (cv) drawArena(cv, this.view, this.frame, this.trails || {}); }
  _reset() { this.view = { az: -0.6, el: 0.85, zoom: 1 }; this._draw(); }
  render() {
    return html`<canvas aria-label="Арена: вид в перспективе; перетаскивайте, чтобы вращать"></canvas>
      <button class="btn sm reset" @click=${this._reset} title="Вернуть вид">вид</button>
      <span class="tip">тяните — вращать · колесо — масштаб</span>`;
  }
}
customElements.define('fs-arena', FsArena);
