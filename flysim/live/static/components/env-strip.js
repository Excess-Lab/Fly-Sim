// Environment strip: time of day, day/night, weather, temperature, light, wind, O2/CO2.
import { LitElement, html, css, nothing } from '../vendor/lit-core.min.js';
import { shared } from '../styles.js';
import { fmtHM, fmtLux, fmtClock } from '../lib/api.js';
import { icons, weatherIcon } from '../lib/icons.js';

class FsEnvStrip extends LitElement {
  static properties = { frame: { attribute: false } };
  static styles = [shared, css`
    :host { display: block; }
    .strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); background: var(--surface);
      border: 1px solid var(--border); border-radius: var(--r); box-shadow: var(--shadow); overflow: hidden; }
    .it { display: flex; gap: var(--sp-3); align-items: center; padding: var(--sp-3) var(--sp-4); min-width: 0;
      border-right: 1px solid var(--border); border-bottom: 1px solid var(--border); margin: 0 -1px -1px 0; }
    .ic { width: 30px; height: 30px; border-radius: 8px; display: grid; place-items: center; flex: none; background: var(--surface-2); color: var(--text-2); }
    .ic svg { width: 17px; height: 17px; }
    .ic.day { background: rgba(250, 204, 21, .18); color: #b45309; }
    .ic.night { background: rgba(99, 102, 241, .16); color: #6366f1; }
    .k { font-size: var(--fs-xs); color: var(--text-3); white-space: nowrap; }
    .v { font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .s { font-size: var(--fs-xs); color: var(--text-2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .txt { min-width: 0; }
    .arrow { display: inline-block; transition: transform .3s; }
  `];

  _item(icon, k, v, s, cls = '') {
    return html`<div class="it"><span class="ic ${cls}">${icon}</span><div class="txt"><div class="k">${k}</div>
      <div class="v">${v}</div>${s ? html`<div class="s">${s}</div>` : nothing}</div></div>`;
  }

  render() {
    const fr = this.frame;
    if (!fr) return nothing;
    const env = fr.env || {}, w = env.weather;
    const wv = env.wind || [0, 0];
    const speed = w ? w.wind_ms : Math.hypot(wv[0] || 0, wv[1] || 0);
    const dir = w ? w.wind_dir_deg : Math.atan2(wv[1] || 0, wv[0] || 0) * 180 / Math.PI;
    const day = Math.floor((fr.clock_h || 0) / 24) + 1;
    const arrow = html`<span class="arrow" style="transform:rotate(${-dir}deg)">${icons.arrow}</span>`;
    return html`<div class="strip" role="group" aria-label="Условия среды">
      ${w ? this._item(w.day ? icons.sun : icons.moon, 'Время суток', fmtHM(w.hour),
            `${w.day ? 'день' : 'ночь'} · сутки ${day} · солнце ${Math.round(w.sun_deg)}°`, w.day ? 'day' : 'night')
          : this._item(icons.box, 'Мир', fmtClock(fr.clock_h), 'контейнер: свет постоянный')}
      ${w ? this._item(weatherIcon(w), 'Погода', w.label, `день года ${Math.round(w.doy)}`) : nothing}
      ${this._item(icons.thermo, 'Температура', `${env.temp_c ?? '—'} °C`, '')}
      ${w ? this._item(icons.sun, 'Освещённость', fmtLux(w.lux), '') : nothing}
      ${this._item(speed > 0.005 ? arrow : icons.wind, 'Ветер', speed > 0.005 ? `${(+speed).toFixed(2)} м/с` : 'штиль',
          speed > 0.005 ? `в сторону ${Math.round(((dir % 360) + 360) % 360)}°` : '')}
      ${this._item(icons.air, 'Воздух', `O₂ ${env.o2_pct ?? '—'}%`, `CO₂ ${env.co2_ppm ?? '—'} ppm`)}
    </div>`;
  }
}
customElements.define('fs-env-strip', FsEnvStrip);
