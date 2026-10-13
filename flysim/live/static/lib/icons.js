// Minimal line icons (24×24, stroke = currentColor).
import { svg, html } from '../vendor/lit-core.min.js';

const ic = body => html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"
  stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;

export const icons = {
  plus: ic(svg`<path d="M12 5v14M5 12h14"/>`),
  play: ic(svg`<path d="M7 5l12 7-12 7z"/>`),
  pause: ic(svg`<path d="M8 5v14M16 5v14"/>`),
  stop: ic(svg`<rect x="6" y="6" width="12" height="12" rx="1.5"/>`),
  trash: ic(svg`<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>`),
  external: ic(svg`<path d="M14 4h6v6M20 4l-9 9M18 14v6H4V6h6"/>`),
  copy: ic(svg`<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a1 1 0 0 1 1-1h10"/>`),
  download: ic(svg`<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>`),
  x: ic(svg`<path d="M6 6l12 12M18 6L6 18"/>`),
  back: ic(svg`<path d="M15 6l-6 6 6 6"/>`),
  fwd: ic(svg`<path d="M9 6l6 6-6 6"/>`),
  back10: ic(svg`<path d="M12 6l-6 6 6 6M19 6l-6 6 6 6"/>`),
  fwd10: ic(svg`<path d="M5 6l6 6-6 6M12 6l6 6-6 6"/>`),
  sun: ic(svg`<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>`),
  moon: ic(svg`<path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/>`),
  auto: ic(svg`<circle cx="12" cy="12" r="8"/><path d="M12 4v16" /><path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor" stroke="none"/>`),
  cloud: ic(svg`<path d="M7 18h10a4 4 0 0 0 .5-7.97A6 6 0 0 0 6 11a3.5 3.5 0 0 0 1 7z"/>`),
  rain: ic(svg`<path d="M7 15h10a4 4 0 0 0 .5-7.97A6 6 0 0 0 6 8a3.5 3.5 0 0 0 1 7zM9 18l-1 3M13 18l-1 3M17 18l-1 3"/>`),
  storm: ic(svg`<path d="M7 15h10a4 4 0 0 0 .5-7.97A6 6 0 0 0 6 8a3.5 3.5 0 0 0 1 7zM13 15l-2 4h3l-2 4"/>`),
  fog: ic(svg`<path d="M4 9h16M6 13h12M4 17h16"/>`),
  thermo: ic(svg`<path d="M14 14.8V5a2 2 0 0 0-4 0v9.8a4 4 0 1 0 4 0z"/>`),
  wind: ic(svg`<path d="M3 9h11a3 3 0 1 0-3-3M3 15h15a3 3 0 1 1-3 3"/>`),
  box: ic(svg`<path d="M4 8l8-4 8 4v8l-8 4-8-4zM4 8l8 4 8-4M12 12v8"/>`),
  air: ic(svg`<circle cx="8" cy="10" r="3"/><circle cx="16" cy="14" r="4"/>`),
  arrow: ic(svg`<path d="M4 12h15M14 7l5 5-5 5"/>`),
  id: ic(svg`<rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="12" r="2.5"/><path d="M14 10h4M14 14h4"/>`),
  list: ic(svg`<path d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01"/>`),
};

/** Weather state → icon (falls back on day/night). */
export function weatherIcon(w) {
  if (!w) return icons.box;
  const s = String(w.state || '');
  if (/storm|thunder|гроз/.test(s)) return icons.storm;
  if (/rain|drizzle|shower/.test(s)) return icons.rain;
  if (/fog|mist/.test(s)) return icons.fog;
  if (/cloud|overcast/.test(s)) return icons.cloud;
  return w.day ? icons.sun : icons.moon;
}
