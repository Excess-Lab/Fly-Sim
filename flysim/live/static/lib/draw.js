// Canvas drawing: arena scene, whole-brain view, chronicle timeline, region dominance over time.
import { api, b64, cssVar, sexColor, SCHEMA, CAT_COLORS, CAT_RU, fmtClock, daylight } from './api.js';

/** Resize the backing store of a canvas to its CSS size × devicePixelRatio. */
export function fit(c) {
  const r = c.getBoundingClientRect(), d = devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * d)), h = Math.max(1, Math.round(r.height * d));
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
}

/** Drag to rotate (dx, dy) and wheel to zoom; returns a detach function. */
export function attachOrbit(cv, onMove, redraw) {
  let last = null;
  const down = e => { last = [e.clientX, e.clientY]; cv.setPointerCapture?.(e.pointerId); cv.style.cursor = 'grabbing'; };
  const move = e => { if (!last) return; onMove(e.clientX - last[0], e.clientY - last[1], 0); last = [e.clientX, e.clientY]; redraw(); };
  const up = () => { last = null; cv.style.cursor = ''; };
  const wheel = e => { e.preventDefault(); onMove(0, 0, e.deltaY); redraw(); };
  cv.addEventListener('pointerdown', down);
  cv.addEventListener('pointermove', move);
  cv.addEventListener('pointerup', up);
  cv.addEventListener('pointercancel', up);
  cv.addEventListener('wheel', wheel, { passive: false });
  return () => {
    cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move);
    cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up);
    cv.removeEventListener('wheel', wheel);
  };
}

// ---------- brain point clouds (static per dataset, cached) ----------
const brains = {};
export function brainOf(ds) {
  if (!brains[ds]) {
    brains[ds] = { ready: false, promise: api('/api/brain/' + encodeURIComponent(ds)).then(b => {
      Object.assign(brains[ds], { ready: true, n: b.n, xyz: b64(b.xyz, Float32Array), region: b64(b.region, Uint8Array) });
      return brains[ds];
    }).catch(e => { delete brains[ds]; throw e; }) };
  }
  return brains[ds].promise;
}
export const brainReady = ds => brains[ds]?.ready ? brains[ds] : null;

const rnd = seed => { const x = Math.sin(seed * 12.9898) * 43758.5453; return x - Math.floor(x); };

// ---------- arena ----------
/**
 * view: {az, el, zoom}; fr: frame; trails: {name: [[x,y,z]…]}.
 * At night (garden climate) the ground and fruit are dimmed by log-illuminance; flies stay legible.
 */
export function drawArena(cv, view, fr, trails) {
  fit(cv);
  const sx = cv.getContext('2d'), W = cv.width, H = cv.height, dpr = devicePixelRatio || 1;
  sx.clearRect(0, 0, W, H);
  sx.fillStyle = cssVar('--scene-bg'); sx.fillRect(0, 0, W, H);
  if (!fr) return;
  const A = fr.arena_mm || 55, s = Math.min(W, H) / (2.6 * A) * view.zoom;
  const ca = Math.cos(view.az), sa = Math.sin(view.az), ce = Math.cos(view.el), se = Math.sin(view.el);
  const proj = p => { const x = p[0] * ca - p[1] * sa, y = p[0] * sa + p[1] * ca; return [W / 2 + x * s, H / 2 + (y * ce - p[2] * se) * s * 0.95]; };
  const font = (px, w = 400) => `${w} ${px * dpr}px ${cssVar('--font') || 'sans-serif'}`;
  const muted = cssVar('--text-2'), ink = cssVar('--text');

  // floor outline + grid
  sx.strokeStyle = cssVar('--scene-grid'); sx.lineWidth = dpr;
  for (let g = -A; g <= A + 1e-6; g += A / 4) {
    for (const [a, b] of [[[g, -A, 0], [g, A, 0]], [[-A, g, 0], [A, g, 0]]]) {
      const p = proj(a), q = proj(b); sx.beginPath(); sx.moveTo(...p); sx.lineTo(...q); sx.stroke();
    }
  }
  sx.strokeStyle = cssVar('--border-strong'); sx.beginPath();
  [[-A, -A], [A, -A], [A, A], [-A, A], [-A, -A]].forEach(([x, y], k) => { const p = proj([x, y, 0]); k ? sx.lineTo(...p) : sx.moveTo(...p); });
  sx.stroke();

  // hot zones
  for (const z of fr.zones || []) {
    sx.fillStyle = z.active === false ? 'rgba(220,38,38,0.07)' : 'rgba(220,38,38,0.22)'; sx.beginPath();
    for (let k = 0; k <= 48; k++) {
      const a = k / 48 * 2 * Math.PI, p = proj([z.center[0] + z.radius * Math.cos(a), z.center[1] + z.radius * Math.sin(a), 0]);
      k ? sx.lineTo(...p) : sx.moveTo(...p);
    }
    sx.fill();
  }

  // apples with cut wound and Penicillium colonies (white rim, blue-green centre) spreading from the wound
  const apples = fr.apples || (fr.apple ? [fr.apple] : []);
  const labels = [];
  apples.forEach((ap, ai) => {
    const p = proj(ap.center), R = Math.max(2, ap.radius * s);
    const sh = proj([ap.center[0], ap.center[1], 0]);
    sx.fillStyle = 'rgba(0,0,0,0.12)'; sx.beginPath(); sx.ellipse(sh[0], sh[1], R, R * 0.35, 0, 0, 7); sx.fill();
    const gr = sx.createRadialGradient(p[0] - R * .35, p[1] - R * .4, R * .08, p[0], p[1], R);
    gr.addColorStop(0, ap.color_light); gr.addColorStop(0.7, ap.color); gr.addColorStop(1, 'rgba(40,25,10,0.9)');
    sx.fillStyle = gr; sx.beginPath(); sx.arc(p[0], p[1], R, 0, 7); sx.fill();
    const wx = p[0] + R * 0.25, wy = p[1] - R * 0.15;
    if (ap.cut) { sx.fillStyle = 'rgba(245,230,180,0.6)'; sx.beginPath(); sx.ellipse(wx, wy, R * 0.28, R * 0.18, 0.4, 0, 7); sx.fill(); }
    const m = ap.mold || 0, nSpots = Math.round(m * 60);
    for (let k = 0; k < nSpots; k++) {
      const a = rnd(ai * 97 + k) * 6.283, rr = R * Math.sqrt(rnd(ai * 31 + k * 7)) * Math.min(1, 0.3 + m);
      const cx = ap.cut ? wx : p[0] + R * 0.5 * (rnd(ai + k * 1.7) - 0.5), cy = ap.cut ? wy : p[1] + R * 0.5 * (rnd(ai + k * 2.3) - 0.5);
      const x = cx + rr * Math.cos(a), y = cy + rr * Math.sin(a) * 0.9;
      if (Math.hypot(x - p[0], y - p[1]) > R * 0.96) continue;
      const rs = R * (0.03 + 0.07 * rnd(k * 3.1 + ai)) * (0.6 + m);
      sx.fillStyle = 'rgba(240,240,235,0.85)'; sx.beginPath(); sx.arc(x, y, rs * 1.3, 0, 7); sx.fill();
      sx.fillStyle = 'rgba(70,130,120,0.92)'; sx.beginPath(); sx.arc(x, y, rs, 0, 7); sx.fill();
    }
    labels.push([`яблоко ${ai + 1}${m > 0.05 ? ' · плесень ' + Math.round(m * 100) + '%' : ''}`, p[0] - R * 0.6, p[1] + R + 13 * dpr]);
  });

  // brood on the first apple
  const br = fr.brood || { egg: 0, larva: 0, pupa: 0 }, nb = (br.egg || 0) + (br.larva || 0) + (br.pupa || 0), ap0 = apples[0];
  if (ap0) for (let k = 0; k < Math.min(nb, 80); k++) {
    const p = proj(ap0.center), R = ap0.radius * s, a = k * 2.4, rr = R * (0.25 + 0.6 * ((k * 37) % 10) / 10);
    sx.fillStyle = k < br.egg ? '#fffbe8' : (k < br.egg + br.larva ? '#f1ecd2' : '#8a5a3c');
    sx.beginPath(); sx.arc(p[0] + rr * Math.cos(a), p[1] + rr * Math.sin(a) * .8, 2.2 * dpr, 0, 7); sx.fill();
  }

  // night: dim the ground and fruit by log-illuminance (flies and labels are drawn on top)
  const w = fr.env?.weather;
  if (w) {
    const dark = 1 - daylight(w.lux);
    if (dark > 0.02) { sx.fillStyle = `rgba(${cssVar('--night') || '12,18,42'},${(0.72 * dark).toFixed(3)})`; sx.fillRect(0, 0, W, H); }
  }

  sx.fillStyle = muted; sx.font = font(11);
  for (const [t, x, y] of labels) sx.fillText(t, x, y);
  for (const c of fr.corpses || []) {
    const q = proj(c.pos); sx.fillStyle = 'rgba(120,120,120,.85)'; sx.beginPath(); sx.arc(q[0], q[1], 3 * dpr, 0, 7); sx.fill();
    sx.fillStyle = muted; sx.font = font(11); sx.fillText(`✝ ${c.name} (${c.cause})`, q[0] + 4 * dpr, q[1] - 4 * dpr);
  }

  // flies, trails, heading, wings, proboscis, copulation ring
  for (const [n, f] of Object.entries(fr.flies || {})) {
    const col = sexColor(f.sex), tr = trails[n] || [];
    sx.strokeStyle = col; sx.globalAlpha = .35; sx.lineWidth = 1.5 * dpr; sx.beginPath();
    tr.forEach((pp, k) => { const q = proj(pp); k ? sx.lineTo(...q) : sx.moveTo(...q); }); sx.stroke(); sx.globalAlpha = 1;
    const q = proj(f.pos), h = proj([f.pos[0] + 3 * Math.cos(f.yaw), f.pos[1] + 3 * Math.sin(f.yaw), f.pos[2]]), r = Math.max(5 * dpr, 1.25 * s);
    if (f.mode === 'fly') { const z = proj([f.pos[0], f.pos[1], 0]); sx.fillStyle = 'rgba(0,0,0,.12)'; sx.beginPath(); sx.arc(z[0], z[1], r, 0, 7); sx.fill(); }
    sx.fillStyle = col; sx.beginPath(); sx.arc(q[0], q[1], r, 0, 7); sx.fill();
    sx.strokeStyle = col; sx.lineWidth = 2 * dpr; sx.beginPath(); sx.moveTo(...q); sx.lineTo(...h); sx.stroke();
    if (f.wing > .3) { sx.lineWidth = dpr; sx.beginPath(); sx.arc(q[0], q[1], r * 2, 0, 7); sx.stroke(); }
    if (f.proboscis) { sx.fillStyle = '#ea580c'; sx.beginPath(); sx.arc(h[0], h[1], 3 * dpr, 0, 7); sx.fill(); }
    if (f.copulating) { sx.strokeStyle = cssVar('--danger'); sx.lineWidth = 2 * dpr; sx.beginPath(); sx.arc(q[0], q[1], r * 2.6, 0, 7); sx.stroke(); }
    sx.fillStyle = ink; sx.font = font(12, 600); sx.fillText(n, q[0] + r + 4 * dpr, q[1] - r);
  }

  // wind arrow (world direction, rotates with the view)
  const wv = fr.env?.wind;
  if (wv && Math.hypot(wv[0], wv[1]) > 0.005) drawWind(sx, W, dpr, wv, view, ink, muted, font);
}

function drawWind(sx, W, dpr, wv, view, ink, muted, font) {
  const sp = Math.hypot(wv[0], wv[1]);
  const cx = W - 34 * dpr, cy = 34 * dpr, R = 20 * dpr;
  // direction in screen space: project the unit wind vector with the same rotation as the floor
  const ux = wv[0] / sp, uy = wv[1] / sp, ca = Math.cos(view.az), sa = Math.sin(view.az), ce = Math.cos(view.el);
  let dx = ux * ca - uy * sa, dy = (ux * sa + uy * ca) * ce;
  const l = Math.hypot(dx, dy) || 1; dx /= l; dy /= l;
  sx.fillStyle = cssVar('--surface'); sx.globalAlpha = .85; sx.beginPath(); sx.arc(cx, cy, R + 4 * dpr, 0, 7); sx.fill(); sx.globalAlpha = 1;
  sx.strokeStyle = cssVar('--border-strong'); sx.lineWidth = dpr; sx.beginPath(); sx.arc(cx, cy, R + 4 * dpr, 0, 7); sx.stroke();
  const len = R * Math.min(1, 0.45 + sp / 2);
  const x0 = cx - dx * len, y0 = cy - dy * len, x1 = cx + dx * len, y1 = cy + dy * len;
  sx.strokeStyle = ink; sx.fillStyle = ink; sx.lineWidth = 2 * dpr; sx.lineCap = 'round';
  sx.beginPath(); sx.moveTo(x0, y0); sx.lineTo(x1, y1); sx.stroke();
  const hx = -dx, hy = -dy, a = 0.5, hl = 7 * dpr;
  sx.beginPath(); sx.moveTo(x1, y1);
  sx.lineTo(x1 + hl * (hx * Math.cos(a) - hy * Math.sin(a)), y1 + hl * (hx * Math.sin(a) + hy * Math.cos(a)));
  sx.lineTo(x1 + hl * (hx * Math.cos(-a) - hy * Math.sin(-a)), y1 + hl * (hx * Math.sin(-a) + hy * Math.cos(-a)));
  sx.closePath(); sx.fill(); sx.lineCap = 'butt';
  sx.fillStyle = muted; sx.font = font(10.5, 500); sx.textAlign = 'center';
  sx.fillText(`${sp.toFixed(2)} м/с`, cx, cy + R + 17 * dpr); sx.textAlign = 'start';
}

// ---------- brain ----------
/** view: {az, el, zoom}; f: fly frame record (idx/val base64 active neurons). */
export function drawBrain(cv, view, f) {
  const B = brainReady(f.ds || f.sex);
  fit(cv);
  const W = cv.width, H = cv.height, c = cv.getContext('2d'), dpr = devicePixelRatio || 1;
  c.fillStyle = cssVar('--brain-bg'); c.fillRect(0, 0, W, H);
  c.font = `${12 * dpr}px ${cssVar('--font') || 'sans-serif'}`;
  if (!B) { c.fillStyle = '#8b8b92'; c.fillText('загрузка положений нейронов…', 12 * dpr, 22 * dpr); return; }
  const ca = Math.cos(view.az), sa = Math.sin(view.az), ce = Math.cos(view.el), se = Math.sin(view.el), S = Math.min(W, H) * .5 * view.zoom;
  const P = i => {
    const x = B.xyz[3 * i], y = B.xyz[3 * i + 1], z = B.xyz[3 * i + 2];
    const X = x * ca - z * sa, Z = x * sa + z * ca; return [W / 2 + X * S, H / 2 + (y * ce - Z * se) * S, x];
  };
  c.fillStyle = 'rgba(170,170,190,.10)';
  const st = Math.max(1, Math.floor(B.n / 24000));
  for (let i = 0; i < B.n; i += st) { const q = P(i); if (!isNaN(q[2])) c.fillRect(q[0], q[1], dpr, dpr); }
  if (f.idx) {
    const idx = b64(f.idx, Uint32Array), val = b64(f.val, Uint8Array);
    for (let j = 0; j < idx.length; j++) {
      const i = idx[j], q = P(i); if (isNaN(q[2])) continue;
      c.fillStyle = SCHEMA.region_colors[B.region[i]]; c.globalAlpha = Math.min(1, .35 + val[j] / 4);
      const r = (1.1 + Math.min(2, val[j] / 3)) * dpr; c.fillRect(q[0] - r / 2, q[1] - r / 2, r, r);
    }
    c.globalAlpha = 1;
  } else {
    c.fillStyle = '#8b8b92';
    c.fillText('облегчённый кадр: полная активность мозга хранится раз в секунду и вокруг событий', 12 * dpr, 22 * dpr);
  }
}

// ---------- chronicle ----------
/** Dot timeline of events by category; returns a function mapping a click x (CSS px) to the nearest event. */
export function drawTimeline(cv, events, clockRange) {
  fit(cv);
  const c = cv.getContext('2d'), W = cv.width, H = cv.height, dpr = devicePixelRatio || 1;
  c.clearRect(0, 0, W, H);
  const t0 = clockRange[0], t1 = Math.max(clockRange[1], ...events.map(e => e.clock_h), t0 + 1e-6);
  const cats = Object.keys(CAT_RU), pad = 10 * dpr, top = 8 * dpr, bottom = 20 * dpr;
  c.strokeStyle = cssVar('--border'); c.lineWidth = dpr;
  for (let k = 0; k < cats.length; k++) {
    const y = top + (H - top - bottom) * (k + 0.5) / cats.length;
    c.beginPath(); c.moveTo(pad, y); c.lineTo(W - pad, y); c.stroke();
  }
  c.fillStyle = cssVar('--text-3'); c.font = `${10.5 * dpr}px ${cssVar('--font') || 'sans-serif'}`;
  c.fillText(fmtClock(t0), pad, H - 5 * dpr);
  c.textAlign = 'right'; c.fillText(fmtClock(t1), W - pad, H - 5 * dpr); c.textAlign = 'start';
  const X = t => pad + (W - 2 * pad) * (t - t0) / (t1 - t0);
  for (const e of events) {
    const k = Math.max(0, cats.indexOf(e.cat)), y = top + (H - top - bottom) * (k + 0.5) / cats.length;
    c.fillStyle = CAT_COLORS[e.cat] || '#999'; c.globalAlpha = 0.85;
    c.beginPath(); c.arc(X(e.clock_h), y, (1 + (e.imp || 1)) * dpr, 0, 7); c.fill();
  }
  c.globalAlpha = 1;
  return cssX => {
    const tt = t0 + (cssX * dpr - pad) / (W - 2 * pad) * (t1 - t0);
    let best = null, bd = Infinity;
    for (const e of events) { const d = Math.abs(e.clock_h - tt); if (d < bd) { bd = d; best = e; } }
    return best;
  };
}

/** Stacked region dominance per series row for one fly; black lines = main events. */
export function drawRegions(cv, series, fly, events, noSensory) {
  fit(cv);
  const c = cv.getContext('2d'), W = cv.width, H = cv.height, dpr = devicePixelRatio || 1;
  c.clearRect(0, 0, W, H);
  const rows = (series || []).filter(r => r.flies[fly] && r.flies[fly].regions);
  if (!rows.length) {
    c.fillStyle = cssVar('--text-3'); c.font = `${12 * dpr}px ${cssVar('--font') || 'sans-serif'}`;
    c.fillText('нет данных об областях мозга для этой мухи', 10 * dpr, 22 * dpr); return;
  }
  const R = rows[0].flies[fly].regions.length;
  for (let k = 0; k < rows.length; k++) {
    const reg = rows[k].flies[fly].regions.map((v, r) => noSensory && r === 1 ? 0 : v), tot = reg.reduce((a, b) => a + b, 0) || 1;
    let y = H;
    const x = W * k / rows.length, w = Math.max(1, W / rows.length + 0.5);
    for (let r = 0; r < R; r++) { const h = H * reg[r] / tot; c.fillStyle = SCHEMA.region_colors[r]; c.fillRect(x, y - h, w, h); y -= h; }
  }
  c.strokeStyle = cssVar('--text'); c.lineWidth = dpr;
  for (const e of events.filter(e => (e.fly === fly || (e.others || []).includes(fly)) && e.imp >= 3)) {
    const k = rows.findIndex(r => r.i >= e.frame); if (k < 0) continue;
    const x = Math.round(W * k / rows.length) + 0.5;
    c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke();
  }
}
