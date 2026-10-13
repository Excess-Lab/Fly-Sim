// Shared design tokens and component styles for the Fly-Sim web UI.
// Tokens live on :root (document level) so they reach every shadow root; `shared` is
// adopted by each Lit component for the common building blocks (cards, buttons, chips…).
import { css } from './vendor/lit-core.min.js';

const LIGHT = `
  --bg: #f5f5f2; --surface: #ffffff; --surface-2: #f0efea; --surface-3: #e7e5df;
  --border: #e3e1db; --border-strong: #cfccc4;
  --text: #1c1c1e; --text-2: #5f5f66; --text-3: #8b8b92;
  --accent: #b4410f; --accent-hover: #9a370c; --accent-ink: #ffffff; --accent-soft: rgba(180, 65, 15, 0.10);
  --ok: #15803d; --ok-soft: rgba(21, 128, 61, 0.12);
  --warn: #b45309; --warn-soft: rgba(180, 83, 9, 0.13);
  --danger: #c81e1e; --danger-soft: rgba(200, 30, 30, 0.10);
  --info: #2563eb; --info-soft: rgba(37, 99, 235, 0.10);
  --male: #2563eb; --female: #db2777; --neutral: #6b7280;
  --brain-bg: #0e0e12; --scene-bg: #fbfbf9; --scene-grid: #e6e4de; --night: 12, 18, 42;
  --shadow: 0 1px 2px rgba(0,0,0,.04), 0 2px 8px rgba(0,0,0,.04);
  --shadow-l: 0 12px 40px rgba(0,0,0,.18);
  color-scheme: light;`;
const DARK = `
  --bg: #111113; --surface: #19191c; --surface-2: #202024; --surface-3: #2a2a2f;
  --border: #2a2a2f; --border-strong: #3a3a40;
  --text: #ececef; --text-2: #a3a3ab; --text-3: #75757d;
  --accent: #f08a4b; --accent-hover: #f59e63; --accent-ink: #1a0d05; --accent-soft: rgba(240, 138, 75, 0.14);
  --ok: #4ade80; --ok-soft: rgba(74, 222, 128, 0.13);
  --warn: #fbbf24; --warn-soft: rgba(251, 191, 36, 0.13);
  --danger: #f87171; --danger-soft: rgba(248, 113, 113, 0.14);
  --info: #60a5fa; --info-soft: rgba(96, 165, 250, 0.14);
  --male: #60a5fa; --female: #f472b6; --neutral: #9ca3af;
  --brain-bg: #08080a; --scene-bg: #141417; --scene-grid: #26262b; --night: 0, 0, 10;
  --shadow: 0 1px 2px rgba(0,0,0,.3);
  --shadow-l: 0 12px 40px rgba(0,0,0,.5);
  color-scheme: dark;`;

export const tokensCSS = `
:root {
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px; --sp-5: 24px; --sp-6: 32px; --sp-7: 48px;
  --r-s: 6px; --r: 10px; --r-l: 14px; --r-pill: 999px;
  --fs-xs: 11.5px; --fs-s: 12.5px; --fs: 14px; --fs-m: 15px; --fs-l: 18px; --fs-xl: 22px;
  ${LIGHT}
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { ${DARK} } }
:root[data-theme="dark"] { ${DARK} }
html, body { margin: 0; background: var(--bg); color: var(--text); }
body { font: var(--fs)/1.45 var(--font); -webkit-font-smoothing: antialiased; overflow-x: hidden; }
`;

export function installTokens() {
  if (document.getElementById('fs-tokens')) return;
  const s = document.createElement('style');
  s.id = 'fs-tokens';
  s.textContent = tokensCSS;
  document.head.appendChild(s);
}

export const shared = css`
  :host { box-sizing: border-box; }
  *, *::before, *::after { box-sizing: border-box; }
  [hidden] { display: none !important; }
  h1, h2, h3, h4 { margin: 0; font-weight: 600; letter-spacing: -0.01em; }
  h1 { font-size: var(--fs-xl); line-height: 1.2; }
  h2 { font-size: var(--fs-l); }
  h3 { font-size: var(--fs-m); }
  h4 { font-size: var(--fs); }
  p { margin: 0; }
  a { color: inherit; }
  button, input, select, textarea { font: inherit; color: inherit; }
  .muted { color: var(--text-2); }
  .faint { color: var(--text-3); }
  .small { font-size: var(--fs-s); }
  .xs { font-size: var(--fs-xs); }
  .num { font-variant-numeric: tabular-nums; }
  .mono { font-family: var(--mono); font-size: var(--fs-s); }
  .row { display: flex; gap: var(--sp-2); align-items: center; flex-wrap: wrap; }
  .stack { display: grid; gap: var(--sp-3); align-content: start; min-width: 0; }
  .spacer { flex: 1; }

  .page-head { display: flex; gap: var(--sp-4); justify-content: space-between; align-items: flex-end;
    flex-wrap: wrap; margin-bottom: var(--sp-5); }
  .page-head .sub { color: var(--text-2); margin-top: var(--sp-1); max-width: 70ch; }
  .section-title { display: flex; align-items: baseline; gap: var(--sp-2); margin: var(--sp-6) 0 var(--sp-3); }
  .section-title:first-child { margin-top: 0; }
  .section-title .count { color: var(--text-3); font-size: var(--fs-s); }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r);
    box-shadow: var(--shadow); min-width: 0; }
  .card-h { display: flex; gap: var(--sp-2); justify-content: space-between; align-items: center; flex-wrap: wrap;
    padding: var(--sp-3) var(--sp-4); border-bottom: 1px solid var(--border); }
  .card-h h3, .card-h h4 { display: flex; align-items: center; gap: var(--sp-2); }
  .card-h .hint { color: var(--text-3); font-size: var(--fs-s); }
  .card-b { padding: var(--sp-3) var(--sp-4); }
  .card-f { padding: var(--sp-3) var(--sp-4); border-top: 1px solid var(--border); }
  .grid { display: grid; gap: var(--sp-3); grid-template-columns: repeat(auto-fill, minmax(min(100%, 320px), 1fr)); }

  .btn { display: inline-flex; align-items: center; justify-content: center; gap: 6px; height: 32px;
    padding: 0 var(--sp-3); border-radius: var(--r-s); border: 1px solid var(--border-strong);
    background: var(--surface); color: var(--text); font-weight: 500; font-size: var(--fs-s);
    cursor: pointer; text-decoration: none; white-space: nowrap; transition: background .12s, border-color .12s; }
  .btn:hover { background: var(--surface-2); }
  .btn:focus-visible, .focusable:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .btn:disabled { opacity: .5; cursor: default; }
  .btn.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
  .btn.primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
  .btn.ghost { border-color: transparent; background: transparent; }
  .btn.ghost:hover { background: var(--surface-2); }
  .btn.danger { color: var(--danger); }
  .btn.danger:hover { background: var(--danger-soft); border-color: var(--danger); }
  .btn.danger.solid { background: var(--danger); border-color: var(--danger); color: #fff; }
  .btn.sm { height: 28px; padding: 0 var(--sp-2); font-size: var(--fs-xs); }
  .btn.icon { width: 32px; padding: 0; }
  .btn.sm.icon { width: 28px; }
  .btn.on { background: var(--text); color: var(--surface); border-color: var(--text); }
  .btn svg { width: 15px; height: 15px; flex: none; }

  .chip { display: inline-flex; align-items: center; gap: 6px; height: 22px; padding: 0 var(--sp-2);
    border-radius: var(--r-pill); font-size: var(--fs-xs); font-weight: 500; white-space: nowrap;
    background: var(--surface-2); color: var(--text-2); }
  .chip::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .chip.plain::before { display: none; }
  .chip.running { background: var(--ok-soft); color: var(--ok); }
  .chip.running::before { animation: pulse 1.6s ease-in-out infinite; }
  .chip.paused { background: var(--warn-soft); color: var(--warn); }
  .chip.finished { background: var(--surface-2); color: var(--text-2); }
  .chip.error { background: var(--danger-soft); color: var(--danger); }
  .chip.info { background: var(--info-soft); color: var(--info); }
  @keyframes pulse { 50% { opacity: .3; } }
  @media (prefers-reduced-motion: reduce) { .chip.running::before { animation: none; } }

  .pill { display: inline-flex; gap: 6px; align-items: baseline; padding: 3px var(--sp-2); border-radius: var(--r-s);
    background: var(--surface-2); font-size: var(--fs-xs); color: var(--text-2); white-space: nowrap; }
  .pill b { color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums; }

  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; flex: none; }
  .legend { display: flex; gap: var(--sp-1) var(--sp-3); flex-wrap: wrap; font-size: var(--fs-xs); color: var(--text-2); }
  .legend span { display: inline-flex; gap: 5px; align-items: center; }

  .stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: var(--sp-3) var(--sp-3); }
  .stat { font-size: var(--fs-xs); color: var(--text-2); min-width: 0; }
  .stat b { display: block; color: var(--text); font-size: var(--fs); font-weight: 600; font-variant-numeric: tabular-nums;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .stat .norm { color: var(--text-3); margin-top: 2px; }
  .meter { height: 3px; background: var(--surface-3); border-radius: 3px; margin-top: 4px; overflow: hidden; }
  .meter i { display: block; height: 100%; background: var(--text-2); border-radius: 3px; }

  .stackbar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; background: var(--surface-3); }

  .select, .input { height: 32px; border: 1px solid var(--border-strong); background: var(--surface);
    border-radius: var(--r-s); padding: 0 var(--sp-2); font-size: var(--fs-s); max-width: 100%; min-width: 0; }
  .select:focus-visible, .input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }

  .err { white-space: pre-wrap; background: var(--danger-soft); color: var(--danger); padding: var(--sp-3);
    border-radius: var(--r-s); font-size: var(--fs-s); overflow: auto; max-height: 240px; }
  .empty { color: var(--text-3); padding: var(--sp-5); text-align: center; border: 1px dashed var(--border-strong);
    border-radius: var(--r); font-size: var(--fs-s); }
  .badge { display: inline-flex; align-items: center; height: 20px; padding: 0 7px; border-radius: var(--r-pill);
    font-size: 11px; font-weight: 500; background: var(--surface-2); color: var(--text-2); white-space: nowrap; }
  .badge.warn { background: var(--warn-soft); color: var(--warn); }

  table.kv { border-collapse: collapse; width: 100%; font-size: var(--fs-s); }
  table.kv th, table.kv td { border-bottom: 1px solid var(--border); padding: 6px var(--sp-2); text-align: left; vertical-align: top; }
  table.kv th { color: var(--text-2); font-weight: 500; width: 40%; }

  canvas { display: block; width: 100%; }
`;

