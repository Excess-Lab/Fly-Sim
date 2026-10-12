"""Chronicle: the complete, browsable history of any run (web session, free simulation or
experiment) — the data behind the "god panel".

Layout of a run directory  results/runs/<run_id>/
  frames.jsonl     one frame per line (0.1 s of simulated time), written with a delay of WINDOW
                   frames so that we already know whether an important event happened nearby:
                   full frames (whole-brain spikes) every FULL_EVERY frames and within +-WINDOW
                   frames of every event with importance >= 2; light frames otherwise
  events.jsonl     every detected event (category, importance, flies, frame index, text)
  manifest.json    provenance (model version, mode, coverage, conditions)
  timeseries.csv, apple.csv, events.txt — tabular logs
"""
import base64
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from flysim.viz.export import _apple_colors

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "runs"
MAX_ACTIVE = 3000
APPLE_KEYS = ("sugar", "ethanol", "acetic", "yeast", "aab", "mold", "patulin")


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def make_frame(world, t_session, counts: dict, log, status: str = "", index: int = 0) -> dict:
    """Snapshot of the whole world + every fly (+ sparse whole-brain spikes)."""
    from flysim.sim.provenance import mechanism_mode
    w = world
    fr = {"i": index, "t": round(t_session, 2), "clock_h": round(w.clock_h, 4),
          "apples": [dict(zip(("color", "color_light"), _apple_colors(vars(a.s))),
                          center=a.center.tolist(), radius=a.p.radius_mm, mold=round(float(a.s.mold), 3),
                          summary=a.summary(), cut=bool(getattr(a, "cut", True)),
                          state={k: round(float(getattr(a.s, k)), 4) for k in APPLE_KEYS})
                     for a in w.apples],
          "arena_mm": w.arena,
          "env": {"o2_pct": round(w.o2_pct, 2), "co2_ppm": round(w.co2_ppm), "temp_c": round(w.ambient_c, 1)},
          "zones": [{"center": z.center, "radius": z.radius, "temp_c": z.temp_c, "active": z.active}
                    for z in w.zones],
          "brood": w.population.counts(w.clock_h) if getattr(w, "population", None) else
          {"egg": 0, "larva": 0, "pupa": 0},
          "born": getattr(getattr(w, "population", None), "born", 0),
          "plants": w.plants.summary() if getattr(w, "plants", None) else None,
          "status": status, "flies": {},
          "corpses": getattr(w, "corpses", [])[-100:]}
    fr["apple"] = fr["apples"][0]
    for f in w.flies:
        c = counts.get(f.cfg.name)
        if c is None:
            c = np.zeros(f.brain.con.n, np.int32)
        nz = np.flatnonzero(c)
        if len(nz) > MAX_ACTIVE:
            nz = nz[np.argsort(c[nz])[-MAX_ACTIVE:]]
        b, r = f.body, f.motor.rate
        fr["flies"][f.cfg.name] = {
            "sex": f.cfg.sex, "pos": [round(float(v), 2) for v in b.pos], "yaw": round(float(b.yaw), 3),
            "mode": b.mode, "surface": b.surface, "proboscis": bool(b.proboscis),
            "wing": round(float(b.wing_ext), 2), "hunger": round(f.meta.hunger, 3),
            "energy": round(f.meta.energy, 3), "crop_ul": round(f.meta.crop_ul, 3),
            "health": round(f.meta.health, 3), "temp": round(float(getattr(f, "last_temp", 25.0)), 1),
            "mated": f.mated, "copulating": f.copulating, "age_d": round(f.age_d, 2),
            "rates": {k: round(float(v), 1) for k, v in r.items()},
            "mbon_w_min": round(float(f.brain.kc_mbon_strength().min()), 3) if f.brain.pl_on else None,
            "n_active": int(np.count_nonzero(c)), "schema": f.schema.summary(),
            "dataset": getattr(f, "dataset_name", f.cfg.sex), "mode_mech": mechanism_mode(f.cfg),
            "idx": _b64(nz.astype(np.uint32)), "val": _b64(np.minimum(c[nz], 255).astype(np.uint8)),
        }
        fr["flies"][f.cfg.name]["mode"] = b.mode
    fr["events"] = log.events[-60:] if log is not None else []
    return fr


class Chronicle:
    WINDOW = 20        # frames (2 s) kept in memory before writing
    FULL_EVERY = 10    # whole-brain frame at least once per second

    def __init__(self, run_id: str | None = None, root: Path = RUNS, meta: dict | None = None):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.dir / "frames.jsonl", "ab")
        self.offsets = []          # byte offset of every written frame
        self.pending = deque()     # (index, frame) not yet written
        self.n = 0
        self.important = []        # frame indices of events with importance >= 2
        (self.dir / "meta.json").write_text(json.dumps(meta or {}, ensure_ascii=False, default=str))

    def note_events(self, events):
        for e in events:
            if e.get("imp", 1) >= 2:
                self.important.append(e["frame"])

    def add(self, frame: dict):
        frame["i"] = self.n
        self.pending.append(frame)
        self.n += 1
        while len(self.pending) > self.WINDOW:
            self._write(self.pending.popleft())

    def _full(self, i):
        if i % self.FULL_EVERY == 0:
            return True
        return any(abs(i - k) <= self.WINDOW for k in self.important[-200:])

    def _write(self, fr):
        if not self._full(fr["i"]):
            fr = dict(fr)
            fr["flies"] = {n: {k: v for k, v in f.items() if k not in ("idx", "val")}
                           for n, f in fr["flies"].items()}
            fr["light"] = True
        fr = {k: v for k, v in fr.items() if k != "events"}
        line = (json.dumps(fr, ensure_ascii=False, default=float) + "\n").encode()
        self.offsets.append(self.fh.tell())
        self.fh.write(line)

    def flush(self):
        while self.pending:
            self._write(self.pending.popleft())
        self.fh.flush()

    def read(self, i: int) -> dict | None:
        if i < 0 or i >= self.n:
            return None
        if i >= len(self.offsets):
            for fr in self.pending:
                if fr["i"] == i:
                    return fr
            return None
        self.fh.flush()
        with open(self.dir / "frames.jsonl", "rb") as r:
            r.seek(self.offsets[i])
            return json.loads(r.readline())

    def save_logs(self, log, manifest: dict | None = None):
        self.flush()
        log.save(self.dir)
        if manifest is not None:
            (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str,
                                                               ensure_ascii=False))
        (self.dir / "index.json").write_text(json.dumps({"n": self.n, "offsets": self.offsets}))

    def close(self):
        self.flush()
        self.fh.close()


class SavedRun:
    """Read-only access to a finished run directory (same API as Chronicle.read)."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.run_id = self.dir.name
        idx = self.dir / "index.json"
        if idx.exists():
            d = json.loads(idx.read_text())
            self.offsets, self.n = d["offsets"], d["n"]
        else:  # rebuild the index by scanning
            self.offsets = []
            with open(self.dir / "frames.jsonl", "rb") as r:
                pos = 0
                for line in r:
                    self.offsets.append(pos)
                    pos += len(line)
            self.n = len(self.offsets)

    def read(self, i: int):
        if i < 0 or i >= len(self.offsets):
            return None
        with open(self.dir / "frames.jsonl", "rb") as r:
            r.seek(self.offsets[i])
            return json.loads(r.readline())

    def events(self):
        p = self.dir / "events.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def list_runs():
    out = []
    if not RUNS.exists():
        return out
    for d in sorted(RUNS.iterdir(), reverse=True):
        if not (d / "frames.jsonl").exists():
            continue
        meta = {}
        for name in ("manifest.json", "meta.json"):
            if (d / name).exists():
                try:
                    meta = json.loads((d / name).read_text())
                    break
                except json.JSONDecodeError:
                    pass
        out.append({"run_id": d.name, "size_mb": round((d / "frames.jsonl").stat().st_size / 1e6, 1),
                    "title": meta.get("title") or meta.get("protocol", {}).get("name", d.name)
                    if isinstance(meta.get("protocol", {}), dict) else d.name})
    return out
