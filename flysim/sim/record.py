"""Run a world and record behaviour, internal state and whole-brain activity (sparse)."""
import json
import time
from pathlib import Path

import numpy as np

from flysim.sim.eventlog import EventLog

KEY_RATES = ["fwd", "back", "turn_DNa02", "gf", "takeoff", "flight_power", "land", "per",
             "song", "receptive", "arousal_P1"]


class Recorder:
    def __init__(self, world, frame_dt: float = 0.05, title: str = "эксперимент"):
        from flysim.sim.chronicle import Chronicle
        self.world = world
        self.frame_dt = frame_dt
        self.chron = Chronicle(run_id=f"{time.strftime('%Y%m%d-%H%M%S')}-{abs(hash(title)) % 10000:04d}",
                               meta={"title": title})
        self._chron_every = max(1, int(round(0.1 / frame_dt)))   # chronicle at 10 frames/s
        self._k = 0
        self.frames = {f.cfg.name: [] for f in world.flies}
        self.spikes = {f.cfg.name: [] for f in world.flies}  # per frame: (idx uint32, cnt uint16)
        self.log = EventLog()

    def snap(self, t):
        from flysim.sim.chronicle import make_frame
        n_ev = len(self.log.events)
        self.log.frame_idx = self.chron.n
        self.log.frame(self.world, t, {f.cfg.name: f.counts_acc for f in self.world.flies})
        self.chron.note_events(self.log.events[n_ev:])
        self._k += 1
        if self._k % self._chron_every == 0:
            fr = make_frame(self.world, t, {f.cfg.name: f.counts_acc.copy() for f in self.world.flies},
                            self.log, index=self.chron.n)
            self.chron.add(fr)
        for f in self.world.flies:
            b = f.body
            mb = f.brain.kc_mbon_strength() if f.brain.pl_on else np.ones(1)
            self.frames[f.cfg.name].append({
                "t": round(t, 3), "pos": [round(float(v), 3) for v in b.pos],
                "yaw": round(float(b.yaw), 4), "mode": b.mode, "surface": b.surface,
                "proboscis": bool(b.proboscis), "wing": round(float(b.wing_ext), 3),
                "hunger": round(f.meta.hunger, 4), "energy": round(f.meta.energy, 4),
                "crop_ul": round(f.meta.crop_ul, 4), "temp": round(float(f.last_temp), 2),
                "rates": {k: round(float(f.motor.rate[k]), 2) for k in KEY_RATES},
                "turn_lr": round(float(f.cmd["turn"]), 3),
                "mbon_w_min": round(float(mb.min()), 4), "mbon_w_mean": round(float(mb.mean()), 4),
                "copulating": bool(getattr(f, "copulating", False)),
            })
            c = f.pop_counts()
            nz = np.flatnonzero(c)
            self.spikes[f.cfg.name].append((nz.astype(np.uint32), c[nz].astype(np.uint16)))

    def run(self, seconds: float, dt: float = 0.01, log_every: float = 5.0, t0: float = 0.0):
        w = self.world
        n = int(round(seconds / dt))
        per_frame = max(1, int(round(self.frame_dt / dt)))
        start = time.time()
        for k in range(n):
            w.step(dt)
            t = t0 + (k + 1) * dt
            if (k + 1) % per_frame == 0:
                self.snap(t)
            if log_every and (k + 1) % int(round(log_every / dt)) == 0:
                s = " | ".join(
                    f"{f.cfg.name}: {f.body.mode}/{f.body.surface} pos=({f.body.pos[0]:.0f},"
                    f"{f.body.pos[1]:.0f},{f.body.pos[2]:.0f}) PER={f.body.proboscis} "
                    f"hunger={f.meta.hunger:.2f}" for f in w.flies)
                print(f"t={t:6.1f}s wall={time.time() - start:6.0f}s  {s}", flush=True)
        return self

    def save(self, out_dir: Path, meta: dict | None = None, protocol: dict | None = None):
        from flysim.sim.provenance import manifest
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        man = manifest(self.world, protocol or {"name": (meta or {}).get("experiment", "run"),
                                                "parameters": meta or {}})
        (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
        if meta is not None:
            meta["experiment_id"] = man["experiment_id"]
            meta["mode"] = man["mode"]
            meta["model_version"] = man["model_version"]
        apple = self.world.apple
        info = {
            "meta": meta or {}, "frame_dt": self.frame_dt, "arena": self.world.arena,
            "apple": {"center": apple.center.tolist(), "radius": apple.p.radius_mm,
                      "summary": apple.summary(), "state": vars(apple.s)},
            "apples": [{"center": a.center.tolist(), "radius": a.p.radius_mm,
                        "summary": a.summary(), "state": vars(a.s)} for a in self.world.apples],
            "zones": [vars(z) for z in self.world.zones],
            "flies": {f.cfg.name: {"sex": f.cfg.sex, "config": vars(f.cfg), "events": f.events}
                      for f in self.world.flies},
        }
        (out_dir / "run.json").write_text(json.dumps(info, default=float, indent=1))
        (out_dir / "frames.json").write_text(json.dumps(self.frames))
        self.log.save(out_dir)
        title = (meta or {}).get("experiment", "эксперимент")
        (self.chron.dir / "meta.json").write_text(json.dumps({"title": title, "meta": meta or {}},
                                                             ensure_ascii=False, default=str))
        self.chron.save_logs(self.log, man)
        for name, fr in self.spikes.items():
            lens = np.array([len(i) for i, _ in fr], np.int64)
            np.savez_compressed(out_dir / f"spikes_{name}.npz", lens=lens,
                                idx=np.concatenate([i for i, _ in fr]) if fr else np.zeros(0),
                                cnt=np.concatenate([c for _, c in fr]) if fr else np.zeros(0))
        return out_dir
