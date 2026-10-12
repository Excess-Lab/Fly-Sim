"""Live mode: run the world and watch it in the browser while it is being computed.

    .venv/bin/python -m flysim.live --males 1 --females 1 --apple-days 2

then open http://localhost:8800 . Behaviour sessions are simulated neuron-by-neuron; between
sessions the world fast-forwards (apple rots, flies get hungry, eggs develop, new adults
eclose with their own brains).
"""
import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from flysim.agent.fly import Fly, FlyConfig
from flysim.brain.factory import load
from flysim.viz.export import REGION_COLORS, REGION_NAMES, _apple_colors, _positions, _regions
from flysim.world.apple import Apple, AppleParams
from flysim.world.population import Population
from flysim.world.world import World

PAGE = Path(__file__).with_name("live.html")
FRAME_DT = 0.1
MAX_ACTIVE = 3000


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


class LiveSim:
    def __init__(self, args):
        self.args = args
        apple = Apple(AppleParams(radius_mm=30.0))
        apple.age(args.apple_days * 24)
        self.world = World(apple, arena=55.0)
        self.world.population = Population(max_adults=args.max_adults)
        self.frames = []
        self.lock = threading.Lock()
        self.status = "загрузка мозгов…"
        self.n_made = {"male": 0, "female": 0}
        self.brains_static = {}
        self.paused = False
        for i in range(args.males):
            self.add_fly("male", pos=(-45 + 4 * i, -6 + 3 * i, 0), yaw=0.3)
        for i in range(args.females):
            self.add_fly("female", pos=(-40 + 4 * i, 5 - 3 * i, 0), yaw=np.pi + 0.3)

    def add_fly(self, sex, pos, yaw, starved_h=12.0, note=""):
        self.n_made[sex] += 1
        name = f"{'♂' if sex == 'male' else '♀'}{self.n_made[sex]}"
        f = Fly(FlyConfig(name, sex, starved_h=starved_h, seed=hash(name) % 10000), pos=pos, yaw=yaw)
        f.calibrate(self.world, 0.5)
        f.born_note = note
        self.world.flies.append(f)
        if note:
            f.events.append((round(f.t, 3), note))
        return f

    def brain_static(self, sex):
        if sex not in self.brains_static:
            _, ann = load(sex)
            self.brains_static[sex] = {"n": int(len(ann)), "xyz": b64(_positions(ann)),
                                       "region": b64(_regions(ann))}
        return self.brains_static[sex]

    def snapshot(self, t_session):
        w = self.world
        col, light = _apple_colors(vars(w.apple.s))
        fr = {"i": len(self.frames), "t": round(t_session, 2), "clock_h": round(w.clock_h, 4),
              "apple": {"summary": w.apple.summary(), "color": col, "color_light": light,
                        "center": w.apple.center.tolist(), "radius": w.apple.p.radius_mm},
              "brood": w.population.counts(w.clock_h), "status": self.status, "flies": {}}
        for f in w.flies:
            c = f.pop_counts()
            nz = np.flatnonzero(c)
            if len(nz) > MAX_ACTIVE:
                nz = nz[np.argsort(c[nz])[-MAX_ACTIVE:]]
            b = f.body
            fr["flies"][f.cfg.name] = {
                "sex": f.cfg.sex, "pos": [round(float(v), 2) for v in b.pos], "yaw": round(float(b.yaw), 3),
                "mode": b.mode, "surface": b.surface, "proboscis": bool(b.proboscis),
                "wing": round(float(b.wing_ext), 2), "hunger": round(f.meta.hunger, 3),
                "energy": round(f.meta.energy, 3), "crop_ul": round(f.meta.crop_ul, 3),
                "temp": round(float(getattr(f, "last_temp", 25.0)), 1),
                "mated": f.mated, "copulating": f.copulating, "age_d": round(f.age_d, 2),
                "rates": {k: round(float(v), 1) for k, v in f.motor.rate.items()},
                "mbon_w_min": round(float(f.brain.kc_mbon_strength().min()), 3) if f.brain.pl_on else None,
                "events": f.events[-8:],
                "idx": b64(nz.astype(np.uint32)), "val": b64(np.minimum(c[nz], 255).astype(np.uint8)),
            }
        with self.lock:
            self.frames.append(fr)
            if len(self.frames) > 6000:
                self.frames = self.frames[-6000:]

    def fast_forward(self, hours):
        w = self.world
        self.status = f"перемотка {hours:.0f} ч: яблоко гниёт, яйца развиваются…"
        pop = w.population
        ready = pop.fast_forward(w.clock_h, hours, w.apple, w.flies)
        w.apple.age(hours)
        for f in w.flies:
            for _ in range(int(hours)):
                f.meta.step(3600.0)
            f.age_d += hours / 24
            f.copulating = False
        w.clock_h += hours
        # deaths
        for f in list(w.flies):
            if f.age_d > pop.lifespan_d:
                w.flies.remove(f)
        # eclosion (limited by RAM)
        for b in ready:
            b.alive = False
            if len(w.flies) >= pop.max_adults:
                pop.born += 1
                continue
            sex = "female" if pop.rng.random() < 0.5 else "male"
            ang = pop.rng.uniform(0, 2 * np.pi)
            pos = (w.apple.center[0] + 36 * np.cos(ang), w.apple.center[1] + 36 * np.sin(ang), 0)
            f = self.add_fly(sex, pos, ang + np.pi, starved_h=6.0,
                             note=f"вылупилась из куколки (мать {b.mother}, отец {b.father})")
            f.age_d = 0.0
            pop.born += 1

    def run(self):
        a = self.args
        session = 0
        while True:
            session += 1
            self.status = f"сессия {session}: нейроны считаются в реальном масштабе (~×0.1)"
            t = 0.0
            per = int(round(FRAME_DT / 0.01))
            k = 0
            while t < a.session:
                if self.paused:
                    time.sleep(0.1)
                    continue
                self.world.step(0.01)
                t += 0.01
                k += 1
                if k % per == 0:
                    self.snapshot(t)
            self.fast_forward(a.ff_hours)
            self.snapshot(0.0)


def make_handler(sim: LiveSim):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                return self._send(PAGE.read_text(), "text/html; charset=utf-8")
            if u.path == "/api/meta":
                return self._send(json.dumps({"arena": sim.world.arena, "frame_dt": FRAME_DT,
                                              "region_names": REGION_NAMES,
                                              "region_colors": REGION_COLORS}))
            if u.path.startswith("/api/brain/"):
                return self._send(json.dumps(sim.brain_static(u.path.rsplit("/", 1)[1])))
            if u.path == "/api/frames":
                since = int(q.get("since", ["0"])[0])
                with sim.lock:
                    base = sim.frames[0]["i"] if sim.frames else 0
                    out = sim.frames[max(0, since - base):][:40]
                return self._send(json.dumps({"frames": out}))
            if u.path == "/api/pause":
                sim.paused = not sim.paused
                return self._send(json.dumps({"paused": sim.paused}))
            self.send_response(404)
            self.end_headers()

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--males", type=int, default=1)
    ap.add_argument("--females", type=int, default=1)
    ap.add_argument("--apple-days", type=float, default=2.0)
    ap.add_argument("--session", type=float, default=60.0, help="seconds simulated per session")
    ap.add_argument("--ff-hours", type=float, default=24.0, help="fast-forward between sessions")
    ap.add_argument("--max-adults", type=int, default=4)
    ap.add_argument("--port", type=int, default=8800)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), None)
    sim_holder = {}

    def boot():
        sim = LiveSim(a)
        sim_holder["sim"] = sim
        srv.RequestHandlerClass = make_handler(sim)
        sim.run()

    srv.RequestHandlerClass = make_handler(type("Booting", (), {
        "world": type("W", (), {"arena": 55.0})(), "frames": [], "lock": threading.Lock(),
        "brain_static": lambda self, s: {}, "paused": False})())
    threading.Thread(target=boot, daemon=True).start()
    print(f"Fly-Sim live: http://localhost:{a.port}  (Ctrl+C — стоп)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
