"""Fly-Sim web app: configure conditions in the browser, start, watch live, download logs.

    .venv/bin/python -m flysim.live            (or: docker compose up)

then open http://localhost:8800 .
Behaviour sessions are simulated neuron-by-neuron; between sessions the world fast-forwards
(the apple ferments and moulds, flies get hungry, eggs develop, new adults eclose with brains).
"""
import argparse
import base64
import io
import json
import threading
import time
import traceback
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from flysim.agent.fly import Fly, FlyConfig
from flysim.brain.factory import load
from flysim.sim.eventlog import EventLog
from flysim.viz.export import REGION_COLORS, REGION_NAMES, _apple_colors, _positions, _regions
from flysim.world.apple import Apple, AppleParams
from flysim.world.population import Population
from flysim.world.world import World, Zone

ROOT = Path(__file__).resolve().parents[2]
PAGE = Path(__file__).with_name("live.html")
FRAME_DT = 0.1
MAX_ACTIVE = 3000

# Every setting with its unit, allowed range and the normal/reference value
SCHEMA = {
    "environment": {
        "o2_pct": {"label": "Кислород", "unit": "%", "min": 0.0, "max": 30.0, "step": 0.1,
                   "default": 20.9, "norm": "20.9 % — обычный воздух; <8 % мухи замедляются, <2 % — ступор"},
        "co2_ppm": {"label": "CO₂ в воздухе", "unit": "ppm", "min": 0, "max": 50000, "step": 10,
                    "default": 420, "norm": "≈420 ppm на улице; мухи избегают CO₂ от ~1000–2000 ppm (0.1–0.2 %)"},
        "temp_c": {"label": "Температура", "unit": "°C", "min": 15.0, "max": 35.0, "step": 0.5,
                   "default": 25.0, "norm": "25 °C — стандарт лабораторий; >30 °C — неприятный жар"},
        "volume_l": {"label": "Объём воздуха в контейнере", "unit": "л", "min": 1, "max": 1000,
                     "step": 1, "default": 10, "norm": "банка ~1 л, садок ~10 л, комната ~1000 л"},
        "ach": {"label": "Вентиляция", "unit": "обм/ч", "min": 0, "max": 20, "step": 0.5,
                "default": 2, "norm": "0 — герметично (CO₂ копится, O₂ падает); комната ~0.5–6"},
    },
    "apple": {
        "age_days": {"label": "Возраст яблока", "unit": "дн", "min": 0.0, "max": 30.0, "step": 0.5,
                     "default": 2.0, "norm": "надрезанное: 1–3 дн брожение, 4–5 дн спирт, 7+ дн уксус и плесень"},
        "cut": {"label": "Надрезанное (повреждённое)", "type": "bool", "default": True,
                "norm": "целое бродит и плесневеет неделями, надрезанное — днями"},
        "radius_mm": {"label": "Радиус", "unit": "мм", "min": 10.0, "max": 40.0, "step": 1.0,
                      "default": 30.0, "norm": "реальное яблоко ~35–40 мм"},
    },
    "flies": {
        "males": {"label": "Самцы", "unit": "", "min": 0, "max": 3, "step": 1, "default": 1,
                  "norm": "Male CNS, 164 587 нейронов каждый"},
        "females": {"label": "Самки", "unit": "", "min": 0, "max": 3, "step": 1, "default": 1,
                    "norm": "FlyWire, 138 639 нейронов каждая"},
        "starved_h": {"label": "Голодали перед стартом", "unit": "ч", "min": 0.0, "max": 48.0,
                      "step": 1.0, "default": 18.0, "norm": "кормовые опыты: 18–24 ч"},
        "virgin": {"label": "Самки девственницы", "type": "bool", "default": True,
                   "norm": "спаренная самка не готова к спариванию и откладывает яйца"},
    },
    "hot_zone": {
        "enabled": {"label": "Горячая зона на полу", "type": "bool", "default": False,
                    "norm": "как в тепловой арене Ofstad et al. 2011"},
        "temp_c": {"label": "Температура зоны", "unit": "°C", "min": 25.0, "max": 45.0,
                   "step": 0.5, "default": 38.0, "norm": ">30 °C — сигнал наказания"},
        "radius_mm": {"label": "Радиус зоны", "unit": "мм", "min": 5.0, "max": 30.0, "step": 1.0,
                      "default": 15.0, "norm": ""},
    },
    "science": {
        "plasticity": {"label": "Обучение (пластичность KC→MBON)", "type": "bool", "default": True,
                       "norm": "выкл. — контроль «только рефлексы»"},
        "us_shortcut": {"label": "Боль → дофамин по данным опытов (A24)", "type": "bool",
                        "default": True, "norm": "выкл. — только путь по коннектому"},
        "spontaneous": {"label": "Спонтанная ходьба (A30)", "type": "bool", "default": True,
                        "norm": "выкл. — двигается только по командам мозга"},
    },
    "run": {
        "session_s": {"label": "Сессия (нейрон-в-нейрон)", "unit": "с", "min": 5, "max": 600,
                      "step": 5, "default": 60, "norm": "~×0.1 от реального времени на M2"},
        "ff_hours": {"label": "Перемотка между сессиями", "unit": "ч", "min": 0, "max": 72,
                     "step": 1, "default": 24, "norm": "0 — без перемотки"},
        "max_adults": {"label": "Макс. мух с мозгом", "unit": "", "min": 1, "max": 6, "step": 1,
                       "default": 4, "norm": "~0.3–0.5 ГБ памяти на муху"},
    },
}

# reference values shown next to the apple state
APPLE_NORMS = {
    "sugar": ("Сахар", "г/100 г", "свежее яблоко 10–13 (USDA)"),
    "ethanol": ("Этанол", "г/100 г", "свежее 0; брожение 1–5"),
    "acetic": ("Уксусная кислота", "г/100 г", "свежее 0; уксус 4–6"),
    "yeast": ("Дрожжи", "доля", "свежее ≈0; брожение → 1"),
    "aab": ("Уксуснокислые бактерии", "доля", "свежее 0; нужны O₂"),
    "mold": ("Плесень", "доля поверхности", "свежее 0; видимые пятна >0.05"),
    "patulin": ("Патулин (токсин)", "отн.", "свежее 0"),
}


def defaults():
    return {g: {k: v["default"] for k, v in items.items()} for g, items in SCHEMA.items()}


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


class Simulation:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        env, ap, fl, hz, sci, run = (cfg[k] for k in ("environment", "apple", "flies", "hot_zone",
                                                      "science", "run"))
        from flysim.world.atmosphere import Atmosphere
        apple = Apple(AppleParams(radius_mm=float(ap["radius_mm"])), cut=bool(ap["cut"]),
                      temp_c=float(env["temp_c"]), o2_pct=float(env["o2_pct"]))
        atm = Atmosphere(volume_l=float(env["volume_l"]), ach=float(env["ach"]),
                         co2_ppm=float(env["co2_ppm"]), o2_pct=float(env["o2_pct"]),
                         outside_co2_ppm=float(env["co2_ppm"]), outside_o2_pct=float(env["o2_pct"]))
        self.world = World(apple, arena=55.0, ambient_c=float(env["temp_c"]),
                           o2_pct=float(env["o2_pct"]), co2_ppm=float(env["co2_ppm"]), atm=atm)
        # the apple has been rotting in this container for age_days before the flies arrive
        self.world.advance_hours(float(ap["age_days"]) * 24)
        self.world.clock_h = 0.0
        self.world.population = Population(max_adults=int(run["max_adults"]))
        if hz["enabled"]:
            self.world.zones.append(Zone(center=(-35.0, 25.0), radius=float(hz["radius_mm"]),
                                         temp_c=float(hz["temp_c"])))
        self.log = EventLog()
        self.frames = []
        self.lock = threading.Lock()
        self.stop_flag = False
        self.paused = False
        self.status = "загрузка мозгов…"
        self.error = None
        self.n_made = {"male": 0, "female": 0}
        self.started = time.time()
        r = apple.p.radius_mm
        k = 0
        for sex, n in (("male", int(fl["males"])), ("female", int(fl["females"]))):
            for _ in range(n):
                ang = np.pi + 0.5 * k - 0.4
                pos = (np.cos(ang) * (r + 12), np.sin(ang) * (r + 12), 0.0)
                f = self.add_fly(sex, pos, ang + np.pi, float(fl["starved_h"]), calibrate=False)
                if sex == "female" and not fl["virgin"]:
                    f.set_mated("(до опыта)")
                k += 1
        self.world.rebuild_physics()

    def add_fly(self, sex, pos, yaw, starved_h, note="", calibrate=True):
        sci = self.cfg["science"]
        self.n_made[sex] += 1
        name = f"{'♂' if sex == 'male' else '♀'}{self.n_made[sex]}"
        f = Fly(FlyConfig(name, sex, starved_h=starved_h, seed=self.n_made[sex] * 7 + (sex == "male"),
                          plasticity=bool(sci["plasticity"]), us_punish_shortcut=bool(sci["us_shortcut"]),
                          spontaneous=bool(sci["spontaneous"]),
                          body="nmf"), pos=pos, yaw=yaw)
        if note:
            f.events.append((round(f.t, 3), note))
        self.world.flies.append(f)
        return f

    def snapshot(self, t_session):
        w = self.world
        counts = {f.cfg.name: f.pop_counts() for f in w.flies}
        self.log.frame(w, t_session, counts)
        s = w.apple.s
        col, light = _apple_colors(vars(s))
        fr = {"i": len(self.frames), "t": round(t_session, 2), "clock_h": round(w.clock_h, 4),
              "apple": {"summary": w.apple.summary(), "color": col, "color_light": light,
                        "center": w.apple.center.tolist(), "radius": w.apple.p.radius_mm,
                        "state": {k: round(float(getattr(s, k)), 4) for k in APPLE_NORMS},
                        "mold": round(float(s.mold), 3)},
              "env": {"o2_pct": round(w.o2_pct, 2), "co2_ppm": round(w.co2_ppm), "temp_c": w.ambient_c},
              "zones": [{"center": z.center, "radius": z.radius, "temp_c": z.temp_c} for z in w.zones],
              "brood": w.population.counts(w.clock_h), "born": w.population.born,
              "status": self.status, "flies": {}}
        for f in w.flies:
            c = counts[f.cfg.name]
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
                "valence": round(r.get("pam", 0.0) - r.get("ppl1", 0.0), 1),
                "mbon_w_min": round(float(f.brain.kc_mbon_strength().min()), 3) if f.brain.pl_on else None,
                "n_active": int(np.count_nonzero(c)),
                "idx": b64(nz.astype(np.uint32)), "val": b64(np.minimum(c[nz], 255).astype(np.uint8)),
            }
        fr["events"] = self.log.events[-60:]
        with self.lock:
            self.frames.append(fr)
            if len(self.frames) > 3000:
                self.frames = self.frames[-3000:]

    def fast_forward(self, hours):
        w = self.world
        self.status = f"перемотка {hours:.0f} ч: яблоко бродит и плесневеет, яйца развиваются"
        pop = w.population
        anoxic_h = {f.cfg.name: 0.0 for f in w.flies}
        # air and fruit evolve together; track hours of anoxia for each fly (A91)
        left = hours
        while left > 1e-9:
            h = min(1.0, left)
            w.advance_hours(h)
            if w.o2_pct < 1.0:
                for f in w.flies:
                    anoxic_h[f.cfg.name] += h
            left -= h
        ready = pop.fast_forward(w.clock_h, hours, w.apple, w.flies)
        for f in w.flies:
            f.pass_time(hours, w, fed_from_fruit=True)
        w.clock_h += hours
        for f in list(w.flies):
            cause = None
            if f.age_d > pop.lifespan_d * (0.3 + 0.7 * f.meta.health):  # A72
                cause = "старость" + (" (ускорена токсином плесени)" if f.meta.health < 0.9 else "")
            elif getattr(f, "starving_h", 0.0) > 48:
                cause = "голод"
            elif anoxic_h.get(f.cfg.name, 0.0) > 4:   # A91: flies recover from <~4 h of anoxia
                cause = "нехватка кислорода"
            if cause:
                w.flies.remove(f)
                self.log._ev(w, f, 0.0, "death", f"погибла: {cause}, возраст {f.age_d:.1f} дн")
        for b in ready:
            b.alive = False
            pop.born += 1  # eclosed adults (with or without a simulated brain)
            if len(w.flies) >= pop.max_adults:
                continue
            sex = "female" if pop.rng.random() < 0.5 else "male"
            ang = pop.rng.uniform(0, 2 * np.pi)
            rr = w.apple.p.radius_mm + 8
            f = self.add_fly(sex, (rr * np.cos(ang), rr * np.sin(ang), 0.0), ang + np.pi, 6.0,
                             note=f"вылупилась (мать {b.mother}, отец {b.father})")
            f.age_d = 0.0
            self.log._ev(w, f, 0.0, "birth", f"новая муха {f.cfg.name} с собственным мозгом")
        w.rebuild_physics()

    def run(self):
        try:
            run = self.cfg["run"]
            session = 0
            while not self.stop_flag:
                session += 1
                self.status = f"сессия {session}: считаются все нейроны"
                t, k = 0.0, 0
                per = int(round(FRAME_DT / 0.01))
                while t < float(run["session_s"]) and not self.stop_flag:
                    if self.paused:
                        time.sleep(0.1)
                        continue
                    self.world.step(0.01)
                    t += 0.01
                    k += 1
                    if k % per == 0:
                        self.snapshot(t)
                if self.stop_flag:
                    break
                if float(run["ff_hours"]) > 0:
                    self.fast_forward(float(run["ff_hours"]))
                    self.snapshot(0.0)
                if not self.world.flies:
                    self.status = "все мухи погибли"
                    self.snapshot(0.0)
                    break
            self.status = "остановлено"
        except Exception as e:  # surface errors in the UI instead of dying silently
            self.error = f"{e}\n{traceback.format_exc()}"
            self.status = "ошибка: " + str(e)

    def manifest(self) -> dict:
        from flysim.sim.provenance import manifest
        return manifest(self.world, {"name": "web_session", "config": self.cfg})

    def logs_zip(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("config.json", json.dumps(self.cfg, ensure_ascii=False, indent=1))
            z.writestr("manifest.json", json.dumps(self.manifest(), ensure_ascii=False, indent=1, default=str))
            z.writestr("events.txt", self.log.events_text())
            z.writestr("events.jsonl", self.log.events_jsonl())
            z.writestr("timeseries.csv", self.log.timeseries_csv())
            z.writestr("apple.csv", self.log.apple_csv())
            for doc in ("ASSUMPTIONS.md", "KNOWN_ISSUES.md"):
                p = ROOT / doc
                if p.exists():
                    z.writestr(doc, p.read_text())
            z.writestr("README.txt", LOG_README)
        return buf.getvalue()


LOG_README = """Fly-Sim — логи прогона

events.txt / events.jsonl  — что происходило («произошло X»): где муха, начала/перестала есть,
                             жар, запахи, контакты, песня, спаривание, яйца, сигналы награды и
                             наказания, изменения памяти, рождение и смерть.
timeseries.csv             — каждые 0.1 с для каждой мухи: положение, режим, голод, энергия, зоб,
                             здоровье, температура, запах/вкус, частоты ключевых нейронов,
                             PPL1 (наказание), PAM (награда), valence_proxy = PAM − PPL1,
                             сила памяти KC→MBON, число активных нейронов.
apple.csv                  — состояние яблока: сахар, спирт, уксус, дрожжи, бактерии, плесень,
                             запахи, яйца/личинки/куколки.
config.json                — условия опыта.
manifest.json              — версия модели, ID эксперимента, режим механизмов
                             (connectome_only / connectome_plus_approximations /
                             experimental_shortcuts), какие обходы и приближения
                             использовались, полнота нервной системы каждой мухи.

ВАЖНО: valence_proxy, PPL1, PAM — функциональные нейронные корреляты ценности стимула.
Они не являются измерением субъективного переживания.
"""


class App:
    def __init__(self):
        self.sim = None
        self.thread = None

    def start(self, cfg):
        self.stop()
        full = defaults()
        for g, items in (cfg or {}).items():
            if g in full:
                full[g].update({k: v for k, v in items.items() if k in full[g]})
        self.sim = Simulation(full)
        self.thread = threading.Thread(target=self.sim.run, daemon=True)
        self.thread.start()

    def stop(self):
        if self.sim:
            self.sim.stop_flag = True
            if self.thread:
                self.thread.join(timeout=5)


_BRAINS = {}


def brain_static(sex):
    if sex not in _BRAINS:
        _, ann = load(sex)
        _BRAINS[sex] = {"n": int(len(ann)), "xyz": b64(_positions(ann)), "region": b64(_regions(ann))}
    return _BRAINS[sex]


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype="application/json", extra=None):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj):
            return self._send(json.dumps(obj, ensure_ascii=False, default=float))

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/start":
                threading.Thread(target=app.start, args=(body,), daemon=True).start()
                return self._json({"ok": True})
            if u.path == "/api/stop":
                app.stop()
                return self._json({"ok": True})
            if u.path == "/api/pause" and app.sim:
                app.sim.paused = not app.sim.paused
                return self._json({"paused": app.sim.paused})
            self.send_response(404)
            self.end_headers()

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            sim = app.sim
            if u.path in ("/", "/index.html"):
                return self._send(PAGE.read_text(), "text/html; charset=utf-8")
            if u.path == "/api/schema":
                return self._json({"schema": SCHEMA, "apple_norms": APPLE_NORMS,
                                   "region_names": REGION_NAMES, "region_colors": REGION_COLORS,
                                   "frame_dt": FRAME_DT})
            if u.path == "/api/state":
                return self._json({"running": bool(sim and not sim.stop_flag and not sim.error),
                                   "status": sim.status if sim else "не запущено",
                                   "error": sim.error if sim else None,
                                   "config": sim.cfg if sim else defaults(),
                                   "manifest": sim.manifest() if sim and sim.world.flies else None})
            if u.path.startswith("/api/brain/"):
                return self._json(brain_static(u.path.rsplit("/", 1)[1]))
            if u.path == "/api/frames":
                if not sim:
                    return self._json({"frames": []})
                since = int(q.get("since", ["0"])[0])
                with sim.lock:
                    base = sim.frames[0]["i"] if sim.frames else 0
                    out = sim.frames[max(0, since - base):][:30]
                return self._json({"frames": out})
            if u.path.startswith("/api/logs/") and sim:
                name = u.path.rsplit("/", 1)[1]
                stamp = time.strftime("%Y%m%d-%H%M")
                if name == "all.zip":
                    return self._send(sim.logs_zip(), "application/zip",
                                      {"Content-Disposition": f'attachment; filename="flysim-logs-{stamp}.zip"'})
                getters = {"events.txt": sim.log.events_text, "events.jsonl": sim.log.events_jsonl,
                           "timeseries.csv": sim.log.timeseries_csv, "apple.csv": sim.log.apple_csv}
                if name in getters:
                    return self._send(getters[name](), "text/plain; charset=utf-8",
                                      {"Content-Disposition": f'attachment; filename="{stamp}-{name}"'})
            self.send_response(404)
            self.end_headers()

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    a = ap.parse_args()
    app = App()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(app))
    print(f"Fly-Sim: откройте http://localhost:{a.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
