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
    "world": {
        "size_m": {"label": "Размер мира (сторона)", "unit": "м", "min": 0.1, "max": 5.0, "step": 0.05,
                   "default": 0.3, "norm": "садок ~0.2–0.3 м, комната 3–5 м; в неподвижном воздухе "
                                         "запахи слабеют как R/r — в большом мире мухи находят еду дольше"},
        "seasonal": {"label": "Смена сезонов (умеренный климат)", "type": "bool", "default": False,
                     "norm": "среднее 10 °C ± 12 °C за год; зимой мухи в холодовой коме, семена проходят стратификацию"},
        "soil": {"label": "Почва под яблоками", "type": "bool", "default": False,
                 "norm": "без почвы семена не прорастут"},
        "moist": {"label": "Влажность почвы", "type": "bool", "default": False,
                  "norm": "семенам нужна влага и для стратификации, и для прорастания"},
    },
    "apples": {
        "list": {"label": "Яблоки", "type": "apples",
                 "default": [{"age_days": 2.0, "cut": True, "radius_mm": 30.0, "x_mm": 0.0, "y_mm": 0.0}],
                 "norm": "надрезанное: 1–3 дн брожение, 4–5 дн спирт, 7+ дн уксус и плесень; "
                         "целое бродит и плесневеет неделями; реальное яблоко ~35–40 мм"},
    },
    "flies": {
        "males": {"label": "Самцы", "unit": "", "min": 0, "max": 10, "step": 1, "default": 1,
                  "norm": "Male CNS, 164 587 нейронов каждый"},
        "females": {"label": "Самки", "unit": "", "min": 0, "max": 10, "step": 1, "default": 1,
                    "norm": "FlyWire, 138 639 нейронов каждая"},
        "starved_h": {"label": "Голодали перед стартом", "unit": "ч", "min": 0.0, "max": 48.0,
                      "step": 1.0, "default": 18.0, "norm": "кормовые опыты: 18–24 ч"},
        "virgin": {"label": "Самки девственницы", "type": "bool", "default": True,
                   "norm": "спаренная самка не готова к спариванию и откладывает яйца"},
        "female_connectome": {"label": "Нервная система самки", "type": "select",
                              "options": {"female": "FlyWire v783 — мозг (проверен)",
                                          "female_banc": "BANC — мозг + VNC (экспериментально)"},
                              "default": "female",
                              "norm": "FlyWire без брюшной нервной цепочки; BANC пока не эквивалентен (KNOWN_ISSUES #6)"},
    },
    "injuries": {
        **{leg: {"label": f"Нога: {name}", "type": "select",
                 "options": {"intact": "цела", "paralyzed": "парализована", "amputated": "отсутствует"},
                 "default": "intact", "norm": ""}
           for leg, name in (("lf", "левая передняя"), ("lm", "левая средняя"), ("lh", "левая задняя"),
                             ("rf", "правая передняя"), ("rm", "правая средняя"), ("rh", "правая задняя"))},
        "wing_left": {"label": "Повреждение левого крыла", "unit": "%", "min": 0, "max": 100, "step": 5,
                      "default": 0, "norm": "снижает подъёмную силу; асимметрия разворачивает в полёте"},
        "wing_right": {"label": "Повреждение правого крыла", "unit": "%", "min": 0, "max": 100, "step": 5,
                       "default": 0, "norm": ""},
        "antenna_left": {"label": "Левый усик удалён", "type": "bool", "default": False,
                         "norm": "молчат обоняние, слух, термосенсоры этой стороны"},
        "antenna_right": {"label": "Правый усик удалён", "type": "bool", "default": False, "norm": ""},
        "eye_left": {"label": "Левый глаз закрыт", "type": "bool", "default": False, "norm": ""},
        "eye_right": {"label": "Правый глаз закрыт", "type": "bool", "default": False, "norm": ""},
        "apply_to": {"label": "Кому", "type": "select",
                     "options": {"all": "всем мухам", "first_male": "первому самцу",
                                 "first_female": "первой самке"}, "default": "all",
                     "norm": "травму задаёт экспериментатор; организм о ней не «знает»"},
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
        "us_shortcut": {"label": "Наказание → дофамин PPL1 по данным опытов (A24)", "type": "bool",
                        "default": True, "norm": "выкл. — только путь по коннектому"},
        "reward_shortcut": {"label": "Сахар → дофамин PAM по данным опытов (A25)", "type": "bool",
                            "default": True, "norm": "выкл. — только путь по коннектому"},
        "spontaneous": {"label": "Спонтанная ходьба (A30)", "type": "bool", "default": True,
                        "norm": "выкл. — двигается только по командам мозга"},
    },
    "run": {
        "session_s": {"label": "Сессия (нейрон-в-нейрон)", "unit": "с", "min": 5, "max": 600,
                      "step": 5, "default": 60, "norm": "~×0.1 от реального времени на M2"},
        "ff_hours": {"label": "Перемотка между сессиями", "unit": "ч", "min": 0, "max": 72,
                     "step": 1, "default": 24, "norm": "0 — без перемотки"},
        "limit_days": {"label": "Лимит мирового времени", "unit": "дн", "min": 0, "max": 3650, "step": 1,
                       "default": 0, "norm": "0 — без лимита; симуляция остановится сама, когда мир проживёт столько"},
        "limit_compute_h": {"label": "Лимит времени счёта", "unit": "ч", "min": 0, "max": 72, "step": 0.5,
                            "default": 0, "norm": "0 — без лимита; сколько реальных часов считать"},
        "max_adults": {"label": "Макс. мух с мозгом", "unit": "", "min": 1, "max": 30, "step": 1,
                       "default": 4, "norm": "~0.4 ГБ памяти и ~10 с счёта на 1 с жизни на муху; "
                                             "на Mac 8 ГБ разумно ≤4–6"},
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
        env, fl, hz, sci, run = (cfg[k] for k in ("environment", "flies", "hot_zone", "science", "run"))
        wd = cfg["world"]
        from flysim.world.plant import PlantWorld
        from flysim.world.atmosphere import Atmosphere
        half = float(wd["size_m"]) * 1000 / 2
        apples = []
        for a in (cfg["apples"]["list"] or [])[:12]:
            r = float(a.get("radius_mm", 30.0))
            x = float(np.clip(a.get("x_mm", 0.0), -half + r, half - r))
            y = float(np.clip(a.get("y_mm", 0.0), -half + r, half - r))
            apples.append(Apple(AppleParams(radius_mm=r), center=(x, y), cut=bool(a.get("cut", True)),
                                temp_c=float(env["temp_c"]), o2_pct=float(env["o2_pct"])))
        if not apples:
            apples = [Apple(AppleParams(radius_mm=30.0))]
        apple = apples[0]
        ages = [float(a.get("age_days", 0.0)) for a in (cfg["apples"]["list"] or [])] or [0.0]
        atm = Atmosphere(volume_l=float(env["volume_l"]), ach=float(env["ach"]),
                         co2_ppm=float(env["co2_ppm"]), o2_pct=float(env["o2_pct"]),
                         outside_co2_ppm=float(env["co2_ppm"]), outside_o2_pct=float(env["o2_pct"]))
        self.world = World(apple, apples=apples, arena=half, ambient_c=float(env["temp_c"]),
                           o2_pct=float(env["o2_pct"]), co2_ppm=float(env["co2_ppm"]), atm=atm,
                           seasonal=bool(wd["seasonal"]), base_temp_c=float(env["temp_c"]),
                           plants=PlantWorld(soil=bool(wd["soil"]), moist=bool(wd["moist"])))
        for k in range(len(apples)):
            self.world.plants.add_apple_seeds(k)
        # each apple has been lying here for its own age before the flies arrive
        for a, age in zip(apples, ages):
            a.age(age * 24)
        self.world.sync_air()
        self.world.clock_h = 0.0
        self.pending = []   # requests from the web page (add fly), applied between steps
        self.world.population = Population(max_adults=int(run["max_adults"]))
        if hz["enabled"]:
            self.world.zones.append(Zone(center=(-35.0, 25.0), radius=float(hz["radius_mm"]),
                                         temp_c=float(hz["temp_c"])))
        self.log = EventLog()
        from flysim.sim.chronicle import Chronicle
        self.chron = Chronicle(meta={"title": "Свободная симуляция", "config": cfg})
        self.run_id = self.chron.run_id
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
        self._spawn_near = apple.center
        for sex, n in (("male", int(fl["males"])), ("female", int(fl["females"]))):
            for _ in range(n):
                ang = np.pi + 0.5 * k - 0.4
                pos = (apple.center[0] + np.cos(ang) * (r + 12), apple.center[1] + np.sin(ang) * (r + 12), 0.0)
                f = self.add_fly(sex, pos, ang + np.pi, float(fl["starved_h"]), calibrate=False)
                if sex == "female" and not fl["virgin"]:
                    f.set_mated("(до опыта)")
                k += 1
        self.world.rebuild_physics()

    def _injuries_for(self, sex, k):
        inj = self.cfg.get("injuries", {})
        who = inj.get("apply_to", "all")
        if who == "first_male" and not (sex == "male" and k == 1):
            return None
        if who == "first_female" and not (sex == "female" and k == 1):
            return None
        return {"legs": {l: inj.get(l, "intact") for l in ("lf", "lm", "lh", "rf", "rm", "rh")},
                "wing_damage": {"left": inj.get("wing_left", 0) / 100, "right": inj.get("wing_right", 0) / 100},
                "antenna": {sd: "removed" if inj.get(f"antenna_{sd}") else "intact" for sd in ("left", "right")},
                "eye": {sd: "covered" if inj.get(f"eye_{sd}") else "open" for sd in ("left", "right")}}

    def add_fly(self, sex, pos, yaw, starved_h, note="", calibrate=True):
        sci = self.cfg["science"]
        self.n_made[sex] += 1
        name = f"{'♂' if sex == 'male' else '♀'}{self.n_made[sex]}"
        f = Fly(FlyConfig(name, sex, starved_h=starved_h, seed=self.n_made[sex] * 7 + (sex == "male"),
                          plasticity=bool(sci["plasticity"]), us_punish_shortcut=bool(sci["us_shortcut"]),
                          us_reward_shortcut=bool(sci.get("reward_shortcut", True)),
                          spontaneous=bool(sci["spontaneous"]),
                          injuries=self._injuries_for(sex, self.n_made[sex]),
                          connectome=(self.cfg["flies"].get("female_connectome", "female")
                                      if sex == "female" else ""),
                          body="nmf"), pos=pos, yaw=yaw)
        if note:
            f.events.append((round(f.t, 3), note))
        self.world.flies.append(f)
        return f

    def snapshot(self, t_session):
        from flysim.sim.chronicle import make_frame
        w = self.world
        counts = {f.cfg.name: f.pop_counts() for f in w.flies}
        n_ev = len(self.log.events)
        self.log.frame_idx = self.chron.n
        self.log.frame(w, t_session, counts)
        self.chron.note_events(self.log.events[n_ev:])
        fr = make_frame(w, t_session, counts, self.log, self.status, index=self.chron.n)
        fr["run_id"] = self.run_id
        self.chron.add(dict(fr))
        with self.lock:
            self.frames.append(fr)
            if len(self.frames) > 600:
                self.frames = self.frames[-600:]

    def fast_forward(self, hours):
        w = self.world
        self.status = f"перемотка {hours:.0f} ч: яблоко бродит и плесневеет, яйца развиваются"
        pop = w.population
        t0 = w.clock_h
        # the world and every fly live through the interval hour by hour (air, fruit, plants,
        # metabolism, ageing and death happen together)
        left = hours
        while left > 1e-9 and w.alive():
            h = min(1.0, left)
            w.advance_hours(h)
            for f in w.flies:
                f.pass_time(h, w, fed_from_fruit=True)
            w.bury_dead()
            left -= h
        if left > 1e-9:            # no flies and no brood: let the rest of the world go on
            w.advance_hours(left)
        ready = pop.fast_forward(t0, hours, w.apples, w.flies)
        if w.plants is not None:
            for te, text in w.plants.events:
                self.log.events.append({"id": len(self.log.events), "clock_h": round(te, 4), "t": 0.0,
                                        "session": self.log.session, "frame": self.chron.n, "fly": "мир",
                                        "sex": "", "type": "plant", "cat": "world", "imp": 2,
                                        "text": text, "others": [], "pos": None, "data": {}})
            w.plants.events.clear()
        self.log.world_events(w, 0.0)
        for b in ready:
            b.alive = False
            pop.born += 1  # eclosed adults (with or without a simulated brain)
            if len(w.flies) >= pop.max_adults:
                continue
            sex = "female" if pop.rng.random() < 0.5 else "male"
            ang = pop.rng.uniform(0, 2 * np.pi)
            ap = w.apples[min(b.apple_idx, len(w.apples) - 1)]
            rr = ap.p.radius_mm + 8
            f = self.add_fly(sex, (ap.center[0] + rr * np.cos(ang), ap.center[1] + rr * np.sin(ang), 0.0),
                             ang + np.pi, 6.0,
                             note=f"вылупилась (мать {b.mother}, отец {b.father})")
            f.age_d = 0.0
            self.log._ev(w, f, 0.0, "birth", f"новая муха {f.cfg.name} с собственным мозгом")

    def _apply_pending(self):
        added = False
        while self.pending:
            req = self.pending.pop(0)
            if req.get("type") == "add_fly":
                w = self.world
                if len(w.flies) >= int(self.cfg["run"]["max_adults"]):
                    self.log._ev(w, w.flies[0], 0.0, "info", "лимит мух с мозгом достигнут") if w.flies else None
                    continue
                ap = w.nearest_apple(np.zeros(3))
                ang = np.random.default_rng().uniform(0, 2 * np.pi)
                rr = ap.p.radius_mm + 10
                f = self.add_fly(req.get("sex", "female"),
                                 (ap.center[0] + rr * np.cos(ang), ap.center[1] + rr * np.sin(ang), 0.0),
                                 ang + np.pi, float(self.cfg["flies"]["starved_h"]),
                                 note="добавлена экспериментатором")
                added = True
        if added:
            self.world.rebuild_physics()

    def limit_reached(self) -> str | None:
        run = self.cfg["run"]
        if float(run.get("limit_days", 0)) > 0 and self.world.clock_h >= float(run["limit_days"]) * 24:
            return f"достигнут лимит мирового времени ({run['limit_days']} дн)"
        if float(run.get("limit_compute_h", 0)) > 0 and \
                time.time() - self.started >= float(run["limit_compute_h"]) * 3600:
            return f"достигнут лимит времени счёта ({run['limit_compute_h']} ч)"
        return None

    def run(self):
        try:
            run = self.cfg["run"]
            session = 0
            while not self.stop_flag:
                lim = self.limit_reached()
                if lim:
                    self.status = lim
                    break
                session += 1
                self.log.session = session
                self.status = f"сессия {session}: считаются все нейроны"
                t, k = 0.0, 0
                per = int(round(FRAME_DT / 0.01))
                while t < float(run["session_s"]) and not self.stop_flag:
                    if self.pending:
                        self._apply_pending()
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
                    if getattr(self.world, "physics_dirty", False) or True:
                        self.world.physics_dirty = False
                        self.world.rebuild_physics()
                    self.snapshot(0.0)
                if not self.world.alive():
                    self.status = "жизнь в мире прекратилась: не осталось ни мух, ни потомства"
                    self.snapshot(0.0)
                    break
            if not self.limit_reached():
                self.status = "остановлено"
        except Exception as e:  # surface errors in the UI instead of dying silently
            self.error = f"{e}\n{traceback.format_exc()}"
            self.status = "ошибка: " + str(e)
        finally:
            try:
                self.snapshot(0.0)
                self.chron.save_logs(self.log, self.manifest())
            except Exception:
                pass

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
            if u.path == "/api/add_fly" and app.sim:
                app.sim.pending.append({"type": "add_fly", "sex": body.get("sex", "female")})
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
            if u.path == "/api/runs":
                from flysim.sim.chronicle import list_runs
                return self._json({"current": sim.run_id if sim else None, "runs": list_runs()})
            if u.path.startswith("/api/run/"):
                parts = u.path.split("/")      # /api/run/<id>/<what>
                rid, what = parts[3], parts[4] if len(parts) > 4 else "info"
                src = sim.chron if (sim and rid == sim.run_id) else None
                if src is None:
                    from flysim.sim.chronicle import RUNS, SavedRun
                    if not (RUNS / rid / "frames.jsonl").exists():
                        self.send_response(404)
                        self.end_headers()
                        return
                    src = SavedRun(RUNS / rid)
                    events = src.events()
                else:
                    events = sim.log.events
                if what == "frame":
                    return self._json(src.read(int(q.get("i", ["0"])[0])) or {})
                if what == "events":
                    mi = int(q.get("min_imp", ["1"])[0])
                    cat = q.get("cat", [""])[0]
                    fly = q.get("fly", [""])[0]
                    ev = [e for e in events if e.get("imp", 1) >= mi and (not cat or e.get("cat") == cat)
                          and (not fly or e.get("fly") == fly or fly in e.get("others", []))]
                    return self._json({"events": ev[-3000:]})
                if what == "info":
                    from flysim.sim.eventlog import CATEGORY_RU
                    cnt, flies, bio = {}, {}, {}
                    for e in events:
                        cnt[e.get("cat", "info")] = cnt.get(e.get("cat", "info"), 0) + 1
                        if e.get("fly") and e["fly"] != "мир":
                            flies[e["fly"]] = e.get("sex", "")
                            if e.get("imp", 1) >= 2:
                                bio.setdefault(e["fly"], []).append(e)
                    first = src.read(0) or {}
                    last = src.read(src.n - 1) or {}
                    return self._json({"run_id": rid, "n_frames": src.n, "counts": cnt,
                                       "category_names": CATEGORY_RU, "flies": flies,
                                       "biographies": {k: v[-80:] for k, v in bio.items()},
                                       "clock_h": [first.get("clock_h", 0), last.get("clock_h", 0)],
                                       "live": bool(sim and rid == sim.run_id and not sim.stop_flag)})
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
