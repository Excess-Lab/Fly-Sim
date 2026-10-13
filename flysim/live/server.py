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
from urllib.parse import parse_qs, unquote, urlparse

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
        "climate": {"label": "Климат", "type": "select",
                    "options": {"container": "контейнер: постоянная температура и свет",
                                "garden": "сад: день и ночь, погода, переменчивый ветер"},
                    "default": "container",
                    "norm": "сад — умеренный пояс 50° с. ш. (A98): солнце и освещённость по часу и дню года, "
                            "суточный ход температуры, смена погоды, ветер у земли с порывами"},
        "start_hour": {"label": "Время суток на старте", "unit": "ч", "min": 0, "max": 23.5, "step": 0.5,
                       "default": 8.0, "norm": "только для сада"},
        "seasonal": {"label": "Смена сезонов (умеренный климат)", "type": "bool", "default": False,
                     "norm": "среднее 10 °C ± 12 °C за год; зимой мухи в холодовой коме, семена проходят стратификацию"},
        "soil": {"label": "Почва под яблоками", "type": "bool", "default": False,
                 "norm": "без почвы семена не прорастут"},
        "moist": {"label": "Влажность почвы", "type": "bool", "default": False,
                  "norm": "семенам нужна влага; влажная почва дышит сильнее (потребляет O₂, выделяет CO₂)"},
        "wind_speed": {"label": "Ветер в контейнере", "unit": "м/с", "min": 0.0, "max": 3.0, "step": 0.05, "default": 0.0,
                       "norm": "0 — неподвижный воздух (садок); сквозняк ~0.1–0.5; мухи летают при ≲2–3 м/с. В саду ветер задаёт погода"},
        "garden_wind": {"label": "Сила ветра в саду", "unit": "×", "min": 0.0, "max": 3.0, "step": 0.1, "default": 1.0,
                        "norm": "1 — как в природе у земли (~0.3–0.6 м/с в ясную погоду, >1 м/с в грозу); 0 — безветрие"},
        "wind_dir_deg": {"label": "Ветер дует в сторону", "unit": "°", "min": 0, "max": 359, "step": 5,
                         "default": 0, "norm": "0° = вдоль оси X"},
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
                    "norm": "BANC: мозг + брюшная нервная цепочка, 181 627 нейронов каждая"},
        "starved_h": {"label": "Голодали перед стартом", "unit": "ч", "min": 0.0, "max": 48.0,
                      "step": 1.0, "default": 18.0, "norm": "кормовые опыты: 18–24 ч"},
        "virgin": {"label": "Самки девственницы", "type": "bool", "default": True,
                   "norm": "спаренная самка не готова к спариванию и откладывает яйца"},
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
                   "step": 0.5, "default": 44.0, "norm": "ноцицепторы лапок отвечают выше ~40 °C (A80); 44 °C — болезненно, как на горячей пластине"},
        "radius_mm": {"label": "Радиус зоны", "unit": "мм", "min": 5.0, "max": 30.0, "step": 1.0,
                      "default": 15.0, "norm": ""},
    },
    "science": {
        "plasticity": {"label": "Обучение (пластичность KC→MBON)", "type": "bool", "default": True,
                       "norm": "выкл. — контроль «только рефлексы»"},
        "us_shortcut": {"label": "Наказание → дофамин PPL1 по данным опытов (A24)", "type": "bool",
                        "default": True, "norm": "действует только для ударов тока; жар идёт через ноцицепторы ног по коннектому"},
        "reward_shortcut": {"label": "Сахар → дофамин PAM по данным опытов (A25)", "type": "bool",
                            "default": True, "norm": "выкл. — только путь по коннектому"},
        "spontaneous": {"label": "Спонтанная ходьба (A30)", "type": "bool", "default": True,
                        "norm": "выкл. — двигается только по командам мозга"},
    },
    "run": {
        "session_s": {"label": "Длина подробного отрезка", "unit": "с", "min": 5, "max": 600,
                      "step": 5, "default": 60, "norm": "сколько секунд жизни считать полностью: каждый нейрон, каждая мышца, физика тела. Потом мир перематывается"},
        "ff_hours": {"label": "Перемотка между отрезками", "unit": "ч", "min": 0, "max": 72,
                     "step": 1, "default": 24, "norm": "часы мира, которые проходят быстро: яблоки гниют, мухи голодают, стареют, развиваются яйца. 0 — без перемотки"},
        "limit_days": {"label": "Лимит мирового времени", "unit": "дн", "min": 0, "max": 3650, "step": 1,
                       "default": 0, "norm": "0 — без лимита; симуляция остановится сама, когда мир проживёт столько"},
        "limit_compute_h": {"label": "Лимит времени счёта", "unit": "ч", "min": 0, "max": 72, "step": 0.5,
                            "default": 0, "norm": "0 — без лимита; сколько реальных часов считать"},
        "max_adults": {"label": "Предел взрослых мух", "unit": "", "min": 1, "max": 30, "step": 1,
                       "default": 4, "norm": "у каждой взрослой мухи свой полный мозг (~0.5 ГБ памяти, ~10 с счёта на 1 с жизни). "
                                             "Мухи, вылупившиеся сверх предела, только учитываются в численности, без мозга и тела. 8 ГБ: ≤4–6"},
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
                           plants=PlantWorld(soil=bool(wd["soil"]), moist=bool(wd["moist"])),
                           wind=(float(wd.get("wind_speed", 0)) * np.cos(np.radians(float(wd.get("wind_dir_deg", 0)))),
                                 float(wd.get("wind_speed", 0)) * np.sin(np.radians(float(wd.get("wind_dir_deg", 0))))))
        if wd.get("climate") == "garden":
            from flysim.world.weather import Weather
            self.world.weather = Weather(seed=int(cfg.get("seed", 0) or 0), seasonal=bool(wd["seasonal"]),
                                         base_temp_c=float(env["temp_c"]),
                                         start_hour=float(wd.get("start_hour", 8.0)),
                                         wind_scale=float(wd.get("garden_wind", 1.0)))
            self.world.apply_weather()
        for k in range(len(apples)):
            self.world.plants.add_apple_seeds(k)
        # each apple has been lying here for its own age before the flies arrive
        for a, age in zip(apples, ages):
            a.age(age * 24)
        self.world.sync_air()
        self.world.clock_h = 0.0
        self.pending = []   # requests from the web page (add fly / apple), applied between steps
        self.passports = {}  # every fly that ever lived in this run (name -> Fly)
        self.title = "Свободная симуляция"
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
                          body="nmf"), pos=pos, yaw=yaw)
        if note:
            f.events.append((round(f.t, 3), note))
        import uuid
        f.uid = f"FLY-{'M' if sex == 'male' else 'F'}-{uuid.uuid4().hex[:10].upper()}"
        f.born_clock_h = self.world.clock_h
        f.parents = None
        f.note = note
        self.world.flies.append(f)
        self.passports[f.cfg.name] = f
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
            f.parents = {"mother": b.mother, "father": b.father}
            f.age_d = 0.0
            self.log._ev(w, f, 0.0, "birth", f"новая муха {f.cfg.name} с собственным мозгом")

    def _apply_pending(self):
        added = False
        while self.pending:
            req = self.pending.pop(0)
            if req.get("type") == "add_apple":
                w = self.world
                r = float(req.get("radius_mm", 30.0))
                x = float(np.clip(req.get("x_mm", 0.0), -w.arena + r, w.arena - r))
                y = float(np.clip(req.get("y_mm", 0.0), -w.arena + r, w.arena - r))
                a = Apple(AppleParams(radius_mm=r), center=(x, y), cut=bool(req.get("cut", True)),
                          temp_c=w.ambient_c, o2_pct=w.o2_pct)
                a.age(float(req.get("age_days", 0.0)) * 24)
                w.apples.append(a)
                if w.plants is not None:
                    w.plants.add_apple_seeds(len(w.apples) - 1)
                self.log.events.append({"id": len(self.log.events), "clock_h": round(w.clock_h, 4),
                                        "t": 0.0, "session": self.log.session, "frame": self.chron.n,
                                        "fly": "мир", "sex": "", "type": "apple", "cat": "world", "imp": 2,
                                        "text": f"экспериментатор положил яблоко {len(w.apples)} "
                                                f"({a.summary()})", "others": [], "pos": [x, y, r], "data": {}})
                added = True
                continue
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
            self.stop_flag = True
            try:
                self.snapshot(0.0)
                self.chron.save_logs(self.log, self.manifest())
            except Exception:
                pass

    def passport(self, name: str) -> dict | None:
        f = self.passports.get(name)
        if f is None:
            return None
        from flysim.sim.provenance import fly_provenance
        dead = next((d for d in self.world.deaths if d["name"] == name), None)
        ev = [e for e in self.log.events if e.get("fly") == name or name in e.get("others", [])]
        cnt = {}
        for e in ev:
            cnt[e.get("cat", "info")] = cnt.get(e.get("cat", "info"), 0) + 1
        prov = fly_provenance(f)
        return {"uid": getattr(f, "uid", ""), "name": name, "sex": f.cfg.sex,
                "status": f"погибла: {dead['cause']}" if dead else "жива",
                "born_world_h": round(getattr(f, "born_clock_h", 0.0), 2),
                "age_d": round(f.age_d, 2), "parents": getattr(f, "parents", None),
                "origin": getattr(f, "note", "") or "создана при запуске",
                "connectome": prov["connectome_coverage"], "mode": prov["mode"],
                "shortcuts": prov["shortcuts"], "approximations": prov["approximations"],
                "seed": f.cfg.seed, "plasticity": f.cfg.plasticity,
                "body": f.schema.summary(), "mated": f.mated, "mate": f.mate,
                "hunger": round(f.meta.hunger, 3), "energy": round(f.meta.energy, 3),
                "health": round(f.meta.health, 3), "bio_age_d": round(f.mortality.bio_age_d, 2),
                "tolerances": {"голод_ч": round(f.mortality.starve_h, 1),
                               "перегрев_мин": round(f.mortality.heat_min, 1),
                               "холод_ч": round(f.mortality.cold_h, 1),
                               "аноксия_ч": round(f.mortality.anoxia_h, 1)},
                "event_counts": cnt, "life": [e for e in ev if e.get("imp", 1) >= 2][-200:]}

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


class ExperimentRun:
    """A prepared experiment protocol running in the background, observable like a simulation."""

    def __init__(self, protocol: str, group: str, seed: int):
        from flysim.sim.protocol import PROTOCOLS
        self.protocol, self.group, self.seed = protocol, group, seed
        self.title = f"Эксперимент: {protocol} / {group} / seed {seed}"
        self.status, self.error, self.result = "подготовка", None, None
        self.stop_flag, self.paused = False, False
        self.recorder = None
        self.spec = PROTOCOLS[protocol]
        self.run_id = None
        self.started = time.time()

    def run(self):
        from flysim.sim import experiments as E
        from flysim.sim import record
        orig = record.Recorder.__init__
        me = self

        def hook(rec, *a, **k):
            orig(rec, *a, **k)
            me.recorder = rec
            me.run_id = rec.chron.run_id
        record.Recorder.__init__ = hook
        try:
            self.status = "идёт эксперимент"
            fn = getattr(E, self.spec["function"])
            self.result = fn(seed=self.seed, **self.spec["groups"][self.group], **self.spec["kwargs"])
            self.status = "завершён"
        except Exception as e:
            self.error = f"{e}\n{traceback.format_exc()}"
            self.status = "ошибка: " + str(e)
        finally:
            record.Recorder.__init__ = orig
            self.stop_flag = True

    @property
    def chron(self):
        return self.recorder.chron if self.recorder else None

    @property
    def log(self):
        return self.recorder.log if self.recorder else None

    @property
    def world(self):
        return self.recorder.world if self.recorder else None


class App:
    def __init__(self):
        self.sims = {}     # run_id -> Simulation | ExperimentRun
        self.threads = {}

    def start(self, cfg):
        full = defaults()
        for g, items in (cfg or {}).items():
            if g in full:
                full[g].update({k: v for k, v in items.items() if k in full[g]})
        sim = Simulation(full)
        if cfg.get("title"):
            sim.title = str(cfg["title"])[:80]
        self.sims[sim.run_id] = sim
        th = threading.Thread(target=sim.run, daemon=True)
        th.start()
        self.threads[sim.run_id] = th
        return sim.run_id

    def delete(self, rid):
        """Remove a finished observation and its saved files (results/runs/<id>)."""
        import re
        import shutil
        from flysim.sim.chronicle import RUNS
        if not re.fullmatch(r"[\w.\-]+", rid or ""):
            return False, "неверный id"
        sim = self.sims.get(rid)
        th = self.threads.get(rid)
        if sim is not None and th is not None and th.is_alive():
            return False, "наблюдение ещё идёт — сначала остановите его"
        self.sims.pop(rid, None)
        self.threads.pop(rid, None)
        d = (RUNS / rid).resolve()
        if d.parent == RUNS.resolve() and d.is_dir():
            shutil.rmtree(d)
        return True, ""

    def start_experiment(self, protocol, group, seed):
        ex = ExperimentRun(protocol, group, int(seed))
        th = threading.Thread(target=ex.run, daemon=True)
        th.start()
        for _ in range(600):           # wait until the recorder (and run id) exists
            if ex.run_id or ex.error:
                break
            time.sleep(0.1)
        rid = ex.run_id or f"exp-{int(time.time())}"
        ex.run_id = rid
        self.sims[rid] = ex
        self.threads[rid] = th
        return rid

    def get(self, rid):
        return self.sims.get(rid)

    def stop(self, rid=None):
        for k, sim in list(self.sims.items()):
            if rid is None or k == rid:
                sim.stop_flag = True

    @property
    def sim(self):          # most recent free simulation (backwards compatible endpoints)
        for sim in reversed(list(self.sims.values())):
            if isinstance(sim, Simulation):
                return sim
        return None


_BRAINS = {}


def brain_static(sex):
    if sex not in _BRAINS:
        _, ann = load(sex)
        _BRAINS[sex] = {"n": int(len(ann)), "xyz": b64(_positions(ann)), "region": b64(_regions(ann))}
    return _BRAINS[sex]


def _source(app, rid):
    """(chronicle-like reader, events list, live object or None) for a run id."""
    sim = app.get(rid)
    if sim is not None and sim.chron is not None:
        return sim.chron, (sim.log.events if sim.log else []), sim
    from flysim.sim.chronicle import RUNS, SavedRun
    if (RUNS / rid / "frames.jsonl").exists():
        sr = SavedRun(RUNS / rid)
        return sr, sr.events(), None
    return None, [], None


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, ctype="application/json", extra=None, code=200):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj, code=200):
            return self._send(json.dumps(obj, ensure_ascii=False, default=float), code=code)

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            parts = [unquote(x) for x in u.path.strip("/").split("/")]
            if u.path == "/api/start":
                return self._json({"run_id": app.start(body)})
            if u.path == "/api/experiment":
                return self._json({"run_id": app.start_experiment(body["protocol"], body["group"],
                                                                  body.get("seed", 100))})
            if len(parts) == 4 and parts[:2] == ["api", "run"] and parts[3] == "delete":
                ok, msg = app.delete(parts[2])
                return self._json({"ok": ok, "error": None if ok else msg}, 200 if ok else 409)
            if len(parts) == 4 and parts[:2] == ["api", "sim"]:
                sim = app.get(parts[2])
                if sim is None:
                    return self._json({"error": "нет такого наблюдения"}, 404)
                what = parts[3]
                if what == "stop":
                    sim.stop_flag = True
                    return self._json({"ok": True})
                if what == "pause":
                    sim.paused = not sim.paused
                    return self._json({"paused": sim.paused})
                if what in ("add_fly", "add_apple") and isinstance(sim, Simulation):
                    sim.pending.append({"type": what, **body})
                    return self._json({"ok": True})
            return self._json({"error": "не найдено"}, 404)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            parts = [unquote(x) for x in u.path.strip("/").split("/")]
            if u.path in ("/", "/index.html"):
                return self._send(PAGE.read_text(), "text/html; charset=utf-8")
            if u.path.startswith("/static/"):
                root = (PAGE.parent / "static").resolve()
                f = (PAGE.parent / u.path.lstrip("/")).resolve()
                if root not in f.parents or not f.is_file():
                    return self._json({"error": "не найдено"}, 404)
                ctype = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                         ".svg": "image/svg+xml"}.get(f.suffix, "application/octet-stream")
                return self._send(f.read_bytes(), ctype)
            if u.path == "/api/schema":
                from flysim.sim.protocol import PROTOCOLS
                return self._json({"schema": SCHEMA, "apple_norms": APPLE_NORMS,
                                   "region_names": REGION_NAMES, "region_colors": REGION_COLORS,
                                   "frame_dt": FRAME_DT, "defaults": defaults(),
                                   "protocols": {k: {"hypothesis": v["hypothesis"], "groups": list(v["groups"]),
                                                     "reference": v["reference"]} for k, v in PROTOCOLS.items()}})
            if u.path == "/api/sims":
                out = []
                for rid, sim in reversed(list(app.sims.items())):
                    w = sim.world
                    out.append({"run_id": rid, "title": sim.title, "status": sim.status,
                                "running": not sim.stop_flag, "paused": bool(getattr(sim, "paused", False)), "error": bool(sim.error),
                                "kind": "experiment" if isinstance(sim, ExperimentRun) else "free",
                                "flies": len(w.flies) if w else 0,
                                "clock_h": round(w.clock_h, 2) if w else 0,
                                "events": len(sim.log.events) if sim.log else 0})
                from flysim.sim.chronicle import list_runs
                live = set(app.sims)
                return self._json({"sims": out, "saved": [r for r in list_runs() if r["run_id"] not in live][:50]})
            if u.path.startswith("/api/brain/"):
                return self._json(brain_static(parts[-1]))
            if len(parts) >= 4 and parts[:2] == ["api", "sim"]:
                sim = app.get(parts[2])
                if sim is None:
                    return self._json({"error": "нет такого наблюдения"}, 404)
                what = parts[3]
                if what == "state":
                    man = None
                    try:
                        man = sim.manifest() if isinstance(sim, Simulation) and sim.world.flies else None
                    except Exception:
                        pass
                    return self._json({"run_id": parts[2], "title": sim.title, "status": sim.status,
                                       "running": not sim.stop_flag, "paused": bool(getattr(sim, "paused", False)), "error": sim.error,
                                       "kind": "experiment" if isinstance(sim, ExperimentRun) else "free",
                                       "config": getattr(sim, "cfg", None), "manifest": man})
                if what == "frames":
                    since = int(q.get("since", ["0"])[0])
                    if isinstance(sim, Simulation):
                        with sim.lock:
                            base = sim.frames[0]["i"] if sim.frames else 0
                            out = sim.frames[max(0, since - base):][:30]
                    else:
                        ch = sim.chron
                        out = [ch.read(i) for i in range(max(since, (ch.n - 30) if ch else 0), ch.n)] if ch else []
                        out = [fr for fr in out if fr]
                        if out and sim.log:
                            out[-1]["events"] = sim.log.events[-60:]
                    return self._json({"frames": out})
                if what == "logs" and len(parts) == 5 and isinstance(sim, Simulation):
                    name = parts[4]
                    stamp = time.strftime("%Y%m%d-%H%M")
                    if name == "all.zip":
                        return self._send(sim.logs_zip(), "application/zip",
                                          {"Content-Disposition": f'attachment; filename="flysim-{parts[2]}.zip"'})
                    getters = {"events.txt": sim.log.events_text, "events.jsonl": sim.log.events_jsonl,
                               "timeseries.csv": sim.log.timeseries_csv, "apple.csv": sim.log.apple_csv}
                    if name in getters:
                        return self._send(getters[name](), "text/plain; charset=utf-8",
                                          {"Content-Disposition": f'attachment; filename="{stamp}-{name}"'})
            if len(parts) >= 4 and parts[:2] == ["api", "run"]:
                rid, what = parts[2], parts[3]
                src, events, live = _source(app, rid)
                if src is None:
                    return self._json({"error": "нет такого прогона"}, 404)
                if what == "frame":
                    return self._json(src.read(int(q.get("i", ["0"])[0])) or {})
                if what == "events":
                    mi = int(q.get("min_imp", ["1"])[0])
                    cat, fly = q.get("cat", [""])[0], q.get("fly", [""])[0]
                    ev = [e for e in events if e.get("imp", 1) >= mi and (not cat or e.get("cat") == cat)
                          and (not fly or e.get("fly") == fly or fly in e.get("others", []))]
                    return self._json({"events": ev})
                if what == "series":
                    n = src.n
                    step = max(1, int(q.get("every", [str(max(1, n // 1500))])[0]))
                    rows = []
                    for i in range(0, n, step):
                        fr = src.read(i)
                        if not fr:
                            continue
                        rows.append({"i": i, "clock_h": fr.get("clock_h"), "t": fr.get("t"),
                                     "env": fr.get("env"),
                                     "flies": {k: {"regions": v.get("regions"), "hunger": v.get("hunger"),
                                                   "energy": v.get("energy"), "ppl1": v.get("rates", {}).get("ppl1"),
                                                   "pam": v.get("rates", {}).get("pam"), "mode": v.get("mode"),
                                                   "surface": v.get("surface"), "proboscis": v.get("proboscis"),
                                                   "n_active": v.get("n_active")}
                                               for k, v in fr.get("flies", {}).items()}})
                    return self._json({"series": rows, "step": step})
                if what == "fly" and len(parts) == 5:
                    if live is not None and isinstance(live, Simulation):
                        pp = live.passport(parts[4])
                        if pp:
                            return self._json(pp)
                    ev = [e for e in events if e.get("fly") == parts[4] or parts[4] in e.get("others", [])]
                    return self._json({"name": parts[4], "life": [e for e in ev if e.get("imp", 1) >= 2],
                                       "event_counts": {}, "status": "архив"})
                if what == "info":
                    from flysim.sim.eventlog import CATEGORY_RU
                    cnt, flies = {}, {}
                    for e in events:
                        cnt[e.get("cat", "info")] = cnt.get(e.get("cat", "info"), 0) + 1
                        if e.get("fly") and e["fly"] != "мир":
                            flies[e["fly"]] = e.get("sex", "")
                    first, last = src.read(0) or {}, src.read(src.n - 1) or {}
                    return self._json({"run_id": rid, "n_frames": src.n, "counts": cnt,
                                       "category_names": CATEGORY_RU, "flies": flies,
                                       "clock_h": [first.get("clock_h", 0), last.get("clock_h", 0)],
                                       "live": bool(live is not None and not live.stop_flag)})
            return self._json({"error": "не найдено"}, 404)

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
