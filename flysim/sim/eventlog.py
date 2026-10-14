"""Human-readable event log + time series of internal states and neural correlates.

Every frame each fly is compared with its previous frame; noteworthy changes become events
("произошло X"). Valence-related columns (PPL1 punishment DANs, PAM reward DANs) are functional
neural correlates, not measurements of subjective experience.
"""
import csv
import io
import json

import numpy as np

HUNGER_LEVELS = (0.25, 0.5, 0.75)

# event type -> (category, importance 1..3)
EVENT_META = {
    "start": ("info", 1), "location": ("movement", 1), "takeoff": ("movement", 1),
    "landing": ("movement", 1), "feeding_start": ("feeding", 1), "feeding_stop": ("feeding", 1),
    "heat_on": ("distress", 2), "heat_off": ("distress", 1), "hunger": ("info", 1),
    "starving": ("distress", 2), "punish_signal": ("distress", 2),
    "distress_sustained": ("distress", 3), "reward_signal": ("reward", 1), "memory": ("memory", 2),
    "arousal": ("social", 1), "song_start": ("courtship", 2), "chase": ("courtship", 2),
    "escape": ("distress", 2), "smell": ("info", 1), "touch": ("social", 1),
    "encounter": ("fight", 2), "lunge": ("fight", 3), "copulation": ("mating", 3),
    "egg": ("mating", 2), "birth": ("life", 3), "death": ("life", 3), "injury": ("life", 2), "transplant": ("life", 3),
    "coma": ("distress", 3), "coma_end": ("distress", 2), "neuron_death": ("life", 3), "dying": ("life", 3),
    "cold_coma": ("distress", 2), "anoxia": ("distress", 3), "behaviour": ("social", 1),
    "apple": ("world", 2), "air": ("world", 2), "weather": ("world", 1), "plant": ("world", 2), "info": ("info", 1),
}
CATEGORY_RU = {"fight": "драки", "courtship": "ухаживание", "mating": "спаривание и яйца",
               "life": "рождения и смерти", "feeding": "кормление", "distress": "дистресс (корреляты)",
               "reward": "награда (корреляты)", "memory": "память", "movement": "перемещения",
               "social": "контакты", "world": "мир", "info": "прочее"}

TS_FIELDS = ["clock_h", "t", "fly", "sex", "x", "y", "z", "yaw", "mode", "surface", "proboscis",
             "wing", "copulating", "mated", "hunger", "energy", "crop_ul", "temp",
             "smell_strongest", "smell_conc", "taste_sugar", "taste_ethanol", "taste_acid",
             "PPL1_punish_Hz", "PAM_reward_Hz", "valence_proxy", "MBON_Hz", "KC_Hz",
             "mbon_w_min", "mbon_w_mean", "n_active_neurons",
             "fwd_Hz", "back_Hz", "DNa02_L_Hz", "DNa02_R_Hz", "GF_Hz", "takeoff_Hz", "land_Hz",
             "MN9_Hz", "pIP10_song_Hz", "P1_arousal_Hz", "DNp37_receptive_Hz", "oviDN_Hz",
             # level 2: motor command derived from the neurons
             "cmd_forward", "cmd_backward", "cmd_turn", "cmd_takeoff", "cmd_land", "cmd_proboscis",
             "drive_L", "drive_R",
             # level 3: physical movement of the body
             "speed_mm_s", "yaw_rate_deg_s", "legs_in_contact", "legs_on_apple", "body_temp_c",
             "neuron_energy", "neurons_blocked", "neurons_dead", "dying"]

APPLE_FIELDS = ["clock_h", "apple", "age_d", "sugar", "ethanol", "acetic", "esters", "yeast", "bacteria",
                "ferm_rate", "odor_ethyl_acetate", "odor_ethanol", "odor_acetic_acid", "odor_CO2",
                "odor_hexyl_acetate", "eggs", "larvae", "pupae"]


class EventLog:
    def __init__(self):
        self.events = []
        self.rows = []
        self.apple_rows = []
        self.prev = {}
        self.frame_idx = 0
        self.session = 0
        self._world_state = {}

    # ------------------------------------------------------------------
    def _ev(self, w, f, t, kind, text, others=(), **data):
        cat, imp = EVENT_META.get(kind, ("info", 1))
        name, sex = (f.cfg.name, f.cfg.sex) if f is not None else ("мир", "")
        pos = [round(float(v), 1) for v in f.body.pos] if f is not None else None
        self.events.append({"id": len(self.events), "clock_h": round(w.clock_h, 5), "t": round(t, 3),
                            "session": self.session, "frame": self.frame_idx, "fly": name, "sex": sex,
                            "type": kind, "cat": cat, "imp": imp, "text": text,
                            "others": list(others), "pos": pos, "data": data})

    def world_events(self, w, t):
        """Events of the world itself: apple stages, air alarms."""
        st = self._world_state
        for k, a in enumerate(w.apples):
            s = a.s
            stage = ("плесень видна" if s.mold > 0.05 else "уксус" if s.acetic > 1.0 else
                     "спиртовое" if (s.sugar < 1.0 and s.ethanol > 2.0) else
                     "брожение" if s.ferm_rate > 0.05 else "свежее")
            key = f"apple{k}"
            if st.get(key) != stage:
                if key in st:
                    self._ev(w, None, t, "apple", f"яблоко {k + 1}: {st[key]} → {stage}", apple=k)
                st[key] = stage
        seen = st.get("deaths", 0)
        for d in getattr(w, "deaths", [])[seen:]:
            self.events.append({"id": len(self.events), "clock_h": d["clock_h"], "t": round(t, 3),
                                "session": self.session, "frame": self.frame_idx, "fly": d["name"],
                                "sex": d["sex"], "type": "death", "cat": "life", "imp": 3,
                                "text": f"{'погиб' if d['sex'] == 'male' else 'погибла'}: {d['cause']} (возраст {d['age_d']} дн)",
                                "others": [], "pos": d["pos"], "data": {"cause": d["cause"]}})
        st["deaths"] = len(getattr(w, "deaths", []))
        wx = getattr(w, "weather", None)
        if wx is not None:
            for te, text in wx.events:
                imp = 2 if ("гроза" in text or "дождь" in text) else 1
                self.events.append({"id": len(self.events), "clock_h": round(te, 4), "t": round(t, 3),
                                    "session": self.session, "frame": self.frame_idx, "fly": "мир",
                                    "sex": "", "type": "weather", "cat": "world", "imp": imp,
                                    "text": text, "others": [], "pos": None, "data": {}})
            wx.events.clear()
            day = wx.is_day(w.clock_h)
            if st.get("day") is not None and st["day"] != day:
                self._ev(w, None, t, "weather", "рассвет" if day else "закат")
            st["day"] = day
        co2_lvl = 2 if w.co2_ppm > 20000 else 1 if w.co2_ppm > 5000 else 0
        o2_lvl = 2 if w.o2_pct < 5 else 1 if w.o2_pct < 15 else 0
        for key, lvl, txt in (("co2", co2_lvl, f"CO₂ в воздухе {w.co2_ppm / 1e4:.2f} %"),
                              ("o2", o2_lvl, f"O₂ в воздухе {w.o2_pct:.1f} %")):
            if lvl > st.get(key, 0):
                self._ev(w, None, t, "air", f"тревога: {txt}")
            st[key] = lvl

    def frame(self, w, t, counts_by_fly=None):
        for f in w.flies:
            r = f.motor.rate
            al, _ = f.antenna_positions()
            odors = w.odor_at(al, f)
            strongest = max(odors.items(), key=lambda kv: kv[1]) if odors else ("", 0.0)
            on_apple = f.body.surface == "apple" and f.body.mode == "walk"
            taste = w.taste_at(f.body.pos) if on_apple else {}
            mb = f.brain.kc_mbon_strength() if f.brain.pl_on else np.ones(1)
            n_act = int(np.count_nonzero(counts_by_fly[f.cfg.name])) if counts_by_fly else -1
            s = {
                "clock_h": round(w.clock_h, 5), "t": round(t, 3), "fly": f.cfg.name, "sex": f.cfg.sex,
                "x": round(float(f.body.pos[0]), 2), "y": round(float(f.body.pos[1]), 2),
                "z": round(float(f.body.pos[2]), 2), "yaw": round(float(f.body.yaw), 3),
                "mode": f.body.mode, "surface": f.body.surface, "proboscis": int(f.body.proboscis),
                "wing": round(float(f.body.wing_ext), 3), "copulating": int(f.copulating),
                "mated": int(f.mated), "hunger": round(f.meta.hunger, 4), "energy": round(f.meta.energy, 4),
                "crop_ul": round(f.meta.crop_ul, 4), "temp": round(float(getattr(f, "last_temp", 25.0)), 2),
                "smell_strongest": strongest[0], "smell_conc": round(float(strongest[1]), 4),
                "taste_sugar": round(taste.get("sugar", 0.0), 3), "taste_ethanol": round(taste.get("ethanol_g", taste.get("ethanol", 0.0)), 3),
                "taste_acid": round(taste.get("acetic_g", taste.get("acid", 0.0)), 3),
                "PPL1_punish_Hz": round(r.get("ppl1", 0.0), 2), "PAM_reward_Hz": round(r.get("pam", 0.0), 2),
                "valence_proxy": round(r.get("pam", 0.0) - r.get("ppl1", 0.0), 2),
                "MBON_Hz": round(r.get("mbon", 0.0), 2), "KC_Hz": round(r.get("kc", 0.0), 3),
                "mbon_w_min": round(float(mb.min()), 4), "mbon_w_mean": round(float(mb.mean()), 4),
                "n_active_neurons": n_act,
                "fwd_Hz": round(r["fwd"], 2), "back_Hz": round(r["back"], 2),
                "DNa02_L_Hz": round(f.motor.rate_l["turn_DNa02"], 2), "DNa02_R_Hz": round(f.motor.rate_r["turn_DNa02"], 2),
                "GF_Hz": round(r["gf"], 2), "takeoff_Hz": round(r["takeoff"], 2), "land_Hz": round(r["land"], 2),
                "MN9_Hz": round(r["per"], 2), "pIP10_song_Hz": round(r["song"], 2),
                "P1_arousal_Hz": round(r["arousal_P1"], 2), "DNp37_receptive_Hz": round(r["receptive"], 2),
                "oviDN_Hz": round(r.get("ovi", 0.0), 2),
            }
            c = f.cmd
            drv = getattr(getattr(f.body, "arena", None), "drive", {}).get(f.cfg.name, [np.nan, np.nan]) \
                if hasattr(f.body, "arena") else [np.nan, np.nan]
            prev = self.prev.get(f.cfg.name)
            if prev is not None and t > prev["t"]:
                dt = t - prev["t"]
                spd = np.hypot(s["x"] - prev["x"], s["y"] - prev["y"]) / dt
                dyaw = (s["yaw"] - prev["yaw"] + np.pi) % (2 * np.pi) - np.pi
                yr = np.degrees(dyaw) / dt
            else:
                spd, yr = 0.0, 0.0
            cont = getattr(f.body, "contacts", {}) or {}
            s.update({"cmd_forward": round(c["forward"], 3), "cmd_backward": round(c["backward"], 3),
                      "cmd_turn": round(c["turn"], 3), "cmd_takeoff": int(c["takeoff"]),
                      "cmd_land": int(c["land"]), "cmd_proboscis": int(c["proboscis"]),
                      "drive_L": round(float(drv[0]), 3), "drive_R": round(float(drv[1]), 3),
                      "speed_mm_s": round(float(spd), 2), "yaw_rate_deg_s": round(float(yr), 1),
                      "legs_in_contact": sum(1 for v in cont.values() if v),
                      "legs_on_apple": sum(1 for v in cont.values() if v == "apple"),
                      "body_temp_c": round(float(getattr(f, "thermal", None).T), 2)
                      if hasattr(f, "thermal") else np.nan})
            ne = getattr(f, "nenergy", None)
            if ne is not None and ne.active:
                s.update({"neuron_energy": round(float(ne.E.mean()), 4),
                          "neurons_blocked": int((ne.sd_t > 1.0).sum()), "neurons_dead": int(ne.dead.sum())})
            else:
                s.update({"neuron_energy": 1.0, "neurons_blocked": 0, "neurons_dead": 0})
            s["dying"] = f.dying or ""
            self.rows.append(s)
            self._events_for(w, f, t, s, self.prev.get(f.cfg.name))
            self.prev[f.cfg.name] = s
        self.world_events(w, t)
        pc = w.population.counts(w.clock_h) if getattr(w, "population", None) else {}
        for k, a in enumerate(w.apples):
            src = a.odor_sources()
            self.apple_rows.append({
                "clock_h": round(w.clock_h, 5), "apple": k, "age_d": round(a.s.age_h / 24, 4),
                "sugar": round(a.s.sugar, 4), "ethanol": round(a.s.ethanol, 4),
                "acetic": round(a.s.acetic, 4), "esters": round(a.s.esters, 5),
                "yeast": round(a.s.yeast, 4), "bacteria": round(a.s.aab, 4),
                "ferm_rate": round(a.s.ferm_rate, 5),
                "odor_ethyl_acetate": round(float(src["ethyl acetate"]), 5),
                "odor_ethanol": round(float(src["ethanol"]), 5),
                "odor_acetic_acid": round(float(src["acetic acid"]), 5),
                "odor_CO2": round(float(src["CO2"]), 5),
                "odor_hexyl_acetate": round(float(src["hexyl acetate"]), 5),
                "eggs": pc.get("egg", 0), "larvae": pc.get("larva", 0), "pupae": pc.get("pupa", 0)})

    def _events_for(self, w, f, t, s, p):
        E = lambda kind, text, **d: self._ev(w, f, t, kind, text, **d)  # noqa: E731
        if p is None:
            E("start", f"начало наблюдения: голод {s['hunger']:.2f}, {s['surface']}")
            if getattr(f, "schema", None) is not None and f.schema.injured():
                hurt = {k: v for k, v in f.schema.legs.items() if v != "intact"}
                E("injury", f"травма (задана экспериментатором): ноги {hurt}, крылья "
                            f"{f.schema.wing_damage}, усики {f.schema.antenna}, глаза {f.schema.eye}")
            return
        dt = max(s["t"] - p["t"], 1e-3)
        if s["mode"] == "fly" and p["mode"] != "fly":
            E("takeoff", "взлетела")
        if p["mode"] == "fly" and s["mode"] != "fly":
            E("landing", f"приземлилась: {'яблоко' if s['surface'] == 'apple' else 'пол'}")
        # sustained punishment-related activity (neural correlate of distress, >=3 s)
        f._ppl1_s = (getattr(f, "_ppl1_s", 0.0) + dt) if s["PPL1_punish_Hz"] > 20 else 0.0
        if f._ppl1_s >= 3.0 and not getattr(f, "_distress_flag", False):
            E("distress_sustained", f"длительный сигнал наказания ≥3 с (PPL1 {s['PPL1_punish_Hz']:.0f} Гц; "
                                    f"нейронный коррелят, не измерение переживания"
                                    + ("; включает обход A24)" if getattr(f, "injected", {}).get("PPL1", 0) > 0 else ")"))
            f._distress_flag = True
        if f._ppl1_s == 0.0:
            f._distress_flag = False
        n = max(1, f.brain.con.n)
        if s["neurons_blocked"] > 0.5 * n >= p.get("neurons_blocked", 0):
            E("coma", f"аноксическая кома: {s['neurons_blocked'] / n:.0%} нейронов в деполяризационном блоке "
                      f"(энергия нейронов {s['neuron_energy']:.2f}, O₂ {w.o2_pct:.1f} %)")
        if p.get("neurons_blocked", 0) > 0.5 * n >= s["neurons_blocked"] and not s["dying"]:
            E("coma_end", f"выход из комы: нейроны восстанавливаются (энергия {s['neuron_energy']:.2f})")
        for frac in (0.01, 0.1, 0.5):
            if s["neurons_dead"] >= frac * n > p.get("neurons_dead", 0):
                E("neuron_death", f"необратимо погибло {s['neurons_dead']:,} нейронов ({s['neurons_dead'] / n:.0%})")
        if s["dying"] and not p.get("dying"):
            E("dying", f"умирает ({s['dying']}): снабжение мозга прекратилось, по нервной системе идёт волна "
                       f"распространяющейся деполяризации — нейроны дают залп и замолкают")
        if s["energy"] <= 0.0 < p["energy"]:
            E("starving", "энергия на нуле — истощение")
        chill = f._chill() if hasattr(f, "_chill") else 1.0
        if chill < 0.2 and not getattr(f, "_coma", False):
            E("cold_coma", f"холодовая кома (тело {s['body_temp_c']} °C)")
            f._coma = True
        elif chill > 0.5:
            f._coma = False
        if w.o2_pct < 2.0 and not getattr(f, "_anox", False):
            E("anoxia", f"ступор от нехватки кислорода (O₂ {w.o2_pct:.1f} %)")
            f._anox = True
        elif w.o2_pct > 3.0:
            f._anox = False
        # social detectors
        for o in w.flies:
            if o is f:
                continue
            d = np.asarray(o.body.pos) - np.asarray(f.body.pos)
            dist = float(np.linalg.norm(d))
            facing = float(np.dot(d[:2], f.body.heading()[:2]) / max(dist, 1e-6))
            key = f"_near_{o.cfg.name}"
            if f.cfg.sex == "male" and o.cfg.sex == "female":
                chasing = dist < 5.0 and facing > 0.6 and f.body.mode == o.body.mode == "walk"
                ts = (getattr(f, key, 0.0) + dt) if chasing else 0.0
                setattr(f, key, ts)
                if ts >= 2.0 and not getattr(f, key + "_ev", False):
                    E("chase", f"преследует {o.cfg.name} ≥2 с (ухаживание?)", others=[o.cfg.name])
                    setattr(f, key + "_ev", True)
                if ts == 0.0:
                    setattr(f, key + "_ev", False)
            if f.cfg.sex == "male" and o.cfg.sex == "male" and f.cfg.name < o.cfg.name:
                close = dist < 3.0 and (f.cmd.get("arousal", 0) > 0.3 or o.cmd.get("arousal", 0) > 0.3)
                if close and not getattr(f, key + "_enc", False):
                    E("encounter", f"стычка с {o.cfg.name} (близко, P1 возбуждён)", others=[o.cfg.name])
                setattr(f, key + "_enc", close)
        if (p["mode"], p["surface"]) != (s["mode"], s["surface"]):
            what = "в полёте" if s["mode"] == "fly" else {
                "apple": "на яблоке", "floor": "на полу", "wall": "на стенке"}.get(s["surface"], str(s["surface"]))
            E("location", f"теперь {what}", mode=s["mode"], surface=s["surface"])
        if s["proboscis"] and not p["proboscis"]:
            if s["surface"] == "apple":
                E("feeding_start", f"выдвинула хоботок и ест (сахар {s['taste_sugar']:.2f}, спирт "
                  f"{s['taste_ethanol']:.2f}, кислота {s['taste_acid']:.2f}; MN9 {s['MN9_Hz']:.0f} Гц)")
            else:
                E("info", f"выдвинула хоботок на полу — пробует субстрат, еды нет (MN9 {s['MN9_Hz']:.0f} Гц)")
        if p["proboscis"] and not s["proboscis"] and p["surface"] == "apple":
            E("feeding_stop", f"перестала есть (голод {s['hunger']:.2f}, зоб {s['crop_ul']:.3f} мкл)")
        if s["temp"] > 30 >= p["temp"]:
            E("heat_on", f"оказалась в жаре {s['temp']:.1f} °C")
        if p["temp"] > 30 >= s["temp"]:
            E("heat_off", "ушла из жары")
        for lvl in HUNGER_LEVELS:
            if (p["hunger"] < lvl) != (s["hunger"] < lvl):
                E("hunger", f"голод {'вырос выше' if s['hunger'] >= lvl else 'упал ниже'} {lvl}", hunger=s["hunger"])
        if s["PPL1_punish_Hz"] > 20 >= p["PPL1_punish_Hz"]:
            inj = getattr(f, "injected", {}).get("PPL1", 0.0)
            E("punish_signal", f"сигнал наказания: PPL1 {s['PPL1_punish_Hz']:.0f} Гц (коррелят, не переживание"
              + (f"; включает {inj:.0f} Гц обхода A24)" if inj > 0 else "; по коннектому)"), shortcut=inj > 0)
        if s["PAM_reward_Hz"] > 10 >= p["PAM_reward_Hz"]:
            inj = getattr(f, "injected", {}).get("PAM", 0.0)
            E("reward_signal", f"сигнал награды: PAM {s['PAM_reward_Hz']:.0f} Гц (коррелят, не переживание"
              + (f"; включает {inj:.0f} Гц обхода A25)" if inj > 0 else "; по коннектому)"), shortcut=inj > 0)
        # memory: report only new lows (lasting depression), in steps of 0.05
        low = getattr(f, "_mem_low", 1.0)
        if s["mbon_w_min"] < low - 0.05:
            E("memory", f"след памяти: самый ослабленный синапс KC→MBON {s['mbon_w_min']:.2f} "
              f"(1 = исходно, ослабление = выученное избегание)")
            f._mem_low = s["mbon_w_min"]
        if s["P1_arousal_Hz"] > 15 >= p["P1_arousal_Hz"]:
            E("arousal", f"возбуждение P1 {s['P1_arousal_Hz']:.0f} Гц")
        if s["wing"] > 0.3 >= p["wing"]:
            E("song_start", f"поёт песню ухаживания (pIP10 {s['pIP10_song_Hz']:.0f} Гц)")
        if s["GF_Hz"] > 30 >= p["GF_Hz"]:
            E("escape", f"гигантское волокно сработало ({s['GF_Hz']:.0f} Гц)")
        if s["smell_conc"] > 0.3 and p["smell_conc"] <= 0.3:
            E("smell", f"сильный запах: {s['smell_strongest']} ({s['smell_conc']:.2f})")
        other = w.touching(f)
        if other is not None and not getattr(f, "_touching", False):
            E("touch", f"контакт с {other.cfg.name}")
        f._touching = other is not None
        # behavioural events recorded by the fly itself (copulation, eggs, lunges)
        seen = getattr(f, "_events_logged", 0)
        for te, e in f.events[seen:]:
            if "->(" in e:
                continue
            kind = ("lunge" if e.startswith("lunge->") else "copulation" if e.startswith("copulation")
                    else "egg" if "egg" in e else "behaviour")
            txt = {"lunge": f"выпад на {e.split('->')[-1]}", "copulation": f"спаривание ({e})",
                   "egg": "отложила яйцо" if e == "laid egg" else e}.get(kind, e)
            if e.startswith("lunged_by"):
                continue
            E(kind, txt, others=[e.split("->")[-1].split("<-")[-1]] if "->" in e or "<-" in e else [])
        f._events_logged = len(f.events)

    # ------------------------------------------------------------------
    def events_jsonl(self) -> str:
        return "\n".join(json.dumps(e, ensure_ascii=False) for e in self.events) + "\n"

    def events_text(self) -> str:
        return "\n".join(f"[д{int(e['clock_h'] // 24)} {e['clock_h'] % 24:6.3f} ч | {e['t']:7.2f} с] "
                         f"{'!' * e.get('imp', 1):3s} {e['fly']:>8}: {e['text']}" for e in self.events) + "\n"

    def counts(self) -> dict:
        c = {}
        for e in self.events:
            c[e["cat"]] = c.get(e["cat"], 0) + 1
        return c

    def _csv(self, rows, fields) -> str:
        buf = io.StringIO()
        wr = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
        return buf.getvalue()

    # ---- streaming to disk (long runs: bounded memory, nothing lost on a hard stop) ----------
    def stream_to(self, out_dir):
        from pathlib import Path
        self._stream = Path(out_dir)
        self._stream.mkdir(parents=True, exist_ok=True)
        self._ev_written = 0
        for name, fields in (("timeseries.csv", TS_FIELDS), ("apple.csv", APPLE_FIELDS)):
            f = self._stream / name
            if not f.exists():
                f.write_text(",".join(fields) + "\n")

    def flush_stream(self):
        """Append rows/events produced since the last call; keep only a short tail in memory."""
        st = getattr(self, "_stream", None)
        if st is None:
            return
        for name, rows, fields in (("timeseries.csv", self.rows, TS_FIELDS),
                                   ("apple.csv", self.apple_rows, APPLE_FIELDS)):
            if rows:
                buf = io.StringIO()
                csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore").writerows(rows)
                with open(st / name, "a") as fh:
                    fh.write(buf.getvalue())
                rows.clear()
        new = self.events[self._ev_written:]
        if new:
            with open(st / "events.jsonl", "a") as fh:
                fh.write("".join(json.dumps(e, ensure_ascii=False, default=float) + "\n" for e in new))
            self._ev_written = len(self.events)

    def timeseries_csv(self) -> str:
        if getattr(self, "_stream", None) is not None:
            self.flush_stream()
            return (self._stream / "timeseries.csv").read_text()
        return self._csv(self.rows, TS_FIELDS)

    def apple_csv(self) -> str:
        if getattr(self, "_stream", None) is not None:
            self.flush_stream()
            return (self._stream / "apple.csv").read_text()
        return self._csv(self.apple_rows, APPLE_FIELDS)

    def save(self, out_dir):
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if getattr(self, "_stream", None) is not None and Path(self._stream).resolve() == out.resolve():
            self.flush_stream()
            (out / "events.txt").write_text(self.events_text())
            return
        (out / "events.jsonl").write_text(self.events_jsonl())
        (out / "events.txt").write_text(self.events_text())
        (out / "timeseries.csv").write_text(self.timeseries_csv())
        (out / "apple.csv").write_text(self.apple_csv())
