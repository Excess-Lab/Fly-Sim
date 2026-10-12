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

TS_FIELDS = ["clock_h", "t", "fly", "sex", "x", "y", "z", "yaw", "mode", "surface", "proboscis",
             "wing", "copulating", "mated", "hunger", "energy", "crop_ul", "temp",
             "smell_strongest", "smell_conc", "taste_sugar", "taste_ethanol", "taste_acid",
             "PPL1_punish_Hz", "PAM_reward_Hz", "valence_proxy", "MBON_Hz", "KC_Hz",
             "mbon_w_min", "mbon_w_mean", "n_active_neurons",
             "fwd_Hz", "back_Hz", "DNa02_L_Hz", "DNa02_R_Hz", "GF_Hz", "takeoff_Hz", "land_Hz",
             "MN9_Hz", "pIP10_song_Hz", "P1_arousal_Hz", "DNp37_receptive_Hz", "oviDN_Hz"]

APPLE_FIELDS = ["clock_h", "age_d", "sugar", "ethanol", "acetic", "esters", "yeast", "bacteria",
                "ferm_rate", "odor_ethyl_acetate", "odor_ethanol", "odor_acetic_acid", "odor_CO2",
                "odor_hexyl_acetate", "eggs", "larvae", "pupae"]


class EventLog:
    def __init__(self):
        self.events = []
        self.rows = []
        self.apple_rows = []
        self.prev = {}

    # ------------------------------------------------------------------
    def _ev(self, w, f, t, kind, text, **data):
        self.events.append({"clock_h": round(w.clock_h, 5), "t": round(t, 3), "fly": f.cfg.name,
                            "sex": f.cfg.sex, "type": kind, "text": text, "data": data})

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
                "taste_sugar": round(taste.get("sugar", 0.0), 3), "taste_ethanol": round(taste.get("ethanol", 0.0), 3),
                "taste_acid": round(taste.get("acid", 0.0), 3),
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
            self.rows.append(s)
            self._events_for(w, f, t, s, self.prev.get(f.cfg.name))
            self.prev[f.cfg.name] = s
        a = w.apple
        src = a.odor_sources()
        pc = w.population.counts(w.clock_h) if getattr(w, "population", None) else {}
        self.apple_rows.append({
            "clock_h": round(w.clock_h, 5), "age_d": round(a.s.age_h / 24, 4), "sugar": round(a.s.sugar, 4),
            "ethanol": round(a.s.ethanol, 4), "acetic": round(a.s.acetic, 4), "esters": round(a.s.esters, 5),
            "yeast": round(a.s.yeast, 4), "bacteria": round(a.s.aab, 4), "ferm_rate": round(a.s.ferm_rate, 5),
            "odor_ethyl_acetate": round(float(src["ethyl acetate"]), 5), "odor_ethanol": round(float(src["ethanol"]), 5),
            "odor_acetic_acid": round(float(src["acetic acid"]), 5), "odor_CO2": round(float(src["CO2"]), 5),
            "odor_hexyl_acetate": round(float(src["hexyl acetate"]), 5),
            "eggs": pc.get("egg", 0), "larvae": pc.get("larva", 0), "pupae": pc.get("pupa", 0)})

    def _events_for(self, w, f, t, s, p):
        E = lambda kind, text, **d: self._ev(w, f, t, kind, text, **d)  # noqa: E731
        if p is None:
            E("start", f"начало наблюдения: голод {s['hunger']:.2f}, {s['surface']}")
            return
        if (p["mode"], p["surface"]) != (s["mode"], s["surface"]):
            what = {("walk", "apple"): "на яблоке", ("walk", "floor"): "на полу", ("fly", "floor"): "в полёте",
                    ("fly", "apple"): "в полёте"}[(s["mode"], s["surface"])]
            E("location", f"теперь {what}", mode=s["mode"], surface=s["surface"])
        if s["proboscis"] and not p["proboscis"]:
            E("feeding_start", f"выдвинула хоботок и ест (сахар {s['taste_sugar']:.2f}, спирт "
              f"{s['taste_ethanol']:.2f}, кислота {s['taste_acid']:.2f}; MN9 {s['MN9_Hz']:.0f} Гц)")
        if p["proboscis"] and not s["proboscis"]:
            E("feeding_stop", f"перестала есть (голод {s['hunger']:.2f}, зоб {s['crop_ul']:.3f} мкл)")
        if s["temp"] > 30 >= p["temp"]:
            E("heat_on", f"оказалась в жаре {s['temp']:.1f} °C")
        if p["temp"] > 30 >= s["temp"]:
            E("heat_off", "ушла из жары")
        for lvl in HUNGER_LEVELS:
            if (p["hunger"] < lvl) != (s["hunger"] < lvl):
                E("hunger", f"голод {'вырос выше' if s['hunger'] >= lvl else 'упал ниже'} {lvl}", hunger=s["hunger"])
        if s["PPL1_punish_Hz"] > 20 >= p["PPL1_punish_Hz"]:
            E("punish_signal", f"сигнал наказания: PPL1 {s['PPL1_punish_Hz']:.0f} Гц (коррелят, не переживание)")
        if s["PAM_reward_Hz"] > 10 >= p["PAM_reward_Hz"]:
            E("reward_signal", f"сигнал награды: PAM {s['PAM_reward_Hz']:.0f} Гц (коррелят, не переживание)")
        if p["mbon_w_min"] - s["mbon_w_min"] > 0.01:
            E("memory", f"память изменилась: KC→MBON min {p['mbon_w_min']:.3f} → {s['mbon_w_min']:.3f}")
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
            E("behaviour", e)
        f._events_logged = len(f.events)

    # ------------------------------------------------------------------
    def events_jsonl(self) -> str:
        return "\n".join(json.dumps(e, ensure_ascii=False) for e in self.events) + "\n"

    def events_text(self) -> str:
        return "\n".join(f"[д{int(e['clock_h'] // 24)} {e['clock_h'] % 24:6.3f} ч | {e['t']:7.2f} с] "
                         f"{e['fly']:>8}: {e['text']}" for e in self.events) + "\n"

    def _csv(self, rows, fields) -> str:
        buf = io.StringIO()
        wr = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
        return buf.getvalue()

    def timeseries_csv(self) -> str:
        return self._csv(self.rows, TS_FIELDS)

    def apple_csv(self) -> str:
        return self._csv(self.apple_rows, APPLE_FIELDS)

    def save(self, out_dir):
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "events.jsonl").write_text(self.events_jsonl())
        (out / "events.txt").write_text(self.events_text())
        (out / "timeseries.csv").write_text(self.timeseries_csv())
        (out / "apple.csv").write_text(self.apple_csv())
