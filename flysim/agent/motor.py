"""Descending/motor neuron readout -> motor commands.

Only neurons with an experimentally established motor role are read out:
  DNa02, DNa01   steering, ipsilateral turning (Rayshubskiy et al. 2020; Chen et al. 2018)
  DNp09 (P9)     forward walking / object-directed turning (Bidaye et al. 2020)
  MDN            backward walking (Bidaye et al. 2014)
  DNp01 (GF)     escape take-off (Allen et al. 2006; von Reyn et al. 2014)
  DNp02/04/11    looming-evoked take-off (von Reyn et al. 2017)
  DNg02          wing-stroke amplitude / flight power (Namiki et al. 2022)
  DNp10, DNp07   landing (Ache et al. 2019)
  MN9            proboscis extension / feeding (Gordon & Scott 2009; Shiu et al. 2024)
  pIP10          courtship song / wing extension, male (von Philipsborn et al. 2011)
  DNp37 (vpoDN)  vaginal plate opening = female receptivity (Wang et al. 2021)
  pC1 'pMP4'     male P1 cluster; aggression/courtship arousal (Hoopfer et al. 2015)
ASSUMPTION(A31): rate->command gains and thresholds are order-of-magnitude choices.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# Proboscis threshold calibrated on behaviour (scripts/calibrate_per.py): the MN9 rate a FED fly
# reaches at 800 mM sucrose, where fed flies extend on 50 % of trials (Inagaki et al. 2014).
# MN9 firing during PER has never been recorded (Shiu et al. 2024).
PER_HZ = {"female_banc": 49.3, "male": 20.7}


@dataclass
class MotorParams:
    """Rate -> command conversion. Each value carries its source (ASSUMPTION A31 where none).
    'scale' = rate (Hz) that gives a full command; 'hz' = threshold rate for an all-or-none act."""
    fwd_scale: float = 40.0
    # Rayshubskiy et al. 2024 (eLife): yaw velocity is linear in the right-left DNa02 difference,
    # ~100 spikes/s <-> ~600 deg/s (full turn)
    turn_scale: float = 100.0
    # MDN fires ~8.5 Hz at rest and rises during backward walking (Dahlhoff et al. 2025 preprint)
    back_rest_hz: float = 10.0
    back_scale: float = 50.0
    flight_scale: float = 40.0
    song_scale: float = 40.0
    arousal_scale: float = 30.0
    # In real flies one giant-fibre spike triggers a short-mode take-off (von Reyn et al. 2014), but
    # the GF is silent at rest; the BANC model GF fires ~6 Hz at rest (artefact), so a single-spike
    # rule makes flies take off constantly. A filtered-rate threshold is used instead (A31).
    gf_takeoff_hz: float = 30.0
    takeoff_hz: float = 40.0
    land_hz: float = 30.0
    per_hz: float = 20.0
    receptive_hz: float = 10.0
    ovi_hz: float = 15.0
    sources: dict = field(default_factory=lambda: {k: "A31 (order of magnitude)" for k in (
        "fwd_scale", "flight_scale", "song_scale", "arousal_scale", "takeoff_hz", "land_hz",
        "per_hz", "receptive_hz", "ovi_hz")} | {
        "turn_scale": "Rayshubskiy et al. 2024", "back_rest_hz": "Dahlhoff et al. 2025 (preprint)",
        "back_scale": "Dahlhoff et al. 2025 (preprint)", "gf_takeoff_hz": "A31 (single-spike rule of von Reyn 2014 unusable: BANC GF fires at rest)"})

GROUPS = {
    "turn_DNa02": r"^DNa02$",
    "turn_DNa01": r"^DNa01$",
    "fwd": r"^DNp09$",
    # forward-walking command DNs used by locomotion="brain" (A102): BDN2 = DNg100,
    # oDN1 = DNg97 (BANC fanc_cell_type synonyms "BDN2, DNg100", "DNg97, oDN1")
    "fwd_bdn2": r"^DNg100$",
    "fwd_odn1": r"^DNg97$",
    "back": r"^MDN",
    "gf": r"^DNp01$",
    "takeoff": r"^DNp(02|04|11)$",
    "flight_power": r"^DNg02",
    "land": r"^DNp(10|07)$",
    "per": r"^(MN9|CB0701)$",
    "song": r"^pIP10$",
    "receptive": r"^DNp37$",
    "ovi": r"^oviDN",          # oviposition descending neurons (Wang et al. 2020)
    # not motor: logged as neural correlates of valence / learning
    "ppl1": r"^PPL1",
    "pam": r"^PAM",
    "mbon": r"^MBON",
    "kc": r"^KC",
}


class MotorReadout:
    def __init__(self, ann: pd.DataFrame, extra_ids: dict | None = None, tau_ms: float = 50.0,
                 p: MotorParams | None = None):
        self.p = p or MotorParams()
        t = pd.Series(ann.type.to_numpy())
        side = ann.side.to_numpy()
        self.groups = {}
        for k, pat in GROUPS.items():
            idx = np.flatnonzero(t.str.match(pat).to_numpy())
            self.groups[k] = idx
        p1 = (t.str.match(r"^pC1").to_numpy()
              & pd.Series(ann.synonyms.to_numpy()).str.contains("pMP4").to_numpy())
        self.groups["arousal_P1"] = np.flatnonzero(p1)
        for k, v in (extra_ids or {}).items():
            self.groups[k] = np.asarray(v)
        self.left = {k: v[side[v] == "left"] for k, v in self.groups.items()}
        self.right = {k: v[side[v] == "right"] for k, v in self.groups.items()}
        self.rate = {k: 0.0 for k in self.groups}
        self.rate_l = {k: 0.0 for k in self.groups}
        self.rate_r = {k: 0.0 for k in self.groups}
        self.tau = tau_ms
        self.base_l = {k: 0.0 for k in self.groups}
        self.base_r = {k: 0.0 for k in self.groups}
        self.age_ms = 0.0
        self.gf_spikes = 0

    def adapt(self, dt_ms: float):
        """ASSUMPTION(A59): intrinsic, always-on adaptation of the left/right steering baseline
        (homeostatic adaptation; fast while the nervous system settles after 'waking', then
        tau = 60 s). It is part of the organism, never triggered by an experiment. Needed because
        the connectome model is strongly left/right asymmetric (symmetric ORN input drives only
        the left DNa02), partly from reconstruction differences."""
        self.age_ms += dt_ms
        tau = min(60_000.0, 300.0 + self.age_ms)
        a = dt_ms / tau
        for k in self.groups:
            self.base_l[k] += a * (self.rate_l[k] - self.base_l[k])
            self.base_r[k] += a * (self.rate_r[k] - self.base_r[k])

    def update(self, counts: np.ndarray, dt_ms: float):
        a = np.exp(-dt_ms / self.tau)
        self.gf_spikes = int(counts[self.groups["gf"]].sum()) if len(self.groups.get("gf", [])) else 0
        for k, idx in self.groups.items():
            inst = counts[idx].mean() / (dt_ms * 1e-3) if len(idx) else 0.0
            self.rate[k] = a * self.rate[k] + (1 - a) * inst
            for side, store, ids in (("l", self.rate_l, self.left[k]),
                                     ("r", self.rate_r, self.right[k])):
                inst_s = counts[ids].mean() / (dt_ms * 1e-3) if len(ids) else 0.0
                store[k] = a * store[k] + (1 - a) * inst_s
        self.adapt(dt_ms)

    def commands(self) -> dict:
        r, L, R = self.rate, self.rate_l, self.rate_r
        P = self.p
        sat = lambda x, s: float(np.clip(x / s, 0.0, 1.0))  # noqa: E731
        # positive turn = left (counter-clockwise); DNa02/DNa01 drive ipsilateral turns
        dl = {k: L[k] - self.base_l[k] for k in L}
        dr = {k: R[k] - self.base_r[k] for k in R}
        turn = (dl["turn_DNa02"] - dr["turn_DNa02"]) + 0.5 * (dl["turn_DNa01"] - dr["turn_DNa01"])
        return {
            "forward": sat(r["fwd"], P.fwd_scale),
            # ASSUMPTION(A102): each forward command DN type is alone sufficient for forward
            # walking (P9: Bidaye 2020; BDN2/DNg100: Tuthill/Brunton preprint 2025-26; oDN1/DNg97:
            # Bidaye lab preprint 2026), so their type-mean rates add; same 40 Hz scale as A31.
            "forward_pop": sat(r["fwd"] + r["fwd_bdn2"] + r["fwd_odn1"], P.fwd_scale),
            "backward": sat(max(0.0, r["back"] - P.back_rest_hz), P.back_scale),
            "turn": float(np.clip(turn / P.turn_scale, -1.0, 1.0)),
            "takeoff": r["gf"] > P.gf_takeoff_hz or r["takeoff"] > P.takeoff_hz,
            "flight_power": sat(r["flight_power"], P.flight_scale),
            "land": r["land"] > P.land_hz,
            "proboscis": r["per"] > P.per_hz,
            "song": sat(r["song"], P.song_scale),
            "receptive": r["receptive"] > P.receptive_hz,
            "arousal": sat(r["arousal_P1"], P.arousal_scale),
            "oviposit": r["ovi"] > P.ovi_hz,
        }
