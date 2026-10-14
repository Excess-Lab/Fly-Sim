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
import numpy as np
import pandas as pd

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
    def __init__(self, ann: pd.DataFrame, extra_ids: dict | None = None, tau_ms: float = 50.0):
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
        sat = lambda x, s: float(np.clip(x / s, 0.0, 1.0))  # noqa: E731
        # positive turn = left (counter-clockwise); DNa02/DNa01 drive ipsilateral turns
        dl = {k: L[k] - self.base_l[k] for k in L}
        dr = {k: R[k] - self.base_r[k] for k in R}
        turn = (dl["turn_DNa02"] - dr["turn_DNa02"]) + 0.5 * (dl["turn_DNa01"] - dr["turn_DNa01"])
        return {
            "forward": sat(r["fwd"], 40.0),
            # ASSUMPTION(A102): each forward command DN type is alone sufficient for forward
            # walking (P9: Bidaye 2020; BDN2/DNg100: Tuthill/Brunton preprint 2025-26; oDN1/DNg97:
            # Bidaye lab preprint 2026), so their type-mean rates add; same 40 Hz scale as A31.
            "forward_pop": sat(r["fwd"] + r["fwd_bdn2"] + r["fwd_odn1"], 40.0),
            "backward": sat(r["back"], 40.0),
            "turn": float(np.clip(turn / 40.0, -1.0, 1.0)),
            "takeoff": r["gf"] > 30.0 or r["takeoff"] > 40.0,
            "flight_power": sat(r["flight_power"], 40.0),
            "land": r["land"] > 30.0,
            "proboscis": r["per"] > 20.0,
            "song": sat(r["song"], 40.0),
            "receptive": r["receptive"] > 10.0,
            "arousal": sat(r["arousal_P1"], 30.0),
            "oviposit": r["ovi"] > 15.0,
        }
