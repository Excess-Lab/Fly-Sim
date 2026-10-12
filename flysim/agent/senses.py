"""Sensory transduction: physical stimuli -> firing rates of identified sensory neurons.

Every stimulus enters the brain only through sensory neurons that exist in the connectome.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from flysim.data.flywire import RAW, Connectome

ODOR = RAW / "odor"


# ------------------------------------------------------------------------------------------
# olfaction
@dataclass
class OlfactionParams:
    # Hallem & Carlson 2006: mean spontaneous rate ~8 Hz; max evoked ~200-300 Hz
    sfr_hz: float = 8.0
    rmax_hz: float = 200.0
    # ASSUMPTION(A50): Hill dose-response with n=1, EC50 = 0.1 in units where 1 = 10^-2 dilution
    # (ORN EC50s typically 10^-4..10^-2; Hallem & Carlson 2006; Si et al. 2019)
    ec50: float = 0.1
    hill: float = 1.0


class Olfaction:
    def __init__(self, ann: pd.DataFrame, p: OlfactionParams | None = None):
        self.p = p or OlfactionParams()
        resp = pd.read_csv(ODOR / "orn_responses.csv", index_col=0)
        resp = resp.drop(columns=[c for c in ["door_name", "InChIKey"] if c in resp.columns])
        self.resp = resp.fillna(0.0).clip(lower=0.0)
        t = ann.type.to_numpy()
        is_orn = pd.Series(t).str.match(r"^ORN_").to_numpy()
        self.idx = np.flatnonzero(is_orn)
        self.glom = np.array([s[4:] for s in t[self.idx]])
        self.side = ann.side.to_numpy()[self.idx]
        # response matrix: odorant x ORN neuron
        cols = [g if g in self.resp.columns else None for g in self.glom]
        self.M = np.zeros((len(self.resp.index), len(self.idx)))
        for j, g in enumerate(cols):
            if g is not None:
                self.M[:, j] = self.resp[g].to_numpy()
        # geosmin activates a single dedicated channel, Or56a -> DA2 (Stensmyr et al. 2012)
        geo = (self.glom == "DA2").astype(float)[None, :]
        self.M = np.vstack([self.M, geo])
        self.odorants = list(self.resp.index) + ["geosmin"]
        self.gain = np.ones(len(self.idx))

    def set_gains(self, gains: dict):
        """gains: {'ORN_DM1': 2.0, ...} from neuromodulation (hunger)."""
        self.gain[:] = 1.0
        for k, v in gains.items():
            if k.startswith("ORN_"):
                self.gain[self.glom == k[4:]] = v

    def rates(self, conc_left: dict, conc_right: dict) -> np.ndarray:
        p = self.p
        out = np.full(len(self.idx), p.sfr_hz)
        for side, conc in (("left", conc_left), ("right", conc_right)):
            m = self.side == side if (self.side == side).any() else np.ones(len(self.idx), bool)
            drive = np.zeros(len(self.idx))
            for od, c in conc.items():
                if c <= 0 or od not in self.odorants:
                    continue
                h = c ** p.hill / (c ** p.hill + p.ec50 ** p.hill)
                drive += self.M[self.odorants.index(od)] * h
            out[m] += p.rmax_hz * np.minimum(drive[m], 1.5)
        return out * self.gain


# ------------------------------------------------------------------------------------------
# gustation
# ASSUMPTION(A51): GRN class -> tastant (FlyWire v3 cell_sub_class labels; Male CNS by type).
# LB1e + LB2 are the Ir94e-like set of Shiu et al. 2024 whose activation suppresses feeding; they
# respond to low salt, which a fruit does not offer (apple Na ~1 mg/100 g), so they stay at
# baseline. Yeast/amino-acid taste (Ir76b) has no identified class in the connectome -> unused.
TASTE_CLASSES = {
    "sugar": r"^LB3[bc]$",
    "water": r"^LB3a$",
    "bitter": r"^LB1[a-d]$",
    "ir94e": r"^(LB1e|LB2[abc])$",
    "salt_high": r"^LB3d$",
}


def fruit_taste_drive(taste: dict) -> dict:
    """Map surface chemistry to GRN class drive (0..1).
    water: ppk28 water cells respond to low osmolarity and are silenced by high osmolarity
      (Cameron et al. 2010); apple juice ~0.6-0.8 Osm -> drive ~0. ASSUMPTION(A51b): linear
      drop from 1 (pure water) to 0 at 0.3 Osm.
    bitter: acids and high ethanol activate bitter GRNs (Charlu et al. 2013; acetic acid
      aversive above ~1 %). ASSUMPTION(A51c): ramps 1->5 g/100 g acetic, 3->8 g/100 g ethanol.
    """
    osm = taste.get("osmolarity", 0.0)
    return {
        "sugar": taste.get("sugar", 0.0),
        "water": max(0.0, 1.0 - osm / 0.3) * taste.get("wet", 0.0),
        "bitter": max(taste.get("bitter", 0.0),
                      min(1.0, max(0.0, (taste.get("acetic_g", 0.0) - 1.0) / 4.0)
                          + max(0.0, (taste.get("ethanol_g", 0.0) - 3.0) / 5.0))),
        "ir94e": 0.0,
        "salt_high": 0.0,
    }


class Gustation:
    def __init__(self, ann: pd.DataFrame):
        t = pd.Series(ann.type.to_numpy())
        self.groups = {k: np.flatnonzero(t.str.match(v).to_numpy()) for k, v in TASTE_CLASSES.items()}
        self.idx = np.concatenate(list(self.groups.values()))
        self.gains = {k: 1.0 for k in self.groups}

    def rates(self, taste: dict, contact: float) -> np.ndarray:
        """taste: normalised 0..1 per class; contact: 0 (none) .. 1 (proboscis on food).
        ASSUMPTION(A52): GRN rate = 150 Hz * concentration * contact, spontaneous ~1 Hz
        (Hiroi et al. 2002; Dahanukar et al. 2007 orders of magnitude)."""
        drive = fruit_taste_drive(taste)
        out = []
        for k, ids in self.groups.items():
            c = drive.get(k, 0.0)
            r = 1.0 + 150.0 * c * contact * self.gains.get(k, 1.0)
            out.append(np.full(len(ids), r))
        return np.concatenate(out) if out else np.zeros(0)


# ------------------------------------------------------------------------------------------
# thermosensation (antennal hot/cold cells)
class Thermo:
    def __init__(self, ann: pd.DataFrame):
        t = pd.Series(ann.type.to_numpy())
        self.hot = np.flatnonzero(t.str.match(r"^TRN_VP2").to_numpy())
        self.cold = np.flatnonzero(t.str.match(r"^TRN_VP3").to_numpy())
        self.idx = np.concatenate([self.hot, self.cold])

    def rates(self, temp_c: float) -> np.ndarray:
        # ASSUMPTION(A53): hot cells ~+15 Hz/degC above 25 C, cold cells ~+10 Hz/degC below
        # (Gallio et al. 2011; Budelli et al. 2019 orders of magnitude)
        hot = 2.0 + 15.0 * max(0.0, temp_c - 25.0)
        cold = 2.0 + 10.0 * max(0.0, 25.0 - temp_c)
        return np.concatenate([np.full(len(self.hot), hot), np.full(len(self.cold), cold)])


# ------------------------------------------------------------------------------------------
# vision: compound eye, one direction per photoreceptor
def _photoreceptor_columns(con: Connectome, ann: pd.DataFrame, sex: str):
    t = pd.Series(ann.type.to_numpy())
    pr = np.flatnonzero(t.str.match(r"^(R1-6|R1-R6|R7|R8)").to_numpy())
    col_xy = np.full((con.n, 2), np.nan)
    if sex == "female":
        c = pd.read_csv(RAW / "annotations" / "codex783" / "column_assignment.csv.gz")
        lut = {int(r): i for i, r in enumerate(con.ids)}
        ii = np.array([lut.get(int(r), -1) for r in c.root_id])
        ok = ii >= 0
        col_xy[ii[ok], 0] = c.x.to_numpy()[ok]
        col_xy[ii[ok], 1] = c.y.to_numpy()[ok]
    else:
        import pyarrow.feather as feather

        from flysim.data.malecns import ANN
        a = feather.read_table(ANN, columns=["bodyId", "assignedOlHex1", "assignedOlHex2"]
                               ).to_pandas().set_index("bodyId").reindex(con.ids)
        col_xy[:, 0] = a.assignedOlHex1.to_numpy(float)
        col_xy[:, 1] = a.assignedOlHex2.to_numpy(float)
    # each photoreceptor inherits the column of its strongest column-assigned partner
    xy = np.full((len(pr), 2), np.nan)
    for k, i in enumerate(pr):
        if not np.isnan(col_xy[i, 0]):
            xy[k] = col_xy[i]
            continue
        lo, hi = con.indptr[i], con.indptr[i + 1]
        tgt, w = con.post[lo:hi], np.abs(con.syn[lo:hi])
        ok = ~np.isnan(col_xy[tgt, 0])
        if ok.any():
            xy[k] = col_xy[tgt[ok][np.argmax(w[ok])]]
    keep = ~np.isnan(xy[:, 0])
    return pr[keep], xy[keep], ann.side.to_numpy()[pr[keep]]


class Vision:
    """Ray-cast luminance per photoreceptor. ASSUMPTION(A54): hex column coordinates are mapped
    linearly onto each eye's field of view (azimuth -15..165 deg lateral, elevation -75..75 deg),
    ~4.6 deg spacing (Land 1997). Photoreceptor 'rate' encodes luminance: bright sky 60 Hz,
    floor 30 Hz, dark objects 5 Hz (graded photoreceptor responses approximated as rates)."""

    def __init__(self, con: Connectome, ann: pd.DataFrame, sex: str):
        self.idx, xy, side = _photoreceptor_columns(con, ann, sex)
        az = np.zeros(len(self.idx))
        el = np.zeros(len(self.idx))
        for s, sign in (("right", -1.0), ("left", 1.0)):
            m = side == s
            if not m.any():
                continue
            x, y = xy[m, 0], xy[m, 1]
            u = (x - x.min()) / max(np.ptp(x), 1e-9)
            v = (y - y.min()) / max(np.ptp(y), 1e-9)
            az[m] = sign * np.radians(-15 + 180 * u)   # +az = left of heading
            el[m] = np.radians(-75 + 150 * v)
        self.dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)

    def rates(self, pos, yaw, spheres) -> np.ndarray:
        """spheres: list of (center xyz, radius). Returns luminance-coded rates."""
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        d = self.dirs @ R.T
        lum = np.where(d[:, 2] > 0, 60.0, 30.0)
        for cen, rad in spheres:
            oc = pos - np.asarray(cen)
            b = d @ oc
            disc = b * b - (oc @ oc - rad * rad)
            hit = (disc > 0) & (-b - np.sqrt(np.maximum(disc, 0)) > 0)
            lum[hit] = 5.0
        return lum


# ------------------------------------------------------------------------------------------
# hearing (Johnston's organ, courtship song) and contact chemosensation
class Hearing:
    def __init__(self, ann: pd.DataFrame):
        t = pd.Series(ann.type.to_numpy())
        self.idx = np.flatnonzero(t.str.match(r"^JO-B").to_numpy())

    def rates(self, song_amp: float) -> np.ndarray:
        # ASSUMPTION(A55): JO-B rate ~ 150 Hz at full near-field song amplitude
        return np.full(len(self.idx), 2.0 + 150.0 * song_amp)


class Interoception:
    """Internal state enters through neurons that sense it.
    IPCs depolarise to glucose (Kreneisz et al. 2010); ASSUMPTION(A56): rate 5..40 Hz with
    haemolymph sugar (energy) level."""

    def __init__(self, ann: pd.DataFrame):
        t = pd.Series(ann.type.to_numpy())
        self.ipc = np.flatnonzero(t.str.match(r"^IPC$").to_numpy())
        self.idx = self.ipc

    def rates(self, energy: float) -> np.ndarray:
        return np.full(len(self.ipc), 5.0 + 35.0 * energy)
