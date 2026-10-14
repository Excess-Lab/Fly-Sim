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
    # per-odorant EC50 (same units as concentrations). CO2: 1 unit = 1 %; ab1C/Gr21a neurons
    # respond from ~0.1-0.2 % with half-maximal responses around ~1 % (Suh et al. 2004;
    # Faucher et al. 2006). geosmin: Or56a is extremely sensitive (Stensmyr et al. 2012).
    ec50_by_odor: dict = None

    def __post_init__(self):
        if self.ec50_by_odor is None:
            self.ec50_by_odor = {"CO2": 1.0, "geosmin": 0.01}


def infer_side(con, ann, idx, side):
    """ASSUMPTION(A50c): neurons without an annotated side take the side of the neurons that
    receive most of their synapses (sensory afferents project ipsilaterally-biased)."""
    side = np.asarray(side, dtype=object).copy()
    if con is None:
        return side
    side_all = ann.side.astype(str).to_numpy()
    for k, i in enumerate(idx):
        if side[k] in ("left", "right"):
            continue
        lo, hi = con.indptr[i], con.indptr[i + 1]
        tg, w = con.post[lo:hi], np.abs(con.syn[lo:hi])
        l_, r_ = w[side_all[tg] == "left"].sum(), w[side_all[tg] == "right"].sum()
        if l_ != r_:
            side[k] = "left" if l_ > r_ else "right"
    return side


class Olfaction:
    def __init__(self, ann: pd.DataFrame, p: OlfactionParams | None = None, con=None):
        self.p = p or OlfactionParams()
        # spontaneous-rate-subtracted DoOR responses: negative = inhibition below spontaneous
        resp = pd.read_csv(ODOR / "orn_responses_sfr_subtracted.csv", index_col=0)
        resp = resp.drop(columns=[c for c in ["door_name", "InChIKey"] if c in resp.columns])
        sfr_rows = [i for i in resp.index if str(i).upper().startswith("SFR")]
        resp = resp.drop(index=sfr_rows)
        raw = pd.read_csv(ODOR / "orn_responses.csv", index_col=0)
        raw_sfr = [i for i in raw.index if str(i).upper().startswith("SFR")]
        sfr_norm = raw.loc[raw_sfr[0]] if raw_sfr else None
        self.resp = resp.fillna(0.0)
        t = ann.type.to_numpy()
        is_orn = pd.Series(t).str.match(r"^ORN_").to_numpy()
        self.idx = np.flatnonzero(is_orn)
        self.glom = np.array([s[4:] for s in t[self.idx]])
        self.side = ann.side.astype(str).to_numpy()[self.idx].astype(object)
        if con is not None:
            # ASSUMPTION(A50c): ORNs without an annotated side (BANC 743, Male CNS 409; no position
            # either) get the side of the antennal lobe that receives more of their synapses -
            # each ORN projects to both lobes with an ipsilateral bias (Gaudry et al. 2013). On
            # ORNs with a known side this rule agrees in 80 % (BANC) / 70 % (Male CNS).
            side_all = ann.side.astype(str).to_numpy()
            for k, i in enumerate(self.idx):
                if self.side[k] in ("left", "right"):
                    continue
                lo, hi = con.indptr[i], con.indptr[i + 1]
                tg, w = con.post[lo:hi], np.abs(con.syn[lo:hi])
                l_, r_ = w[side_all[tg] == "left"].sum(), w[side_all[tg] == "right"].sum()
                if l_ != r_:
                    self.side[k] = "left" if l_ > r_ else "right"
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
        # measured spontaneous rate of each receptor type (DoOR SFR, normalised like the
        # responses; x rmax gives Hz: median ~12 Hz, as in Hallem & Carlson 2006, range 1-47 Hz).
        # Glomeruli without a value get the median. ASSUMPTION(A50)
        sfr = np.full(len(self.idx), np.nan)
        if sfr_norm is not None:
            vals = pd.to_numeric(sfr_norm, errors="coerce")
            for j, g in enumerate(self.glom):
                if g in vals.index and vals[g] == vals[g]:
                    sfr[j] = float(vals[g]) * self.p.rmax_hz
        med = np.nanmedian(sfr) if np.isfinite(sfr).any() else self.p.sfr_hz
        self.sfr = np.where(np.isnan(sfr), med, sfr)

    def set_gains(self, gains: dict):
        """gains: {'ORN_DM1': 2.0, ...} from neuromodulation (hunger)."""
        self.gain[:] = 1.0
        for k, v in gains.items():
            if k.startswith("ORN_"):
                self.gain[self.glom == k[4:]] = v

    def _drive(self, conc: dict) -> np.ndarray:
        p = self.p
        drive = np.zeros(len(self.idx))
        for od, c in conc.items():
            if c <= 0 or od not in self.odorants:
                continue
            e = p.ec50_by_odor.get(od, p.ec50)
            h = c ** p.hill / (c ** p.hill + e ** p.hill)
            drive += self.M[self.odorants.index(od)] * h
        return drive

    def rates(self, conc_left: dict, conc_right: dict) -> np.ndarray:
        """ORN rate = spontaneous + gain * evoked; inhibitory odours can lower it to 0.
        Neurons without an annotated antenna side get the mean of both antennae."""
        p = self.p
        dl, dr = self._drive(conc_left), self._drive(conc_right)
        drive = np.where(self.side == "left", dl, np.where(self.side == "right", dr, 0.5 * (dl + dr)))
        evoked = p.rmax_hz * np.clip(drive, -1.0, 1.5) * self.gain  # hunger scales evoked part only
        return np.maximum(0.0, self.sfr + evoked)


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

    # sugar GRNs saturate: R = 1 + 60 * C / (C + 35 mM) (L-sensilla ~43 Hz at 50 mM sucrose;
    # Dweck & Carlson 2023 curves; constants extrapolated, A52); apple flesh 10 g/100 g sugar
    # ~0.55 M hexose; bitter GRNs up to ~60 Hz (Weiss et al. 2011); spontaneous ~1 Hz (Dweck &
    # Carlson 2023). Other classes: 150 Hz x drive (A52, order of magnitude).
    SUGAR_RMAX, SUGAR_K_M, SUGAR_M_PER_UNIT = 60.0, 0.035, 0.55
    BITTER_RMAX = 60.0

    def rates(self, taste: dict, contact: float) -> np.ndarray:
        """taste: normalised 0..1 per class; contact: 0 (none) .. 1 (proboscis on food).
        Hunger acts presynaptically on sugar GRN output (Fly.sense), not on these rates."""
        drive = fruit_taste_drive(taste)
        out = []
        for k, ids in self.groups.items():
            c = drive.get(k, 0.0) * contact
            if k == "sugar":
                m = c * self.SUGAR_M_PER_UNIT
                r = 1.0 + self.SUGAR_RMAX * m / (m + self.SUGAR_K_M)
            elif k == "bitter":
                r = 1.0 + self.BITTER_RMAX * c * self.gains.get(k, 1.0)
            else:
                r = 1.0 + 150.0 * c * self.gains.get(k, 1.0)
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

    def rates(self, temp_c: float, dt: float = 0.01) -> np.ndarray:
        """Budelli et al. 2019 (Neuron): hot cells are tonic, 17 Hz at 20 C, 37 Hz at 25 C, 74 Hz
        at 30 C (Q10 ~4.4); cold cells are purely phasic: ~30 Hz per cell regardless of the
        steady temperature (92-102 Hz pooled, likely 3 cells), up to ~3x within ~0.2 C of cooling,
        adapting within seconds, and silent for seconds when warmed. ASSUMPTION(A53): cold-cell
        baseline per cell = pooled / 3; adaptation tau 2 s."""
        if not hasattr(self, "_t_slow"):
            self._t_slow = temp_c
        a = 1.0 - np.exp(-dt / 2.0)
        cooling = self._t_slow - temp_c
        self._t_slow += a * (temp_c - self._t_slow)
        hot = min(150.0, 37.0 * 4.4 ** ((temp_c - 25.0) / 10.0))
        cold = 30.0 * (1.0 + 2.0 * float(np.clip(cooling / 0.2, 0.0, 1.0)))
        if cooling < -0.05:            # warming silences cold cells for a few seconds
            cold = 0.0
        return np.concatenate([np.full(len(self.hot), hot), np.full(len(self.cold), cold)])


# ------------------------------------------------------------------------------------------
# vision: compound eye, one direction per photoreceptor
def _photoreceptor_columns(con: Connectome, ann: pd.DataFrame, sex: str):
    """sex: dataset name ("female" = FlyWire, "male" = Male CNS, "female_banc" = BANC)."""
    t = pd.Series(ann.type.to_numpy())
    pr = np.flatnonzero(t.str.match(r"^(R1-6|R1-R6|R7|R8)").to_numpy())
    col_xy = np.full((con.n, 2), np.nan)
    if sex == "female_banc":
        # ASSUMPTION(A54b): BANC has no column assignment. The medulla is retinotopic, so each
        # photoreceptor (BANC: R7/R8 only) takes the soma position of its strongest medulla
        # partner, projected onto that optic lobe's two principal axes. Coarser than columns.
        xyz = ann[["x", "y", "z"]].to_numpy(float)
        side_all = ann.side.astype(str).to_numpy()
        # optic-lobe side from POSITION (BANC side labels of optic-lobe types are inconsistent):
        # left = the x half where most left-annotated neurons are
        okp = ~np.isnan(xyz[:, 0])
        xl = np.nanmedian(xyz[okp & (side_all == "left"), 0])
        xr = np.nanmedian(xyz[okp & (side_all == "right"), 0])
        mid = 0.5 * (xl + xr)
        part = np.full(len(pr), -1)
        for k, i in enumerate(pr):
            lo, hi = con.indptr[i], con.indptr[i + 1]
            tgt, w = con.post[lo:hi], np.abs(con.syn[lo:hi])
            ok = ~np.isnan(xyz[tgt, 0])
            if ok.any():
                part[k] = tgt[ok][np.argmax(w[ok])]
        keep = part >= 0
        pr, part = pr[keep], part[keep]
        xy = np.zeros((len(pr), 2))
        px = xyz[part, 0]
        side = np.where((px > mid) == (xl > xr), "left", "right")
        for s_ in ("left", "right"):
            m = side == s_
            if m.sum() < 10:
                continue
            X = xyz[part[m]] - xyz[part[m]].mean(0)
            v = np.linalg.svd(X, full_matrices=False)[2]
            xy[m] = X @ v[:2].T
        ann_side = ann.side.to_numpy().copy()
        ann_side[pr] = side
        return _orient_eyes(pr, xy, part, ann, ann_side[pr])
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
        # axial hex coordinates (60 deg axes) -> cartesian
        h1, h2 = col_xy[:, 0].copy(), col_xy[:, 1].copy()
        col_xy[:, 0] = h1 + h2 / 2.0
        col_xy[:, 1] = h2 * np.sqrt(3) / 2.0
    # each photoreceptor inherits the column of its strongest column-assigned partner
    xy = np.full((len(pr), 2), np.nan)
    partner = np.full(len(pr), -1)
    for k, i in enumerate(pr):
        if not np.isnan(col_xy[i, 0]):
            xy[k] = col_xy[i]
            partner[k] = i
            continue
        lo, hi = con.indptr[i], con.indptr[i + 1]
        tgt, w = con.post[lo:hi], np.abs(con.syn[lo:hi])
        ok = ~np.isnan(col_xy[tgt, 0])
        if ok.any():
            partner[k] = tgt[ok][np.argmax(w[ok])]
            xy[k] = col_xy[partner[k]]
    keep = ~np.isnan(xy[:, 0])
    pr, xy, partner = pr[keep], xy[keep], partner[keep]
    return _orient_eyes(pr, xy, partner, ann, ann.side.to_numpy()[pr])


def _orient_eyes(pr, xy, partner, ann, side):
    # orient each eye from anatomy: elevation axis = the column axis best correlated with the
    # dorso-ventral position of the partner neurons (y grows ventrally in both datasets)
    py = ann.y.to_numpy()[partner]
    pz = ann.z.to_numpy()[partner]
    for s_ in ("left", "right"):
        m = (side == s_) & ~np.isnan(py)
        if m.sum() < 10:
            continue
        cx = abs(np.corrcoef(xy[m, 0], py[m])[0, 1])
        cy = abs(np.corrcoef(xy[m, 1], py[m])[0, 1])
        if cx > cy:  # swap so that column 1 is the elevation axis
            xy[side == s_] = xy[side == s_][:, ::-1]
        if np.corrcoef(xy[m, 1], py[m])[0, 1] > 0:  # make elevation increase dorsally
            xy[side == s_, 1] *= -1
        mz = m & ~np.isnan(pz)
        if mz.sum() > 10 and np.corrcoef(xy[mz, 0], pz[mz])[0, 1] < 0:  # azimuth: front->back
            xy[side == s_, 0] *= -1
    return pr, xy, side


class Vision:
    """Ray-cast luminance per photoreceptor. ASSUMPTION(A54): hex column coordinates are mapped
    linearly onto each eye's field of view (azimuth -15..165 deg lateral, elevation -75..75 deg),
    ~4.6 deg spacing (Land 1997). Photoreceptor 'rate' encodes luminance: bright sky 60 Hz,
    floor 30 Hz, dark objects 5 Hz (graded photoreceptor responses approximated as rates)."""

    def __init__(self, con: Connectome, ann: pd.DataFrame, sex: str):
        self.idx, xy, side = _photoreceptor_columns(con, ann, sex)
        self.side = np.asarray(side).astype(str)   # per photoreceptor, incl. inferred (A54b)
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

    def rates(self, pos, yaw, spheres, lux=None) -> np.ndarray:
        """spheres: list of (center xyz, radius). Returns luminance-coded rates.
        lux: ambient illuminance (garden, A98); None = constant lab light. Photoreceptor
        responses compress roughly logarithmically over ~6 decades (Juusola & Hardie 2001):
        k = (log10 lux + 1) / 6, clipped to [0, 1], scales the bright part of the response."""
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        d = self.dirs @ R.T
        k = 1.0 if lux is None else float(np.clip((np.log10(max(lux, 1e-3)) + 1) / 6, 0, 1))
        lum = np.where(d[:, 2] > 0, 5.0 + 55.0 * k, 5.0 + 25.0 * k)
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
    def __init__(self, ann: pd.DataFrame, con=None):
        t = pd.Series(ann.type.to_numpy())
        self.idx = np.flatnonzero(t.str.match(r"^JO-B").to_numpy())
        self.side = infer_side(con, ann, self.idx, ann.side.astype(str).to_numpy()[self.idx])

    def rates(self, song_amp: float) -> np.ndarray:
        # ASSUMPTION(A55): JO-B rate ~ 150 Hz at full near-field song amplitude
        return np.full(len(self.idx), 2.0 + 150.0 * song_amp)


class WindSense:
    """Johnston's organ wind-sensitive neurons (JO-C/E, 'wind_gravity'); antennal deflection is
    larger on the side facing the wind (Yorozu et al. 2009; Suver et al. 2019).
    ASSUMPTION(A97): rate = 5 + 60 Hz per m/s, x (0.5 + 0.5 * cos(angle to that antenna))."""

    def __init__(self, ann: pd.DataFrame, con=None):
        sub = pd.Series(ann.cell_sub_class.to_numpy())
        t = pd.Series(ann.type.to_numpy())
        m = (sub == "wind_gravity") | t.str.match(r"^JO-(C|E)").to_numpy()
        self.idx = np.flatnonzero(m.to_numpy() if hasattr(m, "to_numpy") else m)
        self.side = infer_side(con, ann, self.idx, ann.side.astype(str).to_numpy()[self.idx])

    def rates(self, wind_xy, yaw) -> np.ndarray:
        w = np.asarray(wind_xy, float)
        sp = float(np.linalg.norm(w))
        out = np.full(len(self.idx), 5.0)
        if sp <= 0:
            return out
        # direction the wind comes FROM, relative to the heading (+ = from the left)
        frm = np.arctan2(-w[1], -w[0]) - yaw
        for side, sgn in (("left", 1.0), ("right", -1.0)):
            m = self.side == side
            out[m] = 5.0 + 60.0 * sp * (0.5 + 0.5 * np.cos(frm - sgn * np.pi / 4))
        return out


class Interoception:
    """Internal state enters through neurons that sense it.
    IPCs depolarise to glucose (Kreneisz et al. 2010); ASSUMPTION(A56): rate 5..40 Hz with
    haemolymph sugar (energy) level."""

    def __init__(self, ann: pd.DataFrame):
        t = pd.Series(ann.type.to_numpy())
        # insulin-producing cells: "IPC" (FlyWire, Male CNS), "m_NSC_DILP" (BANC)
        self.ipc = np.flatnonzero(t.str.match(r"^(IPC|m_NSC_DILP)$").to_numpy())
        self.idx = self.ipc

    def rates(self, energy: float) -> np.ndarray:
        return np.full(len(self.ipc), 5.0 + 35.0 * energy)
