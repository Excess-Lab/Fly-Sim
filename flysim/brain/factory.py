"""Build a ready-to-use brain for a given sex, with all biological corrections applied."""
from dataclasses import replace

import numpy as np

from flysim.data.flywire import PROCESSED

from flysim.brain.lif import Brain, LIFParams
from flysim.data import annotations as A

# ASSUMPTION(A09): Janelia (Male CNS) detects 1.78x more input synapses per neuron than FlyWire
# for the same cell type (median over 7220 matched types, IQR 1.42-2.35; scripts in repo).
# Synaptic weight per synapse is divided by this factor so that the per-neuron drive matches the
# FlyWire-calibrated model of Shiu et al. 2024.
MALE_SYNAPSE_RATIO = 1.783
# ASSUMPTION(A09b): BANC detects 0.588x the input synapses per neuron of FlyWire for the same cell
# type (median over 7508 matched types, IQR 0.41-0.84) -> weight per synapse / 0.588.
BANC_SYNAPSE_RATIO = 0.588
# ASSUMPTION(A09c): BANC VNC detects 0.456x the input synapses of Male CNS VNC for matched types
# (2612 types) = 0.814x on the FlyWire scale; VNC inputs are therefore scaled by 0.588/0.814.
BANC_VNC_GAIN = 0.588 / (0.456 * MALE_SYNAPSE_RATIO)

_cache = {}
_lock = __import__("threading").Lock()


def load(sex: str):
    if sex not in ("female", "male", "female_banc"):
        raise ValueError(f"unknown sex {sex!r}")
    with _lock:  # never build two multi-GB connectomes at once (8 GB RAM)
        if sex not in _cache:
            _cache[sex] = A.load(sex)
    return _cache[sex]


READOUT_TYPES = r"^(MN9|CB0701|PPL1|PAM|MBON|DM1_lPN|DM4_lPN)"


def male_type_gains(b, ann):
    """ASSUMPTION(A09e): per-cell-type calibration of the male brain against FlyWire (same method
    as A09d). A single global factor left whole pathways silent (e.g. sugar -> MN9 0 Hz); VNC
    neurons keep the global Male CNS scale (no FlyWire VNC reference)."""
    import pandas as pd
    cache = PROCESSED / "male_type_gains.npy"
    if cache.exists():
        g = np.load(cache)
        if len(g) == b.con.n:
            return g
    def insyn(con, w):
        return np.bincount(con.post, weights=np.abs(w), minlength=con.n)
    fcon, fann = load("female")
    fb = Brain(fcon, LIFParams())
    fb.apply_point_neuron_corrections(fann)
    ref = pd.DataFrame({"t": fann.type, "x": insyn(fcon, fb.w)})[lambda d: d.t != ""].groupby("t").x.median()
    mt = ann.flywire_type.str.split(",").str[0].str.strip()
    mt = mt.where(mt != "", ann.type)
    mdf = pd.DataFrame({"t": mt.to_numpy(), "x": insyn(b.con, b.w)})
    mmed = mdf[mdf.t != ""].groupby("t").x.median()
    ratio = (ref / mmed).replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[mmed.reindex(ratio.index) > 20]
    g = np.array(mdf.t.map(ratio).fillna(1.0).clip(0.3, 3.0).to_numpy(), dtype=np.float64, copy=True)
    g[ann.super_class.str.startswith("vnc").to_numpy()] = 1.0
    np.save(cache, g)
    return g


def banc_type_gains(b, ann):
    """ASSUMPTION(A09d): per-cell-type calibration of BANC. Each BANC neuron's synaptic input is
    scaled so that the median total input of its cell type equals that of the same type in the
    validated reference (FlyWire for brain types, Male CNS for VNC types, in FlyWire units).
    Types without a match get the global ratio. Gains clipped to [0.3, 3]."""
    import pandas as pd
    cache = PROCESSED / "banc_type_gains.npy"
    if cache.exists():
        g = np.load(cache)
        if len(g) == b.con.n:
            return g
    def insyn(con, w):
        return np.bincount(con.post, weights=np.abs(w), minlength=con.n)
    bin_ = insyn(b.con, b.w)
    fcon, fann = load("female")
    fb = Brain(fcon, LIFParams())
    fb.apply_point_neuron_corrections(fann)
    fin = insyn(fcon, fb.w)
    mcon, mann = load("male")
    mb = Brain(mcon, replace(LIFParams(), w_syn=LIFParams().w_syn / MALE_SYNAPSE_RATIO))
    mb.apply_point_neuron_corrections(mann)
    min_ = insyn(mcon, mb.w)
    ref = pd.concat([
        pd.DataFrame({"t": fann.type, "x": fin})[lambda d: d.t != ""],
        pd.DataFrame({"t": mann.type, "x": min_})[lambda d: (d.t != "") & mann.super_class.str.startswith("vnc")],
    ]).groupby("t").x.median()
    bt = ann.flywire_type.where(ann.flywire_type != "", ann.type)
    bdf = pd.DataFrame({"t": bt.to_numpy(), "x": bin_})
    bmed = bdf[bdf.t != ""].groupby("t").x.median()
    ratio = (ref / bmed).replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[(bmed.reindex(ratio.index) > 20)]
    glob = float(np.exp(np.log(ratio).median()))
    g = bdf.t.map(ratio).fillna(glob).clip(0.3, 3.0).to_numpy()
    np.save(cache, g)
    return g


def make_brain(sex: str, seed: int = 0) -> tuple[Brain, object]:
    con, ann = load(sex)
    p = LIFParams()
    ratio = {"male": MALE_SYNAPSE_RATIO}.get(sex)   # BANC: per-type gains instead (A09d)
    if ratio:
        # keep the Poisson stimulation kick identical (w_syn * f_poi = 68.75 mV)
        p = replace(p, w_syn=p.w_syn / ratio, f_poi=p.f_poi * ratio)
    b = Brain(con, p, seed=seed)
    b.apply_point_neuron_corrections(ann)
    if sex == "female_banc":
        b.gain[:] = banc_type_gains(b, ann).astype(np.float32)
    elif sex == "male":
        b.gain[:] = male_type_gains(b, ann).astype(np.float32)
    if sex in ("female_banc", "male"):
        # ASSUMPTION(A09f): the neurons used as validation read-outs (proboscis motor neuron,
        # reinforcement DANs, MBONs, tested PNs) are NOT calibrated per type: they get the
        # dataset's global scale, so tests on them are not built in by the calibration.
        ro = ann.type.astype(str).str.match(READOUT_TYPES).to_numpy()
        b.gain[ro] = np.float32(np.median(b.gain) if sex == "female_banc" else 1.0)
    return b, ann
