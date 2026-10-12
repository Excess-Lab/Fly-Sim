"""Build a ready-to-use brain for a given sex, with all biological corrections applied."""
from dataclasses import replace

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


def make_brain(sex: str, seed: int = 0) -> tuple[Brain, object]:
    con, ann = load(sex)
    p = LIFParams()
    ratio = {"male": MALE_SYNAPSE_RATIO, "female_banc": BANC_SYNAPSE_RATIO}.get(sex)
    if ratio:
        # keep the Poisson stimulation kick identical (w_syn * f_poi = 68.75 mV)
        p = replace(p, w_syn=p.w_syn / ratio, f_poi=p.f_poi * ratio)
    b = Brain(con, p, seed=seed)
    b.apply_point_neuron_corrections(ann)
    if sex == "female_banc":
        b.gain[ann.super_class.str.startswith("vnc").to_numpy()] = BANC_VNC_GAIN
    return b, ann
