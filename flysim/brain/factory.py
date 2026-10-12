"""Build a ready-to-use brain for a given sex, with all biological corrections applied."""
from dataclasses import replace

from flysim.brain.lif import Brain, LIFParams
from flysim.data import annotations as A

# ASSUMPTION(A09): Janelia (Male CNS) detects 1.78x more input synapses per neuron than FlyWire
# for the same cell type (median over 7220 matched types, IQR 1.42-2.35; scripts in repo).
# Synaptic weight per synapse is divided by this factor so that the per-neuron drive matches the
# FlyWire-calibrated model of Shiu et al. 2024.
MALE_SYNAPSE_RATIO = 1.783

_cache = {}


def load(sex: str):
    if sex not in _cache:
        _cache[sex] = A.load(sex)
    return _cache[sex]


def make_brain(sex: str, seed: int = 0) -> tuple[Brain, object]:
    con, ann = load(sex)
    p = LIFParams()
    if sex == "male":
        # keep the Poisson stimulation kick identical (w_syn * f_poi = 68.75 mV)
        p = replace(p, w_syn=p.w_syn / MALE_SYNAPSE_RATIO, f_poi=p.f_poi * MALE_SYNAPSE_RATIO)
    b = Brain(con, p, seed=seed)
    b.apply_point_neuron_corrections(ann)
    return b, ann
