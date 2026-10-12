import numpy as np

from flysim.brain.lif import Brain, LIFParams
from flysim.data.flywire import Connectome


def tiny(syn_count: int) -> Connectome:
    # neuron 0 -> neuron 1 with `syn_count` signed synapses
    return Connectome(ids=np.array([10, 11], dtype=np.int64),
                      indptr=np.array([0, 1, 1], dtype=np.int64),
                      post=np.array([1], dtype=np.int32),
                      syn=np.array([syn_count], dtype=np.int32))


def test_silent_without_input():
    counts, _ = Brain(tiny(50)).run(200.0)
    assert counts.sum() == 0


def test_synaptic_delay_and_drive():
    b = Brain(tiny(200))
    b.v[0] = -44.0  # above threshold: neuron 0 spikes on step 0
    _, rast = b.step(200, record=[0, 1])
    t0 = np.flatnonzero(rast[:, 0])[0]
    t1 = np.flatnonzero(rast[:, 1])[0]
    delay_steps = round(LIFParams().t_dly / LIFParams().dt)
    assert t0 == 0
    assert t1 > t0 + delay_steps  # postsynaptic spike only after the 1.8 ms delay


def test_inhibition_prevents_spike():
    b = Brain(tiny(-200))
    b.v[0] = -44.0
    counts, _ = b.step(200)
    assert counts[1] == 0


def test_psp_matches_analytic_solution():
    # single EPSP: v(t) - v0 = A (e^{-t/tau} - e^{-t/tm}), A = g0 tau/(tau - tm)
    p = LIFParams()
    b = Brain(tiny(1))
    b.g[1] = 1.0
    b.step(50)
    t = 5.0
    a = 1.0 * p.tau / (p.tau - p.t_mbr)
    expected = p.v_0 + a * (np.exp(-t / p.tau) - np.exp(-t / p.t_mbr))
    assert abs(b.v[1] - expected) < 1e-9
