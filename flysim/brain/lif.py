"""Event-driven leaky integrate-and-fire brain, numerically matching Shiu et al. 2024.

Model (per neuron):
    dv/dt = (v_0 - v + g) / t_mbr        (unless refractory)
    dg/dt = -g / tau                     (unless refractory)
    spike when v > v_th; reset v = v_rst, g = 0; refractory t_rfc
    presynaptic spike -> g_post += w_syn * signed_synapse_count, after t_dly

ASSUMPTION(A01-A04): see ASSUMPTIONS.md.

Only neurons that spike propagate (event-driven), so cost scales with activity, not with the
15M edges. The state update is exact for the linear ODEs (as Brian2's method='linear').
"""
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from flysim.data.flywire import Connectome


@dataclass
class LIFParams:
    # Kakaria & de Bivort 2017
    v_0: float = -52.0     # mV resting potential
    v_rst: float = -52.0   # mV reset
    v_th: float = -45.0    # mV threshold
    t_mbr: float = 20.0    # ms membrane time constant
    # Juergensen et al. 2021
    tau: float = 5.0       # ms synaptic decay
    # Lazar et al. 2021
    t_rfc: float = 2.2     # ms refractory period
    # Paul et al. 2015
    t_dly: float = 1.8     # ms synaptic delay
    # free parameter, fitted by Shiu et al. 2024
    w_syn: float = 0.275   # mV per synapse
    # stimulation: Poisson input driving v directly (Shiu et al. 2024)
    f_poi: float = 250.0
    dt: float = 0.1        # ms


@njit(cache=True)
def _run(n_steps, dt, v, g, rfc_left, rfc_len, indptr, post, w,
         decay_v, decay_g, coef_a, v_0, v_th, v_rst, delay_steps,
         stim_idx, stim_rate_per_step, stim_w, seed,
         rec_mask, rec_out):
    """rec_out: (n_steps, n_rec) uint8 spike raster for neurons with rec_mask >= 0 (column id)."""
    np.random.seed(seed)
    n = v.shape[0]
    ring = np.zeros((delay_steps + 1, n), dtype=np.float32)
    spk_count = np.zeros(n, dtype=np.int32)
    spiked = np.empty(n, dtype=np.int32)
    for step in range(n_steps):
        slot = step % (delay_steps + 1)
        # 1) integrate (exact solution of the linear system) for non-refractory neurons
        for i in range(n):
            if rfc_left[i] > 0:
                rfc_left[i] -= 1
                continue
            dv = v[i] - v_0
            gi = g[i]
            # v_dev(t+dt) = (dv - A) e^{-dt/tm} + A e^{-dt/tau}, A = g * tau/(tau - tm)
            a = gi * coef_a
            v[i] = v_0 + (dv - a) * decay_v + a * decay_g
            g[i] = gi * decay_g
        # 2) threshold
        ns = 0
        for i in range(n):
            if v[i] > v_th and rfc_left[i] == 0:
                spiked[ns] = i
                ns += 1
        # 3) synaptic delivery due at this step + Poisson drive
        for i in range(n):
            if ring[slot, i] != 0.0:
                g[i] += ring[slot, i]
                ring[slot, i] = 0.0
        for k in range(stim_idx.shape[0]):
            if np.random.random() < stim_rate_per_step[k]:
                v[stim_idx[k]] += stim_w
        # 4) reset + schedule outgoing spikes
        out_slot = (step + delay_steps) % (delay_steps + 1)
        for s in range(ns):
            i = spiked[s]
            v[i] = v_rst
            g[i] = 0.0
            rfc_left[i] = rfc_len[i]
            spk_count[i] += 1
            c = rec_mask[i]
            if c >= 0:
                rec_out[step, c] = 1
            for e in range(indptr[i], indptr[i + 1]):
                ring[out_slot, post[e]] += w[e]
    return spk_count


@dataclass
class Brain:
    con: Connectome
    p: LIFParams = field(default_factory=LIFParams)

    def __post_init__(self):
        n = self.con.n
        self.w = (self.con.syn.astype(np.float32) * np.float32(self.p.w_syn))
        self.v = np.full(n, self.p.v_0, dtype=np.float64)
        self.g = np.zeros(n, dtype=np.float64)
        self.rfc_left = np.zeros(n, dtype=np.int32)
        self.rfc_len = np.full(n, int(round(self.p.t_rfc / self.p.dt)), dtype=np.int32)

    def reset(self):
        self.v[:] = self.p.v_0
        self.g[:] = 0
        self.rfc_left[:] = 0

    def run(self, t_ms: float, stim: dict | None = None, record=None, seed: int = 0):
        """stim: {neuron_index: rate_Hz}. Returns (spike counts per neuron, raster of `record`)."""
        p = self.p
        n_steps = int(round(t_ms / p.dt))
        stim = stim or {}
        stim_idx = np.array(list(stim.keys()), dtype=np.int64)
        stim_rate = np.array([r * p.dt * 1e-3 for r in stim.values()], dtype=np.float64)
        rfc_len = self.rfc_len.copy()
        rfc_len[stim_idx] = 0  # Shiu et al.: no refractory period for Poisson targets
        record = np.asarray(record if record is not None else [], dtype=np.int64)
        rec_mask = np.full(self.con.n, -1, dtype=np.int32)
        rec_mask[record] = np.arange(len(record), dtype=np.int32)
        rec_out = np.zeros((n_steps, len(record)), dtype=np.uint8)
        decay_v = np.exp(-p.dt / p.t_mbr)
        decay_g = np.exp(-p.dt / p.tau)
        coef_a = p.tau / (p.tau - p.t_mbr)
        counts = _run(n_steps, p.dt, self.v, self.g, self.rfc_left, rfc_len,
                      self.con.indptr, self.con.post, self.w,
                      decay_v, decay_g, coef_a, p.v_0, p.v_th, p.v_rst,
                      int(round(p.t_dly / p.dt)),
                      stim_idx, stim_rate, p.w_syn * p.f_poi, seed, rec_mask, rec_out)
        return counts, rec_out
