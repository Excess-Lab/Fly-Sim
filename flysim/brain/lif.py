"""Event-driven leaky integrate-and-fire brain, numerically matching Shiu et al. 2024.

Model (per neuron):
    dv/dt = (v_0 - v + g) / t_mbr        (unless refractory)
    dg/dt = -g / tau                     (unless refractory)
    spike when v > v_th; reset v = v_rst, g = 0; refractory t_rfc
    presynaptic spike -> g_post += w_syn * signed_synapse_count, after t_dly

ASSUMPTION(A01-A04): see ASSUMPTIONS.md.

Only neurons that spike propagate (event-driven), so cost scales with activity, not with the
15M edges. The state update is exact for the linear ODEs (as Brian2's method='linear').
State persists between calls to `step`, so the brain can be advanced in chunks inside a
closed sensorimotor loop.
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
def _step(n_steps, t0, v, g, rfc_left, rfc_len, ring, indptr, post, w, gain,
          decay_v, decay_g, coef_a, v_0, v_th, v_rst,
          stim_idx, stim_p, stim_w, kick_step, kick_idx, rng_state,
          counts, rec_mask, rec_out):
    n = v.shape[0]
    D = ring.shape[0]
    spiked = np.empty(n, dtype=np.int32)
    frozen = np.zeros(n, dtype=np.uint8)
    k = 0
    np.random.seed(rng_state)
    for s in range(n_steps):
        step = t0 + s
        slot = step % D
        # 1) integrate (exact solution of the linear system) for non-refractory neurons
        for i in range(n):
            if rfc_left[i] > 0:
                rfc_left[i] -= 1
                frozen[i] = 1
                continue
            frozen[i] = 0
            dv = v[i] - v_0
            a = g[i] * coef_a
            v[i] = v_0 + (dv - a) * decay_v + a * decay_g
            g[i] = g[i] * decay_g
        # 2) threshold
        ns = 0
        for i in range(n):
            if v[i] > v_th and frozen[i] == 0:
                spiked[ns] = i
                ns += 1
        # 3) synaptic delivery due now, Poisson drive, scheduled kicks
        for i in range(n):
            x = ring[slot, i]
            if x != 0.0:
                # ASSUMPTION(A05): input arriving while refractory is discarded (Brian2 semantics
                # of 'unless refractory' used by Shiu et al. 2024)
                if frozen[i] == 0:
                    g[i] += x * gain[i]
                ring[slot, i] = 0.0
        for j in range(stim_idx.shape[0]):
            if stim_p[j] > 0.0 and np.random.random() < stim_p[j]:
                v[stim_idx[j]] += stim_w
        while k < kick_step.shape[0] and kick_step[k] == s:
            v[kick_idx[k]] += stim_w
            k += 1
        # 4) reset + schedule outgoing spikes
        out_slot = (step + D - 1) % D
        for q in range(ns):
            i = spiked[q]
            v[i] = v_rst
            g[i] = 0.0
            rfc_left[i] = rfc_len[i]
            counts[i] += 1
            c = rec_mask[i]
            if c >= 0:
                rec_out[s, c] = 1
            for e in range(indptr[i], indptr[i + 1]):
                ring[out_slot, post[e]] += w[e]


@dataclass
class Brain:
    con: Connectome
    p: LIFParams = field(default_factory=LIFParams)
    seed: int = 0

    def __post_init__(self):
        n = self.con.n
        p = self.p
        self.w = self.con.syn.astype(np.float32) * np.float32(p.w_syn)
        # per-neuron multiplicative gain on synaptic input: hook for neuromodulation (stage 3)
        self.gain = np.ones(n, dtype=np.float32)
        # Brian2: neuron is frozen while timestep(t - lastspike) < timestep(t_rfc), i.e. for
        # t_rfc/dt - 1 steps after the spike step
        self.base_rfc = np.full(n, max(int(round(p.t_rfc / p.dt)) - 1, 0), dtype=np.int32)
        self.rfc_len = self.base_rfc.copy()
        self.delay_steps = int(round(p.t_dly / p.dt))
        self._decay_v = np.exp(-p.dt / p.t_mbr)
        self._decay_g = np.exp(-p.dt / p.tau)
        self._coef_a = p.tau / (p.tau - p.t_mbr)
        self._rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self, seed: int | None = None):
        n = self.con.n
        self.v = np.full(n, self.p.v_0, dtype=np.float64)
        self.g = np.zeros(n, dtype=np.float64)
        self.rfc_left = np.zeros(n, dtype=np.int32)
        # ring slot k holds input arriving at steps == k (mod D); D = delay+1 so that a spike at
        # step s is delivered at step s + delay
        self.ring = np.zeros((self.delay_steps + 1, n), dtype=np.float32)
        self.t_step = 0
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def set_poisson_targets(self, idx):
        """Neurons that receive Poisson drive have no refractory period (Shiu et al. 2024)."""
        self.rfc_len = self.base_rfc.copy()
        self.rfc_len[np.asarray(idx, dtype=np.int64)] = 0

    def step(self, n_steps: int, stim_idx=None, stim_rate_hz=None, kicks=None, record=None):
        """Advance `n_steps`. stim: Poisson rate per target neuron (Hz).
        kicks: (step_offsets, neuron_idx) deterministic supra-threshold inputs, sorted by step.
        Returns (spike counts per neuron in this chunk, raster of `record`)."""
        p = self.p
        stim_idx = np.asarray(stim_idx if stim_idx is not None else [], dtype=np.int64)
        rates = np.asarray(stim_rate_hz if stim_rate_hz is not None else [], dtype=np.float64)
        rates = np.broadcast_to(rates, stim_idx.shape) if rates.ndim == 0 else rates
        stim_p = rates * p.dt * 1e-3
        if kicks is None:
            ks, ki = np.zeros(0, np.int64), np.zeros(0, np.int64)
        else:
            ks, ki = (np.asarray(x, dtype=np.int64) for x in kicks)
        record = np.asarray(record if record is not None else [], dtype=np.int64)
        rec_mask = np.full(self.con.n, -1, dtype=np.int32)
        rec_mask[record] = np.arange(len(record), dtype=np.int32)
        rec_out = np.zeros((n_steps, len(record)), dtype=np.uint8)
        counts = np.zeros(self.con.n, dtype=np.int32)
        _step(n_steps, self.t_step, self.v, self.g, self.rfc_left, self.rfc_len, self.ring,
              self.con.indptr, self.con.post, self.w, self.gain,
              self._decay_v, self._decay_g, self._coef_a, p.v_0, p.v_th, p.v_rst,
              stim_idx, stim_p, p.w_syn * p.f_poi, ks, ki,
              int(self._rng.integers(2**31 - 1)), counts, rec_mask, rec_out)
        self.t_step += n_steps
        return counts, rec_out

    def run(self, t_ms: float, stim: dict | None = None, record=None, seed: int | None = None):
        """Convenience: fresh trial with constant Poisson stimulus {neuron_index: rate_Hz}."""
        self.reset(seed)
        stim = stim or {}
        idx = np.array(list(stim.keys()), dtype=np.int64)
        self.set_poisson_targets(idx)
        return self.step(int(round(t_ms / self.p.dt)), idx,
                         np.array(list(stim.values()), dtype=np.float64), record=record)
