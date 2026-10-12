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
from numba import njit, prange

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


N_CHUNKS = 64


@njit(cache=True, parallel=True)
def _update_chunks(n, slot, v, g, rfc_left, frozen, ring, gain, decay_v, decay_g, coef_a,
                   v_0, v_th, spk_buf, spk_n):
    """Fused, parallel: integrate -> threshold -> deliver due synaptic input, per neuron.
    Delivered input changes g only, so it cannot affect this step's threshold (as in Brian2)."""
    csize = (n + N_CHUNKS - 1) // N_CHUNKS
    for c in prange(N_CHUNKS):
        lo = c * csize
        hi = min(n, lo + csize)
        m = 0
        for i in range(lo, hi):
            if rfc_left[i] > 0:
                rfc_left[i] -= 1
                frozen[i] = 1
            else:
                frozen[i] = 0
                gi = g[i]
                if gi != 0.0 or v[i] != v_0:
                    a = gi * coef_a
                    v[i] = v_0 + (v[i] - v_0 - a) * decay_v + a * decay_g
                    g[i] = gi * decay_g
                if v[i] > v_th:
                    spk_buf[lo + m] = i
                    m += 1
            x = ring[slot, i]
            if x != 0.0:
                # ASSUMPTION(A05): input arriving while refractory is discarded (Brian2 semantics
                # of 'unless refractory' used by Shiu et al. 2024)
                if frozen[i] == 0:
                    g[i] += x * gain[i]
                ring[slot, i] = 0.0
        spk_n[c] = m


PLAST_EVERY = 10  # steps (1 ms)


@njit(cache=True)
def _plasticity(x_tr, da_tr, w, pe_edge, pe_pre, pe_m, pe_w0, kc_idx, c):
    """Dopamine-gated KC->MBON plasticity, one update per PLAST_EVERY steps.
    c = [dt_s, decay_x, decay_da, eta_dep, eta_rec, 1/tau_forget]
      dw/dt = -eta_dep * x_KC * DA_m * w          coincidence -> depression (Hige 2015, Cohn 2015)
              + eta_rec * x_KC * (w0 - w)          KC activity alone -> recovery (Cohn 2015, Berry 2018)
              - (w - w0) / tau_forget              passive forgetting
    ASSUMPTION(A20-A23): rule form and constants, see ASSUMPTIONS.md.
    """
    dt = c[0]
    for p in range(pe_edge.shape[0]):
        e = pe_edge[p]
        wv = w[e]
        xk = x_tr[pe_pre[p]]
        w0 = pe_w0[p]
        w[e] = wv + (-c[3] * xk * da_tr[pe_m[p]] * wv + c[4] * xk * (w0 - wv)
                     - (wv - w0) * c[5]) * dt
    for k in range(kc_idx.shape[0]):
        x_tr[kc_idx[k]] *= c[1]
    for i in range(da_tr.shape[0]):
        da_tr[i] *= c[2]


@njit(cache=True)
def _step(n_steps, t0, v, g, rfc_left, rfc_len, ring, indptr, post, w, gain,
          decay_v, decay_g, coef_a, v_0, v_th, v_rst,
          stim_idx, stim_p, stim_w, kick_step, kick_idx, rng_state,
          counts, rec_mask, rec_out,
          pl_on, x_tr, da_tr, dan_ptr, dan_m, dan_a, pe_edge, pe_pre, pe_m, pe_w0, kc_idx, pl_c):
    n = v.shape[0]
    D = ring.shape[0]
    csize = (n + N_CHUNKS - 1) // N_CHUNKS
    spk_buf = np.empty(n, dtype=np.int32)
    spk_n = np.zeros(N_CHUNKS, dtype=np.int64)
    frozen = np.zeros(n, dtype=np.uint8)
    k = 0
    np.random.seed(rng_state)
    for s in range(n_steps):
        step = t0 + s
        slot = step % D
        _update_chunks(n, slot, v, g, rfc_left, frozen, ring, gain, decay_v, decay_g, coef_a,
                       v_0, v_th, spk_buf, spk_n)
        # Poisson drive and scheduled kicks act on v after the threshold (Brian2 'synapses' slot)
        for j in range(stim_idx.shape[0]):
            if stim_p[j] > 0.0 and np.random.random() < stim_p[j]:
                v[stim_idx[j]] += stim_w
        while k < kick_step.shape[0] and kick_step[k] == s:
            v[kick_idx[k]] += stim_w
            k += 1
        # reset + schedule outgoing spikes
        out_slot = (step + D - 1) % D
        for c in range(N_CHUNKS):
            for q in range(spk_n[c]):
                i = spk_buf[c * csize + q]
                v[i] = v_rst
                g[i] = 0.0
                rfc_left[i] = rfc_len[i]
                counts[i] += 1
                r = rec_mask[i]
                if r >= 0:
                    rec_out[s, r] = 1
                for e in range(indptr[i], indptr[i + 1]):
                    ring[out_slot, post[e]] += w[e]
                if pl_on:
                    x_tr[i] += 1.0
                    for j in range(dan_ptr[i], dan_ptr[i + 1]):
                        da_tr[dan_m[j]] += dan_a[j]
        if pl_on and (step + 1) % PLAST_EVERY == 0:
            _plasticity(x_tr, da_tr, w, pe_edge, pe_pre, pe_m, pe_w0, kc_idx, pl_c)


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
        self.disable_plasticity()
        self.reset()

    def apply_point_neuron_corrections(self, ann):
        """Remove synapses that cannot drive spikes in a point-neuron model.

        ASSUMPTION(A06): sensory neurons fire from peripheral transduction; central (axo-axonic)
        input onto their axon terminals modulates release but does not generate spikes
        (e.g. presynaptic inhibition of ORNs, Olsen & Wilson 2008). Input to sensory neurons = 0.
        ASSUMPTION(A07): KC->KC synapses are axo-axonic in the MB lobes, while KC spikes start
        near the calyx; they do not drive KC spiking. Removed.
        Without these corrections the model falls into a self-sustaining state with ~70% of KCs
        and ORNs of all glomeruli active, contradicting sparse KC coding (Honegger et al. 2011).

        ASSUMPTION(A08): dopamine, serotonin and octopamine act through metabotropic receptors
        (slow modulation), not fast ionotropic EPSPs; their fast weights are 0. Dopamine keeps its
        role as the teaching signal of KC->MBON plasticity (enable_plasticity).

        ASSUMPTION(A03): sign from the resolved transmitter of the presynaptic neuron:
        acetylcholine +, GABA / glutamate / histamine -.
        """
        nt = ann.nt.str.lower().to_numpy()
        sign = np.where(np.isin(nt, ["gaba", "glutamate", "histamine"]), -1.0, 1.0)
        mono = np.isin(nt, ["dopamine", "serotonin", "octopamine"])
        sensory = ann.super_class.str.contains("sensory").to_numpy()
        kc = ann.type.str.match("^KC").to_numpy()
        pre = np.repeat(np.arange(self.con.n), np.diff(self.con.indptr))
        post = self.con.post
        self.w = (np.abs(self.con.syn) * sign[pre] * self.p.w_syn).astype(np.float32)
        drop = sensory[post] | (kc[pre] & kc[post]) | mono[pre]
        self.w[drop] = 0.0
        self.n_dropped = int(drop.sum())
        return self.n_dropped

    def disable_plasticity(self):
        z64, z32, zf = np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.float32)
        self.pl_on = False
        self.x_tr = np.zeros(1, np.float32)
        self.da_tr = zf.copy()
        self.dan_ptr = np.zeros(self.con.n + 1, np.int64)
        self.dan_m, self.dan_a = z32, zf
        self.pe_edge, self.pe_pre, self.pe_m, self.pe_w0 = z64, z32, z32, zf
        self.kc_idx = z64
        self.pl_c = np.zeros(6, np.float64)

    def enable_plasticity(self, kc, mbon, dan, tau_elig_s=1.0, tau_da_s=0.5,
                          eta_dep=2.0, eta_rec=0.02, tau_forget_s=3 * 3600.0):
        """Make KC->MBON synapses plastic, gated by dopamine.

        Which DAN modulates which MBON is taken from the connectome itself: a DAN spike adds
        dopamine to MBON m in proportion to its synapse count onto m, relative to its strongest
        MBON target (DANs and MBONs share MB compartments; ASSUMPTION A20).
        """
        con = self.con
        kc, mbon, dan = (np.asarray(x, np.int64) for x in (kc, mbon, dan))
        n = con.n
        mslot = np.full(n, -1, np.int64)
        mslot[mbon] = np.arange(len(mbon))
        is_kc = np.zeros(n, bool)
        is_kc[kc] = True
        # plastic edges KC -> MBON
        edges, pres = [], []
        for i in kc:
            lo, hi = con.indptr[i], con.indptr[i + 1]
            tgt = con.post[lo:hi]
            sel = np.flatnonzero(mslot[tgt] >= 0)
            edges.append(lo + sel)
            pres.append(np.full(len(sel), i))
        self.pe_edge = np.concatenate(edges).astype(np.int64)
        self.pe_pre = np.concatenate(pres).astype(np.int32)
        self.pe_m = mslot[con.post[self.pe_edge]].astype(np.int32)
        self.pe_w0 = self.w[self.pe_edge].copy()
        # DAN -> MBON compartment map from synapse counts
        ptr = np.zeros(n + 1, np.int64)
        ms, amts = [], []
        for i in dan:
            lo, hi = con.indptr[i], con.indptr[i + 1]
            tgt, cnt = con.post[lo:hi], np.abs(con.syn[lo:hi]).astype(np.float64)
            sel = mslot[tgt] >= 0
            if sel.any():
                m, a = mslot[tgt[sel]], cnt[sel] / cnt[sel].max()
                ms.append(m)
                amts.append(a)
                ptr[i + 1] = len(m)
        self.dan_ptr = np.cumsum(ptr)
        # entries must be ordered by neuron index to match the CSR pointer
        order = np.argsort([i for i in dan if self.dan_ptr[i + 1] > self.dan_ptr[i]])
        ms = [ms[k] for k in order]
        amts = [amts[k] for k in order]
        self.dan_m = (np.concatenate(ms) if ms else np.zeros(0)).astype(np.int32)
        self.dan_a = (np.concatenate(amts) if amts else np.zeros(0)).astype(np.float32)
        self.x_tr = np.zeros(n, np.float32)
        self.da_tr = np.zeros(len(mbon), np.float32)
        self.kc_idx = kc
        dt_s = PLAST_EVERY * self.p.dt * 1e-3
        self.pl_c = np.array([dt_s, np.exp(-dt_s / tau_elig_s), np.exp(-dt_s / tau_da_s),
                              eta_dep, eta_rec, 1.0 / tau_forget_s])
        self.pl_on = True
        self.mbon_idx = mbon

    def forget(self, seconds: float):
        """Fast-forward a resting period: only passive forgetting acts (ASSUMPTION A23)."""
        if self.pl_on:
            f = np.exp(-seconds * self.pl_c[5])
            self.w[self.pe_edge] = self.pe_w0 + (self.w[self.pe_edge] - self.pe_w0) * f

    def kc_mbon_strength(self) -> np.ndarray:
        """Relative KC->MBON weight per MBON (1 = naive)."""
        num = np.bincount(self.pe_m, self.w[self.pe_edge], minlength=len(self.da_tr))
        den = np.bincount(self.pe_m, self.pe_w0, minlength=len(self.da_tr))
        return num / np.maximum(den, 1e-9)

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
              int(self._rng.integers(2**31 - 1)), counts, rec_mask, rec_out,
              self.pl_on, self.x_tr, self.da_tr, self.dan_ptr, self.dan_m, self.dan_a,
              self.pe_edge, self.pe_pre, self.pe_m, self.pe_w0, self.kc_idx, self.pl_c)
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
