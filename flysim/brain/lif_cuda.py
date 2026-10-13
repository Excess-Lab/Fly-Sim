"""NVIDIA GPU backend for the LIF brain (Numba CUDA).

Same model and same order of operations as the CPU kernel in lif.py, one time step at a time:
  1. integrate / threshold / deliver due synaptic input        (one thread per neuron)
  2. electrical synapses from a snapshot of v                   (one thread per pair)
  3. Poisson drive and scheduled kicks                          (one thread per target)
  4. spikes: reset, propagate to the delay ring, spikelets,
     plasticity traces                                          (one thread per spiking neuron)
  5. every PLAST_EVERY steps: KC->MBON plasticity               (compartments, then edges)
All state stays on the GPU between calls; only spike counts (and recorded rasters) come back.
Differences from the CPU backend: Poisson random numbers come from a GPU generator (same
statistics, different draws), and float32 sums into the delay ring may round in another order.
With deterministic kicks and no Poisson input the two backends give the same spikes (tested
with the CUDA simulator, tests/test_cuda.py).

Use: FLYSIM_BACKEND=cuda (or auto: GPU if available, else CPU).
"""
import math

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64

from flysim.brain.lif import PLAST_EVERY

TPB = 256


def available() -> bool:
    try:
        return cuda.is_available()
    except Exception:
        return False


def _blocks(n):
    return max(1, (n + TPB - 1) // TPB)


@cuda.jit
def k_update(slot, v, g, rfc_left, frozen, ring, gain, decay_v, decay_g, coef_a, v_rest, v_th,
             spk_list, spk_n, wad, decay_w, coef_w):
    i = cuda.grid(1)
    if i >= v.shape[0]:
        return
    if rfc_left[i] > 0:
        rfc_left[i] -= 1
        frozen[i] = 1
    else:
        frozen[i] = 0
        gi = g[i]
        wi = wad[i]
        v0i = v_rest[i]
        if gi != 0.0 or v[i] != v0i or wi != 0.0:
            a = gi * coef_a
            aw = -wi * coef_w
            v[i] = v0i + (v[i] - v0i - a - aw) * decay_v + a * decay_g + aw * decay_w
            g[i] = gi * decay_g
            wad[i] = wi * decay_w
        if v[i] > v_th:
            k = cuda.atomic.add(spk_n, 0, 1)
            spk_list[k] = i
    x = ring[slot, i]
    if x != 0.0:
        if frozen[i] == 0:
            g[i] += x * gain[i]
        ring[slot, i] = 0.0


@cuda.jit
def k_gap_snapshot(v, gj_a, gj_b, gj_k, gj_dv):
    q = cuda.grid(1)
    if q < gj_a.shape[0]:
        gj_dv[q] = gj_k[q] * (v[gj_b[q]] - v[gj_a[q]])


@cuda.jit
def k_gap_apply(v, frozen, gj_a, gj_b, gj_dv):
    q = cuda.grid(1)
    if q < gj_a.shape[0]:
        if frozen[gj_a[q]] == 0:
            cuda.atomic.add(v, gj_a[q], gj_dv[q])
        if frozen[gj_b[q]] == 0:
            cuda.atomic.add(v, gj_b[q], -gj_dv[q])


@cuda.jit
def k_poisson(v, stim_idx, stim_p, stim_w, rng):
    j = cuda.grid(1)
    if j < stim_idx.shape[0]:
        if stim_p[j] > 0.0 and xoroshiro128p_uniform_float64(rng, j) < stim_p[j]:
            cuda.atomic.add(v, stim_idx[j], stim_w)


@cuda.jit
def k_kicks(v, kick_idx, lo, hi, stim_w):
    j = cuda.grid(1) + lo
    if j < hi:
        cuda.atomic.add(v, kick_idx[j], stim_w)


@cuda.jit
def k_spikes(spk_list, spk_n, out_slot, s, v, g, wad, b_w, rfc_left, rfc_len, counts, rec_mask,
             rec_out, indptr, post, w, ring, gk_ptr, gk_tgt, gk_amt, v_rst,
             pl_on, x_tr, kc_new, dan_ptr, dan_m, dan_a, da_tr, da_new):
    q = cuda.grid(1)
    if q >= spk_n[0]:
        return
    i = spk_list[q]
    v[i] = v_rst
    g[i] = 0.0
    wad[i] += b_w
    rfc_left[i] = rfc_len[i]
    counts[i] += 1
    r = rec_mask[i]
    if r >= 0:
        rec_out[s, r] = 1
    for e in range(indptr[i], indptr[i + 1]):
        cuda.atomic.add(ring, (out_slot, post[e]), w[e])
    if pl_on:
        x_tr[i] += 1.0
        kc_new[i] += 1.0
        for j in range(dan_ptr[i], dan_ptr[i + 1]):
            cuda.atomic.add(da_tr, dan_m[j], dan_a[j])
            cuda.atomic.add(da_new, dan_m[j], dan_a[j])


@cuda.jit
def k_spikelets(spk_list, spk_n, rfc_left, gk_ptr, gk_tgt, gk_amt, v):
    # after resets, like the CPU loop (targets reset in this step are refractory and skipped)
    q = cuda.grid(1)
    if q >= spk_n[0]:
        return
    i = spk_list[q]
    for e in range(gk_ptr[i], gk_ptr[i + 1]):
        t = gk_tgt[e]
        if rfc_left[t] == 0:
            cuda.atomic.add(v, t, gk_amt[e])


@cuda.jit
def k_gather(w, idx, out):
    q = cuda.grid(1)
    if q < idx.shape[0]:
        out[q] = w[idx[q]]


@cuda.jit
def k_plast_gate(da_tr, da_new, da_base, c, gate, gate_b, lam):
    i = cuda.grid(1)
    if i < da_tr.shape[0]:
        prev = da_tr[i] - da_new[i]
        da_base[i] = lam * da_base[i] + (1.0 - lam) * da_tr[i]
        thr = c[10] * da_base[i]
        gate[i] = max(0.0, da_tr[i] - thr) / da_tr[i] if da_tr[i] > 0.0 else 0.0
        gate_b[i] = max(0.0, prev - thr) if prev > 0.0 else 0.0


@cuda.jit
def k_plast_edges(x_tr, kc_new, da_new, gate, gate_b, w, pe_edge, pe_pre, pe_m, pe_w0, c):
    p = cuda.grid(1)
    if p >= pe_edge.shape[0]:
        return
    dt = c[0]
    e = pe_edge[p]
    k_i = pe_pre[p]
    m = pe_m[p]
    x_before = x_tr[k_i] - kc_new[k_i]
    da_ph = da_new[m] * gate[m]
    da_before = gate_b[m]
    w0 = pe_w0[p]
    wmax = c[7] * w0
    a = c[3] * x_before * da_ph / dt
    bp = c[6] * da_before * kc_new[k_i] / dt
    r = c[4] * kc_new[k_i] / dt + c[5]
    k = a + bp + r
    if k > 0.0:
        w_eq = (bp * wmax + r * w0) / k
        w[e] = w_eq + (w[e] - w_eq) * math.exp(-k * dt)


@cuda.jit
def k_plast_decay_kc(kc_idx, x_tr, kc_new, decay_x):
    q = cuda.grid(1)
    if q < kc_idx.shape[0]:
        i = kc_idx[q]
        x_tr[i] *= decay_x
        kc_new[i] = 0.0


@cuda.jit
def k_plast_decay_da(da_tr, da_new, decay_da):
    i = cuda.grid(1)
    if i < da_tr.shape[0]:
        da_tr[i] *= decay_da
        da_new[i] = 0.0


STATE = ("v", "g", "rfc_left", "ring", "wad", "x_tr", "da_tr", "kc_new", "da_new", "da_base")
STATIC = ("indptr", "post")
PARAMS = ("rfc_len", "gain", "v_rest", "dan_ptr", "dan_m", "dan_a", "pe_edge", "pe_pre",
          "pe_m", "pe_w0", "kc_idx", "gj_a", "gj_b", "gj_k", "gk_ptr", "gk_tgt", "gk_amt")


class CudaRunner:
    """Holds a Brain's arrays on the GPU and advances it."""

    def __init__(self, brain):
        self.b = brain
        self.d = {}
        self.d["indptr"] = cuda.to_device(np.ascontiguousarray(brain.con.indptr))
        self.d["post"] = cuda.to_device(np.ascontiguousarray(brain.con.post))
        n = brain.con.n
        self.frozen = cuda.device_array(n, np.uint8)
        self.spk_list = cuda.device_array(n, np.int32)
        self.spk_n = cuda.device_array(1, np.int32)
        self.rng = create_xoroshiro128p_states(max(1, n), seed=int(brain.seed) + 1)
        self.state_on_gpu = False

    def _get(self, name):
        if name == "indptr" or name == "post":
            return self.d[name]
        return getattr(self.b, name)

    def push_state(self):
        for k in STATE:
            self.d[k] = cuda.to_device(np.ascontiguousarray(getattr(self.b, k)))
        self.state_on_gpu = True

    def pull_state(self):
        for k in STATE:
            self.d[k].copy_to_host(getattr(self.b, k))

    def push_params(self):
        for k in PARAMS:
            a = np.ascontiguousarray(getattr(self.b, k))
            if a.size == 0:
                a = np.zeros(1, a.dtype)
            self.d[k] = cuda.to_device(a)
        c = np.ascontiguousarray(self.b.pl_c, dtype=np.float64)
        self.d["pl_c"] = cuda.to_device(c if c.size >= 11 else np.zeros(11))
        self.n_gap = len(self.b.gj_a)
        self.n_pe = len(self.b.pe_edge)
        self.n_m = len(self.b.da_tr)
        self.n_kc = len(self.b.kc_idx)

    def pull_weights(self):
        """Only KC->MBON weights change on the GPU (plasticity); copy those back."""
        b = self.b
        if b.pl_on and len(b.pe_edge):
            out = cuda.device_array(len(b.pe_edge), np.float32)
            k_gather[_blocks(len(b.pe_edge)), TPB](self.d["w"], self.d["pe_edge"], out)
            b.w[b.pe_edge] = out.copy_to_host()

    def run(self, n_steps, stim_idx, stim_p, stim_w, ks, ki, counts, rec_mask, rec_out):
        b, d = self.b, self.d
        p = b.p
        if b._gpu_dirty:            # reset / plasticity switched / weights edited on the host
            self.push_state()
            self.d["w"] = cuda.to_device(np.ascontiguousarray(b.w))
            b._gpu_dirty = False
        self.push_params()
        n = b.con.n
        D = b.ring.shape[0]
        d_stim_idx = cuda.to_device(np.ascontiguousarray(stim_idx if len(stim_idx) else np.zeros(1, np.int64)))
        d_stim_p = cuda.to_device(np.ascontiguousarray(stim_p if len(stim_p) else np.zeros(1)))
        n_stim = len(stim_idx)
        d_kick_idx = cuda.to_device(np.ascontiguousarray(ki if len(ki) else np.zeros(1, np.int64)))
        d_counts = cuda.to_device(counts)
        d_rec_mask = cuda.to_device(rec_mask)
        d_rec_out = cuda.to_device(rec_out if rec_out.size else np.zeros((1, 1), np.uint8))
        c = d["pl_c"]
        if b.pl_on:
            gate = cuda.device_array(max(1, self.n_m))
            gate_b = cuda.device_array(max(1, self.n_m))
        zero = np.zeros(1, np.int32)
        kpos = 0
        for s in range(n_steps):
            step = b.t_step + s
            slot = step % D
            self.spk_n.copy_to_device(zero)
            k_update[_blocks(n), TPB](slot, d["v"], d["g"], d["rfc_left"], self.frozen, d["ring"],
                                      d["gain"], b._decay_v, b._decay_g, b._coef_a, d["v_rest"],
                                      p.v_th, self.spk_list, self.spk_n, d["wad"], b._decay_w,
                                      b._coef_w)
            if self.n_gap:
                dv = cuda.device_array(self.n_gap)
                k_gap_snapshot[_blocks(self.n_gap), TPB](d["v"], d["gj_a"], d["gj_b"], d["gj_k"], dv)
                k_gap_apply[_blocks(self.n_gap), TPB](d["v"], self.frozen, d["gj_a"], d["gj_b"], dv)
            if n_stim:
                k_poisson[_blocks(n_stim), TPB](d["v"], d_stim_idx, d_stim_p, stim_w, self.rng)
            lo = kpos
            while kpos < len(ks) and ks[kpos] == s:
                kpos += 1
            if kpos > lo:
                k_kicks[_blocks(kpos - lo), TPB](d["v"], d_kick_idx, lo, kpos, stim_w)
            out_slot = (step + D - 1) % D
            k_spikes[_blocks(n), TPB](self.spk_list, self.spk_n, out_slot, s, d["v"], d["g"], d["wad"],
                                      p.b_w, d["rfc_left"], d["rfc_len"], d_counts, d_rec_mask,
                                      d_rec_out, d["indptr"], d["post"], d["w"], d["ring"],
                                      d["gk_ptr"], d["gk_tgt"], d["gk_amt"], p.v_rst, b.pl_on,
                                      d["x_tr"], d["kc_new"], d["dan_ptr"], d["dan_m"], d["dan_a"],
                                      d["da_tr"], d["da_new"])
            if self.n_gap:
                k_spikelets[_blocks(n), TPB](self.spk_list, self.spk_n, d["rfc_left"], d["gk_ptr"],
                                             d["gk_tgt"], d["gk_amt"], d["v"])
            if b.pl_on and (step + 1) % PLAST_EVERY == 0:
                b.pl_c[9] += 1.0
                lam = max(b.pl_c[8], 1.0 - 1.0 / b.pl_c[9])
                k_plast_gate[_blocks(self.n_m), TPB](d["da_tr"], d["da_new"], d["da_base"], c,
                                                     gate, gate_b, lam)
                k_plast_edges[_blocks(self.n_pe), TPB](d["x_tr"], d["kc_new"], d["da_new"], gate, gate_b,
                                                       d["w"], d["pe_edge"], d["pe_pre"], d["pe_m"],
                                                       d["pe_w0"], c)
                k_plast_decay_kc[_blocks(self.n_kc), TPB](d["kc_idx"], d["x_tr"], d["kc_new"], b.pl_c[1])
                k_plast_decay_da[_blocks(self.n_m), TPB](d["da_tr"], d["da_new"], b.pl_c[2])
        d_counts.copy_to_host(counts)
        if rec_out.size:
            d_rec_out.copy_to_host(rec_out)
        self.pull_weights()
