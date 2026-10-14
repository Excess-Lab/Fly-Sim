"""Energy state of every neuron: hypoxia, anoxic coma, spreading depolarisation and cell death.

ASSUMPTION(A101). Insect neurons get O2 straight from the tracheae; ATP falls within seconds
when O2 is missing. Each neuron i has an energy level E_i in [0, 1]:
    dE/dt = (E_target - E) / tau,  tau = 15 s while falling, 120 s while recovering
    E_target = s(O2) - k_spike * rate_i / 100 Hz          (firing costs ATP)
    s(O2) = [O2 / (O2 + K)] / [20.9 / (20.9 + K)],  K = 1.5 % O2  (s = 1 in normal air)
Membrane: resting potential depolarises as E falls, v_rest = -52 + 6 * (1 - E)^2 mV (pump
failure). Below E_sd = 0.3 the neuron enters spreading depolarisation: a 1 s burst
(v_rest -40 mV, above threshold) followed by depolarisation block (silent; modelled as
v_rest -120 mV and no sensory drive). It leaves the block only when E is back above 0.6
(ion gradients restored), i.e. ~1-1.5 min after reoxygenation (anoxic coma is reversible:
coma within ~10-30 s of anoxia, recovery in minutes; Haddad et al. 1997; Krishnan et al. 1997;
spreading depolarisation in insect CNS: Rodgers et al. 2010; Armstrong et al. 2011;
Spong et al. 2016). A neuron blocked for longer than T_irrev = 60 min dies (irreversible; order of
magnitude - insects survive hours of anoxia, neuronal damage follows prolonged anoxia).
At organismal death the supply stops: starting from a random point the loss of supply spreads
through the CNS at 3 mm/min (spreading depolarisation front), so neurons fail one after another.
In normal air nothing changes (E = 1, v_rest = -52 mV): validated results are unaffected.
"""
import numpy as np

V0 = -52.0
K_O2 = 1.5
TAU_DOWN, TAU_UP = 15.0, 120.0
K_SPIKE = 0.05
E_SD = 0.3
E_RECOVER = 0.6     # leaving the block needs ion gradients restored (hysteresis)
BURST_S = 1.0
T_IRREV_S = 3600.0
SD_SPEED_MM_S = 3.0 / 60.0


def o2_supply(o2_pct):
    return (o2_pct / (o2_pct + K_O2)) / (20.9 / (20.9 + K_O2))


class NeuronEnergy:
    def __init__(self, fly):
        self.f = fly
        n = fly.brain.con.n
        self.E = np.ones(n, np.float32)
        self.sd_t = np.full(n, -1.0)          # time spent in spreading depolarisation (-1 = not)
        self.dead = np.zeros(n, bool)
        self.rate = np.zeros(n, np.float32)
        self.terminal = None                  # (origin distances in mm, elapsed s) after death
        self.base = fly.brain.v_rest.copy()   # per-neuron resting potential (tonic cells keep theirs)
        self.active = False                   # anything away from normal (skip work in normal air)
        self.n_coma_events = 0
        self._in_coma = False

    def start_terminal(self):
        ann = self.f.ann
        xyz = ann[["x", "y", "z"]].to_numpy(float) / 1e6        # nm -> mm
        ok = ~np.isnan(xyz[:, 0])
        rng = np.random.default_rng(self.f.cfg.seed + 101)
        origin = xyz[rng.choice(np.flatnonzero(ok))]
        dist = np.where(ok, np.linalg.norm(np.nan_to_num(xyz) - origin, axis=1), np.nan)
        dist = np.where(np.isnan(dist), np.nanmax(dist), dist)
        self.terminal = {"dist": dist, "t": 0.0, "t_end": float(dist.max()) / SD_SPEED_MM_S + 40.0}

    def step(self, dt, o2_pct, counts):
        """Update energies from the last spike counts; returns (v_rest array, silenced mask) or None
        when every neuron is in its normal state (normal air, alive)."""
        if not self.active and o2_supply(max(o2_pct, 0.0)) >= 0.999 and self.terminal is None:
            return None
        self.active = True
        self.rate = counts / max(dt, 1e-6)
        supply = np.full(len(self.E), o2_supply(max(o2_pct, 0.0)), np.float32)
        if self.terminal is not None:
            self.terminal["t"] += dt
            supply[self.terminal["dist"] <= self.terminal["t"] * SD_SPEED_MM_S] = 0.0
        target = np.clip(supply - K_SPIKE * self.rate / 100.0, 0.0, 1.0)
        tau = np.where(target < self.E, TAU_DOWN, TAU_UP)
        self.E += (target - self.E) * (1.0 - np.exp(-dt / tau)).astype(np.float32)
        sd = (self.E < E_SD) | ((self.sd_t >= 0) & (self.E < E_RECOVER))
        self.sd_t = np.where(sd, np.maximum(self.sd_t, 0.0) + dt, -1.0)
        self.dead |= self.sd_t > T_IRREV_S
        v = self.base + 6.0 * (1.0 - self.E) ** 2
        burst = sd & (self.sd_t <= BURST_S)
        block = (sd & (self.sd_t > BURST_S)) | self.dead
        v = np.where(burst, -40.0, v)
        v = np.where(block, -120.0, v)
        coma = block.mean() > 0.5
        if (self.terminal is None and o2_supply(max(o2_pct, 0.0)) >= 0.999 and not self.dead.any()
                and (self.sd_t < 0).all() and self.E.min() > 0.9):   # firing alone keeps E < 1
            self.active = False            # fully recovered: back to the exact normal model
            self.E[:] = 1.0
            self._in_coma = False
            return self.base.astype(np.float64), np.zeros(len(self.E), bool)
        if coma and not self._in_coma:
            self.n_coma_events += 1
        self._in_coma = coma
        return v.astype(np.float64), block

    def all_failed(self):
        return self.terminal is not None and self.terminal["t"] >= self.terminal["t_end"]

    def summary(self):
        return {"energy_mean": round(float(self.E.mean()), 3),
                "in_depolarisation": int((self.sd_t >= 0).sum()),
                "dead_neurons": int(self.dead.sum()), "coma": bool(self._in_coma)}
