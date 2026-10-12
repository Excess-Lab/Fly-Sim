"""Life cycle: mating -> eggs on the fruit -> larva -> pupa -> new adult with its own brain.

Development of D. melanogaster at 25 C (Ashburner 1989; Ashburner et al. 2005):
  egg ~1 d, larva ~4 d (three instars), pupa ~4.5 d  -> eclosion ~9.5 d after laying.
ASSUMPTION(A63): larval survival rises with yeast (main larval food) and falls with acetic acid
>4 g/100g; egg production of a mated female on fermenting fruit ~2 eggs/h (~50/day; typical
lab fecundity 50-80/day), only while she can reach the fruit.
ASSUMPTION(A64): offspring get the same connectome as their parent of that sex (only one
connectome per sex exists); individuality comes from noise seeds and their own experience.
Larvae are not simulated neuron-by-neuron (the larval connectome is a different nervous system).
"""
from dataclasses import dataclass, field

import numpy as np

EGG_D, LARVA_D, PUPA_D = 1.0, 4.0, 4.5


@dataclass
class Brood:
    laid_h: float
    mother: str
    father: str
    pos: tuple
    alive: bool = True

    def stage(self, now_h):
        age_d = (now_h - self.laid_h) / 24
        if age_d < EGG_D:
            return "egg"
        if age_d < EGG_D + LARVA_D:
            return "larva"
        if age_d < EGG_D + LARVA_D + PUPA_D:
            return "pupa"
        return "adult"


@dataclass
class Population:
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    brood: list = field(default_factory=list)
    born: int = 0
    eggs_per_h: float = 2.0
    max_adults: int = 4            # RAM limit on an 8 GB Mac (each brain ~0.3-0.5 GB)
    lifespan_d: float = 45.0

    def lay(self, now_h, mother, father, pos, n=1):
        for _ in range(n):
            self.brood.append(Brood(now_h, mother, father, tuple(np.round(pos, 2))))

    def survival_per_day(self, apple) -> float:
        """ASSUMPTION(A73): mould competes with larvae and its toxins kill them (Rohlfs 2005;
        Trienens et al. 2010): survival x (1 - 0.8 * mould cover)."""
        s = apple.s
        food = min(1.0, s.yeast + 0.5 * s.sugar / 11.0)
        acid_pen = max(0.0, (s.acetic - 4.0) / 4.0)
        return float(np.clip((0.6 + 0.4 * food - acid_pen) * (1 - 0.8 * s.mold), 0.0, 1.0))

    def fast_forward(self, now_h, hours, apple, flies):
        """Advance the life cycle by `hours` off-line. Returns list of brood ready to eclose."""
        # off-line egg laying by mated females (they live on/near the fruit between sessions)
        for f in flies:
            if f.cfg.sex == "female" and getattr(f, "mated", False):
                lam = self.eggs_per_h * hours * min(1.0, 0.3 + apple.s.yeast)
                k = self.rng.poisson(lam)
                if k:
                    self.lay(now_h, f.cfg.name, getattr(f, "mate", "?"), apple.center, k)
                    f.events.append((round(f.t, 3), f"laid {k} eggs off-line ({hours:.0f} h)"))
        surv_h = self.survival_per_day(apple) ** (hours / 24)
        ready = []
        for b in self.brood:
            if not b.alive:
                continue
            if b.stage(now_h + hours) in ("larva", "pupa") and self.rng.random() > surv_h:
                b.alive = False
            elif b.stage(now_h + hours) == "adult":
                ready.append(b)
        return ready

    def counts(self, now_h):
        c = {"egg": 0, "larva": 0, "pupa": 0}
        for b in self.brood:
            if b.alive:
                st = b.stage(now_h)
                if st in c:
                    c[st] += 1
        return c
