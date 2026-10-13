"""Life cycle: mating -> eggs on the fruit -> larva -> pupa -> new adult with its own brain.

Development of D. melanogaster at 25 C (Ashburner 1989; Ashburner et al. 2005):
  egg ~1 d, larva ~4 d (three instars), pupa ~4.5 d  -> eclosion ~9.5 d after laying.
ASSUMPTION(A63): larval survival rises with yeast (main larval food) and falls with acetic acid
>4 g/100g.
ASSUMPTION(A63b): egg laying between live sessions is SCRIPTED, not produced by the brain:
every living mated female lays Poisson(2 eggs/h x hours x min(1, 0.3 + yeast)) eggs (~50/day;
typical lab fecundity 50-80/day). Her position, oviDN activity and whether she can actually
reach the fruit (cold coma, hypoxia, walls) are ignored. The site is chosen in proportion to
where she laid eggs herself (brain-driven, A61) during live sessions; if she has not laid any
yet, the most yeasty, least mouldy fruit (Becher et al. 2012). Live time spent on each fruit is
not recorded by the organism, so it cannot be used.
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
    apple_idx: int = 0
    offline: bool = False      # laid by the scripted off-line rule (A63b), not by the brain

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

    def lay(self, now_h, mother, father, pos, n=1, apple_idx=0, offline=False):
        for _ in range(n):
            self.brood.append(Brood(now_h, mother, father, tuple(np.round(pos, 2)),
                                    apple_idx=apple_idx, offline=offline))

    def site_weights(self, mother, n_apples):
        """A63b: oviposition-site preference of one female = share of the eggs she laid herself
        (brain-driven, A61) on each fruit. None if she has not laid any yet."""
        w = np.zeros(n_apples)
        for b in self.brood:
            if b.mother == mother and not b.offline and b.apple_idx < n_apples:
                w[b.apple_idx] += 1
        return w / w.sum() if w.sum() > 0 else None

    def survival_per_day(self, apple) -> float:
        """ASSUMPTION(A73): mould competes with larvae and its toxins kill them (Rohlfs 2005;
        Trienens et al. 2010): survival x (1 - 0.8 * mould cover)."""
        s = apple.s
        food = min(1.0, s.yeast + 0.5 * s.sugar / 11.0)
        acid_pen = max(0.0, (s.acetic - 4.0) / 4.0)
        return float(np.clip((0.6 + 0.4 * food - acid_pen) * (1 - 0.8 * s.mold), 0.0, 1.0))

    def fast_forward(self, now_h, hours, apples, flies):
        """Advance the life cycle by `hours` off-line. Returns list of brood ready to eclose."""
        apples = apples if isinstance(apples, (list, tuple)) else [apples]
        # ASSUMPTION(A63b): scripted off-line egg laying (see module docstring)
        default = max(range(len(apples)), key=lambda i: apples[i].s.yeast - 0.2 * apples[i].s.mold)
        for f in flies:
            if f.cfg.sex != "female" or not getattr(f, "mated", False) or getattr(f, "dead", None):
                continue
            w = self.site_weights(f.cfg.name, len(apples))
            idx = self.rng.choice(len(apples), p=w) if w is not None else default
            ap = apples[idx]
            lam = self.eggs_per_h * hours * min(1.0, 0.3 + ap.s.yeast)
            k = self.rng.poisson(lam)
            if k:
                top = ap.center + np.array([0, 0, ap.p.radius_mm])  # on the fruit surface
                self.lay(now_h, f.cfg.name, getattr(f, "mate", "?"), top, k, apple_idx=idx,
                         offline=True)
                f.events.append((round(f.t, 3), f"laid {k} eggs off-line ({hours:.0f} h, scripted A63b)"))
        # mortality applies to the time each individual spends as larva/pupa within the interval
        surv_by_apple = [self.survival_per_day(a) for a in apples]
        ready = []
        for b in self.brood:
            if not b.alive:
                continue
            a0 = (now_h - b.laid_h) / 24
            a1 = (now_h + hours - b.laid_h) / 24
            lo, hi = EGG_D, EGG_D + LARVA_D + PUPA_D
            exposed = max(0.0, min(a1, hi) - max(a0, lo))
            surv_d = surv_by_apple[min(b.apple_idx, len(apples) - 1)]
            if exposed > 0 and self.rng.random() > surv_d ** exposed:
                b.alive = False
            elif a1 >= hi:
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
