"""Mortality: every fly dies of its own cause, with individual tolerance thresholds.

ASSUMPTION(A94), Drosophila melanogaster literature (orders of magnitude):
  ageing     Gompertz hazard mu(x) = a * exp(b x), a = 2e-4 /day, b = 0.12 /day at 25 C
             (median adult lifespan ~50 d; Pletcher et al. 2000; Mair et al. 2003);
             rate scaled by Q10 = 2.5 with temperature (flies live ~2x longer at 18 C);
             mould toxin raises the hazard (health factor, A72)
  starvation energy reserve 1 -> 0 in ~24 h (A10), then death after an individual threshold
             N(40 h, 10 h) at zero energy: median ~64 h from fully fed (range ~45-85 h).
             Well-fed adult females with water die after ~2-3 days of starvation (lab stocks
             roughly 50-70 h; males earlier; selection lines differ widely: review Rion & Kawecki
             2007). Water is always available in the model (no desiccation).
  heat       knockdown at body temperature > 39 C; death after N(15, 5) min above 40 C
             (Huey et al. 1992; Jørgensen et al. 2019)
  cold       death after N(72, 24) h below 5 C (chill injury; MacMillan & Sinclair 2011)
  anoxia     death after N(12, 3) h below 1 % O2. Adult flies fall into reversible anoxic
             stupor within seconds and recover fully after ~4 h of anoxia (Haddad et al. 1997;
             Krishnan et al. 1997), so 4 h is a survival, not a lethal, time; longer anoxia is
             lethal but published LT50 values vary with temperature and strain. 12 +- 3 h is an
             order-of-magnitude choice, not a measured value.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mortality:
    rng: np.random.Generator
    a_per_day: float = 2e-4
    b_per_day: float = 0.12
    q10: float = 2.5
    starve_h: float = field(default=0.0)
    heat_min: float = field(default=0.0)
    cold_h: float = field(default=0.0)
    anoxia_h: float = field(default=0.0)
    # accumulated exposures
    t_starve: float = 0.0
    t_heat: float = 0.0
    t_cold: float = 0.0
    t_anoxia: float = 0.0
    bio_age_d: float = 0.0     # physiological age (temperature-scaled)

    def __post_init__(self):
        r = self.rng
        self.starve_h = max(12.0, r.normal(40.0, 10.0))
        self.heat_min = max(3.0, r.normal(15.0, 5.0))
        self.cold_h = max(12.0, r.normal(72.0, 24.0))
        self.anoxia_h = max(5.0, r.normal(12.0, 3.0))

    def step(self, dt_s: float, energy: float, body_temp_c: float, air_temp_c: float,
             o2_pct: float, health: float = 1.0) -> str | None:
        """Advance exposures by dt; return a cause of death or None."""
        h = dt_s / 3600.0
        rate = self.q10 ** ((air_temp_c - 25.0) / 10.0)
        self.bio_age_d += rate * dt_s / 86400.0
        self.t_starve = self.t_starve + h if energy <= 0.0 else 0.0
        self.t_heat = self.t_heat + dt_s / 60.0 if body_temp_c > 40.0 else max(0.0, self.t_heat - dt_s / 120.0)
        self.t_cold = self.t_cold + h if air_temp_c < 5.0 else max(0.0, self.t_cold - h / 2)
        self.t_anoxia = self.t_anoxia + h if o2_pct < 1.0 else max(0.0, self.t_anoxia - h)
        if self.t_starve >= self.starve_h:
            return "голод"
        if self.t_heat >= self.heat_min:
            return "перегрев"
        if self.t_cold >= self.cold_h:
            return "холод"
        if self.t_anoxia >= self.anoxia_h:
            return "нехватка кислорода"
        # Gompertz ageing hazard (toxins accelerate: health < 1 raises the hazard)
        mu = self.a_per_day * np.exp(self.b_per_day * self.bio_age_d) / max(health, 0.2)
        if self.rng.random() < 1.0 - np.exp(-mu * rate * dt_s / 86400.0):
            return "старость" + (" (ускорена токсином плесени)" if health < 0.9 else "")
        return None
