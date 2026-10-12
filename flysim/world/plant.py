"""Apple seeds -> seedling -> tree (Malus domestica), driven by physiology, not by design.

Literature values (ASSUMPTION A92):
  seeds per apple ~5-10 (use 6); released when the flesh has decomposed
  dormancy: needs cold-moist stratification, ~60-90 days at 1-7 C (chill hours ~1500-2200;
    Lewak 2011; Westwood). Without it seeds do not germinate.
  germination after stratification: soil + moisture + 10-25 C; 50-80 % of viable seeds,
    emergence ~2-3 weeks
  viability loss: anoxia and heavy mould cover kill seeds over weeks; dry storage slowly
  seedling growth: ~30-60 cm in the first season under good conditions, slowing later; growth
    only when 5-30 C (degree-day like)
  juvenile phase: apple trees from seed flower after ~5-10 years (use 6)
  mature tree: tens to hundreds of fruit per year (use 50, autumn), capped by the arena
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Seed:
    apple_idx: int
    chill_h: float = 0.0
    viable: bool = True
    released: bool = False
    stage: str = "in_fruit"     # in_fruit | dormant | stratified | germinated | dead
    age_h_since_germ: float = 0.0
    height_cm: float = 0.0
    will_germinate: bool = True   # fixed fate drawn at stratification (germination rate)


@dataclass
class PlantWorld:
    soil: bool = False
    moist: bool = False
    seeds: list = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1))
    chill_needed_h: float = 1800.0          # ~75 days at 1-7 C
    germ_prob: float = 0.65
    juvenile_years: float = 6.0
    events: list = field(default_factory=list)

    def add_apple_seeds(self, apple_idx, n=6):
        self.seeds += [Seed(apple_idx) for _ in range(n)]

    def step(self, hours: float, apples, temp_c: float, o2_pct: float, clock_h: float):
        for s in self.seeds:
            if s.stage == "dead":
                continue
            a = apples[min(s.apple_idx, len(apples) - 1)]
            if s.stage == "in_fruit":
                # flesh decomposed: sugar gone and microbes have taken over
                if a.s.sugar < 0.5 and (a.s.mold > 0.5 or a.s.aab > 0.5 or a.s.yeast > 0.9):
                    s.stage, s.released = "dormant", True
                    self.events.append((clock_h, f"семя яблока {s.apple_idx} высвободилось из мякоти"))
                continue
            # viability losses (anoxia, heavy mould)
            if o2_pct < 1.0 and self.rng.random() < hours / (14 * 24):
                s.stage, s.viable = "dead", False
                continue
            if a.s.mold > 0.9 and not self.soil and self.rng.random() < hours / (60 * 24):
                s.stage, s.viable = "dead", False
                continue
            if s.stage == "dormant":
                if 1.0 <= temp_c <= 7.0 and self.moist:
                    s.chill_h += hours
                if s.chill_h >= self.chill_needed_h:
                    s.stage = "stratified"
                    s.will_germinate = self.rng.random() < self.germ_prob
                    self.events.append((clock_h, f"семя прошло холодную стратификацию ({s.chill_h / 24:.0f} дн)"))
            elif s.stage == "stratified":
                if self.soil and self.moist and 10.0 <= temp_c <= 25.0 and s.will_germinate:
                    if self.rng.random() < hours / (18 * 24):
                        s.stage = "germinated"
                        self.events.append((clock_h, "семя проросло — появился сеянец яблони"))
            elif s.stage == "germinated":
                s.age_h_since_germ += hours
                if 5.0 <= temp_c <= 30.0 and self.moist and o2_pct > 5:
                    # ~45 cm per growing season early on, slowing with size (logistic to ~8 m)
                    r = 45.0 / (150 * 24)       # cm per hour in the first season
                    s.height_cm += r * hours * (1 - s.height_cm / 800.0)

    def trees(self):
        return [s for s in self.seeds if s.stage == "germinated"]

    def fruiting(self):
        return [s for s in self.trees() if s.age_h_since_germ / (24 * 365) >= self.juvenile_years]

    def summary(self) -> dict:
        c = {}
        for s in self.seeds:
            c[s.stage] = c.get(s.stage, 0) + 1
        t = self.trees()
        return {"seeds": c, "seedlings": len(t),
                "tallest_cm": round(max((s.height_cm for s in t), default=0.0), 1),
                "fruiting_trees": len(self.fruiting()), "soil": self.soil, "moist": self.moist}


def seasonal_temp(clock_h: float, mean_c=10.0, amp_c=12.0, start_doy=180) -> float:
    """Temperate annual cycle (daily mean), day 0 = start_doy (180 = summer)."""
    doy = (start_doy + clock_h / 24.0) % 365.0
    return mean_c + amp_c * np.sin(2 * np.pi * (doy - 110) / 365.0)
