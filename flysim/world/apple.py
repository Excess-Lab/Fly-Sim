"""Rotting apple: fermentation chemistry -> surface taste and odour emission.

Two-stage microbial succession on damaged fruit at ~25 C:
  yeast (Saccharomyces/Hanseniaspora) ferment sugar -> ethanol + CO2 + esters,
  then acetic acid bacteria (Acetobacter) oxidise ethanol -> acetic acid.
Logistic growth with Monod substrate limitation (standard fermentation kinetics).

ASSUMPTION(A40): rate constants are order-of-magnitude values for fruit at room temperature:
yeast doubling ~2-4 h on fruit sugar, visible fermentation within ~1 day, ethanol peaking at
~2-4 % w/w after ~2-3 days, acetic acid rising over days 3-7 (fruit/cider fermentation
literature; Becher et al. 2012 for yeast-fruit volatiles). Apple flesh sugar ~10-13 g/100 g
(fructose ~6, glucose ~2, sucrose ~2.5; USDA FoodData Central).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class AppleParams:
    radius_mm: float = 35.0
    sugar0: float = 11.0          # g / 100 g fresh apple
    mu_yeast: float = 0.25        # 1/h max specific growth
    k_sugar: float = 1.0          # g/100g Monod constant
    yield_eth: float = 0.48       # g ethanol per g sugar (Gay-Lussac 0.51, practical ~0.48)
    mu_aab: float = 0.08          # 1/h acetic acid bacteria growth
    k_eth: float = 0.5
    yield_ac: float = 1.0         # g acetic acid per g ethanol (stoich. 1.30, practical lower)
    ester_frac: float = 0.002     # g esters per g sugar fermented
    lag_aab_h: float = 36.0       # bacteria establish after yeast acidify/produce ethanol
    q_sugar: float = 0.25         # g/100g/h sugar fermented at full yeast biomass
    q_eth: float = 0.08           # g/100g/h ethanol oxidised at full bacterial biomass


@dataclass
class AppleState:
    age_h: float = 0.0
    sugar: float = 11.0           # g/100g
    ethanol: float = 0.0
    acetic: float = 0.0
    esters: float = 0.0
    yeast: float = 0.001          # relative biomass 0..1
    aab: float = 0.0005
    ferm_rate: float = 0.0        # g/100g/h current sugar fermentation


class Apple:
    def __init__(self, p: AppleParams | None = None, center=(0.0, 0.0)):
        self.p = p or AppleParams()
        self.s = AppleState(sugar=self.p.sugar0)
        self.center = np.array([center[0], center[1], self.p.radius_mm])  # resting on the floor

    def age(self, hours: float, dt_h: float = 0.05):
        """Advance fermentation by `hours` (days-scale process, independent of fly timescale)."""
        p, s = self.p, self.s
        for _ in range(int(np.ceil(hours / dt_h))):
            mono_s = s.sugar / (p.k_sugar + s.sugar)
            g_y = p.mu_yeast * s.yeast * (1 - s.yeast) * mono_s
            used = min(s.sugar, p.q_sugar * s.yeast * mono_s * dt_h)  # fermentation ~ biomass
            s.ferm_rate = used / dt_h
            s.sugar -= used
            s.ethanol += p.yield_eth * used
            s.esters += p.ester_frac * used
            s.yeast += g_y * dt_h
            if s.age_h > p.lag_aab_h:
                mono_e = s.ethanol / (p.k_eth + s.ethanol)
                g_a = p.mu_aab * s.aab * (1 - s.aab) * mono_e
                conv = min(s.ethanol, p.q_eth * s.aab * mono_e * dt_h)
                s.ethanol -= conv
                s.acetic += p.yield_ac * conv
                s.aab += g_a * dt_h
            # esters are volatile and hydrolysed: slow loss
            s.esters *= np.exp(-dt_h / 48.0)
            s.age_h += dt_h

    # ---- emission (relative source strengths at the apple surface) ----
    def odor_sources(self) -> dict:
        """Headspace concentration at the surface, in 'dilution units' comparable to the
        DoOR/Hallem stimulus scale (1 = 10^-2 dilution of pure odorant).
        ASSUMPTION(A41): headspace proportional to tissue concentration (Henry's law regime);
        fresh apple emits mostly esters/hexyl acetate/hexanal from the peel."""
        s = self.s
        fresh = max(0.0, 1.0 - s.yeast * 2)
        return {
            "hexyl acetate": 0.05 * fresh + 0.02,
            "hexanal": 0.03 * fresh,
            "ethyl acetate": 4.0 * s.esters + 0.01,
            "isoamyl acetate": 2.0 * s.esters,
            "ethyl butyrate": 1.0 * s.esters,
            "ethanol": 0.08 * s.ethanol,
            "acetic acid": 0.15 * s.acetic,
            "acetaldehyde": 1.5 * s.ferm_rate,
            "2-phenylethanol": 0.02 * s.yeast,
            "acetoin": 0.03 * s.yeast,
            "CO2": 2.0 * s.ferm_rate,  # CO2 only while sugar is actively fermented
        }

    def surface_taste(self) -> dict:
        """Taste at the surface, normalised 0..1 for GRN encoding."""
        s = self.s
        return {
            "sugar": min(1.0, s.sugar / 10.0),
            "water": 0.9,
            "ethanol": min(1.0, s.ethanol / 5.0),
            "acid": min(1.0, s.acetic / 3.0),
            "yeast": min(1.0, s.yeast),       # amino acids / yeast protein
        }

    def nutrition(self) -> dict:
        s = self.s
        return {"sugar_frac": min(1.0, s.sugar / 11.0), "yeast_frac": min(1.0, s.yeast)}

    def summary(self) -> str:
        s = self.s
        return (f"age {s.age_h / 24:.1f} d: sugar {s.sugar:.1f}, ethanol {s.ethanol:.2f}, "
                f"acetic {s.acetic:.2f} g/100g, yeast {s.yeast:.2f}, bacteria {s.aab:.2f}")
