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
    # ASSUMPTION(A40b): environment dependence. Microbial rates Q10 = 2 around 25 C.
    # Acetobacter is an obligate aerobe: growth ~ O2/(O2 + 1 %). Yeast ferments with or without O2.
    # Intact skin blocks microbes: inoculum 1e-6 (rots over weeks) vs cut/damaged 1e-3.
    q10: float = 2.0
    k_o2_pct: float = 1.0
    # ASSUMPTION(A71): blue mould (Penicillium expansum), the main postharvest apple rot.
    # Aerobic; enters through wounds: visible lesions ~3-5 d after wounding at 20-25 C, intact
    # fruit only via micro-wounds (weeks). Logistic growth, consumes sugar, emits geosmin and
    # produces the mycotoxin patulin (Snini et al. 2016; Stensmyr et al. 2012).
    mu_mold: float = 0.06         # 1/h
    q_mold_sugar: float = 0.03    # g/100g/h sugar used at full mould cover


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
    mold: float = 0.0             # fraction of surface covered by mould 0..1
    patulin: float = 0.0          # relative mycotoxin level 0..1


class Apple:
    def __init__(self, p: AppleParams | None = None, center=(0.0, 0.0), cut: bool = True,
                 temp_c: float = 25.0, o2_pct: float = 20.9):
        self.p = p or AppleParams()
        self.s = AppleState(sugar=self.p.sugar0, yeast=1e-3 if cut else 1e-6,
                            aab=5e-4 if cut else 5e-7, mold=1e-4 if cut else 1e-7)
        self.cut = cut
        self.temp_c = temp_c
        self.o2_pct = o2_pct
        self.center = np.array([center[0], center[1], self.p.radius_mm])  # resting on the floor

    def age(self, hours: float, dt_h: float = 0.05):
        """Advance fermentation by `hours` (days-scale process, independent of fly timescale)."""
        p, s = self.p, self.s
        qt = p.q10 ** ((self.temp_c - 25.0) / 10.0)
        qt_mold = qt  # mould invades intact fruit through micro-wounds; only its inoculum is smaller
        if not self.cut:
            qt *= 0.05  # intact skin: microbes only at micro-wounds; apples keep for weeks
        o2f = self.o2_pct / (self.o2_pct + p.k_o2_pct)
        for _ in range(int(np.ceil(hours / dt_h))):
            mono_s = s.sugar / (p.k_sugar + s.sugar)
            g_y = qt * p.mu_yeast * s.yeast * (1 - s.yeast) * mono_s
            used = min(s.sugar, qt * p.q_sugar * s.yeast * mono_s * dt_h)  # fermentation ~ biomass
            s.ferm_rate = used / dt_h
            s.sugar -= used
            s.ethanol += p.yield_eth * used
            s.esters += p.ester_frac * used
            s.yeast += g_y * dt_h
            if s.age_h > p.lag_aab_h:
                mono_e = s.ethanol / (p.k_eth + s.ethanol)
                g_a = qt * o2f * p.mu_aab * s.aab * (1 - s.aab) * mono_e
                conv = min(s.ethanol, qt * o2f * p.q_eth * s.aab * mono_e * dt_h)
                s.ethanol -= conv
                s.acetic += p.yield_ac * conv
                s.aab += g_a * dt_h
            o2_mold = self.o2_pct / (self.o2_pct + 2.0)
            g_m = qt_mold * o2_mold * p.mu_mold * s.mold * (1 - s.mold)
            s.mold += g_m * dt_h
            s.sugar = max(0.0, s.sugar - qt_mold * p.q_mold_sugar * s.mold * dt_h)
            s.patulin += 0.02 * s.mold * dt_h
            s.patulin = min(1.0, s.patulin)
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
            # CO2 at the surface from the gas flux: c_s = Q / (4 pi D R)  (steady diffusion from a
            # sphere), D_CO2 = 0.16 cm^2/s, Q from fermentation + mould respiration; units: % CO2
            "CO2": self.co2_surface_pct(),
            # geosmin: detected at very low concentration by Or56a/DA2 (Stensmyr et al. 2012)
            "geosmin": 1.0 * s.mold,
        }

    def mass_g(self) -> float:
        return 0.85 * 4 / 3 * np.pi * (self.p.radius_mm / 10) ** 3

    def co2_flux_cm3_s(self) -> float:
        s = self.s
        sugar_g_h = (s.ferm_rate + self.p.q_mold_sugar * s.mold * 3.0) * self.mass_g() / 100.0
        mol_h = sugar_g_h / 180.16 * 2.0
        return mol_h * 24450.0 / 3600.0

    def co2_surface_pct(self) -> float:
        return 100.0 * self.co2_flux_cm3_s() / (4 * np.pi * 0.16 * self.p.radius_mm / 10)

    def osmolarity(self) -> float:
        """Osm/kg of the juice from dissolved sugars (hexose 180 g/mol) and acetic acid."""
        s = self.s
        return s.sugar * 10 / 180.0 + s.acetic * 10 / 60.0 * 0.05  # acetic acid is weak (~5% ionised)

    def surface_taste(self) -> dict:
        """Physical surface chemistry. sugar normalised to a sweet meal (1 = ~10 g/100 g)."""
        s = self.s
        return {
            "sugar": min(1.0, s.sugar / 10.0),
            "osmolarity": self.osmolarity(),
            "wet": 1.0,
            "ethanol_g": s.ethanol,
            "acetic_g": s.acetic,
            "yeast": min(1.0, s.yeast),
            "patulin": s.patulin,
            "mold": s.mold,
        }

    def nutrition(self) -> dict:
        s = self.s
        return {"sugar_frac": min(1.0, s.sugar / 11.0), "yeast_frac": min(1.0, s.yeast),
                "patulin": s.patulin}

    def summary(self) -> str:
        s = self.s
        return (f"age {s.age_h / 24:.1f} d: sugar {s.sugar:.1f}, ethanol {s.ethanol:.2f}, "
                f"acetic {s.acetic:.2f} g/100g, yeast {s.yeast:.2f}, bacteria {s.aab:.2f}, "
                f"mould {s.mold:.2f}")
