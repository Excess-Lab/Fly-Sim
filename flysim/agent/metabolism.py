"""Internal energy state and its neuromodulatory effect on sensory gain.

Hunger never drives behaviour directly. It only changes how strongly sensory neurons respond,
which is how starvation acts in real flies; behaviour must emerge from the connectome.

Literature:
- Starved flies: sugar GRN responses up via dopamine (Inagaki et al. 2012 Neuron; 2014 Neuron),
  bitter GRN responses down (Inagaki et al. 2014 Neuron).
- Starvation increases Or42b (DM1) ORN output via sNPF presynaptic facilitation, which is
  necessary for efficient food search (Root et al. 2011 Cell).
- Starvation decreases Or85a (DM5) ORN output via tachykinin (Ko et al. 2015 eLife).
- Satiety: crop/gut distension and haemolymph sugar (Pool & Scott 2014; Dus et al. 2015).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class MetabolismParams:
    # ASSUMPTION(A10): energy reserve in arbitrary units, 1 = fully fed; a fly starved 24 h is
    # behaviourally strongly hungry (most feeding assays use 18-24 h starvation).
    starvation_time_to_empty_h: float = 24.0
    # ASSUMPTION(A11): one full meal after starvation ~0.3 uL (CAFE/MAFE meal sizes ~0.1-0.5 uL,
    # Ja et al. 2007 PNAS; Qi et al. 2015). Crop drains into the gut over ~30 min.
    meal_volume_ul: float = 0.3
    crop_empty_tau_s: float = 1800.0
    # ASSUMPTION(A12): ingestion rate during proboscis contact with a sugar source ~3 nL/s
    # (pharyngeal pumping ~6 Hz, ~0.5 nL/pump; Manzo et al. 2012)
    ingest_nl_per_s: float = 3.0
    # energy content of 1 uL of fully sweet food relative to the reserve (calibrated so that one
    # meal restores a 24 h starved fly): ASSUMPTION(A13)
    energy_per_ul_sugar: float = 1.0 / 0.3
    # ASSUMPTION(A14): hunger modulation. Sugar: dopamine raises sugar-GRN presynaptic output
    # ~1.3-1.4x in starved flies (Inagaki et al. 2012) - applied to the GRN output synapses
    # (Fly.sense); the behavioural ~4.6-fold PER threshold shift (Inagaki 2014) must come from
    # the network. Bitter and Or42b/Or85a: as before (Inagaki 2014; Root 2011; Ko 2015).
    sugar_gain_max: float = 1.4
    sugar_gain_min: float = 1.0
    bitter_gain_min: float = 0.5
    or42b_gain_max: float = 2.0
    or85a_gain_min: float = 0.5


@dataclass
class Metabolism:
    p: MetabolismParams
    energy: float = 1.0       # 0 = starved, 1 = fed
    crop_ul: float = 0.0      # food in crop
    crop_sugar_ul: float = 0.0  # sugar-equivalent volume in crop
    temp_c: float = 25.0
    o2_pct: float = 20.9
    toxin: float = 0.0        # ingested patulin load (relative)
    health: float = 1.0       # 1 = healthy; lowers vigour and lifespan

    def ingest_toxin(self, dose: float, dt_s: float):
        """ASSUMPTION(A72): patulin is toxic to Drosophila (reduces survival and fecundity);
        load decays by detoxification with t1/2 ~ 12 h; health = 1 / (1 + 5*load)."""
        self.toxin += dose
        self.toxin *= np.exp(-dt_s * np.log(2) / (12 * 3600))
        self.health = 1.0 / (1.0 + 5.0 * self.toxin)

    @classmethod
    def starved(cls, hours: float, p: MetabolismParams | None = None):
        p = p or MetabolismParams()
        return cls(p, energy=max(0.0, 1.0 - hours / p.starvation_time_to_empty_h))

    @property
    def hunger(self) -> float:
        """0 = sated, 1 = maximally hungry. A full crop signals satiety before absorption."""
        crop_fill = min(1.0, self.crop_ul / self.p.meal_volume_ul)
        return max(0.0, (1.0 - self.energy) * (1.0 - crop_fill))

    def step(self, dt_s: float, ingest_sugar_frac: float = 0.0, ingesting: bool = False):
        """ingest_sugar_frac: sugar content of food at the proboscis, 0..1 relative to a sweet meal."""
        p = self.p
        if ingesting and self.crop_ul < p.meal_volume_ul:
            dv = p.ingest_nl_per_s * 1e-3 * dt_s
            self.crop_ul += dv
            self.crop_sugar_ul += dv * ingest_sugar_frac
        frac = 1.0 - np.exp(-dt_s / p.crop_empty_tau_s)  # exact for any dt (fast-forward)
        sugar_absorbed = self.crop_sugar_ul * frac
        self.crop_ul -= self.crop_ul * frac
        self.crop_sugar_ul -= sugar_absorbed
        # ASSUMPTION(A15): metabolic rate Q10 = 2 around 25 C; hypoxia lowers it proportionally
        # below 10 % O2 (flies enter hypometabolism, Haddad 1997; Van Voorhies 2009)
        rate = 2.0 ** ((self.temp_c - 25.0) / 10.0) * min(1.0, self.o2_pct / 10.0)
        self.energy = min(1.0, max(0.0, self.energy + sugar_absorbed * p.energy_per_ul_sugar
                                   - rate * dt_s / (p.starvation_time_to_empty_h * 3600)))

    def sensory_gains(self) -> dict:
        h = self.hunger
        p = self.p
        return {
            "sugar": p.sugar_gain_min + (p.sugar_gain_max - p.sugar_gain_min) * h,
            "bitter": 1.0 - (1.0 - p.bitter_gain_min) * h,
            "ORN_DM1": 1.0 + (p.or42b_gain_max - 1.0) * h,
            "ORN_DM5": 1.0 - (1.0 - p.or85a_gain_min) * h,
        }
