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
    # ASSUMPTION(A14): maximal modulation factors. Inagaki 2012/2014 report roughly 2-fold
    # changes of GRN Ca2+ responses between fed and 24 h starved flies.
    sugar_gain_max: float = 2.0
    bitter_gain_min: float = 0.5
    or42b_gain_max: float = 2.0
    or85a_gain_min: float = 0.5


@dataclass
class Metabolism:
    p: MetabolismParams
    energy: float = 1.0       # 0 = starved, 1 = fed
    crop_ul: float = 0.0      # food in crop
    crop_sugar_ul: float = 0.0  # sugar-equivalent volume in crop

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
        frac = dt_s / p.crop_empty_tau_s
        sugar_absorbed = self.crop_sugar_ul * frac
        self.crop_ul -= self.crop_ul * frac
        self.crop_sugar_ul -= sugar_absorbed
        self.energy = min(1.0, max(0.0, self.energy + sugar_absorbed * p.energy_per_ul_sugar
                                   - dt_s / (p.starvation_time_to_empty_h * 3600)))

    def sensory_gains(self) -> dict:
        h = self.hunger
        p = self.p
        return {
            "sugar": 1.0 + (p.sugar_gain_max - 1.0) * h,
            "bitter": 1.0 - (1.0 - p.bitter_gain_min) * h,
            "ORN_DM1": 1.0 + (p.or42b_gain_max - 1.0) * h,
            "ORN_DM5": 1.0 - (1.0 - p.or85a_gain_min) * h,
        }
