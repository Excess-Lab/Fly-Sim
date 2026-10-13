"""Air of the container as a well-mixed gas volume (mass balance), coupled to the apple and flies.

  dC/dt = sources / V  -  ach * (C - C_outside)

Sources (ideal gas, 24.45 L/mol at 25 C, scaled by T/298):
  alcoholic fermentation  C6H12O6 -> 2 C2H5OH + 2 CO2      (no O2 used; 0.489 g CO2 / g sugar)
  aerobic respiration     C6H12O6 + 6 O2 -> 6 CO2 + 6 H2O   (mould, acetic acid bacteria)
  acetic oxidation        C2H5OH + O2 -> CH3COOH + H2O       (Acetobacter, uses O2, no CO2)
  fly respiration         ~4 uL CO2 / h per fly at 25 C (RQ ~1; Van Voorhies et al. 2004)
ASSUMPTION(A90): apple density 0.85 g/cm3; mould respires the sugar it consumes completely.
"""
from dataclasses import dataclass

import numpy as np

MOLAR_L = 24.45   # L/mol at 25 C, 1 atm


@dataclass
class Atmosphere:
    volume_l: float = 1000.0        # container air volume
    ach: float = 6.0                # air changes per hour (ventilation); 0 = sealed
    co2_ppm: float = 420.0
    o2_pct: float = 20.9
    outside_co2_ppm: float = 420.0
    outside_o2_pct: float = 20.9

    def step(self, hours: float, apple, n_flies: int, temp_c: float = 25.0,
             soil_area_m2: float = 0.0, soil_moist: bool = False):
        """Advance by `hours` (sub-stepped). Uses the apple's current microbial rates."""
        apples = apple if isinstance(apple, (list, tuple)) else [apple]
        vm = MOLAR_L * (273.15 + temp_c) / 298.15
        n = max(1, int(np.ceil(hours / 0.05)))
        h = hours / n
        for _ in range(n):
            ferm_g_h = mold_g_h = acet_g_h = 0.0
            for ap in apples:
                s = ap.s
                mass_g = 0.85 * 4 / 3 * np.pi * (ap.p.radius_mm / 10) ** 3
                # amounts actually converted in the apple's last step (substrate-limited)
                ferm_g_h += s.ferm_rate * mass_g / 100.0                       # sugar fermented
                mold_g_h += s.mold_rate * mass_g / 100.0                       # substrate respired
                acet_g_h += s.aab_rate * mass_g / 100.0                        # ethanol oxidised
            # soil respiration (roots absent; microbes): ~4 umol CO2 m-2 s-1 at 25 C in moist soil,
            # Q10 = 2, dry soil ~0.3x; RQ ~1 (Raich & Schlesinger 1992; Lloyd & Taylor 1994) A95
            soil_mol_h = soil_area_m2 * 4e-6 * 3600 * 2.0 ** ((temp_c - 25.0) / 10.0) * (1.0 if soil_moist else 0.3)
            soil_mol_h *= self.o2_pct / (self.o2_pct + 2.0)
            co2_mol_h = ferm_g_h / 180.16 * 2 + mold_g_h / 180.16 * 6 + n_flies * 4e-6 / vm + soil_mol_h
            o2_mol_h = mold_g_h / 180.16 * 6 + acet_g_h / 46.07 + n_flies * 4e-6 / vm + soil_mol_h
            # moles in the container at 1 atm; excess gas vents (mixture leaves proportionally)
            n_tot = self.volume_l / vm
            x_co2, x_o2 = self.co2_ppm * 1e-6, self.o2_pct / 100.0
            n_co2 = x_co2 * n_tot + co2_mol_h * h
            n_o2 = max(0.0, x_o2 * n_tot - o2_mol_h * h)
            n_rest = (1.0 - x_co2 - x_o2) * n_tot + 0.0          # N2, Ar ... (H2O condenses)
            new_tot = n_co2 + n_o2 + n_rest
            x_co2, x_o2 = n_co2 / new_tot, n_o2 / new_tot
            # ventilation: exponential exchange with outside air
            f = np.exp(-self.ach * h)
            x_co2 = self.outside_co2_ppm * 1e-6 + (x_co2 - self.outside_co2_ppm * 1e-6) * f
            x_o2 = self.outside_o2_pct / 100 + (x_o2 - self.outside_o2_pct / 100) * f
            self.co2_ppm = x_co2 * 1e6
            self.o2_pct = x_o2 * 100.0
