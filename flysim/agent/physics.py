"""Physics at the fly's scale that MuJoCo walking does not cover: flight aerodynamics, free fall
and heat exchange.

Flight (quasi-steady, body as a point mass; Dickinson et al. 1999 Science; Fry et al. 2003 Science;
Lehmann & Dickinson 1997):
  mass m = 1.0 mg (female ~1.1, male ~0.8); g = 9.81 m/s^2
  wings produce an aerodynamic force F = k_F * A(t)^2 directed along the stroke plane normal
  (tilted forward by the body pitch), with A = stroke amplitude 0..1 (DNg02 drive). At A=0.65
  (typical hovering amplitude ~140 deg) the force equals the weight (Lehmann & Dickinson 1997:
  flies produce up to ~1.6-1.9 x body weight).
  body drag F_d = -0.5 * rho * C_d * S * |v| v, rho = 1.2 kg/m^3, C_d = 1.0, S = 1.5 mm^2
  (frontal area); without wing force the fly falls and reaches terminal velocity ~3 m/s.
  flapping counter-force (FCF): flapping wings add a drag proportional to airspeed,
  F = -(m / tau_fcf) * (A / A_hover) * v_air, tau_fcf = 0.15 s (Hedrick et al. 2009 Science;
  Fry et al. 2003), so forward flight saturates at ~0.5-1 m/s as in D. melanogaster.
Heat (lumped Newtonian exchange; Stevenson 1985; small insects equilibrate within seconds):
  dT_b/dt = (T_air - T_b)/tau_air + k_cond * contact_frac * (T_surface - T_b)
  tau_air ~ 5 s (convection, 1 mg body), conduction through tarsi ~ 4x faster when all legs touch.
ASSUMPTION(A37, A38): constants as above; see ASSUMPTIONS.md.
"""
from dataclasses import dataclass

import numpy as np

G = 9.81e3          # mm/s^2
RHO = 1.2e-9        # kg/mm^3


@dataclass
class FlightParams:
    mass_kg: float = 1.0e-6
    hover_amp: float = 0.65      # stroke amplitude (0..1) at which lift = weight
    max_force_ratio: float = 1.8  # peak aerodynamic force / weight (Lehmann & Dickinson 1997)
    cd: float = 1.0
    area_mm2: float = 1.5
    pitch_fwd_rad: float = np.radians(25.0)  # stroke-plane tilt for forward flight
    tau_fcf_s: float = 0.15                   # flapping counter-force time constant


class FlightDynamics:
    """Integrates position/velocity of a flying fly under wing force, drag and gravity."""

    def __init__(self, p: FlightParams | None = None):
        self.p = p or FlightParams()
        self.v = np.zeros(3)

    def wing_force(self, amp: float, yaw: float, climb: float) -> np.ndarray:
        """amp 0..1 stroke amplitude; climb -1..1 tilts the stroke plane back/forward."""
        p = self.p
        w = p.mass_kg * G
        k = w / p.hover_amp ** 2
        mag = min(k * amp ** 2, p.max_force_ratio * w)
        tilt = p.pitch_fwd_rad * (1.0 - 0.5 * climb)
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        return mag * (np.cos(tilt) * np.array([0, 0, 1.0]) + np.sin(tilt) * fwd)

    def step(self, pos, yaw, amp, climb, dt, wind=np.zeros(3)):
        p = self.p
        rel = self.v - wind
        drag = -0.5 * RHO * p.cd * p.area_mm2 * np.linalg.norm(rel) * rel   # kg mm/s^2
        f = self.wing_force(amp, yaw, climb) + drag + np.array([0, 0, -p.mass_kg * G])
        # semi-implicit Euler (stable for these time constants at dt = 10 ms substeps)
        sub = 10
        h = dt / sub
        for _ in range(sub):
            rel = self.v - wind
            drag = -0.5 * RHO * p.cd * p.area_mm2 * np.linalg.norm(rel) * rel
            drag = drag - p.mass_kg / p.tau_fcf_s * (amp / p.hover_amp) * rel   # FCF
            f = self.wing_force(amp, yaw, climb) + drag + np.array([0, 0, -p.mass_kg * G])
            self.v = self.v + f / p.mass_kg * h
            pos = pos + self.v * h
        return pos


@dataclass
class HeatParams:
    tau_air_s: float = 5.0
    tau_cond_s: float = 1.2


class BodyTemperature:
    def __init__(self, t0: float = 25.0, p: HeatParams | None = None):
        self.T = t0
        self.p = p or HeatParams()

    def step(self, t_air: float, t_surface: float, contact_frac: float, dt: float) -> float:
        p = self.p
        k_air = 1.0 / p.tau_air_s
        k_c = contact_frac / p.tau_cond_s
        k = k_air + k_c
        t_eq = (k_air * t_air + k_c * t_surface) / k
        self.T = t_eq + (self.T - t_eq) * np.exp(-k * dt)
        return self.T
