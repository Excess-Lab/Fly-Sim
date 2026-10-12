"""Reduced-order fly body: walking on floor/apple surface, flight, landing, contacts.

ASSUMPTION(A30): not a musculoskeletal model. The body executes motor commands from the
descending-neuron readout with literature speeds: walking up to ~20 mm/s and turning up to
~300 deg/s (Strauss & Heisenberg 1990; Mendes et al. 2013), free flight ~300 mm/s
(Fry et al. 2003; scaled down in a small arena). A weak spontaneous locomotor drive (random
walk) stands in for intrinsic activity that the connectome model lacks; experiments include
controls where it is the only driver.
"""
from dataclasses import dataclass, field

import numpy as np

FLY_RADIUS = 1.25  # mm


@dataclass
class BodyParams:
    v_walk: float = 20.0          # mm/s at full DN drive
    v_back: float = 8.0
    w_turn: float = np.radians(300.0)
    v_fly: float = 150.0          # mm/s, ASSUMPTION(A32): arena-scaled
    fly_alt: float = 25.0         # mm cruise altitude
    spont_speed: float = 4.0      # mm/s spontaneous walking (A30)
    spont_turn_sd: float = np.radians(90.0)
    max_flight_s: float = 20.0    # ASSUMPTION(A33): flight bouts end within ~20 s (lands)


@dataclass
class Body:
    pos: np.ndarray
    yaw: float
    p: BodyParams = field(default_factory=BodyParams)
    mode: str = "walk"            # walk | fly
    surface: str = "floor"        # floor | apple
    flight_t: float = 0.0
    proboscis: bool = False
    wing_ext: float = 0.0
    lunge_t: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def heading(self):
        return np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])

    def step(self, cmd: dict, dt: float, apple_c: np.ndarray, apple_r: float, arena: float,
             spontaneous: bool = True, o2_pct: float = 20.9):
        p = self.p
        # ASSUMPTION(A70): hypoxia. Flies keep moving down to ~5-8 % O2, slow down below, and fall
        # into reversible stupor at <~2 % within a minute (Haddad et al. 1997; Krishnan 1997).
        vigor = float(np.clip((o2_pct - 2.0) / 6.0, 0.0, 1.0))
        if vigor <= 0.0:
            if self.mode == "fly":
                self.mode, self.flight_t = "walk", 0.0
                self.pos[2] = 0.0 if self.surface == "floor" else self.pos[2]
            self.proboscis = False
            self.wing_ext = 0.0
            return
        self.proboscis = cmd["proboscis"] and self.mode == "walk"
        self.wing_ext = cmd.get("song", 0.0) if self.mode == "walk" else 0.0
        if self.mode == "walk":
            if cmd["takeoff"]:
                self.mode, self.flight_t = "fly", 0.0
            else:
                fwd = vigor * (p.v_walk * cmd["forward"] - p.v_back * cmd["backward"])
                turn = vigor * p.w_turn * cmd["turn"]
                if spontaneous and not self.proboscis:
                    fwd += vigor * p.spont_speed
                    turn += p.spont_turn_sd * self.rng.standard_normal() / np.sqrt(dt / 0.05)
                if self.proboscis:
                    fwd, turn = 0.0, 0.0  # feeding flies stand still
                self.yaw += turn * dt
                self._walk(fwd * dt, apple_c, apple_r)
        if self.mode == "fly":
            self.flight_t += dt
            power = 0.4 + 0.6 * cmd["flight_power"]
            self.yaw += 3 * p.w_turn * cmd["turn"] * dt
            if spontaneous:
                self.yaw += p.spont_turn_sd * self.rng.standard_normal() * np.sqrt(dt)
            v = p.v_fly * power * self.heading()
            v[2] = 2.0 * (p.fly_alt - self.pos[2])
            landing = cmd["land"] or self.flight_t > p.max_flight_s
            if landing:
                v[2] = -60.0
            self.pos = self.pos + v * dt
            self._land_if_touching(apple_c, apple_r)
        # arena walls
        for k in range(2):
            if abs(self.pos[k]) > arena:
                self.pos[k] = np.sign(self.pos[k]) * arena
                self.yaw += np.pi * 0.5 + self.rng.uniform(0, np.pi)

    def _walk(self, ds, apple_c, apple_r):
        if self.surface == "floor":
            new = self.pos + ds * self.heading()
            new[2] = 0.0
            # walking into the apple: climb onto it (flies readily walk on vertical surfaces)
            if np.linalg.norm(new[:2] - apple_c[:2]) < apple_r * 0.98:
                self.surface = "apple"
                new = self._on_sphere(new, apple_c, apple_r)
            self.pos = new
        else:
            new = self._on_sphere(self.pos + ds * self._tangent(apple_c), apple_c, apple_r)
            if new[2] < 1.0:  # walked down to the floor
                self.surface = "floor"
                new[2] = 0.0
            self.pos = new

    def _tangent(self, apple_c):
        n = self.pos - apple_c
        n /= np.linalg.norm(n)
        h = self.heading()
        t = h - (h @ n) * n
        nt = np.linalg.norm(t)
        return t / nt if nt > 1e-6 else np.array([-n[1], n[0], 0.0])

    @staticmethod
    def _on_sphere(x, c, r):
        d = x - c
        return c + d / max(np.linalg.norm(d), 1e-9) * r

    def _land_if_touching(self, apple_c, apple_r):
        if np.linalg.norm(self.pos - apple_c) <= apple_r:
            self.pos = self._on_sphere(self.pos, apple_c, apple_r)
            self.mode, self.surface = "walk", "apple"
        elif self.pos[2] <= 0.0:
            self.pos[2] = 0.0
            self.mode, self.surface = "walk", "floor"
