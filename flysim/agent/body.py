"""Reduced-order fly body: walking on the floor and over the apple surface, flight, landing.

ASSUMPTION(A30): not a musculoskeletal model. The body executes motor commands from the
descending-neuron readout with literature speeds: walking up to ~20 mm/s and turning up to
~300 deg/s (Strauss & Heisenberg 1990; Mendes et al. 2013), free flight ~300 mm/s
(Fry et al. 2003; scaled down in a small arena). A weak spontaneous locomotor drive (random
walk) stands in for intrinsic activity that the connectome model lacks; experiments include
controls without it.
ASSUMPTION(A35): physical constraints, not decisions: walls deflect the walking direction
(reflection), a walking fly that runs into the apple continues onto its surface (flies walk on
vertical and overhanging surfaces), a feeding fly keeps its legs still.
"""
from dataclasses import dataclass, field

import numpy as np

FLY_RADIUS = 1.25  # mm
BODY_H = 1.0       # mm, height of the body above the substrate
LEG_REACH = 2.0    # mm: a fly on the apple can step to the floor where the gap is smaller


@dataclass
class BodyParams:
    v_walk: float = 20.0          # mm/s at full DN drive
    v_back: float = 8.0
    w_turn: float = np.radians(300.0)
    v_fly: float = 150.0          # mm/s, ASSUMPTION(A32): arena-scaled
    fly_clear: float = 15.0       # mm cruise clearance above the apple top / floor
    spont_speed: float = 4.0      # mm/s spontaneous walking (A30)
    spont_turn_sd: float = np.radians(90.0)
    max_flight_s: float = 20.0    # ASSUMPTION(A33): flight bouts end within ~20 s
    takeoff_s: float = 0.3        # jump/climb-out phase along the surface normal


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
    h: np.ndarray = None          # walking direction, unit tangent of the current surface

    def __post_init__(self):
        if self.h is None:
            self.h = np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])

    def heading(self):
        return np.array([np.cos(self.yaw), np.sin(self.yaw), 0.0])

    def normal(self, apple_c):
        if self.mode == "walk" and self.surface == "apple":
            n = self.pos - apple_c
            return n / np.linalg.norm(n)
        return np.array([0.0, 0.0, 1.0])

    def eye(self, apple_c):
        return self.pos + 0.8 * self.normal(apple_c)

    def teleport(self, pos, yaw):
        self.pos, self.yaw = np.array(pos, float), yaw
        self.mode, self.surface = "walk", "floor"
        self.h = self.heading()

    def _sync_yaw(self):
        if abs(self.h[0]) + abs(self.h[1]) > 1e-6:
            self.yaw = float(np.arctan2(self.h[1], self.h[0]))

    def _rotate_h(self, angle, n):
        # rotate the tangent heading about the surface normal (Rodrigues)
        h = self.h
        self.h = h * np.cos(angle) + np.cross(n, h) * np.sin(angle) + n * (n @ h) * (1 - np.cos(angle))
        self.h -= (self.h @ n) * n
        self.h /= max(np.linalg.norm(self.h), 1e-9)

    def step(self, cmd: dict, dt: float, apple_c: np.ndarray, apple_r: float, arena: float,
             spontaneous: bool = True, o2_pct: float = 20.9):
        p = self.p
        # ASSUMPTION(A70): hypoxia. Flies keep moving down to ~5-8 % O2, slow down below, and fall
        # into reversible stupor at <~2 % within a minute (Haddad et al. 1997; Krishnan 1997).
        vigor = float(np.clip((o2_pct - 2.0) / 6.0, 0.0, 1.0))
        if vigor <= 0.0:
            self.proboscis = False
            self.wing_ext = 0.0
            if self.mode == "fly":  # stuporous flies drop to the substrate below
                self.mode = "walk"
                self._drop(apple_c, apple_r)
            return
        self.proboscis = cmd["proboscis"] and self.mode == "walk"
        self.wing_ext = cmd.get("song", 0.0) if self.mode == "walk" else 0.0
        if self.mode == "walk":
            if cmd["takeoff"]:
                self.mode, self.flight_t = "fly", 0.0
                self._n_takeoff = self.normal(apple_c) if self.surface == "apple" else np.array([0, 0, 1.0])
                self.surface = "air"
                self.h = self.heading()
            else:
                fwd = vigor * (p.v_walk * cmd["forward"] - p.v_back * cmd["backward"])
                turn = vigor * p.w_turn * cmd["turn"]
                if spontaneous:
                    fwd += vigor * p.spont_speed
                    turn += vigor * p.spont_turn_sd * self.rng.standard_normal() / np.sqrt(dt / 0.05)
                if self.proboscis:
                    fwd, turn = 0.0, 0.0
                self._rotate_h(turn * dt, self.normal(apple_c))
                self._walk(fwd * dt, apple_c, apple_r)
                self._sync_yaw()
        if self.mode == "fly":
            self.flight_t += dt
            if self.flight_t < p.takeoff_s:
                self.pos = self.pos + (self._n_takeoff * 60.0 + self.heading() * 30.0) * dt
            else:
                power = 0.4 + 0.6 * cmd["flight_power"]
                self.yaw += vigor * 3 * p.w_turn * cmd["turn"] * dt
                if spontaneous:
                    self.yaw += vigor * p.spont_turn_sd * self.rng.standard_normal() * np.sqrt(dt)
                v = vigor * p.v_fly * power * self.heading()
                cruise = 2 * apple_r + p.fly_clear
                v[2] = 2.0 * (cruise - self.pos[2])
                if cmd["land"] or self.flight_t > p.max_flight_s:
                    v[2] = -60.0
                self.pos = self.pos + v * dt
                self._land_if_touching(apple_c, apple_r)
            self.h = self.heading()
        # arena walls: reflect the walking/flight direction (A35)
        for k in range(2):
            if abs(self.pos[k]) > arena:
                self.pos[k] = np.sign(self.pos[k]) * arena
                if self.surface == "floor" or self.mode == "fly":
                    self.yaw = np.pi - self.yaw if k == 0 else -self.yaw
                    self.h = self.heading()

    # ---------------------------------------------------------------------------------------
    def _walk(self, ds, apple_c, apple_r):
        if ds == 0.0:
            return
        if self.surface == "floor":
            new = self.pos + ds * self.h
            new[2] = 0.0
            # body front hits the apple (3D contact) while moving towards it -> walk onto it
            front = new + np.array([0, 0, BODY_H]) + FLY_RADIUS * self.h
            towards = (apple_c[:2] - new[:2]) @ self.h[:2] > 0
            if np.linalg.norm(front - apple_c) < apple_r and towards:
                self.surface = "apple"
                self.pos = apple_c + (front - apple_c) / np.linalg.norm(front - apple_c) * apple_r
                n = self.normal(apple_c)
                self.h = self.h - (self.h @ n) * n
                if np.linalg.norm(self.h) < 1e-6:
                    self.h = np.array([0.0, 0.0, 1.0]) - n[2] * n
                self.h /= np.linalg.norm(self.h)
            else:
                self.pos = new
        else:
            # move along the great circle in direction h; parallel-transport h
            n = self.normal(apple_c)
            th = ds / apple_r
            pos = apple_c + apple_r * (np.cos(th) * n + np.sin(th) * self.h)
            h = -np.sin(th) * n + np.cos(th) * self.h
            self.pos, self.h = pos, h / np.linalg.norm(h)
            # near the bottom the surface meets the floor: step off when heading down/outwards
            if self.pos[2] < LEG_REACH and self.h[2] <= 0:
                self.surface = "floor"
                self.pos = np.array([self.pos[0], self.pos[1], 0.0])
                hh = np.array([self.h[0], self.h[1], 0.0])
                out = self.pos[:2] - apple_c[:2]
                if np.linalg.norm(hh) < 1e-6:
                    hh[:2] = out
                self.h = hh / max(np.linalg.norm(hh), 1e-9)

    def _drop(self, apple_c, apple_r):
        d = self.pos[:2] - apple_c[:2]
        if np.linalg.norm(d) < apple_r:
            top = apple_c[2] + np.sqrt(max(apple_r ** 2 - d @ d, 0.0))
            self.pos = np.array([self.pos[0], self.pos[1], top])
            self.pos = apple_c + (self.pos - apple_c) / np.linalg.norm(self.pos - apple_c) * apple_r
            self.surface = "apple"
        else:
            self.pos[2] = 0.0
            self.surface = "floor"

    def _land_if_touching(self, apple_c, apple_r):
        if np.linalg.norm(self.pos - apple_c) <= apple_r:
            self.pos = apple_c + (self.pos - apple_c) / np.linalg.norm(self.pos - apple_c) * apple_r
            self.mode, self.surface = "walk", "apple"
            n = self.normal(apple_c)
            self.h = self.heading() - (self.heading() @ n) * n
            if np.linalg.norm(self.h) < 1e-6:
                self.h = np.cross(n, [0, 1.0, 0])
            self.h /= np.linalg.norm(self.h)
        elif self.pos[2] <= 0.0:
            self.pos[2] = 0.0
            self.mode, self.surface = "walk", "floor"
            self.h = self.heading()
