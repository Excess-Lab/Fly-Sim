"""Arena with an apple, optional temperature/bitter zones and shocks; runs flies in closed loop."""
from dataclasses import dataclass, field

import numpy as np

from flysim.world.apple import Apple


@dataclass
class Zone:
    center: tuple
    radius: float
    temp_c: float = 25.0
    bitter: bool = False
    active: bool = True


@dataclass
class World:
    apple: Apple
    arena: float = 60.0                    # half-width, mm
    flies: list = field(default_factory=list)
    zones: list = field(default_factory=list)
    ambient_c: float = 25.0
    odor_scale: float = 1.0
    extra_odor_sources: list = field(default_factory=list)  # (xyz, {odorant: strength}, radius)
    shocks: dict = field(default_factory=dict)              # fly name -> PPL1 drive Hz
    clock_h: float = 0.0                                    # world time since start, hours
    population: object = None                               # flysim.world.population.Population
    copulation_s: float = 60.0  # ASSUMPTION(A62): real ~20 min, shortened inside a session

    # ---- fields ---------------------------------------------------------------------------
    def odor_at(self, x, me) -> dict:
        """Steady-state diffusion from spherical sources: c(r) = c_s * R / r.
        ASSUMPTION(A42): still air, no turbulence or wind."""
        out = {}
        ac, ar = self.apple.center, self.apple.p.radius_mm
        f = ar / max(np.linalg.norm(x - ac), ar)
        for k, v in self.apple.odor_sources().items():
            out[k] = out.get(k, 0.0) + self.odor_scale * v * f
        for pos, comp, rad in self.extra_odor_sources:
            f = rad / max(np.linalg.norm(x - np.asarray(pos)), rad)
            for k, v in comp.items():
                out[k] = out.get(k, 0.0) + v * f
        # ASSUMPTION(A44): males carry cVA (11-cis-vaccenyl acetate) on the cuticle
        for other in self.flies:
            if other is me or other.cfg.sex != "male":
                continue
            d = np.linalg.norm(x - other.body.pos)
            out["cVA"] = out.get("cVA", 0.0) + 0.5 * other.radius / max(d, other.radius)
        return out

    def taste_at(self, x) -> dict:
        return self.apple.surface_taste()

    def temperature_at(self, x) -> float:
        t = self.ambient_c
        for z in self.zones:
            if z.active and np.linalg.norm(np.asarray(x[:2]) - np.asarray(z.center)) < z.radius:
                # heat is felt near the floor (heated floor arena, Ofstad et al. 2011)
                if x[2] < 5.0:
                    t = max(t, z.temp_c)
        return t

    def bitter_zone_at(self, x) -> bool:
        return any(z.active and z.bitter and
                   np.linalg.norm(np.asarray(x[:2]) - np.asarray(z.center)) < z.radius
                   for z in self.zones)

    def shock_at(self, fly) -> float:
        return float(self.shocks.get(fly.cfg.name, 0.0))

    def visual_spheres(self, me):
        s = [(self.apple.center, self.apple.p.radius_mm)]
        for other in self.flies:
            if other is not me:
                s.append((other.body.pos + np.array([0, 0, 0.8]), other.radius))
        return s

    def touching(self, me):
        for other in self.flies:
            if other is not me and np.linalg.norm(other.body.pos - me.body.pos) < 2.5:
                return other
        return None

    def song_at(self, me) -> float:
        """Courtship song is near-field: amplitude falls ~1/r^2 beyond ~2 mm (A55)."""
        amp = 0.0
        for other in self.flies:
            if other is me or other.body.wing_ext <= 0:
                continue
            d = np.linalg.norm(other.body.pos - me.body.pos)
            amp += other.body.wing_ext * min(1.0, (2.0 / max(d, 1e-3)) ** 2)
        return min(amp, 1.0)

    # ---- interactions ---------------------------------------------------------------------
    def interactions(self, dt):
        flies = self.flies
        for i, a in enumerate(flies):
            for b in flies[i + 1:]:
                d = b.body.pos - a.body.pos
                dist = np.linalg.norm(d)
                if dist < 2 * a.radius and dist > 1e-6 and a.body.mode == b.body.mode == "walk":
                    push = (2 * a.radius - dist) / 2 * d / dist
                    a.body.pos -= push
                    b.body.pos += push
        # male behaviours triggered by brain readouts near another fly (A46, A47)
        for m in flies:
            if m.cfg.sex != "male" or m.body.mode != "walk":
                continue
            for o in flies:
                if o is m:
                    continue
                d = o.body.pos - m.body.pos
                dist = np.linalg.norm(d)
                facing = np.dot(d[:2], m.body.heading()[:2]) / max(dist, 1e-6)
                if o.cfg.sex == "male" and dist < 4.0 and facing > 0.7 and \
                        m.cmd["arousal"] > 0.5 and m.body.lunge_t <= 0:
                    m.body.pos = m.body.pos + 0.6 * d   # lunge
                    m.body.lunge_t = 0.5
                    m.events.append((round(m.t, 3), f"lunge->{o.cfg.name}"))
                    o.events.append((round(o.t, 3), f"lunged_by->{m.cfg.name}"))
                if o.cfg.sex == "female" and dist < 3.0 and m.cmd["song"] > 0.3 and \
                        o.cmd["receptive"] and not m.copulating and not o.copulating:
                    m.copulating = o.copulating = True
                    m.copulation_until = m.t + self.copulation_s
                    o.copulation_until = o.t + self.copulation_s
                    m.body.pos = o.body.pos - 1.5 * o.body.heading()
                    m.body.yaw = o.body.yaw
                    o.set_mated(m.cfg.name)
                    m.events.append((round(m.t, 3), f"copulation->{o.cfg.name}"))
                    o.events.append((round(o.t, 3), f"copulation<-{m.cfg.name}"))
            m.body.lunge_t -= dt

    def step(self, dt):
        for f in self.flies:
            f.step(self, dt)
        self.interactions(dt)
        self.clock_h += dt / 3600
