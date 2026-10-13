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
    apples: list = field(default_factory=list)  # all fruit; `apple` is the first one
    flies: list = field(default_factory=list)
    zones: list = field(default_factory=list)
    ambient_c: float = 25.0
    o2_pct: float = 20.9          # normal air 20.9 %
    co2_ppm: float = 420.0        # outdoor air ~420 ppm
    odor_scale: float = 1.0
    extra_odor_sources: list = field(default_factory=list)  # (xyz, {odorant: strength}, radius)
    shocks: dict = field(default_factory=dict)              # fly name -> PPL1 drive Hz
    clock_h: float = 0.0                                    # world time since start, hours
    population: object = None                               # flysim.world.population.Population
    copulation_s: float = 60.0  # ASSUMPTION(A62): real ~20 min, shortened inside a session
    arena_phys: object = None   # NMFArena when flies use NeuroMechFly bodies
    atm: object = None          # flysim.world.atmosphere.Atmosphere (container air)
    plants: object = None       # flysim.world.plant.PlantWorld (seeds, seedlings, trees)
    seasonal: bool = False      # temperate annual temperature cycle (A93)
    base_temp_c: float = 25.0
    max_apples: int = 8         # performance cap for fruit produced by trees
    wind: tuple = (0.0, 0.0)    # m/s (x, y), uniform
    weather: object = None      # flysim.world.weather.Weather (garden climate) or None (container)
    light_lux: float = None     # None = constant lab light (photoreceptor rates as in A54)
    corpses: list = field(default_factory=list)   # dead flies stay where they died
    deaths: list = field(default_factory=list)    # (clock_h, name, sex, cause, pos)

    def __post_init__(self):
        if not self.apples:
            self.apples = [self.apple]
        self.apple = self.apples[0]

    def bury_dead(self):
        dead = [f for f in self.flies if f.dead]
        for f in dead:
            self.flies.remove(f)
            rec = {"clock_h": round(self.clock_h, 4), "name": f.cfg.name, "sex": f.cfg.sex,
                   "cause": f.dead, "pos": [round(float(v), 1) for v in f.body.pos],
                   "age_d": round(f.age_d, 1)}
            self.corpses.append(rec)
            self.deaths.append(rec)
        if dead and self.arena_phys is not None:
            self.rebuild_physics()
        return dead

    def alive(self) -> bool:
        """Is there still life: adult flies or developing brood."""
        if self.flies:
            return True
        pop = getattr(self, "population", None)
        if pop is not None and sum(pop.counts(self.clock_h).values()) > 0:
            return True
        return False

    def nearest_apple(self, x):
        return min(self.apples, key=lambda a: np.linalg.norm(np.asarray(x) - a.center) - a.p.radius_mm)

    def soil_args(self):
        pl = self.plants
        if pl is None or not pl.soil:
            return {}
        return {"soil_area_m2": (2 * self.arena / 1000.0) ** 2, "soil_moist": bool(pl.moist)}

    def sync_air(self):
        if self.atm is not None:
            self.o2_pct, self.co2_ppm = self.atm.o2_pct, self.atm.co2_ppm
        for a in self.apples:
            a.o2_pct = self.o2_pct
            a.temp_c = self.ambient_c

    def advance_hours(self, hours: float):
        """Off-line time: apple chemistry and container air evolve together, hour by hour."""
        left = hours
        while left > 1e-9:
            h = min(1.0, left)
            if self.weather is not None:
                self.weather.step_hours(h, self.clock_h)
                self.apply_weather()
            elif self.seasonal:
                from flysim.world.plant import seasonal_temp
                self.ambient_c = float(seasonal_temp(self.clock_h))
            self.sync_air()
            for a in self.apples:
                a.age(h)
            if self.plants is not None:
                self.plants.step(h, self.apples, self.ambient_c, self.o2_pct, self.clock_h)
                self._tree_fruit()
            self.clock_h += h
            if self.atm is not None:
                self.atm.step(h, self.apples, len(self.flies), self.ambient_c, **self.soil_args())
            left -= h
        self.sync_air()

    def apply_weather(self):
        w = self.weather
        self.ambient_c = w.temp_c(self.clock_h)
        self.wind = w.wind_xy()
        self.light_lux = w.lux(self.clock_h)

    def _tree_fruit(self):
        """Fruiting trees drop new apples in autumn (day-of-year ~260); seeds come with them."""
        from flysim.world.apple import Apple, AppleParams
        trees = self.plants.fruiting()
        doy = (180 + self.clock_h / 24.0) % 365.0
        if not trees or not (259.5 <= doy < 260.5) or len(self.apples) >= self.max_apples:
            return
        rng = self.plants.rng
        for _ in range(min(len(trees) * 50, self.max_apples - len(self.apples))):
            ang, rr = rng.uniform(0, 2 * np.pi), rng.uniform(0.2, 0.8) * self.arena
            a = Apple(AppleParams(radius_mm=30.0), center=(rr * np.cos(ang), rr * np.sin(ang)),
                      cut=rng.random() < 0.5, temp_c=self.ambient_c, o2_pct=self.o2_pct)
            self.apples.append(a)
            self.plants.add_apple_seeds(len(self.apples) - 1)
        self.plants.events.append((self.clock_h, "дерево дало плоды: на землю упали новые яблоки"))
        self.physics_dirty = True

    def rebuild_physics(self):
        """(Re)create the shared MuJoCo world for all flies with body='nmf'."""
        from flysim.agent.body_nmf import NMFArena, NMFBody
        phys = [f for f in self.flies if f.cfg.body == "nmf"]
        if not phys:
            self.arena_phys = None
            return
        self.arena_phys = NMFArena([(a.center, a.p.radius_mm) for a in self.apples],
                                   None, [(f.cfg.name, f.body.pos, f.body.yaw) for f in phys],
                                   half_size=max(1000.0, 2 * self.arena), arena=self.arena,
                                   seeds={f.cfg.name: f.cfg.seed for f in phys})
        for f in phys:
            old = f.body
            f.body = NMFBody(self.arena_phys, f.cfg.name, old.rng)
            f.body.lunge_t = getattr(old, "lunge_t", 0.0)
            self.arena_phys.lesions[f.cfg.name] = f.schema
            ch = getattr(f, "chain", None)
            if ch is not None and ch.available:
                ch.calibrate_signs(self.arena_phys)
                self.arena_phys.chains[f.cfg.name] = ch

    # ---- fields ---------------------------------------------------------------------------
    def odor_at(self, x, me) -> dict:
        """Steady-state diffusion from spherical sources: c(r) = c_s * R / r.
        ASSUMPTION(A42): still air, no turbulence or wind."""
        # ASSUMPTION(A42b): CO2 in odour units of 1 = 1 % CO2 (Gr21a/Gr63a neurons respond from
        # ~0.1 % and drive avoidance at 0.1-1 %; Suh et al. 2004; Jones et al. 2007)
        out = {"CO2": self.co2_ppm / 1e4}
        U = np.array([self.wind[0], self.wind[1], 0.0]) * 1000.0    # mm/s
        u = np.linalg.norm(U)
        # ASSUMPTION(A96): steady advection-diffusion from a source in uniform wind,
        # c = c_s R/r * exp(-u (r - r.e_u) / (2K)), K = molecular 10 mm^2/s + turbulent u * 10 mm (eddy scale)
        K = 10.0 + 10.0 * u
        for a in self.apples:
            ac, ar = a.center, a.p.radius_mm
            d = x - ac
            r = max(np.linalg.norm(d), ar)
            f = ar / r
            if u > 0:
                f *= np.exp(-u * (r - d @ (U / u)) / (2 * K))
            for k, v in a.odor_sources().items():
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
        return self.nearest_apple(x).surface_taste()

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
        s = [(a.center, a.p.radius_mm) for a in self.apples]
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
                if self.arena_phys is not None:
                    continue  # MuJoCo resolves body contacts
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
                    if self.arena_phys is None:
                        m.body.pos = m.body.pos + 0.6 * d   # lunge (reduced body)
                    else:
                        m.lunging = 0.3  # physical body: 0.3 s full forward drive at the opponent
                    m.body.lunge_t = 0.5
                    m.events.append((round(m.t, 3), f"lunge->{o.cfg.name}"))
                    o.events.append((round(o.t, 3), f"lunged_by->{m.cfg.name}"))
                if o.cfg.sex == "female" and dist < 3.0 and m.cmd["song"] > 0.3 and \
                        o.cmd["receptive"] and not m.copulating and not o.copulating:
                    m.copulating = o.copulating = True
                    m.copulation_until = m.t + self.copulation_s
                    o.copulation_until = o.t + self.copulation_s
                    if self.arena_phys is None:
                        m.body.pos = o.body.pos - 1.5 * o.body.heading()
                        m.body.yaw = o.body.yaw
                    o.set_mated(m.cfg.name)
                    m.events.append((round(m.t, 3), f"copulation->{o.cfg.name}"))
                    o.events.append((round(o.t, 3), f"copulation<-{m.cfg.name}"))
            m.body.lunge_t -= dt

    def step(self, dt):
        if self.weather is not None:
            h0 = self.clock_h
            if int(h0) != int(h0 + dt / 3600):     # hourly weather transition inside a session
                self.weather.step_hours(1.0, h0)
            self.weather.step_seconds(dt)
            self.apply_weather()
        for f in self.flies:
            f.step(self, dt)
        if self.arena_phys is not None:
            self.arena_phys.wind = self.wind
            self.arena_phys.advance(dt)
            for f in self.flies:
                if f.cfg.body == "nmf":
                    f.body.sync()
        self.interactions(dt)
        self.bury_dead()
        if self.atm is not None:
            self.atm.step(dt / 3600, self.apples, len(self.flies), self.ambient_c, **self.soil_args())
            self.sync_air()
        # fruit chemistry also runs during detailed sessions (it changes over hours; update each
        # simulated minute so that a world run 1:1 without fast-forward still rots)
        self._live_h = getattr(self, "_live_h", 0.0) + dt / 3600
        if self._live_h >= 1 / 60:
            for a in self.apples:
                a.age(self._live_h, dt_h=self._live_h)
            self._live_h = 0.0
        self.clock_h += dt / 3600
