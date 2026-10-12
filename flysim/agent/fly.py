"""A fly: connectome brain + senses + metabolism + motor readout + body, in closed loop."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from flysim.agent.body import FLY_RADIUS, Body
from flysim.agent.metabolism import Metabolism
from flysim.agent.motor import MotorReadout
from flysim.agent.senses import Gustation, Hearing, Interoception, Olfaction, Thermo, Vision
from flysim.brain.factory import load, make_brain


@dataclass
class FlyConfig:
    name: str
    sex: str = "female"
    starved_h: float = 24.0
    plasticity: bool = True
    # ASSUMPTION(A24): the noxious-stimulus -> PPL1 pathway is not recovered by the connectome
    # model (body nociception enters via ascending/neuromodulatory routes absent from it), so the
    # measured physiology is used: punishment (heat >30 C, shock) excites PPL1 DANs
    # (Mao & Davis 2009; Riemensperger et al. 2005). Set False for the pure-connectome control.
    us_punish_shortcut: bool = True
    # ASSUMPTION(A25): sugar ingestion excites reward PAM DANs (Liu et al. 2012; Yamagata 2015).
    us_reward_shortcut: bool = True
    spontaneous: bool = True
    seed: int = 0
    body: str = "nmf"   # always NeuroMechFly physics; "simple" kept only for unit tests
    injuries: dict = None  # experimenter-set lesions (body_schema.BodySchema.from_config)
    connectome: str = ""   # "" = default for the sex; "female_banc" = female full CNS (BANC)


class Fly:
    BRAIN_DT_MS = 10.0  # closed-loop update period

    def __init__(self, cfg: FlyConfig, pos, yaw):
        self.cfg = cfg
        ds = cfg.connectome or cfg.sex
        con, ann = load(ds)
        self.ann = ann
        self.dataset = ds
        self.dataset_name = {"female": "FlyWire v783 (brain only)",
                             "female_banc": "BANC 888 (brain + VNC, experimental)",
                             "male": "Janelia Male CNS v1.0 (brain + VNC)"}[ds]
        self.brain, _ = make_brain(ds, seed=cfg.seed)
        from flysim.agent.body_schema import BodySchema
        self.schema = BodySchema.from_config(cfg.injuries)
        t = pd.Series(ann.type.to_numpy())
        self.kc = np.flatnonzero(t.str.match(r"^KC").to_numpy())
        self.mbon = np.flatnonzero(t.str.match(r"^MBON").to_numpy())
        self.ppl1 = np.flatnonzero(t.str.match(r"^PPL1").to_numpy())
        self.pam = np.flatnonzero(t.str.match(r"^PAM").to_numpy())
        dan = np.concatenate([self.ppl1, self.pam])
        if cfg.plasticity:
            self.brain.enable_plasticity(self.kc, self.mbon, dan)
        self.olf = Olfaction(ann)
        self.gust = Gustation(ann)
        self.thermo = Thermo(ann)
        self.vision = Vision(con, ann, cfg.sex)
        self.hearing = Hearing(ann)
        self.intero = Interoception(ann)
        # contact pheromone neurons (male: ppk23 receptor class) ASSUMPTION(A57)
        rec = ann["receptor"] if "receptor" in ann.columns else pd.Series([""] * len(ann))
        self.ppk23 = np.flatnonzero((rec == "putative_ppk23").to_numpy())
        # leg nociceptor candidates per leg (male VNC only). ASSUMPTION(A80): adult leg
        # nociception (noxious heat >~40 C; TrpA1/painless md neurons) has no annotated class;
        # the 'unknown_sensory' neurons entering through each leg nerve are used as candidates.
        self.leg_noci = {}
        if ann.super_class.str.contains("vnc").any():
            from flysim.agent.body_nmf import LEG_NERVE, LEGS
            cc = ann.cell_class.to_numpy()
            md = cc == "multidendritic_neuron"
            cand = md if md.any() else (cc == "unknown_sensory")   # A80: male candidates
            en = ann["entry_nerve"].fillna("").to_numpy() if "entry_nerve" in ann.columns else \
                np.array([""] * len(ann))
            for leg in LEGS:
                m = cand & (en == LEG_NERVE[leg[1]]) & \
                    (ann.side.to_numpy() == ("left" if leg[0] == "l" else "right"))
                self.leg_noci[leg] = np.flatnonzero(m)
        self.noci_idx = np.concatenate(list(self.leg_noci.values())).astype(np.int64) if self.leg_noci \
            else np.zeros(0, np.int64)
        # full motor chain through the VNC (male only; A81-A84)
        self.chain = None
        if ann.super_class.str.contains("vnc").any():
            from flysim.agent.motor_chain import MotorChain
            self.chain = MotorChain(self)
        self.prop_idx = self.chain.prop_idx if self.chain is not None else np.zeros(0, np.int64)
        self.stim_idx = np.concatenate([
            self.olf.idx, self.gust.idx, self.thermo.idx, self.vision.idx, self.hearing.idx,
            self.intero.idx, self.ppk23, self.ppl1, self.pam, self.noci_idx,
            self.prop_idx]).astype(np.int64)
        self.brain.set_poisson_targets(np.concatenate([
            self.olf.idx, self.gust.idx, self.thermo.idx, self.vision.idx, self.hearing.idx,
            self.intero.idx, self.ppk23, self.noci_idx, self.prop_idx]))
        self.motor = MotorReadout(ann)
        from flysim.agent.physics import BodyTemperature
        self.thermal = BodyTemperature(25.0)
        self.meta = Metabolism.starved(cfg.starved_h)
        self.body = Body(pos=np.array(pos, float), yaw=yaw, rng=np.random.default_rng(cfg.seed))
        self.counts_acc = np.zeros(con.n, np.int32)
        self.cmd = self.motor.commands()
        self.events = []
        self.t = 0.0
        self.mated = False
        self.mate = None
        self.copulating = False
        self.copulation_until = -1.0
        self.last_egg_t = -1e9
        self.age_d = 3.0
        self.pc1 = np.flatnonzero(t.str.match(r"^pC1").to_numpy())

    def set_mated(self, mate_name: str):
        """ASSUMPTION(A60): male sex peptide silences SPSN -> SAG input to pC1 in the female
        (Feng et al. 2014; Wang et al. 2020). Modelled as reduced synaptic gain onto pC1 (x0.3);
        reduced receptivity and increased egg-laying must then emerge from the connectome."""
        self.mated = True
        self.mate = mate_name
        if self.cfg.sex == "female":
            self.brain.gain[self.pc1] = 0.3

    # ------------------------------------------------------------------
    def antenna_positions(self):
        h = self.body.heading()
        left = np.array([-h[1], h[0], 0.0])
        return self.body.pos + 0.5 * h + 0.3 * left, self.body.pos + 0.5 * h - 0.3 * left

    def sense(self, world) -> np.ndarray:
        gains = self.meta.sensory_gains()
        self.olf.set_gains(gains)
        self.gust.gains = {"sugar": gains["sugar"], "bitter": gains["bitter"]}
        al, ar = self.antenna_positions()
        r_olf = self.olf.rates(world.odor_at(al, self), world.odor_at(ar, self))
        on_apple = self.body.surface == "apple" and self.body.mode == "walk"
        # ASSUMPTION(A43): labellar hairs touch the substrate while standing on food (0.3), full
        # contact with proboscis extended (1.0)
        contact = (1.0 if self.body.proboscis else 0.3) if on_apple else 0.0
        taste = world.taste_at(self.body.pos) if on_apple else {}
        taste = dict(taste)
        if world.bitter_zone_at(self.body.pos):
            taste["bitter"] = 1.0
            contact = max(contact, 0.3)
        r_gust = self.gust.rates(taste, contact)
        # heat exchange: body temperature from air and from the substrate under the legs (A38)
        t_air = world.ambient_c
        t_floor = world.temperature_at(np.array([self.body.pos[0], self.body.pos[1], 0.0]))
        contacts = getattr(self.body, "contacts", None)
        if contacts is None:   # simple body: all legs on its surface
            on_floor = self.body.mode == "walk" and self.body.surface == "floor"
            contacts = {leg: ("floor" if on_floor else ("apple" if self.body.mode == "walk" else None))
                        for leg in ("lf", "lm", "lh", "rf", "rm", "rh")}
        n_floor = sum(1 for c in contacts.values() if c == "floor")
        n_any = sum(1 for c in contacts.values() if c)
        t_surf = (t_floor * n_floor + t_air * (n_any - n_floor)) / n_any if n_any else t_air
        temp = self.thermal.step(t_air, t_surf, n_any / 6.0, 0.01)
        r_th = self.thermo.rates(temp)
        # leg nociceptors: noxious substrate temperature under each leg (>40 C), male VNC
        r_noci = np.zeros(len(self.noci_idx))
        if len(self.noci_idx):
            off = 0
            for leg, ids in self.leg_noci.items():
                hot = t_floor if contacts.get(leg) == "floor" else t_air
                r_noci[off:off + len(ids)] = 2.0 + 20.0 * max(0.0, hot - 40.0)
                off += len(ids)
        ap = world.nearest_apple(self.body.pos)
        r_vis = self.vision.rates(self.body.eye(ap.center), self.body.yaw,
                                  world.visual_spheres(self))
        r_hear = self.hearing.rates(world.song_at(self))
        r_int = self.intero.rates(self.meta.energy)
        touch = world.touching(self)
        r_ppk = np.full(len(self.ppk23), 2.0 + (80.0 if touch is not None else 0.0))
        punish = 0.0
        if self.cfg.us_punish_shortcut:
            punish = 8.0 * max(0.0, temp - 30.0) + world.shock_at(self)
        r_ppl1 = np.full(len(self.ppl1), punish)
        reward = 0.0
        if self.cfg.us_reward_shortcut and self.body.proboscis and on_apple:
            reward = 30.0 * ap.nutrition()["sugar_frac"] * gains["sugar"]
        r_pam = np.full(len(self.pam), reward)
        self.last_temp = temp
        # lesions (body schema): lost or silenced sensors (A85)
        sch = self.schema
        side_olf, side_th = self.olf.side, self.ann.side.to_numpy()
        for sd in ("left", "right"):
            if sch.antenna[sd] != "intact":
                r_olf[side_olf == sd] = 0.0
                r_hear[side_th[self.hearing.idx] == sd] = 0.0
                r_th[side_th[self.thermo.idx] == sd] = 0.0
            if sch.eye[sd] != "open":
                r_vis[side_th[self.vision.idx] == sd] = 5.0
        if len(self.noci_idx):
            off = 0
            for leg, ids in self.leg_noci.items():
                if sch.legs[leg] == "amputated":
                    r_noci[off:off + len(ids)] = 0.0
                off += len(ids)
        if self.chain is not None:
            arena = getattr(self.body, "arena", None)
            r_prop = self.chain.proprio_rates(arena) if arena is not None and self.body.mode == "walk" \
                else np.full(len(self.prop_idx), 5.0)
        else:
            r_prop = np.zeros(0)
        if self.chain is not None and len(r_prop):
            off = 0
            for (leg, kind), ids in self.chain.prop.items():
                if sch.legs[leg] == "amputated":
                    r_prop[off:off + len(ids)] = 0.0
                off += len(ids)
        return np.concatenate([r_olf, r_gust, r_th, r_vis, r_hear, r_int, r_ppk, r_ppl1, r_pam,
                               r_noci, r_prop])

    def step(self, world, dt: float):
        rates = self.sense(world)
        n_steps = int(round(dt * 1000 / self.brain.p.dt))
        counts, _ = self.brain.step(n_steps, self.stim_idx, rates)
        self.counts_acc += counts
        self.motor.update(counts, dt * 1000)
        self.cmd = self.motor.commands()
        if self.chain is not None:
            self.chain.update(counts, dt * 1000)
            # giant fibre -> tergotrochanter (jump) motor neurons -> take-off jump (male VNC)
            if self.chain.jump_rate > 50.0:
                self.cmd["takeoff"] = True
        self.schema.update_function(getattr(self.body, "contacts", {}) or {}, self.chain)
        was = (self.body.mode, self.body.surface, self.body.proboscis)
        if self.copulating and self.t > self.copulation_until:
            self.copulating = False
            self.events.append((round(self.t, 3), "copulation ended"))
        if self.copulating or getattr(self, "lunging", 0) > 0:
            cmd = dict(self.cmd)
            if self.copulating:   # pair stays coupled: no locomotion
                cmd.update(forward=0.0, backward=0.0, turn=0.0, takeoff=False)
            else:
                cmd.update(forward=1.0, turn=0.0, takeoff=False)
                self.lunging -= dt
            near = world.nearest_apple(self.body.pos)
            self.body.step(cmd, dt, near.center, near.p.radius_mm, world.arena,
                           spontaneous=False, o2_pct=world.o2_pct * self.meta.health * self._chill())
        else:
            near = world.nearest_apple(self.body.pos)
            self.body.step(self.cmd, dt, near.center, near.p.radius_mm, world.arena,
                           spontaneous=self.cfg.spontaneous,
                           o2_pct=world.o2_pct * self.meta.health * self._chill())  # poison, cold
        # ASSUMPTION(A61): a mated female standing on the fruit lays an egg when oviDNs fire,
        # at most one egg per 30 s
        if (self.cfg.sex == "female" and self.mated and self.cmd.get("oviposit")
                and self.body.surface == "apple" and self.t - self.last_egg_t > 30.0
                and getattr(world, "population", None) is not None):
            near = world.nearest_apple(self.body.pos)
            world.population.lay(world.clock_h, self.cfg.name, self.mate, self.body.pos,
                                 apple_idx=world.apples.index(near))
            self.last_egg_t = self.t
            self.events.append((round(self.t, 3), "laid egg"))
        self.meta.temp_c = getattr(self, "last_temp", world.ambient_c)  # local (zone) temperature
        self.meta.o2_pct = world.o2_pct
        on_food = self.body.proboscis and self.body.surface == "apple"
        nut = world.nearest_apple(self.body.pos).nutrition()
        self.meta.step(dt, nut["sugar_frac"], ingesting=on_food)
        dose = nut["patulin"] * self.meta.p.ingest_nl_per_s * 1e-3 * dt / self.meta.p.meal_volume_ul \
            if on_food else 0.0
        self.meta.ingest_toxin(dose, dt)
        now = (self.body.mode, self.body.surface, self.body.proboscis)
        if now != was:
            self.events.append((round(self.t, 3), f"{was}->{now}"))
        self.t += dt

    def _chill(self) -> float:
        """ASSUMPTION(A93): chill coma below ~7 C body temperature, full activity from ~15 C
        (David et al. 1998; MacMillan & Sinclair 2011). Applied as a vigour factor."""
        return float(np.clip((self.thermal.T - 7.0) / 8.0, 0.0, 1.0))

    def pass_time(self, hours: float, world, fed_from_fruit: bool = False):
        """The organism lives through `hours` off-line (between sessions / retention intervals):
        its own metabolism runs and memories fade (A23). The experimenter only lets time pass."""
        m = self.meta
        m.temp_c, m.o2_pct = world.ambient_c, world.o2_pct
        best = max(world.apples, key=lambda a: a.nutrition()["sugar_frac"])  # flies find the best fruit
        sugar_frac = best.nutrition()["sugar_frac"]
        patulin = best.nutrition()["patulin"]
        whole, rest = int(hours), hours - int(hours)
        for dt_h in [1.0] * whole + ([rest] if rest > 0 else []):
            # A75: living on the fruit, the fly takes a meal when hungry and sugar is left
            if fed_from_fruit and m.hunger > 0.3 and sugar_frac > 0.05 and world.o2_pct > 2:
                m.crop_ul = m.p.meal_volume_ul
                m.crop_sugar_ul = m.p.meal_volume_ul * sugar_frac
                m.ingest_toxin(patulin, 0.0)
            m.step(3600.0 * dt_h)
            m.ingest_toxin(0.0, 3600.0 * dt_h)
            self.cold_h = (getattr(self, "cold_h", 0.0) + dt_h) if world.ambient_c < 5.0 else 0.0
            self.starving_h = (getattr(self, "starving_h", 0.0) + dt_h) if m.energy <= 0.0 else 0.0
        self.brain.forget(hours * 3600)
        self.age_d += hours / 24
        self.copulating = False

    def calibrate(self, world, seconds: float = 0.0, dt: float = 0.01):
        """Deprecated: steering adaptation is intrinsic now (MotorReadout.adapt, A59)."""
        return

    def pop_counts(self):
        c = self.counts_acc.copy()
        self.counts_acc[:] = 0
        return c

    @property
    def radius(self):
        return FLY_RADIUS
