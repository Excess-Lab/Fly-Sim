"""Full motor chain (male Male CNS): brain -> descending neurons -> VNC -> leg motor neurons ->
muscles -> joints of the NeuroMechFly body, with proprioceptive feedback body -> VNC.

What is reconstructed and what is approximated (ASSUMPTION A81-A84):
  A81  The connectome VNC does not generate a stepping rhythm (tested: scripts/probe_vnc_rhythm.py;
       leg MNs fire tonically, <1 % spectral power at 2-30 Hz). Real rhythm needs intrinsic
       membrane properties, neuromodulation, gap junctions and sensory feedback that the
       connectome LIF lacks. The stepping rhythm therefore still comes from NeuroMechFly's CPG;
       motor-neuron activity is ADDED on top of it through muscles (posture, reflexes,
       withdrawal, jump). Both contributions are logged separately.
  A82  Muscle: first-order activation dynamics da/dt = (u - a)/tau, tau = 10 ms rising /
       30 ms falling (insect leg muscle twitch kinetics), u = MN rate / 100 Hz. Antagonist
       pairs set a joint offset  dtheta = gain * (a_agonist - a_antagonist), gain 0.6 rad.
  A83  Flexion/extension direction of each joint is measured on the NeuroMechFly model
       kinematically (flexion = claw approaches the coxa; promotion = claw moves forward;
       depression = claw moves ventrally), not assumed.
  A84  Proprioceptors per leg (entry nerve + side): chordotonal organ <- femur-tibia angle,
       hair plates <- coxa angle, campaniform sensilla <- leg load (contact). Rates 5-150 Hz.
Female (FlyWire brain only) has no VNC: the chain is unavailable and walking uses DN drive only.
"""
import mujoco
import numpy as np
import pandas as pd
import pyarrow.feather as feather

from flysim.data.malecns import ANN

LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
SUB = {"f": "fl", "m": "ml", "h": "hl"}
NERVE = {"f": "ProLN", "m": "MesoLN", "h": "MetaLN"}
# (joint DoF suffix, agonist muscles, antagonist muscles, kinematic test)
JOINTS = {
    "coxa_pitch": ("c_thorax-{l}_coxa-pitch", ["Sternal anterior rotator MN"],
                   ["Sternal posterior rotator MN", "Pleural remotor/abductor MN"], "promote"),
    "femur_pitch": ("{l}_coxa-{l}_trochanterfemur-pitch",
                    ["Tr flexor MN", "Acc. tr flexor MN", "Sternotrochanter MN", "Tergotr. MN"],
                    ["Tr extensor MN"], "depress"),
    "femur_roll": ("{l}_coxa-{l}_trochanterfemur-roll", ["Fe reductor MN"], [], "depress"),
    "tibia_pitch": ("{l}_trochanterfemur-{l}_tibia-pitch", ["Ti flexor MN", "Acc. ti flexor MN"],
                    ["Ti extensor MN"], "flex"),
    "tarsus_pitch": ("{l}_tibia-{l}_tarsus1-pitch", ["Ta depressor MN", "ltm MN", "ltm1-tibia MN",
                                                     "ltm2-femur MN"], ["Ta levator MN"], "depress"),
}
GAIN = 0.6


class MotorChain:
    def __init__(self, fly, arena=None):
        con, ann = fly.brain.con, fly.ann
        a = feather.read_table(ANN, columns=["bodyId", "subclass", "entryNerve"]).to_pandas() \
            .set_index("bodyId").reindex(con.ids)
        t = ann.type.to_numpy()
        side = ann.side.to_numpy()
        sub = a.subclass.fillna("").to_numpy()
        mn = ann.super_class.to_numpy() == "vnc_motor"
        self.groups = {}   # (leg, joint, 'ag'|'an') -> neuron indices
        for leg in LEGS:
            s = "left" if leg[0] == "l" else "right"
            for j, (_, ag, an, _) in JOINTS.items():
                for tag, names in (("ag", ag), ("an", an)):
                    m = mn & (sub == SUB[leg[1]]) & (side == s) & np.isin(t, names)
                    self.groups[(leg, j, tag)] = np.flatnonzero(m)
        jump = mn & (t == "Tergotr. MN")
        self.jump_mn = np.flatnonzero(jump)
        # proprioceptors
        cc = ann.cell_sub_class.to_numpy()
        en = a.entryNerve.fillna("").to_numpy()
        prop = ann.cell_class.to_numpy() == "mechanosensory_proprioceptive"
        self.prop = {}
        for leg in LEGS:
            s = "left" if leg[0] == "l" else "right"
            base = prop & (en == NERVE[leg[1]]) & (side == s)
            self.prop[(leg, "chordotonal")] = np.flatnonzero(base & (cc == "chordotonal organ"))
            self.prop[(leg, "hair_plate")] = np.flatnonzero(base & (cc == "hair plate"))
            self.prop[(leg, "campaniform")] = np.flatnonzero(base & (cc == "campaniform sensilla"))
        self.prop_idx = np.concatenate(list(self.prop.values())).astype(np.int64)
        self.act = {k: 0.0 for k in self.groups}
        self.mn_rate = {k: 0.0 for k in self.groups}
        self.offsets = {}   # leg -> {dof name: rad}
        self.sign = None
        self.arena = arena
        self.name = fly.cfg.name
        self.jump_rate = 0.0

    @property
    def available(self):
        return sum(len(v) for v in self.groups.values()) > 0

    # ---------------------------------------------------------------------------------------
    def calibrate_signs(self, arena):
        """A83: find the offset sign that flexes / promotes / depresses each joint (kinematics)."""
        m, d = arena.sim.mj_model, arena.sim.mj_data
        qpos0 = d.qpos.copy()
        thorax = arena.thorax[self.name]
        self.sign = {}
        for leg in LEGS:
            def bid(seg):
                return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{self.name}/{leg}_{seg}")
            coxa, claw = bid("coxa"), bid("tarsus5")
            for j, (dof, *_ , test) in JOINTS.items():
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{self.name}/{dof.format(l=leg)}")
                if jid < 0 or coxa < 0 or claw < 0:
                    continue
                adr = m.jnt_qposadr[jid]
                vals = []
                for dq in (+0.15, -0.15):
                    d.qpos[:] = qpos0
                    d.qpos[adr] += dq
                    mujoco.mj_kinematics(m, d)
                    R = d.xmat[thorax].reshape(3, 3)
                    rel = R.T @ (d.xpos[claw] - d.xpos[coxa])
                    if test == "flex":
                        vals.append(-np.linalg.norm(rel))      # shorter leg = more flexed
                    elif test == "promote":
                        vals.append(rel[0])                    # claw forward
                    else:
                        vals.append(-rel[2])                   # claw down
                self.sign[(leg, j)] = (1.0 if vals[0] > vals[1] else -1.0, f"{self.name}/{dof.format(l=leg)}")
        d.qpos[:] = qpos0
        mujoco.mj_kinematics(m, d)

    # ---------------------------------------------------------------------------------------
    def proprio_rates(self, arena) -> np.ndarray:
        """A84: joint angles and leg load -> proprioceptor firing rates."""
        m, d = arena.sim.mj_model, arena.sim.mj_data
        contacts = arena.leg_contacts(self.name)
        out = []
        for leg in LEGS:
            def ang(dof):
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{self.name}/{dof.format(l=leg)}")
                return float(d.qpos[m.jnt_qposadr[jid]]) if jid >= 0 else 0.0
            ti = ang(JOINTS["tibia_pitch"][0])
            cx = ang(JOINTS["coxa_pitch"][0])
            load = 1.0 if contacts.get(leg) else 0.0
            for kind, val in (("chordotonal", 5 + 60 * abs(ti)), ("hair_plate", 5 + 80 * abs(cx)),
                              ("campaniform", 5 + 120 * load)):
                out.append(np.full(len(self.prop[(leg, kind)]), min(val, 150.0)))
        return np.concatenate(out) if out else np.zeros(0)

    def update(self, counts: np.ndarray, dt_ms: float):
        """MN spikes -> muscle activation -> joint offsets (A82)."""
        for k, ids in self.groups.items():
            r = counts[ids].mean() / (dt_ms * 1e-3) if len(ids) else 0.0
            self.mn_rate[k] = r
            u = min(1.0, r / 100.0)
            tau = 10.0 if u > self.act[k] else 30.0
            self.act[k] += (u - self.act[k]) * (1 - np.exp(-dt_ms / tau))
        self.jump_rate = counts[self.jump_mn].mean() / (dt_ms * 1e-3) if len(self.jump_mn) else 0.0
        if self.sign is None:
            return
        off = {}
        for leg in LEGS:
            for j in JOINTS:
                if (leg, j) not in self.sign:
                    continue
                sgn, dof = self.sign[(leg, j)]
                net = self.act[(leg, j, "ag")] - self.act[(leg, j, "an")]
                off[dof] = sgn * GAIN * net
        self.offsets = off

    def summary(self) -> dict:
        return {f"{l}_{j}": round(self.act[(l, j, 'ag')] - self.act[(l, j, 'an')], 3)
                for l in LEGS for j in JOINTS}
