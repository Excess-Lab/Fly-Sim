"""Body schema of each fly: the state of every body part, its measured function, and injuries.

Injuries are set by the experimenter as a property of the world/body (like a lab lesion); the
nervous system is not told. Their consequences reach the brain only physically and through the
sensory neurons that are lost or silenced:
  leg  'paralyzed' : joints held at rest, no tarsal adhesion
  leg  'amputated' : leg folded away from the substrate, its sensory neurons silent
  wing damage d    : aerodynamic force x (1 - d)
  antenna removed  : ORNs, Johnston's organ and thermosensors of that side silent
  eye covered      : photoreceptors of that side in darkness
ASSUMPTION(A85): lesions approximated as above (no regeneration, no pain from the wound itself).
"""
from dataclasses import dataclass, field

LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
LEG_NAMES = {"lf": "левая передняя", "lm": "левая средняя", "lh": "левая задняя",
             "rf": "правая передняя", "rm": "правая средняя", "rh": "правая задняя"}


@dataclass
class BodySchema:
    legs: dict = field(default_factory=lambda: {leg: "intact" for leg in LEGS})
    wing_damage: dict = field(default_factory=lambda: {"left": 0.0, "right": 0.0})
    antenna: dict = field(default_factory=lambda: {"left": "intact", "right": "intact"})
    eye: dict = field(default_factory=lambda: {"left": "open", "right": "open"})
    # measured function (exponential averages, updated every step)
    leg_contact: dict = field(default_factory=lambda: {leg: 0.0 for leg in LEGS})
    leg_motor: dict = field(default_factory=lambda: {leg: 0.0 for leg in LEGS})

    def injured(self) -> bool:
        return (any(v != "intact" for v in self.legs.values())
                or any(v > 0 for v in self.wing_damage.values())
                or any(v != "intact" for v in self.antenna.values())
                or any(v != "open" for v in self.eye.values()))

    def update_function(self, contacts: dict, chain=None, alpha: float = 0.02):
        for leg in LEGS:
            c = 1.0 if contacts.get(leg) else 0.0
            self.leg_contact[leg] += alpha * (c - self.leg_contact[leg])
            if chain is not None:
                act = sum(abs(chain.act.get((leg, j, t), 0.0)) for j in
                          ("coxa_pitch", "femur_pitch", "tibia_pitch", "tarsus_pitch") for t in ("ag", "an"))
                self.leg_motor[leg] += alpha * (act - self.leg_motor[leg])

    def summary(self) -> dict:
        return {
            "legs": {leg: {"state": self.legs[leg], "contact": round(self.leg_contact[leg], 2),
                           "motor": round(self.leg_motor[leg], 2)} for leg in LEGS},
            "wings": {k: round(v, 2) for k, v in self.wing_damage.items()},
            "antenna": dict(self.antenna), "eye": dict(self.eye), "injured": self.injured(),
        }

    @classmethod
    def from_config(cls, cfg: dict | None):
        s = cls()
        cfg = cfg or {}
        for leg, st in (cfg.get("legs") or {}).items():
            if leg in s.legs and st in ("intact", "paralyzed", "amputated"):
                s.legs[leg] = st
        for side, d in (cfg.get("wing_damage") or {}).items():
            s.wing_damage[side] = float(min(max(d, 0.0), 1.0))
        for side, st in (cfg.get("antenna") or {}).items():
            s.antenna[side] = st
        for side, st in (cfg.get("eye") or {}).items():
            s.eye[side] = st
        return s
