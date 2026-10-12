"""Ethical research layer.

We do not assume the simulated fly is sentient; we also do not ignore the uncertainty. Every
potentially aversive manipulation is declared, justified, capped and logged, and stopped when a
neural correlate of punishment stays high for too long. The same principles are recommended for
work with real Drosophila (3Rs: replace, reduce, refine).

Policy (defaults, editable per experiment):
  - aversive stimuli must be declared with a written justification
  - hot floor <= 45 C, shock drive <= 80 Hz on PPL1, single exposure <= 60 s
  - total aversive exposure per fly per session <= 120 s
  - stop rule: PPL1 population > 60 Hz continuously for 20 s -> aversive stimuli switched off
  - experiments with aversive stimuli are marked in manifests, logs and the UI
"""
from dataclasses import dataclass, field


@dataclass
class EthicsPolicy:
    max_floor_temp_c: float = 45.0
    max_shock_hz: float = 80.0
    max_single_exposure_s: float = 60.0
    max_total_exposure_s: float = 120.0
    stop_ppl1_hz: float = 60.0
    stop_after_s: float = 20.0


@dataclass
class EthicsMonitor:
    policy: EthicsPolicy = field(default_factory=EthicsPolicy)
    justification: str = ""
    exposure_s: dict = field(default_factory=dict)      # fly -> seconds of aversive stimulation
    high_ppl1_s: dict = field(default_factory=dict)
    stopped: bool = False
    events: list = field(default_factory=list)

    def declare(self, world, justification: str):
        self.justification = justification
        for z in world.zones:
            if z.temp_c > self.policy.max_floor_temp_c:
                self.events.append(f"zone capped {z.temp_c} -> {self.policy.max_floor_temp_c} C")
                z.temp_c = self.policy.max_floor_temp_c

    def aversive_active(self, world) -> bool:
        return any(z.active and z.temp_c > 30 for z in world.zones) or any(
            v > 0 for v in world.shocks.values())

    def step(self, world, dt: float):
        if self.stopped:
            return
        for name in list(world.shocks):
            if world.shocks[name] > self.policy.max_shock_hz:
                world.shocks[name] = self.policy.max_shock_hz
        for f in world.flies:
            n = f.cfg.name
            hot = getattr(f, "last_temp", 25.0) > 30.0 or world.shocks.get(n, 0) > 0
            if hot:
                self.exposure_s[n] = self.exposure_s.get(n, 0.0) + dt
            ppl1 = f.motor.rate.get("ppl1", 0.0)
            self.high_ppl1_s[n] = self.high_ppl1_s.get(n, 0.0) + dt if ppl1 > self.policy.stop_ppl1_hz else 0.0
            if (self.high_ppl1_s[n] > self.policy.stop_after_s
                    or self.exposure_s.get(n, 0.0) > self.policy.max_total_exposure_s):
                self.stop(world, f"stop rule for {n}: exposure {self.exposure_s.get(n, 0):.0f} s, "
                                 f"PPL1>{self.policy.stop_ppl1_hz:.0f} Hz for {self.high_ppl1_s[n]:.0f} s")
                return

    def stop(self, world, reason: str):
        for z in world.zones:
            z.active = False
        for k in world.shocks:
            world.shocks[k] = 0.0
        self.stopped = True
        self.events.append(reason)

    def report(self) -> dict:
        return {"justification": self.justification, "policy": vars(self.policy),
                "exposure_s": {k: round(v, 2) for k, v in self.exposure_s.items()},
                "stopped": self.stopped, "events": self.events}
