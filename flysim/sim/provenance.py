"""Provenance of every result: which mechanisms produced it, what the nervous system contains,
and an experiment ID from which the run can be reproduced.

Mechanism modes (shown on every result, log and report):
  connectome_only               decisions come only from reconstructed pathways; no shortcuts and
                                no spontaneous locomotor drive
  connectome_plus_approximations  as above, plus documented approximations that do not inject
                                valence (spontaneous drive A30)
  experimental_shortcuts        at least one shortcut that injects activity into neurons by
                                measured physiology instead of a reconstructed pathway (A24, A25)
Always-present approximations (every mode) are listed explicitly: neuron model, sensory
transduction, body controller, flight reflexes, etc.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

ALWAYS_APPROX = [
    ("A01-A09", "point LIF neurons; transmitter signs; point-neuron corrections; male weight calibration"),
    ("A50-A57", "sensory transduction models (odour, taste, temperature, vision, hearing, contact)"),
    ("A10-A15", "metabolism and hunger-dependent sensory gain"),
    ("A31", "descending-neuron -> motor command gains"),
    ("A59", "intrinsic left/right steering adaptation"),
    ("A36-A37", "flight: point-mass aerodynamics with reflex stabilisation layer (A36b)"),
    ("NMF", "walking executed by the NeuroMechFly CPG controller driven by descending drive"),
]


def model_version() -> str:
    try:
        v = subprocess.run(["git", "-C", str(ROOT), "describe", "--tags", "--always", "--dirty"],
                           capture_output=True, text=True, timeout=5).stdout.strip()
        return v or "unknown"
    except Exception:
        return "unknown"


def mechanism_mode(cfg) -> str:
    if cfg.us_punish_shortcut or cfg.us_reward_shortcut:
        return "experimental_shortcuts"
    if cfg.spontaneous:
        return "connectome_plus_approximations"
    return "connectome_only"


def connectome_coverage(fly) -> dict:
    """What this fly's nervous system contains and lacks."""
    ann = fly.ann
    sc = ann.super_class
    has_vnc = bool(sc.str.contains("vnc").any())
    dataset = {"female": "FlyWire v783 (brain only)", "male": "Janelia Male CNS v1.0 (brain + VNC)"}
    dataset = getattr(fly, "dataset_name", None) or dataset[fly.cfg.sex]
    groups = {
        "olfactory_receptor_neurons": len(fly.olf.idx),
        "gustatory_receptor_neurons_labellum": len(fly.gust.idx),
        "photoreceptors": len(fly.vision.idx),
        "thermosensory_antenna": len(fly.thermo.idx),
        "johnstons_organ_JO-B": len(fly.hearing.idx),
        "leg_nociceptor_candidates": int(len(fly.noci_idx)),
        "kenyon_cells": len(fly.kc), "MBONs": len(fly.mbon),
        "PPL1_DANs": len(fly.ppl1), "PAM_DANs": len(fly.pam),
        "descending_neurons": int(sc.str.contains("descending").sum()),
        "leg_motor_neurons": int((sc == "vnc_motor").sum()),
    }
    missing = []
    if not has_vnc:
        missing += ["ventral nerve cord (leg/wing sensory and motor neurons, local VNC circuits)",
                    "leg nociception and leg taste pathways", "VNC locomotor circuits"]
    missing += ["neuromodulator volume transmission (dopamine/serotonin/octopamine act only as "
                "plasticity teaching signal or not at all)", "electrical synapses (gap junctions)",
                "glia", "peripheral and enteric nervous system"]
    return {"dataset": dataset, "neurons": int(fly.brain.con.n),
            "connections": int(len(fly.brain.con.post)), "has_vnc": has_vnc,
            "groups": groups, "missing": missing}


def fly_provenance(fly) -> dict:
    c = fly.cfg
    shortcuts = []
    if c.us_punish_shortcut:
        shortcuts.append("A24: noxious stimulus -> PPL1 punishment DANs (measured physiology)")
    if c.us_reward_shortcut:
        shortcuts.append("A25: sugar ingestion -> PAM reward DANs (measured physiology)")
    approx = list(ALWAYS_APPROX)
    if c.spontaneous:
        approx.append(("A30", "spontaneous locomotor drive"))
    return {"fly": c.name, "sex": c.sex, "seed": c.seed, "mode": mechanism_mode(c),
            "plasticity": c.plasticity, "shortcuts": shortcuts,
            "approximations": [f"{a}: {b}" for a, b in approx],
            "connectome_coverage": connectome_coverage(fly), "body": c.body}


def experiment_id(spec: dict) -> str:
    blob = json.dumps(spec, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def manifest(world, protocol: dict) -> dict:
    """protocol: name, hypothesis, parameters, seed ... (everything needed to reproduce)."""
    m = {"model_version": model_version(), "protocol": protocol,
         "environment": {"o2_pct": world.o2_pct, "co2_ppm": world.co2_ppm,
                         "temp_c": world.ambient_c, "apple": world.apple.summary(),
                         "apple_cut": getattr(world.apple, "cut", True),
                         "zones": [vars(z) for z in world.zones]},
         "flies": [fly_provenance(f) for f in world.flies]}
    modes = {f["mode"] for f in m["flies"]}
    m["mode"] = ("experimental_shortcuts" if "experimental_shortcuts" in modes else
                 "connectome_plus_approximations" if "connectome_plus_approximations" in modes
                 else "connectome_only")
    m["experiment_id"] = experiment_id({"v": m["model_version"], "p": protocol,
                                        "flies": [(f["fly"], f["sex"], f["seed"], f["mode"],
                                                   f["plasticity"]) for f in m["flies"]]})
    return m


def coverage_markdown(cov: dict) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in cov["groups"].items())
    miss = "\n".join(f"- {x}" for x in cov["missing"])
    return (f"**{cov['dataset']}** — {cov['neurons']:,} neurons, {cov['connections']:,} connections\n\n"
            f"| group | neurons |\n|---|---|\n{rows}\n\nMissing:\n{miss}\n")


def jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return str(x)
