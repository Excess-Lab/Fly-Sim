"""How strongly does per-cell-type calibration (A09d BANC, A09e male) rescale each connectome,
and which read-out neurons sit at the clip limits [0.3, 3]?

Validation tests read out MN9, PPL1, MBONs, PNs ... If those very types were rescaled to match
FlyWire, agreement with FlyWire on them is partly built in (circular). This script makes that
visible. It also counts BANC/Male CNS ids that are not neurons (glia, trachea, unannotated
fragments), which are silenced (A08b) and should not be counted as neurons.

  NUMBA_NUM_THREADS=2 python scripts/calibration_report.py
"""
import numpy as np
import pandas as pd

from flysim.brain.factory import load
from flysim.data.flywire import PROCESSED

READOUTS = {"MN9": r"^MN9$", "PPL1": r"^PPL1", "PAM": r"^PAM", "MBON": r"^MBON",
            "KC": r"^KC", "DN (DNa/DNg/DNp)": r"^DN[agp]", "oviDN": r"^oviDN"}
LO, HI = 0.3, 3.0


def report(ds, gain_file):
    con, ann = load(ds)
    g = np.load(PROCESSED / gain_file)
    assert len(g) == con.n, f"{gain_file}: {len(g)} gains for {con.n} ids (stale cache?)"
    sc = ann.super_class.astype(str)
    print(f"\n=== {ds}: {con.n:,} ids, {len(con.post):,} connections")
    vc = sc.value_counts()
    glia = int(sc.str.contains("glia", case=False).sum())
    empty = int((sc == "").sum())
    other_nonneuron = int(sc.str.contains("trachea|not_a_neuron|fragment", case=False).sum())
    print(f"  super_class '' (unannotated): {empty:,}; glia: {glia:,}; trachea/fragments: {other_nonneuron:,}")
    print(f"  -> ids that are neurons with a class: {con.n - empty - glia - other_nonneuron:,}")
    # A08b (Brain.apply_point_neuron_corrections): non-neurons and ids without any transmitter
    # call neither send nor receive synaptic current
    nt = ann.nt.astype(str).str.lower().to_numpy()
    unknown = ~np.isin(nt, ["gaba", "glutamate", "histamine", "acetylcholine", "dopamine",
                            "serotonin", "octopamine", "tyramine"])
    nonneuron = sc.isin(["glia", "not_a_neuron", "trachea", ""]).to_numpy()
    silent = unknown | nonneuron
    print(f"  silenced by A08b: {int(silent.sum()):,} (non-neuron/unannotated {int(nonneuron.sum()):,}, "
          f"plus {int((unknown & ~nonneuron).sum()):,} neurons without a transmitter call) "
          f"-> effective neurons {con.n - int(silent.sum()):,}")
    print("  super_class counts:", ", ".join(f"{k or '<empty>'}={v}" for k, v in vc.items()))
    t = ann.type.astype(str)
    print(f"  gains: median {np.median(g):.2f}, IQR {np.percentile(g, 25):.2f}-{np.percentile(g, 75):.2f}; "
          f"at upper clip {HI}: {int((g >= HI - 1e-6).sum()):,}; at lower clip {LO}: "
          f"{int((g <= LO + 1e-6).sum()):,}; exactly 1.0 (untouched): {int((g == 1.0).sum()):,}")
    for name, pat in READOUTS.items():
        m = t.str.match(pat).to_numpy()
        if not m.any():
            continue
        gg = g[m]
        print(f"  {name:18s} n={m.sum():5d}  gain median {np.median(gg):.2f}  range "
              f"{gg.min():.2f}-{gg.max():.2f}  at clip {HI}: {int((gg >= HI - 1e-6).sum())}")
    clipped = pd.Series(t[g >= HI - 1e-6]).replace("", "<untyped>").value_counts().head(15)
    print("  most frequent types at the upper clip:", ", ".join(f"{k}({v})" for k, v in clipped.items()))


if __name__ == "__main__":
    report("female_banc", "banc_type_gains.npy")
    report("male", "male_type_gains.npy")
