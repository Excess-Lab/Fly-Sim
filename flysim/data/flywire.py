"""FlyWire v783 female brain connectome -> compact CSR graph.

Source: Dorkenwald et al. 2024 / Schlegel et al. 2024 (Nature), via the public files of
Shiu et al. 2024 (github.com/philshiu/Drosophila_brain_model), MIT license.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


@dataclass
class Connectome:
    ids: np.ndarray       # int64 FlyWire root id per neuron index
    indptr: np.ndarray    # int64, CSR over presynaptic neuron
    post: np.ndarray      # int32 postsynaptic index
    syn: np.ndarray       # int32 signed synapse count (+exc / -inh)

    @property
    def n(self) -> int:
        return len(self.ids)

    def index_of(self, flywire_ids) -> np.ndarray:
        lut = {int(f): i for i, f in enumerate(self.ids)}
        missing = [f for f in flywire_ids if int(f) not in lut]
        if missing:
            raise KeyError(f"{len(missing)} ids not in connectome: {missing[:5]}")
        return np.array([lut[int(f)] for f in flywire_ids], dtype=np.int64)

    def filter_ids(self, flywire_ids) -> list:
        s = set(int(i) for i in self.ids)
        return [f for f in flywire_ids if int(f) in s]


def build(version: int = 783) -> Connectome:
    out = PROCESSED / f"flywire_{version}.npz"
    if out.exists():
        z = np.load(out)
        return Connectome(z["ids"], z["indptr"], z["post"], z["syn"])

    comp = pd.read_csv(RAW / f"Completeness_{version}.csv", index_col=0)
    con = pd.read_parquet(
        RAW / f"Connectivity_{version}.parquet",
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    n = len(comp)
    pre = con["Presynaptic_Index"].to_numpy(np.int64)
    order = np.argsort(pre, kind="stable")
    pre = pre[order]
    post = con["Postsynaptic_Index"].to_numpy(np.int32)[order]
    syn = con["Excitatory x Connectivity"].to_numpy(np.int32)[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, pre + 1, 1)
    indptr = np.cumsum(indptr)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    ids = comp.index.to_numpy(np.int64)
    np.savez(out, ids=ids, indptr=indptr, post=post, syn=syn)
    return Connectome(ids, indptr, post, syn)
