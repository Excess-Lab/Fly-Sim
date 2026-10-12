"""Janelia Male CNS v1.0 connectome (brain + optic lobes + VNC) -> compact CSR graph.

Source: gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/ (CC-BY), no auth needed.
Only traced neurons, synapse confidence >= 0.5.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from flysim.data.flywire import PROCESSED, ROOT, Connectome

RAW = ROOT / "data" / "raw" / "malecns"
EDGES = RAW / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"
ANN = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT = RAW / "body-neurotransmitters-male-cns-v1.0.feather"

# ASSUMPTION(A03): sign from predicted transmitter, same convention as for FlyWire:
# acetylcholine excitatory; GABA, glutamate, histamine inhibitory; others excitatory.
INHIBITORY_NT = {"gaba", "glutamate", "histamine"}


def annotations() -> pd.DataFrame:
    ann = feather.read_table(ANN).to_pandas()
    nt = feather.read_table(NT, columns=["body", "consensus_nt", "predicted_nt"]).to_pandas()
    return ann.merge(nt.rename(columns={"body": "bodyId"}), on="bodyId", how="left")


def build() -> Connectome:
    out = PROCESSED / "malecns_v1.npz"
    if out.exists():
        z = np.load(out)
        return Connectome(z["ids"], z["indptr"], z["post"], z["syn"])

    e = feather.read_table(EDGES, columns=["body_pre", "body_post", "weight"]).to_pandas()
    ids = np.unique(np.concatenate([e.body_pre.to_numpy(), e.body_post.to_numpy()]))
    pre = np.searchsorted(ids, e.body_pre.to_numpy()).astype(np.int64)
    post = np.searchsorted(ids, e.body_post.to_numpy()).astype(np.int32)
    w = e.weight.to_numpy().astype(np.int32)
    del e

    nt = feather.read_table(NT, columns=["body", "consensus_nt", "predicted_nt"]).to_pandas()
    nt = nt.set_index("body").reindex(ids)
    label = nt.consensus_nt.where(~nt.consensus_nt.isin([None, "unclear"]), nt.predicted_nt)
    label = label.fillna("acetylcholine").str.lower()
    sign = np.where(label.isin(INHIBITORY_NT).to_numpy(), -1, 1).astype(np.int32)
    w = w * sign[pre]

    order = np.argsort(pre, kind="stable")
    pre, post, w = pre[order], post[order], w[order]
    indptr = np.zeros(len(ids) + 1, dtype=np.int64)
    np.add.at(indptr, pre + 1, 1)
    indptr = np.cumsum(indptr)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=ids.astype(np.int64), indptr=indptr, post=post, syn=w)
    return Connectome(ids.astype(np.int64), indptr, post, w)
