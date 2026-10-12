"""Unified neuron annotations for both connectomes, aligned to Connectome neuron indices.

Columns: type, super_class, cell_class, cell_sub_class, side, nt, x, y, z (soma or root point, nm)
"""
import numpy as np
import pandas as pd

from flysim.data import flywire, malecns
from flysim.data.flywire import Connectome, RAW

FW_ANN = RAW / "annotations" / "Supplemental_file1_neuron_annotations_v2.1.0.tsv"
FW_ANN_NEW = RAW / "annotations" / "Supplemental_file1_neuron_annotations.tsv"

# FlyWire positions are in 4x4x40 nm voxels
_FW_VOXEL = np.array([4.0, 4.0, 40.0])
# Male CNS positions are in 8 nm isotropic voxels
_MCNS_VOXEL = 8.0


def flywire_annotations(con: Connectome) -> pd.DataFrame:
    cols = ["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type",
            "hemibrain_type", "side", "top_nt", "pos_x", "pos_y", "pos_z",
            "soma_x", "soma_y", "soma_z", "known_nt"]
    a = pd.read_csv(FW_ANN, sep="\t", usecols=cols, low_memory=False)
    # newer labels (v3.2.0 retyping, e.g. gustatory LB* subtypes) where available
    b = pd.read_csv(FW_ANN_NEW, sep="\t",
                    usecols=["root_id", "cell_type", "cell_sub_class", "synonyms"],
                    low_memory=False).rename(columns={"cell_type": "type_new",
                                                      "cell_sub_class": "sub_new"})
    a = a.merge(b, on="root_id", how="left").set_index("root_id").reindex(con.ids)
    t = a.type_new.fillna(a.cell_type).fillna(a.hemibrain_type)
    xyz = np.where(a[["soma_x"]].notna().to_numpy(),
                   a[["soma_x", "soma_y", "soma_z"]].to_numpy(float),
                   a[["pos_x", "pos_y", "pos_z"]].to_numpy(float)) * _FW_VOXEL
    nt = resolve_nt(t.fillna("").to_numpy(), a.top_nt.fillna("").to_numpy(),
                    a.known_nt.fillna("").to_numpy())
    return pd.DataFrame({
        "type": t.fillna("").to_numpy(),
        "super_class": a.super_class.fillna("").to_numpy(),
        "cell_class": a.cell_class.fillna("").to_numpy(),
        "cell_sub_class": a.sub_new.fillna(a.cell_sub_class).fillna("").to_numpy(),
        "side": a.side.fillna("").to_numpy(),
        "nt": nt,
        "synonyms": a.synonyms.fillna("").to_numpy(),
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
    })


_FAST = ["gaba", "glutamate", "acetylcholine", "histamine", "dopamine", "serotonin", "octopamine"]


def resolve_nt(types, top_nt, known_nt):
    """Transmitter per neuron. ASSUMPTION(A03): experimentally known transmitter first
    (FlyWire known_nt; inhibitory one if several), else the majority prediction over the cell
    type (Eckstein et al. 2024 recommend type-level aggregation), else the neuron's own
    prediction."""
    df = pd.DataFrame({"type": types, "top": top_nt})
    typed = df[df.type != ""]
    maj = typed.groupby("type").top.agg(lambda x: x.value_counts().index[0])
    out = np.where(df.type != "", df.type.map(maj).fillna(df.top), df.top).astype(object)
    for i, k in enumerate(known_nt):
        if k:
            names = [x.strip() for x in k.lower().split(",")]
            for cand in _FAST:  # GABA/glutamate before ACh: inhibitory wins for co-transmission
                if cand in names:
                    out[i] = cand
                    break
    return out


def malecns_annotations(con: Connectome) -> pd.DataFrame:
    a = malecns.annotations().set_index("bodyId").reindex(con.ids)
    loc = a.somaLocation.where(a.somaLocation.notna(), a.tosomaLocation)
    xyz = np.array([np.asarray(v, float) if v is not None and len(v) == 3 else [np.nan] * 3
                    for v in loc], dtype=float) * _MCNS_VOXEL
    side = a.somaSide.fillna(a.rootSide).fillna("").map(
        {"L": "left", "R": "right", "M": "center"}).fillna("")
    return pd.DataFrame({
        "type": a.type.fillna("").to_numpy(),
        "super_class": a.superclass.fillna("").to_numpy(),
        "cell_class": a["class"].fillna("").to_numpy(),
        "cell_sub_class": a.subclass.fillna("").to_numpy(),
        "side": side.to_numpy(),
        "nt": a.consensus_nt.where(~a.consensus_nt.isin([None, "unclear"]), a.predicted_nt
                                   ).fillna("").to_numpy(),
        "synonyms": a.synonyms.fillna("").to_numpy(),
        "receptor": a.receptorType.fillna("").to_numpy(),
        "flywire_type": a.flywireType.fillna("").to_numpy(),
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
    })


def _male_nt(male: pd.DataFrame) -> np.ndarray:
    """Male CNS transmitter: FlyWire resolved transmitter of the matched cell type (known_nt or
    type consensus) where a match exists, else male type-level majority of consensus_nt."""
    fcon = flywire.build()
    fann = flywire_annotations(fcon)
    fw_type_nt = fann[fann.type != ""].groupby("type").nt.agg(lambda x: x.value_counts().index[0])
    base = resolve_nt(male.type.to_numpy(), male.nt.replace("unclear", "").to_numpy(),
                      np.array([""] * len(male)))
    ft = male.flywire_type.str.split(",").str[0].str.strip()
    matched = ft.map(fw_type_nt)
    return np.where(matched.notna(), matched, base).astype(object)


def load(sex: str):
    """Return (Connectome, annotations) for 'female' (FlyWire brain) or 'male' (Male CNS)."""
    if sex == "female":
        con = flywire.build()
        return con, flywire_annotations(con)
    if sex == "male":
        con = malecns.build()
        ann = malecns_annotations(con)
        cache = flywire.PROCESSED / "malecns_nt_resolved.npy"
        if cache.exists():
            ann["nt"] = np.load(cache, allow_pickle=True)
        else:
            ann["nt"] = _male_nt(ann)
            np.save(cache, ann.nt.to_numpy().astype(object), allow_pickle=True)
        return con, ann
    raise ValueError(sex)


def select(ann: pd.DataFrame, pattern: str, col: str = "type") -> np.ndarray:
    return np.flatnonzero(ann[col].str.match(pattern).to_numpy())
