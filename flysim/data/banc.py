"""BANC: female brain AND nerve cord connectome (Bates et al. 2025), compiled release 888.

Source: gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/banc_888/ (public).
Every BANC neuron carries cross-dataset matches (fafb_match -> FlyWire root id, malecns_match ->
Male CNS body id), which lets us relate the female full CNS neuron-by-neuron to the validated
FlyWire brain and to the male CNS.
"""
import numpy as np
import pandas as pd
import pyarrow.feather as feather

from flysim.data.flywire import PROCESSED, ROOT, Connectome

RAW = ROOT / "data" / "raw" / "banc"
META = RAW / "banc_888_meta.feather"
EDGES = RAW / "banc_888_edgelist_simple_v3.feather"
NON_NEURON = {"glia", "not_a_neuron", "trachea"}

META_COLS = ["banc_888_id", "super_class", "cell_class", "cell_sub_class", "cell_type",
             "fafb_cell_type", "malecns_cell_type", "region", "side", "nerve", "neuromere",
             "neurotransmitter_predicted", "neurotransmitter_verified", "fafb_match",
             "malecns_match", "root_position_nm"]


def meta() -> pd.DataFrame:
    m = feather.read_table(META, columns=META_COLS).to_pandas()
    m["banc_888_id"] = m.banc_888_id.astype(np.int64)
    return m


def build() -> Connectome:
    out = PROCESSED / "banc_888.npz"
    if out.exists():
        z = np.load(out)
        return Connectome(z["ids"], z["indptr"], z["post"], z["syn"])
    m = meta()
    m = m[~m.super_class.isin(NON_NEURON)]
    e = feather.read_table(EDGES, columns=["pre", "post", "count"]).to_pandas()
    pre_id = e.pre.astype(np.int64).to_numpy()
    post_id = e.post.astype(np.int64).to_numpy()
    ids = np.unique(np.concatenate([m.banc_888_id.to_numpy(), pre_id, post_id]))
    pre = np.searchsorted(ids, pre_id).astype(np.int64)
    post = np.searchsorted(ids, post_id).astype(np.int32)
    w = e["count"].to_numpy().astype(np.int32)   # unsigned; signs applied in Brain from annotations
    order = np.argsort(pre, kind="stable")
    pre, post, w = pre[order], post[order], w[order]
    indptr = np.zeros(len(ids) + 1, dtype=np.int64)
    np.add.at(indptr, pre + 1, 1)
    indptr = np.cumsum(indptr)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=ids, indptr=indptr, post=post, syn=w)
    return Connectome(ids, indptr, post, w)


_SUPER = {"ventral_nerve_cord_intrinsic": "vnc_intrinsic", "central_brain_intrinsic": "cb_intrinsic",
          "optic_lobe_intrinsic": "ol_intrinsic"}
_LEG_NERVE = {"prothoracic_leg_nerve": ("fl", "ProLN"), "mesothoracic_leg_nerve": ("ml", "MesoLN"),
              "metathoracic_leg_nerve": ("hl", "MetaLN")}
_PROP = {"chordotonal_organ_neuron": "chordotonal organ", "hair_plate_neuron": "hair plate",
         "campaniform_sensillum_neuron": "campaniform sensilla"}


def annotations(con: Connectome, flywire_nt=None, flywire_type_nt=None, male_type_nt=None) -> pd.DataFrame:
    """Unified annotation table (same columns/conventions as FlyWire and Male CNS tables)."""
    m = meta().drop_duplicates("banc_888_id").set_index("banc_888_id").reindex(con.ids)
    region = m.region.fillna("").to_numpy()
    sc = m.super_class.fillna("").astype(str).map(lambda x: _SUPER.get(x, x)).to_numpy().astype(object)
    in_vnc = region == "ventral_nerve_cord"
    for raw, vnc_name in (("motor", "vnc_motor"), ("sensory", "vnc_sensory")):
        sc[(sc == raw) & in_vnc] = vnc_name
    ctype = m.cell_type.fillna(m.fafb_cell_type).fillna("").astype(str)
    is_mn = sc == "vnc_motor"
    ctype = np.where(is_mn & m.malecns_cell_type.notna().to_numpy(),
                     m.malecns_cell_type.fillna("").to_numpy(), ctype.to_numpy())
    cc = m.cell_class.fillna("").astype(str).to_numpy().astype(object)
    csub = m.cell_sub_class.fillna("").astype(str).to_numpy().astype(object)
    for k, v in _PROP.items():   # proprioceptor naming as in Male CNS
        sel = cc == k
        cc[sel] = "mechanosensory_proprioceptive"
        csub[sel] = v
    nerve = m.nerve.fillna("").astype(str).to_numpy()
    leg = np.array([""] * len(m), dtype=object)
    entry = np.array([""] * len(m), dtype=object)
    for k, (lg, en) in _LEG_NERVE.items():
        sel = np.char.endswith(nerve.astype(str), k)
        leg[sel] = lg
        entry[sel] = en
    # transmitter: verified > FlyWire match (resolved) > BANC prediction
    nt = m.neurotransmitter_verified.fillna("").astype(str).str.split(",").str[0].str.strip().str.lower()
    pred = m.neurotransmitter_predicted.fillna("").astype(str).str.lower()
    # ASSUMPTION(A03c): verified > FlyWire neuron match > FlyWire cell-type consensus (brain) >
    # Male CNS cell-type consensus (VNC) > BANC prediction. Raw per-neuron predictions alone
    # re-created the runaway state (as for FlyWire and Male CNS).
    # A03e: for comparability with the validated FlyWire model, brain neurons whose cell type
    # exists in FlyWire take the FlyWire type consensus first (same convention); verified
    # BANC transmitters are used for everything else (VNC, unmatched types)
    region = m.region.fillna("")
    out = pd.Series([""] * len(m), index=m.index, dtype=object)
    if flywire_type_nt is not None:
        ft0 = m.fafb_cell_type.fillna(m.cell_type).map(flywire_type_nt)
        brain = region.isin(["central_brain", "optic_lobe"])
        out = out.where(~(brain & ft0.notna()), ft0)
    empty = out.isna() | (out == "")
    out = out.where(~empty, nt)
    empty = out.isna() | (out == "")
    if flywire_nt is not None:
        fm = pd.to_numeric(m.fafb_match, errors="coerce").map(flywire_nt)
        out = out.where(~(empty & fm.notna()), fm)
        empty = out == ""
    if flywire_type_nt is not None:
        ft = m.fafb_cell_type.fillna(m.cell_type).map(flywire_type_nt)
        out = out.where(~(empty & ft.notna()), ft)
        empty = out.isna() | (out == "")
    if male_type_nt is not None:
        mt = m.malecns_cell_type.map(male_type_nt)
        out = out.where(~(empty & mt.notna()), mt)
        empty = out.isna() | (out == "")
    nt = out.where(~empty, pred).fillna("").to_numpy()
    pos = m.root_position_nm.fillna("nan, nan, nan").astype(str).str.split(",", expand=True)
    xyz = pos.apply(pd.to_numeric, errors="coerce").to_numpy(float)
    side = m.side.fillna("").astype(str).str.lower().replace({"l": "left", "r": "right"}).to_numpy()
    return pd.DataFrame({
        "type": ctype, "super_class": sc, "cell_class": cc, "cell_sub_class": csub,
        "side": side, "nt": nt, "synonyms": "", "receptor": "",
        "flywire_type": m.fafb_cell_type.fillna("").to_numpy(),
        "fafb_match": pd.to_numeric(m.fafb_match, errors="coerce").to_numpy(),
        "malecns_match": pd.to_numeric(m.malecns_match, errors="coerce").to_numpy(),
        "leg": leg, "entry_nerve": entry,
        "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
    })
