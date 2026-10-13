"""Connectivity of walking descending neurons (no simulation).

1. Inputs: who drives DNp09 (P9), DNg100 (BDN2), DNg97 (oDN1), DNa02, DNa01, MDN ...
   - direct input composition (super_class / cell_class of presynaptic partners)
   - linear multi-hop 'effective input' from sensory modalities (ORN subsets, photoreceptors,
     leg bristles, proprioceptors, IPCs): signed, input-normalised propagation, 1..4 hops
2. Outputs: linear multi-hop signed influence of every DN onto leg motor neurons (VNC).

Linear propagation: A[post, pre] = signed synapses (simulator signs) / total |input synapses| of post;
influence after k hops = A^k. This is a connectivity statistic (no dynamics, no thresholds) -
used only to rank pathways, never as a model parameter.

Usage: python scripts/walk_dn_connectivity.py female_banc|male
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from flysim.brain.factory import load

DS = sys.argv[1] if len(sys.argv) > 1 else "female_banc"
OUT = Path(__file__).resolve().parents[1] / "results" / "walking"
OUT.mkdir(parents=True, exist_ok=True)
con, ann = load(DS)
n = con.n
t = ann.type.astype(str).to_numpy()
side = ann.side.astype(str).to_numpy()
sc = ann.super_class.astype(str).to_numpy()
cc = ann.cell_class.astype(str).to_numpy()
leg = ann["leg"].fillna("").astype(str).to_numpy()
en = ann["entry_nerve"].fillna("").astype(str).to_numpy()
pre = np.repeat(np.arange(n), np.diff(con.indptr)).astype(np.int64)
post = con.post.astype(np.int64)
# signed weights exactly as the simulator uses them (transmitter sign A03, dropped synapses A06-A08)
from types import SimpleNamespace
from flysim.brain.lif import Brain
_ns = SimpleNamespace(con=con, p=SimpleNamespace(w_syn=1.0))
Brain.apply_point_neuron_corrections(_ns, ann)
syn = _ns.w.astype(np.float64)
del _ns
tot_in = np.bincount(post, weights=np.abs(syn), minlength=n)
A = sp.csr_matrix((syn / np.maximum(tot_in[post], 1.0), (post, pre)), shape=(n, n))

DN = np.isin(sc, ["descending", "descending_neuron"])
LEGMN = (sc == "vnc_motor") & (leg != "")
CAND = ["DNp09", "DNg100", "DNg97", "DNa02", "DNa01", "DNa03", "DNb05", "DNg13", "DNb02",
        "MDN", "DNp42"]
tser = pd.Series(t)
ORN = tser.str.match(r"^ORN_").to_numpy()
# attractive food-odour glomeruli (vinegar / ethanol / fruit esters; Semmelhack & Wang 2009,
# Becher et al. 2010): DM1, DM2, DM4, DP1m, VM7d, VA2, DM5 (repulsive at high conc.)
ATTR = tser.str.match(r"^ORN_(DM1|DM2|DM4|DP1m|VM7d|VA2)$").to_numpy()
PR = tser.str.match(r"^(R1-6|R1-R6|R7|R8)").to_numpy()
BRISTLE = np.isin(cc, ["bristle_neuron", "mechanosensory_tactile"])
LEGBRISTLE = BRISTLE & np.isin(en, ["ProLN", "MesoLN", "MetaLN"])
PROP = cc == "mechanosensory_proprioceptive"
IPC = tser.str.match(r"^(IPC|m_NSC_DILP)$").to_numpy()
SOURCES = {"ORN_all": ORN, "ORN_food(DM1,DM2,DM4,DP1m,VM7d,VA2)": ATTR, "photoreceptors": PR,
           "leg_bristles": LEGBRISTLE, "all_bristles": BRISTLE, "proprioceptors": PROP, "IPC": IPC}

res = {"dataset": DS}
# ---- 1. direct input composition ----
direct = {}
for ty in CAND:
    ids = np.flatnonzero(t == ty)
    if not len(ids):
        continue
    m = np.isin(post, ids)
    d = pd.DataFrame({"pre_t": t[pre[m]], "sc": sc[pre[m]], "syn": syn[m]})
    by_sc = d.groupby("sc").syn.agg(lambda x: np.abs(x).sum())
    by_sc = (by_sc / by_sc.sum()).sort_values(ascending=False).round(3).head(6).to_dict()
    top = d.groupby("pre_t").syn.sum().sort_values(key=np.abs, ascending=False).head(10)
    direct[ty] = {"n": int(len(ids)), "in_syn": int(np.abs(syn[m]).sum()),
                  "by_superclass": by_sc, "top_pre_types": {k: int(v) for k, v in top.items()}}
    print(ty, direct[ty])
res["direct_inputs"] = direct

# ---- 1b. multi-hop sensory influence on DNs (k = 1..4, summed) ----
def propagate(src_mask, hops=4):
    x = src_mask.astype(np.float64)
    x = x / max(x.sum(), 1)
    acc = np.zeros(n)
    for _ in range(hops):
        x = A @ x
        acc += x
    return acc

sens = {}
for name, m in SOURCES.items():
    if not m.any():
        continue
    infl = propagate(m)
    sens[name] = {ty: float(infl[t == ty].mean()) for ty in CAND if (t == ty).any()}
    # rank of walking DNs among all DNs
    dn_idx = np.flatnonzero(DN)
    order = dn_idx[np.argsort(-infl[dn_idx])]
    sens[name]["_top_DNs"] = [(t[i], side[i], float(infl[i])) for i in order[:8]]
    sens[name]["_DN_median"] = float(np.median(infl[dn_idx]))
    print("source", name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sens[name].items()
                           if not k.startswith("_")})
    print("   top DNs:", [(a, b, round(c, 5)) for a, b, c in sens[name]["_top_DNs"]])
res["sensory_influence"] = sens

# ---- 2. DN -> leg MN influence (signed, 1..4 hops through the VNC) ----
legmn = np.flatnonzero(LEGMN)
if len(legmn):
    # restrict to the VNC subgraph (DNs, VNC neurons, motor neurons): DN -> leg MN paths
    vnc = np.flatnonzero(DN | np.char.startswith(sc.astype(str), "vnc") | (sc == "ascending")
                         | (sc == "ascending_neuron"))
    pos_in = np.full(n, -1)
    pos_in[vnc] = np.arange(len(vnc))
    As = A[vnc][:, vnc].tocsc()
    dn_idx = np.flatnonzero(DN)
    # backward propagation from each leg MN: z_k = (A^T)^k e_mn gives every neuron's k-hop
    # influence on that MN (dense: |vnc| x |legMN|)
    Z = np.zeros((len(vnc), len(legmn)))
    Z[pos_in[legmn], np.arange(len(legmn))] = 1.0
    acc = np.zeros((len(dn_idx), len(legmn)))
    AsT = As.T.tocsr()
    for _ in range(4):
        Z = AsT @ Z
        acc += Z[pos_in[dn_idx]]
    acc = acc.T            # legMN x DN
    pos_infl = np.clip(acc, 0, None).sum(0)
    neg_infl = np.clip(acc, None, 0).sum(0)
    absinf = np.abs(acc).sum(0)
    df = pd.DataFrame({"type": t[dn_idx], "side": side[dn_idx], "legMN_abs": absinf,
                       "legMN_pos": pos_infl, "legMN_neg": neg_infl})
    df["rank"] = df.legMN_abs.rank(ascending=False).astype(int)
    print("DN -> leg MN influence: median abs", float(df.legMN_abs.median()))
    print(df[df.type.isin(CAND)].sort_values("type").to_string())
    print("top 20 DNs by leg-MN influence:")
    print(df.sort_values("legMN_abs", ascending=False).head(20).to_string())
    df.to_csv(OUT / f"dn_legmn_influence_{DS}.csv", index=False)
    res["legmn_influence_candidates"] = df[df.type.isin(CAND)].round(6).to_dict(orient="records")
    res["legmn_influence_median"] = float(df.legMN_abs.median())
json.dump(res, open(OUT / f"walk_dn_connectivity_{DS}.json", "w"), indent=1, default=str)
