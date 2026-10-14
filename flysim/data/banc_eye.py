"""Complete the BANC left eye by bilateral symmetry.

In BANC 888 ~1600 photoreceptors (R7/R8) are connected to the right optic lobe but only ~245 to
the left, although the left optic lobe itself (Mi1, Tm3, L1-L3, ...) is reconstructed. The fly's
visual system is mirror-symmetric, so each missing left photoreceptor is created as the mirror
image of a right one: for every output synapse of the right photoreceptor onto a cell of type T at
position p, the mirror gets the same synapse count onto the left-lobe cell of type T nearest to
the mirrored position p'. ASSUMPTION(A54d). Synthetic neurons carry `synthetic = True` and ids < 0.
"""
import numpy as np
import pandas as pd

from flysim.data.flywire import Connectome


def _lobe_x(con, ann, idx):
    """x position of each neuron's strongest positioned partner (photoreceptors have no position)."""
    x = ann.x.to_numpy()
    out = np.full(len(idx), np.nan)
    for k, i in enumerate(idx):
        lo, hi = con.indptr[i], con.indptr[i + 1]
        tg, w = con.post[lo:hi], np.abs(con.syn[lo:hi])
        ok = ~np.isnan(x[tg])
        if ok.any():
            out[k] = x[tg[ok][np.argmax(w[ok])]]
    return out


def complete_left_eye(con: Connectome, ann: pd.DataFrame, seed: int = 0):
    from scipy.spatial import cKDTree
    t = ann.type.astype(str).to_numpy()
    pr = np.flatnonzero(pd.Series(t).str.match(r"^(R1-6|R7|R8)").to_numpy())
    xyz = ann[["x", "y", "z"]].to_numpy(float)
    side = ann.side.astype(str).to_numpy()
    # midline and which x is "left": the annotated majority (left = larger x in BANC)
    ok = ~np.isnan(xyz[:, 0])
    xl, xr = np.nanmedian(xyz[ok & (side == "left"), 0]), np.nanmedian(xyz[ok & (side == "right"), 0])
    mid = 0.5 * (xl + xr)
    left_is_high = xl > xr
    lx = _lobe_x(con, ann, pr)
    in_left = (lx > mid) if left_is_high else (lx < mid)
    right_pr, left_pr = pr[~in_left & ~np.isnan(lx)], pr[in_left]
    n_new = len(right_pr) - len(left_pr)
    if n_new <= 0:
        return con, ann, 0
    rng = np.random.default_rng(seed)
    src = rng.choice(right_pr, n_new, replace=False)
    # left-lobe cells per type, for nearest-neighbour lookup of mirrored targets
    left_lobe = ok & ((xyz[:, 0] > mid) if left_is_high else (xyz[:, 0] < mid))
    trees = {}
    new_post, new_syn, new_ptr = [], [], [0]
    for i in src:
        lo, hi = con.indptr[i], con.indptr[i + 1]
        posts, syns = [], []
        for j, s in zip(con.post[lo:hi], con.syn[lo:hi]):
            ty = t[j]
            if not ok[j] or ty == "":
                continue
            if ty not in trees:
                cand = np.flatnonzero(left_lobe & (t == ty))
                trees[ty] = (cKDTree(xyz[cand]), cand) if len(cand) else None
            tr = trees[ty]
            if tr is None:
                continue
            p = xyz[j].copy()
            p[0] = 2 * mid - p[0]
            d, k = tr[0].query(p)
            posts.append(tr[1][k])
            syns.append(s)
        new_post += posts
        new_syn += syns
        new_ptr.append(new_ptr[-1] + len(posts))
    n0 = con.n
    indptr = np.concatenate([con.indptr, con.indptr[-1] + np.asarray(new_ptr[1:], np.int64)])
    post = np.concatenate([con.post, np.asarray(new_post, con.post.dtype)])
    syn = np.concatenate([con.syn, np.asarray(new_syn, con.syn.dtype)])
    ids = np.concatenate([con.ids, -np.arange(1, n_new + 1, dtype=np.int64)])
    con2 = Connectome(ids=ids, indptr=indptr, post=post, syn=syn)
    add = ann.iloc[src].copy()
    add["side"] = "left"
    add[["x", "y", "z"]] = np.nan
    if "synthetic" not in ann.columns:
        ann = ann.assign(synthetic=False)
    add["synthetic"] = True
    ann2 = pd.concat([ann, add], ignore_index=True)
    assert len(ann2) == con2.n == n0 + n_new
    return con2, ann2, n_new
