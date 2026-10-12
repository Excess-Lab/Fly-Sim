"""Pack a recorded run into a single self-contained HTML viewer."""
import base64
import json
from pathlib import Path

import numpy as np

from flysim.brain.factory import load

TEMPLATE = Path(__file__).with_name("template.html")

REGION_NAMES = {
    0: "прочие центральные", 1: "сенсорные", 2: "зрительные доли", 3: "грибовидные тела (память)",
    4: "обонятельный центр (AL/LH)", 5: "командные и моторные", 6: "центральный комплекс",
    7: "брюшная цепочка (VNC)", 8: "дофамин (обучение)",
}
REGION_COLORS = {0: "#9ca3af", 1: "#facc15", 2: "#38bdf8", 3: "#a855f7", 4: "#22c55e",
                 5: "#f97316", 6: "#ef4444", 7: "#14b8a6", 8: "#ec4899"}


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _regions(ann) -> np.ndarray:
    t, sc, cc = ann.type, ann.super_class, ann.cell_class
    r = np.zeros(len(ann), np.uint8)
    r[sc.str.contains("optic|ol_|visual").to_numpy()] = 2
    r[cc.isin(["ALPN", "ALLN", "LHLN", "LHCENT", "ALIN", "ALON"]).to_numpy()] = 4
    r[t.str.match(r"^(ORN|ALPN|.*_[a-z]*PN$|lLN|v2LN)").to_numpy()] = 4
    r[sc.str.contains("vnc").to_numpy()] = 7
    r[t.str.match(r"^(KC|MBON|APL)").to_numpy()] = 3
    r[cc.str.contains("CX").to_numpy() | t.str.match(r"^(EPG|PEN|PFL|ER\d|hDelta|vDelta|FB)").to_numpy()] = 6
    r[sc.str.contains("descending|motor|efferent").to_numpy()] = 5
    r[sc.str.contains("sensory").to_numpy()] = 1
    r[t.str.match(r"^(PAM|PPL|PPM)").to_numpy()] = 8
    return r


def _positions(ann) -> np.ndarray:
    xyz = ann[["x", "y", "z"]].to_numpy(float)
    ok = ~np.isnan(xyz[:, 0])
    c = np.median(xyz[ok], 0)
    s = np.nanmax(np.abs(xyz[ok] - c))
    out = (xyz - c) / s
    return out.astype(np.float32)


def _apple_colors(state: dict):
    y, a = state["yeast"], state["acetic"]
    fresh = np.array([122, 163, 58])
    ferm = np.array([150, 105, 40])
    vin = np.array([90, 55, 30])
    c = fresh * (1 - y) + ferm * y
    c = c * (1 - min(1, a / 4)) + vin * min(1, a / 4)
    light = np.minimum(255, c * 1.35)
    hx = lambda v: "#" + "".join(f"{int(x):02x}" for x in v)  # noqa: E731
    return hx(c), hx(light)


def export(run_dir: Path, max_active_per_frame: int = 6000) -> Path:
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run.json").read_text())
    frames = json.loads((run_dir / "frames.json").read_text())
    data = {"title": run_dir.name, "frame_dt": info["frame_dt"], "arena": info["arena"],
            "meta": info["meta"], "zones": info["zones"], "flies": {},
            "region_names": REGION_NAMES, "region_colors": REGION_COLORS}
    col, light = _apple_colors(info["apple"]["state"])
    data["apple"] = {**info["apple"], "color": col, "color_light": light}
    meta = info["meta"] or {}
    if meta.get("experiment") == "place_avoidance":
        name = next(iter(frames))
        ts = np.array([f["t"] for f in frames[name]])
        for z in data["zones"]:
            z["schedule"] = ((ts > 20.0) & (ts <= 50.0)).tolist()
    if meta.get("experiment") == "tmaze_choice":
        data["odor_sources"] = [{"pos": [-40, 0, 0], "label": "A (наказание)"},
                                {"pos": [40, 0, 0], "label": "B"}]
    for name, fl in info["flies"].items():
        _, ann = load(fl["sex"])
        z = np.load(run_dir / f"spikes_{name}.npz")
        lens, idx, cnt = z["lens"], z["idx"], z["cnt"]
        starts = np.concatenate([[0], np.cumsum(lens)])
        fidx, fval, fstart = [], [], [0]
        for k in range(len(lens)):
            i, c = idx[starts[k]:starts[k + 1]], cnt[starts[k]:starts[k + 1]]
            if len(i) > max_active_per_frame:
                top = np.argsort(c)[-max_active_per_frame:]
                i, c = i[top], c[top]
            fidx.append(i.astype(np.uint32))
            fval.append(np.minimum(c, 255).astype(np.uint8))
            fstart.append(fstart[-1] + len(i))
        data["flies"][name] = {
            "sex": fl["sex"], "events": fl["events"], "frames": frames[name],
            "brain": {"n": int(len(ann)), "xyz": _b64(_positions(ann)), "region": _b64(_regions(ann)),
                      "fstart": _b64(np.array(fstart, np.uint32)),
                      "idx": _b64(np.concatenate(fidx) if fidx else np.zeros(0, np.uint32)),
                      "val": _b64(np.concatenate(fval) if fval else np.zeros(0, np.uint8))},
        }
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(data, default=float))
    out = run_dir / "viewer.html"
    out.write_text(html)
    return out


if __name__ == "__main__":
    import sys
    for d in sys.argv[1:]:
        p = export(Path(d))
        print(p, f"{p.stat().st_size / 1e6:.1f} MB")
