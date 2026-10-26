"""Loss-of-separation detection and short-term conflict prediction.

Pairs are searched with a KD-tree on a local projection (O(n log n) per time
step instead of O(n^2)) and confirmed with the great-circle distance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import haversine_nm, project_local

H_SEP_NM = 5.0       # en-route radar separation
V_SEP_FT = 1000.0    # RVSM vertical separation
MIN_ALT_FT = 10000   # ignore terminal-area traffic (different minima apply)


def add_kinematics(df: pd.DataFrame, lat0: float, lon0: float) -> pd.DataFrame:
    """Add projected position (NM) and velocity (NM/s, ft/s) per flight."""
    df = df.sort_values(["callsign", "time"]).copy()
    df["x"], df["y"] = project_local(df["lat"], df["lon"], lat0, lon0)
    g = df.groupby("callsign")
    dt = g["time"].diff()
    for col, v in (("x", "vx"), ("y", "vy"), ("alt_ft", "vz")):
        df[v] = (g[col].diff() / dt).fillna(0.0)
    return df


def _pairs_at(snap: pd.DataFrame, radius_nm: float) -> np.ndarray:
    if len(snap) < 2:
        return np.empty((0, 2), dtype=int)
    tree = cKDTree(snap[["x", "y"]].to_numpy())
    return tree.query_pairs(radius_nm, output_type="ndarray")


def detect_losses(df: pd.DataFrame, h_nm: float = H_SEP_NM, v_ft: float = V_SEP_FT,
                  min_alt_ft: float = MIN_ALT_FT) -> pd.DataFrame:
    """Every (time, pair) where horizontal AND vertical separation are infringed."""
    lat0, lon0 = df["lat"].mean(), df["lon"].mean()
    df = add_kinematics(df[df["alt_ft"] >= min_alt_ft], lat0, lon0)
    hits = []
    for t, snap in df.groupby("time"):
        snap = snap.reset_index(drop=True)
        # 10% margin covers projection error; exact check follows.
        pairs = _pairs_at(snap, h_nm * 1.1)
        if len(pairs) == 0:
            continue
        a, b = snap.iloc[pairs[:, 0]], snap.iloc[pairs[:, 1]]
        dh = haversine_nm(a["lat"].to_numpy(), a["lon"].to_numpy(),
                          b["lat"].to_numpy(), b["lon"].to_numpy())
        dv = np.abs(a["alt_ft"].to_numpy() - b["alt_ft"].to_numpy())
        m = (dh < h_nm) & (dv < v_ft)
        for i in np.flatnonzero(m):
            p = sorted([a["callsign"].iat[i], b["callsign"].iat[i]])
            hits.append((t, p[0], p[1], dh[i], dv[i]))
    return pd.DataFrame(hits, columns=["time", "ac1", "ac2", "h_nm", "v_ft"])


def group_events(losses: pd.DataFrame, step_s: int = 10) -> pd.DataFrame:
    """Merge consecutive time steps of the same pair into one event."""
    if losses.empty:
        return pd.DataFrame(columns=["ac1", "ac2", "start", "end", "duration_s",
                                     "min_h_nm", "min_v_ft", "severity"])
    losses = losses.sort_values(["ac1", "ac2", "time"])
    new = (losses.groupby(["ac1", "ac2"])["time"].diff() != step_s)
    losses["event"] = new.cumsum()
    ev = losses.groupby("event").agg(
        ac1=("ac1", "first"), ac2=("ac2", "first"), start=("time", "min"),
        end=("time", "max"), min_h_nm=("h_nm", "min"), min_v_ft=("v_ft", "min"))
    ev["duration_s"] = ev["end"] - ev["start"] + step_s
    ev["severity"] = classify_severity(ev["min_h_nm"], ev["min_v_ft"])
    return ev.reset_index(drop=True)


def classify_severity(min_h_nm, min_v_ft, h_nm: float = H_SEP_NM,
                      v_ft: float = V_SEP_FT) -> np.ndarray:
    """Simple severity bands from how much of the minima was kept.

    The closer of the two separations (as a share of its minimum) decides:
    < 50% Serious, < 75% Significant, otherwise Minor. Illustrative only;
    official classification (e.g. the EUROCONTROL RAT) uses more factors.
    """
    kept = np.maximum(np.asarray(min_h_nm) / h_nm, np.asarray(min_v_ft) / v_ft)
    return np.select([kept < 0.5, kept < 0.75], ["Serious", "Significant"], "Minor")


def predict_conflicts(df: pd.DataFrame, lookahead_s: float = 300,
                      h_nm: float = H_SEP_NM, v_ft: float = V_SEP_FT,
                      min_alt_ft: float = MIN_ALT_FT) -> pd.DataFrame:
    """Short-term conflict prediction by linear extrapolation to the closest
    point of approach (CPA). Returns the first alert time per pair."""
    lat0, lon0 = df["lat"].mean(), df["lon"].mean()
    df = add_kinematics(df[df["alt_ft"] >= min_alt_ft], lat0, lon0)
    max_closure = 2 * 0.15  # NM/s, ~540 kt each
    radius = h_nm + max_closure * lookahead_s
    alerts = {}
    for t, snap in df.groupby("time"):
        snap = snap.reset_index(drop=True)
        pairs = _pairs_at(snap, radius)
        if len(pairs) == 0:
            continue
        A, B = snap.iloc[pairs[:, 0]], snap.iloc[pairs[:, 1]]
        dp = B[["x", "y"]].to_numpy() - A[["x", "y"]].to_numpy()
        dv = B[["vx", "vy"]].to_numpy() - A[["vx", "vy"]].to_numpy()
        dv2 = (dv ** 2).sum(1)
        tcpa = np.where(dv2 > 1e-12, -(dp * dv).sum(1) / np.maximum(dv2, 1e-12), 0)
        tcpa = np.clip(tcpa, 0, lookahead_s)
        dcpa = np.linalg.norm(dp + dv * tcpa[:, None], axis=1)
        dz = np.abs((B["alt_ft"].to_numpy() + B["vz"].to_numpy() * tcpa)
                    - (A["alt_ft"].to_numpy() + A["vz"].to_numpy() * tcpa))
        m = (dcpa < h_nm) & (dz < v_ft)
        for i in np.flatnonzero(m):
            key = tuple(sorted([A["callsign"].iat[i], B["callsign"].iat[i]]))
            alerts.setdefault(key, (t, tcpa[i], dcpa[i]))
    rows = [(k[0], k[1], v[0], v[1], v[2]) for k, v in alerts.items()]
    return pd.DataFrame(rows, columns=["ac1", "ac2", "alert_time", "t_to_cpa_s",
                                       "pred_dcpa_nm"])


def warning_times(events: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    """Join alerts to actual events: how many seconds of warning did we give?"""
    first = events.groupby(["ac1", "ac2"], as_index=False)["start"].min()
    j = first.merge(alerts, on=["ac1", "ac2"], how="left")
    j["warning_s"] = j["start"] - j["alert_time"]
    return j
