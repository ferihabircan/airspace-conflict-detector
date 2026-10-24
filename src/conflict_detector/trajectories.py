"""Trajectory generation and loading (OpenSky state-vector format supported)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .geo import haversine_nm, interpolate_great_circle

FT_PER_M = 3.28084

# Core area airports (lat, lon)
AIRPORTS = {
    "EBBR": (50.901, 4.484), "EHAM": (52.309, 4.764), "EDDF": (50.033, 8.570),
    "EDDL": (51.289, 6.767), "EDDK": (50.866, 7.143), "ELLX": (49.627, 6.211),
    "LFPG": (49.010, 2.548), "EGLL": (51.470, -0.454), "EDDM": (48.354, 11.786),
    "LSZH": (47.458, 8.555), "EDDH": (53.630, 9.988), "LFSB": (47.590, 7.529),
}

COLUMNS = ["time", "callsign", "lat", "lon", "alt_ft"]


def _cruise_level(lon_from: float, lon_to: float, rng) -> int:
    """Semicircular rule: eastbound odd, westbound even flight levels."""
    east = lon_to >= lon_from
    levels = [290, 310, 330, 350, 370, 390] if east else [300, 320, 340, 360, 380, 400]
    return int(rng.choice(levels)) * 100


def generate_traffic(n_flights: int = 500, hours: float = 4.0, step_s: int = 10,
                     seed: int = 7) -> pd.DataFrame:
    """Synthetic en-route traffic with climb, cruise and descent phases."""
    rng = np.random.default_rng(seed)
    names = list(AIRPORTS)
    rows = []
    for i in range(n_flights):
        a, b = rng.choice(names, 2, replace=False)
        (la1, lo1), (la2, lo2) = AIRPORTS[a], AIRPORTS[b]
        dist = haversine_nm(la1, lo1, la2, lo2)
        gs_kt = rng.normal(430, 25)
        duration = dist / gs_kt * 3600
        t0 = rng.uniform(0, hours * 3600 - duration) if duration < hours * 3600 else 0
        t = np.arange(0, duration, step_s)
        frac = t / duration
        lat, lon = interpolate_great_circle(la1, lo1, la2, lo2, frac)
        cruise = _cruise_level(lo1, lo2, rng)
        climb = 2200 / 60 * t                    # ft/s * s
        descent = 2000 / 60 * (duration - t)
        alt = np.minimum.reduce([climb, descent, np.full_like(t, cruise)])
        rows.append(pd.DataFrame({
            "time": (t0 + t).round().astype(int), "callsign": f"SYN{i:04d}",
            "lat": lat, "lon": lon, "alt_ft": alt.round(),
        }))
    return pd.concat(rows, ignore_index=True)[COLUMNS]


def load_opensky_states(path: str | Path) -> pd.DataFrame:
    """Load an OpenSky state-vector CSV (columns time, icao24, callsign, lat, lon,
    baroaltitude in metres) into the internal schema."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["lat", "lon", "baroaltitude"])
    cs = df["callsign"].fillna(df["icao24"]).astype(str).str.strip()
    return pd.DataFrame({
        "time": df["time"].astype(int), "callsign": cs,
        "lat": df["lat"], "lon": df["lon"],
        "alt_ft": df["baroaltitude"] * FT_PER_M,
    })


def resample(df: pd.DataFrame, step_s: int = 10) -> pd.DataFrame:
    """Linear interpolation of every flight onto a common time grid."""
    out = []
    for cs, g in df.sort_values("time").groupby("callsign"):
        grid = np.arange(g["time"].min() // step_s * step_s + step_s,
                         g["time"].max() + 1, step_s)
        if len(grid) == 0:
            continue
        out.append(pd.DataFrame({
            "time": grid, "callsign": cs,
            "lat": np.interp(grid, g["time"], g["lat"]),
            "lon": np.interp(grid, g["time"], g["lon"]),
            "alt_ft": np.interp(grid, g["time"], g["alt_ft"]),
        }))
    return pd.concat(out, ignore_index=True)[COLUMNS]
