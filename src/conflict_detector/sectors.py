"""Sector occupancy and overload (hotspot) detection on a lat/lon/FL grid."""
from __future__ import annotations

import numpy as np
import pandas as pd

FL_BANDS = [(245, 345, "LOWER"), (345, 460, "UPPER")]


def assign_sectors(df: pd.DataFrame, cell_deg: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    fl = df["alt_ft"] / 100
    band = pd.Series(pd.NA, index=df.index, dtype="object")
    for lo, hi, name in FL_BANDS:
        band[(fl >= lo) & (fl < hi)] = name
    df = df[band.notna()].assign(band=band[band.notna()])
    df["sector"] = (
        "N" + (np.floor(df["lat"] / cell_deg) * cell_deg).map("{:.0f}".format)
        + "E" + (np.floor(df["lon"] / cell_deg) * cell_deg).map("{:.0f}".format)
        + "_" + df["band"]
    )
    return df


def occupancy(df: pd.DataFrame, window_min: int = 20) -> pd.DataFrame:
    """Number of distinct flights in each sector per time window (entry count)."""
    df = assign_sectors(df)
    df["window"] = (df["time"] // (window_min * 60)) * window_min
    return (df.groupby(["sector", "window"])["callsign"].nunique()
              .rename("flights").reset_index())


def hotspots(occ: pd.DataFrame, capacity: int) -> pd.DataFrame:
    hot = occ[occ["flights"] > capacity].copy()
    hot["overload_pct"] = 100 * (hot["flights"] / capacity - 1)
    return hot.sort_values("overload_pct", ascending=False)
