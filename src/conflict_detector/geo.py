"""Geodesy helpers. Distances in nautical miles, angles in degrees."""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_NM = 3440.065


def haversine_nm(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_NM * np.arcsin(np.sqrt(a))


def interpolate_great_circle(lat1, lon1, lat2, lon2, frac):
    """Points at fraction `frac` (array, 0..1) along the great circle."""
    p1, l1, p2, l2 = map(np.radians, (lat1, lon1, lat2, lon2))
    d = haversine_nm(lat1, lon1, lat2, lon2) / EARTH_RADIUS_NM
    frac = np.asarray(frac, dtype=float)
    if d == 0:
        return np.full_like(frac, lat1), np.full_like(frac, lon1)
    a = np.sin((1 - frac) * d) / np.sin(d)
    b = np.sin(frac * d) / np.sin(d)
    x = a * np.cos(p1) * np.cos(l1) + b * np.cos(p2) * np.cos(l2)
    y = a * np.cos(p1) * np.sin(l1) + b * np.cos(p2) * np.sin(l2)
    z = a * np.sin(p1) + b * np.sin(p2)
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def project_local(lat, lon, lat0: float, lon0: float):
    """Equirectangular projection to a local x/y plane in NM around (lat0, lon0)."""
    x = (np.asarray(lon) - lon0) * 60.0 * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * 60.0
    return x, y
