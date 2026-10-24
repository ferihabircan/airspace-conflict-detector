import itertools

import numpy as np
import pandas as pd
import pytest

from conflict_detector.detect import detect_losses, group_events, predict_conflicts
from conflict_detector.geo import haversine_nm, interpolate_great_circle
from conflict_detector.sectors import hotspots, occupancy
from conflict_detector.trajectories import generate_traffic


def straight(callsign, lat, lon0, lon1, alt, n=60, step=10):
    t = np.arange(n) * step
    return pd.DataFrame({"time": t, "callsign": callsign, "lat": lat,
                         "lon": np.linspace(lon0, lon1, n), "alt_ft": alt})


def test_haversine_known_distance():
    # Brussels -> Amsterdam is about 85 NM
    assert haversine_nm(50.901, 4.484, 52.309, 4.764) == pytest.approx(85, abs=2)


def test_great_circle_endpoints():
    lat, lon = interpolate_great_circle(50, 4, 52, 8, np.array([0, 1]))
    assert lat == pytest.approx([50, 52]) and lon == pytest.approx([4, 8])


def test_head_on_conflict_is_detected_and_predicted():
    a = straight("AAA", 50.0, 4.0, 6.0, 35000)
    b = straight("BBB", 50.0, 6.0, 4.0, 35000)
    events = group_events(detect_losses(pd.concat([a, b])))
    assert len(events) == 1
    alerts = predict_conflicts(pd.concat([a, b]), lookahead_s=300)
    assert alerts["alert_time"].iat[0] < events["start"].iat[0]


def test_vertical_separation_prevents_conflict():
    a = straight("AAA", 50.0, 4.0, 6.0, 35000)
    b = straight("BBB", 50.0, 6.0, 4.0, 36000)
    assert detect_losses(pd.concat([a, b])).empty


def test_kdtree_matches_brute_force():
    traffic = generate_traffic(n_flights=80, hours=1.5, seed=3)
    fast = detect_losses(traffic)
    brute = 0
    hi = traffic[traffic["alt_ft"] >= 10000]
    for _, s in hi.groupby("time"):
        idx = np.array(list(itertools.combinations(range(len(s)), 2)))
        if len(idx) == 0:
            continue
        p, q = s.iloc[idx[:, 0]], s.iloc[idx[:, 1]]
        dh = haversine_nm(p.lat.to_numpy(), p.lon.to_numpy(),
                          q.lat.to_numpy(), q.lon.to_numpy())
        dv = np.abs(p.alt_ft.to_numpy() - q.alt_ft.to_numpy())
        brute += int(((dh < 5) & (dv < 1000)).sum())
    assert len(fast) == brute


def test_hotspots_over_capacity():
    traffic = generate_traffic(n_flights=200, hours=2, seed=1)
    occ = occupancy(traffic)
    hot = hotspots(occ, capacity=3)
    assert (hot["flights"] > 3).all()
    assert len(hot) > 0
