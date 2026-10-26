"""Run the analysis and export what the project website needs to docs/data.json:
downsampled trajectories for the replay map, events, alerts and sector load.

    PYTHONPATH=src python scripts/export_site_data.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from conflict_detector.detect import (detect_losses, group_events, predict_conflicts,
                                      warning_times)
from conflict_detector.sectors import hotspots, occupancy
from conflict_detector.trajectories import AIRPORTS, generate_traffic, resample

OUT = Path(__file__).resolve().parents[1] / "docs" / "data.json"
REPLAY_STEP_S = 30
CAPACITY = 8


def main() -> None:
    traffic = generate_traffic(n_flights=500)
    t0 = time.perf_counter()
    events = group_events(detect_losses(traffic))
    runtime = time.perf_counter() - t0
    alerts = predict_conflicts(traffic, lookahead_s=300)
    warn = warning_times(events, alerts)
    occ = occupancy(traffic)
    hot = hotspots(occ, CAPACITY)

    pos = traffic.set_index(["callsign", "time"])
    ev = events.merge(warn[["ac1", "ac2", "alert_time", "warning_s"]], on=["ac1", "ac2"], how="left")
    ev_rows = []
    for r in ev.itertuples():
        p1, p2 = pos.loc[(r.ac1, r.start)], pos.loc[(r.ac2, r.start)]
        ev_rows.append({
            "ac1": r.ac1, "ac2": r.ac2, "start": int(r.start), "end": int(r.end),
            "duration_s": int(r.duration_s), "min_h_nm": round(float(r.min_h_nm), 2),
            "min_v_ft": round(float(r.min_v_ft)), "severity": r.severity,
            "lat": round(float((p1.lat + p2.lat) / 2), 4),
            "lon": round(float((p1.lon + p2.lon) / 2), 4),
            "fl1": int(p1.alt_ft // 100), "fl2": int(p2.alt_ft // 100),
            "warning_s": None if r.warning_s != r.warning_s else int(r.warning_s),
        })

    # Compact trajectories: per flight, [t0, step, [[lat, lon, FL], ...]]
    flights = {}
    for cs, g in resample(traffic, REPLAY_STEP_S).groupby("callsign"):
        g = g.sort_values("time")
        flights[cs] = [int(g["time"].iat[0]), REPLAY_STEP_S,
                       [[round(a, 3), round(b, 3), int(c // 100)]
                        for a, b, c in zip(g["lat"], g["lon"], g["alt_ft"])]]

    windows = sorted(occ["window"].unique())
    top = (occ.groupby("sector")["flights"].max().sort_values(ascending=False).index[:14])
    heat = (occ[occ["sector"].isin(top)]
            .pivot_table(index="sector", columns="window", values="flights", fill_value=0)
            .reindex(index=top, columns=windows, fill_value=0))

    payload = {
        "summary": {
            "flights": int(traffic["callsign"].nunique()), "positions": int(len(traffic)),
            "events": len(ev_rows), "runtime_s": round(runtime, 2),
            "predicted_pct": round(100 * float(warn["alert_time"].notna().mean()), 1),
            "median_warning_s": float(warn["warning_s"].median()),
            "overloaded_windows": int(len(hot)), "capacity": CAPACITY,
            "duration_s": int(traffic["time"].max()),
        },
        "airports": {k: list(v) for k, v in AIRPORTS.items()},
        "events": ev_rows,
        "flights": flights,
        "heatmap": {"sectors": list(heat.index), "windows": [int(w) for w in windows],
                    "values": heat.astype(int).values.tolist()},
        "hotspots": hot.head(10).round(1).to_dict(orient="records"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(payload["summary"])
    print(ev[["severity"]].value_counts().to_dict())


if __name__ == "__main__":
    main()
