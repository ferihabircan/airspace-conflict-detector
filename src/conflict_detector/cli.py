"""Run the full analysis: python -m conflict_detector.cli [--opensky states.csv]"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .detect import detect_losses, group_events, predict_conflicts, warning_times  # noqa: E402
from .sectors import hotspots, occupancy  # noqa: E402
from .trajectories import AIRPORTS, generate_traffic, load_opensky_states, resample  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--opensky", help="OpenSky state-vector CSV")
    ap.add_argument("--flights", type=int, default=500)
    ap.add_argument("--capacity", type=int, default=8,
                    help="max flights per sector per 20-minute window")
    ap.add_argument("--lookahead", type=int, default=300)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.opensky:
        traffic = resample(load_opensky_states(args.opensky))
    else:
        traffic = generate_traffic(n_flights=args.flights)
    print(f"{traffic['callsign'].nunique()} flights, {len(traffic):,} positions")

    t0 = time.perf_counter()
    events = group_events(detect_losses(traffic))
    t_detect = time.perf_counter() - t0
    alerts = predict_conflicts(traffic, lookahead_s=args.lookahead)
    warn = warning_times(events, alerts)
    occ = occupancy(traffic)
    hot = hotspots(occ, args.capacity)

    events.to_csv(out / "los_events.csv", index=False)
    hot.to_csv(out / "sector_hotspots.csv", index=False)
    summary = {
        "flights": int(traffic["callsign"].nunique()),
        "positions": int(len(traffic)),
        "los_events": int(len(events)),
        "detection_runtime_s": round(t_detect, 2),
        "events_predicted_pct": round(100 * warn["alert_time"].notna().mean(), 1)
        if len(warn) else None,
        "median_warning_s": float(warn["warning_s"].median()) if len(warn) else None,
        "predicted_pairs_total": int(len(alerts)),
        "overloaded_sector_windows": int(len(hot)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(9, 7))
    for _, g in traffic.groupby("callsign"):
        ax.plot(g["lon"], g["lat"], lw=0.3, color="#9bb3d1", alpha=0.6)
    if len(events):
        pos = traffic.set_index(["callsign", "time"])
        pts = [pos.loc[(r.ac1, r.start), ["lon", "lat"]] for r in events.itertuples()]
        ax.scatter([p["lon"] for p in pts], [p["lat"] for p in pts],
                   s=28, color="#d62728", zorder=3, label="Loss of separation")
    for code, (la, lo) in AIRPORTS.items():
        if not args.opensky:
            ax.plot(lo, la, "k^", ms=5)
            ax.annotate(code, (lo, la), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Trajectories and {len(events)} loss-of-separation events (above FL100)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "conflict_map.png", dpi=130)

    top = (occ.pivot_table(index="sector", columns="window", values="flights",
                           fill_value=0)
              .loc[lambda d: d.max(1).sort_values(ascending=False).index[:15]])
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(top.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(top)), top.index, fontsize=7)
    ax.set_xticks(range(len(top.columns)), [f"{int(c)//60:02d}:{int(c)%60:02d}"
                                            for c in top.columns], fontsize=7,
                  rotation=90)
    ax.set_xlabel("20-minute window (elapsed hh:mm)")
    ax.set_title(f"Busiest sectors: flights per window (capacity = {args.capacity})")
    fig.colorbar(im, ax=ax, label="flights")
    fig.tight_layout()
    fig.savefig(out / "sector_occupancy.png", dpi=130)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
