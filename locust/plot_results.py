"""Summarise Locust CSV results into a markdown table and charts.

Reads locust/results/raw/c{containers}_u{users}_stats.csv (Aggregated row) and writes:
  locust/results/summary.csv, locust/results/summary.md, locust/results/*.png
"""
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RAW = RESULTS / "raw"


def load_rows():
    rows = []
    for f in sorted(RAW.glob("c*_u*_stats.csv")):
        m = re.match(r"c(\d+)_u(\d+)_stats\.csv", f.name)
        with f.open() as fh:
            agg = next(r for r in csv.DictReader(fh) if r["Name"] == "Aggregated")
        pred = None
        with f.open() as fh:
            pred = next((r for r in csv.DictReader(fh) if r["Name"] == "POST /api/predict"), None)
        reqs, fails = int(agg["Request Count"]), int(agg["Failure Count"])
        rows.append({
            "containers": int(m.group(1)), "users": int(m.group(2)),
            "requests": reqs, "failures": fails,
            "failure_rate_%": round(100 * fails / reqs, 2) if reqs else 0.0,
            "throughput_rps": round(float(agg["Requests/s"]), 2),
            "avg_ms": round(float(agg["Average Response Time"])),
            "median_ms": round(float(agg["Median Response Time"])),
            "p95_ms": round(float(agg["95%"])),
            "p99_ms": round(float(agg["99%"])),
            "predict_avg_ms": round(float(pred["Average Response Time"])) if pred else None,
        })
    return sorted(rows, key=lambda r: (r["users"], r["containers"]))


def main():
    rows = load_rows()
    if not rows:
        raise SystemExit(f"No results in {RAW}")
    keys = list(rows[0])
    with (RESULTS / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    header = "| Containers | Users | Requests | Failure rate | Throughput (req/s) | Avg (ms) | Median (ms) | p95 (ms) | p99 (ms) |"
    lines = [header, "|" + "---|" * 9]
    for r in rows:
        lines.append(f"| {r['containers']} | {r['users']} | {r['requests']} | {r['failure_rate_%']}% | "
                     f"{r['throughput_rps']} | {r['avg_ms']} | {r['median_ms']} | {r['p95_ms']} | {r['p99_ms']} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    users = sorted({r["users"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for u in users:
        sub = [r for r in rows if r["users"] == u]
        c = [r["containers"] for r in sub]
        axes[0].plot(c, [r["avg_ms"] for r in sub], marker="o", label=f"{u} users")
        axes[1].plot(c, [r["p95_ms"] for r in sub], marker="o", label=f"{u} users")
        axes[2].plot(c, [r["throughput_rps"] for r in sub], marker="o", label=f"{u} users")
    for ax, title, unit in zip(axes, ["Average response time", "95th percentile response time", "Throughput"],
                               ["ms", "ms", "requests / second"]):
        ax.set_title(title); ax.set_xlabel("API containers"); ax.set_ylabel(unit)
        ax.set_xticks(sorted({r["containers"] for r in rows})); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "scaling.png", dpi=120)
    print("wrote", RESULTS / "scaling.png")


if __name__ == "__main__":
    main()
