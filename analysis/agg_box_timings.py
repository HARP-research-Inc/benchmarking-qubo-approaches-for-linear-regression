#!/usr/bin/env python3
"""
agg_box_timings.py  –  aggregate run-time statistics for box-naive / box-opt

Usage
-----
  python analysis/agg_box_timings.py             # pretty table to stdout
  python analysis/agg_box_timings.py --csv out.csv
  python analysis/agg_box_timings.py --db path/to/bench.db
"""

import argparse, sqlite3, sys, csv
from pathlib import Path
from statistics import mean

DEFAULT_DB = Path("results/bench.db")

def fetch(conn, modes):
    """Return dict: {mode: {d: [rows]}} for the requested modes."""
    cur = conn.cursor()
    q = """
        SELECT mode, d, encode_time, anneal_time, total_time
          FROM runs
         WHERE mode IN ({})
           AND encode_time IS NOT NULL
    """.format(",".join("?"*len(modes)))
    buckets = {m: {} for m in modes}
    for mode, d, et, at, tt in cur.execute(q, modes):
        buckets[mode].setdefault(d, []).append((et, at, tt))
    return buckets


def summarise(rows):
    et_avg = mean(r[0] for r in rows)
    at_avg = mean(r[1] for r in rows)
    tt_avg = mean(r[2] for r in rows)
    return round(et_avg, 4), round(at_avg, 4), round(tt_avg, 4), len(rows)


def main():
    ap = argparse.ArgumentParser(description="Aggregate box-naive / box-opt timings")
    ap.add_argument("--db",  default=DEFAULT_DB, type=Path, help="bench.db path")
    ap.add_argument("--csv", metavar="FILE", help="write CSV instead of table")
    args = ap.parse_args()

    if not args.db.exists():
        sys.exit(f"[err] DB not found: {args.db}")

    with sqlite3.connect(args.db) as conn:
        buckets = fetch(conn, ["box-naive", "box-opt"])

    # build summary rows
    rows = []
    for mode in ("box-naive", "box-opt"):
        for d, lst in sorted(buckets[mode].items()):
            et, at, tt, n = summarise(lst)
            rows.append({
                "mode": mode,
                "d":    d,
                "runs": n,
                "encode_avg":  et,
                "anneal_avg":  at,
                "total_avg":   tt,
            })

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, rows[0].keys())
            w.writeheader();  w.writerows(rows)
        print(f"✓ CSV written to {args.csv}")
    else:
        # pretty table to stdout
        col_hdr = "{:<10} {:>6} {:>5} {:>12} {:>12} {:>12}"
        col_row = "{:<10} {:>6d} {:>5d} {:>12.4f} {:>12.4f} {:>12.4f}"
        print(col_hdr.format("mode", "d", "runs",
                             "encode_avg", "anneal_avg", "total_avg"))
        print("-"*65)
        for r in rows:
            print(col_row.format(r["mode"], r["d"], r["runs"],
                                 r["encode_avg"], r["anneal_avg"], r["total_avg"]))


if __name__ == "__main__":
    main()

