#!/usr/bin/env python3
"""
db_summary.py -- quick report of average encode / anneal / total times
for Box-Naive vs Box-Opt, grouped by d (feature count).

Usage:
    python analysis/db_summary.py [bench.db]

If no DB path is given it defaults to results/bench.db
"""

import sqlite3, sys, statistics
from pathlib import Path

DB_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "results/bench.db")

###############################################################################
def fetch_aggregates(conn):
    """
    Returns nested dict:
        stats[mode][d] = {
            'runs': int, 'encode': float, 'anneal': float, 'total': float
        }
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT mode, d,
               COUNT(*)            AS runs,
               AVG(encode_time)    AS encode_avg,
               AVG(anneal_time)    AS anneal_avg,
               AVG(total_time)     AS total_avg
        FROM   runs
        WHERE  mode IN ('box-naive','box-opt')
        GROUP  BY mode, d
        ORDER  BY d, mode
    """)
    stats = {}
    for mode, d, runs, enc, ann, tot in cur.fetchall():
        stats.setdefault(mode, {})[d] = {
            "runs":   runs,
            "encode": enc,
            "anneal": ann,
            "total":  tot,
        }
    return stats


def render_table(stats):
    # header
    print(f"{'mode':<12} {'d':>5} {'runs':>5} "
          f"{'encode_avg':>12} {'anneal_avg':>12} {'total_avg':>11}")
    print("-" * 65)

    modes = ("box-naive", "box-opt")
    all_ds = sorted(
        {d for m in modes for d in stats.get(m, {}).keys()}
    )

    for d in all_ds:
        for mode in modes:
            s = stats.get(mode, {}).get(d)
            if not s:
                continue
            print(f"{mode:<12} {d:>5} {s['runs']:>5} "
                  f"{s['encode']:>12.4f} "
                  f"{s['anneal']:>12.4f} "
                  f"{s['total']:>11.4f}")
    print()

    # extra comparative speed-up table
    print("Speed-up (Opt ÷ Naive)")
    print(f"{'d':>5} {'encode_x':>10} {'total_x':>10}")
    print("-" * 29)
    for d in all_ds:
        n, o = stats["box-naive"].get(d), stats["box-opt"].get(d)
        if n and o:
            enc_mul  = n["encode"] / o["encode"]
            tot_mul  = n["total"]  / o["total"]
            print(f"{d:>5} {enc_mul:>10.2f} {tot_mul:>10.2f}")
    print()


###############################################################################
def main():
    if not DB_PATH.exists():
        sys.exit(f"[err] database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        stats = fetch_aggregates(conn)
        if not stats:
            sys.exit("[err] no Box-Naive / Box-Opt rows found")
        render_table(stats)


if __name__ == "__main__":
    main()

