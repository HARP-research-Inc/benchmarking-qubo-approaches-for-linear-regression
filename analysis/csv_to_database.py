# analysis/csv_to_database.py
import csv, json, re, sqlite3, sys, datetime as dt
from pathlib import Path

DB_PATH  = Path("results/bench_new.db")
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT,          -- ISO timestamp when imported
    mode         TEXT,          -- 'box-naive' / 'box-opt' / 'classical' / 'potok'
    d            INTEGER,
    n            INTEGER,
    k            INTEGER,       -- precision bits (Potok only)  NULL otherwise
    iterations   INTEGER,       -- NULL for classical
    encode_time  REAL,
    anneal_time  REAL,
    total_time   REAL,
    wall_time    REAL,
    error        REAL,
    noise        REAL,
    corr         REAL,
    params       TEXT           -- free-form JSON for extra metadata
);
"""

BOX_QUBO_HEADER = {
    "mode":         "mode",
    "d":            "d",
    "n":            "n",
    "iterations":   "iterations",
    "encode_time":  "encode_time",
    "anneal_time":  "anneal_time",
    "total_time":   "total_time",
    "wall_time":    "wall_time",
    "error":        "error",
}

# potok runs (have extra ‘K’ precision column)
POTOK_HEADER = BOX_QUBO_HEADER | {"k": "k"}

# classical scikit-learn header (unchanged)
CLASSICAL_HEADER = {
    "model":        "mode",
    "d":            "d",
    "n":            "n",
    "noise":        "noise",
    "corr":         "corr",
    "train_time":   "total_time",
    "predict_time": "encode_time",
    "r2":           "anneal_time",
    "mse":          "error",
    "seed":         None,
}

# keep them in one list for the matcher
HEADER_MAPS = [POTOK_HEADER, BOX_QUBO_HEADER, CLASSICAL_HEADER]

# canonicalise filename → mode (hyphen not underscore)
FNAME_MODE_RE = re.compile(r"^(box[-_]naive|box[-_]opt|potok|classical)", re.I)

def normalise_mode(raw: str) -> str:
    raw = raw.lower().replace("_", "-")
    return raw

# ──────────────────────────────────────────────────────────────────────────────
def import_dir(dir_path: Path):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    cur = conn.cursor()

    for csv_file in dir_path.glob("*.csv"):
        with csv_file.open(newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

            # try matching against known header maps
            header_lower = [h.lower() for h in header]
            hdr_map = None
            for m in HEADER_MAPS:
                if set(m.keys()) <= set(header_lower):
                    hdr_map = m
                    break
            if hdr_map is None:
                print(f"[warn] unexpected columns in {csv_file.name}, skipped", file=sys.stderr)
                continue

            col_idx = {name: header_lower.index(name) for name in hdr_map.keys()}

            for row in reader:
                # infer mode from file name if the CSV itself doesn’t carry it
                m = FNAME_MODE_RE.match(csv_file.stem)
                mode_from_fname = normalise_mode(m.group(1)) if m else None

                db_row = {
                    "ts": dt.datetime.utcnow().isoformat(timespec="seconds"),
                    "mode": None,
                    "d": None, "n": None, "k": None,
                    "iterations": None,
                    "encode_time": None, "anneal_time": None,
                    "total_time": None, "wall_time": None,
                    "error": None,
                    "noise": None, "corr": None,
                    "params": json.dumps({"source": csv_file.name})
                }

                # populate from CSV
                for csv_col, db_col in hdr_map.items():
                    if db_col is None:
                        continue
                    val = row[col_idx[csv_col]]
                    if val == "":
                        continue
                    db_row[db_col] = val

                # fill missing mode
                if db_row["mode"] is None:
                    db_row["mode"] = mode_from_fname or "unknown"
                else:
                    db_row["mode"] = normalise_mode(db_row["mode"])

                if db_row["noise"] is None or db_row["corr"] is None:
                    # pattern: “…_NOISE_CORR_rep…”  (eg box_naive_0.05_0.8_rep12.csv)
                    m_nc = re.search(r"_([\d.]+)_([\d.]+)_rep", csv_file.stem)
                    if m_nc:
                        db_row["noise"] = float(m_nc.group(1))
                        db_row["corr"]  = float(m_nc.group(2))
                    else:
                        # default values when nothing available
                        db_row.setdefault("noise", 0.01)
                        db_row.setdefault("corr",  0.0)

                # cast numeric fields
                for key in ("d", "n", "k", "iterations"):
                    if db_row[key] is not None:
                        db_row[key] = int(float(db_row[key]))
                for key in ("encode_time", "anneal_time",
                             "total_time", "wall_time",
                             "error", "noise", "corr"):
                    if db_row[key] is not None:
                        db_row[key] = float(db_row[key])

                cur.execute(
                    """
                    INSERT INTO runs
                    (ts, mode, d, n, k, iterations,
                     encode_time, anneal_time, total_time, wall_time,
                     error, noise, corr, params)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    tuple(db_row[k] for k in
                          ("ts","mode","d","n","k","iterations",
                           "encode_time","anneal_time","total_time","wall_time",
                           "error","noise","corr","params"))
                )

    conn.commit()
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python analysis/csv_to_database.py <results_subdir>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.is_dir():
        print(f"no such directory: {target}", file=sys.stderr)
        sys.exit(1)

    import_dir(target)
    print("✓ import complete")

