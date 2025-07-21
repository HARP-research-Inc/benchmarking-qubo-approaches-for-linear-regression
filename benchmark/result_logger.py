import csv
from pathlib import Path

class ResultLogger:
    """Accumulates dicts → writes once at the end."""
    def __init__(self, out_path):
        self.out_path = Path(out_path)
        self.rows = []

    def add(self, **kwargs):
        self.rows.append(kwargs)

    def flush(self):
        if not self.rows:
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"[ResultLogger] wrote {len(self.rows)} rows to {self.out_path}")

