sqlite3 results/bench.db \
  "SELECT mode, d, avg(total_time) AS t
     FROM runs
    WHERE error < 1e-2
 GROUP BY mode, d
 ORDER BY d, mode;"
