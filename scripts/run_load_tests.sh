#!/usr/bin/env bash
# Runs the same Locust test against 1, 2 and 3 API containers and at several user counts.
# Usage: scripts/run_load_tests.sh [duration=60s] [user counts="10 50"] [container counts="1 2 3"]
# Raw CSVs -> locust/results/raw/, summary + charts via: python locust/plot_results.py
# By default Locust runs in its own container. Set LOCUST_CMD (e.g. LOCUST_CMD=locust) to run a
# locally installed Locust instead, which leaves all of Docker's memory to the API replicas.
set -euo pipefail
cd "$(dirname "$0")/.."

DURATION=${1:-60s}
USERS=${2:-"10 50"}
CONTAINERS=${3:-"1 2 3"}
mkdir -p locust/results/raw

wait_healthy() {  # wait until every api replica reports healthy
  local n=$1
  for _ in $(seq 1 90); do
    healthy=$(docker compose ps api --format '{{.Health}}' | grep -c '^healthy$' || true)
    [ "$healthy" -ge "$n" ] && return 0
    sleep 2
  done
  echo "replicas did not become healthy" >&2; exit 1
}

for n in $CONTAINERS; do
  echo "=== scaling to $n API container(s)"
  docker compose up -d --scale api="$n" --no-recreate api nginx >/dev/null
  wait_healthy "$n"
  sleep 6   # let nginx re-resolve the replica IPs (resolver valid=5s)

  # Verify load balancing: which containers answer 30 health checks?
  echo -n "    replicas answering: "
  for _ in $(seq 1 30); do curl -s localhost:${HTTP_PORT:-8080}/api/health | grep -o '"container":"[a-f0-9]*"'; done | sort | uniq -c | tr '\n' ' '
  echo

  for u in $USERS; do
    echo "    running $u users for $DURATION"
    if [ -n "${LOCUST_CMD:-}" ]; then
      SAMPLES_DIR=data/samples $LOCUST_CMD -f locust/locustfile.py --headless \
        --host "http://localhost:${HTTP_PORT:-8080}" -u "$u" -r 10 -t "$DURATION" --only-summary \
        --csv "locust/results/raw/c${n}_u${u}" > "locust/results/raw/c${n}_u${u}.log" 2>&1 || true   # locust exits 1 when requests fail
    else
      docker compose --profile loadtest run --rm --no-deps locust -f locustfile.py --headless \
        --host http://nginx -u "$u" -r 10 -t "$DURATION" --only-summary \
        --csv "results/raw/c${n}_u${u}" > "locust/results/raw/c${n}_u${u}.log" 2>&1 || true
    fi
    grep -E "Aggregated" "locust/results/raw/c${n}_u${u}.log" | tail -1 || true
  done
done
echo "Done. Now run: python locust/plot_results.py"
