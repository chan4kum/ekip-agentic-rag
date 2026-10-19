#!/usr/bin/env bash
# Live smoke test of the running Compose stack. Used by CI and runnable locally:
#   docker compose up -d --build --wait && bash scripts/smoke.sh
set -euo pipefail

EDGE="${EDGE_URL:-http://localhost:8080}"
pass() { echo "ok   - $1"; }
fail() { echo "FAIL - $1" >&2; exit 1; }

code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
in_api() { docker compose exec -T api python -c "$1"; }

[ "$(code "$EDGE/health")" = 200 ] || fail "edge /health should be 200"
pass "edge /health is 200"

for path in /ready /docs /openapi.json; do
  [ "$(code "$EDGE$path")" = 404 ] || fail "edge $path must not be routed (expected 404)"
done
pass "/ready, /docs and /openapi.json are not exposed at the edge"

body=$(in_api "
import urllib.request as u, urllib.error as e
try:
    print(u.urlopen('http://127.0.0.1:8000/ready').read().decode())
except e.HTTPError as x:
    print(x.read().decode())  # 503 body shows which dependency failed
" || true)
echo "$body" | grep -q '"status":"ready"' || fail "api /ready should be ready: $body"
pass "api /ready reports postgres, redis and opensearch ok"

[ "$(head -c 1100000 /dev/zero | curl -s -o /dev/null -w '%{http_code}' -X POST --data-binary @- "$EDGE/api/v1/search")" = 413 ] \
  || fail "1.1 MB body should be rejected with 413"
pass "oversized body is rejected at the edge (413)"

hits=$(curl -s "$EDGE/api/v1/search?q=torque")
echo "$hits" | grep -q '"chunk_id"' || fail "search for 'torque' should return chunks"
pass "search works end to end through the edge"

in_api "
import urllib.request as u
try:
    u.urlopen('https://api.openai.com/v1/models', timeout=10)
except u.HTTPError:
    pass  # any HTTP answer (401 without a key) proves the tunnel to an allowed host works
" || fail "allowed host should be reachable through the egress proxy"
pass "allowed host is reachable through Squid"

in_api "
import urllib.request as u
try:
    u.urlopen('https://example.com', timeout=10)
except Exception as e:
    raise SystemExit(0 if '403' in str(e) else 1)
raise SystemExit(1)
" || fail "blocked host should be refused by the egress proxy (403)"
pass "blocked host is refused by Squid (403)"

docker compose exec -T -e HTTPS_PROXY= api python -c "
import urllib.request as u
try:
    u.urlopen('https://api.openai.com', timeout=6)
except Exception:
    raise SystemExit(0)
raise SystemExit(1)
" || fail "direct internet access (no proxy) must not work"
pass "api has no direct route to the internet"

docker compose logs --no-color api | sed 's/^[^|]*| //' | python3 -c "
import json, sys
for line in sys.stdin:
    if line.strip():
        json.loads(line)  # raises (non-zero exit) on the first non-JSON line
" || fail "every api log line must be JSON"
pass "api logs are JSON"

echo "smoke test passed"
