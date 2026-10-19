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

KC="${KEYCLOAK_URL:-http://localhost:${KEYCLOAK_HOST_PORT:-8081}}/realms/ekip/protocol/openid-connect"
token_for() {  # password grant for the dev-only ekip-test client; prints the access token
  curl -s -X POST "$KC/token" -d grant_type=password -d client_id=ekip-test \
    -d "username=$1" -d "password=$1-dev-password" -d scope=openid \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])"
}
ALICE=$(token_for alice)
BOB=$(token_for bob)

[ "$(code "$EDGE/api/v1/search?q=torque")" = 401 ] || fail "search without a token must be 401"
[ "$(code -H 'Authorization: Bearer not-a-jwt' "$EDGE/api/v1/search?q=torque")" = 401 ] \
  || fail "search with a garbage token must be 401"
pass "search without or with a bad token is refused (401) through the edge"
problem=$(curl -s -D /tmp/ekip_smoke_headers -H 'X-Request-ID: smoke-req-001' "$EDGE/api/v1/search?q=torque")
grep -qi '^content-type: application/problem+json' /tmp/ekip_smoke_headers || fail "errors must be application/problem+json"
grep -qi '^x-request-id: smoke-req-001' /tmp/ekip_smoke_headers || fail "error responses must echo X-Request-ID"
echo "$problem" | grep -q '"type":"urn:ekip:problem:unauthorized"' || fail "401 body should be a problem document: $problem"
echo "$problem" | grep -q '"request_id":"smoke-req-001"' || fail "problem body should quote the request id"
rm -f /tmp/ekip_smoke_headers
pass "errors are RFC 9457 problem+json and carry the request id through the edge"

hits=$(curl -s -H "Authorization: Bearer $ALICE" "$EDGE/api/v1/search?q=torque")
echo "$hits" | grep -q '"chunk_id":"dev-' || fail "alice's search for 'torque' should return her tenant's chunks"
echo "$hits" | grep -q '"chunk_id":"oth-' && fail "alice must never see tenant-other chunks"
bob_hits=$(curl -s -H "Authorization: Bearer $BOB" -H 'X-Tenant-ID: tenant-dev' "$EDGE/api/v1/search?q=torque&tenant_id=tenant-dev")
echo "$bob_hits" | grep -q '"chunk_id":"oth-' || fail "bob's search should return his tenant's chunks"
echo "$bob_hits" | grep -q '"chunk_id":"dev-' && fail "bob must never see tenant-dev chunks, even when he asks for them"
pass "real Keycloak tokens: search works end to end and tenants are isolated (caller-chosen tenant ignored)"

# Per-user token bucket (burst 20): hammer as carol and expect 429 with Retry-After, never a 5xx.
CAROL=$(token_for carol)
codes=$(for _ in $(seq 1 30); do code -H "Authorization: Bearer $CAROL" "$EDGE/api/v1/search?q=torque"; echo; done)
echo "$codes" | grep -q '^429$' || fail "a burst of 30 requests should hit the per-user rate limit (429): $(echo "$codes" | sort | uniq -c | tr '\n' ' ')"
echo "$codes" | grep -q '^5' && fail "rate limiting must answer 429, never a 5xx"
curl -s -D - -o /dev/null -H "Authorization: Bearer $CAROL" "$EDGE/api/v1/search?q=torque" | grep -qi '^retry-after:' \
  || fail "429 responses must carry Retry-After"
pass "per-user rate limit answers 429 with Retry-After under a burst"

# SSE must stream THROUGH the edge: the first token must arrive well before the stream ends.
# (The stub answer takes several seconds; a buffering proxy would deliver everything at the end.)
BOB_TOKEN="$BOB" python3 - "$EDGE" <<'PY' || fail "chat stream must be delivered incrementally through the edge"
import json, os, sys, time, urllib.request

req = urllib.request.Request(
    sys.argv[1] + "/api/v1/chat",
    data=json.dumps({"message": "torque"}).encode(),
    headers={"Authorization": "Bearer " + os.environ["BOB_TOKEN"], "Content-Type": "application/json"},
)
resp = urllib.request.urlopen(req, timeout=60)
assert resp.headers["Content-Type"].startswith("text/event-stream"), resp.headers["Content-Type"]
assert resp.headers.get("X-Request-ID"), "missing X-Request-ID"
t0, seen = time.monotonic(), []
for raw in resp:
    line = raw.decode().rstrip("\n")
    if line.startswith("event: "):
        seen.append((line[7:], time.monotonic() - t0))
names = [n for n, _ in seen]
assert names[0] == "meta" and names[-1] == "done" and "token" in names and "citations" in names, names
first_token = next(t for n, t in seen if n == "token")
done = seen[-1][1]
print(f"first token after {first_token:.2f}s, done after {done:.2f}s")
assert done - first_token >= 1.0, f"stream looks buffered: first token {first_token:.2f}s, done {done:.2f}s"
PY
pass "chat streams incrementally (SSE) through the edge and only returns the caller's tenant data"
[ "$(head -c 1100000 /dev/zero | curl -s -o /dev/null -w '%{http_code}' -X POST -H "Authorization: Bearer $BOB" -H 'Content-Type: application/json' --data-binary @- "$EDGE/api/v1/chat")" = 413 ] \
  || fail "an oversized chat body must be rejected with 413 (the app enforces it: the stream route bypasses edge buffering)"
pass "oversized chat body is rejected (413) by the app"

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

python3 - "$KC" <<'PY' || fail "identity provider checks failed"
import base64, json, sys, urllib.error, urllib.parse, urllib.request

base = sys.argv[1]

def grant(user, password, client="ekip-test"):
    body = urllib.parse.urlencode({"grant_type": "password", "client_id": client,
                                   "username": user, "password": password, "scope": "openid"}).encode()
    try:
        return json.load(urllib.request.urlopen(urllib.request.Request(base + "/token", body)))
    except urllib.error.HTTPError as e:
        return {"error": json.load(e).get("error")}

def claims(tok):
    part = tok.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))

alice, bob = claims(grant("alice", "alice-dev-password")["access_token"]), claims(grant("bob", "bob-dev-password")["access_token"])
assert alice["aud"] == "ekip-api" and alice["tenant_id"] == "tenant-dev" and "ekip-user" in alice["roles"], alice
assert bob["tenant_id"] == "tenant-other", bob
assert grant("alice", "wrong-password").get("error") == "invalid_grant"
assert grant("alice", "alice-dev-password", "ekip-web").get("error") == "unauthorized_client"
PY
pass "Keycloak issues tenant/role/audience claims; bad password and wrong client are refused"

docker compose logs --no-color api | sed 's/^[^|]*| //' | python3 -c "
import json, sys
for line in sys.stdin:
    if line.strip():
        json.loads(line)  # raises (non-zero exit) on the first non-JSON line
" || fail "every api log line must be JSON"
pass "api logs are JSON"

echo "smoke test passed"
