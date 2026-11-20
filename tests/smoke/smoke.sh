#!/usr/bin/env bash
# Live curl smoke test against a running gateway (needs ComfyUI + GPU; not run in CI).
#   GATEWAY_URL=http://localhost:8001 GATEWAY_API_KEY=... tests/smoke/smoke.sh
set -euo pipefail

URL="${GATEWAY_URL:-http://localhost:8001}"
KEY="${GATEWAY_API_KEY:?set GATEWAY_API_KEY}"
fail=0

check() {  # check <name> <expected status> <actual status>
  if [[ "$2" == "$3" ]]; then echo "ok   $1 ($3)"; else echo "FAIL $1: expected $2 got $3"; fail=1; fi
}

status() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

check "healthz" 200 "$(status "$URL/healthz")"
check "readyz" 200 "$(status "$URL/readyz")"
check "models without key -> 401" 401 "$(status "$URL/v1/models")"
check "models with bad key -> 401" 401 "$(status -H 'Authorization: Bearer wrong' "$URL/v1/models")"
check "models" 200 "$(status -H "Authorization: Bearer $KEY" "$URL/v1/models")"
curl -s -H "Authorization: Bearer $KEY" "$URL/v1/models"; echo
check "unknown model -> 404" 404 "$(status -H "Authorization: Bearer $KEY" "$URL/v1/models/nope")"
check "bad size -> 400" 400 "$(status -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"sdxl","prompt":"x","size":"333x333"}' "$URL/v1/images/generations")"

tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
code="$(curl -s -o "$tmp" -w '%{http_code}' --max-time 100 \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"sdxl","prompt":"a red fox in the snow","quality":"low","response_format":"url"}' \
  "$URL/v1/images/generations")"
check "generation (url)" 200 "$code"
if [[ "$code" == 200 ]]; then
  img_url="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["data"][0]["url"])' "$tmp")"
  echo "image url: $img_url"
  check "image download (no auth)" 200 "$(status "$img_url")"
fi

exit "$fail"
