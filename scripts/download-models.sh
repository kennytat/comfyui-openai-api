#!/usr/bin/env bash
# Download model files listed in scripts/models.manifest.
# Usage: ./scripts/download-models.sh [model_id ...]   (no args = every entry)
# Run ONCE before the first `docker compose up` so the bind-mount dirs are owned by you.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/scripts/models.manifest"
cd "$ROOT"

# Every storage dir the compose stack mounts.
mkdir -p \
  comfyui/storage-cache/dot-cache \
  comfyui/storage-cache/dot-config \
  comfyui/storage-nodes/dot-local \
  comfyui/storage-nodes/custom_nodes \
  comfyui/storage-models/models/checkpoints \
  comfyui/storage-models/models/vae \
  comfyui/storage-models/hf-hub \
  comfyui/storage-models/torch-hub \
  comfyui/storage-user/input/openai-api \
  comfyui/storage-user/output/openai-api \
  comfyui/storage-user/user-profile \
  comfyui/storage-user/user-scripts \
  gateway-data/images

wanted=("$@")

want() {
  [ ${#wanted[@]} -eq 0 ] && return 0
  local w
  for w in "${wanted[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

sha_of() { sha256sum "$1" | cut -d' ' -f1; }

# The token never appears in curl's argv (visible in `ps`): the header is read from a
# process-substitution fd. Redirects are restricted to https as well.
fetch() {
  local url=$1 out=$2
  local opts=(-fL --proto '=https' --proto-redir '=https' --retry 3 -C - -o "$out")
  if [ -n "${HF_TOKEN:-}" ]; then
    curl "${opts[@]}" -H @<(printf 'Authorization: Bearer %s\n' "$HF_TOKEN") "$url"
  else
    curl "${opts[@]}" "$url"
  fi
}

found=0
while read -r id repo file dest sha _; do
  case "${id:-}" in ''|'#'*) continue ;; esac
  want "$id" || continue
  found=1
  if [ -f "$dest" ]; then
    if [ "$sha" = "-" ]; then
      echo "[skip] $id: $dest exists (no sha256 to verify)"
      continue
    fi
    if [ "$(sha_of "$dest")" = "$sha" ]; then
      echo "[skip] $id: sha256 matches"
      continue
    fi
    echo "[redo] $id: sha256 mismatch, re-downloading"
  fi
  mkdir -p "$(dirname "$dest")"
  url="https://huggingface.co/$repo/resolve/main/$file"
  echo "[get ] $id: $url"
  fetch "$url" "$dest.part"
  if [ "$sha" != "-" ] && [ "$(sha_of "$dest.part")" != "$sha" ]; then
    echo "sha256 mismatch for $id" >&2
    rm -f "$dest.part"
    exit 1
  fi
  mv "$dest.part" "$dest"
  echo "[ok  ] $id -> $dest"
done < "$MANIFEST"

if [ "$found" -eq 0 ]; then
  echo "no matching model ids in $MANIFEST: ${wanted[*]:-}" >&2
  exit 1
fi
