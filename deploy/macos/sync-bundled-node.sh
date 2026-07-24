#!/usr/bin/env bash
# Autor: Sergio Martinez de Unlockers Cloud
# URL: https://1lockers.net
#
# Copia Node.js empaquetado al runtime del bot (Cursor SDK requiere Node >= 22).
# Uso: source deploy/macos/sync-bundled-node.sh && sync_bundled_node "$SRC" "$RUNTIME"
sync_bundled_node() {
  local src="${1:?src}"
  local runtime="${2:?runtime}"
  local node_dir
  for node_dir in node-v22.16.0-darwin-arm64 node-v20.20.2-darwin-arm64; do
    if [[ -d "${src}/.tools/${node_dir}" ]]; then
      mkdir -p "${runtime}/.tools"
      /usr/bin/rsync -a "${src}/.tools/${node_dir}/" "${runtime}/.tools/${node_dir}/"
    fi
  done
}
