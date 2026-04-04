#!/usr/bin/env bash
set -euo pipefail

# Run from repository root regardless of current working directory.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST_PATH="$ROOT_DIR/packages/crates/Cargo.toml"

# 1) Verify native FFI build/link first.
cargo build --release --manifest-path "$MANIFEST_PATH"

# 2) Preflight check (packaging + verification, no upload).
cargo publish --dry-run --manifest-path "$MANIFEST_PATH"

# 3) Publish new version to crates.io.
cargo publish --manifest-path "$MANIFEST_PATH"
