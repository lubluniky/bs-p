#!/usr/bin/env bash
set -euo pipefail

# 1) Verify native FFI build/link first.
cargo build --release

# 2) Preflight check (packaging + verification, no upload).
cargo publish --dry-run

# 3) Publish new version to crates.io.
cargo publish
