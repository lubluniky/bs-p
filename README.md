# polymarket-kernel

![Rust](https://img.shields.io/badge/Rust-2024-orange?logo=rust)
![AVX-512](https://img.shields.io/badge/SIMD-AVX--512-blue)
![HFT](https://img.shields.io/badge/Use%20Case-HFT-critical)
![Zero-Allocation](https://img.shields.io/badge/Hot%20Path-Zero%20Allocation-success)

Ultra-low latency computational core for Polymarket market making, based on the Logit Jump-Diffusion framework and an Avellaneda-Stoikov adaptation in logit space.

## Overview

`polymarket-kernel` implements a unified stochastic kernel for prediction markets where probabilities are transformed into log-odds and processed with vectorized math.

The crate is designed for:
- HFT backtesting engines
- live market-making bots
- inventory-aware quoting across large market batches

It exposes an FFI-safe Rust API backed by a C SIMD kernel.

Source paper:
- [Toward Black-Scholes for Prediction Markets (Shaw & Dalen, 2025)](https://arxiv.org/pdf/2510.15205)

## Features

- SoA (Structure of Arrays) layout for contiguous memory access and SIMD-friendly loads
- AVX-512 vectorized quote engine for batch processing
- Custom AVX-512 `log1p` approximation to avoid scalar fallback in spread computation
- Fast sigmoid approximation for `x -> p` mapping in hot paths
- Inventory-aware Avellaneda-Stoikov quoting in logit space
- Zero allocations in the hot path (pre-allocated input/output buffers)
- Numerically safe clamping for stable `logit`/`sigmoid` evaluation

## Quick Start

Install:

```bash
cargo add polymarket-kernel
```

Call `calculate_quotes_logit` with SoA input slices:

```rust
use polymarket_kernel::calculate_quotes_logit;

fn main() {
    // SoA inputs for N markets.
    let x_t = vec![0.15, -0.35, 0.90, -1.20];
    let q_t = vec![10.0, -6.0, 3.0, 0.0];
    let sigma_b = vec![0.22, 0.18, 0.30, 0.15];
    let gamma = vec![0.08, 0.08, 0.08, 0.08];
    let tau = vec![0.50, 0.50, 0.50, 0.50];
    let k = vec![1.40, 1.25, 1.10, 1.80];

    // Pre-allocated outputs (no hot-path allocation inside the kernel).
    let mut bid_p = vec![0.0; x_t.len()];
    let mut ask_p = vec![0.0; x_t.len()];

    calculate_quotes_logit(
        &x_t,
        &q_t,
        &sigma_b,
        &gamma,
        &tau,
        &k,
        &mut bid_p,
        &mut ask_p,
    );

    for i in 0..x_t.len() {
        println!("market {i}: bid={:.6}, ask={:.6}", bid_p[i], ask_p[i]);
    }
}
```

## Benchmark Snapshot

```text
============================================================
 POLYMARKET-KERNEL RAW BENCHMARK
============================================================
 Quote Batch Size        :       8192 markets
 Quote Iterations        :     100000
 AVX-512 Quote Latency   :       6.71 ns/market
------------------------------------------------------------
 SPSC Ring Capacity      :    1048576
 SPSC Messages           :   10000000
 SPSC Throughput         :      29.05 M msgs/sec
============================================================
```

## License

MIT
