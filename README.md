# polymarket-kernel

![Rust](https://img.shields.io/badge/Rust-2024-orange?logo=rust)
![AVX-512](https://img.shields.io/badge/SIMD-AVX--512-blue)
![HFT](https://img.shields.io/badge/Use%20Case-HFT-critical)
![Zero-Allocation](https://img.shields.io/badge/Hot%20Path-Zero%20Allocation-success)

Ultra-low latency computational core for Polymarket market making, now upgraded into a comprehensive decision-support and risk engine for prediction-market microstructure.

## Repository Layout

- `packages/crates`: Rust crate (`polymarket-kernel`)
- `packages/npm`: public npm package
- `packages/bun`: public Bun package
- `packages/python`: public PyPI package
- `docs`: additional project documentation
- `tools`: local release and utility scripts

## Overview

`polymarket-kernel` implements a unified logit-space stochastic framework where probabilities are transformed into log-odds and processed through SIMD-native math.

The crate now combines:
- high-throughput quoting primitives
- inventory-aware execution math
- vectorized analytics and portfolio risk aggregation

The runtime keeps a portable baseline path for any x86_64 CPU and enables AVX-512 acceleration only when the host actually supports it.

Source paper:
- [Toward Black-Scholes for Prediction Markets (Shaw & Dalen, 2025)](https://arxiv.org/pdf/2510.15205)

## Features

### Core Quoting Kernel
- SoA (Structure of Arrays) layout for contiguous memory access and SIMD-friendly loads
- Runtime-dispatched AVX-512 quote acceleration with portable fallback
- Inventory-aware Avellaneda-Stoikov quoting in logit space
- Exact, numerically stable `sigmoid`/`logit` mapping across the public API

### Analytics Capabilities
- Implied Belief Volatility calibration from market bid/ask quotes
- Vectorized Stress-Testing (what-if analysis) for shocked probabilities, PnL shifts, and re-quoted books
- Adaptive Kelly sizing for maker and taker clip recommendations under inventory and risk constraints
- Order Book Microstructure metrics: OBI, VWM, and pressure signal in logit space
- Cross-Market Portfolio Greeks aggregation with optional weighting and correlation matrix support

### Systems Properties
- C kernel in `packages/crates/c_src/*` with FFI-safe Rust bindings in `packages/crates/src/*`
- Portable baseline that runs on any x86_64 CPU
- Zero allocations in the hot path (pre-allocated caller-managed buffers)
- Runtime-dispatched AVX-512 fast path on supported server-class CPUs
- Numerically safe clamping for stable `logit` evaluation without saturating large logits
- Lock-free SPSC ring buffer for market data handoff

## Quick Start

Install:

```bash
cargo add polymarket-kernel
```

Call `calculate_quotes_logit` with SoA input slices:

```rust
use polymarket_kernel::calculate_quotes_logit;

fn main() {
    let x_t = vec![0.15, -0.35, 0.90, -1.20];
    let q_t = vec![10.0, -6.0, 3.0, 0.0];
    let sigma_b = vec![0.22, 0.18, 0.30, 0.15];
    let gamma = vec![0.08, 0.08, 0.08, 0.08];
    let tau = vec![0.50, 0.50, 0.50, 0.50];
    let k = vec![1.40, 1.25, 1.10, 1.80];

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

## Analytics API Example

```rust
use polymarket_kernel::{analytics, GreekOut};

fn main() {
    let n = 4usize;

    let bid_p = vec![0.49, 0.41, 0.62, 0.23];
    let ask_p = vec![0.52, 0.45, 0.66, 0.27];
    let q_t = vec![8.0, -4.0, 2.0, 0.0];
    let gamma = vec![0.08; n];
    let tau = vec![0.5; n];
    let k = vec![1.4; n];

    let mut implied_sigma = vec![0.0; n];
    analytics::implied_belief_volatility_batch(
        &bid_p,
        &ask_p,
        &q_t,
        &gamma,
        &tau,
        &k,
        &mut implied_sigma,
    );

    // `q_t` is retained for API-shape consistency, but the current calibration
    // formula depends on spread, gamma, tau, and k.
    let x_t = vec![0.20, -0.40, 0.70, -1.10];
    let shock_p = vec![0.01, -0.02, 0.03, -0.01];

    let mut out_r_x = vec![0.0; n];
    let mut out_bid = vec![0.0; n];
    let mut out_ask = vec![0.0; n];
    let mut out_greeks = vec![GreekOut::default(); n];
    let mut out_pnl = vec![0.0; n];

    analytics::simulate_shock_logit_batch(
        &x_t,
        &q_t,
        &implied_sigma,
        &gamma,
        &tau,
        &k,
        &shock_p,
        &mut out_r_x,
        &mut out_bid,
        &mut out_ask,
        &mut out_greeks,
        &mut out_pnl,
    );

    println!("implied sigma: {implied_sigma:?}");
    println!("stress pnl shift: {out_pnl:?}");
}
```

## Benchmark Snapshot

```text
============================================================
 POLYMARKET-KERNEL RAW BENCHMARK
============================================================
 Quote Batch Size        :       8192 markets
 Quote Iterations        :     100000
 Runtime-Dispatch Quote :       6.71 ns/market
------------------------------------------------------------
 SPSC Ring Capacity      :    1048576
 SPSC Messages           :   10000000
 SPSC Throughput         :      29.05 M msgs/sec
============================================================
```

## License

MIT
