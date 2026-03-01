# polymarket-kernel

![Rust](https://img.shields.io/badge/Rust-2024-orange?logo=rust)
![AVX-512](https://img.shields.io/badge/SIMD-AVX--512-blue)
![HFT](https://img.shields.io/badge/Use%20Case-HFT-critical)
![Zero-Allocation](https://img.shields.io/badge/Hot%20Path-Zero%20Allocation-success)

Ultra-low latency computational core for Polymarket market making, now extended with decision-support and risk-management analytics on top of the Logit Jump-Diffusion framework and an Avellaneda-Stoikov adaptation in logit space.

## Overview

`polymarket-kernel` implements a unified stochastic kernel for prediction markets where probabilities are transformed into log-odds and processed with vectorized math.

The crate now combines:
- high-throughput quoting primitives
- inventory-aware execution math
- analytics and portfolio-risk aggregation for decision support

The crate is designed for:
- HFT backtesting engines
- live market-making bots
- inventory-aware quoting across large market batches

It exposes an FFI-safe Rust API backed by a C SIMD kernel (`c_src/*`) with Rust bindings in `src/*`.

Source paper:
- [Toward Black-Scholes for Prediction Markets (Shaw & Dalen, 2025)](https://arxiv.org/pdf/2510.15205)

## Features

- SoA (Structure of Arrays) layout for contiguous memory access and SIMD-friendly loads
- AVX-512 vectorized quote engine for batch processing
- Custom AVX-512 `log1p` approximation to avoid scalar fallback in spread computation
- Fast sigmoid approximation for `x -> p` mapping in hot paths
- Inventory-aware Avellaneda-Stoikov quoting in logit space
- Implied Belief Volatility calibration from market bid/ask quotes
- Vectorized logit shock simulation for instant what-if PnL and Greek shifts
- Adaptive Kelly maker/taker clip sizing under inventory and risk limits
- Top-of-book microstructure analytics (OBI, VWM, pressure signal in logit space)
- Cross-market portfolio Greek aggregation with optional correlation matrix
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

## Analytics API Example

```rust
use polymarket_kernel::{analytics, GreekOut};

fn main() {
    let n = 4usize;

    let x_t = vec![0.2, -0.4, 0.7, -1.1];
    let q_t = vec![10.0, -6.0, 3.0, 0.0];
    let sigma_b = vec![0.18, 0.22, 0.27, 0.15];
    let gamma = vec![0.08; n];
    let tau = vec![0.5; n];
    let k = vec![1.4; n];

    let bid = vec![0.49, 0.41, 0.62, 0.23];
    let ask = vec![0.52, 0.45, 0.66, 0.27];
    let shocks = vec![0.01, -0.02, 0.03, -0.01];

    let mut implied_vol = vec![0.0; n];
    analytics::implied_belief_volatility_batch(
        &bid,
        &ask,
        &q_t,
        &gamma,
        &tau,
        &k,
        &mut implied_vol,
    );

    let mut out_r_x = vec![0.0; n];
    let mut out_bid = vec![0.0; n];
    let mut out_ask = vec![0.0; n];
    let mut out_greeks = vec![GreekOut::default(); n];
    let mut out_pnl = vec![0.0; n];

    analytics::simulate_shock_logit_batch(
        &x_t,
        &q_t,
        &sigma_b,
        &gamma,
        &tau,
        &k,
        &shocks,
        &mut out_r_x,
        &mut out_bid,
        &mut out_ask,
        &mut out_greeks,
        &mut out_pnl,
    );

    let delta_x: Vec<f64> = out_greeks.iter().map(|g| g.delta_x).collect();
    let gamma_x: Vec<f64> = out_greeks.iter().map(|g| g.gamma_x).collect();

    let (net_delta, net_gamma) =
        analytics::aggregate_portfolio_greeks(&q_t, &delta_x, &gamma_x, None, None);

    println!("implied_vol={implied_vol:?}");
    println!("net_delta={net_delta:.6}, net_gamma={net_gamma:.6}");
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
