# polymarket-kernel

![Rust](https://img.shields.io/badge/Rust-2024-orange?logo=rust)
![AVX-512](https://img.shields.io/badge/SIMD-AVX--512-blue)
![HFT](https://img.shields.io/badge/Use%20Case-HFT-critical)
![Zero-Allocation](https://img.shields.io/badge/Hot%20Path-Zero%20Allocation-success)

Ультра-низколатентное вычислительное ядро для маркет-мейкинга на Polymarket, расширенное до полноценного decision-support и risk-management движка для prediction markets.

## Overview

`polymarket-kernel` реализует унифицированный стохастический фреймворк в logit-пространстве: вероятности переводятся в log-odds и считаются SIMD-оптимизированной математикой.

Теперь крейт объединяет:
- высокопроизводительное котирование
- inventory-aware execution math
- векторизованную аналитику и портфельную агрегацию рисков

Во время выполнения всегда доступен portable baseline path для любого x86_64 CPU, а AVX-512 включается только если хост действительно поддерживает эту инструкцию.

Источник:
- [Toward Black-Scholes for Prediction Markets (Shaw & Dalen, 2025)](https://arxiv.org/pdf/2510.15205)

## Features

### Core Quoting Kernel
- SoA (Structure of Arrays) layout для contiguous memory access и SIMD-friendly загрузок
- Runtime-dispatched AVX-512 ускорение котирования с portable fallback
- Inventory-aware котирование по Avellaneda-Stoikov в logit-пространстве
- Точная и численно устойчивая связка `sigmoid`/`logit` во всём публичном API

### Analytics Capabilities
- Калибровка Implied Belief Volatility из рыночных bid/ask котировок
- Vectorized Stress-Testing (what-if) с пересчётом котировок, PnL и греков под шоками вероятностей
- Adaptive Kelly sizing для рекомендаций maker/taker clip с учётом инвентаря и риск-лимитов
- Метрики микроструктуры стакана: OBI, VWM и pressure signal в logit-пространстве
- Cross-Market Portfolio Greeks агрегация с опциональными весами и матрицей корреляций

### Systems Properties
- C-ядро в `packages/crates/c_src/*` и FFI-safe Rust bindings в `packages/crates/src/*`
- Portable baseline, который запускается на любом x86_64 CPU
- Ноль аллокаций в hot path (буферы задаются вызывающей стороной)
- Runtime-dispatched AVX-512 fast path на поддерживаемых серверных CPU
- Численно безопасный clamp для стабильного `logit` без saturation больших logits
- Lock-free SPSC ring buffer для передачи рыночных апдейтов

## Quick Start

Установка:

```bash
cargo add polymarket-kernel
```

Вызов `calculate_quotes_logit` с SoA-слайсами:

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

    // `q_t` сохранён для единообразия API, хотя текущая формула калибровки
    // зависит только от spread, gamma, tau и k.
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
