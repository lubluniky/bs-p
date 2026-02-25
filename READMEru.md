# polymarket-kernel

![Rust](https://img.shields.io/badge/Rust-2024-orange?logo=rust)
![AVX-512](https://img.shields.io/badge/SIMD-AVX--512-blue)
![HFT](https://img.shields.io/badge/Use%20Case-HFT-critical)
![Zero-Allocation](https://img.shields.io/badge/Hot%20Path-Zero%20Allocation-success)

Ультра-низколатентное вычислительное ядро для маркет-мейкинга на Polymarket на базе модели Logit Jump-Diffusion и адаптации Avellaneda-Stoikov в logit-пространстве.

## Overview

`polymarket-kernel` реализует унифицированное стохастическое ядро для prediction markets: вероятности переводятся в log-odds, после чего считаются в векторизованной математике.

Крейт рассчитан на:
- HFT backtesting-движки
- live market-making боты
- inventory-aware котирование большого числа рынков в батче

Наружу отдаётся FFI-safe Rust API, под капотом работает C SIMD kernel.

Источник:
- [Toward Black-Scholes for Prediction Markets (Shaw & Dalen, 2025)](https://arxiv.org/pdf/2510.15205)

## Features

- SoA (Structure of Arrays) layout для contiguous memory access и SIMD-friendly загрузок
- AVX-512 векторизация квотинга для пакетной обработки рынков
- Кастомная AVX-512 аппроксимация `log1p`, чтобы не проваливаться в scalar fallback при расчёте спреда
- Быстрая аппроксимация sigmoid для `x -> p` в hot path
- Inventory-aware котирование по Avellaneda-Stoikov в logit-пространстве
- Ноль аллокаций в hot path (только pre-allocated буферы)
- Численно безопасный clamping для стабильных `logit`/`sigmoid`

## Quick Start

Установка:

```bash
cargo add polymarket-kernel
```

Вызов `calculate_quotes_logit` с SoA-слайсами:

```rust
use polymarket_kernel::calculate_quotes_logit;

fn main() {
    // SoA-входы для N рынков.
    let x_t = vec![0.15, -0.35, 0.90, -1.20];
    let q_t = vec![10.0, -6.0, 3.0, 0.0];
    let sigma_b = vec![0.22, 0.18, 0.30, 0.15];
    let gamma = vec![0.08, 0.08, 0.08, 0.08];
    let tau = vec![0.50, 0.50, 0.50, 0.50];
    let k = vec![1.40, 1.25, 1.10, 1.80];

    // Pre-allocated выходы (внутри kernel на hot path аллокаций нет).
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

## License

MIT
