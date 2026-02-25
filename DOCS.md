# Mathematical Notes: Logit Jump-Diffusion Kernel for Polymarket

## Concept

Prediction markets still lack a standard, unified pricing-and-risk framework comparable to what Black-Scholes provides in options.

This crate implements the **Logit Jump-Diffusion** approach described in:

- Shaw & Dalen (2025), *Toward Black-Scholes for Prediction Markets: A Unified Kernel and Market-Maker's Handbook*

The goal is practical and HFT-oriented: produce stable, inventory-aware bid/ask quotes in real time across many markets.

## Why Logit Space

On Polymarket, prices are probabilities:

$$
p \in (0,1)
$$

Direct diffusion in probability space is inconvenient because boundaries at 0 and 1 are hard constraints.

We map probability to log-odds (logit):

$$
x = \log\left(\frac{p}{1-p}\right), \quad p = S(x)=\frac{1}{1+e^{-x}}
$$

Now the state variable lives on the full real line:

$$
x \in (-\infty, +\infty)
$$

This allows standard stochastic calculus tools (diffusion + jumps) without repeatedly hitting hard boundaries.

## Local Sensitivities in Logit Coordinates

For the logistic map $S(x)$:

$$
S'(x)=p(1-p)
$$

$$
S''(x)=p(1-p)(1-2p)
$$

Interpretation:
- $S'(x)$ controls how strongly logit moves transmit into probability moves
- $S''(x)$ captures curvature/asymmetry near extremes (very low or very high probabilities)

## Market-Making Layer (Logit Avellaneda-Stoikov)

To quote in the order book, the kernel adapts Avellaneda-Stoikov directly in logit units.

Inputs per market:
- $x_t$: current logit mid
- $q_t$: current inventory
- $\sigma_b$: belief volatility
- $\gamma$: risk aversion
- $\tau = T-t$: time to resolution
- $k$: order-arrival/liquidity parameter

### Reservation Quote

Inventory shifts the internal fair value:

$$
r_x(t)=x_t - q_t\,\gamma\,\overline{\sigma_b^2}\,(T-t)
$$

Higher long inventory pushes reservation quote down (more aggressive selling), and vice versa.

### Optimal Spread (Approximation)

Total spread in logit units:

$$
2\delta_x(t) \approx \gamma\,\overline{\sigma_b^2}\,(T-t) + \frac{2}{k}\log\left(1+\frac{\gamma}{k}\right)
$$

Half-spread used in code:

$$
\delta_x(t) \approx \frac{1}{2}\gamma\,\overline{\sigma_b^2}\,(T-t) + \frac{1}{k}\log\left(1+\frac{\gamma}{k}\right)
$$

Then:

$$
x^{bid}=r_x-\delta_x, \quad x^{ask}=r_x+\delta_x
$$

$$
p^{bid}=S(x^{bid}), \quad p^{ask}=S(x^{ask})
$$

## HFT Implementation Details

The mathematical model is implemented for throughput-first execution:

- **SoA layout** (`x_t[]`, `q_t[]`, `sigma_b[]`, `gamma[]`, `tau[]`, `k[]`) for contiguous memory streams
- **AVX-512 SIMD** batch evaluation over 8 `f64` lanes per vector
- **Custom AVX-512 `log1p` approximation** to keep spread computation fully vectorized
- **Fast sigmoid approximation** in hot path
- **Zero hot-path allocations** (all buffers supplied by caller)

This design avoids gather penalties from AoS layouts and avoids scalar fallback in the AVX-512 quote path.

## Numerical Guard Rails

For robust production behavior:

- Inputs are clamped where needed (`k >= \epsilon`, `gamma >= 0`, `tau >= 0`)
- Probability outputs are clamped to $(\epsilon, 1-\epsilon)$ to keep `logit` finite
- Branch-minimized math helps preserve predictable latency under load

## Practical Intuition

- Inventory risk term widens/skews quotes as position grows.
- Volatility and time-to-resolution increase risk compensation.
- Arrival-rate parameter $k$ controls how much non-linearity is needed in spread.
- Combined, the kernel balances fill probability vs adverse selection.

In short: this is a production-grade, SIMD-native kernel for quoting prediction markets with mathematically coherent risk control.
