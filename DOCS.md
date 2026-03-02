# Mathematical Notes: Logit Jump-Diffusion Kernel for Polymarket

## Concept

Prediction markets still lack a standard, unified pricing-and-risk framework comparable to what Black-Scholes provides in options.

This crate implements the **Logit Jump-Diffusion** approach described in:

- Shaw & Dalen (2025), *Toward Black-Scholes for Prediction Markets: A Unified Kernel and Market-Maker's Handbook*

The goal is practical and HFT-oriented: produce stable, inventory-aware bid/ask quotes in real time across many markets while exposing fast decision-support analytics.

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

## Decision Support & Analytics

The analytics layer extends the same model assumptions into calibration, scenario analysis, sizing, microstructure diagnostics, and portfolio risk aggregation.

### 1. Implied Belief Volatility Calibration

Given observed market quotes $(p^{bid}, p^{ask})$, define the observed logit spread:

$$
\Delta_x^{mkt} = \operatorname{logit}(p^{ask}) - \operatorname{logit}(p^{bid})
$$

Under the quoting approximation:

$$
\Delta_x^{model}(\sigma_b) = \gamma\tau\sigma_b^2 + \frac{2}{k}\log\left(1+\frac{\gamma}{k}\right)
$$

We recover implied $\sigma_b$ by solving:

$$
f(\sigma_b)=\Delta_x^{model}(\sigma_b)-\Delta_x^{mkt}=0
$$

using Newton-Raphson in vectorized form:

$$
\sigma_{n+1}=\max\left(0,\,\sigma_n-\frac{f(\sigma_n)}{f'(\sigma_n)}\right),\quad f'(\sigma)=2\gamma\tau\sigma
$$

Intuition: this inverts market spread into a latent belief-volatility level used for adaptive risk controls.

### 2. Vectorized Stress-Testing (What-If)

For a shock $\Delta p$, construct shocked probability and logit state:

$$
p' = \operatorname{clip}(p + \Delta p), \quad x' = \operatorname{logit}(p')
$$

Then recompute reservation and spread terms:

$$
r_x' = x' - q_t\gamma\sigma_b^2\tau
$$

$$
\delta_x' = \frac{1}{2}\gamma\sigma_b^2\tau + \frac{1}{k}\log\left(1+\frac{\gamma}{k}\right)
$$

and re-emit shocked quotes, Greeks, and inventory PnL shift:

$$
\Delta \text{PnL} \approx q_t (p' - p)
$$

Intuition: this gives an immediate SIMD what-if map of quote drift and risk under probability shocks.

### 3. Adaptive Kelly / Optimal Sizing

Define edge versus market:

$$
e = p_{user} - p_{mkt}, \quad v = p_{mkt}(1-p_{mkt})
$$

A Kelly-like sizing signal is:

$$
f^* = \frac{e}{v}
$$

The engine scales and clips by risk budget and inventory pressure:

$$
\text{clip}_{taker} = \operatorname{clamp}\left(f^* \cdot \frac{\text{risk\_limit}}{1+\gamma|q_t|},\,-\text{max\_clip},\,\text{max\_clip}\right)
$$

with final bounds from hard inventory limits; maker clips are a conservative fraction of taker clips.

Intuition: convert statistical edge into executable size while respecting inventory convexity and risk caps.

### 4. Order Book Microstructure (OBI/VWM Pressure)

Top-of-book imbalance:

$$
\operatorname{OBI} = \frac{V_b - V_a}{V_b + V_a}
$$

Volume-weighted mid proxy:

$$
\operatorname{VWM} = \frac{p^{ask}V_b + p^{bid}V_a}{V_b + V_a}
$$

Map VWM to logit and build pressure signal:

$$
\text{pressure} = \operatorname{OBI} + \frac{\operatorname{VWM} - \operatorname{mid}}{\operatorname{spread}}
$$

Intuition: combine queue imbalance and price skew into a fast directional microstructure factor.

### 5. Cross-Market Portfolio Greeks

Per-market weighted exposures:

$$
E_i^\Delta = q_i\Delta_i w_i, \quad E_i^\Gamma = q_i\Gamma_i w_i
$$

Without correlation matrix:

$$
\Delta_{net}=\sum_i E_i^\Delta, \quad \Gamma_{net}=\sum_i E_i^\Gamma
$$

With correlation matrix $C$:

$$
\Delta_{net}=\sum_i E_i^\Delta \sum_j C_{ij}E_j^\Delta
$$

$$
\Gamma_{net}=\sum_i E_i^\Gamma \sum_j C_{ij}E_j^\Gamma
$$

Intuition: aggregate cross-market exposures into a portfolio-level risk state in logit coordinates.

## HFT Implementation Details

The mathematical model is implemented for throughput-first execution:

- **SoA layout** (`x_t[]`, `q_t[]`, `sigma_b[]`, `gamma[]`, `tau[]`, `k[]`) for contiguous memory streams
- **AVX-512 SIMD** batch evaluation over 8 `f64` lanes per vector
- **Custom AVX-512 `log1p` approximation** to keep spread computation fully vectorized
- **Fast sigmoid approximation** in hot path
- **Zero hot-path allocations** (all buffers supplied by caller)

This design avoids gather penalties from AoS layouts and avoids scalar fallback in the AVX-512 quote and analytics paths.

## Numerical Guard Rails

For robust production behavior:

- Inputs are clamped where needed (`k >= \epsilon`, `gamma >= 0`, `tau >= 0`)
- Probability outputs are clamped to $(\epsilon, 1-\epsilon)$ to keep `logit` finite
- Branch-minimized math helps preserve predictable latency under load

## Practical Intuition

- Inventory risk term widens/skews quotes as position grows.
- Volatility and time-to-resolution increase risk compensation.
- Arrival-rate parameter $k$ controls how much non-linearity is needed in spread.
- Analytics layer closes the loop from market observables to actionable sizing and portfolio controls.

In short: this is a production-grade, SIMD-native quoting, decision-support, and risk engine for prediction markets.
