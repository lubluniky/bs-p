import argparse
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def quote_spread_probability(
    sigma_b: np.ndarray,
    mid_p: float = 0.55,
    gamma: float = 0.08,
    tau: float = 0.5,
    k: float = 1.4,
) -> np.ndarray:
    mid_x = logit(np.array([mid_p]))[0]
    risk_term = gamma * sigma_b**2 * tau
    half_spread_x = 0.5 * risk_term + np.log1p(gamma / k) / k
    bid = sigmoid(mid_x - half_spread_x)
    ask = sigmoid(mid_x + half_spread_x)
    return ask - bid


def stress_test_pnl(
    shocks: np.ndarray,
    base_p: float,
    inventory: float,
) -> np.ndarray:
    shocked = np.clip(base_p + shocks, 1e-12, 1.0 - 1e-12)
    return inventory * (shocked - base_p)


def adaptive_kelly_sizes(
    edges: np.ndarray,
    market_p: float,
    q_t: float,
    gamma: float,
    risk_limit: float,
    max_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    variance = max(1e-12, market_p * (1.0 - market_p))
    kelly_fraction = edges / variance
    inventory_scale = 1.0 / (1.0 + gamma * abs(q_t))

    taker = np.clip(kelly_fraction * risk_limit * inventory_scale, -max_clip, max_clip)
    long_limit = risk_limit - q_t
    short_limit = -risk_limit - q_t
    taker = np.clip(taker, short_limit, long_limit)
    maker = np.clip(0.5 * taker, short_limit, long_limit)
    return maker, taker


def build_plots(output_path: str | None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sigma_grid = np.linspace(0.05, 0.65, 300)
    spread = quote_spread_probability(sigma_grid)
    axes[0].plot(sigma_grid, spread, color="#1f77b4", lw=2.2)
    axes[0].set_title("Implied Volatility vs Spread Width")
    axes[0].set_xlabel("Belief Volatility ($\\sigma_b$)")
    axes[0].set_ylabel("Quoted Spread (ask - bid)")
    axes[0].grid(alpha=0.25)

    shocks = np.linspace(-0.20, 0.20, 400)
    for inventory, color in [(200.0, "#2ca02c"), (800.0, "#ff7f0e"), (1500.0, "#d62728")]:
        pnl = stress_test_pnl(shocks, base_p=0.53, inventory=inventory)
        axes[1].plot(shocks, pnl, lw=2.0, color=color, label=f"q_t={int(inventory)}")
    axes[1].axhline(0.0, color="black", lw=1.0, alpha=0.5)
    axes[1].set_title("Stress-Test PnL Under Probability Shocks")
    axes[1].set_xlabel("Probability Shock")
    axes[1].set_ylabel("PnL Shift")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    edges = np.linspace(-0.12, 0.12, 400)
    maker, taker = adaptive_kelly_sizes(
        edges,
        market_p=0.52,
        q_t=140.0,
        gamma=0.09,
        risk_limit=1000.0,
        max_clip=220.0,
    )
    axes[2].plot(edges, maker, lw=2.0, label="Maker Clip", color="#9467bd")
    axes[2].plot(edges, taker, lw=2.0, label="Taker Clip", color="#17becf")
    axes[2].axhline(0.0, color="black", lw=1.0, alpha=0.5)
    axes[2].set_title("Adaptive Kelly Sizing vs Edge")
    axes[2].set_xlabel("Edge (belief p - market p)")
    axes[2].set_ylabel("Recommended Order Size")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.25)

    fig.suptitle("Polymarket Kernel Analytics Validation (Mocked)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        fig.savefig(output_path, dpi=160)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock visual validation for analytics module outputs")
    parser.add_argument("--save", type=str, default=None, help="Optional output path for the rendered figure")
    args = parser.parse_args()
    build_plots(args.save)


if __name__ == "__main__":
    main()
