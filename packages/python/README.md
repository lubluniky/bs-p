# bs-p

Python bindings for the `bs-p` C kernel.

## API

- `sigmoid(x: float) -> float`
- `logit(p: float) -> float`
- `calculate_quotes_logit(x_t, q_t, sigma_b, gamma, tau, k) -> dict`
