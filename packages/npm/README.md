# holypolyfoundation-bs-p-npm

Node.js bindings for the `bs-p` C kernel.

## API

- `sigmoid(x: number): number`
- `logit(p: number): number`
- `calculateQuotesLogit(x_t, q_t, sigma_b, gamma, tau, k): { bid_p: Float64Array, ask_p: Float64Array }`
