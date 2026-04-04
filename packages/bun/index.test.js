import { describe, expect, test } from "bun:test";
import { calculateQuotesLogit, logit, sigmoid } from "./index.js";

describe("bs-p bun bindings", () => {
  test("sigmoid/logit sanity", () => {
    expect(Math.abs(sigmoid(0) - 0.5)).toBeLessThan(1e-12);
    expect(Math.abs(logit(0.5))).toBeLessThan(1e-12);
  });

  test("calculateQuotesLogit returns valid arrays", () => {
    const x_t = new Float64Array([0.15, -0.35, 0.9, -1.2]);
    const q_t = new Float64Array([10, -6, 3, 0]);
    const sigma_b = new Float64Array([0.22, 0.18, 0.30, 0.15]);
    const gamma = new Float64Array([0.08, 0.08, 0.08, 0.08]);
    const tau = new Float64Array([0.5, 0.5, 0.5, 0.5]);
    const k = new Float64Array([1.4, 1.25, 1.1, 1.8]);

    const { bid_p, ask_p } = calculateQuotesLogit(x_t, q_t, sigma_b, gamma, tau, k);

    expect(bid_p.length).toBe(x_t.length);
    expect(ask_p.length).toBe(x_t.length);
    for (let i = 0; i < x_t.length; i += 1) {
      expect(Number.isFinite(bid_p[i])).toBe(true);
      expect(Number.isFinite(ask_p[i])).toBe(true);
      expect(ask_p[i] >= bid_p[i]).toBe(true);
    }
  });
});
