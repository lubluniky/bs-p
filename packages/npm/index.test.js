import test from "node:test";
import assert from "node:assert/strict";
import { calculateQuotesLogit, logit, sigmoid } from "./index.js";

test("sigmoid/logit are consistent", () => {
  const p = sigmoid(0);
  assert.ok(Math.abs(p - 0.5) < 1e-12);
  assert.ok(Math.abs(logit(0.5)) < 1e-12);
});

test("calculateQuotesLogit returns bid/ask vectors", () => {
  const x_t = new Float64Array([0.15, -0.35, 0.9, -1.2]);
  const q_t = new Float64Array([10, -6, 3, 0]);
  const sigma_b = new Float64Array([0.22, 0.18, 0.30, 0.15]);
  const gamma = new Float64Array([0.08, 0.08, 0.08, 0.08]);
  const tau = new Float64Array([0.5, 0.5, 0.5, 0.5]);
  const k = new Float64Array([1.4, 1.25, 1.1, 1.8]);

  const { bid_p, ask_p } = calculateQuotesLogit(x_t, q_t, sigma_b, gamma, tau, k);

  assert.equal(bid_p.length, x_t.length);
  assert.equal(ask_p.length, x_t.length);
  for (let i = 0; i < x_t.length; i += 1) {
    assert.ok(Number.isFinite(bid_p[i]));
    assert.ok(Number.isFinite(ask_p[i]));
    assert.ok(ask_p[i] >= bid_p[i]);
  }
});
