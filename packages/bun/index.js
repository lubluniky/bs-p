import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const native = require("./build/Release/bs_p_core.node");

const toF64 = (v) => (v instanceof Float64Array ? v : Float64Array.from(v));

export const sigmoid = native.sigmoid;
export const logit = native.logit;

export function calculateQuotesLogit(x_t, q_t, sigma_b, gamma, tau, k) {
  return native.calculate_quotes_logit(
    toF64(x_t),
    toF64(q_t),
    toF64(sigma_b),
    toF64(gamma),
    toF64(tau),
    toF64(k),
  );
}
