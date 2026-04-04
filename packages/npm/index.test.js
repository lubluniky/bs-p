import test from "node:test";
import assert from "node:assert/strict";
import { healthcheck } from "./index.js";

test("healthcheck reports ok", () => {
  const result = healthcheck();
  assert.equal(result.status, "ok");
});
