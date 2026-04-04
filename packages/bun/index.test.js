import { describe, expect, test } from "bun:test";
import { healthcheck } from "./index.js";

describe("healthcheck", () => {
  test("returns ok", () => {
    expect(healthcheck().status).toBe("ok");
  });
});
