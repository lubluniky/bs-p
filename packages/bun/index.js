export const packageName = "holypolyfoundation-bs-p-bun";
export const version = "0.2.2";

export function healthcheck() {
  return {
    package: packageName,
    version,
    runtime: "bun",
    status: "ok"
  };
}
