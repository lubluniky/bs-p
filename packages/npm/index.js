export const packageName = "holypolyfoundation-bs-p-npm";
export const version = "0.2.2";

export function healthcheck() {
  return {
    package: packageName,
    version,
    status: "ok"
  };
}
