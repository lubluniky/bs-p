"""Python bindings for the bs-p C kernel."""

from ._core import calculate_quotes_logit, logit, sigmoid

__all__ = ["calculate_quotes_logit", "sigmoid", "logit", "healthcheck", "__version__"]

__version__ = "0.2.2"


def healthcheck() -> dict[str, str]:
    return {
        "package": "bs-poly",
        "version": __version__,
        "status": "ok",
    }
