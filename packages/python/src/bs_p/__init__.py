"""Python companion package for bs-p releases."""

__all__ = ["healthcheck", "__version__"]

__version__ = "0.2.2"


def healthcheck() -> dict[str, str]:
    return {
        "package": "bs-p",
        "version": __version__,
        "status": "ok",
    }
