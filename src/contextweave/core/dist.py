from importlib.metadata import PackageNotFoundError, version

DIST_NAME = "contextweave-mcp"
UNKNOWN_VERSION = "0.0.0"


def package_version() -> str:
    """The installed distribution's version, or 0.0.0 when running from an uninstalled tree."""
    try:
        return version(DIST_NAME)
    except PackageNotFoundError:
        return UNKNOWN_VERSION
