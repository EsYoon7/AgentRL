from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('agentrl-worker')
except PackageNotFoundError:
    # package is not installed
    pass
