from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('agentrl-trainer')
except PackageNotFoundError:
    # package is not installed
    pass
