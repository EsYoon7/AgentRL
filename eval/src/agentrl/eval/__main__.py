import asyncio
from argparse import ArgumentParser

from pydantic_settings import CliApp, CliSettingsSource

from ._version import __version__
from .cli import Settings


def main():
    try:
        parser = ArgumentParser()
        parser.add_argument('--version', action='version', version=f'agentrl-eval {__version__}')

        cli_settings = CliSettingsSource(Settings, root_parser=parser)
        CliApp.run(Settings, cli_settings_source=cli_settings)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass  # ignore exceptions caused by graceful exit


if __name__ == '__main__':
    main()
