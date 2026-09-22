"""Configure CLI logging behavior."""

from __future__ import annotations

import logging
import os
import sys


class ColorFormatter(logging.Formatter):
    """Minimal colour formatter for console output."""

    def __init__(self, fmt: str) -> None:
        super().__init__(fmt)

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        level = record.levelname
        if level == "DEBUG":
            return f"\033[90m{msg}\033[0m"
        if level == "INFO":
            return f"\033[36m{msg}\033[0m"
        if level == "WARNING":
            return f"\033[33m{msg}\033[0m"
        if level in {"ERROR", "CRITICAL"}:
            return f"\033[31m{msg}\033[0m"
        return msg


class PathShortener(logging.Filter):
    """Shorten filesystem paths in log arguments for readability."""

    def __init__(self, base_dir: str | None = None) -> None:
        super().__init__()
        self.base_dir = os.fspath(base_dir) if base_dir else os.getcwd()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.INFO:
            return True

        args = record.args
        if not args:
            return True

        if not isinstance(args, tuple):
            args = (args,)

        new_args: list[object] = []
        for arg in args:
            if isinstance(arg, str):
                if os.path.isabs(arg) or (os.sep in arg) or ("/" in arg):
                    arg = os.path.basename(arg.rstrip("/"))
            new_args.append(arg)

        record.args = tuple(new_args)
        return True


def setup_logging(
    verbose: bool = False,
    quiet: bool = False,
    base_dir: str | None = None,
) -> None:
    """Configure root logger for the CLI."""

    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    if quiet:
        level = logging.ERROR

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    formatter = ColorFormatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    handler.addFilter(PathShortener(base_dir=base_dir))
    root.addHandler(handler)
