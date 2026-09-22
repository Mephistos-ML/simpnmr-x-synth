"""Command-line entrypoint for SimpNMR-X-Synth."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from simpnmr_x_synth.app.pipelines.dataset_export import generate_dataset
from simpnmr_x_synth.app.pipelines.dataset_validation import validate_dataset_case
from simpnmr_x_synth.cfg.dataset import DatasetGenerationConfig
from simpnmr_x_synth.cli.set_logging import setup_logging

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(
        prog="simpnmr-x-synth",
        description="Generate SimpNMR-X-compatible susceptibility tensor series.",
    )
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument(
        "--version",
        action="store_true",
        help="show version",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show debug logs",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="show errors only",
    )

    dataset_parser = subparsers.add_parser(
        "dataset",
        help="generate replayable synthetic cases and a paired ML dataset",
    )
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")
    dataset_generate_parser = dataset_subparsers.add_parser(
        "generate",
        help="generate a dataset from YAML",
    )
    dataset_generate_parser.add_argument("config_file", help="YAML config path")
    dataset_generate_parser.add_argument(
        "--output",
        help="output directory (defaults to a sibling directory named after project.name)",
    )
    dataset_validate_parser = dataset_subparsers.add_parser(
        "validate",
        help="compare a completed SimpNMR-X fit with synthetic ground truth",
    )
    dataset_validate_parser.add_argument("case_dir", help="case directory")

    return parser


def main() -> int:
    """CLI entrypoint."""

    parser = build_parser()
    args = parser.parse_args()
    setup_logging(
        verbose=getattr(args, "verbose", False),
        quiet=getattr(args, "quiet", False),
    )

    if args.version:
        from simpnmr_x_synth.__version__ import __version__

        print(__version__)
        return 0

    if args.command == "dataset" and args.dataset_command == "generate":
        config_path = Path(args.config_file).resolve()
        config = DatasetGenerationConfig.from_file(config_path)
        output_dir = (
            Path(args.output).resolve()
            if args.output
            else config_path.parent / config.project.name
        )
        generated_root = generate_dataset(config=config, output_dir=output_dir)
        logger.info("Synthetic dataset written to %s", generated_root)
        return 0

    if args.command == "dataset" and args.dataset_command == "validate":
        report_file = validate_dataset_case(args.case_dir)
        logger.info("Synthetic validation report written to %s", report_file)
        return 0

    parser.print_help()
    return 0
