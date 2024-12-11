from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys

from modules.config.utils.config_exporter import ConfigExporter
from modules.config.utils.setup_config import SetupConfig
from modules.logging import logger
from modules.modelSummary.import_summary import SummaryImporter


class AppArguments:
    _logger = logger

    def __init__(self) -> None:
        """The command line argument parser for the program"""
        self._args = self._verifyArguments(self._createArgParser().parse_args())

    def _createArgParser(self) -> ArgumentParser:
        """
        Create an argument parser with arguments.

        Returns
        -------
        ArgumentParser
            The argument parser.
        """
        parser = ArgumentParser(
            prog=SetupConfig.app_name, description=SetupConfig.app_desc
        )
        parser.add_argument(
            "--version",
            action="version",
            version=f"{'%(prog)s'} v{SetupConfig.app_version}",
        )
        parser.add_argument(
            "--batch",
            "-b",
            action="store_true",
            help="enable batch model training and evaluation using multiple configs",
        )
        parser.add_argument(
            "--batch_config_path",
            "-c",
            default=SetupConfig.arg_export_path,
            help="location of the config folder for batch processing (default: '%(default)s')",
        )
        parser.add_argument(
            "--export",
            "-x",
            help="export the currently active configs and exit. Supply a name as argument",
        )
        parser.add_argument(
            "--export_path",
            "-xp",
            default=SetupConfig.arg_export_path,
            help="location of the folder for exported configs (default: '%(default)s')",
        )
        parser.add_argument(
            "--plot_result",
            "-p",
            help="plot summary results from multiple models. Argument the to search for in summary files (default: '%(default)s')",
        )
        parser.add_argument(
            "--summary_path",
            "-s",
            default=SetupConfig.summary_dir,
            help="location of summary folder to search (default: '%(default)s')",
        )
        return parser

    def _verifyArguments(self, args: Namespace) -> Namespace:
        """
        Verifies `args` to ensure their correctness.

        Parameters
        ----------
        args : Namespace
            Arguments parsed from `sys.argv`.

        Returns
        -------
        Namespace
            Verified arguments.
        """
        config_path = Path(args.batch_config_path)
        if config_path != SetupConfig.arg_batch_config_path:
            # Save argument
            SetupConfig.arg_batch_config_path = config_path.resolve()

        export_path = Path(args.export_path)
        if export_path != SetupConfig.arg_export_path:
            # Save argument
            SetupConfig.arg_export_path = export_path

        load_summary_path = Path(args.summary_path)
        if not (load_summary_path.exists() or load_summary_path.resolve().exists()):
            raise RuntimeError(f"Directory {load_summary_path} does not exists")

        SetupConfig.arg_batch = args.batch
        SetupConfig.arg_export = args.export
        return args

    def executeArguments(self) -> None:
        args = self._args
        if args.export:
            ConfigExporter().exportConfigs(args.export)
            sys.exit()
        if args.plot_result:
            SummaryImporter.plotSummaries(args.summary_path, [args.plot_result])
            sys.exit()

    def getArguments(self) -> Namespace:
        return self._args
