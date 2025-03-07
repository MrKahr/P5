from pathlib import Path


class SetupConfig:
    """This class contains values which are globally applicable across our entire codebase"""

    # General
    app_name = "P5 Project"
    app_desc = "Student project about machine learning at AAU. Title: Estimating the Age of Wounds in Pigs Using Machine Learning - An Experimental Approach for Finding the Best Model"
    app_version = "0.0.1"
    is_release = False
    traceback_limit = 0 if is_release else None
    app_dir = Path.cwd()  # cwd must be set elsewhere. Preferably in the main '.py' file

    # Files
    config_name = "Pipeline Config"
    pipeline_config_file = "pipeline_config.json"
    grid_config_name = "Grid Parameters"
    grid_config_file = "gridparams.json"

    # Logging
    log_dir = Path(app_dir, "logs")
    log_format = "%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s"  # %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_format_color = "%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s"  # %(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s'

    # Config paths
    config_dir = Path(app_dir, "config")
    pipeline_config_path = Path(config_dir, pipeline_config_file).resolve()
    grid_config_path = Path(config_dir, grid_config_file).resolve()

    # Model summary paths
    summary_dir = Path(app_dir, "summary")
    figures_dir = Path(summary_dir, "figures")

    # Program arguments
    arg_batch = False
    arg_export = False
    arg_export_path = Path(app_dir, "configExports")
    arg_batch_config_path = arg_export_path
