from pathlib import Path


class SetupConfig:
    # General
    app_name = "P5"
    is_release = False
    traceback_limit = 0 if is_release else None
    app_dir = Path.cwd()  # cwd must be set elsewhere. Preferably in the main '.py' file

    # Files
    config_name = "Pipeline Config"
    config_file = "pipeline_config.toml"

    # Logging
    log_dir = Path(app_dir, "logs")
    log_format = "%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s"  # %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_format_color = "%(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s"  # %(asctime)s - %(module)s - %(lineno)s - %(levelname)s - %(message)s'

    # Config paths
    config_dir = Path(app_dir, "config")
    app_config_path = Path(config_dir, config_file).resolve()
