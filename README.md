<h1 align="center">
    Estimating the Age of Wounds in Pigs Using Machine Learning: An Experimental Approach for Finding the Best Model
</h1>
<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


## Introduction
This is the program we have used to run experiments for our project.
It uses a config file (or several) to know which model to use and which preprocessing methods to employ. Experiment summaries are delivered in JSON-format, and plots as PNGs.
Some extra code is present and used to generate figures for the project report. It is currently not functional.
The dataset is NOT featured in this version, so the program cannot be run.
The code included here is for review only.

## Using the CLI
The program features a command-line interface for running multiple pipeline configurations in sequence as well as creating score function plots for these pipeline configurations.

Argument | Type | Default | Description
:---: | :---: | :---: | :---
`--batch` / `-b` | flag | N/A | Enable batch processing of pipeline configurations
`--batch_config_path` / `-c` | str | ".\configExports" | Directory of pipeline configurations used during batch processing
`--export` / `-x` | str | N/A | Export the configs from ".\config" in a JSON format compatible with batch processing. Supply a filename as argument, e.g., "7.1_Search-DT"
`--export_path` / `-xp` | str | ".\configExports" | The directory for exported configs.
`--plot_result` / `-p` | str | N/A | Create a score function plot of multiple summaries for comparison of model performance across experiments. Supply a key in the summary dict as argument, e.g., "test_accuracies"
`--summary_path` / `-s` | str | ".\summary" | The directory of configuration summaries created for each pipeline run.


## Install from source
```shell
pip install -r requirements.txt
```

### Requirements
- Python 3.12.7