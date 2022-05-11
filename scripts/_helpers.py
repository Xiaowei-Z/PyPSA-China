
# SPDX-FileCopyrightText: : 2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

import pandas as pd
from pathlib import Path

def configure_logging(snakemake, skip_handlers=False):
    """
    Configure the basic behaviour for the logging module.
    Note: Must only be called once from the __main__ section of a script.
    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.
    Parameters
    ----------
    snakemake : snakemake object
        Your snakemake object containing a snakemake.config and snakemake.log.
    skip_handlers : True | False (default)
        Do (not) skip the default handlers created for redirecting output to STDERR and file.
    """

    import logging

    kwargs = snakemake.config.get('logging', dict())
    kwargs.setdefault("level", "INFO")

    if skip_handlers is False:
        fallback_path = Path(__file__).parent.joinpath('..', 'logs', f"{snakemake.rule}.log")
        logfile = snakemake.log.get('python', snakemake.log[0] if snakemake.log
                                    else fallback_path)
        kwargs.update(
            {'handlers': [
                # Prefer the 'python' log, otherwise take the first log for each
                # Snakemake rule
                logging.FileHandler(logfile),
                logging.StreamHandler()
                ]
            })
    logging.basicConfig(**kwargs)
