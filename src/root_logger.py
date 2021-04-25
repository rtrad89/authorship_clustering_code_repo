# -*- coding: utf-8 -*-
"""
A unified logger for all the modules in the project

The Root logger is a singleton one, which is efficient and
won't instantiate the logger for each module conveniently
"""

import logging
from sys import stdout


logger = logging

# Define the logging configuration and different formats
# File handler
file_handler = logging.FileHandler(filename="app.log")
file_handler.setFormatter(
    logging.Formatter(fmt="%(asctime)s [%(levelname)s]: %(message)s",
                      datefmt="%Y.%m.%d.%a.%H:%M:%S")
    )
# Stdout handler
stdout_handler = logging.StreamHandler(stdout)
stdout_handler.setFormatter(
    logging.Formatter(fmt="%(asctime)s [%(levelname)s]: %(message)s",
                      datefmt="%a-%H:%M:%S")
    )

logger.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stdout_handler]
)
