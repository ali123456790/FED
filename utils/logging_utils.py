# This file provides logging configuration utilities for the FIDS project.
import logging


def setup_logging(config):
    """Set up logging based on the provided configuration."""
    level = config.get("level", "INFO").upper()
    logfile = config.get("file", None)
    handlers = []

    # Console handler if requested
    if config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        handlers.append(console_handler)

    # File handler if logfile is provided
    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        handlers.append(file_handler)

    logging.basicConfig(level=getattr(logging, level), handlers=handlers)
    logging.info("Logging is set up with level {}".format(level)) 