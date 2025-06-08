#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Logging in QUEENS."""

import functools
import inspect
import logging
import sys

from dask.distributed import get_worker

from queens.utils.config_directories import logging_directory_on_dask_worker
from queens.utils.printing import get_str_table

LIBRARY_LOGGER_NAME = "queens"


class LogFilter(logging.Filter):
    """Filters (lets through) all messages with level <= LEVEL.

    Attributes:
        level: Logging level
    """

    def __init__(self, level):
        """Initiatlize the logging filter.

        Args:
            level (int): Logging level
        """
        super().__init__()
        self.level = level

    def filter(self, record):
        """Filter the logging record.

        Args:
            record (LogRecord obj): Logging record object

        Returns:
            LogRecord obj: Filter logging record
        """
        return record.levelno <= self.level


class NewLineFormatter(logging.Formatter):
    """Formatter splitting multiline messages into single line messages.

    A logged message that consists of more than one line - contains a new line char - is split
    into multiple single line messages that all have the same format. Without this the overall
     format of the logging is broken for multiline messages.
    """

    def format(self, record):
        """Override format function.

        Args:
            record (LogRecord obj): Logging record object
        Returns:
            formatted_message (str): Logged message in supplied format split into single lines
        """
        formatted_message = super().format(record)

        if record.message != "":
            parts = formatted_message.split(record.message)
            formatted_message = formatted_message.replace("\n", "\n" + parts[0])

        return formatted_message


def setup_logger(logger=None, debug=False):
    """Set up the main QUEENS logger.

    Args:
        logger (logging.Logger): Logger instance that should be set up
        debug (bool): Indicates debug mode and controls level of logging

    Returns:
        logging.logger: QUEENS logger object
    """
    if logger is None:
        logger = logging.getLogger(LIBRARY_LOGGER_NAME)

        # The default logging level is INFO (for QUEENS)
        # If the parent logger uses a lower level (e.g. pytest) that level is set
        parent_level = logger.parent.getEffectiveLevel()
        logger.setLevel(min(parent_level, logging.INFO))

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        # deactivate logging for specific modules
        logging.getLogger("arviz").setLevel(logging.CRITICAL)
        logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
        logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
        logging.getLogger("numba").setLevel(logging.CRITICAL)

    return logger


def setup_stream_handler(logger):
    """Set up a stream handler.

    Args:
        logger (logging.logger): Logger object to add the stream handler to
    """
    # a plain, minimal formatter for streamhandlers
    stream_formatter = NewLineFormatter("%(message)s")

    # set up logging to stdout
    console_stdout = logging.StreamHandler(stream=sys.stdout)
    # messages lower than and including WARNING go to stdout
    log_filter = LogFilter(logging.WARNING)
    console_stdout.addFilter(log_filter)
    console_stdout.setLevel(logger.level)
    console_stdout.setFormatter(stream_formatter)
    logger.addHandler(console_stdout)

    # set up logging to stderr
    console_stderr = logging.StreamHandler(stream=sys.stderr)
    # messages >= ERROR or messages >= CONSOLE_LEVEL_MIN if CONSOLE_LEVEL_MIN > ERROR go to stderr
    console_stderr.setLevel(max(logger.level, logging.ERROR))
    console_stderr.setFormatter(stream_formatter)

    logger.addHandler(console_stderr)


def setup_file_handler(logger, log_file_path):
    """Set up a file handler.

    Args:
        logger (logging.logger): Logger object to add the stream handler to
        log_file_path (pathlib.Path): Path of the logging file
    """
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_formatter = NewLineFormatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logger.level)
    logger.addHandler(file_handler)


def setup_basic_logging(log_file_path, logger=None, debug=False):
    """Setup basic logging.

    Args:
        log_file_path (Path): Path to the log-file
        logger (logging.Logger): Logger instance that should be set up
        debug (bool): Indicates debug mode and controls level of logging
    """
    logger = setup_logger(logger, debug)
    setup_stream_handler(logger)
    setup_file_handler(logger, log_file_path)


def setup_cli_logging(debug=False):
    """Set up logging for CLI utils.

    Args:
        debug (bool): Indicates debug mode and controls level of logging
    """
    library_logger = setup_logger(debug=debug)
    setup_stream_handler(library_logger)


def setup_cluster_logging():
    """Setup cluster logging."""
    level_min = logging.INFO

    logging.basicConfig(
        level=level_min,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
    )

    console_stdout = logging.StreamHandler(stream=sys.stdout)
    console_stderr = logging.StreamHandler(stream=sys.stderr)

    # messages lower than and including WARNING go to stdout
    log_filter = LogFilter(logging.WARNING)
    console_stdout.addFilter(log_filter)
    console_stdout.setLevel(level_min)

    # messages >= ERROR or messages >= CONSOLE_LEVEL_MIN if CONSOLE_LEVEL_MIN > ERROR go to stderr
    console_stderr.setLevel(max(level_min, logging.ERROR))

    # set a format which is simpler for console use
    formatter = NewLineFormatter("%(name)-12s: %(levelname)-8s %(message)s")
    console_stdout.setFormatter(formatter)
    console_stderr.setFormatter(formatter)

    # add the handlers to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_stdout)
    root_logger.addHandler(console_stderr)


def reset_logging():
    """Reset loggers.

    This is only needed during testing, as otherwise the loggers are not
    destroyed resulting in the same output multiple time. This is taken
    from:

    https://stackoverflow.com/a/56810619
    """
    manager = logging.root.manager
    manager.disabled = logging.NOTSET
    for logger in manager.loggerDict.values():
        if isinstance(logger, logging.Logger) and LIBRARY_LOGGER_NAME in str(logger):
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            logger.disabled = False
            logger.filters.clear()
            handlers = logger.handlers.copy()
            for handler in handlers:
                # Copied from `logging.shutdown`.
                try:
                    handler.acquire()
                    handler.flush()
                    handler.close()
                except (OSError, ValueError):
                    pass
                finally:
                    handler.release()
                logger.removeHandler(handler)


def log_init_args(method):
    """Log arguments of __init__ method.

    Args:
        method (obj): __init__ method
    Returns:
        wrapper (func): Decorated __init__ method
    """

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(method)
        default_kwargs = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        all_keys = list(signature.parameters.keys())
        args_as_kwargs = {all_keys[i]: args[i] for i in range(len(args))}
        all_kwargs = dict(default_kwargs, **args_as_kwargs, **kwargs)

        def key_fun(pair):
            if pair[0] in all_keys:
                return all_keys.index(pair[0])
            return len(all_keys)

        all_kwargs = dict(sorted(all_kwargs.items(), key=key_fun))

        _logger = logging.getLogger(args[0].__module__)
        _logger.info(get_str_table(args[0].__class__.__name__, all_kwargs, use_repr=True))
        method(*args, **kwargs)

    return wrapper


def setup_logger_on_dask_worker(name="worker", experiment_dir=None, level=logging.INFO):
    """Setup a logger on a dask worker.

    Args:
        name (str): Name of the logger.
        experiment_dir (Path): Path to the experiment directory.
        level (int): Logging level.

    Returns:
        logger (logging.Logger): Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger  # Already configured (avoid duplicate handlers)

    logger.setLevel(level)
    formatter = NewLineFormatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M"
    )

    if experiment_dir is not None:
        log_dir = logging_directory_on_dask_worker(experiment_dir)

        try:
            worker = get_worker()
            log_file = log_dir / f"{worker.name}.log"
        except ValueError:
            # Not inside a Dask worker — maybe running locally
            log_file = log_dir / f"{name}_client.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
