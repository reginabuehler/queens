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
"""QUEENS driver module base class."""

import abc
import logging
from pathlib import Path
from typing import final

from queens.utils.logger_settings import setup_logger_on_dask_worker

_logger = logging.getLogger(__name__)


class Driver(metaclass=abc.ABCMeta):
    """Abstract base class for drivers in QUEENS.

    Attributes:
        parameters (Parameters): Parameters object
        files_to_copy (list): files or directories to copy to experiment_dir
        logger_on_dask_worker (logging.Logger): Logger instance used on the dask worker
    """

    def __init__(self, parameters, files_to_copy=None):
        """Initialize Driver object.

        Args:
            parameters (Parameters): Parameters object
            files_to_copy (list): files or directories to copy to experiment_dir
        """
        self.parameters = parameters
        if files_to_copy is None:
            files_to_copy = []
        if not isinstance(files_to_copy, list):
            raise TypeError("files_to_copy must be a list")
        for file_to_copy in files_to_copy:
            if not isinstance(file_to_copy, (str, Path)):
                raise TypeError("files_to_copy must be a list of strings or Path objects")
        self.files_to_copy = files_to_copy

        self.logger_on_dask_worker = None

    @final
    def run(self, sample, job_id, num_procs, experiment_dir, experiment_name):
        """Run driver.

        Args:
            sample (dict): Dict containing sample
            job_id (int): Job ID
            num_procs (int): number of processors
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.

        Returns:
            Result and potentially the gradient
        """
        if self.logger_on_dask_worker is None:
            self.logger_on_dask_worker = setup_logger_on_dask_worker(
                name=type(self).__name__, experiment_dir=experiment_dir, level=logging.INFO
            )
        self.logger_on_dask_worker.info("Running job %i", job_id)

        return self._run(sample, job_id, num_procs, experiment_dir, experiment_name)

    @abc.abstractmethod
    def _run(self, sample, job_id, num_procs, experiment_dir, experiment_name):
        """Abstract method for driver run.

        Args:
            sample (dict): Dict containing sample
            job_id (int): Job ID
            num_procs (int): number of processors
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.

        Returns:
            Result and potentially the gradient
        """
