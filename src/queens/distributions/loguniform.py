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
"""LogUniform Distribution."""

import numpy as np
import scipy.linalg
import scipy.stats

from queens.distributions._distribution import Continuous
from queens.distributions.uniform import Uniform
from queens.utils.logger_settings import log_init_args
from tensorflow.python.ops.gen_array_ops import lower_bound


class LogUniform(Continuous):
    """LogUniform distribution.

    Support in [uniform_distribution.lower_bound, uniform_distribution.upper_bound].

    Attributes:
        uniform_distribution (Uniform): Underlying uniform distribution.
    """

    @log_init_args
    def __init__(self, lower_bound, upper_bound):
        """Initialize loguniform distribution.

        Args:
            lower_bound (array_like): Lower bound(s) of the uniform distribution
            upper_bound (array_like): Upper bound(s) of the uniform distribution
        """

        # ToDo check multivariate
        lower_bound = np.array(lower_bound).reshape(-1)
        upper_bound = np.array(upper_bound).reshape(-1)

        # sanity check
        if (lower_bound <= 0).any():
            raise ValueError(
                f"Provided lower bound is not positive."
                f"Provided lower bound: {lower_bound}"
            )

        self.uniform_distribution = Uniform(lower_bound, upper_bound)

        mean = (upper_bound - lower_bound) / np.log(upper_bound / lower_bound)
        coveriance_first_term = (upper_bound ** 2 - lower_bound ** 2) / ( 2 * np.log(upper_bound / lower_bound))
        covariance_second_term = ((upper_bound - lower_bound) / np.log(upper_bound / lower_bound)) ** 2
        covariance = coveriance_first_term - covariance_second_term

        super().__init__(
            mean=mean, covariance=covariance, dimension=self.uniform_distribution.dimension
        )

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated

        Returns:
            cdf (np.ndarray): cdf at evaluated positions
        """
        # ToDo multivariate
        # return (np.log(x) - np.log(lower_bound)) / (np.log(upper_bound) - np.log(lower_bound))

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws

        Returns:
            samples (np.ndarray): Drawn samples from the distribution
        """
        # ToDo multivariate
        a = self.uniform_distribution.lower_bound
        b = self.uniform_distribution.upper_bound
        normalized_samples = (self.uniform_distribution.draw(num_draws=num_draws) - a) / (b - a)
        return a * np.exp(normalized_samples * np.log(b / a))

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated

        Returns:
            logpdf (np.ndarray): pdf at evaluated positions
        """
        # log_x = np.log(x).reshape(-1, self.dimension)
        # dist = log_x - self.normal_distribution.mean
        # logpdf = (
        #     self.normal_distribution.logpdf_const
        #     - np.sum(log_x, axis=1)
        #     - 0.5 * (np.dot(dist, self.normal_distribution.precision) * dist).sum(axis=1)
        # )
        # return logpdf

    def grad_logpdf(self, x):
        """Gradient of the log pdf with respect to *x*.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated

        Returns:
            grad_logpdf (np.ndarray): Gradient of the log pdf evaluated at positions
        """
        # x = x.reshape(-1, self.dimension)
        # x[x == 0] = np.nan
        # grad_logpdf = (
        #     1
        #     / x
        #     * (
        #         np.dot(
        #             self.normal_distribution.mean.reshape(1, -1) - np.log(x),
        #             self.normal_distribution.precision,
        #         )
        #         - 1
        #     )
        # )
        # return grad_logpdf

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated

        Returns:
            pdf (np.ndarray): pdf at evaluated positions
        """
        # return np.exp(self.logpdf(x))

    def ppf(self, quantiles):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            quantiles (np.ndarray): Quantiles at which the ppf is evaluated

        Returns:
            ppf (np.ndarray): Positions which correspond to given quantiles
        """

        self.check_1d()
        ppf = scipy.stats.reciprocal.ppf(
            quantiles,
            a=self.uniform_distribution.lower_bound,
            b=self.uniform_distribution.upper_bound,
        ).reshape(-1)
        return ppf
