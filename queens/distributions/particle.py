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
"""Discrete particle distribution."""

import itertools
import logging

import numpy as np

from queens.distributions._distribution import Discrete

_logger = logging.getLogger(__name__)


class Particle(Discrete):
    """Discrete probability distributions.

    Similar to particles in SMC, we use the approach where a distribution is approximated as
    particles:

      - The particles are the events of the sample space
      - The weights are denoted by the probabilities of the events

    This class can be used directly, but is also used as parent class for other 1d discrete
    distributions as the computation of expectations is done in the same fashion.

    Attributes:
        mean (np.ndarray): Mean of the distribution
        covariance (np.ndarray): Covariance of the distribution
        dimension (int): Dimensionality of the distribution
        probabilities (np.ndarray): Probabilities associated to all the events in the sample space
        sample_space (np.ndarray): Samples, i.e. possible outcomes of sampling the distribution
    """

    def _compute_mean_and_covariance(self):
        """Compute the mean value and covariance of the mixture model.

        Returns:
            mean (np.ndarray): Mean value of the distribution
            covariance (np.ndarray): Covariance of the distribution
        """
        mean = np.sum(
            self.sample_space
            * np.tile(self.probabilities.reshape(-1, 1), len(self.sample_space[0])),
            axis=0,
        )
        covariance = np.cov(
            self.sample_space, ddof=0, aweights=self.probabilities.flatten(), rowvar=False
        )
        return mean, covariance

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated

        Returns:
            np.ndarray: CDF value of the distribution
        """
        self.check_1d()
        closest_sample_event = np.searchsorted(self.sample_space.flatten(), x.flatten())
        return np.array([np.sum(self.probabilities[: (idx + 1)]) for idx in closest_sample_event])

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws
        """
        samples_per_event = np.random.multinomial(num_draws, self.probabilities)
        samples = (
            [
                [self.sample_space[sample_event]] * repetitions
                for sample_event, repetitions in enumerate(samples_per_event)
                if repetitions
            ],
        )
        samples = np.array(
            list(itertools.chain.from_iterable(*samples)),
        )
        np.random.shuffle(samples)
        return samples.reshape(-1, 1)

    def logpdf(self, x):
        """Log of the probability mass function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated
        """
        return np.log(self.pdf(x))

    def pdf(self, x):
        """Probability mass function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated
        """
        index = np.array([(self.sample_space == xi).all(axis=1).nonzero()[0] for xi in x]).flatten()

        if len(index) != len(x):
            raise ValueError(
                f"At least one event is not part of the sample space {self.sample_space}"
            )

        return self.probabilities[index]

    def ppf(self, quantiles):
        """Percent point function (inverse of cdf-quantiles).

        Args:
            quantiles (np.ndarray): Quantiles at which the ppf is evaluated

        Returns:
            np.ndarray: Event samples corresponding to the quantiles
        """
        self.check_1d()
        indices = np.searchsorted(np.cumsum(self.probabilities), quantiles, side="left")
        indices = np.clip(indices, 0, len(self.probabilities))
        return self.sample_space[indices].reshape(-1)
