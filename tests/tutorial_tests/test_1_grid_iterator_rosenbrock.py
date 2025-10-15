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
"""Integration tests for 1-grid-iterator-rosenbrock tutorial."""

import numpy as np
from testbook import testbook

from test_utils.tutorial_tests import inject_mock_path


@testbook(
    "tutorials/1-grid-iterator-rosenbrock.ipynb",
)
def test_output_tutorial_1(tb, tmp_path):
    """Parameterized test case for tutorial 1: Grid Iterator Rosenbrock.

    The notebook is run with injected lines of codes for testing
    that the final results are as expected.
    """
    optimal_fun = 2.957935e-11
    optimal_x = np.array([0.99999463, 0.99998915]).tolist()

    inject_mock_path(tb, tmp_path)
    # tb.execute_cell([2, 4, 6, 8, 13, 15, 17, 19, 21, 23, 25, 27])
    tb.execute_cell([3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 26, 28])
    tb.inject(
        """np.testing.assert_allclose(X1, X1_QUEENS)
np.testing.assert_allclose(X2, X2_QUEENS)
np.testing.assert_allclose(Z, Z_QUEENS)"""
    )

    tb.execute_cell([32])
    tb.inject(f"np.testing.assert_allclose(optimal_fun, {optimal_fun},atol=1e-12)")
    tb.inject(f"np.testing.assert_allclose(optimal_x, np.array({optimal_x}))")
