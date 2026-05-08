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
"""Utility methods used by the tutorial tests."""


def inject_notebook_directory_to_path(tb, path_to_notebook):
    """Inject the notebook directory into the notebook Python path.

    Args:
        tb (testbook): testbook object for inserting code into the notebook
        path_to_notebook (str | Path): Path to the notebook under test

    Returns:
        None
    """
    tb.inject(
        f"""
        import sys
        from pathlib import Path
        notebook_dir = Path({str(path_to_notebook)!r}).resolve().parent
        if str(notebook_dir) not in sys.path:
            sys.path.insert(0, str(notebook_dir))
        """,
        before=0,
    )


def inject_mock_base_dir(tb, tmp_path):
    """Inject a mock base directory for testing notebooks.

    Args:
        tb (testbook): testbook object for inserting code into the notebook
        tmp_path: pytest tmp_path fixture for experiment directory

    Returns:
        None
    """
    tb.inject(
        f"""
        from unittest.mock import MagicMock
        from pathlib import Path
        import queens.utils.config_directories
        mock_base_dir = Path('{tmp_path}')
        queens.utils.config_directories.base_directory = MagicMock(return_value=mock_base_dir)
        """,
        before=0,
    )
