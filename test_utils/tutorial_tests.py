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


def inject_notebook_execution_context(tb, notebook_dir):
    """Inject the notebook directory as Python path and working directory.

    Args:
        tb (testbook): testbook object for inserting code into the notebook
        notebook_dir (str | Path): Directory conainting the notebook

    Returns:
        None
    """
    tb.inject(
        f"""
        import os
        import sys
        from pathlib import Path
        notebook_dir = Path({str(notebook_dir)!r})
        repo_root = next(
            path for path in (notebook_dir, *notebook_dir.parents)
            if (path / "pyproject.toml").exists()
        )
        for path in (notebook_dir, repo_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        os.chdir(notebook_dir)
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
