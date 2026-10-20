"""
provada/paths.py

Constants and helper functions related to paths in the project.
"""

from pathlib import Path
import sys

# --- Core Project Directories ---
REPO_ROOT = Path(__file__).resolve().parent.parent

PYTHON_PATH = sys.executable


# Helper to build any other path relative to the repository root.
