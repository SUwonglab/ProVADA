"""
provada/paths.py

Constants and helper functions related to paths in the project.
"""

from pathlib import Path
import sys

# --- Core Project Directories ---
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Path to ProteinMPNN run script ---
PROTEIN_MPNN_DIR = REPO_ROOT / "ProteinMPNN"
MPNN_SCRIPT = PROTEIN_MPNN_DIR / "protein_mpnn_run.py"
PARSE_CHAINS_SCRIPT = PROTEIN_MPNN_DIR / "helper_scripts" / "parse_multiple_chains.py"
MAKE_FIXED_POS_SCRIPT = (
    PROTEIN_MPNN_DIR / "helper_scripts" / "make_fixed_positions_dict.py"
)

PYTHON_PATH = sys.executable


# Helper to build any other path relative to the repository root.
def resource_path(*relative_parts: str) -> Path:
    """
    Builds an absolute path starting from the top-level project directory (REPO_ROOT).

    Example:
      # Assuming REPO_ROOT is /path/to/packagedir/, this returns:
      # /path/to/packagedir/configs/foo.yml
      resource_path("configs", "foo.yml")
    """
    return REPO_ROOT.joinpath(*relative_parts)
