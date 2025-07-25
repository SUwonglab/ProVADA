# provada/paths.py

from pathlib import Path


# --- Core Project Directories ---
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Path to ProteinMPNN run script ---
MPNN_SCRIPT = REPO_ROOT / "ProteinMPNN/protein_mpnn_run.py"

# PACKAGE_ROOT is the root of the installable provada package (e.g., /path/to/package_dir/provada/)
PACKAGE_ROOT = REPO_ROOT / "provada"

# --- Paths for Internal Helper Scripts ---
UTILS_DIR = PACKAGE_ROOT / "utils"

# Path to the script that parses PDBs
PARSE_CHAINS_SCRIPT = UTILS_DIR / "pdb_to_mpnn_jsonl.py"

# Path to the script that creates the fixed positions dictionary
MAKE_FIXED_POS_SCRIPT = UTILS_DIR / "define_design_constraints.py"

# --- Helper Function for Validation ---

def get_mpnn_script_path():
    """
    Validates and returns the path to the ProteinMPNN script.
    Raises a FileNotFoundError if the path is not configured or invalid.
    """
    if MPNN_SCRIPT is None or not MPNN_SCRIPT.is_file():
        raise FileNotFoundError(
            "The path to 'protein_mpnn_run.py' is not configured or is invalid.\n"
            "Please edit the MPNN_SCRIPT variable in 'provada/paths.py' to point to your local installation."
        )
    return MPNN_SCRIPT


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