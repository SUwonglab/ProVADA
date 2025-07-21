# provada/paths.py

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_ROOT = PROJECT_ROOT / "provada"

# Paths to data, models, logs, etc.
DATA_DIR   = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR   = Path(os.getenv("MYAPP_LOG_DIR", PROJECT_ROOT / "logs"))

# helper to build any other path
def resource_path(*relative_parts: str) -> Path:
    """
    Return PROJECT_ROOT / <relative_parts...>, e.g.
      resource_path("configs", "foo.yml")
    """
    return PROJECT_ROOT.joinpath(*relative_parts)


# Paths for helper-script paths

# This file contains the path to the script that parse the chains
PARSE_CHAINS_SCRIPT = Path(
    PROJECT_ROOT / "parse_multiple_chains.py"
)

# This file contains the path to the script that makes fixed positions
MAKE_FIXED_POS_SCRIPT = Path(
    PROJECT_ROOT / "make_fixed_positions_dict.py"
)

# This file contains the path to the MPNN script
MPNN_SCRIPT = Path(
    PROJECT_ROOT / "protein_mpnn_run.py"
)



