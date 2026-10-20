#!/bin/bash
#
# Create the ProVADA conda environment.
#
# Model dependencies are not installed here. ProVADA calls its models through
# proto-tools, which builds an isolated environment per tool on first use, so
# this environment holds only what ProVADA itself imports.

set -euo pipefail

BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

ENV_NAME="${1:-provada-env}"

# Print styled header
echo ""
echo -e "${BOLD}${BLUE}╔═════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║Starting ProVADA Environment Installation║${NC}"
echo -e "${BOLD}${BLUE}╚═════════════════════════════════════════╝${NC}"
echo ""

echo "Creating conda environment '$ENV_NAME'..."
conda create -n "$ENV_NAME" python=3.12 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Installing provada and its dependencies..."
pip install uv
# setup.py reads requirements.txt into install_requires, so this covers both.
uv pip install -e .

echo
echo "Done. Activate with: conda activate $ENV_NAME"
