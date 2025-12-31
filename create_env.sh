#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Print styled header
echo ""
echo -e "${BOLD}${BLUE}╔═════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║Starting ProVADA Environment Installation║${NC}"
echo -e "${BOLD}${BLUE}╚═════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Create conda environment
echo -e "${BOLD}📦 Creating conda environment...${NC}"
conda create -n provada-env python=3.12 -y
echo -e "${GREEN}✓${NC} Conda environment created\n"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate provada-env
echo -e "${GREEN}✓${NC} Environment activated\n"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

echo -e "${BOLD}🔧 Installing ffmpeg...${NC}"
conda install ffmpeg -y
echo -e "${GREEN}✓${NC} ffmpeg installed\n"

echo -e "${BOLD}⚙️  Installing uv package manager...${NC}"
pip install uv
echo -e "${GREEN}✓${NC} uv installed\n"

# Install packages
echo -e "${BOLD}🔧 Installing provada in editable mode...${NC}"
uv pip install -e . 
echo -e "${GREEN}✓${NC} Package installed in editable mode\n"

echo -e "${BOLD}📚 Installing requirements...${NC}"
uv pip install -r requirements.txt
echo -e "${GREEN}✓${NC} Requirements installed\n"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Install PyRosetta
echo -e "${BOLD}🧬 Installing PyRosetta...${NC}"
python -c "import pyrosetta_installer as I; I.install_pyrosetta()"
echo -e "${GREEN}✓${NC} PyRosetta installed\n"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
echo -e "${BOLD}${GREEN}🎉 Installation complete!${NC}"
echo -e "${GREEN}✓${NC} The provada-env environment is now active.\n"
echo -e "${YELLOW}ℹ️  To activate this environment in future sessions, run:${NC}"
echo -e "   ${BOLD}conda activate provada-env${NC}\n"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}\n"