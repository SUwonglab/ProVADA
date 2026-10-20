#!/bin/bash
# Environment for ProVADA's PyTorch ProteinMPNN toolkit.
set -euo pipefail
source standalone_helpers.sh

# The fork pins no torch of its own, so install the build matching this host.
proto_install_pytorch

echo "Installing ProteinMPNN fork..."
uv pip install -r requirements.txt

echo "proteinmpnn_pytorch setup complete!"
