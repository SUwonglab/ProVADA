# ProVADA: Conditional Protein Variant Generation via Ensemble-Guided Test-Time Steering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101/2025.07.11.664238-blue)](https://www.biorxiv.org/content/10.1101/2025.07.11.664238v1)

This repository contains the official implementation of **ProVADA** (**Pro**tein **V**ariant **Ada**ptation), a computational method for adapting existing proteins by designing novel variants
conditionally. Starting from a wild-type reference sequence, ProVADA steers the design process to optimize for desired functional properties, as described in our manuscript:

> **ProVADA: Generation of Subcellular Protein Variants via Ensemble-Guided Test-Time Steering**  
> Wenhui Sophia Lu*, Xiaowei Zhang*, Santiago L. Mille-Fragoso, Haoyu Dai, Xiaojing J. Gao, Wing Hung Wong.  
> *bioRxiv* (2025) [doi:10.1101/2025.07.11.664238](https://www.biorxiv.org/content/10.1101/2025.07.11.664238v1)

ProVADA leverages protein language models (ESM-2), structure-based models (ProteinMPNN), and a lightweight classifier to generate sequences optimized for a target property (e.g., subcellular location) while maintaining structural integrity and evolutionary plausibility.

## How ProVADA Works

At its core, ProVADA uses an iterative, population-based sampling algorithm called **MADA** (Mixture-Adaptation Directed Annealing) to explore the sequence space. At each iteration, promising sequences are selected through a down-sample-up-sampling process, partially masked, and then re-completed using ProteinMPNN to generate new proposals. These proposals are accepted or rejected based on a fitness score, guiding the population toward the desired properties.


<img src="./assets/mada_algorithm.png" alt="ProVADA MADA Algorithm Overview" width="700"/>

<br> 

The fitness score combines multiple objectives. A key component is a lightweight property classifier, trained on ESM-2 embeddings, which predicts the probability of a sequence having the target function (e.g., cytosolic vs. extracellular localization). A regularization term penalizes for residue mismatch between the original wild-type sequence and the proposed variants.

<img src="./assets/classifier_schematics.png" alt="Classifier Architecture" width="650"/>



## Installation

Follow the instructions below to install ProVADA.

### 1. Clone the repository
```bash
# 1. Clone this repository from GitHub
git clone https://github.com/SUwonglab/provada.git
cd provada
```

### 2. Create and activate a virtual environment
```bash
conda create -n provada-env python=3.11 -y
conda activate provada-env
pip install uv
uv pip install .
```


### 3. ProteinMPNN Set up
ProVADA uses ProteinMPNN as a local dependency for structure-based sequence design.
Run the command below to populate the ProteinMPNN submodule with the required source code.

```bash
git submodule update --init --recursive
```



## Usage

We provide an example script to run ProVADA on the example Renin protein described in the manuscript.

```bash
python run_provada.py \
    --input_pdb_path 'inputs/renin/renin_af3.pdb' \
    --input_sequence_path 'inputs/renin/example_seq_renin.txt' \
    --classifier_weights 'inputs/renin/logreg_model_weights.pkl' \
    --fixed_positions_files 'inputs/renin/interface_positions.txt' 'inputs/renin/conserved_positions.txt' \
    --force_design_residues 'C' \
    --output_dir './results/test_hard_fix' \
    --trajectory_file 'test_fix_300iter_2.csv' \
    --save_trajectory \
    --greedy
```

### Example


### A Note on Using Your Own Proteins

The ProVADA workflow relies on two key utility scripts to prepare PDB structures for ProteinMPNN: `provada/utils/pdb_to_mpnn_jsonl.py` and `provada/utils/define_design_constraints.py`.

These scripts are designed to be general-purpose. However, due to the high variability in PDB file formatting (e.g., non-standard residue names, HETATMs, complex chain IDs), they **may need to be modified** to work correctly with your specific protein structures.


## Citation

If you use ProVADA in your research, please cite our manuscript:
```bibtex
@article {Lu2025.07.11.664238,
	author = {Lu, Wenhui Sophia and Zhang, Xiaowei and Mille-Fragoso, Luis S. and Dai, Haoyu and Gao, Xiaojing J. and Wong, Wing Hung},
	title = {ProVADA: Generation of Subcellular Protein Variants via Ensemble-Guided Test-Time Steering},
	elocation-id = {2025.07.11.664238},
	year = {2025},
	doi = {10.1101/2025.07.11.664238},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/07/17/2025.07.11.664238},
	eprint = {https://www.biorxiv.org/content/early/2025/07/17/2025.07.11.664238.full.pdf},
	journal = {bioRxiv}
}
