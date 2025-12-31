# Renin Example Input

This is the example input mentioned in the original ProVADA preprint:
[https://www.biorxiv.org/content/10.1101/2025.07.11.664238v1](https://www.biorxiv.org/content/10.1101/2025.07.11.664238v1)

## Example Commands

To run ProVADA with this example input, use the following commands:
```bash
python run_provada.py --config inputs/renin/renin_localization_mpnn.yaml
```
or
```bash
python run_provada.py --config inputs/renin/renin_localization_esm3.yaml
```

## What is Renin?

Renin is a highly specific secreted aspartic protease that functions within the extracellular space as a central regulator of the renin–angiotensin system. Its physiological role is to cleave the plasma protein angiotensinogen with exceptional precision, initiating a tightly controlled proteolytic cascade that regulates blood pressure and electrolyte balance. This unusually narrow substrate specificity has made renin an attractive candidate for repurposing as a programmable protease capable of precisely controlling protein activity through targeted cleavage.

A fundamental challenge in repurposing renin for intracellular applications is that it evolved to function within the secretory pathway and extracellular milieu. When expressed in the cytoplasm, renin fails to adopt a functional protease conformation and exhibits no detectable cytosolic activity. This loss of function reflects the stark physicochemical differences between extracellular and cytoplasmic environments. These compartments differ substantially in redox potential, pH, ionic strength, and protein quality-control mechanisms, all of which strongly influence protein folding, stability, and catalytic competence.

Extracellular proteases such as renin typically contain structural features optimized for secretion and extracellular stability, including multiple disulfide bonds and N-glycosylation sites. While glycosylation is not strictly required for renin’s catalytic activity, both disulfide bonding and post-translational modifications contribute to its proper folding, stability, and trafficking in oxidizing environments. In the reducing cytoplasm, disulfide bonds cannot form reliably and may actively destabilize the protein, while glycosylation is absent, collectively rendering the native renin scaffold incompatible with cytosolic function.

What makes engineering a cytosol-compatible renin particularly challenging—and therefore a compelling test case for ProVADA—is the lack of evolutionary precedents. There are no known close homologs of renin or related aspartic proteases that are naturally adapted to function in the reducing cytosolic environment. As a result, there are no existing biological templates from which to directly infer design solutions. Starting from a scaffold with no detectable cytosolic activity, traditional directed evolution approaches would require prohibitively large libraries and expensive, low-throughput live-cell assays in mammalian systems.

ProVADA addresses this challenge by integrating generative protein models with a specialized fitness oracle that evaluates cytosolic localization compatibility. Rather than relying on random mutagenesis, ProVADA systematically removes extracellular signatures—such as disulfide-forming cysteines and glycosylation motifs—while preserving renin’s core catalytic machinery and overall structural fold. This approach enables the efficient proposal of cytosolic-compatible renin variants in silico, achieving a 9.5-fold improvement in sampling efficiency relative to conventional rejection-sampling strategies, without requiring iterative experimental screening.

## Files in this Directory

- **example_seq_renin.txt** - The wild-type human renin catalytic domain sequence (340 amino acids). This is the starting point for ProVADA's engineering efforts, with a native cytosolic localization probability of only 0.035

- **renin_af3.pdb** - AlphaFold 3 predicted structure of the renin catalytic domain. This structure is used to guide ProteinMPNN generation and ensure structural constraints are maintained during variant design

- **logreg_model.pkl** - Pre-trained logistic regression classifier that predicts cytosolic localization probability. Trained on ESM2 embeddings from a curated dataset of vertebrate proteins, this model serves as the fitness oracle that guides ProVADA toward cytosolic-compatible variants

- **train_localization_classifier.py** - Python script for training the cytosolic localization classifier. Loads UniProt subcellular localization data, embeds sequences using ESM2, and trains an L2-regularized logistic regression model

- **conserved_positions.txt** - List of sequence positions (1-indexed) that must remain fixed during variant generation. These correspond to the catalytic aspartate residues (D38, D226) and other critical positions essential for maintaining aspartic protease activity

- **interface_positions.txt** - List of positions (1-indexed) corresponding to substrate-contacting and active-site residues. Fixing these positions helps preserve renin's substrate specificity, though many are located in the active-site flap region (residues 80-90) which shows high variability in ProVADA outputs

- **renin_localization_mpnn.yaml** - ProVADA configuration file using ProteinMPNN as the sequence generator. Specifies MADA sampler parameters, temperature and masking schedules, and score weights balancing localization probability against sequence divergence.

- **renin_localization_esm3.yaml** - ProVADA configuration file using ESM3 as the sequence generator. Uses a larger population size (200) compared to the ProteinMPNN configuration, with adjusted masking and temperature schedules optimized for ESM3's generation characteristics
