# Nanobody Intracellularization Example Input

This is an additional example input to run nanobody intracellularization with ProVADA. The example protein to be internalized is a nanobody that binds Vimentin termed VimB6 ([source](https://www.nature.com/articles/srep13402)).

## Example Commands

To run ProVADA with this example input, use the following commands:
```bash
python run_provada.py --config inputs/nanobidies/vimb6_localization.yaml
```
The default generative model used is SolubleMPNN.

## Brief introduction to antibody/nanobody intracellularization challenge

As powerful protein-based binders, antibodies and nanobodies are crucial not only for therapeutic applications but also for research tools. Many applications require them to express intracellularly, yet these proteins are secreted and functions extracellularly by nature.

Engineering an antibody or nanobody to function intracellularly (often called **intracellularization**, and intracellularly viable antibodies are called **intrabodies**) is challenging because many variable-domain scaffolds were optimized to fold with stabilizing intradomain disulfide bonds in the secretory pathway, whereas the reducing cytosol/nucleus disfavors disulfide formation and commonly yields misfolded, inactive, or aggregation-prone intrabodies. 

To make viable intrabodies, researchers need to “intracellularize” binders by selecting/engineering variants that are disulfide-independent and cytosol-stable—for example via molecular evolution to produce cysteine-free scFvs that remain functional without the conserved disulfides. 

Therefore, ProVADA can also be applied to accelerate intrabody development. Similar to the Renin example, the same subcellular functionality oracle can be used to efficiently sample variants of the nanobody/antibody scaffold to generate vairants that are adapted to cytosolic environment while not affecting its binding profile.

## Files in this Directory

- **vimb6.pdb**: AlphaFold3 predicted structure of an example nanobody VimB6, which has poor intracellular expression.

- **vimb6_cdr_positions.txt**: List of sequence positions (1-indexed) of the CDR regions in VimB6 using standard Kabat numbering. To remain binding to the target protein, these regions needs to be fixed during ProVADA sampling.
  
- **vimb6_localization.yaml**: ProVADA configuration file using SolubleMPNN as the sequence generator. Specifies MADA sampler parameters, temperature and masking schedules, and score weights balancing localization probability against sequence divergence.

- **vimb6_seq.txt**: Sequence of the anti-vimentin nanobody VimB6, with a low starting cytosolic probability of 0.0122.
