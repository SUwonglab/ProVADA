# Example Inputs

These examples cover the three protein redesign campaigns in the ProVADA
preprint. Each directory contains a reference sequence, a structure, fixed-residue
positions, and a configuration file.

| Example | Design objective |
| --- | --- |
| [Renin](./renin/README.md) | Adapt a secreted protease for activity in the cytosol |
| [FMC63](./fmc63/README.md) | Reduce predicted HLA class I binding of peptides from the anti-CD19 scFv |
| [SunTag scFv](./scfvgcn4/README.md) | Reduce predicted HLA class I binding of peptides from the anti-GCN4 scFv |

All three examples use ProteinMPNN to generate variants while holding selected
residues fixed. The individual READMEs describe the design settings and provide
run commands. Run these commands from the repository root after following the
[installation instructions](../README.md#set-up).

The two deimmunization examples also require netMHCpan; see their READMEs for
setup details.
