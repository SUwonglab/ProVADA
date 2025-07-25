"""
run_provada.py

Main executable for ProVADA
"""

import argparse
import os
import pickle
from pathlib import Path

import torch

from provada import BaseSequenceInfo
from provada.sampler import MADA, SamplerParams
from provada.utils.helpers import get_sequence, read_fixed_positions
from provada.utils.log import get_logger, setup_logger, display_config
from provada.utils.esm_utils import init_ESM


def main(args):
    """
    Main function for running ProVADA.

    Args:
        args: Command line arguments.
    """
    # Setup logger
    setup_logger(verbose=args.verbose, log_filename=args.log_filename)
    logger = get_logger("run_provada")
    logger.info(f"Starting ProVADA run")
    display_config(args, config_name="ProVADA Run Arguments:")

    # Limit GPU visibility
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        logger.info(f"Setting CUDA_VISIBLE_DEVICES to {args.cuda_device}")

    # --- 1. Setup Device ---
    logger.info("Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 2. Create Output Directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")

    # --- 3. Load Inputs ---
    logger.info("Loading input files...")
    input_seq = get_sequence(args.input_sequence_path)
    logger.debug(f"Loaded sequence of length {len(input_seq)} from {args.input_sequence_path}")

    # --- 4. Process Fixed Positions ---
    logger.info("Processing fixed positions...")
    all_fixed_positions = []
    if args.fixed_positions_files:
        for f_path in args.fixed_positions_files:
            logger.debug(f"Reading fixed positions from: {f_path}")
            all_fixed_positions.extend(read_fixed_positions(f_path))

    # Remove duplicates and sort
    fixed_positions = sorted(list(set(all_fixed_positions)))
    if fixed_positions:
        logger.debug(f"Found {len(fixed_positions)} unique fixed positions.")

    # Exclude any residues that should be forced to be designable (e.g., Cysteines)
    hard_fixed_positions = [
        pos for pos in fixed_positions if input_seq[pos - 1] not in args.force_design_residues
    ]
    if args.force_design_residues:
        logger.debug(f"Forcing residues '{args.force_design_residues}' to be designable.")

    logger.debug(f"Final number of hard fixed positions: {len(hard_fixed_positions)}")

    # --- 5. Load Models ---
    logger.info("Loading models...")
    logger.debug(f"Initializing ESM model: {args.esm_model}")
    ESM_model, tokenizer = init_ESM(device=device, model_name=args.esm_model)

    logger.debug(f"Loading classifier weights from: {args.classifier_weights}")
    with open(args.classifier_weights, 'rb') as f:
        loaded_model = pickle.load(f)

    # --- Prepare Sampler and Fitness Parameters ---
    logger.info("Configuring sampler and fitness parameters...")
    sampler_params = SamplerParams(
        mpnn_sample_temp=args.mpnn_temp,
        top_k_frac=args.top_k_frac,
        greedy=args.greedy,
    )

    base_protein_info = BaseSequenceInfo(
        base_seq=input_seq,
        classifier=loaded_model,
        clf_name=args.clf_name,
        target_label=args.target_label,
        ESM_model=ESM_model,
        tokenizer=tokenizer,
        device=device,
        penalty_perplexity=args.penalty_perplexity,
        penalty_MPNN=args.penalty_mpnn,
        input_pdb=args.input_pdb_path,
        save_path=args.output_dir,
        protein_chain=args.protein_chain,
        hard_fixed_positions=hard_fixed_positions,
    )
    logger.debug("BaseSequenceInfo and SamplerParams configured.")

    # --- 7. Run MADA ---
    logger.info(f"Starting MADA run with {args.num_iter} iterations for {args.num_seqs} sequences...")
    results = MADA(
        sequence=input_seq,
        num_seqs=args.num_seqs,
        num_iter=args.num_iter,
        sampler_params=sampler_params,
        base_protein_info=base_protein_info,
        verbose=args.verbose,
        save_sample_traj=args.save_trajectory,
        trajectory_path=args.output_dir,
        trajectory_file=args.trajectory_file,
    )
    logger.debug("MADA run completed.")
    logger.debug(f"Final results dataframe shape: {results.shape}")
    logger.debug(f"Results saved in {args.output_dir}")


def get_cli_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MADA for protein sequence design.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Input Files ---
    inputs = parser.add_argument_group('Input Files')
    inputs.add_argument("--input_pdb_path", type=str, required=True, help="Path to input PDB file")
    inputs.add_argument("--input_sequence_path", type=str, required=True, help="Path to input sequence file")
    inputs.add_argument('--classifier_weights', type=str, required=True, help='Path to the pickled classifier model weights (.pkl).')
    inputs.add_argument(
        '--fixed_positions_files',
        type=str,
        nargs='*',
        default=[],
        help='(Optional) A list of paths to any files containing positions to keep fixed. Accepts multiple files.'
    )
    inputs.add_argument(
        '--force_design_residues',
        type=str,
        default="",
        help=('A string of one-letter amino acid codes (e.g., "C" or "CGP") that '
              'should always be designable, even if they are in a fixed positions file. '
              'For example, "C" means all cysteines are designable. '
              'Default is empty string.')
    )

    # --- Output Settings ---
    outputs = parser.add_argument_group('Output Settings')
    outputs.add_argument('--output_dir', type=str, default="./results", help='Directory to save results, logs, and trajectories.')
    outputs.add_argument('--log_filename', type=str, default="mada_run.log", help='Name for the log file.')
    outputs.add_argument('--save_trajectory', action='store_true', help='Save the full sampling trajectory.')
    outputs.add_argument('--trajectory_file', type=str, default="trajectory.csv", help='Filename for the trajectory CSV.')

    # --- Model & Protein Settings ---
    models = parser.add_argument_group('Model & Protein Settings')
    models.add_argument('--esm_model', type=str, default="facebook/esm2_t33_650M_UR50D", help='Name of the ESM model from HuggingFace.')
    models.add_argument('--protein_chain', type=str, default="A", help='Chain ID of the protein in the PDB file.')
    models.add_argument('--clf_name', type=str, default="logreg", help='Name of the classifier model.')
    models.add_argument('--target_label', type=str, default="cytosolic", help='Target label for the classifier.')
    models.add_argument('--cuda_device', type=int, default=None, help='Specify a CUDA device ID to use (e.g., 0, 1, 2).')

    # --- MADA & Sampler Parameters ---
    sampler = parser.add_argument_group('MADA & Sampler Parameters')
    sampler.add_argument('--num_seqs', type=int, default=30, help='Number of sequences to generate.')
    sampler.add_argument('--num_iter', type=int, default=300, help='Number of MADA iterations.')
    sampler.add_argument('--mpnn_temp', type=float, default=0.5, help='Sampling temperature for ProteinMPNN.')
    sampler.add_argument('--top_k_frac', type=float, default=0.2, help='Top-k fraction for sampling.')
    sampler.add_argument('--greedy', action='store_true', help='Use greedy sampling instead of default.')
    sampler.add_argument('--penalty_perplexity', type=float, default=0.1, help='Penalty for ESM perplexity.')
    sampler.add_argument('--penalty_mpnn', type=float, default=0.1, help='Penalty for ProteinMPNN score.')

    # --- General ---
    general = parser.add_argument_group('General')
    general.add_argument('--verbose', action='store_true', help='Enable verbose logging to console.')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_cli_args()
    main(args)