"""
io.py
"""

from pathlib import Path
from typing import List
import os

from Bio import SeqIO
import numpy as np
from provada.utils.log import get_logger

logger = get_logger(__name__)


class ProteinMPNNOutput:
    """
    ProteinMPNN Output class

    Contains methods for parsing the output of ProteinMPNN.

    The following attributes are saved for both the original(og) and designed(d) sequences:
        sequence (str): The designed protein sequence.

        score (float): The average negative log probability over designed residues only.
            This score represents how well the model thinks the designed sequence fits
            the backbone structure for the residues that were actually redesigned.
            Lower scores indicate better sequence-structure compatibility. This score
            only considers residues in chains that were specified for design (not fixed).

        global_score (float): The average negative log probability over ALL residues
            in the protein structure, including both designed and fixed residues.
            This score represents the overall compatibility of the entire sequence
            with the entire backbone structure. Lower scores indicate better overall
            sequence-structure compatibility. This score considers all residues
            regardless of whether they were designed or held fixed.

    When score_only is True, only the original sequence scores are saved.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize the ProteinMPNNOutput class.

        Args:
            output_dir (Path): Path to the output directory of the MPNN run.
        """
        self.og_sequence: str = None  # Original sequence
        self.og_score: float = None  # Original score
        self.og_global_score: float = None  # Original global score

        self.d_seqs: List[str] = []  # Designed sequences
        self.d_scores: List[float] = []  # Designed scores
        self.d_global_scores: List[float] = []  # Designed global scores

        # Parse the output
        if Path(output_dir / "score_only").exists():
            self.parse_score_only(output_dir / "score_only")

        elif Path(output_dir / "seqs").exists():
            self.parse_seqs(output_dir / "seqs")

        else:
            raise ValueError(f"ProteinMPNN output directory {output_dir} is not valid")

        logger.debug(
            f"Parsed ProteinMPNN output from {output_dir} contains {len(self.d_seqs)} designed sequences"
        )

    def parse_seqs(self, seqs_dir: Path):
        """
        Parse the output fasta files of an MPNN run. For now, this only supports
        the default MPNN output with a single generation.
        """
        # Find the fasta file
        fasta_list = os.listdir(str(seqs_dir))

        if len(fasta_list) != 1:
            raise ValueError(f"Expected 1 fasta file in {seqs_dir}, found {len(fasta_list)}")

        # Read in the fasta file
        sequence_records = list(SeqIO.parse(str(seqs_dir / fasta_list[0]), "fasta"))

        if len(sequence_records) < 2:
            raise ValueError(
                f"Expected at least 2 sequence records in {seqs_dir}, found {len(sequence_records)}"
            )

        def parse_entry(sequence_record) -> dict:
            """
            Parse the description of a fasta sequence record
            """
            desc_split = sequence_record.description.split(" ")
            output = {}
            for item in desc_split:
                if "global_score=" in item:
                    output["global_score"] = float(item.split("=")[1].replace(",", ""))
                elif "score=" in item:
                    output["score"] = float(item.split("=")[1].replace(",", ""))

            output["sequence"] = str(sequence_record.seq)

            return output

        # Parse original sequence
        og_entry = parse_entry(sequence_records[0])

        # Save the og sequence and it's attributes
        self.og_sequence = og_entry["sequence"]
        self.og_score = -og_entry["score"]
        self.og_global_score = -og_entry["global_score"]

        # Parse designed sequences
        for sequence_record in sequence_records[1:]:
            d_entry = parse_entry(sequence_record)
            self.d_seqs.append(d_entry["sequence"])
            self.d_scores.append(-d_entry["score"])
            self.d_global_scores.append(-d_entry["global_score"])

    def parse_score_only(self, score_only_dir: Path):
        """
        Parse the MPNN score of the input sequence. For now, this only supports
        the default MPNN output with a single generation.
        """
        # Find the score file
        score_files = os.listdir(str(score_only_dir))

        if len(score_files) != 1:
            raise ValueError(
                f"Expected 1 score file in {score_only_dir}, found {len(score_files)}"
            )

        # Read in the score file using np
        score_data = np.load(str(score_only_dir / score_files[0]))

        # Save the scores
        self.og_sequence = str(score_data["seq_str"])
        self.og_score = -score_data["score"].item()
        self.og_global_score = -score_data["global_score"].item()

    def get_output_dict(self) -> dict:
        """
        Returns a dictionary representation of the ProteinMPNNOutput object.
        """
        return {
            "filled_seqs": self.d_seqs,
            "new_global_scores": self.d_global_scores,
            "new_scores": self.d_scores,
            "old_global_score": self.og_global_score,
            "old_score": self.og_score,
        }
