import re
import json
import shutil
import tempfile
import subprocess
import sys
import numpy as np
from typing import Optional, List, Tuple, Union
from pathlib import Path
from tempfile import TemporaryDirectory

from provada.utils.helpers import (
    aa_to_arr,
    parse_masked_seq,
    masked_seq_arr_to_str,
)

from provada.utils.log import setup_logger, get_logger

from provada.paths import (
    PARSE_CHAINS_SCRIPT,
    MAKE_FIXED_POS_SCRIPT,
    MPNN_SCRIPT
)

# Default Python command
PYTHON_CMD = sys.executable


def run_mpnn(
    pdb_path: Path,
    out_dir: Path,
    mpnn_script: Path,
    python_cmd: str = "python",
    protein_chain: str = "",
    batch_size: int = 1,
) -> float:
    """
    Invoke ProteinMPNN CLI on a single PDB, write raw output to out_dir,
    parse the 'mean' score, and return it (negated per convention).
    """

    args = [
        python_cmd,
        str(mpnn_script),
        "--pdb_path", str(pdb_path),
        "--pdb_path_chains", protein_chain,
        "--score_only", "1",
        "--out_folder", str(out_dir),
        "--batch_size", str(batch_size),
    ]

    # Run and capture stdout
    proc = subprocess.run(args, stdout=subprocess.PIPE, check=True)
    raw = proc.stdout.decode("utf-8")

    # Parse the penultimate line
    returned_lines = raw.splitlines()

    score_label, score_val = returned_lines[-1:][0].split(",")[1].split(":")
    score_label = score_label.strip()
    assert score_label == "mean", f"unexpected score label {score_label!r}"

    score_val = score_val.strip()   
    score_val = -float(score_val)
    return score_val


def get_mpnn_scores(
    pdb: str,
    protein_chain: str = "",
    *,
    mpnn_script: Optional[str] = None,
    python_cmd: Optional[str] = None,
) -> Optional[float]:
    """
    Score one PDB file via ProteinMPNN.
    WARNING: the score computed here will only be based on the sequence in the PDB file

    Args:
      pdb:            path to a PDB file
      protein_chain:  chain specifier (e.g. "A")
      mpnn_script:    path to protein_mpnn_run.py (overrides default)
      python_cmd:     python interpreter to invoke
    """
    tmp_root = Path("./tmp_")
    tmp_root.mkdir(parents=True, exist_ok=True)

    mpnn_script = Path(mpnn_script or MPNN_SCRIPT)
    python_cmd  = python_cmd  or PYTHON_CMD

    with TemporaryDirectory(dir=tmp_root) as tmpdir:
        score = run_mpnn(
            pdb_path=Path(pdb),
            out_dir=tmp_root,
            mpnn_script=mpnn_script,
            python_cmd=python_cmd,
            protein_chain=protein_chain,
        )

    # Clear temporary directory
    shutil.rmtree(tmp_root)
    return score


def mpnn_masked_gen(
    masked_seq_str: str,
    pdb_path: Union[str, Path],
    protein_chain: str,
    num_seqs_gen: int,
    mpnn_sample_temp: float,
    tmp_root: Optional[Union[str, Path]] = None,
    keep_tmp: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Fill masked positions in `masked_seq_str` via ProteinMPNN, preserving fixed residues.

    Args:
      masked_seq_str:  length-L string, '_' marks masked positions.
      pdb_path:        path to the PDB file.
      protein_chain:   chain identifier (e.g. "A").
      num_seqs_gen:    number of sequences to generate (incl. baseline).
      tmp_root:        base dir for temp work dirs (defaults to ./tmp_).
      keep_tmp:        if True, do not delete the work dir.
      verbose:         if True, output progress messages.

    Returns:
      (filled_seqs,
       new_global_scores,
       new_local_scores,
       old_global_score,
       old_local_score)
    """

    logger = get_logger(__name__)

    pdb_path = Path(pdb_path)
    # Parse masked string → list of AAs + '_'s
    masked_arr = parse_masked_seq(masked_seq_str)

    # Make working dir
    base_tmp = Path(tmp_root or "./tmp_")
    base_tmp.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(dir=base_tmp))
    logger.debug(f"[mpnn_gen] work dir → {work_dir}")

    # Generate parsed.jsonl from PDB
    parsed_jsonl = work_dir / "parsed.jsonl"
    subprocess.run(
        [
            PYTHON_CMD,
            str(PARSE_CHAINS_SCRIPT),
            "--input_path",
            str(pdb_path.parent),
            "--output_path",
            str(parsed_jsonl),
        ],
        check=True,
    )

    # Override sequences for fixed positions
    data = []
    seq_key = f"seq_chain_{protein_chain}"
    with open(parsed_jsonl) as fh:
        for line in fh:
            rec = json.loads(line)
            key = seq_key if seq_key in rec else "seq"
            seq_list = list(rec.get(key, rec.get("seq", "")))
            for idx, aa in enumerate(masked_arr):
                if aa != "_":
                    seq_list[idx] = aa
            new_seq = "".join(seq_list)
            rec[key] = new_seq
            if key != "seq":
                rec["seq"] = new_seq
            data.append(rec)
    with open(parsed_jsonl, "w") as fh:
        for rec in data:
            fh.write(json.dumps(rec) + "\n")

    # Build fixed_positions.jsonl
    fixed_jsonl = work_dir / "fixed_positions.jsonl"
    positions = [i + 1 for i, aa in enumerate(masked_arr) if aa != "_"]
    pos_str = " ".join(map(str, positions))
    subprocess.run(
        [
            PYTHON_CMD,
            str(MAKE_FIXED_POS_SCRIPT),
            "--input_path",
            str(parsed_jsonl),
            "--output_path",
            str(fixed_jsonl),
            "--chain_list",
            protein_chain,
            "--position_list",
            pos_str,
        ],
        check=True,
    )

    # Run ProteinMPNN design
    proc = subprocess.run(
        [
            PYTHON_CMD,
            str(MPNN_SCRIPT),
            "--jsonl_path",
            str(parsed_jsonl),
            "--pdb_path_chains",
            protein_chain,
            "--fixed_positions_jsonl",
            str(fixed_jsonl),
            "--omit_AAs",
            "CX",
            "--out_folder",
            str(work_dir),
            "--batch_size",
            "1",
            "--num_seq_per_target",
            str(num_seqs_gen),
            "--sampling_temp",
            str(mpnn_sample_temp)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    raw_out = proc.stdout.decode()

    # Parse the FASTA results
    result_file = work_dir / "seqs" / (pdb_path.stem + ".fa")
    header_re = re.compile(
        r"score=(?P<score>[-+]?\d+\.\d+).*?global_score=(?P<global_score>[-+]?\d+\.\d+)"
    )
    results = []
    with open(result_file) as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            m = header_re.search(line)
            local = float(m.group("score")) if m else None
            glob  = float(m.group("global_score")) if m else None
            seq   = next(fh).strip()
            results.append({"sequence": seq, "score": local, "global_score": glob})

    # Cleanup
    if not keep_tmp:
        logger.debug(f"[mpnn_gen] deleting work dir at {work_dir}")
        shutil.rmtree(work_dir)
    else:
        logger.debug(f"[mpnn_gen] retained work dir at {work_dir}")

    # Organize outputs
    baseline = results[0]
    old_local_score  = -baseline["score"]
    old_global_score = -baseline["global_score"]

    filled_seqs       = [r["sequence"] for r in results[1:]]
    new_local_scores  = np.asarray([-r["score"] for r in results[1:]])
    new_global_scores = np.asarray([-r["global_score"] for r in results[1:]])

    return {
        'filled_seqs': filled_seqs,
        'new_global_scores': new_global_scores,
        'new_local_scores': new_local_scores,
        'old_global_score': old_global_score,
        'old_local_score': old_local_score
    }


def fill_mpnn(
    input_seq: np.ndarray,
    pdb_path: Union[str, Path],
    protein_chain: str = "",
    num_seqs_gen: int = 1,
    tmp_root: Optional[Union[str, Path]] = None,
    keep_tmp: bool = False,
    mpnn_sample_temp: float = 0.5,
) -> dict:
    """
    This function turns a masked-array into a masked-string.
    Invoke ProteinMPNN to fill in masked positions.
    Convert returned sequences back to integer arrays.

    Returns:
      proposal_seqs:  (num_seqs_gen-1, L) array of filled-in seqs
      new_global_scores,
      new_local_scores,
      old_global_score,
      old_local_score
    """
    pdb_path = Path(pdb_path)

    if isinstance(input_seq, str):
        input_seq = aa_to_arr(input_seq)

    # Ensure input is an array and convert to masked string
    if not isinstance(input_seq, np.ndarray):
        raise ValueError(f"`input_seq` must be np.ndarray, got {type(input_seq)}")
    masked_str = masked_seq_arr_to_str(input_seq)

    # Run MPNN fill
    mpnn_output = mpnn_masked_gen(
        masked_seq_str=masked_str,
        pdb_path=pdb_path,
        protein_chain=protein_chain,
        num_seqs_gen=num_seqs_gen,
        tmp_root=tmp_root,
        keep_tmp=keep_tmp,
        mpnn_sample_temp=mpnn_sample_temp,
    )

    # Convert AA‐strings back into integer arrays
    mpnn_output['filled_seqs'] = np.stack([aa_to_arr(seq) for seq in mpnn_output['filled_seqs']], axis=0)

    return mpnn_output
