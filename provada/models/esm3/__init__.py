"""
esm3/__init__.py

ESM3 model implementation
"""

from typing import List
from esm.models.esm3 import ESM3
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.sdk.api import ESMProtein, GenerationConfig
import torch
from provada.utils.env import device_manager, suppress_console_output
from provada.sequences.vocab import AMINO_ACIDS
from provada.utils.log import get_logger
from tqdm import tqdm

# Set the name of the open model to utilize
OPEN_MODEL_NAME = "esm3-sm-open-v1"

logger = get_logger(__name__)


class ESM3Model:
    def __init__(self):
        logger.info("Loading ESM3 model")
        self.model, self.tokenizer = load()
        self.model.eval()
        self.device = "cpu"

        self.amino_acid_token_ids = torch.tensor(
            [self.tokenizer.get_vocab()[aa] for aa in AMINO_ACIDS], device="cpu"
        )

    def __call__(
        self,
        sequences: List[str],
        batch_size: int = 128,
        device: str = device_manager.get_device(),
        keep_on_device: bool = False,
        return_logits: bool = False,
        progress_bar_msg: str = "Running ESM3 inference",
    ):
        """
        Forward pass through the ESM3 model
        """
        # Move the model to the correct device
        self.model.to(device)
        self.device = str(device)

        # Get the max sequence length
        max_seq_len = max(len(seq) for seq in sequences)

        # Split the sequences into batches
        batches = [sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)]

        all_mean_embeddings = []
        all_logits = []
        all_attention_masks = []

        # For each batch
        for batch_sequences in tqdm(
            batches, desc=progress_bar_msg, total=len(batches), unit="batch"
        ):
            # Tokenize the batch
            batch_inputs = self.tokenizer.batch_encode_plus(
                batch_sequences,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            # Move the inputs to the correct device
            batch_inputs.to(device)

            # Forward pass
            with torch.inference_mode():
                batch_outputs = self.model(
                    sequence_tokens=batch_inputs["input_ids"],
                )

                # Append the embeddings
                embeddings = batch_outputs.embeddings[:, 1:-1, :]  # Remove special tokens
                attention_mask = batch_inputs["attention_mask"][:, 1:-1]

                # Average over embeddings
                masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                seq_lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                batch_mean_embeddings = masked_embeddings.sum(dim=1) / seq_lengths
                all_mean_embeddings.append(batch_mean_embeddings.cpu())

                if return_logits:
                    # Extract logits
                    logits = batch_outputs.sequence_logits[:, 1:-1, :]  # Remove special tokens
                    # Only keep the logits corresponding to the amino acids
                    logits = logits[:, :, self.amino_acid_token_ids.to(device)]

                    # Determine padding length
                    additional_padding_len = max_seq_len - embeddings.size(1)

                    if additional_padding_len > 0:
                        # Pad attention_mask
                        attention_mask = torch.nn.functional.pad(
                            attention_mask, (0, additional_padding_len), value=0
                        )
                        # Pad logits
                        logits = torch.nn.functional.pad(
                            logits, (0, 0, 0, additional_padding_len), value=0.0
                        )

                    # Save attention mask
                    all_attention_masks.append(attention_mask.cpu())

                    # Save logits
                    all_logits.append(logits.cpu())

        # Move model off device
        if self.device != "cpu" and not keep_on_device:
            self.model.to("cpu")
            self.device = "cpu"

        # Concatenate along the batch dimension
        all_mean_embeddings = torch.cat(all_mean_embeddings, dim=0)
        outputs = {"mean_embeddings": all_mean_embeddings}

        if return_logits:
            all_logits = torch.cat(all_logits, dim=0)
            all_attention_masks = torch.cat(all_attention_masks, dim=0)
            outputs["logits"] = all_logits
            outputs["attention_masks"] = all_attention_masks

        return outputs

    def predict_structure(
        self,
        sequences: List[str],
        batch_size: int = 40,
        device: str = device_manager.get_device(),
        keep_on_device: bool = False,
    ):
        """
        Predicts the structures of a list of sequences
        """

        # Move the model to the correct device
        self.model.to(device)
        self.device = str(device)

        # Split the sequences into batches
        max_batch_size = min(batch_size, len(sequences))
        batches = [
            sequences[i : i + max_batch_size] for i in range(0, len(sequences), max_batch_size)
        ]

        all_structures = []

        # For each batch
        for batch_sequences in tqdm(
            batches,
            desc="Predicting structures with ESM3",
            total=len(batches),
            unit="sequence batch",
        ):
            # Create protein and config objects
            esm3_proteins = [ESMProtein(sequence=seq) for seq in batch_sequences]
            structure_configs = [GenerationConfig(track="structure")] * len(esm3_proteins)

            # Generate the structures
            with suppress_console_output(stdout=True, stderr=True):
                structures = self.model.batch_generate(
                    inputs=esm3_proteins,
                    configs=structure_configs,
                )

            # Unpack predicted structures
            for struct in structures:
                all_structures.append(
                    {
                        "sequence": struct.sequence,
                        "pdb_string": struct.to_pdb_string(),
                        "avg_plddt": struct.plddt.mean().item(),
                        "ptm": struct.ptm.item(),
                    }
                )

        # Move model off device
        if self.device != "cpu" and not keep_on_device:
            self.model.to("cpu")
            self.device = "cpu"

        return all_structures


def load():
    """
    Loads the ESM3 model and tokenizer
    """

    # Load in the model and tokenizer
    model = ESM3.from_pretrained(OPEN_MODEL_NAME, device=torch.device("cpu"))
    tokenizer = EsmSequenceTokenizer()

    return model, tokenizer
