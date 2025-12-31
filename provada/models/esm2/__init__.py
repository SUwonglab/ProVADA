"""
esm2/__init__.py

ESM2 model implementation
"""

from typing import List
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from provada.utils.env import device_manager
from provada.sequences.vocab import AMINO_ACIDS
from provada.utils.log import get_logger
from tqdm import tqdm

# Set the name of the open model to utilize
AVAILABLE_MODELS = [
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
]
logger = get_logger(__name__)


class ESM2Model:
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D"):
        logger.info(f"Loading ESM2 model: {model_name}")
        self.model, self.tokenizer = load(model_name)
        self.device = "cpu"
        self.model.eval()

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
        progress_bar_msg: str = "Running ESM2 inference",
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
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    output_hidden_states=True,
                )

                # Extract embeddings
                embeddings = batch_outputs["hidden_states"][-1][  # Remove special tokens
                    :, 1:-1, :
                ]

                # Extract attention mask
                attention_mask = batch_inputs["attention_mask"][:, 1:-1]

                # Calculate mean embeddings
                masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
                seq_lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                batch_mean_embeddings = masked_embeddings.sum(dim=1) / seq_lengths
                all_mean_embeddings.append(batch_mean_embeddings.cpu())

                if return_logits:
                    # Extract logits
                    logits = batch_outputs["logits"][:, 1:-1, :]  # Remove special tokens
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


def load(model_name: str = "esm2_t33_650M_UR50D"):
    """
    Loads the ESM3 model and tokenizer
    """

    # Load in the model and tokenizer
    model = AutoModelForMaskedLM.from_pretrained(
        "facebook/" + model_name,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/" + model_name)

    return model, tokenizer
