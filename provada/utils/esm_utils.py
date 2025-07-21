import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer, EsmTokenizer, EsmForMaskedLM
import numpy as np
from typing import Union

from provada.utils.helpers import arr_to_aa


def init_ESM(device:str = 'cuda', model_name= "facebook/esm2_t33_650M_UR50D") -> tuple:
    """
    Initialize the ESM model and tokenizer.
    Args:
        device: Device to run the model on, e.g. 'cuda' or 'cpu'
        model_name: Name of the ESM model to load, default: "facebook/esm2_t33_650M_UR50D"
    Returns:
        model: The ESM model loaded on the specified device
        tokenizer: The ESM tokenizer
    """
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    return model, tokenizer


def get_embedding_single(sequence, model, tokenizer, device) -> Tensor:
    """
    Get the embedding for a single sequence.
    Args:
        sequence: Amino acid sequence as a string
        model: ESM model
        tokenizer: ESM tokenizer
        device: Device to run the model on, e.g. 'cuda' or 'cpu'
    Returns:
        output: Embedding output as a tensor
    """
    tokens = tokenizer.encode(sequence, return_tensors='pt').to(device)
    model = model.to(device)
    layer_to_hook = model.get_submodule('esm.encoder.emb_layer_norm_after')

    def get_embedding_output(module, input, output):
            embedding_output.append(output)
    
    embedding_output = [] 
    with layer_to_hook.register_forward_hook(get_embedding_output):
        with torch.no_grad():
            model(tokens)

    output = embedding_output[0][0].cpu()

    del tokens, layer_to_hook, embedding_output
    torch.cuda.empty_cache()

    return output



def predict_location_from_emb(emb, clf_name, classifier, device) -> tuple:
    """
    Predict the location of a protein from its embedding using a classifier.
    Args:
        emb: Embedding tensor of shape (L, D) where L is sequence length and D is embedding dimension
        clf_name: Name of the classifier architecture ('logreg', 'cnn', 'mlp')
        classifier: The trained classifier model
        device: Device to run the model on, e.g. 'cuda' or 'cpu'
    Returns:
        cyt_prob: Probability of being in the cytoplasm
        ext_prob: Probability of being extracellular
    """
    if clf_name.lower() == 'logreg':
        # if using logreg, the input should already be mean pooled
        cyt_probabilites = classifier.predict_proba([emb])[0][:,1]
        ext_probabilites = classifier.predict_proba([emb])[1][:,1]
        return cyt_probabilites[0], ext_probabilites[0]
    
    if isinstance(emb, np.ndarray):
        # Convert numpy array to torch tensor
        emb = torch.from_numpy(emb).to(device)
    
    if clf_name.lower() == 'cnn':
        emb = emb[None, :, :]
    elif clf_name.lower() == 'mlp':
        # if using mlp, the input should already be mean pooled
        emb = emb[None, :]
    else:
        raise ValueError("clf_name must be 'logreg'/'cnn'/'mlp'")

    with torch.no_grad():
        output = classifier(emb)

    probabilities = torch.sigmoid(output)
    probabilities = probabilities[0]
    return float(probabilities[0]), float(probabilities[1])


def predict_location_from_seq(seq, clf_name, classifier, ESM_model, tokenizer, device) -> tuple:
    """
    Predict the location of a protein from its sequence using a classifier.
    Args:
        seq: Amino acid sequence as a string or numpy array
        clf_name: Name of the classifier architecture ('logreg', 'cnn', 'mlp')
        classifier: The trained classifier model
        ESM_model: The ESM model for embedding
        tokenizer: The ESM tokenizer
        device: Device to run the model on, e.g. 'cuda' or 'cpu'
    Returns:
        cyt_prob: Probability of being in the cytoplasm
        ext_prob: Probability of being extracellular
    """
    if isinstance(seq, np.ndarray):
        seq = arr_to_aa(seq)

    if not isinstance(seq, str):
        raise ValueError("seq must be a string representing the amino acid sequence")

    # this gets the full embeddings
    emb = get_embedding_single(seq, ESM_model, tokenizer, device)

    if clf_name.lower() == 'logreg':
        emb = np.array(torch.mean(emb, dim=0).cpu().numpy())
    elif clf_name.lower() == 'mlp':
        # if using logreg or mlp, the input should already be mean pooled
        emb = torch.mean(emb, dim=0).to(device)
    elif clf_name.lower() == 'cnn':
        emb = emb.to(device)
    else:
        raise ValueError("clf_name must be 'logreg'/'cnn'/'mlp'")

    cyt_prob, ext_prob = predict_location_from_emb(emb, clf_name, classifier, device)
    
    return cyt_prob, ext_prob


def predict_location_from_seqs(seqs, target_label, clf_name, classifier, ESM_model, tokenizer, device) -> list:
    """
    Predict the location of multiple proteins from their sequences using a classifier.
    Args:
        seqs: List of amino acid sequences as strings or a single sequence string or numpy array
        target_label: Label to filter predictions, e.g. 'extracellular'
        clf_name: Name of the classifier architecture ('logreg', 'cnn', 'mlp')
        classifier: The trained classifier model
        ESM_model: The ESM model for embedding
        tokenizer: The ESM tokenizer
        device: Device to run the model on, e.g. 'cuda' or 'cpu'
    Returns:
        List of probabilities for the target label
    """
    if isinstance(seqs, str):
        # If only one sequence, use the single seq function
        return predict_location_from_seq(seqs, clf_name, classifier, ESM_model, tokenizer, device)
    elif isinstance(seqs, np.ndarray):
        if len(seqs.shape) == 1:
            # If only one sequence, use the single seq function
            seqs = seqs.reshape(1, -1)
    else:
        if not isinstance(seqs, list):
            raise ValueError("seqs must be a list of sequences or a single sequence string or numpy array")
    
    probs = []
    for seq in seqs:
        # if multiple sequences, just call the single seq function multiple times
        prob = predict_location_from_seq(seq, clf_name, classifier, ESM_model, tokenizer, device)[target_label == 'extracellular']
        probs.append(prob)

    # probs is a list of tuples
    return probs




def get_ESM_perplexity_one_pass(
    seq: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str = "cuda",
    B: float = 0.1,
    a: float = 0.1,
    epsilon: float = 1e-3,
) -> float:
    """
    Compute one-pass perplexity for a single sequence using an ESM model.

    Steps:
      1. Tokenize and move inputs to `device`.
      2. Forward-pass through model to get raw logits.
      3. Softmax -> apply linear scaling (p*(B+a)/a - B/a).
      4. Threshold probabilities at `epsilon`.
      5. Compute cross-entropy NLL and exponentiate to get perplexity.

    Args:
        model:     a HuggingFace ESM model returning `.logits`
        tokenizer: matching tokenizer
        seq:       input amino-acid string
        device:    e.g. "cuda" or "cpu"
        B, a:      scaling constants
        epsilon:   min probability clamp

    Returns:
        Perplexity (float)
    """
    if isinstance(seq, np.ndarray):
        seq = arr_to_aa(seq)
    
    if not isinstance(seq, str):
        raise ValueError("seq must be a string representing the amino acid sequence")

    # Tokenize
    tokens: Tensor = tokenizer.encode(seq, return_tensors="pt").to(device)

    # Forward
    with torch.no_grad():
        logits: Tensor = model(tokens).logits  # shape [1, L, V]

    # Softmax + linear transform
    probs: Tensor = F.softmax(logits, dim=-1)
    scale = (B + a) / a
    probs = probs * scale - (B / a)

    # Clamp to epsilon
    probs = torch.clamp_min(probs, epsilon)

    # Compute NLL and perplexity
    vocab_size = probs.size(-1)
    nll = F.cross_entropy(
        probs.view(-1, vocab_size),
        tokens.view(-1),
        ignore_index=-100
    )

    perplexity = torch.exp(nll).item()
    return perplexity



def get_ESM_perplexity_one_pass_multiseqs(seqs: Union[np.ndarray, list, str],
                                          model: PreTrainedModel,
                                          tokenizer: PreTrainedTokenizer,
                                          device: str = "cuda",
                                          B: float = 0.1,
                                          a: float = 0.1,
                                          epsilon: float = 1e-3):
    """
    Compute one-pass perplexity for multiple sequences using an ESM model.
    Args:
        seqs:      List of amino acid sequences as strings or a single sequence string or numpy array
        model:     a HuggingFace ESM model returning `.logits`
        tokenizer: matching tokenizer
        device:    e.g. "cuda" or "cpu"
        B, a:      scaling constants
        epsilon:   min probability clamp
    Returns:
        List of perplexities for each sequence.
        If a single sequence is provided, returns a single perplexity value.
    """
    
    if isinstance(seqs, str):
        # If only one sequence, use the single seq function
        return get_ESM_perplexity_one_pass(seqs, model, tokenizer, device, B, a, epsilon)
    elif isinstance(seqs, np.ndarray):
        if len(seqs.shape) == 1:
            # If only one sequence, use the single seq function
            seqs = seqs.reshape(1, -1)
    else:
        if not isinstance(seqs, list):
            raise ValueError("seqs must be a list of sequences or a single sequence string or numpy array")
    
    pppls = []
    for seq in seqs:
        # If multiple sequences, just call the single seq function
        pppl = get_ESM_perplexity_one_pass(seq, model, tokenizer, device, B, a, epsilon)
        pppls.append(pppl)

    # Outputting a list of perplexities
    return pppls
