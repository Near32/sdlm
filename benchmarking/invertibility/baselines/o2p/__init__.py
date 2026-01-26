"""
O2P (Output-to-Prompt) module.

Provides a unified interface for O2P prompt inversion using a pre-trained
T5-based inverse model.

Unlike SODA/GCG which optimize iteratively, O2P uses a pre-trained inverse
model for one-shot prompt inversion.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import torch


def o2p_optimize_inputs(
    model=None,
    tokenizer=None,
    device: Union[str, torch.device] = "cuda",
    target_text: Optional[str] = None,
    o2p_model_path: Optional[str] = None,
    seq_len: int = 40,
    num_beams: int = 4,
    max_length: int = 32,
    batch_size: int = 1,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], torch.Tensor, List[float], List[float]]:
    """
    O2P prompt inversion using a pre-trained inverse model.

    Unlike SODA/GCG, O2P uses a pre-trained T5-based inverse model to
    generate prompts in one-shot rather than iteratively optimizing.

    Args:
        model: HuggingFace model (subject LLM for model caching compatibility)
        tokenizer: HuggingFace tokenizer
        device: Device to run on
        target_text: The target text we want to invert
        o2p_model_path: Path to the trained O2P inverse model
        seq_len: Length of prompt sequence (used for padding/truncation)
        num_beams: Beam size for T5 generation
        max_length: Maximum generation length for T5 decoder
        batch_size: Batch size (for compatibility)
        kwargs: Additional keyword arguments

    Returns:
        Tuple of:
            - generated_tokens: List of generated token IDs
            - optimized_inputs: Tensor placeholder (not used for O2P)
            - losses: List with single 0.0 value (no iterative loss for O2P)
            - lcs_ratio_history: List with single final LCS ratio

    Raises:
        ValueError: If required parameters are missing
    """
    if isinstance(device, str):
        device = torch.device(device)

    # Validate parameters
    if model is None:
        raise ValueError("model is required for O2P")

    if tokenizer is None:
        raise ValueError("tokenizer is required for O2P")

    if target_text is None:
        raise ValueError("target_text is required for O2P")

    if o2p_model_path is None:
        raise ValueError("o2p_model_path is required for O2P - provide path to trained inverse model")

    # Dispatch to HuggingFace implementation (O2P always uses HF)
    from .o2p_hf import o2p_optimize_inputs_hf
    return o2p_optimize_inputs_hf(
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_text=target_text,
        o2p_model_path=o2p_model_path,
        seq_len=seq_len,
        num_beams=num_beams,
        max_length=max_length,
        kwargs=kwargs,
    )


# Export key components
__all__ = [
    "o2p_optimize_inputs",
]
