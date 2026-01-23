"""
SODA (Soft Discrete Optimization for Adversarial attacks) module.

Provides a unified interface for SODA optimization with different backends
(HuggingFace transformers or transformer_lens) and batching strategies
(single-target or internal batching).
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import torch


def soda_optimize_inputs(
    model=None,
    model_name: Optional[str] = None,
    tokenizer=None,
    device: Union[str, torch.device] = "cuda",
    target_text: Optional[str] = None,
    targets: Optional[List[Dict[str, Any]]] = None,
    backend: str = "hf",
    batching: str = "single",
    seq_len: int = 40,
    epochs: int = 100000,
    learning_rate: float = 0.065,
    temperature: float = 0.05,
    decay_rate: float = 0.9,
    betas: Tuple[float, float] = (0.9, 0.995),
    reset_epoch: int = 50,
    reinit_epoch: int = 1500,
    reg_weight: Optional[float] = None,
    bias_correction: bool = False,
    init_strategy: str = "zeros",
    init_std: float = 0.05,
    max_batch_size: int = 285,
    early_stop_on_exact_match: bool = True,
    lcs_ratio_threshold: float = 1.0,
    batch_size: int = 1,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Union[Tuple[List[int], torch.Tensor, List[float], List[float]], Dict[str, Any]]:
    """
    Unified SODA interface with backend and batching dispatch.

    This function provides a single entry point for all SODA variants:
    - HuggingFace single-target
    - HuggingFace internal batching
    - transformer_lens single-target
    - transformer_lens internal batching

    Args:
        model: HuggingFace model (required if backend="hf")
        model_name: Model name for transformer_lens (required if backend="tl")
        tokenizer: HuggingFace tokenizer (required if backend="hf")
        device: Device to run on
        target_text: Single target text (for batching="single")
        targets: List of target dicts (for batching="internal")
        backend: Model backend - "hf" (HuggingFace) or "tl" (transformer_lens)
        batching: Batching strategy - "single" or "internal"
        seq_len: Length of prompt sequence to optimize
        epochs: Maximum optimization epochs
        learning_rate: Learning rate
        temperature: Softmax temperature
        decay_rate: Embedding decay rate
        betas: Adam beta parameters
        reset_epoch: Optimizer reset frequency
        reinit_epoch: Embedding reinitialization frequency
        reg_weight: Fluency regularization weight
        bias_correction: Use Adam bias correction
        init_strategy: Initialization strategy
        init_std: Standard deviation for normal init
        max_batch_size: Max batch size for internal batching
        early_stop_on_exact_match: Stop early on exact match
        lcs_ratio_threshold: LCS ratio threshold for early stopping
        batch_size: Batch size (for compatibility)
        kwargs: Additional keyword arguments

    Returns:
        For batching="single":
            Tuple of (generated_tokens, optimized_inputs, losses, lcs_ratio_history)
        For batching="internal":
            Dict with results, elapsed_time, etc.

    Raises:
        ValueError: If required parameters are missing or invalid combinations
    """
    if isinstance(device, str):
        device = torch.device(device)

    # Validate parameters
    backend = backend.lower()
    batching = batching.lower()

    if backend not in ("hf", "tl"):
        raise ValueError(f"Invalid backend: {backend}. Must be 'hf' or 'tl'.")

    if batching not in ("single", "internal"):
        raise ValueError(f"Invalid batching: {batching}. Must be 'single' or 'internal'.")

    if backend == "hf" and model is None:
        raise ValueError("model is required for backend='hf'")

    if backend == "hf" and tokenizer is None:
        raise ValueError("tokenizer is required for backend='hf'")

    if backend == "tl" and model_name is None:
        raise ValueError("model_name is required for backend='tl'")

    if batching == "single" and target_text is None:
        raise ValueError("target_text is required for batching='single'")

    if batching == "internal" and targets is None:
        raise ValueError("targets is required for batching='internal'")

    # Dispatch to appropriate implementation
    if backend == "hf" and batching == "single":
        from .soda_hf import soda_optimize_inputs_hf
        return soda_optimize_inputs_hf(
            model=model,
            tokenizer=tokenizer,
            device=device,
            target_text=target_text,
            seq_len=seq_len,
            epochs=epochs,
            learning_rate=learning_rate,
            temperature=temperature,
            decay_rate=decay_rate,
            betas=betas,
            reset_epoch=reset_epoch,
            reinit_epoch=reinit_epoch,
            reg_weight=reg_weight,
            bias_correction=bias_correction,
            init_strategy=init_strategy,
            init_std=init_std,
            early_stop_on_exact_match=early_stop_on_exact_match,
            lcs_ratio_threshold=lcs_ratio_threshold,
            batch_size=batch_size,
            kwargs=kwargs,
        )

    elif backend == "hf" and batching == "internal":
        from .soda_hf_batch import soda_optimize_inputs_hf_batch
        return soda_optimize_inputs_hf_batch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            targets=targets,
            seq_len=seq_len,
            epochs=epochs,
            learning_rate=learning_rate,
            temperature=temperature,
            decay_rate=decay_rate,
            betas=betas,
            reset_epoch=reset_epoch,
            reinit_epoch=reinit_epoch,
            reg_weight=reg_weight,
            bias_correction=bias_correction,
            init_strategy=init_strategy,
            init_std=init_std,
            max_batch_size=max_batch_size,
            early_stop_on_exact_match=early_stop_on_exact_match,
            kwargs=kwargs,
        )

    elif backend == "tl" and batching == "single":
        from .soda_tl import soda_optimize_inputs_tl
        return soda_optimize_inputs_tl(
            model_name=model_name,
            device=device,
            target_text=target_text,
            seq_len=seq_len,
            epochs=epochs,
            learning_rate=learning_rate,
            temperature=temperature,
            decay_rate=decay_rate,
            betas=betas,
            reset_epoch=reset_epoch,
            reinit_epoch=reinit_epoch,
            reg_weight=reg_weight,
            bias_correction=bias_correction,
            init_strategy=init_strategy,
            init_std=init_std,
            early_stop_on_exact_match=early_stop_on_exact_match,
            lcs_ratio_threshold=lcs_ratio_threshold,
            batch_size=batch_size,
            model=model,
            kwargs=kwargs,
        )

    elif backend == "tl" and batching == "internal":
        from .soda_tl_batch import soda_optimize_inputs_tl_batch
        return soda_optimize_inputs_tl_batch(
            model_name=model_name,
            device=device,
            targets=targets,
            seq_len=seq_len,
            epochs=epochs,
            learning_rate=learning_rate,
            temperature=temperature,
            decay_rate=decay_rate,
            betas=betas,
            reset_epoch=reset_epoch,
            reinit_epoch=reinit_epoch,
            reg_weight=reg_weight,
            bias_correction=bias_correction,
            init_strategy=init_strategy,
            init_std=init_std,
            max_batch_size=max_batch_size,
            early_stop_on_exact_match=early_stop_on_exact_match,
            model=model,
            kwargs=kwargs,
        )


# Export key components
__all__ = [
    "soda_optimize_inputs",
]
