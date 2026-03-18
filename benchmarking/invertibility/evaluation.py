"""Evaluation utilities for differentiable prompt reconstruction.

Contains:
  - run_discrete_validation_pass: greedy argmax decode (no soft embeddings)
  - run_embsim_validation_pass: greedy embedding-similarity nearest-token decode
  - build_embsim_wandb_metrics: build standard W&B metrics dict from embsim results
  - build_embsim_epoch_callback: factory for a per-epoch embsim validation callback
    suitable for passing to SODA / GCG training loops.

These functions were extracted from main.py so that baseline modules can use them
without creating a circular import (baselines → main).
"""

import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional

from metrics_registry import lcs_length


def compute_embsim_probs(
    optimized_logits: torch.Tensor,
    stgs_module=None,
    batch_size: int = 1,
    embsim_temperature: float = 1.0,
    probs_source: str = "input_logits",
) -> torch.Tensor:
    """Build the soft token distribution used for embedding-similarity operations."""
    with torch.no_grad():
        if probs_source == "gumbel_soft":
            if stgs_module is None:
                raise ValueError("probs_source='gumbel_soft' requires stgs_module")
            gs_logits = optimized_logits.repeat(batch_size, 1, 1)
            _, _, _, y_soft = stgs_module.forward(gs_logits)
            return y_soft[0:1]
        if probs_source != "input_logits":
            raise ValueError(f"Unknown probs_source: {probs_source}")
        if embsim_temperature == 0.0:
            raise ValueError("embsim_temperature must be non-zero")
        return torch.softmax(optimized_logits / embsim_temperature, dim=-1)


def run_discrete_validation_pass(optimized_logits, allowed_tokens, model, tokenizer,
                                  target_length, device, pre_prompt=None):
    """Greedy decode using only argmax hard token IDs — no soft embeddings."""
    argmax_in_allowed = optimized_logits.argmax(dim=-1)          # (1, seq_len)
    full_vocab_ids = allowed_tokens[argmax_in_allowed]            # (1, seq_len)
    if pre_prompt is not None:
        pre_ids = tokenizer(pre_prompt, return_tensors="pt").input_ids.to(device)
        input_ids = torch.cat([pre_ids, full_vocab_ids.to(device)], dim=1)
    else:
        input_ids = full_vocab_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=target_length,
            do_sample=False,
        )
    disc_gen_ids = out[:, input_ids.shape[1]:]           # strip prompt prefix
    disc_text = tokenizer.decode(disc_gen_ids[0], skip_special_tokens=False)
    disc_prompt_full_ids = full_vocab_ids[0].cpu().tolist()
    return disc_prompt_full_ids, disc_gen_ids[0].cpu().tolist(), disc_text


def run_embsim_validation_pass(optimized_logits, embedding_weights_subset, allowed_tokens,
                               model, tokenizer, target_length, device, pre_prompt=None,
                               stgs_module=None, batch_size=1, similarity="cossim",
                               teacher_forcing=False, target_tokens=None,
                               embsim_temperature: float = 1.0):
    """Greedy decode using the nearest-embedding token at each position.

    Picks the allowed token whose embedding has the highest similarity to
    softmax(logits) @ embedding_weights_subset at each position.

    When stgs_module is provided, uses batch_size GS samples (with fresh Gumbel noise
    and the correct temperature) averaged to form the soft embedding weights, matching
    the distribution used during training.

    Args:
        similarity: 'cossim' (cosine similarity, default) or 'dotproduct' (unnormalized
                    dot product, better aligns with how the model computes logits).
        teacher_forcing: if True and target_tokens is provided, replace autoregressive
                    generation with a single teacher-forced forward pass.  Faster than
                    model.generate() and evaluates the prompt under the same teacher-forced
                    objective used during training.
        target_tokens: list of int token IDs (the reference target) required when
                    teacher_forcing=True; ignored otherwise.
        embsim_temperature: Temperature applied to raw logits before softmax when
                    stgs_module is None (embsim_use_input_logits=True path). Values < 1
                    sharpen the soft distribution toward argmax; values > 1 flatten it.
                    Ignored when stgs_module is provided (STGS controls its own
                    temperature). Default 1.0 (backward-compatible, no behaviour change).
    """
    with torch.no_grad():
        probs = compute_embsim_probs(
            optimized_logits,
            stgs_module=stgs_module,
            batch_size=batch_size,
            embsim_temperature=embsim_temperature,
            probs_source="gumbel_soft" if stgs_module is not None else "input_logits",
        )
        soft_emb = torch.matmul(probs, embedding_weights_subset)     # (1, seq_len, embed_dim)

        if similarity == "dotproduct":
            # Dot product: aligns with how the model computes logits (emb @ E^T)
            sim = torch.matmul(soft_emb, embedding_weights_subset.T)  # (1, seq_len, allowed_vocab)
            argmax_in_allowed = sim.argmax(dim=-1)                    # (1, seq_len)
        elif similarity == "l2":
            soft_sq  = (soft_emb ** 2).sum(dim=-1, keepdim=True)
            emb_sq   = (embedding_weights_subset ** 2).sum(dim=-1)
            cross    = torch.matmul(soft_emb, embedding_weights_subset.T)
            l2_sq    = soft_sq - 2 * cross + emb_sq
            sim      = -l2_sq                                         # (1, seq_len, allowed_vocab)
            argmax_in_allowed = l2_sq.argmin(dim=-1)                  # (1, seq_len)
        else:  # "cossim"
            soft_emb_norm = F.normalize(soft_emb, dim=-1)
            emb_norm = F.normalize(embedding_weights_subset, dim=-1)  # (allowed_vocab, embed_dim)
            sim = torch.matmul(soft_emb_norm, emb_norm.T)             # (1, seq_len, allowed_vocab)
            argmax_in_allowed = sim.argmax(dim=-1)                    # (1, seq_len)
        full_vocab_ids = allowed_tokens[argmax_in_allowed]            # (1, seq_len)

        # Per-position L2 error between soft embedding and the selected token's embedding
        embsim_emb = embedding_weights_subset[argmax_in_allowed]      # (1, seq_len, embed_dim)
        pos_error  = (soft_emb - embsim_emb).norm(dim=-1)            # (1, seq_len)

        # Per-position entropy of the input soft distribution
        _eps = 1e-10
        probs_ent = -(probs * (probs + _eps).log()).sum(dim=-1)[0]   # (seq_len,)
        # Per-position entropy of the similarity-score distribution (softmax of sim)
        sim_probs = sim.softmax(dim=-1)                               # (1, seq_len, allowed_vocab)
        sim_ent   = -(sim_probs * (sim_probs + _eps).log()).sum(dim=-1)[0]  # (seq_len,)
        # Entropy of the clean token distribution (before Gumbel noise)
        clean_probs = torch.softmax(optimized_logits, dim=-1)         # (1, seq_len, allowed_vocab)
        clean_ent   = -(clean_probs * (clean_probs + _eps).log()).sum(dim=-1)[0]  # (seq_len,)
        # Per-position entropy difference: H(probs) − H(clean_softmax)
        # Measures how much GS noise reshapes distribution sharpness; 0 when stgs_module is None
        ent_diff    = probs_ent - clean_ent                           # (seq_len,)

        emb_err_stats = {
            "mean": pos_error.mean().item(),
            "min":  pos_error.min().item(),
            "max":  pos_error.max().item(),
            "probs_entropy_mean":  probs_ent.mean().item(),
            "probs_entropy_min":   probs_ent.min().item(),
            "probs_entropy_max":   probs_ent.max().item(),
            "clean_entropy_mean":  clean_ent.mean().item(),
            "clean_entropy_min":   clean_ent.min().item(),
            "clean_entropy_max":   clean_ent.max().item(),
            "sim_entropy_mean":    sim_ent.mean().item(),
            "sim_entropy_min":     sim_ent.min().item(),
            "sim_entropy_max":     sim_ent.max().item(),
            "entropy_diff_mean":   ent_diff.mean().item(),   # H(probs) − H(clean)
            "entropy_diff_min":    ent_diff.min().item(),
            "entropy_diff_max":    ent_diff.max().item(),
        }

    if pre_prompt is not None:
        pre_ids = tokenizer(pre_prompt, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = torch.cat([pre_ids, full_vocab_ids.to(device)], dim=1)
    else:
        prompt_input_ids = full_vocab_ids.to(device)

    if teacher_forcing and target_tokens is not None:
        # Single teacher-forced forward pass — faster than autoregressive generation.
        # Builds [prompt | target] and reads off next-token predictions at target positions.
        target_ids = torch.tensor(target_tokens, device=device).unsqueeze(0)  # (1, tgt_len)
        tf_input_ids = torch.cat([prompt_input_ids, target_ids], dim=1)
        with torch.no_grad():
            tf_out = model(input_ids=tf_input_ids, attention_mask=torch.ones_like(tf_input_ids))
        prompt_len = prompt_input_ids.shape[1]
        # Logit at position (prompt_len - 1) predicts target_tokens[0], etc.
        tf_logits = tf_out.logits[:, prompt_len - 1 : prompt_len - 1 + target_length, :]
        tf_gen_ids = tf_logits.argmax(dim=-1)[0]  # (target_length,)
        tf_text = tokenizer.decode(tf_gen_ids.cpu(), skip_special_tokens=False)
        return full_vocab_ids[0].cpu().tolist(), tf_gen_ids.cpu().tolist(), tf_text, emb_err_stats

    with torch.no_grad():
        out = model.generate(
            input_ids=prompt_input_ids,
            attention_mask=torch.ones_like(prompt_input_ids),
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=target_length,
            do_sample=False,
        )
    embsim_gen_ids = out[:, prompt_input_ids.shape[1]:]
    embsim_text = tokenizer.decode(embsim_gen_ids[0], skip_special_tokens=False)
    return full_vocab_ids[0].cpu().tolist(), embsim_gen_ids[0].cpu().tolist(), embsim_text, emb_err_stats


def build_embsim_wandb_metrics(emb_err_stats: Dict[str, float], embsim_lcs: float) -> Dict[str, float]:
    """Build the standard W&B metrics dict from embsim validation results.

    Mirrors the inline logging block in the STGS per-epoch embsim section of main.py.
    """
    return {
        "embsim_lcs_ratio":            embsim_lcs,
        "embsim_emb_err_mean":         emb_err_stats["mean"],
        "embsim_emb_err_min":          emb_err_stats["min"],
        "embsim_emb_err_max":          emb_err_stats["max"],
        "embsim_probs_entropy_mean":   emb_err_stats["probs_entropy_mean"],
        "embsim_probs_entropy_min":    emb_err_stats["probs_entropy_min"],
        "embsim_probs_entropy_max":    emb_err_stats["probs_entropy_max"],
        "embsim_clean_entropy_mean":   emb_err_stats["clean_entropy_mean"],
        "embsim_clean_entropy_min":    emb_err_stats["clean_entropy_min"],
        "embsim_clean_entropy_max":    emb_err_stats["clean_entropy_max"],
        "embsim_sim_entropy_mean":     emb_err_stats["sim_entropy_mean"],
        "embsim_sim_entropy_min":      emb_err_stats["sim_entropy_min"],
        "embsim_sim_entropy_max":      emb_err_stats["sim_entropy_max"],
        "embsim_entropy_diff_mean":    emb_err_stats["entropy_diff_mean"],
        "embsim_entropy_diff_min":     emb_err_stats["entropy_diff_min"],
        "embsim_entropy_diff_max":     emb_err_stats["entropy_diff_max"],
    }


def build_embsim_epoch_callback(
    embsim_accumulator: Dict[str, Any],
    model,
    tokenizer,
    device,
    pre_prompt: Optional[str],
    target_tokens_list: List[int],
    target_length: int,
    embedding_weights,
    all_tokens,
    batch_size: int,
    similarity: str,
    teacher_forcing: bool = False,
    embsim_temperature: float = 1.0,
) -> Callable[[int, "torch.Tensor"], Dict[str, float]]:
    """Create a per-epoch embsim validation callback for baseline training loops.

    The returned callback has signature ``(epoch: int, logits: Tensor) -> dict``.
    It:
      1. Runs :func:`run_embsim_validation_pass` with ``stgs_module=None`` (uses
         softmax of input logits directly — the "embsim_use_input_logits" path).
      2. Appends the LCS ratio to ``embsim_accumulator["embsim_lcs_ratio_history"]``.
      3. Stores the latest decoded tokens in ``embsim_accumulator["embsim_generated_tokens"]``.
      4. Returns a ``dict`` of W&B metrics that the caller should ``update`` into its
         ``wandb_log`` dict *before* calling ``wandb.log()``.

    Args:
        embsim_accumulator: Mutable dict for collecting history across epochs.
            Expected keys (initialised as empty by the caller):
            ``"embsim_lcs_ratio_history"`` and ``"embsim_generated_tokens"``.
        model: The frozen LM.
        tokenizer: Tokenizer matching the LM.
        device: Torch device.
        pre_prompt: Optional string prefix prepended before the optimised prompt during
            generation (mirrors the ``pre_prompt`` argument in the STGS path).
        target_tokens_list: Reference token-IDs list for LCS computation.
        target_length: Number of tokens to generate after the prompt.
        embedding_weights: ``model.get_input_embeddings().weight.detach()`` — full vocab.
        all_tokens: ``torch.arange(vocab_size, device=device)`` — identity mapping.
        batch_size: Passed through to :func:`run_embsim_validation_pass`.
        similarity: Similarity metric (``"l2"``, ``"cossim"``, ``"dotproduct"``).
        embsim_temperature: Passed through to :func:`run_embsim_validation_pass`.
            Temperature applied to raw logits before softmax (embsim_use_input_logits
            path). Default 1.0.
    """
    def callback(epoch: int, logits: "torch.Tensor") -> Dict[str, float]:
        _, gen_tokens, _, emb_err_stats = run_embsim_validation_pass(
            logits.detach(),
            embedding_weights,
            all_tokens,
            model,
            tokenizer,
            target_length,
            device,
            pre_prompt,
            stgs_module=None,
            batch_size=batch_size,
            similarity=similarity,
            teacher_forcing=teacher_forcing,
            target_tokens=target_tokens_list,
            embsim_temperature=embsim_temperature,
        )
        embsim_lcs = lcs_length(gen_tokens, target_tokens_list) / max(len(target_tokens_list), 1)
        embsim_accumulator["embsim_lcs_ratio_history"].append(embsim_lcs)
        embsim_accumulator["embsim_generated_tokens"] = gen_tokens
        return build_embsim_wandb_metrics(emb_err_stats, embsim_lcs)

    return callback
