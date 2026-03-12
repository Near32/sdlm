"""STGS token diversity analysis tool.

Samples N times from STGS at multiple temperatures and computes:
  - Distribution-shape metrics: entropy, effective vocab size
  - Empirical diversity metrics: distinct-1/2, mean pairwise Hamming distance
  - STGS diagnostics: signal-to-noise ratio (SNR) mean/min/max

Produces a 3-panel report (metrics curve + heatmap + decoded samples) saved as PNG.
Can be wired as a per_epoch_callback in optimize_inputs via build_diversity_callback.

CLI usage:
    python diversity_analysis.py \\
        --model HuggingFaceTB/SmolLM2-135M \\
        --temperatures 0.1,0.5,1.0,2.0 \\
        --n_samples 20 --seq_len 20 \\
        --logits_source random \\
        --output_dir ./results/diversity/
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F


def str2bool(v):
    """Convert common CLI truthy/falsey strings to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


_STGS_CLASS = None


def _get_stgs_class():
    """Load STGS directly from sdlm/stgs.py to avoid package side effects."""
    global _STGS_CLASS
    if _STGS_CLASS is not None:
        return _STGS_CLASS

    stgs_path = Path(__file__).resolve().parents[2] / "sdlm" / "stgs.py"
    if not stgs_path.exists():
        raise FileNotFoundError(f"Could not find STGS module at: {stgs_path}")

    spec = importlib.util.spec_from_file_location("sdlm_stgs_module", str(stgs_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for: {stgs_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "STGS"):
        raise AttributeError(f"Module {stgs_path} does not define STGS")

    _STGS_CLASS = module.STGS
    return _STGS_CLASS


def _snapshot_rng_state(device: torch.device) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {"cpu": torch.random.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device=device)
    return state


def _restore_rng_state(device: torch.device, state: Dict[str, torch.Tensor]) -> None:
    torch.random.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device=device)


def _preprocess_logits_like_stgs(
    logits: torch.Tensor,
    logits_normalize: str,
    stgs_input_dropout: float,
    eps: float,
) -> torch.Tensor:
    """Mirror STGS preprocessing (normalize + input dropout) for diagnostics."""
    x = logits
    if logits_normalize != "none":
        x = x - x.mean(dim=-1, keepdim=True)
        if logits_normalize == "zscore":
            x = x / (x.std(dim=-1, keepdim=True) + eps)
        elif logits_normalize != "center":
            raise ValueError("--logits_normalize must be one of: none, center, zscore")
    if stgs_input_dropout > 0.0:
        x = F.dropout(x, p=stgs_input_dropout, training=True)
    return x


# ---------------------------------------------------------------------------
# Core sampling
# ---------------------------------------------------------------------------

def _gumbel_sample(
    logits: torch.Tensor,
    temperature: float,
    noise_scale: float = 1.0,
    gradient_estimator: str = "stgs",
    stgs_hard: bool = True,
    stgs_hard_method: str = "categorical",
    stgs_hard_embsim_probs: str = "gumbel_soft",
    stgs_hard_embsim_strategy: str = "nearest",
    stgs_hard_embsim_top_k: int = 8,
    stgs_hard_embsim_rerank_alpha: float = 0.5,
    stgs_hard_embsim_sample_tau: float = 1.0,
    stgs_hard_embsim_margin: float = 0.0,
    stgs_hard_embsim_fallback: str = "argmax",
    logits_normalize: str = "none",
    stgs_input_dropout: float = 0.0,
    stgs_output_dropout: float = 0.0,
    embedding_weights: Optional[torch.Tensor] = None,
    compute_empirical_snr: bool = True,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor | float]]]:
    """Sample from Gumbel-softmax given batched logits.

    Args:
        logits: (N, seq_len, vocab) — will be mutated in-place with Gumbel noise.
        temperature: softmax temperature.
        noise_scale: scale applied to Gumbel noise (0 = no noise → argmax).
        gradient_estimator: "stgs" (Gumbel-Softmax) or "reinforce" (softmax logits).
        stgs_hard: kept for interface parity with STGS; ids are always discrete samples.
        stgs_hard_method: hard token selection method ("categorical" or embsim-*).
        stgs_hard_embsim_probs: "gumbel_soft" or "input_logits" probability source.
        stgs_hard_embsim_strategy: embsim hard-selection strategy.
        stgs_hard_embsim_top_k: candidate size for top-k embsim strategies.
        stgs_hard_embsim_rerank_alpha: blend weight for top-k rerank.
        stgs_hard_embsim_sample_tau: embsim sampling temperature for top-k sample.
        stgs_hard_embsim_margin: margin threshold for ambiguity fallback.
        stgs_hard_embsim_fallback: fallback mode ("argmax" or "categorical").
        logits_normalize: "none", "center", or "zscore" over vocab dim.
        stgs_input_dropout: dropout on logits before noise/softmax.
        stgs_output_dropout: dropout on y_soft before hard selection.
        embedding_weights: optional token embeddings for embsim-* hard methods.
        compute_empirical_snr: whether to compute empirical Gumbel-noise variance and
            empirical SNR using sampled noise.
        eps: clamping epsilon for numerical stability.

    Returns:
        y_soft: (N, seq_len, vocab) soft probabilities after Gumbel-softmax.
        sampled_ids: (N, seq_len) hard token indices.
        stgs_stats: dict with STGS diagnostics, or None for non-STGS estimator.
    """
    x = logits

    stgs_stats: Optional[Dict[str, torch.Tensor | float]] = None
    if gradient_estimator == "stgs":
        STGS = _get_stgs_class()
        if stgs_hard_method in ("embsim-dot", "embsim-cos", "embsim-l2") and embedding_weights is None:
            raise ValueError(
                f"--stgs_hard_method={stgs_hard_method} requires embedding weights "
                "(provide --model or logits_source=model)."
            )
        if compute_empirical_snr and stgs_output_dropout > 0.0:
            raise NotImplementedError(
                "Empirical gumbel-noise variance cannot be recovered when --stgs_output_dropout > 0. "
                "Set --stgs_output_dropout=0 or --stgs_snr_empirical=False."
            )

        emb_weights = None
        if embedding_weights is not None:
            if embedding_weights.dim() != 2:
                raise ValueError(
                    f"embedding_weights must be rank-2 [vocab, dim], got shape {tuple(embedding_weights.shape)}"
                )
            if embedding_weights.shape[0] != logits.shape[-1]:
                raise ValueError(
                    "embedding_weights vocab mismatch: "
                    f"weights vocab={embedding_weights.shape[0]} vs logits vocab={logits.shape[-1]}"
                )
            emb_weights = embedding_weights.to(device=logits.device, dtype=logits.dtype)

        # Snapshot RNG so diagnostics use exactly the same normalization/dropout randomness
        # that STGS forward will use internally.
        device = logits.device
        rng_state = _snapshot_rng_state(device)
        x_for_snr = _preprocess_logits_like_stgs(
            logits=logits,
            logits_normalize=logits_normalize,
            stgs_input_dropout=stgs_input_dropout,
            eps=eps,
        )
        _restore_rng_state(device, rng_state)

        stgs_module = STGS(
            vocab_size=logits.shape[-1],
            stgs_hard=stgs_hard,
            init_temperature=float(temperature),
            learnable_temperature=False,
            input_dropout=stgs_input_dropout,
            output_dropout=stgs_output_dropout,
            stgs_hard_method=stgs_hard_method,
            stgs_hard_embsim_probs=stgs_hard_embsim_probs,
            stgs_hard_embsim_strategy=stgs_hard_embsim_strategy,
            stgs_hard_embsim_top_k=stgs_hard_embsim_top_k,
            stgs_hard_embsim_rerank_alpha=stgs_hard_embsim_rerank_alpha,
            stgs_hard_embsim_sample_tau=stgs_hard_embsim_sample_tau,
            stgs_hard_embsim_margin=stgs_hard_embsim_margin,
            stgs_hard_embsim_fallback=stgs_hard_embsim_fallback,
            logits_normalize=logits_normalize,
            eps=eps,
            device=str(device),
        )
        stgs_module.train()
        sampled_ids_st, _, _, y_soft = stgs_module.forward(
            logits,
            temperature=float(temperature),
            gumbel_noise_scale=float(noise_scale),
            embedding_weights=emb_weights,
        )
        sampled_ids = sampled_ids_st.detach().long()

        # SNR definition requested by user:
        # signal = x/T, noise = (eps_gumbel * gumbel)/T.
        # Equivalent T-free ratio:
        # Var(x/T)/Var((eps_gumbel*g)/T) = Var(x)/Var(eps_gumbel*g).
        signal_var_pos = x_for_snr.float().var(dim=-1).mean(dim=0).detach()  # (seq_len,)
        gumbel_noise_var_theoretical = float((noise_scale ** 2) * (math.pi ** 2 / 6.0))
        stgs_snr_pos_theoretical = signal_var_pos / max(gumbel_noise_var_theoretical, eps)

        if compute_empirical_snr:
            noise_up_to_const = (
                float(temperature) * torch.log(y_soft.float().clamp_min(eps))
                - x_for_snr.float()
            )
            # Softmax has additive invariance; per-position variance is invariant to that shift.
            gumbel_noise_var_empirical_pos = noise_up_to_const.var(dim=-1).mean(dim=0).detach()
            stgs_snr_pos = signal_var_pos / gumbel_noise_var_empirical_pos.clamp_min(eps)
        else:
            gumbel_noise_var_empirical_pos = None
            stgs_snr_pos = None

        stgs_stats = {
            "stgs_snr_pos": stgs_snr_pos,
            "stgs_snr_pos_theoretical": stgs_snr_pos_theoretical,
            "gumbel_noise_var_empirical_pos": gumbel_noise_var_empirical_pos,
            "gumbel_noise_var_theoretical": gumbel_noise_var_theoretical,
        }
        return y_soft, sampled_ids, stgs_stats
    elif gradient_estimator == "reinforce":
        x = _preprocess_logits_like_stgs(
            logits=logits,
            logits_normalize=logits_normalize,
            stgs_input_dropout=stgs_input_dropout,
            eps=eps,
        )
        y_soft = F.softmax(x / temperature, dim=-1)
    else:
        raise ValueError("gradient_estimator must be 'stgs' or 'reinforce'")

    if stgs_output_dropout > 0.0:
        y_soft = F.dropout(y_soft, p=stgs_output_dropout, training=True)

    if stgs_hard_embsim_probs == "input_logits":
        sample_probs = F.softmax(x, dim=-1)
    elif stgs_hard_embsim_probs == "gumbel_soft":
        sample_probs = y_soft
    else:
        raise ValueError("stgs_hard_embsim_probs must be 'gumbel_soft' or 'input_logits'")

    if stgs_hard_method in ("embsim-dot", "embsim-cos", "embsim-l2") and embedding_weights is not None:
        soft_emb = torch.matmul(sample_probs.float(), embedding_weights.float())  # (N,S,D)
        if stgs_hard_method == "embsim-dot":
            emb_scores = torch.matmul(soft_emb, embedding_weights.float().T)
        elif stgs_hard_method == "embsim-l2":
            soft_sq = (soft_emb ** 2).sum(dim=-1, keepdim=True)
            emb_sq = (embedding_weights.float() ** 2).sum(dim=-1)
            cross = torch.matmul(soft_emb, embedding_weights.float().T)
            l2_sq = soft_sq - 2.0 * cross + emb_sq
            emb_scores = -l2_sq
        else:  # embsim-cos
            soft_norm = F.normalize(soft_emb, dim=-1, eps=eps)
            emb_norm = F.normalize(embedding_weights.float(), dim=-1, eps=eps)
            emb_scores = torch.matmul(soft_norm, emb_norm.T)

        if stgs_hard_embsim_strategy == "nearest":
            sampled_ids = emb_scores.argmax(dim=-1)
        elif stgs_hard_embsim_strategy == "topk_rerank":
            k = max(1, min(int(stgs_hard_embsim_top_k), emb_scores.shape[-1]))
            cand_ids = emb_scores.topk(k, dim=-1).indices
            cand_emb_scores = emb_scores.gather(dim=-1, index=cand_ids)
            cand_lm_scores = sample_probs.gather(dim=-1, index=cand_ids)
            alpha = max(0.0, min(float(stgs_hard_embsim_rerank_alpha), 1.0))
            emb_std = cand_emb_scores.std(dim=-1, keepdim=True, unbiased=False)
            lm_std = cand_lm_scores.std(dim=-1, keepdim=True, unbiased=False)
            cand_emb_z = (cand_emb_scores - cand_emb_scores.mean(dim=-1, keepdim=True)) / (emb_std + eps)
            cand_lm_z = (cand_lm_scores - cand_lm_scores.mean(dim=-1, keepdim=True)) / (lm_std + eps)
            fused = alpha * cand_emb_z + (1.0 - alpha) * cand_lm_z
            best_local = fused.argmax(dim=-1, keepdim=True)
            sampled_ids = cand_ids.gather(dim=-1, index=best_local).squeeze(-1)
        elif stgs_hard_embsim_strategy == "topk_sample":
            k = max(1, min(int(stgs_hard_embsim_top_k), emb_scores.shape[-1]))
            cand_ids = emb_scores.topk(k, dim=-1).indices
            cand_emb_scores = emb_scores.gather(dim=-1, index=cand_ids)
            tau = max(float(stgs_hard_embsim_sample_tau), eps)
            cand_probs = F.softmax(cand_emb_scores / tau, dim=-1)
            sampled_local = torch.distributions.Categorical(probs=cand_probs).sample().unsqueeze(-1)
            sampled_ids = cand_ids.gather(dim=-1, index=sampled_local).squeeze(-1)
        elif stgs_hard_embsim_strategy == "margin_fallback":
            nearest_ids = emb_scores.argmax(dim=-1)
            if emb_scores.shape[-1] < 2:
                sampled_ids = nearest_ids
            else:
                top2 = emb_scores.topk(2, dim=-1)
                margin = top2.values[..., 0] - top2.values[..., 1]
                is_ambiguous = margin < float(stgs_hard_embsim_margin)
                if stgs_hard_embsim_fallback == "categorical":
                    fallback_ids = torch.distributions.Categorical(probs=sample_probs).sample()
                else:
                    fallback_ids = sample_probs.argmax(dim=-1)
                sampled_ids = torch.where(is_ambiguous, fallback_ids, nearest_ids)
        elif stgs_hard_embsim_strategy == "lm_topk_restrict":
            k = max(1, min(int(stgs_hard_embsim_top_k), emb_scores.shape[-1]))
            lm_cand_ids = sample_probs.topk(k, dim=-1).indices
            cand_emb_scores = emb_scores.gather(dim=-1, index=lm_cand_ids)
            best_local = cand_emb_scores.argmax(dim=-1, keepdim=True)
            sampled_ids = lm_cand_ids.gather(dim=-1, index=best_local).squeeze(-1)
        else:
            raise ValueError(
                "stgs_hard_embsim_strategy must be one of: "
                "nearest, topk_rerank, topk_sample, margin_fallback, lm_topk_restrict"
            )
    elif stgs_hard_method == "categorical" or not stgs_hard:
        sampled_ids = torch.distributions.Categorical(probs=y_soft).sample()
    else:
        if stgs_hard_method in ("embsim-dot", "embsim-cos", "embsim-l2"):
            raise ValueError(
                f"--stgs_hard_method={stgs_hard_method} requires embedding weights "
                "(provide --model or logits_source=model)."
            )
        raise NotImplementedError(f"Unsupported stgs_hard_method: {stgs_hard_method}")
    return y_soft, sampled_ids, stgs_stats


def sample_diversity(
    logits: torch.Tensor,
    temperatures: List[float],
    n_samples: int,
    noise_scale: float = 1.0,
    gradient_estimator: str = "stgs",
    stgs_hard: bool = True,
    stgs_hard_method: str = "categorical",
    stgs_hard_embsim_probs: str = "gumbel_soft",
    stgs_hard_embsim_strategy: str = "nearest",
    stgs_hard_embsim_top_k: int = 8,
    stgs_hard_embsim_rerank_alpha: float = 0.5,
    stgs_hard_embsim_sample_tau: float = 1.0,
    stgs_hard_embsim_margin: float = 0.0,
    stgs_hard_embsim_fallback: str = "argmax",
    logits_normalize: str = "none",
    stgs_input_dropout: float = 0.0,
    stgs_output_dropout: float = 0.0,
    embedding_weights: Optional[torch.Tensor] = None,
    compute_empirical_snr: bool = True,
    eps: float = 1e-12,
) -> Dict[float, Dict[str, torch.Tensor]]:
    """Sample n_samples sequences at each temperature via Gumbel-softmax.

    Args:
        logits: (1, seq_len, vocab) — the current learnable prompt logits.
        temperatures: list of temperatures to sweep.
        n_samples: number of independent Gumbel draws per temperature.
        noise_scale: Gumbel noise amplitude.
        gradient_estimator: "stgs" or "reinforce".
        stgs_hard: kept for STGS interface parity (does not change ids in this tool).
        stgs_hard_method: hard selection method ("categorical" or embsim-*).
        stgs_hard_embsim_probs: probability source ("gumbel_soft" or "input_logits").
        stgs_hard_embsim_strategy: embsim hard-selection strategy.
        stgs_hard_embsim_top_k: candidate size for top-k embsim strategies.
        stgs_hard_embsim_rerank_alpha: blend weight for top-k rerank.
        stgs_hard_embsim_sample_tau: embsim sampling temperature for top-k sample.
        stgs_hard_embsim_margin: margin threshold for ambiguity fallback.
        stgs_hard_embsim_fallback: fallback mode ("argmax" or "categorical").
        logits_normalize: logit normalization mode before Gumbel noise.
        stgs_input_dropout: dropout on logits before Gumbel/softmax.
        stgs_output_dropout: dropout on y_soft before hard selection.
        embedding_weights: token embedding matrix for embsim-* hard methods.
        compute_empirical_snr: whether to compute empirical noise-variance SNR metrics.
        eps: numerical stability epsilon.

    Returns:
        Dict mapping temperature → sampled tensors and optional STGS diagnostics.
    """
    results: Dict[float, Dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for temp in temperatures:
            expanded = logits.expand(n_samples, -1, -1).clone()
            y_soft, sampled_ids, stgs_stats = _gumbel_sample(
                expanded,
                temp,
                noise_scale=noise_scale,
                gradient_estimator=gradient_estimator,
                stgs_hard=stgs_hard,
                stgs_hard_method=stgs_hard_method,
                stgs_hard_embsim_probs=stgs_hard_embsim_probs,
                stgs_hard_embsim_strategy=stgs_hard_embsim_strategy,
                stgs_hard_embsim_top_k=stgs_hard_embsim_top_k,
                stgs_hard_embsim_rerank_alpha=stgs_hard_embsim_rerank_alpha,
                stgs_hard_embsim_sample_tau=stgs_hard_embsim_sample_tau,
                stgs_hard_embsim_margin=stgs_hard_embsim_margin,
                stgs_hard_embsim_fallback=stgs_hard_embsim_fallback,
                logits_normalize=logits_normalize,
                stgs_input_dropout=stgs_input_dropout,
                stgs_output_dropout=stgs_output_dropout,
                embedding_weights=embedding_weights,
                compute_empirical_snr=compute_empirical_snr,
                eps=eps,
            )
            results[temp] = {
                "y_soft": y_soft.cpu(),
                "sampled_ids": sampled_ids.cpu(),
                "stgs_snr_pos": stgs_stats["stgs_snr_pos"].cpu() if (stgs_stats is not None and stgs_stats["stgs_snr_pos"] is not None) else None,
                "stgs_snr_pos_theoretical": stgs_stats["stgs_snr_pos_theoretical"].cpu() if (stgs_stats is not None and stgs_stats["stgs_snr_pos_theoretical"] is not None) else None,
                "gumbel_noise_var_empirical_pos": stgs_stats["gumbel_noise_var_empirical_pos"].cpu() if (stgs_stats is not None and stgs_stats["gumbel_noise_var_empirical_pos"] is not None) else None,
                "gumbel_noise_var_theoretical": float(stgs_stats["gumbel_noise_var_theoretical"]) if (stgs_stats is not None and stgs_stats["gumbel_noise_var_theoretical"] is not None) else None,
            }
    return results


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def compute_diversity_metrics(
    results_by_temp: Dict[float, Dict[str, torch.Tensor]],
    tokenizer,
) -> Dict[float, Dict[str, float]]:
    """Compute diversity metrics for each temperature.

    Metrics:
        entropy: mean per-position entropy of the aggregate distribution.
        eff_vocab_size: exp(entropy) — number of equally-likely tokens equivalent.
        distinct_1: unique unigram types / total unigrams across all samples.
        distinct_2: unique bigram types / total bigrams across all samples.
        mean_hamming: mean pairwise Hamming distance (fraction of differing positions).
        stgs_snr_mean/min/max: SNR summary stats using empirical gumbel noise variance.
        stgs_snr_theoretical_mean/min/max: SNR summary stats using theoretical noise variance.
        gumbel_noise_var_theoretical: theoretical Var(eps_gumbel * Gumbel(0,1)).
        gumbel_noise_var_empirical_mean/min/max: empirical variance summary from sampled noise.

    Args:
        results_by_temp: output of sample_diversity.
        tokenizer: HuggingFace tokenizer (used to decode token ids for distinct metrics).

    Returns:
        Dict mapping temperature → metric_name → float value.
    """
    eps = 1e-12
    metrics: Dict[float, Dict[str, float]] = {}

    for temp, data in results_by_temp.items():
        y_soft = data["y_soft"]        # (N, S, V)
        sampled_ids = data["sampled_ids"]  # (N, S)
        N, S = sampled_ids.shape

        # --- entropy & eff_vocab_size (over aggregate distribution) ---
        mean_probs = y_soft.mean(dim=0)                         # (S, V)
        token_entropies = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=-1)  # (S,)
        entropy = token_entropies.mean().item()
        eff_vocab_size = math.exp(entropy)

        # --- distinct-1 (unigrams) ---
        all_ids = sampled_ids.flatten().tolist()                # N*S flat list
        unigrams = set(all_ids)
        distinct_1 = len(unigrams) / max(len(all_ids), 1)

        # --- distinct-2 (bigrams) ---
        bigrams: set = set()
        total_bigrams = 0
        for seq in sampled_ids.tolist():                        # (S,) per sample
            for i in range(len(seq) - 1):
                bigrams.add((seq[i], seq[i + 1]))
                total_bigrams += 1
        distinct_2 = len(bigrams) / max(total_bigrams, 1)

        # --- mean pairwise Hamming distance ---
        if N > 1:
            hamming_sum = 0.0
            pair_count = 0
            for i, j in combinations(range(N), 2):
                diff = (sampled_ids[i] != sampled_ids[j]).float().mean().item()
                hamming_sum += diff
                pair_count += 1
            mean_hamming = hamming_sum / pair_count
        else:
            mean_hamming = 0.0

        metrics[temp] = {
            "entropy": entropy,
            "eff_vocab_size": eff_vocab_size,
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "mean_hamming": mean_hamming,
        }
        snr_pos = data.get("stgs_snr_pos")
        if snr_pos is not None:
            snr_f = snr_pos.float()
            metrics[temp]["stgs_snr_mean"] = snr_f.mean().item()
            metrics[temp]["stgs_snr_min"] = snr_f.min().item()
            metrics[temp]["stgs_snr_max"] = snr_f.max().item()

        snr_pos_theoretical = data.get("stgs_snr_pos_theoretical")
        if snr_pos_theoretical is not None:
            snr_th_f = snr_pos_theoretical.float()
            metrics[temp]["stgs_snr_theoretical_mean"] = snr_th_f.mean().item()
            metrics[temp]["stgs_snr_theoretical_min"] = snr_th_f.min().item()
            metrics[temp]["stgs_snr_theoretical_max"] = snr_th_f.max().item()

        gumbel_var_theoretical = data.get("gumbel_noise_var_theoretical")
        if gumbel_var_theoretical is not None:
            metrics[temp]["gumbel_noise_var_theoretical"] = float(gumbel_var_theoretical)

        gumbel_var_emp_pos = data.get("gumbel_noise_var_empirical_pos")
        if gumbel_var_emp_pos is not None:
            gumbel_var_emp_f = gumbel_var_emp_pos.float()
            metrics[temp]["gumbel_noise_var_empirical_mean"] = gumbel_var_emp_f.mean().item()
            metrics[temp]["gumbel_noise_var_empirical_min"] = gumbel_var_emp_f.min().item()
            metrics[temp]["gumbel_noise_var_empirical_max"] = gumbel_var_emp_f.max().item()

    return metrics


def _ordered_metric_names(metrics_by_temp: Dict[float, Dict[str, float]]) -> List[str]:
    preferred = [
        "entropy", "eff_vocab_size", "distinct_1", "distinct_2", "mean_hamming",
        "stgs_snr_mean", "stgs_snr_min", "stgs_snr_max",
        "stgs_snr_theoretical_mean", "stgs_snr_theoretical_min", "stgs_snr_theoretical_max",
        "gumbel_noise_var_theoretical",
        "gumbel_noise_var_empirical_mean", "gumbel_noise_var_empirical_min", "gumbel_noise_var_empirical_max",
    ]
    first_metrics = next(iter(metrics_by_temp.values()))
    ordered = [m for m in preferred if m in first_metrics]
    ordered.extend([m for m in first_metrics.keys() if m not in ordered])
    return ordered


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_diversity_report(
    results_by_temp: Dict[float, Dict[str, torch.Tensor]],
    metrics_by_temp: Dict[float, Dict[str, float]],
    tokenizer,
    top_k: int = 8,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Produce a 3-panel diversity report figure.

    Panel 1 (top): Metrics curves (normalized to [0,1]) vs temperature.
    Panel 2 (middle): Per-temperature heatmaps — top-k tokens × seq_len.
    Panel 3 (bottom): Decoded sample text table.

    Args:
        results_by_temp: output of sample_diversity.
        metrics_by_temp: output of compute_diversity_metrics.
        tokenizer: HuggingFace tokenizer for decoding.
        top_k: number of ranked token rows to show in the heatmap.
        output_path: if given, save PNG here.

    Returns:
        The matplotlib Figure (caller should plt.close() it when done).
    """
    temperatures = sorted(results_by_temp.keys())
    n_temps = len(temperatures)

    # Layout: 4 rows — (1) metrics curve, (2) heatmaps, (3) sample text, (4) metrics table
    fig = plt.figure(figsize=(max(6 * n_temps, 12), 16))
    gs = fig.add_gridspec(
        4, n_temps,
        height_ratios=[2, 3, 2, 1.3],
        hspace=0.45, wspace=0.35,
    )

    # ---- Panel 1: metrics curve (spans all columns) ----------------------
    ax_metrics = fig.add_subplot(gs[0, :])
    metric_names = _ordered_metric_names(metrics_by_temp)
    metric_values: Dict[str, List[float]] = {m: [] for m in metric_names}
    for temp in temperatures:
        for m in metric_names:
            metric_values[m].append(metrics_by_temp[temp][m])

    # Normalize each metric to [0,1] for overlay
    for m in metric_names:
        vals = metric_values[m]
        vmin, vmax = min(vals), max(vals)
        span = vmax - vmin if vmax != vmin else 1.0
        norm_vals = [(v - vmin) / span for v in vals]
        ax_metrics.plot(temperatures, norm_vals, marker="o", label=m)

    ax_metrics.set_xlabel("Temperature (τ)")
    ax_metrics.set_ylabel("Normalized metric [0,1]")
    ax_metrics.set_title("Diversity metrics vs temperature (normalized)")
    ax_metrics.legend(loc="upper left", fontsize=8)
    ax_metrics.grid(True, alpha=0.3)

    # ---- Panel 2: heatmaps -----------------------------------------------
    for col, temp in enumerate(temperatures):
        ax = fig.add_subplot(gs[1, col])
        y_soft = results_by_temp[temp]["y_soft"]  # (N, S, V)
        mean_probs = y_soft.mean(dim=0)             # (S, V)
        seq_len = mean_probs.shape[0]

        # Build (top_k, seq_len) probability matrix
        sorted_idx = torch.argsort(mean_probs, dim=-1, descending=True)[:, :top_k]  # (S, top_k)
        prob_matrix = torch.zeros(top_k, seq_len)
        token_matrix: List[List[str]] = [[""] * seq_len for _ in range(top_k)]

        for pos in range(seq_len):
            for rank in range(top_k):
                tok_id = sorted_idx[pos, rank].item()
                prob = mean_probs[pos, tok_id].item()
                prob_matrix[rank, pos] = prob
                # Decode token; truncate to ~6 chars
                try:
                    tok_str = tokenizer.decode([tok_id]).strip()
                    tok_str = tok_str[:6] if tok_str else f"#{tok_id}"
                except Exception:
                    tok_str = f"#{tok_id}"
                token_matrix[rank][pos] = tok_str

        im = ax.imshow(
            prob_matrix.numpy(),
            cmap="YlOrRd",
            vmin=0.0,
            vmax=float(prob_matrix.max().item()) if prob_matrix.max().item() > 0 else 1.0,
            aspect="auto",
        )
        # Annotate cells
        for rank in range(top_k):
            for pos in range(seq_len):
                ax.text(
                    pos, rank, token_matrix[rank][pos],
                    ha="center", va="center", fontsize=5,
                    color="black",
                )
        ax.set_title(f"τ={temp}", fontsize=9)
        ax.set_xlabel("Position", fontsize=7)
        ax.set_ylabel("Rank" if col == 0 else "", fontsize=7)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([str(r) for r in range(top_k)], fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- Panel 3: decoded sample text table -------------------------------
    # Decode a few samples per temperature; show as a simple text grid
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis("off")

    n_show = min(5, results_by_temp[temperatures[0]]["sampled_ids"].shape[0])
    col_width = 1.0 / n_temps
    row_height = 0.85 / (n_show + 1)

    # Header
    for c, temp in enumerate(temperatures):
        ax_text.text(
            col_width * (c + 0.5), 0.95,
            f"τ={temp}",
            ha="center", va="top",
            fontsize=8, fontweight="bold",
            transform=ax_text.transAxes,
        )
    # Rows
    for row in range(n_show):
        for col, temp in enumerate(temperatures):
            ids = results_by_temp[temp]["sampled_ids"][row].tolist()
            try:
                text = tokenizer.decode(ids, skip_special_tokens=True)
            except Exception:
                text = str(ids[:8])
            # Truncate long text
            if len(text) > 50:
                text = text[:47] + "..."
            y_pos = 0.95 - (row + 1) * row_height
            ax_text.text(
                col_width * (col + 0.05), y_pos,
                text,
                ha="left", va="top",
                fontsize=6,
                fontfamily="monospace",
                transform=ax_text.transAxes,
                wrap=False,
            )

    ax_text.set_title("Decoded samples per temperature", fontsize=9)

    # ---- Panel 4: metrics table text ---------------------------------------
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis("off")

    metric_names = _ordered_metric_names(metrics_by_temp)
    header = f"{'Metric':<20}" + "".join(f"  τ={t:<8}" for t in temperatures)
    lines = [header, "-" * len(header)]
    for m in metric_names:
        row = f"{m:<20}" + "".join(f"  {metrics_by_temp[t][m]:<10.4f}" for t in temperatures)
        lines.append(row)
    table_text = "\n".join(lines)

    ax_table.text(
        0.01, 0.98,
        table_text,
        ha="left", va="top",
        fontsize=8,
        fontfamily="monospace",
        transform=ax_table.transAxes,
    )
    ax_table.set_title("Diversity metrics table", fontsize=9)

    # ---- Save / W&B log --------------------------------------------------
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    try:
        import wandb
        if wandb.run is not None:
            flat_metrics = {
                f"diversity/{temp}/{k}": v
                for temp, d in metrics_by_temp.items()
                for k, v in d.items()
            }
            for temp, data in results_by_temp.items():
                snr_pos = data.get("stgs_snr_pos")
                if snr_pos is not None:
                    for i, v in enumerate(snr_pos.float().tolist()):
                        flat_metrics[f"diversity/{temp}/stgs_snr_pos_{i}"] = float(v)
                snr_pos_theoretical = data.get("stgs_snr_pos_theoretical")
                if snr_pos_theoretical is not None:
                    for i, v in enumerate(snr_pos_theoretical.float().tolist()):
                        flat_metrics[f"diversity/{temp}/stgs_snr_pos_theoretical_{i}"] = float(v)
                gumbel_var_emp = data.get("gumbel_noise_var_empirical_pos")
                if gumbel_var_emp is not None:
                    for i, v in enumerate(gumbel_var_emp.float().tolist()):
                        flat_metrics[f"diversity/{temp}/gumbel_noise_var_empirical_pos_{i}"] = float(v)
            wandb.log({"diversity/report": wandb.Image(fig), **flat_metrics})
    except Exception:
        pass

    return fig


# ---------------------------------------------------------------------------
# Callback factory
# ---------------------------------------------------------------------------

def build_diversity_callback(
    tokenizer,
    temperatures: List[float],
    n_samples: int,
    output_dir: str,
    log_every: int = 10,
    top_k: int = 8,
    noise_scale: float = 1.0,
) -> Callable[[int, torch.Tensor], dict]:
    """Build a per_epoch_callback for the optimize_inputs loop.

    Args:
        tokenizer: HuggingFace tokenizer.
        temperatures: list of temperatures to sweep.
        n_samples: number of Gumbel draws per temperature.
        output_dir: directory to save PNG files.
        log_every: run analysis every N epochs (skips otherwise).
        top_k: rows in heatmap.
        noise_scale: Gumbel noise amplitude.

    Returns:
        Callable (epoch: int, logits: Tensor) -> dict of W&B metrics.
    """
    def callback(epoch: int, logits: torch.Tensor) -> dict:
        if epoch % log_every != 0:
            return {}
        results = sample_diversity(logits, temperatures, n_samples, noise_scale)
        metrics = compute_diversity_metrics(results, tokenizer)
        path = Path(output_dir) / f"diversity_epoch_{epoch:04d}.png"
        fig = plot_diversity_report(results, metrics, tokenizer, top_k=top_k, output_path=str(path))
        plt.close(fig)
        flat = {
            f"diversity/{temp}/{k}": v
            for temp, d in metrics.items()
            for k, v in d.items()
        }
        for temp, data in results.items():
            snr_pos = data.get("stgs_snr_pos")
            if snr_pos is not None:
                for i, v in enumerate(snr_pos.float().tolist()):
                    flat[f"diversity/{temp}/stgs_snr_pos_{i}"] = float(v)
            snr_pos_theoretical = data.get("stgs_snr_pos_theoretical")
            if snr_pos_theoretical is not None:
                for i, v in enumerate(snr_pos_theoretical.float().tolist()):
                    flat[f"diversity/{temp}/stgs_snr_pos_theoretical_{i}"] = float(v)
            gumbel_var_emp = data.get("gumbel_noise_var_empirical_pos")
            if gumbel_var_emp is not None:
                for i, v in enumerate(gumbel_var_emp.float().tolist()):
                    flat[f"diversity/{temp}/gumbel_noise_var_empirical_pos_{i}"] = float(v)
        return flat

    return callback


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_dummy_tokenizer(vocab_size: int):
    """Minimal tokenizer-like object for CLI use without a real model."""
    class _DummyTok:
        def decode(self, ids, skip_special_tokens=False):
            return " ".join(f"t{i}" for i in ids)
    return _DummyTok()


def _resolve_sos_token_id(tokenizer) -> tuple[int, str]:
    """Resolve a start-of-sequence token id with a clear fallback chain."""
    if getattr(tokenizer, "bos_token_id", None) is not None:
        return int(tokenizer.bos_token_id), "bos_token_id"
    if getattr(tokenizer, "cls_token_id", None) is not None:
        return int(tokenizer.cls_token_id), "cls_token_id"
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return int(tokenizer.eos_token_id), "eos_token_id"
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return int(tokenizer.pad_token_id), "pad_token_id"
    raise ValueError(
        "Could not resolve a start token id from tokenizer "
        "(tried bos/cls/eos/pad token ids)."
    )


def _build_logits_from_model(model_id: str, seq_len: int) -> tuple[torch.Tensor, object, torch.Tensor]:
    """Build logits by greedy autoregressive rollout from SoS using MODEL_ID."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sos_token_id, sos_source = _resolve_sos_token_id(tokenizer)
    print(f"[info] logits_source=model using {sos_source}={sos_token_id} on device={device}")

    # Prefix starts with SoS. At each step, capture next-token logits and append argmax token.
    prefix_ids = torch.tensor([[sos_token_id]], device=device, dtype=torch.long)
    logits_steps: List[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(seq_len):
            outputs = model(input_ids=prefix_ids, use_cache=False)
            next_logits = outputs.logits[:, -1, :]  # (1, vocab)
            logits_steps.append(next_logits.detach().cpu())
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # (1, 1)
            prefix_ids = torch.cat([prefix_ids, next_token], dim=1)

    logits = torch.stack(logits_steps, dim=1)  # (1, seq_len, vocab)
    embedding_weights = model.get_input_embeddings().weight.detach().cpu()
    return logits.cpu(), tokenizer, embedding_weights


def _print_metrics_table(metrics_by_temp: Dict[float, Dict[str, float]]) -> None:
    temps = sorted(metrics_by_temp.keys())
    metric_names = _ordered_metric_names(metrics_by_temp)
    header = f"{'Metric':<20}" + "".join(f"  τ={t:<8}" for t in temps)
    print(header)
    print("-" * len(header))
    for m in metric_names:
        row = f"{m:<20}" + "".join(f"  {metrics_by_temp[t][m]:<10.4f}" for t in temps)
        print(row)


def _apply_stgs_logit_processing(
    logits: torch.Tensor,
    logits_top_k: int,
    logits_top_p: float,
    logits_filter_penalty: float,
) -> torch.Tensor:
    """Apply STGS-compatible top-k / top-p filtering before sampling."""
    if logits_top_k < 0:
        raise ValueError("--logits_top_k must be >= 0")
    if not (0.0 < logits_top_p <= 1.0):
        raise ValueError("--logits_top_p must be in (0, 1]")

    processed = logits.clone()

    vocab_size = processed.shape[-1]
    if 0 < logits_top_k < vocab_size:
        topk_vals = torch.topk(processed, k=min(logits_top_k, vocab_size), dim=-1).values
        kth = topk_vals[..., -1:].contiguous()
        mask = (processed < kth).float()
        processed = processed - mask * float(logits_filter_penalty)

    if logits_top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(processed, dim=-1, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = (cumulative_probs - sorted_probs) >= logits_top_p
        mask = torch.zeros_like(processed, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, sorted_mask)
        processed = processed - mask.float() * float(logits_filter_penalty)

    return processed


def main_cli():
    parser = argparse.ArgumentParser(
        description="STGS token diversity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID (optional; needed for real token decoding)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="STGS temperature (used as single value when --temperatures is omitted)")
    parser.add_argument("--temperatures", type=str, default="",
                        help="Comma-separated temperatures; if omitted, defaults to "
                             "0.1,0.5,1.0,2.0 unless --temperature is explicitly set")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of Gumbel draws per temperature")
    parser.add_argument("--seq_len", type=int, default=20,
                        help="Sequence length (used for random/model logits; ignored for .pt logits)")
    parser.add_argument("--logits_source", type=str, default="random",
                        help="'random', 'model', or path to a .pt file containing logits (1, S, V)")
    parser.add_argument("--top_k", type=int, default=8,
                        help="Ranked token rows in heatmap")
    parser.add_argument("--output_dir", type=str, default="./results/diversity/",
                        help="Directory to save the PNG report")
    parser.add_argument("--gumbel_noise_scale", "--noise_scale", dest="gumbel_noise_scale", type=float, default=1.0,
                        help="Gumbel noise amplitude (same naming as STGS scripts)")

    # STGS args with active plumbing in this diversity CLI
    parser.add_argument("--gradient_estimator", type=str, default="stgs", choices=["stgs", "reinforce"])
    parser.add_argument("--stgs_hard", type=str2bool, default=True)
    parser.add_argument("--stgs_hard_method", type=str, default="categorical",
                        choices=["categorical", "embsim-dot", "embsim-cos", "embsim-l2"])
    parser.add_argument("--stgs_hard_embsim_probs", type=str, default="gumbel_soft",
                        choices=["gumbel_soft", "input_logits"])
    parser.add_argument("--stgs_hard_embsim_strategy", type=str, default="nearest",
                        choices=["nearest", "topk_rerank", "topk_sample", "margin_fallback", "lm_topk_restrict"])
    parser.add_argument("--stgs_hard_embsim_top_k", type=int, default=8)
    parser.add_argument("--stgs_hard_embsim_rerank_alpha", type=float, default=0.5)
    parser.add_argument("--stgs_hard_embsim_sample_tau", type=float, default=1.0)
    parser.add_argument("--stgs_hard_embsim_margin", type=float, default=0.0)
    parser.add_argument("--stgs_hard_embsim_fallback", type=str, default="argmax",
                        choices=["argmax", "categorical"])
    parser.add_argument("--eps", type=float, default=1e-10,
                        help="Epsilon value for numerical stability")
    parser.add_argument("--logits_top_k", type=int, default=0,
                        help="Top-k soft filtering of logits before sampling (0 = disabled)")
    parser.add_argument("--logits_top_p", type=float, default=1.0,
                        help="Top-p nucleus soft filtering of logits before sampling (1.0 = disabled)")
    parser.add_argument("--logits_filter_penalty", type=float, default=1e4,
                        help="Penalty value used for filtered logits")
    parser.add_argument("--logits_normalize", type=str, default="none",
                        choices=["none", "center", "zscore"])
    parser.add_argument("--logit_decay", type=float, default=0.0)
    parser.add_argument("--adaptive_gumbel_noise", type=str2bool, default=False)
    parser.add_argument("--adaptive_gumbel_noise_beta", type=float, default=0.9)
    parser.add_argument("--adaptive_gumbel_noise_min_scale", type=float, default=0.0)
    parser.add_argument("--stgs_input_dropout", type=float, default=0.0,
                        help="Dropout applied to logits before Gumbel-softmax")
    parser.add_argument("--stgs_output_dropout", type=float, default=0.0,
                        help="Dropout applied to y_soft before hard selection")
    parser.add_argument("--stgs_snr_empirical", type=str2bool, default=True,
                        help="Compute empirical Gumbel-noise variance and empirical SNR metrics "
                             "(if False, theoretical SNR metrics are still reported)")
    args = parser.parse_args()

    if not (0.0 <= args.stgs_input_dropout < 1.0):
        raise ValueError("--stgs_input_dropout must be in [0, 1)")
    if not (0.0 <= args.stgs_output_dropout < 1.0):
        raise ValueError("--stgs_output_dropout must be in [0, 1)")

    temperature_flag_set = any(
        a == "--temperature" or a.startswith("--temperature=")
        for a in sys.argv[1:]
    )
    if args.temperatures.strip():
        temperatures = [float(t.strip()) for t in args.temperatures.split(",") if t.strip()]
    elif temperature_flag_set:
        temperatures = [float(args.temperature)]
    else:
        temperatures = [0.1, 0.5, 1.0, 2.0]

    embedding_weights: Optional[torch.Tensor] = None

    # Load or generate logits
    if args.logits_source == "random":
        vocab_size = 1000
        logits = torch.randn(1, args.seq_len, vocab_size)
        if args.model is not None:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.model)
                vocab_size = tokenizer.vocab_size
                logits = torch.randn(1, args.seq_len, vocab_size)
            except Exception as e:
                print(f"[warn] Could not load tokenizer from '{args.model}': {e}")
                tokenizer = _make_dummy_tokenizer(vocab_size)
        else:
            tokenizer = _make_dummy_tokenizer(vocab_size)
    elif args.logits_source == "model":
        if args.model is None:
            raise ValueError("--model is required when --logits_source=model")
        logits, tokenizer, embedding_weights = _build_logits_from_model(args.model, args.seq_len)
        vocab_size = logits.shape[-1]
    else:
        logits = torch.load(args.logits_source, map_location="cpu")
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        vocab_size = logits.shape[-1]
        if args.model is not None:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(args.model)
            except Exception as e:
                print(f"[warn] Could not load tokenizer: {e}")
                tokenizer = _make_dummy_tokenizer(vocab_size)
        else:
            tokenizer = _make_dummy_tokenizer(vocab_size)

    logits = _apply_stgs_logit_processing(
        logits=logits,
        logits_top_k=args.logits_top_k,
        logits_top_p=args.logits_top_p,
        logits_filter_penalty=args.logits_filter_penalty,
    )

    if args.logit_decay > 0.0:
        logits = logits * float(args.logit_decay)

    effective_noise_scale = float(args.gumbel_noise_scale)
    if args.adaptive_gumbel_noise:
        # Static approximation of adaptive noise (no training loss loop in this CLI).
        effective_noise_scale = max(
            float(args.adaptive_gumbel_noise_min_scale),
            float(args.gumbel_noise_scale) * float(args.adaptive_gumbel_noise_beta),
        )

    stgs_hard_method = args.stgs_hard_method
    if stgs_hard_method in ("embsim-dot", "embsim-cos", "embsim-l2") and embedding_weights is None:
        if args.model is not None:
            try:
                from transformers import AutoModelForCausalLM
                embs_model = AutoModelForCausalLM.from_pretrained(args.model)
                embedding_weights = embs_model.get_input_embeddings().weight.detach().cpu()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model embeddings required by --stgs_hard_method={stgs_hard_method}"
                ) from e
        else:
            raise ValueError(
                f"--stgs_hard_method={stgs_hard_method} requires embedding weights. "
                "Provide --model or use --logits_source=model."
            )
    if embedding_weights is not None and embedding_weights.shape[0] != logits.shape[-1]:
        raise ValueError(
            "Embedding/logits vocab mismatch: "
            f"embedding vocab={embedding_weights.shape[0]} vs logits vocab={logits.shape[-1]}"
        )

    print(f"Logits shape: {logits.shape}, temperatures: {temperatures}, n_samples: {args.n_samples}")

    results = sample_diversity(
        logits=logits,
        temperatures=temperatures,
        n_samples=args.n_samples,
        noise_scale=effective_noise_scale,
        gradient_estimator=args.gradient_estimator,
        stgs_hard=args.stgs_hard,
        stgs_hard_method=stgs_hard_method,
        stgs_hard_embsim_probs=args.stgs_hard_embsim_probs,
        stgs_hard_embsim_strategy=args.stgs_hard_embsim_strategy,
        stgs_hard_embsim_top_k=args.stgs_hard_embsim_top_k,
        stgs_hard_embsim_rerank_alpha=args.stgs_hard_embsim_rerank_alpha,
        stgs_hard_embsim_sample_tau=args.stgs_hard_embsim_sample_tau,
        stgs_hard_embsim_margin=args.stgs_hard_embsim_margin,
        stgs_hard_embsim_fallback=args.stgs_hard_embsim_fallback,
        logits_normalize=args.logits_normalize,
        stgs_input_dropout=args.stgs_input_dropout,
        stgs_output_dropout=args.stgs_output_dropout,
        embedding_weights=embedding_weights,
        compute_empirical_snr=args.stgs_snr_empirical,
        eps=args.eps,
    )
    metrics = compute_diversity_metrics(results, tokenizer)

    print("\n=== Diversity Metrics ===")
    _print_metrics_table(metrics)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "diversity_report.png"

    fig = plot_diversity_report(results, metrics, tokenizer, top_k=args.top_k,
                                output_path=str(output_path))
    plt.close(fig)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main_cli()
