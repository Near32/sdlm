"""
Partially-conditioned multi-target prompt inversion.

Optimizes a single shared soft prompt p across all (x, R, y) training pairs.

Generative structure:
    R <- f([x, p])          (unconstrained reasoning)
    Y <- f([x, p, R, E])    (final answer given extraction prompt E)

Loss is CE (or composable LossClass losses) on Y tokens against y.
"""

import copy
import sys
import random
import logging
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional, Callable, Sequence, Set, Union

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import wandb

# Invertibility dir first so local main.py takes precedence
sys.path.insert(0, str(Path(__file__).parent))
# Project root for sdlm (installed via pip -e but add root for safety)
sys.path.insert(1, str(Path(__file__).parent.parent.parent))
# tinylora for extraction helpers
sys.path.insert(2, str(Path(__file__).parent.parent / "tinylora"))

from sdlm import STGS
from sdlm.stgs_diff_model import STGSDiffModel
from sdlm.utils.tqdm_utils import tqdm
from rewards import compare_answers
from optimizer_utils import build_lr_scheduler, build_prompt_optimizer

from main import (
    LossClass,
    build_fixed_logits_spec,
    compute_ppo_kl_loss,
    initialize_learnable_inputs,
    initialize_lora_logits,
    snap_free_logits,
    snap_lora_logits,
    compute_position_entropies,
    compute_annealed_temperature,
)
from evaluation import compute_embsim_probs
from pc_weave_logging import (
    build_lm_trace_payload,
    build_trace_segment,
    is_weave_active,
    weave_lm_train_step,
    weave_lm_eval_step,
    weave_epoch_summary,
    weave_prompt_init,
)

logger = logging.getLogger("pc_main")


PC_SUPPORTED_INIT_STRATEGIES = {
    "randn",
    "zeros",
    "normal",
    "one_hot_random",
}

PC_REASONING_GENERATION_BACKENDS = {
    "diff",
    "hf_generate",
}


def _explicitly_requested(explicit_args: Optional[Set[str]], names: List[str]) -> bool:
    if not explicit_args:
        return False
    return any(name in explicit_args for name in names)


def collect_pc_main_incompatibilities(
    config: Dict[str, Any],
    explicit_args: Optional[Set[str]] = None,
) -> List[str]:
    issues: List[str] = []

    reasoning_generation_backend = config.get("reasoning_generation_backend", "diff")
    if reasoning_generation_backend not in PC_REASONING_GENERATION_BACKENDS:
        issues.append(
            "--reasoning_generation_backend only supports "
            f"{', '.join(sorted(PC_REASONING_GENERATION_BACKENDS))}; "
            f"got {reasoning_generation_backend!r}."
        )
    if reasoning_generation_backend == "hf_generate":
        if config.get("teacher_forcing_r", True):
            issues.append("--reasoning_generation_backend='hf_generate' requires --teacher_forcing_r=False.")
        if config.get("bptt", False):
            issues.append("--reasoning_generation_backend='hf_generate' is not compatible with --bptt=True.")

    if config.get("offline_em", False):
        if config.get("teacher_forcing_r", True):
            issues.append(
                "--offline_em=True and --teacher_forcing_r=True are mutually exclusive: "
                "offline EM generates its own reasoning cache, so teacher-forcing must be disabled "
                "(pass --teacher_forcing_r=False)."
            )
        if not config.get("use_batched_forward_pass", False):
            issues.append(
                "--offline_em=True requires --use_batched_forward_pass=True "
                "(serial forward pass does not support the offline reasoning cache)."
            )

    if config.get("gradient_estimator", "stgs") != "stgs":
        issues.append("--gradient_estimator only supports 'stgs' in PC mode.")
    if config.get("method", "stgs") != "stgs":
        issues.append("--method only supports 'stgs' in PC mode.")
    if config.get("teacher_forcing", False):
        issues.append("--teacher_forcing is not compatible with the PC objective; use --teacher_forcing_r instead.")
    if config.get("bptt_teacher_forcing_via_diff_model", False):
        issues.append("--bptt_teacher_forcing_via_diff_model is not implemented for PC optimization.")
    if config.get("bptt_decouple_learnable_temperature", False):
        issues.append("--bptt_decouple_learnable_temperature is not implemented for the PC diff-model path.")
    if config.get("filter_vocab", False):
        issues.append("--filter_vocab is not compatible with the shared multi-target PC prompt.")
    if config.get("early_stop_on_exact_match", False):
        issues.append("--early_stop_on_exact_match is not well-defined for shared-prompt PC optimization.")
    if config.get("run_discrete_validation", False):
        issues.append("--run_discrete_validation is not implemented for PC optimization; use --val_prompt_eval_mode instead.")
    if config.get("run_discrete_embsim_validation", False):
        issues.append("--run_discrete_embsim_validation is not implemented for PC optimization.")
    if config.get("prompt_length_mask_eos_attention", False):
        issues.append("--prompt_length_mask_eos_attention is not implemented for PC optimization.")

    if config.get("fixed_gt_prefix_n", 0) > 0:
        issues.append("--fixed_gt_prefix_n requires prompt-reconstruction ground truth, which PC datasets do not provide.")
    if config.get("fixed_gt_suffix_n", 0) > 0:
        issues.append("--fixed_gt_suffix_n requires prompt-reconstruction ground truth, which PC datasets do not provide.")
    if config.get("fixed_gt_prefix_rank2_n", 0) > 0:
        issues.append("--fixed_gt_prefix_rank2_n requires prompt-reconstruction ground truth, which PC datasets do not provide.")
    if config.get("fixed_gt_suffix_rank2_n", 0) > 0:
        issues.append("--fixed_gt_suffix_rank2_n requires prompt-reconstruction ground truth, which PC datasets do not provide.")

    init_strategy = config.get("init_strategy", "randn")
    if init_strategy not in PC_SUPPORTED_INIT_STRATEGIES:
        issues.append(
            "--init_strategy only supports "
            f"{', '.join(sorted(PC_SUPPORTED_INIT_STRATEGIES))} in PC mode; got {init_strategy!r}."
        )

    if _explicitly_requested(
        explicit_args,
        [
            "plot_every",
            "stgs_grad_variance_samples",
            "stgs_grad_variance_period",
            "stgs_grad_bias_samples",
            "stgs_grad_bias_period",
            "stgs_grad_bias_reference_samples",
            "stgs_grad_bias_reference_batch_size",
            "stgs_grad_bias_reference_use_baseline",
            "stgs_grad_bias_reference_reward_scale",
            "stgs_grad_bias_reference_baseline_beta",
            "reinforce_grad_variance_samples",
            "reinforce_grad_variance_period",
            "reinforce_reward_scale",
            "reinforce_use_baseline",
            "reinforce_baseline_beta",
            "embsim_similarity",
            "embsim_use_input_logits",
            "embsim_teacher_forcing",
            "embsim_temperature",
            "baseline_backend",
            "baseline_model_name",
            "soda_decay_rate",
            "soda_beta1",
            "soda_beta2",
            "soda_reset_epoch",
            "soda_reinit_epoch",
            "soda_reg_weight",
            "soda_bias_correction",
            "soda_init_strategy",
            "soda_init_std",
            "gcg_num_candidates",
            "gcg_top_k",
            "gcg_num_mutations",
            "gcg_pos_choice",
            "gcg_token_choice",
            "gcg_init_strategy",
            "gcg_candidate_batch_size",
            "o2p_model_path",
            "o2p_num_beams",
            "o2p_max_length",
        ],
    ):
        issues.append("REINFORCE, SODA, GCG, O2P, and related diagnostic options are not implemented for PC optimization.")

    return issues


def ensure_pc_main_compatibility(
    config: Dict[str, Any],
    explicit_args: Optional[Set[str]] = None,
) -> None:
    issues = collect_pc_main_incompatibilities(config, explicit_args=explicit_args)
    if issues:
        raise ValueError("Incompatible main.py features for PC optimization:\n- " + "\n- ".join(issues))


def extract_temperature_payload(
    stgs_module: Optional[STGS],
    bptt_module: Optional[STGS],
    device,
) -> Dict[str, Tensor]:
    payload: Dict[str, Tensor] = {}

    def _extract_module(module: Optional[STGS], param_key: str, eff_key: str) -> None:
        if module is None:
            return
        if getattr(module, "learnable_temperature", False) and hasattr(module, "temperature_param"):
            tparam = module.temperature_param.detach().clone()
            payload[param_key] = tparam
            try:
                teff = module.eps + module.init_temperature * (1 + torch.tanh(tparam))
            except Exception:
                teff = tparam
            payload[eff_key] = teff.detach().clone()
        else:
            payload[eff_key] = torch.tensor(
                [float(getattr(module, "init_temperature", 0.0))],
                device=device,
            )

    _extract_module(stgs_module, "stgs_temperature_param", "stgs_effective_temperature")
    _extract_module(bptt_module, "bptt_temperature_param", "bptt_effective_temperature")
    return payload


def _effective_temperature_tensor(stgs_module: STGS) -> Tensor:
    if getattr(stgs_module, "learnable_temperature", False) and hasattr(stgs_module, "temperature_param"):
        return stgs_module.eps + stgs_module.init_temperature * (1 + torch.tanh(stgs_module.temperature_param))
    return torch.tensor(
        [float(getattr(stgs_module, "init_temperature", 1.0))],
        device=getattr(stgs_module, "device", "cpu"),
        dtype=torch.float32,
    )


def _temperature_component_means(temperature_tensor: Tensor) -> List[float]:
    temperature_tensor = temperature_tensor.float()
    if temperature_tensor.ndim > 0 and temperature_tensor.shape[-1] == 1:
        temperature_tensor = temperature_tensor.squeeze(-1)
    if temperature_tensor.ndim == 0:
        return [float(temperature_tensor.item())]
    if temperature_tensor.ndim == 1:
        return [float(value.item()) for value in temperature_tensor]
    return [
        float(temperature_tensor[:, idx].mean().item())
        for idx in range(temperature_tensor.shape[1])
    ]


def _gradient_stats(grad_tensor: Optional[Tensor]) -> Dict[str, float]:
    if grad_tensor is None:
        return {
            "prompt_grad_norm": 0.0,
            "non_zero_grads": 0,
            "grad_mean": 0.0,
            "grad_max": 0.0,
            "grad_min": 0.0,
            "grad_std": 0.0,
        }

    grad_tensor = grad_tensor.detach()
    grad_abs = grad_tensor.abs()
    non_zero = grad_abs > 0
    return {
        "prompt_grad_norm": float(grad_tensor.norm().item()),
        "non_zero_grads": int(non_zero.sum().item()),
        "grad_mean": float(grad_tensor.mean().item()),
        "grad_max": float(grad_tensor.max().item()),
        "grad_min": float(grad_abs[non_zero].min().item()) if non_zero.any() else 0.0,
        "grad_std": float(grad_tensor.std().item()),
    }


# ---------------------------------------------------------------------------
# Weave helpers
# ---------------------------------------------------------------------------

def _decode_trace_text(tokenizer, token_ids: Optional[Tensor], *, skip_special_tokens: bool = False) -> str:
    if token_ids is None:
        return ""
    if isinstance(token_ids, Tensor):
        flat_ids = token_ids.detach().view(-1).tolist()
    else:
        flat_ids = list(token_ids)
    return tokenizer.decode(flat_ids, skip_special_tokens=skip_special_tokens)


def _join_trace_segments(segments: Sequence[Dict[str, Any]]) -> str:
    return "".join(segment.get("text", "") for segment in segments if segment.get("text"))


def _make_trace_segment(
    label: str,
    *,
    text: str,
    token_ids: Optional[Tensor] = None,
    is_exact: bool,
    source: str,
) -> Dict[str, Any]:
    return build_trace_segment(
        label=label,
        text=text,
        token_ids=token_ids,
        is_exact=is_exact,
        source=source,
    )


def _build_pc_lm_trace(
    *,
    tokenizer,
    call_name: str,
    split: str,
    epoch: int,
    lm_name: str,
    input_mode: str,
    output_mode: str,
    input_segments: Sequence[Dict[str, Any]],
    output_segments: Sequence[Dict[str, Any]],
    input_token_ids: Optional[Tensor] = None,
    output_token_ids: Optional[Tensor] = None,
    output_exact_available: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    input_text_approx = _join_trace_segments(input_segments)
    output_text_approx = _join_trace_segments(output_segments)
    return build_lm_trace_payload(
        tokenizer=tokenizer,
        call_name=call_name,
        split=split,
        epoch=epoch,
        lm_name=lm_name,
        input_mode=input_mode,
        output_mode=output_mode,
        input_segments=input_segments,
        output_segments=output_segments,
        input_token_ids=input_token_ids,
        output_token_ids=output_token_ids,
        input_text_approx=input_text_approx,
        output_text_approx=output_text_approx,
        output_exact_available=output_exact_available,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Core forward pass
# ---------------------------------------------------------------------------

def pc_forward_pass(
    diff_model: STGSDiffModel,
    stgs_module: STGS,
    loss_instance: LossClass,
    free_logits: Tensor,            # (1, seq_len, vocab)
    embedding_weights: Tensor,      # (vocab, embed_dim) — frozen
    x_ids: Tensor,                  # (1, x_len)
    x_embeds: Tensor,               # (1, x_len, embed_dim)
    x_text_raw: str,
    R_gt_ids: Optional[Tensor],     # (1, R_len) or None
    R_gt_embeds: Optional[Tensor],  # (1, R_len, embed_dim) or None
    R_gt_text_raw: str,
    E_input_ids: Tensor,            # (1, E_len)
    E_embeds: Tensor,               # (1, E_len, embed_dim)
    Y_tokens: Tensor,               # (1, Y_len) — CE target
    y_target_text_raw: str,
    use_chat_template: bool,
    teacher_forcing_r: bool,
    bptt: bool,
    reasoning_generation_backend: str,
    reasoning_generate_kwargs: Optional[Dict[str, Any]],
    max_new_tokens_reasoning: int,
    max_new_tokens_answer: int,
    gumbel_noise_scale: float = 1.0,
    accumulate_embeds: bool = False,
    tokenizer=None,
    weave_extras: Optional[dict] = None,
) -> Tensor:
    """
    Compute the loss for one (x, R_gt, y) pair.
    Gradients flow back to free_logits through p_embeds.
    Returns a scalar loss tensor.
    """
    Y_len = Y_tokens.shape[1]
    x_text_rendered = _decode_trace_text(tokenizer, x_ids, skip_special_tokens=False) if tokenizer is not None else x_text_raw
    E_text_rendered = _decode_trace_text(tokenizer, E_input_ids, skip_special_tokens=False) if tokenizer is not None else ""
    Y_target_rendered = _decode_trace_text(tokenizer, Y_tokens, skip_special_tokens=False) if tokenizer is not None else y_target_text_raw
    Y_target_clean = _decode_trace_text(tokenizer, Y_tokens, skip_special_tokens=True) if tokenizer is not None else y_target_text_raw

    # Step 0: get soft prompt embeddings via STGS
    _, p_one_hot, _, _ = stgs_module(
        free_logits,
        gumbel_noise_scale=gumbel_noise_scale,
        embedding_weights=embedding_weights,
    )
    p_embeds = p_one_hot @ embedding_weights.detach()   # (1, seq_len, embed_dim)
    p_token_ids = p_one_hot.detach().argmax(dim=-1)
    p_text_argmax = _decode_trace_text(tokenizer, p_token_ids, skip_special_tokens=False) if tokenizer is not None else ""
    p_text_argmax_clean = _decode_trace_text(tokenizer, p_token_ids, skip_special_tokens=True) if tokenizer is not None else ""

    if teacher_forcing_r:
        # Case A: teacher-forced R (or no R when R_gt_embeds is None)
        # Input: [x | p | R_gt | E | Y[0:-1]]
        Y_embeds_full = embedding_weights[Y_tokens.squeeze(0)].unsqueeze(0)  # (1, Y_len, d)
        parts = [x_embeds, p_embeds]
        if R_gt_embeds is not None:
            parts.append(R_gt_embeds)
        parts.append(E_embeds)
        Y_prefix_ids = Y_tokens[:, :-1] if Y_len > 1 else None
        Y_prefix_text = _decode_trace_text(tokenizer, Y_prefix_ids, skip_special_tokens=False) if tokenizer is not None else ""
        if Y_len > 1:
            parts.append(Y_embeds_full[:, :-1, :])
        input_embeds = torch.cat(parts, dim=1)

        output = diff_model.forward(inputs_embeds=input_embeds, output_normal_logits=True)
        y_logits = output.logits[:, -Y_len:, :]   # (1, Y_len, vocab)
        y_pred_token_ids = y_logits.detach().argmax(dim=-1)
        y_pred_text = _decode_trace_text(tokenizer, y_pred_token_ids, skip_special_tokens=False) if tokenizer is not None else ""
        y_pred_text_clean = _decode_trace_text(tokenizer, y_pred_token_ids, skip_special_tokens=True) if tokenizer is not None else ""

        if weave_extras is not None and tokenizer is not None:
            input_segments = [
                _make_trace_segment(
                    "x_rendered",
                    text=x_text_rendered,
                    token_ids=x_ids,
                    is_exact=True,
                    source="chat_template" if use_chat_template else "plain_tokenization",
                ),
                _make_trace_segment(
                    "p_argmax",
                    text=p_text_argmax,
                    token_ids=p_token_ids,
                    is_exact=False,
                    source="argmax_from_stgs_prompt_distribution",
                ),
            ]
            if R_gt_ids is not None:
                input_segments.append(
                    _make_trace_segment(
                        "R_gt",
                        text=_decode_trace_text(tokenizer, R_gt_ids, skip_special_tokens=False),
                        token_ids=R_gt_ids,
                        is_exact=True,
                        source="ground_truth_reasoning",
                    )
                )
            input_segments.append(
                _make_trace_segment(
                    "E",
                    text=E_text_rendered,
                    token_ids=E_input_ids,
                    is_exact=True,
                    source="extraction_prompt",
                )
            )
            if Y_prefix_ids is not None:
                input_segments.append(
                    _make_trace_segment(
                        "Y_prefix",
                        text=Y_prefix_text,
                        token_ids=Y_prefix_ids,
                        is_exact=True,
                        source="teacher_forced_target_prefix",
                    )
                )
            output_segments = [
                _make_trace_segment(
                    "Y_argmax",
                    text=y_pred_text,
                    token_ids=y_pred_token_ids,
                    is_exact=False,
                    source="argmax_from_output_logits",
                )
            ]
            weave_extras.setdefault("lm_traces", []).append(
                _build_pc_lm_trace(
                    tokenizer=tokenizer,
                    call_name="teacher_forced_answer_forward",
                    split="train",
                    epoch=-1,
                    lm_name="diff_model.forward",
                    input_mode="inputs_embeds",
                    output_mode="normal_logits",
                    input_segments=input_segments,
                    output_segments=output_segments,
                    output_token_ids=y_pred_token_ids,
                    output_exact_available=False,
                    metadata={
                        "call_role": "answer",
                        "teacher_forcing_r": True,
                        "reasoning_generation_backend": reasoning_generation_backend,
                        "use_chat_template": use_chat_template,
                        "soft_prompt_present": True,
                        "soft_reasoning_present": False,
                        "x_text_raw": x_text_raw,
                        "x_text_rendered": x_text_rendered,
                        "p_text_argmax": p_text_argmax,
                        "p_text_argmax_clean": p_text_argmax_clean,
                        "R_gt_text": R_gt_text_raw,
                        "E_text": E_text_rendered,
                        "y_target_text": Y_target_rendered,
                        "y_target_text_clean": Y_target_clean,
                        "y_target_raw": y_target_text_raw,
                        "y_target_token_ids": Y_tokens.detach().view(-1).tolist(),
                        "output_kind": "argmax_from_output_logits",
                    },
                )
            )
            weave_extras["y_pred_text"] = y_pred_text_clean

    else:
        # Case B: generate R differentiably
        reasoning_prefix_embeds = torch.cat([x_embeds, p_embeds], dim=1)
        if reasoning_generation_backend == "hf_generate":
            r_out = diff_model.generate(
                inputs_embeds=reasoning_prefix_embeds,
                max_new_tokens=max_new_tokens_reasoning,
                output_diff_one_hots=False,
                return_dict=True,
                generation_backend="hf_generate",
                **(reasoning_generate_kwargs or {}),
            )
            R_soft_embeds = diff_model.model.get_input_embeddings()(r_out.sampled_diff_tokens.long())
            answer_reasoning_exact = True
        else:
            r_out = diff_model.generate(
                inputs_embeds=reasoning_prefix_embeds,
                max_new_tokens=max_new_tokens_reasoning,
                output_diff_one_hots=not accumulate_embeds,
                output_normal_logits=False,
                use_soft_embeds=True,
                use_bpttoken=bptt,
                return_dict=True,
                accumulate_embeds=accumulate_embeds,
            )
            if r_out.sampled_diff_embeds is not None:
                R_soft_embeds = r_out.sampled_diff_embeds
            else:
                R_soft_embeds = r_out.sampled_diff_one_hot @ embedding_weights.detach()
            if not bptt:
                R_soft_embeds = R_soft_embeds.detach()
            answer_reasoning_exact = False

        R_gen_token_ids = r_out.sampled_diff_tokens
        R_gen_text = _decode_trace_text(tokenizer, R_gen_token_ids, skip_special_tokens=False) if tokenizer is not None else ""
        R_gen_text_clean = _decode_trace_text(tokenizer, R_gen_token_ids, skip_special_tokens=True) if tokenizer is not None else ""

        if weave_extras is not None and tokenizer is not None:
            reasoning_input_segments = [
                _make_trace_segment(
                    "x_rendered",
                    text=x_text_rendered,
                    token_ids=x_ids,
                    is_exact=True,
                    source="chat_template" if use_chat_template else "plain_tokenization",
                ),
                _make_trace_segment(
                    "p_argmax",
                    text=p_text_argmax,
                    token_ids=p_token_ids,
                    is_exact=False,
                    source="argmax_from_stgs_prompt_distribution",
                ),
            ]
            reasoning_output_segments = [
                _make_trace_segment(
                    "R_gen",
                    text=R_gen_text,
                    token_ids=R_gen_token_ids,
                    is_exact=True,
                    source="generated_reasoning_tokens",
                )
            ]
            weave_extras.setdefault("lm_traces", []).append(
                _build_pc_lm_trace(
                    tokenizer=tokenizer,
                    call_name="reasoning_generate",
                    split="train",
                    epoch=-1,
                    lm_name="diff_model.generate",
                    input_mode="inputs_embeds",
                    output_mode="generated_tokens",
                    input_segments=reasoning_input_segments,
                    output_segments=reasoning_output_segments,
                    output_token_ids=R_gen_token_ids,
                    output_exact_available=True,
                    metadata={
                        "call_role": "reasoning",
                        "teacher_forcing_r": False,
                        "reasoning_generation_backend": reasoning_generation_backend,
                        "use_chat_template": use_chat_template,
                        "soft_prompt_present": True,
                        "soft_reasoning_present": False,
                        "x_text_raw": x_text_raw,
                        "x_text_rendered": x_text_rendered,
                        "p_text_argmax": p_text_argmax,
                        "p_text_argmax_clean": p_text_argmax_clean,
                        "E_text": E_text_rendered,
                        "y_target_text": Y_target_rendered,
                        "y_target_text_clean": Y_target_clean,
                        "y_target_raw": y_target_text_raw,
                        "y_target_token_ids": Y_tokens.detach().view(-1).tolist(),
                        "output_kind": "generated_tokens",
                    },
                )
            )

        y_out = diff_model.generate(
            inputs_embeds=torch.cat([x_embeds, p_embeds, R_soft_embeds, E_embeds], dim=1),
            max_new_tokens=max_new_tokens_answer,
            output_diff_one_hots=False,
            output_normal_logits=True,
            use_soft_embeds=True,
            return_dict=True,
        )
        actual_len = y_out.logits.shape[1]
        if actual_len < Y_len:
            pad = torch.zeros(
                1, Y_len - actual_len, y_out.logits.shape[-1],
                device=y_out.logits.device, dtype=y_out.logits.dtype,
            )
            y_logits = torch.cat([y_out.logits, pad], dim=1)
        else:
            y_logits = y_out.logits[:, :Y_len, :]
        y_pred_token_ids = y_logits.detach().argmax(dim=-1)
        y_pred_text = _decode_trace_text(tokenizer, y_pred_token_ids, skip_special_tokens=False) if tokenizer is not None else ""
        y_pred_text_clean = _decode_trace_text(tokenizer, y_pred_token_ids, skip_special_tokens=True) if tokenizer is not None else ""

        if weave_extras is not None and tokenizer is not None:
            answer_input_segments = [
                _make_trace_segment(
                    "x_rendered",
                    text=x_text_rendered,
                    token_ids=x_ids,
                    is_exact=True,
                    source="chat_template" if use_chat_template else "plain_tokenization",
                ),
                _make_trace_segment(
                    "p_argmax",
                    text=p_text_argmax,
                    token_ids=p_token_ids,
                    is_exact=False,
                    source="argmax_from_stgs_prompt_distribution",
                ),
                _make_trace_segment(
                    "R_gen",
                    text=R_gen_text,
                    token_ids=R_gen_token_ids,
                    is_exact=answer_reasoning_exact,
                    source=(
                        "generated_reasoning_tokens"
                        if answer_reasoning_exact
                        else "decoded_proxy_for_soft_reasoning_embeds"
                    ),
                ),
                _make_trace_segment(
                    "E",
                    text=E_text_rendered,
                    token_ids=E_input_ids,
                    is_exact=True,
                    source="extraction_prompt",
                ),
            ]
            answer_output_segments = [
                _make_trace_segment(
                    "Y_argmax",
                    text=y_pred_text,
                    token_ids=y_pred_token_ids,
                    is_exact=False,
                    source="argmax_from_output_logits",
                )
            ]
            weave_extras.setdefault("lm_traces", []).append(
                _build_pc_lm_trace(
                    tokenizer=tokenizer,
                    call_name="answer_generate",
                    split="train",
                    epoch=-1,
                    lm_name="diff_model.generate",
                    input_mode="inputs_embeds",
                    output_mode="normal_logits",
                    input_segments=answer_input_segments,
                    output_segments=answer_output_segments,
                    output_token_ids=y_pred_token_ids,
                    output_exact_available=False,
                    metadata={
                        "call_role": "answer",
                        "teacher_forcing_r": False,
                        "reasoning_generation_backend": reasoning_generation_backend,
                        "use_chat_template": use_chat_template,
                        "soft_prompt_present": True,
                        "soft_reasoning_present": not answer_reasoning_exact,
                        "x_text_raw": x_text_raw,
                        "x_text_rendered": x_text_rendered,
                        "p_text_argmax": p_text_argmax,
                        "p_text_argmax_clean": p_text_argmax_clean,
                        "R_gen_text": R_gen_text,
                        "R_gen_text_clean": R_gen_text_clean,
                        "E_text": E_text_rendered,
                        "y_target_text": Y_target_rendered,
                        "y_target_text_clean": Y_target_clean,
                        "y_target_raw": y_target_text_raw,
                        "y_target_token_ids": Y_tokens.detach().view(-1).tolist(),
                        "output_kind": "argmax_from_output_logits",
                    },
                )
            )
            weave_extras["R_gen_text"] = R_gen_text_clean
            weave_extras["y_pred_text"] = y_pred_text_clean

    # Populate Weave trace extras with decoded text
    if weave_extras is not None and tokenizer is not None and "y_pred_text" not in weave_extras:
        weave_extras["y_pred_text"] = _decode_trace_text(
            tokenizer,
            y_logits.detach().argmax(dim=-1),
            skip_special_tokens=True,
        )

    input_dict = {
        "generated_logits":    y_logits,
        "prompt_logits":       free_logits,
        "prompt_argmax_ids":   free_logits.argmax(dim=-1),
        "completion_ids":      Y_tokens,
        "tf_generated_logits": None,
        "prompt_input_logits": free_logits,  # for eos/entropy/commitment losses
    }
    losses_dict = loss_instance.compute_loss(input_dict)
    return losses_dict["sumloss"]


# ---------------------------------------------------------------------------
# Batched helpers
# ---------------------------------------------------------------------------

def _left_pad_embed_sequences(
    sequences: List[Tensor],  # each (1, L_i, d)
    device,
    dtype,
) -> Tuple[Tensor, Tensor]:
    """Left-pad (1, L_i, d) tensors → (B, max_L, d) with attention_mask (B, max_L).

    Zeros are prepended so real tokens are right-aligned.
    attention_mask: 0 = padding, 1 = real token.
    """
    B = len(sequences)
    max_L = max(s.shape[1] for s in sequences)
    d = sequences[0].shape[2]
    padded = torch.zeros(B, max_L, d, device=device, dtype=dtype)
    attention_mask = torch.zeros(B, max_L, device=device, dtype=torch.long)
    for i, seq in enumerate(sequences):
        L_i = seq.shape[1]
        padded[i, max_L - L_i:] = seq[0]
        attention_mask[i, max_L - L_i:] = 1
    return padded, attention_mask


def pc_forward_pass_batched_tf(
    diff_model,
    stgs_module,
    loss_instances: List,                        # one LossClass per sample
    free_logits: Tensor,                         # (1, seq_len, vocab)
    embedding_weights: Tensor,                   # (vocab, d)
    x_embeds_list: List[Tensor],                 # [(1, x_len_i, d), ...]
    R_gt_embeds_list: List[Optional[Tensor]],    # [(1, R_len_i, d) or None, ...]
    E_embeds: Tensor,                            # (1, E_len, d)
    Y_tokens_list: List[Tensor],                 # [(1, Y_len_i), ...]
    gumbel_noise_scale: float = 1.0,
    stgs_noise_mode: str = "shared",             # "shared" | "independent"
) -> Dict[str, Any]:
    """Batched teacher-forcing forward pass (teacher_forcing_r=True, bptt=False).

    Assembles [x_i | p_i | R_gt_i | E | Y_i[:-1]] for each sample,
    left-pads to a common length, and runs a single diff_model.forward call.

    Returns:
        dict with keys:
            "losses":        list of B scalar loss tensors (grad-connected)
            "y_logits_list": list of B (1, Y_len_i, vocab) detached tensors for
                             training accuracy decoding
            "R_tokens":      None (teacher-forcing uses ground-truth R, no generation)
    """
    B = len(x_embeds_list)
    device = x_embeds_list[0].device
    dtype  = x_embeds_list[0].dtype

    # --- Build per-sample or shared p_embeds ---
    if stgs_noise_mode == "independent":
        p_rows = []
        for _ in range(B):
            _, oh, _, _ = stgs_module(
                free_logits, gumbel_noise_scale=gumbel_noise_scale,
                embedding_weights=embedding_weights,
            )
            p_rows.append((oh @ embedding_weights.detach()).squeeze(0))
        p_embeds_batch = torch.stack(p_rows, dim=0)   # (B, seq_len, d)
    else:
        _, oh, _, _ = stgs_module(
            free_logits, gumbel_noise_scale=gumbel_noise_scale,
            embedding_weights=embedding_weights,
        )
        p_embeds_batch = (oh @ embedding_weights.detach()).expand(B, -1, -1)

    # --- Build full sequences [x | p | R_gt | E | Y[:-1]] per sample ---
    sequences = []
    y_lens    = []
    for i in range(B):
        parts = [x_embeds_list[i], p_embeds_batch[i:i+1]]
        if R_gt_embeds_list[i] is not None:
            parts.append(R_gt_embeds_list[i])
        parts.append(E_embeds)
        Y_len_i = Y_tokens_list[i].shape[1]
        y_lens.append(Y_len_i)
        if Y_len_i > 1:
            Y_emb = embedding_weights[Y_tokens_list[i].squeeze(0)].unsqueeze(0)
            parts.append(Y_emb[:, :-1, :])
        sequences.append(torch.cat(parts, dim=1))

    padded, attention_mask = _left_pad_embed_sequences(sequences, device, dtype)
    output = diff_model.forward(
        inputs_embeds=padded,
        attention_mask=attention_mask,
        output_normal_logits=True,
    )
    # output.logits: (B, max_total_len, vocab)

    losses, y_logits_list = [], []
    for i in range(B):
        y_logits_i = output.logits[i:i+1, -y_lens[i]:, :]  # right-aligned
        d = loss_instances[i].compute_loss({
            "generated_logits":    y_logits_i,
            "prompt_logits":       free_logits,
            "prompt_argmax_ids":   free_logits.argmax(dim=-1),
            "completion_ids":      Y_tokens_list[i],
            "tf_generated_logits": None,
            "prompt_input_logits": free_logits,
        })
        losses.append(d["sumloss"])
        y_logits_list.append(y_logits_i.detach())
    return {"losses": losses, "y_logits_list": y_logits_list, "R_tokens": None}


def pc_forward_pass_batched_free(
    diff_model,
    stgs_module,
    loss_instances: List,
    free_logits: Tensor,                         # (1, seq_len, vocab)
    embedding_weights: Tensor,                   # (vocab, d)
    x_embeds_list: List[Tensor],                 # [(1, x_len_i, d), ...]
    E_embeds: Tensor,                            # (1, E_len, d)
    Y_tokens_list: List[Tensor],                 # [(1, Y_len_i), ...]
    max_new_tokens_reasoning: int,
    max_new_tokens_answer: int,
    bptt: bool = False,
    gumbel_noise_scale: float = 1.0,
    accumulate_embeds: bool = False,
    stgs_noise_mode: str = "shared",
    reasoning_generation_backend: str = "diff",
    reasoning_generate_kwargs: Optional[Dict[str, Any]] = None,
    # Offline EM: if provided, skip Stage 1 generation and use cached tokens
    offline_r_cache: Optional[Dict[str, Tensor]] = None,
    x_strs_batch: Optional[List[str]] = None,   # required when offline_r_cache is not None
) -> Dict[str, Any]:
    """Batched free-generation forward pass (teacher_forcing_r=False).

    Stage 1 — batched reasoning generate from [x_i | p_i].
    Stage 2 — batched answer generate from [x_i | p_i | R_i | E].

    bptt=False: R_embeds detached before stage 2 (no gradient through reasoning).
    bptt=True:  R_embeds NOT detached — gradients flow stage-2 loss → R generation
                → p_embeds → STGS → free_logits (same pattern as sdlm_optimizer.py).

    Returns:
        dict with keys:
            "losses":        list of B scalar loss tensors (grad-connected)
            "y_logits_list": list of B (1, Y_len_i, vocab) detached tensors
            "R_tokens":      (B, R_gen_len) int64 tensor of generated reasoning token ids
    """
    B = len(x_embeds_list)
    device = x_embeds_list[0].device
    dtype  = x_embeds_list[0].dtype

    # --- p_embeds ---
    if stgs_noise_mode == "independent":
        p_rows = []
        for _ in range(B):
            _, oh, _, _ = stgs_module(
                free_logits, gumbel_noise_scale=gumbel_noise_scale,
                embedding_weights=embedding_weights,
            )
            p_rows.append((oh @ embedding_weights.detach()).squeeze(0))
        p_embeds_batch = torch.stack(p_rows, dim=0)
    else:
        _, oh, _, _ = stgs_module(
            free_logits, gumbel_noise_scale=gumbel_noise_scale,
            embedding_weights=embedding_weights,
        )
        p_embeds_batch = (oh @ embedding_weights.detach()).expand(B, -1, -1)

    # --- Stage 1: reasoning tokens ---
    if offline_r_cache is not None:
        # Offline EM M-step: use cached reasoning tokens (no gradient through R).
        assert x_strs_batch is not None, (
            "x_strs_batch must be provided when offline_r_cache is not None"
        )
        R_tokens = torch.cat(
            [offline_r_cache[x_str].to(device) for x_str in x_strs_batch], dim=0
        )  # (B, R_gen_len)
        R_embeds = diff_model.model.get_input_embeddings()(R_tokens.long()).detach()
        with torch.no_grad():
            pad_id   = getattr(diff_model, "pad_token_id", 0) or 0
            pad_mask = (R_tokens.long() == pad_id).long()
            R_gen_len = R_tokens.shape[1]
            raw_starts = pad_mask.argmax(dim=1)
            padding_starts = torch.where(
                raw_starts == 0,
                torch.full_like(raw_starts, R_gen_len),
                raw_starts,
            )
    else:
        # Online: generate reasoning fresh from [x_i | p_i].
        xp_seqs = [
            torch.cat([x_embeds_list[i], p_embeds_batch[i:i+1]], dim=1)
            for i in range(B)
        ]
        xp_padded, xp_mask = _left_pad_embed_sequences(xp_seqs, device, dtype)
        _use_hf_backend = reasoning_generation_backend == "hf_generate"
        _r_gen_extra = {
            "generation_backend": "hf_generate", "output_diff_one_hots": False,
            **(reasoning_generate_kwargs or {}),
        } if _use_hf_backend else {}
        r_out = diff_model.generate(
            inputs_embeds=xp_padded,
            attention_mask=xp_mask,
            max_new_tokens=max_new_tokens_reasoning,
            output_normal_logits=False,
            return_dict=True,
            accumulate_embeds=accumulate_embeds and not _use_hf_backend,
            **_r_gen_extra,
        )

        with torch.no_grad():
            R_tokens = r_out.sampled_diff_tokens          # (B, R_gen_len)
            pad_id   = getattr(diff_model, "pad_token_id", 0) or 0
            pad_mask = (R_tokens.long() == pad_id).long()
            R_gen_len = R_tokens.shape[1]
            raw_starts = pad_mask.argmax(dim=1)
            padding_starts = torch.where(
                raw_starts == 0,
                torch.full_like(raw_starts, R_gen_len),
                raw_starts,
            )

        # R soft embeddings.
        # bptt=False: detach so gradients stop here (reasoning is treated as fixed).
        # bptt=True:  keep grad — gradient flows from answer loss back through
        #             reasoning generation to p_embeds → STGS → free_logits.
        if _use_hf_backend:
            R_embeds = diff_model.model.get_input_embeddings()(R_tokens.long()).detach()
        elif r_out.sampled_diff_embeds is not None:
            R_embeds = r_out.sampled_diff_embeds
        else:
            R_embeds = r_out.sampled_diff_one_hot @ embedding_weights.detach()
        if not bptt:
            R_embeds = R_embeds.detach()
    # R_embeds: (B, R_gen_len, d)

    # --- Stage 2: answer generate from [x_i | p_i | R_i[:trim] | E] ---
    xpRE_seqs = []
    for i in range(B):
        trim = int(padding_starts[i].item())
        xpRE_seqs.append(torch.cat([
            x_embeds_list[i],
            p_embeds_batch[i:i+1],
            R_embeds[i:i+1, :trim, :],
            E_embeds,
        ], dim=1))
    xpRE_padded, xpRE_mask = _left_pad_embed_sequences(xpRE_seqs, device, dtype)
    y_out = diff_model.generate(
        inputs_embeds=xpRE_padded,
        attention_mask=xpRE_mask,
        max_new_tokens=max_new_tokens_answer,
        output_normal_logits=True,
        return_dict=True,
    )
    # y_out.logits: (B, max_new_tokens_answer, vocab)

    losses, y_logits_list = [], []
    for i in range(B):
        Y_len_i = Y_tokens_list[i].shape[1]
        actual  = y_out.logits.shape[1]
        if actual < Y_len_i:
            pad = torch.zeros(
                1, Y_len_i - actual, y_out.logits.shape[-1],
                device=device, dtype=y_out.logits.dtype,
            )
            y_logits_i = torch.cat([y_out.logits[i:i+1], pad], dim=1)
        else:
            y_logits_i = y_out.logits[i:i+1, :Y_len_i, :]
        d = loss_instances[i].compute_loss({
            "generated_logits":    y_logits_i,
            "prompt_logits":       free_logits,
            "prompt_argmax_ids":   free_logits.argmax(dim=-1),
            "completion_ids":      Y_tokens_list[i],
            "tf_generated_logits": None,
            "prompt_input_logits": free_logits,
        })
        losses.append(d["sumloss"])
        y_logits_list.append(y_logits_i.detach())
    return {"losses": losses, "y_logits_list": y_logits_list, "R_tokens": R_tokens}


def pc_forward_pass_batched(
    diff_model,
    stgs_module,
    loss_instances: List,
    free_logits: Tensor,
    embedding_weights: Tensor,
    x_embeds_list: List[Tensor],
    R_gt_embeds_list: List[Optional[Tensor]],
    E_embeds: Tensor,
    Y_tokens_list: List[Tensor],
    E_input_ids: Tensor,
    teacher_forcing_r: bool,
    bptt: bool,
    reasoning_generation_backend: str,
    reasoning_generate_kwargs: Optional[Dict[str, Any]],
    max_new_tokens_reasoning: int,
    max_new_tokens_answer: int,
    gumbel_noise_scale: float = 1.0,
    accumulate_embeds: bool = False,
    stgs_noise_mode: str = "shared",
    offline_r_cache: Optional[Dict[str, Tensor]] = None,
    x_strs_batch: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Dispatcher: routes to the appropriate batched implementation.

    teacher_forcing_r=True  → pc_forward_pass_batched_tf (bptt irrelevant: no generation)
    teacher_forcing_r=False → pc_forward_pass_batched_free (bptt controls R_embeds detach)

    offline_r_cache: when provided, Stage 1 reasoning generation is skipped and cached
        token IDs are used instead (offline EM M-step). Requires x_strs_batch.

    There is no serial fallback. bptt=True is handled inside pc_forward_pass_batched_free
    by keeping R_embeds grad-connected, mirroring sdlm_optimizer.py.
    """
    if teacher_forcing_r:
        return pc_forward_pass_batched_tf(
            diff_model=diff_model, stgs_module=stgs_module,
            loss_instances=loss_instances,
            free_logits=free_logits, embedding_weights=embedding_weights,
            x_embeds_list=x_embeds_list, R_gt_embeds_list=R_gt_embeds_list,
            E_embeds=E_embeds, Y_tokens_list=Y_tokens_list,
            gumbel_noise_scale=gumbel_noise_scale,
            stgs_noise_mode=stgs_noise_mode,
        )
    else:
        return pc_forward_pass_batched_free(
            diff_model=diff_model, stgs_module=stgs_module,
            loss_instances=loss_instances,
            free_logits=free_logits, embedding_weights=embedding_weights,
            x_embeds_list=x_embeds_list,
            E_embeds=E_embeds, Y_tokens_list=Y_tokens_list,
            max_new_tokens_reasoning=max_new_tokens_reasoning,
            max_new_tokens_answer=max_new_tokens_answer,
            bptt=bptt,
            gumbel_noise_scale=gumbel_noise_scale,
            accumulate_embeds=accumulate_embeds,
            stgs_noise_mode=stgs_noise_mode,
            reasoning_generation_backend=reasoning_generation_backend,
            reasoning_generate_kwargs=reasoning_generate_kwargs,
            offline_r_cache=offline_r_cache,
            x_strs_batch=x_strs_batch,
        )


# ---------------------------------------------------------------------------
# Offline EM: E-step reasoning cache generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_offline_reasoning_cache(
    diff_model: STGSDiffModel,
    free_logits: Tensor,                         # (1, seq_len, vocab) — detached
    x_embeds_cache: Dict[str, Tensor],           # x_str -> (1, x_len, d)
    train_triples: List[Tuple[str, str, str]],   # (x_str, R_gt_str, y_raw_str)
    stgs_module: "STGS",                         # training module — properties inherited
    embedding_weights: Tensor,                   # (vocab, d)
    offline_em_stgs_method: str,                 # "argmax"|"embsim-dot"|"embsim-cos"|"embsim-l2"|"soft"
    offline_em_stgs_embsim_probs: str,            # "gumbel_soft"|"input_logits"
    offline_em_temperature: Union[str, float],   # "learned" | float
    max_new_tokens_reasoning: int,
    reasoning_generation_backend: str,
    reasoning_generate_kwargs: Optional[Dict[str, Any]],
    cache_batch_size: int,
    device,
) -> Dict[str, Tensor]:
    """E-step: generate reasoning tokens for all training x_strs using the current prompt.

    Returns a dict mapping each unique x_str to a (1, R_len) CPU int64 tensor of
    reasoning token IDs.  These are fed as fixed inputs during the subsequent M-step
    forward passes instead of re-generating reasoning from scratch each iteration.
    """
    # --- Resolve E-step temperature ---
    if offline_em_temperature == "learned":
        e_step_temp: Optional[float] = None      # STGS uses self.init_temperature
    else:
        e_step_temp = float(offline_em_temperature)

    # --- Build p_embeds for the E-step ---
    # Special case: soft distribution from raw input logits (no STGS, no Gumbel noise).
    if offline_em_stgs_method == "soft" and offline_em_stgs_embsim_probs == "input_logits":
        temp = e_step_temp if e_step_temp is not None else stgs_module.init_temperature
        p_probs = F.softmax(free_logits.float() / temp, dim=-1)
        p_embeds = (p_probs.to(embedding_weights.dtype) @ embedding_weights).detach()
    else:
        # Build a lightweight E-step STGS module, inheriting strategy params from the
        # training module but overriding hard_method, embsim_probs, and temperature.
        e_step_stgs = STGS(
            vocab_size=stgs_module.vocab_size,
            stgs_hard=(offline_em_stgs_method != "soft"),
            stgs_hard_method=(
                offline_em_stgs_method if offline_em_stgs_method != "soft" else "argmax"
            ),
            init_temperature=stgs_module.init_temperature,
            learnable_temperature=False,         # no learnable temp in E-step
            stgs_hard_embsim_probs=offline_em_stgs_embsim_probs,
            stgs_hard_embsim_strategy=stgs_module.stgs_hard_embsim_strategy,
            stgs_hard_embsim_top_k=stgs_module.stgs_hard_embsim_top_k,
            stgs_hard_embsim_rerank_alpha=stgs_module.stgs_hard_embsim_rerank_alpha,
            stgs_hard_embsim_sample_tau=stgs_module.stgs_hard_embsim_sample_tau,
            stgs_hard_embsim_margin=stgs_module.stgs_hard_embsim_margin,
            stgs_hard_embsim_fallback=stgs_module.stgs_hard_embsim_fallback,
            logits_normalize=stgs_module.logits_normalize,
            eps=stgs_module.eps,
            device=str(device),
        ).eval()
        if e_step_temp is not None:
            e_step_stgs.init_temperature = e_step_temp

        _, p_one_hot, _, y_soft = e_step_stgs(
            free_logits.detach(),
            temperature=e_step_temp,             # None → uses module's init_temperature
            embedding_weights=embedding_weights,
            gumbel_noise_scale=0.0,              # deterministic: no Gumbel noise in E-step
        )
        if offline_em_stgs_method == "soft":
            p_embeds = (y_soft @ embedding_weights.detach()).detach()
        else:
            p_embeds = (p_one_hot @ embedding_weights.detach()).detach()
    # p_embeds: (1, seq_len, d) — fully detached

    # --- Batch-generate reasoning tokens for every unique x_str ---
    unique_x_strs = list(dict.fromkeys(t[0] for t in train_triples))  # deduplicated, ordered
    _use_hf = reasoning_generation_backend == "hf_generate"
    _r_gen_extra = {
        "generation_backend": "hf_generate", "output_diff_one_hots": False,
        **(reasoning_generate_kwargs or {}),
    } if _use_hf else {}

    cache: Dict[str, Tensor] = {}
    dtype = embedding_weights.dtype
    with tqdm(total=len(unique_x_strs), desc="E-step: caching reasoning", unit="x", leave=False) as pbar:
        for batch_start in range(0, len(unique_x_strs), cache_batch_size):
            batch_x_strs = unique_x_strs[batch_start:batch_start + cache_batch_size]
            xp_seqs = [
                torch.cat([x_embeds_cache[x_str], p_embeds], dim=1)
                for x_str in batch_x_strs
            ]
            xp_padded, xp_mask = _left_pad_embed_sequences(xp_seqs, device, dtype)
            r_out = diff_model.generate(
                inputs_embeds=xp_padded,
                attention_mask=xp_mask,
                max_new_tokens=max_new_tokens_reasoning,
                output_normal_logits=False,
                return_dict=True,
                **_r_gen_extra,
            )
            R_tokens = r_out.sampled_diff_tokens   # (B_batch, R_gen_len)
            for i, x_str in enumerate(batch_x_strs):
                cache[x_str] = R_tokens[i:i+1].cpu()   # (1, R_gen_len)
            pbar.update(len(batch_x_strs))

    return cache


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _tokenize_x(x_str: str, tokenizer, device, use_chat_template: bool) -> Tensor:
    """
    Tokenize input x, optionally applying the model's chat template.

    When use_chat_template=True, wraps x as a user turn and calls
    apply_chat_template with add_generation_prompt=True so the model
    sees [chat(x), p] instead of [x, p].
    Falls back to plain tokenization if the tokenizer has no chat_template.
    """
    if use_chat_template and getattr(tokenizer, "chat_template", None) is not None:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": x_str}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    if use_chat_template:
        logger.warning(
            "use_chat_template=True but tokenizer has no chat_template; "
            "falling back to plain tokenization."
        )
    return tokenizer(x_str, return_tensors="pt").input_ids.to(device)


def _compute_method_results(
    R_text: str,
    Y_text: str,
    y_raw_str: str,
    extraction_fns: Dict[str, Callable],
) -> Dict[str, bool]:
    method_results: Dict[str, bool] = {}
    for method, fn in extraction_fns.items():
        extracted = fn(Y_text) if method == "extractor" else fn(R_text)
        method_results[method] = compare_answers(extracted, y_raw_str)
    return method_results


def _metrics_from_accuracy_counts(
    per_method_correct: Dict[str, int],
    any_correct_count: int,
    all_correct_count: int,
    n_examples: int,
    extraction_fns: Dict[str, Callable],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    denom = max(n_examples, 1)
    for method in extraction_fns:
        metrics[f"accuracy_{method}"] = per_method_correct[method] / denom
    metrics["any_correct"] = any_correct_count / denom
    metrics["all_correct"] = all_correct_count / denom
    metrics["n_eval"] = float(n_examples)
    return metrics


def _accuracy_postfix(metrics: Dict[str, float], extraction_fns: Dict[str, Callable]) -> Dict[str, str]:
    postfix = {
        "any": f"{metrics.get('any_correct', 0.0):.3f}",
        "all": f"{metrics.get('all_correct', 0.0):.3f}",
    }
    for method in extraction_fns:
        postfix[method] = f"{metrics.get(f'accuracy_{method}', 0.0):.3f}"
    return postfix


def evaluate_shared_prompt(
    free_logits: Tensor,
    eval_pairs: List[Tuple[str, str]],      # (x_str, y_raw_str)
    model,                                   # base HF model for discrete eval
    diff_model: STGSDiffModel,               # for soft eval
    stgs_module: STGS,
    embedding_weights: Tensor,               # (vocab, embed_dim)
    tokenizer,
    device,
    E_input_ids: Tensor,                     # (1, E_len)
    E_embeds: Tensor,                        # (1, E_len, embed_dim)
    extraction_fns: Dict[str, Callable],
    max_new_tokens_reasoning: int,
    max_new_tokens_answer: int,
    eval_mode: str = "discrete",             # "discrete" or "soft"
    reasoning_generation_backend: str = "diff",
    reasoning_generate_kwargs: Optional[Dict[str, Any]] = None,
    use_chat_template: bool = False,
    epoch: int = 0,
    eval_split: str = "eval",
    x_ids_cache: Optional[Dict] = None,
    x_embeds_cache: Optional[Dict] = None,
    accumulate_embeds: bool = False,
    prompt_logits_transform: Optional[
        Callable[[Tensor, Sequence[Tuple[str, str]]], Tensor]
    ] = None,
    return_examples: bool = False,
    max_examples: Optional[int] = None,
):
    """
    Evaluate the shared prompt on eval_pairs.

    eval_mode controls prompt representation during evaluation, while
    reasoning_generation_backend controls how the reasoning rollout is generated.
    Returns accuracy metrics per extraction method plus any_correct / all_correct.
    """
    if reasoning_generation_backend not in PC_REASONING_GENERATION_BACKENDS:
        raise ValueError(
            "reasoning_generation_backend must be one of "
            f"{sorted(PC_REASONING_GENERATION_BACKENDS)}, got {reasoning_generation_backend!r}"
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    reasoning_generate_kwargs = dict(reasoning_generate_kwargs or {})

    embedding_layer = model.get_input_embeddings()

    per_method_correct: Dict[str, int] = {m: 0 for m in extraction_fns}
    any_correct_count = 0
    all_correct_count = 0
    n_eval = len(eval_pairs)
    example_rows: List[Dict[str, Any]] = []

    E_text_rendered = _decode_trace_text(tokenizer, E_input_ids, skip_special_tokens=False)
    E_text_clean = _decode_trace_text(tokenizer, E_input_ids, skip_special_tokens=True)
    eval_pbar = tqdm(
        total=n_eval,
        desc=f"{eval_split} eval ({eval_mode})",
        leave=False,
    )

    try:
        with torch.no_grad():
            for sample_count, (x_str, y_raw_str) in enumerate(eval_pairs, start=1):
                effective_prompt_logits = free_logits
                if prompt_logits_transform is not None:
                    effective_prompt_logits = prompt_logits_transform(
                        free_logits,
                        [(x_str, y_raw_str)],
                    )

                p_text_weave = tokenizer.decode(
                    effective_prompt_logits.detach().argmax(dim=-1).squeeze(0).tolist(),
                    skip_special_tokens=False,
                )
                p_text_clean = tokenizer.decode(
                    effective_prompt_logits.detach().argmax(dim=-1).squeeze(0).tolist(),
                    skip_special_tokens=True,
                )

                x_ids = (x_ids_cache or {}).get(x_str) if x_ids_cache else None
                if x_ids is None:
                    x_ids = _tokenize_x(x_str, tokenizer, device, use_chat_template)
                x_text_rendered = _decode_trace_text(tokenizer, x_ids, skip_special_tokens=False)
                reasoning_trace = None
                answer_trace = None

                if eval_mode == "discrete":
                    p_ids = effective_prompt_logits.argmax(dim=-1)   # (1, seq_len)
                    prefix_ids = torch.cat([x_ids, p_ids], dim=1)
                    if reasoning_generation_backend == "hf_generate":
                        r_out = diff_model.generate(
                            input_ids=prefix_ids,
                            max_new_tokens=max_new_tokens_reasoning,
                            output_diff_one_hots=False,
                            return_dict=True,
                            generation_backend="hf_generate",
                            **reasoning_generate_kwargs,
                        )
                    else:
                        r_out = diff_model.generate(
                            input_ids=prefix_ids,
                            max_new_tokens=max_new_tokens_reasoning,
                            output_diff_one_hots=not accumulate_embeds,
                            output_normal_logits=False,
                            use_soft_embeds=True,
                            return_dict=True,
                            accumulate_embeds=accumulate_embeds,
                            generation_backend="diff",
                        )
                    R_ids_only = r_out.sampled_diff_tokens
                    R_text = tokenizer.decode(R_ids_only[0], skip_special_tokens=True)
                    R_text_raw = tokenizer.decode(R_ids_only[0], skip_special_tokens=False)
                    reasoning_trace = _build_pc_lm_trace(
                        tokenizer=tokenizer,
                        call_name="reasoning_generate",
                        split=eval_split,
                        epoch=epoch,
                        lm_name="diff_model.generate",
                        input_mode="input_ids",
                        output_mode="generated_tokens",
                        input_segments=[
                            _make_trace_segment(
                                "x_rendered",
                                text=x_text_rendered,
                                token_ids=x_ids,
                                is_exact=True,
                                source="chat_template" if use_chat_template else "plain_tokenization",
                            ),
                            _make_trace_segment(
                                "p",
                                text=p_text_weave,
                                token_ids=p_ids,
                                is_exact=True,
                                source="prompt_argmax_ids",
                            ),
                        ],
                        output_segments=[
                            _make_trace_segment(
                                "R_gen",
                                text=R_text_raw,
                                token_ids=R_ids_only,
                                is_exact=True,
                                source="generated_reasoning_tokens",
                            )
                        ],
                        input_token_ids=prefix_ids,
                        output_token_ids=R_ids_only,
                        output_exact_available=True,
                        metadata={
                            "call_role": "reasoning",
                            "eval_mode": eval_mode,
                            "reasoning_generation_backend": reasoning_generation_backend,
                            "use_chat_template": use_chat_template,
                            "soft_prompt_present": False,
                            "soft_reasoning_present": False,
                            "x_text_raw": x_str,
                            "x_text_rendered": x_text_rendered,
                            "p_text_argmax": p_text_weave,
                            "p_text_argmax_clean": p_text_clean,
                            "E_text": E_text_rendered,
                            "E_text_clean": E_text_clean,
                            "y_target_raw": y_raw_str,
                        },
                    )

                    # Extractor path
                    xpR_ids = torch.cat([x_ids, p_ids, R_ids_only, E_input_ids], dim=1)
                    Y_ids = model.generate(
                        xpR_ids,
                        max_new_tokens=max_new_tokens_answer,
                        pad_token_id=pad_id,
                        do_sample=False,
                    )
                    Y_output_ids = Y_ids[:, xpR_ids.shape[1]:]
                    Y_text = tokenizer.decode(Y_output_ids[0], skip_special_tokens=True)
                    Y_text_raw = tokenizer.decode(Y_output_ids[0], skip_special_tokens=False)
                    answer_trace = _build_pc_lm_trace(
                        tokenizer=tokenizer,
                        call_name="answer_generate",
                        split=eval_split,
                        epoch=epoch,
                        lm_name="base_model.generate",
                        input_mode="input_ids",
                        output_mode="generated_tokens",
                        input_segments=[
                            _make_trace_segment(
                                "x_rendered",
                                text=x_text_rendered,
                                token_ids=x_ids,
                                is_exact=True,
                                source="chat_template" if use_chat_template else "plain_tokenization",
                            ),
                            _make_trace_segment(
                                "p",
                                text=p_text_weave,
                                token_ids=p_ids,
                                is_exact=True,
                                source="prompt_argmax_ids",
                            ),
                            _make_trace_segment(
                                "R_gen",
                                text=R_text_raw,
                                token_ids=R_ids_only,
                                is_exact=True,
                                source="generated_reasoning_tokens",
                            ),
                            _make_trace_segment(
                                "E",
                                text=E_text_rendered,
                                token_ids=E_input_ids,
                                is_exact=True,
                                source="extraction_prompt",
                            ),
                        ],
                        output_segments=[
                            _make_trace_segment(
                                "Y_gen",
                                text=Y_text_raw,
                                token_ids=Y_output_ids,
                                is_exact=True,
                                source="generated_answer_tokens",
                            )
                        ],
                        input_token_ids=xpR_ids,
                        output_token_ids=Y_output_ids,
                        output_exact_available=True,
                        metadata={
                            "call_role": "answer",
                            "eval_mode": eval_mode,
                            "reasoning_generation_backend": reasoning_generation_backend,
                            "use_chat_template": use_chat_template,
                            "soft_prompt_present": False,
                            "soft_reasoning_present": False,
                            "x_text_raw": x_str,
                            "x_text_rendered": x_text_rendered,
                            "p_text_argmax": p_text_weave,
                            "p_text_argmax_clean": p_text_clean,
                            "R_gen_text": R_text_raw,
                            "R_gen_text_clean": R_text,
                            "E_text": E_text_rendered,
                            "E_text_clean": E_text_clean,
                            "y_target_raw": y_raw_str,
                            "output_kind": "generated_tokens",
                        },
                    )

                else:
                    # Soft eval: use STGSDiffModel with the per-sample prompt distribution
                    _, p_one_hot_eval, _, _ = stgs_module(
                        effective_prompt_logits,
                        gumbel_noise_scale=0.0,
                        embedding_weights=embedding_weights,
                    )
                    p_embeds_eval = p_one_hot_eval @ embedding_weights.detach()
                    x_embeds = (x_embeds_cache or {}).get(x_str) if x_embeds_cache else None
                    if x_embeds is None:
                        x_embeds = embedding_layer(x_ids)

                    reasoning_prefix_embeds = torch.cat([x_embeds, p_embeds_eval], dim=1)
                    if reasoning_generation_backend == "hf_generate":
                        r_out = diff_model.generate(
                            inputs_embeds=reasoning_prefix_embeds,
                            max_new_tokens=max_new_tokens_reasoning,
                            output_diff_one_hots=False,
                            return_dict=True,
                            generation_backend="hf_generate",
                            **reasoning_generate_kwargs,
                        )
                    else:
                        r_out = diff_model.generate(
                            inputs_embeds=reasoning_prefix_embeds,
                            max_new_tokens=max_new_tokens_reasoning,
                            output_diff_one_hots=not accumulate_embeds,
                            output_normal_logits=False,
                            use_soft_embeds=True,
                            return_dict=True,
                            accumulate_embeds=accumulate_embeds,
                            generation_backend="diff",
                        )
                    R_ids_only = r_out.sampled_diff_tokens
                    R_text = tokenizer.decode(R_ids_only[0], skip_special_tokens=True)
                    R_text_raw = tokenizer.decode(R_ids_only[0], skip_special_tokens=False)
                    p_eval_ids = p_one_hot_eval.detach().argmax(dim=-1)
                    p_eval_text = _decode_trace_text(tokenizer, p_eval_ids, skip_special_tokens=False)
                    p_eval_text_clean = _decode_trace_text(tokenizer, p_eval_ids, skip_special_tokens=True)
                    reasoning_trace = _build_pc_lm_trace(
                        tokenizer=tokenizer,
                        call_name="reasoning_generate",
                        split=eval_split,
                        epoch=epoch,
                        lm_name="diff_model.generate",
                        input_mode="inputs_embeds",
                        output_mode="generated_tokens",
                        input_segments=[
                            _make_trace_segment(
                                "x_rendered",
                                text=x_text_rendered,
                                token_ids=x_ids,
                                is_exact=True,
                                source="chat_template" if use_chat_template else "plain_tokenization",
                            ),
                            _make_trace_segment(
                                "p_argmax",
                                text=p_eval_text,
                                token_ids=p_eval_ids,
                                is_exact=False,
                                source="argmax_from_eval_prompt_distribution",
                            ),
                        ],
                        output_segments=[
                            _make_trace_segment(
                                "R_gen",
                                text=R_text_raw,
                                token_ids=R_ids_only,
                                is_exact=True,
                                source="generated_reasoning_tokens",
                            )
                        ],
                        output_token_ids=R_ids_only,
                        output_exact_available=True,
                        metadata={
                            "call_role": "reasoning",
                            "eval_mode": eval_mode,
                            "reasoning_generation_backend": reasoning_generation_backend,
                            "use_chat_template": use_chat_template,
                            "soft_prompt_present": True,
                            "soft_reasoning_present": False,
                            "x_text_raw": x_str,
                            "x_text_rendered": x_text_rendered,
                            "p_text_argmax": p_eval_text,
                            "p_text_argmax_clean": p_eval_text_clean,
                            "E_text": E_text_rendered,
                            "E_text_clean": E_text_clean,
                            "y_target_raw": y_raw_str,
                        },
                    )

                    # Extractor path
                    R_embeds = embedding_layer(R_ids_only.long())
                    y_out = diff_model.generate(
                        inputs_embeds=torch.cat([x_embeds, p_embeds_eval, R_embeds, E_embeds], dim=1),
                        max_new_tokens=max_new_tokens_answer,
                        output_diff_one_hots=True,
                        output_normal_logits=False,
                        use_soft_embeds=True,
                        return_dict=True,
                    )
                    Y_output_ids = y_out.sampled_diff_tokens
                    Y_text = tokenizer.decode(Y_output_ids[0], skip_special_tokens=True)
                    Y_text_raw = tokenizer.decode(Y_output_ids[0], skip_special_tokens=False)
                    answer_trace = _build_pc_lm_trace(
                        tokenizer=tokenizer,
                        call_name="answer_generate",
                        split=eval_split,
                        epoch=epoch,
                        lm_name="diff_model.generate",
                        input_mode="inputs_embeds",
                        output_mode="generated_tokens",
                        input_segments=[
                            _make_trace_segment(
                                "x_rendered",
                                text=x_text_rendered,
                                token_ids=x_ids,
                                is_exact=True,
                                source="chat_template" if use_chat_template else "plain_tokenization",
                            ),
                            _make_trace_segment(
                                "p_argmax",
                                text=p_eval_text,
                                token_ids=p_eval_ids,
                                is_exact=False,
                                source="argmax_from_eval_prompt_distribution",
                            ),
                            _make_trace_segment(
                                "R_gen",
                                text=R_text_raw,
                                token_ids=R_ids_only,
                                is_exact=True,
                                source="generated_reasoning_tokens",
                            ),
                            _make_trace_segment(
                                "E",
                                text=E_text_rendered,
                                token_ids=E_input_ids,
                                is_exact=True,
                                source="extraction_prompt",
                            ),
                        ],
                        output_segments=[
                            _make_trace_segment(
                                "Y_gen",
                                text=Y_text_raw,
                                token_ids=Y_output_ids,
                                is_exact=True,
                                source="generated_answer_tokens",
                            )
                        ],
                        output_token_ids=Y_output_ids,
                        output_exact_available=True,
                        metadata={
                            "call_role": "answer",
                            "eval_mode": eval_mode,
                            "reasoning_generation_backend": reasoning_generation_backend,
                            "use_chat_template": use_chat_template,
                            "soft_prompt_present": True,
                            "soft_reasoning_present": False,
                            "x_text_raw": x_str,
                            "x_text_rendered": x_text_rendered,
                            "p_text_argmax": p_eval_text,
                            "p_text_argmax_clean": p_eval_text_clean,
                            "R_gen_text": R_text_raw,
                            "R_gen_text_clean": R_text,
                            "E_text": E_text_rendered,
                            "E_text_clean": E_text_clean,
                            "y_target_raw": y_raw_str,
                            "output_kind": "generated_tokens",
                        },
                    )

                method_results = _compute_method_results(
                    R_text=R_text,
                    Y_text=Y_text,
                    y_raw_str=y_raw_str,
                    extraction_fns=extraction_fns,
                )
                for method, is_correct in method_results.items():
                    if is_correct:
                        per_method_correct[method] += 1

                any_c = any(method_results.values())
                all_c = all(method_results.values())
                any_correct_count += int(any_c)
                all_correct_count += int(all_c)
                if return_examples and (max_examples is None or len(example_rows) < max_examples):
                    example_rows.append({
                        "sample_index": sample_count - 1,
                        "split": eval_split,
                        "epoch": epoch,
                        "eval_mode": eval_mode,
                        "reasoning_generation_backend": reasoning_generation_backend,
                        "input_text": x_str,
                        "target_answer": y_raw_str,
                        "prompt_text": p_text_weave,
                        "generated_reasoning": R_text,
                        "generated_answer": Y_text,
                        "any_correct": int(any_c),
                        "all_correct": int(all_c),
                        **{f"correct_{method}": int(is_correct) for method, is_correct in method_results.items()},
                    })
                eval_metrics = _metrics_from_accuracy_counts(
                    per_method_correct=per_method_correct,
                    any_correct_count=any_correct_count,
                    all_correct_count=all_correct_count,
                    n_examples=sample_count,
                    extraction_fns=extraction_fns,
                )

                trace_common = {
                    "correct_per_method": method_results,
                    "any_correct": any_c,
                    "all_correct": all_c,
                    "y_target": y_raw_str,
                    "Y_text_clean": Y_text,
                    "R_text_clean": R_text,
                }
                if reasoning_trace is not None:
                    reasoning_trace.update(trace_common)
                    weave_lm_eval_step(**reasoning_trace)
                if answer_trace is not None:
                    answer_trace.update(trace_common)
                    weave_lm_eval_step(**answer_trace)
                eval_pbar.set_postfix(
                    _accuracy_postfix(eval_metrics, extraction_fns),
                    refresh=False,
                )
                eval_pbar.update(1)
    finally:
        eval_pbar.close()

    metrics = _metrics_from_accuracy_counts(
        per_method_correct=per_method_correct,
        any_correct_count=any_correct_count,
        all_correct_count=all_correct_count,
        n_examples=n_eval,
        extraction_fns=extraction_fns,
    )
    if return_examples:
        return metrics, example_rows
    return metrics


def evaluate_shared_prompt_batched(
    free_logits: Tensor,
    eval_pairs: List[Tuple[str, str]],
    model,
    diff_model,
    stgs_module,
    embedding_weights: Tensor,
    tokenizer,
    device,
    E_input_ids: Tensor,
    E_embeds: Tensor,
    extraction_fns: Dict[str, Callable],
    max_new_tokens_reasoning: int,
    max_new_tokens_answer: int,
    eval_mode: str = "discrete",
    reasoning_generation_backend: str = "diff",
    reasoning_generate_kwargs: Optional[Dict] = None,
    use_chat_template: bool = False,
    epoch: int = 0,
    eval_split: str = "eval",
    x_ids_cache: Optional[Dict] = None,
    x_embeds_cache: Optional[Dict] = None,
    accumulate_embeds: bool = False,
    prompt_logits_transform: Optional[Callable] = None,
    eval_inner_batch_size: int = 0,
) -> Dict[str, float]:
    """Batched counterpart of evaluate_shared_prompt.

    Processes all eval_pairs in a single generate call (per stage).
    Falls back to evaluate_shared_prompt when len(eval_pairs) <= 1.
    """
    if len(eval_pairs) <= 1:
        return evaluate_shared_prompt(
            free_logits=free_logits, eval_pairs=eval_pairs,
            model=model, diff_model=diff_model, stgs_module=stgs_module,
            embedding_weights=embedding_weights, tokenizer=tokenizer,
            device=device, E_input_ids=E_input_ids, E_embeds=E_embeds,
            extraction_fns=extraction_fns,
            max_new_tokens_reasoning=max_new_tokens_reasoning,
            max_new_tokens_answer=max_new_tokens_answer,
            eval_mode=eval_mode,
            reasoning_generation_backend=reasoning_generation_backend,
            reasoning_generate_kwargs=reasoning_generate_kwargs or {},
            use_chat_template=use_chat_template,
            epoch=epoch, eval_split=eval_split,
            x_ids_cache=x_ids_cache, x_embeds_cache=x_embeds_cache,
            accumulate_embeds=accumulate_embeds,
            prompt_logits_transform=prompt_logits_transform,
        )

    # Chunked mini-batch evaluation: process eval_pairs in sub-batches and aggregate.
    if eval_inner_batch_size > 0 and len(eval_pairs) > eval_inner_batch_size:
        _shared_kwargs = dict(
            free_logits=free_logits, model=model, diff_model=diff_model,
            stgs_module=stgs_module, embedding_weights=embedding_weights,
            tokenizer=tokenizer, device=device, E_input_ids=E_input_ids,
            E_embeds=E_embeds, extraction_fns=extraction_fns,
            max_new_tokens_reasoning=max_new_tokens_reasoning,
            max_new_tokens_answer=max_new_tokens_answer,
            eval_mode=eval_mode,
            reasoning_generation_backend=reasoning_generation_backend,
            reasoning_generate_kwargs=reasoning_generate_kwargs,
            use_chat_template=use_chat_template, epoch=epoch,
            eval_split=eval_split, x_ids_cache=x_ids_cache,
            x_embeds_cache=x_embeds_cache, accumulate_embeds=accumulate_embeds,
            prompt_logits_transform=prompt_logits_transform,
            eval_inner_batch_size=0,  # no further chunking within each mini-batch
        )
        agg: Dict[str, float] = {}
        total_n = 0
        n_chunks = (len(eval_pairs) + eval_inner_batch_size - 1) // eval_inner_batch_size
        chunk_pbar = tqdm(
            total=n_chunks,
            desc=f"{eval_split} eval chunks ({eval_mode})",
            leave=False,
        )
        try:
            for chunk_start in range(0, len(eval_pairs), eval_inner_batch_size):
                chunk = eval_pairs[chunk_start:chunk_start + eval_inner_batch_size]
                chunk_metrics = evaluate_shared_prompt_batched(eval_pairs=chunk, **_shared_kwargs)
                n = len(chunk)
                total_n += n
                for k, v in chunk_metrics.items():
                    agg[k] = agg.get(k, 0.0) + v * n
                chunk_pbar.update(1)
        finally:
            chunk_pbar.close()
        return {k: v / max(total_n, 1) for k, v in agg.items()}

    with torch.no_grad():
        embedding_layer = model.get_input_embeddings()
        p_ids = free_logits.detach().argmax(dim=-1)   # (1, seq_len)

        # Tokenize all x inputs
        x_ids_list = []
        for x_str, _ in eval_pairs:
            if x_ids_cache and x_str in x_ids_cache:
                x_ids_list.append(x_ids_cache[x_str])
            else:
                x_ids_list.append(_tokenize_x(x_str, tokenizer, device, use_chat_template))

        B = len(eval_pairs)

        if eval_mode == "discrete":
            # Stage 1: left-pad [x_i | p] token ids for batched reasoning generate
            xp_seqs = [torch.cat([xi, p_ids], dim=1) for xi in x_ids_list]
            max_xp  = max(s.shape[1] for s in xp_seqs)
            xp_pad  = torch.full((B, max_xp), tokenizer.pad_token_id, dtype=torch.long, device=device)
            xp_mask = torch.zeros(B, max_xp, dtype=torch.long, device=device)
            for i, s in enumerate(xp_seqs):
                L = s.shape[1]; xp_pad[i, max_xp-L:] = s[0]; xp_mask[i, max_xp-L:] = 1

            _r_gen_kwargs = dict(reasoning_generate_kwargs or {}) if reasoning_generation_backend == "hf_generate" else {}
            r_gen = model.generate(
                input_ids=xp_pad, attention_mask=xp_mask,
                max_new_tokens=max_new_tokens_reasoning,
                pad_token_id=tokenizer.pad_token_id,
                **_r_gen_kwargs,
            )
            R_ids = r_gen.sequences[:, max_xp:]   # (B, max_new_tokens_reasoning)

            # Stage 2: left-pad [x_i | p | R_i | E]
            xpRE_seqs = [
                torch.cat([x_ids_list[i], p_ids, R_ids[i:i+1], E_input_ids], dim=1)
                for i in range(B)
            ]
            max_xpRE = max(s.shape[1] for s in xpRE_seqs)
            xpRE_pad  = torch.full((B, max_xpRE), tokenizer.pad_token_id, dtype=torch.long, device=device)
            xpRE_mask = torch.zeros(B, max_xpRE, dtype=torch.long, device=device)
            for i, s in enumerate(xpRE_seqs):
                L = s.shape[1]; xpRE_pad[i, max_xpRE-L:] = s[0]; xpRE_mask[i, max_xpRE-L:] = 1

            y_gen   = model.generate(
                input_ids=xpRE_pad, attention_mask=xpRE_mask,
                max_new_tokens=max_new_tokens_answer,
                pad_token_id=tokenizer.pad_token_id,
            )
            Y_ids   = y_gen.sequences[:, max_xpRE:]   # (B, max_new_tokens_answer)
            R_for_accuracy = R_ids

        else:  # soft
            if x_embeds_cache:
                x_emb_list = [
                    x_embeds_cache.get(x_str) if x_embeds_cache.get(x_str) is not None else embedding_layer(x_ids_list[i])
                    for i, (x_str, _) in enumerate(eval_pairs)
                ]
            else:
                x_emb_list = [embedding_layer(xid) for xid in x_ids_list]

            _, oh, _, _ = stgs_module(
                free_logits, gumbel_noise_scale=0.0, embedding_weights=embedding_weights,
            )
            p_emb = oh @ embedding_weights.detach()   # (1, seq_len, d)

            xp_seqs = [torch.cat([xe, p_emb], dim=1) for xe in x_emb_list]
            xp_padded, xp_mask = _left_pad_embed_sequences(xp_seqs, device, x_emb_list[0].dtype)
            _use_hf_backend = reasoning_generation_backend == "hf_generate"
            _r_gen_extra = {"generation_backend": "hf_generate", "output_diff_one_hots": False, **(reasoning_generate_kwargs or {})} if _use_hf_backend else {}
            r_out = diff_model.generate(
                inputs_embeds=xp_padded, attention_mask=xp_mask,
                max_new_tokens=max_new_tokens_reasoning,
                output_normal_logits=False, return_dict=True,
                accumulate_embeds=accumulate_embeds and not _use_hf_backend,
                **_r_gen_extra,
            )
            if _use_hf_backend:
                R_emb = embedding_layer(r_out.sampled_diff_tokens.long()).detach()
            else:
                R_emb = (
                    r_out.sampled_diff_embeds if r_out.sampled_diff_embeds is not None
                    else r_out.sampled_diff_one_hot @ embedding_weights.detach()
                ).detach()
            R_ids_soft = r_out.sampled_diff_tokens   # (B, R_gen_len) for text decoding

            xpRE_seqs = [
                torch.cat([x_emb_list[i], p_emb, R_emb[i:i+1], E_embeds], dim=1)
                for i in range(B)
            ]
            xpRE_padded, xpRE_mask = _left_pad_embed_sequences(xpRE_seqs, device, x_emb_list[0].dtype)
            y_out = diff_model.generate(
                inputs_embeds=xpRE_padded, attention_mask=xpRE_mask,
                max_new_tokens=max_new_tokens_answer,
                output_normal_logits=False, return_dict=True,
            )
            Y_ids = y_out.sampled_diff_tokens          # (B, max_new_tokens_answer)
            R_for_accuracy = R_ids_soft

        # Aggregate accuracy
        any_correct    = 0
        all_correct    = 0
        method_correct: Dict[str, int] = {m: 0 for m in extraction_fns}
        for i, (_, y_raw) in enumerate(eval_pairs):
            R_text = tokenizer.decode(R_for_accuracy[i].tolist(), skip_special_tokens=True)
            Y_text = tokenizer.decode(Y_ids[i].tolist(), skip_special_tokens=True)
            res = _compute_method_results(
                R_text=R_text, Y_text=Y_text,
                y_raw_str=y_raw, extraction_fns=extraction_fns,
            )
            for m, ok in res.items():
                if ok: method_correct[m] += 1
            any_correct += int(any(res.values()))
            all_correct += int(all(res.values()))
            if is_weave_active():
                _p_text = tokenizer.decode(p_ids.squeeze(0).tolist(), skip_special_tokens=False)
                _p_text_clean = tokenizer.decode(p_ids.squeeze(0).tolist(), skip_special_tokens=True)
                weave_lm_eval_step(
                    call_name="answer_generate_batched_eval",
                    split=eval_split,
                    epoch=epoch,
                    lm_name="model.generate" if eval_mode == "discrete" else "diff_model.generate",
                    input_mode="input_ids" if eval_mode == "discrete" else "inputs_embeds",
                    output_mode="generated_tokens",
                    call_role="answer",
                    batched=True,
                    eval_mode=eval_mode,
                    reasoning_generation_backend=reasoning_generation_backend,
                    use_chat_template=use_chat_template,
                    x_text_raw=eval_pairs[i][0],
                    p_text_argmax=_p_text,
                    p_text_argmax_clean=_p_text_clean,
                    R_text_clean=R_text,
                    Y_text_clean=Y_text,
                    y_target=y_raw,
                    correct_per_method=res,
                    any_correct=any(res.values()),
                    all_correct=all(res.values()),
                )

        return _metrics_from_accuracy_counts(
            per_method_correct=method_correct,
            any_correct_count=any_correct,
            all_correct_count=all_correct,
            n_examples=B,
            extraction_fns=extraction_fns,
        )


# ---------------------------------------------------------------------------
# LoRA SVD initialization utilities
# ---------------------------------------------------------------------------

def lora_svd_reconstruct(
    token_ids: List[int],
    lora_rank: int,
    embedding_weights: Tensor,
    vocab_size: int,
    device: torch.device,
    spike: float = 10.0,
) -> Dict[str, Any]:
    """Decompose a one-hot logit matrix via rank-r SVD and decode all three
    reconstruction methods (argmax, embsim-l2, embsim-cos).

    Creates a ``(seq_len, vocab_size)`` one-hot matrix with *spike* at each
    target token position, computes the rank-*r* truncated SVD factorisation
    ``lora_A @ lora_B ≈ M``, then decodes the approximation with three
    methods so the caller can compare reconstruction quality.

    Args:
        token_ids:        Target token ID sequence (length = seq_len).
        lora_rank:        Desired LoRA rank; clamped to ``min(rank, seq_len)``.
        embedding_weights: ``(vocab_size, embed_dim)`` embedding table (detached).
        vocab_size:       Vocabulary size.
        device:           PyTorch device for all tensors.
        spike:            Logit value placed at each target token position in
                          the one-hot matrix (default 10.0).

    Returns:
        Dict with keys:

        * ``lora_A`` – ``(1, seq_len, effective_rank)`` tensor (no grad).
        * ``lora_B`` – ``(1, effective_rank, vocab_size)`` tensor (no grad).
        * ``reconstructed`` – ``(seq_len, vocab_size)`` float tensor
          ``= lora_A.squeeze(0) @ lora_B.squeeze(0)``.
        * ``effective_rank`` – actual rank used after clamping.
        * ``recon_error`` – Frobenius norm ``||M - reconstructed||_F``.
        * ``argmax_ids`` – ``List[int]`` from ``argmax(reconstructed, dim=-1)``.
        * ``embsim_l2_ids`` – ``List[int]`` nearest token by L2 distance.
        * ``embsim_cos_ids`` – ``List[int]`` nearest token by cosine similarity.
    """
    seq_len = len(token_ids)
    token_ids_t = torch.tensor(token_ids, dtype=torch.long, device=device)

    # Build one-hot logit matrix.
    oh = torch.zeros(seq_len, vocab_size, device=device)
    oh[torch.arange(seq_len, device=device), token_ids_t] = spike

    # Rank-r SVD decomposition.
    mat = oh.float()
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    r = min(lora_rank, S.shape[0])
    sq = S[:r].sqrt()
    A = (U[:, :r] * sq.unsqueeze(0))    # (seq_len, r)
    B = (sq.unsqueeze(1) * Vh[:r, :])   # (r, vocab_size)
    lora_A = A.unsqueeze(0)             # (1, seq_len, r)
    lora_B = B.unsqueeze(0)             # (1, r, vocab_size)

    reconstructed = (lora_A @ lora_B).squeeze(0).float()  # (seq_len, vocab_size)
    recon_error = float((oh.float() - reconstructed).norm().item())

    # Argmax reconstruction.
    argmax_ids = reconstructed.argmax(dim=-1).cpu().tolist()

    # Embsim-L2 and embsim-cos reconstructions.
    with torch.no_grad():
        _emb = embedding_weights.float().to(device)     # (vocab, embed_dim)
        _probs = torch.softmax(reconstructed.unsqueeze(0), dim=-1)  # (1, seq_len, vocab)
        _soft_emb = torch.matmul(_probs, _emb)          # (1, seq_len, embed_dim)

        # L2.
        _soft_sq = (_soft_emb ** 2).sum(dim=-1, keepdim=True)   # (1, seq_len, 1)
        _emb_sq = (_emb ** 2).sum(dim=-1)                        # (vocab,)
        _cross = torch.matmul(_soft_emb, _emb.T)                 # (1, seq_len, vocab)
        _l2_sq = _soft_sq - 2 * _cross + _emb_sq                 # (1, seq_len, vocab)
        embsim_l2_ids = _l2_sq.argmin(dim=-1).squeeze(0).cpu().tolist()

        # Cosine.
        _soft_norm = F.normalize(_soft_emb, dim=-1, eps=1e-10)   # (1, seq_len, embed_dim)
        _e_norm = F.normalize(_emb, dim=-1, eps=1e-10)            # (vocab, embed_dim)
        _cos_sim = torch.matmul(_soft_norm, _e_norm.T)            # (1, seq_len, vocab)
        embsim_cos_ids = _cos_sim.argmax(dim=-1).squeeze(0).cpu().tolist()

    return {
        "lora_A": lora_A.detach().cpu(),
        "lora_B": lora_B.detach().cpu(),
        "reconstructed": reconstructed.cpu(),
        "effective_rank": r,
        "recon_error": recon_error,
        "argmax_ids": argmax_ids,
        "embsim_l2_ids": embsim_l2_ids,
        "embsim_cos_ids": embsim_cos_ids,
    }


# ---------------------------------------------------------------------------
# Core optimization loop
# ---------------------------------------------------------------------------

def pc_optimize_inputs(
    model,
    diff_model: STGSDiffModel,
    tokenizer,
    device,
    model_precision: str,
    # Dataset
    train_triples: List[Tuple[str, str, str]],   # (x_str, R_gt_str, y_raw_str)
    val_pairs: Optional[List[Tuple[str, str]]],  # (x_str, y_raw_str)
    # Extraction
    extraction_prompt: str,
    extraction_fns: Dict[str, Callable],
    max_new_tokens_reasoning: int = 200,
    max_new_tokens_answer: int = 20,
    teacher_forcing_r: bool = True,
    bptt: bool = False,
    reasoning_generation_backend: str = "diff",
    reasoning_generate_kwargs: Optional[Dict[str, Any]] = None,
    # Loss
    losses: str = "crossentropy",
    # Optimization
    seq_len: int = 20,
    initial_prompt_text: Optional[str] = None,
    initial_prompt_lora_reconstruction: str = "argmax",
    initial_prompt_lora_spike: float = 10.0,
    epochs: int = 2000,
    learning_rate: float = 0.01,
    inner_batch_size: int = 4,
    batch_size: Optional[int] = None,
    # STGS for prompt p
    stgs_module: STGS = None,
    temperature: float = 1.0,
    learnable_temperature: bool = False,
    decouple_learnable_temperature: bool = False,
    stgs_hard: bool = True,
    stgs_hard_method: str = "categorical",
    stgs_hard_embsim_probs: str = "gumbel_soft",
    stgs_hard_embsim_strategy: str = "nearest",
    stgs_hard_embsim_top_k: int = 8,
    stgs_hard_embsim_rerank_alpha: float = 0.5,
    stgs_hard_embsim_sample_tau: float = 1.0,
    stgs_hard_embsim_margin: float = 0.0,
    stgs_hard_embsim_fallback: str = "argmax",
    bptt_temperature: float = 1.0,
    bptt_learnable_temperature: bool = False,
    bptt_decouple_learnable_temperature: bool = False,
    bptt_stgs_hard: bool = True,
    bptt_stgs_hard_method: str = "categorical",
    bptt_stgs_hard_embsim_probs: str = "gumbel_soft",
    bptt_stgs_hard_embsim_strategy: str = "nearest",
    bptt_stgs_hard_embsim_top_k: int = 8,
    bptt_stgs_hard_embsim_rerank_alpha: float = 0.5,
    bptt_stgs_hard_embsim_sample_tau: float = 1.0,
    bptt_stgs_hard_embsim_margin: float = 0.0,
    bptt_stgs_hard_embsim_fallback: str = "argmax",
    bptt_hidden_state_conditioning: bool = False,
    logits_normalize: str = "none",
    bptt_logits_normalize: str = "none",
    stgs_input_dropout: float = 0.0,
    stgs_output_dropout: float = 0.0,
    eps: float = 1e-10,
    bptt_eps: float = 1e-10,
    gumbel_noise_scale: float = 1.0,
    adaptive_gumbel_noise: bool = False,
    adaptive_gumbel_noise_beta: float = 0.9,
    adaptive_gumbel_noise_min_scale: float = 0.0,
    # Logits filtering
    logits_top_k: int = 0,
    logits_top_p: float = 1.0,
    logits_filter_penalty: float = 1e4,
    # Initialization
    init_strategy: str = "randn",
    init_std: float = 0.0,
    init_mlm_model: str = "distilbert-base-uncased",
    init_mlm_top_k: int = 50,
    logits_lora_rank: int = 0,
    logits_lora_b_learning_rate: Optional[float] = None,
    # LR scheduler
    lr_schedule: str = "none",
    lr_warmup_epochs: int = 0,
    lr_schedule_min: float = 0.0,
    lr_schedule_step_size: int = 10,
    lr_schedule_gamma: float = 0.1,
    # Schedule
    max_gradient_norm: float = 0.0,
    temperature_anneal_schedule: str = "none",
    temperature_anneal_min: float = 0.1,
    temperature_anneal_epochs: int = 0,
    temperature_anneal_reg_lambda: float = 0.0,
    temperature_anneal_reg_mode: str = "mse",
    temperature_loss_coupling_lambda: float = 0.0,
    discrete_reinit_epoch: int = 0,
    discrete_reinit_snap: str = "argmax",
    discrete_reinit_prob: float = 1.0,
    discrete_reinit_topk: int = 0,
    discrete_reinit_entropy_threshold: float = 0.0,
    discrete_reinit_embsim_probs: str = "input_logits",
    logit_decay: float = 0.0,
    prompt_length_learnable: bool = False,
    prompt_length_alpha_init: float = 0.0,
    prompt_length_beta: float = 5.0,
    prompt_length_reg_lambda: float = 0.0,
    prompt_length_eos_spike: float = 10.0,
    prompt_length_mask_eos_attention: bool = False,
    ppo_kl_lambda: float = 0.0,
    ppo_kl_mode: str = "soft",
    ppo_kl_epsilon: float = 0.0,
    ppo_ref_update_period: int = 10,
    superposition_metric_every: int = 0,
    superposition_metric_modes: str = "dot,cos,l2",
    superposition_vocab_top_k: int = 256,
    superposition_vocab_source: str = "wikipedia",
    superposition_vocab_dataset_path: Optional[str] = None,
    superposition_vocab_hf_name: str = "lucadiliello/english_wikipedia",
    superposition_vocab_hf_split: str = "train",
    superposition_vocab_num_texts: int = 1000,
    superposition_entropy_temperature: float = 1.0,
    superposition_output_dir: Optional[str] = ".",
    fixed_gt_prefix_n: int = 0,
    fixed_gt_suffix_n: int = 0,
    fixed_gt_prefix_rank2_n: int = 0,
    fixed_gt_suffix_rank2_n: int = 0,
    fixed_prefix_text: Optional[str] = None,
    fixed_suffix_text: Optional[str] = None,
    early_stop_loss_threshold: float = 0.0,
    # Validation
    val_eval_every: int = 100,
    val_prompt_eval_mode: str = "discrete",
    per_epoch_callback: Optional[Callable] = None,
    # Batched forward pass
    use_batched_forward_pass: bool = False,
    batched_stgs_noise_mode: str = "shared",       # "shared" | "independent"
    # Batched evaluation
    use_batched_val_eval: bool = False,
    eval_inner_batch_size: int = 0,
    # Gradient accumulation
    gradient_accumulation_steps: int = 1,
    # SWA (Stochastic Weight Averaging over prompt logits)
    use_swa: bool = False,
    swa_start_epoch: Optional[int] = None,
    swa_freq: int = 1,
    # Multi-Gumbel sampling (N independent draws per mini-batch, losses averaged)
    gumbel_n_samples: int = 1,
    # Pre-training validation
    val_eval_before_training: bool = False,
    test_eval_before_training: bool = False,
    # Chat template
    use_chat_template: bool = False,
    # Memory / speed optimizations
    accumulate_embeds: bool = False,
    prompt_logits_transform: Optional[
        Callable[[Tensor, Sequence[Tuple[str, str]]], Tensor]
    ] = None,
    # MAS rotational metric (opt-in; requires logits_lora_rank > 0)
    mas_metric_every: int = 0,
    mas_hvp_mode: str = "autograd",   # "autograd" | "jvp"
    # Offline EM: cache reasoning at E-step, optimize prompt at M-step
    offline_em: bool = False,
    offline_em_resample_every: int = 0,          # 0 = never resample; N > 0 = resample every N epochs
    offline_em_stgs_method: str = "soft",        # "argmax"|"embsim-dot"|"embsim-cos"|"embsim-l2"|"soft"
    offline_em_stgs_embsim_probs: str = "gumbel_soft",       # "gumbel_soft"|"input_logits"
    offline_em_temperature: Union[str, float] = "learned",   # "learned" | float
    offline_em_cache_batch_size: int = 0,                    # 0 = fall back to inner_batch_size
    # Extra kwargs forwarded to LossClass
    kwargs: dict = {},
) -> Dict:
    """
    Optimize a single shared prompt p across all (x, R_gt, y) training pairs.

    Returns:
        {
          "final_p_logits": Tensor,
          "final_p_tokens": List[int],
          "final_p_text": str,
          "loss_history": List[float],
          "train_accuracy_history": List[Dict],
          "val_accuracy_history": List[Dict],
        }
    """
    compatibility_config = {
        "gradient_estimator": kwargs.get("gradient_estimator", "stgs"),
        "method": kwargs.get("method", "stgs"),
        "teacher_forcing": kwargs.get("teacher_forcing", False),
        "teacher_forcing_r": teacher_forcing_r,
        "bptt": bptt,
        "reasoning_generation_backend": reasoning_generation_backend,
        "bptt_teacher_forcing_via_diff_model": kwargs.get("bptt_teacher_forcing_via_diff_model", False),
        "filter_vocab": kwargs.get("filter_vocab", False),
        "early_stop_on_exact_match": kwargs.get("early_stop_on_exact_match", False),
        "run_discrete_validation": kwargs.get("run_discrete_validation", False),
        "run_discrete_embsim_validation": kwargs.get("run_discrete_embsim_validation", False),
        "prompt_length_mask_eos_attention": prompt_length_mask_eos_attention,
        "fixed_gt_prefix_n": fixed_gt_prefix_n,
        "fixed_gt_suffix_n": fixed_gt_suffix_n,
        "fixed_gt_prefix_rank2_n": fixed_gt_prefix_rank2_n,
        "fixed_gt_suffix_rank2_n": fixed_gt_suffix_rank2_n,
        "init_strategy": init_strategy,
        "superposition_metric_every": kwargs.get("superposition_metric_every", 0),
        "offline_em": offline_em,
        "use_batched_forward_pass": use_batched_forward_pass,
    }
    ensure_pc_main_compatibility(compatibility_config)

    if batch_size is not None:
        inner_batch_size = batch_size
    if reasoning_generation_backend not in PC_REASONING_GENERATION_BACKENDS:
        raise ValueError(
            "reasoning_generation_backend must be one of "
            f"{sorted(PC_REASONING_GENERATION_BACKENDS)}, got {reasoning_generation_backend!r}"
        )
    reasoning_generate_kwargs = dict(reasoning_generate_kwargs or {})
    reserved_reasoning_generate_keys = {
        "input_ids",
        "input_one_hots",
        "inputs_embeds",
        "attention_mask",
        "max_length",
        "max_new_tokens",
        "return_dict",
        "return_dict_in_generate",
        "generation_backend",
        "use_bpttoken",
        "accumulate_embeds",
        "output_diff_one_hots",
        "output_normal_logits",
        "output_stgs_logits",
    }
    overlapping_reasoning_keys = sorted(
        reserved_reasoning_generate_keys.intersection(reasoning_generate_kwargs)
    )
    if overlapping_reasoning_keys:
        raise ValueError(
            "reasoning_generate_kwargs may not override internally managed generation arguments; "
            f"got reserved keys: {', '.join(overlapping_reasoning_keys)}"
        )
    if use_swa and swa_start_epoch is None:
        swa_start_epoch = int(0.75 * epochs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if discrete_reinit_entropy_threshold < 0.0:
        raise ValueError(
            f"discrete_reinit_entropy_threshold must be >= 0, got {discrete_reinit_entropy_threshold}"
        )
    if discrete_reinit_embsim_probs not in {"input_logits", "gumbel_soft"}:
        raise ValueError(
            f"discrete_reinit_embsim_probs must be 'input_logits' or 'gumbel_soft', "
            f"got {discrete_reinit_embsim_probs!r}"
        )

    vocab_size = model.config.vocab_size
    embedding_layer = model.get_input_embeddings()
    embedding_weights = embedding_layer.weight.detach()  # (vocab, embed_dim)
    allowed_tokens = torch.arange(vocab_size, device=device)
    allowed_token_to_idx = {token_id: token_id for token_id in range(vocab_size)}

    if hasattr(diff_model, "stgs"):
        diff_model.stgs.init_temperature = bptt_temperature
        diff_model.stgs.logits_normalize = bptt_logits_normalize
        diff_model.stgs.stgs_hard = bptt_stgs_hard
        diff_model.stgs.stgs_hard_method = bptt_stgs_hard_method
        diff_model.stgs.stgs_hard_embsim_probs = bptt_stgs_hard_embsim_probs
        diff_model.stgs.stgs_hard_embsim_strategy = bptt_stgs_hard_embsim_strategy
        diff_model.stgs.stgs_hard_embsim_top_k = bptt_stgs_hard_embsim_top_k
        diff_model.stgs.stgs_hard_embsim_rerank_alpha = bptt_stgs_hard_embsim_rerank_alpha
        diff_model.stgs.stgs_hard_embsim_sample_tau = bptt_stgs_hard_embsim_sample_tau
        diff_model.stgs.stgs_hard_embsim_margin = bptt_stgs_hard_embsim_margin
        diff_model.stgs.stgs_hard_embsim_fallback = bptt_stgs_hard_embsim_fallback
        diff_model.stgs.eps = bptt_eps
        if bptt_learnable_temperature and not getattr(diff_model.stgs, "learnable_temperature", False):
            raise ValueError("bptt_learnable_temperature=True requires diff_model.stgs to be initialized with learnable temperature support.")
        if bptt_hidden_state_conditioning and getattr(diff_model.stgs, "conditioning_dim", 0) < 1:
            raise ValueError("bptt_hidden_state_conditioning=True requires diff_model.stgs to be initialized with hidden-state conditioning.")

    # -----------------------------------------------------------------------
    # Pre-tokenize and pre-embed all inputs
    # -----------------------------------------------------------------------
    E_input_ids = tokenizer(
        extraction_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    with torch.no_grad():
        E_embeds = embedding_layer(E_input_ids)   # (1, E_len, d)

    # Per-sample precomputed tensors
    x_embeds_cache: Dict[str, Tensor] = {}
    x_ids_cache: Dict[str, Tensor] = {}
    R_gt_ids_cache: Dict[Tuple[str, str], Optional[Tensor]] = {}
    R_gt_embeds_cache: Dict[Tuple[str, str], Optional[Tensor]] = {}
    Y_tokens_cache: Dict[str, Tensor] = {}

    all_x_strs = list({t[0] for t in train_triples})
    for x_str in all_x_strs:
        x_ids = _tokenize_x(x_str, tokenizer, device, use_chat_template)
        x_ids_cache[x_str] = x_ids
        with torch.no_grad():
            x_embeds_cache[x_str] = embedding_layer(x_ids)

    for x_str, R_gt_str, _ in train_triples:
        key = (x_str, R_gt_str)
        if key not in R_gt_embeds_cache:
            if R_gt_str:
                r_ids = tokenizer(R_gt_str, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
                R_gt_ids_cache[key] = r_ids
                with torch.no_grad():
                    R_gt_embeds_cache[key] = embedding_layer(r_ids)
            else:
                R_gt_ids_cache[key] = None
                R_gt_embeds_cache[key] = None

    for _, _, y_raw_str in train_triples:
        if y_raw_str not in Y_tokens_cache:
            Y_tokens_cache[y_raw_str] = tokenizer(
                y_raw_str, return_tensors="pt"
            ).input_ids.to(device)

    # Also precompute for val pairs
    if val_pairs:
        for x_str, _ in val_pairs:
            if x_str not in x_embeds_cache:
                x_ids = _tokenize_x(x_str, tokenizer, device, use_chat_template)
                x_ids_cache[x_str] = x_ids
                with torch.no_grad():
                    x_embeds_cache[x_str] = embedding_layer(x_ids)

    # -----------------------------------------------------------------------
    # Override seq_len from initial_prompt_text (if provided)
    # -----------------------------------------------------------------------
    _initial_prompt_token_ids: Optional[Tensor] = None
    if initial_prompt_text:
        _tok_out = tokenizer(initial_prompt_text, add_special_tokens=False, return_tensors="pt")
        _initial_prompt_token_ids = _tok_out.input_ids[0].to(device)   # (n_tokens,)
        seq_len = int(_initial_prompt_token_ids.shape[0])
        logger.info(
            "initial_prompt_text overrides seq_len → %d tokens: %r",
            seq_len, initial_prompt_text,
        )

    # -----------------------------------------------------------------------
    # Build LossClass cache keyed by (y_raw_str, Y_len)
    # -----------------------------------------------------------------------
    lc_kwargs = dict(kwargs)
    lc_kwargs.setdefault("seq_len", seq_len)

    loss_cache: Dict[Tuple[str, int], LossClass] = {}
    for _, _, y_raw_str in train_triples:
        Y_tokens = Y_tokens_cache[y_raw_str]
        Y_len = Y_tokens.shape[1]
        key = (y_raw_str, Y_len)
        if key not in loss_cache:
            loss_cache[key] = LossClass(
                losses=losses,
                embedding_weights_subset=embedding_weights,
                target_text=y_raw_str,
                target_tokens_mapped=Y_tokens,
                model=model,
                tokenizer=tokenizer,
                kwargs=lc_kwargs,
            )

    # -----------------------------------------------------------------------
    # Fixed prompt spec, learnable prompt init, and prompt assembly
    # -----------------------------------------------------------------------
    fixed_prefix, fixed_suffix, n_free = build_fixed_logits_spec(
        ground_truth_prompt_tokens=None,
        allowed_token_to_idx=allowed_token_to_idx,
        allowed_vocab_size=vocab_size,
        tokenizer=tokenizer,
        device=device,
        seq_len=seq_len,
        fixed_gt_prefix_n=0,
        fixed_gt_suffix_n=0,
        fixed_gt_prefix_rank2_n=0,
        fixed_gt_suffix_rank2_n=0,
        fixed_prefix_text=fixed_prefix_text,
        fixed_suffix_text=fixed_suffix_text,
    )
    prefix_len = fixed_prefix.shape[1] if fixed_prefix is not None else 0

    lora_A = lora_B = None
    if logits_lora_rank > 0:
        lora_A, lora_B = initialize_lora_logits(
            n_free, vocab_size, logits_lora_rank, device,
            init_strategy=init_strategy,
            init_std=init_std,
            embedding_weights=embedding_weights,
            allowed_tokens=allowed_tokens,
            mlm_model_name=init_mlm_model,
            mlm_top_k=init_mlm_top_k,
        )
        free_logits = None
        parameters = [lora_A, lora_B]
    else:
        free_logits = initialize_learnable_inputs(
            vocab_size, n_free, device,
            init_strategy=init_strategy,
            init_std=init_std,
            embedding_weights=embedding_weights,
            allowed_tokens=allowed_tokens,
            mlm_model_name=init_mlm_model,
            mlm_top_k=init_mlm_top_k,
        )
        parameters = [free_logits]

    # One-hot initialization from initial_prompt_text
    if _initial_prompt_token_ids is not None:
        _oh = torch.zeros(1, seq_len, vocab_size, device=device)
        _oh[0, torch.arange(seq_len, device=device), _initial_prompt_token_ids.long()] = initial_prompt_lora_spike

        if logits_lora_rank > 0:
            # SVD rank-r approximation via shared utility.
            _svd = lora_svd_reconstruct(
                token_ids=_initial_prompt_token_ids.cpu().tolist(),
                lora_rank=logits_lora_rank,
                embedding_weights=embedding_weights,
                vocab_size=vocab_size,
                device=device,
                spike=initial_prompt_lora_spike,
            )
            r = _svd["effective_rank"]
            lora_A = _svd["lora_A"].to(device).to(_oh.dtype).requires_grad_(True)
            lora_B = _svd["lora_B"].to(device).to(_oh.dtype).requires_grad_(True)
            parameters = [lora_A, lora_B]

            requested_ids = _initial_prompt_token_ids.cpu().tolist()
            requested_text_dec = tokenizer.decode(requested_ids, skip_special_tokens=False)
            recon_error = _svd["recon_error"]
            argmax_ids = _svd["argmax_ids"]
            embsim_l2_ids = _svd["embsim_l2_ids"]
            embsim_cos_ids = _svd["embsim_cos_ids"]
            argmax_text_dec = tokenizer.decode(argmax_ids, skip_special_tokens=False)
            embsim_l2_text_dec = tokenizer.decode(embsim_l2_ids, skip_special_tokens=False)
            embsim_cos_text_dec = tokenizer.decode(embsim_cos_ids, skip_special_tokens=False)

            # Select the primary reconstruction according to the chosen method.
            _recon_map = {
                "argmax": (argmax_ids, argmax_text_dec),
                "embsim-l2": (embsim_l2_ids, embsim_l2_text_dec),
                "embsim-cos": (embsim_cos_ids, embsim_cos_text_dec),
            }
            recon_ids, reconstructed_text_dec = _recon_map[initial_prompt_lora_reconstruction]
            is_exact = (recon_ids == requested_ids)

            def _label(method: str) -> str:
                return " [PRIMARY]" if method == initial_prompt_lora_reconstruction else ""

            # Loud warning — printed to both stdout and stderr so it is
            # impossible to miss in logs or terminal output.
            _SEP = "!" * 72
            _msg_lines = [
                "",
                _SEP,
                "  LOSSY INIT WARNING: initial_prompt_text + logits_lora_rank > 0",
                f"  Rank-{r} SVD of one-hot logits (seq_len={seq_len}, vocab={vocab_size})",
                f"  Requested          : {requested_text_dec!r}",
                f"  Argmax(lora){_label('argmax')}: {argmax_text_dec!r}",
                f"  EmbsimL2(lora){_label('embsim-l2')}: {embsim_l2_text_dec!r}",
                f"  EmbsimCos(lora){_label('embsim-cos')}: {embsim_cos_text_dec!r}",
                f"  Token-exact [{initial_prompt_lora_reconstruction}]: {is_exact}",
                f"  ||M - AB||_F : {recon_error:.4f}",
                _SEP,
                "",
            ]
            _msg = "\n".join(_msg_lines)
            print(_msg, flush=True)
            print(_msg, file=sys.stderr, flush=True)
            logger.warning(
                "initial_prompt_text with logits_lora_rank=%d: rank-%d SVD approximation. "
                "reconstruction_method=%s  requested=%r  argmax=%r  embsim_l2=%r  embsim_cos=%r  "
                "token_exact=%s  recon_error=%.4f",
                logits_lora_rank, r,
                initial_prompt_lora_reconstruction,
                requested_text_dec, argmax_text_dec, embsim_l2_text_dec, embsim_cos_text_dec,
                is_exact, recon_error,
            )

            if is_weave_active():
                weave_prompt_init(
                    requested_text=requested_text_dec,
                    requested_token_ids=requested_ids,
                    reconstructed_text=reconstructed_text_dec,
                    reconstructed_token_ids=recon_ids,
                    reconstruction_error=recon_error,
                    reconstruction_method=initial_prompt_lora_reconstruction,
                    lora_rank=r,
                    seq_len=seq_len,
                    is_exact=is_exact,
                    embsim_l2_text=embsim_l2_text_dec,
                    embsim_l2_token_ids=embsim_l2_ids,
                    embsim_cos_text=embsim_cos_text_dec,
                    embsim_cos_token_ids=embsim_cos_ids,
                )

        else:
            free_logits = _oh.detach().requires_grad_(True)
            parameters = [free_logits]

    length_alpha = None
    eos_logit_template = None
    if prompt_length_learnable and n_free > 1:
        length_alpha = torch.tensor(float(prompt_length_alpha_init), requires_grad=True, device=device)
        parameters.append(length_alpha)
        eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
        eos_logit_template = torch.zeros(1, 1, vocab_size, device=device)
        eos_logit_template[0, 0, eos_token_id] = prompt_length_eos_spike
    elif prompt_length_learnable and n_free <= 1:
        logger.warning("prompt_length_learnable=True requires at least two free prompt positions; disabling it.")

    # -----------------------------------------------------------------------
    # STGS module for the shared prompt p
    # -----------------------------------------------------------------------
    if stgs_module is None:
        stgs_module = STGS(
            vocab_size=vocab_size,
            stgs_hard=stgs_hard,
            stgs_hard_method=stgs_hard_method,
            init_temperature=temperature,
            learnable_temperature=learnable_temperature,
            nbr_learnable_temperatures=seq_len if decouple_learnable_temperature else None,
            stgs_hard_embsim_probs=stgs_hard_embsim_probs,
            stgs_hard_embsim_strategy=stgs_hard_embsim_strategy,
            stgs_hard_embsim_top_k=stgs_hard_embsim_top_k,
            stgs_hard_embsim_rerank_alpha=stgs_hard_embsim_rerank_alpha,
            stgs_hard_embsim_sample_tau=stgs_hard_embsim_sample_tau,
            stgs_hard_embsim_margin=stgs_hard_embsim_margin,
            stgs_hard_embsim_fallback=stgs_hard_embsim_fallback,
            logits_normalize=logits_normalize,
            input_dropout=stgs_input_dropout,
            output_dropout=stgs_output_dropout,
            eps=eps,
            device=device,
        )
    parameters += list(stgs_module.parameters())

    optimizer = build_prompt_optimizer(
        parameters,
        learning_rate=learning_rate,
        lora_B=lora_B,
        logits_lora_b_learning_rate=logits_lora_b_learning_rate,
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        schedule=lr_schedule,
        total_epochs=epochs,
        warmup_epochs=lr_warmup_epochs,
        lr_min=lr_schedule_min,
        step_size=lr_schedule_step_size,
        gamma=lr_schedule_gamma,
    )

    # -----------------------------------------------------------------------
    # Logits filter helper (top-k / top-p)
    # -----------------------------------------------------------------------
    def _apply_logit_filter(logits: Tensor) -> Tensor:
        """Apply top-k / top-p penalty to free_logits."""
        if logits_top_k <= 0 and logits_top_p >= 1.0:
            return logits
        filtered = logits.clone()
        if logits_top_k > 0:
            # Zero out tokens outside top-k per position
            topk_vals = torch.topk(filtered, min(logits_top_k, filtered.shape[-1]), dim=-1).values
            threshold = topk_vals[..., -1:].expand_as(filtered)
            filtered = torch.where(filtered < threshold, filtered - logits_filter_penalty, filtered)
        if logits_top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > logits_top_p
            sorted_logits[remove_mask] = sorted_logits[remove_mask] - logits_filter_penalty
            filtered.scatter_(-1, sorted_indices, sorted_logits)
        return filtered

    def _get_free_logits() -> Tensor:
        raw = lora_A @ lora_B if lora_A is not None else free_logits
        return _apply_logit_filter(raw)

    def _assemble_prompt_logits() -> Tensor:
        active_free = _get_free_logits()
        if length_alpha is not None:
            lambda_t = n_free * torch.sigmoid(length_alpha)
            i_vals = torch.arange(n_free, 0, -1, device=device, dtype=active_free.dtype)
            gates = torch.sigmoid(prompt_length_beta * (lambda_t - i_vals))
            active_free = gates.view(1, n_free, 1) * active_free + (1.0 - gates.view(1, n_free, 1)) * eos_logit_template
        parts = []
        if fixed_prefix is not None:
            parts.append(fixed_prefix)
        if n_free > 0:
            parts.append(active_free)
        if fixed_suffix is not None:
            parts.append(fixed_suffix)
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def _build_mas_loss_fn(mini_batch, mini_batch_pair_contexts):
        """Return a Callable A_val -> loss for the last sample in mini_batch.

        Used by mas_hvp_mode='jvp' to compute MAS rotational metrics without
        create_graph=True.  Captures caches and frozen params by reference;
        mini_batch / mini_batch_pair_contexts are passed explicitly so they
        are snapshot-frozen at call time.
        """
        _x_str, _R_str, _y_str = mini_batch[-1]
        _loss_inst = loss_cache[(_y_str, Y_tokens_cache[_y_str].shape[1])]
        _lora_B_d  = lora_B.detach()

        def _loss_fn(A_val: Tensor) -> Tensor:
            _active = A_val @ _lora_B_d           # (1, n_free, vocab)
            if length_alpha is not None:
                _lam = n_free * torch.sigmoid(length_alpha.detach())
                _i   = torch.arange(n_free, 0, -1, device=device, dtype=_active.dtype)
                _g   = torch.sigmoid(prompt_length_beta * (_lam - _i))
                _active = _g.view(1, n_free, 1) * _active + (1.0 - _g.view(1, n_free, 1)) * eos_logit_template.detach()
            _parts = []
            if fixed_prefix is not None:
                _parts.append(fixed_prefix.detach())
            if n_free > 0:
                _parts.append(_active)
            _fl = torch.cat(_parts, dim=1) if _parts else _active
            if prompt_logits_transform is not None:
                _fl = prompt_logits_transform(_fl, mini_batch_pair_contexts)
            return pc_forward_pass(
                diff_model=diff_model, stgs_module=stgs_module,
                loss_instance=_loss_inst, free_logits=_fl,
                embedding_weights=embedding_weights,
                x_ids=x_ids_cache[_x_str], x_embeds=x_embeds_cache[_x_str], x_text_raw=_x_str,
                R_gt_ids=R_gt_ids_cache[(_x_str, _R_str)],
                R_gt_embeds=R_gt_embeds_cache[(_x_str, _R_str)], R_gt_text_raw=_R_str,
                E_input_ids=E_input_ids, E_embeds=E_embeds,
                Y_tokens=Y_tokens_cache[_y_str], y_target_text_raw=_y_str,
                use_chat_template=use_chat_template, teacher_forcing_r=teacher_forcing_r,
                bptt=bptt, reasoning_generation_backend=reasoning_generation_backend,
                reasoning_generate_kwargs=reasoning_generate_kwargs,
                max_new_tokens_reasoning=max_new_tokens_reasoning,
                max_new_tokens_answer=max_new_tokens_answer,
                gumbel_noise_scale=current_gumbel_scale,
                accumulate_embeds=accumulate_embeds, tokenizer=tokenizer, weave_extras={},
            )
        return _loss_fn

    superposition_callback = None
    if superposition_metric_every > 0:
        try:
            from superposition_analysis import build_superposition_callback

            superposition_callback = build_superposition_callback(
                tokenizer=tokenizer,
                embedding_weights_subset=embedding_weights,
                allowed_tokens=allowed_tokens,
                output_dir=str(superposition_output_dir or "."),
                log_every=superposition_metric_every,
                modes=superposition_metric_modes,
                vocab_top_k=superposition_vocab_top_k,
                vocab_source=superposition_vocab_source,
                vocab_dataset_path=superposition_vocab_dataset_path,
                vocab_hf_name=superposition_vocab_hf_name,
                vocab_hf_split=superposition_vocab_hf_split,
                vocab_num_texts=superposition_vocab_num_texts,
                entropy_temperature=superposition_entropy_temperature,
            )
        except Exception as exc:
            logger.warning("Superposition metric callback disabled: %s", exc)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    loss_history: List[float] = []
    train_accuracy_history: List[Dict] = []
    val_accuracy_history: List[Dict] = []
    current_gumbel_scale = gumbel_noise_scale
    # SWA state
    swa_logits_accum: Optional[torch.Tensor] = None
    swa_n: int = 0
    ema_loss = None
    anneal_total = temperature_anneal_epochs if temperature_anneal_epochs > 0 else epochs
    _ppo_ref_logits: Optional[Tensor] = None
    global_iteration = 0
    prompt_history_table = None
    exact_match_hits_table = None
    table_log_every = 10
    if wandb.run is not None:
        prompt_history_columns = [
            "epoch",
            "prompt_token_ids",
            "prompt_text",
            "train_loss",
            "temperature",
            "effective_temperature",
            "bptt_effective_temperature",
            "gumbel_noise_scale",
            "grad_norm",
            "prompt_grad_norm",
            "non_zero_grads",
            "discrete_reinit",
            "discrete_reinit_num_positions",
            "train_any_correct",
            "train_all_correct",
            "val_any_correct",
            "val_all_correct",
        ]
        for method_name in extraction_fns:
            prompt_history_columns.append(f"train_{method_name}")
        for method_name in extraction_fns:
            prompt_history_columns.append(f"val_{method_name}")
        prompt_history_table = wandb.Table(columns=prompt_history_columns)
        exact_match_hits_columns = [
            "epoch",
            "split",
            "prompt_token_ids",
            "prompt_text",
            "loss",
            "any_correct",
            "all_correct",
        ]
        for method_name in extraction_fns:
            exact_match_hits_columns.append(method_name)
        exact_match_hits_table = wandb.Table(columns=exact_match_hits_columns)

    epoch_pbar = tqdm(
        total=epochs,
        desc="train epochs",
        leave=True,
    )
    optimizer.zero_grad()

    # Pre-training validation (epoch -1)
    if val_eval_before_training and val_pairs:
        _eval_fn = evaluate_shared_prompt_batched if use_batched_val_eval else evaluate_shared_prompt
        _pre_fl = _assemble_prompt_logits().detach()
        _pre_val_metrics = _eval_fn(
            free_logits=_pre_fl,
            eval_pairs=val_pairs,
            model=model,
            diff_model=diff_model,
            stgs_module=stgs_module,
            embedding_weights=embedding_weights,
            tokenizer=tokenizer,
            device=device,
            E_input_ids=E_input_ids,
            E_embeds=E_embeds,
            extraction_fns=extraction_fns,
            max_new_tokens_reasoning=max_new_tokens_reasoning,
            max_new_tokens_answer=max_new_tokens_answer,
            eval_mode=val_prompt_eval_mode,
            reasoning_generation_backend=reasoning_generation_backend,
            reasoning_generate_kwargs=reasoning_generate_kwargs,
            use_chat_template=use_chat_template,
            epoch=-1,
            eval_split="val",
            x_ids_cache=x_ids_cache,
            x_embeds_cache=x_embeds_cache if val_prompt_eval_mode == "soft" else None,
            accumulate_embeds=accumulate_embeds,
            prompt_logits_transform=prompt_logits_transform,
            eval_inner_batch_size=eval_inner_batch_size,
        )
        _pre_val_metrics["epoch"] = -1
        val_accuracy_history.append(_pre_val_metrics)
        if wandb.run is not None:
            wandb.log({f"val/{k}": v for k, v in _pre_val_metrics.items()})
    if test_eval_before_training and per_epoch_callback is not None:
        _pre_fl = _assemble_prompt_logits().detach()
        _pre_test_metrics = per_epoch_callback(-1, _pre_fl)
        if _pre_test_metrics and wandb.run is not None:
            wandb.log(_pre_test_metrics)

    # -----------------------------------------------------------------------
    # Offline EM: E-step closure + initial cache
    # -----------------------------------------------------------------------
    offline_r_cache: Optional[Dict[str, Tensor]] = None

    def _run_e_step(epoch_label: int) -> Dict[str, Tensor]:
        logger.info("[Offline EM] Generating reasoning cache (E-step at epoch %d)...", epoch_label)
        _fl = _assemble_prompt_logits().detach()
        # Use only the free (non-fixed) portion of the logits for p_embeds, matching what
        # pc_forward_pass_batched_free does during M-step.
        _result = _generate_offline_reasoning_cache(
            diff_model=diff_model,
            free_logits=_fl,
            x_embeds_cache=x_embeds_cache,
            train_triples=train_triples,
            stgs_module=stgs_module,
            embedding_weights=embedding_weights,
            offline_em_stgs_method=offline_em_stgs_method,
            offline_em_stgs_embsim_probs=offline_em_stgs_embsim_probs,
            offline_em_temperature=offline_em_temperature,
            max_new_tokens_reasoning=max_new_tokens_reasoning,
            reasoning_generation_backend=reasoning_generation_backend,
            reasoning_generate_kwargs=reasoning_generate_kwargs,
            cache_batch_size=offline_em_cache_batch_size or inner_batch_size,
            device=device,
        )
        if wandb.run is not None:
            wandb.log({
                "offline_em/cache_refreshed_epoch": epoch_label,
                "offline_em/cache_size": len(_result),
            })
        logger.info("[Offline EM] Cache ready: %d unique x_strs.", len(_result))
        return _result

    if offline_em:
        offline_r_cache = _run_e_step(epoch_label=0)

    try:
        for epoch in range(epochs):
            # Temperature annealing
            current_temperature = compute_annealed_temperature(
                temperature, temperature_anneal_min, epoch, anneal_total,
                temperature_anneal_schedule,
            )
            if not learnable_temperature:
                stgs_module.init_temperature = current_temperature

            # Offline EM: periodic E-step refresh
            if (
                offline_em
                and offline_em_resample_every > 0
                and epoch > 0
                and epoch % offline_em_resample_every == 0
            ):
                offline_r_cache = _run_e_step(epoch_label=epoch)

            discrete_reinit_applied = False
            discrete_reinit_num_positions = 0
            discrete_reinit_entropy_mean = None
            discrete_reinit_entropy_min = None

            # Discrete reinitialization
            should_check_entropy = discrete_reinit_entropy_threshold > 0.0
            period_due = discrete_reinit_epoch > 0 and epoch > 0 and epoch % discrete_reinit_epoch == 0
            threshold_due = discrete_reinit_epoch == 0 and epoch > 0 and should_check_entropy
            if period_due or threshold_due:
                current_logits = _assemble_prompt_logits().detach()
                current_free_logits = current_logits[:, prefix_len:prefix_len + n_free, :]
                reinit_probs = None
                reinit_mask = None
                needs_reinit_probs = (
                    should_check_entropy
                    or discrete_reinit_snap.startswith("embsim-")
                    or discrete_reinit_topk > 0
                )
                if needs_reinit_probs:
                    reinit_probs = compute_embsim_probs(
                        current_free_logits,
                        stgs_module=stgs_module if discrete_reinit_embsim_probs == "gumbel_soft" else None,
                        batch_size=1,
                        embsim_temperature=1.0,
                        probs_source=discrete_reinit_embsim_probs,
                    )
                if should_check_entropy:
                    free_entropies = compute_position_entropies(
                        None if reinit_probs is not None else current_free_logits,
                        eps=eps,
                        probs=reinit_probs,
                    )
                    reinit_mask = free_entropies.squeeze(0) < discrete_reinit_entropy_threshold
                    discrete_reinit_entropy_mean = free_entropies.mean().item()
                    discrete_reinit_entropy_min = free_entropies.min().item()
                    discrete_reinit_num_positions = int(reinit_mask.sum().item())
                    if discrete_reinit_num_positions == 0:
                        reinit_mask = None
                elif period_due:
                    discrete_reinit_num_positions = seq_len

                if reinit_mask is None:
                    discrete_reinit_applied = period_due and not should_check_entropy
                else:
                    discrete_reinit_applied = discrete_reinit_num_positions > 0

                if not discrete_reinit_applied:
                    reinit_mask = None
                if lora_A is not None:
                    if discrete_reinit_applied:
                        snap_lora_logits(
                            lora_A, lora_B, current_logits, embedding_weights,
                            n_free, prefix_len, discrete_reinit_snap, optimizer,
                            discrete_reinit_prob=discrete_reinit_prob,
                            discrete_reinit_topk=discrete_reinit_topk,
                            position_mask=reinit_mask,
                            embsim_probs=reinit_probs,
                        )
                else:
                    if discrete_reinit_applied:
                        snap_free_logits(
                            free_logits, current_logits, embedding_weights,
                            n_free, prefix_len, discrete_reinit_snap,
                            discrete_reinit_prob=discrete_reinit_prob,
                            discrete_reinit_topk=discrete_reinit_topk,
                            position_mask=reinit_mask,
                            embsim_probs=reinit_probs,
                        )

            # Adaptive Gumbel noise
            if adaptive_gumbel_noise and ema_loss is not None:
                current_gumbel_scale = max(
                    adaptive_gumbel_noise_min_scale,
                    adaptive_gumbel_noise_beta * current_gumbel_scale
                    + (1 - adaptive_gumbel_noise_beta) * (ema_loss / (ema_loss + 1.0)),
                )

            epoch_train_triples = list(train_triples)
            random.shuffle(epoch_train_triples)
            train_batches = [
                epoch_train_triples[start:start + inner_batch_size]
                for start in range(0, len(epoch_train_triples), inner_batch_size)
            ]

            epoch_loss_total = 0.0
            epoch_train_examples = 0
            train_method_correct: Dict[str, int] = {m: 0 for m in extraction_fns}
            train_any_correct_count = 0
            train_all_correct_count = 0
            epoch_reg_sums: Dict[str, float] = {}
            total_norm = None

            iteration_pbar = tqdm(
                total=len(train_batches),
                desc=f"train epoch {epoch + 1}/{epochs} iterations",
                leave=False,
            )
            try:
                for iteration, mini_batch in enumerate(train_batches):
                    batch_loss = 0.0
                    reg_log: Dict[str, float] = {}
                    n_batch = len(mini_batch)

                    fl_base = _assemble_prompt_logits()
                    mini_batch_pair_contexts = [(x_str, y_raw_str) for x_str, _, y_raw_str in mini_batch]
                    fl_batch = (
                        prompt_logits_transform(fl_base, mini_batch_pair_contexts)
                        if prompt_logits_transform is not None
                        else fl_base
                    )
                    p_text_batch = tokenizer.decode(
                        fl_batch.detach().argmax(dim=-1).squeeze(0).tolist(),
                        skip_special_tokens=False,
                    )

                    loss_scale = n_batch * max(gradient_accumulation_steps, 1)
                    _is_mas_iter = (
                        mas_metric_every > 0
                        and lora_A is not None
                        and global_iteration % mas_metric_every == 0
                    )
                    if use_batched_forward_pass:
                        _x_emb   = [x_embeds_cache[x]         for x, _, _ in mini_batch]
                        _R_emb   = [R_gt_embeds_cache[(x, R)]  for x, R, _ in mini_batch]
                        _Y_tok   = [Y_tokens_cache[y]          for _, _, y in mini_batch]
                        _losses_ = [
                            loss_cache[(y, Y_tokens_cache[y].shape[1])]
                            for _, _, y in mini_batch
                        ]

                        _bfp_kwargs = dict(
                            diff_model=diff_model, stgs_module=stgs_module,
                            loss_instances=_losses_,
                            free_logits=fl_batch, embedding_weights=embedding_weights,
                            x_embeds_list=_x_emb, R_gt_embeds_list=_R_emb,
                            E_embeds=E_embeds, Y_tokens_list=_Y_tok,
                            E_input_ids=E_input_ids,
                            teacher_forcing_r=teacher_forcing_r, bptt=bptt,
                            reasoning_generation_backend=reasoning_generation_backend,
                            reasoning_generate_kwargs=reasoning_generate_kwargs,
                            max_new_tokens_reasoning=max_new_tokens_reasoning,
                            max_new_tokens_answer=max_new_tokens_answer,
                            gumbel_noise_scale=current_gumbel_scale,
                            accumulate_embeds=accumulate_embeds,
                            stgs_noise_mode=batched_stgs_noise_mode,
                            offline_r_cache=offline_r_cache,
                            x_strs_batch=[x for x, _, _ in mini_batch],
                        )
                        if gumbel_n_samples > 1:
                            _all_batched_outs = [
                                pc_forward_pass_batched(**_bfp_kwargs)
                                for _ in range(gumbel_n_samples)
                            ]
                            _batched_out = _all_batched_outs[0]
                            per_losses = [
                                torch.stack([out["losses"][si] for out in _all_batched_outs]).mean()
                                for si in range(n_batch)
                            ]
                        else:
                            _batched_out = pc_forward_pass_batched(**_bfp_kwargs)
                            per_losses = _batched_out["losses"]
                        y_logits_list  = _batched_out["y_logits_list"]
                        R_tokens_batch = _batched_out["R_tokens"]
                        for si, (loss, y_lg, (x_str, R_gt_str, y_raw_str)) in enumerate(
                            zip(per_losses, y_logits_list, mini_batch)
                        ):
                            # Retain all graphs on MAS iterations when using autograd mode so
                            # compute_rotational_metrics can reuse the graph after the loop.
                            _retain = (si < n_batch - 1) or (_is_mas_iter and mas_hvp_mode == "autograd")
                            (loss / loss_scale).backward(retain_graph=_retain)
                            batch_loss += loss.item()
                            train_Y_text = (
                                tokenizer.decode(
                                    y_lg.argmax(-1).squeeze(0).tolist(),
                                    skip_special_tokens=True,
                                ) if y_lg is not None else ""
                            )
                            if teacher_forcing_r:
                                train_R_text = R_gt_str
                            elif R_tokens_batch is not None:
                                train_R_text = tokenizer.decode(
                                    R_tokens_batch[si].tolist(), skip_special_tokens=True,
                                )
                            else:
                                train_R_text = ""
                            train_method_results = _compute_method_results(
                                R_text=train_R_text, Y_text=train_Y_text,
                                y_raw_str=y_raw_str, extraction_fns=extraction_fns,
                            )
                            for method, ok in train_method_results.items():
                                if ok: train_method_correct[method] += 1
                            train_any_correct_count += int(any(train_method_results.values()))
                            train_all_correct_count += int(all(train_method_results.values()))
                            epoch_train_examples += 1
                            if is_weave_active():
                                weave_lm_train_step(
                                    call_name=(
                                        "teacher_forced_answer_forward_batched"
                                        if teacher_forcing_r else "answer_generate_batched"
                                    ),
                                    split="train",
                                    epoch=epoch,
                                    iteration=iteration,
                                    global_iteration=global_iteration,
                                    loss=loss.item(),
                                    teacher_forcing_r=teacher_forcing_r,
                                    reasoning_generation_backend=reasoning_generation_backend,
                                    x_text_raw=x_str,
                                    p_text_batch_argmax=p_text_batch,
                                    R_gt_text=R_gt_str,
                                    R_text_clean=train_R_text,
                                    Y_text_clean=train_Y_text,
                                    y_target=y_raw_str,
                                    correct_per_method=train_method_results,
                                    any_correct=any(train_method_results.values()),
                                    all_correct=all(train_method_results.values()),
                                    lm_name="diff_model.forward" if teacher_forcing_r else "diff_model.generate",
                                    input_mode="inputs_embeds",
                                    output_mode="normal_logits" if teacher_forcing_r else "generated_tokens",
                                    call_role="answer",
                                    batched=True,
                                )
                            train_metrics_running = _metrics_from_accuracy_counts(
                                per_method_correct=train_method_correct,
                                any_correct_count=train_any_correct_count,
                                all_correct_count=train_all_correct_count,
                                n_examples=epoch_train_examples,
                                extraction_fns=extraction_fns,
                            )
                        # MAS rotational metric — batched path
                        if _is_mas_iter:
                            try:
                                if mas_hvp_mode == "jvp":
                                    from mas_analysis import compute_rotational_metrics_jvp
                                    _mas_raw = compute_rotational_metrics_jvp(
                                        _build_mas_loss_fn(mini_batch, mini_batch_pair_contexts), lora_A
                                    )
                                else:
                                    from mas_analysis import compute_rotational_metrics
                                    _mas_raw = compute_rotational_metrics(per_losses[-1], lora_A)
                                if wandb.run is not None:
                                    wandb.log({f"mas/{k}": v for k, v in _mas_raw.items()})
                            except Exception as exc:
                                logger.warning(
                                    "MAS rotational metric skipped at iteration %d: %s",
                                    global_iteration, exc,
                                )
                    else:
                        _mas_last_loss = None  # last sample's loss tensor for MAS (single-sample estimate)
                        for sample_idx, (x_str, R_gt_str, y_raw_str) in enumerate(mini_batch):
                            x_ids = x_ids_cache[x_str]
                            x_embeds = x_embeds_cache[x_str]
                            R_gt_ids = R_gt_ids_cache[(x_str, R_gt_str)]
                            R_gt_embeds = R_gt_embeds_cache[(x_str, R_gt_str)]
                            Y_tokens = Y_tokens_cache[y_raw_str]
                            Y_len = Y_tokens.shape[1]
                            loss_instance = loss_cache[(y_raw_str, Y_len)]

                            weave_extras: Dict = {}
                            _fp_kwargs = dict(
                                diff_model=diff_model,
                                stgs_module=stgs_module,
                                loss_instance=loss_instance,
                                free_logits=fl_batch,
                                embedding_weights=embedding_weights,
                                x_ids=x_ids,
                                x_embeds=x_embeds,
                                x_text_raw=x_str,
                                R_gt_ids=R_gt_ids,
                                R_gt_embeds=R_gt_embeds,
                                R_gt_text_raw=R_gt_str,
                                E_input_ids=E_input_ids,
                                E_embeds=E_embeds,
                                Y_tokens=Y_tokens,
                                y_target_text_raw=y_raw_str,
                                use_chat_template=use_chat_template,
                                teacher_forcing_r=teacher_forcing_r,
                                bptt=bptt,
                                reasoning_generation_backend=reasoning_generation_backend,
                                reasoning_generate_kwargs=reasoning_generate_kwargs,
                                max_new_tokens_reasoning=max_new_tokens_reasoning,
                                max_new_tokens_answer=max_new_tokens_answer,
                                gumbel_noise_scale=current_gumbel_scale,
                                accumulate_embeds=accumulate_embeds,
                                tokenizer=tokenizer,
                                weave_extras=weave_extras,
                            )
                            if gumbel_n_samples > 1:
                                _g_losses = [
                                    pc_forward_pass(**_fp_kwargs)
                                    for _ in range(gumbel_n_samples)
                                ]
                                loss = torch.stack(_g_losses).mean()
                            else:
                                loss = pc_forward_pass(**_fp_kwargs)
                            # Capture last sample's loss for MAS (single-sample estimate; only on MAS iterations)
                            if _is_mas_iter and sample_idx == n_batch - 1:
                                _mas_last_loss = loss  # unscaled; single sample for MAS
                            # Retain all graphs on MAS iterations when using autograd mode so
                            # compute_rotational_metrics can reuse the graph after the sample loop.
                            _retain = (sample_idx < (n_batch - 1)) or (_is_mas_iter and mas_hvp_mode == "autograd")
                            (loss / loss_scale).backward(retain_graph=_retain)
                            batch_loss += loss.item()

                            # Teacher forcing does not generate reasoning, so training accuracy
                            # falls back to the provided reasoning text in that mode.
                            train_R_text = R_gt_str if teacher_forcing_r else weave_extras.get("R_gen_text", "")
                            train_Y_text = weave_extras.get("y_pred_text", "")
                            train_method_results = _compute_method_results(
                                R_text=train_R_text,
                                Y_text=train_Y_text,
                                y_raw_str=y_raw_str,
                                extraction_fns=extraction_fns,
                            )
                            for method, is_correct in train_method_results.items():
                                if is_correct:
                                    train_method_correct[method] += 1
                            train_any_correct_count += int(any(train_method_results.values()))
                            train_all_correct_count += int(all(train_method_results.values()))
                            epoch_train_examples += 1
                            train_metrics_running = _metrics_from_accuracy_counts(
                                per_method_correct=train_method_correct,
                                any_correct_count=train_any_correct_count,
                                all_correct_count=train_all_correct_count,
                                n_examples=epoch_train_examples,
                                extraction_fns=extraction_fns,
                            )

                            trace_common = {
                                "epoch": epoch,
                                "iteration": iteration,
                                "global_iteration": global_iteration,
                                "loss": loss.item(),
                                "teacher_forcing_r": teacher_forcing_r,
                                "reasoning_generation_backend": reasoning_generation_backend,
                                "x_text_raw": x_str,
                                "p_text_batch_argmax": p_text_batch,
                                "R_gt_text": R_gt_str,
                                "R_text_clean": train_R_text,
                                "Y_text_clean": train_Y_text,
                                "y_target": y_raw_str,
                                "correct_per_method": train_method_results,
                                "any_correct": any(train_method_results.values()),
                                "all_correct": all(train_method_results.values()),
                            }
                            for trace_payload in weave_extras.get("lm_traces", []):
                                trace_payload.update(trace_common)
                                trace_payload["epoch"] = epoch
                                weave_lm_train_step(**trace_payload)

                    # MAS rotational metric — serial path
                    if not use_batched_forward_pass and _is_mas_iter:
                        try:
                            _mas_raw = None
                            if mas_hvp_mode == "jvp":
                                from mas_analysis import compute_rotational_metrics_jvp
                                _mas_raw = compute_rotational_metrics_jvp(
                                    _build_mas_loss_fn(mini_batch, mini_batch_pair_contexts), lora_A
                                )
                            elif _mas_last_loss is not None:
                                from mas_analysis import compute_rotational_metrics
                                _mas_raw = compute_rotational_metrics(_mas_last_loss, lora_A)
                            if _mas_raw is not None and wandb.run is not None:
                                wandb.log({f"mas/{k}": v for k, v in _mas_raw.items()})
                        except Exception as exc:
                            logger.warning(
                                "MAS rotational metric skipped at iteration %d: %s",
                                global_iteration, exc,
                            )

                    mean_batch_loss = batch_loss / max(n_batch, 1)
                    reg_loss = None
                    if length_alpha is not None and prompt_length_reg_lambda > 0.0:
                        lambda_t = n_free * torch.sigmoid(length_alpha)
                        length_reg = prompt_length_reg_lambda * lambda_t
                        reg_loss = length_reg if reg_loss is None else reg_loss + length_reg
                        reg_log["prompt_length_reg"] = float(length_reg.detach().item())

                    if (
                        learnable_temperature
                        and temperature_anneal_reg_lambda > 0.0
                        and temperature_anneal_schedule != "none"
                    ):
                        eff_temp = _effective_temperature_tensor(stgs_module)
                        target_tau = compute_annealed_temperature(
                            temperature, temperature_anneal_min, epoch, anneal_total, temperature_anneal_schedule
                        )
                        target_tau_t = eff_temp.new_tensor(target_tau)
                        if temperature_anneal_reg_mode == "one_sided":
                            temp_reg = F.relu(eff_temp.mean() - target_tau_t).pow(2)
                        else:
                            temp_reg = (eff_temp - target_tau_t).pow(2).mean()
                        scaled_temp_reg = temperature_anneal_reg_lambda * temp_reg
                        reg_loss = scaled_temp_reg if reg_loss is None else reg_loss + scaled_temp_reg
                        reg_log["temperature_anneal_reg"] = float(temp_reg.detach().item())

                    if learnable_temperature and temperature_loss_coupling_lambda > 0.0:
                        eff_temp = _effective_temperature_tensor(stgs_module)
                        coupling_reg = (
                            temperature_loss_coupling_lambda
                            * eff_temp.new_tensor(float(mean_batch_loss))
                            * (1.0 / eff_temp.clamp(min=1e-6)).mean()
                        )
                        reg_loss = coupling_reg if reg_loss is None else reg_loss + coupling_reg
                        reg_log["temperature_loss_coupling_reg"] = float(coupling_reg.detach().item())

                    if ppo_kl_lambda > 0.0:
                        if epoch == 0 and global_iteration == 0:
                            _ppo_ref_logits = _assemble_prompt_logits().detach().clone()
                        eff_tau = float(_effective_temperature_tensor(stgs_module).float().mean().item())
                        ppo_kl = compute_ppo_kl_loss(
                            free_logits=_assemble_prompt_logits(),
                            ref_logits=_ppo_ref_logits,
                            temperature=eff_tau,
                            mode=ppo_kl_mode,
                            epsilon=ppo_kl_epsilon,
                        )
                        scaled_ppo_kl = ppo_kl_lambda * ppo_kl
                        reg_loss = scaled_ppo_kl if reg_loss is None else reg_loss + scaled_ppo_kl
                        reg_log["ppo_kl_loss"] = float(ppo_kl.detach().item())

                    if reg_loss is not None:
                        reg_loss.backward()

                    # Gradient clipping
                    total_norm = None
                    if max_gradient_norm > 0:
                        all_params = [p for p in [free_logits, lora_A, lora_B] if p is not None]
                        all_params += list(stgs_module.parameters())
                        if length_alpha is not None:
                            all_params.append(length_alpha)
                        total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_gradient_norm).item()

                    if wandb.run is not None:
                        _iter_grad_leaf = lora_A if lora_A is not None else free_logits
                        iter_grad_stats = _gradient_stats(
                            _iter_grad_leaf.grad if _iter_grad_leaf is not None else None
                        )
                        iter_log = {
                            "iter/loss": mean_batch_loss,
                            "iter/epoch": epoch,
                            "iter/iteration": iteration,
                            "iter/global_iteration": global_iteration,
                            "iter/batch_size": n_batch,
                            "iter/temperature": current_temperature,
                            "iter/gumbel_noise_scale": current_gumbel_scale,
                            "iter/grad_norm": total_norm if total_norm is not None else iter_grad_stats["prompt_grad_norm"],
                            "iter/discrete_reinit": int(discrete_reinit_applied),
                            "iter/discrete_reinit_num_positions": discrete_reinit_num_positions,
                        }
                        iter_log.update({f"iter/{k}": v for k, v in iter_grad_stats.items()})
                        iter_log.update({f"iter/train/{k}": v for k, v in train_metrics_running.items()})
                        iter_log.update({f"iter/{k}": v for k, v in reg_log.items()})
                        if discrete_reinit_entropy_mean is not None:
                            iter_log["iter/discrete_reinit_entropy_mean"] = discrete_reinit_entropy_mean
                            iter_log["iter/discrete_reinit_entropy_min"] = discrete_reinit_entropy_min
                        if length_alpha is not None:
                            iter_log["iter/effective_prompt_length"] = float(
                                (n_free * torch.sigmoid(length_alpha)).detach().item()
                            )
                            iter_log["iter/length_alpha"] = float(length_alpha.detach().item())
                        iter_eff_temperature = _effective_temperature_tensor(stgs_module).float()
                        if iter_eff_temperature is not None:
                            iter_log["iter/effective_temperature"] = float(iter_eff_temperature.mean().item())
                            if decouple_learnable_temperature:
                                for idx, temp_mean in enumerate(_temperature_component_means(iter_eff_temperature)):
                                    iter_log[f"iter/effective_temperature_{idx}"] = temp_mean
                        iter_bptt_eff_temperature = (
                            _effective_temperature_tensor(diff_model.stgs).float()
                            if hasattr(diff_model, "stgs") and diff_model.stgs is not None
                            else None
                        )
                        if iter_bptt_eff_temperature is not None:
                            iter_log["iter/bptt_effective_temperature"] = float(
                                iter_bptt_eff_temperature.mean().item()
                            )
                            if bptt_decouple_learnable_temperature:
                                for idx, temp_mean in enumerate(_temperature_component_means(iter_bptt_eff_temperature)):
                                    iter_log[f"iter/bptt_effective_temperature_{idx}"] = temp_mean
                        if (
                            learnable_temperature
                            and temperature_anneal_reg_lambda > 0.0
                            and temperature_anneal_schedule != "none"
                        ):
                            iter_log["iter/temperature_anneal_target"] = compute_annealed_temperature(
                                temperature,
                                temperature_anneal_min,
                                epoch,
                                anneal_total,
                                temperature_anneal_schedule,
                            )
                        if adaptive_gumbel_noise and ema_loss is not None:
                            iter_log["iter/gumbel_loss_ema"] = float(ema_loss)
                        wandb.log(iter_log)

                    # Gradient accumulation: only step every N iterations (or at end of epoch)
                    accum_step = (global_iteration + 1) % max(gradient_accumulation_steps, 1)
                    if accum_step == 0 or (iteration == len(train_batches) - 1):
                        # Gradient clipping already done above; now step
                        optimizer.step()
                        optimizer.zero_grad()

                        # Logit decay (only on actual update step)
                        if logit_decay > 0:
                            with torch.no_grad():
                                if lora_A is not None:
                                    lora_A.data.mul_(1 - logit_decay)
                                    lora_B.data.mul_(1 - logit_decay)
                                else:
                                    free_logits.data.mul_(1 - logit_decay)

                    epoch_loss_total += batch_loss
                    for key, value in reg_log.items():
                        epoch_reg_sums[key] = epoch_reg_sums.get(key, 0.0) + value

                    iteration_postfix = {
                        "loss": f"{mean_batch_loss:.4f}",
                    }
                    iteration_postfix.update(_accuracy_postfix(train_metrics_running, extraction_fns))
                    iteration_pbar.set_postfix(iteration_postfix, refresh=False)
                    iteration_pbar.update(1)
                    global_iteration += 1
            finally:
                iteration_pbar.close()

            if ppo_kl_lambda > 0.0 and ppo_ref_update_period > 0 and (epoch + 1) % ppo_ref_update_period == 0:
                _ppo_ref_logits = _assemble_prompt_logits().detach().clone()

            # Cache fl and p_text once per epoch after parameter update
            fl_epoch = _assemble_prompt_logits().detach()

            # SWA: accumulate Welford online mean of prompt logits
            if use_swa and epoch >= swa_start_epoch and (epoch - swa_start_epoch) % swa_freq == 0:
                swa_n += 1
                if swa_n == 1:
                    swa_logits_accum = fl_epoch.clone()
                else:
                    swa_logits_accum = swa_logits_accum + (fl_epoch - swa_logits_accum) / swa_n

            p_token_ids_epoch = fl_epoch.argmax(dim=-1).squeeze(0).tolist()
            p_text_epoch = tokenizer.decode(
                p_token_ids_epoch,
                skip_special_tokens=False,
            )

            epoch_avg_loss = epoch_loss_total / max(epoch_train_examples, 1)
            loss_history.append(epoch_avg_loss)
            train_metrics_epoch = _metrics_from_accuracy_counts(
                per_method_correct=train_method_correct,
                any_correct_count=train_any_correct_count,
                all_correct_count=train_all_correct_count,
                n_examples=epoch_train_examples,
                extraction_fns=extraction_fns,
            )
            train_metrics_epoch.pop("n_eval", None)
            train_metrics_epoch["epoch"] = epoch
            train_metrics_epoch["n_train"] = float(epoch_train_examples)
            train_accuracy_history.append(train_metrics_epoch)
            ema_loss = epoch_avg_loss if ema_loss is None else (
                adaptive_gumbel_noise_beta * ema_loss + (1 - adaptive_gumbel_noise_beta) * epoch_avg_loss
            )
            _grad_leaf = lora_A if lora_A is not None else free_logits
            grad_stats = _gradient_stats(_grad_leaf.grad if _grad_leaf is not None else None)
            eff_temperature = _effective_temperature_tensor(stgs_module).float()
            bptt_eff_temperature = (
                _effective_temperature_tensor(diff_model.stgs).float()
                if hasattr(diff_model, "stgs") and diff_model.stgs is not None
                else None
            )

            # W&B logging
            if wandb.run is not None:
                epoch_reg_log = {
                    key: value / max(len(train_batches), 1)
                    for key, value in epoch_reg_sums.items()
                }
                log_dict = {
                    "train/loss": epoch_avg_loss,
                    "temperature": current_temperature,
                    "gumbel_noise_scale": current_gumbel_scale,
                    "epoch": epoch,
                    "allowed_vocab_size": vocab_size,
                    "vocab_size": vocab_size,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                log_dict.update({f"train/{k}": v for k, v in train_metrics_epoch.items()})
                log_dict.update(epoch_reg_log)
                log_dict.update(grad_stats)
                log_dict["grad_norm"] = total_norm if total_norm is not None else grad_stats["prompt_grad_norm"]
                log_dict["discrete_reinit"] = int(discrete_reinit_applied)
                log_dict["discrete_reinit_num_positions"] = discrete_reinit_num_positions
                if use_swa:
                    log_dict["swa_n"] = swa_n
                if discrete_reinit_entropy_mean is not None:
                    log_dict["discrete_reinit_entropy_mean"] = discrete_reinit_entropy_mean
                    log_dict["discrete_reinit_entropy_min"] = discrete_reinit_entropy_min
                if length_alpha is not None:
                    log_dict["effective_prompt_length"] = float((n_free * torch.sigmoid(length_alpha)).detach().item())
                    log_dict["length_alpha"] = float(length_alpha.detach().item())
                if temperature_anneal_schedule != "none" and not learnable_temperature:
                    log_dict["annealed_temperature"] = current_temperature
                if eff_temperature is not None:
                    log_dict["effective_temperature"] = float(eff_temperature.mean().item())
                    if decouple_learnable_temperature:
                        for idx, temp_mean in enumerate(_temperature_component_means(eff_temperature)):
                            log_dict[f"effective_temperature_{idx}"] = temp_mean
                if bptt_eff_temperature is not None:
                    log_dict["bptt_effective_temperature"] = float(bptt_eff_temperature.mean().item())
                    if bptt_decouple_learnable_temperature:
                        for idx, temp_mean in enumerate(_temperature_component_means(bptt_eff_temperature)):
                            log_dict[f"bptt_effective_temperature_{idx}"] = temp_mean
                if (
                    learnable_temperature
                    and temperature_anneal_reg_lambda > 0.0
                    and temperature_anneal_schedule != "none"
                ):
                    log_dict["temperature_anneal_target"] = compute_annealed_temperature(
                        temperature,
                        temperature_anneal_min,
                        epoch,
                        anneal_total,
                        temperature_anneal_schedule,
                    )
                if adaptive_gumbel_noise and ema_loss is not None:
                    log_dict["gumbel_loss_ema"] = float(ema_loss)
                log_dict["train/p_text"] = p_text_epoch
                log_dict["train/p_token_ids"] = p_token_ids_epoch
                wandb.log(log_dict)

            # Weave epoch-level trace
            weave_epoch_summary(
                epoch=epoch,
                loss=epoch_avg_loss,
                p_text=p_text_epoch,
                temperature=current_temperature,
                n_samples=epoch_train_examples,
            )

            val_any_correct = None
            val_metrics = None
            # Validation
            if val_eval_every > 0 and (epoch + 1) % val_eval_every == 0 and val_pairs:
                _eval_fn = evaluate_shared_prompt_batched if use_batched_val_eval else evaluate_shared_prompt
                val_metrics = _eval_fn(
                    free_logits=fl_epoch,
                    eval_pairs=val_pairs,
                    model=model,
                    diff_model=diff_model,
                    stgs_module=stgs_module,
                    embedding_weights=embedding_weights,
                    tokenizer=tokenizer,
                    device=device,
                    E_input_ids=E_input_ids,
                    E_embeds=E_embeds,
                    extraction_fns=extraction_fns,
                    max_new_tokens_reasoning=max_new_tokens_reasoning,
                    max_new_tokens_answer=max_new_tokens_answer,
                    eval_mode=val_prompt_eval_mode,
                    reasoning_generation_backend=reasoning_generation_backend,
                    reasoning_generate_kwargs=reasoning_generate_kwargs,
                    use_chat_template=use_chat_template,
                    epoch=epoch,
                    eval_split="val",
                    x_ids_cache=x_ids_cache,
                    x_embeds_cache=x_embeds_cache if val_prompt_eval_mode == "soft" else None,
                    accumulate_embeds=accumulate_embeds,
                    prompt_logits_transform=prompt_logits_transform,
                    eval_inner_batch_size=eval_inner_batch_size,
                )
                val_metrics["epoch"] = epoch
                val_accuracy_history.append(val_metrics)
                val_any_correct = val_metrics.get("any_correct")
                if wandb.run is not None:
                    wandb.log({f"val/{k}": v for k, v in val_metrics.items()})

            # Per-epoch callback
            if per_epoch_callback is not None:
                cb_metrics = per_epoch_callback(epoch, fl_epoch)
                if cb_metrics and wandb.run is not None:
                    wandb.log(cb_metrics)

            if superposition_callback is not None and wandb.run is not None:
                _sup_metrics = superposition_callback(epoch, _get_free_logits().detach())
                if _sup_metrics:
                    wandb.log(_sup_metrics)

            if prompt_history_table is not None:
                row = [
                    epoch,
                    p_token_ids_epoch,
                    p_text_epoch,
                    epoch_avg_loss,
                    current_temperature,
                    float(eff_temperature.mean().item()) if eff_temperature is not None else None,
                    float(bptt_eff_temperature.mean().item()) if bptt_eff_temperature is not None else None,
                    current_gumbel_scale,
                    total_norm if total_norm is not None else grad_stats["prompt_grad_norm"],
                    grad_stats["prompt_grad_norm"],
                    grad_stats["non_zero_grads"],
                    int(discrete_reinit_applied),
                    discrete_reinit_num_positions,
                    train_metrics_epoch.get("any_correct"),
                    train_metrics_epoch.get("all_correct"),
                    None if val_metrics is None else val_metrics.get("any_correct"),
                    None if val_metrics is None else val_metrics.get("all_correct"),
                ]
                for method_name in extraction_fns:
                    row.append(train_metrics_epoch.get(f"accuracy_{method_name}"))
                for method_name in extraction_fns:
                    row.append(None if val_metrics is None else val_metrics.get(f"accuracy_{method_name}"))
                prompt_history_table.add_data(*row)
                if epoch % table_log_every == 0 or epoch == epochs - 1:
                    wandb.log({"prompt_history_table": copy.deepcopy(prompt_history_table)})

            if exact_match_hits_table is not None:
                hit_rows = []
                if float(train_metrics_epoch.get("all_correct", 0.0)) >= 1.0:
                    hit_rows.append(("train", train_metrics_epoch))
                if val_metrics is not None and float(val_metrics.get("all_correct", 0.0)) >= 1.0:
                    hit_rows.append(("val", val_metrics))
                for split_name, split_metrics in hit_rows:
                    row = [
                        epoch,
                        split_name,
                        p_token_ids_epoch,
                        p_text_epoch,
                        epoch_avg_loss,
                        split_metrics.get("any_correct"),
                        split_metrics.get("all_correct"),
                    ]
                    for method_name in extraction_fns:
                        row.append(split_metrics.get(f"accuracy_{method_name}"))
                    exact_match_hits_table.add_data(*row)
                if hit_rows:
                    wandb.log({"exact_match_hits": copy.deepcopy(exact_match_hits_table)})

            epoch_postfix = {
                "loss": f"{epoch_avg_loss:.4f}",
                "n_train": epoch_train_examples,
            }
            epoch_postfix.update(_accuracy_postfix(train_metrics_epoch, extraction_fns))
            if current_temperature is not None:
                epoch_postfix["tau"] = f"{current_temperature:.3f}"
            if val_any_correct is not None:
                epoch_postfix["val_any"] = f"{val_any_correct:.3f}"
            epoch_pbar.set_postfix(epoch_postfix, refresh=False)
            epoch_pbar.update(1)

            if lr_scheduler is not None:
                lr_scheduler.step()

            if early_stop_loss_threshold > 0.0 and epoch_avg_loss < early_stop_loss_threshold:
                logger.info(
                    "Stopping early at epoch %s because mean batch loss %.6f < threshold %.6f",
                    epoch,
                    epoch_avg_loss,
                    early_stop_loss_threshold,
                )
                break
    finally:
        epoch_pbar.close()

    # -----------------------------------------------------------------------
    # Build final result
    # -----------------------------------------------------------------------
    final_prompt_logits = _assemble_prompt_logits().detach()
    if use_swa and swa_n > 0:
        final_prompt_logits = swa_logits_accum
        if wandb.run is not None:
            wandb.run.summary["swa_n_snapshots"] = swa_n
    final_learnable_logits = _get_free_logits().detach()
    final_p_tokens = final_prompt_logits.argmax(dim=-1).squeeze(0).tolist()
    final_p_text = tokenizer.decode(final_p_tokens, skip_special_tokens=False)
    final_learnable_temperatures = extract_temperature_payload(
        stgs_module=stgs_module,
        bptt_module=getattr(diff_model, "stgs", None),
        device=device,
    )
    if superposition_callback is not None:
        final_epoch = max(len(loss_history) - 1, 0)
        _sup_final_metrics = superposition_callback(final_epoch, final_learnable_logits, force=True)
        if _sup_final_metrics and wandb.run is not None:
            wandb.log(_sup_final_metrics)
    if wandb.run is not None:
        summary_payload = {
            "final_prompt_text": final_p_text,
            "final_prompt_token_ids": final_p_tokens,
            "final_train_loss": loss_history[-1] if loss_history else None,
            "final_train_any_correct": train_accuracy_history[-1].get("any_correct", 0.0) if train_accuracy_history else 0.0,
            "final_train_all_correct": train_accuracy_history[-1].get("all_correct", 0.0) if train_accuracy_history else 0.0,
            "best_train_any_correct": max((entry.get("any_correct", 0.0) for entry in train_accuracy_history), default=0.0),
            "best_val_any_correct": max((entry.get("any_correct", 0.0) for entry in val_accuracy_history), default=0.0),
        }
        if prompt_history_table is not None:
            wandb.log({"prompt_history_table": copy.deepcopy(prompt_history_table)})
        if exact_match_hits_table is not None and exact_match_hits_table.data:
            wandb.log({"exact_match_hits": copy.deepcopy(exact_match_hits_table)})
        wandb.run.summary.update({k: v for k, v in summary_payload.items() if v is not None})

    return {
        "final_p_logits": final_prompt_logits,
        "swa_logits": swa_logits_accum,
        "learnable_logits": final_learnable_logits,
        "learnable_temperatures": final_learnable_temperatures,
        "final_p_tokens": final_p_tokens,
        "final_p_text": final_p_text,
        "loss_history": loss_history,
        "train_accuracy_history": train_accuracy_history,
        "val_accuracy_history": val_accuracy_history,
    }
