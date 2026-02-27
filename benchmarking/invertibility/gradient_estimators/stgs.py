from typing import Callable, Dict, Optional, List
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ForwardPassResult
from sdlm import STGS

logger = logging.getLogger(__name__)

class STGS_standalone(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        stgs_hard: bool = False,
        init_temperature: float = 1.0,
        learnable_temperature: bool = False,
        nbr_learnable_temperatures: Optional[int] = None,
        conditioning_dim: int = 0,
        eps: float = 1e-12,
        device: str = "cpu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.stgs_hard = stgs_hard
        self.init_temperature = init_temperature
        self.learnable_temperature = learnable_temperature
        self.nbr_learnable_temperatures = nbr_learnable_temperatures
        if self.learnable_temperature:
            if self.nbr_learnable_temperatures is None:
                self.nbr_learnable_temperatures = 1
        self.conditioning_dim = conditioning_dim
        self.eps = eps
        self.device = device

        if self.learnable_temperature:
            if self.conditioning_dim < 1:
                self.temperature_param = nn.Parameter(torch.rand(self.nbr_learnable_temperatures, requires_grad=True, device=self.device))
            else:
                self.tau_fc = nn.Sequential(
                    nn.Linear(self.conditioning_dim, self.nbr_learnable_temperatures, bias=False),
                    nn.Softplus(),
                ).to(device=device)

    def forward(self, x: torch.Tensor, hidden_states: Optional[torch.Tensor] = None):
        if self.learnable_temperature:
            if self.conditioning_dim < 1:
                eff_temperature = self.eps + 1.0 / (F.softplus(self.temperature_param) + 1.0 / (self.eps + self.init_temperature))
                eff_temperature = eff_temperature.reshape(1, -1, 1)
                batch_size = x.shape[0]
                eff_temperature = eff_temperature.repeat(batch_size, 1, 1)
            else:
                assert hidden_states is not None
                batch_size = x.shape[0]
                seq_len = x.shape[1]
                last_hidden_state = hidden_states[-1][:, -1, :].reshape(batch_size, self.conditioning_dim)
                inv_tau0 = 1.0 / (self.eps + self.init_temperature)
                eff_temperature = self.eps + 1.0 / (self.tau_fc(last_hidden_state) + inv_tau0).reshape(batch_size, -1, 1)
                if self.nbr_learnable_temperatures==1 \
                and self.nbr_learnable_temperatures != seq_len:
                    eff_temperature = eff_temperature.repeat(1, seq_len, 1)
        else:
            eff_temperature = torch.tensor([self.init_temperature], device=self.device)

        u = torch.rand_like(x) * (0.999 - self.eps) + self.eps
        gumbels = -torch.log(-torch.log(u))
        gumbel_logits = x + gumbels

        y_soft = F.softmax(gumbel_logits / eff_temperature, dim=-1)

        if self.stgs_hard:
            message_ids = torch.distributions.Categorical(probs=y_soft).sample()
            y_hard = F.one_hot(message_ids, num_classes=self.vocab_size)
            y_hard = y_hard.half() if x.dtype == torch.half else y_hard.float()
            message_one_hot = y_hard - y_soft.detach() + y_soft
        else:
            message_ids = torch.distributions.Categorical(probs=y_soft).sample()
            message_one_hot = y_soft

        message_one_hot = message_one_hot.half() if x.dtype == torch.half else message_one_hot.float()

        return message_ids, message_one_hot, eff_temperature


def _top_k_logit_filter(
    logits: torch.Tensor, k: int, penalty: float = 1e4
) -> torch.Tensor:
    """Suppress logits below the k-th largest at each position with a large penalty.
    Mask is computed without gradients so all positions still receive gradient updates."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    with torch.no_grad():
        top_k_vals = torch.topk(logits, min(k, logits.size(-1)), dim=-1).values
        threshold = top_k_vals[..., -1:]          # (batch, seq, 1)
        mask = (logits < threshold).float()
    return logits - mask * penalty


def _top_p_logit_filter(
    logits: torch.Tensor, p: float, penalty: float = 1e4
) -> torch.Tensor:
    """Suppress logits whose cumulative softmax probability exceeds p (nucleus).
    Mask is computed without gradients so all positions still receive gradient updates."""
    if p >= 1.0 or p <= 0.0:
        return logits
    with torch.no_grad():
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Exclude the token that pushes cumprob over threshold (shift to keep boundary)
        sorted_mask = (cumprobs - F.softmax(sorted_logits, dim=-1)) >= p
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, sorted_mask)
    return logits - mask.float() * penalty


def _build_prompt_attn_mask(
    message_ids: torch.Tensor,
    eos_allowed_idx: int,
    batch_size: int,
    pre_prompt_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a prompt attention mask that zeros out EoS-sampled positions.

    Args:
        message_ids: (batch, seq_len) discrete token indices in allowed_tokens space.
        eos_allowed_idx: Index of EoS token in the allowed vocabulary.
        batch_size: Number of sequences in the batch.
        pre_prompt_len: Number of pre-prompt tokens prepended before the learnable prompt.
        device: Torch device.

    Returns:
        prompt_attn_mask: (batch, pre_prompt_len + seq_len) long tensor; 0 at EoS positions, 1 elsewhere.
    """
    eos_in_prompt = (message_ids == eos_allowed_idx)  # (batch, seq_len)
    prompt_attn = (~eos_in_prompt).long()              # (batch, seq_len)
    if pre_prompt_len > 0:
        pre_attn = torch.ones(batch_size, pre_prompt_len, device=device, dtype=torch.long)
        return torch.cat([pre_attn, prompt_attn], dim=1)
    return prompt_attn


def stgs_forward_pass(
    *,
    model,
    tokenizer,
    loss_instance,
    learnable_inputs: torch.Tensor,
    stgs_module: STGS,
    embedding_weights_subset: torch.Tensor,
    allowed_tokens: torch.Tensor,
    target_length: int,
    batch_size: int,
    device: torch.device,
    model_precision: str,
    pre_prompt: Optional[str],
    bptt: bool,
    bptt_stgs: Optional[STGS],
    filter_vocab: bool,
    teacher_forcing: bool = False,
    target_tokens: Optional[torch.Tensor] = None,
    compute_tf_completion_logits: bool = False,
    logits_top_k: int = 0,
    logits_top_p: float = 1.0,
    logits_filter_penalty: float = 1e4,
    eos_allowed_idx: int = -1,
    mask_eos_attention: bool = False,
) -> ForwardPassResult:
    message_logits = learnable_inputs.repeat(batch_size, 1, 1)
    if model_precision == "half":
        message_logits = message_logits.half()
    # Soft top-k / top-p filtering (gradient flows to all positions via detached mask)
    if logits_top_k > 0:
        message_logits = _top_k_logit_filter(message_logits, logits_top_k, logits_filter_penalty)
    if logits_top_p < 1.0:
        message_logits = _top_p_logit_filter(message_logits, logits_top_p, logits_filter_penalty)
    # Read gumbel_noise_scale set externally (e.g. by adaptive noise controller in main.py)
    gumbel_noise_scale = getattr(stgs_module, 'gumbel_noise_scale', 1.0)
    message_ids, message_one_hot, eff_temperature, y_soft = stgs_module.forward(
        message_logits,
        gumbel_noise_scale=gumbel_noise_scale,
        embedding_weights=embedding_weights_subset,
    )

    input_embeddings = torch.matmul(message_one_hot, embedding_weights_subset)
    embedding_layer = model.get_input_embeddings()

    pre_prompt_len = 0
    if pre_prompt is not None:
        pre_prompt_tokens = tokenizer(pre_prompt, return_tensors="pt").input_ids.to(device)
        pre_prompt_embeds = embedding_layer(pre_prompt_tokens).repeat(batch_size, 1, 1)
        pre_prompt_len = pre_prompt_tokens.shape[1]
        current_embeds = torch.cat([pre_prompt_embeds, input_embeddings], dim=1)
    else:
        current_embeds = input_embeddings

    # Build optional attention mask that suppresses EoS-sampled prompt positions
    prompt_attn_mask = None
    if mask_eos_attention and eos_allowed_idx >= 0:
        prompt_attn_mask = _build_prompt_attn_mask(
            message_ids=message_ids,
            eos_allowed_idx=eos_allowed_idx,
            batch_size=batch_size,
            pre_prompt_len=pre_prompt_len,
            device=device,
        )

    seq_len = current_embeds.shape[1]
    bptt_eff_temperature = torch.tensor(0.0, device=device)

    if teacher_forcing and target_tokens is not None:
        # FAST PATH: Single forward pass with teacher forcing (like SODA)
        # This processes all positions in one call instead of autoregressive loop
        if bptt:
            logger.warning("Teacher forcing is incompatible with bptt=True. Falling back to autoregressive.")
        else:
            # Get target token embeddings (shifted by 1 for next-token prediction)
            # target_tokens shape: (batch_size, target_length)
            # We need embeddings for positions 0 to target_length-2 to predict 1 to target_length-1
            target_embeds = embedding_weights_subset[target_tokens[:, :-1]]
            # Concatenate prompt embeddings with target embeddings
            combined_embeds = torch.cat([current_embeds, target_embeds], dim=1)

            # Single forward pass for all positions
            tf_attention_mask = None
            if prompt_attn_mask is not None:
                tf_attention_mask = torch.cat(
                    [prompt_attn_mask,
                     torch.ones(batch_size, target_length - 1, device=device, dtype=torch.long)],
                    dim=1,
                )
            outputs = model(
                inputs_embeds=combined_embeds,
                attention_mask=tf_attention_mask,
                output_hidden_states=True,
                use_cache=False,  # No need for KV cache in teacher forcing
                return_dict=True,
            )

            logits = outputs.logits
            # Get prompt logits (positions 0 to seq_len-1 predict positions 1 to seq_len)
            prompt_logits = logits[:, :seq_len, :]
            prompt_logits = prompt_logits[..., allowed_tokens]

            # Get generated logits for target prediction positions
            # Position seq_len-1 predicts first target token, position seq_len predicts second, etc.
            generated_logits = logits[:, seq_len-1:seq_len-1+target_length, :]
            generated_logits = generated_logits[..., allowed_tokens]

            # Get completion IDs from argmax of generated logits
            completion_ids = generated_logits.argmax(dim=-1)

            # Build hidden states list for loss computation (one entry per target position)
            all_hidden_states = [
                tuple(hs[:, seq_len-1+i:seq_len+i, :] for hs in outputs.hidden_states)
                for i in range(target_length)
            ]

            # In TF mode, generated_logits is already teacher-forced.
            # Combine prompt_logits and generated_logits to form tf_generated_logits
            # for the prompt + teacher-forced completion perplexity metric.
            tf_generated_logits = torch.cat([prompt_logits, generated_logits], dim=1)

            losses_dict = loss_instance.compute_loss(
                input_dict={
                    "generated_logits": generated_logits,
                    "generated_hidden_states": all_hidden_states,
                    "completion_ids": completion_ids,
                    "prompt_ids": message_ids,
                    "prompt_argmax_ids": message_logits.argmax(dim=-1),
                    "prompt_logits": prompt_logits,
                    "tf_generated_logits": tf_generated_logits,
                    "prompt_input_logits": message_logits,
                },
            )
            loss = losses_dict["sumloss"]

            estimator_state: Dict[str, Optional[torch.Tensor]] = {
                "eff_temperature": eff_temperature,
                "bptt_eff_temperature": bptt_eff_temperature,
            }

            return ForwardPassResult(
                loss=loss,
                backward_loss=loss,
                losses_dict=losses_dict,
                generated_logits=generated_logits,
                prompt_ids=message_ids,
                prompt_logits=prompt_logits,
                completion_ids=completion_ids,
                estimator_state=estimator_state,
            )

    # SLOW PATH: Existing autoregressive generation (unchanged)
    outputs = model(
        inputs_embeds=current_embeds,
        attention_mask=prompt_attn_mask,
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )

    logits = outputs.logits
    hidden_states = outputs.hidden_states
    logits_allowed = logits[..., allowed_tokens]
    prompt_logits = logits_allowed
    past_key_values = outputs.past_key_values

    all_logits = [logits_allowed[:, -1:, :]]
    all_hidden_states = [outputs.hidden_states]

    # Track growing attention mask for autoregressive steps
    ar_attn = prompt_attn_mask

    completion_ids = []
    current_length = 1
    while current_length < target_length:
        if bptt:
            assert bptt_stgs is not None, "BPTT requested but no STGS module provided for it."
            next_token_id, next_token_one_hot, bptt_eff_temperature, bptt_y_soft = bptt_stgs(
                all_logits[-1],
                hidden_states=outputs.hidden_states,
                temperature_param_indices=[current_length-1],
            )
            next_token_embedding = torch.matmul(next_token_one_hot, embedding_weights_subset)
        else:
            next_token_id = torch.argmax(all_logits[-1], dim=-1)
            if filter_vocab:
                next_token_embedding = embedding_weights_subset[next_token_id]
            else:
                next_token_embedding = embedding_layer(next_token_id)

        completion_ids.append(next_token_id)

        if ar_attn is not None:
            ar_attn = torch.cat(
                [ar_attn, torch.ones(batch_size, 1, device=device, dtype=torch.long)],
                dim=1,
            )

        outputs = model(
            inputs_embeds=next_token_embedding,
            past_key_values=past_key_values,
            attention_mask=ar_attn,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values

        next_logits = outputs.logits[..., allowed_tokens]
        all_logits.append(next_logits)
        all_hidden_states.append(outputs.hidden_states)

        current_length += 1

    generated_logits = torch.cat(all_logits, dim=1)
    completion_ids = torch.cat(completion_ids, dim=1)

    # Optionally run a single TF forward pass to get teacher-forced completion logits
    if compute_tf_completion_logits and target_tokens is not None:
        target_embeds = embedding_weights_subset[target_tokens[:, :-1]]
        combined_embeds = torch.cat([current_embeds, target_embeds], dim=1)
        tf_compl_attention_mask = None
        if prompt_attn_mask is not None:
            tf_compl_attention_mask = torch.cat(
                [prompt_attn_mask,
                 torch.ones(batch_size, target_length - 1, device=device, dtype=torch.long)],
                dim=1,
            )
        tf_outputs = model(
            inputs_embeds=combined_embeds,
            attention_mask=tf_compl_attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        tf_logits = tf_outputs.logits
        tf_prompt_logits = tf_logits[:, :seq_len, :][..., allowed_tokens]
        tf_compl_logits = tf_logits[:, seq_len-1:seq_len-1+target_length, :][..., allowed_tokens]
        tf_generated_logits = torch.cat([tf_prompt_logits, tf_compl_logits], dim=1)
    else:
        tf_generated_logits = None

    losses_dict = loss_instance.compute_loss(
        input_dict={
            "generated_logits": generated_logits,
            "generated_hidden_states": all_hidden_states,
            "completion_ids": completion_ids,
            "prompt_ids": message_ids,
            "prompt_argmax_ids": message_logits.argmax(dim=-1),
            "prompt_logits": prompt_logits,
            "tf_generated_logits": tf_generated_logits,
            "prompt_input_logits": message_logits,
        },
    )
    loss = losses_dict["sumloss"]

    estimator_state: Dict[str, Optional[torch.Tensor]] = {
        "eff_temperature": eff_temperature,
        "bptt_eff_temperature": bptt_eff_temperature,
    }

    return ForwardPassResult(
        loss=loss,
        backward_loss=loss,
        losses_dict=losses_dict,
        generated_logits=generated_logits,
        prompt_ids=message_ids,
        prompt_logits=prompt_logits,
        completion_ids=completion_ids,
        estimator_state=estimator_state,
    )


def estimate_stgs_gradient_variance(
    num_samples: int,
    baseline_grad: Optional[torch.Tensor],
    forward_pass_fn,
    learnable_inputs: torch.Tensor,
) -> Dict[str, float]:
    if num_samples < 2 or baseline_grad is None:
        return {}

    grad_mean = baseline_grad.clone()
    grad_m2 = torch.zeros_like(grad_mean)
    sample_count = 1

    grad_samples = [grad_mean]
    pbar_desc = "STGS Var Est"
    for _ in tqdm(range(num_samples - 1), desc=pbar_desc, leave=False):
        sample_result: ForwardPassResult = forward_pass_fn()
        sample_grad = torch.autograd.grad(
            sample_result.backward_loss,
            learnable_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0].detach()

        sample_count += 1
        delta = sample_grad - grad_mean
        grad_mean = grad_mean + delta / sample_count
        grad_m2 = grad_m2 + delta * (sample_grad - grad_mean)
        grad_samples.append(sample_grad)

        del sample_result, sample_grad

    if sample_count < 2:
        return {}

    variance = grad_m2 / (sample_count - 1)
    return {
        "stgs_grad_variance_mean": variance.mean().item(),
        "stgs_grad_variance_max": variance.max().item(),
        "stgs_grad_variance_norm": variance.norm().item(),
        "stgs_grad_mean_norm": grad_mean.norm().item(),
        "stgs_grad_variance_samples": int(sample_count),
        "stgs_grad_samples": grad_samples,
    }


def estimate_stgs_gradient_bias(
    *,
    stgs_num_samples: int,
    reinforce_num_samples: int,
    baseline_grad: Optional[torch.Tensor],
    stgs_forward_pass_fn: Callable[[], ForwardPassResult],
    reinforce_forward_pass_fn: Callable[..., ForwardPassResult],
    learnable_inputs: torch.Tensor,
    reinforce_update_baseline: bool,
    stgs_grad_samples: List[torch.Tensor]=[],
) -> Dict[str, float]:
    if baseline_grad is None:
        return {}

    if stgs_num_samples < 1 or reinforce_num_samples < 1:
        return {}

    if stgs_grad_samples == []:
        stgs_grad_samples = [baseline_grad.clone()]
    
    stgs_sample_count = len(stgs_grad_samples)

    for _ in tqdm(range(stgs_num_samples - stgs_sample_count), desc="STGS Bias (STGS)", leave=False):
        sample_result = stgs_forward_pass_fn()
        sample_grad = torch.autograd.grad(
            sample_result.backward_loss,
            learnable_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0].detach()

        stgs_sample_count += 1
        stgs_grad_samples.append(sample_grad)

        del sample_result, sample_grad

    stgs_grad_mean = torch.stack(stgs_grad_samples).mean(dim=0)

    reinforce_grad_sum = torch.zeros_like(baseline_grad)
    reinforce_sample_count = 0
    reinforce_grad_samples = []
    for _ in tqdm(range(reinforce_num_samples), desc="STGS Bias (REINFORCE)", leave=False):
        sample_result = reinforce_forward_pass_fn(update_baseline=reinforce_update_baseline)
        sample_grad = torch.autograd.grad(
            sample_result.backward_loss,
            learnable_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0].detach()

        reinforce_sample_count += 1
        reinforce_grad_samples.append(sample_grad)

        del sample_result, sample_grad

    if reinforce_sample_count < 1:
        return {}

    reinforce_grad_mean = torch.stack(reinforce_grad_samples).mean(dim=0)
    reinforce_grad_variance = torch.stack(reinforce_grad_samples).var(dim=0)
    bias_tensor = stgs_grad_mean - reinforce_grad_mean

    '''
    return {
        "stgs_grad_bias_norm": bias_tensor.norm().item(),
        "stgs_grad_bias_mean": bias_tensor.mean().item(),
        "stgs_grad_bias_abs_mean": bias_tensor.abs().mean().item(),
        "stgs_grad_bias_abs_max": bias_tensor.abs().max().item(),
        "stgs_grad_mean_norm": stgs_grad_mean.norm().item(),
        "stgs_grad_bias_reference_grad_mean_norm": reinforce_grad_mean.norm().item(),
        "stgs_grad_bias_stgs_samples": int(stgs_sample_count),
        "stgs_grad_bias_reference_samples": int(reinforce_sample_count),
    }
    '''
    return {
        "stgs_grad_bias_norm": bias_tensor.norm().item(),
        "stgs_grad_bias_mean": bias_tensor.mean().item(),
        "stgs_grad_bias_abs_mean": bias_tensor.abs().mean().item(),
        "stgs_grad_bias_abs_max": bias_tensor.abs().max().item(),
        "stgs_grad_mean_norm": stgs_grad_mean.norm().item(),
        "stgs_grad_bias_reference_grad_mean_norm": reinforce_grad_mean.norm().item(),
        "stgs_grad_bias_reference_grad_variance_norm": reinforce_grad_variance.norm().item(),
        "stgs_grad_bias_reference_grad_variance_mean": reinforce_grad_variance.mean().item(),
        "stgs_grad_bias_reference_grad_variance_max": reinforce_grad_variance.max().item(),
        "stgs_grad_bias_stgs_samples": int(stgs_sample_count),
        "stgs_grad_bias_reference_samples": int(reinforce_sample_count),
    }
