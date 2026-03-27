"""
Implementation of the Straight-Through Gumbel-Softmax operation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch import Tensor


class STGS(nn.Module):
    """
    Straight-Through Gumbel-Softmax operation that allows for differentiable
    sampling from a categorical distribution with parameterized temperature.
    
    Args:
        vocab_size: Size of the vocabulary
        stgs_hard: If True, uses hard (discrete) samples in forward pass
        init_temperature: Initial temperature for Gumbel-Softmax
        learnable_temperature: If True, makes temperature a learnable parameter
        conditioning_dim: Dimension of conditioning vector for adaptive temperature (0 for fixed temperature)
        dropout: rate of dropout gating of the gradient, default = 0.
        eps: Small epsilon for numerical stability
        device: Device to run the computation on
    """
    def __init__(
        self,
        vocab_size: int,
        stgs_hard: bool = False,
        init_temperature: float = 1.0,
        learnable_temperature: bool = False,
        nbr_learnable_temperatures: Optional[int] = None,
        conditioning_dim: int = 0,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        output_dropout: float = 0.0,
        output_top_k: int = 0,
        stgs_hard_method: str = "categorical",
        stgs_hard_embsim_probs: str = "gumbel_soft",
        stgs_hard_embsim_strategy: str = "nearest",  # nearest | topk_rerank | topk_sample | margin_fallback | lm_topk_restrict
        stgs_hard_embsim_top_k: int = 8,
        stgs_hard_embsim_rerank_alpha: float = 0.5,
        stgs_hard_embsim_sample_tau: float = 1.0,
        stgs_hard_embsim_margin: float = 0.0,
        stgs_hard_embsim_fallback: str = "argmax",  # argmax | categorical
        logits_normalize: str = "none",   # "none" | "center" | "zscore"
        eps: float = 1e-12,
        device: str = "cpu",
        tokenizer = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.stgs_hard = stgs_hard
        self.stgs_hard_method = stgs_hard_method
        self.stgs_hard_embsim_probs = stgs_hard_embsim_probs
        self.stgs_hard_embsim_strategy = stgs_hard_embsim_strategy
        self.stgs_hard_embsim_top_k = stgs_hard_embsim_top_k
        self.stgs_hard_embsim_rerank_alpha = stgs_hard_embsim_rerank_alpha
        self.stgs_hard_embsim_sample_tau = stgs_hard_embsim_sample_tau
        self.stgs_hard_embsim_margin = stgs_hard_embsim_margin
        self.stgs_hard_embsim_fallback = stgs_hard_embsim_fallback
        self.logits_normalize = logits_normalize
        self.init_temperature = init_temperature
        self.learnable_temperature = learnable_temperature
        self.nbr_learnable_temperatures = nbr_learnable_temperatures
        if self.learnable_temperature:
            if self.nbr_learnable_temperatures is None:
                self.nbr_learnable_temperatures = 1
        self.conditioning_dim = conditioning_dim
        self.dropout = dropout
        assert 0 <= self.dropout <= 1, "Dropout rate must be between 0 and 1"
        self.input_dropout = input_dropout
        assert 0.0 <= self.input_dropout < 1.0, "input_dropout must be in [0, 1)"
        self.output_dropout = output_dropout
        assert 0.0 <= self.output_dropout < 1.0, "output_dropout must be in [0, 1)"
        self.output_top_k = output_top_k
        assert output_top_k >= 0, "output_top_k must be >= 0"
        self.eps = eps
        self.device = device
        self.tokenizer = tokenizer
        self.last_snr: Optional[Tensor] = None

        if self.learnable_temperature:
            if self.conditioning_dim < 1:
                self.temperature_param = nn.Parameter(torch.rand(self.nbr_learnable_temperatures, requires_grad=True, device=self.device))
            else:
                self.tau_fc = nn.Sequential(
                    nn.Linear(self.conditioning_dim, self.nbr_learnable_temperatures, bias=False),
                    nn.Softplus()
                ).to(device=device)

    def forward(
        self,
        x: Tensor,
        temperature: Optional[float] = None,
        hidden_states: Optional[Tensor] = None,
        dropout: Optional[float] = None,
        temperature_param_indices: Optional[List[int]] = None,
        gumbel_noise_scale: float = 1.0,
        input_dropout: Optional[float] = None,
        output_dropout: Optional[float] = None,
        output_top_k: Optional[int] = None,
        embedding_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with STGS sampling.
        
        Args:
            x: Input logits (batch_size, seq_len, vocab_size)
            temperature: Optional temperature to use for Gumbel-Softmax sampling
            hidden_states: Optional hidden states for conditional temperature
            dropout: Optional dropout rate to use for gating the gradient
            
        Returns:
            Tuple of (sampled_tokens, sampled_probs, temperature)
        """
        if temperature is not None:
            eff_temperature = torch.tensor([temperature], device=self.device)
        elif self.learnable_temperature:
            if self.conditioning_dim < 1:
                # PREVIOUSLY:
                #eff_temperature = F.softplus(self.temperature_param) + self.init_temperature
                #eff_temperature = self.eps + 1.0 / (F.softplus(self.temperature_param) + 1.0 / (self.eps + self.init_temperature))
                # TANH:
                eff_temperature = self.eps+self.init_temperature*(1+F.tanh(self.temperature_param))
                # SIGMOID:
                #eff_temperature = self.eps+self.init_temperature*F.sigmoid(self.temperature_param)
                #EXP:
                #eff_temperature = self.eps+torch.exp(self.temperature_param)
                ##
                eff_temperature = eff_temperature.reshape(1, -1, 1)
                batch_size = x.shape[0]
                eff_temperature = eff_temperature.repeat(batch_size, 1, 1)
            else:
                assert hidden_states is not None
                batch_size = x.shape[0]
                seq_len = x.shape[1]
                last_hidden_state = hidden_states[-1][:,-1,:].reshape(batch_size, self.conditioning_dim).float()
                self.inv_tau0 = 1.0/(self.eps+self.init_temperature)
                #eff_temperature = self.eps + 1. / (self.tau_fc(last_hidden_state)+self.inv_tau0).reshape(batch_size, -1, 1)
                # TANH:
                eff_temperature = self.eps+self.init_temperature*(1+F.tanh(self.tau_fc(last_hidden_state)))
                # SIGMOID:
                #eff_temperature = self.eps+self.init_temperature*F.sigmoid(self.tau_fc(last_hidden_state))
                if self.nbr_learnable_temperatures==1 \
                and self.nbr_learnable_temperatures != seq_len:
                    eff_temperature = eff_temperature.repeat(1, seq_len, 1)
            if self.nbr_learnable_temperatures > 1 and temperature_param_indices is not None:
                eff_temperature = eff_temperature[:,temperature_param_indices,...]
        else:
            eff_temperature = torch.tensor([self.init_temperature], device=self.device)
        
        eff_temperature = eff_temperature.to(
            device=x.device,
            dtype=x.dtype,
        )

        # Per-position logit normalization (vocab dim only; no mixing across positions)
        # Mean/std are inside the computational graph — gradients flow through them.
        if self.logits_normalize != "none":
            mean = x.mean(dim=-1, keepdim=True)
            x = x - mean
            if self.logits_normalize == "zscore":
                std = x.std(dim=-1, keepdim=True)
                x = x / (std + self.eps)

        # Signal-to-noise ratio: Var(x/T) / Var(ε·gumbels/T) = Var(x) / (T² · ε² · π²/6)
        # Captures effective discriminability: high T → low SNR even with large logit variance.
        # eff_temperature shapes: (1,) | (batch, 1, 1) | (batch, seq_len, 1)
        _T = eff_temperature.float()
        _T_pos = _T.mean(dim=0).view(-1) if _T.dim() >= 2 else _T.view(-1)  # (seq_len,) or (1,)
        _noise_var = (_T_pos ** 2) * (gumbel_noise_scale ** 2) * (torch.pi ** 2 / 6.0)
        self.last_snr = (x.float().var(dim=-1).mean(dim=0) / _noise_var).detach()
        # shape: (seq_len,)

        # Input-distribution dropout: randomly zero vocab-dimension logits before Gumbel noise
        eff_input_dropout = self.input_dropout if input_dropout is None else input_dropout
        if eff_input_dropout > 0.0:
            x = F.dropout(x, p=eff_input_dropout, training=self.training)

        # Gumbel-Softmax sampling
        with torch.no_grad():
            u = torch.rand_like(x)*(0.999-self.eps)+self.eps
            gumbels = -torch.log( -torch.log(u)).to(
                device=x.device,
                dtype=x.dtype,
            )
        # batch_size x seq_len x vocab_size

        gumbel_logits = (x + gumbel_noise_scale * gumbels).float()
        y_soft = F.softmax(gumbel_logits / eff_temperature, dim=-1)
        # temperary float conversion to prevent from underflow and breaking Simplex assumption
        # batch_size x seq_len x vocab_size

        # Post-softmax dropout: randomly zero vocab-dimension probabilities before hard sampling
        eff_output_dropout = self.output_dropout if output_dropout is None else output_dropout
        if eff_output_dropout > 0.0:
            y_soft = F.dropout(y_soft, p=eff_output_dropout, training=self.training)

        # Post-softmax top-k: zero non-top-k probs and renormalise (mask is detached)
        eff_output_top_k = self.output_top_k if output_top_k is None else output_top_k
        if 0 < eff_output_top_k < self.vocab_size:
            with torch.no_grad():
                topk_vals = torch.topk(y_soft, eff_output_top_k, dim=-1).values
                threshold = topk_vals[..., -1:]   # (batch, seq, 1)
                mask = (y_soft >= threshold).to(y_soft.dtype)
            y_soft = y_soft * mask
            y_soft = y_soft / (y_soft.sum(dim=-1, keepdim=True) + self.eps)

        # Sampling from batched distribution y_soft:
        with torch.no_grad():
            if self.stgs_hard_embsim_probs == "input_logits":
                emb_probs = F.softmax(x.float(), dim=-1)
            else:  # "gumbel_soft" (default)
                emb_probs = y_soft.float()
        if self.stgs_hard_method in ("embsim-dot", "embsim-cos", "embsim-l2") and embedding_weights is not None:
            with torch.no_grad():
                emb_w = embedding_weights.float()
                soft_emb = torch.matmul(emb_probs, emb_w)
                if self.stgs_hard_method == "embsim-dot":
                    emb_scores = torch.matmul(soft_emb, emb_w.T)
                elif self.stgs_hard_method == "embsim-l2":
                    soft_sq = (soft_emb ** 2).sum(dim=-1, keepdim=True)           # (B, S, 1)
                    emb_sq  = (emb_w ** 2).sum(dim=-1)                             # (V,)
                    cross   = torch.matmul(soft_emb, emb_w.T)                      # (B, S, V)
                    l2_sq   = soft_sq - 2 * cross + emb_sq                        # (B, S, V)
                    emb_scores = -l2_sq
                else:  # embsim-cos
                    soft_norm = F.normalize(soft_emb, dim=-1, eps=self.eps)
                    emb_norm  = F.normalize(emb_w, dim=-1, eps=self.eps)
                    emb_scores = torch.matmul(soft_norm, emb_norm.T)

                strategy = self.stgs_hard_embsim_strategy
                if strategy == "nearest":
                    output_ids = emb_scores.argmax(dim=-1)
                elif strategy == "topk_rerank":
                    k = max(1, min(int(self.stgs_hard_embsim_top_k), emb_scores.shape[-1]))
                    cand_ids = emb_scores.topk(k, dim=-1).indices
                    cand_emb_scores = emb_scores.gather(dim=-1, index=cand_ids)
                    cand_lm_scores = emb_probs.gather(dim=-1, index=cand_ids)
                    alpha = max(0.0, min(float(self.stgs_hard_embsim_rerank_alpha), 1.0))
                    emb_std = cand_emb_scores.std(dim=-1, keepdim=True, unbiased=False)
                    lm_std = cand_lm_scores.std(dim=-1, keepdim=True, unbiased=False)
                    cand_emb_z = (cand_emb_scores - cand_emb_scores.mean(dim=-1, keepdim=True)) / (emb_std + self.eps)
                    cand_lm_z = (cand_lm_scores - cand_lm_scores.mean(dim=-1, keepdim=True)) / (lm_std + self.eps)
                    fused = alpha * cand_emb_z + (1.0 - alpha) * cand_lm_z
                    best_local = fused.argmax(dim=-1, keepdim=True)
                    output_ids = cand_ids.gather(dim=-1, index=best_local).squeeze(-1)
                elif strategy == "topk_sample":
                    k = max(1, min(int(self.stgs_hard_embsim_top_k), emb_scores.shape[-1]))
                    cand_ids = emb_scores.topk(k, dim=-1).indices
                    cand_emb_scores = emb_scores.gather(dim=-1, index=cand_ids)
                    tau = max(float(self.stgs_hard_embsim_sample_tau), self.eps)
                    cand_probs = F.softmax(cand_emb_scores / tau, dim=-1)
                    sampled_local = torch.distributions.Categorical(probs=cand_probs).sample().unsqueeze(-1)
                    output_ids = cand_ids.gather(dim=-1, index=sampled_local).squeeze(-1)
                elif strategy == "margin_fallback":
                    nearest_ids = emb_scores.argmax(dim=-1)
                    if emb_scores.shape[-1] < 2:
                        output_ids = nearest_ids
                    else:
                        top2 = emb_scores.topk(2, dim=-1)
                        margin = top2.values[..., 0] - top2.values[..., 1]
                        is_ambiguous = margin < float(self.stgs_hard_embsim_margin)
                        if self.stgs_hard_embsim_fallback == "categorical":
                            fallback_ids = torch.distributions.Categorical(probs=emb_probs).sample()
                        else:
                            fallback_ids = emb_probs.argmax(dim=-1)
                        output_ids = torch.where(is_ambiguous, fallback_ids, nearest_ids)
                elif strategy == "lm_topk_restrict":
                    k = max(1, min(int(self.stgs_hard_embsim_top_k), emb_scores.shape[-1]))
                    lm_cand_ids = emb_probs.topk(k, dim=-1).indices
                    cand_emb_scores = emb_scores.gather(dim=-1, index=lm_cand_ids)
                    best_local = cand_emb_scores.argmax(dim=-1, keepdim=True)
                    output_ids = lm_cand_ids.gather(dim=-1, index=best_local).squeeze(-1)
                else:
                    raise ValueError(f"Unknown stgs_hard_embsim_strategy: {strategy}")
        elif self.stgs_hard_method == "argmax": # "argmax" 
            output_ids = emb_probs.argmax(dim=-1, keepdim=False)
        # batch_size x seq_len
        else:  # "categorical" (default) or fallback
            #output_ids = torch.distributions.Categorical(probs=y_soft).sample()
            output_ids = torch.distributions.Categorical(probs=emb_probs).sample()
        # batch_size x seq_len
        del gumbel_logits
        y_soft = y_soft.to(x.dtype)
        
        eff_dropout = self.dropout if dropout is None else dropout
        backprop_gate = 1.0
        if eff_dropout > 0.0:
            backprop_gate = torch.rand(1,device=x.device) >= eff_dropout
            backprop_gate = backprop_gate.to(dtype=x.dtype, device=x.device).detach()
        # Straight-through: use hard in forward, soft in backward
        if self.stgs_hard:
            with torch.no_grad():
                y_hard = F.one_hot(output_ids, num_classes=self.vocab_size)
                # batch_size x seq_len x vocab_size
                # Type: half or full
                y_hard = y_hard.to(x.dtype)
            # Straight-through trick: 
            output_one_hot = y_hard.detach() + backprop_gate*(y_soft - y_soft.detach())
            # batch_size x seq_len x vocab_size
        else:
            output_one_hot = y_soft if backprop_gate else F.one_hot(output_ids, num_classes=self.vocab_size)
            # batch_size x seq_len x vocab_size

        output_one_hot = output_one_hot.to(
            device=x.device,
            dtype=x.dtype,
        )

        # Differentiable output ids:
        gathered_one_hot = torch.gather(output_one_hot, dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)
        diff_output_ids = output_ids.detach()+backprop_gate*(gathered_one_hot-gathered_one_hot.detach())
        # batch_size x seq_len
        
        return diff_output_ids, output_one_hot, eff_temperature, y_soft
