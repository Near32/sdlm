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
        eps: float = 1e-12,
        device: str = "cpu",
        tokenizer = None,
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
        self.dropout = dropout
        assert 0 <= self.dropout <= 1, "Dropout rate must be between 0 and 1"
        self.eps = eps
        self.device = device
        self.tokenizer = tokenizer

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
                eff_temperature = self.eps + 1.0 / (F.softplus(self.temperature_param) + 1.0 / (self.eps + self.init_temperature))
                eff_temperature = eff_temperature.reshape(1, -1, 1)
                batch_size = x.shape[0]
                eff_temperature = eff_temperature.repeat(batch_size, 1, 1)
            else:
                assert hidden_states is not None
                batch_size = x.shape[0]
                seq_len = x.shape[1]
                last_hidden_state = hidden_states[-1][:,-1,:].reshape(batch_size, self.conditioning_dim).float()
                self.inv_tau0 = 1.0/(self.eps+self.init_temperature)
                eff_temperature = self.eps + 1. / (self.tau_fc(last_hidden_state)+self.inv_tau0).reshape(batch_size, -1, 1)
                if self.nbr_learnable_temperatures==1 \
                and self.nbr_learnable_temperatures != seq_len:
                    eff_temperature = eff_temperature.repeat(1, seq_len, 1)
        else:
            eff_temperature = torch.tensor([self.init_temperature], device=self.device)
        
        eff_temperature = eff_temperature.to(
            device=x.device,
            dtype=x.dtype,
        )

        # Gumbel-Softmax sampling
        with torch.no_grad():
            u = torch.rand_like(x)*(0.999-self.eps)+self.eps
            gumbels = -torch.log( -torch.log(u)).to(
                device=x.device,
                dtype=x.dtype,
            )
        # batch_size x seq_len x vocab_size

        gumbel_logits = (x + gumbels).float()
        y_soft = F.softmax(gumbel_logits / eff_temperature, dim=-1)
        # temperary float conversion to prevent from underflow and breaking Simplex assumption
        # batch_size x seq_len x vocab_size

        # Sampling from batched distribution y_soft:
        output_ids = torch.distributions.Categorical(probs=y_soft).sample()
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
