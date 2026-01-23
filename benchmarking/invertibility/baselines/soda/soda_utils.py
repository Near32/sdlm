"""
SODA utilities - shared components for SODA optimization.
Based on the SODA ICML Experiment Notebook implementation.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional


class DotDict(dict):
    """Dictionary with attribute-style access."""
    def __getattr__(self, name):
        return self.get(name)
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        del self[name]


class CustomAdam(torch.optim.Optimizer):
    """
    Adam optimizer without bias correction, matching the original SODA paper.
    This is critical for reproducing SODA results.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(CustomAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("CustomAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # First moment (m_t)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # Second moment (v_t)

                m, v = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # No bias correction (key difference from standard Adam)
                m_hat = m
                v_hat = v

                denom = v_hat.sqrt().add(group['eps'])
                # theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
                p.data.addcdiv_(m_hat, denom, value=-group['lr'])

        return loss


def compute_lcs_length(seq1: List[int], seq2: List[int]) -> int:
    """
    Compute the length of the Longest Common Subsequence between two sequences.

    Args:
        seq1: First sequence of token IDs
        seq2: Second sequence of token IDs

    Returns:
        Length of the LCS
    """
    m, n = len(seq1), len(seq2)

    # Create a 2D DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill in the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def compute_lcs_ratio(pred_tokens: List[int], target_tokens: List[int]) -> float:
    """
    Compute the LCS ratio between predicted and target token sequences.

    Args:
        pred_tokens: Predicted token IDs
        target_tokens: Target token IDs

    Returns:
        LCS ratio (0.0 to 1.0)
    """
    if len(target_tokens) == 0:
        return 0.0
    lcs = compute_lcs_length(pred_tokens, target_tokens)
    return lcs / len(target_tokens)


def check_exact_match(pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> bool:
    """
    Check if predicted tokens exactly match target tokens.

    Args:
        pred_tokens: Predicted token tensor
        target_tokens: Target token tensor

    Returns:
        True if exact match, False otherwise
    """
    return torch.equal(pred_tokens.cpu(), target_tokens.cpu())


def initialize_embedding_params(
    vocab_size: int,
    seq_len: int,
    device: torch.device,
    init_strategy: str = "zeros",
    init_std: float = 0.05
) -> torch.Tensor:
    """
    Initialize the learnable embedding parameters for SODA.

    Args:
        vocab_size: Size of the vocabulary
        seq_len: Length of the sequence to optimize
        device: Device to place the tensor on
        init_strategy: Initialization strategy ("zeros", "normal")
        init_std: Standard deviation for normal initialization

    Returns:
        Initialized embedding parameter tensor of shape (1, seq_len, vocab_size)
    """
    if init_strategy == "zeros":
        return torch.zeros((1, seq_len, vocab_size), device=device)
    elif init_strategy == "normal":
        embed = torch.empty((1, seq_len, vocab_size), device=device)
        torch.nn.init.normal_(embed, std=init_std)
        return embed
    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy}")


def create_optimizer(
    params: List[torch.Tensor],
    learning_rate: float,
    betas: Tuple[float, float] = (0.9, 0.995),
    bias_correction: bool = False
) -> torch.optim.Optimizer:
    """
    Create an optimizer for SODA optimization.

    Args:
        params: List of parameters to optimize
        learning_rate: Learning rate
        betas: Adam beta parameters
        bias_correction: Whether to use bias correction (False = CustomAdam, True = standard Adam)

    Returns:
        Optimizer instance
    """
    if bias_correction:
        return torch.optim.Adam(params, lr=learning_rate, betas=betas)
    else:
        return CustomAdam(params, lr=learning_rate, betas=betas)


def apply_embedding_decay(params: List[torch.Tensor], decay_rate: float):
    """
    Apply decay to embedding parameters (in-place).

    Args:
        params: List of embedding parameter tensors
        decay_rate: Decay rate to multiply parameters by
    """
    with torch.no_grad():
        for param in params:
            param.mul_(decay_rate)


def reset_optimizer_state(optimizer: torch.optim.Optimizer, params: List[torch.Tensor]):
    """
    Reset the optimizer state for given parameters.

    Args:
        optimizer: The optimizer to reset
        params: The parameters whose states to reset
    """
    for param in params:
        if param in optimizer.state:
            del optimizer.state[param]


def reinitialize_params(params: List[torch.Tensor], std: float = 0.1):
    """
    Reinitialize parameters with normal distribution (in-place).

    Args:
        params: List of parameter tensors to reinitialize
        std: Standard deviation for normal initialization
    """
    with torch.no_grad():
        for param in params:
            param.normal_(std=std)


def compute_fluency_penalty(
    logits: torch.Tensor,
    pred_one_hot: torch.Tensor
) -> torch.Tensor:
    """
    Compute the fluency regularization penalty.

    The penalty encourages the predicted tokens to have high probability under
    the model's own predictions, promoting fluent text.

    Args:
        logits: Model logits of shape (batch, seq_len, vocab_size)
        pred_one_hot: Predicted one-hot distribution of shape (batch, seq_len, vocab_size)

    Returns:
        Fluency penalty tensor of shape (batch,)
    """
    # Get the argmax token indices for positions 1 onwards
    next_token_indices = pred_one_hot[:, 1:, :].argmax(dim=-1).unsqueeze(-1)

    # Get log probabilities from logits (excluding the last position)
    log_probs = logits[:, :-1, :].softmax(dim=-1).log()

    # Gather the log probability of the predicted next tokens
    gathered_log_probs = log_probs.gather(2, next_token_indices).squeeze(-1)

    # Negative mean (we want to maximize probability = minimize negative log prob)
    penalty = -gathered_log_probs.mean(dim=-1)

    return penalty
