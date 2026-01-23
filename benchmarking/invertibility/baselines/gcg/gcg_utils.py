"""
GCG utilities - shared components for GCG optimization.
Based on the SODA ICML Experiment Notebook GCG implementation.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


class DotDict(dict):
    """Dictionary with attribute-style access."""
    def __getattr__(self, name):
        return self.get(name)
    def __setattr__(self, name, value):
        self[name] = value
    def __delattr__(self, name):
        del self[name]


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


def initialize_tokens(
    vocab_size: int,
    seq_len: int,
    device: torch.device,
    init_strategy: str = "zeros",
    init_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Initialize the token sequence for GCG.

    Args:
        vocab_size: Size of the vocabulary
        seq_len: Length of the sequence to optimize
        device: Device to place the tensor on
        init_strategy: Initialization strategy ("zeros" or "random")
        init_tokens: Pre-specified initialization tokens

    Returns:
        Initialized token tensor of shape (1, seq_len)
    """
    if init_tokens is not None:
        return init_tokens.to(device).unsqueeze(0) if init_tokens.dim() == 1 else init_tokens.to(device)

    if init_strategy == "zeros":
        return torch.zeros((1, seq_len), dtype=torch.long, device=device)
    elif init_strategy == "random":
        return torch.randint(0, vocab_size, (1, seq_len), device=device)
    else:
        raise ValueError(f"Unknown init_strategy: {init_strategy}")


def select_mutation_positions(
    grad: torch.Tensor,
    num_mutations: int,
    pos_choice: str = "uniform",
    batch_size: int = 1,
    seq_len: int = None,
) -> torch.Tensor:
    """
    Select positions to mutate based on gradient information.

    Args:
        grad: Gradient tensor of shape (batch, seq_len, vocab_size)
        num_mutations: Number of positions to mutate
        pos_choice: Selection strategy - "uniform", "weighted", or "greedy"
        batch_size: Current batch size
        seq_len: Sequence length

    Returns:
        Position indices tensor of shape (batch, num_mutations)
    """
    if seq_len is None:
        seq_len = grad.shape[1]

    device = grad.device

    if pos_choice == "uniform":
        # Uniform random selection
        index_weights = torch.ones((batch_size, seq_len), device=device)
        new_token_pos = torch.multinomial(index_weights, num_mutations, replacement=False)

    elif pos_choice == "weighted":
        # Weighted by gradient norm
        grad_norm = grad.norm(dim=-1)  # (batch, seq_len)
        # Ensure positive weights
        index_weights = (grad_norm + 1e-6) - grad_norm.min(dim=-1, keepdim=True).values
        new_token_pos = torch.multinomial(index_weights, num_mutations, replacement=False)

    elif pos_choice == "greedy":
        # Select positions with highest gradient norm
        grad_norm = grad.norm(dim=-1)  # (batch, seq_len)
        new_token_pos = grad_norm.topk(num_mutations).indices

    else:
        raise ValueError(f"Unknown pos_choice: {pos_choice}")

    return new_token_pos


def select_token_candidates(
    grad: torch.Tensor,
    new_token_pos: torch.Tensor,
    top_k: int,
    token_choice: str = "uniform",
    num_mutations: int = 1,
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Select token candidates for mutation based on gradient information.

    Args:
        grad: Gradient tensor of shape (batch, seq_len, vocab_size)
        new_token_pos: Positions to mutate, shape (batch, num_mutations)
        top_k: Number of top tokens to consider
        token_choice: Selection strategy - "uniform" or "weighted"
        num_mutations: Number of mutations
        batch_size: Current batch size

    Returns:
        Selected token values tensor of shape (batch, num_mutations)
    """
    device = grad.device
    vocab_size = grad.shape[-1]

    # Get top-k gradient values and indices
    if top_k != vocab_size:
        topk_grad_values, topk_grad_indices = grad.topk(top_k, dim=-1)
    else:
        topk_grad_values = grad
        topk_grad_indices = torch.arange(vocab_size, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)

    # Get values at selected positions
    batch_arrange = torch.arange(batch_size, device=device).unsqueeze(-1)
    topk_grad_indices_at_pos = topk_grad_indices[batch_arrange, new_token_pos]  # (batch, num_mutations, top_k)
    topk_grad_values_at_pos = topk_grad_values[batch_arrange, new_token_pos]  # (batch, num_mutations, top_k)

    if token_choice == "uniform":
        # Uniform random selection from top-k
        chosen_grad_indices = torch.randint(
            0, top_k, (batch_size, num_mutations), device=device
        )

    elif token_choice == "weighted":
        # Weighted sampling based on gradient values
        index_weights = (topk_grad_values_at_pos + 1e-6) - topk_grad_values_at_pos.min(dim=-1, keepdim=True).values
        # Flatten for multinomial
        chosen_grad_indices_flat = torch.multinomial(
            index_weights.view(-1, top_k), num_samples=1, replacement=True
        )[:, 0]
        chosen_grad_indices = chosen_grad_indices_flat.view(batch_size, num_mutations)

    else:
        raise ValueError(f"Unknown token_choice: {token_choice}")

    # Gather the selected token values
    rows_arrange = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_mutations)
    cols_arrange = torch.arange(num_mutations, device=device).unsqueeze(0).expand(batch_size, -1)
    new_token_val = topk_grad_indices_at_pos[rows_arrange, cols_arrange, chosen_grad_indices]

    return new_token_val


def apply_mutations(
    pred_tokens: torch.Tensor,
    new_token_pos: torch.Tensor,
    new_token_val: torch.Tensor,
) -> torch.Tensor:
    """
    Apply token mutations at specified positions.

    Args:
        pred_tokens: Current prediction tokens, shape (batch, seq_len)
        new_token_pos: Positions to mutate, shape (batch, num_mutations)
        new_token_val: New token values, shape (batch, num_mutations)

    Returns:
        Mutated token tensor
    """
    new_pred_tokens = pred_tokens.clone()
    new_pred_tokens = new_pred_tokens.scatter(1, new_token_pos, new_token_val)
    return new_pred_tokens
