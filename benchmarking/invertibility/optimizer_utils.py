from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.optim as optim

LOOKAHEAD_SLOW_BUFFER = "lookahead_slow_param"
LOOKAHEAD_STACK_STATE_KEY = "lookahead_stack"


def get_lookahead_slow_buffer_key(level_index: int) -> str:
    """Return the per-level slow-buffer key stored in optimizer state."""
    if level_index == 0:
        return LOOKAHEAD_SLOW_BUFFER
    return f"{LOOKAHEAD_SLOW_BUFFER}_{level_index}"


def normalize_lookahead_levels(lookahead_k) -> list[int]:
    """Normalize lookahead_k from int/string/list form to an ordered list of levels."""
    if lookahead_k is None:
        return []

    if isinstance(lookahead_k, str):
        text = lookahead_k.strip()
        if not text or text == "0":
            return []
        raw_values = [part.strip() for part in text.split(",")]
        if any(not part for part in raw_values):
            raise ValueError(f"Invalid lookahead_k value {lookahead_k!r}: empty level")
        try:
            levels = [int(part) for part in raw_values]
        except ValueError as exc:
            raise ValueError(
                f"Invalid lookahead_k value {lookahead_k!r}: expected integer or comma-separated integers"
            ) from exc
    elif isinstance(lookahead_k, int):
        levels = [lookahead_k]
    elif isinstance(lookahead_k, Sequence) and not isinstance(lookahead_k, (str, bytes)):
        levels = []
        for value in lookahead_k:
            if isinstance(value, int):
                levels.append(value)
                continue
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    raise ValueError(f"Invalid lookahead_k value {lookahead_k!r}: empty level")
                try:
                    levels.append(int(stripped))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid lookahead_k value {lookahead_k!r}: expected integer entries"
                    ) from exc
                continue
            raise ValueError(
                f"Invalid lookahead_k value {lookahead_k!r}: expected int, string, or sequence of ints"
            )
    else:
        raise ValueError(
            f"Invalid lookahead_k value {lookahead_k!r}: expected int, string, or sequence of ints"
        )

    if not levels:
        return []
    if len(levels) == 1 and levels[0] == 0:
        return []
    if any(level < 0 for level in levels):
        raise ValueError(f"lookahead_k values must be >= 0, got {levels}")
    if any(level == 0 for level in levels):
        raise ValueError(f"lookahead_k=0 disables Lookahead and cannot be mixed with enabled levels: {levels}")
    if any(level == 1 for level in levels):
        raise ValueError("lookahead_k values must be >= 2 when enabled")
    if len(set(levels)) != len(levels):
        raise ValueError(f"lookahead_k values must be unique, got {levels}")
    if levels != sorted(levels):
        raise ValueError(f"lookahead_k values must be strictly increasing, got {levels}")
    return levels


class LookaheadOptimizer(optim.Optimizer):
    """Wrap a base optimizer with periodic slow-weight interpolation."""

    def __init__(
        self,
        base_optimizer: optim.Optimizer,
        k: int = 5,
        alpha: float = 0.5,
        level_index: int = 0,
    ):
        if k < 1:
            raise ValueError(f"lookahead k must be >= 1, got {k}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"lookahead alpha must be in [0, 1], got {alpha}")

        self.base_optimizer = base_optimizer
        self._initializing = True
        super().__init__(base_optimizer.param_groups, base_optimizer.defaults)
        self.k = int(k)
        self.alpha = float(alpha)
        self.level_index = int(level_index)
        self.slow_buffer_key = get_lookahead_slow_buffer_key(self.level_index)
        self._lookahead_step = 0
        self._staged_slow_buffers: dict[torch.Tensor, torch.Tensor] = {}
        self.is_lookahead_optimizer = True
        self._initializing = False
        self._sync_optimizer_refs()

    def _sync_optimizer_refs(self) -> None:
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.defaults = self.base_optimizer.defaults

    def _param_to_index(self) -> dict[torch.Tensor, int]:
        mapping: dict[torch.Tensor, int] = {}
        index = 0
        for group in self.param_groups:
            for param in group["params"]:
                if param in mapping:
                    continue
                mapping[param] = index
                index += 1
        return mapping

    def _index_to_param(self) -> dict[int, torch.Tensor]:
        mapping: dict[int, torch.Tensor] = {}
        index = 0
        seen: set[torch.Tensor] = set()
        for group in self.param_groups:
            for param in group["params"]:
                if param in seen:
                    continue
                seen.add(param)
                mapping[index] = param
                index += 1
        return mapping

    def _collect_missing_slow_buffers(self) -> dict[torch.Tensor, torch.Tensor]:
        pending_buffers: dict[torch.Tensor, torch.Tensor] = {}
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                if self.slow_buffer_key in state:
                    continue
                if param in self._staged_slow_buffers:
                    pending_buffers[param] = self._staged_slow_buffers.pop(param)
                else:
                    pending_buffers[param] = param.detach().clone()
        return pending_buffers

    def _serialize_staged_slow_buffers(self) -> dict[int, torch.Tensor]:
        param_to_index = self._param_to_index()
        serialized: dict[int, torch.Tensor] = {}
        for param, slow_param in self._staged_slow_buffers.items():
            if param in param_to_index:
                serialized[param_to_index[param]] = slow_param.detach().clone()
        return serialized

    def _restore_staged_slow_buffers(self, serialized: dict[int, torch.Tensor]) -> None:
        index_to_param = self._index_to_param()
        self._staged_slow_buffers = {}
        for raw_index, slow_param in serialized.items():
            index = int(raw_index)
            if index not in index_to_param:
                continue
            param = index_to_param[index]
            self._staged_slow_buffers[param] = slow_param.detach().clone().to(
                device=param.device,
                dtype=param.dtype,
            )

    def stage_slow_buffer(self, param: torch.Tensor, slow_param: Optional[torch.Tensor] = None) -> None:
        self._staged_slow_buffers[param] = (
            param.detach().clone() if slow_param is None else slow_param.detach().clone()
        )

    @torch.no_grad()
    def step(self, closure=None):
        pending_buffers = self._collect_missing_slow_buffers()
        loss = self.base_optimizer.step(closure)
        for param, slow_param in pending_buffers.items():
            self.state[param][self.slow_buffer_key] = slow_param.to(
                device=param.device,
                dtype=param.dtype,
            )
        self._lookahead_step += 1

        if self._lookahead_step % self.k == 0:
            for group in self.param_groups:
                for param in group["params"]:
                    state = self.state[param]
                    slow_param = state[self.slow_buffer_key]
                    slow_param.add_(param.detach() - slow_param, alpha=self.alpha)
                    param.copy_(slow_param)

        return loss

    def zero_grad(self, set_to_none: Optional[bool] = None):
        if set_to_none is None:
            return self.base_optimizer.zero_grad()
        return self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        if getattr(self, "_initializing", False):
            return super().add_param_group(param_group)
        self.base_optimizer.add_param_group(param_group)
        self._sync_optimizer_refs()

    def state_dict(self):
        state_dict = self.base_optimizer.state_dict()
        stack = list(state_dict.get(LOOKAHEAD_STACK_STATE_KEY, []))
        stack.append(
            {
                "k": self.k,
                "alpha": self.alpha,
                "level_index": self.level_index,
                "slow_buffer_key": self.slow_buffer_key,
                "lookahead_step": self._lookahead_step,
                "staged_slow_buffers": self._serialize_staged_slow_buffers(),
            }
        )
        state_dict[LOOKAHEAD_STACK_STATE_KEY] = stack
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)

        metadata = None
        stack = list(state_dict.pop(LOOKAHEAD_STACK_STATE_KEY, []))
        if stack:
            metadata = stack.pop()
            if stack:
                state_dict[LOOKAHEAD_STACK_STATE_KEY] = stack
        else:
            legacy_step = state_dict.pop("lookahead_step", None)
            legacy_k = state_dict.pop("lookahead_k", None)
            legacy_alpha = state_dict.pop("lookahead_alpha", None)
            if legacy_step is not None or legacy_k is not None or legacy_alpha is not None:
                metadata = {
                    "k": self.k if legacy_k is None else legacy_k,
                    "alpha": self.alpha if legacy_alpha is None else legacy_alpha,
                    "level_index": self.level_index,
                    "slow_buffer_key": self.slow_buffer_key,
                    "lookahead_step": 0 if legacy_step is None else legacy_step,
                    "staged_slow_buffers": {},
                }

        if metadata is not None:
            self.k = int(metadata.get("k", self.k))
            self.alpha = float(metadata.get("alpha", self.alpha))
            self.level_index = int(metadata.get("level_index", self.level_index))
            self.slow_buffer_key = metadata.get(
                "slow_buffer_key",
                get_lookahead_slow_buffer_key(self.level_index),
            )
            self._lookahead_step = int(metadata.get("lookahead_step", 0))
            staged_slow_buffers = metadata.get("staged_slow_buffers", {})
        else:
            staged_slow_buffers = {}

        self.base_optimizer.load_state_dict(state_dict)
        self._sync_optimizer_refs()
        self._restore_staged_slow_buffers(staged_slow_buffers)


def _unwrap_lookahead_chain(optimizer: optim.Optimizer) -> tuple[list[LookaheadOptimizer], optim.Optimizer]:
    lookahead_wrappers: list[LookaheadOptimizer] = []
    base_optimizer = optimizer
    while getattr(base_optimizer, "is_lookahead_optimizer", False):
        lookahead_wrappers.append(base_optimizer)
        base_optimizer = base_optimizer.base_optimizer
    return lookahead_wrappers, base_optimizer


def reset_optimizer_state_for_positions(
    optimizer: Optional[optim.Optimizer],
    param: torch.Tensor,
    position_mask: Optional[torch.Tensor] = None,
) -> None:
    """Reset optimizer moments for a tensor, optionally only at selected sequence positions."""
    if optimizer is None:
        return

    lookahead_wrappers, base_optimizer = _unwrap_lookahead_chain(optimizer)

    if position_mask is None:
        for wrapper in lookahead_wrappers:
            wrapper.stage_slow_buffer(param)
        base_optimizer.state[param] = {}
        return

    state = base_optimizer.state.get(param)
    if not state:
        for wrapper in lookahead_wrappers:
            wrapper.stage_slow_buffer(param)
        base_optimizer.state[param] = {}
        return

    mask = position_mask.to(device=param.device, dtype=torch.bool).view(1, -1, *([1] * (param.dim() - 2)))
    mask = mask.expand_as(param)

    lookahead_keys = {wrapper.slow_buffer_key for wrapper in lookahead_wrappers}
    for wrapper in lookahead_wrappers:
        slow_param = state.get(wrapper.slow_buffer_key)
        if slow_param is None:
            if param in wrapper._staged_slow_buffers:
                slow_param = wrapper._staged_slow_buffers.pop(param).to(
                    device=param.device,
                    dtype=param.dtype,
                )
            else:
                slow_param = param.detach().clone()
            state[wrapper.slow_buffer_key] = slow_param
        slow_param[mask] = param.detach()[mask]

    for key, value in state.items():
        if key in lookahead_keys:
            continue
        if torch.is_tensor(value) and value.shape == param.shape:
            value[mask] = 0


def build_prompt_optimizer(
    parameters,
    learning_rate: float,
    lora_B: Optional[torch.Tensor] = None,
    logits_lora_b_learning_rate: Optional[float] = None,
    lookahead_k=0,
    lookahead_alpha: float = 0.5,
):
    """Build the prompt optimizer, optionally overriding LoRA matrix B's LR."""
    if lora_B is None or logits_lora_b_learning_rate is None:
        base_optimizer = optim.Adam(parameters, lr=learning_rate)
    else:
        shared_params = [param for param in parameters if param is not lora_B]
        param_groups = []
        if shared_params:
            param_groups.append({"params": shared_params, "lr": learning_rate})
        param_groups.append({"params": [lora_B], "lr": logits_lora_b_learning_rate})
        base_optimizer = optim.Adam(param_groups, lr=learning_rate)

    lookahead_levels = normalize_lookahead_levels(lookahead_k)
    for level_index, k in enumerate(lookahead_levels):
        base_optimizer = LookaheadOptimizer(
            base_optimizer,
            k=k,
            alpha=lookahead_alpha,
            level_index=level_index,
        )
    return base_optimizer


def build_lr_scheduler(
    optimizer,
    schedule: str,
    total_epochs: int,
    warmup_epochs: int = 0,
    lr_min: float = 0.0,
    step_size: int = 10,
    gamma: float = 0.1,
):
    """Build an epoch-based LR scheduler. Returns None when schedule='none'.

    Args:
        optimizer: The prompt optimizer returned by build_prompt_optimizer().
        schedule: One of 'none', 'cosine', 'linear', 'step', 'exponential'.
        total_epochs: Total number of training epochs.
        warmup_epochs: Epochs of linear warmup from ~0 to base LR before main schedule.
        lr_min: Minimum LR for cosine/linear schedules (eta_min / end LR).
        step_size: Period in epochs for StepLR (ignored by other schedules).
        gamma: Multiplicative decay factor for step/exponential schedules.
    """
    if schedule == "none":
        return None

    warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))
    main_epochs = max(1, total_epochs - warmup_epochs)

    if schedule == "cosine":
        main_sched = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_epochs, eta_min=lr_min
        )
    elif schedule == "linear":
        base_lr = optimizer.param_groups[0]["lr"]
        end_factor = max(lr_min / base_lr, 1e-8) if base_lr > 0 else 1e-8
        main_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=end_factor, total_iters=main_epochs
        )
    elif schedule == "step":
        main_sched = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif schedule == "exponential":
        main_sched = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError(
            f"Unknown lr_schedule: {schedule!r}. "
            "Choose from: none, cosine, linear, step, exponential"
        )

    if warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_epochs]
        )
    return main_sched
