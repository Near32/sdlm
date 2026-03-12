"""
Weave (W&B Traces) logging for the PC multi-target pipeline.

Provides three @weave.op()-traced functions that capture text-level inputs and
outputs from each LM call during training and evaluation:

  weave_lm_train_step  — one training sample forward pass
  weave_lm_eval_step   — one eval/test sample inference
  weave_epoch_summary  — epoch-level aggregates

Each function is a pure Python function that receives already-decoded text
representations; the actual tensor computation happens *before* these calls.
If weave is not installed, or init_weave() was never called, all functions
fall back to no-ops.

Usage (in batch_optimize_pc_main.py):
    from pc_weave_logging import init_weave
    init_weave(project="sdlm-pc-mt-lmi", entity=None)

Usage (in pc_main.py):
    from pc_weave_logging import weave_lm_train_step, weave_lm_eval_step, weave_epoch_summary
    weave_lm_train_step(epoch=epoch, x_text=x_str, ...)
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("pc_weave_logging")

# ---------------------------------------------------------------------------
# Conditional weave import + decorator factory
# ---------------------------------------------------------------------------

try:
    import weave as _weave
    _WEAVE_AVAILABLE = True
except ImportError:
    _weave = None
    _WEAVE_AVAILABLE = False
    logger.debug("weave not installed; Weave tracing disabled.")


def _op(name: str):
    """Return @weave.op(name=name) if weave is available, else identity."""
    if _WEAVE_AVAILABLE:
        return _weave.op(name=name)
    return lambda f: f


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

_weave_initialized = False


def init_weave(project: str, entity: Optional[str] = None) -> bool:
    """
    Initialize Weave for the given W&B project.

    Args:
        project: W&B project name (same as --wandb_project).
        entity:  W&B entity/team (same as --wandb_entity). Optional.

    Returns:
        True if Weave was successfully initialized, False otherwise.
    """
    global _weave_initialized
    if not _WEAVE_AVAILABLE:
        logger.warning("weave not installed; skipping Weave init.")
        return False
    try:
        project_name = f"{entity}/{project}" if entity else project
        _weave.init(project_name)
        _weave_initialized = True
        logger.info(f"Weave initialized: project={project_name!r}")
        return True
    except Exception as exc:
        logger.warning(f"Weave init failed: {exc}")
        return False


def is_weave_active() -> bool:
    """True if Weave was successfully initialized."""
    return _weave_initialized


# ---------------------------------------------------------------------------
# Traced ops
# ---------------------------------------------------------------------------

@_op(name="pc/lm_train_step")
def weave_lm_train_step(
    epoch: int,
    x_text: str,
    p_text: str,
    R_gt_text: str,          # ground-truth CoT; "" when not using TF
    R_gen_text: str,         # model-generated R; "" when using TF
    E_text: str,             # extraction prompt
    y_target: str,           # ground-truth answer string
    y_pred: str,             # argmax of y_logits, decoded
    loss: float,
    teacher_forcing_r: bool,
) -> Dict[str, Any]:
    """
    Weave trace entry for one training sample's LM call.

    The 'input' key holds a human-readable rendering of the full LM context
    ([x], [p], [R], [E]).  'output' holds the model's Y prediction vs target.
    """
    if teacher_forcing_r:
        R_used, R_label = R_gt_text, "R_gt"
    else:
        R_used, R_label = R_gen_text, "R_gen"

    input_repr = f"[x] {x_text}\n[p] {p_text}"
    if R_used:
        input_repr += f"\n[{R_label}] {R_used}"
    input_repr += f"\n[E] {E_text}"

    return {
        "input": input_repr,
        "y_pred": y_pred,
        "y_target": y_target,
        "loss": loss,
    }


@_op(name="pc/lm_eval_step")
def weave_lm_eval_step(
    split: str,              # "val" or "test"
    epoch: int,
    x_text: str,
    p_text: str,
    E_text: str,
    R_text: str,             # generated reasoning
    Y_text: str,             # generated answer (from extractor path)
    y_target: str,
    correct_per_method: Dict[str, bool],
    any_correct: bool,
    eval_mode: str,          # "discrete" or "soft"
) -> Dict[str, Any]:
    """
    Weave trace entry for one eval/test sample.

    Records the full input context, both generated texts, and per-method
    extraction correctness.
    """
    return {
        "input": f"[x] {x_text}\n[p] {p_text}",
        "R_text": R_text,
        "Y_text": Y_text,
        "y_target": y_target,
        "any_correct": any_correct,
        "eval_mode": eval_mode,
        **{f"correct/{m}": v for m, v in correct_per_method.items()},
    }


@_op(name="pc/epoch_summary")
def weave_epoch_summary(
    epoch: int,
    loss: float,
    p_text: str,
    temperature: float,
    n_samples: int,
) -> Dict[str, Any]:
    """
    Weave trace entry summarising one full training epoch.
    """
    return {
        "epoch": epoch,
        "loss": loss,
        "p_text": p_text,
        "temperature": temperature,
        "n_samples": n_samples,
    }
