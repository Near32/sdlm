"""
Weave (W&B Traces) logging for the PC multi-target pipeline.

The PC path can hit multiple LM boundaries per sample: teacher-forced forwards,
reasoning rollouts, and answer generation. This module provides helpers that
log the most exact textual representation available for each LM call. When the
input is driven by token ids, the full rendered LM input is decoded exactly.
When the input is driven by embeddings or soft prompts, the payload marks the
trace as approximate and records the best available textual proxy per segment.
"""

import logging
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
_CALL_LINK_LOGGER_NAME = "weave.trace.weave_client"
_CALL_LINK_HANDLER_ATTR = "_pc_weave_call_link_handler"


def _configure_call_link_logger(call_link_log_path: str | PathLike[str]) -> None:
    """Route Weave call-link INFO records to a file instead of the terminal."""
    log_path = Path(call_link_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    call_link_logger = logging.getLogger(_CALL_LINK_LOGGER_NAME)
    existing_handler = getattr(call_link_logger, _CALL_LINK_HANDLER_ATTR, None)
    target_path = str(log_path.resolve())

    if existing_handler is not None:
        current_path = getattr(existing_handler, "baseFilename", None)
        if current_path == target_path:
            call_link_logger.propagate = False
            call_link_logger.setLevel(logging.INFO)
            return
        call_link_logger.removeHandler(existing_handler)
        existing_handler.close()

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    call_link_logger.addHandler(file_handler)
    call_link_logger.setLevel(logging.INFO)
    call_link_logger.propagate = False
    setattr(call_link_logger, _CALL_LINK_HANDLER_ATTR, file_handler)


def init_weave(
    project: str,
    entity: Optional[str] = None,
    call_link_log_path: Optional[str | PathLike[str]] = None,
) -> bool:
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
        if call_link_log_path is not None:
            _configure_call_link_logger(call_link_log_path)
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
# Trace payload helpers
# ---------------------------------------------------------------------------

def _flatten_token_ids(token_ids: Optional[Any]) -> Optional[List[int]]:
    if token_ids is None:
        return None
    if hasattr(token_ids, "detach"):
        token_ids = token_ids.detach().cpu().tolist()
    elif hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    flattened: List[int] = []

    def _visit(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                _visit(item)
            return
        flattened.append(int(value))

    _visit(token_ids)
    return flattened


def _decode_tokens(tokenizer, token_ids: Optional[Any], *, skip_special_tokens: bool) -> Optional[str]:
    flat_ids = _flatten_token_ids(token_ids)
    if tokenizer is None or flat_ids is None:
        return None
    return tokenizer.decode(flat_ids, skip_special_tokens=skip_special_tokens)


def build_trace_segment(
    *,
    label: str,
    text: str,
    token_ids: Optional[Any] = None,
    is_exact: bool,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    segment = {
        "label": label,
        "text": text,
        "is_exact": bool(is_exact),
    }
    flat_ids = _flatten_token_ids(token_ids)
    if flat_ids is not None:
        segment["token_ids"] = flat_ids
    if source:
        segment["source"] = source
    return segment


def build_lm_trace_payload(
    *,
    tokenizer,
    call_name: str,
    split: str,
    epoch: int,
    lm_name: str,
    input_mode: str,
    output_mode: str,
    input_segments: Optional[Sequence[Dict[str, Any]]] = None,
    output_segments: Optional[Sequence[Dict[str, Any]]] = None,
    input_token_ids: Optional[Any] = None,
    output_token_ids: Optional[Any] = None,
    input_text_approx: Optional[str] = None,
    output_text_approx: Optional[str] = None,
    output_exact_available: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    input_segments_list = list(input_segments or [])
    output_segments_list = list(output_segments or [])
    input_token_ids_list = _flatten_token_ids(input_token_ids)
    output_token_ids_list = _flatten_token_ids(output_token_ids)

    input_text_exact = _decode_tokens(
        tokenizer,
        input_token_ids_list,
        skip_special_tokens=False,
    )
    input_text_clean = _decode_tokens(
        tokenizer,
        input_token_ids_list,
        skip_special_tokens=True,
    )
    if input_text_approx is None:
        input_text_approx = "".join(
            segment.get("text", "")
            for segment in input_segments_list
            if segment.get("text")
        )

    output_text_exact = None
    output_text_clean = None
    if output_exact_available:
        output_text_exact = _decode_tokens(
            tokenizer,
            output_token_ids_list,
            skip_special_tokens=False,
        )
        output_text_clean = _decode_tokens(
            tokenizer,
            output_token_ids_list,
            skip_special_tokens=True,
        )
    elif output_text_approx is None:
        output_text_approx = "".join(
            segment.get("text", "")
            for segment in output_segments_list
            if segment.get("text")
        )

    payload = {
        "call_name": call_name,
        "split": split,
        "epoch": epoch,
        "lm_name": lm_name,
        "input_mode": input_mode,
        "output_mode": output_mode,
        "input_exact_available": input_text_exact is not None,
        "output_exact_available": bool(output_exact_available),
        "input_text": input_text_exact if input_text_exact is not None else input_text_approx,
        "input_text_exact": input_text_exact,
        "input_text_approx": None if input_text_exact is not None else input_text_approx,
        "input_text_clean": input_text_clean if input_text_clean is not None else input_text_approx,
        "input_token_ids": input_token_ids_list,
        "input_segments": input_segments_list,
        "output_text": output_text_exact if output_text_exact is not None else output_text_approx,
        "output_text_exact": output_text_exact,
        "output_text_approx": None if output_text_exact is not None else output_text_approx,
        "output_text_clean": output_text_clean if output_text_clean is not None else output_text_approx,
        "output_token_ids": output_token_ids_list,
        "output_segments": output_segments_list,
    }
    if metadata:
        payload.update(metadata)
    return payload


# ---------------------------------------------------------------------------
# Traced ops
# ---------------------------------------------------------------------------

@_op(name="pc/lm_call")
def weave_lm_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Trace one LM boundary payload."""
    return payload


def weave_lm_train_step(**payload: Any) -> Dict[str, Any]:
    """Compatibility wrapper for training LM call traces."""
    return weave_lm_call(payload)


def weave_lm_eval_step(**payload: Any) -> Dict[str, Any]:
    """Compatibility wrapper for eval LM call traces."""
    return weave_lm_call(payload)


@_op(name="pc/prompt_init")
def weave_prompt_init(
    requested_text: str,
    requested_token_ids: List[int],
    reconstructed_text: str,
    reconstructed_token_ids: List[int],
    reconstruction_error: float,
    reconstruction_method: str,
    lora_rank: int,
    seq_len: int,
    is_exact: bool,
    embsim_l2_text: str,
    embsim_l2_token_ids: List[int],
    embsim_cos_text: str,
    embsim_cos_token_ids: List[int],
) -> Dict[str, Any]:
    """
    Trace the initial prompt text vs its LoRA rank-r SVD approximation.

    Called once at optimization start when initial_prompt_text is set and
    logits_lora_rank > 0.  Records the intended text, the primary reconstructed
    text (selected by reconstruction_method), and all three decode methods
    (argmax, embsim-l2, embsim-cos) for comparison.

    Args:
        reconstruction_method: The decode method used as the primary reconstruction
            ('argmax', 'embsim-l2', or 'embsim-cos'); drives reconstructed_text /
            reconstructed_token_ids and the is_exact flag.
    """
    return {
        "requested_text": requested_text,
        "requested_token_ids": requested_token_ids,
        "reconstructed_text": reconstructed_text,
        "reconstructed_token_ids": reconstructed_token_ids,
        "reconstruction_error": reconstruction_error,
        "reconstruction_method": reconstruction_method,
        "lora_rank": lora_rank,
        "seq_len": seq_len,
        "is_exact": is_exact,
        "embsim_l2_text": embsim_l2_text,
        "embsim_l2_token_ids": embsim_l2_token_ids,
        "embsim_cos_text": embsim_cos_text,
        "embsim_cos_token_ids": embsim_cos_token_ids,
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
