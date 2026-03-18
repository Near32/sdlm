from __future__ import annotations

import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INVERTIBILITY_DIR = REPO_ROOT / "benchmarking" / "invertibility"
if str(INVERTIBILITY_DIR) not in sys.path:
    sys.path.insert(0, str(INVERTIBILITY_DIR))

import pc_weave_logging


class _StubWeave:
    def __init__(self):
        self.init_calls: list[str] = []

    def init(self, project_name: str) -> None:
        self.init_calls.append(project_name)


def _reset_call_link_logger() -> logging.Logger:
    logger = logging.getLogger(pc_weave_logging._CALL_LINK_LOGGER_NAME)
    managed_handler = getattr(logger, pc_weave_logging._CALL_LINK_HANDLER_ATTR, None)
    if managed_handler is not None:
        logger.removeHandler(managed_handler)
        managed_handler.close()
        delattr(logger, pc_weave_logging._CALL_LINK_HANDLER_ATTR)
    logger.propagate = True
    logger.setLevel(logging.NOTSET)
    return logger


def test_init_weave_routes_call_links_to_file(tmp_path, monkeypatch):
    stub_weave = _StubWeave()
    monkeypatch.setattr(pc_weave_logging, "_WEAVE_AVAILABLE", True)
    monkeypatch.setattr(pc_weave_logging, "_weave", stub_weave)
    pc_weave_logging._weave_initialized = False

    call_link_logger = _reset_call_link_logger()
    log_path = tmp_path / "weave_call_links.log"

    try:
        assert pc_weave_logging.init_weave(
            project="proj",
            entity="team",
            call_link_log_path=log_path,
        )
        assert stub_weave.init_calls == ["team/proj"]
        assert call_link_logger.propagate is False

        file_handlers = [
            handler
            for handler in call_link_logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert Path(file_handlers[0].baseFilename) == log_path.resolve()

        call_link_logger.info("🍩 https://wandb.ai/example/project/r/call/123")
        file_handlers[0].flush()
        assert "https://wandb.ai/example/project/r/call/123" in log_path.read_text()
    finally:
        _reset_call_link_logger()


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        flat = []
        for item in ids:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        if skip_special_tokens:
            flat = [item for item in flat if item >= 0]
        return "|".join(str(item) for item in flat)


def test_build_lm_trace_payload_prefers_exact_token_rendering():
    tokenizer = _FakeTokenizer()
    payload = pc_weave_logging.build_lm_trace_payload(
        tokenizer=tokenizer,
        call_name="answer_generate",
        split="test",
        epoch=3,
        lm_name="base_model.generate",
        input_mode="input_ids",
        output_mode="generated_tokens",
        input_segments=[
            pc_weave_logging.build_trace_segment(
                label="x_rendered",
                text="x",
                token_ids=[1, 2],
                is_exact=True,
                source="chat_template",
            )
        ],
        output_segments=[
            pc_weave_logging.build_trace_segment(
                label="Y_gen",
                text="y",
                token_ids=[5, 6],
                is_exact=True,
                source="generated_answer_tokens",
            )
        ],
        input_token_ids=[1, 2, 3],
        output_token_ids=[5, 6],
        output_exact_available=True,
        metadata={"extra": "value"},
    )

    assert payload["input_exact_available"] is True
    assert payload["output_exact_available"] is True
    assert payload["input_text"] == "1|2|3"
    assert payload["input_text_exact"] == "1|2|3"
    assert payload["output_text"] == "5|6"
    assert payload["output_text_exact"] == "5|6"
    assert payload["extra"] == "value"


def test_build_lm_trace_payload_marks_embedding_inputs_as_approximate():
    tokenizer = _FakeTokenizer()
    payload = pc_weave_logging.build_lm_trace_payload(
        tokenizer=tokenizer,
        call_name="teacher_forced_answer_forward",
        split="train",
        epoch=1,
        lm_name="diff_model.forward",
        input_mode="inputs_embeds",
        output_mode="normal_logits",
        input_segments=[
            pc_weave_logging.build_trace_segment(
                label="x_rendered",
                text="10|11",
                token_ids=[10, 11],
                is_exact=True,
                source="plain_tokenization",
            ),
            pc_weave_logging.build_trace_segment(
                label="p_argmax",
                text="12|13",
                token_ids=[12, 13],
                is_exact=False,
                source="argmax_from_stgs_prompt_distribution",
            ),
        ],
        output_segments=[
            pc_weave_logging.build_trace_segment(
                label="Y_argmax",
                text="14|15",
                token_ids=[14, 15],
                is_exact=False,
                source="argmax_from_output_logits",
            )
        ],
        output_token_ids=[14, 15],
        output_exact_available=False,
    )

    assert payload["input_exact_available"] is False
    assert payload["input_text"] == "10|1112|13"
    assert payload["input_text_exact"] is None
    assert payload["input_text_approx"] == "10|1112|13"
    assert payload["output_exact_available"] is False
    assert payload["output_text"] == "14|15"
    assert payload["output_text_exact"] is None


def test_init_weave_keeps_single_managed_handler_for_same_file(tmp_path, monkeypatch):
    stub_weave = _StubWeave()
    monkeypatch.setattr(pc_weave_logging, "_WEAVE_AVAILABLE", True)
    monkeypatch.setattr(pc_weave_logging, "_weave", stub_weave)
    pc_weave_logging._weave_initialized = False

    call_link_logger = _reset_call_link_logger()
    log_path = tmp_path / "weave_call_links.log"

    try:
        assert pc_weave_logging.init_weave(project="proj", call_link_log_path=log_path)
        first_handler = getattr(call_link_logger, pc_weave_logging._CALL_LINK_HANDLER_ATTR)

        assert pc_weave_logging.init_weave(project="proj", call_link_log_path=log_path)
        second_handler = getattr(call_link_logger, pc_weave_logging._CALL_LINK_HANDLER_ATTR)

        assert first_handler is second_handler
        file_handlers = [
            handler
            for handler in call_link_logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        assert len(file_handlers) == 1

        call_link_logger.info("🍩 https://wandb.ai/example/project/r/call/456")
        second_handler.flush()
        log_text = log_path.read_text()
        assert log_text.count("https://wandb.ai/example/project/r/call/456") == 1
    finally:
        _reset_call_link_logger()
