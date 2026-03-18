from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from transformers import PretrainedConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_stgs_diff_model_class():
    package_name = "test_sdlm_pkg"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO_ROOT / "sdlm")]
    saved_package = sys.modules.get(package_name)
    saved_stgs = sys.modules.get(f"{package_name}.stgs")
    saved_diff = sys.modules.get(f"{package_name}.stgs_diff_model")
    try:
        sys.modules[package_name] = package

        stgs_spec = importlib.util.spec_from_file_location(
            f"{package_name}.stgs",
            REPO_ROOT / "sdlm" / "stgs.py",
        )
        if stgs_spec is None or stgs_spec.loader is None:
            raise RuntimeError("Could not load sdlm.stgs for test.")
        stgs_module = importlib.util.module_from_spec(stgs_spec)
        sys.modules[f"{package_name}.stgs"] = stgs_module
        stgs_spec.loader.exec_module(stgs_module)

        diff_spec = importlib.util.spec_from_file_location(
            f"{package_name}.stgs_diff_model",
            REPO_ROOT / "sdlm" / "stgs_diff_model.py",
        )
        if diff_spec is None or diff_spec.loader is None:
            raise RuntimeError("Could not load sdlm.stgs_diff_model for test.")
        diff_module = importlib.util.module_from_spec(diff_spec)
        sys.modules[f"{package_name}.stgs_diff_model"] = diff_module
        diff_spec.loader.exec_module(diff_module)
        return diff_module.STGSDiffModel
    finally:
        if saved_package is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = saved_package
        if saved_stgs is None:
            sys.modules.pop(f"{package_name}.stgs", None)
        else:
            sys.modules[f"{package_name}.stgs"] = saved_stgs
        if saved_diff is None:
            sys.modules.pop(f"{package_name}.stgs_diff_model", None)
        else:
            sys.modules[f"{package_name}.stgs_diff_model"] = saved_diff


STGSDiffModel = _load_stgs_diff_model_class()


class DummyTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __len__(self):
        return self.vocab_size


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_size: int = 8):
        super().__init__()
        self.config = PretrainedConfig()
        self.config.vocab_size = vocab_size
        self.config.hidden_size = hidden_size
        self.config.is_encoder_decoder = False
        self.embeddings = torch.nn.Embedding(vocab_size, hidden_size)

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            super().to(device)
        return self

    def get_input_embeddings(self):
        return self.embeddings

    def prepare_inputs_for_generation(self, input_ids=None, inputs_embeds=None, **kwargs):
        return {}


def _build_diff_model() -> STGSDiffModel:
    model = DummyModel()
    tokenizer = DummyTokenizer(model.config.vocab_size)
    return STGSDiffModel(
        model=model,
        tokenizer=tokenizer,
        stgs_kwargs={
            "hard": False,
            "temperature": 1.0,
            "learnable_temperature": False,
            "hidden_state_conditioning": False,
            "dropout": 0.0,
        },
    )


def test_hf_generate_backend_passes_kwargs_through(monkeypatch):
    diff_model = _build_diff_model()
    recorded = {}
    returned_sequences = torch.tensor([[7, 8, 9]])

    def fake_generate(**kwargs):
        recorded.update(kwargs)
        return types.SimpleNamespace(sequences=returned_sequences)

    monkeypatch.setattr(diff_model.model, "generate", fake_generate, raising=False)

    prefix_embeds = torch.randn(1, 3, diff_model.model.config.hidden_size)
    output = diff_model.generate(
        inputs_embeds=prefix_embeds,
        max_new_tokens=3,
        output_diff_one_hots=False,
        return_dict=True,
        generation_backend="hf_generate",
        do_sample=True,
        top_p=0.9,
        top_k=12,
    )

    assert torch.equal(output.sampled_diff_tokens, returned_sequences)
    assert output.base_generate_output.sequences is returned_sequences
    assert recorded["inputs_embeds"] is prefix_embeds
    assert "input_ids" not in recorded
    assert recorded["max_new_tokens"] == 3
    assert recorded["do_sample"] is True
    assert recorded["top_p"] == pytest.approx(0.9)
    assert recorded["top_k"] == 12
    assert recorded["return_dict_in_generate"] is True


def test_hf_generate_backend_strips_input_prefix(monkeypatch):
    diff_model = _build_diff_model()
    input_ids = torch.tensor([[2, 3]])
    returned_sequences = torch.tensor([[2, 3, 4, 5]])

    monkeypatch.setattr(
        diff_model.model,
        "generate",
        lambda **kwargs: types.SimpleNamespace(sequences=returned_sequences),
        raising=False,
    )

    output = diff_model.generate(
        input_ids=input_ids,
        max_new_tokens=2,
        output_diff_one_hots=False,
        return_dict=True,
        generation_backend="hf_generate",
    )

    assert torch.equal(output.sampled_diff_tokens, torch.tensor([[4, 5]]))


def test_hf_generate_backend_rejects_diff_only_outputs():
    diff_model = _build_diff_model()
    prefix_embeds = torch.randn(1, 3, diff_model.model.config.hidden_size)

    with pytest.raises(ValueError, match="tokens-only outputs"):
        diff_model.generate(
            inputs_embeds=prefix_embeds,
            return_dict=True,
            generation_backend="hf_generate",
            output_diff_one_hots=True,
        )


def test_hf_generate_backend_requires_inputs_embeds_support(monkeypatch):
    diff_model = _build_diff_model()
    prefix_embeds = torch.randn(1, 2, diff_model.model.config.hidden_size)

    monkeypatch.setattr(
        diff_model.model,
        "prepare_inputs_for_generation",
        lambda input_ids=None, **kwargs: {},
    )

    with pytest.raises(ValueError, match="support model.generate\\(inputs_embeds=...\\)"):
        diff_model.generate(
            inputs_embeds=prefix_embeds,
            output_diff_one_hots=False,
            return_dict=True,
            generation_backend="hf_generate",
        )
