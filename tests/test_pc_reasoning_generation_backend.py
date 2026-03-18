from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_pc_main():
    stub_modules = {
        "sdlm": types.ModuleType("sdlm"),
        "sdlm.stgs_diff_model": types.ModuleType("sdlm.stgs_diff_model"),
        "sdlm.utils": types.ModuleType("sdlm.utils"),
        "sdlm.utils.tqdm_utils": types.ModuleType("sdlm.utils.tqdm_utils"),
        "rewards": types.ModuleType("rewards"),
        "main": types.ModuleType("main"),
        "evaluation": types.ModuleType("evaluation"),
        "pc_weave_logging": types.ModuleType("pc_weave_logging"),
    }
    stub_modules["sdlm"].STGS = type("STGS", (), {})
    stub_modules["sdlm.stgs_diff_model"].STGSDiffModel = type("STGSDiffModel", (), {})
    stub_modules["rewards"].compare_answers = lambda *args, **kwargs: False

    main_module = stub_modules["main"]
    main_module.LossClass = type("LossClass", (), {})
    for name in (
        "build_fixed_logits_spec",
        "compute_ppo_kl_loss",
        "initialize_learnable_inputs",
        "initialize_lora_logits",
        "snap_free_logits",
        "snap_lora_logits",
        "compute_position_entropies",
        "compute_annealed_temperature",
    ):
        setattr(main_module, name, lambda *args, **kwargs: None)

    stub_modules["evaluation"].compute_embsim_probs = lambda *args, **kwargs: None
    stub_modules["sdlm.utils.tqdm_utils"].tqdm = lambda iterable=None, **kwargs: iterable

    weave_module = stub_modules["pc_weave_logging"]
    weave_module.build_lm_trace_payload = lambda *args, **kwargs: {}
    weave_module.build_trace_segment = lambda *args, **kwargs: {}
    weave_module.weave_lm_train_step = lambda *args, **kwargs: None
    weave_module.weave_lm_eval_step = lambda *args, **kwargs: None
    weave_module.weave_epoch_summary = lambda *args, **kwargs: None

    saved_modules = {}
    for name, module in stub_modules.items():
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    module_name = "pc_main_test_module"
    saved_target = sys.modules.get(module_name)
    try:
        module_path = REPO_ROOT / "benchmarking" / "invertibility" / "pc_main.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load spec for {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if saved_target is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = saved_target
        for name, old_module in saved_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def test_pc_main_rejects_hf_generate_with_teacher_forcing():
    pc_main = _load_pc_main()

    issues = pc_main.collect_pc_main_incompatibilities(
        {
            "teacher_forcing_r": True,
            "bptt": False,
            "reasoning_generation_backend": "hf_generate",
        }
    )

    assert any("requires --teacher_forcing_r=False" in issue for issue in issues)


def test_pc_main_rejects_hf_generate_with_bptt():
    pc_main = _load_pc_main()

    issues = pc_main.collect_pc_main_incompatibilities(
        {
            "teacher_forcing_r": False,
            "bptt": True,
            "reasoning_generation_backend": "hf_generate",
        }
    )

    assert any("not compatible with --bptt=True" in issue for issue in issues)


def test_pc_main_rejects_unknown_reasoning_backend():
    pc_main = _load_pc_main()

    issues = pc_main.collect_pc_main_incompatibilities(
        {
            "teacher_forcing_r": False,
            "bptt": False,
            "reasoning_generation_backend": "unknown_backend",
        }
    )

    assert any("--reasoning_generation_backend only supports" in issue for issue in issues)


class _DummyProgress:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def set_postfix(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return None

    def close(self):
        return None

    def set_description(self, *args, **kwargs):
        return None


def _load_pc_main_runtime():
    stub_modules = {
        "sdlm": types.ModuleType("sdlm"),
        "sdlm.stgs_diff_model": types.ModuleType("sdlm.stgs_diff_model"),
        "sdlm.utils": types.ModuleType("sdlm.utils"),
        "sdlm.utils.tqdm_utils": types.ModuleType("sdlm.utils.tqdm_utils"),
        "rewards": types.ModuleType("rewards"),
        "main": types.ModuleType("main"),
        "evaluation": types.ModuleType("evaluation"),
        "pc_weave_logging": types.ModuleType("pc_weave_logging"),
        "optimizer_utils": types.ModuleType("optimizer_utils"),
    }
    stub_modules["sdlm"].STGS = type("STGS", (), {})
    stub_modules["sdlm.stgs_diff_model"].STGSDiffModel = type("STGSDiffModel", (), {})
    stub_modules["rewards"].compare_answers = lambda extracted, target: extracted == target

    main_module = stub_modules["main"]
    main_module.LossClass = type("LossClass", (), {})
    main_module.build_fixed_logits_spec = lambda *args, **kwargs: (None, None, kwargs["seq_len"])
    main_module.compute_ppo_kl_loss = lambda *args, **kwargs: torch.tensor(0.0)
    main_module.initialize_learnable_inputs = (
        lambda vocab_size, n_free, device, **kwargs: torch.nn.Parameter(
            torch.zeros((1, n_free, vocab_size), device=device)
        )
    )
    main_module.initialize_lora_logits = lambda *args, **kwargs: None
    main_module.snap_free_logits = lambda *args, **kwargs: None
    main_module.snap_lora_logits = lambda *args, **kwargs: None
    main_module.compute_position_entropies = lambda *args, **kwargs: torch.zeros((1, 1))
    main_module.compute_annealed_temperature = lambda temperature, *args, **kwargs: temperature

    stub_modules["evaluation"].compute_embsim_probs = lambda *args, **kwargs: None
    stub_modules["optimizer_utils"].build_prompt_optimizer = lambda *args, **kwargs: None
    stub_modules["sdlm.utils.tqdm_utils"].tqdm = lambda *args, **kwargs: _DummyProgress(*args, **kwargs)

    weave_module = stub_modules["pc_weave_logging"]
    weave_module.build_lm_trace_payload = lambda *args, **kwargs: {}
    weave_module.build_trace_segment = lambda *args, **kwargs: {}
    weave_module.weave_lm_train_step = lambda *args, **kwargs: None
    weave_module.weave_lm_eval_step = lambda *args, **kwargs: None
    weave_module.weave_epoch_summary = lambda *args, **kwargs: None

    saved_modules = {}
    for name, module in stub_modules.items():
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    module_name = "pc_main_runtime_test_module"
    saved_target = sys.modules.get(module_name)
    try:
        module_path = REPO_ROOT / "benchmarking" / "invertibility" / "pc_main.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load spec for {module_name} from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if saved_target is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = saved_target
        for name, old_module in saved_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    chat_template = None

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        del add_special_tokens, return_tensors
        token = max(2, len(text) % 7 + 2)
        return types.SimpleNamespace(input_ids=torch.tensor([[token]], dtype=torch.long))

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return " ".join(str(token) for token in ids)


class _FakeModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 8, hidden_size: int = 4):
        super().__init__()
        self.config = types.SimpleNamespace(vocab_size=vocab_size, hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.generate_calls = []

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, input_ids, **kwargs):
        self.generate_calls.append({"input_ids": input_ids.clone(), **kwargs})
        max_new_tokens = kwargs["max_new_tokens"]
        suffix = torch.full((input_ids.shape[0], max_new_tokens), 6, dtype=torch.long)
        return torch.cat([input_ids, suffix], dim=1)


class _FakeDiffModel:
    def __init__(self, vocab_size: int = 8):
        self.calls = []
        self.stgs = types.SimpleNamespace(
            init_temperature=1.0,
            logits_normalize="none",
            stgs_hard=True,
            stgs_hard_method="categorical",
            stgs_hard_embsim_probs="gumbel_soft",
            stgs_hard_embsim_strategy="nearest",
            stgs_hard_embsim_top_k=8,
            stgs_hard_embsim_rerank_alpha=0.5,
            stgs_hard_embsim_sample_tau=1.0,
            stgs_hard_embsim_margin=0.0,
            stgs_hard_embsim_fallback="argmax",
            eps=1e-10,
            learnable_temperature=False,
            conditioning_dim=0,
        )
        self.vocab_size = vocab_size

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        max_new_tokens = kwargs["max_new_tokens"]
        token_id = 4 if len(self.calls) == 1 else 5
        sampled = torch.full((1, max_new_tokens), token_id, dtype=torch.long)
        sampled_one_hot = torch.nn.functional.one_hot(
            sampled,
            num_classes=self.vocab_size,
        ).to(torch.float32)
        return types.SimpleNamespace(
            sampled_diff_tokens=sampled,
            sampled_diff_one_hot=sampled_one_hot,
        )


class _FakeEvalSTGS:
    def __call__(self, logits, gumbel_noise_scale, embedding_weights):
        del gumbel_noise_scale
        token_ids = logits.argmax(dim=-1)
        one_hot = torch.nn.functional.one_hot(
            token_ids,
            num_classes=embedding_weights.shape[0],
        ).to(embedding_weights.dtype)
        return None, one_hot, None, None


def test_evaluate_shared_prompt_uses_reasoning_backend_for_soft_eval():
    pc_main = _load_pc_main_runtime()
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    diff_model = _FakeDiffModel(vocab_size=model.config.vocab_size)
    free_logits = torch.zeros((1, 2, model.config.vocab_size), dtype=torch.float32)
    free_logits[0, 0, 2] = 1.0
    free_logits[0, 1, 3] = 1.0
    embedding_weights = model.get_input_embeddings().weight.detach()
    E_input_ids = torch.tensor([[2]], dtype=torch.long)
    E_embeds = model.get_input_embeddings()(E_input_ids)

    pc_main.evaluate_shared_prompt(
        free_logits=free_logits,
        eval_pairs=[("question", "target")],
        model=model,
        diff_model=diff_model,
        stgs_module=_FakeEvalSTGS(),
        embedding_weights=embedding_weights,
        tokenizer=tokenizer,
        device="cpu",
        E_input_ids=E_input_ids,
        E_embeds=E_embeds,
        extraction_fns={"extractor": lambda text: text},
        max_new_tokens_reasoning=2,
        max_new_tokens_answer=1,
        eval_mode="soft",
        reasoning_generation_backend="hf_generate",
        reasoning_generate_kwargs={"top_p": 0.7, "do_sample": True},
    )

    assert len(diff_model.calls) == 2
    reasoning_call, answer_call = diff_model.calls
    assert reasoning_call["generation_backend"] == "hf_generate"
    assert reasoning_call["top_p"] == pytest.approx(0.7)
    assert reasoning_call["do_sample"] is True
    assert "inputs_embeds" in reasoning_call
    assert "generation_backend" not in answer_call


def test_pc_optimize_inputs_propagates_reasoning_backend_to_validation(monkeypatch):
    pc_main = _load_pc_main_runtime()

    class FakeTrainSTGS(torch.nn.Module):
        def __init__(self, vocab_size, init_temperature=1.0, eps=1e-10, device="cpu", **kwargs):
            super().__init__()
            del kwargs
            self.vocab_size = vocab_size
            self.init_temperature = init_temperature
            self.eps = eps
            self.device = device
            self.learnable_temperature = False

    class FakeLoss:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class FakeOptimizer:
        def zero_grad(self):
            return None

        def step(self):
            return None

    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    diff_model = _FakeDiffModel(vocab_size=model.config.vocab_size)
    captured_eval_kwargs = []

    monkeypatch.setattr(pc_main, "STGS", FakeTrainSTGS)
    monkeypatch.setattr(pc_main, "LossClass", FakeLoss)
    monkeypatch.setattr(pc_main, "build_prompt_optimizer", lambda *args, **kwargs: FakeOptimizer())
    monkeypatch.setattr(
        pc_main,
        "pc_forward_pass",
        lambda **kwargs: torch.zeros((), dtype=torch.float32, requires_grad=True),
    )
    monkeypatch.setattr(pc_main, "wandb", types.SimpleNamespace(run=None, log=lambda *args, **kwargs: None))
    monkeypatch.setattr(
        pc_main,
        "evaluate_shared_prompt",
        lambda **kwargs: captured_eval_kwargs.append(kwargs) or {"any_correct": 0.0},
    )

    pc_main.pc_optimize_inputs(
        model=model,
        diff_model=diff_model,
        tokenizer=tokenizer,
        device="cpu",
        model_precision="full",
        train_triples=[("train-x", "train-r", "train-y")],
        val_pairs=[("val-x", "val-y")],
        extraction_prompt="answer:",
        extraction_fns={"extractor": lambda text: text},
        teacher_forcing_r=False,
        reasoning_generation_backend="hf_generate",
        reasoning_generate_kwargs={"top_p": 0.6},
        seq_len=2,
        epochs=1,
        inner_batch_size=1,
        val_eval_every=1,
        val_prompt_eval_mode="soft",
    )

    assert len(captured_eval_kwargs) == 1
    assert captured_eval_kwargs[0]["reasoning_generation_backend"] == "hf_generate"
    assert captured_eval_kwargs[0]["reasoning_generate_kwargs"] == {"top_p": 0.6}
