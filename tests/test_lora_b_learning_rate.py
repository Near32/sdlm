from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
INVERTIBILITY_DIR = REPO_ROOT / "benchmarking" / "invertibility"
if str(INVERTIBILITY_DIR) not in sys.path:
    sys.path.insert(0, str(INVERTIBILITY_DIR))

import optimizer_utils


def _group_param_ids(param_group) -> set[int]:
    return {id(param) for param in param_group["params"]}


def _load_module(module_name: str, relative_path: str, stub_modules: dict[str, types.ModuleType]):
    saved_modules: dict[str, types.ModuleType | None] = {}
    for name, module in stub_modules.items():
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    saved_target = sys.modules.get(module_name)
    try:
        module_path = REPO_ROOT / relative_path
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


def _stub_batch_optimize_main_dependencies() -> dict[str, types.ModuleType]:
    metrics_registry = types.ModuleType("metrics_registry")
    metrics_registry.compute_all_metrics = lambda *args, **kwargs: {}

    metrics_aggregator = types.ModuleType("metrics_aggregator")
    metrics_aggregator.MetricsAggregator = type("MetricsAggregator", (), {})

    metrics_logging = types.ModuleType("metrics_logging")
    metrics_logging.MetricsLogger = type("MetricsLogger", (), {})

    evaluation_utils = types.ModuleType("evaluation_utils")
    evaluation_utils.evaluate_generated_output = lambda *args, **kwargs: {}
    evaluation_utils.evaluate_prompt_reconstruction = lambda *args, **kwargs: {}

    main_stub = types.ModuleType("main")
    main_stub.optimize_inputs = lambda *args, **kwargs: {}

    return {
        "metrics_registry": metrics_registry,
        "metrics_aggregator": metrics_aggregator,
        "metrics_logging": metrics_logging,
        "evaluation_utils": evaluation_utils,
        "main": main_stub,
    }


def _stub_batch_optimize_pc_dependencies() -> dict[str, types.ModuleType]:
    batch_optimize_main = types.ModuleType("batch_optimize_main")
    batch_optimize_main.setup_model_and_tokenizer = lambda *args, **kwargs: (None, None)

    sdlm = types.ModuleType("sdlm")
    sdlm_utils = types.ModuleType("sdlm.utils")
    sdlm_utils_tqdm = types.ModuleType("sdlm.utils.tqdm_utils")
    sdlm.STGS = type("STGS", (), {})
    sdlm.utils = sdlm_utils
    sdlm_utils.tqdm_utils = sdlm_utils_tqdm
    sdlm_utils_tqdm.tqdm = lambda iterable=None, **kwargs: iterable

    sdlm_stgs_diff_model = types.ModuleType("sdlm.stgs_diff_model")
    sdlm_stgs_diff_model.STGSDiffModel = type("STGSDiffModel", (), {})

    pc_main = types.ModuleType("pc_main")
    pc_main.ensure_pc_main_compatibility = lambda *args, **kwargs: None
    pc_main.pc_optimize_inputs = lambda *args, **kwargs: {}
    pc_main.evaluate_shared_prompt = lambda *args, **kwargs: {}
    pc_main.evaluate_shared_prompt_batched = lambda *args, **kwargs: {}

    pc_weave_logging = types.ModuleType("pc_weave_logging")
    pc_weave_logging.init_weave = lambda *args, **kwargs: None

    product_of_speaker = types.ModuleType("product_of_speaker")
    product_of_speaker.build_pc_product_of_speaker_transform = lambda *args, **kwargs: None

    rewards = types.ModuleType("rewards")
    rewards.extract_gsm8k_answer = lambda *args, **kwargs: None
    rewards.extract_boxed_answer = lambda *args, **kwargs: None
    rewards.extract_final_number = lambda *args, **kwargs: None
    rewards.compare_answers = lambda *args, **kwargs: None

    return {
        "batch_optimize_main": batch_optimize_main,
        "sdlm": sdlm,
        "sdlm.utils": sdlm_utils,
        "sdlm.utils.tqdm_utils": sdlm_utils_tqdm,
        "sdlm.stgs_diff_model": sdlm_stgs_diff_model,
        "pc_main": pc_main,
        "pc_weave_logging": pc_weave_logging,
        "product_of_speaker": product_of_speaker,
        "rewards": rewards,
    }


def test_build_prompt_optimizer_keeps_single_group_without_override():
    lora_a = torch.nn.Parameter(torch.randn(1, 2, 3))
    lora_b = torch.nn.Parameter(torch.randn(1, 3, 4))
    extra = torch.nn.Parameter(torch.randn(1))

    optimizer = optimizer_utils.build_prompt_optimizer(
        [lora_a, lora_b, extra],
        learning_rate=0.1,
        lora_B=lora_b,
        logits_lora_b_learning_rate=None,
    )

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert _group_param_ids(optimizer.param_groups[0]) == {id(lora_a), id(lora_b), id(extra)}


def test_build_prompt_optimizer_overrides_only_lora_b_lr():
    lora_a = torch.nn.Parameter(torch.randn(1, 2, 3))
    lora_b = torch.nn.Parameter(torch.randn(1, 3, 4))
    extra = torch.nn.Parameter(torch.randn(1))

    optimizer = optimizer_utils.build_prompt_optimizer(
        [lora_a, lora_b, extra],
        learning_rate=0.1,
        lora_B=lora_b,
        logits_lora_b_learning_rate=0.025,
    )

    assert len(optimizer.param_groups) == 2

    shared_group, lora_b_group = optimizer.param_groups
    assert shared_group["lr"] == pytest.approx(0.1)
    assert _group_param_ids(shared_group) == {id(lora_a), id(extra)}
    assert lora_b_group["lr"] == pytest.approx(0.025)
    assert _group_param_ids(lora_b_group) == {id(lora_b)}


def test_build_prompt_optimizer_wraps_adam_with_lookahead():
    lora_a = torch.nn.Parameter(torch.randn(1, 2, 3))
    lora_b = torch.nn.Parameter(torch.randn(1, 3, 4))
    extra = torch.nn.Parameter(torch.randn(1))

    optimizer = optimizer_utils.build_prompt_optimizer(
        [lora_a, lora_b, extra],
        learning_rate=0.1,
        lora_B=lora_b,
        logits_lora_b_learning_rate=0.025,
        lookahead_k=4,
        lookahead_alpha=0.3,
    )

    assert isinstance(optimizer, optimizer_utils.LookaheadOptimizer)
    assert optimizer.k == 4
    assert optimizer.alpha == pytest.approx(0.3)
    assert len(optimizer.param_groups) == 2


def test_normalize_lookahead_levels_accepts_scalar_and_nested_inputs():
    assert optimizer_utils.normalize_lookahead_levels(0) == []
    assert optimizer_utils.normalize_lookahead_levels("0") == []
    assert optimizer_utils.normalize_lookahead_levels("8") == [8]
    assert optimizer_utils.normalize_lookahead_levels("8,80,800") == [8, 80, 800]
    assert optimizer_utils.normalize_lookahead_levels([8, "80", 800]) == [8, 80, 800]


@pytest.mark.parametrize(
    "lookahead_k",
    ["1", "-2", "8,0", "8,8", "80,8", "8,,80", object()],
)
def test_normalize_lookahead_levels_rejects_invalid_inputs(lookahead_k):
    with pytest.raises(ValueError):
        optimizer_utils.normalize_lookahead_levels(lookahead_k)


def test_build_prompt_optimizer_wraps_adam_with_nested_lookahead():
    param = torch.nn.Parameter(torch.randn(1, 2, 3))

    optimizer = optimizer_utils.build_prompt_optimizer(
        [param],
        learning_rate=0.1,
        lookahead_k=[4, 16],
        lookahead_alpha=0.3,
    )

    assert isinstance(optimizer, optimizer_utils.LookaheadOptimizer)
    assert optimizer.k == 16
    assert optimizer.alpha == pytest.approx(0.3)
    assert optimizer.slow_buffer_key == optimizer_utils.get_lookahead_slow_buffer_key(1)

    inner_optimizer = optimizer.base_optimizer
    assert isinstance(inner_optimizer, optimizer_utils.LookaheadOptimizer)
    assert inner_optimizer.k == 4
    assert inner_optimizer.alpha == pytest.approx(0.3)
    assert inner_optimizer.slow_buffer_key == optimizer_utils.LOOKAHEAD_SLOW_BUFFER


def test_lookahead_optimizer_syncs_every_k_steps():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    base_optimizer = torch.optim.SGD([param], lr=1.0)
    optimizer = optimizer_utils.LookaheadOptimizer(base_optimizer, k=2, alpha=0.5)

    param.grad = torch.ones_like(param)
    optimizer.step()
    assert param.item() == pytest.approx(0.0)

    param.grad = torch.ones_like(param)
    optimizer.step()
    assert param.item() == pytest.approx(0.0)

    param.grad = torch.ones_like(param)
    optimizer.step()
    assert param.item() == pytest.approx(-1.0)


def test_nested_lookahead_optimizer_syncs_each_level_on_its_own_schedule():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    inner_optimizer = optimizer_utils.LookaheadOptimizer(
        torch.optim.SGD([param], lr=1.0),
        k=2,
        alpha=0.5,
        level_index=0,
    )
    optimizer = optimizer_utils.LookaheadOptimizer(
        inner_optimizer,
        k=4,
        alpha=0.5,
        level_index=1,
    )

    for _ in range(4):
        param.grad = torch.ones_like(param)
        optimizer.step()

    assert param.item() == pytest.approx(0.0)
    assert optimizer.state[param][optimizer_utils.LOOKAHEAD_SLOW_BUFFER].item() == pytest.approx(-1.0)
    assert optimizer.state[param][optimizer_utils.get_lookahead_slow_buffer_key(1)].item() == pytest.approx(0.0)


def test_lookahead_optimizer_state_dict_roundtrip_preserves_state():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = optimizer_utils.LookaheadOptimizer(torch.optim.SGD([param], lr=1.0), k=2, alpha=0.25)

    for _ in range(2):
        param.grad = torch.ones_like(param)
        optimizer.step()

    state_dict = optimizer.state_dict()

    restored_param = torch.nn.Parameter(param.detach().clone())
    restored_optimizer = optimizer_utils.LookaheadOptimizer(
        torch.optim.SGD([restored_param], lr=1.0),
        k=5,
        alpha=0.9,
    )
    restored_optimizer.load_state_dict(state_dict)

    assert restored_optimizer.k == 2
    assert restored_optimizer.alpha == pytest.approx(0.25)
    assert restored_optimizer._lookahead_step == 2
    assert restored_optimizer.state[restored_param][optimizer_utils.LOOKAHEAD_SLOW_BUFFER].item() == pytest.approx(
        optimizer.state[param][optimizer_utils.LOOKAHEAD_SLOW_BUFFER].item()
    )

    param.grad = torch.ones_like(param)
    restored_param.grad = torch.ones_like(restored_param)
    optimizer.step()
    restored_optimizer.step()

    assert restored_param.item() == pytest.approx(param.item())


def test_nested_lookahead_state_dict_roundtrip_preserves_all_levels():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = optimizer_utils.LookaheadOptimizer(
        optimizer_utils.LookaheadOptimizer(
            torch.optim.SGD([param], lr=1.0),
            k=2,
            alpha=0.25,
            level_index=0,
        ),
        k=4,
        alpha=0.25,
        level_index=1,
    )

    for _ in range(4):
        param.grad = torch.ones_like(param)
        optimizer.step()

    state_dict = optimizer.state_dict()

    restored_param = torch.nn.Parameter(param.detach().clone())
    restored_optimizer = optimizer_utils.LookaheadOptimizer(
        optimizer_utils.LookaheadOptimizer(
            torch.optim.SGD([restored_param], lr=1.0),
            k=10,
            alpha=0.9,
            level_index=0,
        ),
        k=20,
        alpha=0.9,
        level_index=1,
    )
    restored_optimizer.load_state_dict(state_dict)

    assert restored_optimizer.k == 4
    assert restored_optimizer.base_optimizer.k == 2
    assert restored_optimizer._lookahead_step == 4
    assert restored_optimizer.base_optimizer._lookahead_step == 4

    param.grad = torch.ones_like(param)
    restored_param.grad = torch.ones_like(restored_param)
    optimizer.step()
    restored_optimizer.step()

    assert restored_param.item() == pytest.approx(param.item())
    assert restored_optimizer.state[restored_param][optimizer_utils.LOOKAHEAD_SLOW_BUFFER].item() == pytest.approx(
        optimizer.state[param][optimizer_utils.LOOKAHEAD_SLOW_BUFFER].item()
    )
    assert restored_optimizer.state[restored_param][optimizer_utils.get_lookahead_slow_buffer_key(1)].item() == pytest.approx(
        optimizer.state[param][optimizer_utils.get_lookahead_slow_buffer_key(1)].item()
    )


def test_reset_optimizer_state_for_positions_resyncs_lookahead_buffer():
    param = torch.nn.Parameter(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
    optimizer = optimizer_utils.build_prompt_optimizer(
        [param],
        learning_rate=0.1,
        lookahead_k=2,
        lookahead_alpha=0.5,
    )

    for _ in range(2):
        param.grad = torch.ones_like(param)
        optimizer.step()

    state = optimizer.state[param]
    exp_avg_before = state["exp_avg"].clone()
    slow_before = state[optimizer_utils.LOOKAHEAD_SLOW_BUFFER].clone()

    with torch.no_grad():
        param[:, 0, :].fill_(42.0)

    optimizer_utils.reset_optimizer_state_for_positions(optimizer, param, torch.tensor([True, False]))

    assert torch.count_nonzero(state["exp_avg"][:, 0, :]) == 0
    assert torch.equal(state["exp_avg"][:, 1, :], exp_avg_before[:, 1, :])
    assert torch.equal(state[optimizer_utils.LOOKAHEAD_SLOW_BUFFER][:, 0, :], param.detach()[:, 0, :])
    assert torch.equal(state[optimizer_utils.LOOKAHEAD_SLOW_BUFFER][:, 1, :], slow_before[:, 1, :])

    optimizer_utils.reset_optimizer_state_for_positions(optimizer, param)

    assert optimizer.state[param] == {}
    staged_snapshot = param.detach().clone()

    param.grad = torch.ones_like(param)
    optimizer.step()

    assert "exp_avg" in optimizer.state[param]
    assert optimizer_utils.LOOKAHEAD_SLOW_BUFFER in optimizer.state[param]
    assert torch.equal(
        optimizer.state[param][optimizer_utils.LOOKAHEAD_SLOW_BUFFER],
        staged_snapshot,
    )


def test_reset_optimizer_state_for_positions_resyncs_nested_lookahead_buffers():
    param = torch.nn.Parameter(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
    optimizer = optimizer_utils.build_prompt_optimizer(
        [param],
        learning_rate=0.1,
        lookahead_k=[2, 4],
        lookahead_alpha=0.5,
    )

    for _ in range(4):
        param.grad = torch.ones_like(param)
        optimizer.step()

    state = optimizer.state[param]
    exp_avg_before = state["exp_avg"].clone()
    inner_key = optimizer_utils.LOOKAHEAD_SLOW_BUFFER
    outer_key = optimizer_utils.get_lookahead_slow_buffer_key(1)
    inner_before = state[inner_key].clone()
    outer_before = state[outer_key].clone()

    with torch.no_grad():
        param[:, 0, :].fill_(99.0)

    optimizer_utils.reset_optimizer_state_for_positions(optimizer, param, torch.tensor([True, False]))

    assert torch.count_nonzero(state["exp_avg"][:, 0, :]) == 0
    assert torch.equal(state["exp_avg"][:, 1, :], exp_avg_before[:, 1, :])
    assert torch.equal(state[inner_key][:, 0, :], param.detach()[:, 0, :])
    assert torch.equal(state[outer_key][:, 0, :], param.detach()[:, 0, :])
    assert torch.equal(state[inner_key][:, 1, :], inner_before[:, 1, :])
    assert torch.equal(state[outer_key][:, 1, :], outer_before[:, 1, :])

    optimizer_utils.reset_optimizer_state_for_positions(optimizer, param)

    assert optimizer.state[param] == {}
    staged_snapshot = param.detach().clone()

    param.grad = torch.ones_like(param)
    optimizer.step()

    assert "exp_avg" in optimizer.state[param]
    assert torch.equal(optimizer.state[param][inner_key], staged_snapshot)
    assert torch.equal(optimizer.state[param][outer_key], staged_snapshot)


def test_batch_optimize_parser_accepts_lora_b_lr():
    batch_optimize_main = _load_module(
        "batch_optimize_main",
        "benchmarking/invertibility/batch_optimize_main.py",
        _stub_batch_optimize_main_dependencies(),
    )

    args = batch_optimize_main.parse_args(
        ["--dataset_path", "dummy.json", "--logits_lora_b_learning_rate", "0.03125"]
    )

    assert args.logits_lora_b_learning_rate == pytest.approx(0.03125)


def test_batch_optimize_parser_accepts_lookahead_args():
    batch_optimize_main = _load_module(
        "batch_optimize_main",
        "benchmarking/invertibility/batch_optimize_main.py",
        _stub_batch_optimize_main_dependencies(),
    )

    args = batch_optimize_main.parse_args(
        ["--dataset_path", "dummy.json", "--lookahead_k", "8,80,800", "--lookahead_alpha", "0.2"]
    )

    assert args.lookahead_k == "8,80,800"
    assert args.lookahead_alpha == pytest.approx(0.2)


def test_invert_sentences_parser_accepts_lora_b_lr(monkeypatch):
    batch_optimize_main = types.ModuleType("batch_optimize_main")
    batch_optimize_main.optimize_for_target = lambda *args, **kwargs: {}
    batch_optimize_main.setup_model_and_tokenizer = lambda *args, **kwargs: (None, None)

    metrics_registry = types.ModuleType("metrics_registry")
    metrics_registry.lcs_length = lambda *args, **kwargs: 0

    invert_sentences = _load_module(
        "invert_sentences",
        "benchmarking/invertibility/invert_sentences.py",
        {
            "batch_optimize_main": batch_optimize_main,
            "metrics_registry": metrics_registry,
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["invert_sentences.py", "--sentences", "hello world", "--logits_lora_b_learning_rate", "0.125"],
    )

    args = invert_sentences.parse_args()

    assert args.logits_lora_b_learning_rate == pytest.approx(0.125)


def test_batch_optimize_pc_parser_accepts_lora_b_lr(monkeypatch):
    batch_optimize_pc_main = _load_module(
        "batch_optimize_pc_main",
        "benchmarking/invertibility/batch_optimize_pc_main.py",
        _stub_batch_optimize_pc_dependencies(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["batch_optimize_pc_main.py", "--logits_lora_b_learning_rate", "0.0625"],
    )

    args = batch_optimize_pc_main.parse_args()

    assert args.logits_lora_b_learning_rate == pytest.approx(0.0625)


def test_batch_optimize_pc_parser_accepts_reasoning_backend_args(monkeypatch):
    batch_optimize_pc_main = _load_module(
        "batch_optimize_pc_main",
        "benchmarking/invertibility/batch_optimize_pc_main.py",
        _stub_batch_optimize_pc_dependencies(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_optimize_pc_main.py",
            "--teacher_forcing_r",
            "False",
            "--reasoning_generation_backend",
            "hf_generate",
            "--reasoning_generate_do_sample",
            "True",
            "--reasoning_generate_top_p",
            "0.85",
            "--reasoning_generate_num_beams",
            "3",
        ],
    )

    args = batch_optimize_pc_main.parse_args()

    assert args.reasoning_generation_backend == "hf_generate"
    assert args.reasoning_generate_do_sample is True
    assert args.reasoning_generate_top_p == pytest.approx(0.85)
    assert args.reasoning_generate_num_beams == 3


def test_collect_reasoning_generate_kwargs_only_emits_explicit_values():
    batch_optimize_pc_main = _load_module(
        "batch_optimize_pc_main",
        "benchmarking/invertibility/batch_optimize_pc_main.py",
        _stub_batch_optimize_pc_dependencies(),
    )

    kwargs = batch_optimize_pc_main.collect_reasoning_generate_kwargs(
        {
            "reasoning_generate_do_sample": True,
            "reasoning_generate_top_p": 0.9,
            "reasoning_generate_top_k": None,
            "reasoning_generate_num_beams": 4,
        }
    )

    assert kwargs == {
        "do_sample": True,
        "top_p": 0.9,
        "num_beams": 4,
    }


def test_batch_optimize_pc_parser_accepts_prompt_eval_mode_args(monkeypatch):
    batch_optimize_pc_main = _load_module(
        "batch_optimize_pc_main",
        "benchmarking/invertibility/batch_optimize_pc_main.py",
        _stub_batch_optimize_pc_dependencies(),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_optimize_pc_main.py",
            "--val_prompt_eval_mode",
            "soft",
            "--test_prompt_eval_mode",
            "discrete",
        ],
    )

    args = batch_optimize_pc_main.parse_args()

    assert args.val_prompt_eval_mode == "soft"
    assert args.test_prompt_eval_mode == "discrete"


def test_batch_pc_optimize_propagates_reasoning_backend_to_all_eval_calls(tmp_path, monkeypatch):
    batch_optimize_pc_main = _load_module(
        "batch_optimize_pc_main_runtime",
        "benchmarking/invertibility/batch_optimize_pc_main.py",
        _stub_batch_optimize_pc_dependencies(),
    )

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total")

        def set_description(self, *args, **kwargs):
            return None

        def update(self, *args, **kwargs):
            return None

        def set_postfix(self, *args, **kwargs):
            return None

        def close(self):
            return None

    class FakeTokenizer:
        def __init__(self, vocab_size: int = 8):
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.chat_template = None
            self.vocab_size = vocab_size

        def __len__(self):
            return self.vocab_size

        def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
            del text, add_special_tokens
            return types.SimpleNamespace(input_ids=torch.tensor([[2, 3]], dtype=torch.long))

        def decode(self, ids, skip_special_tokens=False):
            del skip_special_tokens
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return " ".join(str(token) for token in ids)

    class FakeModel(torch.nn.Module):
        def __init__(self, vocab_size: int = 8, hidden_size: int = 4):
            super().__init__()
            self.config = types.SimpleNamespace(vocab_size=vocab_size, hidden_size=hidden_size)
            self.embedding = torch.nn.Embedding(vocab_size, hidden_size)

        def get_input_embeddings(self):
            return self.embedding

    class FakeDiffModel:
        def __init__(self, model, tokenizer, stgs_kwargs=None, stgs_logits_generation=None):
            del tokenizer, stgs_kwargs, stgs_logits_generation
            self.model = model

    class FakeSTGS:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class FakeArtifact:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def add_file(self, *args, **kwargs):
            return None

    class FakeRun:
        def __init__(self):
            self.summary = {}

        def log_artifact(self, *args, **kwargs):
            return None

    class FakeConfig:
        def update(self, *args, **kwargs):
            return None

    class FakeTable:
        def __init__(self, columns=None):
            self.columns = columns or []
            self.data = []

        def add_data(self, *args):
            self.data.append(args)

    class FakeWandb:
        def __init__(self):
            self.run = FakeRun()
            self.config = FakeConfig()

        def init(self, *args, **kwargs):
            del args, kwargs
            return self.run

        def log(self, *args, **kwargs):
            return None

        def finish(self):
            return None

        Artifact = FakeArtifact
        Table = FakeTable

    evaluate_calls = []
    optimize_kwargs = {}

    def fake_evaluate_shared_prompt(**kwargs):
        evaluate_calls.append(kwargs)
        metrics = {"any_correct": 0.5, "n_eval": 1.0}
        if kwargs.get("return_examples"):
            return metrics, []
        return metrics

    def fake_pc_optimize_inputs(**kwargs):
        optimize_kwargs.update(kwargs)
        final_logits = torch.zeros((1, kwargs["seq_len"], kwargs["model"].config.vocab_size))
        per_epoch_callback = kwargs.get("per_epoch_callback")
        if per_epoch_callback is not None:
            per_epoch_callback(0, final_logits)
        return {
            "final_p_logits": final_logits,
            "final_p_text": "prompt",
            "final_p_tokens": [0] * kwargs["seq_len"],
            "val_accuracy_history": [{"any_correct": 0.5, "epoch": 0}],
            "train_accuracy_history": [],
            "loss_history": [0.0],
            "learnable_logits": final_logits,
            "learnable_temperatures": {},
        }

    batch_optimize_pc_main.setup_model_and_tokenizer = lambda *args, **kwargs: (FakeModel(), FakeTokenizer())
    batch_optimize_pc_main.STGSDiffModel = FakeDiffModel
    batch_optimize_pc_main.STGS = FakeSTGS
    batch_optimize_pc_main.tqdm = lambda *args, **kwargs: FakeProgress(*args, **kwargs)
    batch_optimize_pc_main.wandb = FakeWandb()
    batch_optimize_pc_main.init_weave = lambda *args, **kwargs: None
    batch_optimize_pc_main.load_hf_gsm8k = lambda **kwargs: {
        "train": [{"input": "x-train", "reasoning": "r-train", "answer": "1"}],
        "validation": [{"input": "x-val", "answer": "2"}],
        "test": [{"input": "x-test", "answer": "3"}],
    }
    batch_optimize_pc_main.evaluate_shared_prompt = fake_evaluate_shared_prompt
    batch_optimize_pc_main.pc_optimize_inputs = fake_pc_optimize_inputs

    config = {
        "model_name": "fake-model",
        "device": "cpu",
        "model_precision": "full",
        "output_dir": str(tmp_path),
        "run_name": "pc-batch-test",
        "hf_dataset": "openai/gsm8k",
        "hf_dataset_subset": "main",
        "train_size": 1,
        "val_size": 1,
        "test_size": 1,
        "seq_len": 2,
        "epochs": 1,
        "teacher_forcing_r": False,
        "bptt": False,
        "reasoning_generation_backend": "hf_generate",
        "reasoning_generate_kwargs": {"do_sample": True, "top_p": 0.8},
        "val_prompt_eval_mode": "soft",
        "test_prompt_eval_mode": "soft",
        "test_eval_every": 1,
        "test_eval_size": 1,
        "wandb_project": "unit-test",
    }

    batch_optimize_pc_main.batch_pc_optimize(config)

    assert optimize_kwargs["reasoning_generation_backend"] == "hf_generate"
    assert optimize_kwargs["reasoning_generate_kwargs"] == {"do_sample": True, "top_p": 0.8}
    assert optimize_kwargs["val_prompt_eval_mode"] == "soft"
    assert len(evaluate_calls) == 3
    assert [call["eval_mode"] for call in evaluate_calls] == ["soft", "soft", "soft"]
    for call in evaluate_calls:
        assert call["reasoning_generation_backend"] == "hf_generate"
        assert call["reasoning_generate_kwargs"] == {"do_sample": True, "top_p": 0.8}
