"""Tests for pc_forward_pass_batched_tf, pc_forward_pass_batched_free,
pc_forward_pass_batched, and evaluate_shared_prompt_batched."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarking" / "invertibility"))
import pc_main


def _fake_stgs(vocab_size, seq_len, device="cpu"):
    embed_dim = 8
    embedding_weights = torch.randn(vocab_size, embed_dim, device=device)

    class _FakeSTGS:
        def __call__(self, logits, gumbel_noise_scale=1.0, embedding_weights=None):
            B, L, V = logits.shape
            one_hot = F.one_hot(logits.argmax(-1), V).float()
            return None, one_hot, None, None

    return _FakeSTGS(), embedding_weights


def _make_diff_model(vocab_size=32, embed_dim=8, device="cpu"):
    calls = []

    class _FakeDiffModel:
        pad_token_id = 0

        def forward(self, inputs_embeds, attention_mask=None, output_normal_logits=False, **kw):
            calls.append(("forward", inputs_embeds.shape, attention_mask))
            B, T, _ = inputs_embeds.shape
            class NS: pass
            ns = NS(); ns.logits = torch.zeros(B, T, vocab_size, device=device)
            return ns

        def generate(self, inputs_embeds=None, attention_mask=None, max_new_tokens=5, **kw):
            calls.append(("generate", inputs_embeds.shape, attention_mask))
            B = inputs_embeds.shape[0]
            tokens = torch.zeros(B, max_new_tokens, dtype=torch.long, device=device)
            one_hot = F.one_hot(tokens, vocab_size).float()
            embeds = one_hot @ torch.randn(vocab_size, embed_dim, device=device)
            class NS: pass
            ns = NS()
            ns.sampled_diff_tokens = tokens
            ns.sampled_diff_one_hot = one_hot
            ns.sampled_diff_embeds = embeds
            ns.logits = torch.zeros(B, max_new_tokens, vocab_size, device=device)
            return ns

    return _FakeDiffModel(), calls


class _FakeLoss:
    def compute_loss(self, d):
        return {"sumloss": d["generated_logits"].sum() * 0}


def test_batched_tf_single_forward_call():
    vocab_size, seq_len = 32, 4
    stgs, emb_w = _fake_stgs(vocab_size, seq_len)
    diff_model, calls = _make_diff_model(vocab_size)
    free_logits = torch.randn(1, seq_len, vocab_size)
    B = 3
    x_embeds  = [torch.randn(1, 5 + i, 8) for i in range(B)]
    R_embeds  = [torch.randn(1, 3, 8) for _ in range(B)]
    E_embeds  = torch.randn(1, 2, 8)
    Y_tokens  = [torch.zeros(1, 2 + i, dtype=torch.long) for i in range(B)]

    losses, yl = pc_main.pc_forward_pass_batched_tf(
        diff_model=diff_model, stgs_module=stgs,
        loss_instances=[_FakeLoss()] * B,
        free_logits=free_logits, embedding_weights=emb_w,
        x_embeds_list=x_embeds, R_gt_embeds_list=R_embeds,
        E_embeds=E_embeds, Y_tokens_list=Y_tokens,
        gumbel_noise_scale=1.0, stgs_noise_mode="shared",
    )
    fwd = [c for c in calls if c[0] == "forward"]
    assert len(fwd) == 1, "Expected exactly 1 batched forward call"
    assert fwd[0][1][0] == B
    assert len(losses) == B
    assert len(yl) == B


def test_batched_tf_attention_mask():
    vocab_size, seq_len = 32, 4
    stgs, emb_w = _fake_stgs(vocab_size, seq_len)
    diff_model, calls = _make_diff_model(vocab_size)
    free_logits = torch.randn(1, seq_len, vocab_size)
    x_lens, R_lens, Y_lens, E_len = [3, 5], [2, 4], [2, 3], 2
    B = 2
    pc_main.pc_forward_pass_batched_tf(
        diff_model=diff_model, stgs_module=stgs,
        loss_instances=[_FakeLoss()] * B,
        free_logits=free_logits, embedding_weights=emb_w,
        x_embeds_list=[torch.randn(1, l, 8) for l in x_lens],
        R_gt_embeds_list=[torch.randn(1, l, 8) for l in R_lens],
        E_embeds=torch.randn(1, E_len, 8),
        Y_tokens_list=[torch.zeros(1, l, dtype=torch.long) for l in Y_lens],
        gumbel_noise_scale=1.0, stgs_noise_mode="shared",
    )
    _, _, mask = calls[0]
    for i in range(B):
        real_len = x_lens[i] + seq_len + R_lens[i] + E_len + (Y_lens[i] - 1)
        assert mask[i].sum().item() == real_len
        assert mask[i, -real_len:].all()
        assert not mask[i, :-real_len].any()


def test_batched_free_two_generate_calls():
    vocab_size, seq_len = 32, 4
    stgs, emb_w = _fake_stgs(vocab_size, seq_len)
    diff_model, calls = _make_diff_model(vocab_size)
    free_logits = torch.randn(1, seq_len, vocab_size)
    B = 2
    losses, yl = pc_main.pc_forward_pass_batched_free(
        diff_model=diff_model, stgs_module=stgs,
        loss_instances=[_FakeLoss()] * B,
        free_logits=free_logits, embedding_weights=emb_w,
        x_embeds_list=[torch.randn(1, 4 + i, 8) for i in range(B)],
        E_embeds=torch.randn(1, 2, 8),
        Y_tokens_list=[torch.zeros(1, 3, dtype=torch.long) for _ in range(B)],
        max_new_tokens_reasoning=5, max_new_tokens_answer=5,
        gumbel_noise_scale=1.0, stgs_noise_mode="shared",
    )
    gen = [c for c in calls if c[0] == "generate"]
    assert len(gen) == 2, "Expected stage-1 + stage-2 generate calls"
    assert gen[0][1][0] == B
    assert gen[1][1][0] == B
    assert len(losses) == B


def test_independent_noise_mode_accepted():
    vocab_size, seq_len = 32, 4
    stgs, emb_w = _fake_stgs(vocab_size, seq_len)
    diff_model, _ = _make_diff_model(vocab_size)
    B = 2
    losses, _ = pc_main.pc_forward_pass_batched_tf(
        diff_model=diff_model, stgs_module=stgs,
        loss_instances=[_FakeLoss()] * B,
        free_logits=torch.randn(1, seq_len, vocab_size),
        embedding_weights=emb_w,
        x_embeds_list=[torch.randn(1, 4, 8) for _ in range(B)],
        R_gt_embeds_list=[None] * B,
        E_embeds=torch.randn(1, 2, 8),
        Y_tokens_list=[torch.zeros(1, 2, dtype=torch.long) for _ in range(B)],
        gumbel_noise_scale=1.0, stgs_noise_mode="independent",
    )
    assert len(losses) == B


def test_dispatcher_bptt_true_still_batched():
    """bptt=True with TF=False: still 2 batched generate calls (B inputs each)."""
    vocab_size, seq_len = 32, 4
    stgs, emb_w = _fake_stgs(vocab_size, seq_len)
    diff_model, calls = _make_diff_model(vocab_size)
    B = 3
    losses, _ = pc_main.pc_forward_pass_batched(
        diff_model=diff_model, stgs_module=stgs,
        loss_instances=[_FakeLoss()] * B,
        free_logits=torch.randn(1, seq_len, vocab_size),
        embedding_weights=emb_w,
        x_embeds_list=[torch.randn(1, 4, 8)] * B,
        R_gt_embeds_list=[None] * B,
        E_embeds=torch.randn(1, 2, 8),
        E_input_ids=torch.zeros(1, 2, dtype=torch.long),
        Y_tokens_list=[torch.zeros(1, 2, dtype=torch.long)] * B,
        teacher_forcing_r=False, bptt=True,
        reasoning_generation_backend="diff", reasoning_generate_kwargs={},
        max_new_tokens_reasoning=5, max_new_tokens_answer=5,
        stgs_noise_mode="shared",
    )
    gen = [c for c in calls if c[0] == "generate"]
    assert len(gen) == 2, "bptt=True should still use 2 batched generate calls"
    assert gen[0][1][0] == B
    assert gen[1][1][0] == B
    assert len(losses) == B


def test_evaluate_shared_prompt_batched_discrete_one_generate_call_per_stage():
    """Discrete batched eval: model.generate called once per stage (B inputs)."""
    vocab_size, seq_len = 32, 4
    stgs, emb_w = _fake_stgs(vocab_size, seq_len)
    diff_model, _ = _make_diff_model(vocab_size)
    free_logits = torch.randn(1, seq_len, vocab_size)
    B = 3

    base_calls = []
    class _FakeBase:
        config = type("C", (), {"vocab_size": vocab_size})()
        def get_input_embeddings(self):
            class EL:
                weight = emb_w
                def __call__(self, ids): return emb_w[ids]
            return EL()
        def generate(self, input_ids=None, attention_mask=None, max_new_tokens=5, **kw):
            base_calls.append(input_ids.shape)
            B_ = input_ids.shape[0]
            return type("O", (), {
                "sequences": torch.zeros(B_, input_ids.shape[1]+max_new_tokens, dtype=torch.long)
            })()

    class _FakeTok:
        pad_token_id = 0
        eos_token_id = 0
        def __call__(self, text, **kw):
            return type("E", (), {"input_ids": torch.zeros(1,3,dtype=torch.long)})()
        def decode(self, ids, **kw): return ""

    eval_pairs = [(f"q{i}", f"{i}") for i in range(B)]
    metrics = pc_main.evaluate_shared_prompt_batched(
        free_logits=free_logits, eval_pairs=eval_pairs,
        model=_FakeBase(), diff_model=diff_model, stgs_module=stgs,
        embedding_weights=emb_w, tokenizer=_FakeTok(), device="cpu",
        E_input_ids=torch.zeros(1,2,dtype=torch.long),
        E_embeds=torch.randn(1,2,8),
        extraction_fns={},
        max_new_tokens_reasoning=5, max_new_tokens_answer=5,
        eval_mode="discrete",
    )
    # Two generate calls: stage-1 (reasoning) and stage-2 (answer)
    assert len(base_calls) == 2
    assert base_calls[0][0] == B
    assert base_calls[1][0] == B
