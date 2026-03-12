import torch
import torch.nn.functional as F
import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from sdlm.stgs_diff_model import STGSDiffModel


class DummyTokenizer:
    def __init__(self, vocab_size: int, pad_token_id: int = 0, eos_token_id: int = 1):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __len__(self):
        return self.vocab_size


def _build_diff_model(*, stgs_hard: bool, vocab_size: int = 32) -> STGSDiffModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=32,
        n_ctx=32,
        n_embd=24,
        n_layer=1,
        n_head=2,
        pad_token_id=0,
        eos_token_id=1,
    )
    base_model = GPT2LMHeadModel(config)
    tokenizer = DummyTokenizer(vocab_size=vocab_size, pad_token_id=0, eos_token_id=1)
    diff_model = STGSDiffModel(
        model=base_model,
        tokenizer=tokenizer,
        stgs_kwargs={
            "hard": stgs_hard,
            "temperature": 1.0,
            "learnable_temperature": False,
            "hidden_state_conditioning": False,
            "dropout": 0.0,
        },
        stgs_logits_generation=True,
        device=None,
    )
    diff_model.eval()
    return diff_model


def test_teacher_forcing_overrides_straight_through_output_when_hard():
    diff_model = _build_diff_model(stgs_hard=True, vocab_size=32)

    input_ids = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=torch.long)
    teacher_forced_ids = torch.tensor([[10, 11, 12, 13], [14, 15, 16, 17]], dtype=torch.long)

    outputs = diff_model(
        input_ids=input_ids,
        output_diff_tokens=True,
        output_diff_one_hots=True,
        output_stgs_logits=True,
        return_dict=True,
        teacher_forced_token_ids=teacher_forced_ids,
    )

    assert torch.equal(outputs.sampled_diff_tokens.long(), teacher_forced_ids)
    assert torch.equal(outputs.sampled_diff_one_hot.argmax(dim=-1), teacher_forced_ids)
    assert torch.allclose(
        outputs.sampled_diff_one_hot,
        F.one_hot(teacher_forced_ids, num_classes=32).to(outputs.sampled_diff_one_hot.dtype),
        atol=0.0,
        rtol=0.0,
    )


def test_teacher_forcing_shape_mismatch_raises():
    diff_model = _build_diff_model(stgs_hard=True, vocab_size=32)
    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    bad_shape_teacher_forced_ids = torch.tensor([[10, 11, 12]], dtype=torch.long)

    with pytest.raises(ValueError, match="teacher_forced_token_ids must have shape"):
        diff_model(
            input_ids=input_ids,
            output_diff_tokens=True,
            output_diff_one_hots=True,
            return_dict=True,
            teacher_forced_token_ids=bad_shape_teacher_forced_ids,
        )


def test_teacher_forcing_is_ignored_when_not_hard_for_backward_compatibility():
    diff_model = _build_diff_model(stgs_hard=False, vocab_size=32)
    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    teacher_forced_ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)

    torch.manual_seed(1234)
    out_without_tf = diff_model(
        input_ids=input_ids,
        output_diff_tokens=True,
        output_diff_one_hots=True,
        return_dict=True,
    )

    torch.manual_seed(1234)
    out_with_tf = diff_model(
        input_ids=input_ids,
        output_diff_tokens=True,
        output_diff_one_hots=True,
        return_dict=True,
        teacher_forced_token_ids=teacher_forced_ids,
    )

    assert torch.equal(out_without_tf.sampled_diff_tokens, out_with_tf.sampled_diff_tokens)
    assert torch.allclose(out_without_tf.sampled_diff_one_hot, out_with_tf.sampled_diff_one_hot)
