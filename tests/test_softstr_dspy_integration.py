import os
import types

os.environ.setdefault("SDLM_QUIET", "1")

import dspy
import pytest
import torch
from dspy.clients.lm import BaseLM

import sdlm
from sdlm.core.tensor_string import TensorString


class DummyTokenizer:
    def __init__(self, vocab_size=32):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors="pt", padding=False, truncation=False, max_length=None):
        del padding, truncation, max_length
        token_ids = [2 + (ord(char) % (self.vocab_size - 2)) for char in text]
        if not token_ids:
            token_ids = [self.eos_token_id]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in (self.pad_token_id, self.eos_token_id):
                continue
            chars.append(chr(((int(token_id) - 2) % 26) + ord("a")))
        return "".join(chars)

    def convert_ids_to_tokens(self, token_ids):
        return [str(token_id) for token_id in token_ids]

    def tokenize(self, text):
        return list(text)


class DummyEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, embed_dim))


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size=32, embed_dim=8):
        super().__init__()
        self.embedding = DummyEmbedding(vocab_size, embed_dim)
        self.config = types.SimpleNamespace(hidden_size=embed_dim)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds=None, return_dict=True, use_cache=False, **kwargs):
        del return_dict, use_cache, kwargs
        logits = torch.matmul(inputs_embeds, self.embedding.weight.T)
        return types.SimpleNamespace(logits=logits)

    __call__ = forward

    def to(self, device):
        del device
        return self

    def eval(self):
        return self

    def train(self):
        return self


class RecordingLM(BaseLM):
    def __init__(self, output_text):
        self.kwargs = {}
        self.history = []
        self.model_type = "chat"
        self.output_text = output_text
        self.last_messages = None

    def forward(self, prompt=None, messages=None, **kwargs):
        del prompt, kwargs
        self.last_messages = messages
        return [self.output_text]

    def __call__(self, prompt=None, messages=None, **kwargs):
        return self.forward(prompt=prompt, messages=messages, **kwargs)


class SoftQASignature(dspy.Signature):
    question: sdlm.SoftStr = dspy.InputField()
    answer: sdlm.SoftStr = dspy.OutputField()


@pytest.fixture()
def dummy_tensorstring_model():
    TensorString._default_model_name = "dummy"
    TensorString._global_device = torch.device("cpu")
    TensorString._tokenizer_cache["dummy"] = DummyTokenizer()
    TensorString._model_cache["dummy"] = DummyModel()
    yield
    TensorString._tokenizer_cache.pop("dummy", None)
    TensorString._model_cache.pop("dummy", None)


def test_adapter_keeps_softstr_content(dummy_tensorstring_model):
    adapter = sdlm.SDLMChatAdapter()
    learnable = sdlm.soft_str("hello", model_name="dummy", requires_grad=True)

    messages = adapter.format(SoftQASignature, demos=[], inputs={"question": learnable})

    assert isinstance(messages[-1]["content"], sdlm.SoftStr)
    assert messages[-1]["content"].requires_grad
    assert "[[ ## question ## ]]" in messages[-1]["content"].text


def test_adapter_autowraps_plain_strings_for_softstr_fields(dummy_tensorstring_model):
    adapter = sdlm.SDLMChatAdapter()

    messages = adapter.format(SoftQASignature, demos=[], inputs={"question": "plain prompt"})

    assert isinstance(messages[-1]["content"], sdlm.SoftStr)
    assert "plain prompt" in messages[-1]["content"].text


def test_predict_round_trips_softstr(dummy_tensorstring_model):
    lm = RecordingLM("[[ ## answer ## ]]\nsoft output\n\n[[ ## completed ## ]]")
    dspy.configure(lm=lm, adapter=sdlm.SDLMChatAdapter())

    predictor = dspy.Predict(SoftQASignature)
    result = predictor(question="hello world")

    assert isinstance(lm.last_messages[-1]["content"], sdlm.SoftStr)
    assert isinstance(result.answer, sdlm.SoftStr)
    assert result.answer.text == "soft output"


def test_differentiable_lm_builds_prompt_from_softstr(monkeypatch, dummy_tensorstring_model):
    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()
    monkeypatch.setattr(
        "sdlm.core.differentiable_lm.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: dummy_model,
    )
    monkeypatch.setattr(
        "sdlm.core.differentiable_lm.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: dummy_tokenizer,
    )

    lm = sdlm.DifferentiableLM("dummy", device="cpu", max_tokens=2)
    learnable = sdlm.SoftStr.from_tensor_string(
        sdlm.from_string("soft", model_name="dummy", requires_grad=True)
    )

    prompt = lm._messages_to_prompt([{"role": "user", "content": learnable}])

    assert isinstance(prompt, TensorString)
    assert prompt.requires_grad
    assert "User:" in str(prompt)
