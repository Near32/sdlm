import logging
from string import Formatter
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)

_EPS = 1e-10
_FORMATTER = Formatter()

DEFAULT_POS_PROMPT = (
    "You are scoring candidate prompt tokens for an inversion task. "
    "Return the next prompt token that best helps reconstruct the hidden prompt."
)


def _top_k_logit_filter(
    logits: torch.Tensor,
    k: int,
    penalty: float = 1e4,
) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    with torch.no_grad():
        top_k_vals = torch.topk(logits, min(k, logits.size(-1)), dim=-1).values
        threshold = top_k_vals[..., -1:]
        mask = logits < threshold
    return logits - mask.to(logits.dtype) * penalty


def _top_p_logit_filter(
    logits: torch.Tensor,
    p: float,
    penalty: float = 1e4,
) -> torch.Tensor:
    if p >= 1.0:
        return logits
    if p <= 0.0:
        raise ValueError(f"pos_lm_top_p must be in (0, 1], got {p}")
    with torch.no_grad():
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = (cumprobs - sorted_probs) >= p
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, sorted_indices, sorted_mask)
    return logits - mask.to(logits.dtype) * penalty


def _load_pos_model_and_tokenizer(
    *,
    pos_lm_name: Optional[str],
    base_model,
    base_tokenizer,
    base_model_name: Optional[str],
) -> Tuple[object, object, bool]:
    if not pos_lm_name or pos_lm_name == base_model_name:
        return base_model, base_tokenizer, True

    logger.info("Loading product-of-speaker LM: %s", pos_lm_name)
    pos_model = AutoModelForCausalLM.from_pretrained(
        pos_lm_name,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    pos_model.eval()
    for param in pos_model.parameters():
        param.requires_grad = False

    pos_tokenizer = AutoTokenizer.from_pretrained(pos_lm_name)
    if pos_tokenizer.get_vocab() != base_tokenizer.get_vocab():
        raise ValueError(
            "product_of_speaker requires the auxiliary LM tokenizer to share the same "
            "token-to-id vocabulary as the optimized model."
        )

    return pos_model, pos_tokenizer, False


def _extract_template_fields(template: str) -> Set[str]:
    fields: Set[str] = set()
    for _, field_name, _, _ in _FORMATTER.parse(template):
        if field_name is None:
            continue
        if field_name == "":
            raise ValueError("PoS templates do not support anonymous '{}' placeholders.")
        if any(ch in field_name for ch in ".[]"):
            raise ValueError(
                f"PoS templates support only simple placeholder names, got {field_name!r}."
            )
        fields.add(field_name)
    return fields


def _render_template(
    *,
    template: str,
    allowed_fields: Set[str],
    values: Dict[str, str],
    template_name: str,
) -> str:
    fields = _extract_template_fields(template)
    unsupported = sorted(fields - allowed_fields)
    if unsupported:
        raise ValueError(
            f"{template_name} contains unsupported placeholders {unsupported}; "
            f"allowed placeholders are {sorted(allowed_fields)}."
        )
    if not fields:
        return template
    missing = sorted(field for field in fields if field not in values)
    if missing:
        raise ValueError(
            f"{template_name} requires values for placeholders {missing}, but they were not provided."
        )
    return template.format(**{field: values[field] for field in fields})


def render_pos_single_target_prompt(
    *,
    pos_prompt_template: Optional[str],
    pos_prompt: Optional[str],
    target_text: str,
) -> str:
    if pos_prompt_template is not None:
        return _render_template(
            template=pos_prompt_template,
            allowed_fields={"target"},
            values={"target": target_text},
            template_name="pos_prompt_template",
        )
    if pos_prompt is not None:
        return pos_prompt
    return DEFAULT_POS_PROMPT


def _render_pos_pairs_block(
    *,
    pos_pair_template: str,
    pair_contexts: Sequence[Tuple[str, str]],
    pos_pair_separator: str,
) -> str:
    rendered_pairs = []
    for partial_conditioning_text, target_text in pair_contexts:
        rendered_pairs.append(
            _render_template(
                template=pos_pair_template,
                allowed_fields={"partial_conditioning_text", "target"},
                values={
                    "partial_conditioning_text": partial_conditioning_text,
                    "target": target_text,
                },
                template_name="pos_pair_template",
            )
        )
    return pos_pair_separator.join(rendered_pairs)


def render_pos_pc_prompt(
    *,
    pos_prompt_template: Optional[str],
    pos_pair_template: Optional[str],
    pair_contexts: Sequence[Tuple[str, str]],
    pos_pair_separator: str,
    pos_prompt: Optional[str],
) -> str:
    if pos_prompt_template is not None:
        prompt_fields = _extract_template_fields(pos_prompt_template)
        if "pairs" in prompt_fields:
            if pos_pair_template is None:
                raise ValueError(
                    "pos_prompt_template uses {pairs}, but pos_pair_template was not provided."
                )
            pairs_text = _render_pos_pairs_block(
                pos_pair_template=pos_pair_template,
                pair_contexts=pair_contexts,
                pos_pair_separator=pos_pair_separator,
            )
        else:
            pairs_text = ""
        return _render_template(
            template=pos_prompt_template,
            allowed_fields={"pairs"},
            values={"pairs": pairs_text},
            template_name="pos_prompt_template",
        )
    if pos_prompt is not None:
        return pos_prompt
    return DEFAULT_POS_PROMPT


class ProductOfSpeakerFactory:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        allowed_tokens: torch.Tensor,
        seq_len: int,
        base_model_name: Optional[str],
        pos_lm_name: Optional[str] = None,
        pos_use_chat_template: bool = False,
        pos_lm_temperature: float = 1.0,
        pos_lm_top_p: float = 1.0,
        pos_lm_top_k: int = 0,
        logits_filter_penalty: float = 1e4,
    ):
        if pos_lm_temperature <= 0.0:
            raise ValueError(
                f"pos_lm_temperature must be > 0, got {pos_lm_temperature}"
            )
        if pos_lm_top_k < 0:
            raise ValueError(f"pos_lm_top_k must be >= 0, got {pos_lm_top_k}")

        self.pos_model, self.pos_tokenizer, reused_base_model = _load_pos_model_and_tokenizer(
            pos_lm_name=pos_lm_name,
            base_model=model,
            base_tokenizer=tokenizer,
            base_model_name=base_model_name,
        )

        self.allowed_tokens = allowed_tokens.detach().long()
        if self.allowed_tokens.numel() == 0:
            raise ValueError("allowed_tokens must be non-empty for product_of_speaker")
        if int(self.allowed_tokens.max().item()) >= self.pos_model.config.vocab_size:
            raise ValueError(
                "allowed_tokens contains token ids outside the auxiliary LM vocabulary"
            )

        self.seq_len = seq_len
        self.pos_use_chat_template = pos_use_chat_template
        self.pos_lm_temperature = pos_lm_temperature
        self.pos_lm_top_p = pos_lm_top_p
        self.pos_lm_top_k = pos_lm_top_k
        self.logits_filter_penalty = logits_filter_penalty
        self.embedding_weight = self.pos_model.get_input_embeddings().weight.detach()
        self.embedding_allowed = self.embedding_weight[
            self.allowed_tokens.to(self.embedding_weight.device)
        ]
        self._cached_prompt_text: Optional[str] = None
        self._cached_logits: Optional[torch.Tensor] = None
        self._cached_past_key_values = None

        logger.info(
            "Product-of-speaker factory enabled: lm=%s, reuse_base_model=%s, "
            "use_chat_template=%s, temperature=%.4f, top_k=%d, top_p=%.4f, max_token_len=%d",
            pos_lm_name or base_model_name or "<active-model>",
            reused_base_model,
            pos_use_chat_template,
            pos_lm_temperature,
            pos_lm_top_k,
            pos_lm_top_p,
            seq_len,
        )

    def _tokenize_prompt(self, prompt_text: str) -> torch.Tensor:
        if self.pos_use_chat_template:
            if getattr(self.pos_tokenizer, "chat_template", None) is not None:
                prompt_ids = torch.tensor(
                    self.pos_tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt_text}],
                        tokenize=True,
                        add_generation_prompt=True,
                    ),
                    dtype=torch.long,
                ).unsqueeze(0)
            else:
                logger.warning(
                    "pos_use_chat_template=True but tokenizer has no chat_template; "
                    "falling back to plain tokenization."
                )
                prompt_ids = self.pos_tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids
        else:
            prompt_ids = self.pos_tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids
        if prompt_ids.numel() == 0:
            fallback_token_id = (
                self.pos_tokenizer.bos_token_id
                or self.pos_tokenizer.eos_token_id
                or self.pos_tokenizer.pad_token_id
                or 0
            )
            prompt_ids = torch.tensor([[fallback_token_id]], dtype=torch.long)
        return prompt_ids

    def _prime_prompt(self, prompt_text: str) -> Tuple[torch.Tensor, Tuple]:
        if (
            prompt_text == self._cached_prompt_text
            and self._cached_logits is not None
            and self._cached_past_key_values is not None
        ):
            return self._cached_logits, self._cached_past_key_values

        prompt_ids = self._tokenize_prompt(prompt_text)
        with torch.no_grad():
            prompt_outputs = self.pos_model(
                input_ids=prompt_ids.to(self.embedding_weight.device),
                use_cache=True,
                return_dict=True,
            )

        cached_logits = prompt_outputs.logits[:, -1:, :].detach()
        cached_past_key_values = tuple(
            tuple(value.detach() for value in layer)
            for layer in prompt_outputs.past_key_values
        )

        self._cached_prompt_text = prompt_text
        self._cached_logits = cached_logits
        self._cached_past_key_values = cached_past_key_values
        return cached_logits, cached_past_key_values

    def transform_logits(self, base_logits: torch.Tensor, prompt_text: str) -> torch.Tensor:
        if base_logits.dim() != 3 or base_logits.shape[0] != 1:
            raise ValueError(
                "product_of_speaker expects logits with shape (1, seq_len, vocab)"
            )
        if base_logits.shape[1] != self.seq_len:
            raise ValueError(
                f"product_of_speaker expected seq_len={self.seq_len}, got {base_logits.shape[1]}"
            )

        cached_logits, cached_past_key_values = self._prime_prompt(prompt_text)
        speaker_probs = torch.softmax(base_logits, dim=-1)
        lm_logits_step = cached_logits.to(base_logits.device, dtype=base_logits.dtype)
        past_key_values = cached_past_key_values
        combined_probs_steps = []

        for pos in range(self.seq_len):
            lm_allowed = lm_logits_step[..., self.allowed_tokens.to(lm_logits_step.device)]
            lm_allowed = lm_allowed / self.pos_lm_temperature
            if self.pos_lm_top_k > 0:
                lm_allowed = _top_k_logit_filter(
                    lm_allowed,
                    self.pos_lm_top_k,
                    self.logits_filter_penalty,
                )
            if self.pos_lm_top_p < 1.0:
                lm_allowed = _top_p_logit_filter(
                    lm_allowed,
                    self.pos_lm_top_p,
                    self.logits_filter_penalty,
                )

            lm_probs = torch.softmax(lm_allowed, dim=-1)
            speaker_step = speaker_probs[:, pos : pos + 1, :]
            combined = speaker_step * lm_probs
            combined_mass = combined.sum(dim=-1, keepdim=True)
            fallback = speaker_step / speaker_step.sum(dim=-1, keepdim=True).clamp_min(_EPS)
            combined = torch.where(
                combined_mass > _EPS,
                combined / combined_mass.clamp_min(_EPS),
                fallback,
            )
            combined_probs_steps.append(combined)

            if pos + 1 >= self.seq_len:
                continue

            step_embeds = torch.matmul(
                combined.to(self.embedding_allowed.device),
                self.embedding_allowed.to(combined.dtype),
            )
            step_outputs = self.pos_model(
                inputs_embeds=step_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            lm_logits_step = step_outputs.logits[:, -1:, :]
            past_key_values = step_outputs.past_key_values

        combined_probs = torch.cat(combined_probs_steps, dim=1)
        return torch.log(combined_probs.clamp_min(_EPS))


def build_product_of_speaker_callback(
    *,
    model,
    tokenizer,
    allowed_tokens: torch.Tensor,
    seq_len: int,
    base_model_name: Optional[str],
    pos_lm_name: Optional[str] = None,
    pos_prompt: Optional[str] = None,
    pos_use_chat_template: bool = False,
    pos_lm_temperature: float = 1.0,
    pos_lm_top_p: float = 1.0,
    pos_lm_top_k: int = 0,
    logits_filter_penalty: float = 1e4,
) -> Callable[[torch.Tensor], torch.Tensor]:
    factory = ProductOfSpeakerFactory(
        model=model,
        tokenizer=tokenizer,
        allowed_tokens=allowed_tokens,
        seq_len=seq_len,
        base_model_name=base_model_name,
        pos_lm_name=pos_lm_name,
        pos_use_chat_template=pos_use_chat_template,
        pos_lm_temperature=pos_lm_temperature,
        pos_lm_top_p=pos_lm_top_p,
        pos_lm_top_k=pos_lm_top_k,
        logits_filter_penalty=logits_filter_penalty,
    )
    prompt_text = pos_prompt or DEFAULT_POS_PROMPT
    return lambda base_logits: factory.transform_logits(base_logits, prompt_text)


def build_pc_product_of_speaker_transform(
    *,
    model,
    tokenizer,
    allowed_tokens: torch.Tensor,
    seq_len: int,
    base_model_name: Optional[str],
    pos_lm_name: Optional[str] = None,
    pos_prompt: Optional[str] = None,
    pos_prompt_template: Optional[str] = None,
    pos_pair_template: Optional[str] = None,
    pos_pair_separator: str = "\n\n",
    pos_use_chat_template: bool = False,
    pos_lm_temperature: float = 1.0,
    pos_lm_top_p: float = 1.0,
    pos_lm_top_k: int = 0,
    logits_filter_penalty: float = 1e4,
) -> Callable[[torch.Tensor, Sequence[Tuple[str, str]]], torch.Tensor]:
    factory = ProductOfSpeakerFactory(
        model=model,
        tokenizer=tokenizer,
        allowed_tokens=allowed_tokens,
        seq_len=seq_len,
        base_model_name=base_model_name,
        pos_lm_name=pos_lm_name,
        pos_use_chat_template=pos_use_chat_template,
        pos_lm_temperature=pos_lm_temperature,
        pos_lm_top_p=pos_lm_top_p,
        pos_lm_top_k=pos_lm_top_k,
        logits_filter_penalty=logits_filter_penalty,
    )

    def transform(base_logits: torch.Tensor, pair_contexts: Sequence[Tuple[str, str]]) -> torch.Tensor:
        prompt_text = render_pos_pc_prompt(
            pos_prompt_template=pos_prompt_template,
            pos_pair_template=pos_pair_template,
            pair_contexts=pair_contexts,
            pos_pair_separator=pos_pair_separator,
            pos_prompt=pos_prompt,
        )
        return factory.transform_logits(base_logits, prompt_text)

    return transform
