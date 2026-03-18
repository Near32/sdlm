"""
DSPy adapter that preserves `SoftStr` values until the LM boundary.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, get_args, get_origin

from dspy.adapters.base import split_message_content_for_custom_types
from dspy.adapters.chat_adapter import ChatAdapter, format_field_value

from .soft_str import SoftStr
from .tensor_string import TensorString


def _annotation_uses_softstr(annotation: Any) -> bool:
    try:
        if isinstance(annotation, type) and issubclass(annotation, SoftStr):
            return True
    except TypeError:
        return False

    origin = get_origin(annotation)
    if origin is None:
        return False
    return any(_annotation_uses_softstr(arg) for arg in get_args(annotation))


class SDLMChatAdapter(ChatAdapter):
    """
    Chat adapter that leaves differentiable prompt segments as `SoftStr`.
    """

    def _coerce_field_value(self, annotation: Any, value: Any) -> Any:
        if value is None:
            return value
        if _annotation_uses_softstr(annotation):
            return SoftStr.coerce(value)
        return value

    def _join_parts(
        self,
        parts: Iterable[Any],
        separator: str = "\n\n",
    ) -> str | SoftStr:
        filtered_parts = [part for part in parts if part not in (None, "")]
        if not filtered_parts:
            return ""

        joined_text = separator.join(str(part) for part in filtered_parts).strip()

        has_soft = any(isinstance(part, (SoftStr, TensorString)) for part in filtered_parts)
        if not has_soft:
            return joined_text

        seed = next(part for part in filtered_parts if isinstance(part, (SoftStr, TensorString)))
        seed_soft = SoftStr.coerce(seed)
        result = None
        for index, part in enumerate(filtered_parts):
            if index > 0:
                sep_tensor = TensorString(
                    separator,
                    model_name=seed_soft.model_name,
                    device=seed_soft.device,
                )
                result = sep_tensor if result is None else result + sep_tensor
            if isinstance(part, SoftStr):
                next_tensor = part.to_tensor_string(
                    model_name=seed_soft.model_name,
                    device=seed_soft.device,
                )
            elif isinstance(part, TensorString):
                next_tensor = SoftStr.coerce(part, model_name=seed_soft.model_name).to_tensor_string(
                    model_name=seed_soft.model_name,
                    device=seed_soft.device,
                )
            else:
                next_tensor = TensorString(
                    str(part),
                    model_name=seed_soft.model_name,
                    device=seed_soft.device,
                )
            result = next_tensor if result is None else result + next_tensor

        if result is None:
            return ""
        rendered = SoftStr.from_tensor_string(result)
        rendered.text = joined_text
        return rendered

    def _format_section(self, field_name: str, field_info: Any, value: Any) -> str | SoftStr:
        coerced_value = self._coerce_field_value(field_info.annotation, value)
        if isinstance(coerced_value, SoftStr):
            return self._join_parts([f"[[ ## {field_name} ## ]]\n", coerced_value], separator="")
        return f"[[ ## {field_name} ## ]]\n{format_field_value(field_info=field_info, value=coerced_value)}"

    def format_user_message_content(
        self,
        signature,
        inputs,
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str | SoftStr:
        sections = [prefix]
        for key, field in signature.input_fields.items():
            if key in inputs:
                sections.append(self._format_section(key, field, inputs.get(key)))

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                sections.append(output_requirements)

        sections.append(suffix)
        return self._join_parts(sections)

    def format_assistant_message_content(
        self,
        signature,
        outputs,
        missing_field_message=None,
    ) -> str | SoftStr:
        sections = []
        for key, field in signature.output_fields.items():
            sections.append(
                self._format_section(
                    key,
                    field,
                    outputs.get(key, missing_field_message),
                )
            )
        sections.append("[[ ## completed ## ]]\n")
        return self._join_parts(sections)

    def format(self, signature, demos, inputs):
        inputs_copy = {
            key: self._coerce_field_value(signature.input_fields[key].annotation, value)
            if key in signature.input_fields
            else value
            for key, value in dict(inputs).items()
        }

        history_field_name = self._get_history_field_name(signature)
        if history_field_name:
            signature_without_history = signature.delete(history_field_name)
            conversation_history = self.format_conversation_history(
                signature_without_history,
                history_field_name,
                inputs_copy,
            )
        else:
            signature_without_history = signature
            conversation_history = []

        messages = []
        system_message = (
            f"{self.format_field_description(signature)}\n"
            f"{self.format_field_structure(signature)}\n"
            f"{self.format_task_description(signature)}"
        )
        messages.append({"role": "system", "content": system_message})
        messages.extend(self.format_demos(signature, demos))

        if history_field_name:
            content = self.format_user_message_content(
                signature_without_history,
                inputs_copy,
                main_request=True,
            )
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": content})
        else:
            content = self.format_user_message_content(signature, inputs_copy, main_request=True)
            messages.append({"role": "user", "content": content})

        processed_messages = []
        for message in messages:
            if isinstance(message.get("content"), str):
                processed_messages.extend(split_message_content_for_custom_types([message]))
            else:
                processed_messages.append(message)
        return processed_messages
