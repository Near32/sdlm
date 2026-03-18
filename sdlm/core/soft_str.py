"""
DSPy-facing differentiable text type for SDLM.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from dspy.adapters.types.base_type import Type
from pydantic import ConfigDict, PrivateAttr, model_validator

from .tensor_string import TensorString


class SoftStr(Type):
    """
    Differentiable text wrapper used in DSPy signatures.

    `SoftStr` keeps a string view for DSPy compatibility while lazily materializing
    a `TensorString` when tensor-backed operations are needed.
    """

    text: str = ""
    model_name: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _tensor_string: Optional[TensorString] = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _coerce_value(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return {"text": value.text, "model_name": value.model_name}
        if isinstance(value, TensorString):
            return {"text": str(value), "model_name": value._model_name}
        if isinstance(value, str):
            return {"text": value}
        return value

    @classmethod
    def from_string(
        cls,
        text: str,
        model_name: Optional[str] = None,
    ) -> "SoftStr":
        return cls(text=text, model_name=model_name)

    @classmethod
    def from_tensor_string(cls, tensor_string: TensorString) -> "SoftStr":
        instance = cls(text=str(tensor_string), model_name=tensor_string._model_name)
        instance._tensor_string = tensor_string
        return instance

    @classmethod
    def coerce(
        cls,
        value: Union["SoftStr", TensorString, str],
        model_name: Optional[str] = None,
    ) -> "SoftStr":
        if isinstance(value, cls):
            if model_name is None or value.model_name == model_name:
                return value
            if value._tensor_string is None or not value._tensor_string.requires_grad:
                return cls.from_string(value.text, model_name=model_name)
            raise ValueError(
                "Cannot retokenize a gradient-carrying SoftStr for a different model. "
                "Create the SoftStr with the target LM tokenizer."
            )
        if isinstance(value, TensorString):
            if model_name is not None and value._model_name != model_name and value.requires_grad:
                raise ValueError(
                    "Cannot retokenize a gradient-carrying TensorString for a different model."
                )
            if model_name is not None and value._model_name != model_name:
                return cls.from_string(str(value), model_name=model_name)
            return cls.from_tensor_string(value)
        return cls.from_string(str(value), model_name=model_name)

    def format(self) -> str:
        return self.text

    @property
    def tensor_string(self) -> TensorString:
        if self._tensor_string is None:
            self._tensor_string = TensorString(self.text, model_name=self.model_name)
        return self._tensor_string

    @property
    def tensor(self):
        return self.tensor_string.tensor

    @property
    def requires_grad(self) -> bool:
        return self._tensor_string.requires_grad if self._tensor_string is not None else False

    @property
    def device(self):
        return self.tensor_string.device

    def requires_grad_(self, requires_grad: bool = True) -> "SoftStr":
        self.tensor_string.requires_grad_(requires_grad)
        self.text = str(self.tensor_string)
        return self

    def to_tensor_string(
        self,
        model_name: Optional[str] = None,
        device: Optional[Any] = None,
    ) -> TensorString:
        target_model_name = model_name or self.model_name
        if self._tensor_string is not None:
            current_model_name = self._tensor_string._model_name
            if target_model_name is not None and current_model_name != target_model_name:
                if self._tensor_string.requires_grad:
                    raise ValueError(
                        "Cannot retokenize a gradient-carrying SoftStr for a different model."
                    )
                tensor_string = TensorString(
                    self.text,
                    model_name=target_model_name,
                    device=device,
                )
                return tensor_string
            if device is not None:
                return self._tensor_string.to(device)
            return self._tensor_string
        return TensorString(self.text, model_name=target_model_name, device=device)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"SoftStr(text={self.text!r}, model_name={self.model_name!r})"

