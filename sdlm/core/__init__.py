"""
sdlm/core/__init__.py

Core components for SDLM - Soft Differentiable Language Models.
"""

from .tensor_string import TensorString
from .soft_str import SoftStr
from .dspy_adapter import SDLMChatAdapter
from .patching import DSPyPatcher, TensorStringContext
from .differentiable_lm import DifferentiableLM

__all__ = [
    'TensorString',
    'SoftStr',
    'SDLMChatAdapter',
    'DSPyPatcher', 
    'TensorStringContext',
    'DifferentiableLM'
]
