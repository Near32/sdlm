"""
SDLM (Soft Differentiable Language Models)

A library for differentiable text generation using Straight-Through Gumbel-Softmax.
This module provides tensor-centric enhanced strings where PyTorch tensors are the 
primary representation, with full gradient flow preservation through most string
 operations.
"""

from .stgs import STGS
from .stgs_diff_model import STGSDiffModel
from .stgs_diff_string import STGSDiffString

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Union, List
import warnings

from .core.tensor_string import TensorString
from .core.patching import DSPyPatcher, TensorStringContext
from .core.soft_str import SoftStr
from .core.dspy_adapter import SDLMChatAdapter
from .core.differentiable_lm import DifferentiableLM, setup_dspy_with_differentiable_lm
from .utils.factory_functions import (
    TensorStr,
    from_string,
    from_tensor,
    from_token_ids,
    empty_tensor_string,
)

# ==========================================
# Main SDLM Manager Class
# ==========================================

class SDLMManager:
    """
    Main SDLM system manager with DSPy compatibility helpers.
    
    This class manages the activation/deactivation of TensorString behavior
    and provides fine-grained control over which packages are affected.
    """
    
    def __init__(self):
        self.dspy_patcher = DSPyPatcher()
        self.context_manager = TensorStringContext()
        self.is_dspy_patched = False
        self.default_model = "distilbert/distilgpt2"
        self.default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configuration
        self._configure_defaults()
    
    def _configure_defaults(self):
        """
        Configure default settings.
        """
        TensorString._default_model_name = self.default_model
        TensorString.set_global_device(self.default_device)
    
    def configure_default_model(
        self, 
        model_name: str,
        model: Optional[AutoModelForCausalLM] = None,
        default: Optional[str] = 'default',
    ):
        """
        Configure the default model for TensorString operations.
        """
        self.default_model = model_name
        TensorString.configure_model(model_name=model_name, model=model)
        print(f"SDLM: Configured {default} model: {model_name}")
    
    def configure_fluency_model(
        self,
        model_name: str,
        model: Optional[AutoModelForCausalLM] = None,
    ):
        """
        Configure the fluency model for TensorString and STGSDiffString.
        """
        # Fluency model is the default model
        self.configure_default_model(model_name, model=model, default='fluency')
                 
    def get_fluency_model(self) -> AutoModelForCausalLM:
        return TensorString.fluency_model()

    def get_fluency_tokenizer(self) -> AutoTokenizer:
        return TensorString.fluency_tokenizer()
    
    def get_default_model(self) -> AutoModelForCausalLM:
        if self.default_model not in TensorString._model_cache:
            TensorString.configure_model(self.default_model)
        return TensorString._model_cache[self.default_model]
    
    def get_default_tokenizer(self) -> AutoTokenizer:
        if self.default_model not in TensorString._tokenizer_cache:
            TensorString.configure_model(self.default_model)
        return TensorString._tokenizer_cache[self.default_model]
    
    def set_device(
        self, 
        device: Union[str, torch.device],
    ):
        """
        Set the default device for tensor operations.
        """
        self.default_device = torch.device(device)
        TensorString.set_global_device(self.default_device)
        print(f"SDLM: Set device: {self.default_device}")
    
    def activate_for_dspy(self):
        """
        Configure DSPy to use SDLM's adapter path instead of monkey-patching.
        """
        if self.is_dspy_patched:
            print("SDLM: DSPy integration already active")
            return

        import dspy

        warnings.warn(
            "sdlm.activate_for_dspy() now configures the SDLMChatAdapter. "
            "The old monkey-patching path is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        dspy.configure(adapter=SDLMChatAdapter())
        self.is_dspy_patched = True
        print("✅ SDLM: Activated DSPy integration with SDLMChatAdapter")
    
    def deactivate_dspy(self):
        """
        Disable the SDLM DSPy adapter and restore unpatched DSPy settings.
        """
        if not self.is_dspy_patched:
            print("SDLM: DSPy integration not active")
            return

        import dspy

        dspy.configure(adapter=None)
        if self.dspy_patcher.is_active:
            self.dspy_patcher.deactivate()
        self.is_dspy_patched = False
        print("✅ SDLM: DSPy integration deactivated")
    
    def with_tensor_strings(self):
        """
        Get context manager for temporary global TensorString activation.
        
        Usage:
            with sdlm.with_tensor_strings():
                text = "This becomes TensorString"
        """
        return self.context_manager
    
    @property
    def is_active(self):
        """
        Check if any patching is currently active.
        """
        return self.is_dspy_patched
    
    @property
    def status(self):
        """
        Get current status of SDLM system.
        """
        return {
            "dspy_patched": self.is_dspy_patched,
            "default_model": self.default_model,
            "device": str(self.default_device),
            "tensor_string_available": True
        }

# ==========================================
# Global SDLM Instance
# ==========================================

# Create global SDLM instance
_manager = SDLMManager()

# ==========================================
# Public API Functions
# ==========================================

def activate_for_dspy():
    """
    Activate SDLM's DSPy adapter path.
    
    This is the recommended way to enable gradient-based prompt optimization
    while preserving compatibility with other packages.
    """
    _manager.activate_for_dspy()

def deactivate():
    """
    Deactivate SDLM's DSPy integration helpers.
    """
    _manager.deactivate_dspy()

def with_tensor_strings():
    """
    Context manager for temporary TensorString activation.
    
    Example:
        with sdlm.with_tensor_strings():
            text = "This becomes a TensorString"
            assert isinstance(text, TensorString)
    """
    return _manager.with_tensor_strings()

def configure_default_model(
    model_name: str,
    model: Optional[AutoModelForCausalLM] = None,
    ):
    """
    Configure the default model for all TensorString operations.
    """
    _manager.configure_default_model(model_name, model=model)

def set_device(device: Union[str, torch.device]):
    """
    Set the default device for tensor operations.
    """
    _manager.set_device(device)

def get_status():
    """
    Get current status of the SDLM system.
    """
    return _manager.status


# ==========================================
# Factory Functions
# ==========================================

def soft_str(
    text: str,
    model_name: Optional[str] = None,
    requires_grad: bool = False,
) -> SoftStr:
    """
    Create a `SoftStr` value for DSPy signatures.
    """
    value = SoftStr.from_string(text, model_name=model_name)
    if requires_grad:
        value.requires_grad_(True)
    return value

def from_string(
    text: str, 
    model_name: Optional[str] = None, 
    requires_grad: bool = False, 
    device: Optional[torch.device] = None,
) -> TensorString:
    """
    Create a TensorString from text.
    
    Args:
        text: Input text
        model_name: Model for tokenization (uses default if None)
        requires_grad: Whether tensor should require gradients
        device: Device for tensor (uses default if None)
        
    Returns:
        TensorString with gradient capabilities
    """
    result = TensorString(value=text, model_name=model_name, device=device)
    if requires_grad:
        result.requires_grad_(True)
    return result

def from_tensor(
    tensor: torch.Tensor, 
    model_name: Optional[str] = None, 
    requires_grad: Optional[bool] = None,
) -> TensorString:
    """
    Create a TensorString from a PyTorch tensor.
    
    Args:
        tensor: Input tensor (1D token IDs or 2D one-hot/probabilities)
        model_name: Model for decoding (uses default if None)
        requires_grad: Whether to enable gradients (preserves if None)
        
    Returns:
        TensorString from tensor
    """
    if requires_grad is not None:
        tensor = tensor.clone().requires_grad_(requires_grad)
    
    return TensorString(tensor=tensor, model_name=model_name)

def from_token_ids(
    token_ids: List[int], 
    model_name: Optional[str] = None, 
    requires_grad: bool = False,
) -> TensorString:
    """
    Create a TensorString from token IDs.
    
    Args:
        token_ids: List of token IDs
        model_name: Model to use (uses default if None)
        requires_grad: Whether tensor should require gradients
        
    Returns:
        TensorString from token IDs
    """
    device = _manager.default_device
    tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
    
    if requires_grad:
        # Convert to one-hot for gradient support
        # Vocab size will be determined by TensorString
        pass  # TensorString constructor will handle this
    
    return TensorString(tensor=tensor, model_name=model_name)

def empty_tensor_string(
    length: int, 
    model_name: Optional[str] = None, 
    requires_grad: bool = True,
) -> TensorString:
    """
    Create an empty learnable TensorString for gradient-based generation.
    
    Args:
        length: Sequence length
        model_name: Model to use (uses default if None)
        requires_grad: Whether tensor should require gradients
        
    Returns:
        Empty learnable TensorString
    """
    model_name = model_name or _manager.default_model
    device = _manager.default_device
    
    # Get vocab size from temporary instance
    temp = TensorString(value="", model_name=model_name)
    vocab_size = temp.vocab_size
    
    # Create random learnable tensor
    tensor = torch.randn(length, vocab_size, requires_grad=requires_grad, device=device)
    tensor = F.softmax(tensor, dim=1)  # Initialize as probability distribution
    
    return TensorString(tensor=tensor, model_name=model_name)

# ==========================================
# Differentiable Language Model Factory
# ==========================================

def create_differentiable_lm(model_name: str = "distilbert/distilgpt2", **kwargs) -> DifferentiableLM:
    """
    Create a differentiable language model for use with DSPy.
    
    Args:
        model_name: HuggingFace model name
        **kwargs: Additional arguments for DifferentiableLM
        
    Returns:
        DifferentiableLM: Language model with gradient flow support
    """
    return DifferentiableLM(model_name, **kwargs)

# ==========================================
# Batch Operations
# ==========================================

def batch_from_strings(texts: List[str], model_name: Optional[str] = None,
                      requires_grad: bool = False, device: Optional[torch.device] = None,
                      max_length: Optional[int] = None, padding: bool = True) -> List[TensorString]:
    """
    Create multiple TensorStrings efficiently from a list of texts.
    
    Args:
        texts: List of text strings
        model_name: Model to use for tokenization
        requires_grad: Whether tensors should require gradients
        device: Device to place tensors on
        max_length: Maximum sequence length for padding/truncation
        padding: Whether to pad sequences to max_length
        
    Returns:
        List[TensorString]: List of TensorString instances
    """
    if not texts:
        return []
    
    device = device or _sdlm_instance.device
    model_name = model_name or _sdlm_instance.default_model
    
    # Create first string to get tokenizer
    first_string = create_tensor_string_safe(texts[0], model_name, device)
    tokenizer = first_string.tokenizer
    
    # Batch tokenization
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=padding,
        truncation=max_length is not None,
        max_length=max_length
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_masks = encoded.get('attention_mask', None)
    if attention_masks is not None:
        attention_masks = attention_masks.to(device)
    
    # Create TensorString instances
    results = []
    for i in range(len(texts)):
        token_ids = input_ids[i]
        attention_mask = attention_masks[i] if attention_masks is not None else None
        
        # Convert to one-hot
        one_hot = F.one_hot(token_ids, num_classes=first_string.vocab_size).float()
        if requires_grad:
            one_hot.requires_grad_(True)
        
        tensor_string = TensorString(
            tensor=one_hot,
            model_name=model_name,
            attention_mask=attention_mask,
            device=device
        )
        results.append(tensor_string)
    
    return results

# ==========================================
# Optimization Utilities
# ==========================================

def create_prompt_optimizer(learning_rate: float = 0.01):
    """Create an optimizer for learnable prompt components."""
    from .optimization.prompt_optimizer import PromptOptimizer
    return PromptOptimizer(learning_rate)

def differentiable_similarity(str1: TensorString, str2: TensorString) -> torch.Tensor:
    """Compute differentiable similarity between two TensorStrings."""
    return str1.similarity_to(str2)


# ==========================================
# Gradient-Based String Operations
# ==========================================

def differentiable_concatenate(
    tensor_strings: List[TensorString], 
    weights: Optional[torch.Tensor] = None,
) -> TensorString:
    """
    Concatenate TensorStrings with optional learned weights.
    
    Args:
        tensor_strings: List of TensorStrings to concatenate
        weights: Optional weights for each string (should sum to 1)
        
    Returns:
     TensorString: Concatenated result with gradient flow
    """
    if not tensor_strings:
        return empty_tensor_string()
    
    if len(tensor_strings) == 1:
        return tensor_strings[0]
    
    if weights is not None:
        if len(weights) != len(tensor_strings):
            raise ValueError("Number of weights must match number of strings")
        
        # Weighted combination instead of simple concatenation
        # Pad all to same length first
        max_len = max(ts.tensor.shape[0] for ts in tensor_strings)
        padded_strings = [ts.pad_to_length(max_len) for ts in tensor_strings]
        
        # Stack and apply weights
        stacked = torch.stack([ts.tensor for ts in padded_strings])  # (num_strings, max_len, vocab_size)
        weights = weights.view(-1, 1, 1)  # Broadcast over sequence and vocab dimensions
        
        weighted_sum = (stacked * weights).sum(dim=0)  # (max_len, vocab_size)
        
        return TensorString(
            tensor=weighted_sum,
            model_name=tensor_strings[0]._model_name,
            device=tensor_strings[0].device
        )
    else:
        # Simple concatenation
        result = tensor_strings[0]
        for ts in tensor_strings[1:]:
            result = result + ts
        return result

def learnable_string_interpolation(
    str1: TensorString, 
    str2: TensorString, 
    alpha: torch.Tensor,
) -> TensorString:
    """
    Interpolate between two TensorStrings with a learnable parameter.
    
    Args:
        str1: First TensorString
        str2: Second TensorString  
        alpha: Interpolation parameter (0 = str1, 1 = str2)
        
    Returns:
        TensorString: Interpolated result
    """
    # Ensure same length
    max_len = max(str1.tensor.shape[0], str2.tensor.shape[0])
    str1_padded = str1.pad_to_length(max_len)
    str2_padded = str2.pad_to_length(max_len)
    
    # Interpolate in tensor space
    alpha = torch.sigmoid(alpha)  # Constrain to [0, 1]
    interpolated = (1 - alpha) * str1_padded.tensor + alpha * str2_padded.tensor
    
    return TensorString(
        tensor=interpolated,
        model_name=str1._model_name,
        device=str1.device
    )

# ==========================================
# Convenience Aliases
# ==========================================

# Shorter aliases for common functions
String = from_string  # Alias for backwards compatibility
TensorStr = from_string  # Alternative name
SoftText = SoftStr  # Backwards-compatible alias while SoftStr is adopted.

# ==========================================
# Module Configuration
# ==========================================

TensorString._default_model_name = _manager.default_model

# Auto-detect and configure device
if torch.cuda.is_available():
    set_device('cuda')
    print(f"SDLM configured with CUDA device: {torch.cuda.get_device_name()}")
else:
    set_device('cpu')
    print("SDLM configured with CPU device")

# ==========================================
# Module Exports
# ==========================================

__all__ = [
    # Core classes
    'TensorString',
    'SoftStr',
    'SDLMChatAdapter',
    'SDLMManager',
    'DifferentiableLM',
    
    # Activation/deactivation
    'activate_for_dspy',
    'deactivate',
    'with_tensor_strings',
    
    # Configuration
    'configure_default_model',
    'set_device',
    'get_status',
    
    # Factory functions
    'soft_str',
    'from_string',
    'from_tensor', 
    'from_token_ids',
    'empty_tensor_string',
    'String',
    'TensorStr',
    'SoftText',
    
    # Batch operations
    'batch_from_strings',
    
    # Language models
    'create_differentiable_lm',
    'setup_dspy_with_differentiable_lm',
    
    # Optimization
    'create_prompt_optimizer',
    'differentiable_similarity',
]

# ==========================================
# Version and Metadata
# ==========================================

__version__ = "2.1.0"
__author__ = "SDLM Development Team"
__email__ = "sdlm@example.com"
__description__ = "Soft Differentiable Language Models with DSPy integration"
__url__ = "https://github.com/sdlm/sdlm"

# ==========================================
# Module Initialization Message
# ==========================================

def _print_welcome_message():
    """Print welcome message with usage instructions."""
    print("=" * 60)
    print("🚀 SDLM - v" + __version__)
    print("=" * 60)
    #print("📖 Quick Start:")
    #print("   import sdlm")
    #print("   sdlm.activate_for_dspy()  # Enable for DSPy only")
    #print("   # Your DSPy code now has gradient flow!")
    #print()
    #print("🔧 Configuration:")
    print(f"   Model: {_manager.default_model}")
    print(f"   Device: {_manager.default_device}")
    #print()
    #print("📚 Documentation: https://sdlm.readthedocs.io")
    print("=" * 60)

# Print welcome message on import (can be disabled with environment variable)
import os
if not os.environ.get('SDLM_QUIET', False):
    _print_welcome_message()

# ==========================================
# Optional: Auto-activation for DSPy
# ==========================================

# Check if user wants auto-activation
if os.environ.get('SDLM_AUTO_ACTIVATE_DSPY', False):
    print("SDLM: Auto-activating DSPy integration...")
    activate_for_dspy()
