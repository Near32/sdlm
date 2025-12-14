"""
TextGrad-style differentiable text variables for SDLM.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable
from transformers import PreTrainedTokenizer
from sdlm.stgs_diff_string import STGSDiffString


class Variable(STGSDiffString):
    """
    A differentiable text variable that supports gradient-based optimization.
    
    This class extends STGSDiffString to add gradient tracking and optimization
    capabilities similar to TextGrad.
    """
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        initial_string: Optional[str] = None,
        init_strategy: Optional[str] = "random",
        initial_ids: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        temperature: float = 0.1,
        logit_scaler: float = 10.0,
        hard: bool = False,
        learnable_temperature: bool = False,
        decouple_learnable_temperature: bool = False,
        device: Optional[str] = None,
        constraint: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize a Variable.
        
        Args:
            tokenizer: Tokenizer to use for text processing
            initial_string: Initial text content
            init_strategy: strategy to initialize the variable's logits ("random", "fluency").
            initial_ids: Initial token IDs
            name: Optional name for the variable (for debugging)
            temperature: Sampling temperature for Gumbel-Softmax
            hard: Whether to use hard sampling
            learnable_temperature: Whether to make temperature learnable
            device: Device to use (cuda/cpu)
            constraint: Optional constraint function to apply to gradients
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        super().__init__(
            initial_string=initial_string,
            initial_ids=initial_ids,
            init_strategy=init_strategy,
            tokenizer=tokenizer,
            temperature=temperature,
            logit_scaler=logit_scaler,
            hard=hard,
            learnable_temperature=learnable_temperature,
            decouple_learnable_temperature=learnable_temperature,
            device=device
        )
        
        self.name = name or f"var_{id(self)}"
        self._grad = None
        self._constraint = constraint

    def reset(
        self,
        initial_string: Optional[str] = None,
        initial_ids: Optional[torch.Tensor] = None,
    ):
        """
        Reset the Variable to a new initial state.
        
        Args:
            initial_string: The new initial string to represent
            initial_ids: The new initial token IDs to represent
        """
        super().reset(initial_string, initial_ids)
    
    def backward(self, grad_output: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute gradient with respect to the string's logits.
        
        Args:
            grad_output: Upstream gradient
            
        Returns:
            Gradient tensor
        """
        if grad_output is not None:
            self._grad = grad_output
            
            # Apply constraint if provided
            if self._constraint is not None:
                self._grad = self._constraint(self._grad)
                
        return self._grad
        
    def apply_gradient(self, lr: float = 0.1) -> None:
        """
        Apply the accumulated gradient to update the string.
        
        Args:
            lr: Learning rate for the update
        """
        if self._grad is not None:
            with torch.no_grad():
                self.logits.data.add_(self._grad * lr)
            
    def zero_grad(self) -> None:
        """Reset the gradient buffer."""
        self._grad = None
        
    def add_constraint(self, constraint: Callable) -> None:
        """
        Add a constraint function that modifies the gradient.
        
        Args:
            constraint: Function that takes a gradient tensor and returns a modified gradient tensor
        """
        self._constraint = constraint
        
    def __call__(self) -> str:
        """Return the current string value."""
        return self.get_string()
        
    def __str__(self) -> str:
        """String representation of the variable."""
        return f"Variable(name='{self.name}', value='{self.get_string()}')"
        
    def clone(self) -> 'Variable':
        """
        Create a deep copy of this Variable.
        
        Returns:
            A new Variable instance with the same parameters and state.
        """
        # Create a new instance with the same parameters
        clone = Variable(
            initial_string=self.get_string(),
            tokenizer=self.tokenizer,
            name=f"{self.name}_clone",
            temperature=self.stgs.init_temperature.item() \
                if isinstance(self.stgs.init_temperature, torch.Tensor) \
                else self.stgs.init_temperature,
            hard=self.stgs.stgs_hard,
            learnable_temperature=self.stgs.learnable_temperature,
            device=self.device,
            constraint=self._constraint
        )
        
        # Copy the logits
        with torch.no_grad():
            clone.logits.data.copy_(self.logits.data)
            
        # Copy the gradient if it exists
        if self._grad is not None:
            clone._grad = self._grad.clone()
            
        return clone
