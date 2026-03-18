"""
Differentiable language model implementation using input embeddings for gradient flow.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import dspy
from dspy.clients.lm import BaseLM
from typing import Any, Dict, List, Optional, Union
import warnings

from .dspy_adapter import SDLMChatAdapter
from .soft_str import SoftStr
from .tensor_string import TensorString

class DifferentiableLM(BaseLM):
    """
    Differentiable language model that preserves gradients through generation.
    
    Uses transformer input_embeds parameter to accept soft token distributions
    and maintain gradient flow for gradient-based prompt optimization.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        temperature: float = 1.0,
        max_new_tokens: int = 50,
        max_tokens: int = 1000,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize differentiable language model.
        
        Args:
            model_name: HuggingFace model name
            temperature: Temperature for generation (higher = more random)
            max_new_tokens: Maximum number of tokens to generate
            max_tokens: Maximum number of tokens to generate
            device: Device to run model on (auto-detect if None)
        """
        """
        super().__init__(
            model_name=model_name,
            device=device,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        """
        self.model_type = "chat"
        self.cache = True
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []
     
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model and tokenizer
        print(f"Loading {model_name} on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Handle missing pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode for inference
        
        print(f"✅ {model_name} loaded successfully")
    
    def __call__(
        self,
        prompt=None,
        messages=None,
        **kwargs
    ) -> List[TensorString]:
        """
        Main DSPy interface
        """
        if messages is not None:
            prompt = self._messages_to_prompt(messages)
        elif prompt is None:
            raise ValueError("Prompt or messages must be provided.")

        prompt = self._coerce_to_tensor_string(prompt)
        
        max_tokens = kwargs.pop('max_tokens', self.kwargs['max_tokens'])
        temperature = kwargs.pop('temperature', self.kwargs['temperature'])
        n = kwargs.pop('n', 1) # number of completions

        # Our gradient-preserving generation
        responses = []
        for _ in range(n):
            response = self._generate_with_gradients(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )
            responses.append(response)
        
        self._log_to_history(prompt, responses, kwargs)
        
        return responses  # Return TensorString
    
    def basic_request(
        self,
        prompt,
        **kwargs
    ) -> List[TensorString]:
        """
        Legacy interface - may not be used by modern DSPy
        """
        return self.__call__(prompt, **kwargs)
    
    def _messages_to_prompt(
        self,
        messages: List[Dict[str, Any]]
    ) -> TensorString:
        """
        Convert chat messages format to prompt string.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            str: Formatted prompt string
        """
        prompt_parts: List[TensorString] = []
        prompt_text_parts: List[str] = []

        for message in messages:
            role = message.get("role", "user")
            raw_content = message.get("content", "")
            content = self._coerce_to_tensor_string(raw_content)
            role_prefix = TensorString(f"{role.capitalize()}: ", model_name=self.model_name, device=self.device)
            prompt_parts.append(role_prefix + content)
            prompt_text_parts.append(f"{role.capitalize()}: {self._render_content_as_text(raw_content)}")

        prompt_parts.append(TensorString("Assistant:", model_name=self.model_name, device=self.device))
        prompt_text_parts.append("Assistant:")
        prompt = TensorString("\n", model_name=self.model_name, device=self.device).join(prompt_parts)
        object.__setattr__(prompt, "_cached_str", "\n".join(prompt_text_parts))
        return prompt

    def _coerce_to_tensor_string(
        self,
        value: Any,
    ) -> TensorString:
        if isinstance(value, SoftStr):
            return value.to_tensor_string(model_name=self.model_name, device=self.device)
        if isinstance(value, TensorString):
            if value._model_name != self.model_name:
                if value.requires_grad:
                    raise ValueError(
                        "Gradient-carrying TensorString inputs must use the same tokenizer as the LM."
                    )
                return TensorString(str(value), model_name=self.model_name, device=self.device)
            return value.to(self.device)
        if isinstance(value, list):
            return self._content_blocks_to_tensor_string(value)
        return TensorString(str(value), model_name=self.model_name, device=self.device)

    def _content_blocks_to_tensor_string(self, content_blocks: List[Any]) -> TensorString:
        parts: List[TensorString] = []
        for block in content_blocks:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(
                        TensorString(
                            block.get("text", ""),
                            model_name=self.model_name,
                            device=self.device,
                        )
                    )
                else:
                    parts.append(
                        TensorString(
                            str(block),
                            model_name=self.model_name,
                            device=self.device,
                        )
                    )
            else:
                parts.append(self._coerce_to_tensor_string(block))
        if not parts:
            return TensorString("", model_name=self.model_name, device=self.device)
        result = parts[0]
        for part in parts[1:]:
            result = result + part
        return result

    def _render_content_as_text(self, value: Any) -> str:
        if isinstance(value, SoftStr):
            return value.text
        if isinstance(value, TensorString):
            return str(value)
        if isinstance(value, list):
            text_parts = []
            for block in value:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    else:
                        text_parts.append(str(block))
                else:
                    text_parts.append(self._render_content_as_text(block))
            return "".join(text_parts)
        return str(value)
    
    def _generate_with_gradients(
        self,
        prompt: TensorString,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> TensorString:
        """
        Generate tokens while preserving gradients through input_embeds.
        
        Args:
            prompt: TensorString prompt with potential gradients
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            **kwargs: Additional arguments to pass to model
            
        Returns:
            TensorString with generated content and preserved gradients
        """
        try:
            # Handle empty prompts
            if prompt.tensor.shape[0] == 0:
                return self._generate_from_empty_prompt(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            
            # Convert TensorString to input embeddings
            input_embeds = self._tensor_string_to_embeddings(prompt)
            
            # Generate using input_embeds (preserves gradients)
            with torch.set_grad_enabled(True):
                generated_embeds = self._generate_embeddings_autoregressive(
                    input_embeds=input_embeds,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                
                # Convert back to TensorString
                return self._embeddings_to_tensor_string(
                    embeddings=generated_embeds,
                    model_name=prompt._model_name
                )
        
        except Exception as e:
            warnings.warn(f"Generation failed: {e}. Returning fallback response.")
            return self._create_fallback_response(prompt, max_tokens)
    
    def _tensor_string_to_embeddings(
        self,
        tensor_string: TensorString
    ) -> torch.Tensor:
        """
        Convert TensorString to input embeddings for transformer.
        
        This is where the magic happens - we use matrix multiplication to convert
        soft token distributions to embeddings while preserving gradients.
        
        Args:
            tensor_string: Input TensorString with shape (seq_len, vocab_size)
            
        Returns:
            Embeddings tensor with shape (1, seq_len, embed_dim)
        """
        # Get embedding matrix from model
        embedding_matrix = self.model.get_input_embeddings().weight  # (vocab_size, embed_dim)
        
        # Convert soft tokens to embeddings via matrix multiplication
        # This preserves gradients: d/dx[softmax(x) @ W] flows back to x
        input_embeds = torch.matmul(tensor_string.tensor, embedding_matrix)  # (seq_len, embed_dim)
        
        # Add batch dimension for transformer
        input_embeds = input_embeds.unsqueeze(0)  # (1, seq_len, embed_dim)
        
        return input_embeds.to(self.device)
    
    def _generate_embeddings_autoregressive(
        self,
        input_embeds: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        annealing: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate token embeddings autoregressively using input_embeds.
        
        Args:
            input_embeds: Input embeddings (1, seq_len, embed_dim)
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            annealing: Whether to use annealing
            **kwargs: Additional arguments to pass to model
            
        Returns:
            Generated embeddings including input + generated tokens
        """
        batch_size, seq_len, embed_dim = input_embeds.shape
        embedding_matrix = self.model.get_input_embeddings().weight
        
        # Start with input embeddings
        current_embeds = input_embeds
        all_embeds = [current_embeds]
        
        # Generate tokens one by one
        for step in range(max_new_tokens):
            # Forward pass through transformer
            outputs = self.model(
                inputs_embeds=current_embeds,
                return_dict=True,
                use_cache=False,  # Disable cache for gradient flow
                **kwargs,
            )
            
            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Convert to probabilities using Gumbel softmax for differentiability
            if annealing:
                tau = max(0.1, temperature * (0.9 ** step))
            else:
                tau = temperature
            
            next_token_probs = F.gumbel_softmax(
                next_token_logits, 
                tau=tau,
                hard=False,  # Keep soft for gradients
                dim=-1
            )
            
            # Convert probabilities back to embeddings
            next_token_embed = torch.matmul(
                next_token_probs.unsqueeze(1),  # (batch_size, 1, vocab_size)
                embedding_matrix.unsqueeze(0)   # (1, vocab_size, embed_dim)
            )  # (batch_size, 1, embed_dim)
            
            # Append to sequence
            current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)
            all_embeds.append(next_token_embed)
            
            # Optional: Early stopping based on EOS token probability
            eos_prob = next_token_probs[0, self.tokenizer.eos_token_id].item()
            if eos_prob > 0.8:  # High confidence EOS
                break
        
        # Return full sequence of embeddings
        return torch.cat(all_embeds, dim=1)  # (batch_size, total_seq_len, embed_dim)
    
    def _embeddings_to_tensor_string(
        self,
        embeddings: torch.Tensor,
        model_name: str
    ) -> TensorString:
        """
        Convert embeddings back to TensorString.
        
        Args:
            embeddings: Generated embeddings (1, seq_len, embed_dim)
            model_name: Model name for TensorString
            
        Returns:
            TensorString with preserved gradients
        """
        # Remove batch dimension
        embeddings = embeddings.squeeze(0)  # (seq_len, embed_dim)
        
        # Project embeddings back to vocabulary space
        embedding_matrix = self.model.get_input_embeddings().weight  # (vocab_size, embed_dim)
        
        # Compute logits by matrix multiplication with embedding matrix transpose
        logits = torch.matmul(embeddings, embedding_matrix.T)  # (seq_len, vocab_size)
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Create TensorString from probabilities
        return TensorString(tensor=probs, model_name=model_name, device=self.device)
    
    def _generate_from_empty_prompt(
        self,
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> TensorString:
        """Generate from empty prompt (unconditional generation)."""
        # Create minimal starting embedding
        embed_dim = self.model.config.hidden_size
        start_embed = torch.randn(1, 1, embed_dim, device=self.device) * 0.1
        
        # Generate
        generated_embeds = self._generate_embeddings_autoregressive(
            input_embeds=start_embed,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        
        return self._embeddings_to_tensor_string(generated_embeds, self.model_name)
    
    def _create_fallback_response(
        self,
        prompt: TensorString,
        max_tokens: int
    ) -> TensorString:
        """Create fallback response when generation fails."""
        fallback_text = f"[Generation failed for prompt: {str(prompt)[:50]}...]"
        return TensorString(fallback_text, model_name=prompt._model_name, device=self.device)

    def _log_to_history(
        self,
        prompt: Union[str, TensorString],
        responses: List[TensorString],
        kwargs: Dict
    ):
        """
        Log interaction to history (DSPy requirement).
        WARNING: make sure to log actually strings, not TensorString objects!
        This is in order to not have PyTorch tensors in the history, as it could
        cause a memory leak.

        Args:
            prompt: Prompt string or TensorString
            responses: List of TensorString responses
            kwargs: Generation parameters
        """
        history_entry = {
            "prompt": str(prompt),
            "responses": [str(response) for response in responses],
            "kwargs": kwargs,
            "model": self.model_name
        }
        self.history.append(history_entry)
        
        # Also update global history if available
        if hasattr(self, 'update_global_history'):
            self.update_global_history(history_entry)
        
    def generate_batch(
        self,
        prompts: List[TensorString],
        **kwargs
    ) -> List[TensorString]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of TensorString prompts
            **kwargs: Generation parameters
            
        Returns:
            List of TensorString responses
        """
        responses = []
        
        for prompt in prompts:
            response = self._generate_with_gradients(
                prompt, 
                kwargs.get('max_tokens', self.max_new_tokens),
                kwargs.get('temperature', self.temperature)
            )
            responses.append(response)
        
        self._log_to_history(prompts, responses, kwargs)
        return responses
    
    # ==========================================
    # Additional DSPy Interface Methods
    # ==========================================
    
    def generate(
        self,
        prompt: Union[str, TensorString],
        **kwargs
    ) -> TensorString:
        """
        Simple generate interface for direct use.
        
        Args:
            prompt: Input prompt string
            **kwargs: Generation parameters
            
        Returns:
            TensorString: Generated response
        """
        responses = self.__call__(prompt=prompt, **kwargs)
        return responses[0] if responses else ""
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> TensorString:
        """
        Chat interface for conversation format.
        
        Args:
            messages: List of message dicts
            **kwargs: Generation parameters
            
        Returns:
            TensorString: Generated response
        """
        responses = self.__call__(messages=messages, **kwargs)
        return responses[0] if responses else ""
    
    # ==========================================
    # Configuration and Utility Methods
    # ==========================================

    def set_temperature(
        self,
        temperature: float
    ):
        """
        Set generation temperature.
        """
        self.temperature = temperature
    
    def set_max_tokens(
        self,
        max_tokens: int
    ):
        """
        Set maximum generation length.
        """
        self.max_new_tokens = max_tokens
    
    def to(
        self,
        device: str
    ):
        """
        Move model to specified device.
        """
        self.device = torch.device(device)
        self.model.to(self.device)
        return self
    
    def eval(self):
        """
        Set model to evaluation mode.
        """
        self.model.eval()
        return self
    
    def train(self):
        """
        Set model to training mode.
        """
        self.model.train()
        return self
    
    @property
    def vocab_size(self) -> int:
        """
        Get vocabulary size.
        """
        return self.tokenizer.vocab_size
    
    def __repr__(self):
        """
        String representation.
        """
        return (f"DifferentiableLM(model='{self.model_name}', "
                f"device='{self.device}', temp={self.temperature}, "
                f"max_tokens={self.max_new_tokens})")

# ==========================================
# Utility Functions
# ==========================================

def create_differentiable_lm(
    model_name: str = "gpt2",
    **kwargs
) -> DifferentiableLM:
    """
    Factory function for creating differentiable language models.
    
    Args:
        model_name: HuggingFace model name
        **kwargs: Additional parameters for DifferentiableLM
        
    Returns:
        Configured DifferentiableLM instance
    """
    return DifferentiableLM(model_name, **kwargs)

def setup_dspy_with_differentiable_lm(
    model_name: str = "gpt2",
    **kwargs
):
    """
    Setup DSPy with a differentiable language model.
    
    Args:
        model_name: HuggingFace model name
        **kwargs: Additional parameters for DifferentiableLM
        
    Returns:
        Configured DifferentiableLM instance
    """
    lm = create_differentiable_lm(model_name, **kwargs)
    dspy.configure(lm=lm, adapter=SDLMChatAdapter())
    print(f"✅ DSPy configured with DifferentiableLM({model_name}) and SDLMChatAdapter")
    return lm
