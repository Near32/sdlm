import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from transformers import PreTrainedTokenizer
from .stgs import STGS


class STGSDiffString(nn.Module):
    """
    A class that represents a differentiable string using the STGS (Straight-Through Gumbel-Softmax)
    operation. It maintains both a one-hot encoded tensor representation of tokens and the 
    corresponding string. It does not include special tokens (e.g. BOS, EOS, etc.) in the 
    input_ids and logits.
    """
    
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        initial_string: Optional[str] = None,
        initial_ids: Optional[torch.Tensor] = None,
        init_strategy: str = "random",
        logit_scaler: float = 10.0,
        temperature: float = 0.1,
        hard: bool = False,
        learnable_temperature: bool = False,
        decouple_learnable_temperature: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """
        Initialize the STGSDiffString.
        
        Args:
            tokenizer: A tokenizer with encode() and decode() methods
            initial_string: The initial string to represent
            initial_ids: The initial token IDs to represent
            init_strategy: The initialization strategy to use
            logit_scaler: Scalar to scale the logits
            temperature: Initial temperature for Gumbel-Softmax
            hard: If True, uses straight-through estimator with hard samples
            learnable_temperature: If True, makes temperature a learnable parameter
            device: Device to store tensors on
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.decouple_learnable_temperature = decouple_learnable_temperature

        assert initial_string is not None or initial_ids is not None
        if initial_ids is not None:
            self.input_ids = initial_ids.to(device)
            initial_string = tokenizer.decode(initial_ids[0])
        else:
            # Encode the initial string to get token IDs
            self.input_ids = tokenizer.encode(
                initial_string, 
                return_tensors='pt',
                add_special_tokens=False,
        ).to(device)
        
        # Initialize parameters
        self.logit_scaler = logit_scaler
        self.vocab_size = len(tokenizer) #tokenizer.vocab_size
        self.seq_len = len(self.input_ids[0])
        
        # Initialize logits for the one-hot distribution
        self.init_strategy = init_strategy
        if "random" in self.init_strategy:
            # Start with one-hot encoding of the input_ids
            logits = self.logit_scaler*F.one_hot(self.input_ids, num_classes=self.vocab_size).float().to(device)
            if "normalised" in self.init_strategy:
                logits = (logits-logits.mean())/logits.std()
            self.logits = nn.Parameter(
                logits.detach(),
                requires_grad=True,
            )
        elif "fluency" in self.init_strategy:
            # Start with a logit distribution similar to the fluency model's next_token distribution:
            import sdlm
            fluency_model = sdlm._manager.get_fluency_model()
            fluency_tokenizer = sdlm._manager.get_fluency_tokenizer()
            fm_input = fluency_tokenizer.encode(
                initial_string, 
                return_tensors='pt',
                add_special_tokens=False,
            ).to(fluency_model.device)
            fm_output = fluency_model(
                input_ids=fm_input,
                return_dict=True,
            )
            # remove batch dimension and last token prediction: 
            logits = fm_output.logits[0][:-1].to(self.input_ids.device)
            # add initial token as max logit:
            logit_max = logits.max()
            first_token_logits = logits.mean()*torch.ones((1, self.vocab_size), device=logits.device)
            first_token_logits[0,self.input_ids[0][0]] = (1+logit_max)
            # (1, vocab_size)
            logits = torch.cat([
                first_token_logits,
                logits,
            ], dim=0)
            # (1+(seq_len-1), vocab_size)
            # add slight bump on initial_string logits:
            logits = logits.scatter(
                dim=-1,
                index=self.input_ids[0].unsqueeze(1).to(logits.device),
                src=(1+logit_max)*torch.ones_like(logits),
            )
            assert all(logits.argmax(dim=-1) == self.input_ids[0])
            if "unnormalised" not in self.init_strategy:
                logits = (logits-logits.mean())/logits.std()
            self.logits = nn.Parameter(
                logits.detach().to(self.device).unsqueeze(0), # adding the batch dim...
                requires_grad=True,
            )
            
        
        # Initialize STGS module
        self.stgs = STGS(
            vocab_size=self.vocab_size,
            stgs_hard=hard,
            init_temperature=temperature,
            learnable_temperature=learnable_temperature,
            nbr_learnable_temperatures=self.logits.shape[1] if decouple_learnable_temperature else None, 
            device=device
        )

        self.eff_temperature = 0.0
    
    def reset(
        self,
        initial_string: Optional[str] = None,
        initial_ids: Optional[torch.Tensor] = None,
    ):
        """
        Reset the STGSDiffString to a new initial state.
        
        Args:
            initial_string: The new initial string to represent
            initial_ids: The new initial token IDs to represent
        """
        assert initial_string is not None or initial_ids is not None
        if initial_ids is not None:
            self.input_ids = initial_ids.to(self.device)
        else:
            # Encode the initial string to get token IDs
            self.input_ids = self.tokenizer.encode(
                initial_string, 
                return_tensors='pt',
                add_special_tokens=False,
            ).to(self.device)
        
        # Initialize logits for the one-hot distribution
        del self.logits
        self.logits = nn.Parameter(
            self.logit_scaler*F.one_hot(self.input_ids, num_classes=self.vocab_size).float().to(self.device),
            requires_grad=True,
        )
    
    def forward(
        self,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Forward pass that returns both the differentiable input_ids and the decoded string.

        Args:
            temperature: Temperature to use for Gumbel-Softmax sampling

        Returns:
            Tuple containing:
                - input_ids: Differentiable token IDs using STGS
                - one_hot: Differentiable one-hot encoding of the input_ids
                - decoded_string: The decoded string from the current distribution
        """
        # Apply STGS to get differentiable samples
        diff_input_ids, diff_one_hot, eff_temperature, _ = self.stgs(
            self.logits,
            temperature=temperature,
        )
        if self.decouple_learnable_temperature:
            self.eff_temperatures = eff_temperature.mean(dim=0)
        self.eff_temperature = (eff_temperature.sum()/len(eff_temperature)).item()

        # Decode the string
        decoded_string = self.tokenizer.decode(diff_input_ids.long()[0].tolist())
        
        return diff_input_ids, diff_one_hot, decoded_string

    def update(self):
        """Update the current logits based on a previously computed gradient."""
        if self.logits.grad is None:
            raise ValueError("Gradient not computed. Call .backward() on a loss before updating.")
        
        raise NotImplementedError
        #TODO: consider using a learning rate and optimizer
        #TODO: consider adding fluency constraints-derived gradient contributions
        #TODO: consider increasing/decreasing the length of the string based on fluency

        self.logits.data.add_(self.logits.grad.data)
        self.logits.grad.zero_()    
    
    def get_string(self) -> str:
        """Get the current string representation."""
        _, _, decoded_string = self.forward(temperature=0.1)
        return decoded_string
    
    def get_input_ids(self) -> torch.Tensor:
        """Get the current differentiable input_ids."""
        diff_input_ids, _, _ = self.forward(temperature=1.0)
        return diff_input_ids
    
    def get_one_hot(self) -> torch.Tensor:
        """Get the current differentiable one-hot encoding."""
        _, diff_one_hot, _ = self.forward(temperature=1.0)
        return diff_one_hot
    
    def __str__(self) -> str:
        return self.get_string()
    
    def __len__(self) -> int:
        return self.seq_len
    
    def to(self, device, *args, **kwargs):
        """Move the module to the specified device."""
        super().to(device, *args, **kwargs)
        self.device = device
        self.logits.data = self.logits.data.to(device)
        self.input_ids = self.input_ids.to(device)
        self.stgs = self.stgs.to(device)
        return self
    
    def parameters(self, recurse: bool = True):
        """Return the parameters of the model."""
        return [self.logits]
    
    def state_dict(self, *args, **kwargs):
        """Return the state dictionary."""
        return {'logits': self.logits}
    
    def load_state_dict(self, state_dict, strict: bool = True):
        """Load the state dictionary."""
        self.logits.data.copy_(state_dict['logits'])
        
    def extra_repr(self) -> str:
        return f'seq_len={self.seq_len}, vocab_size={self.vocab_size}, device={self.device}'
