# SDLM: Soft Differentiable Language Models

SDLM is a Python library that provides differentiable text generation capabilities using the Straight-Through Gumbel-Softmax trick. It allows for gradient-based optimization through discrete token generation, making it useful for tasks that require end-to-end training with discrete text outputs.

DSPy integration is implemented locally in this repository through `SoftStr`, `SDLMChatAdapter`, and `DifferentiableLM`. DSPy is expected to be installed as a pinned Python dependency rather than vendored as a submodule.

## Features

- `STGS`: Core implementation of the Straight-Through Gumbel-Softmax operation
- `STGSDiffModel`: Wrapper for HuggingFace models to enable differentiable text generation
- `STGSDiffString`: A differentiable string representation for gradient-based text manipulation

## Installation

```bash
pip install -e .
```

## Usage

```python
from sdlm import STGSDiffModel, STGSDiffString
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize with a pretrained model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Create differentiable model
diff_model = STGSDiffModel(
    model=base_model,
    tokenizer=tokenizer,
    temperature=0.7,
    hard=True,
    learnable_temperature=True
)

# Generate text with gradient flow
outputs = diff_model.generate(input_ids=input_ids, max_length=50)
```

## Project Structure

```
sdlm/
├── sdlm/                      # Package source
│   ├── __init__.py
│   ├── stgs_diff_model.py     # STGSDiffModel implementation
│   └── stgs_diff_string.py    # STGSDiffString implementation
├── tests/                     # Unit tests
├── examples/                  # Example scripts
└── README.md                 # This file
```

## Development

### Testing

Run the test suite with:

```bash
pytest tests/
```

### Test Status

#### ✅ Completed Tests

##### STGS Class Tests
- [x] Test initialization with different parameters
- [x] Test forward pass with fixed temperature
- [x] Test forward pass with learnable temperature
- [x] Test gradient flow through STGS operation
- [x] Test hard vs soft sampling modes
- [x] Test device handling (CPU/GPU)

##### STGSDiffModel Tests
- [x] Test initialization with different model types
- [x] Test forward pass with and without labels
- [x] Test gradient flow through generation
- [x] Test batch processing
- [x] Test attention mask handling
- [x] Test with different model architectures

##### STGSDiffString Tests
- [x] Test initialization with different strings
- [x] Test string manipulation operations
- [x] Test gradient flow through string operations
- [x] Test device handling
- [x] Test serialization/deserialization

#### 🚧 Upcoming Tests

##### Enhanced Test Coverage
- [ ] Add comprehensive batch processing tests with varying batch sizes
- [ ] Add edge case tests for empty strings and max sequence lengths
- [ ] Add integration tests for the full text generation pipeline
- [ ] Add performance benchmarks for different model sizes
- [ ] Test compatibility with different tokenizers and model architectures
- [ ] Add tests for mixed-precision training scenarios
- [ ] Test gradient checkpointing for memory efficiency

##### STGSDiffString Tests
- [ ] Test string update strategy based on gradient flow
- [ ] Test strategy to increase string length
- [ ] Test strategy to decrease string length
- [ ] Test gradient flow through STGSDiffModel 

##### Documentation & Examples
- [ ] Add detailed API documentation
- [ ] Create interactive notebooks for key use cases
- [ ] Add example scripts for common NLP tasks
- [ ] Document best practices for training and inference

## License

MIT

## Citation

If you use SDLM in your research, please cite:

```
@misc{sdlm,
  author = {Kevin Denamganaï},
  title = {SDLM: Soft Differentiable Language Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Near32/sdlm}}
}
```
