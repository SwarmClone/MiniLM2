# AGENTS.md - Guidelines for AI Coding Agents

## Project Overview

MiniLM2 is a PyTorch-based language model training framework implementing nGPT architecture. It uses UV for package management and targets Python 3.10+.

## Build/Lint/Test Commands

### Package Management (UV)
```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>

# Run a command in the virtual environment
uv run <command>

# Activate the virtual environment
source .venv/bin/activate
```

### Running Code
```bash
# Train a model
python -m minilm2.llm.train <config_path>

# Run SFT training
python -m minilm2.llm.sft <config_path>

# Evaluate model
python -m minilm2.llm.eval_pretrained <model_path>

# Run inference
python -m minilm2.llm.infer_sft <model_path>

# Train tokenizer
python -m minilm2.utils.train_tokenizer <text_path> <tokenizer_path>

# Test tokenizer
python -m minilm2.utils.test_tokenizer <tokenizer_path>

# Test model
python -m minilm2.utils.test_pretrain_model <model_path>
python -m minilm2.utils.test_sft_model <model_path>
```

### Testing

This project uses standalone test scripts (not pytest). To run tests:
```bash
# Run individual test scripts
python -m minilm2.utils.test_tokenizer <path>
python -m minilm2.utils.test_pretrain_model <path>
python -m minilm2.utils.test_sft_model <path>
python -m minilm2.utils.test_dialogue_model <path>
python -m minilm2.utils.test_speed
python -m minilm2.utils.test_batch
```

**Note**: No formal test framework is configured. Tests are executable scripts in `minilm2/utils/`.

## Code Style Guidelines

### Imports
- Group imports in this order: stdlib, third-party, local
- Use absolute imports for local modules: `from minilm2.llm.dataset import ...`
- Use relative imports within the same package: `from . import config`
- Wildcard imports are used for utils: `from minilm2.utils.train_utils import *`

### Type Hints
- Use Python 3.10+ type hints: `list[int]`, `str | None`, `dict[str, int]`
- Use `from __future__ import annotations` if needed for forward references
- Function signatures should include type hints for parameters and return types

### Naming Conventions
- **Classes**: PascalCase (e.g., `PreTrainDataset`, `NGPTConfig`)
- **Functions/Methods**: snake_case (e.g., `get_lr_schedule`, `collate_fn`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `SPECIAL_TOKENS`, `DEVICE`)
- **Variables**: snake_case (e.g., `train_config`, `model_type`)
- **Private**: Prefix with underscore (e.g., `_wrap`, `_decode`)

### Formatting
- 4 spaces for indentation (no tabs)
- No strict line length limit, but keep lines reasonable (<120 chars)
- Trailing commas in multi-line collections
- No automated formatter configured (no Black/Ruff)

### Comments
- Chinese comments are acceptable and exist throughout the codebase
- Use docstrings for classes and public methods (Google style preferred)
- Inline comments should explain "why", not "what"

### Configuration
- Use dataclasses for configuration: `@dataclass class Config`
- Configs stored in JSON files in `models/` directory
- Default config at `models/defaults.json`

### Error Handling
- Use early returns with error messages for CLI scripts
- Example: `if len(sys.argv) < 2: print('Usage: ...'); exit(1)`
- Use try/except for user input and file operations
- Handle KeyboardInterrupt and EOFError for interactive scripts

### Architecture Patterns

#### Module Structure
```
minilm2/
├── __init__.py          # Exports Config
├── config.py            # Global configuration dataclass
├── llm/                 # LLM training and modeling
│   ├── __init__.py      # Package-level config instance
│   ├── train.py         # Pre-training script
│   ├── sft.py           # SFT training script
│   ├── dataset.py       # Dataset classes
│   ├── modeling_*.py    # Model implementations
│   └── ...
└── utils/               # Utilities and preprocessing
    ├── __init__.py
    ├── train_utils.py   # Shared training utilities
    ├── preprocess_*.py  # Data preprocessing scripts
    └── test_*.py        # Test/validation scripts
```

#### Key Patterns
- CLI entry points use `if __name__ == '__main__':` pattern
- Accept config via JSON file path as command line argument
- Use `pathlib.Path` for path operations
- Models inherit from `PreTrainedModel` and use Transformers integration
- Tokenizers use HuggingFace `tokenizers` library with custom wrapper

### Dependencies
- **Core**: torch, transformers, tokenizers, numpy
- **Data**: ijson, tiktoken, modelscope
- **Training**: flash-linear-attention, flash-muon (optional)
- **Dev**: ipykernel, ipython, matplotlib, tqdm

### Important Notes
- The project uses Chinese PyPI mirror (Tsinghua University)
- PyTorch wheels use SJTU CUDA 13.0 mirror
- Models are compiled with `model.compile()` for training
- Use `torch.amp` for mixed precision training
- Distributed training uses NCCL backend

## File Locations
- Configs: `models/*.json`
- Checkpoints: `models/checkpoint_*.pt/`
- Datasets: `datasets/` (gitignored, large files)
- Logs: `logs/` directory expected but not in repo
