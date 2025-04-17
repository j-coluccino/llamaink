# LlamaInk

<div align="center">
  
![LlamaInk Logo](https://raw.githubusercontent.com/j-coluccino/llamaink/main/assets/llamaink_logo.png)

</div>

**Privacy-first DBT documentation generator using local LLMs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python: 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🔍 Overview

LlamaInk is a tool for automatically generating comprehensive documentation for DBT models in data engineering repositories. Unlike cloud-based solutions, LlamaInk runs entirely locally, ensuring your proprietary code never leaves your machine.

Here is a quick video of LlamaInk in action:

### Key Features


- **No code transmission**: Your code never leaves your computer
- **Simple CLI interface**: Easy to use commands for generating documentation
- **Local LLM integration**: Uses quantized Llama models optimized for consumer hardware
- **CPU & GPU support**: Works on systems with or without dedicated GPUs
- **Open source & verifiable**: All code is open for review and enhancement

## 🚀 Installation

### Requirements

- Python 3.8 or higher
- 8GB+ RAM (16GB+ recommended for larger models)
- At least 4GB of free disk space for models
- CUDA-compatible GPU (optional, but improves performance)


### Installation from Source

```bash
# Clone the repository
git clone https://github.com/j-coluccino/llamaink.git
cd llamaink

# Install the package
pip install .

# For LLM installation
pip install -e ".[llm]"
```

## 📘 Usage

### Basic Usage

```bash
# Navigate to your DBT project
cd your-dbt-project

# Generate documentation
llamaink --repo .

# Specify output directory
llamaink --repo . --output docs/generated

# Use a different model
llamaink --repo . --model /path/to/your/model.gguf
```

### Command Line Options

```
LlamaInk - Privacy-First DBT Documentation Generator

Repository Options:
  --repo, -r              Path to the repository (default: current directory)

Output Options:
  --output, -o            Output directory for documentation (default: docs)

Model Options:
  --model, -m             Path to the Llama model (overrides config)
  --cpu-only              Force CPU-only mode, even if GPU is available
  --context-size          Context size for the model (overrides config)

Setup and Configuration:
  --init                  Initialize LlamaInk (download models, setup config)
  --config, -c            Path to custom configuration file
  --check-model           Check model compatibility without generating documentation

Logging Options:
  --verbose, -v           Enable verbose output downloading prompt.txt and response_output.txt
  --quiet, -q             Minimal output (errors only)
  --debug                 Enable debug output (very verbose)
  --log-file              Path to log file
  --no-progress           Disable progress bars
```

## 🔧 Configuration

LlamaInk uses a simple JSON configuration file located at `~/.llamaink/llamaink.json`. You can customize this file to change default behaviors.

Example configuration:

```json
{
  "model_path": "/path/to/your/model.gguf",
  "context_size": 8192,
  "use_gpu": true,
  "output_dir": "docs",
  "exclude_patterns": [
    ".git",
    "node_modules",
    "__pycache__",
    "*.pyc"
  ]
}
```

## 🖥️ Supported Models

LlamaInk works with quantized Llama models in GGUF format. Recommended models include:

- **Meta-Llama-3-8B-Instruct (Q2_K)**: Fast, efficient model for most documentation needs (default and only one tested)

When using `--init`, you'll be guided through downloading a compatible model.

## 🔄 How It Works

1. **Repository Scanning**: LlamaInk analyzes your repository to identify DBT models, schemas, and configurations
2. **Context Collection**: The tool collects relevant information from your DBT files
3. **Documentation Generation**: A local LLM processes the context and generates comprehensive documentation
4. **Output**: Generates Markdown files with model documentation, relationships, and lineage

## 📊 Example Outputs

The documentation generated includes:

- Model descriptions and purposes
- Field definitions and data types
- Relationships between models
- Data lineage and transformation logic
- Key usage patterns

## 🔒 Privacy & Security

LlamaInk is designed with privacy as the core principle:

- All code analysis happens on your local machine
- No data is ever transmitted to external servers
- Models run fully locally (no API keys required)
- Open source code that can be audited for security

## 🤝 Contributing

Contributions will be welcome! I am going to set up a contribution guideline soon after setting up proper CI/CD.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [DBT](https://www.getdbt.com/) - The data transformation tool LlamaInk supports
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient CPU inference for LLMs
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Python bindings for llama.cpp

## 🙏 Acknowledgements

- Meta's Llama models for providing the foundation for local LLM processing
- llama.cpp and llama-cpp-python for enabling efficient local inference
- DBT for creating an excellent data transformation tool

## ⚠️ Limitations of Local LLM Processing
While LlamaInk brings the power of LLMs to your machine, be aware of these limitations:

- Memory Requirements: Expect 4-16GB RAM usage depending on model size
- Speed: Generation takes 30 seconds to several minutes (slower than cloud alternatives)
- Context Size: Limited to 2048-8192 tokens, which may not accommodate very large codebases
- Hardware Impact: Significant battery drain and device heating during processing
- LLMs are not perfect: Expect some inaccuracies in generated documentation!

LlamaInk adapts to your hardware, but for best results, start with smaller repositories and be patient during model loading.