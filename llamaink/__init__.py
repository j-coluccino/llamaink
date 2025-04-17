"""
LlamaInk - Privacy-First DBT Documentation Generator Using Local LLMs

LlamaInk is a tool for automatically generating comprehensive documentation
for DBT models in data engineering repositories.

Features:
- Privacy-first approach: All analysis happens locally
- Uses quantized Llama models optimized for consumer hardware
- Simple CLI interface for generating DBT documentation
- Open source and verifiable with no external dependencies
"""

__version__ = "0.1.0"

from llamaink.analyser.repository import RepositoryScanner
from llamaink.llm.engine import LlamaEngine
from llamaink.generators.dbt_docs import DBTDocumentationGenerator
from llamaink.config import Config

__all__ = [
    "RepositoryScanner",
    "LlamaEngine",
    "DBTDocumentationGenerator",
    "Config",
]
