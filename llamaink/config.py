import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, ClassVar, Dict, Any, cast, Union

logger = logging.getLogger(__name__)


@dataclass
class LlamaInkConfig:
    """Configuration for LlamaInk."""

    model_path: str = ""
    context_size: int = 8192
    use_gpu: bool = True
    llama_binary_path: str = ""

    exclude_patterns: List[str] = field(
        default_factory=lambda: [
            ".git",
            ".github",
            "node_modules",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".DS_Store",
            ".idea",
            ".vscode",
            "venv",
            "env",
            ".env",
            ".venv",
            ".tox",
            "dist",
            "build",
            "*.egg-info",
        ]
    )

    output_dir: str = "docs"

    CONFIG_FILE_NAME: ClassVar[str] = "llamaink.json"

    @classmethod
    def get_default_model_path(cls) -> str:
        """Get the default model path."""
        home_dir = os.path.expanduser("~")
        return os.path.join(
            home_dir, ".llamaink", "models", "Meta-Llama-3-8B-Instruct.Q2_K.gguf"
        )

    @classmethod
    def from_file(cls, config_path: Optional[str] = None) -> "LlamaInkConfig":
        """
        Load configuration from a file.

        Args:
            config_path: Path to the configuration file, or None to use default

        Returns:
            Loaded configuration
        """
        if config_path is None:
            current_dir_config = os.path.join(os.getcwd(), cls.CONFIG_FILE_NAME)
            home_dir = os.path.expanduser("~")
            home_config = os.path.join(home_dir, ".llamaink", cls.CONFIG_FILE_NAME)

            if os.path.exists(current_dir_config):
                config_path = current_dir_config
            else:
                config_path = home_config

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                config_data_dict = cast(Dict[str, Any], config_data)
                config = cls(**config_data_dict)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            except Exception as e:
                logger.warning(
                    f"Error loading configuration from {config_path}: {str(e)}"
                )

        config = cls()

        if not config.model_path:
            config.model_path = cls.get_default_model_path()

        return config

    def save(self, config_path: Optional[str] = None) -> bool:
        """
        Save configuration to a file.

        Args:
            config_path: Path to save the configuration file, or None to use default

        Returns:
            True if successful, False otherwise
        """
        if config_path is None:
            home_dir = os.path.expanduser("~")
            config_dir = os.path.join(home_dir, ".llamaink")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, self.CONFIG_FILE_NAME)

        try:
            config_dict: Dict[str, Union[str, int, bool, List[str]]] = {
                k: v
                for k, v in self.__dict__.items()
                if not k.startswith("_") and k != "CONFIG_FILE_NAME"
            }

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2)

            logger.info(f"Saved configuration to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {str(e)}")
            return False


class Config:
    """
    Configuration manager for LlamaInk.

    This class handles loading, saving, and initializing configuration.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file, or None to use default
        """
        self.config = LlamaInkConfig.from_file(config_path)

    def initialize(self) -> bool:
        """
        Initialize LlamaInk configuration.

        This creates a default configuration file and sets up directories.

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            home_dir = os.path.expanduser("~")
            config_dir = os.path.join(home_dir, ".llamaink")
            models_dir = os.path.join(config_dir, "models")

            os.makedirs(config_dir, exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)

            config_path = os.path.join(config_dir, LlamaInkConfig.CONFIG_FILE_NAME)
            self.config.save(config_path)

            print(f"Initialized LlamaInk in {config_dir}")
            print(f"Model directory: {models_dir}")
            print(f"Configuration file: {config_path}")
            print("\nTo use LlamaInk, you need to:")
            print("1. Download a compatible quantized Llama model")
            print("2. Place it in the model directory or configure its path")
            print("3. Run 'llamaink --repo <path>' to generate DBT documentation")

            return True
        except Exception as e:
            logger.error(f"Error initializing LlamaInk: {str(e)}")
            return False

    def get_config(self) -> LlamaInkConfig:
        """Get the current configuration."""
        return self.config
