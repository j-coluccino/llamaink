import argparse
import os
import sys
import logging
import time
from typing import Dict, List, Tuple, Any, cast
import platform

from llamaink.analyser.repository import RepositoryScanner
from llamaink.llm.engine import LlamaEngine
from llamaink.llm.model_verification import ModelVerifier
from llamaink.generators.dbt_docs import DBTDocumentationGenerator
from llamaink.config import Config
from llamaink.utils.logging_config import configure_logging, ProgressDisplay

logger = logging.getLogger(__name__)

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"


def supports_ansi_colors() -> bool:
    """Check if the terminal supports ANSI colors."""
    if platform.system() == "Windows":
        try:
            import ctypes

            if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32"):
                kernel32 = ctypes.windll.kernel32
                return kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7) != 0
            return False
        except Exception:
            return False
    else:
        return True


def colorize(text: str, color: str) -> str:
    """Colorize text if supported."""
    if not supports_ansi_colors():
        return text
    return f"{color}{text}{ANSI_RESET}"


def setup_argparser() -> argparse.ArgumentParser:
    """Set up the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description=colorize(
            "LlamaInk - Privacy-first DBT documentation generator using local LLMs",
            ANSI_BOLD + ANSI_CYAN,
        )
    )

    repo_group = parser.add_argument_group("Repository Options")
    repo_group.add_argument(
        "--repo",
        "-r",
        default=".",
        help="Path to the repository (default: current directory)",
    )

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        "-o",
        help="Output directory for documentation (default: specified in config)",
    )

    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model", "-m", help="Path to the Llama model (overrides config)"
    )
    model_group.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU-only mode, even if GPU is available",
    )
    model_group.add_argument(
        "--context-size", type=int, help="Context size for the model (overrides config)"
    )

    model_group.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate (default: 4096)",
    )

    setup_group = parser.add_argument_group("Setup and Configuration")
    setup_group.add_argument(
        "--init",
        action="store_true",
        help="Initialize LlamaInk (setup config, etc.)",
    )
    setup_group.add_argument("--config", "-c", help="Path to custom configuration file")
    setup_group.add_argument(
        "--check-model",
        action="store_true",
        help="Check model compatibility without generating documentation",
    )

    logging_group = parser.add_argument_group("Logging Options")
    log_level = logging_group.add_mutually_exclusive_group()
    log_level.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    log_level.add_argument(
        "--quiet", "-q", action="store_true", help="Minimal output (errors only)"
    )
    log_level.add_argument(
        "--debug", action="store_true", help="Enable debug output (very verbose)"
    )
    logging_group.add_argument("--log-file", help="Path to log file")
    logging_group.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars"
    )

    return parser


def save_docs(docs: Dict[str, str], output_dir: str) -> bool:
    """
    Save generated documentation to files.

    Args:
        docs: Dictionary mapping file paths to content
        output_dir: Base output directory

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        saved_files: List[str] = []

        progress = ProgressDisplay(
            f"Saving documentation to {output_dir}",
            total=len(docs),
            disable="LLAMAINK_NO_PROGRESS" in os.environ,
        )

        for path, content in docs.items():
            file_path = os.path.join(output_dir, path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            saved_files.append(path)
            progress.update(1, path)

        progress.close()

        for path in saved_files:
            logger.info(f"Saved {os.path.join(output_dir, path)}")

        return True
    except Exception as e:
        logger.error(f"Error saving documentation: {str(e)}")
        return False


def check_repository_path(repo_path: str) -> bool:
    """
    Check if the repository path is valid.

    Args:
        repo_path: Path to check

    Returns:
        True if valid, False otherwise
    """
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_path):
        logger.error(f"Error: {repo_path} is not a valid directory")
        return False

    if len(os.listdir(repo_path)) == 0:
        logger.error(f"Error: {repo_path} is empty")
        return False

    return True


def print_system_info() -> None:
    """Print system information for debugging."""
    verifier = ModelVerifier()
    system_info = verifier._get_system_info()

    print("\nSystem Information:")
    print(f"  OS: {system_info['os']}")
    print(f"  Architecture: {system_info['architecture']}")
    print(f"  Python: {system_info['python_version']}")
    print(f"  CPU cores: {system_info['cpu_count']}")
    print(f"  RAM: {system_info['ram_gb']:.1f} GB")
    print(f"  GPU detected: {system_info['has_gpu']}")
    if system_info["has_metal"]:
        print(f"  Apple Silicon detected: {system_info['has_metal']}")

    try:
        import llama_cpp

        print(f"  llama-cpp-python: {llama_cpp.__version__}")
    except ImportError:
        print("  llama-cpp-python: Not installed")

    print(f"  llamaink version: {get_llamaink_version()}")

    config = Config().get_config()
    if os.path.exists(config.model_path):
        model_size = os.path.getsize(config.model_path) / (1024 * 1024 * 1024)
        print(
            f"  Default model: {os.path.basename(config.model_path)} ({model_size:.2f} GB)"
        )
    else:
        print("  Default model: Not found")

    print()


def get_llamaink_version() -> str:
    """Get the LlamaInk version."""
    try:
        import llamaink

        return getattr(llamaink, "__version__", "unknown")
    except (ImportError, AttributeError):
        return "development"


def print_repository_summary(dbt_files: List[str]) -> None:
    """Print a summary of the repository analysis."""
    print("\nRepository Analysis Summary:")
    print(f"  DBT files: {len(dbt_files)} files")
    print(f"  Total: {len(dbt_files)} files\n")


def print_banner() -> None:
    """Print a banner for the CLI."""
    if not supports_ansi_colors():
        print("\nLlamaInk - Privacy-First DBT Documentation Generator")
        print("Version: " + get_llamaink_version())
        print("=" * 50 + "\n")
        return

    banner = rf"""
{ANSI_BOLD}{ANSI_BLUE}╔═══════════════════════════════════════════════════════╗
║                                                       ║                   
║    _      _                       _____       _       ║
║   | |    | |                     |_   _|     | |      ║
║   | |    | | __ _ _ __ ___   __ _  | |  _ __ | | __   ║
║   | |    | |/ _` | '_ ` _ \ / _` | | | | '_ \| |/ /   ║
║   | |____| | (_| | | | | | | (_| |_| |_| | | |   <    ║
║   |______|_|\__,_|_| |_| |_|\__,_|_____|_| |_|_|\_\   ║
║                                                       ║
║   LlamaInk - Privacy-First DBT Documentation Generator║
║   Version: v{get_llamaink_version()}                                     ║
╚═══════════════════════════════════════════════════════╝{ANSI_RESET}
"""
    print(banner)


def check_context_compatibility(
    config: Any, verification: Dict[str, Any]
) -> Tuple[bool, int]:
    """
    Check if the configured context size is compatible with the model.

    Args:
        config: LlamaInk configuration
        verification: Model verification results

    Returns:
        Tuple of (is_compatible, recommended_size)
    """
    recommended_size = config.context_size

    metadata = verification.get("metadata", {})
    model_context = metadata.get("context_length")

    if model_context:
        try:
            model_max_context = int(model_context)

            if config.context_size > model_max_context:
                logger.warning(
                    f"Configured context size ({config.context_size}) exceeds "
                    f"model maximum ({model_max_context}). Adjusting."
                )
                recommended_size = model_max_context
                return False, recommended_size

        except (ValueError, TypeError):
            pass

    system_info = verification.get("system_info", {})
    ram_gb = cast(float, system_info.get("ram_gb", 8.0))
    has_gpu = cast(bool, system_info.get("has_gpu", False))

    if has_gpu:
        if ram_gb < 8 and config.context_size > 2048:
            logger.warning(
                f"Limited GPU memory detected ({ram_gb:.1f} GB). "
                f"Reducing context size from {config.context_size} to 2048."
            )
            recommended_size = 2048
            return False, recommended_size
    else:
        if ram_gb < 16 and config.context_size > 2048:
            logger.warning(
                f"Limited system RAM detected ({ram_gb:.1f} GB). "
                f"Reducing context size from {config.context_size} to 2048."
            )
            recommended_size = 2048
            return False, recommended_size

    return True, recommended_size


def main() -> int:
    """Main entry point for the CLI."""
    print_banner()

    parser = setup_argparser()
    args = parser.parse_args()

    log_level = "normal"
    if args.verbose:
        log_level = "verbose"
    elif args.quiet:
        log_level = "quiet"
    elif args.debug:
        log_level = "debug"

    configure_logging(
        level_name=log_level,
        log_file=args.log_file,
        disable_progress_bars=args.no_progress,
    )

    try:
        config_manager = Config(args.config)
        config = config_manager.get_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        print(f"Error: {str(e)}")
        return 1

    if args.init:
        success = config_manager.initialize()
        if not success:
            logger.error("Initialization failed")
            return 1
        logger.info("Initialization successful")
        return 0

    if args.debug:
        print_system_info()

    if args.model:
        config.model_path = args.model
    if args.cpu_only:
        config.use_gpu = False
    if args.context_size:
        config.context_size = args.context_size
    if args.max_tokens:
        config.max_tokens = args.max_tokens
    if args.output:
        config.output_dir = args.output

    if args.check_model:
        verifier = ModelVerifier()
        result = verifier.verify_model(config.model_path)

        print(f"\nModel verification for: {result['model_path']}")
        print(f"  Exists: {result['exists']}")
        if result["exists"]:
            print(f"  Format: {result['format']}")
            print(f"  Size: {result['size_mb']:.2f} MB ({result['size_gb']:.2f} GB)")
            print(f"  Quantization: {result['quantization']}")
            print(f"  Compatible with system: {result['compatible_with_system']}")

            if result["warnings"]:
                print("\nWarnings:")
                for warning in result["warnings"]:
                    print(f"  - {warning}")

            if result["recommendations"]:
                print("\nRecommendations:")
                for recommendation in result["recommendations"]:
                    print(f"  - {recommendation}")

        return 0

    repo_path = os.path.abspath(args.repo)
    if not check_repository_path(repo_path):
        return 1

    logger.info(f"Analyzing repository: {repo_path}")

    try:
        scanner = RepositoryScanner(repo_path, config.exclude_patterns)
        dbt_files = scanner.scan()

        if args.verbose or args.debug:
            print_repository_summary(dbt_files)

        if not dbt_files:
            logger.warning("No DBT files found in the repository")
            print(f"\n{ANSI_YELLOW}No DBT files found in the repository.{ANSI_RESET}")
            print("Please check that the repository contains DBT models.")
            return 1

        logger.info("Initializing LLM engine...")
        start_time = time.time()

        verifier = ModelVerifier()
        verification = verifier.verify_model(config.model_path)

        if not verification["exists"]:
            logger.error(f"Model not found: {config.model_path}")
            print(f"\nError: Model not found at {config.model_path}")
            print(
                "Run 'llamaink --init' to setup config or specify a model with --model"
            )
            return 1

        if not verification["compatible_with_system"] and verification["warnings"]:
            for warning in verification["warnings"]:
                logger.warning(warning)

        is_compatible, recommended_ctx_size = check_context_compatibility(
            config, verification
        )
        if not is_compatible:
            logger.warning(f"Adjusting context size to {recommended_ctx_size}")
            config.context_size = recommended_ctx_size

        engine = LlamaEngine(
            model_path=config.model_path,
            context_size=config.context_size,
            use_gpu=config.use_gpu,
            verbose=args.verbose or args.debug,
            config=config,
        )

        if not engine.initialize():
            logger.error("Failed to initialize LLM engine")
            return 1

        logger.info(f"LLM engine initialized in {time.time() - start_time:.2f} seconds")

        try:
            start_time = time.time()
            logger.info("Generating DBT documentation...")

            doc_generator = DBTDocumentationGenerator(engine)
            docs = doc_generator.generate(repo_path, dbt_files)

            generation_time = time.time() - start_time
            logger.info(f"Documentation generated in {generation_time:.2f} seconds")

            output_dir = args.output or config.output_dir
            if output_dir != ".":
                output_dir = os.path.join(repo_path, output_dir)

            logger.info(f"Saving documentation to {output_dir}...")
            if not save_docs(docs, output_dir):
                logger.error("Failed to save documentation")
                return 1

            num_files = len(docs)
            logger.info(
                f"Documentation generated successfully! {num_files} files created."
            )

            print(
                f"\n{ANSI_GREEN}DBT documentation generated successfully!{ANSI_RESET}"
            )
            print(f"Generated {num_files} files in {output_dir}")

            for path in docs.keys():
                file_path = os.path.join(output_dir, path)
                print(f"  DBT docs: {file_path}")

            print("\nUse these files to understand and document your DBT models.")

            return 0
        except Exception as e:
            logger.error(f"Error generating documentation: {str(e)}")
            if args.verbose or args.debug:
                import traceback

                traceback.print_exc()
            return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if args.verbose or args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
