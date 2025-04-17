import os
import logging
import tempfile
import platform
import shutil
import subprocess
import sys
from typing import Optional, Dict, Any, List, cast

logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME: str = "Meta-Llama-3-8B-Instruct.Q2_K.gguf"
MODEL_URLS: Dict[str, str] = {
    "Meta-Llama-3-8B-Instruct.Q2_K.gguf": "https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q2_K.gguf",
}


class LlamaEngine:
    """
    Engine for running local Llama models.

    This class handles the initialization, configuration, and execution of locally run
    Llama models, with optimizations for CPU and GPU execution.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        context_size: int = 8192,
        use_gpu: bool = True,
        threads: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the LLM engine.

        Args:
            model_path: Path to the model file, or None to use default location
            context_size: Context window size in tokens
            use_gpu: Whether to use GPU acceleration if available
            threads: Number of CPU threads to use, or None for auto-detection
            verbose: Whether to print verbose output
        """
        self.model_path: str = model_path or self._get_default_model_path()
        self.context_size: int = context_size
        self.use_gpu: bool = use_gpu and self._is_gpu_available()
        self.threads: int = threads or self._get_optimal_threads()
        self.verbose: bool = verbose
        self.initialized: bool = False
        self.llm: Any = None

        self.llama_binary_path: Optional[str] = self._find_llama_binary()
        self.llama_cpp_python_available: bool = self._check_llama_cpp_python()

    def initialize(self) -> bool:
        """
        Initialize the LLM engine.

        Returns:
            True if initialization was successful, False otherwise
        """
        if not os.path.exists(self.model_path):
            logger.info(f"Model not found at {self.model_path}")
            success = self._download_model()
            if not success:
                return False

        if self.llama_cpp_python_available:
            return self._initialize_with_python_bindings()

        elif self.llama_binary_path:
            logger.info(f"Using llama.cpp binary at {self.llama_binary_path}")
            self.initialized = True
            return True
        else:
            logger.error("No valid LLM implementation available")
            return False

    def generate(self, prompt: str, max_tokens: int = 4096) -> str:
        """
        Generate text from the model.

        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum number of tokens to generate

        Returns:
            Generated text

        Raises:
            RuntimeError: If the engine is not initialized
        """
        if not self.initialized:
            raise RuntimeError("LLM engine not initialized. Call initialize() first.")

        if self.verbose:
            logger.debug(f"Generating text with max_tokens={max_tokens}")
            logger.debug(f"Prompt preview: {prompt[:100]}...")

        from llamaink.llm.context_manager import ContextManager

        context_manager = ContextManager(self.context_size)
        adjusted_prompt, adjusted_max_tokens, modifications = (
            context_manager.ensure_context_fits(prompt, max_tokens)
        )

        if modifications["input_modified"] or modifications["output_modified"]:
            logger.warning(
                f"Adjusted prompt and/or max_tokens to fit within context window ({self.context_size}). "
                f"Original tokens: input={modifications['original_input_tokens']}, "
                f"output={modifications['original_max_tokens']}. "
                f"Available context: {modifications['available_context']}"
            )

        if self.llm is not None:
            return self._generate_with_python_bindings(
                adjusted_prompt, adjusted_max_tokens
            )

        return self._generate_with_binary(adjusted_prompt, adjusted_max_tokens)

    def _get_default_model_path(self) -> str:
        """
        Get the default model path based on user's home directory.

        Returns:
            Default path for the model file
        """
        home_dir = os.path.expanduser("~")
        model_dir = os.path.join(home_dir, ".llamaink", "models")
        return os.path.join(model_dir, DEFAULT_MODEL_NAME)

    def _is_gpu_available(self) -> bool:
        """
        Check if compatible GPU is available.

        Returns:
            True if a compatible GPU is available, False otherwise
        """
        try:
            if platform.system() in ["Windows", "Linux"]:
                nvidia_result = subprocess.run(
                    ["nvidia-smi"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=2,
                )
                if nvidia_result.returncode == 0:
                    logger.info("NVIDIA GPU detected")
                    return True

            if platform.system() == "Darwin":
                process = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                )
                if process.stdout and "Apple" in process.stdout:
                    logger.info("Apple Silicon detected")
                    return True

            return False
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _get_optimal_threads(self) -> int:
        """
        Get the optimal number of threads based on CPU cores.

        Returns:
            Recommended number of threads to use
        """
        import multiprocessing

        cores = multiprocessing.cpu_count()
        return max(1, cores - 2)

    def _check_llama_cpp_python(self) -> bool:
        """
        Check if llama-cpp-python is available.

        Returns:
            True if available, False otherwise
        """
        try:
            import llama_cpp

            logger.info("llama-cpp-python is available")
            return True
        except ImportError:
            logger.info("llama-cpp-python not found, will fallback to binary execution")
            return False

    def _initialize_with_python_bindings(self) -> bool:
        """
        Initialize using llama-cpp-python bindings.

        Returns:
            True if successful, False otherwise
        """
        try:
            import llama_cpp

            logger.info(f"Initializing LLM with model: {self.model_path}")
            logger.info(f"Context size: {self.context_size}, Threads: {self.threads}")

            n_gpu_layers = 32 if self.use_gpu else 0
            if self.use_gpu:
                logger.info(f"Using GPU with {n_gpu_layers} layers")

            self.llm = llama_cpp.Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_threads=self.threads,
                n_gpu_layers=n_gpu_layers,
                verbose=self.verbose,
            )

            logger.info("Testing model initialization...")
            self.llm("Test")

            logger.info("LLM initialized successfully with Python bindings")
            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize with Python bindings: {str(e)}")
            self.llm = None
            return False

    def _generate_with_python_bindings(self, prompt: str, max_tokens: int) -> str:
        """
        Generate text using llama-cpp-python bindings.

        Args:
            prompt: The prompt text
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            from llamaink.utils.logging_config import ProgressDisplay

            if self.verbose:
                prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
                logger.debug(f"Generating with prompt preview: {prompt_preview}")
                logger.debug(
                    f"Generation parameters: max_tokens={max_tokens}, temp=0.7, top_p=0.9"
                )

            generation_params: Dict[str, Any] = {
                "max_tokens": min(max_tokens, 8192),
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stop": ["<|endoftext|>", "</s>", "<|eot_id|>"],
                "seed": 42,
            }

            progress: Optional[ProgressDisplay] = None
            if max_tokens > 500:
                progress = ProgressDisplay("Generating documentation")

            response = self.llm(prompt, **generation_params)
            with open("response_output.txt", "w", encoding="utf-8") as output_file:
                output_file.write(str(response))

            if isinstance(response, dict) and "choices" in response:
                output = response["choices"][0]["text"]
            elif (
                hasattr(response, "choices")
                and len(getattr(response, "choices", [])) > 0
            ):
                choice = response.choices[0]
                if hasattr(choice, "text"):
                    output = choice.text
                else:
                    output = str(choice)
            elif hasattr(response, "completion"):
                output = response.completion
            else:
                logger.debug(
                    f"Using string representation for response type: {type(response)}"
                )
                output_parts = str(response).split("```", 2)
                if len(output_parts) >= 2:
                    output = output_parts[0] + "```" + output_parts[1] + "```"
                else:
                    output = output_parts[0]
            return output.strip()

        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")

            if "out of memory" in str(e).lower():
                logger.error(
                    "Memory error detected. Try reducing context size or using a smaller model."
                )
            elif "cuda" in str(e).lower() and "error" in str(e).lower():
                logger.error("CUDA error detected. Trying to fall back to CPU...")

                try:
                    old_n_gpu_layers = self.llm.n_gpu_layers
                    self.llm.n_gpu_layers = 0
                    response = self.llm(
                        prompt, **{"max_tokens": max_tokens, "temperature": 0.7}
                    )
                    self.llm.n_gpu_layers = old_n_gpu_layers
                    return cast(str, response).strip()
                except Exception as e2:
                    logger.error(f"CPU fallback also failed: {str(e2)}")

            raise RuntimeError(f"LLM generation failed: {str(e)}")

    def _find_llama_binary(self) -> Optional[str]:
        """
        Find the llama.cpp binary using multiple strategies.

        Returns:
            Path to the binary if found, None otherwise
        """
        system = platform.system()
        binary_name = "llama" if system != "Windows" else "llama.exe"

        possible_locations: List[str] = []

        path_var = os.environ.get("PATH", "")
        path_separator = ";" if system == "Windows" else ":"

        for path_dir in path_var.split(path_separator):
            binary_path = os.path.join(path_dir, binary_name)
            possible_locations.append(binary_path)

        if system == "Linux" or system == "Darwin":
            possible_locations.extend(
                [
                    "/usr/local/bin/llama",
                    "/usr/bin/llama",
                    os.path.expanduser("~/llama.cpp/build/bin/llama"),
                    os.path.expanduser("~/.local/bin/llama"),
                ]
            )
        elif system == "Windows":
            possible_locations.extend(
                [
                    os.path.expanduser("~\\llama.cpp\\build\\bin\\llama.exe"),
                    "C:\\Program Files\\llama.cpp\\llama.exe",
                ]
            )

        package_dir = self._get_package_binary_dir()
        if package_dir:
            possible_locations.append(os.path.join(package_dir, binary_name))

        for location in possible_locations:
            if os.path.exists(location) and os.access(location, os.X_OK):
                logger.info(f"Found llama binary at: {location}")
                return location

        logger.warning("Could not find llama binary in any standard location")
        return None

    def _get_package_binary_dir(self) -> Optional[str]:
        """
        Get the directory containing packaged binaries.

        Returns:
            Path to the binary directory, or None if not found
        """
        try:
            import llamaink

            package_dir = os.path.dirname(llamaink.__file__)
            binary_dir = os.path.join(package_dir, "bin")

            if os.path.isdir(binary_dir):
                return binary_dir
            return None
        except ImportError:
            return None

    def _generate_with_binary(self, prompt: str, max_tokens: int) -> str:
        """
        Generate text using llama.cpp binary.

        Args:
            prompt: The prompt text
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        if self.llama_binary_path is None:
            raise RuntimeError("No llama binary available")

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False, suffix=".txt"
        ) as f:
            f.write(prompt)
            input_file = f.name

        try:
            binary_path = self.llama_binary_path
            simple_path = binary_path.replace(
                os.path.basename(binary_path), os.path.basename(binary_path) + "-simple"
            )

            if os.path.exists(simple_path) and os.access(simple_path, os.X_OK):
                logger.info(f"Using simplified binary at {simple_path}")
                binary_path = simple_path

            cmd: List[str] = [
                binary_path,
                "-m",
                self.model_path,
                "-f",
                input_file,
                "-n",
                str(max_tokens),
                "--ctx-size",
                str(self.context_size),
                "--threads",
                str(self.threads),
                "--temp",
                "0.7",
                "--top-p",
                "0.9",
                "--repeat-penalty",
                "1.1",
                "--seed",
                "42",
                "-p",
                prompt,
            ]

            if self.use_gpu:
                cmd.extend(["--gpu-layers", "32"])

            if self.verbose:
                logger.debug(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            output = result.stdout.strip()

            if not output or all(
                line.startswith("-") for line in output.split("\n") if line.strip()
            ):
                logger.warning(
                    "Invalid output from llama binary, trying alternative methods"
                )

                output_file = input_file + ".out"
                if os.path.exists(output_file):
                    with open(output_file, "r", encoding="utf-8") as f:
                        output = f.read().strip()

                cmd = [
                    binary_path,
                    "-m",
                    self.model_path,
                    "-n",
                    str(max_tokens),
                    "--ctx-size",
                    str(self.context_size),
                    "--threads",
                    str(self.threads),
                    "--temp",
                    "0.7",
                    "--top-p",
                    "0.9",
                    "--repeat-penalty",
                    "1.1",
                    "-p",
                    prompt,
                ]

                if not output or all(
                    line.startswith("-") for line in output.split("\n") if line.strip()
                ):
                    cmd = cmd
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, check=True
                    )
                    output = result.stdout.strip()

                if self.use_gpu:
                    cmd.extend(["--gpu-layers", "32"])

                result = subprocess.run(cmd, capture_output=True, text=True, check=True)

                output = result.stdout.strip()

                if output.startswith(prompt):
                    output = output[len(prompt) :].strip()

            return output

        except subprocess.CalledProcessError as e:
            error_msg = f"LLM inference failed: {e.stderr}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        finally:
            try:
                if os.path.exists(input_file):
                    os.unlink(input_file)
                output_file = input_file + ".out"
                if os.path.exists(output_file):
                    os.unlink(output_file)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {str(e)}")

    def _download_model(self) -> bool:
        """
        Handles model download or user-provided model path.

        Returns:
            True if model setup was successful, False otherwise
        """
        model_dir = os.path.dirname(self.model_path)
        os.makedirs(model_dir, exist_ok=True)

        print("=" * 80)
        print("LlamaInk requires a quantized Llama model to work.")
        print("\nOptions:")
        print("1. Download a model automatically")
        print("2. Provide a path to an existing model")
        print("3. Abort setup")
        print("=" * 80)

        while True:
            choice = input("Choose an option (1-3): ").strip()

            if choice == "1":
                return self._auto_download_model()
            elif choice == "2":
                return self._use_existing_model()
            elif choice == "3":
                print("Setup aborted.")
                return False
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

    def _auto_download_model(self) -> bool:
        """
        Automatically download a model from a predefined source.

        Returns:
            True if download was successful, False otherwise
        """
        try:
            try:
                import requests
                from tqdm import tqdm
            except ImportError:
                print("Required packages missing. Installing requests and tqdm...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "requests", "tqdm"]
                )
                import requests
                from tqdm import tqdm

            print("\nAvailable models:")
            for i, model_name in enumerate(MODEL_URLS.keys(), 1):
                print(f"{i}. {model_name}")

            choice = input(
                f"Select a model (1-{len(MODEL_URLS)}), or press Enter for default: "
            )

            if not choice.strip():
                selected_model = DEFAULT_MODEL_NAME
            else:
                try:
                    idx = int(choice) - 1
                    model_keys = list(MODEL_URLS.keys())
                    if idx < 0 or idx >= len(model_keys):
                        print("Invalid selection. Using default model.")
                        selected_model = DEFAULT_MODEL_NAME
                    else:
                        selected_model = model_keys[idx]
                except (ValueError, IndexError):
                    print("Invalid selection. Using default model.")
                    selected_model = DEFAULT_MODEL_NAME

            url = MODEL_URLS[selected_model]
            target_path = os.path.join(os.path.dirname(self.model_path), selected_model)

            print(f"Downloading {selected_model}...")
            print(f"This may take a while depending on your internet connection.")

            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))

            with open(target_path, "wb") as f, tqdm(
                desc=selected_model,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024 * 1024):
                    size = f.write(data)
                    bar.update(size)

            self.model_path = target_path
            print(f"Model downloaded successfully to: {target_path}")
            return True

        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            return False

    def _use_existing_model(self) -> bool:
        """
        Use a model file provided by the user.

        Returns:
            True if model setup was successful, False otherwise
        """
        user_model_path = input("Enter the full path to your model file: ").strip()

        if not user_model_path:
            print("No path provided. Aborting.")
            return False

        if not os.path.exists(user_model_path):
            print(f"Error: File not found at {user_model_path}")
            return False

        try:
            target_path = os.path.join(
                os.path.dirname(self.model_path), os.path.basename(user_model_path)
            )

            print(f"Copying model from {user_model_path} to {target_path}...")
            shutil.copy2(user_model_path, target_path)

            self.model_path = target_path
            print(f"Model copied successfully to: {target_path}")
            return True

        except Exception as e:
            print(f"Error copying model: {str(e)}")
            return False
