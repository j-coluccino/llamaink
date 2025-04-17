import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import subprocess

from llamaink.llm.engine import LlamaEngine


class TestLlamaEngine(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_model_path = os.path.join(self.temp_dir.name, "test_model.gguf")

        with open(self.test_model_path, "w") as f:
            f.write("DUMMY MODEL DATA")

        self.engine = LlamaEngine(
            model_path=self.test_model_path,
            context_size=2048,
            use_gpu=False,
            threads=4,
            verbose=False,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init(self):
        """Test engine initialization."""
        engine = LlamaEngine(
            model_path=self.test_model_path,
            context_size=4096,
            use_gpu=True,
            threads=8,
            verbose=True,
        )

        self.assertEqual(engine.model_path, self.test_model_path)
        self.assertEqual(engine.context_size, 4096)
        self.assertEqual(engine.threads, 8)
        self.assertTrue(engine.verbose)

        self.assertFalse(engine.initialized)
        self.assertIsNone(engine.llm)

    def test_get_default_model_path(self):
        """Test getting the default model path."""
        path = self.engine._get_default_model_path()

        self.assertIn(".llamaink", path)
        self.assertIn("models", path)
        self.assertIn("Meta-Llama-3-8B-Instruct.Q2_K.gguf", path)

    @patch("llamaink.llm.engine.platform.system")
    @patch("subprocess.run")
    def test_is_gpu_available_nvidia(self, mock_run, mock_system):
        """Test GPU availability detection for NVIDIA."""

        mock_system.return_value = "Linux"
        mock_run.return_value = MagicMock(returncode=0)

        result = self.engine._is_gpu_available()

        self.assertTrue(result)
        mock_run.assert_called_once_with(
            ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2
        )

    @patch("llamaink.llm.engine.platform.system")
    @patch("subprocess.run")
    def test_is_gpu_available_apple(self, mock_run, mock_system):
        """Test GPU availability detection for Apple Silicon."""

        mock_system.return_value = "Darwin"
        mock_run.side_effect = subprocess.SubprocessError()

        process_mock = MagicMock()
        process_mock.stdout = "Apple M1 Pro"
        with patch("subprocess.run", return_value=process_mock) as mock_run_apple:
            result = self.engine._is_gpu_available()

            mock_run_apple.assert_called_once()
            self.assertTrue(result)

    @patch("multiprocessing.cpu_count")
    def test_get_optimal_threads(self, mock_cpu_count):
        """Test optimal thread count calculation."""

        mock_cpu_count.return_value = 8

        threads = self.engine._get_optimal_threads()

        self.assertEqual(threads, 6)

        mock_cpu_count.return_value = 1
        threads = self.engine._get_optimal_threads()
        self.assertEqual(threads, 1)

    @patch("llamaink.llm.engine.os.path.exists")
    def test_find_llama_binary(self, mock_exists):
        """Test finding the llama.cpp binary."""

        def mock_path_exists(path):
            return path == "/usr/local/bin/llama"

        mock_exists.side_effect = mock_path_exists

        with patch("os.access", return_value=True):
            binary_path = self.engine._find_llama_binary()

            self.assertEqual(binary_path, "/usr/local/bin/llama")

    @patch("llamaink.llm.engine.os.path.exists", return_value=False)
    def test_find_llama_binary_not_found(self, mock_exists):
        """Test binary not found."""
        binary_path = self.engine._find_llama_binary()

        self.assertIsNone(binary_path)

    def test_check_llama_cpp_python(self):
        """Test checking for llama-cpp-python."""

        with patch.dict("sys.modules", {"llama_cpp": MagicMock()}):
            result = self.engine._check_llama_cpp_python()
            self.assertTrue(result)

        with patch.dict("sys.modules", {"llama_cpp": None}):
            with patch("importlib.import_module", side_effect=ImportError()):
                result = self.engine._check_llama_cpp_python()
                self.assertFalse(result)

    @patch("os.path.exists", return_value=True)
    @patch("llamaink.llm.engine.LlamaEngine._initialize_with_python_bindings")
    def test_initialize_with_python_bindings(self, mock_init_py, mock_exists):
        """Test initialization with Python bindings."""

        self.engine.llama_cpp_python_available = True
        mock_init_py.return_value = True

        def side_effect():

            self.engine.initialized = True
            return True

        mock_init_py.side_effect = side_effect

        result = self.engine.initialize()

        self.assertTrue(result)
        self.assertTrue(self.engine.initialized)
        mock_init_py.assert_called_once()

    @patch("llamaink.llm.engine.os.path.exists", return_value=True)
    def test_initialize_with_binary(self, mock_exists):
        """Test initialization with llama.cpp binary."""

        self.engine.llama_cpp_python_available = False
        self.engine.llama_binary_path = "/usr/local/bin/llama"

        result = self.engine.initialize()

        self.assertTrue(result)
        self.assertTrue(self.engine.initialized)

    @patch("llamaink.llm.engine.os.path.exists", return_value=True)
    def test_initialize_no_implementation(self, mock_exists):
        """Test initialization with no valid implementation."""

        self.engine.llama_cpp_python_available = False
        self.engine.llama_binary_path = None

        result = self.engine.initialize()

        self.assertFalse(result)
        self.assertFalse(self.engine.initialized)

    @patch("llamaink.llm.engine.os.path.exists", return_value=False)
    @patch("llamaink.llm.engine.LlamaEngine._download_model", return_value=True)
    def test_initialize_download_model(self, mock_download, mock_exists):
        """Test initialization with model download."""

        result = self.engine.initialize()

        mock_download.assert_called_once()

    def test_generate_not_initialized(self):
        """Test generate without initialization."""
        self.engine.initialized = False

        with self.assertRaises(RuntimeError):
            self.engine.generate("Test prompt")

    @patch("llamaink.llm.context_manager.ContextManager.ensure_context_fits")
    def test_generate_context_adjustment(self, mock_ensure_fits):
        """Test context window adjustment during generation."""

        self.engine.initialized = True

        mock_ensure_fits.return_value = (
            "Adjusted prompt",
            1024,
            {
                "input_modified": True,
                "output_modified": True,
                "original_input_tokens": 3000,
                "original_max_tokens": 2048,
                "available_context": 1500,
            },
        )

        with patch.object(
            self.engine, "_generate_with_python_bindings", return_value="Generated text"
        ) as mock_gen_py:
            self.engine.llm = MagicMock()
            result = self.engine.generate("Test prompt", max_tokens=2048)

            mock_gen_py.assert_called_once_with("Adjusted prompt", 1024)
            self.assertEqual(result, "Generated text")

        with patch.object(
            self.engine, "_generate_with_binary", return_value="Generated with binary"
        ) as mock_gen_bin:
            self.engine.llm = None
            result = self.engine.generate("Test prompt", max_tokens=2048)

            mock_gen_bin.assert_called_once_with("Adjusted prompt", 1024)
            self.assertEqual(result, "Generated with binary")

    @patch("llamaink.llm.engine.tempfile.NamedTemporaryFile")
    @patch("llamaink.llm.engine.subprocess.run")
    def test_generate_with_binary(self, mock_run, mock_tempfile):
        """Test generation using the llama.cpp binary."""

        mock_file = MagicMock()
        mock_file.name = "/tmp/input.txt"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        mock_process = MagicMock()
        mock_process.stdout = "Generated output"
        mock_run.return_value = mock_process

        self.engine.llama_binary_path = "/usr/local/bin/llama"

        result = self.engine._generate_with_binary("Test prompt", 100)

        mock_run.assert_called_once()

        cmd_args = mock_run.call_args[0][0]

        self.assertEqual(cmd_args[0], "/usr/local/bin/llama")
        self.assertEqual(cmd_args[2], self.test_model_path)

        self.assertEqual(result, "Generated output")

    @patch("requests.get")
    @patch("builtins.input", return_value="1")
    @patch("builtins.print")
    def test_auto_download_model(self, mock_print, mock_input, mock_requests_get):
        """Test automatic model download."""

        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1048576"}
        mock_response.iter_content.return_value = [b"x" * 1024] * 1024
        mock_requests_get.return_value = mock_response

        with patch("tqdm.tqdm"):

            result = self.engine._auto_download_model()

            self.assertTrue(result)

            mock_requests_get.assert_called_once()

            mock_input.assert_called_once()

    @patch("builtins.input", return_value="1")
    @patch("os.path.exists")
    @patch("llamaink.llm.engine.LlamaEngine._auto_download_model")
    @patch("llamaink.llm.engine.LlamaEngine._initialize_with_python_bindings")
    def test_initialize_with_download(
        self, mock_init_py, mock_download, mock_exists, mock_input
    ):
        """Test initialization with model download."""

        mock_exists.return_value = False

        mock_download.return_value = True

        mock_init_py.return_value = True

        def init_side_effect():
            self.engine.initialized = True
            return True

        mock_init_py.side_effect = init_side_effect

        result = self.engine.initialize()

        self.assertTrue(result)
        self.assertTrue(self.engine.initialized)

        mock_download.assert_called_once()

        mock_init_py.assert_called_once()

    @patch("llamaink.llm.engine.shutil.copy2")
    @patch("builtins.input", return_value="/path/to/existing/model.gguf")
    @patch("os.path.exists", return_value=True)
    def test_use_existing_model(self, mock_exists, mock_input, mock_copy):
        """Test using an existing model file."""
        result = self.engine._use_existing_model()

        self.assertTrue(result)

        mock_copy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
