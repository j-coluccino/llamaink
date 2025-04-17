import unittest
from unittest.mock import patch, MagicMock
from llamaink.llm.model_verification import ModelVerifier


class TestModelVerifier(unittest.TestCase):
    def setUp(self):
        with patch.object(ModelVerifier, "_get_system_info") as mock_get_system_info:
            mock_get_system_info.return_value = {
                "os": "Linux",
                "architecture": "x86_64",
                "python_version": "3.8.0",
                "cpu_count": 8,
                "ram_gb": 16.0,
                "has_gpu": True,
                "has_metal": False,
            }
            self.verifier = ModelVerifier()

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=5 * 1024 * 1024 * 1024)  # 5 GB
    def test_verify_model_exists(self, mock_getsize, mock_exists):
        """Test verifying an existing model."""
        with patch.object(
            self.verifier,
            "_extract_model_metadata",
            return_value={"context_length": "8192"},
        ):
            result = self.verifier.verify_model("/path/to/model.gguf")

            self.assertTrue(result["exists"])
            self.assertEqual(result["size_gb"], 5.0)
            self.assertTrue(result["size_ok"])

            self.assertEqual(result["format"], "GGUF format (recommended)")
            self.assertTrue(result["format_compatible"])

            self.assertTrue(result["compatible_with_system"])
            self.assertEqual(len(result["warnings"]), 0)

    @patch("os.path.exists", return_value=False)
    def test_verify_model_not_exists(self, mock_exists):
        """Test verifying a non-existent model."""
        result = self.verifier.verify_model("/path/to/nonexistent.gguf")

        self.assertFalse(result["exists"])
        self.assertGreater(len(result["warnings"]), 0)
        self.assertGreater(len(result["recommendations"]), 0)

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=5 * 1024 * 1024)  # 5 MB (too small)
    def test_verify_model_too_small(self, mock_getsize, mock_exists):
        """Test verifying a suspiciously small model."""
        minimum_size = self.verifier.MINIMUM_MODEL_SIZE_MB

        if minimum_size <= 5.0:
            mock_getsize.return_value = int((minimum_size - 1) * 1024 * 1024)

        result = self.verifier.verify_model("/path/to/model.gguf")

        self.assertTrue(result["exists"])
        self.assertAlmostEqual(
            result["size_mb"], mock_getsize.return_value / (1024 * 1024), places=1
        )

        self.assertFalse(result["size_ok"])

        size_warnings = [w for w in result["warnings"] if "small" in w.lower()]
        self.assertTrue(len(size_warnings) > 0)

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=5 * 1024 * 1024 * 1024)  # 5 GB
    def test_verify_model_unsupported_format(self, mock_getsize, mock_exists):
        """Test verifying a model with unsupported format."""
        result = self.verifier.verify_model("/path/to/model.unsupported")

        self.assertTrue(result["exists"])
        self.assertFalse(result["format_compatible"])
        self.assertGreater(len(result["warnings"]), 0)
        self.assertIn("Model file format not recognized", result["warnings"][0])

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=5 * 1024 * 1024 * 1024)  # 5 GB
    def test_verify_model_with_insufficient_ram(self, mock_getsize, mock_exists):
        """Test verifying a model with insufficient RAM."""
        self.verifier.system_info["ram_gb"] = 4.0

        result = self.verifier.verify_model("/path/to/model.q8_0.gguf")

        self.assertTrue(result["exists"])
        self.assertEqual(result["quantization"], "q8_0")
        self.assertFalse(result["compatible_with_system"])
        self.assertGreater(len(result["warnings"]), 0)
        self.assertIn("RAM", result["warnings"][0])

    @patch("platform.system", return_value="Linux")
    @patch("subprocess.run")
    def test_detect_gpu_linux(self, mock_run, mock_system):
        """Test GPU detection on Linux."""
        with patch.multiple(self.verifier, _detect_gpu=self.verifier._detect_gpu):
            with patch("subprocess.run") as mock_nvidia_check:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_nvidia_check.return_value = mock_result

                self.assertTrue(self.verifier._detect_gpu())

        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = FileNotFoundError("No nvidia-smi")

            with patch("os.path.exists", return_value=False):
                with patch("platform.system", return_value="Linux"):
                    with patch("platform.machine", return_value="x86_64"):
                        self.assertFalse(self.verifier._detect_gpu())

        def mock_path_exists(path):
            return path == "/dev/dri/renderD128"

        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = FileNotFoundError("No nvidia-smi")

            with patch("os.path.exists", side_effect=mock_path_exists):
                with patch("platform.system", return_value="Linux"):
                    with patch("platform.machine", return_value="x86_64"):
                        self.assertTrue(self.verifier._detect_gpu())


if __name__ == "__main__":
    unittest.main()
