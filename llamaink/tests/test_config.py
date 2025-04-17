import os
import json
import tempfile
import unittest
from unittest.mock import patch, mock_open
from llamaink.config import Config, LlamaInkConfig


class TestLlamaInkConfig(unittest.TestCase):
    def setUp(self):

        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name

        self.test_config = {
            "model_path": "/test/model/path.gguf",
            "context_size": 4096,
            "use_gpu": False,
            "llama_binary_path": "/test/binary/path",
            "exclude_patterns": [".git", "node_modules"],
            "output_dir": "test_docs",
        }

    def tearDown(self):

        self.temp_dir.cleanup()

    def test_default_config_values(self):
        """Test that default config values are set correctly."""
        config = LlamaInkConfig()
        self.assertEqual(config.context_size, 8192)
        self.assertTrue(config.use_gpu)
        self.assertEqual(config.output_dir, "docs")
        self.assertIn(".git", config.exclude_patterns)

    def test_get_default_model_path(self):
        """Test getting the default model path."""
        expected_path = os.path.join(
            os.path.expanduser("~"),
            ".llamaink",
            "models",
            "Meta-Llama-3-8B-Instruct.Q2_K.gguf",
        )
        self.assertEqual(LlamaInkConfig.get_default_model_path(), expected_path)

    def test_from_file_existing(self):
        """Test loading config from an existing file."""
        config_path = os.path.join(self.temp_path, "llamaink.json")

        with open(config_path, "w") as f:
            json.dump(self.test_config, f)

        config = LlamaInkConfig.from_file(config_path)

        self.assertEqual(config.model_path, self.test_config["model_path"])
        self.assertEqual(config.context_size, self.test_config["context_size"])
        self.assertEqual(config.use_gpu, self.test_config["use_gpu"])
        self.assertEqual(config.exclude_patterns, self.test_config["exclude_patterns"])

    def test_from_file_nonexistent(self):
        """Test loading config from a non-existent file."""
        config = LlamaInkConfig.from_file("/path/does/not/exist.json")

        self.assertEqual(config.model_path, LlamaInkConfig.get_default_model_path())
        self.assertEqual(config.context_size, 8192)

    @patch("llamaink.config.os.path.dirname")
    @patch("llamaink.config.os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_config(self, mock_file, mock_makedirs, mock_dirname):
        """Test saving config to a file."""

        mock_dirname.return_value = os.path.join(self.temp_path, "new_dir")

        config = LlamaInkConfig(**self.test_config)
        config_path = os.path.join(self.temp_path, "new_dir", "config.json")

        with patch("json.dump") as mock_json_dump:
            result = config.save(config_path)

        self.assertTrue(result)

        mock_file.assert_called_once_with(config_path, "w", encoding="utf-8")
        mock_json_dump.assert_called_once()


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch.object(LlamaInkConfig, "from_file")
    def test_init(self, mock_from_file):
        """Test Config initialization."""

        test_config = LlamaInkConfig()
        mock_from_file.return_value = test_config

        config_manager = Config("test_path.json")

        mock_from_file.assert_called_once_with("test_path.json")

        self.assertEqual(config_manager.config, test_config)

    @patch.object(LlamaInkConfig, "save")
    @patch("os.makedirs")
    def test_initialize(self, mock_makedirs, mock_save):
        """Test Config initialization method."""

        mock_save.return_value = True

        config_manager = Config()
        result = config_manager.initialize()

        self.assertTrue(result)
        self.assertEqual(mock_makedirs.call_count, 2)
        mock_save.assert_called_once()

    def test_get_config(self):
        """Test get_config method."""
        config_manager = Config()
        config = config_manager.get_config()

        self.assertIsInstance(config, LlamaInkConfig)


if __name__ == "__main__":
    unittest.main()
