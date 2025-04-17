import os
import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import tempfile
import argparse

from llamaink.cli import (
    setup_argparser,
    save_docs,
    check_repository_path,
    main,
    supports_ansi_colors,
    colorize,
    print_banner,
)


class TestCliHelpers(unittest.TestCase):
    def test_supports_ansi_colors(self):
        """Test ANSI color support detection."""

        result = supports_ansi_colors()
        self.assertIsInstance(result, bool)

    def test_colorize(self):
        """Test text colorization."""
        text = "Test Text"
        color = "\033[32m"

        with patch("llamaink.cli.supports_ansi_colors", return_value=True):
            result = colorize(text, color)
            self.assertEqual(result, f"{color}{text}\033[0m")

        with patch("llamaink.cli.supports_ansi_colors", return_value=False):
            result = colorize(text, color)
            self.assertEqual(result, text)

    @patch("llamaink.cli.get_llamaink_version", return_value="0.1.0")
    @patch("llamaink.cli.supports_ansi_colors", return_value=False)
    @patch("builtins.print")
    def test_print_banner_no_ansi(self, mock_print, mock_supports_ansi, mock_version):
        """Test banner printing without ANSI support."""
        print_banner()

        mock_print.assert_any_call(
            "\nLlamaInk - Privacy-First DBT Documentation Generator"
        )
        mock_print.assert_any_call("Version: 0.1.0")
        mock_print.assert_any_call("=" * 50 + "\n")

    @patch("llamaink.cli.get_llamaink_version", return_value="0.1.0")
    @patch("llamaink.cli.supports_ansi_colors", return_value=True)
    @patch("builtins.print")
    def test_print_banner_with_ansi(self, mock_print, mock_supports_ansi, mock_version):
        """Test banner printing with ANSI support."""
        print_banner()

        calls = mock_print.call_args_list

        self.assertTrue(any("LlamaInk" in str(call_args) for call_args in calls))

        self.assertTrue(any("0.1.0" in str(call_args) for call_args in calls))


class TestArgParser(unittest.TestCase):
    def test_setup_argparser(self):
        """Test argument parser setup."""
        parser = setup_argparser()

        self.assertIsInstance(parser, argparse.ArgumentParser)

        args = parser.parse_args(["--repo", "/test/repo", "--verbose"])

        self.assertEqual(args.repo, "/test/repo")
        self.assertTrue(args.verbose)
        self.assertFalse(args.quiet)

        with self.assertRaises(SystemExit):
            parser.parse_args(["--verbose", "--quiet"])


class TestSaveDocs(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

        self.docs = {
            "file1.md": "# File 1 Content",
            "subdir/file2.md": "# File 2 Content",
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("llamaink.cli.os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("llamaink.cli.logger")
    def test_save_docs_success(self, mock_logger, mock_open, mock_makedirs):
        """Test successful document saving."""
        result = save_docs(self.docs, self.output_dir)

        self.assertTrue(result)

        mock_makedirs.assert_any_call(self.output_dir, exist_ok=True)
        mock_makedirs.assert_any_call(
            os.path.dirname(os.path.join(self.output_dir, "subdir/file2.md")),
            exist_ok=True,
        )

        expected_calls = [
            call(os.path.join(self.output_dir, "file1.md"), "w", encoding="utf-8"),
            call(
                os.path.join(self.output_dir, "subdir/file2.md"), "w", encoding="utf-8"
            ),
        ]
        mock_open.assert_has_calls(expected_calls, any_order=True)

        mock_open().write.assert_any_call("# File 1 Content")
        mock_open().write.assert_any_call("# File 2 Content")

        mock_logger.info.assert_any_call(
            f"Saved {os.path.join(self.output_dir, 'file1.md')}"
        )
        mock_logger.info.assert_any_call(
            f"Saved {os.path.join(self.output_dir, 'subdir/file2.md')}"
        )

    @patch("llamaink.cli.os.makedirs")
    @patch("llamaink.cli.logger")
    def test_save_docs_failure(self, mock_logger, mock_makedirs):
        """Test document saving failure."""

        mock_makedirs.side_effect = Exception("Permission denied")

        result = save_docs(self.docs, self.output_dir)

        self.assertFalse(result)
        mock_logger.error.assert_called_once()


class TestRepositoryCheck(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_path = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("llamaink.cli.logger")
    def test_check_repository_path_valid(self, mock_logger):
        """Test valid repository path check."""

        with open(os.path.join(self.repo_path, "test_file.txt"), "w") as f:
            f.write("Test content")

        result = check_repository_path(self.repo_path)

        self.assertTrue(result)

    @patch("llamaink.cli.logger")
    def test_check_repository_path_nonexistent(self, mock_logger):
        """Test nonexistent repository path check."""
        nonexistent_path = os.path.join(self.repo_path, "nonexistent")

        result = check_repository_path(nonexistent_path)

        self.assertFalse(result)
        mock_logger.error.assert_called_once()

    @patch("llamaink.cli.logger")
    def test_check_repository_path_empty(self, mock_logger):
        """Test empty repository path check."""

        result = check_repository_path(self.repo_path)

        self.assertFalse(result)
        mock_logger.error.assert_called_once()


class TestMain(unittest.TestCase):
    @patch("llamaink.cli.print_banner")
    @patch("llamaink.cli.setup_argparser")
    @patch("llamaink.cli.configure_logging")
    @patch("llamaink.cli.Config")
    def test_main_initialization(
        self,
        mock_config_class,
        mock_configure_logging,
        mock_setup_argparser,
        mock_print_banner,
    ):
        """Test main function initialization."""

        mock_args = MagicMock()
        mock_args.init = True
        mock_args.verbose = False
        mock_args.quiet = False
        mock_args.debug = False
        mock_args.log_file = None
        mock_args.no_progress = False

        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = mock_args
        mock_setup_argparser.return_value = mock_parser

        mock_config_instance = MagicMock()
        mock_config_instance.initialize.return_value = True
        mock_config_class.return_value = mock_config_instance

        result = main()

        self.assertEqual(result, 0)
        mock_print_banner.assert_called_once()
        mock_setup_argparser.assert_called_once()
        mock_configure_logging.assert_called_once()
        mock_config_instance.initialize.assert_called_once()

    @patch("llamaink.cli.print_banner")
    @patch("llamaink.cli.setup_argparser")
    @patch("llamaink.cli.configure_logging")
    @patch("llamaink.cli.Config")
    @patch("llamaink.cli.check_repository_path", return_value=False)
    def test_main_invalid_repository(
        self,
        mock_check_repo,
        mock_config_class,
        mock_configure_logging,
        mock_setup_argparser,
        mock_print_banner,
    ):
        """Test main function with invalid repository."""

        mock_args = MagicMock()
        mock_args.init = False
        mock_args.repo = "/fake/repo"
        mock_args.verbose = False
        mock_args.quiet = False
        mock_args.debug = False
        mock_args.log_file = None
        mock_args.no_progress = False
        mock_args.check_model = False

        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = mock_args
        mock_setup_argparser.return_value = mock_parser

        mock_config_instance = MagicMock()
        mock_config_class.return_value = mock_config_instance

        result = main()

        self.assertEqual(result, 1)
        mock_check_repo.assert_called_with("/fake/repo")

    @patch("llamaink.cli.print_banner")
    @patch("llamaink.cli.setup_argparser")
    @patch("llamaink.cli.configure_logging")
    @patch("llamaink.cli.Config")
    @patch("llamaink.cli.ModelVerifier")
    def test_main_check_model(
        self,
        mock_verifier_class,
        mock_config_class,
        mock_configure_logging,
        mock_setup_argparser,
        mock_print_banner,
    ):
        """Test main function with model check."""

        mock_args = MagicMock()
        mock_args.init = False
        mock_args.check_model = True
        mock_args.model = None
        mock_args.verbose = False
        mock_args.quiet = False
        mock_args.debug = False
        mock_args.log_file = None
        mock_args.no_progress = False

        mock_parser = MagicMock()
        mock_parser.parse_args.return_value = mock_args
        mock_setup_argparser.return_value = mock_parser

        mock_config_instance = MagicMock()
        mock_config_instance.get_config.return_value = MagicMock(
            model_path="/test/model.gguf"
        )
        mock_config_class.return_value = mock_config_instance

        mock_verifier_instance = MagicMock()
        mock_verifier_instance.verify_model.return_value = {
            "model_path": "/test/model.gguf",
            "exists": True,
            "format": "GGUF",
            "size_mb": 1024,
            "size_gb": 1.0,
            "quantization": "q4_k",
            "compatible_with_system": True,
            "warnings": [],
            "recommendations": [],
        }
        mock_verifier_class.return_value = mock_verifier_instance

        with patch("builtins.print") as mock_print:
            result = main()

        self.assertEqual(result, 0)
        mock_verifier_instance.verify_model.assert_called_with("/test/model.gguf")

        self.assertTrue(
            any(
                "model verification" in str(call_args).lower()
                for call_args in mock_print.call_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
