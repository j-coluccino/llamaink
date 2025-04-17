import os
import logging
import unittest
from unittest.mock import patch, MagicMock
from llamaink.utils.logging_config import (
    configure_logging,
    ProgressDisplay,
    LOGGING_LEVELS,
)


class TestLoggingConfig(unittest.TestCase):
    def setUp(self):
        self.root_handlers = logging.root.handlers.copy()
        self.root_level = logging.root.level

        if "LLAMAINK_NO_PROGRESS" in os.environ:
            del os.environ["LLAMAINK_NO_PROGRESS"]

    def tearDown(self):
        logging.root.handlers = self.root_handlers
        logging.root.level = self.root_level

        if "LLAMAINK_NO_PROGRESS" in os.environ:
            del os.environ["LLAMAINK_NO_PROGRESS"]

    def test_configure_logging_levels(self):
        """Test configuring different logging levels."""
        for level_name, level in LOGGING_LEVELS.items():
            logging.root.handlers = []

            configure_logging(level_name=level_name)

            self.assertEqual(logging.root.level, level)

            self.assertGreater(len(logging.root.handlers), 0)

    @patch("logging.FileHandler")
    def test_configure_logging_with_file(self, mock_file_handler):
        """Test configuring logging with a log file."""
        mock_instance = MagicMock()
        mock_file_handler.return_value = mock_instance

        configure_logging(log_file="test.log")

        mock_file_handler.assert_called_once_with("test.log", encoding="utf-8")

        mock_instance.setFormatter.assert_called_once()
        mock_instance.setLevel.assert_called_once()

    def test_configure_logging_disable_progress(self):
        """Test disabling progress bars."""
        configure_logging(disable_progress_bars=True)

        self.assertEqual(os.environ.get("LLAMAINK_NO_PROGRESS"), "1")

        configure_logging(disable_progress_bars=False)

        self.assertNotIn("LLAMAINK_NO_PROGRESS", os.environ)

    @patch("tqdm.tqdm")
    def test_progress_display_with_tqdm(self, mock_tqdm):
        """Test progress display with tqdm available."""
        mock_instance = MagicMock()
        mock_tqdm.return_value = mock_instance

        progress = ProgressDisplay("Test progress", total=100)

        mock_tqdm.assert_called_once()
        self.assertEqual(mock_tqdm.call_args[1]["desc"], "Test progress")
        self.assertEqual(mock_tqdm.call_args[1]["total"], 100)

        progress.update(10, "Processing")
        mock_instance.update.assert_called_once_with(10)
        mock_instance.set_postfix_str.assert_called_once_with("Processing")

        progress.close()
        mock_instance.close.assert_called_once()

    def test_progress_display_disabled(self):
        """Test disabled progress display."""
        os.environ["LLAMAINK_NO_PROGRESS"] = "1"

        progress = ProgressDisplay("Test progress", total=100)

        progress.update(10)
        progress.close()


if __name__ == "__main__":
    unittest.main()
