import logging
import os
import sys
from typing import Optional, Dict, Any

LOGGING_LEVELS: Dict[str, int] = {
    "quiet": logging.ERROR,
    "normal": logging.INFO,
    "verbose": logging.DEBUG,
    "debug": 5,
}


def _configure_llama_cpp_logger(level: int) -> None:
    """
    Configure the llama-cpp-python logger to a specific level.

    Args:
        level: The logging level to set
    """
    llama_logger = logging.getLogger("llama_cpp")

    if level <= 5:
        llama_logger.setLevel(logging.DEBUG)
    else:
        llama_logger.setLevel(logging.WARNING)


def configure_logging(
    level_name: str = "normal",
    log_file: Optional[str] = None,
    disable_progress_bars: bool = False,
) -> None:
    """
    Configure application-wide logging settings.

    Args:
        level_name: Friendly name for log level ("quiet", "normal", "verbose", "debug")
        log_file: Optional file path to write logs to
        disable_progress_bars: Whether to disable progress bar outputs
    """
    logging.addLevelName(5, "DEBUG_DETAIL")

    level = LOGGING_LEVELS.get(level_name.lower(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            console_handler.setLevel(logging.DEBUG)
            root_logger.error(f"Could not create log file: {str(e)}")

    _configure_llama_cpp_logger(level)

    for lib_logger_name in ["urllib3", "matplotlib", "PIL"]:
        logging.getLogger(lib_logger_name).setLevel(logging.WARNING)

    if disable_progress_bars:
        os.environ["LLAMAINK_NO_PROGRESS"] = "1"
    else:
        os.environ.pop("LLAMAINK_NO_PROGRESS", None)


class ProgressDisplay:
    """
    Helper class for displaying progress and status information to users.
    This provides a consistent interface that works in both interactive and
    non-interactive environments.
    """

    def __init__(
        self, desc: str = "", total: Optional[int] = None, disable: bool = False
    ) -> None:
        """
        Initialize a progress display.

        Args:
            desc: Description of the task
            total: Total number of steps (if known)
            disable: Whether to disable display
        """
        self.desc: str = desc
        self.total: Optional[int] = total
        self.disable: bool = disable or os.environ.get("LLAMAINK_NO_PROGRESS") == "1"
        self.current: int = 0
        self._tqdm: Any = None

        if not self.disable and total is not None:
            try:
                from tqdm import tqdm

                self._tqdm = tqdm(total=total, desc=desc, unit="files")
            except ImportError:
                self._tqdm = None
                logging.debug("tqdm not available, using simple progress display")

        if not self.disable and self._tqdm is None:
            logging.info(f"{desc}...")

    def update(self, amount: int = 1, status: Optional[str] = None) -> None:
        """
        Update the progress display.

        Args:
            amount: Number of steps to increment
            status: Optional status message to display
        """
        if self.disable:
            return

        self.current += amount

        if self._tqdm is not None:
            self._tqdm.update(amount)
            if status:
                self._tqdm.set_postfix_str(status)
        elif status and self.total:
            percent = int(100 * self.current / self.total)
            logging.info(f"{self.desc}: {percent}% - {status}")
        elif status:
            logging.info(f"{self.desc}: {status}")

    def close(self, message: Optional[str] = None) -> None:
        """
        Close the progress display.

        Args:
            message: Optional completion message
        """
        if self.disable:
            return

        if self._tqdm is not None:
            self._tqdm.close()

        if message:
            logging.info(message)
