import logging
from pathlib import Path


#self.logger = Logger("TransformationApplier")
class Logger:
    def __init__(self, name: str, log_file: str = "app_debug.log", log_level: str = "INFO"):
        """
        Initialize a logger with file and console handlers.

        Args:
            name (str): Name of the logger (usually __name__).
            log_file (str): Name of the log file (default: app_debug.log).
            log_level (str): Logging level (default: INFO).
        """
        # Define log directory and ensure it exists
        self.log_dir = Path(__file__).parent.parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / log_file

        # Set up the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Avoid adding handlers if logger already has them
        if self.logger.handlers:
            return

        # Define log format
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(threadName)s - %(name)s - %(message)s"
        )

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log an error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log a critical message."""
        self.logger.critical(message)

