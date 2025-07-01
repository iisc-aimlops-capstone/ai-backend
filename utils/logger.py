import logging
from typing import Optional

def get_logger(name: str, log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Creates a logger with optional file handler.
    Args:
        name (str): Logger name.
        log_level (str): Log level, e.g., 'INFO', 'DEBUG'.
        log_file (Optional[str]): Path to log file. If None, logs only to console.
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), "INFO"))

    formatter = logging.Formatter(
        "[%(asctime)s] << %(levelname)s >> [%(filename)s:%(funcName)s]: \n%(message)s\n---"
        )

    # Stream handler (Console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler (if path provided)
    if log_file!=None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger