import logging
import os
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO, 
                 format_str: str = '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                 disable_info: bool = True):  # Changed default to True for --quiet by default
    """
    Setup logger with specified configuration
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        format_str: Log message format
        disable_info: If True, INFO logs are filtered out even if level is DEBUG (default: True)
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Only configure handlers if the logger doesn't already have any
    if not logger.handlers:
        logger.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(format_str)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # If disable_info is True and level is DEBUG, use a filter to remove INFO logs
        if disable_info and level <= logging.INFO:
            class NoInfoFilter(logging.Filter):
                def filter(self, record):
                    return record.levelno != logging.INFO
            
            console_handler.addFilter(NoInfoFilter())
        
        logger.addHandler(console_handler)
        
        # File handler if log_file is provided
        if log_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            # File handler always logs everything at the specified level
            logger.addHandler(file_handler)
    
    return logger