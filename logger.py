"""
Logging utilities for the OpenAI client.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a logger with specified configuration.
    
    Args:
        name: Name of the logger
        level: Logging level, defaults to INFO if not specified
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist
    if not logger.handlers:
        # Set level
        if level is not None:
            logger.setLevel(level)
        else:
            logger.setLevel(logging.INFO)
        
        # Create handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger