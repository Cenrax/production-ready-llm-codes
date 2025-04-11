"""
Pytest configuration file.
"""

import os
import pytest
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@pytest.fixture(scope="session")
def openai_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return api_key

@pytest.fixture(autouse=True)
def mock_env_variables():
    """Set up environment variables for testing."""
    # We don't need to mock the API key since we want to use the real one
    yield