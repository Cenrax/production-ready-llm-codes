"""
Tests for the OpenAI client.
"""

import os
import json
import base64
from unittest.mock import patch, mock_open
import pytest

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai_client import OpenAIClient


@pytest.fixture
def client(openai_api_key):
    """Create an OpenAI client for testing."""
    return OpenAIClient(api_key=openai_api_key)


@pytest.fixture
def test_messages():
    """Return sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]


class TestOpenAIClient:
    """Test suite for OpenAIClient."""

    def test_init(self, openai_api_key):
        """Test client initialization."""
        # Test with custom API key
        client = OpenAIClient(api_key="custom-key")
        assert client.api_key == "custom-key"
        
        # Test using environment variable
        client = OpenAIClient()
        assert client.api_key == openai_api_key
        
        # Test missing API key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                OpenAIClient()

    def test_chat_completion(self, client, test_messages):
        """Test chat completion functionality."""
        # Call the method with a simple prompt
        result = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'test successful' and nothing else."}
            ],
            model="gpt-4o",
            temperature=0.7,
            max_tokens=20
        )
        
        # Verify the result has expected structure
        assert "id" in result
        assert "model" in result
        assert "choices" in result
        assert len(result["choices"]) > 0
        assert "message" in result["choices"][0]
        assert "content" in result["choices"][0]["message"]
        assert "test successful" in result["choices"][0]["message"]["content"].lower()

    def test_create_embedding(self, client):
        """Test embedding creation functionality."""
        # Call the method
        result = client.create_embedding(text="Hello world", model="text-embedding-3-small")
        
        # Verify the result is a list of floats
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

    @patch("builtins.open", mock_open(read_data=b"image_data"))
    @patch("base64.b64encode")
    def test_chat_with_image(self, mock_b64encode, client, test_messages):
        """Test chat with image functionality."""
        # Skip this test to avoid unnecessary API costs
        pytest.skip("Skipping image test to avoid API costs")
        
        # Configure the mocks
        mock_b64encode.return_value = b"base64_encoded_image"
        
        # Call the method
        result = client.chat_with_image(
            messages=test_messages,
            image_path="test.jpg",
            model="gpt-4o-vision-preview"
        )
        
        # Verify the result has expected structure
        assert "id" in result
        assert "choices" in result
        assert "message" in result["choices"][0]
        assert "content" in result["choices"][0]["message"]

    def test_chat_with_image_file_not_found(self, client, test_messages):
        """Test exception handling when image file is not found."""
        # Verify exception handling when file is not found
        with pytest.raises(FileNotFoundError):
            client.chat_with_image(
                messages=test_messages,
                image_path="nonexistent.jpg"
            )