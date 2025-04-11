"""
Tests for the app module.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the app module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import parse_arguments, setup_logging, create_default_messages, main


class TestApp:
    """Test suite for app module."""

    def test_parse_arguments(self):
        """Test argument parsing."""
        with patch('sys.argv', ['app.py']):
            args = parse_arguments()
            assert args.model == "gpt-4o"
            assert args.temperature == 0.7
            assert args.max_tokens == 1000
            assert args.message == "Tell me about the role of AI in healthcare."
            assert args.image is None
            assert args.api_key is None
            assert args.verbose is False
        
        with patch('sys.argv', ['app.py', '--model', 'gpt-3.5-turbo', '--temperature', '0.5', 
                               '--max-tokens', '500', '--message', 'Hello', '--image', 'test.jpg',
                               '--api-key', 'test-key', '--verbose']):
            args = parse_arguments()
            assert args.model == "gpt-3.5-turbo"
            assert args.temperature == 0.5
            assert args.max_tokens == 500
            assert args.message == "Hello"
            assert args.image == "test.jpg"
            assert args.api_key == "test-key"
            assert args.verbose is True

    def test_setup_logging(self, caplog):
        """Test logging setup."""
        # Test with verbose=False
        setup_logging(verbose=False)
        assert caplog.get_records('setup') == []
        
        # Test with verbose=True
        setup_logging(verbose=True)
        assert caplog.get_records('setup') == []

    def test_create_default_messages(self):
        """Test default message creation."""
        messages = create_default_messages("Test message")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "medical information" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test message"

    @patch('app.OpenAIClient')
    @patch('app.parse_arguments')
    def test_main_text_only(self, mock_parse_args, mock_openai_client):
        """Test main function with text-only request."""
        # Configure mocks
        mock_args = MagicMock()
        mock_args.model = "gpt-4o"
        mock_args.temperature = 0.7
        mock_args.max_tokens = 1000
        mock_args.message = "Test message"
        mock_args.image = None
        mock_args.api_key = None
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        mock_client_instance.chat_completion.return_value = mock_response
        
        # Call the function
        result = main()
        
        # Verify the result
        assert result == 0
        mock_client_instance.chat_completion.assert_called_once()
        args, kwargs = mock_client_instance.chat_completion.call_args
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 1000
        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][1]["content"] == "Test message"

    @patch('app.OpenAIClient')
    @patch('app.parse_arguments')
    @patch('os.path.exists')
    def test_main_with_image(self, mock_path_exists, mock_parse_args, mock_openai_client):
        """Test main function with image request."""
        # Configure mocks
        mock_args = MagicMock()
        mock_args.model = "gpt-4o"
        mock_args.temperature = 0.7
        mock_args.max_tokens = 1000
        mock_args.message = "Describe this image"
        mock_args.image = "test.jpg"
        mock_args.api_key = None
        mock_args.verbose = False
        mock_parse_args.return_value = mock_args
        
        mock_path_exists.return_value = True
        
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Image description"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        mock_client_instance.chat_with_image.return_value = mock_response
        
        # Call the function
        result = main()
        
        # Verify the result
        assert result == 0
        mock_client_instance.chat_with_image.assert_called_once()
        args, kwargs = mock_client_instance.chat_with_image.call_args
        assert kwargs["model"] == "gpt-4o-vision-preview"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 1000
        assert kwargs["image_path"] == "test.jpg"
        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][1]["content"] == "Describe this image"

    @patch('app.OpenAIClient')
    @patch('app.parse_arguments')
    def test_main_value_error(self, mock_parse_args, mock_openai_client):
        """Test main function with ValueError."""
        # Configure mocks
        mock_args = MagicMock()
        mock_parse_args.return_value = mock_args
        
        mock_openai_client.side_effect = ValueError("API key required")
        
        # Call the function
        result = main()
        
        # Verify the result
        assert result == 1

    @patch('app.OpenAIClient')
    @patch('app.parse_arguments')
    @patch('os.path.exists')
    def test_main_file_not_found(self, mock_path_exists, mock_parse_args, mock_openai_client):
        """Test main function with FileNotFoundError."""
        # Configure mocks
        mock_args = MagicMock()
        mock_args.model = "gpt-4o"
        mock_args.image = "nonexistent.jpg"
        mock_parse_args.return_value = mock_args
        
        mock_path_exists.return_value = True
        
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat_with_image.side_effect = FileNotFoundError("File not found")
        
        # Call the function
        result = main()
        
        # Verify the result
        assert result == 1

    @patch('app.OpenAIClient')
    @patch('app.parse_arguments')
    def test_main_general_exception(self, mock_parse_args, mock_openai_client):
        """Test main function with general exception."""
        # Configure mocks
        mock_args = MagicMock()
        mock_args.image = None
        mock_parse_args.return_value = mock_args
        
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance
        mock_client_instance.chat_completion.side_effect = Exception("API Error")
        
        # Call the function
        result = main()
        
        # Verify the result
        assert result == 1
