"""
Example main script for using the OpenAIClient.
"""

import os
import argparse
import logging
from typing import List, Dict, Any

from openai_client import OpenAIClient
from logger import get_logger

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OpenAI API Client Example")
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o", 
        help="Model to use for chat completion"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="Sampling temperature (0.0-2.0)"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=1000, 
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--message", 
        type=str, 
        default="Tell me about the role of AI in healthcare.", 
        help="Message to send to the API"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        help="Path to image file for vision model queries"
    )
    parser.add_argument(
        "--api-key", 
        type=str, 
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Set up logging level based on verbosity."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def create_default_messages(user_message: str) -> List[Dict[str, str]]:
    """Create default messages list with system and user messages.
    
    Args:
        user_message: Message from the user
        
    Returns:
        List of message dictionaries
    """
    return [
        {
            "role": "system",
            "content": "You are a helpful AI assistant specializing in medical information."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]


def main():
    """Main function to demonstrate OpenAIClient usage."""
    args = parse_arguments()
    setup_logging(args.verbose)
    
    try:
        # Initialize the OpenAI client
        client = OpenAIClient(api_key=args.api_key)
        logger.info(f"Using model: {args.model}")
        
        # Create messages
        messages = create_default_messages(args.message)
        
        # Handle different types of requests
        if args.image and os.path.exists(args.image):
            logger.info(f"Processing request with image: {args.image}")
            response = client.chat_with_image(
                messages=messages,
                image_path=args.image,
                model="gpt-4o-vision-preview" if args.model == "gpt-4o" else args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
        else:
            logger.info("Processing text-only request")
            response = client.chat_completion(
                messages=messages,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
        
        # Extract and print the response content
        content = response["choices"][0]["message"]["content"]
        print("\nResponse:")
        print("-" * 40)
        print(content)
        print("-" * 40)
        
        # Print usage information
        if response.get("usage"):
            print(f"\nToken Usage:")
            print(f"  Prompt tokens: {response['usage']['prompt_tokens']}")
            print(f"  Completion tokens: {response['usage']['completion_tokens']}")
            print(f"  Total tokens: {response['usage']['total_tokens']}")
        
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        return 1
    except FileNotFoundError:
        logger.error(f"Image file not found: {args.image}")
        return 1
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)