import os
import json
import base64
from typing import Dict, List, Optional, Any, Union
import logging
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from logger import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    """Client for interacting with OpenAI API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Please set OPENAI_API_KEY environment variable "
                "or provide api_key parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info("OpenAI client initialized")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate a chat completion response.
        
        Args:
            messages: List of message dictionaries (role and content).
            model: Model name to use for completion.
            temperature: Sampling temperature between 0 and 2.
            max_tokens: Maximum number of tokens to generate.
            top_p: Nucleus sampling parameter.
            frequency_penalty: Frequency penalty parameter.
            presence_penalty: Presence penalty parameter.
            
        Returns:
            Response from the API as a dictionary.
        """
        logger.debug(f"Generating chat completion with model: {model}")
        
        # Convert messages to the expected format
        formatted_messages: List[ChatCompletionMessageParam] = []
        for msg in messages:
            if isinstance(msg["content"], list):
                # Handle message with image content
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            else:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            
            # Convert to dictionary for consistent return format
            response_dict = {
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                        },
                        "finish_reason": choice.finish_reason,
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
            }
            
            logger.debug(f"Chat completion generated successfully. Usage: {response_dict['usage']}")
            return response_dict
        
        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            raise
    
    def create_embedding(
        self,
        text: str,
        model: str = "text-embedding-3-large",
    ) -> List[float]:
        """Create an embedding for text.
        
        Args:
            text: Text to embed.
            model: Model name to use for embedding.
            
        Returns:
            List of embedding values.
        """
        logger.debug(f"Creating embedding with model: {model}")
        
        try:
            response = self.client.embeddings.create(
                model=model,
                input=text,
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Embedding created successfully. Dimensions: {len(embedding)}")
            return embedding
        
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise
    
    def chat_with_image(
        self,
        messages: List[Dict[str, Any]],
        image_path: str,
        model: str = "gpt-4o-vision-preview",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """Generate a chat completion with image input.
        
        Args:
            messages: List of message dictionaries (role and content).
            image_path: Path to the image file.
            model: Model name to use for completion.
            temperature: Sampling temperature between 0 and 2.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            Response from the API as a dictionary.
        """
        logger.debug(f"Generating chat completion with image using model: {model}")
        
        try:
            # Read image file and encode as base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create messages with image content
            formatted_messages = []
            
            # Add previous messages
            for msg in messages[:-1]:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add the last message with the image
            last_msg = messages[-1]
            formatted_messages.append({
                "role": last_msg["role"],
                "content": [
                    {"type": "text", "text": last_msg["content"]},
                    {"type": "image_url", 
                     "image_url": {
                         "url": f"data:image/jpeg;base64,{base64_image}",
                         "detail": "high"
                     }
                    }
                ]
            })
            
            response = self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Convert to dictionary for consistent return format
            response_dict = {
                "id": response.id,
                "object": response.object,
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                        },
                        "finish_reason": choice.finish_reason,
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
            }
            
            logger.debug(f"Chat completion with image generated successfully. Usage: {response_dict['usage']}")
            return response_dict
        
        except Exception as e:
            logger.error(f"Error generating chat completion with image: {str(e)}")
            raise