"""
Azure OpenAI client with streaming support.
Generates tutoring responses using Azure OpenAI with streaming output.
"""

import os
import logging
from typing import Iterator, List, Dict, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)


class LLMClient:
    """Azure OpenAI client with streaming chat completions."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        system_prompt_path: Optional[str] = None
    ):
        """
        Initialize Azure OpenAI client.
        
        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            deployment: Deployment name
            api_version: API version
            system_prompt_path: Path to system prompt file
        """
        self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        self.endpoint = endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        self.deployment = deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT')
        self.api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')
        self.system_prompt_path = system_prompt_path or os.getenv('SYSTEM_PROMPT_PATH', './config/system_prompt.txt')
        
        if not all([self.api_key, self.endpoint, self.deployment]):
            raise ValueError("Azure OpenAI credentials not fully provided")
        
        # Initialize client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
        
        # Load system prompt
        self.system_prompt = self._load_system_prompt()
        logger.info("Azure OpenAI client initialized")
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from file."""
        try:
            if os.path.exists(self.system_prompt_path):
                with open(self.system_prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                    logger.info(f"Loaded system prompt from {self.system_prompt_path}")
                    return prompt
            else:
                logger.warning(f"System prompt file not found at {self.system_prompt_path}, using default")
                return "You are Jarvis, a helpful AI tutoring assistant."
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")
            return "You are Jarvis, a helpful AI tutoring assistant."
    
    def generate_response_stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Iterator[str]:
        """
        Generate streaming response from Azure OpenAI.
        
        Args:
            messages: Conversation history (list of message dicts)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Yields:
            Text chunks as they arrive
        """
        try:
            # Prepend system prompt
            full_messages = [{"role": "system", "content": self.system_prompt}] + messages
            
            logger.info(f"Generating response for {len(messages)} messages")
            
            # Create streaming completion
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Stream chunks
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def extract_image_content(self, image_url: str, max_tokens: int = 1000) -> str:
        """
        One-shot GPT-4V call to extract all image content as plain text.
        Called once on the vision turn; result stored in history so the image
        is never re-sent — eliminates per-turn image re-encoding cost/latency.
        """
        extraction_prompt = (
            "You are an image content extractor. Describe every question, equation, "
            "diagram, number, and piece of text visible in this image with complete "
            "accuracy. Write all math in plain spoken form (e.g. '2x plus 5 equals 15'). "
            "Be exhaustive — do not skip any content."
        )
        messages = [
            {"role": "system", "content": extraction_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
            ]},
        ]
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            max_tokens=max_tokens,
            stream=False,
        )
        extracted = response.choices[0].message.content.strip()
        logger.info(f"📷 Image extracted ({len(extracted)} chars)")
        return extracted

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 15
    ) -> str:
        """
        Generate complete response (non-streaming).
        
        Args:
            messages: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Complete response text
        """
        try:
            # Collect all chunks
            chunks = list(self.generate_response_stream(messages, temperature, max_tokens))
            response = ''.join(chunks)
            logger.info(f"Generated response: {response[:100]}...")
            return response
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise


if __name__ == "__main__":
    # Test LLM client
    print("Testing Azure OpenAI client...\n")
    
    try:
        client = LLMClient()
        
        # Test messages
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        print("Streaming response:")
        print("-" * 60)
        
        for chunk in client.generate_response_stream(messages):
            print(chunk, end='', flush=True)
        
        print("\n" + "-" * 60)
        print("\nTest complete!")
    
    except Exception as e:
        print(f"Error: {e}")
