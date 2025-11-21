"""
Azure OpenAI Client with Streaming Support and Conversation History
Replaces the Azure Prompt Flow client with direct Azure OpenAI API calls.
"""

from openai import AzureOpenAI
from typing import Generator, List, Dict, Optional
import os
from chippy_system_prompt import get_chippy_system_prompt


class AzureOpenAIClient:
    """
    Client for streaming responses from Azure OpenAI base model.
    Includes conversation history management for contextual responses.
    """
    
    def __init__(self, 
                 endpoint: str, 
                 api_key: str, 
                 deployment: str,
                 api_version: str = "2024-12-01-preview",
                 grade: int = 5,
                 topic: str = "general"):
        """
        Initialize the Azure OpenAI client.
        
        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: API key for authentication
            deployment: Deployment name (e.g., "baseModelGPT-4.1")
            api_version: API version
            grade: Student's grade level
            topic: Current topic being taught
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment = deployment
        self.api_version = api_version
        self.grade = grade
        self.topic = topic
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        
        # Conversation history (stores messages)
        self.conversation_history: List[Dict[str, str]] = []
        
        # Conversation mode tracking
        self.conversation_mode = "greeting"  # greeting, tutoring, or closing
        self.exchange_count = 0
        
        print(f"✅ Azure OpenAI Client initialized")
        print(f"   Endpoint: {endpoint}")
        print(f"   Deployment: {deployment}")
        print(f"   Grade Level: {grade}")
    
    def set_mode(self, mode: str):
        """
        Set the conversation mode (greeting, tutoring, closing).
        
        Args:
            mode: Conversation mode
        """
        if mode in ["greeting", "tutoring", "closing"]:
            self.conversation_mode = mode
            print(f"📝 Conversation mode set to: {mode}")
        else:
            print(f"⚠️  Unknown mode: {mode}, keeping current mode: {self.conversation_mode}")
    
    def set_topic(self, topic: str):
        """Update the topic being taught."""
        self.topic = topic
        print(f"📚 Topic updated to: {topic}")
    
    def add_to_history(self, role: str, content: str):
        """
        Add a message to conversation history.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Keep history manageable (last 20 messages = 10 exchanges)
        if len(self.conversation_history) > 20:
            # Keep first message if it was important, remove oldest exchanges
            self.conversation_history = self.conversation_history[-20:]
    
    def clear_history(self):
        """Clear conversation history (e.g., for new session)."""
        self.conversation_history = []
        self.exchange_count = 0
        print("🗑️  Conversation history cleared")
    
    def get_streaming_response(self, user_text: str) -> Generator[str, None, None]:
        """
        Get streaming response from Azure OpenAI, yielding text chunks as they arrive.
        
        Args:
            user_text: The user's input text
            
        Yields:
            Text chunks as they arrive from the API
        """
        try:
            # Add user message to history
            self.add_to_history("user", user_text)
            self.exchange_count += 1
            
            # Auto-transition modes based on exchange count
            if self.exchange_count <= 3 and self.conversation_mode != "greeting":
                self.conversation_mode = "greeting"
            elif self.exchange_count > 3 and self.conversation_mode == "greeting":
                self.conversation_mode = "tutoring"
            
            # Get system prompt based on current mode
            system_prompt = get_chippy_system_prompt(
                grade=self.grade,
                topic=self.topic,
                mode=self.conversation_mode
            )
            
            # Prepare messages with system prompt + history
            messages = [
                {"role": "system", "content": system_prompt}
            ] + self.conversation_history
            
            # Make streaming request to Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_tokens=800,  # Limit response length for conciseness
                temperature=0.7,  # Balanced creativity
                top_p=0.9,
                stream=True  # Enable streaming
            )
            
            # Accumulate the full response for history
            full_response = []
            
            # Stream the response
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response.append(content)
                    yield content
            
            # Add assistant response to history
            complete_response = "".join(full_response)
            self.add_to_history("assistant", complete_response)
            
        except Exception as e:
            print(f"❌ Azure OpenAI error: {e}")
            error_message = "I'm having trouble connecting to my AI brain. Let's try again!"
            self.add_to_history("assistant", error_message)
            yield error_message
    
    def get_complete_response(self, user_text: str) -> str:
        """
        Get complete response (non-streaming fallback).
        
        Args:
            user_text: The user's input text
            
        Returns:
            Complete response text
        """
        try:
            # Use streaming but collect all chunks
            chunks = []
            for chunk in self.get_streaming_response(user_text):
                chunks.append(chunk)
            return "".join(chunks)
            
        except Exception as e:
            print(f"❌ Azure OpenAI error: {e}")
            return "I'm having technical difficulties. Let's continue anyway!"
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the current conversation.
        
        Returns:
            Summary string
        """
        if not self.conversation_history:
            return "No conversation yet."
        
        exchange_count = len(self.conversation_history) // 2
        return f"Conversation: {exchange_count} exchanges, Mode: {self.conversation_mode}, Topic: {self.topic}"
    
    def close(self):
        """Close the client connection."""
        self.client.close()
        print("🔌 Azure OpenAI client closed")