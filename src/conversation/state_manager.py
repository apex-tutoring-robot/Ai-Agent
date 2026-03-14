"""
Conversation State Manager.
Manages conversation history and context for multi-turn interactions.
"""

import logging
import threading
from typing import List, Dict, Optional
from collections import deque

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)


class ConversationStateManager:
    """Thread-safe conversation state manager."""
    
    def __init__(self, max_history: int = 20):
        """
        Initialize conversation state manager.
        
        Args:
            max_history: Maximum number of messages to keep in history
        """
        self.max_history = max_history
        self._messages: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
        
        logger.info(f"Conversation state manager initialized (max history: {max_history})")
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to conversation history.
        
        Args:
            content: User message content
        """
        with self._lock:
            self._messages.append({
                "role": "user",
                "content": content
            })
            logger.debug(f"Added user message: {content[:50]}...")
    
    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to conversation history.
        
        Args:
            content: Assistant message content
        """
        with self._lock:
            self._messages.append({
                "role": "assistant",
                "content": content
            })
            logger.debug(f"Added assistant message: {content[:50]}...")
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in conversation history.
        
        Returns:
            List of message dictionaries
        """
        with self._lock:
            return list(self._messages)
    
    def get_recent_messages(self, count: int) -> List[Dict[str, str]]:
        """
        Get recent messages from conversation history.
        
        Args:
            count: Number of recent messages to retrieve
        
        Returns:
            List of recent message dictionaries
        """
        with self._lock:
            messages = list(self._messages)
            return messages[-count:] if count < len(messages) else messages
    
    def clear_history(self) -> None:
        """Clear all conversation history."""
        with self._lock:
            self._messages.clear()
            logger.info("Conversation history cleared")
    
    def get_message_count(self) -> int:
        """Get total number of messages in history."""
        with self._lock:
            return len(self._messages)
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation.
        
        Returns:
            String summary of conversation
        """
        with self._lock:
            if not self._messages:
                return "No conversation history"
            
            user_msgs = sum(1 for msg in self._messages if msg["role"] == "user")
            assistant_msgs = sum(1 for msg in self._messages if msg["role"] == "assistant")
            
            return (f"Conversation: {len(self._messages)} messages "
                   f"({user_msgs} user, {assistant_msgs} assistant)")
    
    def __len__(self) -> int:
        """Get conversation length."""
        return self.get_message_count()
    
    def __str__(self) -> str:
        """String representation."""
        return self.get_conversation_summary()


if __name__ == "__main__":
    # Test conversation state manager
    print("Testing Conversation State Manager\n")
    
    manager = ConversationStateManager(max_history=5)
    
    # Simulate conversation
    manager.add_user_message("Hello! Can you help me with math?")
    manager.add_assistant_message("Of course! I'd be happy to help you with math. What topic are you working on?")
    
    manager.add_user_message("I need help with fractions.")
    manager.add_assistant_message("Great! Fractions are an important concept. What specifically would you like to know?")
    
    manager.add_user_message("How do I add fractions?")
    manager.add_assistant_message("To add fractions, you need to have a common denominator...")
    
    print(f"Conversation summary: {manager}")
    print(f"\nTotal messages: {len(manager)}")
    
    print("\nFull conversation history:")
    print("-" * 60)
    for i, msg in enumerate(manager.get_messages(), 1):
        role = msg["role"].capitalize()
        content = msg["content"]
        print(f"{i}. [{role}]: {content}")
    
    print("\n" + "-" * 60)
    print("\nRecent messages (last 2):")
    for msg in manager.get_recent_messages(2):
        print(f"[{msg['role']}]: {msg['content']}")
    
    print("\n" + "-" * 60)
    print("Testing max history limit (adding more messages)...")
    
    manager.add_user_message("What about multiplication?")
    manager.add_assistant_message("Let me explain multiplication...")
    
    print(f"\nMessages after limit: {len(manager)} (should be capped at 5)")
    print("\nRemaining messages:")
    for i, msg in enumerate(manager.get_messages(), 1):
        print(f"{i}. [{msg['role']}]: {msg['content'][:50]}...")
