"""
Privacy manager for CHIPPY.
Handles anonymization of user inputs before sending to cloud services.
"""

import re
import hashlib
from typing import Dict, Tuple, List

class PrivacyManager:
    """
    Privacy management class for anonymizing and restoring personal data.
    Ensures that sensitive information isn't sent to cloud services.
    """
    
    def __init__(self, session_id: str):
        """
        Initialize the privacy manager with a session ID.
        
        Args:
            session_id: A unique identifier for the current session
        """
        self.session_id = session_id
        self.dummy_names = ['Alex', 'Sam', 'Jordan', 'Casey', 'Taylor', 'Riley']
        self.placeholder_mappings: Dict[str, str] = {}
        self.name_index = 0
    
    def _get_placeholder_name(self, original_name: str) -> str:
        """Get a consistent placeholder name for an original name."""
        if original_name in self.placeholder_mappings:
            return self.placeholder_mappings[original_name]
        
        # Use next available dummy name
        placeholder = self.dummy_names[self.name_index % len(self.dummy_names)]
        self.name_index += 1
        self.placeholder_mappings[original_name] = placeholder
        return placeholder
    
    def anonymize_for_llm(self, user_input: str) -> Tuple[str, Dict[str, str]]:
        """
        Anonymize user input by replacing personal identifiers with placeholders.
        
        Args:
            user_input: The original user input text
            
        Returns:
            Tuple containing anonymized text and mapping for restoration
        """
        # Very basic name detection - in production would use NER models
        potential_names = re.findall(r'\b[A-Z][a-z]+\b', user_input)
        
        anonymized_text = user_input
        for name in potential_names:
            placeholder = self._get_placeholder_name(name)
            anonymized_text = anonymized_text.replace(name, placeholder)
        
        return anonymized_text, self.placeholder_mappings
    
    def restore_personal_response(self, llm_response: str) -> str:
        """
        Restore personal context in LLM responses by replacing placeholders with real names.
        
        Args:
            llm_response: The response from the LLM containing placeholder names
            
        Returns:
            Response with original names restored
        """
        restored_response = llm_response
        
        # Create reverse mapping (placeholder → original)
        reverse_mappings = {v: k for k, v in self.placeholder_mappings.items()}
        
        # Replace all placeholders with original names
        for placeholder, original in reverse_mappings.items():
            restored_response = restored_response.replace(placeholder, original)
            
        return restored_response