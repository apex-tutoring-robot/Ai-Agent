import unittest
from src.privacy_manager import PrivacyManager

class TestPrivacyManager(unittest.TestCase):
    def setUp(self):
        self.session_id = "test_session"
        self.privacy_manager = PrivacyManager(self.session_id)

    def test_anonymize_for_llm(self):
        user_input = "My name is John Doe."
        sanitized_input, mapping = self.privacy_manager.anonymize_for_llm(user_input)
        self.assertIn("Alex", sanitized_input)
        self.assertEqual(mapping["John Doe"], "Alex")

    def test_restore_personal_response(self):
        llm_response = "Hello Alex, how can I help you?"
        mapping = {"Alex": "John Doe"}
        restored_response = self.privacy_manager.restore_personal_response(llm_response, mapping)
        self.assertIn("John Doe", restored_response)

if __name__ == '__main__':
    unittest.main()