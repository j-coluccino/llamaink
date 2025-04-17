import unittest
from llamaink.llm.prompts import get_prompt_for_file_type


class TestPrompts(unittest.TestCase):
    def test_get_prompt_for_dbt(self):
        """Test getting a DBT prompt."""
        context = "File: model.sql\n```sql\nSELECT * FROM table\n```"
        repo_name = "test_repo"

        prompt = get_prompt_for_file_type("dbt", context, repo_name=repo_name)

        self.assertIn("<|begin_of_text|>", prompt)
        self.assertIn("system", prompt)
        self.assertIn("user", prompt)
        self.assertIn("assistant", prompt)

        self.assertIn(context, prompt)

        self.assertIn(repo_name, prompt)

        self.assertIn("# DBT Documentation", prompt)
        self.assertIn("Models section", prompt)
        self.assertIn("Entity Relationships", prompt)
        self.assertIn("Data Flow", prompt)

    def test_unsupported_file_type(self):
        """Test behavior with an unsupported file type."""
        with self.assertRaises(ValueError) as context:
            get_prompt_for_file_type("unsupported", "context")

        self.assertIn("Unsupported file type", str(context.exception))


if __name__ == "__main__":
    unittest.main()
