import unittest
from unittest.mock import patch, MagicMock
from llamaink.generators.dbt_docs import DBTDocumentationGenerator
from llamaink.llm.prompts import DISCLAIMER
from llamaink.llm.engine import LlamaEngine


class TestDBTDocumentationGenerator(unittest.TestCase):
    def setUp(self):

        self.mock_engine = MagicMock(spec=LlamaEngine)
        self.mock_engine.generate.return_value = "# Generated DBT Documentation"
        self.mock_engine.verbose = False

        self.generator = DBTDocumentationGenerator(self.mock_engine)

    def test_generate(self):
        """Test the main documentation generation method."""
        repo_path = "/fake/repo"
        dbt_files = ["models/test.sql", "dbt_project.yml"]

        with patch.object(
            self.generator, "_get_file_content", return_value="File content"
        ):
            docs = self.generator.generate(repo_path, dbt_files)

            self.assertIn("dbt.md", docs)
            self.assertEqual(
                docs["dbt.md"], "# Generated DBT Documentation" + DISCLAIMER
            )

            self.assertNotIn("prompt.txt", docs)

    def test_generate_verbose(self):
        """Test documentation generation in verbose mode."""
        repo_path = "/fake/repo"
        dbt_files = ["models/test.sql", "dbt_project.yml"]

        self.mock_engine.verbose = True

        with patch.object(
            self.generator, "_get_file_content", return_value="File content"
        ):
            docs = self.generator.generate(repo_path, dbt_files)

            self.assertIn("prompt.txt", docs)

    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="SELECT * FROM table",
    )
    def test_get_file_content(self, mock_open, mock_exists):
        """Test extracting file content."""
        repo_path = "/fake/repo"
        dbt_files = ["models/test.sql", "models/schema.yml"]

        content = self.generator._get_file_content(repo_path, dbt_files)

        self.assertEqual(mock_open.call_count, 2)

        self.assertIn("File: models/test.sql", content)
        self.assertIn("```sql", content)
        self.assertIn("SELECT * FROM table", content)

    def test_sort_files_by_priority(self):
        """Test file sorting by priority."""
        dbt_files = [
            "models/test.sql",
            "dbt_project.yml",
            "schema.yml",
            "other_file.txt",
            "models/schema.yml",
        ]

        sorted_files = self.generator._sort_files_by_priority(dbt_files)

        self.assertEqual(sorted_files[0], "dbt_project.yml")

        self.assertTrue(
            sorted_files.index("schema.yml") < sorted_files.index("models/test.sql")
            or sorted_files.index("models/schema.yml")
            < sorted_files.index("models/test.sql")
        )

        self.assertEqual(sorted_files[-1], "other_file.txt")


if __name__ == "__main__":
    unittest.main()
