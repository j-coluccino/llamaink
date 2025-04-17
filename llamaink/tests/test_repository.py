import os
import tempfile
import unittest
from pathlib import Path
from llamaink.analyser.repository import RepositoryScanner


class TestRepositoryScanner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_path = self.temp_dir.name

        os.makedirs(os.path.join(self.repo_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.repo_path, ".git"), exist_ok=True)
        os.makedirs(os.path.join(self.repo_path, "node_modules"), exist_ok=True)

        with open(os.path.join(self.repo_path, "dbt_project.yml"), "w") as f:
            f.write("name: test_project\nversion: 1.0.0\n")

        with open(os.path.join(self.repo_path, "models", "test_model.sql"), "w") as f:
            f.write("{{ config(materialized='table') }}\nSELECT * FROM source_table")

        with open(os.path.join(self.repo_path, "models", "schema.yml"), "w") as f:
            f.write("version: 2\nmodels:\n  - name: test_model\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init(self):
        """Test scanner initialization."""
        exclude_patterns = [".temp", "logs"]
        scanner = RepositoryScanner(self.repo_path, exclude_patterns)

        self.assertEqual(scanner.repo_path, Path(self.repo_path).resolve())

        self.assertIn(".temp", scanner.exclude_patterns)
        self.assertIn("logs", scanner.exclude_patterns)

        self.assertNotIn(".git", scanner.exclude_patterns)

        scanner_with_defaults = RepositoryScanner(self.repo_path, None)
        self.assertIn(".git", scanner_with_defaults.exclude_patterns)

    def test_add_gitignore_patterns(self):
        """Test adding patterns from .gitignore file."""
        gitignore_content = "*.log\nbuild/\n/dist/\n"
        with open(os.path.join(self.repo_path, ".gitignore"), "w") as f:
            f.write(gitignore_content)

        scanner = RepositoryScanner(self.repo_path)

        self.assertIn("*.log", scanner.exclude_patterns)
        self.assertIn("build", scanner.exclude_patterns)
        self.assertIn("dist", scanner.exclude_patterns)

    def test_should_exclude(self):
        """Test path exclusion logic."""
        scanner = RepositoryScanner(self.repo_path)

        self.assertTrue(
            scanner._should_exclude(os.path.join(self.repo_path, ".git", "config"))
        )
        self.assertTrue(
            scanner._should_exclude(
                os.path.join(self.repo_path, "node_modules", "package.json")
            )
        )
        self.assertTrue(
            scanner._should_exclude(os.path.join(self.repo_path, "file.pyc"))
        )

        self.assertFalse(
            scanner._should_exclude(
                os.path.join(self.repo_path, "models", "test_model.sql")
            )
        )
        self.assertFalse(
            scanner._should_exclude(os.path.join(self.repo_path, "dbt_project.yml"))
        )

    def test_is_dbt_file_by_pattern(self):
        """Test DBT file pattern recognition."""
        scanner = RepositoryScanner(self.repo_path)

        self.assertTrue(scanner._is_dbt_file_by_pattern("dbt_project.yml"))
        self.assertTrue(scanner._is_dbt_file_by_pattern("models/test_model.sql"))
        self.assertTrue(scanner._is_dbt_file_by_pattern("models/schema.yml"))
        self.assertTrue(scanner._is_dbt_file_by_pattern("snapshots/test_snapshot.sql"))

        self.assertFalse(scanner._is_dbt_file_by_pattern("README.md"))
        self.assertFalse(scanner._is_dbt_file_by_pattern("src/main.py"))

    def test_is_dbt_file(self):
        """Test DBT file content recognition."""
        scanner = RepositoryScanner(self.repo_path)

        dbt_file_path = os.path.join(self.repo_path, "test_dbt.sql")
        with open(dbt_file_path, "w") as f:
            f.write("{{ ref('some_model') }}\nSELECT * FROM table")

        non_dbt_file_path = os.path.join(self.repo_path, "regular.sql")
        with open(non_dbt_file_path, "w") as f:
            f.write("SELECT * FROM table")

        self.assertTrue(scanner._is_dbt_file(Path(dbt_file_path)))
        self.assertFalse(scanner._is_dbt_file(Path(non_dbt_file_path)))

    def test_scan(self):
        """Test full repository scanning."""
        scanner = RepositoryScanner(self.repo_path)
        dbt_files = scanner.scan()

        self.assertIn("dbt_project.yml", dbt_files)
        self.assertIn(
            os.path.join("models", "test_model.sql").replace("\\", "/"), dbt_files
        )
        self.assertIn(
            os.path.join("models", "schema.yml").replace("\\", "/"), dbt_files
        )

        for file_path in dbt_files:
            self.assertNotIn(".git", file_path)
            self.assertNotIn("node_modules", file_path)


if __name__ == "__main__":
    unittest.main()
