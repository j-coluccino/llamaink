import os
import pathlib
from typing import List, Optional
import re


class RepositoryScanner:
    """
    Scans a repository to identify and categorize DBT files.
    Focuses specifically on DBT models, schemas, and configurations.
    """

    def __init__(
        self, repo_path: str, exclude_patterns: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the repository scanner.

        Args:
            repo_path: Path to the repository root
            exclude_patterns: List of glob patterns to exclude from scanning
        """
        self.repo_path = pathlib.Path(repo_path).resolve()

        default_exclude_patterns = [
            ".git",
            ".svn",
            ".hg",
            ".bzr",
            "node_modules",
            "__pycache__",
            "venv",
            "env",
            ".env",
            ".venv",
            ".tox",
            "dist",
            "build",
            "*.egg-info",
            ".DS_Store",
            ".idea",
            ".vscode",
            "*.swp",
            "*.swo",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "*.so",
            "*.dll",
            "*.class",
            "*.o",
            ".cache",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".coverage",
            "package-lock.json",
            "yarn.lock",
            "Pipfile.lock",
            "poetry.lock",
            ".env*",
            "*.pem",
            "*.key",
            "secrets.*",
            "*.log",
            "logs/",
        ]

        self.exclude_patterns: List[str] = exclude_patterns or default_exclude_patterns

        self._add_gitignore_patterns()

        self.dbt_file_patterns = [
            r"dbt_project\.yml",
            r"models/.*\.sql$",
            r"models/.*\.yml$",
            # r"seeds/.*\.csv$",
            r"snapshots/.*\.sql$",
            r"macros/.*\.sql$",
            r"analyses/.*\.sql$",
            r"tests/.*\.sql$",
            r"schema\.yml$",
            r"dbt_modules/.*",
            r"manifest\.json$",
        ]

    def _add_gitignore_patterns(self) -> None:
        """
        Add patterns from .gitignore file if it exists.
        """
        gitignore_path = self.repo_path / ".gitignore"
        if gitignore_path.exists():
            try:
                with open(gitignore_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            pattern = line.lstrip("/")

                            if pattern.endswith("/"):
                                pattern = pattern[:-1]

                            if pattern not in self.exclude_patterns:
                                self.exclude_patterns.append(pattern)
            except Exception as e:
                print(f"Error reading .gitignore: {e}")

    def scan(self) -> List[str]:
        """
        Scan repository and identify DBT files.

        Returns:
            List of DBT file paths
        """
        dbt_files: List[str] = []

        for path in self._walk_repository():
            path_str = str(path)
            rel_path = os.path.relpath(path_str, str(self.repo_path))

            try:
                if os.path.getsize(path_str) > 100 * 1024 * 1024:
                    continue
            except Exception:
                continue

            if self._is_dbt_file_by_pattern(rel_path) or self._is_dbt_file(path):
                dbt_files.append(rel_path)

        return dbt_files

    def _walk_repository(self) -> List[pathlib.Path]:
        """
        Walk through repository, respecting exclude patterns.

        Returns:
            List of Path objects for all non-excluded files
        """
        files: List[pathlib.Path] = []
        for root, dirs, filenames in os.walk(str(self.repo_path)):
            dirs[:] = [
                d for d in dirs if not self._should_exclude(os.path.join(root, d))
            ]

            for filename in filenames:
                file_path = os.path.join(root, filename)
                if not self._should_exclude(file_path):
                    files.append(pathlib.Path(file_path))
        return files

    def _should_exclude(self, path: str) -> bool:
        """
        Check if a path should be excluded.

        Args:
            path: The file or directory path to check

        Returns:
            True if the path should be excluded, False otherwise
        """
        rel_path = os.path.relpath(path, str(self.repo_path))

        for pattern in self.exclude_patterns:
            if pattern.endswith("/") and os.path.isdir(path):
                if rel_path.startswith(pattern[:-1]):
                    return True
            elif "*" in pattern:
                pattern_regex = pattern.replace(".", "\\.").replace("*", ".*")
                if re.match(f"^{pattern_regex}$", rel_path):
                    return True
            else:
                if pattern == rel_path or f"/{pattern}/" in f"/{rel_path}/":
                    return True

        return False

    def _is_dbt_file_by_pattern(self, rel_path: str) -> bool:
        """
        Check if a file matches DBT file patterns.

        Args:
            rel_path: Path to the file relative to the repository

        Returns:
            True if the file matches DBT patterns, False otherwise
        """
        for pattern in self.dbt_file_patterns:
            if re.search(pattern, rel_path):
                return True

        return False

    def _is_dbt_file(self, path: pathlib.Path) -> bool:
        """
        Check if a file is a DBT model, schema, or configuration.

        Args:
            path: Path to the file

        Returns:
            True if the file is related to DBT, False otherwise
        """
        path_str = str(path)

        if path.name == "dbt_project.yml":
            return True

        if path.suffix == ".sql":
            try:
                with open(path_str, "r", encoding="utf-8") as f:
                    content = f.read(512).lower()
                    dbt_patterns = [
                        r"{{[\s-]*ref\s*\(",  # {{ ref( pattern
                        r"{{[\s-]*config\s*\(",  # {{ config( pattern
                        r"{{[\s-]*source\s*\(",  # {{ source( pattern
                        r"{{[\s-]*macro\s*\(",  # {{ macro( pattern
                        r"\{\%\s*set\s+",  # {% set pattern
                        r"\{\%\s*if\s+",  # {% if pattern
                        r"\{\%\s*for\s+",  # {% for pattern
                    ]
                    for pattern in dbt_patterns:
                        if re.search(pattern, content):
                            return True
            except Exception:
                pass

        for dbt_dir in ["models", "seeds", "snapshots", "macros", "analyses", "tests"]:
            if f"/{dbt_dir}/" in f"/{path_str}/":
                if path.suffix == ".sql":
                    return True

                if path.suffix in [".yml", ".yaml"]:
                    return True

        return False
