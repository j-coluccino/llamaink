import logging
import re
from typing import Dict, Tuple, List, Any

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages prompt context windows to prevent exceeding model limits.

    This utility helps ensure that prompts don't exceed the model's context
    window, and provides strategies for handling prompts that are too large.
    """

    def __init__(self, context_size: int = 8192) -> None:
        """
        Initialize the context manager.

        Args:
            context_size: Maximum context size in tokens
        """
        self.context_size: int = context_size
        self.reserved_output_tokens: int = max(512, int(self.context_size * 0.2))

    def estimate_token_count(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        This is a rough estimate - actual token count depends on the tokenizer.

        Args:
            text: Text to estimate token count for

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def ensure_context_fits(
        self, prompt: str, max_tokens: int
    ) -> Tuple[str, int, Dict[str, Any]]:
        """
        Ensure the prompt and output tokens fit within context size.

        Args:
            prompt: The input prompt
            max_tokens: Requested output tokens

        Returns:
            Tuple of (adjusted_prompt, adjusted_max_tokens, modifications)
        """
        est_input_tokens = self.estimate_token_count(prompt)
        available_context = self.context_size - self.reserved_output_tokens

        modifications: Dict[str, Any] = {
            "input_modified": False,
            "output_modified": False,
            "original_input_tokens": est_input_tokens,
            "original_max_tokens": max_tokens,
            "available_context": available_context,
        }

        if est_input_tokens + max_tokens > self.context_size:
            adj_max_tokens = min(max_tokens, self.context_size - est_input_tokens)

            if est_input_tokens > available_context:
                logger.warning(
                    f"Prompt is too large ({est_input_tokens} tokens) for context window "
                    f"({available_context} available tokens). Trimming prompt."
                )
                prompt = self.trim_prompt(prompt, available_context)
                modifications["input_modified"] = True

            if adj_max_tokens != max_tokens:
                logger.warning(
                    f"Requested max_tokens ({max_tokens}) adjusted to {adj_max_tokens} "
                    f"to fit within context window."
                )
                max_tokens = adj_max_tokens
                modifications["output_modified"] = True

        return prompt, max_tokens, modifications

    def trim_prompt(self, prompt: str, target_token_count: int) -> str:
        """
        Trim a prompt to fit within a target token count.

        This method tries to preserve structure by keeping important sections.

        Args:
            prompt: The prompt to trim
            target_token_count: Target token count

        Returns:
            Trimmed prompt
        """
        est_tokens = self.estimate_token_count(prompt)
        if est_tokens <= target_token_count:
            return prompt

        conservative_target = int(target_token_count * 0.95)

        sections = self._split_into_sections(prompt)

        trimmed_prompt = self._selective_section_trimming(sections, conservative_target)

        if self.estimate_token_count(trimmed_prompt) > target_token_count:
            target_chars = target_token_count * 4

            if len(prompt) > target_chars:
                prefix_size = int(target_chars * 0.7)
                suffix_size = target_chars - prefix_size

                trimmed_prompt = (
                    prompt[:prefix_size]
                    + "\n\n[...content trimmed...]\n\n"
                    + prompt[-suffix_size:]
                )

        return trimmed_prompt

    def _split_into_sections(self, prompt: str) -> List[Tuple[str, int]]:
        """
        Split a prompt into meaningful sections with priority.

        Returns:
            List of (section_text, priority) tuples
        """
        sections: List[Tuple[str, int]] = []

        parts = re.split(r"\n\n+", prompt)

        for i, part in enumerate(parts):
            priority = 5

            if re.match(r"^#+\s", part) or "IMPORTANT" in part:
                priority = 1

            elif "context" in part.lower() or "file:" in part.lower():
                priority = 2

            elif i < 3:
                priority = 3

            elif i == len(parts) - 1:
                priority = 3

            sections.append((part, priority))

        return sections

    def _selective_section_trimming(
        self, sections: List[Tuple[str, int]], target_token_count: int
    ) -> str:
        """
        Trim sections selectively based on priority.

        Args:
            sections: List of (section_text, priority) tuples
            target_token_count: Target token count

        Returns:
            Trimmed prompt text
        """
        sorted_sections = sorted(sections, key=lambda x: x[1])

        kept_sections: List[str] = []
        current_tokens = 0

        for section, priority in sorted_sections:
            if priority <= 2:
                kept_sections.append(section)
                current_tokens += self.estimate_token_count(section)

        for section, priority in sorted_sections:
            if priority > 2:
                section_tokens = self.estimate_token_count(section)
                if current_tokens + section_tokens <= target_token_count:
                    kept_sections.append(section)
                    current_tokens += section_tokens

        return "\n\n".join(kept_sections)
