import unittest
from llamaink.llm.context_manager import ContextManager


class TestContextManager(unittest.TestCase):
    def setUp(self):
        self.context_manager = ContextManager(context_size=4096)

    def test_init(self):
        """Test initialization with different context sizes."""
        manager = ContextManager(context_size=2048)
        self.assertEqual(manager.context_size, 2048)
        self.assertEqual(manager.reserved_output_tokens, 512)

        manager = ContextManager(context_size=16384)
        self.assertEqual(manager.context_size, 16384)
        self.assertEqual(manager.reserved_output_tokens, 3276)

    def test_estimate_token_count(self):
        """Test token count estimation."""
        text = "This is a sample text with 10 words and about 50 characters."
        estimated_tokens = self.context_manager.estimate_token_count(text)

        self.assertEqual(estimated_tokens, len(text) // 4)

        large_text = "x" * 1000
        self.assertEqual(self.context_manager.estimate_token_count(large_text), 250)

    def test_ensure_context_fits_no_adjustment(self):
        """Test when prompt and tokens fit within context."""
        prompt = "Short prompt"
        max_tokens = 1000

        adjusted_prompt, adjusted_max_tokens, modifications = (
            self.context_manager.ensure_context_fits(prompt, max_tokens)
        )

        self.assertEqual(adjusted_prompt, prompt)
        self.assertEqual(adjusted_max_tokens, max_tokens)
        self.assertFalse(modifications["input_modified"])
        self.assertFalse(modifications["output_modified"])

    def test_ensure_context_fits_output_adjustment(self):
        """Test when max_tokens needs adjustment."""
        prompt = "x" * 3000
        max_tokens = 4000

        adjusted_prompt, adjusted_max_tokens, modifications = (
            self.context_manager.ensure_context_fits(prompt, max_tokens)
        )

        self.assertEqual(adjusted_prompt, prompt)
        self.assertLess(adjusted_max_tokens, max_tokens)
        self.assertFalse(modifications["input_modified"])
        self.assertTrue(modifications["output_modified"])

    def test_ensure_context_fits_input_adjustment(self):
        """Test when prompt needs trimming."""
        prompt = "x" * 20000
        max_tokens = 1000

        adjusted_prompt, adjusted_max_tokens, modifications = (
            self.context_manager.ensure_context_fits(prompt, max_tokens)
        )

        self.assertNotEqual(adjusted_prompt, prompt)
        self.assertLess(len(adjusted_prompt), len(prompt))
        self.assertTrue(modifications["input_modified"])

    def test_trim_prompt_not_needed(self):
        """Test when trimming is not needed."""
        prompt = "Short prompt that doesn't need trimming"
        target_tokens = 100

        trimmed = self.context_manager.trim_prompt(prompt, target_tokens)
        self.assertEqual(trimmed, prompt)

    def test_trim_prompt_section_trimming(self):
        """Test selective section trimming."""

        sections = [
            "# IMPORTANT Header",
            "Context section with critical info",
            "Low priority section 1",
            "Low priority section 2",
            "Low priority section 3",
            "Final summary section",
        ]
        prompt = "\n\n".join(sections)

        target_tokens = (
            self.context_manager.estimate_token_count(
                sections[0] + "\n\n" + sections[1] + "\n\n" + sections[5]
            )
            + 5
        )

        trimmed = self.context_manager.trim_prompt(prompt, target_tokens)

        self.assertIn(sections[0], trimmed)
        self.assertIn(sections[1], trimmed)

        self.assertTrue(any(section not in trimmed for section in sections[2:5]))

    def test_trim_prompt_content_trimming(self):
        """Test content trimming when section trimming is insufficient."""

        long_prompt = "x" * 10000

        target_tokens = 500

        trimmed = self.context_manager.trim_prompt(long_prompt, target_tokens)

        self.assertLess(len(trimmed), len(long_prompt))

        estimated_tokens = self.context_manager.estimate_token_count(trimmed)
        self.assertLessEqual(estimated_tokens, target_tokens * 1.05)

    def test_split_into_sections(self):
        """Test splitting prompt into sections with priorities."""

        prompt = "# Important Header\n\nThis is a critical section.\n\nIMPORTANT note here.\n\nRegular content.\n\nMore regular content.\n\nFinal summary."

        sections = self.context_manager._split_into_sections(prompt)

        self.assertEqual(len(sections), 6)

        priorities = [priority for _, priority in sections]

        self.assertEqual(priorities[0], 1)

        self.assertLessEqual(priorities[1], 3)

        self.assertEqual(priorities[2], 1)


if __name__ == "__main__":
    unittest.main()
