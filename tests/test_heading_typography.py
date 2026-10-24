"""
Unit tests for ProductMind X Chatbot Heading Typography and Markdown Parsing.
Verifies:
1. Extraction of prominent main headings from standalone bold lines (**Title**) and markdown (# Title).
2. Removal of literal Markdown markers (**, #) from the visible title.
3. Prevention of code block (<pre><code>) rendering around headings.
4. Correct preservation of document hierarchy (H1 title, H2/H3 sections, body text, lists, and citations).
5. Robust handling of edge cases (long titles, no titles, gateway notices).
"""

import unittest
from ui.views import parse_chat_response


class TestHeadingTypography(unittest.TestCase):

    def test_screenshot_heading_structure(self):
        """Test with exact screenshot heading structure."""
        content = (
            "** 🥊 HEAD-TO-HEAD COMPARISON TABLE**\n\n"
            "| Metric / Feature | Sony WH-1000XM5 | Apple AirPods Pro (2nd Gen) | Advantage / Winner |\n"
            "| --- | --- | --- | --- |\n"
            "| Price | $398.00 [Metadata #B09XS7JWHH] | $249.00 [Metadata #B0BDHW4P5S] | AirPods Pro 2 |"
        )
        notice, title, body = parse_chat_response(content)

        # Main heading is cleanly extracted without asterisks
        self.assertNotIn("**", title)
        self.assertIn("Head-to-Head Comparison", title)
        # Body starts directly with the table
        self.assertTrue(body.startswith("| Metric / Feature"))
        # Preserves citations in body
        self.assertIn("[Metadata #B09XS7JWHH]", body)
        self.assertIn("[Metadata #B0BDHW4P5S]", body)

    def test_user_reported_aspect_heading(self):
        """Test with user reported heading: **Noise-Cancellation Performance – Evidence-Based Findings**."""
        content = (
            "**Noise-Cancellation Performance – Evidence-Based Findings**\n\n"
            "### [Observation]\n"
            "Three reviewers explicitly complain about ANC: \"crackling noise\" [Review #REV-B09XS7-U1001-0000].\n\n"
            "### [Calculation]\n"
            "70% negative sentiment on NC across 13 verified reviews."
        )
        notice, title, body = parse_chat_response(content)

        self.assertEqual(title, "Noise-Cancellation Performance – Evidence-Based Findings")
        self.assertNotIn("**", title)
        # Body preserves section headings
        self.assertIn("### [Observation]", body)
        self.assertIn("### [Calculation]", body)
        self.assertIn("[Review #REV-B09XS7-U1001-0000]", body)

    def test_markdown_h1_heading(self):
        """Test when model returns standard Markdown # Heading."""
        content = (
            "# Noise-Cancellation Performance – Evidence-Based Findings\n\n"
            "The Sony WH-1000XM5 features an 8-mic ANC system."
        )
        notice, title, body = parse_chat_response(content)

        self.assertEqual(title, "Noise-Cancellation Performance – Evidence-Based Findings")
        self.assertNotIn("#", title)
        self.assertTrue(body.startswith("The Sony WH-1000XM5"))

    def test_long_heading_wrapping(self):
        """Test long heading string wraps naturally without breaking parser."""
        long_title = "Comprehensive Head-to-Head Acoustic, Ergonomic, and Defect Comparison: Sony WH-1000XM5 vs Apple AirPods Pro 2"
        content = f"**{long_title}**\n\nDetailed breakdown below."
        notice, title, body = parse_chat_response(content)

        self.assertEqual(title, long_title)
        self.assertEqual(body, "Detailed breakdown below.")

    def test_response_with_no_explicit_title(self):
        """Test response with direct paragraph and lists (no explicit title)."""
        content = (
            "The Sony WH-1000XM5 provides 30 hours of continuous battery life [Metadata #B09XS7JWHH].\n\n"
            "- Quick-charge: 3 min -> 3 hours\n"
            "- USB-C PD compatible"
        )
        notice, title, body = parse_chat_response(content)

        self.assertEqual(title, "")
        self.assertEqual(notice, "")
        self.assertEqual(body, content)
        self.assertIn("[Metadata #B09XS7JWHH]", body)

    def test_gateway_notice_with_heading(self):
        """Test response beginning with blockquote rate-limit notice."""
        content = (
            "> ⚠️ **AI Gateway Rate-Limited:** Serving deterministic grounded analysis.\n\n"
            "**Noise-Cancellation Performance – Evidence-Based Findings**\n\n"
            "Analysis of 13 customer records."
        )
        notice, title, body = parse_chat_response(content)

        self.assertTrue(notice.startswith("> ⚠️"))
        self.assertEqual(title, "Noise-Cancellation Performance – Evidence-Based Findings")
        self.assertEqual(body, "Analysis of 13 customer records.")


if __name__ == "__main__":
    unittest.main()
