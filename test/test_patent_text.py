import unittest

from src.data.patent_text import (
    PATENT_DOCUMENT_TEMPLATE_NAME,
    format_named_text_template,
    format_patent_document_text,
    format_patent_document_text_prefix,
)


class PatentTextFormattingTest(unittest.TestCase):
    def test_format_patent_document_text_includes_description(self) -> None:
        text = format_patent_document_text(
            {
                "title": "Valve Assembly",
                "abstract": "A valve assembly with a hinged seal.",
                "claims": "1. A valve assembly...",
                "description": "Detailed description.",
            }
        )

        self.assertEqual(
            text,
            "Title: Valve Assembly\n"
            "Abstract: A valve assembly with a hinged seal.\n"
            "Claims: 1. A valve assembly...\n"
            "Description: Detailed description.",
        )

    def test_format_patent_document_text_skips_empty_fields(self) -> None:
        text = format_patent_document_text(
            {
                "title": "Valve Assembly",
                "abstract": "",
                "claims": None,
                "description": "Detailed description.",
            }
        )

        self.assertEqual(
            text,
            "Title: Valve Assembly\nDescription: Detailed description.",
        )

    def test_named_template_dispatches_patent_template(self) -> None:
        text = format_named_text_template(
            PATENT_DOCUMENT_TEMPLATE_NAME,
            {
                "title": "Valve Assembly",
                "abstract": "A valve assembly with a hinged seal.",
                "claims": "1. A valve assembly...",
                "description": "Detailed description.",
            },
        )

        self.assertIn("Description: Detailed description.", text)

    def test_patent_document_prefix_truncates_without_full_description(self) -> None:
        prefix = format_patent_document_text_prefix(
            {
                "title": "Valve Assembly",
                "abstract": "A valve assembly with a hinged seal.",
                "claims": "1. A valve assembly...",
                "description": "Detailed description " * 50,
            },
            char_budget=96,
        )

        self.assertTrue(prefix.truncated)
        self.assertIn("Title: Valve Assembly", prefix.text)
        self.assertIn("Abstract:", prefix.text)
        self.assertNotIn("Detailed description Detailed description Detailed", prefix.text)


if __name__ == "__main__":
    unittest.main()
