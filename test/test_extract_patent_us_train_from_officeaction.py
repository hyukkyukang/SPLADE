import unittest

from script.preprocess.patent.extract_patent_us_train_from_officeaction import (
    compose_doc_text,
    project_question_text,
    select_segment_positive_ids,
    select_positive_ids,
    split_rejection_text,
)


class ExtractPatentUsTrainFromOfficeactionTest(unittest.TestCase):
    def test_split_rejection_text_claim_blocks(self) -> None:
        text = (
            "Claim Rejections - 35 USC § 103\n"
            "Preamble text.\n"
            "Claims 1-3 are rejected under 35 U.S.C. 103 as being unpatentable over A.\n"
            "Analysis for claims 1-3.\n"
            "Claim 4 is rejected under 35 U.S.C. 103 as being unpatentable over B.\n"
            "Analysis for claim 4.\n"
        )

        segments = split_rejection_text(text, mode="claims_blocks")

        self.assertEqual(len(segments), 2)
        self.assertTrue(segments[0].startswith("Claims 1-3 are rejected under"))
        self.assertTrue(segments[1].startswith("Claim 4 is rejected under"))

    def test_split_rejection_text_falls_back_to_full_text(self) -> None:
        text = "Claim Rejections - 35 USC § 103\nNo explicit claim header."

        segments = split_rejection_text(text, mode="claims_blocks_or_full_text")

        self.assertEqual(segments, [text])

    def test_select_positive_ids_prefers_application_matches(self) -> None:
        section = {
            "DedupApplicationCheck103": ["US14685948", "US13933733"],
            "DedupPublicationCheck103": ["US20160307145A1", "US20140039954A1"],
        }

        positive_ids, field_name = select_positive_ids(
            section,
            section_suffix="103",
            positive_id_mode="first_nonempty",
            prefer_application_matches=True,
        )

        self.assertEqual(positive_ids, ["US14685948", "US13933733"])
        self.assertEqual(field_name, "DedupApplicationCheck103")

    def test_select_positive_ids_can_merge_multiple_nonempty_fields(self) -> None:
        section = {
            "DedupApplicationCheck102": ["US07118728"],
            "OpenSearchApplicationMatches102": ["US07118728", "US01234567"],
        }

        positive_ids, field_name = select_positive_ids(
            section,
            section_suffix="102",
            positive_id_mode="merge_nonempty",
            prefer_application_matches=True,
        )

        self.assertEqual(positive_ids, ["US07118728", "US01234567"])
        self.assertEqual(field_name, "DedupApplicationCheck102")

    def test_project_question_text_first_line(self) -> None:
        text = "Claims 1-3 are rejected under 35 U.S.C. 103.\nDetails follow."

        projected = project_question_text(text, mode="first_line")

        self.assertEqual(projected, "Claims 1-3 are rejected under 35 U.S.C. 103.")

    def test_select_segment_positive_ids_filters_by_segment_markers(self) -> None:
        section = {
            "CitedPublicationNumbers103": ["US 1234567 A", "US 7654321 A"],
            "SearchPublicationNumbers103": ["US01234567", "US07654321"],
            "DedupApplicationCheck103": ["US10000001", "US10000002"],
        }
        segment = "Claim 1 is rejected under 35 U.S.C. 103 over US 1234567 A."

        positive_ids, field_name = select_segment_positive_ids(
            section,
            segment,
            section_suffix="103",
            positive_id_mode="first_nonempty",
            prefer_application_matches=True,
            selection_scope="segment_filtered_or_section",
        )

        self.assertEqual(positive_ids, ["US10000001"])
        self.assertEqual(field_name, "DedupApplicationCheck103")

    def test_compose_doc_text_joins_requested_columns(self) -> None:
        row = {
            "title": "Title",
            "abstract": "Abstract line 1\nAbstract line 2",
            "claims": "Claims",
            "description": "Description",
        }

        text = compose_doc_text(
            row,
            columns=["title", "abstract", "claims"],
            normalize=True,
        )

        self.assertEqual(text, "Title Abstract line 1 Abstract line 2 Claims")


if __name__ == "__main__":
    unittest.main()
