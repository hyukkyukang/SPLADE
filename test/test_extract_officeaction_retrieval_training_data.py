import json
import unittest
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from script.preprocess.patent.extract_officeaction_retrieval_training_data import (
    build_patent_record_lookup,
    expand_claim_expression,
    extract_evidence_refs,
    iter_candidate_units,
    materialize_examples,
    parse_claims_text,
    select_description_snippets,
)


class ExtractOfficeactionRetrievalTrainingDataTest(unittest.TestCase):
    def test_expand_claim_expression_handles_ranges(self) -> None:
        self.assertEqual(
            expand_claim_expression("13-14, 19, 25, 28-29, 37, and 40"),
            ["13", "14", "19", "25", "28", "29", "37", "40"],
        )

    def test_parse_claims_text_extracts_numbered_claims(self) -> None:
        claims_text = (
            "1. An implantable medical device comprising a processor and a sensor.\n"
            "2. The implantable medical device of claim 1, wherein the sensor is a temperature sensor.\n"
            "3. The implantable medical device of claim 1, wherein the processor stores measurements.\n"
        )

        claim_map = parse_claims_text(claims_text)

        self.assertEqual(len(claim_map), 3)
        self.assertTrue(claim_map["1"].startswith("1. An implantable medical device"))
        self.assertIn("claim 1", claim_map["2"])

    def test_parse_claims_text_falls_back_to_sequential_claims(self) -> None:
        claims_text = (
            "An apparatus comprising a processor and a connector.\n"
            " . The apparatus of claim 1, wherein the connector is an SPE connector.\n"
            " . The apparatus of claim 1, wherein the processor converts Ethernet data.\n"
        )

        claim_map = parse_claims_text(claims_text)

        self.assertEqual(len(claim_map), 3)
        self.assertTrue(claim_map["1"].startswith("1. An apparatus comprising"))
        self.assertTrue(claim_map["2"].startswith("2. The apparatus of claim 1"))

    def test_extract_evidence_refs_collects_common_patterns(self) -> None:
        text = (
            "Balczewski discloses the device (Abstract and Figure 1; Para 13). "
            "Calfee teaches weighted statistics (Col. 11, lines 15-25)."
        )

        refs = extract_evidence_refs(text)

        self.assertEqual(
            refs,
            ["Abstract", "Figure 1", "Para 13", "Col. 11, lines 15-25"],
        )

    def test_select_description_snippets_chunks_long_single_block(self) -> None:
        long_description = " ".join(
            [
                "The system assigns resources to tasks based on skills and availability."
                for _ in range(80)
            ]
        )

        snippets = select_description_snippets(
            abstract="",
            description=long_description,
            claim_texts=["1. A system that assigns resources to tasks based on availability."],
            rationale_text="assigns resources to tasks based on availability",
            max_snippets=2,
        )

        self.assertEqual(len(snippets), 2)
        self.assertTrue(all(len(snippet) <= 900 for snippet in snippets))
        self.assertTrue(all("assigns resources to tasks" in snippet for snippet in snippets))

    def test_end_to_end_materializes_claim_aware_example(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            officeaction_path = tmp_path / "officeaction.jsonl"
            corpus_path = tmp_path / "patent.parquet"

            officeaction_row = {
                "patentApplicationNumber": "US17328125",
                "inventionTitle": "IMPLANTABLE MEDICAL DEVICE WITH TEMPERATURE SENSOR",
                "filingDate": "2021-05-24T00:00:00.000Z",
                "ClaimRejections102": {
                    "text": "",
                    "CitedPublicationNumbers102": [],
                    "CitedApplicationNumbers102": [],
                    "SearchPublicationNumbers102": [],
                    "SearchApplicationNumbers102": [],
                    "OpenSearchPublicationMatches102": [],
                    "OpenSearchApplicationMatches102": [],
                    "DedupPublicationCheck102": [],
                    "DedupApplicationCheck102": [],
                },
                "ClaimRejections103": {
                    "text": (
                        "Claims 1-2 are rejected under 35 U.S.C. 103 as being unpatentable over "
                        'US 2012/0046708 "Balczewski et al." hereinafter "Balczewski" in view of '
                        'US 4,803,987 Calfee et al., hereinafter "Calfee".\n'
                        "Regarding claim 1, Balczewski discloses an implantable medical device "
                        "with a processor and a temperature sensor (Abstract; Figure 1; Para 13).\n"
                        "Balczewski does not disclose a weighted statistic.\n"
                        "However, Calfee teaches a weighted statistic (Col. 11, lines 15-25).\n"
                    ),
                    "CitedPublicationNumbers103": [
                        "US 2012/0046708",
                        "US 4,803,987",
                    ],
                    "CitedApplicationNumbers103": [],
                    "SearchPublicationNumbers103": [
                        "US20120046708",
                        "US04803987",
                    ],
                    "SearchApplicationNumbers103": [],
                    "OpenSearchPublicationMatches103": [
                        "US20120046708A1",
                        "US04803987A",
                    ],
                    "OpenSearchApplicationMatches103": [
                        "US13287751",
                        "US06872824",
                    ],
                    "DedupPublicationCheck103": [
                        "US20120046708A1",
                        "US04803987A",
                    ],
                    "DedupApplicationCheck103": [
                        "US13287751",
                        "US06872824",
                    ],
                },
                "patentPublicationNumber": "US20210369202A1",
                "patentAbstract": "Implantable device that senses body temperature and performs statistical analysis.",
                "patentCPCList": ["A61B 5/686"],
            }
            officeaction_path.write_text(
                json.dumps(officeaction_row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            patent_table = pa.Table.from_pylist(
                [
                    {
                        "doc_id": "US17328125",
                        "application_id": "US17328125",
                        "title": "IMPLANTABLE MEDICAL DEVICE WITH TEMPERATURE SENSOR",
                        "abstract": "An implantable device senses body temperature and performs statistical analysis.",
                        "claims": (
                            "1. An implantable medical device comprising a processor, a memory, "
                            "and a temperature sensor configured to sense body temperature and "
                            "calculate a reference value.\n"
                            "2. The implantable medical device of claim 1, wherein the processor "
                            "calculates a weighted statistic.\n"
                        ),
                        "description": (
                            "[0001] The implantable medical device calibrates an implanted "
                            "temperature sensor without an external programmer.\n"
                            "[0002] The processor stores repeated temperature measurements and "
                            "computes a reference value and variability statistics.\n"
                        ),
                    }
                ]
            )
            pq.write_table(patent_table, corpus_path)

            candidate_units = list(iter_candidate_units(officeaction_path))
            patent_lookup = build_patent_record_lookup(
                glob(str(corpus_path)),
                {"US17328125"},
            )
            examples, stats, tier_counter = materialize_examples(
                candidate_units,
                patent_lookup,
                max_description_snippets=2,
                require_claim_text=True,
                min_quality_tier="silver",
            )

            self.assertEqual(len(examples), 1)
            example = examples[0]
            self.assertEqual(example["examined_app_id"], "US17328125")
            self.assertEqual(example["claim_ids"], ["1"])
            self.assertIn("IMPLANTABLE MEDICAL DEVICE WITH TEMPERATURE SENSOR", example["query_text"])
            self.assertIn("1. An implantable medical device", example["query_text"])
            self.assertEqual(len(example["positives"]), 2)
            self.assertEqual(example["positives"][0]["doc_id"], "US13287751")
            self.assertEqual(example["positives"][0]["role"], "primary")
            self.assertEqual(example["positives"][1]["doc_id"], "US06872824")
            self.assertEqual(example["positives"][1]["role"], "supporting")
            self.assertEqual(example["quality_tier"], "gold")
            self.assertEqual(tier_counter["gold"], 1)
            self.assertEqual(stats.examples_written, 1)


if __name__ == "__main__":
    unittest.main()
