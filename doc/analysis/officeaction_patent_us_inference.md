# Office Action JSONL Schema and Inferred Mapping to `Hyukkyu/patent-us`

This note documents two things:

1. The observed schema of `officeaction_102_103_20250105-20250330_cpc.jsonl`
2. The current inferred method by which its contents were likely transformed into the Hugging Face dataset `Hyukkyu/patent-us`

Important caveat:

- The final publishing step is verified from local code.
- The earlier conversion from the office-action JSONL into `usc102103_train.json` is not present in this repository.
- Therefore, the extraction pipeline below is partly inferred from the JSONL contents, the published dataset schema, and the local publish scripts.

## File Summary

- File: `officeaction_102_103_20250105-20250330_cpc.jsonl`
- Size: about `2.0 GB`
- Rows: `129,484`
- Format: JSON Lines, one JSON object per line
- Parse errors observed in full scan: `0`

## Top-Level Schema

Each row is a JSON object with the following observed keys:

```json
{
  "patentApplicationNumber": "string",
  "inventionTitle": "string",
  "filingDate": "string (ISO-8601 UTC datetime)",
  "ClaimRejections102": {
    "text": "string",
    "CitedPublicationNumbers102": ["string", "..."],
    "CitedApplicationNumbers102": ["string", "..."],
    "SearchPublicationNumbers102": ["string", "..."],
    "SearchApplicationNumbers102": ["string", "..."],
    "OpenSearchPublicationMatches102": ["string", "..."],
    "OpenSearchApplicationMatches102": ["string", "..."],
    "DedupPublicationCheck102": ["string", "..."],
    "DedupApplicationCheck102": ["string", "..."]
  },
  "ClaimRejections103": {
    "text": "string",
    "CitedPublicationNumbers103": ["string", "..."],
    "CitedApplicationNumbers103": ["string", "..."],
    "SearchPublicationNumbers103": ["string", "..."],
    "SearchApplicationNumbers103": ["string", "..."],
    "OpenSearchPublicationMatches103": ["string", "..."],
    "OpenSearchApplicationMatches103": ["string", "..."],
    "DedupPublicationCheck103": ["string", "..."],
    "DedupApplicationCheck103": ["string", "..."]
  },
  "patentPublicationNumber": "string",
  "patentAbstract": "string",
  "patentCPCList": ["string", "..."]
}
```

## Example Data

The examples below are real sampled rows from the JSONL, with long text fields truncated for readability.

### Example 1: Row with populated `ClaimRejections103`

```json
{
  "patentApplicationNumber": "US15257351",
  "inventionTitle": "Apparatus and method for maintaining pet waste",
  "filingDate": "2016-09-06T00:00:00.000Z",
  "ClaimRejections102": {
    "text": "",
    "CitedPublicationNumbers102": [],
    "CitedApplicationNumbers102": [],
    "SearchPublicationNumbers102": [],
    "SearchApplicationNumbers102": [],
    "OpenSearchPublicationMatches102": [],
    "OpenSearchApplicationMatches102": [],
    "DedupPublicationCheck102": [],
    "DedupApplicationCheck102": []
  },
  "ClaimRejections103": {
    "text": "Claim Rejections - 35 USC \u00a7 103\\nIn the event the determination of the status of the application ... Claims 1,2,5,6,8,20,23 are rejected under pre-AIA 35 U.S.C. 103(a) as being unpatentable over Pierson ...",
    "CitedPublicationNumbers103": [
      "US 20050183401 A1",
      "US 5178099 A",
      "US 5551375 A",
      "US 5823137 A",
      "US 5572950 A",
      "US 5911194 A"
    ],
    "CitedApplicationNumbers103": [],
    "SearchPublicationNumbers103": [
      "US20050183401A1",
      "US05178099",
      "US05551375",
      "US05823137",
      "US05572950",
      "US05911194"
    ],
    "SearchApplicationNumbers103": [],
    "OpenSearchPublicationMatches103": [],
    "OpenSearchApplicationMatches103": [],
    "DedupPublicationCheck103": [],
    "DedupApplicationCheck103": []
  },
  "patentPublicationNumber": "",
  "patentAbstract": "",
  "patentCPCList": [
    "A01K 1/0114",
    "A01K 1/011",
    "A01K 1/0151"
  ]
}
```

What this example shows:

- `ClaimRejections103.text` can contain a long multi-paragraph office-action narrative.
- `CitedPublicationNumbers103` contains raw extracted prior-art references.
- `SearchPublicationNumbers103` contains normalized search-friendly publication IDs.
- `patentPublicationNumber` and `patentAbstract` may be empty even when rejection text is populated.

### Example 2: Row with populated `ClaimRejections102`

```json
{
  "patentApplicationNumber": "US16206952",
  "inventionTitle": "MULTIPURPOSE CONTAINER SYSTEM",
  "filingDate": "2018-11-30T00:00:00.000Z",
  "ClaimRejections102": {
    "text": "Claim Rejections - 35 USC \u00a7 102\\nThe following is a quotation of the appropriate paragraphs of 35 U.S.C. 102 ... Claim(s) 16, 17, 24, and 25 is/are rejected under 35 U.S.C. 102(a)(1) as being anticipated by US Patent No. 4,763,763 ...",
    "CitedPublicationNumbers102": [
      "US Patent No. 4,763,763"
    ],
    "CitedApplicationNumbers102": [],
    "SearchPublicationNumbers102": [
      "US04763763"
    ],
    "SearchApplicationNumbers102": [],
    "OpenSearchPublicationMatches102": [
      "US04763763A"
    ],
    "OpenSearchApplicationMatches102": [
      "US07118728"
    ],
    "DedupPublicationCheck102": [
      "US04763763A"
    ],
    "DedupApplicationCheck102": [
      "US07118728"
    ]
  },
  "ClaimRejections103": {
    "text": "",
    "CitedPublicationNumbers103": [],
    "CitedApplicationNumbers103": [],
    "SearchPublicationNumbers103": [],
    "SearchApplicationNumbers103": [],
    "OpenSearchPublicationMatches103": [],
    "OpenSearchApplicationMatches103": [],
    "DedupPublicationCheck103": [],
    "DedupApplicationCheck103": []
  },
  "patentPublicationNumber": "US20230257163A1",
  "patentAbstract": "A container system includes a container having a top and a bottom, with an attachment collar secured to the container ...",
  "patentCPCList": [
    "B65D 25/20",
    "B65D 1/12",
    "B65D 1/34",
    "B65D 25/28",
    "B65D 43/0202"
  ]
}
```

What this example shows:

- `ClaimRejections102` can be populated while `ClaimRejections103` is empty.
- The publication/application match arrays can already contain apparently resolved identifiers.
- Rows with a non-empty `patentPublicationNumber` typically also have a non-empty `patentAbstract`.

### Example 3: Row with empty CPC list

```json
{
  "patentApplicationNumber": "US13164732",
  "inventionTitle": "SYSTEMS OF COMPUTERIZED AGENTS AND USER-DIRECTED SEMANTIC NETWORKING",
  "patentPublicationNumber": "US20110314382A1",
  "patentAbstract": "non-empty",
  "patentCPCList": []
}
```

What this example shows:

- `patentCPCList` may be present but empty.
- Empty CPC does not imply missing publication metadata.

### Example 4: Row with missing `patentCPCList`

```json
{
  "patentApplicationNumber": "US15802908",
  "inventionTitle": "Method and Apparatus for Transporting Video Signal Over Wireless Interface",
  "filingDate": "2017-11-03T00:00:00.000Z",
  "ClaimRejections102": { "...": "present" },
  "ClaimRejections103": { "...": "present" },
  "patentPublicationNumber": "",
  "patentAbstract": ""
}
```

What this example shows:

- `patentCPCList` is sometimes omitted entirely.
- In the observed file, rows missing `patentCPCList` also had empty `patentPublicationNumber` and `patentAbstract`.

## Field Behavior

### Always-present top-level fields

- `patentApplicationNumber`
- `inventionTitle`
- `filingDate`
- `ClaimRejections102`
- `ClaimRejections103`
- `patentPublicationNumber`
- `patentAbstract`

### Optional top-level field

- `patentCPCList`
  - Present in `127,515` rows
  - Missing in `1,969` rows
  - Present but empty in `40` rows

### Text population

- `ClaimRejections102.text`
  - Non-empty in `50,784` rows
  - Empty in `78,700` rows

- `ClaimRejections103.text`
  - Non-empty in `113,055` rows
  - Empty in `16,429` rows

- `patentPublicationNumber`
  - Non-empty in `89,705` rows
  - Empty in `39,779` rows

- `patentAbstract`
  - Non-empty in `89,705` rows
  - Empty in `39,779` rows

Observed pattern:

- `patentPublicationNumber` and `patentAbstract` move together in this file.
- Rows with missing `patentCPCList` also have empty `patentPublicationNumber` and empty `patentAbstract`.

### `patentCPCList`

- Element type: always string when present
- Average length: about `4.80`
- Maximum observed length: `115`

### Uniqueness

- `patentApplicationNumber` is not unique per row
- Total rows: `129,484`
- Unique `patentApplicationNumber`: `128,700`

This suggests the file is at least partly office-action-level, not purely application-level.

## Meaning of the Citation-related Fields

Based on names and sampled values, the nested arrays likely represent a citation normalization pipeline:

- `CitedPublicationNumbers102/103`
  - Raw publication citations extracted from office-action text
  - Example: `"US 20050183401 A1"`

- `SearchPublicationNumbers102/103`
  - Normalized publication search keys
  - Example: `"US20050183401A1"`

- `OpenSearchPublicationMatches102/103`
  - Publication identifiers matched via search/enrichment

- `OpenSearchApplicationMatches102/103`
  - Application identifiers matched via search/enrichment

- `DedupPublicationCheck102/103`
  - Deduplicated publication matches

- `DedupApplicationCheck102/103`
  - Deduplicated application matches

In this file:

- `CitedApplicationNumbers102/103` are always empty
- `SearchApplicationNumbers102/103` are always empty

## Verified Local Code About `Hyukkyu/patent-us`

The local publishing script shows that the Hugging Face `train` split is created from `usc102103_train.json`, not directly from the JSONL:

- `question_id = "q_" + sha1(row["question"])`
- `label_id = ["d_" + sha1(ctx["text"]) for ctx in row["positive_ctxs"]]`

See:

- `script/preprocess/patent/publish_patent_id_datasets.py`

That means the intermediate training file almost certainly has rows shaped roughly like:

```json
{
  "question": "some training query text",
  "positive_ctxs": [
    {"text": "some positive patent document text"},
    {"text": "another positive patent document text"}
  ]
}
```

Everything else in `usc102103_train.json` would be ignored by the publisher for the HF `train` split.

## Inferred Mapping From the Office-action JSONL

The most likely fields used from `officeaction_102_103_20250105-20250330_cpc.jsonl` are:

### Likely source of `question`

- `ClaimRejections102.text`
- `ClaimRejections103.text`

Reason:

- These are the only large free-text rejection narratives in the file.
- The HF train publisher only needs a question string.
- The HF train split has `298,114` rows, which is much larger than the `129,484` JSONL rows, so the source was likely segmented into multiple training examples per office-action row.

The strongest inference is:

- Each non-empty `ClaimRejections102.text` and `ClaimRejections103.text` was split into smaller rejection-level snippets.
- Each snippet became one intermediate `question`.

Examples of plausible segmentation units:

- A rejection block beginning with text like `Claims 1-3 are rejected under 35 U.S.C. 103 ...`
- A claim-group-specific rationale block within the longer office-action text

This is consistent with the observed scale:

- JSONL rows: `129,484`
- Observed rejection-like spans in text via a simple scan: about `337,971`
- HF train rows: `298,114`

That relationship strongly suggests rejection-level segmentation rather than one-row-per-office-action conversion.

### Likely source of positive patent references

Most likely primary fields:

- `ClaimRejections102.DedupPublicationCheck102`
- `ClaimRejections103.DedupPublicationCheck103`

Likely fallback fields if dedup arrays were empty:

- `OpenSearchPublicationMatches102/103`
- `SearchPublicationNumbers102/103`
- `CitedPublicationNumbers102/103`

Reason:

- The names imply a progression from raw extracted citation -> normalized search ID -> matched publication ID -> deduplicated final ID.
- The final publisher hashes `positive_ctxs[*].text`, so there must have been a step that resolved these publication IDs into actual patent document text.

### Likely source of patent document text for positives

The positive document text likely did not come from the office-action JSONL itself.

Instead, the likely process was:

1. Extract rejection text from `ClaimRejections102.text` / `ClaimRejections103.text`
2. Extract cited or matched publication IDs from the nested citation arrays
3. Retrieve the corresponding patent document text from a patent corpus or OpenSearch-backed patent store
4. Store that retrieved document text into `positive_ctxs[*].text`
5. Publish only hashed IDs to `Hyukkyu/patent-us`

This is supported by the local patent corpus export scripts, which build a separate US patent text corpus with:

- `doc_id`
- `title`
- `abstract`
- `claims`
- `description`
- `application_id`

## Current Inferred End-to-end Processing Outline

The current best reconstruction of the missing extraction script is:

1. Read each JSONL row from `officeaction_102_103_20250105-20250330_cpc.jsonl`
2. For each row, inspect:
   - `ClaimRejections102.text`
   - `ClaimRejections103.text`
3. Skip empty rejection texts
4. Split each non-empty rejection text into smaller rejection-level training units
5. For each training unit, collect the related cited prior-art identifiers from:
   - preferably `DedupPublicationCheck102/103`
   - otherwise `OpenSearchPublicationMatches102/103`
   - otherwise normalized or raw citation fields
6. Resolve those publication identifiers to actual patent documents in an external patent corpus / OpenSearch index
7. Build an intermediate row:

```json
{
  "question": "rejection snippet text",
  "positive_ctxs": [
    {"text": "resolved patent document text"},
    {"text": "resolved patent document text"}
  ]
}
```

8. Write these rows into `usc102103_train.json`
9. Run the local publisher:
   - hash `question` into `question_id`
   - hash each `positive_ctxs[*].text` into `label_id`
   - dedupe duplicate label texts per row
10. Push the resulting ID-only dataset to `Hyukkyu/patent-us`

## Confidence by Field

### High confidence

- `ClaimRejections102.text` is used
- `ClaimRejections103.text` is used
- Some publication-based citation field is used to choose positives

### Medium confidence

- `DedupPublicationCheck102/103` is the primary positive-source field
- Rejection text is split into multiple training examples per office-action row

### Low confidence

- Exact segmentation rules for creating one `question` from longer rejection text
- Exact fallback order among `DedupPublicationCheck*`, `OpenSearchPublicationMatches*`, `SearchPublicationNumbers*`, and `CitedPublicationNumbers*`
- Whether application-based match fields were used as backup during retrieval

## Fields Likely Not Used as Primary Inputs to the HF Train Dataset

- `inventionTitle`
- `filingDate`
- `patentPublicationNumber`
- `patentAbstract`
- `patentCPCList`
- `CitedApplicationNumbers102/103`
- `SearchApplicationNumbers102/103`

These may have been useful for filtering or metadata, but they do not look like the main source for training `question` or `positive_ctxs`.

## Bottom Line

The strongest current inference is:

- `question` in the missing `usc102103_train.json` was derived from segmented content of:
  - `ClaimRejections102.text`
  - `ClaimRejections103.text`

- `positive_ctxs[*].text` was derived by resolving cited prior-art publications, most likely starting from:
  - `DedupPublicationCheck102`
  - `DedupPublicationCheck103`

The office-action JSONL is therefore best understood as the supervision source, while the actual positive patent texts were likely fetched from a separate patent corpus.
