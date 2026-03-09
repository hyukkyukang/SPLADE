# Office Action JSONL Schema and Current Inferred Mapping to `Hyukkyu/patent-us`

This note documents two things:

1. The observed schema of `officeaction_102_103_20250105-20250330_cpc.jsonl`
2. The current inferred method by which its contents were transformed into the live Hugging Face dataset `Hyukkyu/patent-us`

Important caveats:

- As of `2026-03-09`, the live Hugging Face dataset uses raw application-style IDs, not hashed text IDs.
- A local publishing script in this repository still hashes `question` and `positive_ctxs[*].text`, so it appears to describe an older dataset revision or a different publishing path.
- The exact upstream extraction script is still not present in this repository.
- Therefore, the downstream mapping below is based on the JSONL contents, the live Hugging Face dataset, the local notebook `make_reject_paragraph_by_industry.ipynb`, and direct overlap checks.

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
- `DedupApplicationCheck103` can still be empty even when rejection text and normalized publication references are present.
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
- The application-match arrays can already contain apparently resolved identifiers.
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

## Live Hugging Face Dataset State (`2026-03-09`)

The live dataset `Hyukkyu/patent-us` no longer uses hashed IDs.

Observed live metadata:

- Features:
  - `question_id: string`
  - `label_id: list[string]`
- Splits:
  - `train = 68,734`
  - `test = 22,638`
- `question_id` values are raw application-style IDs such as `US15257351`
- `label_id` values are lists of raw application-style IDs such as `["US05551375", "US05823137", ...]`

Implication:

- The current live dataset is application-to-application linkage data.
- It is not an ID-only text-hash dataset in its current form.

## What the Notebook Confirms

The notebook `make_reject_paragraph_by_industry.ipynb` parses the office-action JSONL and builds records with this logic:

```python
if DedupApplicationCheck102:
    question_id = patentApplicationNumber
    label_id = DedupApplicationCheck102
    reject = ClaimRejections102.text
elif DedupApplicationCheck103:
    question_id = patentApplicationNumber
    label_id = DedupApplicationCheck103
    reject = ClaimRejections103.text
else:
    skip
```

This is strong evidence that the relevant office-action fields for the live Hugging Face dataset are:

- `patentApplicationNumber`
- `ClaimRejections102.DedupApplicationCheck102`
- `ClaimRejections103.DedupApplicationCheck103`

The notebook also shows that:

- `patentCPCList` is used only for industry filtering
- `ClaimRejections102.text` / `ClaimRejections103.text` are carried as supporting text
- the downstream dataset structure is based on application IDs, not text hashing

## Notebook Limitations

The notebook is informative, but it is not the full production extractor.

Known limitations:

- It prefers `102` over `103` and drops `103` when both are present.
- It does not union labels across both rejection sections.
- It does not fall back beyond `DedupApplicationCheck102/103`.
- It does not split rejection text into paragraphs despite the notebook name.
- It filters by only the first CPC in `patentCPCList`.
- Its CPC wildcard handling is weak:
  - it strips `*` and then uses exact `isin(...)`
  - entries like `A61K*` become `A61K`, which do not exactly match derived values like `A61K36/`

## Direct Comparison to the Live Hugging Face Dataset

### Strict notebook-style extraction across the whole office-action file

If the notebook logic is applied to the whole JSONL without industry filtering:

- Candidate rows: `65,167`
- Candidate unique `question_id`: `64,734`

Compared with live HF `train`:

- HF `train` rows: `68,734`
- Shared `question_id` values: `64,456`
- Exact row overlap under strict notebook logic: `50,528`

This is already strong evidence that the office-action file is the source.

### Unioning `102` and `103` labels makes the match much stronger

If `label_id` is built as the deduplicated union of:

- `DedupApplicationCheck102`
- `DedupApplicationCheck103`

then the overlap improves materially.

Against live HF `train`:

- Exact order-sensitive matches: `62,651`
- Set-equality matches ignoring label order: `62,745`

Against live HF `test`:

- All `22,638 / 22,638` rows match by label set
- The remaining difference is label ordering only

This makes the likely production rule much clearer:

- `question_id = patentApplicationNumber`
- `label_id = union(DedupApplicationCheck102, DedupApplicationCheck103)`

### Additional fallback fields are still needed for part of `train`

Not all live HF `train` rows are covered by the dedup-application arrays alone.

Useful observations:

- All live HF `train` `question_id` values exist in the office-action file
- Some unmatched rows have empty `DedupApplicationCheck102/103` but non-empty:
  - `OpenSearchApplicationMatches102/103`
  - `SearchPublicationNumbers102/103`
  - rejection text in `ClaimRejections102.text` or `ClaimRejections103.text`

Using a simple extra fallback:

- if both dedup-application arrays are empty, use `OpenSearchApplicationMatches102/103`

improves `train` overlap to:

- Exact order-sensitive matches: `62,893`
- Set-equality matches ignoring label order: `62,987`

This still leaves a remaining gap, which most likely comes from rows where:

- application-level match arrays are empty
- publication-level identifiers exist
- those publication identifiers were later resolved to application IDs in the real pipeline

## Current Best Inferred Extraction Logic

The current best reconstruction of the live dataset creation logic is:

1. Read each JSONL row from `officeaction_102_103_20250105-20250330_cpc.jsonl`
2. Set:
   - `question_id = patentApplicationNumber`
3. Build candidate labels from both rejection sections, not just one:
   - `DedupApplicationCheck102`
   - `DedupApplicationCheck103`
4. Deduplicate and union those application IDs into `label_id`
5. If both dedup-application arrays are empty, fall back to:
   - `OpenSearchApplicationMatches102/103`
6. If application-match arrays are still empty but publication identifiers exist, resolve publication IDs to application IDs using:
   - `SearchPublicationNumbers102/103`
   - possibly `OpenSearchPublicationMatches102/103`
   - possibly `DedupPublicationCheck102/103`
7. Write the resulting rows as:

```json
{
  "question_id": "US15257351",
  "label_id": [
    "US05551375",
    "US05823137",
    "US05572950",
    "US05911194"
  ]
}
```

This is now the simplest explanation that fits:

- the live Hugging Face dataset format
- the notebook logic
- and the observed overlap statistics

## Confidence by Field

### High confidence

- `question_id` comes from `patentApplicationNumber`
- `label_id` is application-based, not publication-based in the final dataset
- `DedupApplicationCheck102/103` is part of the real extraction path
- `102` and `103` are combined more symmetrically than the notebook does

### Medium confidence

- `OpenSearchApplicationMatches102/103` is a real fallback source
- publication-number fields are used only when application-match fields are missing

### Low confidence

- Exact fallback priority among:
  - `OpenSearchApplicationMatches*`
  - `SearchPublicationNumbers*`
  - `OpenSearchPublicationMatches*`
  - `DedupPublicationCheck*`
- Exact label ordering rule used in the final dataset
- How duplicated `patentApplicationNumber` rows in the JSONL are merged when the same application appears more than once

## Fields Likely Not Used as Primary Inputs to the Live HF ID Mapping

- `inventionTitle`
- `filingDate`
- `patentPublicationNumber`
- `patentAbstract`
- `patentCPCList`

These may still be useful for filtering, enrichment, or analysis, but they do not look like the primary source of `question_id` or `label_id`.

## Bottom Line

The clearest current interpretation is:

- The live `Hyukkyu/patent-us` dataset is built from the office-action JSONL using raw application IDs.
- `question_id` is the examined application's `patentApplicationNumber`.
- `label_id` is a deduplicated application-ID list derived primarily from:
  - `DedupApplicationCheck102`
  - `DedupApplicationCheck103`
- The real extractor likely unions `102` and `103`, then falls back to other application/publication match fields when the dedup arrays are empty.

The notebook `make_reject_paragraph_by_industry.ipynb` is therefore not the full production extractor, but it reveals the correct core field mapping and makes the current live dataset much easier to explain.
