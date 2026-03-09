# Office Action Retrieval Training Extraction Spec

## Goal

Build high-quality retrieval supervision from `officeaction_102_103_20250105-20250330_cpc.jsonl`.

The office action is the label source.
The examined patent document is the query source.

Each training example should represent one rejected claim group or one claim-specific rationale unit.

## Core Principle

Do not use office-action prose as the final query text.

Use the office action to determine:

- which claims were rejected
- under `102` or `103`
- which cited patents are positive
- which cited passages were relied upon
- how strong each positive should be weighted

Use the examined patent document to build the query text:

- title
- explicit rejected claim text
- supporting description snippets from the same patent

## Input Sources

### Office-action JSONL

- `patentApplicationNumber`
- `filingDate`
- `patentCPCList`
- `ClaimRejections102`
- `ClaimRejections103`

Important nested fields:

- `text`
- `DedupApplicationCheck102/103`
- `OpenSearchApplicationMatches102/103`
- `SearchPublicationNumbers102/103`
- `OpenSearchPublicationMatches102/103`
- `DedupPublicationCheck102/103`
- `CitedPublicationNumbers102/103`

### Patent corpus

Use the local patent corpus parquet shards to resolve the examined patent by:

- `application_id`
- `doc_id`

Needed columns:

- `title`
- `abstract`
- `claims`
- `description`

## Extraction Unit

The preferred unit is:

- one claim-specific rationale block if the office action has `Regarding claim ...`

Fallback unit:

- one rejected claim group block such as `Claims 1-3 are rejected under ...`

Do not use one whole office-action row as one training example unless no finer structure is available.

## Pipeline

### 1. Parse office-action sections separately

Treat:

- `ClaimRejections102`
- `ClaimRejections103`

as separate supervision channels.

Store `statute = 102` or `103` on every derived example.

### 2. Split each section into rejection blocks

Use block boundaries like:

- `Claim 1 is rejected under 35 U.S.C. 102 ...`
- `Claims 1-3 are rejected under 35 U.S.C. 103 ...`

Each block should contain:

- claim group
- rejection header
- local rationale text

### 3. Parse rejected claim IDs

Convert claim expressions like:

- `1`
- `1-3`
- `1, 3, 5-7`
- `13-14, 19, 25, 28-29, 37, and 40`

into normalized claim ID lists.

### 4. Parse claim-specific rationale units

Within each block, split on patterns like:

- `Regarding claim 1,`
- `Regarding claims 7-9,`
- `As to claim 12,`

If present, these units are preferred over the whole claim group.

### 5. Resolve positive references

Build aligned reference records from the section arrays.

Use this fallback priority for final positive document IDs:

1. `DedupApplicationCheck*`
2. `OpenSearchApplicationMatches*`
3. publication-based fields resolved into application IDs:
   - `DedupPublicationCheck*`
   - `OpenSearchPublicationMatches*`
   - `SearchPublicationNumbers*`
   - `CitedPublicationNumbers*`

Drop unresolved positives from the final label set.

### 6. Assign positive roles

Infer roles from the rejection header:

- `over A` -> `primary`
- `in view of B` -> `supporting`
- `further in view of C` -> `supporting`

If the header cannot be parsed cleanly:

- first cited reference -> `primary`
- remaining references -> `supporting`

### 7. Build query text from the examined patent

Resolve the examined patent using `patentApplicationNumber`.

Extract:

- `title`
- explicit text of the rejected claims
- short description snippets selected from the patent description

Description snippets should be selected by lexical overlap with:

- the rejected claim text
- the local office-action rationale

### 8. Keep evidence metadata from the office action

Extract passage-like evidence references such as:

- `Abstract`
- `Figure 1`
- `Para 13`
- `[0071]`
- `Col. 11, lines 15-25`

Store these on each positive label.

### 9. Score example quality

Recommended tiers:

- `gold`
  - query claim text found
  - positive application IDs resolved
  - evidence refs present
- `silver`
  - query claim text found
  - positive application IDs resolved
- `bronze`
  - claim group parsed
  - only publication-level positives available
- `drop`
  - no useful claim/query text or no usable positives

Use `gold` and `silver` for main retrieval training.
Use `bronze` only if more volume is needed.

### 10. Weight positives

Recommended defaults:

- `102` primary: `1.0`
- `102` supporting: `0.7`
- `103` primary: `0.8`
- `103` supporting: `0.5`
- fallback-resolved positives: lower than direct application matches

## Output Schema

```json
{
  "query_id": "US17328125__103__claims_1__block_0__unit_0",
  "examined_app_id": "US17328125",
  "officeaction_line": 147,
  "statute": "103",
  "claim_ids": ["1"],
  "query_title": "IMPLANTABLE MEDICAL DEVICE WITH TEMPERATURE SENSOR",
  "query_claim_texts": [
    "1. An implantable medical device ..."
  ],
  "query_description_snippets": [
    "The specification explains ...",
    "In one embodiment ..."
  ],
  "query_text": "Title + explicit claim text + selected description snippets",
  "positives": [
    {
      "doc_id": "US13287751",
      "role": "primary",
      "weight": 0.8,
      "confidence": "gold",
      "source_field": "DedupApplicationCheck103",
      "raw_citation": "US 2012/0046708 Balczewski et al.",
      "publication_ids": ["US20120046708A1"],
      "evidence_text": "Regarding claim 1, Balczewski discloses ...",
      "evidence_refs": [
        "Abstract",
        "Figure 1",
        "Para 7",
        "Para 9",
        "Para 13"
      ]
    }
  ],
  "quality_tier": "gold",
  "cpc": [
    "A61B 5/686",
    "A61B 5/4836"
  ]
}
```

## Non-Goals

This extractor should not try to exactly reproduce the live Hugging Face dataset.

Its purpose is different:

- maximize training signal quality
- preserve claim-aware supervision
- preserve evidence metadata
- preserve multi-positive structure

## Validation Checks

Track these metrics after extraction:

- rows scanned
- sections with text
- claim blocks parsed
- claim-specific rationale units parsed
- examples with claim text found
- examples with resolved positive application IDs
- examples with evidence refs
- `102` vs `103` distribution
- quality tier distribution
- average positives per example

## Recommended First Training Slice

For the first high-quality run, keep only:

- `gold` and `silver`
- examples with explicit rejected claim text
- examples with at least one resolved application-ID positive
- examples with no unresolved query patent

This will trade volume for label quality and is the best first slice for training a retrieval model.
