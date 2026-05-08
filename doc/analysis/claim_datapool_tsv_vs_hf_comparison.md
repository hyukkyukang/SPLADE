# Claim Datapool TSV vs HF Passage Generation

## Goal

Check whether our local passage generation from the Hugging Face corpus matches the original GCP TSV generation logic used by the DPR pipeline.

Compared assets:

- Original TSV:
  - `/home/user/SPLADE/data/datapool_US_claims_0917.tsv`
- Original TSV-generation script from GCP:
  - `/home/user/SPLADE/.cache/gcs/patent_data/scripts/biencoder/datapool/get_all_claim_datapool.py`
- Local passage builder:
  - `/home/user/SPLADE/src/data/patent_passages.py`
- Local generated passage corpus:
  - `/home/user/SPLADE/data/corpus/patent_us_claim_passages_small/passages.parquet`
- Local HF corpus:
  - `/home/user/SPLADE/.cache/hf/patent-us-corpus-small/data/*.parquet`

## Summary

The logic is similar, but the generated passages are **not exactly the same**.

The biggest differences are:

1. `passage_id` is built differently.
2. The source `claims` text is not identical between the GCP TSV pipeline and the HF corpus pipeline.
3. Therefore the final chunk text and word counts differ even when the same patent `doc_id` exists in both corpora.

The title-prefixing and sentence-chunking logic are broadly aligned. The main mismatch for the example below is **source text content**, not title handling.

## Original GCP Script Logic

From `/home/user/SPLADE/.cache/gcs/patent_data/scripts/biencoder/datapool/get_all_claim_datapool.py`:

- Source corpus is pulled from OpenSearch index:
  - `src_index = "patent_search_index_1027"`
- Source fields:
  - `publ_id`
  - `appl_id`
  - `US_title`
  - `US_abstract`
  - `US_claims`
- Text cleanup:
  - replace `\n` with spaces
  - split on whitespace
  - re-join with single spaces
- Sentence chunking:
  - `nltk.sent_tokenize`
  - chunk budget `max_words=300`
- Final passage text:
  - `title_words + chunk_words`
  - target `MAX_WORDS = 100`
  - if the combined title+chunk is too long, trim the title first
  - if the chunk alone is already too long, keep the chunk and drop the title
- Output TSV row:
  1. `doc_id`
  2. `final_text`
  3. `appl_id`
  4. `publ_id + "&&&claim&&&" + chunk_idx`

Important detail:

- The GCP script writes `passage_id` from `publ_id`, not from `doc_id`.

## Local Passage Builder Logic

From `/home/user/SPLADE/src/data/patent_passages.py`:

- Source fields come from the HF corpus row:
  - `doc_id`
  - `title`
  - `claims`
- Text cleanup:
  - replace `\n` with spaces
  - split on whitespace
  - re-join with single spaces
- Sentence chunking:
  - NLTK `sent_tokenize` when available
  - fallback regex only if NLTK is unavailable
  - chunk budget `max_words=300`
- Final passage text:
  - same general `title + chunk` logic
  - same `100`-word title-prefixed target behavior
- Output row:
  - `passage_id = f"{doc_id}&&&claim&&&{chunk_idx}"`
  - `parent_doc_id = doc_id`
  - `source_doc_id = doc_id`

Important detail:

- The local builder uses `doc_id` as the base of `passage_id`, not `publ_id`.

## Example Comparison

Chosen overlapping patent:

- `doc_id = US17572861`

HF corpus metadata for this patent:

- `doc_id = US17572861`
- `publication_number = 20220223697`
- `title = SEMICONDUCTOR DEVICE AND METHOD OF MANUFACTURE`

TSV row metadata for the same patent:

- column 1 `doc_id = US17572861`
- column 3 `appl_id = US17572861`
- column 4 `passage_id = US20220223697A1&&&claim&&&0`

### Passage Count

- GCP TSV passages for this patent: `3`
- Local generated passages for this patent: `3`

So the coarse chunk count matched for this example.

### First Passage ID

GCP TSV:

```text
US20220223697A1&&&claim&&&0
```

Local:

```text
US17572861&&&claim&&&0
```

This shows the first exact mismatch:

- GCP TSV passage ids are based on `publ_id`
- local passage ids are based on `doc_id`

### First Passage Text

GCP TSV first passage:

```text
A semiconductor device comprising: a substrate having a first epitaxial layer arranged thereon and a voltage blocking element arranged in the first epitaxial layer; a second epitaxial layer arranged on the first epitaxial layer, and a vertical switching element arranged in the second epitaxial layer. The semiconductor device as claimed in claim 1, wherein the voltage blocking element is connected to a source of the vertical switching element. The semiconductor device as claimed in claim 1, wherein the voltage blocking element is connected to a potential other than a source, between the source and a drain, of the vertical switching element. ...
```

- word count: `251`

Local first passage:

```text
1. A semiconductor device comprising: a substrate having a first epitaxial layer arranged thereon and a voltage blocking element arranged in the first epitaxial layer; a second epitaxial layer arranged on the first epitaxial layer, and a vertical switching element arranged in the second epitaxial layer. 2. The semiconductor device as claimed in claim 1, wherein the voltage blocking element is connected to a source of the vertical switching element. 3. The semiconductor device as claimed in claim 1, wherein the voltage blocking element is connected to a potential other than a source, between the source and a drain, of the vertical switching element. ...
```

- word count: `262`

### Key Observation

The local version preserves numbered claims:

- `1.`
- `2.`
- `3.`

The GCP TSV version for the same overlapping patent does not preserve them in the same way:

- starts with `A semiconductor device ...`
- then `The semiconductor device ...`

This means the mismatch is not just `passage_id`.

The source `claims` content itself is different between:

- OpenSearch `US_claims` used by the GCP script
- HF corpus `claims` used by our local builder

## What Causes the Difference

### 1. Different source corpus field content

This is the main cause for the text mismatch.

For the same overlapping patent `US17572861`:

- HF corpus `claims` begins with:

```text
1. A semiconductor device comprising: ...

2. The semiconductor device as claimed in claim 1, ...

3. The semiconductor device as claimed in claim 1, ...
```

- GCP TSV generated text begins with:

```text
A semiconductor device comprising: ...
The semiconductor device as claimed in claim 1, ...
The semiconductor device as claimed in claim 1, ...
```

So the upstream `claims` field feeding the GCP script is not text-identical to the HF `claims` field.

### 2. Different passage id construction

GCP script:

```text
passage_id = publ_id + "&&&claim&&&" + chunk_idx
```

Local builder:

```text
passage_id = doc_id + "&&&claim&&&" + chunk_idx
```

For this example:

- `publ_id`-based:
  - `US20220223697A1&&&claim&&&0`
- `doc_id`-based:
  - `US17572861&&&claim&&&0`

### 3. Chunk boundary shifts follow from source-text differences

Even if the sentence splitter and word-budget logic are the same, adding or removing claim numbering changes:

- word counts
- where chunk boundaries fall
- where title trimming activates

In this example:

- GCP first passage: `251` words
- Local first passage: `262` words

So even with the same nominal `300 words` chunking rule, the exact chunk payload differs.

## What Did Not Cause the Difference Here

### Title prefixing

For this patent, the first chunk is already much longer than `100` words.

So in both pipelines:

- the title would be dropped because there is no room for it

That means the mismatch in this example is **not** because one side added title and the other side did not.

### Sentence splitting strategy

Both pipelines use NLTK sentence splitting in the intended path.

So for this example, the visible mismatch is not best explained by sentence-splitter differences. The stronger explanation is the underlying claim text mismatch.

## Practical Conclusion

If the goal is exact apples-to-apples reproduction against the original TSV pipeline, then matching only the chunking algorithm is not enough.

We also need to match:

1. the exact upstream source text field used by OpenSearch (`US_claims`)
2. the exact `passage_id` construction using `publ_id`

At the moment, our local HF-based builder is:

- logically similar
- but not byte-for-byte equivalent to the TSV-generation pipeline

## Recommended Next Step

For exact reproduction, do one of the following:

1. Use the downloaded TSV itself as the corpus source for encoding.
2. Or rebuild the local passage corpus from a source that exposes the same `publ_id` and `US_claims` content as the OpenSearch pipeline.

Using only the HF corpus `claims` field will not guarantee identical passage text.
