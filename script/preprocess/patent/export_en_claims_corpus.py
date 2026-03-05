import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import nltk
from nltk.tokenize import sent_tokenize
from opensearchpy import OpenSearch
from tqdm import tqdm


def clean_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split())


def split_into_chunks_by_sentence_nltk(text: str, max_words: int = 300) -> list[str]:
    sentences = sent_tokenize(text)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if word_count == 0:
            continue
        if current_len + word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk).strip())
            current_chunk = [sentence]
            current_len = word_count
        else:
            current_chunk.append(sentence)
            current_len += word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())
    return chunks


def iter_scrolled_hits(
    client: OpenSearch,
    index: str,
    query: dict,
    source_fields: list[str],
    batch_size: int,
    scroll_time: str,
    slice_spec: dict | None = None,
) -> Iterable[dict]:
    search_body = {"track_total_hits": True, "query": query}
    if slice_spec is not None:
        search_body["slice"] = slice_spec
    response = client.search(
        index=index,
        scroll=scroll_time,
        size=batch_size,
        _source=source_fields,
        body=search_body,
    )
    scroll_id = response["_scroll_id"]
    try:
        hits = response["hits"]["hits"]
        while hits:
            for hit in hits:
                yield hit
            response = client.scroll(scroll_id=scroll_id, scroll=scroll_time)
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]
    finally:
        try:
            client.clear_scroll(body={"scroll_id": [scroll_id]})
        except Exception:
            pass


def build_opensearch_client(host: str, port: int, timeout: int) -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": host, "port": str(port)}],
        http_compress=True,
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=timeout,
        max_retries=10,
        retry_on_timeout=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export patent claim corpus.")
    parser.add_argument("--host", default="10.4.43.27", help="OpenSearch host")
    parser.add_argument("--port", type=int, default=9200, help="OpenSearch port")
    parser.add_argument("--index", default="patent_search", help="OpenSearch index/alias")
    parser.add_argument("--batch-size", type=int, default=5000, help="Scroll batch size")
    parser.add_argument("--scroll-time", default="10m", help="Scroll keep-alive")
    parser.add_argument("--timeout", type=int, default=120, help="OpenSearch timeout seconds")
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=Path("outputs/patent/us_en_claims_corpus.tsv"),
        help="Output TSV file path",
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=Path("outputs/patent/us_en_claims_corpus_stats.json"),
        help="Output stats manifest JSON path",
    )
    parser.add_argument(
        "--org-code",
        default="US",
        help="Patent organization code filter (e.g., US, KR)",
    )
    parser.add_argument(
        "--lang-types",
        nargs="*",
        default=None,
        help="Optional lang_type filters, e.g. --lang-types EN en",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional max number of patent docs to export",
    )
    parser.add_argument(
        "--chunk-max-words",
        type=int,
        default=300,
        help="Max words per sentence chunk before title prepend",
    )
    parser.add_argument(
        "--final-max-words",
        type=int,
        default=100,
        help="Max words in final output text (title + claim chunk)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar output",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=None,
        help="Optional sliced-scroll parallelism: total slice count.",
    )
    parser.add_argument(
        "--slice-id",
        type=int,
        default=None,
        help="Optional sliced-scroll parallelism: this worker's slice id (0-indexed).",
    )
    args = parser.parse_args()

    if (args.num_slices is None) != (args.slice_id is None):
        raise ValueError("`--num-slices` and `--slice-id` must be provided together.")
    if args.num_slices is not None and not (0 <= args.slice_id < args.num_slices):
        raise ValueError("`--slice-id` must satisfy 0 <= slice-id < num-slices.")

    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    output_path = args.output_tsv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = args.stats_json
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    client = build_opensearch_client(host=args.host, port=args.port, timeout=args.timeout)

    filters: list[dict] = [{"term": {"org_code": args.org_code}}]
    if args.lang_types:
        filters.append({"terms": {"lang_type": args.lang_types}})
    query = {"bool": {"filter": filters}}

    total_docs = (
        None
        if args.num_slices is not None
        else client.count(index=args.index, body={"query": query})["count"]
    )
    if args.max_docs is not None:
        target_docs = args.max_docs if total_docs is None else min(total_docs, args.max_docs)
    else:
        target_docs = total_docs
    slice_spec = (
        None
        if args.num_slices is None
        else {"id": int(args.slice_id), "max": int(args.num_slices)}
    )

    title_field = f"{args.org_code}_title"
    claims_field = f"{args.org_code}_claims"
    source_fields = ["publ_id", "appl_id", title_field, claims_field, "lang_type", "org_code"]
    docs_seen = 0
    chunks_written = 0
    docs_with_claims = 0

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        pbar = None if args.no_progress else tqdm(total=target_docs, desc="Exporting docs")

        for doc in iter_scrolled_hits(
            client=client,
            index=args.index,
            query=query,
            source_fields=source_fields,
            batch_size=args.batch_size,
            scroll_time=args.scroll_time,
            slice_spec=slice_spec,
        ):
            if args.max_docs is not None and docs_seen >= args.max_docs:
                break

            docs_seen += 1
            if pbar is not None:
                pbar.update(1)

            source = doc.get("_source", {})
            doc_id = doc.get("_id", "")
            title = clean_text(source.get(title_field, ""))
            content = clean_text(source.get(claims_field, ""))
            if not content:
                continue

            docs_with_claims += 1
            chunks = split_into_chunks_by_sentence_nltk(
                content,
                max_words=args.chunk_max_words,
            )

            for chunk_idx, chunk in enumerate(chunks):
                chunk_words = chunk.split()
                if not chunk_words:
                    continue

                if title:
                    title_words = title.split()
                    merged_words = title_words + chunk_words
                    if len(merged_words) <= args.final_max_words:
                        final_words = merged_words
                    else:
                        space_for_title = args.final_max_words - len(chunk_words)
                        if space_for_title > 0:
                            final_words = title_words[:space_for_title] + chunk_words
                        else:
                            final_words = chunk_words
                else:
                    final_words = chunk_words

                final_text = " ".join(final_words)
                writer.writerow(
                    [
                        doc_id,
                        final_text,
                        source.get("appl_id", ""),
                        f"{source.get('publ_id', '')}&&&claim&&&{chunk_idx}",
                    ]
                )
                chunks_written += 1
        if pbar is not None:
            pbar.close()

    stats = {
        "source_index": args.index,
        "filters": {
            "org_code": args.org_code,
            "lang_type": args.lang_types if args.lang_types else None,
        },
        "slice": slice_spec,
        "total_matching_docs": total_docs,
        "processed_docs": docs_seen,
        "docs_with_non_empty_claims": docs_with_claims,
        "claim_chunk_rows": chunks_written,
        "output_tsv": str(output_path.as_posix()),
    }
    with stats_path.open("w", encoding="utf-8") as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)

    matched_docs_text = f"{total_docs:,}" if total_docs is not None else "N/A (slice worker)"
    print(f"Patent docs matched query: {matched_docs_text}")
    print(f"Docs processed: {docs_seen:,}")
    print(f"Docs with non-empty claims: {docs_with_claims:,}")
    print(f"Claim chunks written: {chunks_written:,}")
    print(f"Output file: {output_path.resolve().as_posix()}")
    print(f"Stats manifest: {stats_path.resolve().as_posix()}")


if __name__ == "__main__":
    main()
