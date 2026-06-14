"""
build_corpus.py
Fetches FastAPI documentation from GitHub raw content, cleans markdown,
and chunks into passages. Outputs corpus.jsonl to data/.
"""
import re
import json
import time
import requests
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/tiangolo/fastapi/master/"

# 55 documentation pages — tutorial, advanced, deployment, and core pages
DOC_PATHS = [
    "docs/en/docs/index.md",
    "docs/en/docs/features.md",
    "docs/en/docs/python-types.md",
    "docs/en/docs/async.md",
    "docs/en/docs/benchmarks.md",
    "docs/en/docs/tutorial/first-steps.md",
    "docs/en/docs/tutorial/path-params.md",
    "docs/en/docs/tutorial/query-params.md",
    "docs/en/docs/tutorial/query-params-str-validations.md",
    "docs/en/docs/tutorial/path-params-numeric-validations.md",
    "docs/en/docs/tutorial/body.md",
    "docs/en/docs/tutorial/body-multiple-params.md",
    "docs/en/docs/tutorial/body-updates.md",
    "docs/en/docs/tutorial/response-model.md",
    "docs/en/docs/tutorial/response-status-code.md",
    "docs/en/docs/tutorial/extra-models.md",
    "docs/en/docs/tutorial/request-files.md",
    "docs/en/docs/tutorial/request-forms-and-files.md",
    "docs/en/docs/tutorial/cookie-params.md",
    "docs/en/docs/tutorial/header-params.md",
    "docs/en/docs/tutorial/handling-errors.md",
    "docs/en/docs/tutorial/path-operation-configuration.md",
    "docs/en/docs/tutorial/background-tasks.md",
    "docs/en/docs/tutorial/middleware.md",
    "docs/en/docs/tutorial/cors.md",
    "docs/en/docs/tutorial/sql-databases.md",
    "docs/en/docs/tutorial/bigger-applications.md",
    "docs/en/docs/tutorial/static-files.md",
    "docs/en/docs/tutorial/encoder.md",
    "docs/en/docs/tutorial/schema-extra-example.md",
    "docs/en/docs/tutorial/extra-data-types.md",
    "docs/en/docs/tutorial/debugging.md",
    "docs/en/docs/tutorial/testing.md",
    "docs/en/docs/tutorial/dependencies/index.md",
    "docs/en/docs/tutorial/dependencies/classes-as-dependencies.md",
    "docs/en/docs/tutorial/dependencies/sub-dependencies.md",
    "docs/en/docs/tutorial/dependencies/dependencies-in-path-operation-decorators.md",
    "docs/en/docs/tutorial/dependencies/global-dependencies.md",
    "docs/en/docs/tutorial/security/index.md",
    "docs/en/docs/tutorial/security/oauth2-jwt.md",
    "docs/en/docs/advanced/response-directly.md",
    "docs/en/docs/advanced/response-cookies.md",
    "docs/en/docs/advanced/response-headers.md",
    "docs/en/docs/advanced/additional-status-codes.md",
    "docs/en/docs/advanced/additional-responses.md",
    "docs/en/docs/advanced/path-operation-advanced-configuration.md",
    "docs/en/docs/advanced/middleware.md",
    "docs/en/docs/advanced/events.md",
    "docs/en/docs/advanced/settings.md",
    "docs/en/docs/advanced/response-directly.md",
    "docs/en/docs/advanced/custom-response.md",
    "docs/en/docs/deployment/index.md",
    "docs/en/docs/deployment/versions.md",
    "docs/en/docs/deployment/https.md",
    "docs/en/docs/deployment/manually.md",
    "docs/en/docs/deployment/server-workers.md",
    "docs/en/docs/deployment/docker.md",
]

# Deduplicate
DOC_PATHS = list(dict.fromkeys(DOC_PATHS))


def clean_markdown(text: str) -> str:
    """Strip markdown syntax, preserve the prose and key technical terms."""
    # Remove code blocks but mark them (they contain key technical terms)
    text = re.sub(r'```[^\n]*\n(.*?)```', lambda m: ' ' + m.group(1).replace('\n', ' ') + ' ', text, flags=re.DOTALL)
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove frontmatter
    text = re.sub(r'^---.*?---\s*', '', text, flags=re.DOTALL)
    # Remove template tags like {!> ...!}
    text = re.sub(r'\{[!>][^}]*\}', '', text)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    # Convert headings to plain text (keep the content, remove #)
    text = re.sub(r'^#{1,6}\s+(.+?)\s*\{[^}]*\}', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove image markdown
    text = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', r'\1', text)
    # Convert links to just their text
    text = re.sub(r'\[([^\]]+)\]\([^\)]*\)', r'\1', text)
    # Remove bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalise whitespace within lines (not newlines)
    lines = text.split('\n')
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    text = '\n'.join(lines)
    return text.strip()


def chunk_page(doc_id: str, page_path: str, text: str) -> list[dict]:
    """
    Split a cleaned page into paragraph-level chunks.
    - Split on double newlines (paragraph boundaries)
    - Merge very short paragraphs (<80 chars) with the next
    - Split very long paragraphs (>600 chars) at sentence boundaries
    - Add a 1-sentence overlap between adjacent chunks for context continuity
    """
    # Split into raw paragraphs
    raw = [p.strip() for p in re.split(r'\n\s*\n+', text)]
    raw = [p for p in raw if len(p) > 30]  # drop near-empty

    # Merge very short consecutive paragraphs
    merged = []
    buf = ""
    for p in raw:
        if len(buf) < 80:
            buf = (buf + " " + p).strip() if buf else p
        else:
            merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)

    # Further split very long chunks at sentence boundaries
    final_paras = []
    for para in merged:
        if len(para) <= 600:
            final_paras.append(para)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= 600:
                    current = (current + " " + sent).strip() if current else sent
                else:
                    if current:
                        final_paras.append(current)
                    current = sent
            if current:
                final_paras.append(current)

    # Build chunk objects with 1-sentence lookahead overlap
    chunks = []
    for i, para in enumerate(final_paras):
        # Add first sentence of next paragraph as trailing context
        if i + 1 < len(final_paras):
            next_sents = re.split(r'(?<=[.!?])\s+', final_paras[i + 1])
            if next_sents:
                overlap = next_sents[0]
                text_with_overlap = para + " " + overlap
            else:
                text_with_overlap = para
        else:
            text_with_overlap = para

        chunk_id = f"{doc_id}__chunk_{i:03d}"
        chunks.append({
            "chunk_id":  chunk_id,
            "doc_id":    doc_id,
            "page_path": page_path,
            "chunk_idx": i,
            "text":      text_with_overlap,
        })

    return chunks


def fetch_and_chunk(path: str) -> list[dict]:
    """Fetch one doc page, clean, and return its chunks."""
    url = BASE_URL + path
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"  SKIP {path} (HTTP {r.status_code})")
            return []
    except Exception as e:
        print(f"  ERROR {path}: {e}")
        return []

    # Derive a short stable doc_id from the path
    doc_id = path.replace("docs/en/docs/", "").replace(".md", "").replace("/", "__")
    cleaned = clean_markdown(r.text)
    chunks = chunk_page(doc_id, path, cleaned)
    return chunks


def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "corpus.jsonl"

    all_chunks = []
    seen_doc_ids = set()

    print(f"Fetching {len(DOC_PATHS)} documentation pages...")
    for i, path in enumerate(DOC_PATHS):
        doc_id = path.replace("docs/en/docs/", "").replace(".md", "").replace("/", "__")
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)

        print(f"  [{i+1:2}/{len(DOC_PATHS)}] {path.split('docs/en/docs/')[-1]}", end="")
        chunks = fetch_and_chunk(path)
        all_chunks.extend(chunks)
        print(f" → {len(chunks)} chunks")
        time.sleep(0.15)  # polite rate limiting

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nCorpus built: {len(all_chunks)} chunks from {len(seen_doc_ids)} pages")
    print(f"Saved to {out_path}")

    # Quick stats
    lengths = [len(c["text"]) for c in all_chunks]
    print(f"Chunk length — min: {min(lengths)}, max: {max(lengths)}, "
          f"mean: {sum(lengths)//len(lengths)}")


if __name__ == "__main__":
    main()
