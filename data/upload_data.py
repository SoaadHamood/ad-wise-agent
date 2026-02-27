import os
import csv
import re
import time
from dataclasses import dataclass
from typing import Iterator, List, Dict

from dotenv import load_dotenv
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec


# =========================
# CONFIG - EDIT THESE
# =========================

INPUT_CSV = r"C:\data\amazon_category_ads.csv"
CATEGORY_COL = "category"
ADS_COL = "ads"

PINECONE_INDEX_NAME = "amazon-ads-index"
PINECONE_NAMESPACE = "amazon_ads"  # keep one namespace for everything

PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dims

# Chunking
ADS_PER_CHUNK = 200
ADS_OVERLAP = 10

# Batching
EMBED_BATCH_SIZE = 128
UPSERT_BATCH_SIZE = 500

# Safety for huge CSV cells / metadata
CSV_FIELD_LIMIT = 800 * 1024 * 1024  # 800MB
MAX_TEXT_CHARS_FOR_METADATA = 20000  # store only a safe amount of text per vector

# Retry
MAX_RETRIES = 5
RETRY_SLEEP_SECONDS = 2


# =========================
# Helpers
# =========================

def set_csv_field_limit(limit: int) -> None:
    try:
        csv.field_size_limit(limit)
    except OverflowError:
        csv.field_size_limit(50 * 1024 * 1024)


def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120]


def truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def iter_ads_from_comma_blob(blob: str) -> Iterator[str]:
    """
    Stream split comma-separated ads WITHOUT creating a giant list.
    Assumes commas inside each ad were removed earlier.
    """
    if not blob:
        return
    start = 0
    n = len(blob)
    for i, ch in enumerate(blob):
        if ch == ",":
            ad = blob[start:i].strip()
            if ad:
                yield ad
            start = i + 1
    if start < n:
        ad = blob[start:].strip()
        if ad:
            yield ad


def chunk_ads(ads_iter: Iterator[str], size: int, overlap: int) -> Iterator[List[str]]:
    buf: List[str] = []
    for ad in ads_iter:
        buf.append(ad)
        if len(buf) >= size:
            yield buf[:size]
            if overlap > 0:
                buf = buf[size - overlap :]
            else:
                buf = []
    if buf:
        yield buf  # leftover chunk


def ensure_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            deletion_protection="disabled",
        )


def batched(items, batch_size: int):
    batch = []
    for x in items:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def upsert_with_retry(index, vectors: List[Dict], namespace: str):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return index.upsert(vectors=vectors, namespace=namespace)
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP_SECONDS * attempt)
    raise last_err


@dataclass
class ChunkDoc:
    vec_id: str
    embed_text: str
    metadata: Dict


def iter_chunks_from_csv(path: str) -> Iterator[ChunkDoc]:
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or CATEGORY_COL not in reader.fieldnames or ADS_COL not in reader.fieldnames:
            raise RuntimeError(f"CSV must contain columns: {CATEGORY_COL}, {ADS_COL}. Found: {reader.fieldnames}")

        for row in reader:
            category = (row.get(CATEGORY_COL) or "").strip()
            blob = row.get(ADS_COL) or ""
            if not category or not blob.strip():
                continue

            cat_slug = slugify(category)

            ads_iter = iter_ads_from_comma_blob(blob)
            chunk_index = 0

            for ads_chunk in chunk_ads(ads_iter, ADS_PER_CHUNK, ADS_OVERLAP):
                if not ads_chunk:
                    continue
                chunk_index += 1

                ads_blob = ",".join(ads_chunk)
                embed_text = f"category: {category}\nads: {ads_blob}"

                vec_id = f"{cat_slug}__chunk_{chunk_index:06d}"

                meta_text = truncate(ads_blob, MAX_TEXT_CHARS_FOR_METADATA)

                metadata = {
                    "category": category,
                    "chunk_index": chunk_index,
                    "ad_count": len(ads_chunk),
                    # this is the actual content you will use in RAG:
                    "ads_blob": meta_text,
                }

                yield ChunkDoc(vec_id=vec_id, embed_text=embed_text, metadata=metadata)


def main():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY in .env / environment")

    set_csv_field_limit(CSV_FIELD_LIMIT)

    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=api_key)
    ensure_index(pc, PINECONE_INDEX_NAME, dim)
    index = pc.Index(PINECONE_INDEX_NAME)

    chunk_iter = iter_chunks_from_csv(INPUT_CSV)

    total_upserted = 0
    buffer: List[ChunkDoc] = []

    def flush(buf: List[ChunkDoc]):
        nonlocal total_upserted
        if not buf:
            return

        texts = [d.embed_text for d in buf]
        embs = model.encode(
            texts,
            batch_size=min(64, len(texts)),
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        vectors = []
        for d, vec in zip(buf, embs):
            vectors.append({
                "id": d.vec_id,
                "values": vec.tolist(),
                "metadata": d.metadata,
            })

        for up in batched(vectors, UPSERT_BATCH_SIZE):
            upsert_with_retry(index, up, PINECONE_NAMESPACE)
            total_upserted += len(up)

    for doc in tqdm(chunk_iter, desc="Building & uploading chunks"):
        buffer.append(doc)
        if len(buffer) >= EMBED_BATCH_SIZE:
            flush(buffer)
            buffer = []

    flush(buffer)

    print(f"✅ Done. Upserted vectors: {total_upserted:,}")
    print(f"Index: {PINECONE_INDEX_NAME} | Namespace: {PINECONE_NAMESPACE}")


if __name__ == "__main__":
    main()