"""Sentence cosine similarities. Only the OpenAI backend sends sentence tokens off-device."""

import hashlib
import json
import os
from contextlib import closing
from functools import lru_cache
from pathlib import Path
import re
import sqlite3
import tempfile

import numpy as np
from filelock import FileLock


BACKENDS = ("tfidf", "sbert", "legalbert", "openai")
MODEL_IDS = {
    "tfidf": "sklearn-smoothed-tfidf-unigrams",
    "sbert": "sentence-transformers/all-MiniLM-L6-v2",
    "legalbert": "nlpaueb/legal-bert-base-uncased",
    "openai": "text-embedding-3-large",
}
MODEL_REVISIONS = {
    "sbert": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "legalbert": "15b570cbf88259610b082a167dacc190124f60f6",
}
# Published sequence limits including special tokens. Long local inputs are truncated.
MAX_LENGTH = {"sbert": 256, "legalbert": 512}
OPENAI_DIMENSIONS = 3072
OPENAI_CONTEXT = 8191
# Stay below the service's 300,000 token / 2,048 input caps.
OPENAI_BATCH_TOKENS = 24_000
OPENAI_BATCH_ITEMS = 128
TFIDF_TOKEN_PATTERN = r"(?u)\b\w+\b"


def _tfidf(sentences):
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not any(re.search(r"\w", sentence) for sentence in sentences):
        return np.zeros((len(sentences), len(sentences)), dtype=np.float32)
    vectorizer = TfidfVectorizer(token_pattern=TFIDF_TOKEN_PATTERN, dtype=np.float32)
    vectors = vectorizer.fit_transform(sentences)
    return (vectors @ vectors.T).toarray()


@lru_cache(maxsize=1)
def _local_model(backend):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    model_id = MODEL_IDS[backend]
    revision = MODEL_REVISIONS[backend]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision).eval().to(device)
    return tokenizer, model, device


def _local_embeddings(backend, sentences):
    import torch

    tokenizer, model, device = _local_model(backend)
    # Adjacent lengths in each batch limit the padding cost on long legal sentences.
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    embeddings = np.empty((len(sentences), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), 16):
            indices = order[start:start + 16]
            encoded = tokenizer(
                [sentences[i] for i in indices], padding=True, truncation=True,
                max_length=MAX_LENGTH[backend], return_tensors="pt",
            ).to(device)
            tokens = model(**encoded).last_hidden_state
            mask = encoded.attention_mask.unsqueeze(-1)
            pooled = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
            embeddings[indices] = pooled.cpu().numpy()
    return embeddings


def _openai_batches(chunks):
    batch = []
    total = 0
    for digest, ids in chunks:
        if batch and (len(batch) == OPENAI_BATCH_ITEMS or total + len(ids) > OPENAI_BATCH_TOKENS):
            yield batch
            batch = []
            total = 0
        batch.append((digest, ids))
        total += len(ids)
    if batch:
        yield batch


def _openai_embeddings(sentences, cache_dir):
    from openai import OpenAI, APIStatusError
    from dotenv import dotenv_values
    import tiktoken

    model_id = MODEL_IDS["openai"]
    tokenizer = tiktoken.encoding_for_model(model_id)
    chunks = {}
    documents = []
    for sentence in sentences:
        ids = tokenizer.encode_ordinary(sentence)
        parts = []
        for offset in range(0, len(ids), OPENAI_CONTEXT):
            part = ids[offset:offset + OPENAI_CONTEXT]
            digest = hashlib.sha256(np.asarray(part, dtype="<i4").tobytes()).digest()
            chunks[digest] = part
            parts.append((digest, len(part)))
        documents.append(parts)

    # Persist each successful API batch independently so interrupted runs resume.
    embeddings = {}
    database = cache_dir / f"{model_id}-{OPENAI_DIMENSIONS}.sqlite3"
    # The same token chunk can occur in different judgments. Lock the shared
    # DB through the remote request and commit so concurrent processes reuse it.
    with FileLock(str(database) + ".lock"), closing(sqlite3.connect(database)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS vectors (digest BLOB PRIMARY KEY, vector BLOB NOT NULL)")
        missing = []
        for digest, part in chunks.items():
            row = db.execute("SELECT vector FROM vectors WHERE digest = ?", (digest,)).fetchone()
            if row:
                embeddings[digest] = np.frombuffer(row[0], dtype="<f4")
            else:
                missing.append((digest, part))
        if missing:
            key = os.getenv("OPENAI_API_KEY") or dotenv_values(Path.cwd() / ".env", interpolate=False).get("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("Set OPENAI_API_KEY to run uncached OpenAI embeddings.")
            client = OpenAI(api_key=key, max_retries=5, timeout=120)
            try:
                for batch in _openai_batches(missing):
                    result = client.embeddings.create(
                        model=model_id, input=[ids for _, ids in batch],
                        dimensions=OPENAI_DIMENSIONS, encoding_format="float",
                    )
                    indices = sorted(item.index for item in result.data)
                    if indices != list(range(len(batch))):
                        raise RuntimeError("OpenAI returned incomplete or duplicate embedding indices.")
                    with db:
                        for item in result.data:
                            digest = batch[item.index][0]
                            vector = np.asarray(item.embedding, dtype="<f4")
                            if vector.shape != (OPENAI_DIMENSIONS,) or not np.isfinite(vector).all():
                                raise ValueError("OpenAI returned an invalid embedding.")
                            embeddings[digest] = vector
                            db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?)", (digest, vector.tobytes()))
            except APIStatusError as error:
                # Authentication responses may echo part of the submitted key.
                raise RuntimeError(f"OpenAI embeddings request failed (HTTP {error.status_code}, code {error.code}).") from None
            finally:
                client.close()
    return np.stack([
        np.average([embeddings[digest] for digest, _ in parts], axis=0,
                   weights=[length for _, length in parts])
        for parts in documents
    ]).astype(np.float32)


def similarity_matrix(backend: str, sentences: list[str], cache_dir: Path) -> np.ndarray:
    """Return the symmetric cosine matrix in sentence order, including the diagonal.

    TF-IDF fits one document at a time. The transformer encoders use masked mean
    pooling. Oversized OpenAI inputs are token-weighted averages of chunk embeddings.
    The caller handles negative cosines, self-edges and rhetorical role weighting.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Choose from {BACKENDS}.")
    if any(not s.strip() for s in sentences):
        raise ValueError("Each input sentence must contain text.")
    if not sentences:
        return np.empty((0, 0), dtype=np.float32)
    if backend == "openai":
        settings = ["token-weighted-chunk-mean", OPENAI_CONTEXT, OPENAI_DIMENSIONS]
    elif backend == "tfidf":
        settings = ["smoothed-unigram-tfidf", TFIDF_TOKEN_PATTERN]
    else:
        settings = ["attention-masked-mean", MAX_LENGTH[backend]]
    identity = [2, backend, MODEL_IDS[backend], MODEL_REVISIONS.get(backend), settings, sentences]
    key = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
    directory = Path(cache_dir) / "similarity" / backend
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (key + ".npy")
    if target.is_file():
        return np.load(target, allow_pickle=False)
    if backend == "tfidf":
        matrix = _tfidf(sentences)
    else:
        if backend == "openai":
            vectors = _openai_embeddings(sentences, directory)
        else:
            vectors = _local_embeddings(backend, sentences)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= np.maximum(norms, np.finfo(vectors.dtype).tiny)
        matrix = vectors @ vectors.T
    np.clip(matrix, -1, 1, out=matrix)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{backend} produced non-finite sentence similarities.")
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, matrix, allow_pickle=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return matrix
