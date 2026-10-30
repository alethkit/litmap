from __future__ import annotations
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from tqdm import tqdm

from litmap.zotero import get_all_items, Item, ZOTERO_DB
from litmap.sources import (
    text_for_item, has_local_fulltext, can_fetch,
    extract_pdf_text as _extract_pdf_text,  # re-exported for backward compatibility
)

EMBEDDINGS_DB = Path.home() / "LitLake" / "embeddings.db"
MODEL_NAME = "Alibaba-NLP/gte-modernbert-base"
DIMS = 768
_BATCH_SIZE = 32
# GTE-ModernBERT context window; leave a small margin
_MAX_TOKENS = 8000
_model = None
_tokenizer = None


def pick_device() -> str:
    """Torch device for the embedding model.

    Honour LITMAP_DEVICE if set (e.g. "cpu", "mps", "cuda"); otherwise use
    Apple's MPS backend when it is actually available and fall back to CPU.
    MPS can be missing inside sandboxes, containers and VMs even on a Mac.
    """
    import os
    forced = os.environ.get("LITMAP_DEVICE", "").strip()
    if forced:
        return forced
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device=pick_device())
    return _model


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def init_db(db_path: Path = EMBEDDINGS_DB) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS embeddings (
            zotero_key  TEXT PRIMARY KEY,
            vector      BLOB NOT NULL,
            embedded_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fulltext_embeddings (
            zotero_key  TEXT PRIMARY KEY,
            vector      BLOB NOT NULL,
            embedded_at TEXT NOT NULL,
            n_tokens    INTEGER,
            n_chunks    INTEGER
        );
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.execute("INSERT OR IGNORE INTO meta VALUES ('model', ?)", (MODEL_NAME,))
    conn.execute("INSERT OR IGNORE INTO meta VALUES ('dims', ?)", (str(DIMS),))
    # Migration: databases created before HTML/URL sources lack the `source` column.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(fulltext_embeddings)").fetchall()}
    if "source" not in cols:
        conn.execute("ALTER TABLE fulltext_embeddings ADD COLUMN source TEXT")
    conn.commit()
    conn.close()


def embed_text(text: str, db_path: Path = EMBEDDINGS_DB) -> np.ndarray:
    model = _get_model()
    vecs = model.encode([text], normalize_embeddings=True)
    return np.array(vecs[0], dtype=np.float32)


def get_embedding(zotero_key: str, db_path: Path = EMBEDDINGS_DB) -> Optional[np.ndarray]:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT vector FROM embeddings WHERE zotero_key = ?", (zotero_key,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32).copy()


def load_all_embeddings(
    db_path: Path = EMBEDDINGS_DB,
    scope_keys: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    conn = sqlite3.connect(db_path)
    if scope_keys:
        placeholders = ",".join("?" * len(scope_keys))
        rows = conn.execute(
            f"SELECT zotero_key, vector FROM embeddings WHERE zotero_key IN ({placeholders})",
            scope_keys,
        ).fetchall()
    else:
        rows = conn.execute("SELECT zotero_key, vector FROM embeddings").fetchall()
    conn.close()
    if not rows:
        return np.empty((0, DIMS), dtype=np.float32), []
    keys = [r[0] for r in rows]
    matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return matrix, keys


def sync(
    db_path: Path = EMBEDDINGS_DB,
    zotero_db: Path = ZOTERO_DB,
    force: bool = False,
) -> int:
    init_db(db_path)
    all_items = get_all_items(zotero_db)
    prune_stale(db_path, {i.key for i in all_items})
    if force:
        items_to_embed = all_items
    else:
        existing_keys = _existing_keys(db_path)
        items_to_embed = [i for i in all_items if i.key not in existing_keys]
    if not items_to_embed:
        return 0
    _embed_and_store(items_to_embed, db_path)
    return len(items_to_embed)


def prune_stale(db_path: Path, valid_keys: set[str]) -> int:
    """Delete cached vectors whose Zotero key is no longer a paper.

    Covers items that were deleted/trashed in Zotero and, after the item-type
    fix, attachments that earlier versions wrongly embedded as papers.
    Does nothing when valid_keys is empty (e.g. an unreadable Zotero DB), so a
    transient failure can never wipe the cache. Returns the number of
    rows removed from the embeddings table.
    """
    if not valid_keys:
        return 0
    conn = sqlite3.connect(db_path)
    try:
        removed = 0
        for table in ("embeddings", "fulltext_embeddings"):
            stale = [
                r[0] for r in conn.execute(f"SELECT zotero_key FROM {table}").fetchall()
                if r[0] not in valid_keys
            ]
            for start in range(0, len(stale), 500):
                chunk = stale[start:start + 500]
                conn.execute(
                    f"DELETE FROM {table} WHERE zotero_key IN ({','.join('?' * len(chunk))})",
                    chunk,
                )
            if table == "embeddings":
                removed = len(stale)
        conn.commit()
    finally:
        conn.close()
    return removed


def _existing_keys(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT zotero_key FROM embeddings").fetchall()
    conn.close()
    return {r[0] for r in rows}


def _embed_and_store(items: list[Item], db_path: Path) -> None:
    model = _get_model()
    texts = [f"{i.title} {i.abstract} {i.keywords}".strip() for i in items]
    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc).isoformat()
    with tqdm(total=len(items), desc="Syncing new papers", unit="paper") as bar:
        for batch_start in range(0, len(items), _BATCH_SIZE):
            batch_items = items[batch_start:batch_start + _BATCH_SIZE]
            batch_texts = texts[batch_start:batch_start + _BATCH_SIZE]
            vecs = model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
            for item, vec in zip(batch_items, vecs):
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings (zotero_key, vector, embedded_at) VALUES (?, ?, ?)",
                    (item.key, np.array(vec, dtype=np.float32).tobytes(), now),
                )
                bar.update(1)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Full-text embedding (PDF, HTML snapshot, or fetched README / web page)
# ---------------------------------------------------------------------------


def _truncate_to_tokens(text: str, max_tokens: int = _MAX_TOKENS) -> tuple[str, int]:
    """Truncate text to at most max_tokens tokens. Returns (truncated_text, n_tokens)."""
    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    n_tokens = len(tokens)
    if n_tokens <= max_tokens:
        return text, n_tokens
    truncated_ids = tokens[:max_tokens]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return truncated_text, max_tokens


def _embed_fulltext_single(
    item: Item,
    max_tokens: int = _MAX_TOKENS,
    fetch_missing: bool = False,
) -> Optional[tuple[np.ndarray, int, int, str]]:
    """Extract, chunk, embed and average a single item's full text.

    Text comes from the item's PDF, else its HTML snapshot, else (only when
    fetch_missing=True) a GitHub README or the page at its URL.
    Strategy: split into non-overlapping chunks of max_tokens tokens,
    embed each chunk, then return the L2-normalised mean vector.
    Returns (vector, n_tokens, n_chunks, source) or None if no text extracted.
    """
    raw_text, source = text_for_item(item, fetch_missing=fetch_missing)
    if not raw_text.strip():
        return None

    tokenizer = _get_tokenizer()
    model = _get_model()

    all_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
    n_tokens = len(all_tokens)
    if n_tokens == 0:
        return None

    # Split into chunks
    chunk_ids = [
        all_tokens[start:start + max_tokens]
        for start in range(0, n_tokens, max_tokens)
    ]
    chunk_texts = [
        tokenizer.decode(chunk, skip_special_tokens=True)
        for chunk in chunk_ids
    ]
    n_chunks = len(chunk_texts)

    # Encode one chunk at a time to avoid OOM on MPS with long documents
    chunk_vecs = []
    for chunk_text in chunk_texts:
        vec = model.encode([chunk_text], normalize_embeddings=True, show_progress_bar=False)
        chunk_vecs.append(vec[0])
    vecs = np.stack(chunk_vecs)
    mean_vec = np.mean(vecs, axis=0).astype(np.float32)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm

    return mean_vec, n_tokens, n_chunks, source


def _existing_fulltext_keys(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT zotero_key FROM fulltext_embeddings").fetchall()
    conn.close()
    return {r[0] for r in rows}


def load_all_fulltext_embeddings(
    db_path: Path = EMBEDDINGS_DB,
    scope_keys: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """Load full-text embeddings (falls back to title+abstract embeddings if absent)."""
    conn = sqlite3.connect(db_path)
    if scope_keys:
        placeholders = ",".join("?" * len(scope_keys))
        # Prefer fulltext; fall back to title+abstract
        rows = conn.execute(
            f"""
            SELECT COALESCE(ft.zotero_key, e.zotero_key) AS zotero_key,
                   COALESCE(ft.vector, e.vector) AS vector
            FROM embeddings e
            LEFT JOIN fulltext_embeddings ft ON ft.zotero_key = e.zotero_key
            WHERE e.zotero_key IN ({placeholders})
            """,
            scope_keys,
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT COALESCE(ft.zotero_key, e.zotero_key) AS zotero_key,
                   COALESCE(ft.vector, e.vector) AS vector
            FROM embeddings e
            LEFT JOIN fulltext_embeddings ft ON ft.zotero_key = e.zotero_key
            """
        ).fetchall()
    conn.close()
    if not rows:
        return np.empty((0, DIMS), dtype=np.float32), []
    keys = [r[0] for r in rows]
    matrix = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return matrix, keys


def sync_fulltext(
    db_path: Path = EMBEDDINGS_DB,
    zotero_db: Path = ZOTERO_DB,
    force: bool = False,
    max_tokens: int = _MAX_TOKENS,
    collection: Optional[str] = None,
    fetch_missing: bool = False,
) -> tuple[int, int]:
    """Embed full text for Zotero items.

    Candidates are items with a local PDF or HTML snapshot. With
    fetch_missing=True, items that have only a URL are also candidates: a
    github.com repository URL yields its README, any other URL yields the
    visible text of the page. Local files are never bypassed by a fetch.

    If collection is given, only items in that collection are processed.
    Skips items already in fulltext_embeddings unless force=True.
    Returns (n_embedded, n_skipped), where n_skipped counts items with no
    usable text source (or whose source yielded no text).
    """
    init_db(db_path)
    if collection:
        from litmap.zotero import get_collection
        all_items = get_collection(collection, zotero_db)
    else:
        all_items = get_all_items(zotero_db)
    candidates = [
        i for i in all_items
        if has_local_fulltext(i) or (fetch_missing and can_fetch(i))
    ]
    n_skipped = len(all_items) - len(candidates)

    if not force:
        existing = _existing_fulltext_keys(db_path)
        candidates = [i for i in candidates if i.key not in existing]

    if not candidates:
        return 0, n_skipped

    conn = sqlite3.connect(db_path)
    now = datetime.now(timezone.utc).isoformat()
    n_embedded = 0

    with tqdm(total=len(candidates), desc="Embedding full text", unit="item") as bar:
        for item in candidates:
            src = item.fulltext_path
            bar.set_postfix({"item": (src.name if src else item.url)[:40]})
            result = _embed_fulltext_single(item, max_tokens=max_tokens, fetch_missing=fetch_missing)
            if result is not None:
                vec, n_tokens, n_chunks, source = result
                conn.execute(
                    """INSERT OR REPLACE INTO fulltext_embeddings
                       (zotero_key, vector, embedded_at, n_tokens, n_chunks, source)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (item.key, vec.tobytes(), now, n_tokens, n_chunks, source),
                )
                conn.commit()
                n_embedded += 1
            else:
                n_skipped += 1
            bar.update(1)

    conn.close()
    return n_embedded, n_skipped
