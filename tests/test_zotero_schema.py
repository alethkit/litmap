"""Zotero schema drift: item-type IDs differ between versions, and trashed
items must never surface as papers."""
import sqlite3

import numpy as np
import pytest

from litmap.zotero import get_all_items, get_collection, get_subcollection_map
from litmap.embedder import init_db, sync, prune_stale, DIMS
from unittest.mock import patch


@pytest.fixture
def zotero9_db(tmp_path):
    """Zotero 9-style schema: attachment=3, note=28, annotation=1, document=14.

    Under the old hardcoded exclusion (14, 26) the 'document' item would be
    dropped and the attachment/note would be embedded as papers.
    """
    db_path = tmp_path / "zotero9.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
        INSERT INTO itemTypes VALUES (1, 'annotation');
        INSERT INTO itemTypes VALUES (3, 'attachment');
        INSERT INTO itemTypes VALUES (14, 'document');
        INSERT INTO itemTypes VALUES (22, 'journalArticle');
        INSERT INTO itemTypes VALUES (28, 'note');

        CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
        INSERT INTO fields VALUES (1, 'title');
        INSERT INTO fields VALUES (2, 'abstractNote');
        INSERT INTO fields VALUES (6, 'date');
        INSERT INTO fields VALUES (8, 'DOI');
        INSERT INTO fields VALUES (13, 'url');

        CREATE TABLE creatorTypes (creatorTypeID INTEGER PRIMARY KEY, creatorType TEXT);
        INSERT INTO creatorTypes VALUES (8, 'author');

        CREATE TABLE items (itemID INTEGER PRIMARY KEY, itemTypeID INTEGER,
                            libraryID INTEGER DEFAULT 1, key TEXT);
        INSERT INTO items VALUES (1, 22, 1, 'PAPER001');   -- journal article
        INSERT INTO items VALUES (2, 14, 1, 'DOCUMENT1');  -- 'document' type: a real item
        INSERT INTO items VALUES (3, 3,  1, 'ATTACH001');  -- standalone PDF attachment
        INSERT INTO items VALUES (4, 28, 1, 'NOTE00001');  -- standalone note
        INSERT INTO items VALUES (5, 22, 1, 'TRASHED01');  -- in the trash

        CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO itemDataValues VALUES (1, 'A Real Paper');
        INSERT INTO itemDataValues VALUES (2, 'A Document Item');
        INSERT INTO itemDataValues VALUES (3, 'orphan.pdf');
        INSERT INTO itemDataValues VALUES (4, 'My note title');
        INSERT INTO itemDataValues VALUES (5, 'Deleted Paper');

        CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
        INSERT INTO itemData VALUES (1, 1, 1);
        INSERT INTO itemData VALUES (2, 1, 2);
        INSERT INTO itemData VALUES (3, 1, 3);
        INSERT INTO itemData VALUES (4, 1, 4);
        INSERT INTO itemData VALUES (5, 1, 5);

        CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
        CREATE TABLE itemCreators (itemID INTEGER, creatorID INTEGER,
                                   creatorTypeID INTEGER, orderIndex INTEGER);

        CREATE TABLE collections (collectionID INTEGER PRIMARY KEY, collectionName TEXT,
                                  parentCollectionID INTEGER, libraryID INTEGER DEFAULT 1);
        INSERT INTO collections VALUES (1, 'Omega', NULL, 1);
        CREATE TABLE collectionItems (collectionID INTEGER, itemID INTEGER);
        INSERT INTO collectionItems VALUES (1, 1);
        INSERT INTO collectionItems VALUES (1, 2);
        INSERT INTO collectionItems VALUES (1, 3);
        INSERT INTO collectionItems VALUES (1, 4);
        INSERT INTO collectionItems VALUES (1, 5);

        CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INTEGER,
                                      linkMode INTEGER, contentType TEXT, path TEXT);
        INSERT INTO itemAttachments VALUES (3, NULL, 0, 'application/pdf', 'storage:orphan.pdf');

        CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY, dateDeleted TEXT);
        INSERT INTO deletedItems VALUES (5, '2026-01-01 00:00:00');
    """)
    conn.commit()
    conn.close()
    return db_path


def test_excluded_types_resolved_by_name_not_id(zotero9_db):
    keys = {i.key for i in get_all_items(zotero9_db)}
    assert "PAPER001" in keys
    assert "DOCUMENT1" in keys, "'document' items are real papers and must not be excluded"
    assert "ATTACH001" not in keys
    assert "NOTE00001" not in keys


def test_trashed_items_are_excluded(zotero9_db):
    assert "TRASHED01" not in {i.key for i in get_all_items(zotero9_db)}
    assert "TRASHED01" not in {i.key for i in get_collection("Omega", zotero9_db)}
    assert "TRASHED01" not in get_subcollection_map(zotero9_db)


def test_subcollection_map_uses_runtime_types(zotero9_db):
    mapping = get_subcollection_map(zotero9_db)
    assert set(mapping) == {"PAPER001", "DOCUMENT1"}


def test_legacy_schema_without_deleteditems_still_works(zotero_db):
    # the shared fixture has no deletedItems table and uses attachment=14
    assert len(get_all_items(zotero_db)) == 3


def _store(db_path, key):
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO embeddings VALUES (?, ?, 'x')", (key, np.ones(DIMS, np.float32).tobytes()))
    conn.execute("INSERT INTO fulltext_embeddings (zotero_key, vector, embedded_at) VALUES (?, ?, 'x')",
                 (key, np.ones(DIMS, np.float32).tobytes()))
    conn.commit(); conn.close()


def test_prune_stale_removes_non_paper_keys(tmp_path):
    db_path = tmp_path / "emb.db"; init_db(db_path)
    for k in ("PAPER001", "ATTACH001", "TRASHED01"):
        _store(db_path, k)
    removed = prune_stale(db_path, {"PAPER001"})
    assert removed == 2
    conn = sqlite3.connect(db_path)
    assert {r[0] for r in conn.execute("SELECT zotero_key FROM embeddings")} == {"PAPER001"}
    assert {r[0] for r in conn.execute("SELECT zotero_key FROM fulltext_embeddings")} == {"PAPER001"}
    conn.close()


def test_prune_stale_is_a_noop_with_no_valid_keys(tmp_path):
    db_path = tmp_path / "emb.db"; init_db(db_path)
    _store(db_path, "PAPER001")
    assert prune_stale(db_path, set()) == 0
    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1
    conn.close()


def test_sync_prunes_attachments_embedded_by_older_versions(zotero9_db, tmp_path):
    db_path = tmp_path / "emb.db"; init_db(db_path)
    _store(db_path, "ATTACH001")
    with patch("litmap.embedder._get_model") as mock_model:
        mock_model.return_value.encode.return_value = np.ones((2, DIMS), dtype=np.float32)
        n = sync(db_path, zotero9_db)
    assert n == 2
    conn = sqlite3.connect(db_path)
    assert {r[0] for r in conn.execute("SELECT zotero_key FROM embeddings")} == {"PAPER001", "DOCUMENT1"}
    conn.close()
