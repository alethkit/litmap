"""Tests for non-PDF full-text sources: HTML snapshots and URL fallbacks."""
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from litmap.sources import (
    extract_html_text, github_repo_from_url, text_for_item,
)
from litmap.zotero import Item, get_collection, get_item
from litmap.embedder import init_db, sync_fulltext, DIMS


LONG_PROSE = " ".join(f"word{i}" for i in range(120))


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

def test_extract_html_text_strips_scripts_styles_and_chrome():
    html = f"""
    <html><head><title>T</title><style>body{{color:red}}</style></head>
    <body>
      <nav>Home About Contact</nav>
      <script>var x = "SCRIPTTEXT";</script>
      <p>First paragraph.</p><p>Second paragraph.</p>
      <footer>Copyright FOOTERTEXT</footer>
    </body></html>"""
    text = extract_html_text(html)
    assert "SCRIPTTEXT" not in text
    assert "color:red" not in text
    assert "FOOTERTEXT" not in text
    assert "Home About" not in text
    # block boundaries become newlines, so adjacent paragraphs don't fuse
    assert "First paragraph.\n\nSecond paragraph." in text


def test_extract_html_text_prefers_article_region():
    html = f"<html><body><div>SIDEBAR noise here</div><article>{LONG_PROSE}</article></body></html>"
    text = extract_html_text(html)
    assert "SIDEBAR" not in text
    assert "word0" in text and "word119" in text


def test_extract_html_text_falls_back_when_article_is_a_stub():
    html = f"<html><body><article>Loading…</article><div>{LONG_PROSE}</div></body></html>"
    text = extract_html_text(html)
    assert "word50" in text


def test_extract_html_text_decodes_entities():
    assert "a & b < c" in extract_html_text("<p>a &amp; b &lt; c</p>")


# ---------------------------------------------------------------------------
# GitHub URL parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://github.com/geohot/qira", ("geohot", "qira")),
    ("https://github.com/geohot/qira/", ("geohot", "qira")),
    ("https://github.com/geohot/qira.git", ("geohot", "qira")),
    ("https://github.com/tklengyel/drakvuf/tree/main/src", ("tklengyel", "drakvuf")),
    ("https://github.com/letoram/senseye/wiki/Home", ("letoram", "senseye")),
    ("http://www.github.com/a/b#readme", ("a", "b")),
    ("https://github.com/LaurieWired", None),            # user page, no repo
    ("https://github.com/topics/fuzzing", None),
    ("https://gitlab.com/owner/repo", None),
    ("https://sean.heelan.io/", None),
    ("", None),
])
def test_github_repo_from_url(url, expected):
    assert github_repo_from_url(url) == expected


# ---------------------------------------------------------------------------
# Dispatcher: source preference and network gating
# ---------------------------------------------------------------------------

def _item(**kw) -> Item:
    base = dict(key="K", title="t", abstract="", authors=[], year="", doi="")
    base.update(kw)
    return Item(**base)


def test_text_for_item_prefers_pdf_over_html(tmp_path):
    pdf = tmp_path / "a.pdf"; pdf.write_bytes(b"%PDF-")
    page = tmp_path / "a.html"; page.write_text(f"<p>{LONG_PROSE}</p>")
    with patch("litmap.sources.extract_pdf_text", return_value=LONG_PROSE.upper()):
        text, source = text_for_item(_item(pdf_path=pdf, html_path=page))
    assert source == "pdf"
    assert text == LONG_PROSE.upper()


def test_text_for_item_uses_html_when_pdf_unreadable(tmp_path):
    pdf = tmp_path / "a.pdf"; pdf.write_bytes(b"not a pdf")
    page = tmp_path / "a.html"; page.write_text(f"<p>{LONG_PROSE}</p>")
    text, source = text_for_item(_item(pdf_path=pdf, html_path=page))
    assert source == "html"
    assert "word3" in text


def test_text_for_item_never_touches_network_unless_asked():
    with patch("litmap.sources._http_get", side_effect=AssertionError("network used")):
        text, source = text_for_item(_item(url="https://github.com/geohot/qira"), fetch_missing=False)
    assert (text, source) == ("", "")


def test_text_for_item_fetches_github_readme_when_asked():
    with patch("litmap.sources.fetch_github_readme", return_value=LONG_PROSE) as fetch:
        text, source = text_for_item(_item(url="https://github.com/geohot/qira"), fetch_missing=True)
    fetch.assert_called_once_with("geohot", "qira")
    assert source == "github"


def test_text_for_item_falls_back_to_page_text_for_non_github_urls():
    with patch("litmap.sources._http_get", return_value=(f"<article>{LONG_PROSE}</article>".encode(), "text/html")):
        text, source = text_for_item(_item(url="https://example.org/post"), fetch_missing=True)
    assert source == "url"
    assert "word7" in text


def test_text_for_item_rejects_too_short_fetches():
    with patch("litmap.sources._http_get", return_value=(b"<p>tiny</p>", "text/html")):
        assert text_for_item(_item(url="https://example.org/"), fetch_missing=True) == ("", "")


# ---------------------------------------------------------------------------
# Zotero layer: HTML snapshot and URL resolved onto Item
# ---------------------------------------------------------------------------

@pytest.fixture
def zotero_db_with_snapshot(zotero_db, tmp_path):
    """Extend the shared fixture: item 1 gets an HTML snapshot, item 2 a URL."""
    storage = tmp_path / "storage" / "SNAP0001"
    storage.mkdir(parents=True)
    (storage / "clip.html").write_text(f"<html><body><article>{LONG_PROSE}</article></body></html>")

    conn = sqlite3.connect(zotero_db)
    conn.executescript("""
        INSERT INTO fields VALUES (13, 'url');
        INSERT INTO items VALUES (10, 14, 1, 'SNAP0001');
        INSERT INTO itemAttachments VALUES (10, 1, 1, 'text/html', 'storage:clip.html');
        -- a linked (non-stored) HTML attachment must be ignored
        INSERT INTO items VALUES (11, 14, 1, 'LINK0001');
        INSERT INTO itemAttachments VALUES (11, 2, 3, 'text/html', NULL);
        INSERT INTO itemDataValues VALUES (401, 'https://github.com/geohot/qira');
        INSERT INTO itemData VALUES (2, 13, 401);
    """)
    conn.commit()
    conn.close()
    return zotero_db


def test_get_collection_resolves_html_snapshot_and_url(zotero_db_with_snapshot):
    items = {i.key: i for i in get_collection("My Papers", zotero_db_with_snapshot)}
    a, b = items["AAAA0001"], items["AAAA0002"]
    assert a.pdf_path is None
    assert a.html_path is not None and a.html_path.name == "clip.html"
    assert a.fulltext_path == a.html_path
    assert b.html_path is None            # linked attachment, nothing on disk
    assert b.url == "https://github.com/geohot/qira"


def test_get_item_without_url_field_in_schema_still_works(zotero_db):
    # the base fixture has no 'url' field row at all
    item = get_item("AAAA0001", zotero_db)
    assert item is not None and item.url == ""


# ---------------------------------------------------------------------------
# sync_fulltext end to end (model mocked)
# ---------------------------------------------------------------------------

def _fake_tokenizer():
    tok = MagicMock()
    tok.encode.side_effect = lambda text, add_special_tokens=False: list(range(len(text.split())))
    tok.decode.side_effect = lambda ids, skip_special_tokens=True: " ".join("w" for _ in ids)
    return tok


def _fake_model():
    model = MagicMock()
    model.encode.return_value = np.ones((1, DIMS), dtype=np.float32)
    return model


def _sources(db_path):
    conn = sqlite3.connect(db_path)
    rows = dict(conn.execute("SELECT zotero_key, source FROM fulltext_embeddings").fetchall())
    conn.close()
    return rows


def test_sync_fulltext_embeds_html_snapshot_without_network(zotero_db_with_snapshot, tmp_path):
    db_path = tmp_path / "emb.db"
    init_db(db_path)
    with patch("litmap.embedder._get_model", return_value=_fake_model()), \
         patch("litmap.embedder._get_tokenizer", return_value=_fake_tokenizer()), \
         patch("litmap.sources._http_get", side_effect=AssertionError("network used")):
        n, skipped = sync_fulltext(db_path, zotero_db_with_snapshot, collection="My Papers")
    assert n == 1
    assert skipped == 1                      # the URL-only item
    assert _sources(db_path) == {"AAAA0001": "html"}


def test_sync_fulltext_fetch_missing_covers_url_only_items(zotero_db_with_snapshot, tmp_path):
    db_path = tmp_path / "emb.db"
    init_db(db_path)
    with patch("litmap.embedder._get_model", return_value=_fake_model()), \
         patch("litmap.embedder._get_tokenizer", return_value=_fake_tokenizer()), \
         patch("litmap.sources.fetch_github_readme", return_value=LONG_PROSE):
        n, skipped = sync_fulltext(db_path, zotero_db_with_snapshot, collection="My Papers", fetch_missing=True)
    assert n == 2 and skipped == 0
    assert _sources(db_path) == {"AAAA0001": "html", "AAAA0002": "github"}


def test_sync_fulltext_skips_already_embedded(zotero_db_with_snapshot, tmp_path):
    db_path = tmp_path / "emb.db"
    init_db(db_path)
    with patch("litmap.embedder._get_model", return_value=_fake_model()), \
         patch("litmap.embedder._get_tokenizer", return_value=_fake_tokenizer()):
        sync_fulltext(db_path, zotero_db_with_snapshot, collection="My Papers")
        n, _ = sync_fulltext(db_path, zotero_db_with_snapshot, collection="My Papers")
    assert n == 0


def test_init_db_adds_source_column_to_legacy_db(embeddings_db):
    conn = sqlite3.connect(embeddings_db)
    before = {r[1] for r in conn.execute("PRAGMA table_info(fulltext_embeddings)")}
    conn.close()
    assert "source" not in before
    init_db(embeddings_db)
    conn = sqlite3.connect(embeddings_db)
    after = {r[1] for r in conn.execute("PRAGMA table_info(fulltext_embeddings)")}
    conn.close()
    assert "source" in after
