from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from typing import Optional

ZOTERO_DB = Path.home() / "Zotero" / "zotero.sqlite"
# Item types that never represent a citable paper. Their numeric itemTypeIDs
# are NOT stable across Zotero versions (e.g. attachment is 14 in Zotero 6/7
# but 3 in Zotero 9), so they are resolved by name at query time.
EXCLUDED_TYPE_NAMES = ("attachment", "note", "annotation")
# Last-resort fallback if the itemTypes table cannot be read.
_FALLBACK_EXCLUDED_TYPES = (14, 26)


@dataclass
class Item:
    key: str
    title: str
    abstract: str
    authors: list[str]
    year: str
    doi: str
    keywords: str = ""
    pdf_path: Optional[Path] = None
    # Local HTML snapshot (Zotero "Save to Zotero" web clip), used for
    # full-text embedding when no PDF is attached.
    html_path: Optional[Path] = None
    url: str = ""

    @property
    def fulltext_path(self) -> Optional[Path]:
        """Best local full-text source: PDF if present, else HTML snapshot."""
        return self.pdf_path or self.html_path


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _field_ids(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        "SELECT fieldName, fieldID FROM fields "
        "WHERE fieldName IN ('title','abstractNote','date','DOI','keywords','url')"
    ).fetchall()
    return {r["fieldName"]: r["fieldID"] for r in rows}


def _excluded_types_sql(conn: sqlite3.Connection) -> str:
    """SQL tuple literal of itemTypeIDs for attachments/notes/annotations."""
    try:
        placeholders = ",".join("?" * len(EXCLUDED_TYPE_NAMES))
        rows = conn.execute(
            f"SELECT itemTypeID FROM itemTypes WHERE typeName IN ({placeholders})",
            EXCLUDED_TYPE_NAMES,
        ).fetchall()
        ids = tuple(int(r[0]) for r in rows)
    except sqlite3.Error:
        ids = ()
    if not ids:
        ids = _FALLBACK_EXCLUDED_TYPES
    return "(" + ",".join(str(t) for t in ids) + ")"


def _trash_filter_sql(conn: sqlite3.Connection) -> str:
    """Exclude items sitting in Zotero's trash (table may be absent in old DBs)."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='deletedItems'"
    ).fetchone()
    return "AND i.itemID NOT IN (SELECT itemID FROM deletedItems)" if row else ""


def _item_select(conn: sqlite3.Connection) -> str:
    return _ITEM_SELECT.format(
        excluded=_excluded_types_sql(conn),
        trash_filter=_trash_filter_sql(conn),
    )


def _author_type_id(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT creatorTypeID FROM creatorTypes WHERE creatorType = 'author'"
    ).fetchone()
    return row["creatorTypeID"] if row else 1


def _resolve_storage_path(raw_path: str, att_key: str, zotero_base: Optional[Path]) -> Optional[Path]:
    """Map a Zotero `storage:<filename>` attachment path to a file on disk.

    Zotero stores attachments in storage/<8-char-key>/<filename>. Returns None
    for linked files, missing files, or when no base directory is known.
    """
    if not raw_path or zotero_base is None or not raw_path.startswith("storage:"):
        return None
    filename = raw_path[len("storage:"):]
    candidate = zotero_base / "storage" / (att_key or "") / filename
    return candidate if candidate.exists() else None


def _rows_to_items(rows, zotero_base: Optional[Path] = None) -> list[Item]:
    items = []
    for r in rows:
        d = dict(r)
        authors = [a.strip() for a in (r["authors"] or "").split(";") if a.strip()]
        pdf_path = _resolve_storage_path(d.get("pdf_path") or "", d.get("att_key") or "", zotero_base)
        html_path = _resolve_storage_path(d.get("html_path") or "", d.get("html_key") or "", zotero_base)
        items.append(Item(
            key=r["key"],
            title=r["title"] or "",
            abstract=r["abstract"] or "",
            authors=authors,
            year=(r["year"] or "")[:4],
            doi=r["doi"] or "",
            keywords=r["keywords"] or "",
            pdf_path=pdf_path,
            html_path=html_path,
            url=d.get("url") or "",
        ))
    return items


_ITEM_SELECT = """
    SELECT
        i.key,
        tv.value  AS title,
        av.value  AS abstract,
        GROUP_CONCAT(c.lastName || ', ' || c.firstName, '; ' ORDER BY ic.orderIndex) AS authors,
        dv.value  AS year,
        doiv.value AS doi,
        kv.value  AS keywords,
        att.path  AS pdf_path,
        atti.key  AS att_key,
        hatt.path AS html_path,
        hatti.key AS html_key,
        uv.value  AS url
    FROM items i
    LEFT JOIN itemData    td   ON td.itemID   = i.itemID AND td.fieldID   = :title_id
    LEFT JOIN itemDataValues tv ON tv.valueID = td.valueID
    LEFT JOIN itemData    ad   ON ad.itemID   = i.itemID AND ad.fieldID   = :abs_id
    LEFT JOIN itemDataValues av ON av.valueID = ad.valueID
    LEFT JOIN itemData    dd   ON dd.itemID   = i.itemID AND dd.fieldID   = :date_id
    LEFT JOIN itemDataValues dv ON dv.valueID = dd.valueID
    LEFT JOIN itemData    doid ON doid.itemID = i.itemID AND doid.fieldID = :doi_id
    LEFT JOIN itemDataValues doiv ON doiv.valueID = doid.valueID
    LEFT JOIN itemData    kd   ON kd.itemID   = i.itemID AND kd.fieldID   = :keywords_id
    LEFT JOIN itemDataValues kv ON kv.valueID = kd.valueID
    LEFT JOIN itemData    ud   ON ud.itemID   = i.itemID AND ud.fieldID   = :url_id
    LEFT JOIN itemDataValues uv ON uv.valueID = ud.valueID
    LEFT JOIN itemCreators ic ON ic.itemID = i.itemID AND ic.creatorTypeID = :author_type
    LEFT JOIN creators c ON c.creatorID = ic.creatorID
    LEFT JOIN (
        SELECT parentItemID, MIN(itemID) AS itemID, path
        FROM itemAttachments
        WHERE contentType = 'application/pdf'
        GROUP BY parentItemID
    ) att ON att.parentItemID = i.itemID
    LEFT JOIN items atti ON atti.itemID = att.itemID
    LEFT JOIN (
        -- Web snapshots saved by the Zotero connector. Only stored files
        -- (path LIKE 'storage:%') can be read locally; linked URLs are skipped.
        SELECT parentItemID, MIN(itemID) AS itemID, path
        FROM itemAttachments
        WHERE contentType = 'text/html' AND path LIKE 'storage:%'
        GROUP BY parentItemID
    ) hatt ON hatt.parentItemID = i.itemID
    LEFT JOIN items hatti ON hatti.itemID = hatt.itemID
    WHERE i.itemTypeID NOT IN {excluded}
      {trash_filter}
      AND tv.value IS NOT NULL
"""


def get_all_items(db_path: Path = ZOTERO_DB) -> list[Item]:
    zotero_base = db_path.parent
    with _connect(db_path) as conn:
        fids = _field_ids(conn)
        atid = _author_type_id(conn)
        rows = conn.execute(
            _item_select(conn) + " GROUP BY i.itemID",
            {"title_id": fids["title"], "abs_id": fids["abstractNote"],
             "date_id": fids["date"], "doi_id": fids["DOI"],
             "keywords_id": fids.get("keywords"), "url_id": fids.get("url"),
             "author_type": atid},
        ).fetchall()
    return _rows_to_items(rows, zotero_base)


def get_collection(name: str, db_path: Path = ZOTERO_DB) -> list[Item]:
    zotero_base = db_path.parent
    with _connect(db_path) as conn:
        fids = _field_ids(conn)
        atid = _author_type_id(conn)
        rows = conn.execute(
            _item_select(conn) + """
              AND i.itemID IN (
                  SELECT ci.itemID FROM collectionItems ci
                  JOIN collections col ON col.collectionID = ci.collectionID
                  WHERE col.collectionName = :name
              )
            GROUP BY i.itemID
            """,
            {"title_id": fids["title"], "abs_id": fids["abstractNote"],
             "date_id": fids["date"], "doi_id": fids["DOI"],
             "keywords_id": fids.get("keywords"), "url_id": fids.get("url"),
             "author_type": atid, "name": name},
        ).fetchall()
    return _rows_to_items(rows, zotero_base)


def get_item(key_or_doi: str, db_path: Path = ZOTERO_DB) -> Optional[Item]:
    zotero_base = db_path.parent
    with _connect(db_path) as conn:
        fids = _field_ids(conn)
        atid = _author_type_id(conn)
        rows = conn.execute(
            _item_select(conn) + """
              AND (i.key = :val OR doiv.value = :val)
            GROUP BY i.itemID
            LIMIT 1
            """,
            {"title_id": fids["title"], "abs_id": fids["abstractNote"],
             "date_id": fids["date"], "doi_id": fids["DOI"],
             "keywords_id": fids.get("keywords"), "url_id": fids.get("url"),
             "author_type": atid, "val": key_or_doi},
        ).fetchall()
    items = _rows_to_items(rows, zotero_base)
    return items[0] if items else None


def get_subcollection_map(db_path: Path = ZOTERO_DB) -> dict[str, list[str]]:
    """Return {zotero_key: [collection_names]} for every non-attachment item.

    Lists only the collections the item directly belongs to — parents of
    a child collection are not included by inheritance.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT i.key AS key, col.collectionName AS name
            FROM items i
            JOIN collectionItems ci ON ci.itemID = i.itemID
            JOIN collections col ON col.collectionID = ci.collectionID
            WHERE i.itemTypeID NOT IN {_excluded_types_sql(conn)}
              {_trash_filter_sql(conn)}
            ORDER BY i.key, col.collectionName
            """
        ).fetchall()
    mapping: dict[str, list[str]] = {}
    for r in rows:
        mapping.setdefault(r["key"], []).append(r["name"])
    return mapping
