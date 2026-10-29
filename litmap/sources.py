"""Full-text sources for embedding: local PDFs, local HTML snapshots, and
(optionally) remote fetches for items that have only a URL.

Every public function here returns plain text, or "" when nothing usable
could be extracted. Callers decide what to do with empty results.
"""
from __future__ import annotations

import re
import urllib.request
import urllib.error
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

from litmap.zotero import Item

# Elements whose text content is never prose we want to embed.
_SKIP_TAGS = frozenset({
    "script", "style", "noscript", "template", "svg", "head", "iframe",
    "nav", "header", "footer", "aside", "form", "button",
})
# Block-level elements: emit a newline so words from adjacent blocks don't fuse.
_BLOCK_TAGS = frozenset({
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "td", "th", "table", "section", "article", "main", "pre",
    "blockquote", "figcaption", "dd", "dt", "hr",
})
# If the page has one of these, its text is a much better signal than the
# whole document (which is mostly chrome on modern sites).
_CONTENT_TAGS = ("article", "main")

_MIN_WORDS = 40                    # below this, treat extraction as failed
_FETCH_TIMEOUT = 20                # seconds
_FETCH_MAX_BYTES = 20 * 1024 * 1024
_USER_AGENT = "litmap/0.1 (+https://github.com/ElliotRoe/lit-lake; local literature indexer)"

_GITHUB_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?(?:[/#?].*)?$"
)


class _TextExtractor(HTMLParser):
    """Collect visible text; prefer <article>/<main> content when present."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._all: list[str] = []
        self._content: list[str] = []
        self._content_depth = 0      # >0 while inside <article>/<main>
        self.saw_content_tag = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag in _CONTENT_TAGS:
            self._content_depth += 1
            self.saw_content_tag = True
        if tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in _CONTENT_TAGS and self._content_depth > 0:
            self._content_depth -= 1
        if tag in _BLOCK_TAGS:
            self._emit("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._emit(data)

    def _emit(self, s: str) -> None:
        self._all.append(s)
        if self._content_depth > 0:
            self._content.append(s)

    def text(self) -> str:
        parts = self._content if self.saw_content_tag and self._content else self._all
        return _normalise("".join(parts))


def _normalise(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_html_text(html: str) -> str:
    """Visible text of an HTML document, with scripts/styles/chrome removed."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return ""
    text = parser.text()
    # Prefer <article>/<main>, but fall back to the whole page if that region
    # turned out to be a stub (e.g. a teaser wrapper around a JS-rendered body).
    if parser.saw_content_tag and len(text.split()) < _MIN_WORDS:
        text = _normalise("".join(parser._all))
    return text


def extract_html_file_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    # Zotero snapshots are UTF-8 in practice; be lenient about stray bytes.
    return extract_html_text(raw.decode("utf-8", errors="replace"))


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF using PyMuPDF. Returns empty string on failure."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Remote fallbacks (opt-in: only used when sync_fulltext(fetch_missing=True))
# ---------------------------------------------------------------------------

def github_repo_from_url(url: str) -> Optional[tuple[str, str]]:
    """('owner', 'repo') for a github.com repository URL, else None.

    User/organisation pages (github.com/<user>) and non-GitHub hosts return None.
    """
    m = _GITHUB_RE.match(url or "")
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    if owner.lower() in {"orgs", "topics", "search", "settings", "marketplace", "features"}:
        return None
    return owner, repo


def _http_get(url: str, headers: Optional[dict] = None) -> tuple[bytes, str]:
    """GET url -> (body, content_type). Raises on any failure."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
        body = resp.read(_FETCH_MAX_BYTES + 1)
        if len(body) > _FETCH_MAX_BYTES:
            raise ValueError("response too large")
        return body, resp.headers.get("Content-Type", "")


def fetch_github_readme(owner: str, repo: str) -> str:
    """README of the repository's default branch as plain text, or ""."""
    # The API resolves the default branch and README filename for us, so we
    # don't have to guess main/master or README.md vs README.rst.
    try:
        body, _ = _http_get(
            f"https://api.github.com/repos/{owner}/{repo}/readme",
            headers={"Accept": "application/vnd.github.raw+json"},
        )
    except Exception:
        return ""
    return body.decode("utf-8", errors="replace").strip()


def fetch_url_text(url: str) -> str:
    """Visible text of a web page (or PDF) at url, or "" on failure."""
    if not url.lower().startswith(("http://", "https://")):
        return ""
    try:
        body, ctype = _http_get(url)
    except Exception:
        return ""
    ctype = ctype.lower()
    if "pdf" in ctype or body[:5] == b"%PDF-":
        try:
            import fitz
            doc = fitz.open(stream=body, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception:
            return ""
    return extract_html_text(body.decode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def has_local_fulltext(item: Item) -> bool:
    return item.pdf_path is not None or item.html_path is not None


def can_fetch(item: Item) -> bool:
    return bool(item.url) and item.url.lower().startswith(("http://", "https://"))


def text_for_item(item: Item, fetch_missing: bool = False) -> tuple[str, str]:
    """Best available full text for an item.

    Returns (text, source) where source is one of 'pdf', 'html', 'github',
    'url', or '' when nothing usable was found. Local files always win over
    network fetches; the network is only touched when fetch_missing is True
    and the item has no local full text.
    """
    if item.pdf_path is not None:
        text = extract_pdf_text(item.pdf_path)
        if _usable(text):
            return text, "pdf"
    if item.html_path is not None:
        text = extract_html_file_text(item.html_path)
        if _usable(text):
            return text, "html"
    if fetch_missing and can_fetch(item):
        gh = github_repo_from_url(item.url)
        if gh:
            text = fetch_github_readme(*gh)
            if _usable(text):
                return text, "github"
        text = fetch_url_text(item.url)
        if _usable(text):
            return text, "url"
    return "", ""


def _usable(text: str) -> bool:
    return len(text.split()) >= _MIN_WORDS
