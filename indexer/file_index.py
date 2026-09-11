"""
indexer/file_index.py

A pre-built SQLite index of (filename, path, last_modified) so voice search
is a fast lookup, not a live disk scan. This is the piece that directly
prevents the "takes 5 minutes" problem you flagged — never os.walk the
whole disk at query time.
"""

from pathlib import Path


class FileIndex:
    def __init__(self, db_path: str = "file_index.db", roots: list[str] | None = None) -> None:
        self.db_path = db_path
        self.roots = roots or []

    def build(self) -> None:
        """
        TODO (implementation milestone, days 5-6):
        - Create SQLite table (filename TEXT, path TEXT, mtime REAL) with
          an index on filename
        - Walk each root in self.roots ONCE (this is the only allowed
          full walk, and it's scoped to configured roots, not the whole
          drive) and populate the table
        - Called once at startup; kept fresh afterward by watcher.py
        """
        raise NotImplementedError("Index build lands in days 5-6 milestone")

    def search(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        """
        TODO:
        - Pull candidate filenames from SQLite (cheap prefilter, e.g. LIKE
          on a normalized substring) then rank with rapidfuzz for the
          actual fuzzy score
        - Return [(filename, full_path), ...] best matches first
        - This must stay fast (milliseconds) — it's on the hot path for
          every "open X" command
        """
        raise NotImplementedError("Fuzzy search lands in days 5-6 milestone")

    def upsert(self, path: Path) -> None:
        """Called by watcher.py on file create/rename events to keep the
        index current without a full rebuild."""
        raise NotImplementedError("Incremental update lands alongside watcher.py")

    def remove(self, path: Path) -> None:
        """Called by watcher.py on file delete events."""
        raise NotImplementedError("Incremental update lands alongside watcher.py")
