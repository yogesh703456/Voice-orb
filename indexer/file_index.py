from __future__ import annotations

import contextlib
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    from rapidfuzz import fuzz, process as fuzzy_process
except ImportError:
    fuzz = None
    fuzzy_process = None


SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "AppData"}


class FileIndex:
    def __init__(
        self,
        db_path: str = "file_index.db",
        roots: list[str] | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.roots = [Path(root).expanduser() for root in (roots or [])]
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextlib.contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """A connection that is ALWAYS closed after the block, not just
        committed. `with connection:` alone commits but leaves the
        connection open, relying on garbage collection to release it --
        on Windows that keeps a file lock on the SQLite database (seen as
        WinError 32 when anything tries to clean up the directory) and
        pins memory for the life of the process."""
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _create_tables(self) -> None:
        with self._session() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_paths (
                    path TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    is_folder INTEGER NOT NULL,
                    modified_time REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_indexed_paths_name
                ON indexed_paths(name COLLATE NOCASE)
                """
            )

    def build(self) -> None:
        """Rebuild the index from configured roots."""
        with self._session() as connection:
            connection.execute("DELETE FROM indexed_paths")

            for root in self.roots:
                if not root.exists() or not root.is_dir():
                    print(f"[INDEX] Skipping missing root: {root}")
                    continue

                print(f"[INDEX] Scanning: {root}")

                for directory, directory_names, file_names in os.walk(
                    root,
                    followlinks=False,
                ):
                    directory_names[:] = [
                        name for name in directory_names
                        if name not in SKIP_DIRS
                    ]

                    for name in directory_names + file_names:
                        path = Path(directory) / name
                        try:
                            modified_time = path.stat().st_mtime
                        except OSError:
                            continue

                        connection.execute(
                            """
                            INSERT OR REPLACE INTO indexed_paths
                            (path, name, is_folder, modified_time)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                str(path),
                                path.name,
                                int(path.is_dir()),
                                modified_time,
                            ),
                        )

        print("[INDEX] Build complete.")

    def search(
        self,
        query: str,
        limit: int = 5,
        folders_only: bool | None = None,
    ) -> list[tuple[str, str]]:
        """
        Return [(name, full_path)].

        folders_only:
          True  = folders only
          False = files only
          None  = both
        """
        cleaned = query.strip().casefold()
        if len(cleaned) < 3:
            return []

        where = ["name COLLATE NOCASE LIKE ?"]
        values: list[object] = [f"%{cleaned}%"]

        if folders_only is not None:
            where.append("is_folder = ?")
            values.append(int(folders_only))

        sql = f"""
            SELECT name, path
            FROM indexed_paths
            WHERE {" AND ".join(where)}
            ORDER BY
                CASE WHEN lower(name) = ? THEN 0 ELSE 1 END,
                length(name),
                name COLLATE NOCASE
            LIMIT ?
        """

        with self._session() as connection:
            rows = connection.execute(
                sql,
                [*values, cleaned, limit],
            ).fetchall()

            exact_results = [
                (str(row["name"]), str(row["path"]))
                for row in rows
            ]

            if exact_results:
                return exact_results

            # No substring match: use only high-confidence fuzzy suggestions.
            if fuzzy_process is None or fuzz is None:
                return []

            type_clause = ""
            type_values: list[object] = []

            if folders_only is not None:
                type_clause = "WHERE is_folder = ?"
                type_values.append(int(folders_only))

            candidates = connection.execute(
                f"""
                SELECT name, path
                FROM indexed_paths
                {type_clause}
                LIMIT 10000
                """,
                type_values,
            ).fetchall()

        names = [str(row["name"]).casefold() for row in candidates]

        matches = fuzzy_process.extract(
            cleaned,
            names,
            scorer=fuzz.WRatio,
            score_cutoff=90,
            limit=limit,
        )

        return [
            (str(candidates[index]["name"]), str(candidates[index]["path"]))
            for _, _, index in matches
        ]

    def upsert(self, path: Path) -> None:
        """Add or update one file/folder in the index."""
        path = path.expanduser()

        if not path.exists():
            self.remove(path)
            return

        try:
            modified_time = path.stat().st_mtime
        except OSError:
            return

        with self._session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO indexed_paths
                (path, name, is_folder, modified_time)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(path),
                    path.name,
                    int(path.is_dir()),
                    modified_time,
                ),
            )

    def remove(self, path: Path) -> None:
        """Remove a deleted file/folder from the index."""
        with self._session() as connection:
            connection.execute(
                "DELETE FROM indexed_paths WHERE path = ?",
                (str(path.expanduser()),),
            )
