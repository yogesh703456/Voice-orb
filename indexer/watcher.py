<<<<<<< HEAD
"""
indexer/watcher.py

Keeps FileIndex current after the initial build, using `watchdog` to react
to filesystem events in the configured roots instead of ever re-scanning
from scratch.
"""
=======
from __future__ import annotations

from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
>>>>>>> 45763e9 (Initial commit)

from indexer.file_index import FileIndex


<<<<<<< HEAD
class IndexWatcher:
    def __init__(self, index: FileIndex, roots: list[str]) -> None:
        self.index = index
        self.roots = roots

    def start(self) -> None:
        """
        TODO (implementation milestone, days 5-6, alongside file_index.py):
        - from watchdog.observers import Observer
        - from watchdog.events import FileSystemEventHandler
        - Handler maps on_created/on_moved -> index.upsert(path),
          on_deleted -> index.remove(path)
        - Schedule an Observer per root, run in a background thread so it
          never blocks the main pipeline
        """
        raise NotImplementedError("Watcher wiring lands in days 5-6 milestone")
=======
class _IndexEventHandler(FileSystemEventHandler):
    def __init__(self, index: FileIndex) -> None:
        self.index = index

    def on_created(self, event) -> None:
        self.index.upsert(Path(event.src_path))

    def on_modified(self, event) -> None:
        self.index.upsert(Path(event.src_path))

    def on_deleted(self, event) -> None:
        self.index.remove(Path(event.src_path))

    def on_moved(self, event) -> None:
        self.index.remove(Path(event.src_path))
        self.index.upsert(Path(event.dest_path))


class IndexWatcher:
    def __init__(self, index: FileIndex, roots: list[str]) -> None:
        self.index = index
        self.roots = [Path(root).expanduser() for root in roots]
        self.observer = Observer()

    def start(self) -> None:
        handler = _IndexEventHandler(self.index)
        watched = 0

        for root in self.roots:
            if root.exists() and root.is_dir():
                self.observer.schedule(handler, str(root), recursive=True)
                watched += 1

        if not watched:
            print("[INDEX] No valid folders available to watch.")
            return

        self.observer.start()
        print(f"[INDEX] Watching {watched} folder(s) for changes.")

    def stop(self) -> None:
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join(timeout=5)
>>>>>>> 45763e9 (Initial commit)
