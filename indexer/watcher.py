"""
indexer/watcher.py

Keeps FileIndex current after the initial build, using `watchdog` to react
to filesystem events in the configured roots instead of ever re-scanning
from scratch.
"""

from indexer.file_index import FileIndex


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
