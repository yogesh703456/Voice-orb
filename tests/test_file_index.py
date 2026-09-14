from indexer.file_index import FileIndex


def test_index_finds_folder(tmp_path):
    project = tmp_path / "Rock Paper Scissor"
    project.mkdir()

    index = FileIndex(
        db_path=str(tmp_path / "test_index.sqlite3"),
        roots=[str(tmp_path)],
    )

    index.build()

    results = index.search(
        "rock paper scissor",
        folders_only=True,
    )

    assert ("Rock Paper Scissor", str(project)) in results


def test_index_ignores_wrong_folder(tmp_path):
    (tmp_path / "orb").mkdir()
    target = tmp_path / "Rock Paper Scissor"
    target.mkdir()

    index = FileIndex(
        db_path=str(tmp_path / "test_index.sqlite3"),
        roots=[str(tmp_path)],
    )

    index.build()

    results = index.search(
        "rock paper scissor",
        folders_only=True,
    )

    assert ("Rock Paper Scissor", str(target)) in results
    assert ("orb", str(tmp_path / "orb")) not in results