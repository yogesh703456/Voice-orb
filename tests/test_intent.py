from core.intent import IntentType, parse


def test_open_app_command():
    intent = parse("open chrome")

    assert intent.type == IntentType.LAUNCH_APP
    assert intent.query == "chrome"


def test_search_command():
    intent = parse("search for resume")

    assert intent.type == IntentType.SEARCH_FILES
    assert intent.query == "resume"


def test_delete_command_requires_delete_intent():
    intent = parse("delete my file")

    assert intent.type == IntentType.DELETE_ITEM


def test_unknown_command():
    intent = parse("make me breakfast")

    assert intent.type == IntentType.UNKNOWN