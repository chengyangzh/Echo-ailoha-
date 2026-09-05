from agent_harness.cli import _read_multiline


def test_multiline_input_becomes_one_message(monkeypatch):
    lines = iter(["first paragraph", "", "second paragraph", "/end"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    assert _read_multiline() == "first paragraph\n\nsecond paragraph"
