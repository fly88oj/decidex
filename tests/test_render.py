"""Tests for state/instruction rendering and the evaluation pipeline."""


from decidex.render import estimate_tokens, render_state, render_text


class TestRenderText:
    def test_string_passthrough(self):
        assert render_text("Does this convey urgency?") == "Does this convey urgency?"

    def test_object_renders_keyed_lines(self):
        out = render_text({"a": 1, "b": "two"})
        assert out == "a: 1\nb: two"

    def test_object_with_structured_values_inline(self):
        out = render_text({"question": "Is this the same person?", "record": {"name": "John"}})
        assert "question: Is this the same person?" in out
        assert 'record: {"name":"John"}' in out

    def test_array_renders_bullets(self):
        out = render_text(["first", "second"])
        assert out == "- first\n- second"

    def test_backtick_paths_preserved(self):
        instructions = "Does `ticket.messages[0].text` request a refund?"
        assert "`ticket.messages[0].text`" in render_text(instructions)


class TestRenderState:
    def test_string_state_passthrough(self):
        assert render_state("plain text") == "plain text"

    def test_structured_state_json(self):
        out = render_state({"order": {"id": "A-104"}})
        assert '"id": "A-104"' in out

    def test_unicode_preserved(self):
        assert "中文" in render_state({"msg": "中文测试"})


class TestEstimateTokens:
    def test_positive_and_nonzero(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("abcde") == 2
