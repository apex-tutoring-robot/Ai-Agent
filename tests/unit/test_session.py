"""Tests for session.Session - the per-conversation bookkeeping dataclass."""

from session import Session


class TestSessionDefaults:
    def test_generates_unique_session_ids(self):
        s1 = Session()
        s2 = Session()
        assert s1.session_id != s2.session_id

    def test_starts_with_zero_turns_and_empty_tool_calls(self):
        s = Session()
        assert s.turn_count == 0
        assert s.tool_calls == []
        assert s.active_task is None


class TestNewTurn:
    def test_increments_turn_count(self):
        s = Session()
        s.new_turn()
        s.new_turn()
        assert s.turn_count == 2


class TestRecordToolCall:
    def test_appends_to_tool_calls_with_expected_shape(self):
        s = Session()
        s.record_tool_call("search_curriculum", {"query": "fractions"}, "some result text")

        assert len(s.tool_calls) == 1
        call = s.tool_calls[0]
        assert call["tool"] == "search_curriculum"
        assert call["arguments"] == {"query": "fractions"}
        assert call["result"] == "some result text"
        assert "at" in call

    def test_records_multiple_calls_in_order(self):
        s = Session()
        s.record_tool_call("search_curriculum", {"query": "a"}, "r1")
        s.record_tool_call("set_volume", {"level": "louder"}, "r2")

        assert [c["tool"] for c in s.tool_calls] == ["search_curriculum", "set_volume"]
