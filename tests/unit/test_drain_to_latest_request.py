"""
Tests for JarvisBot._drain_to_latest_request() - see _speaker_loop.
Several utterances can queue up while a previous turn is still working
(e.g. during generate_teaching_plan's long "thinking" wait); this makes
sure only the most recent one gets acted on instead of each queued
utterance getting its own full response in order.

Regression test for direct feedback: an interruption should mean "stop
and work on the latest part," not "queue it and get to it eventually."

Only touches self._request_queue, so JarvisBot.__new__ + manually
setting that one attribute is enough - no need to construct a real
JarvisBot (which needs live Azure credentials, PyAudio devices, etc.),
same pattern used for other single-attribute pure-logic tests in this
suite (e.g. TextToSpeechClient.__new__ in test_tts_speech_rate.py).
"""

import queue

import pytest

from main import JarvisBot


def make_bot(*queued) -> JarvisBot:
    bot = JarvisBot.__new__(JarvisBot)
    bot._request_queue = queue.Queue()
    for item in queued:
        bot._request_queue.put(item)
    return bot


class TestDrainToLatestRequest:
    def test_returns_first_when_nothing_else_queued(self):
        bot = make_bot()
        assert bot._drain_to_latest_request("hello") == "hello"

    def test_returns_the_latest_queued_utterance(self):
        bot = make_bot("second", "third")
        assert bot._drain_to_latest_request("first") == "third"

    def test_drains_the_queue_completely(self):
        bot = make_bot("second", "third")
        bot._drain_to_latest_request("first")
        assert bot._request_queue.empty()

    def test_a_single_newer_item_is_returned(self):
        bot = make_bot("second")
        assert bot._drain_to_latest_request("first") == "second"

    def test_a_termination_sentinel_is_put_back_not_consumed(self):
        bot = make_bot(None)
        result = bot._drain_to_latest_request("first")
        # The real utterance is still returned/acted on this round...
        assert result == "first"
        # ...but the sentinel must survive for the caller's own
        # termination check on its next loop iteration.
        assert bot._request_queue.get_nowait() is None

    def test_newer_utterances_before_a_sentinel_are_still_taken(self):
        bot = make_bot("second", None)
        result = bot._drain_to_latest_request("first")
        assert result == "second"
        assert bot._request_queue.get_nowait() is None

    def test_a_sentinel_stops_the_drain_immediately(self):
        # Nothing queued after the sentinel should be consumed by this
        # call - it belongs to whatever happens after shutdown, not this
        # drain. Queue.put() appends to the END, not back to its
        # original position, so the untouched item stays queued ahead of
        # the re-added sentinel - still fully preserved, just not in its
        # original relative order, which is an acceptable tradeoff for
        # this rare edge case (multiple utterances AND a shutdown signal
        # queued at once).
        bot = make_bot("second", None, "should_not_be_reached")
        bot._drain_to_latest_request("first")
        assert bot._request_queue.get_nowait() == "should_not_be_reached"
        assert bot._request_queue.get_nowait() is None
