"""
Tests for JarvisBot._should_defer_idle_check() - see _handle_wake_word's
idle/check-in loop and _listener_loop's barge-in handling in main.py.

Root-cause fix for a Discord bug report: "Are you still there?" firing
on essentially every interruption. _speaker_busy alone only protects a
turn that's actively being processed, and clears the instant playback
stops on a barge-in - before the student's follow-up utterance (the
whole reason playback stopped) has even finished being transcribed.
_should_defer_idle_check also honors a time.time() deadline
(_interruption_recovery_deadline) that _listener_loop opens the moment
it confirms a barge-in, giving that follow-up the same idle_timeout
grace a normal finished turn gets instead of relying on frame-by-frame
VAD alone to keep last_speech_time fresh through STT's own
finalization latency.

Only touches _speaker_busy/_interruption_recovery_deadline, so
JarvisBot.__new__ + manually setting those two attributes is enough -
same pattern as test_drain_to_latest_request.py.
"""

import threading
import time

from main import JarvisBot


def make_bot(speaker_busy: bool, deadline: float) -> JarvisBot:
    bot = JarvisBot.__new__(JarvisBot)
    bot._speaker_busy = threading.Event()
    if speaker_busy:
        bot._speaker_busy.set()
    bot._interruption_recovery_deadline = deadline
    return bot


class TestShouldDeferIdleCheck:
    def test_false_when_idle_and_no_recent_interruption(self):
        bot = make_bot(speaker_busy=False, deadline=0.0)
        assert bot._should_defer_idle_check() is False

    def test_true_while_speaker_busy_regardless_of_deadline(self):
        bot = make_bot(speaker_busy=True, deadline=0.0)
        assert bot._should_defer_idle_check() is True

    def test_true_while_within_the_interruption_recovery_window(self):
        bot = make_bot(speaker_busy=False, deadline=time.time() + 5.0)
        assert bot._should_defer_idle_check() is True

    def test_false_once_the_recovery_deadline_has_passed(self):
        bot = make_bot(speaker_busy=False, deadline=time.time() - 0.1)
        assert bot._should_defer_idle_check() is False

    def test_default_zero_deadline_never_defers_on_its_own(self):
        # 0.0 is the __init__ default before any interruption has ever
        # happened - must not accidentally satisfy `now < deadline`.
        bot = make_bot(speaker_busy=False, deadline=0.0)
        assert bot._should_defer_idle_check() is False
