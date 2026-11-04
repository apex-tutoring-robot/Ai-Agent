"""
Tests for the two places a startup/restart failure in
WakeWordDetector.start() gets reported to the screen instead of just a
log line nobody at a remote deployment can read - see
ui_signals.fatal_error and error_banner.py's own module docstring for
the real incident (a silent startup crash) this exists to make visible:

  - JarvisBot.run()'s outer except - the very first start() call.
  - JarvisBot._run_wake_word_detector_catching_errors - the thread
    target _restart_wake_word spawns for every start() call after that.

Uses JarvisBot.__new__(JarvisBot) + manually setting only the attributes
each method reads/writes, same pattern as test_system_command.py -
__init__ needs live Azure credentials, PyAudio devices, etc.
"""

from unittest.mock import Mock

from main import JarvisBot


def make_bot_for_run(start_side_effect) -> JarvisBot:
    bot = JarvisBot.__new__(JarvisBot)
    bot._is_running = False
    bot.ui_signals = Mock()
    bot.wake_word_detector = Mock()
    bot.wake_word_detector.start.side_effect = start_side_effect
    bot._handle_wake_word = Mock()
    bot._speak_boot_greeting = Mock()
    bot.stop = Mock()
    return bot


class TestRunEmitsFatalErrorOnStartupFailure:
    def test_wake_word_start_failure_emits_fatal_error(self):
        bot = make_bot_for_run(RuntimeError("mic stream failed"))
        bot.run()
        bot.ui_signals.fatal_error.emit.assert_called_once()
        message = bot.ui_signals.fatal_error.emit.call_args[0][0]
        assert "mic stream failed" in message

    def test_stop_is_still_called_on_startup_failure(self):
        bot = make_bot_for_run(RuntimeError("boom"))
        bot.run()
        bot.stop.assert_called_once()

    def test_no_ui_signals_does_not_crash(self):
        bot = make_bot_for_run(RuntimeError("boom"))
        bot.ui_signals = None
        bot.run()  # must not raise
        bot.stop.assert_called_once()

    def test_keyboard_interrupt_does_not_emit_fatal_error(self):
        # A deliberate Ctrl+C shutdown isn't a fatal/silent failure -
        # it's expected, visible (whoever pressed it is right there),
        # and already has its own log line.
        bot = make_bot_for_run(KeyboardInterrupt())
        bot.run()
        bot.ui_signals.fatal_error.emit.assert_not_called()


class TestRestartWakeWordDetectorCatchesErrors:
    def _make_bot(self, start_side_effect=None) -> JarvisBot:
        bot = JarvisBot.__new__(JarvisBot)
        bot.wake_word_detector = Mock()
        bot.wake_word_detector.start.side_effect = start_side_effect
        bot._handle_wake_word = Mock()
        bot.ui_signals = Mock()
        return bot

    def test_start_failure_is_caught_and_emits_fatal_error(self):
        bot = self._make_bot(RuntimeError("model load failed"))
        bot._run_wake_word_detector_catching_errors()  # must not raise
        bot.ui_signals.fatal_error.emit.assert_called_once()
        message = bot.ui_signals.fatal_error.emit.call_args[0][0]
        assert "model load failed" in message

    def test_no_ui_signals_does_not_crash(self):
        bot = self._make_bot(RuntimeError("boom"))
        bot.ui_signals = None
        bot._run_wake_word_detector_catching_errors()  # must not raise

    def test_success_does_not_emit_anything(self):
        bot = self._make_bot(start_side_effect=None)  # start() returns normally
        bot._run_wake_word_detector_catching_errors()
        bot.ui_signals.fatal_error.emit.assert_not_called()


class TestRestartWakeWordUsesTheErrorCatchingWrapper:
    def test_thread_target_is_the_wrapper_not_start_directly(self, monkeypatch):
        bot = JarvisBot.__new__(JarvisBot)
        bot._is_running = True
        bot.wake_word_detector = Mock()
        bot.wake_word_detector.is_running.return_value = False

        captured = {}

        class FakeThread:
            def __init__(self, target=None, daemon=None):
                captured["target"] = target

            def start(self):
                pass

        monkeypatch.setattr("main.threading.Thread", FakeThread)

        bot._restart_wake_word()

        # If this were wake_word_detector.start directly (the old
        # behavior), a failure there would kill the daemon thread
        # silently - no log line readable on a remote device, no signal
        # ever emitted. Routing through the wrapper is what makes
        # TestRestartWakeWordDetectorCatchesErrors's guarantees apply.
        assert captured["target"] == bot._run_wake_word_detector_catching_errors
