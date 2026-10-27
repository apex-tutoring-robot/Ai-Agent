"""
Tests for the voice-triggered shutdown/restart feature:
_SHUTDOWN_PATTERN/_RESTART_PATTERN, _try_handle_system_command,
_handle_system_command_confirmation, and _execute_system_command (see
main.py - intercepted in _speaker_loop before normal tutoring turns).

This triggers a REAL system command (x728off / sudo reboot) on
physical hardware, so the two things that matter most to get right are
covered directly: (1) the trigger phrases are narrow enough that
ordinary tutoring language ("let's restart this problem") doesn't
false-positive, and (2) nothing actually executes without a separate,
explicit "yes" confirmation.

Uses JarvisBot.__new__(JarvisBot) + manually setting only the
attributes each method reads/writes, with _speak_fixed_phrase and
_execute_system_command mocked out - same pattern as
test_tts_speech_rate.py's make_client, since __init__ needs live Azure
credentials, PyAudio devices, etc.
"""

from unittest.mock import Mock, patch

import pytest

from main import JarvisBot


def make_bot(pending_system_command=None) -> JarvisBot:
    bot = JarvisBot.__new__(JarvisBot)
    bot._pending_system_command = pending_system_command
    bot._speak_fixed_phrase = Mock()
    return bot


class TestShutdownPattern:
    @pytest.mark.parametrize("text", [
        "shut down",
        "Shut Down please",
        "shutdown",
        "power off",
        "please power off",
        "turn off yourself",
        "turn off your self",
        "turn off jarvis",
        "turn off the robot",
    ])
    def test_matches_clear_shutdown_phrases(self, text):
        assert JarvisBot._SHUTDOWN_PATTERN.search(text)

    @pytest.mark.parametrize("text", [
        "turn off the lights",
        "can you turn off the music",
        "the light switch is off",
        "I powered through my homework",
    ])
    def test_does_not_match_unrelated_turn_off_phrases(self, text):
        # "turn off" alone (without yourself/jarvis/the robot) is common
        # enough in an ordinary word problem or request that it must NOT
        # trigger a real shutdown on its own.
        assert not JarvisBot._SHUTDOWN_PATTERN.search(text)


class TestRestartPattern:
    @pytest.mark.parametrize("text", [
        "reboot",
        "please reboot",
        "restart yourself",
        "restart your self",
        "restart jarvis",
        "restart the robot",
        "restart the system",
    ])
    def test_matches_clear_restart_phrases(self, text):
        assert JarvisBot._RESTART_PATTERN.search(text)

    @pytest.mark.parametrize("text", [
        "let's restart this problem",
        "can we restart from the beginning",
        "I want to restart my attempt",
    ])
    def test_does_not_match_ordinary_tutoring_restart_phrases(self, text):
        # "restart" alone is ordinary tutoring language ("restart this
        # problem") far more often than it means "reboot the hardware" -
        # must require an explicit target (yourself/jarvis/the robot/the
        # system) to trigger.
        assert not JarvisBot._RESTART_PATTERN.search(text)


class TestTryHandleSystemCommand:
    def test_shutdown_phrase_sets_pending_and_asks_to_confirm(self):
        bot = make_bot()
        handled = bot._try_handle_system_command("shut down", None, 0)
        assert handled is True
        assert bot._pending_system_command == "shutdown"
        bot._speak_fixed_phrase.assert_called_once()
        assert "shut down" in bot._speak_fixed_phrase.call_args[0][0].lower()

    def test_restart_phrase_sets_pending_and_asks_to_confirm(self):
        bot = make_bot()
        handled = bot._try_handle_system_command("please reboot", None, 0)
        assert handled is True
        assert bot._pending_system_command == "restart"
        bot._speak_fixed_phrase.assert_called_once()
        assert "restart" in bot._speak_fixed_phrase.call_args[0][0].lower()

    def test_unrelated_text_is_not_handled(self):
        bot = make_bot()
        handled = bot._try_handle_system_command("what is 2 plus 2?", None, 0)
        assert handled is False
        assert bot._pending_system_command is None
        bot._speak_fixed_phrase.assert_not_called()

    def test_nothing_executes_on_the_trigger_turn_alone(self):
        # The trigger phrase only asks for confirmation - it must never
        # call the actual system command itself.
        bot = make_bot()
        with patch.object(JarvisBot, "_execute_system_command") as mock_execute:
            bot._try_handle_system_command("shut down", None, 0)
            mock_execute.assert_not_called()


class TestHandleSystemCommandConfirmation:
    def test_yes_executes_shutdown_and_clears_pending(self):
        bot = make_bot(pending_system_command="shutdown")
        with patch.object(JarvisBot, "_execute_system_command") as mock_execute:
            bot._handle_system_command_confirmation("yes", None, 0)
            mock_execute.assert_called_once_with("shutdown")
        assert bot._pending_system_command is None

    def test_yes_executes_restart_and_clears_pending(self):
        bot = make_bot(pending_system_command="restart")
        with patch.object(JarvisBot, "_execute_system_command") as mock_execute:
            bot._handle_system_command_confirmation("yeah", None, 0)
            mock_execute.assert_called_once_with("restart")
        assert bot._pending_system_command is None

    def test_no_cancels_without_executing(self):
        bot = make_bot(pending_system_command="shutdown")
        with patch.object(JarvisBot, "_execute_system_command") as mock_execute:
            bot._handle_system_command_confirmation("no", None, 0)
            mock_execute.assert_not_called()
        assert bot._pending_system_command is None
        bot._speak_fixed_phrase.assert_called_once()

    def test_unclear_answer_cancels_rather_than_executing(self):
        # Fail-safe default: an ambiguous reply must NOT be treated as
        # confirmation for an action this consequential.
        bot = make_bot(pending_system_command="shutdown")
        with patch.object(JarvisBot, "_execute_system_command") as mock_execute:
            bot._handle_system_command_confirmation("umm what", None, 0)
            mock_execute.assert_not_called()
        assert bot._pending_system_command is None


class TestExecuteSystemCommand:
    def test_shutdown_runs_x728off(self):
        bot = JarvisBot.__new__(JarvisBot)
        with patch("main.subprocess.run") as mock_run:
            bot._execute_system_command("shutdown")
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["x728off"]

    def test_restart_runs_sudo_reboot(self):
        bot = JarvisBot.__new__(JarvisBot)
        with patch("main.subprocess.run") as mock_run:
            bot._execute_system_command("restart")
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == ["sudo", "reboot"]

    def test_a_failed_command_is_logged_not_raised(self):
        bot = JarvisBot.__new__(JarvisBot)
        with patch("main.subprocess.run", side_effect=OSError("not found")):
            bot._execute_system_command("shutdown")  # must not raise
