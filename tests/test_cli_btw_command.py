"""Tests for the /btw CLI slash command."""

import sys
from unittest.mock import MagicMock, patch

from tests.test_cli_init import _make_cli


_CLI_IMPORT_STUBS = {
    "fire": MagicMock(),
    "openai": MagicMock(),
    "firecrawl": MagicMock(),
    "fal_client": MagicMock(),
}


class TestBtwCommand:
    def test_btw_requires_question(self):
        with patch.dict(sys.modules, _CLI_IMPORT_STUBS):
            cli = _make_cli()
            cli._handle_btw_command("/btw")

        assert cli._pending_input.empty()
        assert cli._btw_tasks == {}

    def test_btw_starts_ephemeral_thread_without_touching_pending_input(self):
        with patch.dict(sys.modules, _CLI_IMPORT_STUBS):
            cli = _make_cli()
            cli.conversation_history = [{"role": "user", "content": "Current context"}]

            fake_thread = MagicMock()

            with patch.object(cli, "_ensure_runtime_credentials", return_value=True), \
                 patch.object(cli, "_resolve_turn_agent_config", return_value={
                     "model": cli.model,
                     "runtime": {
                         "api_key": "key",
                         "base_url": "https://openrouter.ai/api/v1",
                         "provider": "openrouter",
                         "api_mode": "chat_completions",
                         "command": None,
                         "args": None,
                     },
                 }), \
                 patch("cli.threading.Thread", return_value=fake_thread) as mock_thread:
                cli.process_command("/btw what owns titles?")

        assert cli._pending_input.empty()
        assert len(cli._btw_tasks) == 1
        fake_thread.start.assert_called_once()
        kwargs = mock_thread.call_args.kwargs
        assert kwargs["daemon"] is True
        assert kwargs["name"].startswith("btw-task-")
