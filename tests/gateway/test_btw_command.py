"""Tests for /btw gateway slash command."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/btw", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()
    runner._active_btw_tasks = {}

    mock_store = MagicMock()
    mock_store.get_or_create_session.return_value = MagicMock(session_id="sess_123")
    mock_store.load_transcript.return_value = [{"role": "user", "content": "main context"}]
    runner.session_store = mock_store

    from gateway.hooks import HookRegistry
    runner.hooks = HookRegistry()

    return runner


class TestHandleBtwCommand:
    @pytest.mark.asyncio
    async def test_no_question_shows_usage(self):
        runner = _make_runner()
        event = _make_event(text="/btw")
        result = await runner._handle_btw_command(event)
        assert "Usage:" in result
        assert "/btw" in result

    @pytest.mark.asyncio
    async def test_valid_question_starts_task(self):
        runner = _make_runner()
        created_tasks = []

        def capture_task(coro, *args, **kwargs):
            coro.close()
            mock_task = MagicMock()
            mock_task.done.return_value = False
            created_tasks.append(mock_task)
            return mock_task

        with patch("gateway.run.asyncio.create_task", side_effect=capture_task):
            event = _make_event(text="/btw what owns titles?")
            result = await runner._handle_btw_command(event)

        assert "💬 /btw started" in result
        assert "what owns titles?" in result
        assert len(created_tasks) == 1
        assert len(runner._active_btw_tasks) == 1

    @pytest.mark.asyncio
    async def test_rejects_second_active_btw_for_same_session(self):
        runner = _make_runner()
        task = MagicMock()
        task.done.return_value = False
        session_key = runner._session_key_for_source(_make_event().source)
        runner._active_btw_tasks[session_key] = task

        result = await runner._handle_btw_command(_make_event(text="/btw another one"))
        assert "already running" in result


class TestRunBtwTask:
    @pytest.mark.asyncio
    async def test_no_credentials_sends_error(self):
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = _make_event().source
        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": None}):
            await runner._run_btw_task("what owns titles?", source, "telegram:12345:67890", "btw_test")

        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        sent = call_args[1].get("content", call_args[0][1] if len(call_args[0]) > 1 else "")
        assert "/btw failed" in sent

    @pytest.mark.asyncio
    async def test_successful_task_sends_prefixed_result(self):
        runner = _make_runner()
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock()
        mock_adapter.extract_media = MagicMock(return_value=([], "Session titles live in hermes_state.py"))
        mock_adapter.extract_images = MagicMock(return_value=([], "Session titles live in hermes_state.py"))
        runner.adapters[Platform.TELEGRAM] = mock_adapter

        source = _make_event().source
        mock_result = {"final_response": "Session titles live in hermes_state.py", "messages": []}

        fake_loop = MagicMock()
        fake_loop.run_in_executor = AsyncMock(return_value=mock_result)

        with patch("gateway.run._resolve_runtime_agent_kwargs", return_value={"api_key": "test-key"}), \
             patch("gateway.run._load_gateway_config", return_value={}), \
             patch("gateway.run._resolve_gateway_model", return_value="anthropic/claude-opus-4.6"), \
             patch("gateway.run.asyncio.get_event_loop", return_value=fake_loop), \
             patch("run_agent.AIAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run_conversation.return_value = mock_result
            MockAgent.return_value = mock_agent_instance

            await runner._run_btw_task("what owns titles?", source, "telegram:12345:67890", "btw_test")

        mock_adapter.send.assert_called_once()
        content = mock_adapter.send.call_args[1].get("content", "")
        assert "💬 /btw" in content
        assert "what owns titles?" in content
        assert "Session titles live in hermes_state.py" in content


class TestBtwRegistry:
    def test_btw_is_known_gateway_command(self):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "btw" in GATEWAY_KNOWN_COMMANDS

    def test_btw_in_commands_dict(self):
        from hermes_cli.commands import COMMANDS
        assert "/btw" in COMMANDS

    def test_btw_in_session_category(self):
        from hermes_cli.commands import COMMANDS_BY_CATEGORY
        assert "/btw" in COMMANDS_BY_CATEGORY["Session"]
