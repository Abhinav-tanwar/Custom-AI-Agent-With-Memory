from pathlib import Path
from unittest.mock import Mock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest


def test_app_renders_without_api_key(monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setenv("OPENAI_API_KEY", "")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "Custom AI Agent With Memory"
    assert any("Add OPENAI_API_KEY" in item.value for item in app.info)


def test_chat_and_new_conversation(monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    fake = Mock()
    fake.reply.return_value = ("A useful response", [])
    fake.store.list.return_value = []
    with patch("memory_agent.agent.PersonalAgent", return_value=fake):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
        app.chat_input[0].set_value("Hello").run()
        assert not app.exception
        assert app.session_state["messages"][-1]["content"] == "A useful response"
        next(button for button in app.button if button.label == "New conversation").click().run()
        assert app.session_state["messages"] == []
