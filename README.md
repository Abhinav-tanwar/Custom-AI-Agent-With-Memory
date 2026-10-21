# Custom AI Agent With Memory

![Custom AI Agent With Memory — charcoal and amber project header](assets/readme-header.svg)

A voice and text assistant with a persistent, inspectable memory library. Save selected preferences or project details, start a fresh conversation, and retrieve relevant context without storing every message as a fact.

**Python · LangChain · OpenAI · Whisper · ChromaDB · Streamlit**

![Memory agent workspace](screenshots/workspace.png)
*Actual local workspace with conversation, voice input and explicit memory controls.*

## Verified results

| Check | Observed result |
| --- | --- |
| Automated validation | **17 tests passed**, using fake embeddings and mocked model/audio clients |
| Persistent memory | **Save → reopen store → recall in fresh conversation → delete** verified live |
| Voice integration | Live Whisper transcription verified using synthetic speech |

[Validation record](VALIDATION.md). Live checks used synthetic context and temporary storage. They demonstrate integration, not a recall or transcription accuracy benchmark. Physical microphone capture depends on browser permission and hardware.

## Run locally

Requires **Python 3.12** and an OpenAI API key with API credit.

```sh
git clone https://github.com/Abhinav-tanwar/Custom-AI-Agent-With-Memory.git
cd Custom-AI-Agent-With-Memory
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` privately in `.env`, then:

```sh
.venv/bin/python -m streamlit run app.py
```

Open **http://127.0.0.1:8030**.

1. Save a preference in **Memory library**.
2. Ask a related question in **Conversation**; expand **Context used** to inspect retrieval.
3. Start a new conversation and ask again to check persistence.
4. Record or upload audio, choose **Transcribe voice note**, edit the draft and send it.

## How it works

`Text / Whisper transcript → bounded conversation + semantic memory recall → LangChain → OpenAI answer`

- Explicit save and forget operations manage local ChromaDB memory.
- Duplicate detection, relevance filtering and optional retrieval keep context focused.
- Inspect retrieved sources and export conversations or the memory library.

## Verify

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

## Scope and data

Single-user localhost application without shared-user authentication. Saved memories persist on disk; chat history lives in the current Streamlit session. Memory text is sent to OpenAI for embeddings; selected context goes to the chat model; requested audio goes to Whisper. Deleting a memory does not remove prior chat history, exports or backups. Changing embedding models requires a fresh store or deliberate migration. Keep `.env` and `data/` private.

[Architecture, configuration, data handling and limitations](docs/guide.md)

<sub>macOS/Linux commands shown. On Windows, use `py -3.12 -m venv .venv` and replace `.venv/bin/python` with `.venv\Scripts\python.exe`.</sub>
