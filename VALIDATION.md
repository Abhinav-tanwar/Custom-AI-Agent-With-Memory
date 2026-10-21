# Validation

Validated on Python 3.12 on macOS.

- 17 automated tests pass using fake embeddings and mocked model/audio clients.
- Ruff checks pass.
- Streamlit workspace renders with the charcoal, ivory, and amber theme.
- Live OpenAI embeddings and chat: a synthetic project detail was saved in temporary Chroma storage, recalled after reopening the store, used in an answer in a fresh conversation, and deleted successfully.
- Live Whisper transcription: synthetic speech was transcribed successfully. Physical microphone recording still depends on the browser's microphone permission and the user's hardware.

Live checks used synthetic data and temporary storage. No personal conversation or memory data is included in the repository. Model responses are nondeterministic; these checks demonstrate integration behavior, not a general accuracy benchmark.

`requirements-lock.txt` records the tested macOS environment. The bounded requirements files are used for installation and Linux CI; the full snapshot may contain platform-specific packages.
