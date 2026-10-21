"""Streamlit entry point for Custom AI Agent With Memory."""

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

from memory_agent.agent import PersonalAgent
from memory_agent.store import MemoryStore

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Custom AI Agent With Memory", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.stApp{background:#181713} .block-container{max-width:1150px;padding-top:3rem}
h1{letter-spacing:-1.6px} [data-testid="stSidebar"]{background:#222019}
.stButton>button{border-radius:10px} [data-testid="stChatMessage"]{border:1px solid #454033;border-radius:16px}
.eyebrow{color:#d6af69;font-size:12px;letter-spacing:2px;font-weight:700}
.hero{color:#c7bfae;font-size:18px;max-width:760px;line-height:1.7}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def build_agent():
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return None
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), api_key=key
    )
    path = Path(os.getenv("MEMORY_DATA_DIR", "data"))
    if not path.is_absolute():
        path = ROOT / path
    store = MemoryStore(path, embeddings)
    return PersonalAgent(
        ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=key,
            temperature=0.4,
            max_tokens=1200,
            timeout=45,
            max_retries=1,
        ),
        store,
        OpenAI(api_key=key, timeout=45, max_retries=1),
    )


try:
    agent = build_agent()
except Exception:
    st.error("Local memory could not be opened. Check the data directory and restart the app.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "draft" not in st.session_state:
    st.session_state.draft = ""

with st.sidebar:
    st.markdown("### ◈  MEMORY AGENT")
    st.caption("A little context. A better conversation.")
    st.divider()
    use_memory = st.toggle("Use saved memories", value=True)
    st.caption(
        "Saved memories stay on this device. Relevant text is sent to OpenAI when used; embeddings and audio also use OpenAI."
    )
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.download_button(
        "Export conversation",
        json.dumps(st.session_state.messages, indent=2),
        file_name="conversation.json",
        mime="application/json",
        use_container_width=True,
    )
    st.divider()
    st.caption(
        "Single-user local workspace · Conversations stay in session memory. Save selected details in the Memory library."
    )

st.markdown('<div class="eyebrow">PERSONAL CONTEXT, KEPT WITH INTENTION</div>', unsafe_allow_html=True)
st.title("Custom AI Agent With Memory")
st.markdown(
    '<p class="hero">Think out loud. Pick up where you left off. An assistant that remembers the details you choose to keep.</p>',
    unsafe_allow_html=True,
)
chat, library, about = st.tabs(["Conversation", "Memory library", "How it works"])

with chat:
    if not agent:
        st.info("Add OPENAI_API_KEY to the local .env file, then restart the app to connect.")
    if not st.session_state.messages:
        st.markdown("#### Make room for your next idea")
        st.caption(
            "Plan a project, work through a decision, or ask a question. Your memory library adds context across conversations."
        )
        cols = st.columns(3)
        for col, label in zip(
            cols,
            ["Plan my next project", "Help me prioritize my week", "What do you remember about me?"],
            strict=True,
        ):
            if col.button(label, use_container_width=True):
                st.session_state.draft = label
                st.rerun()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("memories"):
                with st.expander(f"Context used · {len(message['memories'])} memories"):
                    for memory in message["memories"]:
                        st.markdown("• " + memory["text"])
    with st.expander("Voice note · record or upload"):
        st.caption(
            "Transcription sends your audio to OpenAI Whisper. Review the transcript before sending it to chat. Audio is not saved locally."
        )
        recording = st.audio_input("Record a voice note")
        upload = st.file_uploader("Or upload audio", type=["wav", "mp3", "m4a", "webm", "mp4", "ogg", "flac"])
        audio = recording or upload
        if st.button("Transcribe voice note", disabled=not agent or audio is None):
            try:
                with st.spinner("Transcribing…"):
                    st.session_state.draft = agent.transcribe(audio.getvalue(), audio.name)
                st.rerun()
            except Exception:
                st.error(
                    "Transcription failed. Check the audio format, size (under 20 MB), connection, and API billing."
                )
    if st.session_state.draft:
        draft = st.text_area("Review your message", value=st.session_state.draft, max_chars=8000)
        send_draft = st.button("Send message", disabled=not agent)
        if st.button("Discard draft"):
            st.session_state.draft = ""
            st.rerun()
    else:
        draft, send_draft = "", False
    typed = st.chat_input("Ask something, or bring an idea…", disabled=not agent, max_chars=8000)
    prompt = draft if send_draft else typed
    if prompt:
        try:
            with st.spinner("Connecting the dots…"):
                answer, memories = agent.reply(prompt, st.session_state.messages, use_memory)
            st.session_state.messages.extend(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer, "memories": memories},
                ]
            )
            st.session_state.draft = ""
            st.rerun()
        except Exception:
            st.error(
                "The response could not be completed. Check your connection, API key, and billing, then retry. Your conversation has not changed."
            )

with library:
    st.subheader("Keep what matters")
    st.caption(
        "Save preferences, goals, and project context explicitly. Saving text sends it to OpenAI for embedding. Avoid passwords and sensitive details."
    )
    memory = st.text_area(
        "A detail to remember",
        placeholder="I prefer concise explanations with practical Python examples.",
        max_chars=2000,
    )
    if st.button("Save memory", type="primary", disabled=not agent or not memory.strip()):
        try:
            agent.store.save(memory)
            st.success("Memory saved. It can now provide context in future conversations.")
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Memory could not be saved. Check your API connection and local storage.")
    rows = agent.store.list() if agent else []
    st.metric("Saved memories", len(rows))
    query = st.text_input("Filter library", placeholder="Find a saved detail…")
    for row in rows:
        if query.lower() not in row["text"].lower():
            continue
        with st.container(border=True):
            st.write(row["text"])
            st.caption("Saved " + row["created_at"][:10])
            if st.button("Forget this detail", key=row["id"]):
                agent.store.delete(row["id"])
                st.rerun()
    if not rows:
        st.info("Your library is empty. Add your first detail above.")
    st.download_button("Export memories", json.dumps(rows, indent=2), "memories.json", "application/json")
    confirm = st.checkbox("I want to delete all saved memories")
    if st.button("Delete all memories", disabled=not agent or not rows or not confirm):
        agent.store.clear()
        st.rerun()
    st.caption(
        "Forgetting a memory removes it from retrieval. Existing conversation context and exported files may still contain it; start a new conversation after deletion. Local database deletion is not forensic secure erasure."
    )

with about:
    st.subheader("Memory you can understand")
    st.markdown("""1. **Choose what to keep.** Only the Save memory control creates persistent records.
2. **Retrieve relevant context.** OpenAI embeddings and ChromaDB find related saved details.
3. **Have a grounded conversation.** LangChain passes bounded chat history and retrieved context to the model.
4. **Stay in control.** Inspect used context, export your data, or forget details at any time.

**Voice:** OpenAI Whisper turns a recording into an editable message before it is sent.

**Storage:** ChromaDB lives in the ignored `data/` directory. Chat history is temporary and resets with the session. This is a local, single-user application with no sign-in or multi-user isolation. Keep it bound to localhost; add authentication and tenant isolation before any shared deployment.

**Limitations:** Recall is approximate, memories may become outdated, and answers can be incorrect. This agent has no external action tools and does not execute code. Forgetting local records does not revoke data already sent to the API.
""")
