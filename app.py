import streamlit as st
import uuid
import chromadb
import os
from styles import load_custom_css

from DocumentProcessor import (
    get_files_from_folder,
    get_file_hash,
    load_cache,
    extract_txt_detailed,
    extract_pdf_detailed,
    extract_docx_detailed,
    save_cache
)
from ChatEngine import get_embedding_function, answer_question_with_groq

st.set_page_config(
    page_title="Biomedical Document Chatbot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css()

CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")
CHROMA_FOLDER = "./chroma_db"

os.makedirs(CACHE_FOLDER, exist_ok=True)
os.makedirs(CHROMA_FOLDER, exist_ok=True)

if "collection" not in st.session_state:
    st.session_state.collection = None

if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.active_chat = None

client = chromadb.Client(
    settings=chromadb.Settings(
        persist_directory=CHROMA_FOLDER
    )
)

collections = client.list_collections()

if collections:
    collection = client.get_collection(
        name=collections[0].name,
        embedding_function=get_embedding_function()
    )
    st.session_state.collection = collection
else:
    with st.spinner("📚 Processing documents..."):
        files = get_files_from_folder()
        if not files:
            st.error("No documents found")
            st.stop()

        collection = client.create_collection(
            name="biomed_docs",
            embedding_function=get_embedding_function()
        )

        all_chunks = []
        all_meta = {}

        for path in files:
            name = os.path.basename(path)
            ext = name.split(".")[-1].lower()
            key = f"{get_file_hash(path)}_{ext}"
            cached = load_cache(key)

            if cached:
                info = cached
            else:
                if ext == "pdf":
                    info, _ = extract_pdf_detailed(path)
                elif ext in ["doc", "docx"]:
                    info, _ = extract_docx_detailed(path)
                elif ext == "txt":
                    info, _ = extract_txt_detailed(path)
                else:
                    continue
                save_cache(key, info)

            for c in info["chunks"]:
                if isinstance(c, dict):
                    all_chunks.append(c["content"])
                    all_meta[len(all_chunks) - 1] = c["metadata"]
                else:
                    all_chunks.append(c)
                    all_meta[len(all_chunks) - 1] = {
                        "source": name,
                        "page": "N/A",
                        "is_table": "False",
                        "table_number": "N/A"
                    }

        for i in range(0, len(all_chunks), 500):
            collection.add(
                documents=all_chunks[i:i+500],
                metadatas=[all_meta[j] for j in range(i, min(i+500, len(all_chunks)))],
                ids=[f"chunk_{j}" for j in range(i, min(i+500, len(all_chunks)))]
            )

        st.session_state.collection = collection

if not st.session_state.chats:
    cid = f"chat_{uuid.uuid4().hex[:6]}"
    st.session_state.chats[cid] = {
        "title": "New Chat",
        "messages": [],
        "context": []
    }
    st.session_state.active_chat = cid

st.markdown("""
<div class="main-card">
    <h1 style='text-align:center;margin:0;'>🧬 Biomedical Document Chatbot</h1>
    <p style='text-align:center;margin-top:10px;'>Your smart assistant for the Master Biomedical Engineering (MBE) program at Hochschule Anhalt</p>
</div>

<div class="main-card">
<h2 style="color:#00d9ff; text-align:center;">What does it do?</h2>

<p style="font-size:1.1rem; text-align:center;">Ask any question about your Master's in Biomedical Engineering – it answers directly from the official documents only.</p>

<h3 style="color:#00d9ff;">Features</h3>
<ul style="font-size:1.05rem; line-height:1.8;">
    <li>Chat naturally and ask follow-ups (e.g., "summarize that")</li>
    <li>Works in English, German, and Arabic</li>
    <li>Fast and accurate responses</li>
    <li>Remembers conversation context</li>
</ul>

<h3 style="color:#00d9ff;">Documents it uses</h3>
<ul style="font-size:1.05rem; line-height:1.8;">
    <li>Study & Examination Regulations (SPO)</li>
    <li>Module Handbook (June 2024)</li>
    <li>Guide to Writing Scientific Papers</li>
    <li>Final Thesis Guidelines</li>
</ul>

<h3 style="color:#00d9ff;">Try these</h3>
<ul style="font-size:1.05rem; line-height:1.8;">
    <li>What are the requirements for registering the master's thesis?</li>
    <li>Tell me about the internship</li>
    <li>What are the thesis rules?</li>
</ul>

<p style="text-align:center; font-size:1.1rem; margin-top:2rem;">Just type your question and get clear answers! 🚀</p>
</div>
""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("# 🧬 BioMed Doc Chat")

    files = get_files_from_folder()

    if st.button("# + New Chat", use_container_width=True):
        cid = f"chat_{uuid.uuid4().hex[:6]}"
        st.session_state.chats[cid] = {
            "title": "New Chat",
            "messages": [],
            "context": []
        }
        st.session_state.active_chat = cid
        st.rerun()
    st.markdown("---")
    st.markdown("### 💬 Chats")
    for cid in list(st.session_state.chats.keys()):
        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(f"💬 {st.session_state.chats[cid]['title']}", key=f"open_{cid}", use_container_width=True):
                st.session_state.active_chat = cid
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{cid}", use_container_width=True):
                del st.session_state.chats[cid]
                if st.session_state.active_chat == cid:
                    st.session_state.active_chat = None
                st.rerun()
chat = st.session_state.chats[st.session_state.active_chat]
messages = chat["messages"]

for m in messages:
    if m["role"] == "user":
        st.markdown(f"<div class='chat-message user-message'>👤 {m['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-message assistant-message'>🤖<br>{m['content']}</div>", unsafe_allow_html=True)

query = st.chat_input("Ask anything about your documents...")

if query:
    messages.append({"role": "user", "content": query})
    if chat["title"] == "New Chat":
        chat["title"] = query[:40]

    res = st.session_state.collection.query(
        query_texts=[query],
        n_results=10
    )

    chunks = [{"content": d, "metadata": m} for d, m in zip(res["documents"][0], res["metadatas"][0])]

    answer = answer_question_with_groq(query, chunks, messages)
    messages.append({"role": "assistant", "content": answer})
    chat["context"] = chunks

    st.rerun()
