"""
DocMind AI — RAG Chatbot
────────────────────────
Model  : llama-3.1-8b-instant (via Groq, free)
Store  : ChromaDB
Embed  : sentence-transformers/all-MiniLM-L6-v2 (local)
UI     : Gradio 6.15+
"""

import os
import shutil

import gradio as gr
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_BASE   = "https://api.groq.com/openai/v1"
LLM_MODEL   = "llama-3.1-8b-instant"
CHROMA_DIR  = "/tmp/chroma_db"
CHUNK_SIZE  = 600
CHUNK_OVER  = 150
TOP_K       = 4
MAX_TOKENS  = 600

SYSTEM_PROMPT = (
    "You are DocMind, a brilliant and precise AI assistant. "
    "Your sole job is to answer the user's question using ONLY the document "
    "context provided. Be accurate, well-structured, and concise. "
    "If the answer is not in the context, say: "
    "'I couldn't find this information in the uploaded documents.' "
    "Never fabricate or infer beyond the given context."
)

# ── Global state ───────────────────────────────────────────────────────────────
vectordb:   Chroma | None = None
llm_client: OpenAI | None = None
doc_names:  list[str]     = []


# ── Gradio 6 message helpers ───────────────────────────────────────────────────
def user_msg(text: str) -> dict:
    return {"role": "user", "content": text}

def bot_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


def clean_key(raw: str) -> str:
    """Aggressively clean API key — strips spaces, newlines, quotes, BOMs."""
    if not raw:
        return ""
    return raw.strip().strip("\n").strip("\r").strip('"').strip("'").strip()


# ── Core logic ─────────────────────────────────────────────────────────────────
def process_pdfs(pdf_files, api_key: str):
    global vectordb, llm_client, doc_names

    api_key = clean_key(api_key)
    print(f"[DEBUG] Key prefix: '{api_key[:8]}...'  length: {len(api_key)}")

    if not api_key:
        return "❌  API key missing — paste your Groq key above.", gr.update(visible=False)
    if not pdf_files:
        return "❌  No files detected — upload at least one PDF.", gr.update(visible=False)

    try:
        # Test the key before processing PDFs
        test_client = OpenAI(base_url=GROQ_BASE, api_key=api_key)
        test_client.models.list()   # raises 401 immediately if key is bad
        llm_client = test_client

        splitter  = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVER
        )
        all_splits = []
        names      = []

        for pdf_path in pdf_files:
            loader = PyPDFLoader(pdf_path)
            pages  = loader.load()
            chunks = splitter.split_documents(pages)
            all_splits.extend(chunks)
            names.append(os.path.basename(pdf_path))

        embedding = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)

        vectordb  = Chroma.from_documents(
            documents=all_splits,
            embedding=embedding,
            persist_directory=CHROMA_DIR,
        )
        doc_names = names

        file_bullets = "\n".join(f"  • {n}" for n in names)
        status = (
            f"✅  Ready to answer questions!\n\n"
            f"📄  {len(names)} file(s) · {len(all_splits)} chunks indexed\n\n"
            f"{file_bullets}"
        )
        return status, gr.update(value=_doc_badges(names), visible=True)

    except Exception as exc:
        return f"❌  {exc}", gr.update(visible=False)


def _doc_badges(names: list[str]) -> str:
    badges = "".join(
        f"<span style='display:inline-block;margin:3px 4px;padding:3px 10px;"
        f"background:rgba(139,92,246,0.18);border:1px solid rgba(139,92,246,0.4);"
        f"border-radius:20px;font-size:0.78em;color:#c4b5fd;'>📄 {n}</span>"
        for n in names
    )
    return f"<div style='margin-top:6px;line-height:2;'>{badges}</div>"


def respond(message: str, history: list):
    global vectordb, llm_client

    message = message.strip()
    if not message:
        return history, ""

    if vectordb is None:
        return history + [
            user_msg(message),
            bot_msg("⚠️ No documents loaded. Upload PDFs and click **Process Documents** first."),
        ], ""

    if llm_client is None:
        return history + [
            user_msg(message),
            bot_msg("⚠️ No API key set. Enter your Groq API key and process your documents."),
        ], ""

    try:
        docs    = vectordb.similarity_search(message, k=TOP_K)
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {message}"},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.3,
        )
        answer = response.choices[0].message.content.strip()
        return history + [user_msg(message), bot_msg(answer)], ""

    except Exception as exc:
        return history + [
            user_msg(message),
            bot_msg(f"❌ **Error:** {exc}"),
        ], ""


def clear_all():
    global vectordb, doc_names
    vectordb  = None
    doc_names = []
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    return [], "", "🗑️  Session cleared. Upload new PDFs to begin.", gr.update(visible=False)


# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: #080a12 !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    color: #e2e8f0 !important; margin: 0 !important;
}
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; padding: 24px !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.35); border-radius: 3px; }
#hdr {
    background: linear-gradient(135deg, #0f0a2e 0%, #1e0a4a 35%, #0a1f3d 70%, #080a12 100%);
    border: 1px solid rgba(139,92,246,0.2); border-radius: 20px;
    padding: 40px 56px 36px; margin-bottom: 24px;
    text-align: center; position: relative; overflow: hidden;
}
#hdr::before {
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse 60% 50% at 50% 0%, rgba(139,92,246,0.12), transparent);
    pointer-events:none;
}
#hdr .logo-row { display:flex; align-items:center; justify-content:center; gap:14px; margin-bottom:12px; }
#hdr .logo { font-size:2.6em; line-height:1; }
#hdr h1 {
    font-size:2.5em; font-weight:900; letter-spacing:-0.03em;
    background: linear-gradient(135deg,#e0d7ff 0%,#a78bfa 40%,#60a5fa 80%,#c4b5fd 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; margin:0;
}
#hdr .tagline { color:#94a3b8; font-size:1.05em; margin:8px 0 18px; line-height:1.6; }
#hdr .pill-row { display:flex; align-items:center; justify-content:center; gap:8px; flex-wrap:wrap; }
#hdr .pill { display:inline-flex; align-items:center; gap:5px; padding:5px 14px; border-radius:20px; font-size:0.76em; font-weight:600; letter-spacing:0.03em; }
#hdr .pill-purple { background:rgba(139,92,246,0.15); border:1px solid rgba(139,92,246,0.35); color:#a78bfa; }
#hdr .pill-blue   { background:rgba(59,130,246,0.13);  border:1px solid rgba(59,130,246,0.3);  color:#7dd3fc; }
#hdr .pill-green  { background:rgba(16,185,129,0.12);  border:1px solid rgba(16,185,129,0.3);  color:#6ee7b7; }
#hdr .pill-orange { background:rgba(249,115,22,0.12);  border:1px solid rgba(249,115,22,0.3);  color:#fdba74; }
#sidebar-col {
    background:#0e1120; border:1px solid rgba(139,92,246,0.15);
    border-radius:18px; padding:24px 20px;
}
.panel-title {
    font-size:0.7em !important; font-weight:700 !important;
    text-transform:uppercase !important; letter-spacing:0.1em !important;
    color:#6d28d9 !important; margin-bottom:10px !important;
}
label span { color:#94a3b8 !important; font-size:0.82em !important; font-weight:500 !important; }
input[type="text"], input[type="password"], textarea {
    background:#131929 !important; border:1px solid rgba(99,102,241,0.25) !important;
    border-radius:10px !important; color:#e2e8f0 !important;
    font-family:'Inter',sans-serif !important; font-size:0.93em !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
input:focus, textarea:focus {
    border-color:#7c3aed !important;
    box-shadow:0 0 0 3px rgba(124,58,237,0.12) !important; outline:none !important;
}
#process-btn {
    background:linear-gradient(135deg,#5b21b6 0%,#1d4ed8 100%) !important;
    border:none !important; border-radius:12px !important; color:#fff !important;
    font-weight:700 !important; font-size:0.92em !important; padding:14px 0 !important;
    box-shadow:0 4px 20px rgba(91,33,182,0.3) !important; transition:all 0.2s !important;
}
#process-btn:hover {
    opacity:0.88 !important; transform:translateY(-2px) !important;
    box-shadow:0 8px 28px rgba(91,33,182,0.45) !important;
}
#clear-btn {
    background:rgba(239,68,68,0.07) !important; border:1px solid rgba(239,68,68,0.25) !important;
    border-radius:12px !important; color:#fca5a5 !important;
    font-weight:600 !important; transition:all 0.2s !important;
}
#clear-btn:hover { background:rgba(239,68,68,0.14) !important; border-color:rgba(239,68,68,0.5) !important; }
#send-btn {
    background:linear-gradient(135deg,#5b21b6 0%,#1d4ed8 100%) !important;
    border:none !important; border-radius:12px !important; color:#fff !important;
    font-weight:700 !important; box-shadow:0 4px 16px rgba(91,33,182,0.3) !important;
    min-width:100px !important; transition:all 0.2s !important;
}
#send-btn:hover { opacity:0.88 !important; box-shadow:0 6px 22px rgba(91,33,182,0.45) !important; }
#status-box textarea {
    background:#0a1a10 !important; border:1px solid rgba(16,185,129,0.2) !important;
    color:#6ee7b7 !important; font-family:'SF Mono','Fira Code',monospace !important;
    font-size:0.82em !important; line-height:1.7 !important; border-radius:10px !important;
}
#chatbot {
    background:#0e1120 !important; border:1px solid rgba(99,102,241,0.18) !important;
    border-radius:18px !important;
}
#chatbot .message {
    font-size:0.94em !important; line-height:1.7 !important;
    padding:14px 18px !important; border-radius:14px !important;
}
#msg-input textarea {
    background:#131929 !important; border:1px solid rgba(99,102,241,0.28) !important;
    border-radius:14px !important; color:#e2e8f0 !important; font-size:0.95em !important;
    padding:13px 18px !important; resize:none !important;
    transition:border-color 0.2s, box-shadow 0.2s !important;
}
#msg-input textarea:focus {
    border-color:#7c3aed !important; box-shadow:0 0 0 3px rgba(124,58,237,0.1) !important;
}
.divider { border:none; border-top:1px solid rgba(255,255,255,0.06); margin:18px 0; }
.howto {
    background:linear-gradient(135deg,#0f0a2e22,#0a1f3d22);
    border:1px solid rgba(99,102,241,0.15); border-radius:12px;
    padding:16px 18px; font-size:0.83em; color:#64748b; line-height:2;
}
.howto b { color:#94a3b8; }
.howto .step { display:flex; align-items:flex-start; gap:8px; margin:3px 0; }
.howto .num {
    min-width:20px; height:20px; background:rgba(139,92,246,0.2); border-radius:50%;
    display:inline-flex; align-items:center; justify-content:center;
    font-size:0.75em; font-weight:700; color:#a78bfa; margin-top:2px;
}
"""

# ── Layout ─────────────────────────────────────────────────────────────────────
with gr.Blocks(title="DocMind AI") as demo:

    gr.HTML("""
    <div id="hdr">
        <div class="logo-row"><span class="logo">🧠</span><h1>DocMind AI</h1></div>
        <p class="tagline">
            Hi! I'm a <strong style="color:#c4b5fd;">super-powered AI assistant</strong> built on RAG.<br>
            Upload any PDF — I'll read it and answer every question you have about it, instantly.
        </p>
        <div class="pill-row">
            <span class="pill pill-orange">⚡ Llama 3.1 · 8B via Groq</span>
            <span class="pill pill-blue">🗄️ ChromaDB Vector Store</span>
            <span class="pill pill-green">🔍 Semantic Search</span>
            <span class="pill pill-purple">📄 Multi-PDF Support</span>
        </div>
    </div>
    """)

    with gr.Row(equal_height=False, variant="panel"):

        # ── Sidebar ──────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300, elem_id="sidebar-col"):
            gr.HTML("<p class='panel-title'>🔑 &nbsp;Authentication</p>")
            api_key = gr.Textbox(
                label="Groq API Key",
                placeholder="gsk_••••••••••••••••••••••••",
                type="password",
                info="Free at console.groq.com — never stored.",
            )
            gr.HTML("<hr class='divider'>")
            gr.HTML("<p class='panel-title'>📂 &nbsp;Document Upload</p>")
            pdf_upload = gr.File(
                label="Drag & drop PDFs or click to browse",
                file_types=[".pdf"],
                file_count="multiple",
                type="filepath",
                height=150,
            )
            process_btn = gr.Button(
                "🚀  Process Documents",
                elem_id="process-btn", variant="primary", size="lg",
            )
            doc_badges = gr.HTML(visible=False)
            gr.HTML("<hr class='divider'>")
            gr.HTML("<p class='panel-title'>📋 &nbsp;Status</p>")
            status_box = gr.Textbox(
                label="", interactive=False, elem_id="status-box",
                lines=5, placeholder="Waiting for documents…", show_label=False,
            )
            gr.HTML("<hr class='divider'>")
            gr.HTML("""
            <div class="howto">
                <b>How to use</b>
                <div class="step"><span class="num">1</span><span>Get a free key at <b>console.groq.com</b></span></div>
                <div class="step"><span class="num">2</span><span>Paste your <b>gsk_...</b> key above</span></div>
                <div class="step"><span class="num">3</span><span>Upload one or more PDFs</span></div>
                <div class="step"><span class="num">4</span><span>Click <b>Process Documents</b></span></div>
                <div class="step"><span class="num">5</span><span>Ask anything in the chat!</span></div>
            </div>
            """)
            gr.HTML("<hr class='divider'>")
            clear_btn = gr.Button("🗑️  Clear Session & Reset", elem_id="clear-btn", size="sm")

        # ── Chat ─────────────────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="", elem_id="chatbot", height=560, show_label=False,
                placeholder=(
                    "<div style='text-align:center;padding:80px 32px;color:#1e2a40;'>"
                    "<div style='font-size:3em;margin-bottom:16px;'>🧠</div>"
                    "<div style='font-size:1.2em;font-weight:700;color:#334155;margin-bottom:8px;'>"
                    "DocMind AI is ready</div>"
                    "<div style='font-size:0.9em;color:#475569;max-width:340px;margin:0 auto;line-height:1.6;'>"
                    "Process your PDFs on the left, then ask me anything.</div></div>"
                ),
            )
            with gr.Row(equal_height=True):
                msg_input = gr.Textbox(
                    placeholder="Ask anything about your documents…  (Enter to send)",
                    show_label=False, scale=5, elem_id="msg-input",
                    lines=1, max_lines=6, autofocus=True,
                )
                send_btn = gr.Button("Send ➤", elem_id="send-btn", scale=1, variant="primary")
            gr.HTML("""
            <div style="text-align:center;margin-top:10px;font-size:0.75em;color:#1e2a40;">
                Answers grounded in your documents only &nbsp;·&nbsp; Llama 3.1-8B via Groq
            </div>
            """)

    # ── Events ────────────────────────────────────────────────────────────────
    process_btn.click(fn=process_pdfs, inputs=[pdf_upload, api_key], outputs=[status_box, doc_badges])
    send_btn.click(fn=respond,         inputs=[msg_input, chatbot],   outputs=[chatbot, msg_input])
    msg_input.submit(fn=respond,       inputs=[msg_input, chatbot],   outputs=[chatbot, msg_input])
    clear_btn.click(fn=clear_all,      outputs=[chatbot, msg_input, status_box, doc_badges])


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        demo.launch(
            server_name="0.0.0.0", server_port=7860,
            inbrowser=True, show_error=True,
            css=CSS,
            theme=gr.themes.Base(
                primary_hue=gr.themes.colors.violet,
                secondary_hue=gr.themes.colors.blue,
                neutral_hue=gr.themes.colors.slate,
                font=gr.themes.GoogleFont("Inter"),
            ),
        )
    except Exception as e:
        print(f"\n❌  Launch error: {e}")
        input("Press Enter to exit…")