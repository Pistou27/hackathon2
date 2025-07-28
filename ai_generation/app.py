import warnings, logging, io, re
from pathlib import Path

import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util

from generation import TextGenerator
from summarization import TextSummarizer
from ethical_filter import ethical_filter   # v1 ou v2

# ───────── Masquage warnings PDF ──────────
warnings.filterwarnings("ignore", message="Could get FontBBox")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

st.set_page_config(page_title="Gen+Résumé+Filtre avec doc", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    body,.stApp{background:#FFDAB9;color:#5B3A29}
    textarea,input{background:#FFE5B4;color:#5B3A29}
    button{background:#FFB07C;color:#5B3A29;border:none}
    button:hover{background:#FFA07A}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🧠 Génération + Résumé + Filtre (avec contexte document)")

# ========== 0. Helpers ========================================================
_DOC_ASK_RGX = re.compile(
    r"(parle\s*[- ]*moi|résume|de quoi .* (doc|document|fichier))",
    flags=re.IGNORECASE,
)

def is_doc_query(prompt: str) -> bool:
    return bool(_DOC_ASK_RGX.search(prompt))

def chunk_by_tokens(text: str, tokenizer, max_tok=120, overlap=20):
    """Découpage robuste en passages ~120 tokens."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cur, cur_len, chunks = [], 0, []
    for sent in sentences:
        tok = tokenizer.encode(sent, add_special_tokens=False)
        if cur_len + len(tok) > max_tok:
            if cur:
                chunks.append(" ".join(cur))
            # overlap
            cur = cur[-overlap//10:]
            cur_len = sum(len(tokenizer.encode(x, add_special_tokens=False)) for x in cur)
        cur.append(sent)
        cur_len += len(tok)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def auto_retrieve(question, all_chunks, emb_mat, model, k=3):
    """Retourne k passages les + proches de la question."""
    q_vec = model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(q_vec, emb_mat, top_k=k)[0]
    return "\n\n".join(all_chunks[h["corpus_id"]] for h in hits)

# ========== 1. Barre latérale paramètres =====================================
with st.sidebar:
    st.header("🛠️  Paramètres")
    model_choice = st.selectbox(
        "Modèle génération",
        ["gpt2", "distilgpt2", "gpt2-medium", "./gpt2_lora_finetuned"],
        format_func=lambda x: "LoRA-finetuned" if x.startswith("./") else x,
    )
    max_tokens = st.slider("Max nouveaux tokens", 50, 400, 180, 10)
    quant8 = st.checkbox("Quantization 8-bit (GPU uniquement)",
                         value=False, disabled=not torch.cuda.is_available())
    mode = st.radio(
    "🧭  Mode",
    ["Doc-QA", "Post LinkedIn"],
    horizontal=True)

# ========== 2. Lazy loaders ===================================================
@st.cache_resource(show_spinner=False)
def get_generator(path, tokens, q8):
    return TextGenerator(path, tokens, quantize_8bit=q8)

@st.cache_resource(show_spinner=False)
def get_summarizer():
    return TextSummarizer(max_input_length=512, max_summary_length=80)

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

generator  = get_generator(model_choice, max_tokens, quant8)
summarizer = get_summarizer()
embedder   = get_embedder()

# ========== 3. Upload / parsing document =====================================
st.subheader("📄 1) Uploader un document (facultatif)")
uploaded = st.file_uploader("Fichier (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
doc_text, chunks, embeddings = "", [], None

if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".txt":
        doc_text = uploaded.read().decode("utf-8", errors="ignore")

    elif suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(io.BytesIO(uploaded.read())) as pdf:
            doc_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    elif suffix == ".docx":
        import docx
        doc = docx.Document(io.BytesIO(uploaded.read()))
        doc_text = "\n".join(p.text for p in doc.paragraphs)

    # ---- découpage par tokens ---------------------------------------------
    chunks = chunk_by_tokens(doc_text, generator.tokenizer)
    st.success(f"{len(chunks)} passages indexés.")
    if chunks:
        st.caption(f"🔍 Premier extrait :\n\n{chunks[0][:200]}…")

    # embeddings
    embeddings = embedder.encode(chunks, convert_to_tensor=True, show_progress_bar=False)

    # Recherche manuelle
    query = st.text_input("🔍 Chercher dans le doc", "")
    if query:
        retrieved = auto_retrieve(query, chunks, embeddings, embedder, k=5)
        st.markdown("**Passages les plus proches :**")
        for idx, passage in enumerate(retrieved.split("\n\n")):
            if st.button(f"➕ Utiliser passage {idx}", key=f"use_{idx}"):
                st.session_state["selected_passage"] = passage
            st.caption(passage[:400] + "…")
            st.write("---")

# ========== 4. Prompt zone ====================================================
default_passage = st.session_state.get("selected_passage", "")
prompt = st.text_area("✍️ 2) Prompt",
                      value=(default_passage + "\n" if default_passage else ""),
                      height=180)
if default_passage:
    st.info("✅ Passage inséré depuis le document.")

# ========== 5. Boutons ========================================================
left, right = st.columns(2)
submit = left.button("🚀 Lancer")
if right.button("🧹 Effacer passage"):
    st.session_state.pop("selected_passage", None)
    st.experimental_rerun()

# ========== 6. Pipeline =======================================================
if submit:
    if not prompt.strip():
        st.warning("Le prompt est vide.")
        st.stop()

    # --- Cas A : demande de résumé global -----------------------------------
    if doc_text and is_doc_query(prompt):
        st.toast("⏳ Résumé global du document…", icon="📑")
        doc_short = doc_text[:15000]  # ~4k tokens
        summary = summarizer.summarize(doc_short)
        st.subheader("📰 Résumé automatique du document")
        st.write(summary)

        res = ethical_filter(summary)
        st.subheader("🔍 Filtrage éthique")
        st.write(res)
        st.toast("✨ Terminé", icon="✅")
        st.stop()

    # --- Cas B : question précise -> auto-retrieve si rien choisi ----------
    if doc_text and not default_passage:
        auto_passages = auto_retrieve(prompt, chunks, embeddings, embedder, k=3)
        prompt = (
            "Voici des extraits du document :\n"
            f"{auto_passages}\n\n"
            "Réponds à la demande suivante en t'appuyant uniquement sur ces extraits.\n"
            f"Demande : {prompt}"
        )

    # Troncature à 1 000 tokens
    ids = generator.tokenizer.encode(prompt)
    if len(ids) > 1000:
        prompt = generator.tokenizer.decode(ids[-1000:])
        st.warning("Prompt tronqué à 1 000 tokens.")

    st.toast("⏳ Génération…", icon="🧠")
    gen_text = generator.generate(prompt)
    st.subheader("📝 Texte généré")
    st.write(gen_text)

    summary = summarizer.summarize(gen_text)
    st.subheader("📰 Résumé")
    st.write(summary)

    res = ethical_filter(summary)
    st.subheader("🔍 Filtrage éthique du résumé")
    st.write(res)

    st.toast("✨ Terminé", icon="✅")