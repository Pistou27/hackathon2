import streamlit as st
from ethical_filter import ethical_filter
from generation import TextGenerator
from summarization import TextSummarizer

# Configuration de la page
st.set_page_config(page_title="Génération, Résumé & Filtrage", page_icon="🧠")

st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #FFDAB9;
        color: #5B3A29;
    }
    textarea, input, .stTextArea, .stTextInput {
        background-color: #FFE5B4;
        color: #5B3A29;
    }
    button {
        background-color: #FFB07C;
        color: #5B3A29;
        border: none;
    }
    button:hover {
        background-color: #FFA07A;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Génération + Résumé + Filtrage de texte")

st.markdown("""
Ce mini outil permet :
- De générer un texte à partir d'un *prompt* avec `gpt2-medium`
- De résumer automatiquement le texte généré avec `distilBART`
- De filtrer le résumé pour détecter les contenus toxiques avec `unitary/toxic-bert`
""")

generator = TextGenerator(max_length=200)
summarizer = TextSummarizer(max_input_length=512, max_summary_length=80)

prompt = st.text_area("✍️ Entrez un prompt :", height=150)
submit = st.button("Générer, résumer et filtrer")

if submit:
    if not prompt.strip():
        st.warning("Merci de saisir un prompt.")
    else:
        # 1. Génération
        generated_text = generator.generate(prompt)
        st.subheader("📝 Texte généré")
        st.write(generated_text)

        # 2. Résumé
        summary = summarizer.summarize(generated_text)
        st.subheader("📰 Résumé du texte généré")
        st.write(summary)

        # 3. Filtrage éthique sur résumé
        result = ethical_filter(summary)
        st.subheader("🔍 Filtrage éthique du résumé")
        st.write(f"**Statut :** `{result['status']}`")
        if result["flagged"]:
            st.error(f"⚠️ Contenu toxique détecté (label: `{result['label']}`, score: `{result['score']:.2f}`)")
        else:
            st.success("✅ Aucun contenu toxique détecté.")
