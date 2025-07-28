import streamlit as st
from ethical_filter import ethical_filter
from generation import TextGenerator  # ✅ Import de ton générateur

# Configuration de la page
st.set_page_config(page_title="🧠 Filtrage Éthique & Génération", page_icon="🧠")

st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #FFDAB9; /* Peach Puff */
        color: #5B3A29; /* Brun foncé pour le texte */
    }
    textarea, input, .stTextArea, .stTextInput {
        background-color: #FFE5B4; /* pêche clair pour les champs */
        color: #5B3A29;
    }
    button {
        background-color: #FFB07C; /* pêche moyen */
        color: #5B3A29;
        border: none;
    }
    button:hover {
        background-color: #FFA07A; /* un peu plus vif au hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Génération & Filtrage de Texte avec BERT + GPT2")

st.markdown("""
Ce mini outil permet :
- 🧠 De générer un texte à partir d'un *prompt* avec `distilGPT2`
- 🚨 De filtrer automatiquement les contenus toxiques avec `unitary/toxic-bert`
""")

# Chargement du générateur de texte (avec distilGPT2)
generator = TextGenerator(max_length=80)

# Formulaire utilisateur
with st.form("text_gen_form"):
    prompt = st.text_area("✍️ Donne un prompt pour générer du texte :", height=150)
    submit = st.form_submit_button("Générer et analyser")

if submit:
    if not prompt.strip():
        st.warning("Merci de saisir un prompt.")
    else:
        generated = generator.generate(prompt)
        result = ethical_filter(generated)

        st.subheader("📝 Texte généré")
        st.write(generated)

        st.subheader("🔍 Filtrage éthique")
        st.write(f"**Statut :** `{result['status']}`")

        if result["flagged"]:
            st.error(f"⚠️ Contenu toxique détecté (label: `{result['label']}`, score: `{result['score']:.2f}`)")
        else:
            st.success("✅ Aucun contenu toxique détecté.")
