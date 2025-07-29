import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import os

st.set_page_config(page_title="📈 Évaluations", layout="wide")
st.title("📈 Évaluations du pipeline de génération")

db_path = "runs/history.sqlite"

# Lancement du script d'évaluation
if st.button("🚀 Lancer l’évaluation sur 50 exemples"):
    with st.spinner("Exécution en cours…"):
        result = subprocess.run(["python", "evaluate_pipeline.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("✅ Évaluation terminée avec succès.")
        else:
            st.error(f"❌ Erreur :\n{result.stderr}")

# Affichage des résultats
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM evals ORDER BY ts DESC", conn)
    conn.close()

    if not df.empty:
        st.dataframe(df)

        # Moyennes
        col1, col2, col3 = st.columns(3)
        col1.metric("ROUGE-L", f"{df['rouge'].mean():.4f}")
        col2.metric("Perplexité", f"{df['ppl'].mean():.2f}")
        col3.metric("Similarité", f"{df['sim'].mean():.4f}")

        # Courbes
        st.subheader("📊 Évolution des scores")
        df["ts"] = pd.to_datetime(df["ts"])

        st.line_chart(df.set_index("ts")[["rouge", "ppl", "sim"]])
    else:
        st.info("Aucun résultat enregistré.")
else:
    st.warning("Aucune base de données d’évaluation trouvée.")