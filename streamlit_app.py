# --- streamlit_app.py ---
import streamlit as st
import pandas as pd
import numpy as np
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from tqdm import tqdm

# --- Seiteneinstellungen & Header ---
st.set_page_config(page_title="🔗 Keyword-URL Matching Tool", layout="wide")
st.title("🔗 Keyword-URL Matching Tool")

with st.expander("ℹ️ Was macht dieses Tool?", expanded=True):
    st.markdown(
        """
        Dieses Tool berechnet die semantische Ähnlichkeit zwischen deinen Webseiten (basierend auf. Title Tag Meta Description & Body Content) und einer Liste an Keywords.
        Du bekommst zwei Resultate als CSV:

        1. Die am besten passenden URLs für jedes Keyword
        2. Die am besten passenden Keywords für jede URL

        **Wichtig:** Die Berechnung basiert auf OpenAI-Embeddings. Kontrolliere die Resultate kritisch.
        """
    )

# --- API Key ---
st.subheader("🔑 OpenAI API Key")
api_key = st.text_input("Bitte gib deinen API-Key ein", type="password")
st.markdown("[💡 API-Key generieren](https://platform.openai.com/account/api-keys)", unsafe_allow_html=True)

if not api_key:
    st.warning("Bitte gib deinen OpenAI API Key ein.")
    st.stop()
client = OpenAI(api_key=api_key)

# --- Datei Upload ---
st.subheader("⬆️ Inhalte & Keywords hochladen")

content_file = st.file_uploader("1. Lade die Datei mit den Webseiteninformationen hoch", type="csv")
st.markdown("""📄 **Erwartetes Format:**  
**Spalte 1:** URL  
**Spalte 2:** Title Tag  
**Spalte 3:** Meta Description  
**Spalte 4:** Content  
""", unsafe_allow_html=True)

keywords_file = st.file_uploader("2. Lade die Datei mit den Keywords hoch", type="csv")
st.markdown("""🗝️ **Erwartetes Format:**  
**Spalte 1:** Keyword (beginnend in Zeile 2)  
""", unsafe_allow_html=True)

# --- Einstellungen ---
st.subheader("⚙️ Einstellungen")
max_urls = st.number_input("🔢 Maximale URLs pro Keyword (0 = unbegrenzt)", min_value=0, value=3)
max_keywords = st.number_input("🔢 Maximale Keywords pro URL (0 = unbegrenzt)", min_value=0, value=3)
threshold = st.slider("📈 Similarity-Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
model = st.selectbox("🤖 Embedding-Modell wählen", ["text-embedding-3-small", "text-embedding-3-large"], index=1)

# --- Analyse starten Button ---
start = st.button("🚀 Analyse starten")

if not (api_key and content_file and keywords_file and start):
    st.stop()

# --- Tokenizer & Embedding Funktion ---
encoding = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 8100

def truncate(text):
    tokens = encoding.encode(text)
    return encoding.decode(tokens[:MAX_TOKENS]) if len(tokens) > MAX_TOKENS else text

def get_embedding(text):
    text = text.replace("\n", " ").strip()
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Fehler bei Embedding: {e}")
        return None

# --- Inhalte vorbereiten ---
df_content = pd.read_csv(content_file)
df_content.columns = [c.strip().lower() for c in df_content.columns]
df_content = df_content.rename(columns={"title tag": "title_tag", "meta description": "meta_description"})
df_content = df_content.dropna(subset=["url", "content"])
df_content["text"] = (
    "URL: " + df_content["url"].fillna("") + "\n" +
    "Title Tag: " + df_content["title_tag"].fillna("") + "\n" +
    "Meta Description: " + df_content["meta_description"].fillna("") + "\n" +
    "Content:\n" + df_content["content"].fillna("")
).apply(truncate)

st.info("🔄 Erstelle Embeddings für Seiten")
df_content["embedding"] = df_content["text"].apply(get_embedding)
df_content = df_content[df_content["embedding"].notnull()]

# --- Keywords vorbereiten ---
df_keywords = pd.read_csv(keywords_file)
df_keywords.columns = ["keyword"]
df_keywords = df_keywords.dropna(subset=["keyword"])

st.info("🔄 Erstelle Embeddings für Keywords")
df_keywords["embedding"] = df_keywords["keyword"].apply(get_embedding)
df_keywords = df_keywords[df_keywords["embedding"].notnull()]

# --- Similarity berechnen ---
st.success("✅ Berechne Ähnlichkeiten")
page_embeddings = np.vstack(df_content["embedding"].values)
keyword_embeddings = np.vstack(df_keywords["embedding"].values)
similarities = cosine_similarity(keyword_embeddings, page_embeddings)

# --- Ergebnis 1: URLs pro Keyword ---
st.subheader("📄 1. Beste URLs pro Keyword")
rows_kw_url = []
for i, kw in enumerate(df_keywords["keyword"]):
    sim_row = similarities[i]
    top_idxs = sim_row.argsort()[::-1]
    count = 0
    for j in top_idxs:
        score = sim_row[j]
        if score >= threshold:
            rows_kw_url.append({"Keyword": kw, "URL": df_content.iloc[j]["url"], "Similarity": round(score, 4)})
            count += 1
            if max_urls != 0 and count >= max_urls:
                break

df_kw_url = pd.DataFrame(rows_kw_url)
st.dataframe(df_kw_url)
st.download_button("📥 CSV herunterladen (Keywords → URLs)", df_kw_url.to_csv(index=False).encode("utf-8"), "1_keywords_to_urls.csv")

# --- Ergebnis 2: Keywords pro URL ---
st.subheader("📄 2. Beste Keywords pro URL")
rows_url_kw = []
for j, page_url in enumerate(df_content["url"]):
    sim_column = similarities[:, j]
    top_idxs = sim_column.argsort()[::-1]
    count = 0
    for i in top_idxs:
        score = sim_column[i]
        if score >= threshold:
            rows_url_kw.append({"URL": page_url, "Keyword": df_keywords.iloc[i]["keyword"], "Similarity": round(score, 4)})
            count += 1
            if max_keywords != 0 and count >= max_keywords:
                break

df_url_kw = pd.DataFrame(rows_url_kw)
st.dataframe(df_url_kw)
st.download_button("📥 CSV herunterladen (URLs → Keywords)", df_url_kw.to_csv(index=False).encode("utf-8"), "2_urls_to_keywords.csv")
