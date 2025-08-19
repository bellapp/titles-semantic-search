import streamlit as st
import os
import requests
from elasticsearch import Elasticsearch
import cohere
import voyageai
import vertexai
from google.oauth2 import service_account
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from vertexai.language_models import TextEmbeddingModel

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Job Title Similarity Comparison - Developer- ")

# --- CONFIGURATION ---
try:
    ES_HOST = st.secrets["connections"]["elasticsearch"]["host"]
    LOCAL_API_URL = st.secrets["connections"]["custom_api"]["url"]
    GOOGLE_PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
    GOOGLE_LOCATION = st.secrets["gcp_service_account"]["location"]
except (KeyError, FileNotFoundError):
    st.error("Secrets for Elasticsearch, Custom API, or GCP are not set.")
    st.stop()

# --- Index Names ---
TITLES_DICTIONARY_OPENAI   = "developer_titles_openai"
TITLES_DICTIONARY_COHERE   = "developer_titles_cohere"
TITLES_DICTIONARY_VOYAGEAI = "developer_titles_voyageai"
TITLES_DICTIONARY_GOOGLE   = "developer_titles_google"
TITLES_DICTIONARY_E5       = "embedding_dictionary_e5_base"
TITLES_DICTIONARY_E5_QUANT = "embedding_dictionary_e5_base-quant"

# --- Model Names ---
MODEL_NAME_LOCAL   = 'paraphrase-multilingual-mpnet-base-v2'
MODEL_NAME_OPENAI  = "text-embedding-3-small"     # 1536 dims
MODEL_NAME_COHERE  = "embed-multilingual-v3.0"
MODEL_NAME_VOYAGEAI = "voyage-3-large"
MODEL_NAME_GOOGLE  = "text-multilingual-embedding-002"
MODEL_E5_MULTI_BASE = "intfloat__multilingual-e5-base"
MODEL_E5_MULTI_BASE_QUANT = "multilingual-e5-base-quant"

# --- 1. Loading / Clients ---
@st.cache_resource
def load_st_model():
    return SentenceTransformer(MODEL_NAME_LOCAL)

@st.cache_resource
def load_openai_client():
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API key not found in secrets."); return None
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_cohere_client():
    if "COHERE_API_KEY" not in st.secrets:
        st.error("Cohere API key not found in secrets."); return None
    return cohere.Client(st.secrets["COHERE_API_KEY"])

@st.cache_resource
def load_voyageai_client():
    if "VOYAGEAI_API_KEY" not in st.secrets:
        st.error("VoyageAI API key not found in secrets."); return None
    return voyageai.Client(st.secrets["VOYAGEAI_API_KEY"])

@st.cache_resource
def load_google_client():
    try:
        gcp_creds_dict = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(gcp_creds_dict)
        vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION, credentials=credentials)
        return TextEmbeddingModel.from_pretrained(MODEL_NAME_GOOGLE)
    except Exception as e:
        st.error(f"Failed to init Google Client: {e}")
        return None

@st.cache_resource
def load_es_client():
    try:
        return Elasticsearch(
            hosts=[ES_HOST],
            api_key=st.secrets["connections"]["elasticsearch"]["api_key"]
        )
    except Exception as e:
        st.error(f"Failed to connect ES: {e}")
        return None

# --- 2. Search Functions ---
def find_similar_titles_api(query_title: str, top_k: int):
    payload = {"mode": "synonyms", "inputs": [{"input_type": "titles", "input_value": query_title, "size": top_k}]}
    try:
        response = requests.post(LOCAL_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        for result in data.get("results", []):
            if result.get("input_type") == "titles":
                return [{"title": v, "similarity_score": None} for v in result.get("variants", [])]
        return []
    except Exception as e:
        st.error(f"Custom API Error: {e}")
        return []

def find_similar_titles_openai(es, client, query_title: str, top_k: int):
    if not es or not client: return []
    try:
        query_vector = client.embeddings.create(input=[query_title], model=MODEL_NAME_OPENAI).data[0].embedding
        knn_query = {"field": "title_embedding", "query_vector": query_vector, "k": top_k, "num_candidates": 100}
        resp = es.search(index=TITLES_DICTIONARY_OPENAI, knn=knn_query, size=top_k)
        return [{"title": h["_source"]["title"], "similarity_score": h["_score"]} for h in resp['hits']['hits']]
    except Exception as e:
        st.error(f"OpenAI Error: {e}")
        return []

def find_similar_titles_cohere(es, client, query_title: str, top_k: int):
    if not es or not client: return []
    try:
        resp = client.embed(texts=[query_title], model=MODEL_NAME_COHERE, input_type="search_query")
        query_vector = resp.embeddings[0]
        knn_query = {"field": "title_embedding", "query_vector": query_vector, "k": top_k, "num_candidates": 100}
        resp = es.search(index=TITLES_DICTIONARY_COHERE, knn=knn_query, size=top_k)
        return [{"title": h["_source"]["title"], "similarity_score": h["_score"]} for h in resp['hits']['hits']]
    except Exception as e:
        st.error(f"Cohere Error: {e}"); return []

def find_similar_titles_voyage(es, client, query_title: str, top_k: int):
    if not es or not client: return []
    try:
        resp = client.embed(texts=[query_title], model=MODEL_NAME_VOYAGEAI, input_type="query")
        query_vector = resp.embeddings[0]
        knn_query = {"field": "title_embedding", "query_vector": query_vector, "k": top_k, "num_candidates": 100}
        resp = es.search(index=TITLES_DICTIONARY_VOYAGEAI, knn=knn_query, size=top_k)
        return [{"title": h["_source"]["title"], "similarity_score": h["_score"]} for h in resp['hits']['hits']]
    except Exception as e:
        st.error(f"VoyageAI Error: {e}"); return []

def find_similar_titles_google(es, model, query_title: str, top_k: int):
    if not es or not model: return []
    try:
        resp = model.get_embeddings(texts=[query_title])
        query_vector = resp[0].values
        knn_query = {"field": "title_embedding", "query_vector": query_vector, "k": top_k, "num_candidates": 100}
        resp = es.search(index=TITLES_DICTIONARY_GOOGLE, knn=knn_query, size=top_k)
        return [{"title": h["_source"]["title"], "similarity_score": h["_score"]} for h in resp['hits']['hits']]
    except Exception as e:
        st.error(f"Google Error: {e}"); return []

def find_similar_titles_e5(es, query_title: str, top_k: int, index_name: str, model_id: str):
    if not es: return []
    try:
        knn_query = {
            "field": "title_embedding",
            "query_vector_builder": {
                "text_embedding": {
                    "model_id": model_id,
                    "model_text": query_title
                }
            },
            "k": top_k,
            "num_candidates": 100
        }
        resp = es.search(index=index_name, knn=knn_query, size=top_k)
        return [{"title": h["_source"]["title"], "similarity_score": h["_score"]}
                for h in resp['hits']['hits']]
    except Exception as e:
        st.error(f"E5 Error: {e}")
        return []

# --- 3. UI ---
st.title("Job Title Similarity Comparison")
st.subheader("Search Parameters")

col1, col2, col3 = st.columns([4,2,1])
with col1:
    query_title_input = st.text_input("Job Title", value="senior java developer", label_visibility="collapsed")
with col2:
    top_k_input = st.slider("Number of Results", min_value=5, max_value=100, value=20, label_visibility="collapsed")
with col3:
    search_button = st.button("Run Comparison", type="primary", use_container_width=True)

st.markdown("---")

if search_button and query_title_input:
    es         = load_es_client()
    client_oa  = load_openai_client()
    client_coh = load_cohere_client()
    client_voy = load_voyageai_client()
    client_g   = load_google_client()
    model_loc  = load_st_model()

    with st.spinner("Running searches across all models..."):
        results_api     = find_similar_titles_api(query_title_input, top_k_input)
        results_openai  = find_similar_titles_openai(es, client_oa, query_title_input, top_k_input)
        results_cohere  = find_similar_titles_cohere(es, client_coh, query_title_input, top_k_input)
        results_voyage  = find_similar_titles_voyage(es, client_voy, query_title_input, top_k_input)
        results_google  = find_similar_titles_google(es, client_g, query_title_input, top_k_input)
        results_e5 = find_similar_titles_e5(es, query_title_input, top_k_input,TITLES_DICTIONARY_E5, MODEL_E5_MULTI_BASE)
        results_e5quant = find_similar_titles_e5(es, query_title_input, top_k_input,    TITLES_DICTIONARY_E5_QUANT, MODEL_E5_MULTI_BASE_QUANT)

        # results_e5      = find_similar_titles_e5(es, query_title_input, top_k_input, TITLES_DICTIONARY_E5)
        # results_e5quant = find_similar_titles_e5(es, query_title_input, top_k_input, TITLES_DICTIONARY_E5_QUANT)

    st.header(f"Comparison Results for: '{query_title_input}'")
    res_cols = st.columns(7)

    with res_cols[0]:
        st.subheader("Actual solution (LLM)")
        if results_api:
            for item in results_api: st.markdown(f"{item['title']}<hr>", unsafe_allow_html=True)
        else: st.warning("No results.")

    with res_cols[1]:
        st.subheader("OpenAI")
        if results_openai:
            for it in results_openai: st.markdown(f"**{it['similarity_score']:.4f}** | {it['title']}<hr>", unsafe_allow_html=True)
        else: st.warning("No results.")

    with res_cols[2]:
        st.subheader("Cohere")
        if results_cohere:
            for it in results_cohere: st.markdown(f"**{it['similarity_score']:.4f}** | {it['title']}<hr>", unsafe_allow_html=True)
        else: st.warning("No results.")

    with res_cols[3]:
        st.subheader("VoyageAI")
        if results_voyage:
            for it in results_voyage: st.markdown(f"**{it['similarity_score']:.4f}** | {it['title']}<hr>", unsafe_allow_html=True)
        else: st.warning("No results.")

    with res_cols[4]:
        st.subheader("Google Vertex AI")
        if results_google:
            for it in results_google: st.markdown(f"**{it['similarity_score']:.4f}** | {it['title']}<hr>", unsafe_allow_html=True)
        else: st.warning("No results.")

    with res_cols[5]:
        st.subheader("E5 Base (Elastic)")
        if results_e5:
            for it in results_e5: st.markdown(f"**{it['similarity_score']:.4f}** | {it['title']}<hr>", unsafe_allow_html=True)
        else: st.warning("No results.")

    with res_cols[6]:
        st.subheader("E5 Base-Quant (Elastic)")
        if results_e5quant:
            for it in results_e5quant: st.markdown(f"**{it['similarity_score']:.4f}** | {it['title']}<hr>", unsafe_allow_html=True)
        else: st.warning("No results.")
else:
    st.info("Enter a job title and click *Run Comparison* to see results.")
