import streamlit as st
import os
import requests
from elasticsearch import Elasticsearch
import cohere
import voyageai
import vertexai
import json # <--- IMPORT THE JSON LIBRARY
from google.oauth2 import service_account
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from vertexai.language_models import TextEmbeddingModel

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Job Title Similarity Comparison")

# --- 1. CONFIGURATION (Now from st.secrets) ---
try:
    ES_HOST = st.secrets["connections"]["elasticsearch"]["host"]
    LOCAL_API_URL = st.secrets["connections"]["custom_api"]["url"]
    GOOGLE_PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
    GOOGLE_LOCATION = st.secrets["gcp_service_account"]["location"]
except (KeyError, FileNotFoundError):
    st.error("Secrets for Elasticsearch, Custom API, or GCP are not set. Please configure them in your Streamlit Cloud settings.")
    st.stop()

# Index and Model names
TITLES_DICTIONARY_LOCAL = "embedding_dictionary_local_model"
TITLES_DICTIONARY_OPENAI = "embedding_dictionary_openai"
TITLES_DICTIONARY_COHERE = "embedding_dictionary_cohere"
TITLES_DICTIONARY_VOYAGEAI = "embedding_dictionary_voyageai"
TITLES_DICTIONARY_GOOGLE = "embedding_dictionary_google"

MODEL_NAME_LOCAL = 'paraphrase-multilingual-mpnet-base-v2'
MODEL_NAME_OPENAI = "text-embedding-3-small"
MODEL_NAME_COHERE = "embed-multilingual-v3.0"
MODEL_NAME_VOYAGEAI = "voyage-3-large"
MODEL_NAME_GOOGLE = "text-multilingual-embedding-002"

# --- 2. EFFICIENT LOADING WITH STREAMLIT CACHING ---
@st.cache_resource
def load_st_model():
    return SentenceTransformer(MODEL_NAME_LOCAL)

@st.cache_resource
def load_openai_client():
    if "OPENAI_API_KEY" not in st.secrets: st.error("OpenAI API key not found in secrets."); return None
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_cohere_client():
    if "COHERE_API_KEY" not in st.secrets: st.error("Cohere API key not found in secrets."); return None
    return cohere.Client(st.secrets["COHERE_API_KEY"])

@st.cache_resource
def load_voyageai_client():
    if "VOYAGEAI_API_KEY" not in st.secrets: st.error("VoyageAI API key not found in secrets."); return None
    return voyageai.Client(st.secrets["VOYAGEAI_API_KEY"])

# --- THIS FUNCTION IS NOW CORRECTED ---
@st.cache_resource
def load_google_client():
    """Initializes Google Vertex AI client using a service account from st.secrets."""
    try:
        # Get the service account dict from secrets
        gcp_creds_dict = dict(st.secrets["gcp_service_account"])
        # Optional: st.write(gcp_creds_dict)  # For debugging, remove in production

        # Use the dictionary to create credentials
        credentials = service_account.Credentials.from_service_account_info(gcp_creds_dict)

        vertexai.init(project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION, credentials=credentials)
        return TextEmbeddingModel.from_pretrained(MODEL_NAME_GOOGLE)
    except Exception as e:
        st.error(f"Failed to init Google Client. Check GCP secrets & permissions. Error: {e}")
        return None

@st.cache_resource
def load_es_client():
    """Initializes Elasticsearch client using an API Key from st.secrets."""
    try:
        return Elasticsearch(
            hosts=[ES_HOST],
            api_key=st.secrets["connections"]["elasticsearch"]["api_key"]
            # api_key=(
            #     st.secrets["connections"]["elasticsearch"]["id"],
            #     st.secrets["connections"]["elasticsearch"]["api_key"] # This looks for "api_key"
            # )
        )
    except Exception as e:
        st.error(f"Failed to connect to Elasticsearch. Check your host and API Key in secrets. Error: {e}")
        return None

# --- 3. SEARCH FUNCTIONS (No changes needed here) ---
def find_similar_titles_api(query_title: str, top_k: int):
    payload = {"mode": "synonyms", "inputs": [{"input_type": "titles", "input_value": query_title, "size": top_k}]}
    try:
        response = requests.post(LOCAL_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        for result in data.get("results", []):
            if result.get("input_type") == "titles":
                return [{"title": variant, "similarity_score": None} for variant in result.get("variants", [])]
        return []
    except Exception as e:
        st.error(f"Custom API Error: {e}"); return []

def find_similar_titles_local(es, model, query_title: str, top_k: int):
    if not es or not model: return []
    try:
        query_vector = model.encode(query_title)
        knn_query = {"field": "title_embedding", "query_vector": query_vector.tolist(), "k": top_k, "num_candidates": 100}
        response = es.search(index=TITLES_DICTIONARY_LOCAL, knn=knn_query, size=top_k)
        return [{"title": hit['_source']['title'], "similarity_score": hit['_score']} for hit in response['hits']['hits']]
    except Exception as e:
        st.error(f"Sentence Transformer Error: {e}"); return []

def find_similar_titles_openai(es, client, query_title: str, top_k: int):
    if not es or not client: return []
    try:
        query_vector = client.embeddings.create(input=[query_title], model=MODEL_NAME_OPENAI).data[0].embedding
        knn_query = {"field": "title_embedding", "query_vector": query_vector, "k": top_k, "num_candidates": 100}
        response = es.search(index=TITLES_DICTIONARY_OPENAI, knn=knn_query, size=top_k)
        return [{"title": hit['_source']['title'], "similarity_score": hit['_score']} for hit in response['hits']['hits']]
    except Exception as e:
        st.error(f"OpenAI Error: {e}"); return []

def find_similar_titles_cohere(es, client, query_title: str, top_k: int):
    if not es or not client: return []
    try:
        response = client.embed(texts=[query_title], model=MODEL_NAME_COHERE, input_type="search_query")
        query_vector = response.embeddings[0]
        knn_query = {"field": "title_embedding", "query_vector": query_vector, "k": top_k, "num_candidates": 100}
        response = es.search(index=TITLES_DICTIONARY_COHERE, knn=knn_query, size=top_k)
        return [{"title": hit['_source']['title'], "similarity_score": hit['_score']} for hit in response['hits']['hits']]
    except Exception as e:
        st.error(f"Cohere Error: {e}"); return []

def find_similar_titles_voyage(es, client, query_title: str, top_k: int):
    if not es or not client: return []
    try:
        response = client.embed(texts=[query_title], model=MODEL_NAME_VOYAGEAI, input_type="query")
        query_vector = response.embeddings[0]
        knn_query = {"field": "title_embedding", "query_vector": query_vector, "k": top_k, "num_candidates": 100}
        response = es.search(index=TITLES_DICTIONARY_VOYAGEAI, knn=knn_query, size=top_k)
        return [{"title": hit['_source']['title'], "similarity_score": hit['_score']} for hit in response['hits']['hits']]
    except Exception as e:
        st.error(f"Voyage Error: {e}"); return []

def find_similar_titles_google(es, model, query_title: str, top_k: int):
    if not es or not model: return []
    try:
        response = model.get_embeddings(texts=[query_title])
        query_vector = response[0].values
        knn_query = {"field": "title_embedding", "query_vector": query_vector, "k": top_k, "num_candidates": 100}
        response = es.search(index=TITLES_DICTIONARY_GOOGLE, knn=knn_query, size=top_k)
        return [{"title": hit['_source']['title'], "similarity_score": hit['_score']} for hit in response['hits']['hits']]
    except Exception as e:
        st.error(f"Google Vertex AI Error: {e}"); return []

# --- 4. STREAMLIT UI (No changes needed here) ---
st.title("Job Title Similarity Comparison")


st.subheader("Search Parameters")
col_input1, col_input2, col_input3 = st.columns([4, 2, 1])

with col_input1:
    query_title_input = st.text_input("Job Title", value="senior java developer", label_visibility="collapsed", placeholder="Enter a Job Title (e.g., senior java developer)")

with col_input2:
    top_k_input = st.slider("Number of Results", min_value=5, max_value=100, value=15, label_visibility="collapsed")

with col_input3:
    search_button = st.button("Run Comparison", type="primary", use_container_width=True)

st.markdown("---")

if search_button and query_title_input:
    # Load all models and clients
    model_local = load_st_model()
    client_openai = load_openai_client()
    client_cohere = load_cohere_client()
    client_voyageai = load_voyageai_client()
    model_google = load_google_client()
    es = load_es_client()

    with st.spinner("Running all search models... This may take a moment."):
        # Run all searches
        results_api = find_similar_titles_api(query_title_input, top_k_input)
        results_local = find_similar_titles_local(es, model_local, query_title_input, top_k_input)
        results_openai = find_similar_titles_openai(es, client_openai, query_title_input, top_k_input)
        results_cohere = find_similar_titles_cohere(es, client_cohere, query_title_input, top_k_input)
        results_voyageai = find_similar_titles_voyage(es, client_voyageai, query_title_input, top_k_input)
        results_google = find_similar_titles_google(es, model_google, query_title_input, top_k_input)

    st.header(f"Comparison Results for: '{query_title_input}'")

    res_col1, res_col2, res_col3, res_col4, res_col5, res_col6 = st.columns(6)

    with res_col1:
        st.subheader("Local Synonym API")
        if results_api:
            for item in results_api:
                st.markdown(f"{item['title']}<hr>", unsafe_allow_html=True)
        else: st.warning("No results found.")

    with res_col2:
        st.subheader("Sentence Transformer")
        if results_local:
            for item in results_local:
                st.markdown(f"**{item['similarity_score']:.4f}** | {item['title']}<hr>", unsafe_allow_html=True)
        else: st.warning("No results found.")

    with res_col3:
        st.subheader("OpenAI")
        if results_openai:
            for item in results_openai:
                st.markdown(f"**{item['similarity_score']:.4f}** | {item['title']}<hr>", unsafe_allow_html=True)
        else: st.warning("No results found.")

    with res_col4:
        st.subheader("Cohere")
        if results_cohere:
            for item in results_cohere:
                st.markdown(f"**{item['similarity_score']:.4f}** | {item['title']}<hr>", unsafe_allow_html=True)
        else: st.warning("No results found.")

    with res_col5:
        st.subheader("Voyage AI")
        if results_voyageai:
            for item in results_voyageai:
                st.markdown(f"**{item['similarity_score']:.4f}** | {item['title']}<hr>", unsafe_allow_html=True)
        else: st.warning("No results found.")

    with res_col6:
        st.subheader("Google Vertex AI")
        if results_google:
            for item in results_google:
                st.markdown(f"**{item['similarity_score']:.4f}** | {item['title']}<hr>", unsafe_allow_html=True)
        else:
            st.warning("No results found.")
else:
    st.info("Enter a job title above and click 'Run Comparison' to see the results.")
