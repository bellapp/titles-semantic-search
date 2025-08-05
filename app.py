import streamlit as st
import os
import requests
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import cohere
import voyageai

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Job Title Similarity Comparison")

# --- 1. CONFIGURATION (Copied from your script) ---
ES_HOST = "http://localhost:9210"
LOCAL_API_URL = "http://localhost:8080/api/intelligence/generate/similars"

# Index names for each model
TITLES_DICTIONARY_LOCAL = "embedding_dictionary_local_model"
TITLES_DICTIONARY_OPENAI = "embedding_dictionary_openai"
TITLES_DICTIONARY_COHERE = "embedding_dictionary_cohere"
TITLES_DICTIONARY_VOYAGEAI = "embedding_dictionary_voyageai"

# Model names
MODEL_NAME_LOCAL = 'paraphrase-multilingual-mpnet-base-v2'
MODEL_NAME_OPENAI = "text-embedding-3-small"
MODEL_NAME_COHERE = "embed-multilingual-v3.0"
MODEL_NAME_VOYAGEAI = "voyage-3-large"
# Load environment variables from .env file
load_dotenv()

# --- 2. EFFICIENT LOADING WITH STREAMLIT CACHING ---
# Cache the models and clients so they don't reload on every interaction.
@st.cache_resource
def load_st_model():
    return SentenceTransformer(MODEL_NAME_LOCAL, use_auth_token=st.secrets["HF_TOKEN"]))

@st.cache_resource
def load_openai_client():
    # api_key = os.environ.get("OPENAI_API_KEY")
    api_key = st.secrets["OPENAI_API_KEY"]
    if not api_key:
        st.error("OPENAI_API_KEY not set in .env file!")
        return None
    return OpenAI(api_key=api_key)

@st.cache_resource
def load_cohere_client():
    # api_key = os.environ.get("COHERE_API_KEY")
    api_key = st.secrets["COHERE_API_KEY"]
    if not api_key:
        st.error("COHERE_API_KEY not set in .env file!")
        return None
    return cohere.Client(api_key)

@st.cache_resource
def load_voyageai_client():
    # api_key = os.environ.get("VOYAGEAI_API_KEY")
    api_key = st.secrets["VOYAGEAI_API_KEY"]
    if not api_key:
        st.error("VOYAGEAI_API_KEY not set in .env file!")
        return None
    return voyageai.Client(api_key)

@st.cache_resource
def load_es_client():
    return Elasticsearch(ES_HOST)

# --- 3. SEARCH FUNCTIONS (Adapted for Streamlit) ---
# These are your functions, slightly adapted to take clients as arguments.

def find_similar_titles_api(query_title: str, top_k: int):
    # ... (Function is identical to the previous script)
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
        st.error(f"Local API Error: {e}")
        return []

def find_similar_titles_local(es, model, query_title: str, top_k: int):
    # ... (Function is identical to the previous script)
    try:
        query_vector = model.encode(query_title)
        knn_query = {"field": "title_embedding", "query_vector": query_vector.tolist(), "k": top_k, "num_candidates": 100}
        response = es.search(index=TITLES_DICTIONARY_LOCAL, knn=knn_query, size = top_k)
        return [{"title": hit['_source']['title'], "similarity_score": hit['_score']} for hit in response['hits']['hits']]
    except Exception as e:
        st.error(f"Sentence Transformer Error: {e}")
        return []

def find_similar_titles_openai(es, client, query_title: str, top_k: int):
    # ... (Function is identical to the previous script)
    if not client: return []
    try:
        query_vector = client.embeddings.create(input=[query_title], model=MODEL_NAME_OPENAI).data[0].embedding
        knn_query = {"field": "title_embedding", "query_vector": query_vector, "k": top_k, "num_candidates": 100}
        response = es.search(index=TITLES_DICTIONARY_OPENAI, knn=knn_query, size = top_k)
        return [{"title": hit['_source']['title'], "similarity_score": hit['_score']} for hit in response['hits']['hits']]
    except Exception as e:
        st.error(f"OpenAI Error: {e}")
        return []

def find_similar_titles_cohere(es, client, query_title: str, top_k: int):
    # ... (Function is identical to the previous script)
    if not client: return []
    try:
        response = client.embed(texts=[query_title], model=MODEL_NAME_COHERE, input_type="search_query")
        query_vector = response.embeddings[0]
        knn_query = {"field": "title_embedding", "query_vector": query_vector, "k": top_k, "num_candidates": 100}
        response = es.search(index=TITLES_DICTIONARY_COHERE, knn=knn_query, size = top_k)
        return [{"title": hit['_source']['title'], "similarity_score": hit['_score']} for hit in response['hits']['hits']]
    except Exception as e:
        st.error(f"Cohere Error: {e}")
        return []

def find_similar_titles_voyage(es, client, query_title: str, top_k: int):
    # ... (Function is identical to the previous script)
    if not client: return []
    try:
        response = client.embed(texts=[query_title], model=MODEL_NAME_VOYAGEAI, input_type="query")
        query_vector = response.embeddings[0]
        knn_query = {"field": "title_embedding", "query_vector": query_vector, "k": top_k, "num_candidates": 100}
        response = es.search(index=TITLES_DICTIONARY_VOYAGEAI, knn=knn_query, size = top_k)
        return [{"title": hit['_source']['title'], "similarity_score": hit['_score']} for hit in response['hits']['hits']]
    except Exception as e:
        st.error(f"Voyage Error: {e}")
        return []
# --- 4. STREAMLIT UI ---
st.title("Job Title Similarity Comparison")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Search Parameters")
    query_title_input = st.text_input("Enter a Job Title:", "senior java developer")
    top_k_input = st.slider("Number of Results (Top K):", min_value=5, max_value=100, value=15)
    search_button = st.button("Run Comparison", type="primary")

# --- Main Area for Results ---
if search_button:
    # Load all models and clients
    model_local = load_st_model()
    client_openai = load_openai_client()
    client_cohere = load_cohere_client()
    client_voyageai = load_voyageai_client()
    es = load_es_client()

    with st.spinner("Running all search models... This may take a moment."):
        # Run all four searches
        results_api = find_similar_titles_api(query_title_input, top_k_input)
        results_local = find_similar_titles_local(es, model_local, query_title_input, top_k_input)
        results_openai = find_similar_titles_openai(es, client_openai, query_title_input, top_k_input)
        results_cohere = find_similar_titles_cohere(es, client_cohere, query_title_input, top_k_input)
        results_voyageai = find_similar_titles_voyage(es, client_voyageai, query_title_input, top_k_input)


    st.header(f"Comparison Results for: '{query_title_input}'")

    # Create 4 columns for the output grid
    col1, col2, col3, col4, col5 = st.columns(5)

    # --- Column 1: Local Synonym API ---
    with col1:
        st.subheader("Local Synonym API")
        if results_api:
            for item in results_api:
                st.markdown(f"{item['title']}")
                st.markdown("---")
        else:
            st.warning("No results found.")

    # --- Column 2: Sentence Transformer (Local) ---
    with col2:
        st.subheader("Sentence Transformer")
        if results_local:
            for item in results_local:
                # st.markdown(f"**{item['similarity_score']:.4f}** | {item['title']}")
                st.markdown(f"{item['title']}")
                st.markdown("---")
        else:
            st.warning("No results found.")

    # --- Column 3: OpenAI (Cloud) ---
    with col3:
        st.subheader("OpenAI")
        if results_openai:
            for item in results_openai:
                # st.markdown(f"**{item['similarity_score']:.4f}** | {item['title']}")
                st.markdown(f"{item['title']}")
                st.markdown("---")
        else:
            st.warning("No results found.")

    # --- Column 4: Cohere (Cloud) ---
    with col4:
        st.subheader("Cohere")
        if results_cohere:
            for item in results_cohere:
                # st.markdown(f"**{item['similarity_score']:.4f}** | {item['title']}")
                st.markdown(f"{item['title']}")
                st.markdown("---")
        else:
            st.warning("No results found.")
    # --- Column 5: Cohere (Cloud) ---
    with col5:
        st.subheader("Voyage AI")
        if results_voyageai:
            for item in results_voyageai:
                # st.markdown(f"**{item['similarity_score']:.4f}** | {item['title']}")
                st.markdown(f"{item['title']}")
                st.markdown("---")
        else:
            st.warning("No results found.")
else:
    st.info("Enter a job title in the sidebar and click 'Run Comparison' to see the results.")
