import streamlit as st
import json
import time
import pandas as pd
from datetime import datetime
import os
import sys

# Add the parent directory to sys.path to import the search function
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the search function and dependencies
from elasticsearch import Elasticsearch
import requests
import math
from voyageai import Client as VoyageClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Voyage AI Semantic Search Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .search-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    # .result-card {
    #     background-color: green;
    #     padding: 1rem;
    #     border-radius: 8px;
    #     border-left: 4px solid green;
    #     margin-bottom: 1rem;
    #     box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    # }
    .metric-container {
        background-color: grey;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .result-header {
        background-color: #28a745;
        color: white;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .result-number {
        font-size: 1.1rem;
    }
    .result-score {
        font-size: 1.1rem;
    }
</style>

<script>
// Enhanced Ctrl+Enter handler for Streamlit forms
let keyHandler = function(event) {
    if (event.ctrlKey && event.key === 'Enter') {
        event.preventDefault();
        event.stopPropagation();
        
        // Try multiple selectors for form submit buttons
        const selectors = [
            'button[data-testid="baseButton-formSubmit"]',
            'button[kind="formSubmit"]', 
            'form button[type="submit"]',
            'form .stButton button',
            'button:contains("🚀 Search")',
            '.stForm button[data-testid="baseButton-primary"]'
        ];
        
        for (let selector of selectors) {
            try {
                const btn = document.querySelector(selector);
                if (btn && btn.textContent.includes('🚀 Search')) {
                    btn.click();
                    return;
                }
            } catch (e) {
                continue;
            }
        }
        
        // Fallback: look for any button with "Search" text inside a form
        const forms = document.querySelectorAll('form, .stForm');
        for (let form of forms) {
            const searchBtn = Array.from(form.querySelectorAll('button')).find(btn => 
                btn.textContent.includes('🚀 Search') || btn.textContent.includes('Search')
            );
            if (searchBtn) {
                searchBtn.click();
                return;
            }
        }
    }
};

// Attach to document
document.addEventListener('keydown', keyHandler, true);

// Also attach to textareas specifically
setTimeout(() => {
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        textarea.addEventListener('keydown', keyHandler, true);
    });
}, 1000);

// Re-attach handlers when Streamlit updates the DOM
const observer = new MutationObserver(() => {
    const textareas = document.querySelectorAll('textarea:not([data-key-handler])');
    textareas.forEach(textarea => {
        textarea.setAttribute('data-key-handler', 'true');
        textarea.addEventListener('keydown', keyHandler, true);
    });
});

observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'trigger_search' not in st.session_state:
    st.session_state.trigger_search = False

# Configuration
@st.cache_data
def load_config():
    """Load configuration from environment variables."""
    config = {
        'elastic_host': os.getenv("ES_WIGGLI_VECTOR_HOST"),
        'elastic_api_key': os.getenv("ES_WIGGLI_VECTOR_API_KEY"),
        'voyage_api_key': os.getenv("VOYAGEAI_API_KEY"),
        'index_name': "profile_v7_demo_dev_compl_embedded_voyageai",
        'voyage_model': "voyage-3-large",
        'parser_url': os.getenv("PARSER_URL")
    }
    return config

# Helper functions (copied from the original code)
def cosine_similarity(v1, v2):
    if not v1 or not v2:
        return None
    n = min(len(v1), len(v2))
    v1 = v1[:n]; v2 = v2[:n]
    dot = sum(a*b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a*a for a in v1))
    n2 = math.sqrt(sum(b*b for b in v2))
    if n1 == 0 or n2 == 0:
        return None
    return dot / (n1 * n2)

def extract_values(query_json, key):
    try:
        block = (query_json or {}).get(key) or {}
        return [ (item.get("value") or "").strip()
                 for item in block.get("values", [])
                 if (item.get("value") or "").strip() ]
    except Exception:
        return []

def build_field_query_texts(structured_json, fallback_text, fields_to_embed):
    titles_text = " | ".join(extract_values(structured_json, "titles")) or ""
    skills_text = " | ".join(extract_values(structured_json, "skills")) or ""
    industries_text = " | ".join(extract_values(structured_json, "industries")) or ""
    
    # Get filter value from structured_json
    filter_value = structured_json.get("filter", "all") if structured_json else "all"
    
    # Initialize the result dictionary (only include fields with non-empty content)
    result = {}
    
    # Add skills only if not empty
    if skills_text.strip():
        result[fields_to_embed["skills"]] = skills_text
    
    # Title field is mandatory - stop search if no titles found
    if not titles_text.strip():
        # No titles found - return empty result to stop search
        return {}, True  # Empty result, titles_missing=True
    
    # Apply filter logic for titles (we have valid titles)
    if filter_value == "current":
        # Only search in current titles
        result[fields_to_embed["flat_current_titles"]] = titles_text
    elif filter_value == "past":
        # Only search in past titles
        result[fields_to_embed["flat_past_titles"]] = titles_text
    else:  # filter_value == "all" or any other value
        # Search in both current and past titles
        result[fields_to_embed["flat_current_titles"]] = titles_text
        result[fields_to_embed["flat_past_titles"]] = titles_text
    
    # Apply filter logic for industries (only if industries_text is not empty)
    if industries_text.strip():
        if filter_value == "current":
            # Only search in current industries
            result[fields_to_embed["flat_current_industries"]] = industries_text
        elif filter_value == "past":
            # Only search in past industries
            result[fields_to_embed["flat_past_industries"]] = industries_text
        else:  # filter_value == "all" or any other value
            # Search in both current and past industries
            result[fields_to_embed["flat_current_industries"]] = industries_text
            result[fields_to_embed["flat_past_industries"]] = industries_text
    
    return result, False  # result, titles_missing=False

def get_profile_id(src: dict):
    for k in ("profile_id", "profileId", "id", "profileID"):
        v = src.get(k)
        if v not in (None, ""):
            return v
    return None

# Main search function
def semantic_profile_search(user_query, title_threshold, skill_threshold, industry_threshold, 
                          title_weight, skill_weight, industry_weight, result_size, sort_mode, config):
    """
    Enhanced semantic search function with configurable parameters.
    """
    # Fields mapping
    FIELDS_TO_EMBED = {
        "flat_current_titles": "flat_current_titles_embedding",
        "flat_past_titles": "flat_past_titles_embedding",
        "skills": "skills_embedding",
        "flat_current_industries": "flat_current_industries_embedding",
        "flat_past_industries": "flat_past_industries_embedding"
    }
    
    # KNN parameters
    K = min(200, result_size * 4)
    NUM_CANDIDATES = min(1000, result_size * 10)
    RRF_SIZE = min(500, result_size * 5)
    RRF_WINDOW = min(500, result_size * 5)
    
    try:
        # Parse user query
        with st.spinner("Parsing query..."):
            payload = json.dumps({"job_description": user_query, "type": "prompt"})
            headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
            response = requests.post(config['parser_url'], headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            structured_query_json = response.json()
            print('query JSon after parsing', structured_query_json)
        
        # Initialize clients
        with st.spinner("Connecting to services..."):
            es = Elasticsearch(hosts=[config['elastic_host']], api_key=config['elastic_api_key'])
            if not es.ping():
                raise ConnectionError("Could not connect to Elasticsearch")
            
            voyage_client = VoyageClient(api_key=config['voyage_api_key'])
        
        # Generate embeddings
        with st.spinner("Generating embeddings with Voyage AI..."):
            field_texts, titles_missing = build_field_query_texts(structured_query_json, user_query, FIELDS_TO_EMBED)
            print('====== field_texts after building=== ', field_texts)
            
            # Stop search if no titles found
            if titles_missing:
                st.error("❌ **Search stopped**: No job titles found in your query. Please specify job titles to search for candidates.")
                st.info("💡 **Example**: Try queries like 'java developer', 'senior engineer', 'project manager', etc.")
                return [], 0, {
                    "filter_value": structured_query_json.get("filter", "all") if structured_query_json else "all",
                    "error": "no_titles_found",
                    "extracted_query": structured_query_json,
                    "field_texts": {}
                }
            
            # Get unique texts and generate embeddings (filter out empty strings)
            texts_to_embed = list(set(text for text in field_texts.values() if text.strip()))
            
            # Validate that we have texts to embed (titles are mandatory, so this should not happen)
            if not texts_to_embed:
                raise ValueError("No valid text found for embedding generation - this should not happen since titles are mandatory")
            
            embeddings_response = voyage_client.embed(
                texts=texts_to_embed,
                model=config['voyage_model']
            )
            
            # Map embeddings to fields
            text_to_embedding = dict(zip(texts_to_embed, embeddings_response.embeddings))
            vector_by_field = {}
            for field_name, text in field_texts.items():
                if text.strip() and text in text_to_embedding:
                    vector_by_field[field_name] = text_to_embedding[text]
        
        # Perform search
        with st.spinner("Searching profiles..."):
            # Validate that we have vectors for search
            if not vector_by_field:
                raise ValueError("No valid embeddings generated for search")
            
            knn_retrievers = []
            for vector_field, qvec in vector_by_field.items():
                if isinstance(qvec, list) and len(qvec) > 0:
                    knn_retrievers.append({
                        "knn": {
                            "field": vector_field,
                            "query_vector": qvec,
                            "k": K,
                            "num_candidates": NUM_CANDIDATES
                        }
                    })
            
            if not knn_retrievers:
                raise ValueError("No valid KNN retrievers created for search")
            
            fields_to_return = [
                "title", "headline", "flat_current_titles", "flat_past_titles",
                "flat_current_industries", "flat_past_industries", "skills", 
                "ai_summary_for_embedding", "profile_id", "profileId", "id", "profileID"
            ] + list(vector_by_field.keys())
            
            sem_response = es.search(
                index=config['index_name'],
                retriever={"rrf": {"retrievers": knn_retrievers, "rank_window_size": RRF_WINDOW}},
                source=fields_to_return,
                size=RRF_SIZE
            )
            
            hits = sem_response.get("hits", {}).get("hits", [])
        
        # Process results
        with st.spinner("Processing results..."):
            def field_cosines(src):
                # Calculate title similarities (only for fields that exist in vector_by_field)
                title_scores = []
                if FIELDS_TO_EMBED["flat_current_titles"] in vector_by_field:
                    t1 = cosine_similarity(vector_by_field[FIELDS_TO_EMBED["flat_current_titles"]], 
                                         src.get(FIELDS_TO_EMBED["flat_current_titles"]))
                    if t1 is not None:
                        title_scores.append(t1)
                
                if FIELDS_TO_EMBED["flat_past_titles"] in vector_by_field:
                    t2 = cosine_similarity(vector_by_field[FIELDS_TO_EMBED["flat_past_titles"]], 
                                         src.get(FIELDS_TO_EMBED["flat_past_titles"]))
                    if t2 is not None:
                        title_scores.append(t2)
                
                best_title = max(title_scores) if title_scores else None
                
                # Calculate skill similarity (only if skills were searched)
                sk = None
                if FIELDS_TO_EMBED["skills"] in vector_by_field:
                    profile_skills_embedding = src.get(FIELDS_TO_EMBED["skills"])
                    profile_skills_raw = src.get("skills", "")
                    
                    # Handle skills field whether it's string or list
                    if isinstance(profile_skills_raw, list):
                        profile_skills_text = ", ".join(str(skill) for skill in profile_skills_raw if skill)
                    else:
                        profile_skills_text = str(profile_skills_raw) if profile_skills_raw else ""
                    
                    sk = cosine_similarity(vector_by_field[FIELDS_TO_EMBED["skills"]], 
                                         profile_skills_embedding)
                    
                    # Debug: Print when there's a mismatch
                    if sk is not None and sk > 0 and not profile_skills_text.strip():
                        print(f"DEBUG: Skill similarity {sk:.4f} but empty skills text. Profile has embedding: {profile_skills_embedding is not None}")
                
                # Calculate industry similarities (only for fields that exist in vector_by_field)
                industry_scores = []
                if FIELDS_TO_EMBED["flat_current_industries"] in vector_by_field:
                    i1 = cosine_similarity(vector_by_field[FIELDS_TO_EMBED["flat_current_industries"]], 
                                         src.get(FIELDS_TO_EMBED["flat_current_industries"]))
                    if i1 is not None:
                        industry_scores.append(i1)
                
                if FIELDS_TO_EMBED["flat_past_industries"] in vector_by_field:
                    i2 = cosine_similarity(vector_by_field[FIELDS_TO_EMBED["flat_past_industries"]], 
                                         src.get(FIELDS_TO_EMBED["flat_past_industries"]))
                    if i2 is not None:
                        industry_scores.append(i2)
                
                best_ind = max(industry_scores) if industry_scores else None
                
                return best_title, sk, best_ind
            
            results = []
            for h in hits:
                src = h.get("_source", {}) or {}
                title_cos, skill_cos, ind_cos = field_cosines(src)
                
                # Apply thresholds only for fields that were actually searched
                threshold_passed = True
                
                # Check title threshold - always required since title field is mandatory
                if title_cos is None or title_cos < title_threshold:
                    threshold_passed = False
                
                # Check skill threshold only if skills were searched
                if FIELDS_TO_EMBED["skills"] in vector_by_field:
                    # First check if profile has actual skills text
                    profile_skills = src.get('skills', '')
                    if isinstance(profile_skills, list):
                        has_skills_text = bool(profile_skills and any(str(s).strip() for s in profile_skills))
                    else:
                        has_skills_text = bool(str(profile_skills).strip())
                    
                    # If query has skills but profile doesn't, skip this profile
                    if not has_skills_text:
                        threshold_passed = False
                    elif skill_cos is None or skill_cos < skill_threshold:
                        threshold_passed = False
                
                # Check industry threshold only if industries were searched
                if any(field in vector_by_field for field in [FIELDS_TO_EMBED["flat_current_industries"], FIELDS_TO_EMBED["flat_past_industries"]]):
                    # First check if profile has actual industry text
                    profile_current_industries = src.get('flat_current_industries', '')
                    profile_past_industries = src.get('flat_past_industries', '')
                    has_industry_text = bool(str(profile_current_industries).strip() or str(profile_past_industries).strip())
                    
                    # If query has industries but profile doesn't, skip this profile
                    if not has_industry_text:
                        threshold_passed = False
                    elif ind_cos is None or ind_cos < industry_threshold:
                        threshold_passed = False
                
                if not threshold_passed:
                    continue
                
                # Calculate weighted score
                weighted = 0.0
                total_weight = 0.0
                
                # Include weights for all searched fields (profiles already filtered for required text)
                if title_cos is not None: 
                    weighted += title_weight * title_cos
                    total_weight += title_weight
                
                if skill_cos is not None: 
                    weighted += skill_weight * skill_cos
                    total_weight += skill_weight
                
                if ind_cos is not None: 
                    weighted += industry_weight * ind_cos
                    total_weight += industry_weight
                
                # Normalize by the actual total weight of searched fields only
                if total_weight > 0:
                    weighted = weighted / total_weight
                else:
                    weighted = 0.0
                # Clean source data
                redacted = {k: v for k, v in src.items() if k not in vector_by_field}
                
                results.append({
                    "doc_id": h.get("_id"),
                    "profile_id": get_profile_id(src),
                    "rrf_score": h.get("_score"),
                    "weighted_score": weighted,
                    "cosines": {
                        "title": title_cos,
                        "skill": skill_cos,
                        "industry": ind_cos
                    },
                    "source": redacted
                })
            
            # Sort results
            if sort_mode == "cosine":
                results.sort(key=lambda r: r["weighted_score"], reverse=True)
            else:  # "rrf"
                results.sort(key=lambda r: (r["rrf_score"] is None, r["rrf_score"]), reverse=True)
            
            # Trim to requested size
            results = results[:result_size]
            
            # Get filter information for debugging
            filter_info = {
                "filter_value": structured_query_json.get("filter", "all") if structured_query_json else "all",
                "active_fields": list(vector_by_field.keys()),
                "field_texts": field_texts,
                "extracted_query": structured_query_json
            }
            
            return results, len(hits), filter_info
    
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return [], 0, {}

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🔍  Profile Semantic Search Demo (Voyage AI) </h1>', unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    
    # Check configuration - all variables are mandatory
    missing_configs = []
    if not config['elastic_host']:
        missing_configs.append("ES_WIGGLI_VECTOR_HOST")
    if not config['elastic_api_key']:
        missing_configs.append("ES_WIGGLI_VECTOR_API_KEY") 
    if not config['voyage_api_key']:
        missing_configs.append("VOYAGEAI_API_KEY")
    if not config['parser_url']:
        missing_configs.append("PARSER_URL")
    
    if missing_configs:
        st.error(f"❌ Missing required configuration! Please set these environment variables: {', '.join(missing_configs)}")
        st.info("💡 Create a `.env` file based on `env_example.txt` with your actual values.")
        st.stop()
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("🔧 Search Parameters")
        
        # Use form for better key handling
        with st.form(key="search_form", clear_on_submit=False):
            # Search query
            user_query = st.text_area(
                "Search Query",
                placeholder="e.g., je cherche un lead technique java dans la santé",
                height=100,
                help="Enter your search query in natural language. Press Ctrl+Enter to search quickly!",
                key="search_query"
            )
            
            # Keyboard shortcut info
            st.caption("💡 **Tip:** Press `Ctrl + Enter` anywhere to search quickly!")
            
            st.divider()
            
            # Result size
            result_size = st.number_input("Result Size", min_value=1, max_value=50, value=10, step=1,
                                         help="Number of results to return")
            
            st.subheader("🎯 Similarity Thresholds")
            title_threshold = st.number_input("Title Threshold", min_value=0.0, max_value=1.0, 
                                            value=0.6, step=0.05, format="%.2f",
                                            help="Minimum similarity for titles")
            skill_threshold = st.number_input("Skill Threshold", min_value=0.0, max_value=1.0, 
                                            value=0.5, step=0.05, format="%.2f",
                                            help="Minimum similarity for skills")
            industry_threshold = st.number_input("Industry Threshold", min_value=0.0, max_value=1.0, 
                                               value=0.5, step=0.05, format="%.2f",
                                               help="Minimum similarity for industries")
            
            st.subheader("⚖️ Cosine Weights")
            title_weight = st.number_input("Title Weight", min_value=0.1, max_value=5.0, 
                                         value=3.0, step=0.1, format="%.1f",
                                         help="Weight for title similarity")
            skill_weight = st.number_input("Skill Weight", min_value=0.1, max_value=5.0, 
                                         value=2.0, step=0.1, format="%.1f",
                                         help="Weight for skill similarity")
            industry_weight = st.number_input("Industry Weight", min_value=0.1, max_value=5.0, 
                                            value=1.0, step=0.1, format="%.1f",
                                            help="Weight for industry similarity")
            
            st.subheader("📊 Sort Mode")
            sort_mode = st.radio(
                "Sort Results By",
                options=["cosine", "rrf"],
                help="Choose sorting method: cosine (weighted similarity) or rrf (Elasticsearch RRF score)"
            )
            
            st.divider()
            
            # Search button with form submit
            search_clicked = st.form_submit_button("🚀 Search", type="primary", use_container_width=True)
    
    # Main content area
    if search_clicked and user_query.strip():
        start_time = time.time()
        
        # Perform search
        results, total_candidates, filter_info = semantic_profile_search(
            user_query, title_threshold, skill_threshold, industry_threshold,
            title_weight, skill_weight, industry_weight, result_size, sort_mode, config
        )
        
        search_time = time.time() - start_time
        
        # Store results in session state
        st.session_state.search_results = results
        st.session_state.search_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'query': user_query,
            'results_count': len(results),
            'search_time': search_time
        })
        
        # Display search info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔍 Results Found", len(results))
        with col2:
            st.metric("⏱️ Search Time", f"{search_time:.2f}s")
        with col3:
            st.metric("📊 Total Candidates", total_candidates)
        with col4:
            st.metric("✅ Pass Rate", f"{(len(results)/total_candidates*100):.1f}%" if total_candidates > 0 else "0%")
        
        # Display query analysis
        if filter_info:
            with st.expander("🔍 Query Analysis", expanded=False):
                # Check if there was an error (no titles found)
                if filter_info.get('error') == 'no_titles_found':
                    st.markdown("**❌ Search Error:** No titles found in query")
                    if filter_info.get('extracted_query'):
                        st.markdown("**📋 Extracted Query Data:**")
                        st.json(filter_info.get('extracted_query'))
                else:
                    # Normal successful search analysis
                    st.markdown(f"**🎯 Filter Applied:** `{filter_info.get('filter_value', 'all')}`")
                    st.markdown(f"**📊 Active Search Fields:** `{len(filter_info.get('active_fields', []))}`")
                    
                    # Show extracted query data
                    if filter_info.get('extracted_query'):
                        st.markdown("**📋 Extracted Query Data:**")
                        st.json(filter_info.get('extracted_query'))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Active Fields:**")
                        for field in filter_info.get('active_fields', []):
                            field_name = field.split('_embedding')[0]  # Remove '_embedding' suffix
                            st.markdown(f"- `{field_name}`")
                    
                    with col2:
                        st.markdown("**Field Texts:**")
                        st.json(filter_info.get('field_texts', {}))
        
        # Display results
        if results:
            st.subheader(f"📋 Search Results ({len(results)})")
            
            for i, result in enumerate(results, 1):
                with st.container():
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    
                    # Result header in green container
                    st.markdown(f'''
                    <div class="result-header">
                        <div class="result-number">Result #{i}</div>
                        <div class="result-score">Score: {result['weighted_score']:.4f}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Profile ID (if available)
                    if result['profile_id']:
                        st.markdown(f"*Profile ID: {result['profile_id']}*")
                    
                    # Similarity scores
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        title_score = result["cosines"]["title"]
                        title_display = f"{title_score:.4f}" if title_score is not None else "N/A"
                        st.markdown(f'<div class="metric-container">Title: {title_display}</div>', 
                                  unsafe_allow_html=True)
                    with col2:
                        skill_score = result["cosines"]["skill"]
                        skill_display = f"{skill_score:.4f}" if skill_score is not None else "N/A"
                        st.markdown(f'<div class="metric-container">Skill: {skill_display}</div>', 
                                  unsafe_allow_html=True)
                    with col3:
                        industry_score = result["cosines"]["industry"]
                        industry_display = f"{industry_score:.4f}" if industry_score is not None else "N/A"
                        st.markdown(f'<div class="metric-container">Industry: {industry_display}</div>', 
                                  unsafe_allow_html=True)
                    
                    # Profile information
                    source = result['source']
                    
                    if source.get('headline'):
                        st.markdown(f"**Headline:** {source['headline']}")
                    
                    if source.get('flat_current_titles'):
                        st.markdown(f"**Current Titles:** {source['flat_current_titles']}")
                    
                    if source.get('flat_past_titles'):
                        st.markdown(f"**Past Titles:** {source['flat_past_titles']}")
                    
                    if source.get('flat_current_industries'):
                        st.markdown(f"**Current Industries:** {source['flat_current_industries']}")
                    
                    if source.get('skills'):
                        skills_raw = source['skills']
                        
                        # Handle skills whether it's a string or list
                        if isinstance(skills_raw, list):
                            skills_text = ", ".join(str(skill) for skill in skills_raw if skill)
                        else:
                            skills_text = str(skills_raw) if skills_raw else ""
                        
                        # Truncate if too long
                        if len(skills_text) > 200:
                            skills_text = skills_text[:200] + "..."
                        
                        if skills_text.strip():  # Only display if not empty
                            st.markdown(f"**Skills:** {skills_text}")
                    
                    # Full data in expander
                    with st.expander(f"📄 Full Profile Data #{i}"):
                        st.json(source)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("---")
        else:
            st.warning("No results found matching your criteria. Try adjusting the thresholds.")
    
    elif search_clicked:
        st.warning("Please enter a search query.")
    
    # Search history
    if st.session_state.search_history:
        with st.expander("📈 Search History", expanded=False):
            df = pd.DataFrame(st.session_state.search_history)
            st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
