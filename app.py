
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
from google import genai
import getpass

st.set_page_config(
    page_title="ReformulateAI",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
<style>
.main-header {font-size: 28px; font-weight: 600; color: #1a1a1a; margin-bottom: 4px;}
.sub-header {font-size: 15px; color: #666; margin-bottom: 2rem;}
.result-box {background: #f0f7eb; border-left: 4px solid #3b6d11; padding: 1rem; border-radius: 8px; margin-top: 1rem;}
.stat-box {background: #f7f7f5; border-radius: 8px; padding: 1rem; text-align: center;}
.stat-num {font-size: 24px; font-weight: 600; color: #1a1a1a;}
.stat-lbl {font-size: 12px; color: #888; margin-top: 2px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌿 ReformulateAI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered clean label ingredient substitution for food scientists</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="stat-box"><div class="stat-num">162+</div><div class="stat-lbl">Ingredients indexed</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-box"><div class="stat-num">12</div><div class="stat-lbl">Additive categories</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><div class="stat-num">3</div><div class="stat-lbl">Regulatory markets</div></div>', unsafe_allow_html=True)

st.markdown("---")

with st.sidebar:
    st.markdown("### Settings")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste your API key here")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    Built by a food formulation scientist using AI and a proprietary ingredient database.
    
    Covers substitutions for:
    - Synthetic colours
    - Preservatives
    - Emulsifiers
    - Stabilisers & thickeners
    - Artificial sweeteners
    - Antioxidants
    """)

@st.cache_resource
def load_database():
    df = pd.read_csv("CleanLabel.csv")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client_db = chromadb.Client()
    try:
        collection = client_db.get_collection("ingredients")
    except:
        collection = client_db.create_collection("ingredients")
        for index, row in df.iterrows():
            text = f"""Ingredient: {row["A: Ingredient Name"]}
            Function: {row["E: Primary Function"]}
            Applications: {row["H: Application Categories"]}
            Clean Label Status: {row["O: Clean Label Status"]}
            Alternatives: {row["P: Clean Label Alternatives"]}
            Description: {row["W: Functional Description"]}"""
            vector = embedder.encode(text).tolist()
            collection.add(
                embeddings=[vector],
                documents=[text],
                metadatas=[{
                    "name": str(row["A: Ingredient Name"]),
                    "function": str(row["E: Primary Function"]),
                    "applications": str(row["H: Application Categories"]),
                    "clean_label_status": str(row["O: Clean Label Status"]),
                    "alternatives": str(row["P: Clean Label Alternatives"]),
                    "substitution_ratio": str(row["Q: Substitution Ratio"]),
                    "sensory_impact": str(row["R: Sensory Impact of Substitution"]),
                    "processing": str(row["S: Processing Considerations"]),
                    "ghana_fda": str(row["T: Regulatory Status Ghana FDA"]),
                    "eu_status": str(row["U: Regulatory Status EU"]),
                    "usa_status": str(row["V: Regulatory Status FDA USA"]),
                    "description": str(row["W: Functional Description"])
                }],
                ids=[str(index)]
            )
    return df, embedder, collection

st.markdown("### Reformulation request")

ingredient = st.text_input("Ingredient to replace", placeholder="e.g. Carrageenan (E407)")

col1, col2 = st.columns(2)
with col1:
    application = st.text_input("Product application", placeholder="e.g. Chocolate dairy alternative")
with col2:
    processing = st.text_input("Processing conditions (optional)", placeholder="e.g. UHT 140C, pH 6.8")

function_needed = st.text_area("What function does this ingredient perform?", placeholder="e.g. Stabilises cocoa suspension, prevents phase separation...", height=80)

st.markdown("*Quick examples:*")
ex1, ex2, ex3 = st.columns(3)
if ex1.button("Carrageenan in dairy"):
    ingredient = "Carrageenan (E407)"
    application = "Chocolate dairy alternative"
    function_needed = "Stabilisation and cocoa suspension"
if ex2.button("Sodium benzoate in juice"):
    ingredient = "Sodium Benzoate (E211)"
    application = "Tropical fruit juice"
    function_needed = "Antimicrobial preservation"
if ex3.button("Tartrazine in drink"):
    ingredient = "Tartrazine (E102)"
    application = "Carbonated soft drink"
    function_needed = "Bright yellow colouration"

if st.button("🔍 Get clean label recommendation", type="primary"):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not ingredient or not application or not function_needed:
        st.warning("Please fill in ingredient, application and function.")
    else:
        with st.spinner("Searching ingredient database and generating recommendation..."):
            df, embedder, collection = load_database()
            search_query = f"Replace {ingredient} in {application}. Function: {function_needed}"
            query_vector = embedder.encode(search_query).tolist()
            results = collection.query(query_embeddings=[query_vector], n_results=5)
            context = ""
            for i, meta in enumerate(results["metadatas"][0]):
                context += f"""
CANDIDATE {i+1}: {meta["name"]}
Function: {meta["function"]}
Clean Label Status: {meta["clean_label_status"]}
Alternatives: {meta["alternatives"]}
Substitution Ratio: {meta["substitution_ratio"]}
Sensory Impact: {meta["sensory_impact"]}
Processing Notes: {meta["processing"]}
Ghana FDA: {meta["ghana_fda"]}
EU: {meta["eu_status"]}
USA: {meta["usa_status"]}
---"""
            prompt = f"""You are a senior food formulation scientist with 15 years experience in clean label reformulation.

REFORMULATION REQUEST:
- Replace: {ingredient}
- Product: {application}
- Function needed: {function_needed}
- Processing: {processing if processing else "Not specified"}

CANDIDATES FROM DATABASE:
{context}

Provide recommendations covering:
1. TOP RECOMMENDATION with exact usage level
2. WHY IT WORKS - functional mechanism
3. SUBSTITUTION RATIO - precise replacement ratio
4. SENSORY IMPACT - changes in taste, texture, appearance
5. PROCESSING ADJUSTMENTS needed
6. POTENTIAL CHALLENGES - honest assessment
7. ALTERNATIVE OPTIONS - 2nd and 3rd choices
8. REGULATORY NOTE - Ghana FDA, EU, USA status"""

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

        st.success("Recommendation ready")
        st.markdown("### Reformulation Recommendation")
        st.markdown(response.text)
        st.markdown("---")
        st.caption("For formulation guidance only. Always validate through bench trials before commercialisation.")

st.markdown("---")
st.caption("ReformulateAI · Built with food science expertise + AI · Powered by Gemini")
