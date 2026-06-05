import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
from google import genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(
    page_title="ReformulateAI",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
<style>
.main-header {font-size: 28px; font-weight: 600; color: #1a1a1a; margin-bottom: 4px;}
.sub-header {font-size: 15px; color: #666; margin-bottom: 2rem;}
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
    texts = []
    for _, row in df.iterrows():
        text = f"""Ingredient: {row["A: Ingredient Name"]}
        Function: {row["E: Primary Function"]}
        Applications: {row["H: Application Categories"]}
        Clean Label Status: {row["O: Clean Label Status"]}
        Alternatives: {row["P: Clean Label Alternatives"]}
        Description: {row["W: Functional Description"]}"""
        texts.append(text)
    embeddings = embedder.encode(texts)
    return df, embedder, embeddings

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
            df, embedder, embeddings = load_database()
            search_query = f"Replace {ingredient} in {application}. Function: {function_needed}"
            query_vector = embedder.encode([search_query])
            similarities = cosine_similarity(query_vector, embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:5]
            context = ""
            for i in top_indices:
                row = df.iloc[i]
                context += f"""
CANDIDATE: {row["A: Ingredient Name"]}
Function: {row["E: Primary Function"]}
Clean Label Status: {row["O: Clean Label Status"]}
Alternatives: {row["P: Clean Label Alternatives"]}
Substitution Ratio: {row["Q: Substitution Ratio"]}
Sensory Impact: {row["R: Sensory Impact of Substitution"]}
Processing Notes: {row["S: Processing Considerations"]}
Ghana FDA: {row["T: Regulatory Status Ghana FDA"]}
EU: {row["U: Regulatory Status EU"]}
USA: {row["V: Regulatory Status FDA USA"]}
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
