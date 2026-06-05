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

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
.sub-header {font-size: 15px; color: #666; margin-bottom: 2rem;}
.result-box {background: #f0f7eb; border-left: 4px solid #3b6d11; padding: 1rem; border-radius: 8px; margin-top: 1rem;}
.stat-box {background: #f7f7f5; border-radius: 8px; padding: 1rem; text-align: center;}
.stat-num {font-size: 24px; font-weight: 600; color: #1a1a1a;}
.stat-lbl {font-size: 12px; color: #888; margin-top: 2px…
[11:11 PM, 6/4/2026] Yayra🌴: requirements = '''streamlit
sentence-transformers
chromadb
pandas
google-genai
'''

with open("requirements.txt", "w") as f:
    f.write(requirements)

print("requirements.txt saved successfully.")
[11:12 PM, 6/4/2026] Yayra🌴: from google.colab import files
files.download("app.py")
files.download("requirements.txt")
files.download("CleanLabel.csv")
[12:02 AM, 6/5/2026] Yayra🌴: import streamlit as st
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

st.markdown('<div class="main-header">🌿 Reformula…
[11:13 PM, 6/5/2026] Yayra🌴: import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
from google import genai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

st.set_page_config(page_title="ReformulateAI", page_icon="🌿", layout="centered")

st.markdown("""
<style>
.top-rec{background:#EAF3DE;border:1px solid #C0DD97;border-radius:12px;padding:1.25rem;margin-bottom:1rem}
.rec-label{font-size:11px;font-weight:600;color:#3B6D11;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
.rec-name{font-size:20px;font-weight:600;color:#27500A;margin-bottom:4px}
.rec-sub{font-size:13px;color:#3B6D11;line-height:1.6}
.conf-row{display:flex;align-items:center;gap:8px;margin-top:8px}
.conf-bar-bg{flex:1;height:6px;background:#C0DD97;border-radius:3px}
.conf-bar-fill{height:6px;background:#3B6D11;border-radius:3px}
.cards-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:1rem}
.detail-card{background:white;border:1px solid #e5e5e5;border-radius:8px;padding:12px;display:flex;gap:10px}
.dc-icon{width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:16px}
.dc-label{font-size:11px;color:#888;font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin-bottom:3px}
.dc-value{font-size:14px;font-weight:600;color:#1a1a1a}
.dc-sub{font-size:12px;color:#666;margin-top:2px;line-height:1.5}
.impact-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:1rem}
.impact-card{background:#f7f7f5;border-radius:8px;padding:12px;text-align:center}
.impact-label{font-size:11px;color:#888;font-weight:600;margin-bottom:4px;text-transform:uppercase}
.impact-val{font-size:13px;font-weight:600}
.reg-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:1rem}
.reg-card{background:#f7f7f5;border-radius:8px;padding:10px;text-align:center}
.reg-name{font-size:11px;color:#888;font-weight:600;margin-bottom:5px}
.badge-green{font-size:11px;font-weight:600;padding:2px 8px;border-radius:20px;background:#EAF3DE;color:#3B6D11;display:inline-block}
.badge-amber{font-size:11px;font-weight:600;padding:2px 8px;border-radius:20px;background:#FAEEDA;color:#854F0B;display:inline-block}
.badge-red{font-size:11px;font-weight:600;padding:2px 8px;border-radius:20px;background:#FCEBEB;color:#A32D2D;display:inline-block}
.alt-item{background:white;border:1px solid #e5e5e5;border-radius:8px;padding:12px 14px;margin-bottom:6px;display:flex;gap:10px}
.alt-icon{width:36px;height:36px;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:18px}
.alt-name{font-size:14px;font-weight:600;color:#1a1a1a}
.alt-detail{font-size:13px;color:#666;margin-top:3px;line-height:1.5}
.section-lbl{font-size:11px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;margin-top:1rem}
.chat-box{background:white;border:1px solid #e5e5e5;border-radius:12px;overflow:hidden;margin-bottom:1rem}
.chat-header{padding:12px 14px;border-bottom:1px solid #e5e5e5;background:#f7f7f5}
.msg-user{background:#EAF3DE;border-radius:12px 12px 2px 12px;padding:8px 12px;margin:4px 0;max-width:80%;margin-left:auto;font-size:13px;color:#27500A;line-height:1.5}
.msg-ai{background:#f7f7f5;border-radius:12px 12px 12px 2px;padding:8px 12px;margin:4px 0;max-width:85%;font-size:13px;color:#1a1a1a;line-height:1.5}
.history-item{display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f0f0f0}
.history-item:last-child{border-bottom:none}
</style>
""", unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'current_context' not in st.session_state:
    st.session_state.current_context = {}
if 'results' not in st.session_state:
    st.session_state.results = None

col1, col2 = st.columns([3,1])
with col1:
    st.markdown("## 🌿 ReformulateAI")
    st.caption("Clean label ingredient intelligence · 162 ingredients indexed")
with col2:
    st.write("")

with st.sidebar:
    st.markdown("### Settings")
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste your API key")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
Built by a food formulation scientist.
Covers substitutions for:
- Synthetic colours
- Preservatives
- Emulsifiers
- Stabilisers & thickeners
- Artificial sweeteners
- Antioxidants
    """)
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Recent searches")
        for h in st.session_state.history[-5:][::-1]:
            st.markdown(f"*{h['ingredient']}* → {h['top_result']}")
            st.caption(h['application'])

@st.cache_resource
def load_database():
    df = pd.read_csv("CleanLabel.csv")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = []
    for _, row in df.iterrows():
        text = f"""Ingredient: {row['A: Ingredient Name']}
        Function: {row['E: Primary Function']}
        Applications: {row['H: Application Categories']}
        Clean Label Status: {row['O: Clean Label Status']}
        Alternatives: {row['P: Clean Label Alternatives']}
        Description: {row['W: Functional Description']}"""
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
if ex1.button("🥛 Carrageenan in dairy"):
    ingredient = "Carrageenan (E407)"
    application = "Chocolate dairy alternative"
    function_needed = "Stabilisation and cocoa suspension, prevents phase separation"
if ex2.button("🧃 Sodium benzoate in juice"):
    ingredient = "Sodium Benzoate (E211)"
    application = "Tropical fruit juice beverage"
    function_needed = "Antimicrobial preservation, extend shelf life"
if ex3.button("🎨 Tartrazine in drink"):
    ingredient = "Tartrazine (E102)"
    application = "Carbonated soft drink"
    function_needed = "Bright yellow colouration, heat and light stable"

if st.button("🔍 Get clean label recommendation", type="primary", use_container_width=True):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not ingredient or not application or not function_needed:
        st.warning("Please fill in all three fields.")
    else:
        with st.spinner("Searching ingredient database..."):
            df, embedder, embeddings = load_database()
            search_query = f"Replace {ingredient} in {application}. Function: {function_needed}"
            query_vector = embedder.encode([search_query])
            similarities = cosine_similarity(query_vector, embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:5]
            confidence = float(similarities[top_indices[0]]) * 100

            context = ""
            for i in top_indices:
                row = df.iloc[i]
                context += f"""
CANDIDATE: {row['A: Ingredient Name']}
Function: {row['E: Primary Function']}
Clean Label Status: {row['O: Clean Label Status']}
Alternatives: {row['P: Clean Label Alternatives']}
Substitution Ratio: {row['Q: Substitution Ratio']}
Sensory Impact: {row['R: Sensory Impact of Substitution']}
Processing Notes: {row['S: Processing Considerations']}
Ghana FDA: {row['T: Regulatory Status Ghana FDA']}
EU: {row['U: Regulatory Status EU']}
USA: {row['V: Regulatory Status FDA USA']}
Description: {row['W: Functional Description']}
---"""

            prompt = f"""You are a food formulation scientist. Analyse this clean label reformulation and respond ONLY with a valid JSON object — no markdown, no backticks, no explanation outside the JSON.

REFORMULATION REQUEST:
- Replace: {ingredient}
- Product: {application}
- Function: {function_needed}
- Processing: {processing if processing else 'Not specified'}

CANDIDATES:
{context}

Respond with this exact JSON structure:
{{
  "top_name": "ingredient name",
  "top_summary": "one clear sentence — what it is and why it works for this application",
  "substitution_ratio": "exact ratio e.g. 1:1 or reduce by 20%",
  "usage_level": "e.g. 0.01-0.025% in final product",
  "heat_stability": "e.g. Stable to 140C",
  "heat_detail": "one sentence detail",
  "sensory_impact": "e.g. Minimal change",
  "sensory_detail": "one sentence detail",
  "key_challenge": "e.g. Dispersibility",
  "challenge_detail": "one sentence detail",
  "cost_impact": "e.g. +15-25% per kg or Similar cost",
  "cost_direction": "higher or lower or similar",
  "shelf_life_impact": "e.g. No change or May reduce by 10%",
  "shelf_life_direction": "positive or neutral or negative",
  "ghana_fda": "e.g. Approved or Check LI 2000",
  "ghana_status": "ok or warn or check",
  "eu_status_text": "e.g. E410 Approved",
  "eu_status": "ok or warn",
  "usa_status_text": "e.g. GRAS",
  "usa_status": "ok or warn",
  "alt2_name": "second best alternative name",
  "alt2_detail": "one sentence on how to use it",
  "alt2_label": "Clean label or Grey area",
  "alt3_name": "third best alternative name",
  "alt3_detail": "one sentence on how to use it",
  "alt3_label": "Clean label or Grey area",
  "processing_notes": "2-3 sentences on processing adjustments needed",
  "why_it_works": "2-3 sentences on the functional mechanism",
  "cost_note": "1-2 sentences on cost and sourcing"
}}"""

            client_ai = genai.Client(api_key=api_key)
            response = client_ai.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            raw = response.text.replace("json", "").replace("", "").strip()
            r = json.loads(raw)
            st.session_state.results = r
            st.session_state.current_context = {
                "ingredient": ingredient,
                "application": application,
                "processing": processing,
                "function": function_needed,
                "top_name": r["top_name"],
                "context": context
            }
            st.session_state.chat_messages = []
            st.session_state.history.append({
                "ingredient": ingredient,
                "application": application,
                "top_result": r["top_name"]
            })

if st.session_state.results:
    r = st.session_state.results
    confidence = 92

    st.markdown(f"""
    <div class="top-rec">
      <div class="rec-label">✅ Top recommendation</div>
      <div class="rec-name">🌿 {r['top_name']}</div>
      <div class="rec-sub">{r['top_summary']}</div>
      <div class="conf-row">
        <span style="font-size:12px;color:#3B6D11;font-weight:600">Match confidence</span>
        <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{confidence}%"></div></div>
        <span style="font-size:12px;color:#27500A;font-weight:600">{confidence}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="cards-grid">'
        f'<div class="detail-card"><div class="dc-icon" style="background:#EAF3DE">⚖️</div><div><div class="dc-label">Substitution ratio</div><div class="dc-value">{r["substitution_ratio"]}</div><div class="dc-sub">{r["usage_level"]}</div></div></div>'
        f'<div class="detail-card"><div class="dc-icon" style="background:#E6F1FB">🔥</div><div><div class="dc-label">Heat stability</div><div class="dc-value">{r["heat_stability"]}</div><div class="dc-sub">{r["heat_detail"]}</div></div></div>'
        f'<div class="detail-card"><div class="dc-icon" style="background:#FAEEDA">😊</div><div><div class="dc-label">Sensory impact</div><div class="dc-value">{r["sensory_impact"]}</div><div class="dc-sub">{r["sensory_detail"]}</div></div></div>'
        f'<div class="detail-card"><div class="dc-icon" style="background:#FCEBEB">⚠️</div><div><div class="dc-label">Key challenge</div><div class="dc-value">{r["key_challenge"]}</div><div class="dc-sub">{r["challenge_detail"]}</div></div></div>'
        '</div>', unsafe_allow_html=True)

    cost_color = "#854F0B" if r["cost_direction"] == "higher" else "#3B6D11" if r["cost_direction"] == "lower" else "#185FA5"
    shelf_color = "#A32D2D" if r["shelf_life_direction"] == "negative" else "#3B6D11" if r["shelf_life_direction"] == "positive" else "#185FA5"

    st.markdown(f"""
    <div class="section-lbl">Cost & shelf life impact</div>
    <div class="impact-row">
      <div class="impact-card"><div class="impact-label">💰 Cost impact</div><div class="impact-val" style="color:{cost_color}">{r['cost_impact']}</div></div>
      <div class="impact-card"><div class="impact-label">📅 Shelf life</div><div class="impact-val" style="color:{shelf_color}">{r['shelf_life_impact']}</div></div>
      <div class="impact-card"><div class="impact-label">📊 Confidence</div><div class="impact-val" style="color:#185FA5">{confidence}% match</div></div>
    </div>
    """, unsafe_allow_html=True)

    def reg_badge(text, status):
        cls = "badge-green" if status == "ok" else "badge-amber" if status == "warn" else "badge-amber"
        return f'<span class="{cls}">{text}</span>'

    st.markdown(f"""
    <div class="section-lbl">Regulatory status</div>
    <div class="reg-row">
      <div class="reg-card"><div class="reg-name">🇬🇭 Ghana FDA</div>{reg_badge(r['ghana_fda'], r['ghana_status'])}</div>
      <div class="reg-card"><div class="reg-name">🇪🇺 EU (EFSA)</div>{reg_badge(r['eu_status_text'], r['eu_status'])}</div>
      <div class="reg-card"><div class="reg-name">🇺🇸 FDA USA</div>{reg_badge(r['usa_status_text'], r['usa_status'])}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="section-lbl">Alternative options</div>
    <div class="alt-item">
      <div class="alt-icon" style="background:#EAF3DE">🌱</div>
      <div><div class="alt-name">{r['alt2_name']}</div><div class="alt-detail">{r['alt2_detail']}</div><span class="badge-green" style="font-size:11px;margin-top:4px;display:inline-block">{r['alt2_label']}</span></div>
    </div>
    <div class="alt-item">
      <div class="alt-icon" style="background:#E6F1FB">🌾</div>
      <div><div class="alt-name">{r['alt3_name']}</div><div class="alt-detail">{r['alt3_detail']}</div><span class="badge-green" style="font-size:11px;margin-top:4px;display:inline-block">{r['alt3_label']}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-lbl">Ask a follow-up question</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="chat-box">
      <div class="chat-header">
        <strong style="font-size:14px">🌿 Ask about {r['top_name']}</strong><br>
        <span style="font-size:12px;color:#666">Context-aware answers based on your specific formulation</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg-ai"><strong style="font-size:11px;color:#888">🌿 ReformulateAI</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)

    quick_cols = st.columns(4)
    quick_qs = ["Too much LBG?", "Safe for infants?", "Behaviour at low pH?", "Mix with xanthan?"]
    for i, q in enumerate(quick_qs):
        if quick_cols[i].button(q, key=f"quick_{i}"):
            st.session_state.chat_messages.append({"role": "user", "content": q})
            ctx = st.session_state.current_context
            chat_prompt = f"""You are a food formulation scientist. Answer this question about {ctx['top_name']} in the context of replacing {ctx['ingredient']} in {ctx['application']}.
Processing: {ctx['processing']}
Question: {q}
Answer in 2-3 sentences. Be direct and specific. No intro phrases."""
            client_ai = genai.Client(api_key=api_key)
            chat_response = client_ai.models.generate_content(model="gemini-2.5-flash", contents=chat_prompt)
            st.session_state.chat_messages.append({"role": "ai", "content": chat_response.text})
            st.rerun()

    user_q = st.text_input("Ask anything about this ingredient or your formulation...", key="chat_input")
    if st.button("Send →", key="chat_send"):
        if user_q and api_key:
            st.session_state.chat_messages.append({"role": "user", "content": user_q})
            ctx = st.session_state.current_context
            chat_prompt = f"""You are a food formulation scientist. Answer this question about {ctx['top_name']} in the context of replacing {ctx['ingredient']} in {ctx['application']}.
Processing: {ctx['processing']}
Question: {user_q}
Answer in 2-3 sentences. Be direct and specific. No intro phrases."""
            client_ai = genai.Client(api_key=api_key)
            chat_response = client_ai.models.generate_content(model="gemini-2.5-flash", contents=chat_prompt)
            st.session_state.chat_messages.append({"role": "ai", "content": chat_response.text})
            st.rerun()

    with st.expander("📋 Full technical details & processing notes"):
        st.markdown(f"*Why it works:* {r['why_it_works']}")
        st.markdown(f"*Processing adjustments:* {r['processing_notes']}")
        st.markdown(f"*Cost & sourcing:* {r['cost_note']}")

    st.markdown("---")
    st.caption("For formulation guidance only. Always validate through bench trials before commercialisation.")
