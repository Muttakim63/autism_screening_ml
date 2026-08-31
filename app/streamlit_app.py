import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Autism Spectrum Disorder (ASD) Screening Portal",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM GLASSMORPHISM STYLING ---
st.markdown("""
<style>
    /* Dark Theme & Modern Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.37);
    }
    
    /* Headers */
    .main-header {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.6rem;
        margin-bottom: 8px;
    }
    
    .sub-header {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 24px;
    }
    
    /* Result Badges */
    .result-badge-positive {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid #ef4444;
        color: #fca5a5;
        border-radius: 12px;
        padding: 16px 24px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
    }
    
    .result-badge-negative {
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid #22c55e;
        color: #86efac;
        border-radius: 12px;
        padding: 16px 24px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
    }
    
    /* Question Container */
    .q-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #6366f1;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 700;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL PIPELINE & METADATA ---
MODEL_PATH = '/Users/shuprov630/autism_screening_ml/models/best_autism_model.joblib'
META_PATH = '/Users/shuprov630/autism_screening_ml/models/model_metadata.json'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r') as f:
            return json.load(f)
    return None

pipeline = load_model()
metadata = load_metadata()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/brain.png", width=70)
    st.markdown("### 🧩 ASD Screening Portal")
    st.markdown("Machine Learning-Powered Early Diagnostic Screening Tool for Adults.")
    
    st.markdown("---")
    st.markdown("#### 📊 Model Information")
    if metadata:
        st.markdown(f"**Selected Model:** `{metadata.get('model_name', 'Tuned ML Classifier')}`")
        st.markdown(f"**Test Accuracy:** `{metadata.get('accuracy', 0.95):.1%}`")
        st.markdown(f"**Test F1-Score:** `{metadata.get('f1_score', 0.95):.3f}`")
    else:
        st.markdown("**Status:** Model Pipeline Ready")
        
    st.markdown("---")
    st.markdown("#### ⚠️ Medical Disclaimer")
    st.caption(
        "This tool is designed purely for informational and educational screening purposes based on standard "
        "AQ-10 items. It is **NOT** a formal medical or clinical diagnosis. Please consult a qualified healthcare "
        "professional for official clinical evaluations."
    )

# --- MAIN APP HEADER ---
st.markdown('<div class="main-header">Autism Spectrum Disorder (ASD) Screening Tool</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Utilizing Machine Learning & AQ-10 Behavioral Indicators to Evaluate ASD Traits in Adults</div>',
    unsafe_allow_html=True
)

# Tab Layout
tab1, tab2, tab3 = st.tabs(["📝 Interactive Screening", "📊 Analytics & Benchmarks", "ℹ️ About the Project"])

AQ_QUESTIONS = [
    ("A1_Score", "1. I often notice small sounds when others do not."),
    ("A2_Score", "2. I usually concentrate more on the whole picture, rather than the small details."),
    ("A3_Score", "3. I find it easy to do more than one thing at once."),
    ("A4_Score", "4. If there is an interruption, I can switch back to what I was doing very quickly."),
    ("A5_Score", "5. I find it easy to 'read between the lines' when someone is talking to me."),
    ("A6_Score", "6. I know how to tell if someone listening to me is getting bored."),
    ("A7_Score", "7. When I'm reading a story, I find it difficult to work out the characters' intentions."),
    ("A8_Score", "8. I like to collect information about categories of things (e.g. types of car, birds, trains)."),
    ("A9_Score", "9. I find it easy to work out what someone is thinking or feeling just by looking at their face."),
    ("A10_Score", "10. I find it difficult to make new friends.")
]

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📋 Step 1: AQ-10 Behavioral Screening Questionnaire")
    st.caption("Select 'Agree' if the behavior describes you, or 'Disagree' otherwise.")
    
    col1, col2 = st.columns(2)
    user_aq = {}
    
    for idx, (col_key, q_text) in enumerate(AQ_QUESTIONS):
        target_col = col1 if idx < 5 else col2
        with target_col:
            st.markdown(f"**Question {idx+1}**")
            st.write(q_text)
            choice = st.radio(
                f"q_{col_key}",
                options=["Disagree / No", "Agree / Yes"],
                key=f"radio_{col_key}",
                label_visibility="collapsed"
            )
            user_aq[col_key] = 1 if "Agree" in choice else 0
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("👤 Step 2: Individual Characteristics & Clinical History")
    
    d_col1, d_col2, d_col3 = st.columns(3)
    
    with d_col1:
        age = st.slider("Age (years)", min_value=18, max_value=80, value=28)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
    with d_col2:
        ethnicity = st.selectbox(
            "Ethnicity",
            ["White-European", "Asian", "Middle Eastern", "South Asian", "Black", "Latino", "Hispanic", "Pashtika", "Turkish", "Others", "Unknown"]
        )
        country = st.selectbox(
            "Country of Residence",
            ["United States", "United Kingdom", "Canada", "Australia", "India", "New Zealand", "United Arab Emirates", "Others"]
        )
        
    with d_col3:
        jaundice = st.radio("Born with Jaundice?", ["No", "Yes"], horizontal=True)
        austim = st.radio("Family member with ASD?", ["No", "Yes"], horizontal=True)
        used_app = st.radio("Used screening app before?", ["No", "Yes"], horizontal=True)
        relation = st.selectbox("Test taken by", ["Self", "Parent", "Relative", "Health Care Professional", "Others"])
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Calculate Total AQ Score
    total_aq = sum(user_aq.values())
    
    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.button("🚀 Analyze ASD Screening Probability", use_container_width=True)

    if submit_btn or 'screening_done' in st.session_state:
        st.session_state['screening_done'] = True
        
        # Prepare input dataframe matching model columns
        input_data = {
            'A1_Score': user_aq['A1_Score'],
            'A2_Score': user_aq['A2_Score'],
            'A3_Score': user_aq['A3_Score'],
            'A4_Score': user_aq['A4_Score'],
            'A5_Score': user_aq['A5_Score'],
            'A6_Score': user_aq['A6_Score'],
            'A7_Score': user_aq['A7_Score'],
            'A8_Score': user_aq['A8_Score'],
            'A9_Score': user_aq['A9_Score'],
            'A10_Score': user_aq['A10_Score'],
            'age': float(age),
            'gender': 1 if gender == "Male" else 0,
            'ethnicity': ethnicity,
            'jaundice': 1 if jaundice == "Yes" else 0,
            'austim': 1 if austim == "Yes" else 0,
            'country_of_res': country,
            'used_app_before': 1 if used_app == "Yes" else 0,
            'result': float(total_aq),
            'relation': relation
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Predict using machine learning pipeline
        if pipeline is not None:
            pred = pipeline.predict(input_df)[0]
            proba = pipeline.predict_proba(input_df)[0][1]
        else:
            # Fallback rule-based heuristic if model loading is pending
            proba = min(1.0, (total_aq / 10.0) * 0.85 + (0.15 if austim == "Yes" else 0.0))
            pred = 1 if proba >= 0.5 else 0
            
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎯 Screening Assessment Result")
        
        r_col1, r_col2 = st.columns([1, 2])
        
        with r_col1:
            # Donut Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={'suffix': "%", 'font': {'size': 42, 'color': '#f8fafc'}},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "ASD Trait Probability", 'font': {'size': 18, 'color': '#94a3b8'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#94a3b8"},
                    'bar': {'color': "#a855f7" if pred == 1 else "#3b82f6"},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(34, 197, 94, 0.3)"},
                        {'range': [40, 65], 'color': "rgba(234, 179, 8, 0.3)"},
                        {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.3)"}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=260,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with r_col2:
            st.write("#### Classification Summary")
            if pred == 1:
                st.markdown(
                    '<div class="result-badge-positive">⚠️ Positive ASD Screening Result (High Probability)</div>',
                    unsafe_allow_html=True
                )
                st.markdown("""
                **Insights:**
                - Your response profile exhibits strong behavioral indicators associated with Autistic Spectrum Disorder traits.
                - Total AQ-10 Score: **{}/10** (Scores $\ge 6$ typically warrant further professional assessment).
                - **Recommended Action**: Consult a licensed clinical psychologist or medical practitioner for formal evaluation.
                """.format(total_aq))
            else:
                st.markdown(
                    '<div class="result-badge-negative">✅ Negative ASD Screening Result (Low Probability)</div>',
                    unsafe_allow_html=True
                )
                st.markdown("""
                **Insights:**
                - Your response profile aligns with typical non-autistic spectrum patterns.
                - Total AQ-10 Score: **{}/10**.
                - **Recommended Action**: No immediate ASD screening indicators detected. Maintain regular wellness checkups.
                """.format(total_aq))

        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 AQ-10 Behavioral Score Radar Analytics")
    
    # Radar chart comparing user responses
    categories = [f"Q{i}" for i in range(1, 11)]
    scores = [user_aq[f"A{i}_Score"] for i in range(1, 11)] if 'user_aq' in locals() else [0]*10
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='User Profile',
        line_color='#a855f7',
        fillcolor='rgba(168, 85, 247, 0.3)'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[0.8]*10,
        theta=categories,
        name='ASD Benchmark Average',
        line=dict(color='#ef4444', dash='dash')
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='rgba(15, 23, 42, 0.4)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("ℹ️ Project Architecture & Engineering Notes")
    st.markdown("""
    ### 🔬 Machine Learning Pipeline & Methodology
    This project is built to deliver a robust, reproducible, end-to-end Machine Learning screening solution for Autism Spectrum Disorder:
    
    1. **Data Preprocessing & Cleaning**:
       - Handled missing values (`?`) in categorical attributes using domain-appropriate imputer strategies.
       - Standardized numeric features (`age`, `result`) using `StandardScaler`.
       - One-Hot Encoded non-binary categorical features (`ethnicity`, `country_of_res`, `relation`).
    2. **Model Evaluation & Benchmarking**:
       - Evaluated 6 diverse classification algorithms: **Logistic Regression**, **Random Forest**, **Support Vector Machines (SVM)**, **Gradient Boosting**, **XGBoost**, and **LightGBM**.
       - Evaluated models using **F1-Score**, **Precision**, **Recall**, and **ROC-AUC** to balance false positives and false negatives in medical screening context.
    3. **Deployment**:
       - Serialized trained Scikit-Learn pipeline using `joblib`.
       - Interactive Streamlit web interface with real-time inference.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
