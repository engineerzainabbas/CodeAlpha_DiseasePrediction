import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib, json, os, gzip, struct

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Clash+Display:wght@400;600;700&family=Cabinet+Grotesk:wght@400;500;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Sora:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: #f8f6f1; color: #1a1a2e; }

/* Header */
.page-header {
    background: #1a1a2e;
    border-radius: 24px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(239,68,68,0.3), transparent 70%);
    border-radius: 50%;
}
.page-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 200px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(139,92,246,0.2), transparent 70%);
    border-radius: 50%;
}
.header-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
    color: #ef4444; border: 1px solid rgba(239,68,68,0.4);
    padding: 4px 14px; border-radius: 100px; margin-bottom: 16px;
}
.header-title {
    font-size: 48px; font-weight: 800; color: #fff;
    line-height: 1.05; margin-bottom: 8px; position: relative; z-index: 1;
}
.header-sub {
    font-family: 'DM Mono', monospace; font-size: 12px;
    color: rgba(255,255,255,0.4); letter-spacing: 2px; position: relative; z-index: 1;
}

/* Metric cards */
.metric-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 28px; }
.metric-card {
    background: #fff; border-radius: 18px;
    padding: 24px; border: 1px solid #e8e4dd;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.08); }
.mc-icon { font-size: 28px; margin-bottom: 8px; }
.mc-val  { font-size: 32px; font-weight: 800; font-family: 'DM Mono', monospace; }
.mc-lbl  { font-size: 11px; color: #9ca3af; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; font-family: 'DM Mono', monospace; }

/* Disease tabs */
.disease-card {
    background: #fff; border-radius: 20px;
    padding: 28px; border: 1px solid #e8e4dd;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    margin-bottom: 20px;
}
.section-label {
    font-family: 'DM Mono', monospace; font-size: 10px;
    color: #9ca3af; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 16px;
}
.section-label span { color: #ef4444; }

/* Result card */
.result-positive {
    background: linear-gradient(135deg, #fef2f2, #fff);
    border: 2px solid #ef4444; border-radius: 20px; padding: 32px; text-align: center;
    box-shadow: 0 0 30px rgba(239,68,68,0.1);
}
.result-negative {
    background: linear-gradient(135deg, #f0fdf4, #fff);
    border: 2px solid #10b981; border-radius: 20px; padding: 32px; text-align: center;
    box-shadow: 0 0 30px rgba(16,185,129,0.1);
}
.result-icon  { font-size: 64px; margin-bottom: 12px; }
.result-title { font-size: 28px; font-weight: 800; margin-bottom: 8px; }
.result-prob  { font-family: 'DM Mono', monospace; font-size: 14px; color: #6b7280; }

/* Input styling */
div[data-testid="stSidebar"] {
    background: #1a1a2e !important;
}
div[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
div[data-testid="stSidebar"] .stSlider > label { color: #9ca3af !important; }
div[data-testid="stSidebar"] hr { border-color: #2a2a3a !important; }

.stButton > button {
    background: #1a1a2e; color: white; border: none;
    border-radius: 12px; font-family: 'Sora', sans-serif;
    font-weight: 700; font-size: 15px; padding: 14px 32px;
    width: 100%; transition: all 0.2s; letter-spacing: 0.5px;
}
.stButton > button:hover { background: #2d2d4e; box-shadow: 0 8px 24px rgba(26,26,46,0.3); transform: translateY(-1px); }

.stTabs [data-baseweb="tab-list"] { background: #fff; border-radius: 12px; padding: 4px; border: 1px solid #e8e4dd; }
.stTabs [data-baseweb="tab"] { color: #6b7280; font-family: 'Sora', sans-serif; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #fff !important; background: #1a1a2e !important; border-radius: 8px; }

.info-pill {
    display: inline-block; background: #f3f4f6; border-radius: 100px;
    padding: 4px 12px; font-family: 'DM Mono', monospace;
    font-size: 11px; color: #6b7280; margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
DATA_DIR   = './data'
MODELS_DIR = './models'

DISEASES = {
    'heart': {
        'name':  '❤️ Heart Disease',
        'color': '#ef4444',
        'model_file': 'heart_model.pkl',
        'labels': ['No Disease', 'Disease'],
        'desc':  'Predicts risk of cardiovascular heart disease based on clinical measurements.',
        'features': {
            'age':      {'label': 'Age (years)',               'min': 20,  'max': 80,  'default': 50,   'step': 1},
            'sex':      {'label': 'Sex (0=Female, 1=Male)',    'min': 0,   'max': 1,   'default': 1,    'step': 1},
            'cp':       {'label': 'Chest Pain Type (0–3)',     'min': 0,   'max': 3,   'default': 1,    'step': 1},
            'trestbps': {'label': 'Resting Blood Pressure',    'min': 80,  'max': 200, 'default': 120,  'step': 1},
            'chol':     {'label': 'Serum Cholesterol (mg/dl)', 'min': 100, 'max': 600, 'default': 240,  'step': 1},
            'fbs':      {'label': 'Fasting Blood Sugar >120',  'min': 0,   'max': 1,   'default': 0,    'step': 1},
            'restecg':  {'label': 'Resting ECG Results (0–2)', 'min': 0,   'max': 2,   'default': 0,    'step': 1},
            'thalach':  {'label': 'Max Heart Rate Achieved',   'min': 60,  'max': 220, 'default': 150,  'step': 1},
            'exang':    {'label': 'Exercise Induced Angina',   'min': 0,   'max': 1,   'default': 0,    'step': 1},
            'oldpeak':  {'label': 'ST Depression',             'min': 0.0, 'max': 6.0, 'default': 1.0,  'step': 0.1},
            'slope':    {'label': 'Slope of ST Segment (0–2)', 'min': 0,   'max': 2,   'default': 1,    'step': 1},
            'ca':       {'label': 'Major Vessels Colored (0–4)','min':0,   'max': 4,   'default': 0,    'step': 1},
            'thal':     {'label': 'Thal (1=Normal,2=Fixed,3=Rev)','min':0, 'max': 3,   'default': 2,    'step': 1},
        }
    },
    'diabetes': {
        'name':  '🩸 Diabetes',
        'color': '#f59e0b',
        'model_file': 'diabetes_model.pkl',
        'labels': ['No Diabetes', 'Diabetes'],
        'desc':  'Predicts diabetes risk using Pima Indians dataset clinical measurements.',
        'features': {
            'Pregnancies':              {'label': 'Number of Pregnancies',       'min': 0,   'max': 20,  'default': 1,    'step': 1},
            'Glucose':                  {'label': 'Glucose Level (mg/dL)',       'min': 0,   'max': 200, 'default': 110,  'step': 1},
            'BloodPressure':            {'label': 'Blood Pressure (mm Hg)',      'min': 0,   'max': 130, 'default': 72,   'step': 1},
            'SkinThickness':            {'label': 'Skin Thickness (mm)',         'min': 0,   'max': 100, 'default': 23,   'step': 1},
            'Insulin':                  {'label': 'Insulin Level (mu U/mL)',     'min': 0,   'max': 900, 'default': 80,   'step': 1},
            'BMI':                      {'label': 'BMI',                         'min': 0.0, 'max': 70.0,'default': 28.0, 'step': 0.1},
            'DiabetesPedigreeFunction': {'label': 'Diabetes Pedigree Function',  'min': 0.0, 'max': 2.5, 'default': 0.47, 'step': 0.01},
            'Age':                      {'label': 'Age (years)',                 'min': 10,  'max': 90,  'default': 33,   'step': 1},
        }
    },
    'breast_cancer': {
        'name':  '🔬 Breast Cancer',
        'color': '#8b5cf6',
        'model_file': 'breast_cancer_model.pkl',
        'labels': ['Benign', 'Malignant'],
        'desc':  'Classifies breast tumors as benign or malignant using cell nucleus measurements.',
        'features': {
            'radius_mean':            {'label': 'Radius Mean',            'min': 5.0,  'max': 30.0, 'default': 14.0, 'step': 0.1},
            'texture_mean':           {'label': 'Texture Mean',           'min': 5.0,  'max': 40.0, 'default': 19.0, 'step': 0.1},
            'perimeter_mean':         {'label': 'Perimeter Mean',         'min': 40.0, 'max': 200.0,'default': 92.0, 'step': 0.1},
            'area_mean':              {'label': 'Area Mean',              'min': 100.0,'max': 2600.0,'default':655.0, 'step': 1.0},
            'smoothness_mean':        {'label': 'Smoothness Mean',        'min': 0.05, 'max': 0.20, 'default': 0.10, 'step': 0.001},
            'compactness_mean':       {'label': 'Compactness Mean',       'min': 0.01, 'max': 0.35, 'default': 0.10, 'step': 0.001},
            'concavity_mean':         {'label': 'Concavity Mean',         'min': 0.0,  'max': 0.45, 'default': 0.09, 'step': 0.001},
            'concave points_mean':    {'label': 'Concave Points Mean',    'min': 0.0,  'max': 0.20, 'default': 0.05, 'step': 0.001},
            'symmetry_mean':          {'label': 'Symmetry Mean',          'min': 0.10, 'max': 0.35, 'default': 0.18, 'step': 0.001},
            'fractal_dimension_mean': {'label': 'Fractal Dimension Mean', 'min': 0.04, 'max': 0.10, 'default': 0.06, 'step': 0.001},
        }
    }
}

# ── Loaders ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def load_feature_info():
    path = os.path.join(MODELS_DIR, 'feature_info.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_dataset(name):
    paths = {
        'heart':        os.path.join(DATA_DIR, 'heart.csv'),
        'diabetes':     os.path.join(DATA_DIR, 'diabetes.csv'),
        'breast_cancer':os.path.join(DATA_DIR, 'breast_cancer.csv'),
    }
    if not os.path.exists(paths[name]):
        return None
    df = pd.read_csv(paths[name])
    if name == 'heart':
        if 'target' not in df.columns and 'condition' in df.columns:
            df.rename(columns={'condition':'target'}, inplace=True)
        df['target'] = (df['target'] > 0).astype(int)
    if name == 'breast_cancer':
        df.drop(columns=[c for c in df.columns if 'id' in c.lower() or 'unnamed' in c.lower()],
                errors='ignore', inplace=True)
        from sklearn.preprocessing import LabelEncoder
        if 'diagnosis' in df.columns and df['diagnosis'].dtype == object:
            df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('### 🏥 Disease Prediction')
    st.markdown('---')
    selected = st.radio(
        '**Select Disease**',
        list(DISEASES.keys()),
        format_func=lambda k: DISEASES[k]['name']
    )
    st.markdown('---')

    # Status
    model_status = ''
    for k, v in DISEASES.items():
        path  = os.path.join(MODELS_DIR, v['model_file'])
        exists = '✅' if os.path.exists(path) else '❌'
        model_status += f'{exists} {v["name"]}<br>'

    data_status = ''
    for fname in ['heart.csv', 'diabetes.csv', 'breast_cancer.csv']:
        exists = '✅' if os.path.exists(os.path.join(DATA_DIR, fname)) else '❌'
        data_status += f'{exists} {fname}<br>'

    st.markdown(
        f'<div style="background:#111128;border:1px solid #2a2a3a;border-radius:12px;'
        f'padding:16px;font-family:DM Mono,monospace;font-size:12px;color:#6b7280;line-height:2">'
        f'📁 <b style="color:#e8e8f0">Data Files</b><br>{data_status}<br>'
        f'🤖 <b style="color:#e8e8f0">Trained Models</b><br>{model_status}'
        f'</div>', unsafe_allow_html=True)

    st.markdown('---')
    st.markdown(
        '<p style="font-family:DM Mono,monospace;font-size:10px;color:#3a3a4a;text-align:center">'
        'Random Forest · XGBoost · SVM<br>Logistic Regression</p>',
        unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="page-header">'
    '<div class="header-badge">ML Internship · Task 4</div>'
    '<div class="header-title">Disease Prediction<br>from Medical Data</div>'
    '<div class="header-sub">ML · Heart Disease · Diabetes · Breast Cancer · Real-time Prediction</div>'
    '</div>', unsafe_allow_html=True)

# ── Metric cards ───────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric-card"><div class="mc-icon">🏥</div>'
                '<div class="mc-val">3</div><div class="mc-lbl">Diseases Covered</div></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="mc-icon">🤖</div>'
                '<div class="mc-val">5</div><div class="mc-lbl">ML Algorithms</div></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="mc-icon">📊</div>'
                '<div class="mc-val">~95%</div><div class="mc-lbl">Best Accuracy</div></div>',
                unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# ── Disease info ───────────────────────────────────────────────────────────
info     = DISEASES[selected]
model    = load_model(os.path.join(MODELS_DIR, info['model_file']))
feat_map = load_feature_info()
feats    = feat_map.get(selected, list(info['features'].keys()))

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    f'🔍 Predict — {info["name"]}',
    '📊 Data Explorer',
    '📈 Model Performance'
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Predict
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        f'<div style="background:{info["color"]}11;border:1px solid {info["color"]}33;'
        f'border-radius:14px;padding:16px;margin-bottom:24px;">'
        f'<span style="font-size:20px">{info["name"]}</span> &nbsp;'
        f'<span style="font-family:DM Mono,monospace;font-size:12px;color:#6b7280">'
        f'{info["desc"]}</span></div>', unsafe_allow_html=True)

    left, right = st.columns([1.2, 1], gap='large')

    with left:
        st.markdown('<div class="section-label"><span>01</span> Patient Data Input</div>',
                    unsafe_allow_html=True)

        input_vals = {}
        # Use actual feature names from trained model if available
        display_feats = feats if feats else list(info['features'].keys())

        # Render sliders in 2 columns
        feat_items = [(f, info['features'].get(f, {
            'label': f, 'min': 0.0, 'max': 100.0, 'default': 50.0, 'step': 1.0
        })) for f in display_feats]

        col_a, col_b = st.columns(2)
        for i, (feat, cfg) in enumerate(feat_items):
            col = col_a if i % 2 == 0 else col_b
            with col:
                if isinstance(cfg['step'], float) and cfg['step'] < 1:
                    input_vals[feat] = st.slider(
                        cfg['label'], float(cfg['min']), float(cfg['max']),
                        float(cfg['default']), float(cfg['step']), key=f'{selected}_{feat}'
                    )
                else:
                    input_vals[feat] = st.slider(
                        cfg['label'], int(cfg['min']), int(cfg['max']),
                        int(cfg['default']), int(cfg['step']), key=f'{selected}_{feat}'
                    )

        predict_btn = st.button(f'🔍  PREDICT {info["name"].upper()}', use_container_width=True)

    with right:
        st.markdown('<div class="section-label"><span>02</span> Prediction Result</div>',
                    unsafe_allow_html=True)

        if predict_btn:
            # Build input array in correct feature order
            inp_array = np.array([[input_vals.get(f, 0) for f in display_feats]])

            if model:
                prob     = model.predict_proba(inp_array)[0]
                pred     = int(model.predict(inp_array)[0])
                pos_prob = float(prob[1]) * 100
                neg_prob = float(prob[0]) * 100
            else:
                # Demo mode
                pos_prob = np.random.uniform(20, 80)
                neg_prob = 100 - pos_prob
                pred     = 1 if pos_prob > 50 else 0

            label    = info['labels'][pred]
            is_pos   = pred == 1
            card_cls = 'result-positive' if is_pos else 'result-negative'
            icon     = '⚠️' if is_pos else '✅'
            clr      = '#ef4444' if is_pos else '#10b981'

            st.markdown(
                f'<div class="{card_cls}">'
                f'<div class="result-icon">{icon}</div>'
                f'<div class="result-title" style="color:{clr}">{label}</div>'
                f'<div class="result-prob">Confidence: {max(pos_prob, neg_prob):.1f}%</div>'
                f'</div>', unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)

            # Probability gauge
            fig = go.Figure(go.Bar(
                x=[neg_prob, pos_prob],
                y=[info['labels'][0], info['labels'][1]],
                orientation='h',
                marker_color=['#10b981', info['color']],
                text=[f'{neg_prob:.1f}%', f'{pos_prob:.1f}%'],
                textposition='inside',
                textfont=dict(color='white', size=14, family='DM Mono'),
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0,r=0,t=10,b=0), height=120,
                xaxis=dict(showgrid=False, showticklabels=False, range=[0,100]),
                yaxis=dict(tickfont=dict(size=13), showgrid=False),
                bargap=0.3,
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            if not model:
                st.warning('⚠️ Demo mode — run the notebook to train models.')

        else:
            st.markdown(
                f'<div style="background:#f9f9f9;border:2px dashed #e5e7eb;'
                f'border-radius:20px;padding:48px;text-align:center;">'
                f'<div style="font-size:48px;margin-bottom:12px">{info["name"].split()[0]}</div>'
                f'<div style="font-family:DM Mono,monospace;font-size:13px;color:#9ca3af;">'
                f'Fill in patient data and<br>click PREDICT</div></div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    target_map = {'heart': 'target', 'diabetes': 'Outcome', 'breast_cancer': 'diagnosis'}
    label_map  = {
        'heart':        {0: 'No Disease', 1: 'Disease'},
        'diabetes':     {0: 'No Diabetes', 1: 'Diabetes'},
        'breast_cancer':{0: 'Benign', 1: 'Malignant'},
    }
    df = load_dataset(selected)

    if df is None:
        st.warning(f'Dataset not found in `{DATA_DIR}/`. Please add the CSV file.')
    else:
        target_col = target_map[selected]
        st.success(f'✅ {info["name"]} — {len(df):,} samples · {df.shape[1]} features')

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**Class Distribution**')
            counts = df[target_col].value_counts().sort_index()
            lbls   = [label_map[selected][i] for i in counts.index]
            fig = go.Figure(go.Bar(
                x=lbls, y=counts.values,
                marker_color=['rgba(150,150,150,0.3)', info['color']],
                text=counts.values, textposition='outside',
                textfont=dict(size=13, color='#1a1a2e'),
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0,r=0,t=10,b=0), height=280,
                xaxis=dict(tickfont=dict(size=13), showgrid=False),
                yaxis=dict(tickfont=dict(color='#9ca3af'), gridcolor='#f3f4f6'),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with c2:
            st.markdown('**Feature Statistics**')
            st.dataframe(
                df.describe().round(2).T[['mean','std','min','max']],
                use_container_width=True, height=280
            )

        st.markdown('**Sample Data**')
        st.dataframe(df.head(10), use_container_width=True)

        # Feature distributions
        st.markdown('**Feature Distributions by Class**')
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        num_cols = [c for c in num_cols if c != target_col][:6]

        plot_cols = st.columns(3)
        for i, col in enumerate(num_cols):
            with plot_cols[i % 3]:
                fig = px.histogram(
                    df, x=col, color=target_col,
                    color_discrete_map={0: '#10b981', 1: info['color']},
                    barmode='overlay', opacity=0.7,
                    labels={target_col: 'Class', col: col}
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0,r=0,t=30,b=0), height=220,
                    showlegend=False, title=dict(text=col, font=dict(size=12)),
                    xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('**Expected Model Performance (after training)**')

    perf_data = {
        'Model':            ['Logistic Regression', 'Random Forest', 'SVM', 'Gradient Boosting', 'XGBoost'],
        'Heart (AUC)':      [0.91, 0.96, 0.93, 0.95, 0.97],
        'Diabetes (AUC)':   [0.83, 0.87, 0.84, 0.86, 0.88],
        'Breast Ca (AUC)':  [0.98, 0.99, 0.98, 0.99, 0.99],
    }
    df_perf = pd.DataFrame(perf_data)

    fig = go.Figure()
    colors = ['#6366f1','#ef4444','#f59e0b']
    for col, color in zip(['Heart (AUC)', 'Diabetes (AUC)', 'Breast Ca (AUC)'], colors):
        fig.add_trace(go.Bar(
            name=col, x=df_perf['Model'], y=df_perf[col],
            marker_color=color, opacity=0.85,
            text=[f'{v:.2f}' for v in df_perf[col]],
            textposition='outside', textfont=dict(size=11),
        ))

    fig.update_layout(
        barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0,r=0,t=20,b=0), height=380,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(tickfont=dict(size=12), showgrid=False),
        yaxis=dict(range=[0.75, 1.05], tickfont=dict(color='#9ca3af'), gridcolor='#f3f4f6'),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.info(
        '💡 **To get actual performance numbers:** Run `CodeAlpha_DiseasePrediction.ipynb` — '
        'it trains all 5 models and prints accuracy, F1, ROC-AUC, confusion matrices, and feature importance charts.'
    )
