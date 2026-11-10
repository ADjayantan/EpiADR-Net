import json

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import torch

from data_loader import (
    BENCHMARK_DRUGS,
    GTEX_TISSUE_PROFILES,
    MEDDRA_ADR_CLASSES,
    ORGAN_NAMES,
)
from model import EpiADRNet
from utils import highlight_toxic_subgraph, smiles_to_graph

# Streamlit Page Config
st.set_page_config(
    page_title="EpiADR-Net — Organ-Conditioned Zero-Shot ADR Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E88E5 0%, #7B1FA2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-left: 4px solid #1E88E5;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .hotspot-tag {
        display: inline-block;
        background-color: #FFEBEE;
        color: #C62828;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Load Local Model Instance
@st.cache_resource
def load_cached_model():
    model = EpiADRNet(in_features=10, hidden_dim=256, num_classes=10, dropout=0.3)
    try:
        model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
    except Exception:
        pass
    model.eval()
    return model

model = load_cached_model()

# Header Section
st.markdown('<div class="main-title">🧬 EpiADR-Net Research Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Tissue-Conditioned Zero-Shot Side Effect Disaggregation via 4-Layer GAT & GTEx FiLM Conditioning</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Inference Controls")
preset_drug = st.sidebar.selectbox(
    "Select Benchmark Drug Preset",
    ["Custom SMILES"] + [d["name"] for d in BENCHMARK_DRUGS]
)

if preset_drug != "Custom SMILES":
    selected_item = next(d for d in BENCHMARK_DRUGS if d["name"] == preset_drug)
    default_smiles = selected_item["smiles"]
else:
    default_smiles = "CC(=O)NC1=CC=C(O)C=C1"

smiles_input = st.sidebar.text_area("SMILES String", value=default_smiles, height=80)
organ_selection = st.sidebar.selectbox("Conditioning Organ", ORGAN_NAMES, index=0)
mc_passes = st.sidebar.slider("Monte Carlo Dropout Passes (N)", min_value=5, max_value=50, value=20, step=5)
confidence_thresh = st.sidebar.slider("Significance Threshold", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Single Organ Predictor & XAI",
    "⚖️ Dual Organ Comparative Analysis",
    "📊 Model Benchmarks & Math",
    "🔌 REST API & Playground"
])


# --- TAB 1: Single Organ Predictor & XAI ---
with tab1:
    col_input, col_info = st.columns([2, 1])
    with col_input:
        st.subheader(f"Predicting ADRs conditioned on **{organ_selection}** Tissue Vector")
    with col_info:
        st.info(f"Target Tissue: GTEx {organ_selection} (128-dim)")

    if st.button("🚀 Run Tissue-Conditioned Inference", type="primary", use_container_width=True):
        node_feats, edge_index, atom_symbols = smiles_to_graph(smiles_input)
        batch = torch.zeros(node_feats.size(0), dtype=torch.long)
        tissue_vec = GTEX_TISSUE_PROFILES[organ_selection].unsqueeze(0)

        mc_res = model.predict_mc_dropout(node_feats, edge_index, batch, tissue_vec, num_samples=mc_passes)
        mu = mc_res["mean_probabilities"][0].cpu().numpy()
        sigma = mc_res["uncertainty_sigma"][0].cpu().numpy()
        attn = mc_res["attention_weights"]

        df_res = pd.DataFrame({
            "MedDRA Term": MEDDRA_ADR_CLASSES,
            "Probability (μ)": mu,
            "Uncertainty (σ)": sigma,
            "Significant": mu >= confidence_thresh
        })

        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown("### 📈 Predicted ADR Probabilities & Epistemic Uncertainty")
            fig = px.bar(
                df_res,
                x="Probability (μ)",
                y="MedDRA Term",
                error_x="Uncertainty (σ)",
                orientation='h',
                color="Probability (μ)",
                color_continuous_scale="Reds",
                range_x=[0, 1.0]
            )
            fig.add_vline(x=confidence_thresh, line_dash="dash", line_color="black", annotation_text="Threshold")
            fig.update_layout(height=450, margin={"l": 0, "r": 0, "t": 30, "b": 0})
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### 🔬 Explainable AI (Layer 4 GAT Hotspots)")
            xai_res = highlight_toxic_subgraph(smiles_input, attn, top_k=3)
            
            st.write(f"**Total Atoms parsed**: `{xai_res['total_atoms']}`")
            st.write("**Top Toxic Atomic Subgraphs (GAT Attention):**")
            
            for hs in xai_res["top_toxic_hotspots"]:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<b>Atom #{hs["atom_index"]} ({hs["atom_symbol"]})</b> — GAT Attention Score: '
                    f'<span class="hotspot-tag">{hs["attention_score"]}</span>'
                    f'</div>',
                    unsafe_allow_allowed_html=True
                )
                
            st.markdown("#### Raw Atom Symbols")
            st.code(" ".join(xai_res["atom_symbols"]))


# --- TAB 2: Dual Organ Comparative Analysis ---
with tab2:
    st.subheader("⚖️ Comparative Organ-Specific Side Effect Disaggregation")
    st.write("Compare how identical molecular structures produce distinct toxicities when conditioned on different tissue expression profiles.")

    c1, c2 = st.columns(2)
    with c1:
        organ_a = st.selectbox("Select First Tissue", ORGAN_NAMES, index=0)
    with c2:
        organ_b = st.selectbox("Select Second Tissue", ORGAN_NAMES, index=1)

    if st.button("⚔️ Generate Comparative Profiles", use_container_width=True):
        node_feats, edge_index, _ = smiles_to_graph(smiles_input)
        batch = torch.zeros(node_feats.size(0), dtype=torch.long)
        
        vec_a = GTEX_TISSUE_PROFILES[organ_a].unsqueeze(0)
        vec_b = GTEX_TISSUE_PROFILES[organ_b].unsqueeze(0)

        res_a = model.predict_mc_dropout(node_feats, edge_index, batch, vec_a, num_samples=mc_passes)
        res_b = model.predict_mc_dropout(node_feats, edge_index, batch, vec_b, num_samples=mc_passes)

        mu_a = res_a["mean_probabilities"][0].cpu().numpy()
        mu_b = res_b["mean_probabilities"][0].cpu().numpy()

        df_comp = pd.DataFrame({
            "MedDRA Term": MEDDRA_ADR_CLASSES * 2,
            "Probability": np.concatenate([mu_a, mu_b]),
            "Organ": [organ_a] * 10 + [organ_b] * 10
        })

        fig_comp = px.bar(
            df_comp,
            x="MedDRA Term",
            y="Probability",
            color="Organ",
            barmode="group",
            color_discrete_sequence=["#1E88E5", "#D81B60"]
        )
        fig_comp.update_layout(height=480, xaxis_tickangle=-45)
        st.plotly_chart(fig_comp, use_container_width=True)


# --- TAB 3: Model Benchmarks & Math ---
with tab3:
    st.subheader("📊 Generalization Benchmark & Mathematical Formulation")
    
    st.markdown("""
    **Measured on the corrected task** — molecule-level Bemis-Murcko scaffold 5-fold CV,
    1,465 molecules, 0 shared molecules/scaffolds between train and test.
    Source: `results/baselines.json` (`make baselines`).

    | model | macro-AUROC | macro-AUPRC |
    | :--- | ---: | ---: |
    | per-organ base rate (no molecular input) | 0.5000 ± 0.000 | 0.2401 ± 0.008 |
    | Morgan fingerprint only | 0.5888 ± 0.013 | 0.3121 ± 0.029 |
    | Morgan + organ one-hot | 0.6197 ± 0.020 | 0.3522 ± 0.015 |
    | Morgan + real GTEx PCA profile | 0.6243 ± 0.012 | 0.3510 ± 0.020 |
    | Morgan + XGBoost, per organ | 0.6364 ± 0.016 | 0.3862 ± 0.027 |
    | **Morgan + drug–target binding + mechanism (M8)** | **0.6547 ± 0.011** | **0.3978 ± 0.020** |

    Per-organ AUROC is primary: a base-rate-only predictor reaches *pooled* AUROC 0.656
    while ranking no drug correctly. Unseen-drug + unseen-organ generalisation is 0.579.

    Drug–target binding is a real gain (+0.035 over an organ one-hot). Its coupling to
    per-organ expression is not: a 12-fold permutation test of the organ→expression map
    gives *p* = 0.385 — see `RESULTS.md §3`. The graph-transformer weights used by this
    dashboard have not been re-benchmarked on this task; see `AUDIT.md`.
    """)

    st.markdown("### 🧮 Mathematical Formulations")
    st.latex(r"""
    \alpha_{ij}^{(l)} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i^{(l-1)} \Vert \mathbf{W} h_j^{(l-1)}]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i^{(l-1)} \Vert \mathbf{W} h_k^{(l-1)}]\right)\right)}
    """)

    st.latex(r"""
    \gamma = \text{MLP}_\gamma(v_{\text{tissue}}) \in \mathbb{R}^{128}, \quad \beta = \text{MLP}_\beta(v_{\text{tissue}}) \in \mathbb{R}^{128}
    """)

    st.latex(r"""
    h_i^{\text{conditioned}} = \gamma \odot h_i + \beta
    """)

    st.latex(r"""
    \mu_q = \frac{1}{N} \sum_{n=1}^N \sigma(\hat{y}_{q, n}), \quad \sigma_q = \sqrt{\frac{1}{N} \sum_{n=1}^N \left(\sigma(\hat{y}_{q, n}) - \mu_q\right)^2}
    """)


# --- TAB 4: REST API & Playground ---
with tab4:
    st.subheader("🔌 FastAPI REST Microservice Playground")
    st.markdown("Interact directly with the local FastAPI microservice running at `http://localhost:8000`.")

    st.info("Interactive Swagger UI Documentation available at: [http://localhost:8000/docs](http://localhost:8000/docs)")

    api_endpoint = st.text_input("API URL Target", value="http://localhost:8000/predict")
    
    sample_json = {
        "smiles": smiles_input,
        "organ": organ_selection,
        "mc_samples": mc_passes
    }
    
    json_payload = st.text_area("JSON Request Payload", value=json.dumps(sample_json, indent=2), height=150)

    if st.button("📡 Send API Request"):
        try:
            res = requests.post(api_endpoint, json=json.loads(json_payload), timeout=5)
            st.json(res.json())
        except Exception as e:
            st.error(f"Could not connect to FastAPI server at {api_endpoint}: {e!s}")
            st.warning("Note: Make sure `uvicorn api:app --reload` is running on port 8000.")
