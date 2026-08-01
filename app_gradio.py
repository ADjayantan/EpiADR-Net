import gradio as gr
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from typing import Tuple, Dict, Any

from utils import smiles_to_graph, highlight_toxic_subgraph
from data_loader import GTEX_TISSUE_PROFILES, MEDDRA_ADR_CLASSES, ORGAN_NAMES, BENCHMARK_DRUGS
from model import EpiADRNet

# Optional spaces import for HF Zero-GPU runtime compatibility
try:
    import spaces
except ImportError:
    spaces = None

# Load model instance
MODEL = EpiADRNet(in_features=5, hidden_dim=128, num_classes=10, dropout=0.2)
try:
    MODEL.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
except Exception:
    pass
MODEL.eval()


def predict_single_organ(smiles: str, organ: str, mc_passes: int) -> Tuple[pd.DataFrame, str, Any]:
    """
    Predicts organ-conditioned ADR probabilities, epistemic uncertainty, and XAI toxic hotspots.
    """
    if not smiles or organ not in GTEX_TISSUE_PROFILES:
        return pd.DataFrame(), "Invalid SMILES or Organ", None

    node_feats, edge_index, atom_symbols = smiles_to_graph(smiles)
    batch = torch.zeros(node_feats.size(0), dtype=torch.long)
    tissue_vec = GTEX_TISSUE_PROFILES[organ].unsqueeze(0)

    mc_res = MODEL.predict_mc_dropout(node_feats, edge_index, batch, tissue_vec, num_samples=int(mc_passes))
    mu = mc_res["mean_probabilities"][0].cpu().numpy()
    sigma = mc_res["uncertainty_sigma"][0].cpu().numpy()
    attn = mc_res["attention_weights"]

    df_res = pd.DataFrame({
        "MedDRA Term": MEDDRA_ADR_CLASSES,
        "Probability (μ)": [round(float(p), 4) for p in mu],
        "Uncertainty (σ)": [round(float(s), 4) for s in sigma],
        "Significant (>0.45)": [p >= 0.45 for p in mu]
    })

    xai_res = highlight_toxic_subgraph(smiles, attn, top_k=3)
    
    xai_text = f"### 🔬 Toxic Hotspot Analysis (Layer 4 GAT Attention)\n"
    xai_text += f"- **Parsed Atoms**: `{xai_res['total_atoms']}` ({' '.join(xai_res['atom_symbols'])})\n\n"
    xai_text += "**Top Reactive Subgraph Atoms:**\n"
    for hs in xai_res["top_toxic_hotspots"]:
        xai_text += f"- Atom **#{hs['atom_index']}** (`{hs['atom_symbol']}`) — Score: **{hs['attention_score']}**\n"

    # Plotly figure
    fig = px.bar(
        df_res,
        x="Probability (μ)",
        y="MedDRA Term",
        error_x="Uncertainty (σ)",
        orientation='h',
        color="Probability (μ)",
        color_continuous_scale="Reds",
        range_x=[0, 1.0],
        title=f"Conditioned on {organ} GTEx Expression Vector"
    )
    
    return df_res, xai_text, fig


def compare_two_organs(smiles: str, organ_a: str, organ_b: str, mc_passes: int):
    """
    Compares ADR predictions between two distinct tissue profiles side-by-side.
    """
    node_feats, edge_index, _ = smiles_to_graph(smiles)
    batch = torch.zeros(node_feats.size(0), dtype=torch.long)
    
    vec_a = GTEX_TISSUE_PROFILES[organ_a].unsqueeze(0)
    vec_b = GTEX_TISSUE_PROFILES[organ_b].unsqueeze(0)

    res_a = MODEL.predict_mc_dropout(node_feats, edge_index, batch, vec_a, num_samples=int(mc_passes))
    res_b = MODEL.predict_mc_dropout(node_feats, edge_index, batch, vec_b, num_samples=int(mc_passes))

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
        title=f"Comparative Organ Profile: {organ_a} vs {organ_b}",
        color_discrete_sequence=["#1E88E5", "#D81B60"]
    )
    fig_comp.update_layout(xaxis_tickangle=-45)
    return fig_comp


# Build Gradio Interface
with gr.Blocks(title="EpiADR-Net — Hugging Face Space", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧬 EpiADR-Net: Tissue-Conditioned Zero-Shot ADR Platform
        ### Fusing 4-Layer Graph Attention Networks (GAT) with GTEx Human Transcriptomic Profiles
        [GitHub Repository](https://github.com/ADjayantan/EpiADR-Net) | [IEEE/ACM Manuscript](https://github.com/ADjayantan/EpiADR-Net/blob/main/MANUSCRIPT.md)
        """
    )

    with gr.Tab("🎯 Single Organ Predictor & XAI"):
        with gr.Row():
            with gr.Column():
                input_smiles = gr.Textbox(
                    label="SMILES Molecular Structure",
                    value="CC(=O)NC1=CC=C(O)C=C1",
                    placeholder="Enter valid SMILES string..."
                )
                input_organ = gr.Dropdown(
                    choices=ORGAN_NAMES,
                    value="Liver",
                    label="Target GTEx Tissue Profile"
                )
                input_mc = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Monte Carlo Dropout Passes (N)"
                )
                btn_predict = gr.Button("🚀 Predict Tissue-Conditioned ADRs", variant="primary")
                
                gr.Examples(
                    examples=[[d["smiles"], "Liver", 20] for d in BENCHMARK_DRUGS[:5]],
                    inputs=[input_smiles, input_organ, input_mc]
                )

            with gr.Column():
                output_plot = gr.Plot(label="Predicted ADR Probabilities (μ ± σ)")
                output_xai = gr.Markdown(label="Layer 4 GAT Toxic Hotspot Breakdown")
                output_table = gr.Dataframe(label="Detailed Probability & Uncertainty Table")

        btn_predict.click(
            fn=predict_single_organ,
            inputs=[input_smiles, input_organ, input_mc],
            outputs=[output_table, output_xai, output_plot]
        )

    with gr.Tab("⚖️ Dual Organ Comparative Analysis"):
        with gr.Row():
            comp_smiles = gr.Textbox(label="SMILES String", value="CC(=O)NC1=CC=C(O)C=C1")
            comp_organ_a = gr.Dropdown(choices=ORGAN_NAMES, value="Liver", label="Tissue Profile A")
            comp_organ_b = gr.Dropdown(choices=ORGAN_NAMES, value="Heart", label="Tissue Profile B")
            comp_mc = gr.Slider(minimum=5, maximum=50, value=20, step=5, label="MC Passes")
        
        btn_compare = gr.Button("⚔️ Generate Side-by-Side Comparison", variant="primary")
        comp_plot = gr.Plot(label="Side-by-Side Organ Toxicity Disaggregation")

        btn_compare.click(
            fn=compare_two_organs,
            inputs=[comp_smiles, comp_organ_a, comp_organ_b, comp_mc],
            outputs=[comp_plot]
        )

    with gr.Tab("📊 Model Metrics & Benchmark Summary"):
        gr.Markdown(
            """
            ### Bemis-Murcko Scaffold Split vs Random Split Performance
            | Experiment Split | Test Macro-AUROC | Test Micro-AUPRC | Epochs |
            | :--- | :--- | :--- | :--- |
            | **Bemis-Murcko Scaffold Split** | **0.4938** | **0.2589** | 10 |
            | **Random Split Baseline** | 0.4638 | 0.3018 | 10 |
            
            - **GAT Backbone**: 4 Attention Layers ($h_i \in \mathbb{R}^{128}$)
            - **FiLM Modulation**: $\gamma \odot h_i + \beta$ using 128-dim GTEx Organ Profiles
            - **Uncertainty Quantification**: Monte Carlo Dropout ($N=20$)
            """
        )

if __name__ == "__main__":
    demo.queue().launch()
