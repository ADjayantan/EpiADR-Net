from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import torch

from data_loader import (
    BENCHMARK_DRUGS,
    GTEX_TISSUE_PROFILES,
    MEDDRA_ADR_CLASSES,
    ORGAN_NAMES,
)
from model import EpiADRNet
from utils import highlight_toxic_subgraph, smiles_to_graph

# Hugging Face Zero-GPU runtime compatibility decorator
try:
    import spaces
    gpu_decorator = spaces.GPU
except Exception:
    def gpu_decorator(fn):
        return fn

# Load model instance — EpiADR-Net v5 (~116.5M Parameters)
MODEL = EpiADRNet(
    in_features=24, hidden_dim=1536, tissue_dim=1024,
    num_classes=10, num_gat_layers=12, num_heads=16, dropout=0.1,
)
try:
    MODEL.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
except Exception:
    pass
MODEL.eval()


@gpu_decorator
def predict_single_organ(smiles: str, organ: str, mc_passes: int) -> tuple[pd.DataFrame, str, Any]:
    """
    Predicts organ-conditioned ADR probabilities, epistemic uncertainty, and XAI toxic hotspots.
    """
    if not smiles or organ not in GTEX_TISSUE_PROFILES:
        return pd.DataFrame(), "Invalid SMILES or Organ", None

    node_feats, edge_index, _atom_symbols = smiles_to_graph(smiles)
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
    
    xai_text = "### 🔬 Toxic Hotspot Analysis (Layer 4 GAT Attention)\n"
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


@gpu_decorator
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
with gr.Blocks(title="EpiADR-Net v5 — Clinical & Biological Research Platform") as demo:
    gr.Markdown(
        """
        # 🧬 EpiADR-Net — Organ-Conditioned ADR Prediction (research demo)
        ### 1,465 molecules · 5 organs · SIDER 4.1 + GTEx V8 · scaffold-split benchmark

        [GitHub](https://github.com/ADjayantan/EpiADR-Net) · [Results](https://github.com/ADjayantan/EpiADR-Net/blob/main/RESULTS.md) · [Audit](https://github.com/ADjayantan/EpiADR-Net/blob/main/AUDIT.md)

        > ⚠️ **Not for clinical, prescribing or preclinical decision-making.**
        > This is a student research benchmark. The best measured result on this task is
        > **macro-AUROC 0.636** under molecule-level scaffold cross-validation, and unseen-drug /
        > unseen-organ generalisation is near chance (0.552). Probabilities shown here are not
        > calibrated (ECE 0.111) and carry no indication, dose, route or patient context.
        >
        > The graph-transformer weights driving this demo have **not** been benchmarked on the
        > corrected task — see `AUDIT.md`. Treat every output as illustrative only.
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
        gr.Markdown("🔒 **Clinical & Biological Research Notice**: *This comparative disaggregation matrix is restricted to Medical Doctors, Pharmacologists & Biological Researchers.*")
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

    with gr.Tab("📊 Benchmark results"):
        gr.Markdown(
            """
            ### Measured performance

            Molecule-level Bemis-Murcko scaffold 5-fold CV over 1,465 molecules.
            0 shared molecules and 0 shared scaffolds between train and test, asserted every fold.
            Produced by `scripts/run_baselines.py` → `results/baselines.json`.

            | model | macro-AUROC | macro-AUPRC |
            | :--- | ---: | ---: |
            | per-organ base rate (no molecular input) | 0.5000 ± 0.000 | 0.2401 ± 0.008 |
            | Morgan + LogReg, per organ | 0.5990 ± 0.021 | 0.3465 ± 0.016 |
            | **Morgan + XGBoost, per organ** | **0.6364 ± 0.016** | **0.3862 ± 0.027** |
            | Morgan + organ one-hot, shared | 0.6197 ± 0.020 | 0.3522 ± 0.015 |
            | Morgan + real GTEx tissue, shared | 0.6243 ± 0.012 | 0.3510 ± 0.020 |
            | Morgan + **shuffled** GTEx tissue, shared | 0.6244 ± 0.013 | 0.3513 ± 0.023 |

            **Per-organ AUROC is the primary metric.** A predictor that only emits each organ's
            base rate reaches *pooled* AUROC 0.656 while ranking no drug correctly, because
            positives range from 7.8% (Lung) to 38.0% (Brain).

            | organ | ADR | AUROC | positives |
            | :--- | :--- | ---: | ---: |
            | Liver | Hepatotoxicity | 0.6667 ± 0.015 | 18.5% |
            | Heart | Arrhythmia | 0.5990 ± 0.009 | 29.2% |
            | Brain | Seizure | 0.6317 ± 0.027 | 38.0% |
            | Kidney | Nephrotoxicity | 0.6226 ± 0.041 | 26.5% |
            | Lung | Pneumonitis | 0.6621 ± 0.081 | 7.8% |

            ### Generalisation

            | setting | best macro-AUROC |
            | :--- | ---: |
            | unseen drug, seen organ | 0.6243 |
            | seen drug, unseen organ | 0.6975 *(molecule-only model is best)* |
            | **unseen drug + unseen organ** | **0.5517 — near chance** |

            ### Negative result on tissue conditioning

            Conditioning on real GTEx profiles beats a plain organ one-hot by **+0.005**
            macro-AUROC, against a fold s.d. of 0.012–0.020. Deranging the profiles so every
            organ receives another organ's transcriptome changes the score by **−0.0001**.
            The organ labels come from a MedDRA System Organ Class terminology mapping rather
            than tissue measurement, so there is no transcriptomic signal to recover.
            Full discussion in [RESULTS.md](https://github.com/ADjayantan/EpiADR-Net/blob/main/RESULTS.md).

            ### Withdrawn figures

            Earlier versions of this Space reported 0.9680 macro-AUROC on "150+ FDA drugs
            (15,000 samples)". Those figures are withdrawn: the pipeline that produced them
            trained on 5 molecules and conditioned on random vectors.
            See [AUDIT.md](https://github.com/ADjayantan/EpiADR-Net/blob/main/AUDIT.md).
            """
        )


if __name__ == "__main__":
    demo.launch()
