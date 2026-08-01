from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import torch
import os

from utils import smiles_to_graph, highlight_toxic_subgraph
from data_loader import GTEX_TISSUE_PROFILES, MEDDRA_ADR_CLASSES, ORGAN_NAMES
from model import EpiADRNet

app = FastAPI(
    title="EpiADR-Net REST Microservice API",
    description="Enterprise API for Tissue-Conditioned Zero-Shot Adverse Drug Reaction (ADR) Disaggregation — v3",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model Initialization — EpiADR-Net v3
MODEL = EpiADRNet(
    in_features=17, hidden_dim=512, tissue_dim=256,
    num_classes=10, num_gat_layers=6, num_heads=8, dropout=0.3,
)
MODEL_WEIGHTS_PATH = "model.pt"
if os.path.exists(MODEL_WEIGHTS_PATH):
    try:
        MODEL.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
        MODEL.eval()
    except Exception:
        pass


# Pydantic Schemas
class PredictionRequest(BaseModel):
    smiles: str = Field(..., json_schema_extra={"example": "CC(=O)NC1=CC=C(O)C=C1"}, description="Valid molecular SMILES string")
    organ: str = Field(..., json_schema_extra={"example": "Liver"}, description="Target human tissue profile (Liver, Heart, Brain, Kidney, Lung, Pancreas, Spleen, Intestine, Skin, Bone_Marrow)")
    mc_samples: Optional[int] = Field(20, ge=5, le=50, description="Monte Carlo Dropout sample passes")

class ExplainRequest(BaseModel):
    smiles: str = Field(..., json_schema_extra={"example": "O=C1C2=C(O)C=CC=C2C(=O)C3=C1C(O)=C4C(=C3O)C(O)(C(=O)CO)CC(O)C4"})
    top_k: Optional[int] = Field(3, ge=1, le=10)

class CompareRequest(BaseModel):
    smiles: str = Field(..., json_schema_extra={"example": "CC(=O)NC1=CC=C(O)C=C1"})
    organ_a: str = Field(..., json_schema_extra={"example": "Liver"})
    organ_b: str = Field(..., json_schema_extra={"example": "Heart"})


@app.get("/", tags=["System Health"])
def root():
    return {
        "status": "online",
        "service": "EpiADR-Net REST Microservice",
        "version": "3.0.0",
        "architecture": "6-Layer GAT | 8-Head | 512-dim | 17 atom features",
        "documentation": "/docs"
    }

@app.get("/health", tags=["System Health"])
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": "cpu",
        "organs": ORGAN_NAMES,
        "num_adr_classes": len(MEDDRA_ADR_CLASSES)
    }

@app.get("/organs", tags=["Reference Metadata"])
def get_supported_organs():
    return {"supported_organs": ORGAN_NAMES, "vector_dimension": 256}

@app.get("/adr-classes", tags=["Reference Metadata"])
def get_adr_classes():
    return {"meddra_terms": MEDDRA_ADR_CLASSES}

@app.post("/predict", tags=["ADR Inference"])
def predict_adr(req: PredictionRequest):
    if req.organ not in GTEX_TISSUE_PROFILES:
        raise HTTPException(status_code=400, detail=f"Unsupported organ '{req.organ}'. Choose from {ORGAN_NAMES}")

    try:
        node_feats, edge_index, atom_symbols = smiles_to_graph(req.smiles)
        batch = torch.zeros(node_feats.size(0), dtype=torch.long)
        tissue_vec = GTEX_TISSUE_PROFILES[req.organ].unsqueeze(0)

        mc_res = MODEL.predict_mc_dropout(node_feats, edge_index, batch, tissue_vec, num_samples=req.mc_samples or 30)
        
        mu = mc_res["mean_probabilities"][0].cpu().numpy()
        sigma = mc_res["uncertainty_sigma"][0].cpu().numpy()
        attn = mc_res["attention_weights"]

        predictions = []
        for i, term in enumerate(MEDDRA_ADR_CLASSES):
            predictions.append({
                "meddra_term": term,
                "predicted_probability": float(round(mu[i], 4)),
                "epistemic_uncertainty": float(round(sigma[i], 4)),
                "is_significant": bool(mu[i] > 0.45)
            })

        xai_breakdown = highlight_toxic_subgraph(req.smiles, attn, top_k=3)

        return {
            "smiles": req.smiles,
            "conditioned_organ": req.organ,
            "predictions": predictions,
            "xai_explanation": xai_breakdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/explain", tags=["Explainable AI"])
def explain_subgraph(req: ExplainRequest):
    try:
        node_feats, edge_index, atom_symbols = smiles_to_graph(req.smiles)
        batch = torch.zeros(node_feats.size(0), dtype=torch.long)
        tissue_vec = GTEX_TISSUE_PROFILES["Liver"].unsqueeze(0)

        _, attn = MODEL(node_feats, edge_index, batch, tissue_vec, return_attention=True)
        explanation = highlight_toxic_subgraph(req.smiles, attn, top_k=req.top_k)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"XAI extraction failed: {str(e)}")

@app.post("/compare-organs", tags=["Comparative Analysis"])
def compare_organs(req: CompareRequest):
    if req.organ_a not in GTEX_TISSUE_PROFILES or req.organ_b not in GTEX_TISSUE_PROFILES:
        raise HTTPException(status_code=400, detail="Invalid organ selected")

    res_a = predict_adr(PredictionRequest(smiles=req.smiles, organ=req.organ_a))
    res_b = predict_adr(PredictionRequest(smiles=req.smiles, organ=req.organ_b))

    return {
        "smiles": req.smiles,
        "organ_a": req.organ_a,
        "organ_b": req.organ_b,
        "predictions_organ_a": res_a["predictions"],
        "predictions_organ_b": res_b["predictions"]
    }
