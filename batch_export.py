import argparse
import pandas as pd
import torch
import os
from data_loader import GTEX_TISSUE_PROFILES, MEDDRA_ADR_CLASSES, ORGAN_NAMES
from model import EpiADRNet
from utils import smiles_to_graph

def batch_export_predictions(input_csv: str, output_csv: str, organ: str, mc_samples: int = 30):
    if organ not in GTEX_TISSUE_PROFILES:
        raise ValueError(f"Unsupported organ '{organ}'. Choose from {ORGAN_NAMES}")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file {input_csv} not found.")

    print(f"Loading EpiADR-Net v5 model...")
    model = EpiADRNet(
        in_features=24, hidden_dim=1536, tissue_dim=1024,
        num_classes=10, num_gat_layers=12, num_heads=16, dropout=0.1,
    )
    
    if os.path.exists("model.pt"):
        model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
    else:
        print("Warning: 'model.pt' not found. Using untrained weights for demonstration.")
    
    model.eval()
    
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if "smiles" not in df.columns:
        raise ValueError("Input CSV must contain a 'smiles' column.")
    
    results = []
    tissue_vec = GTEX_TISSUE_PROFILES[organ].unsqueeze(0)
    
    print(f"Running inference for organ: {organ}...")
    for idx, row in df.iterrows():
        smiles = row["smiles"]
        try:
            node_feats, edge_index, _ = smiles_to_graph(smiles)
            batch = torch.zeros(node_feats.size(0), dtype=torch.long)
            
            mc_res = model.predict_mc_dropout(node_feats, edge_index, batch, tissue_vec, num_samples=mc_samples)
            mu = mc_res["mean_probabilities"][0].cpu().numpy()
            sigma = mc_res["uncertainty_sigma"][0].cpu().numpy()
            
            res_dict = {"smiles": smiles, "organ": organ}
            for i, term in enumerate(MEDDRA_ADR_CLASSES):
                res_dict[f"{term}_Prob"] = round(float(mu[i]), 4)
                res_dict[f"{term}_Uncertainty"] = round(float(sigma[i]), 4)
                
            results.append(res_dict)
        except Exception as e:
            print(f"Failed to process SMILES '{smiles}': {e}")
            
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"Successfully exported {len(out_df)} predictions to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Export EpiADR-Net Predictions to CSV")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file containing a 'smiles' column")
    parser.add_argument("--output", type=str, default="predictions_export.csv", help="Output CSV file path")
    parser.add_argument("--organ", type=str, default="Liver", help="Target organ for tissue conditioning")
    parser.add_argument("--mc-samples", type=int, default=30, help="Number of Monte Carlo samples for uncertainty")
    args = parser.parse_args()
    
    batch_export_predictions(args.input, args.output, args.organ, args.mc_samples)
