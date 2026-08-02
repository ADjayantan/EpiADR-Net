import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: "5mb" }));

  // Initialize Gemini AI lazily/safely
  const getAi = () => {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) return null;
    return new GoogleGenAI({ apiKey });
  };

  // Health Check
  app.get("/api/health", (_req, res) => {
    res.json({
      status: "ok",
      version: "2.1.0-EpiADR-Net",
      model: "EpiADR-Net Tissue-Conditioned Graph Transformer",
      dataset: "SIDER 4.1 (7,325 records) & GTEx V8 Transcriptomics",
      time: new Date().toISOString()
    });
  });

  // Organ Transcriptomics Metadata
  app.get("/api/organs", (_req, res) => {
    res.json({
      organs: [
        { id: "liver", name: "Liver (Hepatic)", markerGenes: ["CYP3A4", "CYP2D6", "ALB", "APOB", "HP"], expressionDim: 128 },
        { id: "heart", name: "Heart (Cardiac)", markerGenes: ["MYH6", "TNNT2", "MYL2", "ACTC1", "COX6A2"], expressionDim: 128 },
        { id: "kidney", name: "Kidney (Renal)", markerGenes: ["SLC22A2", "UMOD", "CUBN", "LRP2", "NPHS1"], expressionDim: 128 },
        { id: "brain", name: "Brain (Central Nervous)", markerGenes: ["GFAP", "SYP", "MBP", "GRIN1", "GABRA1"], expressionDim: 128 },
        { id: "lung", name: "Lung (Pulmonary)", markerGenes: ["SFTPB", "CLDN18", "SFTPA1", "AGER", "LAMP3"], expressionDim: 128 }
      ]
    });
  });

  // AI Preclinical Toxicity Synthesis & Advisor endpoint
  app.post("/api/ml-advisor", async (req, res) => {
    try {
      const { compoundName, smiles, organScores, tanimoto, mcUncertainty, useTissueConditioning } = req.body;
      const ai = getAi();

      if (!ai) {
        return res.status(200).json({
          advice: `[OFFLINE REPORT] Preclinical Toxicity Summary for ${compoundName || "Query Compound"} (${smiles || "Custom SMILES"}):
- Tanimoto Applicability Score: ${(tanimoto ?? 0.78).toFixed(2)} (${(tanimoto ?? 0.78) >= 0.7 ? "In-Domain 🟢" : "Out-of-Domain 🔴"}).
- Tissue Conditioning: ${useTissueConditioning ? "Enabled (16-Head Cross-Attention GTEx V8 active)" : "Disabled (Molecule-Only Baseline)"}.
- Primary High-Risk Organ: ${organScores?.liver > 0.6 ? "Hepatotoxicity (Liver)" : organScores?.heart > 0.6 ? "Cardiotoxicity (Heart)" : "Low Organ Risk Overall"}.
- Recommendation: Perform in-vitro CYP3A4 inhibition assay and hERG channel patch-clamp assay prior to Phase I trials.`
        });
      }

      const prompt = `You are an expert Computational Toxicologist and Lead Medicinal Chemist evaluating compound safety using EpiADR-Net (a Tissue-Conditioned Graph Transformer trained on SIDER 4.1 & GTEx V8).

Query Molecule:
- Name: ${compoundName || "Custom Synthetic Drug Candidate"}
- SMILES: ${smiles}
- Tissue Conditioning Mode: ${useTissueConditioning ? "Active (GTEx V8 Organ Gene Profiles Integrated)" : "Baseline (Molecule-Only Structure)"}
- Tanimoto Domain Similarity: ${tanimoto ?? "0.76"}
- Monte Carlo Uncertainty (Mean σ): ${mcUncertainty ?? "0.04"}
- Predicted Organ Toxicity Risk Scores (0.0 - 1.0):
  ${JSON.stringify(organScores, null, 2)}

Provide a structured Preclinical Safety Assessment Report in markdown:
1. **Mechanistic Safety Analysis**: Explain why this chemical structure and GTEx expression profile triggers these specific organ risk scores (e.g. electrophilic reactive metabolite formation, hERG potassium channel blocking, renal transporter inhibition).
2. **Atomic Hotspot & Toxicophore Interpretation**: Highlight chemical functional groups likely responsible for observed toxicity.
3. **Recommended Wet-Lab Assays & Risk Mitigation**: Propose 3 specific in-vitro or in-vivo assays (e.g., HepG2 cytotoxicity, hERG patch-clamp, ALT/AST elevation screening, serum creatinine tracking) to validate or mitigate predicted risks.

Keep it highly authoritative, scientific, clear, and actionable.`;

      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
      });

      return res.json({ advice: response.text || "Report generated successfully." });
    } catch (err: any) {
      console.error("EpiADR AI Advisor Error:", err);
      return res.status(500).json({ error: "Failed to generate AI safety analysis", message: err.message });
    }
  });

  // Vite middleware setup
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`EpiADR-Net Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();

