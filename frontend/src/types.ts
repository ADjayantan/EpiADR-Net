export type OrganType = 'liver' | 'heart' | 'kidney' | 'brain' | 'lung';

export interface OrganProfile {
  id: OrganType;
  name: string;
  description: string;
  markerGenes: string[];
  geneExpressionValues: number[]; // 128-dim GTEx transcriptomic vector
}

export type MedDRAToxicityClass =
  | 'hepatotoxicity'
  | 'cardiotoxicity'
  | 'nephrotoxicity'
  | 'neurotoxicity'
  | 'pulmotoxicity'
  | 'gastrointestinal'
  | 'dermatological'
  | 'hematological'
  | 'metabolic'
  | 'systemic_fatigue';

export interface DrugRecord {
  id: string;
  name: string;
  smiles: string;
  cid?: number;
  fdaApproved: boolean;
  indication: string;
  adrLabels: Record<MedDRAToxicityClass, number>; // 0 or 1
  organScores: Record<OrganType, number>; // Ground truth / benchmark risk score (0.0 to 1.0)
  tanimotoBaseline: number; // Tanimoto similarity to SIDER 4.1 reference set
}

export interface EpiADRHyperparameters {
  useTissueConditioning: boolean; // True = Cross-Attention GTEx V8, False = Baseline Molecule-Only
  crossAttentionHeads: number; // Default 16
  learningRate: number; // Default 0.001
  epochs: number; // Default 50
  batchSize: number; // Default 64
  optimizer: 'adam' | 'adamw' | 'sgd';
  posWeight: number; // Class imbalance weight (e.g. 2.5)
  regularizationL2: number;
  dropoutRate: number; // Default 0.2
  mcDropoutPasses: number; // N=30 for uncertainty quantification
}

export interface EpochTrainingMetric {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAUROC: number;
  valAUROC: number;
}

export interface ModelTrainingSummary {
  isTrained: boolean;
  trainingTimeMs: number;
  finalTrainLoss: number;
  finalValLoss: number;
  finalValAUROC: number;
  finalF1Score: number;
  epochHistory: EpochTrainingMetric[];
  confusionMatrix: {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  };
}

export interface AtomicHotspot {
  atomIndex: number;
  symbol: string;
  attentionWeight: number; // 0.0 to 1.0 (XAI Toxicity Hotspot contribution)
  subgraphName?: string;
}

export interface ToxicityPredictionResult {
  compoundName: string;
  smiles: string;
  useTissueConditioning: boolean;
  
  // Organ-Specific Zero-Shot Toxicity Scores (0.0 - 1.0)
  organScores: Record<OrganType, {
    meanRisk: number; // μ
    uncertaintySigma: number; // σ from MC Dropout
    riskLevel: 'Low' | 'Moderate' | 'High' | 'Severe';
  }>;

  // MedDRA Clinical Toxicity Classes Risk
  meddraScores: Record<MedDRAToxicityClass, number>;

  // Tanimoto Structural Applicability Domain
  tanimotoSimilarity: number;
  applicabilityDomain: 'High Confidence (In-Domain)' | 'Moderate Confidence' | 'Out-of-Domain (Novel Scaffold)';
  domainColor: 'green' | 'yellow' | 'red';

  // Atomic Hotspots (Explainable AI)
  atomicHotspots: AtomicHotspot[];

  // Top Driving Toxicophores
  toxicophores: string[];
}
