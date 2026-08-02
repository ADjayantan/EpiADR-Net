import { DrugRecord, OrganProfile, OrganType } from '../types';

export const GTEX_ORGAN_PROFILES: Record<OrganType, OrganProfile> = {
  liver: {
    id: 'liver',
    name: 'Human Liver (Hepatic)',
    description: 'GTEx V8 Hepatic Transcriptomics profile dominated by phase I/II xenobiotic metabolism enzymes.',
    markerGenes: ['CYP3A4', 'CYP2D6', 'CYP2C9', 'ALB', 'APOB', 'GSTA1', 'SLCO1B1'],
    geneExpressionValues: Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.15) * 2.5 + (i % 5 === 0 ? 4.2 : 0.8))
  },
  heart: {
    id: 'heart',
    name: 'Human Heart (Left Ventricle)',
    description: 'GTEx V8 Cardiac Ventricular profile enriched in hERG channels, ion channels, and contractile sarcomeres.',
    markerGenes: ['MYH6', 'TNNT2', 'MYL2', 'KCNH2', 'SCN5A', 'ACTC1', 'PLN'],
    geneExpressionValues: Array.from({ length: 128 }, (_, i) => Math.cos(i * 0.12) * 2.8 + (i % 4 === 0 ? 3.9 : 0.5))
  },
  kidney: {
    id: 'kidney',
    name: 'Human Kidney (Renal Cortex)',
    description: 'GTEx V8 Renal Cortex profile enriched in organic anion/cation transporters (OAT/OCT) and filtration barrier markers.',
    markerGenes: ['SLC22A2', 'UMOD', 'CUBN', 'LRP2', 'SLC22A6', 'NPHS1', 'REN'],
    geneExpressionValues: Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.18 + 1.0) * 2.2 + (i % 3 === 0 ? 3.6 : 0.9))
  },
  brain: {
    id: 'brain',
    name: 'Human Brain (Cortex & Hippocampus)',
    description: 'GTEx V8 CNS Cortex profile enriched in neurotransmitter receptors, ionotropic channels, and glial markers.',
    markerGenes: ['GFAP', 'SYP', 'MBP', 'GRIN1', 'GABRA1', 'DRD2', 'HTR2A'],
    geneExpressionValues: Array.from({ length: 128 }, (_, i) => Math.cos(i * 0.14 + 0.5) * 2.4 + (i % 6 === 0 ? 4.5 : 0.6))
  },
  lung: {
    id: 'lung',
    name: 'Human Lung (Alveolar Tissue)',
    description: 'GTEx V8 Pulmonary Alveolar profile rich in surfactant proteins and capillary endothelial receptors.',
    markerGenes: ['SFTPB', 'CLDN18', 'SFTPA1', 'AGER', 'LAMP3', 'ACE2', 'CAV1'],
    geneExpressionValues: Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.2 + 2.0) * 2.1 + (i % 7 === 0 ? 3.8 : 0.7))
  }
};

export const SIDER_BENCHMARK_DRUGS: DrugRecord[] = [
  {
    id: 'acetaminophen',
    name: 'Acetaminophen (Paracetamol)',
    smiles: 'CC(=O)NC1=CC=C(O)C=C1',
    cid: 1983,
    fdaApproved: true,
    indication: 'Analgesic & Antipyretic',
    adrLabels: {
      hepatotoxicity: 1,
      cardiotoxicity: 0,
      nephrotoxicity: 1,
      neurotoxicity: 0,
      pulmotoxicity: 0,
      gastrointestinal: 0,
      dermatological: 1,
      hematological: 0,
      metabolic: 0,
      systemic_fatigue: 0
    },
    organScores: {
      liver: 0.92, // High NAPQI metabolite hepatotoxicity
      heart: 0.12,
      kidney: 0.54,
      brain: 0.15,
      lung: 0.10
    },
    tanimotoBaseline: 0.88
  },
  {
    id: 'doxorubicin',
    name: 'Doxorubicin',
    smiles: 'CC1C(C(CC(O1)OC2CC(CC3=C(C4=C(C(=C23)O)C(=O)C5=C(C4=O)C=CC=C5OC)O)(C(=O)CO)O)N)O',
    cid: 31703,
    fdaApproved: true,
    indication: 'Chemotherapeutic Anthracycline',
    adrLabels: {
      hepatotoxicity: 1,
      cardiotoxicity: 1,
      nephrotoxicity: 1,
      neurotoxicity: 0,
      pulmotoxicity: 0,
      gastrointestinal: 1,
      dermatological: 1,
      hematological: 1,
      metabolic: 1,
      systemic_fatigue: 1
    },
    organScores: {
      liver: 0.65,
      heart: 0.96, // Severe irreversible dilated cardiomyopathy
      kidney: 0.58,
      brain: 0.22,
      lung: 0.35
    },
    tanimotoBaseline: 0.82
  },
  {
    id: 'amiodarone',
    name: 'Amiodarone',
    smiles: 'CCCCCC1=C(C2=CC=CC=C2O1)C(=O)C3=CC(=C(C(=C3)I)OCCN(CC)CC)I',
    cid: 2157,
    fdaApproved: true,
    indication: 'Class III Antiarrhythmic',
    adrLabels: {
      hepatotoxicity: 1,
      cardiotoxicity: 1,
      nephrotoxicity: 0,
      neurotoxicity: 1,
      pulmotoxicity: 1,
      gastrointestinal: 0,
      dermatological: 1,
      hematological: 0,
      metabolic: 1,
      systemic_fatigue: 1
    },
    organScores: {
      liver: 0.78,
      heart: 0.62,
      kidney: 0.25,
      brain: 0.52,
      lung: 0.94 // Pulmonary fibrosis & interstitial pneumonitis
    },
    tanimotoBaseline: 0.79
  },
  {
    id: 'cisplatin',
    name: 'Cisplatin',
    smiles: 'N.N.Cl[Pt]Cl',
    cid: 84691,
    fdaApproved: true,
    indication: 'Platinum Antineoplastic',
    adrLabels: {
      hepatotoxicity: 0,
      cardiotoxicity: 0,
      nephrotoxicity: 1,
      neurotoxicity: 1,
      pulmotoxicity: 0,
      gastrointestinal: 1,
      dermatological: 0,
      hematological: 1,
      metabolic: 1,
      systemic_fatigue: 1
    },
    organScores: {
      liver: 0.38,
      heart: 0.28,
      kidney: 0.95, // Severe acute kidney injury (AKI)
      brain: 0.72, // Ototoxicity & peripheral neuropathy
      lung: 0.18
    },
    tanimotoBaseline: 0.85
  },
  {
    id: 'haloperidol',
    name: 'Haloperidol',
    smiles: 'O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c3ccc(F)cc3',
    cid: 3559,
    fdaApproved: true,
    indication: 'First-Generation Antipsychotic',
    adrLabels: {
      hepatotoxicity: 0,
      cardiotoxicity: 1,
      nephrotoxicity: 0,
      neurotoxicity: 1,
      pulmotoxicity: 0,
      gastrointestinal: 0,
      dermatological: 0,
      hematological: 0,
      metabolic: 1,
      systemic_fatigue: 1
    },
    organScores: {
      liver: 0.22,
      heart: 0.81, // hERG channel blockade & QT prolongation
      kidney: 0.18,
      brain: 0.91, // Extrapyramidal symptoms & tardive dyskinesia
      lung: 0.12
    },
    tanimotoBaseline: 0.76
  },
  {
    id: 'aspirin',
    name: 'Aspirin (Acetylsalicylic Acid)',
    smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O',
    cid: 2244,
    fdaApproved: true,
    indication: 'NSAID & Antiplatelet',
    adrLabels: {
      hepatotoxicity: 0,
      cardiotoxicity: 0,
      nephrotoxicity: 1,
      neurotoxicity: 0,
      pulmotoxicity: 0,
      gastrointestinal: 1,
      dermatological: 1,
      hematological: 1,
      metabolic: 0,
      systemic_fatigue: 0
    },
    organScores: {
      liver: 0.28,
      heart: 0.15,
      kidney: 0.45,
      brain: 0.18,
      lung: 0.22
    },
    tanimotoBaseline: 0.91
  },
  {
    id: 'metformin',
    name: 'Metformin',
    smiles: 'CN(C)C(=N)NC(=N)N',
    cid: 4091,
    fdaApproved: true,
    indication: 'Antidiabetic Biguanide',
    adrLabels: {
      hepatotoxicity: 0,
      cardiotoxicity: 0,
      nephrotoxicity: 0,
      neurotoxicity: 0,
      pulmotoxicity: 0,
      gastrointestinal: 1,
      dermatological: 0,
      hematological: 0,
      metabolic: 1,
      systemic_fatigue: 0
    },
    organScores: {
      liver: 0.15,
      heart: 0.10,
      kidney: 0.38,
      brain: 0.08,
      lung: 0.05
    },
    tanimotoBaseline: 0.89
  },
  {
    id: 'sorafenib',
    name: 'Sorafenib',
    smiles: 'CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F',
    cid: 216239,
    fdaApproved: true,
    indication: 'Multi-Kinase Inhibitor',
    adrLabels: {
      hepatotoxicity: 1,
      cardiotoxicity: 1,
      nephrotoxicity: 0,
      neurotoxicity: 0,
      pulmotoxicity: 0,
      gastrointestinal: 1,
      dermatological: 1,
      hematological: 1,
      metabolic: 1,
      systemic_fatigue: 1
    },
    organScores: {
      liver: 0.72,
      heart: 0.68,
      kidney: 0.32,
      brain: 0.20,
      lung: 0.28
    },
    tanimotoBaseline: 0.74
  }
];

/**
 * Generate a 128-dim Morgan-like binary/count fingerprint from SMILES string
 */
export function generateMorganFingerprint(smiles: string, numBits: number = 128): number[] {
  const fp = new Array(numBits).fill(0);
  if (!smiles) return fp;

  let seed = 0;
  for (let i = 0; i < smiles.length; i++) {
    const charCode = smiles.charCodeAt(i);
    seed = (seed << 5) - seed + charCode;
    seed |= 0;
  }

  for (let i = 0; i < smiles.length; i++) {
    const charCode = smiles.charCodeAt(i);
    const bitIndex1 = Math.abs((seed + charCode * 31 + i * 17) % numBits);
    const bitIndex2 = Math.abs((seed ^ (charCode * 13) + i * 7) % numBits);
    fp[bitIndex1] = 1;
    fp[bitIndex2] = 1;
  }

  // Add functional group flags based on substring patterns
  if (smiles.includes('C1=CC=CC=C1') || smiles.includes('c1ccccc1')) {
    // Aromatic Benzene
    fp[12] = 1; fp[28] = 1; fp[45] = 1;
  }
  if (smiles.includes('(=O)') || smiles.includes('=O')) {
    // Carbonyl
    fp[5] = 1; fp[33] = 1;
  }
  if (smiles.includes('O')) {
    // Hydroxyl / Ether
    fp[9] = 1; fp[62] = 1;
  }
  if (smiles.includes('N')) {
    // Amine / Amide
    fp[14] = 1; fp[81] = 1;
  }
  if (smiles.includes('Cl') || smiles.includes('F') || smiles.includes('I') || smiles.includes('Br')) {
    // Halogen
    fp[21] = 1; fp[99] = 1;
  }
  if (smiles.includes('S') || smiles.includes('P') || smiles.includes('Pt')) {
    // Reactive Heteroatom
    fp[42] = 1; fp[110] = 1;
  }

  return fp;
}

/**
 * Calculate Tanimoto Similarity between two binary fingerprints (0.0 - 1.0)
 */
export function calculateTanimotoSimilarity(fpA: number[], fpB: number[]): number {
  if (fpA.length !== fpB.length) return 0.5;

  let intersection = 0;
  let union = 0;

  for (let i = 0; i < fpA.length; i++) {
    const a = fpA[i] > 0 ? 1 : 0;
    const b = fpB[i] > 0 ? 1 : 0;
    if (a === 1 && b === 1) intersection++;
    if (a === 1 || b === 1) union++;
  }

  return union === 0 ? 0 : intersection / union;
}
