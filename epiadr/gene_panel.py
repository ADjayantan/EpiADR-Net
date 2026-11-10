"""
Pre-specified pharmacological gene panel for the mechanism-feature experiment.

The panel is fixed BEFORE any model is fitted. It is chosen by mechanism, not by
measured performance, so that the mechanism result cannot be a product of gene
selection on the test set. Every gene is here because it is a documented route
by which a small molecule produces one of the five organ toxicities in the task,
or because it governs the exposure that drives them.

Grouping is documentation, not a feature: the model sees one flat vector.
"""
from __future__ import annotations

GENE_PANEL: dict[str, list[str]] = {
    # Phase I metabolism — determines hepatic exposure and reactive-metabolite load.
    "cyp": [
        "CYP1A2", "CYP2B6", "CYP2C8", "CYP2C9", "CYP2C19",
        "CYP2D6", "CYP2E1", "CYP3A4", "CYP3A5",
    ],
    # Phase II conjugation — detoxification capacity.
    "phase2": ["UGT1A1", "NAT2", "GSTP1", "SULT1A1", "TPMT", "COMT"],
    # Transporters — ABCB11/BSEP inhibition is the canonical cholestatic injury
    # route; SLC22A6/8 (OAT1/3) and SLC22A2 (OCT2) drive renal tubular exposure;
    # ABCB1/ABCG2 gate brain and biliary penetration.
    "transporter": [
        "ABCB1", "ABCG2", "ABCC2", "ABCB11", "SLCO1B1", "SLCO1B3",
        "SLC22A1", "SLC22A2", "SLC22A6", "SLC22A8", "SLC47A1",
    ],
    # Cardiac electrophysiology — KCNH2 (hERG) block is the textbook mechanism
    # for drug-induced arrhythmia.
    "cardiac_ion_channel": ["KCNH2", "SCN5A", "CACNA1C", "KCNQ1", "KCNE1", "KCNJ11"],
    # CNS receptors and transporters — seizure threshold, sedation, serotonergic
    # and dopaminergic adverse effects.
    "cns": [
        "GABRA1", "GABRA2", "GRIN1", "GRIN2B", "HTR1A", "HTR2A", "HTR2C",
        "DRD1", "DRD2", "CHRM1", "CHRM2", "CHRM3", "HRH1",
        "SLC6A2", "SLC6A3", "SLC6A4", "OPRM1", "CNR1", "ADRA1A", "ADRA2A",
    ],
    # Eicosanoid and inflammatory signalling — NSAID renal and GI injury,
    # bronchospasm, drug-induced pneumonitis.
    "inflammation": [
        "PTGS1", "PTGS2", "ALOX5", "TBXA2R", "LTB4R", "TNF", "IL6", "PTGER2",
    ],
    # Nuclear receptors — induction, steatosis, and endocrine adverse effects.
    "nuclear_receptor": [
        "NR1I2", "NR1I3", "PPARA", "PPARG", "ESR1", "AR", "NR3C1", "THRB",
    ],
    # Mitochondrial and genotoxic stress — hepatocellular and cardiac injury.
    "stress": ["TOP2A", "TOP2B", "XDH", "AOX1", "SOD1", "CAT"],
    # Kinases — targeted-therapy cardiotoxicity and pneumonitis.
    "kinase": ["EGFR", "KDR", "ABL1", "SRC", "ERBB2", "MTOR"],
    # Renal and vascular handling.
    "renal_vascular": ["ACE", "NOS3", "AQP2", "SLC12A1", "REN", "AGTR1"],
    # Other high-frequency small-molecule targets that modulate organ exposure.
    "other": ["ACHE", "MAOA", "MAOB", "CA2", "HMGCR", "ADORA1", "ADORA2A", "PDE4D"],
}

#: Flat, de-duplicated, order-stable panel.
PANEL: list[str] = list(dict.fromkeys(g for genes in GENE_PANEL.values() for g in genes))

#: Task organ -> the HPA consensus tissue whose expression represents it.
#: "cerebral cortex" stands for Brain, matching the GTEx aggregation in
#: DATA_PROVENANCE.md, which averaged cortical and sub-cortical regions.
ORGAN_TO_HPA_TISSUE: dict[str, str] = {
    "Liver": "liver",
    "Heart": "heart muscle",
    "Brain": "cerebral cortex",
    "Kidney": "kidney",
    "Lung": "lung",
}
