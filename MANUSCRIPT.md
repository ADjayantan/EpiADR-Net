# Organ-conditioned adverse drug reaction prediction: drug–target activity helps, tissue transcriptomics does not

**A. D. Jayantan**
Department of Computer Science and Engineering, VSB College of Engineering Technical Campus, Coimbatore
Repository: [github.com/ADjayantan/EpiADR-Net](https://github.com/ADjayantan/EpiADR-Net)

> **Status.** Working draft. All figures in this document are produced by scripts
> in the repository and written to `results/*.json`. An earlier version of this
> manuscript reported a Macro-AUROC of 0.9680 on "150+ FDA-approved drugs and 10
> human organs (15,000 samples)". Those figures were withdrawn; the pipeline that
> produced them trained on five molecules and conditioned on random vectors. The
> audit is published alongside this document as `AUDIT.md`.

---

## Abstract

Organ-specific adverse drug reaction (ADR) prediction is commonly framed as
conditioning a molecular representation on tissue transcriptomics, on the
intuition that a drug's effect in an organ depends on that organ's expression
programme. We construct a leakage-audited benchmark for this task from SIDER 4.1
and GTEx V8 — 1 465 molecules across five organs, with molecule-level
Bemis-Murcko scaffold cross-validation and assertions that no molecule and no
scaffold is shared between train and test — and we report a negative result.

Conditioning on real 128-dimensional GTEx organ profiles improves macro-AUROC by
**+0.005** over a five-dimensional organ one-hot, against a fold standard
deviation of 0.012–0.020. Deranging the profiles, so that every organ is given
another organ's transcriptome, changes macro-AUROC by **−0.0001**: the model uses
the tissue vector as an organ index and nothing else.

We then test the mechanistic route that this failure suggests. Adding PubChem
drug–target activity over a pre-specified 94-gene pharmacological panel raises
macro-AUROC to **0.6547 ± 0.011** (+0.035 over the one-hot) — a real gain, and
the best result we measure. Multiplying that binding by each target's per-organ
expression, however, adds nothing identifiable: a 12-fold permutation test of
the organ→expression assignment places the correct map at **p = 0.385**, with 4
of 12 wrong assignments scoring higher. The drug side of the product carries
signal; the tissue side does not. In the strictest generalisation setting — an
unseen drug evaluated in an unseen organ — performance remains near chance
(0.579).

We trace the cause to label provenance rather than to model capacity. Organ
assignment in SIDER-derived datasets is produced by mapping MedDRA System Organ
Class terms to organs, a terminological operation that never consults tissue
measurement. With five organs, organ identity is five bits of information, which
a one-hot already carries in full. We argue that transcriptomic conditioning on
this class of label is unfalsifiable by construction, and that demonstrating it
requires either labels resolved by biology rather than terminology, or features
that couple molecule and tissue mechanistically.

---

## 1. Introduction

Adverse drug reactions are a leading cause of late-stage attrition and
post-market withdrawal, and whether a reaction manifests in liver, heart, brain,
kidney or lung is clinically decisive. A natural modelling response is to
condition a molecular encoder on a representation of the target tissue. Several
public datasets make this appear straightforward: SIDER provides drug–side-effect
associations, MedDRA organises side-effect terms into System Organ Classes, and
GTEx provides per-tissue expression profiles.

This paper reports what happens when that pipeline is built carefully and
evaluated honestly. The result is negative, and the negative result is specific
and diagnosable: it is a property of how the labels were produced, and it will
reproduce for any study that assembles organ labels the same way.

**Contributions.**

1. A reproducible organ-conditioned ADR benchmark with molecule-level scaffold
   splits and machine-checked leakage assertions (§3).
2. A demonstration that pooled AUROC is not a valid headline metric on this task:
   a predictor with no molecular input reaches pooled AUROC 0.656 (§4.1).
3. A tissue-derangement ablation showing that transcriptomic conditioning
   contributes nothing beyond organ identity (§4.2).
4. A three-setting generalisation protocol (S1/S2/S3) showing that the "zero-shot"
   framing common in this literature is near chance when tested strictly (§4.3).
5. A mechanistic-coupling experiment with real drug–target and per-gene
   expression data, separating the drug side of the product (which helps) from
   the tissue side (which does not), with a permutation test (§4.4).
6. An account of why, grounded in label provenance (§5).

## 2. Task formulation

Raw SIDER-derived records associate a molecule with ten binary MedDRA-derived ADR
labels and replicate the identical label vector across five organ rows. Ten
labels replicated five times is not an organ-conditioned task: the target does
not vary with the conditioning variable, so no conditioning signal is
identifiable. In our data, **0 of 1 465** molecules have a label vector that
differs across their organ rows.

We therefore use the organ partition the ADR names already imply:

| scope | mapping |
|:--|:--|
| organ-specific | Liver→Hepatotoxicity, Heart→Arrhythmia, Brain→Seizure, Kidney→Nephrotoxicity, Lung→Pneumonitis |
| systemic | Nausea, Headache, Dizziness, Fatigue, Rash |

A row is `(molecule, organ) → y`, where `y` is the organ-specific ADR of *that*
organ. The target now varies across a molecule's five rows, and the derangement
ablation of §4.2 becomes meaningful.

Positive rates vary sharply by organ: Lung 7.8 %, Liver 18.5 %, Kidney 26.5 %,
Heart 29.2 %, Brain 38.0 %. Organ-specific ADRs are weakly positively correlated
across organs (mean pairwise *r* = 0.196); 538 of 1 465 molecules are negative in
all five organs and 17 are positive in all five.

## 3. Experimental protocol

**Split.** Bemis-Murcko scaffolds are computed per molecule and assigned whole to
folds, so all five organ rows of a molecule move together. `verify_no_leakage`
asserts zero shared molecules and zero shared scaffolds on every fold and raises
otherwise; the assertion runs inside the experiment scripts and in CI.

**Metric.** Per-organ AUROC, macro-averaged. Within one organ the base rate is
constant, so per-organ AUROC measures only drug ranking. AUPRC is reported
alongside because positives are the minority class in every organ, and expected
calibration error because the deployed interface shows a probability.

**Models.** Morgan fingerprints (ECFP4, 2048 bit) with logistic regression and
with XGBoost, fitted either per organ or as a single model over four feature
configurations: molecule only; molecule + organ one-hot; molecule + real GTEx
profile; molecule + deranged GTEx profile. Seed 42 throughout.

## 4. Results

### 4.1 Pooled AUROC is not a valid headline

| predictor | macro-AUROC | pooled-AUROC |
|:--|--:|--:|
| per-organ base rate, no molecular input | **0.5000** | **0.6561** |
| Morgan + LogReg, per organ | 0.5990 ± 0.021 | 0.6358 |
| Morgan + XGBoost, per organ | **0.6364 ± 0.016** | 0.6904 |

A predictor that emits each organ's training base rate and ignores the molecule
entirely reaches pooled AUROC 0.656 — above the pooled score of a fingerprint
logistic regression — while ranking no drug correctly. Pooled AUROC on this task
partly measures the Lung/Brain prevalence gap. We report per-organ AUROC
throughout and recommend that work on this task do the same.

### 4.2 Transcriptomic conditioning does not survive a derangement ablation

| features | macro-AUROC | macro-AUPRC |
|:--|--:|--:|
| molecule only | 0.5888 ± 0.013 | — |
| + organ one-hot (5 dim) | 0.6197 ± 0.020 | 0.3522 ± 0.015 |
| + real GTEx profile (128 dim) | 0.6243 ± 0.012 | 0.3510 ± 0.020 |
| + **deranged** GTEx profile (128 dim) | 0.6244 ± 0.013 | 0.3513 ± 0.023 |

Real profiles beat a one-hot by +0.005, inside one fold standard deviation.
Replacing them with a derangement — every organ receives a different organ's
transcriptome, so the vector is uninformative about which organ the row describes
while all marginals are preserved — changes macro-AUROC by −0.0001. The model
treats the 128-dimensional profile as a lookup key for organ identity.

Five independent per-organ models (0.6364) outperform the shared
tissue-conditioned model (0.6243), so the shared-conditioning architecture is not
buying parameter efficiency either.

### 4.3 Strict generalisation is near chance

| setting | molecule only | + one-hot | + GTEx |
|:--|--:|--:|--:|
| S1 unseen drug, seen organ | 0.5888 | 0.6197 | 0.6243 |
| S2 seen drug, unseen organ | **0.6975** | 0.6905 | 0.6965 |
| S3 unseen drug + unseen organ | 0.5402 | 0.5464 | **0.5517** |

S2 is frequently the setting reported as "zero-shot across tissues". It is the
easiest of the three, because the molecule is not held out: the model has seen
that drug's toxicity in four other organs, and organ ADRs are positively
correlated. The decisive observation is that a **molecule-only model with no
organ input at all is the best S2 model (0.6975)** — whatever is transferring is
drug-level toxicity, not organ-specific biology.

S3, which holds out both drug and organ, is the setting the zero-shot claim
actually describes. All models sit between 0.540 and 0.552.

### 4.4 Drug–target activity helps; its coupling to tissue does not

§4.2 leaves one route untested. Any feature that is a function of the organ
alone is bounded by the one-hot; a mechanism feature is not, because it is a
product $m_g(d, o) = \text{binds}(d, g)\cdot\text{expr}(g, o)$ whose mask
depends on the drug and whose magnitude depends on the organ.

We fix a 94-gene pharmacological panel in advance, by mechanism rather than by
measured performance: hERG/KCNH2 for arrhythmia, ABCB11/BSEP for cholestatic
injury, SLC22A6/OAT1 for renal tubular exposure, GABA-A and NMDA subunits for
seizure threshold, the CYPs for hepatic exposure, and the major transporters and
nuclear receptors that govern exposure. Binding comes from PubChem BioAssay
outcomes (6 582 drug–target hits; 58.6 % of molecules hit at least one panel
gene); per-gene expression comes from Human Protein Atlas consensus tissue nTPM,
used in place of our own PCA-reduced profiles because 128 principal components
cannot be inverted back to genes. The expression table reproduces textbook
selectivity (KCNH2 peaks in heart, CYP3A4 and ABCB11 in liver, SLC22A6 in
kidney, GRIN1 in brain).

| model | macro-AUROC | vs one-hot |
|:--|--:|--:|
| M0 Morgan only | 0.5888 ± 0.013 | −0.031 |
| M1 Morgan + organ one-hot | 0.6197 ± 0.020 | — |
| M2 Morgan + GTEx PCA profile | 0.6243 ± 0.012 | +0.005 |
| M4 Morgan + binding + one-hot | 0.6464 ± 0.018 | +0.027 |
| M5 Morgan + coupling (replacing binding) | 0.6191 ± 0.020 | −0.001 |
| M7 coupling only, no fingerprint | 0.5734 ± 0.015 | −0.046 |
| **M8 Morgan + binding + one-hot + coupling** | **0.6547 ± 0.011** | **+0.035** |

Two observations. First, binding is a real gain: M4 and M8 are the only features
we test that beat the one-hot by more than fold noise. Second, the coupling must
be *added to* the binding vector, not substituted for it — M5 replaces binding
with the product and so zeroes out binding wherever expression is low, losing
information a gradient-boosted tree could otherwise recover from M4's separate
columns as the split `organ == Liver AND binds ABCB11`.

**The permutation test.** M5 beat its single deranged control by +0.0088, which
would have been reported as the first positive signal for tissue conditioning in
this work. We instead constructed the null properly, refitting M8 under 12
independent derangements of the organ→expression map:

| | macro-AUROC |
|:--|--:|
| real assignment | 0.6547 |
| null, 12 derangements | 0.6518 ± 0.0050 (range 0.6410–0.6588) |
| derangements ≥ real | 4 / 12 |
| empirical *p* | **0.385** |

The effect is +0.0029 and not significant. The mechanism block's contribution
survives permuting which organ each gene's expression belongs to, so it is
acting as additional drug-level descriptors — a nonlinear re-encoding of the
binding vector — rather than as tissue biology.

This is a cautionary result in its own right. With a fold standard deviation of
~0.02 and a null spread of ±0.005, a single-derangement ablation cannot separate
+0.008 from noise, and would have produced a false positive here. We recommend a
permutation test over several derangements as the minimum control for
tissue-conditioning claims of this magnitude.

## 5. Why: label provenance, not model capacity

`DATA_PROVENANCE.md` records the organ assignment rule: MedDRA System Organ Class
terms are mapped to physiological organs (hepatic injury terms → Liver, cardiac
arrhythmia terms → Heart, and so on). The organ label therefore encodes *which
ontological class a side-effect term belongs to*. It is a fact about MedDRA's
hierarchy. No measurement of tissue biology enters its construction.

Transcriptomic features cannot recover signal that was never encoded. With five
organs, organ identity carries $\log_2 5 \approx 2.3$ bits, fully expressed by a
five-dimensional one-hot; the remaining 123 dimensions of the GTEx profile
describe biology on which the label has no dependence. Scaling the conditioning
pathway — more heads, deeper cross-attention, higher-dimensional profiles — cannot
change this, because the limitation is in the target, not the function class.

This has a methodological consequence. A study that conditions on
transcriptomics, reports an improvement over a molecule-only baseline, and omits
both the one-hot control and a derangement ablation cannot distinguish "the model
uses tissue biology" from "the model knows which organ this is". We recommend
both controls as a minimum for claims of this kind.

This also explains the asymmetry in §4.4. The binding vector describes the
**drug**, which the labels do encode — SIDER records that a compound causes
hepatotoxicity, and that is a fact about the compound. The expression factor
describes the **organ**, which the labels encode only as an ontological class.
The product therefore inherits signal from one factor and noise from the other,
which is exactly what the permutation test measures.

**What would make the hypothesis testable.** Route (ii) — mechanistic coupling —
is now tested and does not rescue it. What remains is route (i): organ labels
resolved by measurement rather than terminology, such as organ-specific signals
mined from FAERS/OFFSIDES case narratives, or *in vitro* organ-specific toxicity
assays. Until the labels change, every organ-side feature is bounded above by
what a five-dimensional one-hot already encodes.

## 6. Limitations

Single label source (SIDER 4.1); no external validation on OFFSIDES or FAERS, so
the negative result is established for terminology-mapped labels and not for ADR
labels in general. Five organs and one ADR per organ. The graph transformer in
the repository has not been re-benchmarked on the corrected task, so this paper
compares feature sets rather than architectures; a GNN could plausibly raise the
absolute numbers, but the permutation test is architecture-independent in its
logic. PubChem "active" pools curated pharmacology with high-throughput screens,
so the binding matrix means "has measured activity against" rather than a
calibrated affinity, and 41.4 % of molecules hit no panel gene and contribute a
zero binding vector; a richer affinity source could raise the binding gain,
though it would not change which factor the permutation test implicates. The
94-gene panel is pre-specified and was not tuned. ECE is 0.111, so probabilities
should not be read as risk estimates.

## 7. Conclusion

Built carefully, with molecule-level scaffold splits and leakage assertions,
organ-conditioned ADR prediction on SIDER-derived labels reaches macro-AUROC
**0.6547 ± 0.011** once drug–target activity is included — a real +0.035 over an
organ one-hot. Conditioning on tissue transcriptomics adds nothing identifiable,
whether as a GTEx profile (+0.005, derangement −0.0001) or as an explicit
binding × expression product (permutation test *p* = 0.385), and
unseen-drug/unseen-organ generalisation stays near chance. The drug side of the
mechanism carries signal; the tissue side does not, because the labels resolve
organs by terminology rather than measurement. We publish the benchmark, the
audit of our own earlier incorrect figures, and the permutation protocol that
caught a false positive we were about to report, so that the next attempt at
this hypothesis can be checked.

## Data and code availability

All code, data and result JSON: [github.com/ADjayantan/EpiADR-Net](https://github.com/ADjayantan/EpiADR-Net).
Reproduce with `make results`. SIDER 4.1 under CC BY-NC-SA 4.0; GTEx V8 under the
GTEx Data Use Agreement. Sources, URLs, download dates and mapping rules are
recorded in `DATA_PROVENANCE.md`.
