# EpiADR-Net — Viva Defence

Every number below is in `results/*.json` and regenerable with `make results`.
If you cannot point at the script that produced a number, do not say the number.

---

## The 30-second answer

> I built a benchmark for organ-conditioned adverse drug reaction prediction:
> given a drug's structure and a target organ, predict that organ's
> characteristic ADR. 1 465 molecules, 5 organs, SIDER labels, molecule-level
> scaffold cross-validation with leakage assertions.
>
> Two findings. Drug–target activity helps — adding PubChem binding data over a
> 94-gene pharmacological panel takes macro-AUROC from 0.620 with a plain organ
> one-hot to 0.655, which is my best result. Tissue transcriptomics does not.
> Multiplying that binding by each target's per-organ expression adds nothing: a
> 12-fold permutation test of the organ→expression map gives p = 0.385, and four
> of the twelve *wrong* assignments scored higher than the correct one.
>
> I traced the cause to the labels: organ assignment comes from a MedDRA
> terminology mapping, not from tissue measurement. The binding vector describes
> the drug, which the labels encode; the expression factor describes the organ,
> which they encode only as an ontological class.
>
> I found all of this by auditing my own earlier pipeline, which reported 0.742
> and was wrong.

---

## Q1. What is the headline number?

**Macro-AUROC 0.6547 ± 0.011** — model M8: Morgan fingerprint (ECFP4, 2048 bit)
+ PubChem drug–target binding over a 94-gene panel + organ one-hot + mechanism
coupling, XGBoost, molecule-level Bemis-Murcko scaffold 5-fold CV.
`results/mechanism.json`.

The progression, all on the same splits:

| | macro-AUROC |
|:--|--:|
| Morgan only | 0.5888 ± 0.013 |
| + organ one-hot | 0.6197 ± 0.020 |
| + GTEx PCA profile | 0.6243 ± 0.012 |
| + drug–target binding | 0.6464 ± 0.018 |
| **+ mechanism coupling (M8)** | **0.6547 ± 0.011** |

The fingerprint-only per-organ reference is 0.6364 ± 0.016 (`results/baselines.json`).

## Q2. Why per-organ AUROC and not pooled AUROC?

Because pooled AUROC is inflated by base rates and I can show it numerically.

A predictor that takes **no molecular input at all** and outputs each organ's
training base rate scores **pooled AUROC 0.656**. Its macro-AUROC is exactly
0.500 — it ranks no drug correctly. Pooled AUROC on this task rewards separating
Lung (7.8 % positive) from Brain (38.0 %), which requires no chemistry.

Within one organ the base rate is constant, so per-organ AUROC measures only what
the task is about: ranking drugs by toxicity.

## Q3. How do you prevent data leakage?

Three mechanisms, all machine-checked:

1. **Molecule-level, not row-level.** Each molecule has five rows (one per
   organ). Splitting by row would put the same molecule in train and test.
   `epiadr/splits.py` assigns whole molecules.
2. **Scaffold-level.** Bemis-Murcko scaffolds are assigned whole to a fold, so
   structurally related molecules cannot straddle the split.
3. **Asserted, not assumed.** `verify_no_leakage()` raises unless shared
   molecules and shared scaffolds are both zero. It runs on every fold inside the
   experiment scripts and in the test suite (`tests/test_honest_pipeline.py`).

## Q4. Does tissue conditioning work? *(the question your examiner wants)*

**No — and I tested it twice, the second time properly.**

First with the GTEx profile, where a single derangement was enough:

| features | macro-AUROC |
|:--|--:|
| molecule only | 0.5888 ± 0.013 |
| + organ one-hot (5 dim) | 0.6197 ± 0.020 |
| + real GTEx profile (128 dim) | 0.6243 ± 0.012 |
| + **deranged** GTEx profile | 0.6244 ± 0.013 |

A derangement gives every organ another organ's transcriptome — the marginal
distribution is identical, but the vector no longer says which organ the row is.
Macro-AUROC changes by **−0.0001**. The model was using the 128-dimensional
profile as a lookup key for organ identity, nothing more. The one-hot row
confirms it from the other side: 5 dimensions buy the same 0.62 that 128 do.

Then with explicit mechanistic coupling and real drug–target data — see Q4b.

## Q4b. You added drug–target data. What happened?

**The binding half helped. The tissue half did not.**

Adding PubChem binding over the 94-gene panel: 0.6197 → 0.6464 (+0.027), and
0.6547 with the coupling on top (+0.035). That is the first feature in the
project to beat the one-hot by more than fold noise.

But the coupling specifically — `binds(drug, gene) × expression(gene, organ)` —
does not survive a proper control. A single derangement said +0.0088, which
looked like a real signal. So I built the null properly: refit M8 under **12
independent derangements**.

| | macro-AUROC |
|:--|--:|
| real organ→expression map | 0.6547 |
| null, 12 derangements | 0.6518 ± 0.0050 (range 0.6410–0.6588) |
| derangements ≥ real | **4 / 12** |
| empirical *p* | **0.385** |

Not significant. The mechanism block is acting as extra drug-level descriptors —
a nonlinear re-encoding of the binding vector — not as tissue biology.

**If you say one thing from this slide, say this:** the single-derangement
ablation nearly made me report a false positive. With a fold s.d. of 0.02 and a
null spread of 0.005, one control draw cannot separate +0.008 from noise.

## Q4c. Why did your hand-crafted coupling feature *hurt* in one version?

Model M5 replaced the raw binding vector with the product and scored 0.6191,
below M4's 0.6464 which had the same information uncoupled. Two reasons:

1. The product zeroes out binding wherever expression is low, so the raw "this
   drug hits CYP2C9" signal is destroyed.
2. A gradient-boosted tree can already form the interaction from separate
   columns — the split `organ == Liver AND binds ABCB11` is two levels of a
   tree. Hand-coding the product removed the parts and added nothing.

The coupling only helps when **added alongside** the binding vector (M8), and
then by +0.008 — which the permutation test then shows is not organ-specific.

This is a general lesson: hand-engineering an interaction a model can already
learn usually loses information.

## Q5. Why doesn't tissue conditioning work? Is your model too small?

No — it is the labels, and capacity cannot fix it.

`DATA_PROVENANCE.md` records the rule: MedDRA **System Organ Class** terms were
mapped to organs (hepatic terms → Liver, arrhythmia terms → Heart, …). The organ
label is therefore a fact about MedDRA's ontology, not a measurement of tissue
biology. Expression data cannot recover a signal that was never put in.

Information-theoretically: five organs is about 2.3 bits. A one-hot carries all
of them. The other 123 dimensions describe biology the label does not depend on.

This also explains the split in Q4b. The binding vector describes the **drug**,
which the labels encode — SIDER records that a compound causes hepatotoxicity,
and that is a fact about the compound. The expression factor describes the
**organ**, which the labels encode only as an ontological class. The product
inherits signal from one factor and noise from the other.

## Q6. Then what would make the hypothesis testable?

Two routes were available. I tested one of them.

1. **Mechanistic coupling** — drug–target binding weighted by per-tissue
   expression. **Tested (Q4b). Does not rescue it**, p = 0.385.
2. **Labels resolved by biology, not terminology** — organ-specific signals from
   FAERS/OFFSIDES case narratives, or *in vitro* organ-specific toxicity assays.
   **Untested.** This is what is left.

Until the labels change, every organ-side feature is bounded above by what a
five-dimensional one-hot already encodes.

## Q7. Your README says "zero-shot". Did you test it?

Yes — three settings, and the strict one is near chance.

| setting | best macro-AUROC |
|:--|--:|
| S1 unseen drug, seen organ | 0.6547 (M8) |
| S2 seen drug, unseen organ | 0.6997 (binding + one-hot) |
| S3 **unseen drug + unseen organ** | **0.5786 (M8)** |

The trap is S2, which is often reported as "zero-shot across tissues". It is the
*easiest* setting: the molecule is not held out, so the model has seen that
drug's toxicity in four other organs, and organ ADRs correlate (mean pairwise
*r* = 0.196). Proof that nothing organ-specific transfers: in S2 the **best**
model is molecule-only with no organ input at all (0.6975).

S3 holds out both. **0.579** with the best feature set — still near chance. That
is the honest zero-shot number.

## Q8. Why is a 116 M-parameter graph transformer not your headline model?

Because it has not been benchmarked on the corrected task, so I am not claiming
anything for it. The fingerprint model is the reference because it is the
strongest result that has actually been measured. Any GNN has to beat
**0.6364 ± 0.016** on the same splits to justify its parameters — that is the
bar, and it is written into the README.

On 1 465 molecules, a 116 M-parameter model is about 79 000 parameters per
molecule. That ratio is itself the argument for starting from a strong, cheap
baseline.

## Q9. Earlier versions of this repo reported 0.742 and 0.968. What happened?

I audited my own pipeline and found four blocking faults. They are documented in
`AUDIT.md` and reproducible with `python scripts/audit_pipeline.py`:

1. **`data/adr_records.csv` was never read.** `load_labeled_records()` and
   `GTExTissueLoader` were defined with full validation and called by nothing.
   Training fell back to a hard-coded list.
2. **The tissue vectors were `np.random.dirichlet` noise**, 1024-dimensional,
   while the documented GTEx profiles are 128-dimensional and sat unused.
3. **The target did not depend on the organ.** `_build_target_vector()` never
   received the organ argument, so all organs of a molecule got an identical
   label. In the real CSV, 0 of 1 465 molecules had labels varying across organs.
4. **The benchmark trained on five molecules** — paracetamol, aspirin, ibuprofen,
   diclofenac, naproxen — duplicated into 400 rows. Five-fold CV held out one
   molecule per fold.

The `± 0.031` was also not emitted by any script in the repository.

**Say this plainly.** Finding it, documenting it, and publishing the corrected
numbers is a stronger result than the number it replaces. An examiner who opens
`data_loader.py` will find the fault in two minutes; the only question is whether
you found it first.

## Q10. So what is the contribution?

1. A reproducible, leakage-audited benchmark where every reported number is
   emitted by a script into `results/*.json`.
2. The demonstration that pooled AUROC is invalid here (0.656 with no molecular
   input), so future work on this task reports per-organ.
3. **A real improvement**: drug–target activity over a pre-specified
   pharmacological panel, +0.035 macro-AUROC over an organ one-hot.
4. **The permutation test as a required control** for any tissue-conditioning
   claim — a single derangement nearly produced a false positive here. Without
   it, "the model uses biology" and "the model knows which organ this is" are
   indistinguishable.
5. A diagnosed negative result on the tissue half, with the one untested route
   named.

---

## Numbers to have memorised

| | |
|:--|:--|
| molecules / rows / organs | 1 465 / 7 325 / 5 |
| **best macro-AUROC (M8)** | **0.6547 ± 0.011** |
| base-rate predictor | macro 0.500, pooled **0.656** |
| Morgan only → + one-hot | 0.5888 → 0.6197 |
| + drug–target binding | 0.6464 (**+0.027** over one-hot) |
| + mechanism coupling | 0.6547 (**+0.035** over one-hot) |
| one-hot vs real GTEx PCA | 0.6197 → 0.6243 (+0.005) |
| real vs deranged GTEx PCA | **−0.0001** |
| **permutation test, 12 derangements** | **p = 0.385, 4/12 beat real** |
| S3 true zero-shot | **0.5786** |
| panel / binding coverage | 94 genes, 6 582 hits, 58.6 % of molecules |
| positives: Lung / Brain | 7.8 % / 38.0 % |
| cross-organ ADR correlation | mean *r* = 0.196 |
| reference ECE | 0.111 (not well calibrated — say so) |

## Things not to say

- Do **not** call it a foundation model. 1 465 molecules.
- Do **not** quote 0.742 or 0.968. Withdrawn.
- Do **not** say the mechanism coupling "works". It gains +0.003 over the
  permutation null at p = 0.385. The *binding* works; the *coupling* does not.
- Do **not** say "state of the art". No external leaderboard comparison was run.
- Do **not** say the probabilities are clinically usable. ECE 0.111.
- If asked something not measured, say: *"I didn't measure that."*
