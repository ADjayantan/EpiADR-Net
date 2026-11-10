# EpiADR-Net — Viva Defence

Every number below is in `results/*.json` and regenerable with `make results`.
If you cannot point at the script that produced a number, do not say the number.

---

## The 30-second answer

> I built a benchmark for organ-conditioned adverse drug reaction prediction:
> given a drug's structure and a target organ, predict that organ's
> characteristic ADR. 1 465 molecules, 5 organs, SIDER labels, GTEx
> transcriptomics. The best honest result is macro-AUROC 0.636 with
> fingerprints and gradient boosting, under molecule-level scaffold
> cross-validation.
>
> The main finding is negative. Conditioning on real GTEx tissue profiles does
> not beat a five-dimensional organ one-hot, and if I shuffle the profiles so
> every organ gets the wrong transcriptome, the score moves by 0.0001. I traced
> it to the labels: organ assignment comes from a MedDRA terminology mapping, not
> from tissue measurement, so there is no transcriptomic signal to recover. I
> found this by auditing my own earlier pipeline, which reported 0.742 and was
> wrong.

---

## Q1. What is the headline number?

**Macro-AUROC 0.6364 ± 0.016** — Morgan fingerprint (ECFP4, 2048 bit) + XGBoost,
one model per organ, molecule-level Bemis-Murcko scaffold 5-fold CV.
`results/baselines.json`.

Per organ: Liver 0.667, Lung 0.662, Brain 0.632, Kidney 0.623, Heart 0.599.

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

**No, and the ablation is the evidence.**

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

## Q5. Why doesn't it work? Is your model too small?

No — it is the labels, and capacity cannot fix it.

`DATA_PROVENANCE.md` records the rule: MedDRA **System Organ Class** terms were
mapped to organs (hepatic terms → Liver, arrhythmia terms → Heart, …). The organ
label is therefore a fact about MedDRA's ontology, not a measurement of tissue
biology. Expression data cannot recover a signal that was never put in.

Information-theoretically: five organs is about 2.3 bits. A one-hot carries all
of them. The other 123 dimensions describe biology the label does not depend on.

## Q6. Then what would make the hypothesis testable?

Two routes, both named in the manuscript:

1. **Labels resolved by biology, not terminology** — organ-specific signals from
   FAERS/OFFSIDES case data, or *in vitro* organ-specific toxicity assays.
2. **Mechanistic coupling features** — drug–target binding from ChEMBL/DrugBank
   weighted by that target's per-tissue expression in GTEx:
   $\sum_t \text{affinity}(d,t)\cdot\text{expression}(t,\text{organ})$.
   This is **not** derivable from an organ identifier, so it can beat the one-hot
   control in principle.

Neither is claimed in the current results.

## Q7. Your README says "zero-shot". Did you test it?

Yes — three settings, and the strict one is near chance.

| setting | best macro-AUROC |
|:--|--:|
| S1 unseen drug, seen organ | 0.6243 |
| S2 seen drug, unseen organ | 0.6975 |
| S3 **unseen drug + unseen organ** | **0.5517** |

The trap is S2, which is often reported as "zero-shot across tissues". It is the
*easiest* setting: the molecule is not held out, so the model has seen that
drug's toxicity in four other organs, and organ ADRs correlate (mean pairwise
*r* = 0.196). Proof that nothing organ-specific transfers: in S2 the **best**
model is molecule-only with no organ input at all (0.6975).

S3 holds out both. 0.552. That is the honest zero-shot number.

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
3. The derangement ablation as a required control for any tissue-conditioning
   claim — without it, "the model uses biology" and "the model knows which organ
   this is" are indistinguishable.
4. A diagnosed negative result, with a concrete, falsifiable proposal for what
   would change it.

---

## Numbers to have memorised

| | |
|:--|:--|
| molecules / rows / organs | 1 465 / 7 325 / 5 |
| best macro-AUROC | **0.6364 ± 0.016** |
| base-rate predictor | macro 0.500, pooled **0.656** |
| one-hot vs real GTEx | 0.6197 → 0.6243 (+0.005) |
| real vs deranged GTEx | 0.6243 → 0.6244 (**−0.0001**) |
| S3 true zero-shot | **0.5517** |
| positives: Lung / Brain | 7.8 % / 38.0 % |
| cross-organ ADR correlation | mean *r* = 0.196 |
| reference-model ECE | 0.111 (not well calibrated — say so) |

## Things not to say

- Do **not** call it a foundation model. 1 465 molecules.
- Do **not** quote 0.742 or 0.968. Withdrawn.
- Do **not** say "state of the art". No external leaderboard comparison was run.
- Do **not** say the probabilities are clinically usable. ECE 0.111.
- If asked something not measured, say: *"I didn't measure that."*
