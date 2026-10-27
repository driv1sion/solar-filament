# Solar Filament Segmentation Documentation Suite (v22 SOTA)
### IEEE BigData 2026 SABID Challenge | MAGFiLO Benchmark (*Nature Scientific Data 2024*)

Welcome to the comprehensive, publication-grade documentation suite for the **Solar Filament Segmentation Challenge 2026**. This documentation is split into four modular, deep-dive technical guides designed for presentations, defense panels, and reproducible research.

---

## Documentation Modules

```
docs/
├── README.md                                # Master documentation hub and sitemap (this file)
├── 01_problem_dataset_and_metrics.md        # Part 1: Physics Domain, Dataset & Metric Formulations
├── 02_experiment_evolution_and_ablation.md  # Part 2: Model Selection, Experiment Progression & Ablations
├── 03_final_architecture_and_components.md # Part 3: End-to-End Pipeline & Engineering Components
└── 04_limitations_and_future_directions.md  # Part 4: Model Weaknesses, Failure Modes & Future Roadmap
```

---

### [Part 1: Problem Statement, Solar Physics Domain, Dataset Exploration & Evaluation Metrics](file:///Users/sadhumitha/Documents/projects/solar-filament/docs/01_problem_dataset_and_metrics.md)
* **Astrophysical Phenomenon:** What solar filaments vs. prominences are; magnetic flux ropes and Polarity Inversion Lines (PILs); hydrogen Balmer-alpha ($H\alpha$ at $656.3\text{ nm}$) absorption mechanics; Coronal Mass Ejection (CME) space weather hazards.
* **The GONG Telescope Array:** Microclimates and optical characteristics across all 6 global observatories (`Bh`, `Ch`, `Lh`, `Mh`, `Th`, `Uh`).
* **5 Deep Learning Challenges (First Principles):** Curvilinear anisotropy ($1500\text{px}$ length vs. $3\text{px}$ width), extreme class imbalance ($< 0.8\%$ foreground), inter-observatory photometric domain shift, multi-annotator epistemic label variance, and 27-day solar synodic rotation data leakage.
* **Panoptic Quality (PQ):** Mathematical factorization $\text{PQ} = \text{SQ} \times \text{RQ}$, uniqueness theorem for $\text{IoU} > 0.50$, severe non-linear penalties (one-to-many fragmentation, many-to-one over-merging, $0.49$ vs. $0.51$ knife-edge cliff), and detailed comparisons with Semantic Dice, COCO mAP, and clDice.

---

### [Part 2: Evolution of Experiments, Model Selection & Quantitative Ablation Study](file:///Users/sadhumitha/Documents/projects/solar-filament/docs/02_experiment_evolution_and_ablation.md)
* **Model Selection Rationale:** Why we selected `timm-efficientnet-b3`, `resnet34`, and `tu-convnext_tiny` with `scSE` attention; why pure Vision Transformers (ViT/Swin/DeiT) were rejected due to $O(N^2)$ memory explosion on $2048 \times 2048$ resolution; why SegFormer, DeepLabV3+, and Mask R-CNN failed.
* **Milestone Progression (v1 to v22):** Step-by-step chronology across 22 iterations with tagged notebook links ([`solar-filament-v4.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v4.ipynb), [`solar-filament-v17.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v17.ipynb), [`solar-filament-v22.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v22.ipynb), etc.).
* **Metric Trade-Off Dynamics:** Threshold ($\tau$) trade-off between Recall/RQ and Precision/SQ; grouping radius ($r$) trade-off between bridging and over-merging; minimum area filter trade-offs.
* **Forensic Autopsy of Failed Experiments:** Why Gaussian ASCE pre-processing failed in v5; why watershed instance demixing fractured flux ropes in v8; why continuous topological clDice skeleton loss caused gradient instability and noise chasing in v17; why multi-scale crop resizing blurred sharp 1:1 boundaries in v17.
* **Official 10-Variant Ablation Table:** Full quantitative metrics table matching LaTeX and CSV records.

---

### [Part 3: Final Architecture, Engineering Design & Pipeline Components](file:///Users/sadhumitha/Documents/projects/solar-filament/docs/03_final_architecture_and_components.md)
* **End-to-End Visual Architecture:** Comprehensive Mermaid diagrams detailing data flow from raw observation to competition submission.
* **Curvilinear-Guided Training Sampling:** $70\%$ positive filament-centered sampling $+ 30\%$ unconstrained background sampling on native 1:1 scale $768 \times 768$ crops.
* **Curvilinear Compound Boundary Loss:** Continuous Soft-Dice ($0.35$), Lovasz-Hinge submodular Jaccard surrogate ($0.40$), and Binary Cross-Entropy ($0.25$) with mathematical proofs.
* **Pure CNN Tri-Backbone Diversity Ensemble:** 69.7M parameters with mathematical proof of why Logit-Space linear blending outperforms post-sigmoid probability averaging.
* **12-View Zoom TTA Pyramid:** Scale magnification $[1.0, 1.125, 1.25] \times 4\text{ flips}$ (36 total passes).
* **Physics-Informed Post-Processing Engine:** Station-calibrated thresholding, curvilinear dilation grouping ($r = 10\text{px}$), TAC-SR v2 extremity recovery ($p_{25} \le 2.0\text{px}$), solar-physics morphometric pruning, and 3-pixel tube Collinear Endpoint Gap Bridging (CGB).
* **Strict Two-Pass Physical Density Calibration:** Eliminating 415 excess noise candidates (pruning from raw 1,405 down to 990 instances) to match physical ground-truth density ($5.50\text{ fil/img}$).

---

### [Part 4: Model Limitations, Failure Modes & Future Research Directions](file:///Users/sadhumitha/Documents/projects/solar-filament/docs/04_limitations_and_future_directions.md)
* **Metric Bottlenecks:** Analysis of why Recognition Quality ($\text{RQ} = 0.6455$) lags behind Segmentation Quality ($\text{SQ} = 0.6732$).
* **Observatory Disparity:** Environmental breakdown of why Cerro Tololo (`Ch`: $0.3903\text{ PQ}$) suffers from optical gain degradation compared to Learmonth (`Lh`: $0.4577\text{ PQ}$) and Udaipur (`Uh`: $0.4518\text{ PQ}$).
* **Persistent Failure Modes:** Active region over-merging, thin barb erosion on thick bodies, solar limb darkening distortion, and atmospheric seeing turbulence.
* **Forensic Analysis of the Val (0.4346) vs. Public Leaderboard (0.37) Delta:** Detailed decomposition showing how the $\text{IoU} > 0.50$ cliff effect, test ground-truth consensus binarization strictness, and observatory distribution shifts account for the score difference.
* **Future Research Roadmap:** Linear-complexity State-Space Models (Vision Mamba / VMamba / SegMamba) for native $2048 \times 2048$ processing; multi-modal physics fusion with SDO/HMI line-of-sight magnetograms ($B_{\text{LOS}}$); curvilinear query-based Panoptic Transformers (Mask2Former); and spatio-temporal video modeling.

---

## Quick Reference Summary (v22 SOTA)

```
=====================================================================================================
                                 SOTA v22 CONFIGURATION CHEAT SHEET
=====================================================================================================
Pipeline Version:       v22 SOTA (Final Verified Pipeline)
Super-Ensemble Params:  69,727,884 parameters across 3 complementary backbones
Backbone 1:             timm-efficientnet-b3   (Weight: 0.40 | Best Val PQ: 0.4147 | SQ: 0.6772, RQ: 0.6124)
Backbone 2:             resnet34               (Weight: 0.20 | Best Val PQ: 0.4183 | SQ: 0.6802, RQ: 0.6149)
Backbone 3:             tu-convnext_tiny+scSE  (Weight: 0.40 | Best Val PQ: 0.4282 | SQ: 0.6812, RQ: 0.6287)
Ensemble Fusion:        Logit-Space Linear Blend: 0.40*L1 + 0.20*L2 + 0.40*L3
Model Checkpointing:    True Panoptic Quality (PQ) evaluated on 67 full-res observations in RAM (6.7s cache)
Peak Validation PQ:     0.4346 (SQ: 0.6732, RQ: 0.6455, Mean Dice: 0.6741)
Public Leaderboard:     0.37 (Official verified score)
Test-Time Augmentation: 12-View Zoom Pyramid: Scales [1.0, 1.125, 1.25] x 4 Orthogonal Flips
Optimal Base Threshold: 0.55 (Mh, Th, Uh) | 0.51 (Seeing-degraded: Bh, Ch, Lh via delta=0.04)
Grouping Radius (r):    10px (Elliptical structuring element)
Min Area Threshold:     450px (Mh, Th, Uh) | 420px (Bh, Ch, Lh)
TAC-SR Extremity Gate:  p25(dist) <= 2.0px or median(dist) <= 2.5px | Gate: locked at 0.35
Physics Pruning Rule:   Prune if (Solidity > 0.76 and AR < 1.75 and p90_prob < 0.62)
Tube CGB Bridging:      Max Gap: 35px | Collinear: >= 0.70 | Line Integral >= 0.09 (Ch) / 0.15 (others)
Test Density Target:    Strict 5.50 filaments / image (Two-pass outlier-aware calibration)
Training Time & VRAM:   ~34 mins sequential training | Peak VRAM ~2.2 GiB (Zero OOM / Timeout risk)
=====================================================================================================
```
