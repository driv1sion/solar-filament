# Solar Filament Segmentation Documentation Suite (v28 SOTA)
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
* **Model Selection Rationale:** Why we selected `timm-efficientnet-b3`, `resnet50+scSE`, and `tu-convnext_tiny+scSE`; why pure Vision Transformers (ViT/Swin/DeiT) were rejected due to $O(N^2)$ memory explosion on $2048 \times 2048$ resolution; why SegFormer, DeepLabV3+, and Mask R-CNN failed.
* **Milestone Progression (v1 to v28):** Step-by-step chronology across 28 iterations with tagged notebook links ([`solar-filament-v4.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v4.ipynb), [`solar-filament-v22.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/results/solar-filament-v22.ipynb), [`solar-filament-v28.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v28.ipynb)).
* **v23 to v28 Innovations:** $1024 \times 1024$ native crops with Focal Loss, ResNet-50 Bottleneck capacity upgrade with linear warmup, directional longitudinal tube dilation along the PIL, 5-Fold out-of-fold LightGBM instance reranker, and robust quantile normalization.
* **Metric Trade-Off Dynamics:** Threshold ($\tau$) trade-off between Recall/RQ and Precision/SQ; grouping radius ($r$) trade-off between bridging and over-merging; minimum area filter trade-offs.
* **Official 12-Variant Ablation Table:** Full quantitative metrics table matching LaTeX and CSV records (`results/ablation_study.csv`).

---

### [Part 3: Final Architecture, Engineering Design & Pipeline Components](file:///Users/sadhumitha/Documents/projects/solar-filament/docs/03_final_architecture_and_components.md)
* **End-to-End Visual Architecture:** Clean, modular flowchart detailing data flow from raw observation to competition submission.
* **Solar Ingestion & Pre-Processing:** Global Robust Quantile Normalization ($P_2-P_{98}$ dynamic range stretch) standardizing the 6 GONG stations, synthetic 3-channel replication, and soft consensus target generation.
* **Curvilinear-Guided Training Sampling:** $70\%$ positive filament-centered sampling $+ 30\%$ unconstrained background sampling on native 1:1 scale $1024 \times 1024$ crops with $D_4$ symmetry augmentations.
* **Curvilinear Compound Boundary Loss:** Continuous Soft-Dice ($0.40$), Lov\'asz-Hinge submodular Jaccard surrogate ($0.40$), and Focal Loss ($0.20$, $\alpha=0.75, \gamma=2.0$) with RetinaNet prior bias init ($b = -2.19$).
* **Pure CNN Tri-Backbone Diversity Ensemble:** 79.1M parameters (EfficientNet-B3 12.2M, ResNet-50-scSE 33.8M, ConvNeXt-Tiny-scSE 33.1M) with logit-space simplex blending.
* **12-View Zoom TTA Pyramid:** Scale magnification $[1.0, 1.125, 1.25] \times 4\text{ flips}$ (36 total passes).
* **Physics-Informed Post-Processing Engine:** Directional longitudinal tube dilation ($\theta \pm 15^\circ$, $21\text{px} \times 5\text{px}$ kernel), TAC-SR v2 extremity recovery ($P_{25} \le 2.0\text{px}$), solar-physics morphometric pruning, and 3-pixel tube Collinear Endpoint Gap Bridging (CGB).
* **Supervised 2nd-Stage Reranker & Density Prior:** 5-fold OOF LightGBM classifier on 9 morphometric features ($P(\text{TP}) \ge 0.38$) with physical density calibration ($\approx 5.35\text{--}5.50$ fil/img = 990 test instances) and single-frame noise burst guardrail ($\le 10$ fil/img).

---

### [Part 4: Model Limitations, Failure Modes & Future Research Directions](file:///Users/sadhumitha/Documents/projects/solar-filament/docs/04_limitations_and_future_directions.md)
* **Metric Bottlenecks:** Analysis of why Recognition Quality ($\text{RQ} = 0.6368$) lags behind Segmentation Quality ($\text{SQ} = 0.6738$).
* **Observatory Disparity:** Environmental breakdown of why Cerro Tololo (`Ch`: $0.3870\text{ PQ}$) suffers from optical gain degradation compared to Learmonth (`Lh`: $0.4413\text{ PQ}$) and Big Bear (`Bh`: $0.4379\text{ PQ}$).
* **Persistent Failure Modes:** Active region over-merging, thin barb erosion on thick bodies, solar limb darkening distortion, and atmospheric seeing turbulence.
* **Forensic Analysis of the Val (0.4346) vs. Public Leaderboard (0.37) Delta:** Detailed decomposition showing how the $\text{IoU} > 0.50$ cliff effect, test ground-truth consensus binarization strictness, and observatory distribution shifts account for the score difference.
* **Future Research Roadmap:** Linear-complexity State-Space Models (Vision Mamba / VMamba / SegMamba) for native $2048 \times 2048$ processing; multi-modal physics fusion with SDO/HMI line-of-sight magnetograms ($B_{\text{LOS}}$); curvilinear query-based Panoptic Transformers (Mask2Former); and spatio-temporal video modeling.

---

## Quick Reference Summary (v28 SOTA)

```
=====================================================================================================
                                 SOTA v28 CONFIGURATION CHEAT SHEET
=====================================================================================================
Pipeline Version:       v28 SOTA (Final Submission Pipeline)
Super-Ensemble Params:  79,127,884 parameters across 3 complementary backbones
Backbone 1:             timm-efficientnet-b3      (Weight: 0.30 | Receptive field: 1024x1024 native)
Backbone 2:             resnet50 + scSE           (Weight: 0.25 | Bottleneck blocks, 33.8M params)
Backbone 3:             tu-convnext_tiny + scSE   (Weight: 0.45 | 7x7 depthwise macro-continuity)
Ensemble Fusion:        Logit-Space Linear Blend: 0.30*L1 + 0.25*L2 + 0.45*L3 (5-fold simplex optimized)
Compound Loss:          0.40 Soft-Dice + 0.40 Lovasz-Hinge + 0.20 Focal Loss (alpha=0.75, gamma=2.0)
Model Checkpointing:    True Panoptic Quality (PQ) evaluated on 67 full-res observations in RAM (6.7s cache)
Peak Validation PQ:     0.4346 (SQ: 0.6738, RQ: 0.6368, Mean Dice: 0.6784)
Public Leaderboard:     0.37 (Official verified score)
Test-Time Augmentation: 12-View Zoom Pyramid: Scales [1.0, 1.125, 1.25] x 4 Orthogonal Flips (36 passes)
Preprocessing:          Global Robust Quantile Normalization (P2 - P98 dynamic range stretch)
Base Threshold (tau*):  0.53 (Optimized via Bayesian Optuna search)
Grouping Kernel:        Directional Longitudinal Tube Dilation (theta +- 15 deg, 21x5px elliptical)
Min Area Threshold:     380px (Normalized across stations via quantile stretch)
TAC-SR Extremity Gate:  p25(dist) <= 2.0px | Gate: locked at 0.30 (Ch) / 0.34 (Bh, Lh) / 0.35 (Mh, Th, Uh)
Physics Pruning Rule:   Prune if (Solidity > 0.70 and AR < 1.70 and p90_prob < 0.58)
Tube CGB Bridging:      Max Gap: 35px | Collinear: >= 0.70 | Line Integral >= 0.11
2nd-Stage Reranker:     5-Fold OOF-Trained LightGBM (9 features, acceptance threshold P(TP) >= 0.38)
Physical Density Prior: GONG solar cycle consensus (5.35 - 5.50 fil/img = 990 filaments across 180 frames)
Burst Frame Guardrail:  Single-frame cap <= 10 fil/img (prunes granulation noise bursts on turbulent frames)
=====================================================================================================
```
