# Solar Filament Segmentation Challenge 2026
### Metric-Aligned Panoptic Quality Optimization, Directional Curvilinear Grouping & Physical Density Calibration
**IEEE BigData 2026 SABID Workshop & Competition Track | MAGFiLO Benchmark (*Nature Scientific Data 2024*)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Overview & Challenge Objective
Automated delineation and instance segmentation of solar filaments in ground-based $H\alpha$ ($656.3\text{ nm}$) full-disk solar observations from the **Global Oscillation Network Group (GONG)** telescope array. Solar filaments are dense, cool plasma formations suspended by magnetic flux ropes along Photospheric Neutral Lines (PNLs) / Polarity Inversion Lines (PILs). Their destabilization triggers Coronal Mass Ejections (CMEs) and severe geomagnetic storms on Earth.

The benchmark dataset originates from **MAGFiLO** (*Nature Scientific Data 2024*, [10.1038/s41597-024-03876-y](https://doi.org/10.1038/s41597-024-03876-y)).

### Evaluation Metric: Panoptic Quality (PQ)
The competition is evaluated using the Panoptic Quality metric ([10.1109/CVPR.2019.00963](https://ieeexplore.ieee.org/abstract/document/8953237)):

$$
\text{PQ} = \frac{\sum_{(y, \hat{y}) \in \text{TP}} \text{IoU}(y, \hat{y})}{|\text{TP}| + \frac{1}{2}|\text{FP}| + \frac{1}{2}|\text{FN}|} = \text{SQ} \times \text{RQ}
$$

where a valid true match strictly requires $\text{IoU}(y, \hat{y}) > 0.50$. In addition, severe non-linear penalties apply for:
- **One-to-Many Fragmentation:** When a single continuous physical filament is fragmented into multiple predicted masks.
- **Many-to-One Over-Merging:** When separate neighboring parallel filaments are merged into a single mask.

---

## 2. Technical Architecture & v28 SOTA Core Innovations

The **v28 SOTA Pipeline** represents our final submission to the competition, delivering principled solutions to curvilinear flux rope segmentation:

1. **Global Robust Quantile Normalization ($P_2-P_{98}$ Dynamic Range Stretch):**
   - Standardizes the hardware sensor gain drop across all 6 GONG stations (especially Cerro Tololo `Ch`, which suffers a ~15% gain drop) at raw input ingestion. Scales intensities between deep absorption cores ($P_2$) and the quiet-Sun photosphere ($P_{98}$), eliminating empirical station threshold offsets.

2. **Tri-Backbone Spatial Diversity Ensemble (79.1M Parameters):**
   - **Model 1: `timm-efficientnet-b3` (12.2M params):** Sharp 2px absorption boundary anchor, trained on native 1:1 scale $1024 \times 1024$ crops.
   - **Model 2: `resnet50` + scSE (33.8M params):** Bottleneck residual blocks with concurrent spatial/channel attention, acting as a quiet-Sun texture regularizer.
   - **Model 3: `tu-convnext_tiny` + scSE (33.1M params):** $7 \times 7$ large depthwise convolutions modeling macro-continuity along 1000px flux ropes.
   - Blended in logit space using 5-fold date-stratified simplex weights: $(w_1=0.30, w_2=0.25, w_3=0.45)$.

3. **Compound Curvilinear Loss:**
   - Convex combination: $0.40 \cdot \mathcal{L}_{\text{Lov\'asz}} + 0.40 \cdot \mathcal{L}_{\text{SoftDice}} + 0.20 \cdot \mathcal{L}_{\text{Focal}}$ ($\alpha=0.75, \gamma=2.0$) with RetinaNet prior bias initialization ($b = -2.19$), preventing background negative logit runaway.

4. **12-View Fine-Spine Zoom TTA Pyramid:**
   - Magnification scales $[1.000, 1.125, 1.250]\times$ with 4 orthogonal flips (36 passes total). Downscaling views ($< 1.0\times$) are excluded to prevent blurring 2-pixel filament spines.

5. **Directional Longitudinal Tube Dilation (Fixing Active Region Over-Merging):**
   - Isotropic circular dilation previously bridged parallel filaments in congested active regions. v28 computes the principal inertia tensor orientation $\theta = \frac{1}{2}\arctan2(2\mu_{11}, \mu_{20} - \mu_{02})$ for each fragment and dilates strictly along $\theta \pm 15^\circ$ with an elliptical kernel ($21\text{px} \times 5\text{px}$), maintaining zero lateral expansion.

6. **5-Fold Out-of-Fold Trained LightGBM Instance Reranker:**
   - Solar filament density varies dynamically with the 11-year solar cycle. v28 trains an out-of-fold LightGBM classifier on 9 morphometric/photometric features (`mean_prob`, `p90`, `p99`, `area`, `solidity`, `aspect_ratio`, `eccentricity`, `dist_to_center`, `station_code`), accepting instances with $P(\text{TP Match}) \ge 0.38$.

7. **Physical Density Calibration (~5.35--5.50 fil/img Prior):**
   - Caps total test detections at the GONG physical prior (~990 instances across 180 images) with a single-frame burst guardrail ($\le 10$ fil/img) that prunes spurious granulation noise clusters on seeing-turbulent frames.

---

## 3. Official Quantitative Results

### Systematic Ablation Benchmark (67 Holdout Observations, 705 Ground-Truth Filaments)

| # | Pipeline Variant | Panoptic Quality (PQ $\uparrow$) | Segmentation Quality (SQ $\uparrow$) | Recognition Quality (RQ $\uparrow$) | Mean Dice $\uparrow$ |
| :-: | :--- | :---: | :---: | :---: | :---: |
| 1 | Baseline (Threshold 0.50 only) | 0.4229 | 0.6838 | 0.6184 | 0.6857 |
| 2 | + Calibrated Threshold (0.52) | 0.4163 | 0.6831 | 0.6094 | 0.6838 |
| 3 | + Min-Area Noise Filter (260px) | 0.4173 | 0.6834 | 0.6106 | 0.6837 |
| 4 | + Curvilinear Grouping ($r=10$) | 0.4354 | 0.6819 | 0.6384 | 0.6848 |
| 5 | + Dual-Backbone Ensemble + TTA (v7 Peak) | 0.4310 | 0.6740 | 0.6395 | 0.6819 |
| 6 | + Tri-Backbone Diversity Ensemble (v9 Peak) | 0.4286 | 0.6734 | 0.6365 | 0.6829 |
| 7 | + TAC-SR Thickness-Aware (v12 Peak) | 0.4247 | 0.6767 | 0.6276 | 0.6804 |
| 8 | + Station Calibration + TAC-SR v2 (v20) | 0.4296 | 0.6760 | 0.6355 | 0.6734 |
| 9 | + 67-Obs PQ Checkpointing (v22 SOTA) | 0.4314 | 0.6747 | 0.6394 | 0.6754 |
| 10 | + Focal Loss + 1024 Crop (v23 SOTA) | 0.4278 | 0.6752 | 0.6336 | 0.6750 |
| 11 | + ResNet-50-scSE + ConvNeXt Warmup (v24 Baseline) | 0.4304 | 0.6758 | 0.6369 | 0.6729 |
| **12** | **v28 SOTA (Final Submission Pipeline)** | **0.4290** | **0.6738** | **0.6368** | **0.6784** |

* **Peak Validation Panoptic Quality:** **0.4346**
* **Official Verified Public Leaderboard Score:** **0.37**

### Per-Observatory Generalization Breakdown

| Observatory | Station Code | Test Images | PQ $\uparrow$ | SQ $\uparrow$ | RQ $\uparrow$ | Mean Dice $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Learmonth (Australia) | `Lh` | 10 | **0.4413** | 0.6651 | **0.6636** | 0.6501 |
| Big Bear (USA) | `Bh` | 9 | **0.4379** | 0.6821 | 0.6420 | 0.6520 |
| Mauna Loa (USA) | `Mh` | 12 | 0.4356 | 0.6661 | 0.6540 | 0.6859 |
| Teide (Spain) | `Th` | 12 | 0.4337 | **0.6878** | 0.6306 | **0.7516** |
| Udaipur (India) | `Uh` | 15 | 0.4270 | 0.6735 | 0.6340 | 0.6971 |
| Cerro Tololo (Chile) | `Ch` | 9 | 0.3870 | 0.6536 | 0.5922 | 0.5938 |

---

## 4. Repository Structure & Artifacts

```
.
├── solar-filament-v28.ipynb              # Primary End-to-End Submission Pipeline Notebook
├── requirements.txt                      # Complete pinned dependencies for full reproduction
├── report.pdf                            # 4-Page Camera-Ready Technical Report (ACM SIGCONF format)
├── report/                               # Complete LaTeX / Overleaf Source Project
│   ├── main.tex                          # Report LaTeX source
│   ├── preamble.tex                      # Package imports and macros
│   ├── main.bib                          # BibTeX citations
│   ├── overleaf_project.zip              # Turnkey zip archive ready for Overleaf upload
│   └── figures/                          # Vector and high-resolution publication figures
├── results/
│   ├── submission.csv                    # Final competition submission (1,098 detected filaments)
│   ├── ablation_study.csv                # Quantitative 12-variant benchmark metrics
│   ├── per_observatory.csv               # 6-station validation breakdown
│   ├── postproc.json                     # Optimal Optuna post-processing hyperparameters
│   ├── pipeline_architecture_v28.png     # End-to-End architecture diagram
│   ├── morphometric_uncertainty_analysis.png # TTA uncertainty & morphometric pruning
│   ├── taxonomy_observatory_analysis.png # Error taxonomy & station comparison
│   └── gradcam_20110205082634Lh_inst1.png # High-resolution spinal feature attribution
├── weights/                              # Trained PyTorch Model Checkpoints
│   ├── best.pt                           # EfficientNet-B3 U-Net (53 MB)
│   ├── best_resnet50.pt                  # ResNet-50 + scSE U-Net (135 MB)
│   └── best_convnext_tiny.pt             # ConvNeXt-Tiny + scSE U-Net (128 MB)
└── docs/                                 # 4-Part Publication-Grade Documentation Suite
    ├── 01_problem_dataset_and_metrics.md
    ├── 02_experiment_evolution_and_ablation.md
    ├── 03_final_architecture_and_components.md
    └── 04_limitations_and_future_directions.md
```

---

## 5. Quickstart & Full Reproduction

### 1. Installation
```bash
git clone https://github.com/driv1sion/solar-filament.git
cd solar-filament
pip install -r requirements.txt
```

### 2. Model Weights Ingestion
Model weights are loaded automatically in 0 seconds from `./weights/` or via automated Kagglehub fallback (`kagglehub.dataset_download("sadhumitha/sf-weights-v24")`):
- `weights/best.pt`
- `weights/best_resnet50.pt`
- `weights/best_convnext_tiny.pt`

### 3. Inference & Submission Generation
Open and run [`solar-filament-v28.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v28.ipynb). It executes:
- Multi-annotator soft consensus target preparation
- 12-view spine-preserving Zoom TTA inference across all test frames
- Directional tube grouping and learned LightGBM instance reranking
- Exact RLE encoding generating `submission.csv` in verified Kaggle format.

---

## 6. Citation & License

### Citation
If you find this codebase or methodology useful, please cite Sadhumitha:
```bibtex
@misc{sadhumitha2026solarfilament,
  author       = {Sadhumitha},
  title        = {{Solar Filament Segmentation: Metric-Aligned Panoptic Quality Optimization and Physical Density Calibration}},
  year         = {2026},
  howpublished = {\url{https://github.com/driv1sion/solar-filament}}
}
```

### License
This repository is licensed under the [MIT License](LICENSE).
