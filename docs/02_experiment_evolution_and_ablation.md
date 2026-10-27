# Part 2: Evolution of Experiments, Model Selection & Quantitative Ablation Study

---

## 1. Overview & Research Philosophy

Our development from initial baseline to the final **v22 SOTA** pipeline was strictly hypothesis-driven and guided by empirical evaluation on the MAGFiLO benchmark. 

Every design decision was categorized into one of three operational phases:
1. **Domain Pre-Processing:** Preparing raw 1-channel $2048 \times 2048$ $H\alpha$ frames for neural ingestion while preserving physical absorption gradients.
2. **Deep Learning Training:** Network backbone design, multi-scale receptive field optimization, loss function formulation, and metric-aligned checkpointing.
3. **Inference & Post-Processing:** Test-Time Augmentation (TTA), logit-space ensembling, curvilinear grouping, spine recovery, solar-physics morphometric filtering, and physical density calibration.

---

## 2. Chronological Pipeline Progression (v1 to v22)

The table below documents the empirical evolution of the solar filament segmentation pipeline. 
*(Note: Versions v1–v12 were evaluated on a truncated 40-image validation cache containing ~380 filaments. Beginning with v17, evaluation was expanded to the full, un-truncated 67-observation date-stratified holdout set containing ~850 filaments across all 6 GONG stations).*

| Version | Phase | Primary Innovations & Architectural Changes | Val PQ | Val SQ | Val RQ | Matched TP | FP | Frag | Key Empirical Outcome & Notebook Reference |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | Train | Single UNet (EffNet-B3), Fixed Threshold $\tau = 0.50$ | 0.3364 | 0.6852 | 0.4909 | ~285 | 140+ | 24 | Massive false-positive noise specks and severe spinal fragmentation. Baseline benchmark. |
| **v4** | Post | Introduced 12-View Multi-Scale Zoom TTA Pyramid ($[1.0, 1.125, 1.25] \times 4\text{ flips}$) | 0.4353 | 0.6762 | 0.6430 | 337 | 110 | 18 | Scale magnification recovered faint barbs; +52 TPs. Notebook: [`solar-filament-v4.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v4.ipynb) |
| **v5** | Pre | Adaptive Solar Contrast Enhancement (ASCE) & unsharp masking | 0.3920 | 0.6510 | 0.6021 | 310 | 185 | 22 | **FAILED EXPERIMENT:** Heavy Gaussian background subtraction amplified granular noise. Notebook: [`solar-filament-v5.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v5.ipynb) |
| **v6** | Train | Dual-Backbone Super-Ensemble (`timm-efficientnet-b3` + `resnet34`) | 0.4360 | 0.6770 | 0.6424 | 350 | 102 | 14 | ResNet stabilized quiet-sun texture noise; +13 genuine filaments. Notebook: [`solar-filament-v6.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v6.ipynb) |
| **v7** | Post | Collinear Endpoint Gap Bridging (CGB) + Compactness Filter | 0.4407 | 0.6736 | 0.6542 | 350 | 97 | 11 | First time breaking 0.44 PQ; bridged neutral line fissures. Notebook: [`solar-filament-v7.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v7.ipynb) |
| **v8** | Post | Unconstrained Watershed Instance Demixing | 0.4180 | 0.6610 | 0.6324 | 340 | 145 | 19 | **FAILED EXPERIMENT:** Watershed over-segmented single undulating flux ropes. Notebook: [`solar-filament-v8.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v8.ipynb) |
| **v9** | Train | Tri-Backbone Diversity Ensemble (+ `tu-convnext_tiny` with `scSE`) | 0.4480 | 0.6733 | 0.6654 | 350 | 90 | 8 | $7 \times 7$ depthwise kernels slashed fragmentation to 8. Notebook: [`solar-filament-v9.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v9.ipynb) |
| **v11** | Post | Rebalanced Ensemble Weights: $(0.40 / 0.20 / 0.40) + \text{Zoom}$ | 0.4493 | 0.6752 | 0.6654 | 348 | 90 | 8 | Macro-continuity prioritization boosted RQ. Notebook: [`solar-filament-v11.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v11.ipynb) |
| **v12** | Post | High-Precision Frontier ($\tau = 0.56$) + $35\text{px}$ CGB | 0.4508 | 0.6748 | 0.6679 | 350 | 88 | 7 | Historic peak on truncated 40-image validation cache. Notebook: [`solar-filament-v12.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v12.ipynb) |
| **v16** | Post | Station Calibration Introduced ($\Delta_{\text{station}} = 0.04$ for Bh, Lh, Ch) | 0.4397 | 0.6748 | 0.6515 | 348 | 94 | 6 | Localized calibration proved viable on seeing-degraded sites. Notebook: [`solar-filament-v16.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v16.ipynb) |
| **v17 Audit** | Train | Full 67-Img Holdout + Continuous clDice Loss + Multi-Scale Crop Blur | 0.4208 | 0.6664 | 0.6314 | 450 | 258 | 4 | **FAILED EXPERIMENT:** clDice and multi-scale blur hurt stability! +102 FPs. Notebook: [`solar-filament-v17.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v17.ipynb) |
| **v18** | Train | Native 1:1 Scale Crops ($768\text{px}$) Restored + Curvilinear Tube CGB | 0.4283 | 0.6760 | 0.6336 | 524 | 122 | 8 | Supervised ensemble proved highly effective (+50.6% TPs on full holdout). Notebook: [`solar-filament-v18.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v18.ipynb) |
| **v19** | Train | Direct Supervised Deployment (No noisy pseudo-labels) | 0.4300 | 0.6755 | 0.6366 | 522 | 114 | 8 | Clean supervised baseline established on full holdout. Notebook: [`solar-filament-v19.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v19.ipynb) |
| **v20** | Post | Unconstrained Noise Pruning (`min_area = 420px`) + TAC-SR v2 | 0.4322 | 0.6696 | 0.6454 | 525 | 105 | 9 | Found optimal area filter; cut FPs to 105; recovered +10 borderline barbs. Notebook: [`solar-filament-v20.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v20.ipynb) |
| **v21** | Post | Station-Gated TAC-SR ($p_{25}$) + Adaptive Ch CGB ($0.09$) + Density Ceiling $6.65$ | 0.4325 | 0.6664 | 0.6489 | 532 | 90 | 7 | Station-adaptive refinement; bridged faint neutral lines on Cerro Tololo. Notebook: [`solar-filament-v21.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v21.ipynb) |
| **v22 SOTA** | Train/Post | **Full 67-Obs Metric-Aligned PQ Checkpointing + Strict 5.5 fil/img Calibration (Grid Peak)** | **0.4346** | **0.6732** | **0.6455** | **534** | **~82** | **~6** | **Definitive Peak: Checkpoints saved on true instance PQ (EffNet: 0.4147, ResNet: 0.4183, ConvNeXt: 0.4282). Public LB: 0.37.** Notebook: [`solar-filament-v22.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v22.ipynb) |

---

## 3. Why These Pretrained Models? (Model Selection Rationale)

Choosing backbone architectures for full-disk solar segmentation is fundamentally constrained by resolution, memory, and spatial inductive biases:

```
+---------------------------------------------------------------------------------------------------+
|                                  MODEL ARCHITECTURE SELECTION MATRIX                              |
+----------------------+-----------------------------+----------------------------------------------+
| Architecture         | Decision                    | Technical Justification                      |
+----------------------+-----------------------------+----------------------------------------------+
| timm-efficientnet-b3 | SELECTED (Weight: 0.40)     | Depthwise separable convolutions provide     |
|                      |                             | sharp boundary gradients with zero lateral   |
|                      |                             | dilation bloat; compound scaled depth/width. |
+----------------------+-----------------------------+----------------------------------------------+
| resnet34             | SELECTED (Weight: 0.20)     | Classic residual identity mappings provide   |
|                      |                             | strong inductive bias against high-frequency |
|                      |                             | quiet-sun granular background texture noise. |
+----------------------+-----------------------------+----------------------------------------------+
| tu-convnext_tiny     | SELECTED (Weight: 0.40)     | Large 7x7 depthwise convolutions mimic ViT   |
| + scSE Attention     |                             | receptive fields without quadratic memory;   |
|                      |                             | maintains continuous PIL macro-connectivity. |
+----------------------+-----------------------------+----------------------------------------------+
| Vision Transformers  | REJECTED                    | Quadratic self-attention complexity O(N^2)   |
| (ViT / Swin / DeiT)  |                             | makes native 2048x2048 processing explode    |
|                      |                             | GPU VRAM (>24 GB OOM on Kaggle T4/P100).     |
+----------------------+-----------------------------+----------------------------------------------+
| SegFormer (B0-B4)    | REJECTED                    | Hierarchical patch embedding downsamples 4x  |
|                      |                             | immediately, destroying 2px-wide barbs.      |
+----------------------+-----------------------------+----------------------------------------------+
| DeepLabV3+           | REJECTED                    | Atrous/dilated convolutions cause grid-like  |
| (ASPP)               |                             | checkerboard artifacts along fine spines.    |
+----------------------+-----------------------------+----------------------------------------------+
| Mask R-CNN           | REJECTED                    | Region Proposal Networks generate rectangular|
|                      |                             | bounding boxes, which are >90% empty space   |
|                      |                             | for diagonal snakes; NMS over-suppresses.    |
+----------------------+-----------------------------+----------------------------------------------+
```

### 3.1 Why ConvNeXt-Tiny + scSE Attention?
ConvNeXt (Liu et al., CVPR 2022) modernizes standard ResNets by adopting Vision Transformer design choices:
1. **$7 \times 7$ Depthwise Convolutions:** By moving to $7 \times 7$ kernels in the depthwise stage, the Effective Receptive Field is over $4\times$ larger than standard ResNet blocks. This allows ConvNeXt to perceive continuity along a $500\text{px}$ flux rope even across faint absorption gaps.
2. **Inverted Bottleneck Design:** Similar to MobileNetV2 and Transformers, ConvNeXt expands channel dimension by $4\times$ in the hidden layer, preventing information bottlenecking along narrow features.
3. **Spatial & Channel Squeeze-and-Excitation (`scSE`) Decoder Attention:** Roy et al. (IEEE TMI 2018) proved that concurrently exciting spatial features ($s\text{SE}$) and recalibrating channel dependencies ($c\text{SE}$) sharpens boundary localization in biomedical tubular structures. In our UNet decoder, `scSE` forces the network to route high-level semantic spinal features directly along fine low-level edge contours.

### 3.2 Why Not Pure Vision Transformers (ViT, Swin, DeiT)?
- Standard Vision Transformers compute self-attention across flattened spatial tokens:
  $$\text{Complexity} = O(N^2 \cdot D) \quad \text{where } N = \frac{H \cdot W}{P^2}$$
- For a $2048 \times 2048$ image with standard $16 \times 16$ patches, the sequence length is:
  $$N = \left(\frac{2048}{16}\right) \times \left(\frac{2048}{16}\right) = 128 \times 128 = 16{,}384\text{ tokens}$$
- The self-attention matrix requires computing and storing:
  $$N^2 = (16{,}384)^2 = 268{,}435{,}456\text{ attention weights per head per layer}$$
- Under FP16 precision, this exceeds **$16\text{ GB}$ of VRAM** for a single forward pass, rendering training and inference completely infeasible within Kaggle's 16 GB GPU constraints.
- While Swin Transformers restrict attention to local windows, window boundary partitions artificially slice continuous $1000\text{px}$ filaments, introducing border artifacts.

---

## 4. Fundamental Metric Trade-Offs in Solar Segmentation

In standard classification, increasing a threshold monotonically trades Precision for Recall. Under **Panoptic Quality ($\text{PQ} = \text{SQ} \times \text{RQ}$)**, hyperparameter interactions are non-linear and coupled:

```
+----------------------------------------------------------------------------------------------------+
|                                    KEY METRIC TRADEOFF DYNAMICS                                    |
+----------------------+-----------------------------+-----------------------------------------------+
| Parameter Changed    | Increasing Parameter (+)    | Decreasing Parameter (-)                      |
+----------------------+-----------------------------+-----------------------------------------------+
| Binarization         | Higher Precision; cleans FPs| Higher Recall; recovers faint filaments       |
| Threshold (tau)      | BUT erodes extremities,     | BUT dilates boundaries (drops SQ) and triggers|
|                      | dropping IoU below 0.50 (FN)| granulation false positives (crushes RQ).     |
+----------------------+-----------------------------+-----------------------------------------------+
| Curvilinear Grouping | Bridges broken flux ropes   | Prevents over-merging parallel filaments      |
| Radius (r)           | (increases TP, cuts Frag)   | BUT leaves physical neutral line gaps         |
|                      | BUT risks over-merging (FN).| disconnected, exploding fragmentation.        |
+----------------------+-----------------------------+-----------------------------------------------+
| Minimum Area Filter  | Eliminates noise specks     | Preserves small active-region filaments (TP)  |
| (min_area)           | (cuts FP denominator)       | BUT retains hundreds of chromospheric noise   |
|                      | BUT prunes small real barbs.| specks, inflating the FP penalty.             |
+----------------------+-----------------------------+-----------------------------------------------+
| TAC-SR Boundary      | Recovers borderline barbs   | Zero boundary dilation bloat (protects SQ)    |
| Gate (pos_region)    | (0.49 -> 0.51 IoU conversion| BUT loses thin extremity barbs, dropping      |
|                      | BUT risks lateral overgrowth| borderline filaments below 0.50 IoU.          |
+----------------------+-----------------------------+-----------------------------------------------+
```

### 4.1 The Threshold ($\tau$) Dilemma: Precision vs. The 0.50 IoU Cliff
- **Lowering $\tau$ ($0.50 \to 0.46$):** 
  - *Intended Benefit:* Faint filaments on seeing-degraded stations (`Ch`, `Bh`) are detected, reducing False Negatives.
  - *Unintended Penalty:* Lateral boundaries expand by 2–4 pixels. Because Segmentation Quality (SQ) averages $\text{IoU}$ across matched pairs, boundary overgrowth inflates the denominator of $\text{IoU} = \frac{|y \cap \hat{y}|}{|y \cup \hat{y}|}$, pulling SQ down from $0.675 \to 0.640$. Furthermore, quiet-sun granulation blobs cross the threshold, inflating FPs.
- **Raising $\tau$ ($0.50 \to 0.56$):**
  - *Intended Benefit:* Eliminates false positive granulation blobs, improving Recognition Quality.
  - *Unintended Penalty:* Tapering barbs are eroded. A filament with $|y| = 1000\text{px}$ loses 50 boundary pixels; its overlap drops from $510\text{px} \to 470\text{px}$. Its $\text{IoU}$ drops to $0.47 \le 0.50$, instantly triggering the **knife-edge double penalty** ($1\text{ FN} + 1\text{ FP}$).
- **The Empirical Optimum:** $\tau = 0.55$ for high-contrast stations (`Mh`, `Th`, `Uh`), and $\tau = 0.51$ for seeing-attenuated stations (`Bh`, `Ch`, `Lh`), discovered in our v22 grid search.

### 4.2 The Grouping Radius ($r$) Dilemma: Fragmentation vs. Over-Merging
- **Increasing $r$ ($r = 0 \to 15\text{px}$):**
  - *Intended Benefit:* Genuine magnetic flux ropes with small photometric gaps are bridged into a single component ID, reducing one-to-many fragmentation.
  - *Unintended Penalty:* In active solar regions (e.g., active sunspot complexes), two separate, oppositely sheared filaments frequently lie within $20\text{ pixels}$ of each other. A dilation radius of $15\text{px}$ merges them into a single instance ID. The union mask fails the $\text{IoU} > 0.50$ matching criterion for both, turning $2\text{ TPs}$ into $2\text{ FNs} + 1\text{ FP}$.
- **The Empirical Optimum:** $r = 10\text{px}$ (elliptical structuring element), paired with oriented **Collinear Endpoint Gap Bridging (CGB)** which bridges gaps along the magnetic inversion line without isotropic lateral expansion.

---

## 5. Forensic Post-Mortem of Failed Experiments

Understanding what **failed** and why is essential for technical depth:

### 5.1 Failed Experiment 1: Adaptive Solar Contrast Enhancement (ASCE) in v5
- **Notebook:** [`solar-filament-v5.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v5.ipynb)
- **The Hypothesis:** Solar images suffer from limb darkening (the optical path through the solar atmosphere is longer at the limb, causing radial intensity falloff). Applying large-scale Gaussian background subtraction ($\sigma = 45$) followed by contrast stretching should equalize limb darkening and normalize inter-telescope gain.
- **The Empirical Result:** Validation PQ collapsed from $0.4353 \to 0.3920$ ($-0.0433\text{ PQ}$), with False Positives jumping from $110 \to 185$.
- **The Forensic Reason:** Chromospheric granulation is a physical texture consisting of convective cells ($1{,}000\text{ km}$ across). Heavy unsharp masking and local contrast stretching amplified the intensity variance of dark intergranular lanes. The network could no longer differentiate between a faint, genuine active-region filament and normal convective granulation boundaries.

### 5.2 Failed Experiment 2: Unconstrained Watershed Demixing in v8
- **Notebook:** [`solar-filament-v8.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v8.ipynb)
- **The Hypothesis:** In crowded regions, connected components sometimes merge adjacent filaments. Applying morphological erosion to find local distance transform peaks followed by watershed propagation should cleanly split touching filaments.
- **The Empirical Result:** Validation PQ dropped from $0.4407 \to 0.4180$, and fragmentation instances increased from $11 \to 19$.
- **The Forensic Reason:** Solar filaments are not convex cells (like biological nuclei). They undulate, pinch, and widen along their magnetic trajectory. The distance transform of a single, continuous filament naturally exhibits multiple local maxima separated by narrow "waists". Watershed treated these natural morphological narrowings as boundaries between two different objects, catastrophically fracturing single filaments into multiple pieces.

### 5.3 Failed Experiment 3: Continuous Topological Skeleton Loss (clDice) in v17
- **Notebook:** [`solar-filament-v17.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v17.ipynb)
- **The Hypothesis:** clDice (Shit et al., CVPR 2021) computes soft-skeletonization via iterated min/max pooling layers:
  $$\text{skel}(X) = \text{ReLU}(X - \text{dilate}(\text{erode}(X)))$$
  Because filaments are curvilinear, optimizing skeleton precision and sensitivity should enforce spinal continuity and eliminate fragmentation.
- **The Empirical Result:** Validation PQ dropped from $0.4397 \to 0.4208$, and False Positives surged to an all-time high of **$258\text{ FPs}$**.
- **The Forensic Reason:**
  1. **Non-Convex Gradient Instability:** Iterating soft morphological operations 6 times creates deep, non-convex computational graphs. In regions with faint, diffuse absorption (e.g., quiet-sun granulation), gradients exploded or vanished, destabilizing the encoder weights.
  2. **Topological Equivalence Fallacy:** clDice treats all 1-pixel disconnections equally. It penalizes a 1-pixel break along an ambiguous background noise speck just as heavily as a disconnection along a primary $800\text{px}$ flux rope. The network exhausted capacity chasing noisy background artifacts at the expense of edge precision on primary filaments.

### 5.4 Failed Experiment 4: Multi-Scale Crop Resizing in v17
- **Notebook:** [`solar-filament-v17.ipynb`](file:///Users/sadhumitha/Documents/projects/solar-filament/solar-filament-v17.ipynb)
- **The Hypothesis:** Sampling crops dynamically at $[0.75, 1.0, 1.25] \times 768$ and resizing back to $768 \times 768$ during training would teach the network multi-scale invariance.
- **The Empirical Result:** Segmentation Quality (SQ) dropped from $0.676 \to 0.666$.
- **The Forensic Reason:** Solar filaments are observed by fixed focal-length telescopes at a constant astronomical distance (1 AU). The physical cross-sectional width of chromospheric flux ropes is inherently $3\text{--}15\text{ pixels}$. Downsampling a crop to $0.75\times$ via bilinear interpolation smoothed out the sharp high-frequency absorption gradient along the spine, teaching the network that blurred, low-gradient boundaries are acceptable.

---

## 6. Official Ablation Study Table

The complete quantitative ablation table across all 10 major pipeline variants (matching [`results/table_ablation.tex`](file:///Users/sadhumitha/Documents/projects/solar-filament/results/table_ablation.tex) and [`results/ablation_study.csv`](file:///Users/sadhumitha/Documents/projects/solar-filament/results/ablation_study.csv)):

```
+---------------------------------------------------------------------------------------------------+
|                                 FINAL PIPELINE ABLATION BENCHMARK                                 |
+----+-----------------------------------------------------+--------+--------+--------+-----------+
| Var| Pipeline Variant Description                        |  PQ    |  SQ    |  RQ    | Mean Dice |
+----+-----------------------------------------------------+--------+--------+--------+-----------+
|  1 | Baseline (Fixed Threshold 0.50 only)                | 0.3639 | 0.6815 | 0.5340 |  0.6853   |
|  2 | + Calibrated Threshold (0.52)                       | 0.3609 | 0.6824 | 0.5288 |  0.6833   |
|  3 | + Min-Area Noise Filter (260px)                     | 0.4103 | 0.6828 | 0.6009 |  0.6830   |
|  4 | + Curvilinear Dilation Grouping (r=10)              | 0.4275 | 0.6783 | 0.6302 |  0.6840   |
|  5 | + Dual-Backbone Ensemble & Zoom TTA (v7 Peak)       | 0.4224 | 0.6661 | 0.6342 |  0.6832   |
|  6 | + Tri-Backbone Diversity Ensemble (v9 Peak)         | 0.4209 | 0.6658 | 0.6321 |  0.6842   |
|  7 | + Thickness-Aware Spine Recovery (v12 Peak)         | 0.4253 | 0.6672 | 0.6374 |  0.6787   |
|  8 | + Full Network Station Calibration & TAC-SR v2 (v20)| 0.4296 | 0.6669 | 0.6441 |  0.6766   |
|  9 | + Station-Gated TAC-SR (p25) & Density Calib (v21)  | 0.4325 | 0.6664 | 0.6489 |  0.6805   |
| 10 | + Full 67-Obs PQ Checkpointing & 5.5 fil/img (v22)  | 0.4346 | 0.6732 | 0.6455 |  0.6741   |
+----+-----------------------------------------------------+--------+--------+--------+-----------+
```

In [Part 3: Final Architecture and Components](file:///Users/sadhumitha/Documents/projects/solar-filament/docs/03_final_architecture_and_components.md), we detail the exact mathematical formulas, loss formulations, and algorithmic implementations powering every module of our winning v22 pipeline.
