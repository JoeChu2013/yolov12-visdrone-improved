# Conversation & Development History

## Project Goal
Optimization of YOLOv12 for VisDrone (UAV Aerial Photography) dataset.
Target: Maximize mAP@0.5 and small object detection performance.

## Session Summary (Date: 2026-01-30)

### 1. Initial Analysis & Ablation Study
We conducted a 6-group ablation study on YOLOv12-turbo-n (Nano scale):
1.  **Baseline (CIoU)**: mAP 28.23%
2.  **Baseline + SIoU**: mAP 28.64% (+0.41%)
3.  **Baseline + CA (Coordinate Attention)**: mAP 29.07% (+0.84%)
4.  **Baseline + BiFPN**: mAP 30.20% (+1.97%)
5.  **Baseline + BiFPN + CA**: mAP 30.01% (+1.78%)
6.  **Improved (BiFPN + CA + SIoU)**: mAP 30.52% (+2.29%)

**Conclusion**: The combination of weighted feature fusion (BiFPN), attention mechanism (CA), and angle-aware loss (SIoU) yielded significant improvements.

### 2. SOTA Comparison
We compared our improved model against YOLOv8n, v9t, v10n, and v11n.
- **Result**: While our Improved model (30.55%) beat the baseline (28.32%), it was still slightly behind YOLOv9t (33.83%) and YOLOv11n (33.71%).
- **Analysis**: VisDrone contains extremely small objects (<8x8 pixels) which are lost in deep feature maps (P3/P4/P5). Standard YOLO models downsample too aggressively for this specific dataset.

### 3. Final Optimization: P2 Head
To surpass SOTA and maximize performance for drone imagery, we decided to implement a **P2 Detection Head**.
- **Architecture**: `ultralytics/cfg/models/v12/yolov12-p2-improved.yaml`
- **Change**: Added a high-resolution detection head (stride 4) to detect tiny objects.
- **Expectation**: This should provide the highest mAP, potentially exceeding 35-38%.

## Session Summary (Date: 2026-03-06)

### 4. P2 Head Results (SOTA Achieved!)
After adding the P2 Head, our model performance skyrocketed:
- **Baseline YOLOv12n**: 28.32% mAP
- **Improved (BiFPN+CA+SIoU)**: 30.55% mAP
- **Improved + P2 Head (Ours)**: **34.51% mAP** (+6.19% improvement)

We generated a comprehensive comparison table (`model_comparison_final.csv`) and visual charts (`final_visual_matrix.png`) showing our model outperforming all baselines (YOLOv8n-v11n).

### 5. Learning Rate Ablation Study
We investigated the impact of learning rate (LR) on convergence:
- **Experiments**: LR = 0.01, 0.005, 0.001, 0.0001
- **Findings**:
    - `0.01`: Converged fastest, achieved ~34.5% mAP.
    - `0.005`: Slightly slower but stable.
    - `0.001`: Significantly slower convergence.
    - `0.1`: Initially tested but abandoned due to divergence/poor performance (~20% mAP). Replaced with 0.005.

### 6. New Ablation Study (No SIoU, focus on P2)
We updated the ablation study to specifically isolate the contribution of the P2 Head without SIoU (using standard CIoU).
New results in `ablation_study_final_v2.csv`:
- **+ P2 (New)**: Adding *only* the P2 head boosted mAP from 28.29% (Baseline) to **32.41%**. This proves P2 is the most critical factor for small object detection.
- **+ BiFPN + CA + P2**: Combining all modules yields the best result (**34.51%**).

### 7. Tiny P2P3 Focused Model (Ongoing Experiment)
We designed a new lightweight architecture: **YOLOv12-Tiny-P2P3-Focused**.
- **Goal**: Maximize efficiency for drone imagery by removing redundant large-object detection heads.
- **Changes**:
    - **Removed P4 & P5 Heads**: No detection for large objects (not needed for VisDrone).
    - **Asymmetric BiFPN**: Kept top-down path for semantic context, but removed bottom-up path for P4/P5 to save compute.
    - **Focus**: Computation concentrated on P2 (Tiny) and P3 (Small) layers.
- **Status**: Currently training via `run_sequential_tasks.bat`.

## How to Continue (From Home)
1.  **Pull Repository**: `git pull origin main`
2.  **Check Training**:
    - Verify if `runs/detect/YOLOv12-Tiny-P2P3-Focused` has completed.
    - Check `ablation_study_final_v2.csv` for the latest P2 ablation data.
3.  **Next Steps**:
    - Analyze the trade-off between speed and accuracy for the Tiny P2P3 model.
    - Finalize the paper with the generated charts in the root directory.
