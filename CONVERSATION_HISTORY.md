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

## Current Status
- **Training**: Started `train_final_p2.py` for the P2 Improved model.
- **Automation**: Created `run_and_sync.bat` to automatically train, evaluate, and push results to Git.

## How to Continue (From Home)
1.  **Pull/Clone** the repository.
2.  Check `runs/detect/YOLOv12-P2-Improved` for results.
3.  The final comparison table will be in `model_comparison_final.csv`.
4.  If training was interrupted, run `run_and_sync.bat` to resume.
