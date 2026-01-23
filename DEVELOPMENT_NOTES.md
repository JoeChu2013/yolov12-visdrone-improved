# YOLOv12-Improved Development Notes & Ablation Study

## Project Overview
This project aims to improve YOLOv12 for VisDrone (UAV aerial photography) detection tasks. 
Key improvements include:
1. **BiFPN**: Bidirectional Feature Pyramid Network for weighted multi-scale feature fusion.
2. **Coordinate Attention (CA)**: Attention mechanism to focus on small targets.
3. **SIoU Loss**: Improved loss function for better convergence and localization.

## Ablation Study Results (VisDrone Dataset)

**Environment**:
- Input Size: 640x640
- Batch Size: 16
- Optimizer: SGD (lr=0.01)
- Epochs: 100

| Model | Params (M) | FPS (GPU) | mAP@0.5 | Precision | Recall | Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (CIoU)** | 2.52 | 70.6 | 28.23% | 38.87% | 29.05% | - |
| **Baseline + SIoU** | 2.52 | 72.7 | 28.64% | 38.57% | 29.75% | +0.41% |
| **Baseline + CA** | 2.53 | 65.3 | 29.07% | 39.23% | 29.76% | +0.84% |
| **Baseline + BiFPN** | 2.92 | 62.8 | 30.20% | 42.10% | 30.12% | +1.97% |
| **Baseline + BiFPN + CA** | 2.94 | 62.5 | 30.01% | 39.85% | 30.76% | +1.78% |
| **Improved (BiFPN+CA+SIoU)** | **2.94** | **60.7** | **30.52%** | **41.25%** | **30.74%** | **+2.29%** |

## Key Files Created
- `ultralytics/cfg/models/v12/yolov12-improved.yaml`: Final improved model configuration.
- `ultralytics/cfg/models/v12/yolov12-bifpn.yaml`: BiFPN only configuration.
- `ultralytics/cfg/models/v12/yolov12-ca.yaml`: CA only configuration.
- `train_ablation.py`: Script to run all 6 ablation experiments automatically.
- `analyze_final_results.py`: Script to parse training logs and generate comparison tables.
- `ultralytics/utils/loss.py`: Modified to support dynamic IoU switching (SIoU/CIoU) via env vars.

## How to Resume/Run
1. Install dependencies: `pip install ultralytics` (plus torch, etc.)
2. Run ablation training:
   ```bash
   python train_ablation.py
   ```
3. Analyze results:
   ```bash
   python analyze_final_results.py
   ```

## Future Work
- Verify performance on larger input sizes (e.g., 1024x1024).
- Experiment with hyperparameter tuning for the Improved model.
- Deploy to Edge devices (Jetson/Raspberry Pi) to verify FPS in real-world scenarios.
