# YOLOv12 VisDrone Project Structure

This directory contains the results and source code for the "Improved YOLOv12 for UAV Detection" project.

## Directory Structure (runs/detect/)

After training, the results are stored in `runs/detect/{experiment_name}`. Key files include:

### 1. Visualization Images
These images are **highly recommended for your paper** to demonstrate performance visually.

*   **`results.png`**: (Essential) Shows the convergence curves of Loss, Precision, Recall, and mAP over epochs. Good for proving training stability.
*   **`confusion_matrix.png`**: Shows classification accuracy per category. Helps analyze which classes are easily confused (e.g., Pedestrian vs. People).
*   **`F1_curve.png`**: F1-Score curve at different confidence thresholds.
*   **`PR_curve.png`**: Precision-Recall curve. Area under this curve is mAP.
*   **`val_batch*_pred.jpg`**: (Essential) Actual detection results on validation batch. Use these to show qualitative comparison (e.g., "Our model detects small cars that baseline missed").
*   **`labels.jpg`**: Data distribution of the dataset (object sizes, locations). Good for "Dataset Analysis" section.

### 2. Data Files
*   **`results.csv`**: Raw training metrics (Loss, P, R, mAP) for every epoch.
*   **`args.yaml`**: The exact hyperparameters used for this training run.
*   **`weights/best.pt`**: The trained model weights.

## Code Files
*   **`train_ablation.py`**: Script used for the 6-group ablation study.
*   **`train_final_p2.py`**: Script for the final "P2 Head" improved model.
*   **`compare_sota.py`**: Script to compare against YOLOv8/v9/v10/v11.
*   **`ultralytics/cfg/models/v12/yolov12-p2-improved.yaml`**: The core architecture file for your improved model.

## Git Repository
This folder is a git repository. You can push it to GitHub using:
```bash
git remote add origin <your-repo-url>
git push -u origin main
```
