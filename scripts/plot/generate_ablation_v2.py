import pandas as pd
import torch
from ultralytics import YOLO
import time
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def get_best_metrics(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if 'metrics/mAP50(B)' in df.columns:
            best_idx = df['metrics/mAP50(B)'].idxmax()
            return df.iloc[best_idx]
        return None
    except Exception as e:
        return None

def get_model_info(weights_path, yaml_path=None):
    try:
        if weights_path and os.path.exists(weights_path):
            model = YOLO(weights_path)
        elif yaml_path:
            model = YOLO(yaml_path)
        else:
            return 0, 0
            
        n_params = sum(x.numel() for x in model.model.parameters())
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.model.to(device)
        model.model.eval()
        input_tensor = torch.randn(1, 3, 640, 640).to(device)
        
        for _ in range(10): _ = model.model(input_tensor)
        t0 = time.time()
        n_iters = 50
        for _ in range(n_iters): _ = model.model(input_tensor)
        t1 = time.time()
        
        fps = n_iters / (t1 - t0)
        return n_params, fps
    except Exception as e:
        print(f"Error analyzing model: {e}")
        return 0, 0

def main():
    # New Experiment List
    experiments = [
        {
            "name": "Baseline",
            "csv": "runs/detect/ablation_baseline_ciou/results.csv",
            "weights": "runs/detect/ablation_baseline_ciou/weights/best.pt",
            "yaml": "ultralytics/cfg/models/v12/yolov12.yaml",
            "components": ["-", "-", "-"]
        },
        {
            "name": "+ BiFPN",
            "csv": "runs/detect/ablation_bifpn_ciou/results.csv",
            "weights": "runs/detect/ablation_bifpn_ciou/weights/best.pt",
            "yaml": "ultralytics/cfg/models/v12/yolov12-bifpn.yaml",
            "components": ["✓", "-", "-"]
        },
        {
            "name": "+ CA",
            "csv": "runs/detect/ablation_ca_ciou/results.csv",
            "weights": "runs/detect/ablation_ca_ciou/weights/best.pt",
            "yaml": "ultralytics/cfg/models/v12/yolov12-ca.yaml",
            "components": ["-", "✓", "-"]
        },
        {
            "name": "+ P2 (New)", # Need to simulate/train this or estimate?
            # User asked to "Add" this data. Since we haven't trained pure P2 model,
            # we need to train it now.
            "csv": "runs/detect/ablation_p2_ciou/results.csv",
            "weights": "runs/detect/ablation_p2_ciou/weights/best.pt",
            "yaml": "ultralytics/cfg/models/v12/yolov12-p2.yaml",
            "components": ["-", "-", "✓"]
        },
        {
            "name": "+ BiFPN + CA",
            "csv": "runs/detect/ablation_bifpn_ca_ciou/results.csv",
            "weights": "runs/detect/ablation_bifpn_ca_ciou/weights/best.pt",
            "yaml": "ultralytics/cfg/models/v12/yolov12-improved.yaml",
            "components": ["✓", "✓", "-"]
        },
        {
            "name": "+ BiFPN + CA + P2 (Symmetric)",
            "csv": "runs/detect/YOLOv12-P2-Improved/results.csv",
            "weights": "runs/detect/YOLOv12-P2-Improved/weights/best.pt",
            "yaml": "ultralytics/cfg/models/v12/yolov12-p2-improved.yaml",
            "components": ["✓", "✓", "✓"]
        },
        {
            "name": "+ Asym-BiFPN + CA + P2 (Tiny)",
            "csv": "runs/detect/YOLOv12-Tiny-P2P3-Focused/results.csv",
            "weights": "runs/detect/YOLOv12-Tiny-P2P3-Focused/weights/best.pt",
            "yaml": "ultralytics/cfg/models/v12/yolov12-tiny-p2p3.yaml",
            "components": ["Asym", "✓", "✓"]
        }
    ]

    print(f"{'Model':<35} | {'BiFPN':<10} | {'CA':<6} | {'P2':<6} | {'Params (M)':<10} | {'FPS':<8} | {'mAP@0.5':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 125)

    rows = []
    
    # We need to run training for P2 if not exists
    p2_exp = experiments[3]
    if not os.path.exists(p2_exp["weights"]):
        print("\nTraining Pure P2 Model for Ablation Study...")
        model = YOLO(p2_exp["yaml"])
        model.train(
            data='ultralytics/cfg/datasets/VisDrone.yaml',
            epochs=100, # Match others
            batch=16,
            imgsz=640,
            lr0=0.01,
            optimizer='SGD',
            name="ablation_p2_ciou",
            device='0',
            project='runs/detect',
            exist_ok=True,
            pretrained=True,
            plots=False
        )

    for exp in experiments:
        metrics = get_best_metrics(exp["csv"])
        map50 = "N/A"
        p = "N/A"
        r = "N/A"
        
        if metrics is not None:
            map50 = f"{metrics.get('metrics/mAP50(B)', 0):.2%}"
            p = f"{metrics.get('metrics/precision(B)', 0):.2%}"
            r = f"{metrics.get('metrics/recall(B)', 0):.2%}"
            
        params, fps = get_model_info(exp["weights"], exp["yaml"])
        params_m = f"{params/1e6:.2f}"
        fps_str = f"{fps:.1f}"
        
        bifpn, ca, p2 = exp["components"]
        print(f"{exp['name']:<35} | {bifpn:<10} | {ca:<6} | {p2:<6} | {params_m:<10} | {fps_str:<8} | {map50:<10} | {p:<10} | {r:<10}")
        
        rows.append({
            "Model": exp["name"],
            "BiFPN": bifpn,
            "CA": ca,
            "P2": p2,
            "Params (M)": params_m,
            "FPS": fps_str,
            "mAP@0.5": map50,
            "Precision": p,
            "Recall": r
        })

    df = pd.DataFrame(rows)
    df.to_csv("ablation_study_final_v2.csv", index=False)
    print("\nSaved table to ablation_study_final_v2.csv")

if __name__ == "__main__":
    main()
