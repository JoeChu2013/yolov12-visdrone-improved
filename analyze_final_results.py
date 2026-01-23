import pandas as pd
import torch
from ultralytics import YOLO
import time
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def get_last_metrics(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        # Sort by epoch just in case, or take last
        # metrics/mAP50(B) is the key
        last_row = df.iloc[-1]
        return last_row
    except Exception as e:
        return None

def get_model_info(yaml_path, weights_path=None):
    try:
        # Load model
        if weights_path and os.path.exists(weights_path):
            model = YOLO(weights_path)
        else:
            model = YOLO(yaml_path)
            
        # Params
        n_params = sum(x.numel() for x in model.model.parameters())
        
        # FPS Measurement
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.model.to(device)
        model.model.eval()
        
        input_tensor = torch.randn(1, 3, 640, 640).to(device)
        
        # Warmup
        for _ in range(10):
            _ = model.model(input_tensor)
            
        # Measure
        t0 = time.time()
        n_iters = 100
        for _ in range(n_iters):
            _ = model.model(input_tensor)
        t1 = time.time()
        
        fps = n_iters / (t1 - t0)
        
        return n_params, fps
    except Exception as e:
        print(f"Error analyzing {yaml_path}: {e}")
        return 0, 0

def main():
    experiments = [
        {
            "name": "Baseline (CIoU)",
            "config": "ultralytics/cfg/models/v12/yolov12.yaml",
            "csv": "runs/detect/ablation_baseline_ciou/results.csv",
            "weights": "runs/detect/ablation_baseline_ciou/weights/best.pt"
        },
        {
            "name": "Baseline + BiFPN",
            "config": "ultralytics/cfg/models/v12/yolov12-bifpn.yaml",
            "csv": "runs/detect/ablation_bifpn_ciou/results.csv",
            "weights": "runs/detect/ablation_bifpn_ciou/weights/best.pt"
        },
        {
            "name": "Baseline + CA",
            "config": "ultralytics/cfg/models/v12/yolov12-ca.yaml",
            "csv": "runs/detect/ablation_ca_ciou/results.csv",
            "weights": "runs/detect/ablation_ca_ciou/weights/best.pt"
        },
        {
            "name": "Baseline + SIoU",
            "config": "ultralytics/cfg/models/v12/yolov12.yaml",
            "csv": "runs/detect/ablation_baseline_siou/results.csv",
            "weights": "runs/detect/ablation_baseline_siou/weights/best.pt"
        },
        {
            "name": "Baseline + BiFPN + CA",
            "config": "ultralytics/cfg/models/v12/yolov12-improved.yaml",
            "csv": "runs/detect/ablation_bifpn_ca_ciou/results.csv",
            "weights": "runs/detect/ablation_bifpn_ca_ciou/weights/best.pt"
        },
        {
            "name": "Baseline + BiFPN + CA + SIoU",
            "config": "ultralytics/cfg/models/v12/yolov12-improved.yaml",
            "csv": "runs/detect/ablation_bifpn_ca_siou/results.csv",
            "weights": "runs/detect/ablation_bifpn_ca_siou/weights/best.pt"
        }
    ]

    print(f"{'Model':<30} | {'Params (M)':<10} | {'FPS':<8} | {'mAP@0.5':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 90)

    for exp in experiments:
        # Get Metrics
        metrics = get_last_metrics(exp["csv"])
        map50 = "N/A"
        p = "N/A"
        r = "N/A"
        
        if metrics is not None:
            map50_val = metrics.get('metrics/mAP50(B)', 0)
            p_val = metrics.get('metrics/precision(B)', 0)
            r_val = metrics.get('metrics/recall(B)', 0)
            
            map50 = f"{map50_val:.4f}"
            p = f"{p_val:.4f}"
            r = f"{r_val:.4f}"
            
        # Get Params & FPS
        # Use weights if available for accurate inference speed (though architecture defines it)
        params, fps = get_model_info(exp["config"], exp["weights"])
        
        params_m = f"{params/1e6:.2f}"
        fps_str = f"{fps:.1f}"
        
        print(f"{exp['name']:<30} | {params_m:<10} | {fps_str:<8} | {map50:<10} | {p:<10} | {r:<10}")

if __name__ == "__main__":
    main()
