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
        last_row = df.iloc[-1]
        return last_row
    except Exception as e:
        return None

def get_model_info(yaml_path):
    try:
        model = YOLO(yaml_path)
        n_params = sum(x.numel() for x in model.model.parameters())
        
        # Measure FPS (Dummy inference on CPU to be safe and consistent)
        input_tensor = torch.randn(1, 3, 640, 640)
        
        # Use CPU for stable relative comparison if GPU is busy or not available
        device = 'cpu' 
        # If GPU is available and reliable, use it
        if torch.cuda.is_available():
            device = 'cuda'
            
        model.model.to(device)
        input_tensor = input_tensor.to(device)
        model.model.eval()
        
        # Warmup
        for _ in range(5):
            _ = model.model(input_tensor)
            
        # Timing
        start_time = time.time()
        n_iters = 30
        for _ in range(n_iters):
            _ = model.model(input_tensor)
        end_time = time.time()
        
        fps = n_iters / (end_time - start_time)
        
        return n_params, fps
    except Exception as e:
        print(f"Error analyzing {yaml_path}: {e}")
        return None, None

def main():
    experiments = [
        {
            "name": "Baseline",
            "config": "ultralytics/cfg/models/v12/yolov12.yaml",
            "log": "logs/yolov12_turbo_n.csv", # Provided baseline log
            "note": ""
        },
        {
            "name": "Baseline + BiFPN",
            "config": "ultralytics/cfg/models/v12/yolov12-bifpn.yaml",
            "log": None,
            "note": "Requires training"
        },
        {
            "name": "Baseline + CA",
            "config": "ultralytics/cfg/models/v12/yolov12-ca.yaml",
            "log": None,
            "note": "Requires training"
        },
        {
            "name": "Baseline + SIoU",
            "config": "ultralytics/cfg/models/v12/yolov12-siou.yaml",
            "log": None, # Assuming no specific log for just SIoU yet
            "note": "Requires training"
        },
        {
            "name": "Baseline + BiFPN + CA",
            "config": "ultralytics/cfg/models/v12/yolov12-improved.yaml",
            "log": "runs/detect/train2/results.csv", # Existing run
            "note": ""
        },
        {
            "name": "Baseline + BiFPN + CA + SIoU",
            "config": "ultralytics/cfg/models/v12/yolov12-bifpn-ca-siou.yaml",
            "log": None,
            "note": "Requires training"
        }
    ]

    print(f"{'Experiment':<35} | {'Params (M)':<12} | {'FPS':<10} | {'mAP@0.5':<10} | {'Note':<15}")
    print("-" * 95)

    for exp in experiments:
        params, fps = get_model_info(exp["config"])
        
        map50 = "N/A"
        if exp["log"]:
            metrics = get_last_metrics(exp["log"])
            if metrics is not None:
                # Try standard keys
                val = metrics.get('metrics/mAP50(B)', None)
                if val is not None:
                    map50 = f"{val:.4f}"
        
        if params:
            params_m = f"{params/1e6:.2f} M"
            fps_str = f"{fps:.2f}"
        else:
            params_m = "Error"
            fps_str = "Error"

        print(f"{exp['name']:<35} | {params_m:<12} | {fps_str:<10} | {map50:<10} | {exp['note']:<15}")

if __name__ == "__main__":
    main()
