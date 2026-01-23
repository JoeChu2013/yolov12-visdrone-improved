import pandas as pd
import torch
from ultralytics import YOLO
import time
import os

def get_last_metrics(csv_path):
    if not os.path.exists(csv_path):
        return None
    # Ultralytics results.csv usually has spaces in column names
    try:
        df = pd.read_csv(csv_path)
        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]
        last_row = df.iloc[-1]
        return last_row
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def get_model_info(yaml_path):
    try:
        # Load model from yaml (build from scratch)
        model = YOLO(yaml_path)
        # Get simplified model info
        # Note: model.info() prints to stdout, we need to access properties if possible
        # Or just infer on dummy input to get speed
        
        # Calculate Params
        n_params = sum(x.numel() for x in model.model.parameters())
        
        # Measure FPS (Dummy inference)
        input_tensor = torch.randn(1, 3, 640, 640)
        if torch.cuda.is_available():
            model.model.to('cuda')
            input_tensor = input_tensor.to('cuda')
        
        model.model.eval()
        
        # Warmup
        for _ in range(10):
            _ = model.model(input_tensor)
            
        # Timing
        start_time = time.time()
        for _ in range(50):
            _ = model.model(input_tensor)
        end_time = time.time()
        
        fps = 50 / (end_time - start_time)
        
        return n_params, fps
    except Exception as e:
        print(f"Error analyzing {yaml_path}: {e}")
        return None, None

def main():
    print("=== Ablation Study Results ===\n")
    
    # 1. Improved Model Results (from training log)
    improved_csv = 'runs/detect/train2/results.csv'
    improved_metrics = get_last_metrics(improved_csv)
    
    # 2. Baseline Model Results (try to find a matching log or use generic)
    # Assuming the improved model was 'n' scale based on typical defaults or file size
    baseline_csv = 'logs/yolov12_turbo_n.csv' 
    baseline_metrics = get_last_metrics(baseline_csv)

    # Display Metrics
    headers = ["Model", "Precision (P)", "Recall (R)", "mAP50", "mAP50-95", "Params (M)", "FPS"]
    print(f"{headers[0]:<20} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} {headers[4]:<15} {headers[5]:<15} {headers[6]:<15}")
    
    # Analyze Improved Model
    imp_params, imp_fps = get_model_info('ultralytics/cfg/models/v12/yolov12-improved.yaml')
    if improved_metrics is not None:
        # Keys might vary, let's print columns to be sure if this fails, but usually:
        # metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
        p = improved_metrics.get('metrics/precision(B)', 0)
        r = improved_metrics.get('metrics/recall(B)', 0)
        map50 = improved_metrics.get('metrics/mAP50(B)', 0)
        map95 = improved_metrics.get('metrics/mAP50-95(B)', 0)
        
        print(f"{'YOLOv12-Improved':<20} {p:<15.4f} {r:<15.4f} {map50:<15.4f} {map95:<15.4f} {imp_params/1e6:<15.2f} {imp_fps:<15.2f}")
    else:
        print(f"{'YOLOv12-Improved':<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {imp_params/1e6:<15.2f} {imp_fps:<15.2f}")

    # Analyze Baseline Model
    base_params, base_fps = get_model_info('ultralytics/cfg/models/v12/yolov12.yaml')
    if baseline_metrics is not None:
        p = baseline_metrics.get('metrics/precision(B)', 0)
        r = baseline_metrics.get('metrics/recall(B)', 0)
        map50 = baseline_metrics.get('metrics/mAP50(B)', 0)
        map95 = baseline_metrics.get('metrics/mAP50-95(B)', 0)
        
        print(f"{'YOLOv12-Base':<20} {p:<15.4f} {r:<15.4f} {map50:<15.4f} {map95:<15.4f} {base_params/1e6:<15.2f} {base_fps:<15.2f}")
    else:
        print(f"{'YOLOv12-Base':<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {base_params/1e6:<15.2f} {base_fps:<15.2f}")

if __name__ == "__main__":
    main()
