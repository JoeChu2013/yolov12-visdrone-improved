import os
import pandas as pd
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

def main():
    name = "YOLOv12-P2-Improved"
    model_cfg = "ultralytics/cfg/models/v12/yolov12-p2-improved.yaml"
    device = "0" if torch.cuda.is_available() else "cpu"
    
    print(f"\n=== Training Final Improved Model: {name} ===")
    
    # 1. Train
    # Using SIoU by env var
    os.environ['YOLO_IOU_TYPE'] = 'SIoU'
    
    save_dir = os.path.join("runs", "detect", name)
    weights_path = os.path.join(save_dir, "weights", "best.pt")
    
    if not os.path.exists(weights_path):
        model = YOLO(model_cfg)
        model.train(
            data='ultralytics/cfg/datasets/VisDrone.yaml',
            epochs=100,
            batch=16, # P2 head adds memory, might need smaller batch if OOM, but 16 usually ok for 'n' scale
            imgsz=640,
            lr0=0.01,
            optimizer='SGD',
            name=name,
            device=device,
            project='runs/detect',
            exist_ok=True,
            pretrained=True, # Will load matching backbone weights
            val=True,
            plots=True
        )
    else:
        print("Model already trained.")
        model = YOLO(weights_path)
    
    # 2. Evaluate
    metrics = model.val(data='ultralytics/cfg/datasets/VisDrone.yaml', split='val')
    
    print("\n=== Final Results ===")
    print(f"mAP@0.5: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.4f}")
    
    # 3. Add to Comparison Table
    csv_path = "model_comparison_final.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Check if already exists
        if name not in df['Model'].values:
            params = sum(x.numel() for x in model.model.parameters()) / 1e6
            new_row = {
                "Model": name,
                "Precision": f"{metrics.results_dict['metrics/precision(B)']:.4f}",
                "Recall": f"{metrics.results_dict['metrics/recall(B)']:.4f}",
                "mAP@0.5": f"{metrics.results_dict['metrics/mAP50(B)']:.4f}",
                "Params (M)": f"{params:.2f}"
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(csv_path, index=False)
            print("Added to comparison CSV.")
            print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
