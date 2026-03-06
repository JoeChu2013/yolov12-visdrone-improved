import os
import torch
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

def main():
    name = "YOLOv12-Tiny-P2P3-Focused"
    model_cfg = "ultralytics/cfg/models/v12/yolov12-tiny-p2p3.yaml"
    device = "0" if torch.cuda.is_available() else "cpu"
    
    print(f"\n=== Training Tiny Object Focused Model: {name} ===")
    
    save_dir = os.path.join("runs", "detect", name)
    weights_path = os.path.join(save_dir, "weights", "best.pt")
    
    if not os.path.exists(weights_path):
        model = YOLO(model_cfg)
        model.train(
            data='ultralytics/cfg/datasets/VisDrone.yaml',
            epochs=100,
            batch=32, # Much smaller model, can use larger batch!
            imgsz=640,
            lr0=0.01,
            optimizer='SGD',
            name=name,
            device=device,
            project='runs/detect',
            exist_ok=True,
            pretrained=True,
            plots=True
        )
    else:
        print("Model already trained.")

    # Evaluate
    model = YOLO(weights_path)
    metrics = model.val(data='ultralytics/cfg/datasets/VisDrone.yaml', split='val')
    
    print(f"\nTiny P2P3 Model Results: mAP@0.5={metrics.results_dict['metrics/mAP50(B)']:.4f}")

if __name__ == "__main__":
    main()
