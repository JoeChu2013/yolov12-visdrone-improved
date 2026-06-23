import os
import pandas as pd
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

def train_and_eval(name, model_yaml_or_pt, device="0", epochs=100):
    print(f"\n=== Processing Model: {name} ===")
    
    save_dir = os.path.join("runs", "detect", name)
    weights_path = os.path.join(save_dir, "weights", "best.pt")
    
    # 1. Train if not exists
    if not os.path.exists(weights_path):
        print(f"Starting training for {name}...")
        # For standard models (v8, v9, v10, v11), we can load from .pt directly which implies config + weights
        # For custom models (v12-improved), we load yaml
        if model_yaml_or_pt.endswith('.yaml'):
             model = YOLO(model_yaml_or_pt)
             pretrained = True # Load backbone if possible
        else:
             model = YOLO(model_yaml_or_pt) # e.g. yolov8n.pt
             pretrained = True

        model.train(
            data='ultralytics/cfg/datasets/VisDrone.yaml',
            epochs=epochs,
            batch=16,
            imgsz=640,
            lr0=0.01,
            optimizer='SGD',
            name=name,
            device=device,
            project='runs/detect',
            exist_ok=True,
            pretrained=pretrained,
            val=False, # Speed up, val at end
            plots=False
        )
    else:
        print(f"Model {name} already trained. Skipping training.")

    # 2. Validation (to get metrics)
    # Reload best model
    if os.path.exists(weights_path):
        model = YOLO(weights_path)
        metrics = model.val(data='ultralytics/cfg/datasets/VisDrone.yaml', split='val', plots=False)
        return model, metrics
    return None, None

def get_params(model):
    return sum(x.numel() for x in model.model.parameters())

def run_inference_comparison(models_dict, image_path='datasets/VisDrone/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg'):
    # Check if image exists, if not pick a random one from dataset
    if not os.path.exists(image_path):
        # Try to find a jpg in visdrone path
        search_path = 'datasets/VisDrone/VisDrone2019-DET-val/images'
        if os.path.exists(search_path):
            files = [f for f in os.listdir(search_path) if f.endswith('.jpg')]
            if files:
                image_path = os.path.join(search_path, files[0])
            else:
                print("No validation images found for visualization.")
                return

    print(f"Generating inference comparison on {image_path}...")
    
    imgs = []
    names = []
    
    for name, model in models_dict.items():
        if model is None: continue
        # Inference
        results = model.predict(image_path, imgsz=640, conf=0.25)
        # Plot
        res_plotted = results[0].plot()
        # Convert BGR to RGB
        res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        imgs.append(res_plotted)
        names.append(name)
        
    # Plot grid
    n = len(imgs)
    if n == 0: return
    
    cols = 3
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(imgs[i])
        plt.title(names[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig('model_comparison_vis.png')
    print("Visualization saved to model_comparison_vis.png")

def main():
    # Define Models
    # Using 'n' (nano) scale for all for fair comparison
    # v12-improved is based on 'n' scale turbo
    models_config = {
        "YOLOv8n": "yolov8n.pt",
        "YOLOv9t": "yolov9t.pt", # v9t is comparable to n
        "YOLOv10n": "yolov10n.pt",
        "YOLOv11n": "yolo11n.pt",
        "YOLOv12n": "ultralytics/cfg/models/v12/yolov12.yaml", # Use our v12 yaml
        "YOLOv12-Improved": "ultralytics/cfg/models/v12/yolov12-improved.yaml"
    }
    
    device = "0" if torch.cuda.is_available() else "cpu"
    
    results_data = []
    trained_models = {}

    for name, cfg in models_config.items():
        # Special handling for v12/improved which we might have trained already or need yaml
        # For v12n and improved, we prefer using the weights we just trained if available to save time?
        # But user wants "comparison", let's assume we re-verify or use existing logic
        
        # Check if we have pre-trained ablation weights for v12 variants to save time
        if name == "YOLOv12n":
            # Map to ablation_baseline_ciou if exists
            if os.path.exists("runs/detect/ablation_baseline_ciou/weights/best.pt"):
                print(f"Using existing weights for {name}")
                model = YOLO("runs/detect/ablation_baseline_ciou/weights/best.pt")
                metrics = model.val(data='ultralytics/cfg/datasets/VisDrone.yaml', split='val')
                trained_models[name] = model
            else:
                model, metrics = train_and_eval(name, cfg, device)
                if model: trained_models[name] = model
        
        elif name == "YOLOv12-Improved":
            # Map to ablation_bifpn_ca_siou
            if os.path.exists("runs/detect/ablation_bifpn_ca_siou/weights/best.pt"):
                print(f"Using existing weights for {name}")
                model = YOLO("runs/detect/ablation_bifpn_ca_siou/weights/best.pt")
                metrics = model.val(data='ultralytics/cfg/datasets/VisDrone.yaml', split='val')
                trained_models[name] = model
            else:
                model, metrics = train_and_eval(name, cfg, device)
                if model: trained_models[name] = model
                
        else:
            # Standard models (v8, v9, v10, v11)
            model, metrics = train_and_eval(name, cfg, device)
            if model: trained_models[name] = model

        # Collect Data
        if name in trained_models and metrics:
            p = metrics.results_dict['metrics/precision(B)']
            r = metrics.results_dict['metrics/recall(B)']
            map50 = metrics.results_dict['metrics/mAP50(B)']
            params = get_params(trained_models[name])
            
            results_data.append({
                "Model": name,
                "Precision": p,
                "Recall": r,
                "mAP@0.5": map50,
                "Params (M)": params / 1e6
            })

    # Create Table
    df = pd.DataFrame(results_data)
    # Format
    df["Precision"] = df["Precision"].apply(lambda x: f"{x:.4f}")
    df["Recall"] = df["Recall"].apply(lambda x: f"{x:.4f}")
    df["mAP@0.5"] = df["mAP@0.5"].apply(lambda x: f"{x:.4f}")
    df["Params (M)"] = df["Params (M)"].apply(lambda x: f"{x:.2f}")
    
    print("\n=== Model Comparison Table ===")
    print(df.to_markdown(index=False))
    
    # Save CSV
    df.to_csv("model_comparison_final.csv", index=False)
    
    # Run Visualization
    run_inference_comparison(trained_models)

if __name__ == "__main__":
    main()
