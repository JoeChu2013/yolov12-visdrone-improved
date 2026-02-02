import os
import pandas as pd
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

def calculate_f1(p, r):
    if p + r == 0: return 0
    return 2 * (p * r) / (p + r)

def get_flops(model, imgsz=640):
    try:
        # Use thop or ultralytics built-in info
        # Ultralytics model.info() prints FLOPs but doesn't return it easily in API
        # Let's approximate or use what's available
        # Actually model.info() returns (n_l, n_p, n_g, flops)
        return model.model.info(verbose=False)[3] 
    except:
        return 0

def main():
    # Define Models
    models_config = {
        "YOLOv8n": "yolov8n.pt",
        "YOLOv9t": "yolov9t.pt",
        "YOLOv10n": "yolov10n.pt",
        "YOLOv11n": "yolo11n.pt",
        "YOLOv12n (Base)": "runs/detect/ablation_baseline_ciou/weights/best.pt",
        "YOLOv12-Improved (BiFPN+CA+SIoU)": "runs/detect/ablation_bifpn_ca_siou/weights/best.pt",
        "YOLOv12-P2-Improved (Ours)": "runs/detect/YOLOv12-P2-Improved/weights/best.pt"
    }
    
    device = "0" if torch.cuda.is_available() else "cpu"
    results_data = []
    loaded_models = {}

    print("Gathering metrics and FLOPs...")
    
    for name, path in models_config.items():
        if not os.path.exists(path) and not path.endswith('.pt'):
            print(f"Skipping {name}: path {path} not found.")
            continue
            
        try:
            model = YOLO(path)
            loaded_models[name] = model
            
            # Get metrics (assume training/val finished, or run quick val)
            # For efficiency, if it's a 'runs/' path, we try to read results.csv first
            # If not, we run val()
            
            p, r, map50, map5095 = 0, 0, 0, 0
            
            # Try reading from CSV if available (faster)
            if "runs/detect" in path:
                csv_path = os.path.dirname(os.path.dirname(path)) + "/results.csv"
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df.columns = [c.strip() for c in df.columns]
                    # Get best mAP epoch or last? Usually best.pt corresponds to best mAP50
                    # Let's find row with max mAP50
                    best_row = df.loc[df['metrics/mAP50(B)'].idxmax()]
                    p = best_row['metrics/precision(B)']
                    r = best_row['metrics/recall(B)']
                    map50 = best_row['metrics/mAP50(B)']
                else:
                    # Run val
                    metrics = model.val(data='ultralytics/cfg/datasets/VisDrone.yaml', split='val', plots=False)
                    p = metrics.results_dict['metrics/precision(B)']
                    r = metrics.results_dict['metrics/recall(B)']
                    map50 = metrics.results_dict['metrics/mAP50(B)']
            else:
                # Standard models (v8/v9/etc), run val to get VisDrone metrics (pretrained COCO weights won't give VisDrone metrics unless we trained them)
                # Wait, did we train v8/v9/v10/v11 in compare_sota.py?
                # Yes, they should be in runs/detect/YOLOv8n etc.
                # Let's adjust paths to point to trained weights if possible
                trained_path = os.path.join("runs", "detect", name, "weights", "best.pt")
                if os.path.exists(trained_path):
                    model = YOLO(trained_path) # Reload trained
                    loaded_models[name] = model
                    csv_path = os.path.join("runs", "detect", name, "results.csv")
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        df.columns = [c.strip() for c in df.columns]
                        best_row = df.loc[df['metrics/mAP50(B)'].idxmax()]
                        p = best_row['metrics/precision(B)']
                        r = best_row['metrics/recall(B)']
                        map50 = best_row['metrics/mAP50(B)']
                else:
                    # Fallback: User might have just asked for comparison but we only trained some?
                    # Assuming compare_sota.py finished, they should exist.
                    # If not, we skip or put N/A
                    pass

            # Params & FLOPs
            params = sum(x.numel() for x in model.model.parameters()) / 1e6
            flops = get_flops(model)
            
            f1 = calculate_f1(p, r)
            
            results_data.append({
                "Model": name,
                "Precision": p,
                "Recall": r,
                "F1-Score": f1,
                "mAP@0.5": map50,
                "Params (M)": params,
                "FLOPs (G)": flops
            })
            
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # Create Table
    df_res = pd.DataFrame(results_data)
    # Sort by mAP
    df_res = df_res.sort_values(by="mAP@0.5", ascending=False)
    
    # Format
    df_display = df_res.copy()
    for col in ["Precision", "Recall", "F1-Score", "mAP@0.5"]:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}")
    for col in ["Params (M)", "FLOPs (G)"]:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}")
        
    print("\n=== Comprehensive Model Comparison ===")
    print(df_display.to_markdown(index=False))
    df_display.to_csv("final_comprehensive_comparison.csv", index=False)

    # Visualization
    print("\nGenerating Visual Comparison...")
    # Use a challenging image
    img_path = 'datasets/VisDrone/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg'
    if not os.path.exists(img_path):
        # Fallback
        search = 'datasets/VisDrone/VisDrone2019-DET-val/images'
        if os.path.exists(search):
            img_path = os.path.join(search, os.listdir(search)[0])
    
    if os.path.exists(img_path):
        plt.figure(figsize=(20, 10))
        num_models = len(loaded_models)
        cols = 4
        rows = (num_models + cols - 1) // cols
        
        for i, (name, model) in enumerate(loaded_models.items()):
            res = model.predict(img_path, conf=0.25, verbose=False)[0]
            plotted = res.plot()
            plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            
            plt.subplot(rows, cols, i+1)
            plt.imshow(plotted)
            plt.title(f"{name}\nmAP: {df_res[df_res['Model']==name]['mAP@0.5'].values[0]:.2%}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig('final_visual_comparison.png')
        print("Visual comparison saved to final_visual_comparison.png")

if __name__ == "__main__":
    main()
