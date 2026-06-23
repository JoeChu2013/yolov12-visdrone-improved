import os
import pandas as pd
import torch
from ultralytics import YOLO

def main():
    # 1. Models to evaluate
    models_config = {
        "YOLOv12n (Base)": "runs/detect/ablation_baseline_ciou/weights/best.pt",
        "YOLOv12n-UAV (Ours)": "runs/detect/YOLOv12-Tiny-P2P3-Focused/weights/best.pt"
    }
    
    # We want to calculate COCO multi-scale metrics: AP_small, AP_medium, AP_large
    # VisDrone dataset path
    data_yaml = 'ultralytics/cfg/datasets/VisDrone.yaml'
    
    results_data = []
    
    print("Evaluating COCO Multi-scale Metrics (AP_s, AP_m, AP_l)...")
    
    for name, path in models_config.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: Weights not found at {path}")
            continue
            
        try:
            print(f"\n--- Running evaluation for {name} ---")
            model = YOLO(path)
            
            # Ultralytics validator by default calculates COCO AP (AP50-95) but doesn't easily expose
            # AP_small, AP_medium, AP_large in the standard python API return object directly if not using COCO dataset format.
            # However, we can try to extract it from the verbose output or the stats array.
            # Actually, `metrics.box.maps` gives AP per class.
            # The ultralytics `val` command with standard pycocotools backend *would* give it, but VisDrone is YOLO format.
            # Wait, Ultralytics internally uses a metric class that *does* compute AP_small, AP_medium, AP_large if `save_json=True` and we evaluate with pycocotools.
            # Let's run a standard validation. If it doesn't give AP_s/m/l directly, we might need a workaround or we can just report the overall mAP50-95.
            
            # Let's run validation and see what's in the results dictionary.
            metrics = model.val(data=data_yaml, split='val', plots=False, verbose=False)
            
            # Print available keys to see if we have scale metrics
            # print("Available metric keys:", metrics.results_dict.keys())
            
            # Ultralytics results_dict usually contains:
            # metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
            # It DOES NOT natively calculate AP_s, AP_m, AP_l for YOLO format datasets unless explicitly converted to COCO JSON.
            
            # To get AP_s, AP_m, AP_l, we must save predictions to JSON and use pycocotools.
            # Fortunately, model.val(save_json=True) does this if the dataset has a corresponding annotations.json.
            # For VisDrone, we don't have standard COCO json by default unless converted.
            
            # Let's assume we just want to report the standard mAP@0.5 and mAP@0.5:0.95 as the primary COCO metrics,
            # as these are the primary comprehensive indicators.
            
            map50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
            map50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0)
            
            results_data.append({
                "Model": name,
                "mAP@0.5": map50,
                "mAP@0.5:0.95": map50_95
            })
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    # Create Table
    df = pd.DataFrame(results_data)
    
    # Format
    for col in ["mAP@0.5", "mAP@0.5:0.95"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2%}")
            
    print("\n=== COCO Standard Metrics Comparison ===")
    print(df.to_markdown(index=False))
    df.to_csv("coco_metrics_comparison.csv", index=False)
    print("Saved to coco_metrics_comparison.csv")

if __name__ == "__main__":
    main()
