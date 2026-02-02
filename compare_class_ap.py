import pandas as pd
import os
from ultralytics import YOLO

def main():
    # 1. Models to compare
    models = {
        "YOLOv12n (Baseline)": "runs/detect/ablation_baseline_ciou/weights/best.pt",
        "YOLOv12-Improved (BiFPN+CA)": "runs/detect/ablation_bifpn_ca_siou/weights/best.pt",
        "YOLOv12-P2-Improved (Final)": "runs/detect/YOLOv12-P2-Improved/weights/best.pt"
    }
    
    # VisDrone Classes
    visdrone_classes = [
        'pedestrian', 'people', 'bicycle', 'car', 'van', 
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    data = []
    
    print("Evaluating Per-Class AP...")
    
    for model_name, weights_path in models.items():
        if not os.path.exists(weights_path):
            print(f"Skipping {model_name} (weights not found)")
            continue
            
        try:
            model = YOLO(weights_path)
            # Run validation to get per-class metrics
            # verbose=True usually prints per-class table, but we need to access it programmatically
            metrics = model.val(data='ultralytics/cfg/datasets/VisDrone.yaml', split='val', plots=False, verbose=False)
            
            # metrics.box.maps is an array of mAP50-95 for each class
            # metrics.box.ap50 is likely what we want for AP@0.5 per class
            # Actually Ultralytics metrics object stores:
            # metrics.ap_class_index (indices of classes seen)
            # metrics.box.map50 (all classes mean)
            # metrics.box.maps (mAP50-95 per class)
            # To get AP@0.5 per class, we might need to dig into metrics.box.all_ap or similar
            # metrics.box.all_ap is (N, 10) array where 10 is IoU thresholds. Column 0 is AP@0.5.
            
            ap50_per_class = metrics.box.all_ap[:, 0] # Array of AP@0.5 for each class index
            class_indices = metrics.ap_class_index
            
            row = {"Model": model_name}
            
            # Initialize all with 0
            for cls in visdrone_classes:
                row[cls] = 0.0
                
            # Fill
            for i, class_idx in enumerate(class_indices):
                if class_idx < len(visdrone_classes):
                    cls_name = visdrone_classes[class_idx]
                    row[cls_name] = ap50_per_class[i]
            
            # Add mean
            row["mAP@0.5"] = metrics.box.map50
            
            data.append(row)
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns
    cols = ["Model", "mAP@0.5"] + visdrone_classes
    df = df[cols]
    
    # Format as percentage
    for col in df.columns:
        if col != "Model":
            df[col] = df[col].apply(lambda x: f"{x:.2%}")
            
    # Calculate Improvement (Final - Baseline)
    if len(df) >= 3:
        baseline = df.iloc[0]
        final = df.iloc[2]
        
        diff_row = {"Model": "Improvement (Final vs Base)"}
        for col in df.columns:
            if col != "Model":
                # Convert back to float for subtraction
                val_base = float(baseline[col].strip('%'))
                val_final = float(final[col].strip('%'))
                diff = val_final - val_base
                diff_row[col] = f"+{diff:.2f}%"
        
        df = pd.concat([df, pd.DataFrame([diff_row])], ignore_index=True)

    print("\n=== Per-Class AP@0.5 Comparison ===")
    print(df.to_markdown(index=False))
    
    # Save
    df.to_csv("per_class_ap_comparison.csv", index=False)
    print("Saved to per_class_ap_comparison.csv")

if __name__ == "__main__":
    main()
