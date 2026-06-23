import os
import pandas as pd
from ultralytics import YOLO
import torch

def is_completed(save_dir, target_epochs=100):
    csv_path = os.path.join(save_dir, "results.csv")
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        if 'epoch' in df.columns:
            last_epoch = df['epoch'].iloc[-1]
            # epoch is 1-indexed in CSV usually? Let's check. 
            # In logs: 1, 2... 100. So if last is 100, it's done.
            return last_epoch >= target_epochs
    except:
        return False
    return False

def train_experiment(name, model_yaml, iou_type, device="0"):
    print(f"\n=== Checking Experiment: {name} ===")
    
    # Set Environment Variable for IoU
    os.environ['YOLO_IOU_TYPE'] = iou_type
    
    save_dir = os.path.join("runs", "detect", name)
    weights_dir = os.path.join(save_dir, "weights")
    last_pt = os.path.join(weights_dir, "last.pt")
    
    # 1. Check if completed
    if is_completed(save_dir, 100):
        print(f"Experiment {name} already completed (100 epochs). Skipping...")
        return

    # 2. Check if can resume
    if os.path.exists(last_pt):
        print(f"Resuming experiment {name} from {last_pt}...")
        try:
            model = YOLO(last_pt)
            model.train(resume=True)
            return
        except Exception as e:
            print(f"Failed to resume {name}: {e}. Restarting...")

    # 3. Start New
    print(f"Starting experiment {name}...")
    model = YOLO(model_yaml)
    
    # Accelerated training config
    # val=False to speed up (validate at end)
    # plots=False to speed up
    model.train(
        data='ultralytics/cfg/datasets/VisDrone.yaml',
        epochs=100,
        batch=16,
        imgsz=640,
        lr0=0.01,
        optimizer='SGD',
        name=name,
        device=device,
        project='runs/detect',
        exist_ok=True,
        pretrained=True,
        val=False,   # Speed up
        plots=False, # Speed up
    )
    
    # Final Validation to ensure we get metrics
    print(f"Running final validation for {name}...")
    # Load best (or last if val was disabled)
    best_pt = os.path.join(weights_dir, "best.pt")
    if not os.path.exists(best_pt):
        best_pt = last_pt
    
    if os.path.exists(best_pt):
        val_model = YOLO(best_pt)
        val_model.val(data='ultralytics/cfg/datasets/VisDrone.yaml', split='val')

def main():
    experiments = [
        # 1. Baseline (CIoU)
        {
            "name": "ablation_baseline_ciou",
            "model": "ultralytics/cfg/models/v12/yolov12.yaml",
            "iou": "CIoU"
        },
        # 2. Baseline + BiFPN (CIoU)
        {
            "name": "ablation_bifpn_ciou",
            "model": "ultralytics/cfg/models/v12/yolov12-bifpn.yaml",
            "iou": "CIoU"
        },
        # 3. Baseline + CA (CIoU)
        {
            "name": "ablation_ca_ciou",
            "model": "ultralytics/cfg/models/v12/yolov12-ca.yaml",
            "iou": "CIoU"
        },
        # 4. Baseline + SIoU
        {
            "name": "ablation_baseline_siou",
            "model": "ultralytics/cfg/models/v12/yolov12.yaml",
            "iou": "SIoU"
        },
        # 5. Baseline + BiFPN + CA (CIoU)
        {
            "name": "ablation_bifpn_ca_ciou",
            "model": "ultralytics/cfg/models/v12/yolov12-improved.yaml",
            "iou": "CIoU"
        },
        # 6. Baseline + BiFPN + CA + SIoU
        {
            "name": "ablation_bifpn_ca_siou",
            "model": "ultralytics/cfg/models/v12/yolov12-improved.yaml",
            "iou": "SIoU"
        }
    ]
    
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for exp in experiments:
        try:
            train_experiment(exp["name"], exp["model"], exp["iou"], device)
        except Exception as e:
            print(f"Error in experiment {exp['name']}: {e}")

if __name__ == "__main__":
    main()
