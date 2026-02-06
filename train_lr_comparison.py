import os
import torch
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt

def is_completed(csv_path, target_epochs=50):
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        if df.empty: return False
        df.columns = [c.strip() for c in df.columns]
        last_epoch = df['epoch'].iloc[-1]
        # CSV epochs are 1-based usually in Ultralytics logs if using 'epoch' column? 
        # Actually they are 0-based or 1-based depending on version. 
        # Let's assume if we have 50 rows it's close enough.
        return len(df) >= target_epochs
    except:
        return False

def main():
    # 3 Learning Rates to compare
    lrs = [0.01, 0.001, 0.0001]
    
    model_cfg = "ultralytics/cfg/models/v12/yolov12-p2-improved.yaml"
    device = "0" if torch.cuda.is_available() else "cpu"
    
    results = {}
    
    print("=== Starting Learning Rate Comparison Experiment (Resume Supported) ===")
    
    for lr in lrs:
        name = f"lr_experiment_{lr}"
        print(f"\nProcessing {name} (lr0={lr})...")
        
        save_dir = os.path.join("runs", "detect", name)
        csv_path = os.path.join(save_dir, "results.csv")
        weights_dir = os.path.join(save_dir, "weights")
        last_pt = os.path.join(weights_dir, "last.pt")
        
        # 1. Check if fully completed
        if is_completed(csv_path, 50):
            print(f"  -> Already completed (50 epochs). Skipping.")
            results[lr] = pd.read_csv(csv_path)
            continue
            
        # 2. Check if can resume
        if os.path.exists(last_pt):
            print(f"  -> Resuming from {last_pt}...")
            try:
                model = YOLO(last_pt)
                model.train(resume=True)
                # After resume, load results
                if os.path.exists(csv_path):
                    results[lr] = pd.read_csv(csv_path)
                continue
            except Exception as e:
                print(f"  -> Failed to resume: {e}. Restarting...")
        
        # 3. Start New
        print(f"  -> Starting new training...")
        model = YOLO(model_cfg)
        model.train(
            data='ultralytics/cfg/datasets/VisDrone.yaml',
            epochs=50, 
            batch=16,
            imgsz=640,
            lr0=lr,
            lrf=0.01,
            optimizer='SGD',
            name=name,
            device=device,
            project='runs/detect',
            exist_ok=True,
            pretrained=True,
            plots=False
        )
        
        if os.path.exists(csv_path):
            results[lr] = pd.read_csv(csv_path)

    # Plotting
    if results:
        print("\nGenerating comparison plots...")
        plt.figure(figsize=(12, 6))
        
        # mAP@0.5 Convergence
        plt.subplot(1, 2, 1)
        for lr, df in results.items():
            if df is None or df.empty: continue
            df.columns = [c.strip() for c in df.columns]
            plt.plot(df['epoch'], df['metrics/mAP50(B)'], label=f'lr0={lr}')
        
        plt.title('mAP@0.5 Convergence by Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('mAP@0.5')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss Convergence
        plt.subplot(1, 2, 2)
        for lr, df in results.items():
            if df is None or df.empty: continue
            df.columns = [c.strip() for c in df.columns]
            plt.plot(df['epoch'], df['train/box_loss'], label=f'lr0={lr}')
            
        plt.title('Training Box Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Box Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lr_convergence_comparison.png', dpi=300)
        print("Comparison plot saved to lr_convergence_comparison.png")

if __name__ == "__main__":
    main()
