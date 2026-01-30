import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_comparison():
    # Paths
    ciou_path = 'runs/detect/ablation_baseline_ciou/results.csv'
    siou_path = 'runs/detect/ablation_baseline_siou/results.csv'
    
    if not os.path.exists(ciou_path) or not os.path.exists(siou_path):
        print("Error: Results files not found.")
        return

    # Load Data
    df_ciou = pd.read_csv(ciou_path)
    df_siou = pd.read_csv(siou_path)
    
    # Strip columns
    df_ciou.columns = [c.strip() for c in df_ciou.columns]
    df_siou.columns = [c.strip() for c in df_siou.columns]

    # Extract Data
    epochs = df_ciou['epoch']
    
    # Box Loss is the most relevant for IoU loss comparison
    box_loss_ciou = df_ciou['train/box_loss']
    box_loss_siou = df_siou['train/box_loss']
    
    val_box_loss_ciou = df_ciou['val/box_loss']
    val_box_loss_siou = df_siou['val/box_loss']
    
    map_ciou = df_ciou['metrics/mAP50(B)']
    map_siou = df_siou['metrics/mAP50(B)']

    # Plotting
    plt.figure(figsize=(15, 5))

    # 1. Train Box Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, box_loss_ciou, label='CIoU (Baseline)', color='blue', linestyle='--')
    plt.plot(epochs, box_loss_siou, label='SIoU (Improved)', color='red')
    plt.title('Training Box Loss Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Box Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Val Box Loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_box_loss_ciou, label='CIoU (Baseline)', color='blue', linestyle='--')
    plt.plot(epochs, val_box_loss_siou, label='SIoU (Improved)', color='red')
    plt.title('Validation Box Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Box Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. mAP@0.5
    plt.subplot(1, 3, 3)
    plt.plot(epochs, map_ciou, label='CIoU (Baseline)', color='blue', linestyle='--')
    plt.plot(epochs, map_siou, label='SIoU (Improved)', color='red')
    plt.title('mAP@0.5 Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('siou_vs_ciou_convergence.png')
    print("Comparison plot saved to siou_vs_ciou_convergence.png")
    
    # Print numerical comparison at key epochs
    print("\n=== Detailed Performance Comparison ===")
    print(f"{'Epoch':<10} | {'Metric':<15} | {'CIoU':<10} | {'SIoU':<10} | {'Diff':<10}")
    print("-" * 65)
    
    checkpoints = [10, 30, 50, 80, 100]
    for ep in checkpoints:
        if ep > len(epochs): continue
        idx = ep - 1
        
        # Loss (Lower is better)
        loss_c = box_loss_ciou[idx]
        loss_s = box_loss_siou[idx]
        diff_loss = loss_s - loss_c
        indicator_loss = "▼" if diff_loss < 0 else "▲"
        
        print(f"{ep:<10} | {'Train Box Loss':<15} | {loss_c:.4f}     | {loss_s:.4f}     | {diff_loss:.4f} {indicator_loss}")
        
        # mAP (Higher is better)
        map_c = map_ciou[idx]
        map_s = map_siou[idx]
        diff_map = map_s - map_c
        indicator_map = "▲" if diff_map > 0 else "▼"
        
        print(f"{ep:<10} | {'mAP@0.5':<15} | {map_c:.4f}     | {map_s:.4f}     | {diff_map:.4f} {indicator_map}")
        print("-" * 65)

if __name__ == "__main__":
    plot_comparison()
