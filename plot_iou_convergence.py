import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Paths
    # Baseline (CIoU)
    path_ciou = 'runs/detect/ablation_baseline_ciou/results.csv'
    # SIoU only (or Improved which uses SIoU)
    # Ideally we compare "Baseline+CIoU" vs "Baseline+SIoU"
    # Do we have a pure SIoU run? 
    # Yes, ablation_baseline_siou
    path_siou = 'runs/detect/ablation_baseline_siou/results.csv'
    
    # If pure SIoU run doesn't exist, we can use the intermediate improved (BiFPN+CA+SIoU) 
    # but that mixes factors. Let's check if pure SIoU exists.
    if not os.path.exists(path_siou):
        # Fallback to the improved model if specific siou ablation missing
        # But user specifically asked for IoU comparison.
        # Let's assume we have it or use the closest proxy.
        # Actually in our conversation history we did 6 groups, one was Baseline+SIoU.
        pass

    data_exists = True
    if not os.path.exists(path_ciou):
        print(f"CIoU data missing: {path_ciou}")
        data_exists = False
    if not os.path.exists(path_siou):
        print(f"SIoU data missing: {path_siou}")
        data_exists = False
        
    if not data_exists:
        return

    # Load
    df_c = pd.read_csv(path_ciou)
    df_s = pd.read_csv(path_siou)
    
    # Strip columns
    df_c.columns = [c.strip() for c in df_c.columns]
    df_s.columns = [c.strip() for c in df_s.columns]
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    # 1. mAP@0.5 Convergence (Speed of Accuracy)
    plt.subplot(2, 2, 1)
    plt.plot(df_c['epoch'], df_c['metrics/mAP50(B)'], label='CIoU', color='blue', linestyle='--')
    plt.plot(df_s['epoch'], df_s['metrics/mAP50(B)'], label='SIoU', color='red')
    plt.title('mAP@0.5 Convergence Speed')
    plt.xlabel('Epoch')
    plt.ylabel('mAP@0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Box Loss Convergence (Speed of Regression Learning)
    plt.subplot(2, 2, 2)
    plt.plot(df_c['epoch'], df_c['val/box_loss'], label='CIoU', color='blue', linestyle='--')
    plt.plot(df_s['epoch'], df_s['val/box_loss'], label='SIoU', color='red')
    plt.title('Validation Box Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Box Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Precision Convergence
    plt.subplot(2, 2, 3)
    plt.plot(df_c['epoch'], df_c['metrics/precision(B)'], label='CIoU', color='blue', linestyle='--')
    plt.plot(df_s['epoch'], df_s['metrics/precision(B)'], label='SIoU', color='red')
    plt.title('Precision Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Recall Convergence
    plt.subplot(2, 2, 4)
    plt.plot(df_c['epoch'], df_c['metrics/recall(B)'], label='CIoU', color='blue', linestyle='--')
    plt.plot(df_s['epoch'], df_s['metrics/recall(B)'], label='SIoU', color='red')
    plt.title('Recall Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('iou_loss_convergence_detailed.png', dpi=300)
    print("Detailed convergence comparison saved to iou_loss_convergence_detailed.png")

if __name__ == "__main__":
    main()
