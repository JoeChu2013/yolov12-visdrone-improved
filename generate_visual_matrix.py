import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import random
from matplotlib import font_manager

def main():
    # 1. Models to Compare (Rows)
    models_config = {
        "YOLOv8n": "yolov8n.pt",
        "YOLOv10n": "yolov10n.pt",
        "YOLOv12n": "runs/detect/ablation_baseline_ciou/weights/best.pt",
        "Improved YOLOv12": "runs/detect/YOLOv12-P2-Improved/weights/best.pt"
    }

    # 2. Select 4 Images for different scenarios (Columns)
    # Trying to find images that represent: Low Light, Occlusion, Dense, etc.
    # Since we can't manually pick easily without seeing, we'll pick 4 random distinct images from validation set
    val_img_dir = 'datasets/VisDrone/VisDrone2019-DET-val/images'
    if not os.path.exists(val_img_dir):
        print("Validation directory not found.")
        return

    all_images = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
    if len(all_images) < 4:
        print("Not enough images.")
        return
        
    # Fixed seed for reproducibility or pick specific ones if known
    random.seed(42)
    selected_images = random.sample(all_images, 4)
    
    # Scenarios labels for the bottom
    scenarios = ["Scenario A", "Scenario B", "Scenario C", "Scenario D"]

    # 3. Setup Plot
    # Grid: (Num Models + 1 for Ground Truth/Original) x 4 Images
    num_rows = len(models_config) + 1
    num_cols = 4
    
    plt.figure(figsize=(16, 3 * num_rows))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Configure Font for Chinese support if needed (SimHei usually on Windows)
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    # Load Models
    loaded_models = {}
    for name, path in models_config.items():
        if os.path.exists(path) or path.endswith('.pt'):
            try:
                loaded_models[name] = YOLO(path)
            except:
                print(f"Failed to load {name}")

    # --- ROW 1 to N-1: Model Predictions ---
    row_idx = 0
    for model_name, model in loaded_models.items():
        for col_idx, img_path in enumerate(selected_images):
            # Predict
            results = model.predict(img_path, conf=0.25, verbose=False)[0]
            
            # Plot with boxes (hide conf/labels to look cleaner like the example? 
            # The example shows boxes and maybe labels. Let's keep default plot but maybe thinner line)
            # Ultralytics plot() returns numpy array
            res_plotted = results.plot(line_width=2, font_size=10) 
            res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            # Subplot index: row * cols + col + 1
            ax = plt.subplot(num_rows, num_cols, row_idx * num_cols + col_idx + 1)
            plt.imshow(res_plotted)
            
            # Y-Label only on first column
            if col_idx == 0:
                plt.ylabel(model_name, fontsize=14, labelpad=10, rotation=0, ha='right', va='center')
            
            # Remove ticks
            plt.xticks([])
            plt.yticks([])
            
            # Remove frame?
            for spine in ax.spines.values():
                spine.set_visible(False)
                
        row_idx += 1

    # --- ROW N: Original Image (Ground Truth style or just Raw) ---
    # The example shows "Original" (Raw) at the bottom
    for col_idx, img_path in enumerate(selected_images):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax = plt.subplot(num_rows, num_cols, row_idx * num_cols + col_idx + 1)
        plt.imshow(img)
        
        if col_idx == 0:
            plt.ylabel("Original", fontsize=14, labelpad=10, rotation=0, ha='right', va='center')
            
        plt.xticks([])
        plt.yticks([])
        
        # Scenario Label at bottom
        plt.xlabel(scenarios[col_idx], fontsize=12, labelpad=5)
        
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Save
    save_path = "final_visual_matrix.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Comparison matrix saved to {save_path}")

if __name__ == "__main__":
    main()
