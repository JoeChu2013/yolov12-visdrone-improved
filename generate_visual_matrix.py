import os
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import random
from matplotlib import font_manager

def get_dense_crop_region(model, img_path, crop_w=800, crop_h=450):
    """
    Finds the densest bounding box region to create a zoomed-in crop.
    This solves the issue of small objects being invisible in the full matrix.
    """
    res = model.predict(img_path, conf=0.15, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    if len(boxes) == 0:
        cx, cy = w // 2, h // 2
    else:
        # Use the median of box centers to avoid outliers skewing the crop
        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
        cx = int(np.median(centers_x))
        cy = int(np.median(centers_y))
        
    x1 = max(0, cx - crop_w // 2)
    y1 = max(0, cy - crop_h // 2)
    
    if x1 + crop_w > w: x1 = w - crop_w
    if y1 + crop_h > h: y1 = h - crop_h
    
    return max(0, x1), max(0, y1), min(w, crop_w), min(h, crop_h)

def main():
    # 1. Models to Compare (Rows)
    models_config = {
        "YOLOv8n": "runs/detect/YOLOv8n/weights/best.pt",
        "YOLOv9t": "runs/detect/YOLOv9t/weights/best.pt",
        "YOLOv10n": "runs/detect/YOLOv10n/weights/best.pt",
        "YOLOv11n": "runs/detect/YOLOv11n/weights/best.pt",
        "YOLOv12n (基线模型)": "runs/detect/ablation_baseline_ciou/weights/best.pt",
        "YOLOv12n-ACA (本文算法)": "runs/detect/YOLOv12-Tiny-P2P3-Focused/weights/best.pt"
    }

    # 2. Select 4 Images for different scenarios (Columns)
    val_img_dir = 'datasets/VisDrone/VisDrone2019-DET-val/images'
    if not os.path.exists(val_img_dir):
        print("Validation directory not found.")
        return

    all_images = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
    if len(all_images) < 4:
        print("Not enough images.")
        return
        
    random.seed(42)
    selected_images = random.sample(all_images, 4)
    scenarios = ["场景 A", "场景 B", "场景 C", "场景 D"]

    # Load Models
    loaded_models = {}
    for name, path in models_config.items():
        if os.path.exists(path) or path.endswith('.pt'):
            try:
                loaded_models[name] = YOLO(path)
            except:
                print(f"Failed to load {name}")

    # Configure Font for Chinese support
    try:
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
    except:
        pass

    # Instead of separate single-column images, we will generate 4 images in total.
    # Each image corresponds to one scenario.
    # For each scenario, the 7 rows will be wrapped into a grid of 2 rows x 4 columns.
    # Actually, we have 7 items (Original + 6 Models).
    # We can layout them as:
    # Row 1: Original, YOLOv8n, YOLOv9t, YOLOv10n
    # Row 2: YOLOv11n, Base, Ours, (Empty or Legend)
    # Let's just make it a 2x4 grid per scenario image.
    
    models_list = list(loaded_models.items())
    
    for col_idx, img_path in enumerate(selected_images):
        
        # Original VisDrone images are typically 16:9 or 4:3.
        # A 2x4 grid of wide images requires a slightly taller figure than square crops.
        fig = plt.figure(figsize=(7.5, 3.0)) 
        plt.subplots_adjust(wspace=0.03, hspace=0.2, top=0.85, bottom=0.02, left=0.02, right=0.98)
        
        # Super title for the scenario
        plt.suptitle(scenarios[col_idx], fontfamily='SimSun', fontsize=12, fontweight='bold', y=0.98)
        
        # 1. Original Image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax = plt.subplot(2, 4, 1)
        plt.imshow(img)
        plt.title("原始图像", fontdict={'family': 'SimSun', 'size': 10, 'weight': 'bold'}, pad=6)
        plt.xticks([]); plt.yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
            
        # 2 to 7. Model Predictions
        for i in range(6):
            model_name, model = models_list[i]
            results = model.predict(img_path, conf=0.25, verbose=False)[0]
            # Make the line width slightly thicker (2) so boxes are visible in full-scale images
            res_plotted = results.plot(line_width=2, font_size=10) 
            
            res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            ax = plt.subplot(2, 4, i + 2)
            plt.imshow(res_plotted)
            
            plt.title(model_name, fontdict={'family': 'SimSun', 'size': 10, 'weight': 'bold'}, pad=6)
            
            plt.xticks([]); plt.yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            
        save_path = f"final_visual_matrix_scenario_{chr(65+col_idx)}.png"
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved {save_path}")
        plt.close(fig)

if __name__ == "__main__":
    main()
