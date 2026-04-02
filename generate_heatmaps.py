import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import random

def get_feature_heatmap(model_path, img_path):
    model = YOLO(model_path)
    
    # List to store feature maps
    features = []
    
    # Define a hook function to capture the input to the Detect layer
    def hook(module, input):
        # input[0] is typically a list of tensors from different scales
        if isinstance(input[0], list):
            for tensor in input[0]:
                features.append(tensor.detach().cpu())
        elif isinstance(input[0], torch.Tensor):
            features.append(input[0].detach().cpu())
            
    # Register the pre-hook on the last layer (Detect)
    detect_layer = model.model.model[-1]
    handle = detect_layer.register_forward_pre_hook(hook)
    
    # Read image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run prediction to trigger the hook
    _ = model.predict(img_path, imgsz=640, verbose=False)
    
    # Remove hook
    handle.remove()
    
    # Process features into a single heatmap
    heatmap_combined = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    
    for feat in features:
        # feat shape: [1, C, H, W]
        # Take the maximum activation across the channel dimension
        act, _ = torch.max(feat, dim=1)
        act = act.squeeze().numpy()
        
        # ReLU
        act = np.maximum(act, 0)
        
        # Normalize
        if np.max(act) > 0:
            act /= np.max(act)
            
        # Resize to original image size
        act_resized = cv2.resize(act, (img.shape[1], img.shape[0]))
        
        # Combine by taking the maximum activation across different scales
        heatmap_combined = np.maximum(heatmap_combined, act_resized)
        
    # Final normalization
    if np.max(heatmap_combined) > 0:
        heatmap_combined /= np.max(heatmap_combined)
        
    # Convert to colormap (JET)
    heatmap_img = np.uint8(255 * heatmap_combined)
    colormap = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
    
    # Superimpose on original image
    overlay = cv2.addWeighted(img_rgb, 0.6, colormap, 0.4, 0)
    return overlay

def main():
    base_model = "runs/detect/ablation_baseline_ciou/weights/best.pt"
    tiny_model = "runs/detect/YOLOv12-Tiny-P2P3-Focused/weights/best.pt"
    
    val_img_dir = 'datasets/VisDrone/VisDrone2019-DET-val/images'
    if not os.path.exists(val_img_dir):
        print("Validation directory not found.")
        return
        
    all_images = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
    random.seed(45) # Different seed to pick interesting images
    selected_images = random.sample(all_images, 2)
    
    plt.figure(figsize=(18, 10))
    
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
        
    for i, img_p in enumerate(selected_images):
        img_orig = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        print(f"Processing image {i+1}...")
        
        hm_base = get_feature_heatmap(base_model, img_p)
        hm_tiny = get_feature_heatmap(tiny_model, img_p)
        
        plt.subplot(2, 3, i*3 + 1)
        plt.imshow(img_orig)
        plt.title("Original Image", fontsize=14)
        plt.axis('off')
        
        plt.subplot(2, 3, i*3 + 2)
        plt.imshow(hm_base)
        plt.title("YOLOv12n (Baseline) Feature Activation", fontsize=14)
        plt.axis('off')
        
        plt.subplot(2, 3, i*3 + 3)
        plt.imshow(hm_tiny)
        plt.title("YOLOv12n-UAV (Ours) Feature Activation", fontsize=14)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig("heatmap_comparison.png", dpi=300, bbox_inches='tight')
    print("Heatmap comparison saved to heatmap_comparison.png")

if __name__ == "__main__":
    main()
