import os
import random
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set up Chinese font for matplotlib
try:
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
except:
    pass

def normalize_heatmap(t):
    t = t.astype(np.float32)
    if t.max() > t.min():
        t = (t - t.min()) / (t.max() - t.min())
    return t

def to_heatmap(tensor):
    t = tensor.detach().cpu().numpy()
    if t.ndim == 4:
        t = t[0]
    t = np.mean(np.abs(t), axis=0)
    t = normalize_heatmap(t)
    t = np.uint8(255 * t)
    hm = cv2.applyColorMap(t, cv2.COLORMAP_JET)
    return hm

def capture_layers(model, img_path, pick=3):
    feats = []
    handles = []
    for m in model.model.model:
        if hasattr(m, 'register_forward_hook'):
            h = m.register_forward_hook(lambda mod, inp, out: feats.append(out) if isinstance(out, torch.Tensor) and out.ndim == 4 else None)
            handles.append(h)
    _ = model(img_path, verbose=False)
    for h in handles:
        h.remove()
    candidates = []
    for f in feats:
        if isinstance(f, torch.Tensor) and f.ndim == 4:
            candidates.append(f)
    if len(candidates) == 0:
        return []
    candidates = sorted(candidates, key=lambda x: x.shape[-1]*x.shape[-2], reverse=True)
    selected = []
    seen_sizes = set()
    for f in candidates:
        sz = (int(f.shape[-2]), int(f.shape[-1]))
        if sz in seen_sizes:
            continue
        selected.append(f)
        seen_sizes.add(sz)
        if len(selected) >= pick:
            break
    return selected

def render_feature_responses(model_path, img_path, out_path):
    # Setup fonts
    try:
        import platform
        if platform.system() == "Windows":
            font_path = "C:/Windows/Fonts/simsun.ttc"
        else:
            font_path = "/usr/share/fonts/truetype/simsun.ttf" # Adjust if needed
        my_font = fm.FontProperties(fname=font_path)
    except:
        my_font = None

    model = YOLO(model_path)
    img = cv2.imread(img_path)
    if img is None:
        return
        
    # Get intermediate features
    # We want to capture specific P3, P4, P5 layer outputs from backbone
    feats = []
    def get_features(m, i, o):
        if isinstance(o, torch.Tensor) and o.ndim == 4:
            feats.append(o)
            
    handles = []
    # Register hooks on C3k2/A2C2f modules in backbone which correspond to P3/P4/P5
    for i, m in enumerate(model.model.model):
        if i in [4, 7, 10]: # These are typical backbone output stages (P3, P4, P5) in YOLOv12
            h = m.register_forward_hook(get_features)
            handles.append(h)
            
    _ = model.predict(img_path, verbose=False)
    
    for h in handles:
        h.remove()
        
    if len(feats) < 3:
        # Fallback to the original method if specific layers weren't captured
        print("Fallback to auto-capture")
        feats = capture_layers(model, img_path, pick=3)
        
    if len(feats) == 0:
        return
        
    plt.figure(figsize=(7.5, 4.8))
    
    # 1. Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('原始输入图像', fontproperties=my_font, fontsize=9, pad=10)
    plt.axis('off')
    
    # Titles for the feature maps
    titles = ['浅层特征响应 (P3)', '中层特征响应 (P4)', '深层特征响应 (P5)']
    
    # 2,3,4. Feature Maps
    for i in range(min(3, len(feats))):
        f = feats[i]
        
        # Convert tensor to heatmap
        t = f.detach().cpu().numpy()
        if t.ndim == 4:
            t = t[0]
        # Mean across channels
        t = np.mean(np.abs(t), axis=0)
        t = normalize_heatmap(t)
        
        # Resize to original image size for better visualization
        t_resized = cv2.resize(t, (img.shape[1], img.shape[0]))
        
        # Apply colormap
        t_uint8 = np.uint8(255 * t_resized)
        hm = cv2.applyColorMap(t_uint8, cv2.COLORMAP_JET)
        hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.3, hm, 0.7, 0)
        
        plt.subplot(2, 2, i + 2)
        plt.imshow(overlay)
        plt.title(titles[i] if i < len(titles) else f'层级 {i+1}', fontproperties=my_font, fontsize=9, pad=10)
        plt.axis('off')
        
    plt.suptitle("基线模型 YOLOv12n 多尺度特征响应可视化 (小目标随下采样衰减)", 
                 fontproperties=my_font, fontsize=11, fontweight='bold', y=0.98)
    plt.subplots_adjust(wspace=0.05, hspace=0.15, top=0.88)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    model_path = 'runs/detect/ablation_baseline_ciou/weights/best.pt'
    val_dir = 'datasets/VisDrone/VisDrone2019-DET-val/images'
    if not os.path.exists(model_path) or not os.path.exists(val_dir):
        print('paths not found')
        return
    imgs = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.jpg')]
    if len(imgs) == 0:
        print('no images')
        return
    random.seed(0)
    sample = random.choice(imgs)
    render_feature_responses(model_path, sample, 'feature_responses_baseline.png')
    print('Saved feature_responses_baseline.png')

if __name__ == '__main__':
    main()

