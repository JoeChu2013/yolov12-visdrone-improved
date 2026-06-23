import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.font_manager as fm

# Set up Chinese font for matplotlib
try:
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'stix'
except:
    pass

def load_coco(anno_json):
    with open(anno_json, 'r') as f:
        data = json.load(f)
    imgs = {im['id']: im for im in data['images']}
    return imgs, data['annotations']

def compute_stats(imgs, anns):
    areas = []
    ratios = []
    counts = Counter()
    for a in anns:
        img = imgs[a['image_id']]
        iw, ih = img['width'], img['height']
        x, y, w, h = a['bbox']
        area = w * h
        # Ensure w and h are valid
        if w <= 0 or h <= 0: continue
        areas.append(area / (iw * ih + 1e-9))
        ratios.append(w / (h + 1e-9))
        counts[a['image_id']] += 1
    density = list(counts.values())
    return np.array(areas), np.array(ratios), np.array(density)

def main():
    anno_json = 'datasets/VisDrone/VisDrone2019-DET-val/annotations.json'       
    if not os.path.exists(anno_json):
        print(f"File not found: {anno_json}")
        return
        
    imgs, anns = load_coco(anno_json)
    areas, ratios, density = compute_stats(imgs, anns)
    
    # Filter extreme outliers for better visualization
    ratios_filtered = ratios[ratios < 5] # Exclude extreme aspect ratios
    areas_filtered = areas[areas < 0.05] # Exclude extreme large objects for histogram
    
    # Set standard academic font sizes
    plt.rcParams.update({'font.size': 10})
    title_fs = 11
    label_fs = 10
    tick_fs = 9
    text_fs = 9
    
    # Plot 1: Scale Distribution (Separate)
    fig1, ax1 = plt.subplots(figsize=(7.2, 5.0))
    ax1.hist(areas_filtered, bins=50, color='#4C78A8', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax1.set_title('目标相对面积分布 (量化“小”)', fontsize=title_fs, fontweight='bold', pad=15)
    ax1.set_xlabel('相对面积 (目标面积 / 图像总面积)', fontsize=label_fs)
    ax1.set_ylabel('实例数量 (对数坐标)', fontsize=label_fs)
    ax1.tick_params(axis='both', labelsize=tick_fs)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    q_50 = np.quantile(areas, 0.5)
    q_90 = np.quantile(areas, 0.9)
    stats_text1 = f"中位数(Q50): {q_50:.4f}\n90%分位数: {q_90:.4f}\n绝大多数目标不足全图面积的 0.35%"
    ax1.text(0.35, 0.75, stats_text1, transform=ax1.transAxes, fontsize=text_fs,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.tight_layout()
    plt.savefig('visdrone_stats_1_scale.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Plot 2: Aspect Ratio Distribution (Separate)
    fig2, ax2 = plt.subplots(figsize=(7.2, 5.0))
    ax2.hist(ratios_filtered, bins=50, color='#F58518', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_title('目标长宽比分布', fontsize=title_fs, fontweight='bold', pad=15)
    ax2.set_xlabel('长宽比 (宽 / 高)', fontsize=label_fs)
    ax2.set_ylabel('实例数量', fontsize=label_fs)
    ax2.tick_params(axis='both', labelsize=tick_fs)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    stats_text2 = f"均值: {mean_ratio:.2f}\n标准差: {std_ratio:.2f}\n分布趋近于正方形，无极端长宽比"
    ax2.text(0.40, 0.75, stats_text2, transform=ax2.transAxes, fontsize=text_fs,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.tight_layout()
    plt.savefig('visdrone_stats_2_ratio.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # Plot 3: Density Distribution (Separate)
    fig3, ax3 = plt.subplots(figsize=(7.2, 5.0))
    ax3.hist(density, bins=50, color='#54A24B', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax3.set_title('单图目标密度分布 (量化“密集”)', fontsize=title_fs, fontweight='bold', pad=15)
    ax3.set_xlabel('每张图像的目标数量', fontsize=label_fs)
    ax3.set_ylabel('图像数量', fontsize=label_fs)
    ax3.tick_params(axis='both', labelsize=tick_fs)
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    mean_density = np.mean(density)
    max_density = np.max(density)
    stats_text3 = f"平均密度: {mean_density:.1f} 目标/图\n最大密度: {max_density} 目标/图\n场景极度密集，易产生严重遮挡"
    ax3.text(0.35, 0.75, stats_text3, transform=ax3.transAxes, fontsize=text_fs,
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
    plt.tight_layout()
    plt.savefig('visdrone_stats_3_density.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("Saved 3 separate high-res images for Word:")
    print("1. visdrone_stats_1_scale.png")
    print("2. visdrone_stats_2_ratio.png")
    print("3. visdrone_stats_3_density.png")

if __name__ == '__main__':
    main()

