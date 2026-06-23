import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib.font_manager as fm
import math

def draw_ciou_vs_siou_trajectory():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    
    # Configure Chinese font (SimSun) and English font (Times New Roman)
    try:
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix' # STIX is visually very similar to Times New Roman
    except:
        pass
        
    # Common parameters for drawing
    gt_box = (6, 6, 4, 3) # cx, cy, w, h
    pred_box_start = (2, 2, 4, 3) # cx, cy, w, h
    
    colors = {
        'gt': '#2ca02c',      # Green
        'pred': '#d62728',    # Red
        'traj': '#1f77b4',    # Blue
        'angle': '#ff7f0e'    # Orange
    }

    def draw_box(ax, box, color, label, is_dashed=False):
        cx, cy, w, h = box
        x, y = cx - w/2, cy - h/2
        ls = '--' if is_dashed else '-'
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.1, linestyle=ls)
        ax.add_patch(rect)
        # Draw center point
        ax.plot(cx, cy, marker='o', markersize=4, color=color)
        ax.text(cx, cy + h/2 + 0.3, label, color=color, fontsize=10, fontweight='bold', ha='center', fontfamily='SimSun')

    # --- Plot 1: CIoU Trajectory (Wandering) ---
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Draw GT and Start Box
    draw_box(ax1, gt_box, colors['gt'], '真实框 ($\mathrm{GT}$)')
    draw_box(ax1, pred_box_start, colors['pred'], '初始预测框', is_dashed=True)
    
    # CIoU trajectory points (Wandering/Oscillating approach)
    # CIoU only minimizes distance and aspect ratio, doesn't penalize angle
    ciou_pts_x = [2.0, 3.5, 4.2, 4.8, 5.2, 6.0]
    ciou_pts_y = [2.0, 2.5, 4.0, 3.5, 5.0, 6.0]
    
    # Draw trajectory line
    ax1.plot(ciou_pts_x, ciou_pts_y, color=colors['traj'], linewidth=1.5, linestyle='-', marker='s', markersize=3, label='回归轨迹')
    
    # Add arrows to trajectory
    for i in range(len(ciou_pts_x)-1):
        ax1.annotate('', xy=(ciou_pts_x[i+1], ciou_pts_y[i+1]), xytext=(ciou_pts_x[i], ciou_pts_y[i]),
                     arrowprops=dict(arrowstyle="->", color=colors['traj'], lw=1.2))
                     
    # Draw direct distance line to show angle deviation
    ax1.plot([2, 6], [2, 6], color='gray', linestyle=':', linewidth=1.2)
    
    # Very simple titles for academic papers
    ax1.set_title('(a) $\mathrm{CIoU}$', fontsize=12, pad=10, fontfamily='serif', fontweight='bold')
    ax1.legend(loc='upper left', prop={'family': 'SimSun', 'size': 9})
    ax1.tick_params(axis='both', which='major', labelsize=8)

    # --- Plot 2: SIoU Trajectory (Angle-first) ---
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # Draw GT and Start Box
    draw_box(ax2, gt_box, colors['gt'], '真实框 ($\mathrm{GT}$)')
    draw_box(ax2, pred_box_start, colors['pred'], '初始预测框', is_dashed=True)
    
    # SIoU trajectory points (Angle-first approach)
    # SIoU penalizes angle first, so it moves horizontally or vertically to align axes, then approaches
    siou_pts_x = [2.0, 4.5, 5.8, 6.0, 6.0]
    siou_pts_y = [2.0, 2.8, 3.0, 4.5, 6.0]
    
    # Draw trajectory line
    ax2.plot(siou_pts_x, siou_pts_y, color=colors['traj'], linewidth=1.5, linestyle='-', marker='s', markersize=3, label='回归轨迹')
    
    # Add arrows to trajectory
    for i in range(len(siou_pts_x)-1):
        ax2.annotate('', xy=(siou_pts_x[i+1], siou_pts_y[i+1]), xytext=(siou_pts_x[i], siou_pts_y[i]),
                     arrowprops=dict(arrowstyle="->", color=colors['traj'], lw=1.2))
                     
    # Draw axis alignment lines (Angle Cost representation)
    ax2.plot([2, 6], [2, 2], color=colors['angle'], linestyle='--', linewidth=1.5, label='$\mathrm{X}$ 轴 / $\mathrm{Y}$ 轴 对齐约束')
    ax2.plot([6, 6], [2, 6], color=colors['angle'], linestyle='--', linewidth=1.5)
    
    # Add Angle Cost annotation
    ax2.annotate('先趋向轴向对齐\n(最小化角度代价 $\Lambda$)', xy=(4, 2.2), xytext=(4, 0.8),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.0, headwidth=4),
                 fontsize=9, ha='center', color=colors['angle'], fontfamily='SimSun')
                 
    ax2.annotate('再进行距离逼近\n(最小化距离代价 $\Delta$)', xy=(6.2, 4.5), xytext=(8.5, 4.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.0, headwidth=4),
                 fontsize=9, ha='center', color=colors['traj'], fontfamily='SimSun')

    ax2.set_title('(b) $\mathrm{SIoU}$', fontsize=12, pad=10, fontfamily='serif', fontweight='bold')
    ax2.legend(loc='upper left', prop={'family': 'SimSun', 'size': 9})
    ax2.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    out_path = 'siou_vs_ciou_trajectory.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    draw_ciou_vs_siou_trajectory()