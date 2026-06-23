import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Setup Font for Academic Papers
    try:
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
    except:
        pass
        
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
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    
    prop_legend = {'family': 'serif', 'size': 9}
    font_title = {'family': 'SimSun', 'size': 11, 'weight': 'bold'}
    font_label = {'family': 'SimSun', 'size': 10}

    # 1. mAP@0.5 Convergence (Speed of Accuracy)
    ax = axes[0, 0]
    ax.plot(df_c['epoch'], df_c['metrics/mAP50(B)'], label=r'$\mathrm{CIoU}$', color='blue', linestyle='--', linewidth=1.5)
    ax.plot(df_s['epoch'], df_s['metrics/mAP50(B)'], label=r'$\mathrm{SIoU}$', color='red', linewidth=1.5)
    ax.set_title(r'(a) $\mathrm{mAP@0.5}$ 收敛曲线', fontdict=font_title, pad=10)
    ax.set_xlabel('训练轮次', fontdict=font_label)
    ax.set_ylabel(r'$\mathrm{mAP@0.5}$', fontdict={'family': 'serif', 'size': 10})
    ax.legend(prop=prop_legend)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=9)
    
    # 2. Box Loss Convergence (Speed of Regression Learning)
    ax = axes[0, 1]
    ax.plot(df_c['epoch'], df_c['val/box_loss'], label=r'$\mathrm{CIoU}$', color='blue', linestyle='--', linewidth=1.5)
    ax.plot(df_s['epoch'], df_s['val/box_loss'], label=r'$\mathrm{SIoU}$', color='red', linewidth=1.5)
    ax.set_title('(b) 验证集边界框损失收敛曲线', fontdict=font_title, pad=10)
    ax.set_xlabel('训练轮次', fontdict=font_label)
    ax.set_ylabel('边界框损失', fontdict=font_label)
    ax.legend(prop=prop_legend)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=9)
    
    # 3. Precision Convergence
    ax = axes[1, 0]
    ax.plot(df_c['epoch'], df_c['metrics/precision(B)'], label=r'$\mathrm{CIoU}$', color='blue', linestyle='--', linewidth=1.5)
    ax.plot(df_s['epoch'], df_s['metrics/precision(B)'], label=r'$\mathrm{SIoU}$', color='red', linewidth=1.5)
    ax.set_title('(c) 精确率收敛曲线', fontdict=font_title, pad=10)
    ax.set_xlabel('训练轮次', fontdict=font_label)
    ax.set_ylabel('精确率', fontdict=font_label)
    ax.legend(prop=prop_legend)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=9)

    # 4. Recall Convergence
    ax = axes[1, 1]
    ax.plot(df_c['epoch'], df_c['metrics/recall(B)'], label=r'$\mathrm{CIoU}$', color='blue', linestyle='--', linewidth=1.5)
    ax.plot(df_s['epoch'], df_s['metrics/recall(B)'], label=r'$\mathrm{SIoU}$', color='red', linewidth=1.5)
    ax.set_title('(d) 召回率收敛曲线', fontdict=font_title, pad=10)
    ax.set_xlabel('训练轮次', fontdict=font_label)
    ax.set_ylabel('召回率', fontdict=font_label)
    ax.legend(prop=prop_legend)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout()
    plt.savefig('siou_vs_ciou_convergence.png', dpi=300, bbox_inches='tight')
    print("Detailed convergence comparison saved to siou_vs_ciou_convergence.png")

if __name__ == "__main__":
    main()
