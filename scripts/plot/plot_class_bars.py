import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Setup Font for Academic Papers
    try:
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
    except:
        pass

    # 1. Load Data
    try:
        df = pd.read_csv("per_class_ap_comparison.csv")
    except:
        print("CSV file not found. Run compare_class_ap.py first.")
        return

    # Filter out "Improvement" row if present
    df = df[~df['Model'].str.contains('Improvement')]
    
    # Classes
    classes = [c for c in df.columns if c not in ['Model', 'mAP@0.5']]
    
    # Chinese Class Names Mapping
    class_name_map = {
        'pedestrian': '行人',
        'people': '人群',
        'bicycle': '自行车',
        'car': '轿车',
        'van': '厢式货车',
        'truck': '卡车',
        'tricycle': '三轮车',
        'awning-tricycle': '遮阳三轮车',
        'bus': '公交车',
        'motor': '摩托车'
    }
    
    # Prepare Data for Plotting
    models = df['Model'].tolist()
    # Shorten model names for legend
    short_names = {
        "YOLOv12n (Baseline)": "YOLOv12n (基线模型)",
        "YOLOv12n-UAV (Ours)": "YOLOv12n-ACA (本文算法)",
        "YOLOv12n-ACA (Ours)": "YOLOv12n-ACA (本文算法)"
    }
    models = [short_names.get(m, m) for m in models]
    
    # Colors
    colors = ['#CCCCCC', '#FF6347'] # Grey, Tomato Red (Highlight Ours)
    
    # 2. Plotting
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    
    x = np.arange(len(classes))
    width = 0.35 # Slightly wider since there are only 2 bars now
    
    prop_legend = {'family': 'SimSun', 'size': 9}
    if 'stix' in plt.rcParams['mathtext.fontset']:
        # To mix English/Math and Chinese, we rely on SimSun fallback and stix.
        pass

    # Plot bars for each model
    for i, model_name in enumerate(df['Model']):
        # Extract values, strip % and convert to float
        values = [float(df.iloc[i][cls].strip('%')) for cls in classes]
        
        # Calculate offset
        offset = (i - 0.5) * width
        
        ax.bar(x + offset, values, width, label=models[i], color=colors[i], edgecolor='white')
        
        # Add values on top of "Ours" bars for emphasis
        if i == 1: # Ours
            for j, v in enumerate(values):
                ax.text(x[j] + offset, v + 1, f"{v:.1f}", ha='center', va='bottom', 
                        fontdict={'family': 'serif', 'size': 8, 'weight': 'bold'})

    # Formatting
    ax.set_ylabel('AP@0.5 (%)', fontdict={'family': 'serif', 'size': 10})
    ax.set_title('各类别检测精度对比 (AP@0.5)', fontdict={'family': 'SimSun', 'size': 11, 'weight': 'bold'}, pad=15)
    
    chinese_classes = [class_name_map.get(c.lower(), c) for c in classes]
    ax.set_xticks(x)
    ax.set_xticklabels(chinese_classes, fontdict={'family': 'SimSun', 'size': 9}, rotation=30, ha='right')
    
    ax.legend(prop=prop_legend, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 90) # Car is around 78%
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('per_class_bar_chart.png', dpi=300, bbox_inches='tight')
    print("Chart saved to per_class_bar_chart.png")

if __name__ == "__main__":
    main()
