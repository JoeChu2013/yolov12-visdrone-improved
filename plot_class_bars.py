import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. Load Data
    try:
        df = pd.read_csv("per_class_ap_comparison.csv")
    except:
        print("CSV file not found. Run compare_class_ap.py first.")
        return

    # Filter out "Improvement" row if present
    df = df[df['Model'] != 'Improvement (Final vs Base)']
    
    # Classes
    classes = [c for c in df.columns if c not in ['Model', 'mAP@0.5']]
    
    # Prepare Data for Plotting
    models = df['Model'].tolist()
    # Shorten model names for legend
    short_names = {
        "YOLOv12n (Baseline)": "Baseline",
        "YOLOv12-Improved (BiFPN+CA)": "Intermediate",
        "YOLOv12-P2-Improved (Final)": "Ours (P2)"
    }
    models = [short_names.get(m, m) for m in models]
    
    # Colors
    colors = ['#CCCCCC', '#87CEFA', '#FF6347'] # Grey, Light Blue, Tomato Red (Highlight Ours)
    
    # 2. Plotting
    plt.figure(figsize=(14, 6))
    
    x = np.arange(len(classes))
    width = 0.25
    
    # Plot bars for each model
    for i, model_name in enumerate(df['Model']):
        # Extract values, strip % and convert to float
        values = [float(df.iloc[i][cls].strip('%')) for cls in classes]
        
        # Calculate offset
        offset = (i - 1) * width
        
        plt.bar(x + offset, values, width, label=models[i], color=colors[i], edgecolor='white')
        
        # Add values on top of "Ours" bars for emphasis
        if i == 2: # Ours
            for j, v in enumerate(values):
                plt.text(x[j] + offset, v + 1, f"{v:.1f}", ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')

    # Formatting
    plt.ylabel('AP@0.5 (%)', fontsize=12)
    plt.title('Per-Class Performance Comparison', fontsize=14, pad=20)
    plt.xticks(x, [c.capitalize() for c in classes], rotation=0, fontsize=10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.ylim(0, 90) # Car is around 78%
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('per_class_bar_chart.png', dpi=300)
    print("Chart saved to per_class_bar_chart.png")

if __name__ == "__main__":
    main()
