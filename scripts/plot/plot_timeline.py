import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def draw_yolo_timeline():
    # Setup fonts
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.family'] = 'serif'
    
    events = [
        {"date": "Jul 18, 2021", "version": "YOLOX", "pos": "bottom", "color": "#FF9933"},
        {"date": "Jun, 2022", "version": "YOLOv6", "pos": "top", "color": "#FFCC66"},
        {"date": "Jul 6, 2022", "version": "YOLOv7", "pos": "bottom", "color": "#FF9999"},
        {"date": "Jan 10, 2023", "version": "YOLOv8", "pos": "top", "color": "#FF99CC"},
        {"date": "Jan 13, 2023", "version": "YOLOv6 3.0", "pos": "bottom", "color": "#CC99FF"},
        {"date": "May 2, 2023", "version": "YOLO-NAS", "pos": "top", "color": "#9999FF"},
        {"date": "Jan 30, 2024", "version": "YOLO-World", "pos": "bottom", "color": "#66CCCC"},
        {"date": "Feb 21, 2024", "version": "YOLOv9", "pos": "top", "color": "#99FF99"},
        {"date": "May 23, 2024", "version": "YOLOv10", "pos": "bottom", "color": "#CCCC00"},
        {"date": "Sep 30, 2024", "version": "YOLOv11", "pos": "top", "color": "#FF9933"},
        {"date": "Feb 18, 2025", "version": "YOLOv12", "pos": "bottom", "color": "#CC99FF"}
    ]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.axis('off')
    
    # Set limits
    ax.set_xlim(0, len(events) + 1)
    ax.set_ylim(-4.5, 4.5)

    # Draw central timeline (dashed)
    ax.plot([0.5, len(events) + 0.5], [0, 0], color='black', linestyle='--', linewidth=1.5, zorder=1)

    # Plot events
    for i, event in enumerate(events):
        x = i + 1
        y_direction = 1 if event['pos'] == 'top' else -1
        
        # Line length
        line_len = 2.2 * y_direction
        
        # Draw dashed vertical line
        ax.plot([x, x], [0, line_len], color='black', linestyle='--', linewidth=1.0, zorder=2)
        
        # Draw dot on timeline
        ax.plot(x, 0, marker='o', markersize=6, color=event['color'], zorder=3)
        
        # Draw date pill
        date_y = 0.8 * y_direction
        bbox_props = dict(boxstyle="round,pad=0.3,rounding_size=0.4", fc=event['color'], ec="none", alpha=0.9)
        ax.text(x, date_y, event['date'], ha='center', va='center', fontsize=8, 
                fontweight='bold', bbox=bbox_props, zorder=4)
        
        # Draw Version Text
        text_y = 2.4 * y_direction
        va_align = 'bottom' if event['pos'] == 'top' else 'top'
        ax.text(x, text_y, event['version'], ha='center', va=va_align, 
                fontsize=9, fontweight='bold', zorder=5)

    plt.tight_layout()
    out_path = 'yolo_timeline_clean.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    draw_yolo_timeline()