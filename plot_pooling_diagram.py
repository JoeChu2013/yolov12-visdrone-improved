import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_pooling_comparison():
    # Configure font (SimSun for Chinese, Times New Roman style for math)
    try:
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
    except:
        pass

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.set_aspect('equal')
    ax.axis('off')

    # Data
    input_matrix = np.array([
        [1, 2, 3, 2],
        [3, 4, 2, 5],
        [5, 4, 4, 6],
        [8, 7, 4, 2]
    ])

    max_matrix = np.array([
        [4, 5],
        [8, 6]
    ])

    avg_matrix = np.array([
        [2.5, 3],
        [6, 4]
    ])

    # Colors for quadrants
    colors = {
        (0, 0): '#E87D32', # Top-Left Orange
        (0, 1): '#88C25B', # Top-Right Light Green
        (1, 0): '#249942', # Bottom-Left Dark Green
        (1, 1): '#19A9F0'  # Bottom-Right Blue
    }

    cell_size = 1.2
    x_input = 2.5
    y_input = 2.0

    x_max = 11.0
    y_max = 5.0

    x_avg = 11.0
    y_avg = 0.5

    def draw_grid(x_start, y_start, matrix, title):
        rows, cols = matrix.shape
        for r in range(rows):
            for c in range(cols):
                x = x_start + c * cell_size
                y = y_start + (rows - 1 - r) * cell_size

                # Determine quadrant for coloring
                quad_r = 0 if r < rows/2 else 1
                quad_c = 0 if c < cols/2 else 1
                facecolor = colors[(quad_r, quad_c)]

                rect = patches.Rectangle((x, y), cell_size, cell_size,
                                         linewidth=0.8, edgecolor='#c0392b', facecolor=facecolor)
                ax.add_patch(rect)

                # Format value
                val = matrix[r, c]
                val_str = f"{val:g}"
                ax.text(x + cell_size/2, y + cell_size/2, f"$\\mathbf{{{val_str}}}$",
                        fontsize=8.5, ha='center', va='center', color='white')

        # Draw outer border
        rect = patches.Rectangle((x_start, y_start), cols * cell_size, rows * cell_size,
                                 linewidth=1.5, edgecolor='#c0392b', facecolor='none')
        ax.add_patch(rect)

        if title:
            # Place title below the grid
            ax.text(x_start + (cols*cell_size)/2, y_start - 0.6, title,
                    fontsize=8, ha='center', va='center', fontweight='bold')

    # Draw grids
    draw_grid(x_input, y_input, input_matrix, "输入特征图")
    draw_grid(x_max, y_max, max_matrix, "最大池化")
    draw_grid(x_avg, y_avg, avg_matrix, "平均池化")

    # Draw connecting arrows
    x_in_rc = x_input + 4 * cell_size
    y_in_rc = y_input + 2 * cell_size

    x_max_lc = x_max
    y_max_lc = y_max + 1 * cell_size

    x_avg_lc = x_avg
    y_avg_lc = y_avg + 1 * cell_size

    # Arrow to MaxPool
    ax.annotate("", xy=(x_max_lc-0.2, y_max_lc), xytext=(x_in_rc+0.2, y_in_rc),
                arrowprops=dict(arrowstyle="-|>", color="#5C81B8", lw=1.2, mutation_scale=12))
    ax.text((x_in_rc + x_max_lc)/2 - 1, (y_in_rc + y_max_lc)/2 + 0.4, "最大化操作",
            fontsize=8, fontfamily='SimSun', fontweight='bold', rotation=22)

    # Arrow to AvgPool
    ax.annotate("", xy=(x_avg_lc-0.2, y_avg_lc), xytext=(x_in_rc+0.2, y_in_rc),
                arrowprops=dict(arrowstyle="-|>", color="#5C81B8", lw=1.2, mutation_scale=12))
    ax.text((x_in_rc + x_avg_lc)/2 - 1.2, (y_in_rc + y_avg_lc)/2 - 0.8, "平均化操作",
            fontsize=8, fontfamily='SimSun', fontweight='bold', rotation=-22)

    # Draw X and Y Axes
    ax_x_start = x_input - 0.8
    ax_y_start = y_input - 0.8
    
    # X Axis
    ax.annotate("", xy=(x_input + 4*cell_size + 0.8, ax_y_start), xytext=(ax_x_start, ax_y_start), 
                arrowprops=dict(arrowstyle="-|>", color="#5C81B8", lw=1.2, mutation_scale=12))
    ax.text(x_input + 4*cell_size + 0.8, ax_y_start - 0.3, "$\mathbf{x}$", 
            fontsize=8.5)
            
    # Y Axis
    ax.annotate("", xy=(ax_x_start, y_input + 4*cell_size + 0.8), xytext=(ax_x_start, ax_y_start), 
                arrowprops=dict(arrowstyle="-|>", color="#5C81B8", lw=1.2, mutation_scale=12))
    ax.text(ax_x_start - 0.4, y_input + 4*cell_size + 0.8, "$\mathbf{y}$", 
            fontsize=8.5)

    # Add explanatory annotation
    desc = r"池化窗口: $\mathbf{2 \times 2}$    步长: $\mathbf{2}$"
    ax.text((x_input + x_max + 2*cell_size)/2, y_max + 2*cell_size + 0.5, desc, 
            fontsize=8, ha='center', va='center',
            bbox=dict(facecolor='#f1f2f6', edgecolor='gray', boxstyle='round,pad=0.4', alpha=0.8))

    plt.xlim(0, 16)
    plt.ylim(-1.5, 9.5)
    
    out_path = 'pooling_comparison_diagram.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    draw_pooling_comparison()