import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_convolution_process():
    # Configure font (SimSun for Chinese, Times New Roman style for math)
    try:
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
    except:
        pass

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.set_aspect('equal')
    ax.axis('off')

    # Data from the user's reference image
    input_matrix = np.array([
        [1, 1, 1, 1, 1],
        [-1, 0, -3, 0, 1],
        [2, 1, 1, -1, 0],
        [0, -1, 1, 2, 1],
        [1, 2, 1, 1, 1]
    ])

    kernel_matrix = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, -1]
    ])

    output_matrix = np.array([
        [0, -2, -1],
        [2, 2, 4],
        [-1, 0, 0]
    ])

    # Drawing parameters
    cell_size = 1.0
    gap_x = 2.0
    
    x_input = 0
    y_input = 0
    
    x_kernel = x_input + 5 * cell_size + gap_x
    y_kernel = y_input + 1 * cell_size
    
    x_output = x_kernel + 3 * cell_size + gap_x
    y_output = y_input + 1 * cell_size

    # Function to draw a matrix grid
    def draw_grid(x_start, y_start, matrix, title=None, highlight_rect=None, is_input=False):
        rows, cols = matrix.shape
        for r in range(rows):
            for c in range(cols):
                x = x_start + c * cell_size
                y = y_start + (rows - 1 - r) * cell_size # Draw from top to bottom
                
                # Check if this cell is part of the highlight
                is_highlighted = False
                if highlight_rect is not None:
                    hr_r, hr_c, hr_rows, hr_cols = highlight_rect
                    if hr_r <= r < hr_r + hr_rows and hr_c <= c < hr_c + hr_cols:
                        is_highlighted = True
                
                # Cell background and border
                facecolor = '#f5d6d6' if is_highlighted else 'white'
                linewidth = 0.8
                edgecolor = '#c0392b'
                
                rect = patches.Rectangle((x, y), cell_size, cell_size, 
                                         linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, alpha=1.0)
                ax.add_patch(rect)
                
                # Cell text (Main value)
                val = matrix[r, c]
                ax.text(x + cell_size/2, y + cell_size/2, str(val), 
                        fontsize=8.5, ha='center', va='center', fontfamily='serif')
                
                # Small multiplier text for the highlighted input region
                if is_input and is_highlighted:
                    # Map input (r, c) to kernel (k_r, k_c)
                    k_r = r - highlight_rect[0]
                    k_c = c - highlight_rect[1]
                    k_val = kernel_matrix[k_r, k_c]
                    # Format multiplier
                    mult_text = f"$\\times {k_val}$" if k_val >= 0 else f"$\\times ({k_val})$"
                    ax.text(x + cell_size - 0.05, y + 0.05, mult_text, 
                            fontsize=5.5, ha='right', va='bottom', color='blue', fontfamily='serif')

        if title:
            ax.text(x_start + (cols*cell_size)/2, y_start - 0.6, title, 
                    fontsize=8, ha='center', va='center', fontweight='bold')
                    
        # If input, draw a thick border around the highlighted region
        if is_input and highlight_rect is not None:
            hr_r, hr_c, hr_rows, hr_cols = highlight_rect
            hx = x_start + hr_c * cell_size
            hy = y_start + (rows - hr_r - hr_rows) * cell_size
            rect = patches.Rectangle((hx, hy), hr_cols * cell_size, hr_rows * cell_size, 
                                     linewidth=2, edgecolor='#e74c3c', facecolor='none', alpha=0.7)
            ax.add_patch(rect)
            
        # Draw a thick border around the highlighted output cell
        if not is_input and highlight_rect is not None:
            hr_r, hr_c, hr_rows, hr_cols = highlight_rect
            hx = x_start + hr_c * cell_size
            hy = y_start + (rows - hr_r - hr_rows) * cell_size
            rect = patches.Rectangle((hx, hy), hr_cols * cell_size, hr_rows * cell_size, 
                                     linewidth=2, edgecolor='#e74c3c', facecolor='none', alpha=0.7)
            ax.add_patch(rect)

    # 1. Draw Input Matrix (Highlighting the top-right 3x3 region for calculation of -1)
    # The output -1 is at index (0, 2), which corresponds to input region (0, 2) to (2, 4)
    draw_grid(x_input, y_input, input_matrix, title="输入特征图 ($\mathrm{Input}$)", highlight_rect=(0, 2, 3, 3), is_input=True)

    # Draw Convolution Symbol
    ax.text(x_kernel - gap_x/2, y_kernel + 1.5 * cell_size, r'$\otimes$', 
            fontsize=14, ha='center', va='center')

    # 2. Draw Kernel Matrix
    draw_grid(x_kernel, y_kernel, kernel_matrix, title="卷积核 ($\mathrm{Kernel}$)\n[权值共享]")

    # Draw Equals Symbol
    ax.text(x_output - gap_x/2, y_kernel + 1.5 * cell_size, r'$=$', 
            fontsize=14, ha='center', va='center')

    # 3. Draw Output Matrix (Highlighting the resulting cell)
    draw_grid(x_output, y_output, output_matrix, title="输出特征图 ($\mathrm{Output}$)", highlight_rect=(0, 2, 1, 1))

    # Add Calculation Formula Annotation at the bottom
    formula = r"计算过程: $(1 \times 1) + (1 \times 0) + (1 \times 0) + (-3 \times 0) + (0 \times 0) + (1 \times 0) + (1 \times 0) + (-1 \times 0) + (0 \times -1) = -1$"
    ax.text(x_kernel + 1.5, y_input - 1.5, formula, fontsize=8, ha='center', va='center', 
            bbox=dict(facecolor='#f1f2f6', edgecolor='gray', boxstyle='round,pad=0.4', alpha=0.8))

    plt.xlim(x_input - 1, x_output + 3 * cell_size + 1)
    plt.ylim(y_input - 2.5, y_input + 5 * cell_size + 1)
    
    out_path = 'convolution_process_diagram.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    draw_convolution_process()