import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import os

def draw_ca_integration_v2():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_aspect('equal')
    ax.axis('off')

    # --- Configuration ---
    x_bb = 2  # Backbone center x
    x_neck_td = 8 # Neck Top-Down column
    x_neck_bu = 10 # Neck Bottom-Up column
    x_head = 12 # Head column

    y_input = 0
    y_c1 = 1.5
    y_c2 = 3.0 # P2
    y_c3 = 4.5 # P3
    y_c4 = 6.0 # P4
    y_c5 = 7.5 # P5

    h_block = 0.8
    w_input = 6
    w_c1 = 5
    w_c2 = 4
    w_c3 = 3
    w_c4 = 2
    w_c5 = 1.5
    w_ca = 0.8

    c_bb = '#B0B0B0'
    c_ca = '#FFD700'
    c_neck_td = '#87CEFA'
    c_neck_bu = '#90EE90'
    c_head = '#FF69B4'

    # Configure font
    try:
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
    except:
        pass

    def draw_rect(x, y, w, h, color, label, fontsize=10):
        rect = patches.Rectangle((x - w/2, y - h/2), w, h,
                                 linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold', fontfamily='SimSun')
        return rect

    def draw_circle(x, y, r, color, label=None, fontsize=12):
        circle = patches.Circle((x, y), r, linewidth=1, edgecolor='black', facecolor=color, zorder=10)
        ax.add_patch(circle)
        if label:
            ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontfamily='SimSun')
        return circle

    def draw_arrow(x1, y1, x2, y2, color='black', style='->'):
        ax.add_patch(patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color, lw=1.5, mutation_scale=15))

    # --- 1. Backbone Section ---
    draw_rect(x_bb, y_input, w_input, h_block, '#808080', '输入图像\n(Input)', fontsize=12)
    draw_rect(x_bb, y_c1, w_c1, h_block, c_bb, 'C1\n(P1)', fontsize=12)
    draw_rect(x_bb, y_c2, w_c2, h_block, c_bb, 'C2\n(P2)', fontsize=12)
    
    # C3 + CA
    r_c3 = draw_rect(x_bb, y_c3, w_c3, h_block, c_bb, 'C3\n(P3)', fontsize=12)
    # CA Block
    r_ca3 = draw_rect(x_bb + w_c3/2 + w_ca/2 + 0.1, y_c3, w_ca, h_block, c_ca, 'CA', fontsize=12)

    # C4 + CA
    r_c4 = draw_rect(x_bb, y_c4, w_c4, h_block, c_bb, 'C4\n(P4)', fontsize=12)
    r_ca4 = draw_rect(x_bb + w_c4/2 + w_ca/2 + 0.1, y_c4, w_ca, h_block, c_ca, 'CA', fontsize=12)

    # C5 (NO CA)
    r_c5 = draw_rect(x_bb, y_c5, w_c5, h_block, c_bb, 'C5\n(P5)', fontsize=12)

    draw_arrow(x_bb, y_input + h_block/2, x_bb, y_c1 - h_block/2)
    draw_arrow(x_bb, y_c1 + h_block/2, x_bb, y_c2 - h_block/2)
    draw_arrow(x_bb, y_c2 + h_block/2, x_bb, y_c3 - h_block/2)
    draw_arrow(x_bb, y_c3 + h_block/2, x_bb, y_c4 - h_block/2)
    draw_arrow(x_bb, y_c4 + h_block/2, x_bb, y_c5 - h_block/2)

    # --- 2. Neck Section ---
    r_node = 0.4
    n_p5_td = (x_neck_td, y_c5)
    n_p4_td = (x_neck_td, y_c4)
    n_p3_td = (x_neck_td, y_c3)
    n_p2_td = (x_neck_td, y_c2)

    draw_circle(*n_p5_td, r_node, c_neck_td, '$P_5^{td}$')
    draw_circle(*n_p4_td, r_node, c_neck_td, '$P_4^{td}$')
    draw_circle(*n_p3_td, r_node, c_neck_td, '$P_3^{td}$')
    draw_circle(*n_p2_td, r_node, c_neck_td, '$P_2^{td}$')

    # --- 3. Connections ---
    # Backbone -> Neck
    draw_arrow(x_bb + w_c5/2 + 0.1, y_c5, n_p5_td[0]-r_node, n_p5_td[1])
    ax.text((x_bb + w_c5/2 + 0.1 + n_p5_td[0]-r_node)/2, y_c5 + 0.2, "直通路径\n(Direct Pass)", ha='center', fontsize=10, fontfamily='SimSun')
    
    draw_arrow(x_bb + w_c4/2 + w_ca + 0.1, y_c4, n_p4_td[0]-r_node, n_p4_td[1])
    draw_arrow(x_bb + w_c3/2 + w_ca + 0.1, y_c3, n_p3_td[0]-r_node, n_p3_td[1])
    
    draw_arrow(x_bb + w_c2/2 + 0.1, y_c2, n_p2_td[0]-r_node, n_p2_td[1])
    ax.text((x_bb + w_c2/2 + 0.1 + n_p2_td[0]-r_node)/2, y_c2 + 0.2, "直通路径\n(Direct Pass)", ha='center', fontsize=10, fontfamily='SimSun')

    # Top-Down Path
    draw_arrow(n_p5_td[0], n_p5_td[1]-r_node, n_p4_td[0], n_p4_td[1]+r_node)
    draw_arrow(n_p4_td[0], n_p4_td[1]-r_node, n_p3_td[0], n_p3_td[1]+r_node)
    draw_arrow(n_p3_td[0], n_p3_td[1]-r_node, n_p2_td[0], n_p2_td[1]+r_node)

    # --- 4. Heads ---
    r_head = 0.5
    draw_arrow(n_p2_td[0]+r_node, n_p2_td[1], x_head-r_head, y_c2)
    draw_circle(x_head, y_c2, r_head, c_head, '检测头\n(微小目标)')

    draw_arrow(n_p3_td[0]+r_node, n_p3_td[1], x_head-r_head, y_c3)
    draw_circle(x_head, y_c3, r_head, c_head, '检测头\n(小目标)')

    # --- Annotations ---
    rect_bb = patches.Rectangle((x_bb - w_input/2 - 0.5, y_input - 0.5), w_input + 2.0, y_c5 + 1.0,
                                linewidth=1.5, edgecolor='#666666', facecolor='none', linestyle='--')
    ax.add_patch(rect_bb)
    ax.text(x_bb - w_input/2, y_c5 + 0.5, "骨干网络 (Backbone)", fontsize=14, fontweight='bold', fontfamily='SimSun')

    rect_neck = patches.Rectangle((x_neck_td - 1.0, y_c2 - 1.0), 2.0, (y_c5 - y_c2) + 2.0,
                                  linewidth=1.5, edgecolor='#666666', facecolor='none', linestyle='-.')
    ax.add_patch(rect_neck)
    ax.text(x_neck_td, y_c5 + 0.5, "特征颈部 (Neck)", fontsize=14, fontweight='bold', ha='center', fontfamily='SimSun')

    rect_head = patches.Rectangle((x_head - 1.0, y_c2 - 1.0), 2.0, (y_c3 - y_c2) + 2.0,
                                  linewidth=1.5, edgecolor='#666666', facecolor='none', linestyle=':')
    ax.add_patch(rect_head)
    ax.text(x_head, y_c3 + 1.0, "检测头 (Head)", fontsize=14, fontweight='bold', ha='center', fontfamily='SimSun')

    ax.text(x_bb + w_c5/2 + 1.5, y_c5, "← 坐标注意力 (CA)\n   嵌入于特征提取末端\n   执行多尺度融合前提纯", fontsize=12, color='#D79B00', fontweight='bold', ha='left', va='center', fontfamily='SimSun')

    plt.title("坐标注意力 (CA) 骨干网络嵌入架构图", fontsize=18, fontweight='bold', fontfamily='SimSun', y=0.98)
    plt.tight_layout()
    plt.savefig("fig_3_5_ca_backbone_flow.png", dpi=300, bbox_inches='tight')
    print("Saved fig_3_5_ca_backbone_flow.png")

if __name__ == "__main__":
    draw_ca_integration_v2()