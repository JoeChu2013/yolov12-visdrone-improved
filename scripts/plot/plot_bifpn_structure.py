import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_bifpn_p2_tiny():
    try:
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
    except:
        pass

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(-1, 5)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Define Node Coordinates (x, y)
    # y corresponds to Level (2, 3, 4, 5)
    # x corresponds to Column (Backbone=0, Top-Down=2, Bottom-Up=4)
    
    # Backbone Nodes (C)
    nodes_c = {
        'C5': (0, 5),
        'C4': (0, 4),
        'C3': (0, 3),
        'C2': (0, 2)
    }

    # Top-Down Nodes (P_td) - Blue Path
    nodes_td = {
        'P5_in': (1, 5), # Intermediate
        'P4_td': (2, 4),
        'P3_td': (2, 3),
        'P2_out': (2, 2) # This acts as P2 Output directly in this config
    }

    # Bottom-Up Nodes (P_out) - Red Path
    nodes_bu = {
        'P3_out': (3, 3)
        # P4_out, P5_out are OMITTED as per request
    }
    
    # Heads
    nodes_head = {
        'Head_P2': (4, 2),
        'Head_P3': (4, 3)
    }

    # --- Draw Edges ---

    # Style definitions
    style_backbone = {'color': 'gray', 'linestyle': '--', 'arrowstyle': '->', 'alpha': 0.5}
    style_td = {'color': 'blue', 'linestyle': '-', 'arrowstyle': '->', 'lw': 2}
    style_bu = {'color': 'red', 'linestyle': '-', 'arrowstyle': '->', 'lw': 2}
    style_head = {'color': 'black', 'linestyle': '-', 'arrowstyle': '->', 'lw': 2}
    
    # 1. Backbone Connections to BiFPN (Lateral connections)
    ax.annotate("", xy=(nodes_td['P5_in'][0]-0.3, nodes_td['P5_in'][1]), xytext=(nodes_c['C5'][0]+0.3, nodes_c['C5'][1]), arrowprops=style_backbone)
    ax.annotate("", xy=(nodes_td['P4_td'][0]-0.3, nodes_td['P4_td'][1]), xytext=(nodes_c['C4'][0]+0.3, nodes_c['C4'][1]), arrowprops=style_backbone)
    ax.annotate("", xy=(nodes_td['P3_td'][0]-0.3, nodes_td['P3_td'][1]), xytext=(nodes_c['C3'][0]+0.3, nodes_c['C3'][1]), arrowprops=style_backbone)
    ax.annotate("", xy=(nodes_td['P2_out'][0]-0.3, nodes_td['P2_out'][1]), xytext=(nodes_c['C2'][0]+0.3, nodes_c['C2'][1]), arrowprops=style_backbone)

    # 2. Top-Down Path (Blue)
    # P5 -> P4
    ax.annotate("", xy=(nodes_td['P4_td'][0], nodes_td['P4_td'][1]+0.25), xytext=(nodes_td['P5_in'][0], nodes_td['P5_in'][1]-0.25), arrowprops=style_td)
    # P4 -> P3
    ax.annotate("", xy=(nodes_td['P3_td'][0], nodes_td['P3_td'][1]+0.25), xytext=(nodes_td['P4_td'][0], nodes_td['P4_td'][1]-0.25), arrowprops=style_td)
    # P3 -> P2
    ax.annotate("", xy=(nodes_td['P2_out'][0], nodes_td['P2_out'][1]+0.25), xytext=(nodes_td['P3_td'][0], nodes_td['P3_td'][1]-0.25), arrowprops=style_td)

    # 3. Bottom-Up Path (Red)
    # P2 -> P3
    ax.annotate("", xy=(nodes_bu['P3_out'][0]-0.3, nodes_bu['P3_out'][1]-0.25), xytext=(nodes_td['P2_out'][0]+0.3, nodes_td['P2_out'][1]+0.25), arrowprops=style_bu)
    # P3_td -> P3_out (The fusion connection)
    ax.annotate("", xy=(nodes_bu['P3_out'][0]-0.3, nodes_bu['P3_out'][1]), xytext=(nodes_td['P3_td'][0]+0.3, nodes_td['P3_td'][1]), arrowprops=dict(color='red', linestyle='-', arrowstyle='->', lw=2))

    # 4. Truncated Path (Dashed/Ghost) - Optional visualization
    # P3 -> P4 (Dashed)
    ax.annotate("", xy=(3, 4), xytext=(nodes_bu['P3_out'][0], nodes_bu['P3_out'][1]+0.25), arrowprops=dict(color='gray', linestyle=':', arrowstyle='->', lw=1.5))
    ax.text(3.1, 3.8, "路径截断\n(无底向上 P4/P5)", color='gray', fontsize=9.5, ha='left')

    # 5. Heads (Output)
    ax.annotate("", xy=(nodes_head['Head_P2'][0]-0.3, nodes_head['Head_P2'][1]), xytext=(nodes_td['P2_out'][0]+0.3, nodes_td['P2_out'][1]), arrowprops=style_head)
    ax.annotate("", xy=(nodes_head['Head_P3'][0]-0.3, nodes_head['Head_P3'][1]), xytext=(nodes_bu['P3_out'][0]+0.3, nodes_bu['P3_out'][1]), arrowprops=style_head)

    # --- Draw Nodes ---
    
    def draw_node(name, pos, color='white', ec='black', box=True):
        if box:
            rect = patches.FancyBboxPatch((pos[0]-0.3, pos[1]-0.25), 0.6, 0.5, boxstyle="round,pad=0.1", 
                                          linewidth=1.5, edgecolor=ec, facecolor=color)
            ax.add_patch(rect)
            ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=10.5, fontweight='bold')
        else:
            circle = patches.Circle(pos, 0.15, linewidth=1.5, edgecolor=ec, facecolor=color)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1]-0.35, name, ha='center', va='top', fontsize=10.5)

    # Backbone
    for name, pos in nodes_c.items():
        draw_node(name, pos, color='#E0E0E0', ec='gray')
    
    # Top-Down
    for name, pos in nodes_td.items():
        draw_node(name, pos, color='#E6F3FF', ec='blue') # Light Blue

    # Bottom-Up
    for name, pos in nodes_bu.items():
        draw_node(name, pos, color='#FFE6E6', ec='red') # Light Red
        
    # Heads
    for name, pos in nodes_head.items():
        draw_node("检测头\n(Detect Head)", pos, color='#FFFFE0', ec='orange')
        ax.text(pos[0], pos[1]-0.45, name.replace('_', ' '), ha='center', va='top', fontsize=10, fontweight='bold')

    # --- Legend & Titles ---
    ax.text(0, 5.5, "骨干网络\n(Backbone)", ha='center', fontsize=13, fontweight='bold')
    ax.text(2, 5.5, "自顶向下路径\n(Top-Down Path)", ha='center', fontsize=13, fontweight='bold', color='blue')
    ax.text(3, 5.5, "自底向上路径\n(Bottom-Up Path)", ha='center', fontsize=13, fontweight='bold', color='red')
    ax.text(4, 5.5, "预测头\n(Prediction Heads)", ha='center', fontsize=13, fontweight='bold')

    plt.title("非对称 BiFPN 拓扑架构图", fontsize=15, pad=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig("bifpn_p2_tiny_topology.png", dpi=300)
    print("Diagram saved to bifpn_p2_tiny_topology.png")

if __name__ == "__main__":
    draw_bifpn_p2_tiny()
