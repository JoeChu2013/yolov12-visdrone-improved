import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_yolov12_uav_architecture():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Configuration
    box_w = 1.8
    box_h = 0.8
    
    # Colors
    c_backbone = '#D9EAD3'  # Light Green/Blue
    c_ca = '#FFF2CC'        # Light Yellow
    c_td = '#FCE5CD'        # Light Orange
    c_bu = '#D0E0E3'        # Light Green
    c_head = '#F4CCCC'      # Light Pink
    c_ghost = '#F3F3F3'     # Light Gray
    c_edge = '#434343'
    
    # Define Nodes: {id: (x, y, label, color, type)}
    nodes = {
        # Backbone
        'Input': (0, 0, 'Input Image\n(640x640)', '#EEEEEE', 'box'),
        'Stem': (0, 1.2, 'Conv Stem', c_backbone, 'box'),
        'P2': (2, 2.5, 'C3k2 Block\n(P2, Stride=4)', c_backbone, 'box'),
        'P3': (2, 4.5, 'C3k2 Block\n(P3, Stride=8)', c_backbone, 'box'),
        'P4': (2, 6.5, 'A2C2f Block\n(P4, Stride=16)', c_backbone, 'box'),
        'P5': (2, 8.5, 'A2C2f Block\n(P5, Stride=32)', c_backbone, 'box'),
        
        # CA Modules
        'CA3': (4.5, 4.5, 'CoordAtt\n(CA)', c_ca, 'box'),
        'CA4': (4.5, 6.5, 'CoordAtt\n(CA)', c_ca, 'box'),
        'CA5': (4.5, 8.5, 'CoordAtt\n(CA)', c_ca, 'box'),
        
        # Neck - Top Down
        'TD5': (8, 8.5, 'Proj (P5)', c_td, 'box'),
        'TD4': (8, 6.5, 'BiFPN_Add\nA2C2f (P4)', c_td, 'box'),
        'TD3': (8, 4.5, 'BiFPN_Add\nA2C2f (P3)', c_td, 'box'),
        'TD2': (8, 2.5, 'BiFPN_Add\nC3k2 (P2)', c_td, 'box'),
        
        # Neck - Bottom Up (Asymmetric)
        'BU3': (11, 4.5, 'BiFPN_Add\nA2C2f (P3)', c_bu, 'box'),
        
        # Ghost Nodes for Asymmetry Visualization
        'Ghost4': (11, 6.5, 'Truncated\n(No BU P4)', c_ghost, 'dashed'),
        'Ghost5': (11, 8.5, 'Truncated\n(No BU P5)', c_ghost, 'dashed'),
        
        # Heads
        'Head2': (14, 2.5, 'Detect Head\n(Tiny Objects)', c_head, 'box'),
        'Head3': (14, 4.5, 'Detect Head\n(Small Objects)', c_head, 'box'),
    }
    
    # Draw Background Grouping Boxes
    def draw_group(x, y, w, h, label, color):
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', linestyle='-.', alpha=0.6)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.3, label, ha='center', va='top', fontsize=14, fontweight='bold', color=color)

    draw_group(-1.5, -0.8, 4.5, 10.5, "Backbone", '#6AA84F')
    draw_group(3.5, 3.5, 2.0, 6.0, "Attention", '#B45F06')
    draw_group(6.5, 1.5, 6.0, 8.0, "Neck (Asymmetric BiFPN)", '#2986CC')
    draw_group(13.0, 1.5, 2.0, 4.0, "Head", '#C90076')

    # Helper to draw boxes
    def draw_node(node_id):
        x, y, label, color, style = nodes[node_id]
        ls = '--' if style == 'dashed' else '-'
        alpha = 0.5 if style == 'dashed' else 1.0
        
        rect = patches.FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h, 
                                      boxstyle="round,pad=0.1", fc=color, ec=c_edge, lw=1.5, linestyle=ls, alpha=alpha, zorder=5)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold', alpha=alpha, zorder=6)

    # Draw all nodes
    for nid in nodes:
        draw_node(nid)

    # Helper to draw arrows
    arrow_props = dict(arrowstyle='->', color=c_edge, lw=1.5, mutation_scale=15)
    dashed_arrow_props = dict(arrowstyle='->', color='gray', lw=1.5, mutation_scale=15, linestyle='--')
    
    def connect(n1, n2, style='straight', offset_x1=0, offset_y1=0, offset_x2=0, offset_y2=0, dashed=False):
        x1, y1 = nodes[n1][0] + offset_x1, nodes[n1][1] + offset_y1
        x2, y2 = nodes[n2][0] + offset_x2, nodes[n2][1] + offset_y2
        
        props = dashed_arrow_props if dashed else arrow_props
        
        if style == 'straight':
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=props, zorder=1)
        elif style == 'vertical_then_horizontal':
            ax.plot([x1, x1, x2], [y1, y2, y2], color=props['color'], lw=props['lw'], ls=props.get('linestyle', '-'), zorder=1)
            ax.annotate("", xy=(x2, y2), xytext=(x2-0.1, y2), arrowprops=props, zorder=1)
        elif style == 'horizontal_then_vertical':
            ax.plot([x1, x2, x2], [y1, y1, y2], color=props['color'], lw=props['lw'], ls=props.get('linestyle', '-'), zorder=1)
            ax.annotate("", xy=(x2, y2), xytext=(x2, y2+0.1 if y1>y2 else y2-0.1), arrowprops=props, zorder=1)

    # --- Draw Connections ---
    
    # Backbone internal
    connect('Input', 'Stem', offset_y1=box_h/2, offset_y2=-box_h/2)
    connect('Stem', 'P2', style='vertical_then_horizontal', offset_y1=box_h/2, offset_x2=-box_w/2)
    connect('P2', 'P3', offset_y1=box_h/2, offset_y2=-box_h/2)
    connect('P3', 'P4', offset_y1=box_h/2, offset_y2=-box_h/2)
    connect('P4', 'P5', offset_y1=box_h/2, offset_y2=-box_h/2)
    
    # Downsample labels
    ax.text(2.1, 3.5, "Conv(s=2)", fontsize=8, color='gray')
    ax.text(2.1, 5.5, "Conv(s=2)", fontsize=8, color='gray')
    ax.text(2.1, 7.5, "Conv(s=2)", fontsize=8, color='gray')

    # Backbone to CA
    connect('P3', 'CA3', offset_x1=box_w/2, offset_x2=-box_w/2)
    connect('P4', 'CA4', offset_x1=box_w/2, offset_x2=-box_w/2)
    connect('P5', 'CA5', offset_x1=box_w/2, offset_x2=-box_w/2)
    
    # CA to Top-Down
    connect('CA5', 'TD5', offset_x1=box_w/2, offset_x2=-box_w/2)
    connect('CA4', 'TD4', offset_x1=box_w/2, offset_x2=-box_w/2)
    connect('CA3', 'TD3', offset_x1=box_w/2, offset_x2=-box_w/2)
    connect('P2', 'TD2', style='vertical_then_horizontal', offset_x1=box_w/2, offset_x2=-box_w/2, offset_y1=-0.2) # P2 goes to TD2 directly

    # Top-Down Path
    connect('TD5', 'TD4', offset_y1=-box_h/2, offset_y2=box_h/2)
    connect('TD4', 'TD3', offset_y1=-box_h/2, offset_y2=box_h/2)
    connect('TD3', 'TD2', offset_y1=-box_h/2, offset_y2=box_h/2)
    
    # Upsample labels
    ax.text(8.1, 7.5, "Upsample", fontsize=8, color='gray')
    ax.text(8.1, 5.5, "Upsample", fontsize=8, color='gray')
    ax.text(8.1, 3.5, "Upsample", fontsize=8, color='gray')

    # Bottom-Up Path (Asymmetric)
    # TD2 -> BU3
    connect('TD2', 'BU3', style='horizontal_then_vertical', offset_x1=box_w/2, offset_x2=-box_w/2+0.5, offset_y1=0.2)
    ax.text(9.5, 3.5, "Downsample", fontsize=8, color='gray', rotation=90)
    
    # TD3 -> BU3 (Skip connection)
    connect('TD3', 'BU3', offset_x1=box_w/2, offset_x2=-box_w/2)

    # Ghost Connections
    connect('BU3', 'Ghost4', style='horizontal_then_vertical', offset_x1=box_w/2, offset_x2=-box_w/2+0.5, dashed=True)
    connect('TD4', 'Ghost4', offset_x1=box_w/2, offset_x2=-box_w/2, dashed=True)
    connect('Ghost4', 'Ghost5', style='horizontal_then_vertical', offset_x1=box_w/2, offset_x2=-box_w/2+0.5, dashed=True)
    connect('TD5', 'Ghost5', offset_x1=box_w/2, offset_x2=-box_w/2, dashed=True)

    # To Heads
    connect('TD2', 'Head2', style='straight', offset_x1=box_w/2, offset_x2=-box_w/2)
    connect('BU3', 'Head3', offset_x1=box_w/2, offset_x2=-box_w/2)

    # Title
    plt.title("Architecture of YOLOv12-UAV (Tiny-P2P3-Focused)", fontsize=18, fontweight='bold', y=0.95)
    
    # Save
    plt.xlim(-2, 16)
    plt.ylim(-1, 10)
    plt.tight_layout()
    plt.savefig("yolov12_uav_architecture.png", dpi=300, bbox_inches='tight')
    print("Architecture diagram saved as yolov12_uav_architecture.png")

if __name__ == "__main__":
    draw_yolov12_uav_architecture()
