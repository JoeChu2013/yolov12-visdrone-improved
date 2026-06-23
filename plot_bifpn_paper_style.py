import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_paper_style_bifpn_detailed():
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Constants
    r = 0.35 # Node radius
    dx = 3.0 # Column spacing (wider for arrows)
    dy = 2.0 # Row spacing
    
    # Coordinates mapping
    # Rows: P2 (bottom) to P5 (top)
    y_map = {'P2': 0*dy, 'P3': 1*dy, 'P4': 2*dy, 'P5': 3*dy}
    
    # Columns: 
    # 0: Backbone (Input)
    # 1: Top-Down Path (Feature Fusion)
    # 2: Bottom-Up Path (Output)
    x_map = {0: 0*dx, 1: 1*dx, 2: 2*dx}
    
    # Colors (Paper style)
    c_blue = '#5B9BD5'   # Backbone
    c_orange = '#ED7D31' # Top-Down
    c_green = '#70AD47'  # Bottom-Up / Output
    c_gray = '#D9D9D9'   # Ghost/Truncated
    
    nodes = []
    
    # --- Define Nodes ---
    
    # 1. Backbone (Column 0)
    nodes.append({'id': 'C5', 'x': x_map[0], 'y': y_map['P5'], 'color': c_blue, 'label': '$C_5$'})
    nodes.append({'id': 'C4', 'x': x_map[0], 'y': y_map['P4'], 'color': c_blue, 'label': '$C_4$'})
    nodes.append({'id': 'C3', 'x': x_map[0], 'y': y_map['P3'], 'color': c_blue, 'label': '$C_3$'})
    nodes.append({'id': 'C2', 'x': x_map[0], 'y': y_map['P2'], 'color': c_blue, 'label': '$C_2$'})

    # 2. Top-Down (Column 1)
    # P4_td fuses C4 and P5
    nodes.append({'id': 'P4_td', 'x': x_map[1], 'y': y_map['P4'], 'color': c_orange, 'label': '$P_4^{td}$'})
    # P3_td fuses C3 and P4_td
    nodes.append({'id': 'P3_td', 'x': x_map[1], 'y': y_map['P3'], 'color': c_orange, 'label': '$P_3^{td}$'})
    # P2_td fuses C2 and P3_td (This is also P2 Output)
    nodes.append({'id': 'P2_td', 'x': x_map[1], 'y': y_map['P2'], 'color': c_green, 'label': '$P_2^{out}$'})

    # 3. Bottom-Up (Column 2)
    # P3_out fuses P3_td and P2_td
    nodes.append({'id': 'P3_out', 'x': x_map[2], 'y': y_map['P3'], 'color': c_green, 'label': '$P_3^{out}$'})
    
    # Ghost Nodes for Truncation
    nodes.append({'id': 'P4_ghost', 'x': x_map[2], 'y': y_map['P4'], 'color': c_gray, 'label': '$P_4^{out}$', 'style': 'dashed'})
    nodes.append({'id': 'P5_ghost', 'x': x_map[2], 'y': y_map['P5'], 'color': c_gray, 'label': '$P_5^{out}$', 'style': 'dashed'})

    node_dict = {n['id']: n for n in nodes}

    # --- Draw Edges ---
    arrow_args = dict(arrowstyle='->', color='black', lw=1.5, mutation_scale=15)
    
    def draw_arrow(n1, n2, connection_type='straight'):
        x1, y1 = n1['x'], n1['y']
        x2, y2 = n2['x'], n2['y']
        
        # Adjust start/end to be on circle edge
        import math
        angle = math.atan2(y2-y1, x2-x1)
        start_x = x1 + r * math.cos(angle)
        start_y = y1 + r * math.sin(angle)
        end_x = x2 - r * math.cos(angle)
        end_y = y2 - r * math.sin(angle)
        
        patch = patches.FancyArrowPatch((start_x, start_y), (end_x, end_y), **arrow_args)
        ax.add_patch(patch)

    # 1. Backbone -> Top-Down (Lateral)
    draw_arrow(node_dict['C5'], node_dict['P4_td']) # C5 downsamples/projects to P4_td input? No, typically P5->P4_td
    # Actually in BiFPN: P4_td = Conv(C4) + Resize(P5_in). 
    # Here P5_in is C5.
    
    draw_arrow(node_dict['C4'], node_dict['P4_td'])
    draw_arrow(node_dict['C3'], node_dict['P3_td'])
    draw_arrow(node_dict['C2'], node_dict['P2_td'])
    
    # 2. Top-Down Path
    draw_arrow(node_dict['P4_td'], node_dict['P3_td'])
    draw_arrow(node_dict['P3_td'], node_dict['P2_td'])
    
    # 3. Bottom-Up Path
    draw_arrow(node_dict['P2_td'], node_dict['P3_out'])
    draw_arrow(node_dict['P3_td'], node_dict['P3_out']) # Skip connection P3_td -> P3_out
    
    # 4. Heads
    # From P2_td
    ax.annotate("Detect\n(Tiny)", xy=(node_dict['P2_td']['x'], node_dict['P2_td']['y']-r), 
                xytext=(node_dict['P2_td']['x'], node_dict['P2_td']['y']-1.0),
                ha='center', va='center', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # From P3_out
    ax.annotate("Detect\n(Small)", xy=(node_dict['P3_out']['x']+r, node_dict['P3_out']['y']), 
                xytext=(node_dict['P3_out']['x']+1.2, node_dict['P3_out']['y']),
                ha='center', va='center', fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # --- Draw Nodes ---
    for n in nodes:
        linestyle = '--' if n.get('style') == 'dashed' else '-'
        alpha = 0.5 if n.get('style') == 'dashed' else 1.0
        circle = patches.Circle((n['x'], n['y']), r, facecolor=n['color'], edgecolor='black', linewidth=1.5, linestyle=linestyle, alpha=alpha, zorder=10)
        ax.add_patch(circle)
        ax.text(n['x'], n['y'], n['label'], ha='center', va='center', fontsize=11, fontweight='bold', color='white' if n['color'] in [c_blue, c_green, c_orange] else 'black')

    # --- Draw Truncated Connections (Dashed) ---
    # P3_out -> P4_ghost
    n1 = node_dict['P3_out']
    n2 = node_dict['P4_ghost']
    ax.add_patch(patches.FancyArrowPatch((n1['x'], n1['y']+r), (n2['x'], n2['y']-r), arrowstyle='->', color='gray', lw=1.5, linestyle='--'))
    
    # P4_ghost -> P5_ghost
    n1 = node_dict['P4_ghost']
    n2 = node_dict['P5_ghost']
    ax.add_patch(patches.FancyArrowPatch((n1['x'], n1['y']+r), (n2['x'], n2['y']-r), arrowstyle='->', color='gray', lw=1.5, linestyle='--'))
    
    ax.text(x_map[2]+0.6, (y_map['P4']+y_map['P5'])/2, "Truncated Path\n(Saved Computation)", ha='left', va='center', color='gray', fontsize=10, fontstyle='italic')

    # --- Titles ---
    ax.text(x_map[0], y_map['P5']+0.8, "Backbone", ha='center', fontsize=12, fontweight='bold')
    ax.text(x_map[1], y_map['P5']+0.8, "Top-Down FPN", ha='center', fontsize=12, fontweight='bold')
    ax.text(x_map[2], y_map['P5']+0.8, "Bottom-Up PANet", ha='center', fontsize=12, fontweight='bold')

    plt.title("Modified BiFPN: Tiny-P2P3 Focused Architecture", fontsize=16, y=0.98)
    
    # Adjust limits
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1.5, 7)
    
    plt.tight_layout()
    plt.savefig("bifpn_p2_tiny_paper_style_v2.png", dpi=300, bbox_inches='tight')
    print("Saved bifpn_p2_tiny_paper_style_v2.png")

if __name__ == "__main__":
    draw_paper_style_bifpn_detailed()
