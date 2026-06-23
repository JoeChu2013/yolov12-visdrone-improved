import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_ca_backbone_diagram():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Coordinates
    x_center = 4
    dy = 2.0
    
    # Backbone Blocks (Simplified)
    blocks = [
        {'id': 'Stem', 'label': 'Stem (P1/P2)', 'y': 8},
        {'id': 'P3_Block', 'label': 'C3k2 Block (P3)', 'y': 6},
        {'id': 'P4_Block', 'label': 'A2C2f Block (P4)', 'y': 4},
        {'id': 'P5_Block', 'label': 'A2C2f Block (P5)', 'y': 2},
    ]
    
    # CA Modules (Inserted after blocks)
    ca_modules = [
        {'id': 'CA_P3', 'y': 5, 'label': 'CoordAtt (P3)'},
        {'id': 'CA_P4', 'y': 3, 'label': 'CoordAtt (P4)'},
        {'id': 'CA_P5', 'y': 1, 'label': 'CoordAtt (P5)'},
    ]

    # Draw Arrows
    arrow_args = dict(arrowstyle='->', color='black', lw=2)
    
    # Main Flow
    ax.annotate("", xy=(x_center, 7.2), xytext=(x_center, 7.8), arrowprops=arrow_args) # Stem -> P3
    ax.annotate("", xy=(x_center, 5.8), xytext=(x_center, 6.2), arrowprops=arrow_args) # P3 -> CA
    ax.annotate("", xy=(x_center, 4.8), xytext=(x_center, 5.2), arrowprops=arrow_args) # CA -> P4
    ax.annotate("", xy=(x_center, 3.8), xytext=(x_center, 4.2), arrowprops=arrow_args) # P4 -> CA
    ax.annotate("", xy=(x_center, 2.8), xytext=(x_center, 3.2), arrowprops=arrow_args) # CA -> P5
    ax.annotate("", xy=(x_center, 1.8), xytext=(x_center, 2.2), arrowprops=arrow_args) # P5 -> CA
    ax.annotate("", xy=(x_center, 0.5), xytext=(x_center, 1.2), arrowprops=arrow_args) # CA -> SPPF/Neck

    # Draw Blocks
    w_block = 4
    h_block = 0.8
    
    for b in blocks:
        rect = patches.FancyBboxPatch((x_center - w_block/2, b['y'] - h_block/2), w_block, h_block, 
                                      boxstyle="round,pad=0.1", fc='#DAE8FC', ec='#6C8EBF', lw=2)
        ax.add_patch(rect)
        ax.text(x_center, b['y'], b['label'], ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw CA Modules (Distinct Color)
    w_ca = 3
    h_ca = 0.6
    
    for ca in ca_modules:
        rect = patches.FancyBboxPatch((x_center - w_ca/2, ca['y'] - h_ca/2), w_ca, h_ca, 
                                      boxstyle="round,pad=0.1", fc='#FFE6CC', ec='#D79B00', lw=2)
        ax.add_patch(rect)
        ax.text(x_center, ca['y'], ca['label'], ha='center', va='center', fontsize=12, fontweight='bold', color='#D79B00')
        
        # Add annotation about CA function
        ax.text(x_center + 2.5, ca['y'], "Spatial + Channel\nAttention", ha='left', va='center', fontsize=9, color='gray', fontstyle='italic')

    # Add Side Connections (To Neck)
    # P3 Output
    ax.annotate("", xy=(x_center+3, 5), xytext=(x_center+1.6, 5), arrowprops=dict(arrowstyle='->', color='#D79B00', lw=2))
    ax.text(x_center+3.2, 5, "To Neck (P3)", ha='left', va='center', fontsize=10, fontweight='bold')
    
    # P4 Output
    ax.annotate("", xy=(x_center+3, 3), xytext=(x_center+1.6, 3), arrowprops=dict(arrowstyle='->', color='#D79B00', lw=2))
    ax.text(x_center+3.2, 3, "To Neck (P4)", ha='left', va='center', fontsize=10, fontweight='bold')
    
    # P5 Output
    ax.annotate("", xy=(x_center+3, 1), xytext=(x_center+1.6, 1), arrowprops=dict(arrowstyle='->', color='#D79B00', lw=2))
    ax.text(x_center+3.2, 1, "To Neck (P5)", ha='left', va='center', fontsize=10, fontweight='bold')

    # Downsampling (Conv Stride 2) annotations
    ax.text(x_center-2.5, 5.5, "Conv (Stride 2)", ha='right', va='center', fontsize=9)
    ax.text(x_center-2.5, 3.5, "Conv (Stride 2)", ha='right', va='center', fontsize=9)

    plt.title("Fig 3.5: Integration of Coordinate Attention (CA) in YOLOv12 Backbone", fontsize=14, y=0.95)
    plt.ylim(0, 9)
    plt.xlim(0, 9)
    plt.tight_layout()
    plt.savefig("ca_backbone_integration.png", dpi=300)
    print("Saved ca_backbone_integration.png")

if __name__ == "__main__":
    draw_ca_backbone_diagram()
