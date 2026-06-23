import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_yolov12_baseline_architecture():
    try:
        plt.rcParams['font.sans-serif'] = ['SimSun']
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.fontset'] = 'stix'
    except:
        pass
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Configuration
    box_w = 2.15
    box_h = 0.95
    
    # Colors
    c_backbone = '#D9EAD3'  # Light Green
    c_sppf = '#FFF2CC'      # Light Yellow
    c_td = '#FCE5CD'        # Light Orange
    c_bu = '#D0E0E3'        # Light Cyan
    c_head = '#F4CCCC'      # Light Pink
    c_edge = '#434343'
    
    # Define Nodes: {id: (x, y, label, color)}
    nodes = {
        # Backbone
        'Input': (0, -0.5, '输入图像\n(640x640)', '#EEEEEE'),
        'Stem': (0, 1.5, '卷积主干\n(Stem)', c_backbone),
        'P2': (0, 3.5, 'C3k2模块\n(P2)', c_backbone),
        'P3': (0, 5.5, 'C3k2模块\n(P3)', c_backbone),
        'P4': (0, 7.5, 'A2C2f模块\n(P4)', c_backbone),
        'P5': (0, 9.5, 'A2C2f模块\n(P5)', c_backbone),
        
        # Neck
        'SPPF': (2.5, 9.5, 'SPPF模块\n(空间池化)', c_sppf),
        'TD4': (5.0, 7.5, '特征融合\nA2C2f', c_td),
        'TD3': (7.5, 5.5, '特征融合\nC3k2', c_td),
        'BU4': (10.0, 7.5, '特征融合\nA2C2f', c_bu),
        'BU5': (12.5, 9.5, '特征融合\nA2C2f', c_bu),
        
        # Heads
        'Head3': (15.0, 5.5, '检测头\n(小目标)', c_head),
        'Head4': (15.0, 7.5, '检测头\n(中目标)', c_head),
        'Head5': (15.0, 9.5, '检测头\n(大目标)', c_head),
    }
    
    def draw_group(x, y, w, h, label, color):
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=color, facecolor='none', linestyle='-.', alpha=0.6)
        ax.add_patch(rect)
        ax.text(
            x + w/2,
            y + h + 0.12,
            label,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color=color,
            zorder=10,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5),
        )

    draw_group(-1.35, -1.2, 2.7, 11.7, "骨干网络", '#6AA84F')
    draw_group(1.3, 4.3, 12.4, 6.2, "特征颈部 (PANet)", '#2986CC')
    draw_group(13.75, 4.3, 2.5, 6.2, "检测头", '#C90076')

    for nid, (x, y, label, color) in nodes.items():
        rect = patches.FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h, 
                                      boxstyle="round,pad=0.1", fc=color, ec=c_edge, lw=1.2, zorder=5)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=9.0, fontweight='bold', zorder=6)

    arrow_props = dict(arrowstyle='->', color=c_edge, lw=1.2, mutation_scale=12)

    def connect_vertical(n1, n2, label=None):
        x1, y1 = nodes[n1][0], nodes[n1][1] + box_h/2
        x2, y2 = nodes[n2][0], nodes[n2][1] - box_h/2
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_props, zorder=1)
        if label:
            ax.text(x1+0.1, (y1+y2)/2, label, fontsize=8, color='gray', va='center')

    def connect_straight(n1, n2):
        x1, y1 = nodes[n1][0] + box_w/2, nodes[n1][1]
        x2, y2 = nodes[n2][0] - box_w/2, nodes[n2][1]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_props, zorder=1)

    def connect_branch(n1, n2, branch_x, in_side='top', label=None):
        y1 = nodes[n1][1]
        x2, y2 = nodes[n2][0], nodes[n2][1]
        
        if in_side == 'top':
            py2 = y2 + box_h/2
        elif in_side == 'bottom':
            py2 = y2 - box_h/2
            
        ax.plot([branch_x, branch_x], [y1, py2], color=c_edge, lw=1.2, zorder=1)
        ax.annotate("", xy=(branch_x, py2), xytext=(branch_x, py2+0.01 if y1<py2 else py2-0.01), arrowprops=arrow_props, zorder=1)
        if label:
            ax.text(branch_x+0.1, (y1+py2)/2, label, fontsize=8, color='gray', va='center')

    def draw_junction(x, y):
        circle = patches.Circle((x, y), radius=0.08, color=c_edge, zorder=3)
        ax.add_patch(circle)

    # Backbone
    connect_vertical('Input', 'Stem')
    connect_vertical('Stem', 'P2', '下采样')
    connect_vertical('P2', 'P3', '下采样')
    connect_vertical('P3', 'P4', '下采样')
    connect_vertical('P4', 'P5', '下采样')
    
    connect_straight('P5', 'SPPF')
    
    # Horizontal paths
    connect_straight('P4', 'TD4')
    connect_straight('P3', 'TD3')
    
    connect_straight('SPPF', 'BU5')
    connect_straight('TD4', 'BU4')
    connect_straight('TD3', 'Head3')
    
    connect_straight('BU4', 'Head4')
    connect_straight('BU5', 'Head5')
    
    # Branches
    connect_branch('SPPF', 'TD4', branch_x=nodes['TD4'][0], in_side='top', label='上采样')
    draw_junction(nodes['TD4'][0], nodes['SPPF'][1])
    
    connect_branch('TD4', 'TD3', branch_x=nodes['TD3'][0], in_side='top', label='上采样')
    draw_junction(nodes['TD3'][0], nodes['TD4'][1])
    
    connect_branch('TD3', 'BU4', branch_x=nodes['BU4'][0], in_side='bottom', label='下采样')
    draw_junction(nodes['BU4'][0], nodes['TD3'][1])
    
    connect_branch('BU4', 'BU5', branch_x=nodes['BU5'][0], in_side='bottom', label='下采样')
    draw_junction(nodes['BU5'][0], nodes['BU4'][1])

    plt.title("YOLOv12 基线网络架构图", fontsize=15, fontweight='bold', y=1.02)
    
    plt.xlim(-2, 16.5)
    plt.ylim(-2, 11)
    plt.tight_layout()
    plt.savefig("yolov12_baseline_architecture.png", dpi=300, bbox_inches='tight')
    print("Saved yolov12_baseline_architecture.png")

if __name__ == "__main__":
    draw_yolov12_baseline_architecture()
