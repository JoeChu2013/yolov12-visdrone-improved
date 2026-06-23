import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


def draw_asymmetric_bifpn_topology():
    try:
        plt.rcParams["font.sans-serif"] = ["SimSun"]
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["mathtext.fontset"] = "stix"
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300)
    ax.set_aspect("equal")
    ax.axis("off")

    r = 0.28
    c_edge = "#444444"
    c_backbone = "#D9EAD3"
    c_td = "#FCE5CD"
    c_bu = "#D0E0E3"
    c_head = "#F4CCCC"
    c_ghost = "#F3F3F3"

    nodes = {
        # Backbone inputs
        "P5": (0.0, 6.0, r"$P_5$", c_backbone, "solid"),
        "P4": (0.0, 4.5, r"$P_4$", c_backbone, "solid"),
        "P3": (0.0, 3.0, r"$P_3$", c_backbone, "solid"),
        "P2": (0.0, 1.5, r"$P_2$", c_backbone, "solid"),

        # Top-down path
        "P5_td": (2.4, 6.0, r"$P_5^{td}$", c_td, "solid"),
        "P4_td": (2.4, 4.5, r"$P_4^{td}$", c_td, "solid"),
        "P3_td": (2.4, 3.0, r"$P_3^{td}$", c_td, "solid"),
        "P2_td": (2.4, 1.5, r"$P_2^{td}$", c_td, "solid"),

        # Bottom-up path
        "P3_out": (4.9, 3.0, r"$P_3^{out}$", c_bu, "solid"),
        "P4_out": (4.9, 4.5, r"$P_4^{out}$", c_ghost, "dashed"),
        "P5_out": (4.9, 6.0, r"$P_5^{out}$", c_ghost, "dashed"),

        # Heads
        "Head2": (7.2, 1.5, "检测头\n(微小目标)", c_head, "solid"),
        "Head3": (7.2, 3.0, "检测头\n(小目标)", c_head, "solid"),
    }

    def draw_group(x, y, w, h, label, color, label_offset=0.12):
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1.3, edgecolor=color, facecolor="none",
            linestyle="-.", alpha=0.7
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h + label_offset, label,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=color, zorder=10,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.8)
        )

    def draw_circle_node(node_id):
        x, y, label, color, style = nodes[node_id]
        ls = "--" if style == "dashed" else "-"
        alpha = 0.5 if style == "dashed" else 1.0
        circle = patches.Circle(
            (x, y), r, facecolor=color, edgecolor=c_edge,
            linewidth=1.3, linestyle=ls, alpha=alpha, zorder=5
        )
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold", alpha=alpha, zorder=6)

    def draw_head(node_id):
        x, y, label, color, _ = nodes[node_id]
        box = patches.FancyBboxPatch(
            (x - 0.62, y - 0.32), 1.24, 0.64,
            boxstyle="round,pad=0.08", fc=color, ec=c_edge, lw=1.2, zorder=5
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=9.2, fontweight="bold", zorder=6)

    def connect(n1, n2, color=c_edge, lw=1.4, ls="-", label=None, label_dx=0.0, label_dy=0.0):
        x1, y1 = nodes[n1][0], nodes[n1][1]
        x2, y2 = nodes[n2][0], nodes[n2][1]
        angle = math.atan2(y2 - y1, x2 - x1)
        start = (x1 + r * math.cos(angle), y1 + r * math.sin(angle))
        end_r = 0.62 if n2.startswith("Head") else r
        end = (x2 - end_r * math.cos(angle), y2 - end_r * math.sin(angle))
        arrow = patches.FancyArrowPatch(
            start, end, arrowstyle="->", mutation_scale=12,
            lw=lw, color=color, linestyle=ls, zorder=2
        )
        ax.add_patch(arrow)
        if label:
            mx = (start[0] + end[0]) / 2 + label_dx
            my = (start[1] + end[1]) / 2 + label_dy
            ax.text(
                mx, my, label, fontsize=8, color=color, ha="center", va="center",
                zorder=10, bbox=dict(facecolor="white", edgecolor="none", pad=0.8)
            )

    # Group boxes
    draw_group(-0.65, 0.95, 1.3, 5.6, "输入特征层", "#6AA84F", label_offset=0.16)
    draw_group(1.75, 0.95, 1.3, 5.6, "自顶向下\n路径", "#B45F06", label_offset=0.24)
    draw_group(4.2, 2.65, 1.5, 3.7, "非对称\n自底向上路径", "#2986CC", label_offset=0.2)
    draw_group(6.45, 0.95, 1.5, 2.45, "检测头", "#C90076", label_offset=0.16)

    # Nodes
    for node_id in ["P5", "P4", "P3", "P2", "P5_td", "P4_td", "P3_td", "P2_td", "P3_out", "P4_out", "P5_out"]:
        draw_circle_node(node_id)
    draw_head("Head2")
    draw_head("Head3")

    # Lateral inputs
    connect("P5", "P5_td")
    connect("P4", "P4_td")
    connect("P3", "P3_td")
    connect("P2", "P2_td")

    # Top-down path
    connect("P5_td", "P4_td", label="上采样", label_dx=0.45)
    connect("P4_td", "P3_td", label="上采样", label_dx=0.45)
    connect("P3_td", "P2_td", label="上采样", label_dx=0.45)

    # Bottom-up path (asymmetric)
    connect("P2_td", "P3_out", color="#2F5597", lw=1.6, label="下采样", label_dx=0.25, label_dy=-0.18)
    connect("P3_td", "P3_out", color="#2F5597", lw=1.6)

    # Truncated ghost path
    connect("P3_out", "P4_out", color="gray", ls="--", lw=1.2)
    connect("P4_td", "P4_out", color="gray", ls="--", lw=1.2)
    connect("P4_out", "P5_out", color="gray", ls="--", lw=1.2)
    connect("P5_td", "P5_out", color="gray", ls="--", lw=1.2)
    ax.text(
        5.7, 5.2, "路径截断", fontsize=8.5, color="gray", ha="left", va="center",
        zorder=10, bbox=dict(facecolor="white", edgecolor="none", pad=0.8)
    )
    ax.text(
        5.7, 4.82, "(省略 P4/P5 底向上分支)", fontsize=8.2, color="gray", ha="left", va="center",
        zorder=10, bbox=dict(facecolor="white", edgecolor="none", pad=0.8)
    )

    # Heads
    connect("P2_td", "Head2")
    connect("P3_out", "Head3")

    # Fusion hints
    ax.text(
        2.4, 6.82, "保留完整高到低语义传递", fontsize=8.4, color="#B45F06", ha="center",
        zorder=10, bbox=dict(facecolor="white", edgecolor="none", pad=0.8)
    )
    ax.text(
        4.9, 2.02, "仅保留关键 P2→P3 回流", fontsize=8.6, color="#2F5597", ha="center",
        zorder=10, bbox=dict(facecolor="white", edgecolor="none", pad=0.8)
    )

    plt.title("非对称 BiFPN 拓扑架构图", fontsize=15, fontweight="bold", y=1.02)
    ax.set_xlim(-1.0, 8.1)
    ax.set_ylim(0.6, 7.2)
    plt.tight_layout()
    plt.savefig("asymmetric_bifpn_topology.png", dpi=300, bbox_inches="tight")
    print("Saved asymmetric_bifpn_topology.png")


if __name__ == "__main__":
    draw_asymmetric_bifpn_topology()
