import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# --- 1. 全局字体与排版设置 (严格适配 Word A4) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimSun']  # 中文宋体
plt.rcParams['mathtext.fontset'] = 'stix'     # 英文/数字 Times New Roman 风格
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10                # 基础字号 10pt

# --- 2. 生成模拟的统计特征数据 ---
# 根据学术界公认特征：COCO 目标相对稀疏且面积分布广；VisDrone 极度密集且目标极小
np.random.seed(42)

# MS COCO: 实例数较少 (均值约 7.7)，目标面积较大
coco_density = np.random.lognormal(mean=np.log(7), sigma=0.8, size=10000)
coco_density = coco_density[coco_density < 80]
# 面积比例 (对数正态分布)
coco_area = np.random.lognormal(mean=np.log(0.05), sigma=1.5, size=10000)
coco_area = coco_area[(coco_area > 0.0001) & (coco_area < 1.0)]

# VisDrone: 实例数极多 (均值约 54)，密集场景，目标面积极小
visdrone_density = np.random.lognormal(mean=np.log(54), sigma=0.6, size=10000)
visdrone_density = visdrone_density[visdrone_density < 200]
# 面积比例极小
visdrone_area = np.random.lognormal(mean=np.log(0.001), sigma=1.2, size=10000)
visdrone_area = visdrone_area[(visdrone_area > 0.00001) & (visdrone_area < 0.1)]

# --- 3. 创建画布 ---
# 宽度 7.2 英寸，高度 3.2 英寸，完美适配 A4 页面横向 1:1 插入
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)

color_coco = '#4C72B0'      # 经典学术蓝
color_visdrone = '#C44E52'  # 经典学术红

# =======================================================
# 子图 (a): 实例密度分布 (Instances per Image)
# =======================================================
ax1 = axes[0]
kde_coco_d = gaussian_kde(coco_density)
kde_vis_d = gaussian_kde(visdrone_density)
x_d = np.linspace(0, 150, 500)

ax1.plot(x_d, kde_coco_d(x_d), color=color_coco, linewidth=2, label='MS COCO')
ax1.fill_between(x_d, kde_coco_d(x_d), alpha=0.3, color=color_coco)
ax1.plot(x_d, kde_vis_d(x_d), color=color_visdrone, linewidth=2, label='VisDrone')
ax1.fill_between(x_d, kde_vis_d(x_d), alpha=0.3, color=color_visdrone)

ax1.set_title('(a) 每张图像实例数量分布', fontsize=11, pad=10)
ax1.set_xlabel('实例数量 ($\mathbf{Instances\ per\ Image}$)', fontsize=10)
ax1.set_ylabel('频次密度 ($\mathbf{Density}$)', fontsize=10)
ax1.set_xlim(0, 150)
ax1.set_ylim(bottom=0)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper right', frameon=True, fontsize=9)

# =======================================================
# 子图 (b): 目标相对面积分布 (Relative Object Area)
# =======================================================
ax2 = axes[1]
# 使用对数尺度展示小目标差异，由于是核密度估计，我们将面积转换为 log10 取值
kde_coco_a = gaussian_kde(np.log10(coco_area))
kde_vis_a = gaussian_kde(np.log10(visdrone_area))
x_a = np.linspace(-4, 0, 500)

ax2.plot(x_a, kde_coco_a(x_a), color=color_coco, linewidth=2, label='MS COCO')
ax2.fill_between(x_a, kde_coco_a(x_a), alpha=0.3, color=color_coco)
ax2.plot(x_a, kde_vis_a(x_a), color=color_visdrone, linewidth=2, label='VisDrone')
ax2.fill_between(x_a, kde_vis_a(x_a), alpha=0.3, color=color_visdrone)

ax2.set_title('(b) 目标相对面积分布', fontsize=11, pad=10)
ax2.set_xlabel('相对面积 ($\mathbf{Relative\ Area}\ 10^x$)', fontsize=10)
ax2.set_ylabel('频次密度 ($\mathbf{Density}$)', fontsize=10)
ax2.set_xlim(-4, 0)
ax2.set_ylim(bottom=0)

# 自定义 X 轴刻度标签为指数形式
xticks = [-4, -3, -2, -1, 0]
ax2.set_xticks(xticks)
ax2.set_xticklabels([f'$10^{{{t}}}$' for t in xticks])

ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper right', frameon=True, fontsize=9)

# --- 4. 调整布局并保存 ---
plt.tight_layout()
plt.savefig('coco_vs_visdrone_characteristics.png', bbox_inches='tight', dpi=300)
print("图表已成功保存为 coco_vs_visdrone_characteristics.png")
