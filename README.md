# YOLOv12n-ACA for VisDrone UAV Detection

[中文说明](#中文说明) | [English](#english)

## 中文说明

本仓库基于官方 `YOLOv12` 代码框架，面向 **VisDrone 无人机场景小目标检测** 进行了针对性改造，形成了本文使用的轻量化改进模型 **YOLOv12n-ACA**。仓库同时保留了训练脚本、对比实验脚本、可视化生成脚本以及论文中使用的主要图表。

## 项目目标

无人机航拍场景中的检测任务具有以下典型难点：

- 目标尺寸极小，远距离目标容易被高层语义淹没
- 场景密集，车辆和行人常出现遮挡与紧邻分布
- 背景复杂，俯视视角下不同类别车辆的外观差异被压缩
- 边缘设备部署受限，模型需要兼顾精度与复杂度

为解决上述问题，本项目以 `YOLOv12n` 为基线，围绕小目标表征、多尺度融合和回归优化进行了系统改进。

## 改动方案

### 1. 骨干增强

- 在高层特征提取阶段引入坐标注意力思想，增强空间位置信息建模能力
- 强化小目标的局部结构响应，减轻纯语义特征对细节信息的覆盖
- 对应可视化图：`yolov12_uav_architecture.png`、`fig_3_5_ca_backbone_flow.png`

### 2. 非对称 BiFPN

- 保留完整的自顶向下语义传递路径
- 仅保留对微小目标最有价值的 `P2 -> P3` 自底向上回流
- 裁剪冗余的 `P4/P5` 底向上分支，降低计算量和冗余融合
- 对应拓扑图：`asymmetric_bifpn_topology.png`

### 3. Tiny P2/P3 Focused Detection Head

- 检测头聚焦于 `P2` 与 `P3` 两个尺度
- 重点服务 VisDrone 中占比更高的微小目标和小目标
- 相关训练脚本：`train_tiny_p2p3.py`
- 相关模型配置：`ultralytics/cfg/models/v12/yolov12-tiny-p2p3.yaml`

### 4. 回归与实验分析增强

- 引入 SIoU / IoU 收敛分析脚本，对不同损失函数的优化特性进行可视化
- 补充类别对比、热力图、失败案例、数据集统计等论文支撑材料
- 相关脚本包括：
  - `plot_iou_convergence.py`
  - `generate_heatmaps.py`
  - `generate_failure_cases.py`
  - `generate_feature_responses.py`
  - `dataset_statistics.py`

## 主要效果

`final_comprehensive_comparison.csv` 中汇总了本文模型与主流轻量化检测模型的综合对比结果：

| 模型 | Precision | Recall | F1-Score | mAP@0.5 | Params (M) | FLOPs (G) |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| YOLOv9t | 46.51% | 33.32% | 38.83% | 33.76% | 2.01 | 7.70 |
| YOLOv11n | 45.44% | 33.56% | 38.61% | 33.70% | 2.59 | 6.30 |
| YOLOv12n-ACA (Ours) | 45.64% | 33.12% | 38.39% | 33.66% | 1.88 | 5.42 |
| YOLOv10n | 44.97% | 33.43% | 38.35% | 33.47% | 2.71 | 8.10 |
| YOLOv8n | 44.44% | 33.09% | 37.93% | 33.40% | 3.01 | 8.10 |
| YOLOv12n (Base) | 39.48% | 28.88% | 33.36% | 28.29% | 2.52 | 6.50 |

从结果可以看到：

- 本文模型在 `mAP@0.5` 上相较基线 `YOLOv12n` 提升明显
- 参数量降至 `1.88M`，为对比模型中最低
- FLOPs 仅 `5.42G`，具备更强的边缘部署潜力
- 在综合精度-复杂度权衡上，`YOLOv12n-ACA` 具有较强竞争力

## 论文图表与可视化结果

仓库中已经整理出论文常用图表，可直接复用或继续微调：

### 架构与机制图

- `yolov12_baseline_architecture.png`
- `yolov12_uav_architecture.png`
- `fig_3_5_ca_backbone_flow.png`
- `asymmetric_bifpn_topology.png`
- `siou_vs_ciou_trajectory.png`

### 数据集与统计图

- `coco_vs_visdrone_characteristics.png`
- `visdrone_stats_1_scale.png`
- `visdrone_stats_2_ratio.png`
- `visdrone_stats_3_density.png`
- `dataset_comparison_academic.png`

### 训练与实验结果图

- `siou_vs_ciou_convergence.png`
- `per_class_bar_chart.png`
- `heatmap_comparison.png`
- `feature_responses_baseline.png`
- `failure_modes_baseline.png`
- `final_visual_matrix_scenario_A.png`
- `final_visual_matrix_scenario_B.png`
- `final_visual_matrix_scenario_C_updated.png`
- `final_visual_matrix_scenario_D.png`

## 关键脚本说明

### 训练与评估

- `train_tiny_p2p3.py`：训练 YOLOv12n-ACA 主模型
- `compare_sota.py`：与 YOLOv8/v9/v10/v11 等模型对比
- `eval_coco_multiscale.py`：多尺度 COCO 指标评估
- `convert_yolo_to_coco.py`：YOLO 检测结果转 COCO 格式

### 可视化与论文素材生成

- `generate_visual_matrix.py`：多模型定性检测结果矩阵
- `regenerate_scene_c_highway.py`：重生成高速公路密集车流场景 C
- `generate_heatmaps.py`：特征热力图对比
- `generate_feature_responses.py`：骨干特征响应图
- `generate_failure_cases.py`：失败模式可视化
- `plot_yolov12_base_arch.py`：基线架构图
- `plot_yolov12_uav_arch.py`：改进架构图
- `plot_asymmetric_bifpn_topology.py`：非对称 BiFPN 拓扑图

## 环境安装

建议使用 Python 3.11 和 CUDA 环境。

```bash
conda create -n yolov12 python=3.11
conda activate yolov12
pip install -r requirements.txt
pip install -e .
```

## 训练示例

```bash
python train_tiny_p2p3.py
```

训练完成后，结果默认保存在：

```text
runs/detect/YOLOv12-Tiny-P2P3-Focused
```

主要文件包括：

- `weights/best.pt`
- `results.csv`
- `results.png`
- `confusion_matrix.png`

## 结果复现示例

### 1. 重新生成场景 C 高速公路密集车流检测对比图

```bash
python regenerate_scene_c_highway.py
```

### 2. 重新生成改进架构图

```bash
python plot_yolov12_uav_arch.py
```

### 3. 重新生成非对称 BiFPN 拓扑图

```bash
python plot_asymmetric_bifpn_topology.py
```

## 仓库说明

- 本仓库以论文实验和可视化整理为主
- 部分脚本用于中间分析、图表重绘或论文排版适配
- 如需面向工程部署，可进一步清理中间文件并补充统一的数据处理入口

## 致谢

本项目代码基于以下开源工作进行扩展：

- [YOLOv12](https://github.com/sunsmarterjie/yolov12)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

感谢原作者提供的优秀基础实现。

## 引用说明

如果本仓库中的改进方案、实验脚本、可视化结果或论文图表对您的研究有帮助，欢迎在论文或项目中引用。若引用的是本文对应的改进工作，建议同时说明其基于 YOLOv12 框架并面向 VisDrone 无人机场景进行了定制化改造。

### BibTeX

```bibtex
@misc{yolov12n_aca_visdrone,
  title        = {YOLOv12n-ACA for VisDrone UAV Detection},
  author       = {JoeChu2013},
  year         = {2026},
  howpublished = {\url{https://github.com/JoeChu2013/yolov12-visdrone-improved}},
  note         = {Improved YOLOv12n for UAV small object detection on VisDrone, including asymmetric BiFPN, ACA enhancement, evaluation scripts, and paper-ready visualizations}
}
```

如果您同时参考了原始 YOLOv12 框架，也建议一并引用其原始论文。

---

## English

This repository is built on top of the official `YOLOv12` codebase and focuses on **small-object detection in UAV aerial imagery on VisDrone**. The final model used in the paper is named **YOLOv12n-ACA**. The repository also includes training scripts, evaluation scripts, visualization utilities, and paper-ready figures.

### Project Motivation

VisDrone-style UAV detection is challenging because:

- targets are extremely small and easily submerged by high-level semantics
- crowded scenes lead to occlusion and dense object layouts
- overhead viewpoints reduce inter-class appearance differences
- edge deployment requires a strong balance between accuracy and complexity

### Main Modifications

#### 1. Backbone Enhancement

- Coordinate-attention style enhancement is introduced in high-level feature extraction
- Spatial position modeling is strengthened for tiny-object perception
- Related figures: `yolov12_uav_architecture.png`, `fig_3_5_ca_backbone_flow.png`

#### 2. Asymmetric BiFPN

- The full top-down semantic pathway is preserved
- Only the most useful bottom-up return path `P2 -> P3` is retained
- Redundant `P4/P5` bottom-up branches are removed to reduce computation
- Related topology figure: `asymmetric_bifpn_topology.png`

#### 3. Tiny P2/P3 Focused Detection Head

- Detection is focused on `P2` and `P3` feature levels
- The design targets VisDrone tiny and small objects more directly
- Training script: `train_tiny_p2p3.py`
- Model config: `ultralytics/cfg/models/v12/yolov12-tiny-p2p3.yaml`

#### 4. Additional Analysis and Visualization

- IoU/SIoU convergence plots
- qualitative comparison matrices
- feature heatmaps and failure-case visualization
- dataset characteristic and scale distribution plots

### Main Results

The overall comparison is summarized in `final_comprehensive_comparison.csv`.

| Model | Precision | Recall | F1-Score | mAP@0.5 | Params (M) | FLOPs (G) |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| YOLOv9t | 46.51% | 33.32% | 38.83% | 33.76% | 2.01 | 7.70 |
| YOLOv11n | 45.44% | 33.56% | 38.61% | 33.70% | 2.59 | 6.30 |
| YOLOv12n-ACA (Ours) | 45.64% | 33.12% | 38.39% | 33.66% | 1.88 | 5.42 |
| YOLOv10n | 44.97% | 33.43% | 38.35% | 33.47% | 2.71 | 8.10 |
| YOLOv8n | 44.44% | 33.09% | 37.93% | 33.40% | 3.01 | 8.10 |
| YOLOv12n (Base) | 39.48% | 28.88% | 33.36% | 28.29% | 2.52 | 6.50 |

Key takeaways:

- `YOLOv12n-ACA` improves substantially over the `YOLOv12n` baseline
- the final model uses only `1.88M` parameters
- FLOPs are reduced to `5.42G`, making it more suitable for edge-side UAV deployment
- the model achieves a strong accuracy-complexity trade-off

### Frequently Used Paper Figures

#### Architecture Figures

- `yolov12_baseline_architecture.png`
- `yolov12_uav_architecture.png`
- `fig_3_5_ca_backbone_flow.png`
- `asymmetric_bifpn_topology.png`
- `siou_vs_ciou_trajectory.png`

#### Dataset and Statistics

- `coco_vs_visdrone_characteristics.png`
- `visdrone_stats_1_scale.png`
- `visdrone_stats_2_ratio.png`
- `visdrone_stats_3_density.png`
- `dataset_comparison_academic.png`

#### Experimental Visualizations

- `siou_vs_ciou_convergence.png`
- `per_class_bar_chart.png`
- `heatmap_comparison.png`
- `feature_responses_baseline.png`
- `failure_modes_baseline.png`
- `final_visual_matrix_scenario_A.png`
- `final_visual_matrix_scenario_B.png`
- `final_visual_matrix_scenario_C_updated.png`
- `final_visual_matrix_scenario_D.png`

### Installation

```bash
conda create -n yolov12 python=3.11
conda activate yolov12
pip install -r requirements.txt
pip install -e .
```

### Training

```bash
python train_tiny_p2p3.py
```

### Reproducing Key Figures

```bash
python regenerate_scene_c_highway.py
python plot_yolov12_uav_arch.py
python plot_asymmetric_bifpn_topology.py
```

### Acknowledgements

This project extends:

- [YOLOv12](https://github.com/sunsmarterjie/yolov12)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

### Citation

If this repository, its modified architecture, evaluation scripts, or paper-ready visualizations are helpful to your research, please consider citing it. If your work also builds on the original YOLOv12 framework, citing the original YOLOv12 paper is also recommended.

```bibtex
@misc{yolov12n_aca_visdrone,
  title        = {YOLOv12n-ACA for VisDrone UAV Detection},
  author       = {JoeChu2013},
  year         = {2026},
  howpublished = {\url{https://github.com/JoeChu2013/yolov12-visdrone-improved}},
  note         = {Improved YOLOv12n for UAV small object detection on VisDrone, including asymmetric BiFPN, ACA enhancement, evaluation scripts, and paper-ready visualizations}
}
```
