import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def main():
    models_config = {
        "YOLOv8n": "runs/detect/YOLOv8n/weights/best.pt",
        "YOLOv9t": "runs/detect/YOLOv9t/weights/best.pt",
        "YOLOv10n": "runs/detect/YOLOv10n/weights/best.pt",
        "YOLOv11n": "runs/detect/YOLOv11n/weights/best.pt",
        "YOLOv12n（基线模型）": "runs/detect/ablation_baseline_ciou/weights/best.pt",
        "YOLOv12n-ACA（本文算法）": "runs/detect/YOLOv12-Tiny-P2P3-Focused/weights/best.pt",
    }

    img_path = "datasets/VisDrone/VisDrone2019-DET-val/images/0000276_04401_d_0000529.jpg"
    save_path = "final_visual_matrix_scenario_C_updated.png"

    plt.rcParams["font.sans-serif"] = ["SimSun"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"

    loaded_models = {name: YOLO(path) for name, path in models_config.items()}
    models_list = list(loaded_models.items())

    fig = plt.figure(figsize=(7.5, 3.0))
    plt.subplots_adjust(wspace=0.03, hspace=0.2, top=0.85, bottom=0.02, left=0.02, right=0.98)
    plt.suptitle("场景 C（高速公路密集车流）", fontfamily="SimSun", fontsize=12, fontweight="bold", y=0.98)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(2, 4, 1)
    plt.imshow(img)
    plt.title("原始图像", fontdict={"family": "SimSun", "size": 10, "weight": "bold"}, pad=6)
    plt.xticks([])
    plt.yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, (model_name, model) in enumerate(models_list):
        results = model.predict(img_path, conf=0.25, verbose=False)[0]
        plotted = results.plot(line_width=2, font_size=10)
        plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

        ax = plt.subplot(2, 4, i + 2)
        plt.imshow(plotted)
        plt.title(model_name, fontdict={"family": "SimSun", "size": 10, "weight": "bold"}, pad=6)
        plt.xticks([])
        plt.yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
