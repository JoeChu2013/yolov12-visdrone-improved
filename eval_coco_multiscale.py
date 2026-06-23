import os
import pandas as pd
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_with_pycocotools(anno_json, pred_json):
    coco_gt = COCO(anno_json)
    coco_dt = coco_gt.loadRes(pred_json)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.maxDets = [1, 10, 100, 500] # Increase maxDets for VisDrone     
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # To get metrics for maxDets=500:
    # shape of precision is (T, R, K, A, M) where M is len(maxDets)=4
    # We want T=all, R=all, K=all, A=all, M=3 (500)
    import numpy as np
    
    def _summarizeDets(ap=1, iouThr=None, areaRng='all', maxDets=500):
        p = coco_eval.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            s = coco_eval.eval['precision']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            s = coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        return mean_s

    map_05_095 = _summarizeDets(1, maxDets=500)
    map_05 = _summarizeDets(1, iouThr=.5, maxDets=500)
    ap_s = _summarizeDets(1, areaRng='small', maxDets=500)
    ap_m = _summarizeDets(1, areaRng='medium', maxDets=500)
    ap_l = _summarizeDets(1, areaRng='large', maxDets=500)

    return {
        "mAP@0.5:0.95": map_05_095,
        "mAP@0.5": map_05,
        "AP_Small": ap_s,
        "AP_Medium": ap_m,
        "AP_Large": ap_l
    }

def main():
    anno_json = 'datasets/VisDrone/VisDrone2019-DET-val/annotations.json'

    models_to_eval = {
        "YOLOv12n (Base)": "runs/detect/val25/predictions.json",
        "YOLOv12n-UAV (Ours)": "runs/detect/val26/predictions.json"
    }

    if not os.path.exists(anno_json):
        print(f"Error: Annotations JSON not found at {anno_json}")
        return

    results_data = []

    for name, pred_json in models_to_eval.items():
        if not os.path.exists(pred_json):
            print(f"Skipping {name}: predictions.json not found at {pred_json}")
            continue

        try:
            print(f"\n--- Running pycocotools evaluation for {name} ---")
            coco_metrics = evaluate_with_pycocotools(anno_json, pred_json)

            results_data.append({
                "Model": name,
                "mAP@0.5": coco_metrics["mAP@0.5"],
                "mAP@0.5:0.95": coco_metrics["mAP@0.5:0.95"],
                "AP_S (Small)": coco_metrics["AP_Small"],
                "AP_M (Medium)": coco_metrics["AP_Medium"],
                "AP_L (Large)": coco_metrics["AP_Large"]
            })

        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    if results_data:
        df = pd.DataFrame(results_data)
        
        # Format as percentages
        cols_to_format = ["mAP@0.5", "mAP@0.5:0.95", "AP_S (Small)", "AP_M (Medium)", "AP_L (Large)"]
        for col in cols_to_format:
            df[col] = df[col].apply(lambda x: f"{x*100:.2f}%" if x >= 0 else "N/A")
            
        print("\n=== COCO Multi-Scale Metrics Comparison ===")
        print(df.to_markdown(index=False))
        df.to_csv("coco_multiscale_comparison.csv", index=False)
        print("Saved to coco_multiscale_comparison.csv")

if __name__ == "__main__":
    main()
