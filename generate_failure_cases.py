import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from pycocotools.coco import COCO

def iou_xywh(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    iw = max(0.0, xi2 - xi1)
    ih = max(0.0, yi2 - yi1)
    inter = iw * ih
    uni = w1 * h1 + w2 * h2 - inter
    if uni <= 0:
        return 0.0
    return inter / uni

def get_crop_around_box(img, box, crop_size=400):
    h, w = img.shape[:2]
    bx, by, bw, bh = box[:4]
    cx, cy = bx + bw/2, by + bh/2

    x1 = int(max(0, cx - crop_size/2))
    y1 = int(max(0, cy - crop_size/2))
    x2 = int(min(w, x1 + crop_size))
    y2 = int(min(h, y1 + crop_size))

    # Adjust if near borders
    if x2 - x1 < crop_size:
        if x1 == 0: x2 = min(w, crop_size)
        else: x1 = max(0, w - crop_size)
    if y2 - y1 < crop_size:
        if y1 == 0: y2 = min(h, crop_size)
        else: y1 = max(0, h - crop_size)

    crop = img[y1:y2, x1:x2].copy()

    # Adjust box relative to crop
    adj_box = [bx - x1, by - y1, bw, bh]
    return crop, adj_box

def draw_box_with_label(img, box, color, label):
    x, y, w, h = [int(v) for v in box[:4]]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("simsun.ttc", 20)
    except:
        font = ImageFont.load_default()
        
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    rgb_color = (color[2], color[1], color[0])
    draw.rectangle([x, max(0, y - th - 10), x + tw + 4, y], fill=rgb_color)
    draw.text((x + 2, max(0, y - th - 10) + 2), label, font=font, fill=(255, 255, 255))
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def collect_failure_crops(model_path, anno_json, img_dir, max_per_type=4, seed=42):
    random.seed(seed)
    coco = COCO(anno_json)
    img_ids = coco.getImgIds()
    model = YOLO(model_path)
    
    miss_crops = []
    fp_crops = []
    loc_crops = []
    
    random.shuffle(img_ids)
    
    for img_id in img_ids:
        if len(miss_crops) >= max_per_type and len(fp_crops) >= max_per_type and len(loc_crops) >= max_per_type:
            break
            
        info = coco.loadImgs([img_id])[0]
        path = os.path.join(img_dir, info["file_name"])
        if not os.path.exists(path):
            continue
            
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        gts = []
        for a in anns:
            gts.append([a["bbox"][0], a["bbox"][1], a["bbox"][2], a["bbox"][3], 1.0, int(a["category_id"])])
            
        if not gts: continue
            
        res = model.predict(path, verbose=False)[0]
        preds = []
        if res.boxes is not None and res.boxes.xywh is not None:
            xywh = res.boxes.xywh.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy()
            for i in range(xywh.shape[0]):
                cx, cy, w, h = xywh[i]
                preds.append([cx - w/2, cy - h/2, w, h, float(conf[i]), int(cls[i]) + 1])
                
        # Matching
        matched_gt = set()
        matched_pred = set()
        for pi, p in enumerate(preds):
            best_iou, best_gi = 0.0, -1
            for gi, g in enumerate(gts):
                if gi in matched_gt or int(p[5]) != int(g[5]): continue
                iou = iou_xywh(p[:4], g[:4])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_gi >= 0 and best_iou >= 0.5:
                matched_gt.add(best_gi)
                matched_pred.add(pi)

        img = cv2.imread(path)
        if img is None: continue

        # 1. Missed (False Negatives) - Yellow
        if len(miss_crops) < max_per_type:
            misses = [gts[i] for i in range(len(gts)) if i not in matched_gt]   
            if misses:
                # Pick a random miss that is small but visible (e.g. area > 100)
                valid_misses = [m for m in misses if m[2]*m[3] > 100]
                if valid_misses:
                    m = random.choice(valid_misses)
                    crop, adj_box = get_crop_around_box(img, m)
                    crop = draw_box_with_label(crop, adj_box, (0, 215, 255), "漏检 (真实框)")
                    miss_crops.append(crop)

        # 2. False Positives - Red
        if len(fp_crops) < max_per_type:
            fps = [preds[i] for i in range(len(preds)) if i not in matched_pred and preds[i][4] > 0.3]
            if fps:
                f = random.choice(fps)
                crop, adj_box = get_crop_around_box(img, f)
                crop = draw_box_with_label(crop, adj_box, (0, 0, 255), "误检")
                fp_crops.append(crop)

        # 3. Localization Errors (IoU between 0.1 and 0.5) - Blue
        if len(loc_crops) < max_per_type:
            for pi, p in enumerate(preds):
                if pi in matched_pred: continue
                for gi, g in enumerate(gts):
                    if gi in matched_gt or int(p[5]) != int(g[5]): continue     
                    iou = iou_xywh(p[:4], g[:4])
                    if 0.1 < iou < 0.5:
                        crop, adj_pbox = get_crop_around_box(img, p)
                        # Also draw GT on the same crop for reference
                        adj_gbox = [g[0] - (p[0] - adj_pbox[0]), g[1] - (p[1] - adj_pbox[1]), g[2], g[3]]
                        crop = draw_box_with_label(crop, adj_gbox, (0, 255, 0), "真实框")
                        crop = draw_box_with_label(crop, adj_pbox, (255, 0, 0), f"预测框 IoU:{iou:.2f}")
                        loc_crops.append(crop)
                        break
                if len(loc_crops) >= max_per_type: break

    return miss_crops, fp_crops, loc_crops

def create_grid(miss_crops, fp_crops, loc_crops, out_path):
    rows = [miss_crops, fp_crops, loc_crops]
    row_labels = ["漏检", "误检", "定位偏移"]
    
    cell_h, cell_w = 400, 400
    top_margin = 50
    left_margin = 250  # Increased for longer text
    
    grid_h = len(rows) * cell_h + top_margin
    grid_w = 4 * cell_w + left_margin
    
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
    
    # Initialize fonts for titles
    try:
        font_title = ImageFont.truetype("simsun.ttc", 36)
        font_label = ImageFont.truetype("simsun.ttc", 40)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        
    img_pil = Image.fromarray(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Draw column titles
    for c_idx in range(4):
        title = f"示例 {c_idx + 1}"
        bbox = draw.textbbox((0, 0), title, font=font_title)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = left_margin + c_idx * cell_w + (cell_w - tw) // 2
        y = top_margin // 2 - th // 2
        draw.text((x, y), title, font=font_title, fill=(0, 0, 0))
    
    for r_idx, (row_crops, label) in enumerate(zip(rows, row_labels)):
        # Draw row label
        bbox = draw.textbbox((0, 0), label, font=font_label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (left_margin - tw) // 2
        y = top_margin + r_idx * cell_h + cell_h // 2 - th // 2
        draw.text((x, y), label, font=font_label, fill=(0, 0, 0))
                    
        for c_idx, crop in enumerate(row_crops):
            if crop is None: continue
            crop_resized = cv2.resize(crop, (cell_w, cell_h))
            y_start = top_margin + r_idx * cell_h
            x_start = left_margin + c_idx * cell_w
            
            # Place resized crop into PIL image by converting temporarily
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            img_pil.paste(Image.fromarray(crop_rgb), (x_start, y_start))
            
    final_grid = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, final_grid)

def main():
    anno_json = 'datasets/VisDrone/VisDrone2019-DET-val/annotations.json'       
    img_dir = 'datasets/VisDrone/VisDrone2019-DET-val/images'
    model_path = 'runs/detect/ablation_baseline_ciou/weights/best.pt'
    
    if not os.path.exists(model_path):
        print('baseline weights not found')
        return
        
    print("Collecting failure crops (this may take a moment)...")
    miss, fps, loc = collect_failure_crops(model_path, anno_json, img_dir, max_per_type=4, seed=45)
    create_grid(miss, fps, loc, 'failure_modes_baseline.png')
    print('Saved highly-visible layout to failure_modes_baseline.png')

if __name__ == '__main__':
    main()

