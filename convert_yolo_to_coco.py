import os
import json
import cv2

def convert_yolo_to_coco():
    val_img_dir = 'datasets/VisDrone/VisDrone2019-DET-val/images'
    val_lbl_dir = 'datasets/VisDrone/VisDrone2019-DET-val/labels'
    out_json = 'datasets/VisDrone/VisDrone2019-DET-val/annotations.json'
    
    classes = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": c} for i, c in enumerate(classes)]     
    }
    
    annotation_id = 0
    images = [f for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
    
    for img_name in images:
        img_path = os.path.join(val_img_dir, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        img_id_str = img_name.replace('.jpg', '')
        
        coco_data["images"].append({
            "id": img_id_str,
            "file_name": img_name,
            "width": w,
            "height": h
        })
        
        lbl_name = img_name.replace('.jpg', '.txt')
        lbl_path = os.path.join(val_lbl_dir, lbl_name)
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1]) * w
                        y_center = float(parts[2]) * h
                        box_w = float(parts[3]) * w
                        box_h = float(parts[4]) * h
                        
                        x_min = x_center - box_w / 2
                        y_min = y_center - box_h / 2
                        
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": img_id_str,
                            "category_id": cls_id + 1,  # YOLO adds +1 for COCO format output
                            "bbox": [x_min, y_min, box_w, box_h],
                            "area": box_w * box_h,
                            "iscrowd": 0
                        })
                        annotation_id += 1
                        
    with open(out_json, 'w') as f:
        json.dump(coco_data, f)
    print(f"Converted {len(images)} images and {annotation_id} annotations to {out_json}")

if __name__ == "__main__":
    convert_yolo_to_coco()
