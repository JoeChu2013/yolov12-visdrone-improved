import json
import os

def fix_ids():
    anno_path = 'datasets/VisDrone/VisDrone2019-DET-val/annotations.json'
    pred_base = 'runs/detect/val25/predictions.json'
    pred_ours = 'runs/detect/val26/predictions.json'
    
    with open(anno_path, 'r') as f:
        anno = json.load(f)
        
    img_id_map = {}
    for i, img in enumerate(anno['images']):
        img_id_map[img['id']] = i + 1
        img['id'] = i + 1
        
    for ann in anno['annotations']:
        ann['image_id'] = img_id_map[ann['image_id']]
        
    with open('datasets/VisDrone/VisDrone2019-DET-val/annotations_int.json', 'w') as f:
        json.dump(anno, f)
        
    def fix_pred(pred_path, out_path):
        if not os.path.exists(pred_path): return
        with open(pred_path, 'r') as f:
            preds = json.load(f)
        for p in preds:
            if p['image_id'] in img_id_map:
                p['image_id'] = img_id_map[p['image_id']]
        with open(out_path, 'w') as f:
            json.dump(preds, f)
            
    fix_pred(pred_base, 'runs/detect/val25/predictions_int.json')
    fix_pred(pred_ours, 'runs/detect/val26/predictions_int.json')
    print("Fixed IDs")

if __name__ == '__main__':
    fix_ids()
