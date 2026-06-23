import json

def fix_categories():
    with open('datasets/VisDrone/VisDrone2019-DET-val/annotations_int.json', 'r') as f:
        anno = json.load(f)
        
    for cat in anno['categories']:
        cat['id'] += 1
        
    for ann in anno['annotations']:
        ann['category_id'] += 1
        
    with open('datasets/VisDrone/VisDrone2019-DET-val/annotations_int_cat.json', 'w') as f:
        json.dump(anno, f)
        
    def fix_pred(pred_path, out_path):
        with open(pred_path, 'r') as f:
            preds = json.load(f)
        for p in preds:
            p['category_id'] += 1
        with open(out_path, 'w') as f:
            json.dump(preds, f)
            
    fix_pred('runs/detect/val25/predictions_int.json', 'runs/detect/val25/predictions_int_cat.json')
    fix_pred('runs/detect/val26/predictions_int.json', 'runs/detect/val26/predictions_int_cat.json')
    print("Fixed category IDs")

if __name__ == '__main__':
    fix_categories()
