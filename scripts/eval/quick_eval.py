from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def eval(anno, pred):
    cocoGt = COCO(anno)
    cocoDt = cocoGt.loadRes(pred)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.maxDets = [1, 10, 500]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    return cocoEval.stats

print("Evaluating Base:")
stats_base = eval('datasets/VisDrone/VisDrone2019-DET-val/annotations_int.json', 'runs/detect/val25/predictions_int.json')
print("\nEvaluating Ours:")
stats_ours = eval('datasets/VisDrone/VisDrone2019-DET-val/annotations_int.json', 'runs/detect/val26/predictions_int.json')
