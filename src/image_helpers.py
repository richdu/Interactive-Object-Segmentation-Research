import numpy as np

def iou(mask1, mask2):
    return float((mask1&mask2).sum())/((mask1|mask2).sum())

def get_largest_annotation(coco, img_id):
    annIds = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(annIds)
    areas = [x['area'] for x in anns]
    idx_max = np.argmax(areas)
    return anns[idx_max]
