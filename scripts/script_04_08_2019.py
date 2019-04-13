import sys
import os
import traceback
import numpy as np
import cv2
import time

sys.path.append('../external/cocoapi/PythonAPI/')
from pycocotools.coco import COCO
sys.path.append('../src/')
from image_helpers import iou, get_largest_annotation
from grabcut_model import GrabCutModel

def write_helper(img, image_array, iou_array, mask_array, user_input_array):
    print("Writing: {0:012d}".format(img['id']))
    fp = os.path.join(output_folder, "{0:012d}_image_array.npy".format(img['id']))
    np.save(fp, np.array(image_array))
    print(fp)
    fp = os.path.join(output_folder, "{0:012d}_iou_array.npy".format(img['id']))
    np.save(fp, np.array(iou_array))
    print(fp)
    fp = os.path.join(output_folder, "{0:012d}_mask_array.npy".format(img['id']))
    np.save(fp, np.array(mask_array))
    print(fp)
    fp = os.path.join(output_folder, "{0:012d}_user_input_array.npy".format(img['id']))
    np.save(fp, np.array(user_input_array))
    print(fp)

# Image dirs
dataDir = '../../data/input/coco'
print(os.listdir(dataDir))
dataType = 'val2017'
annDir = '{}/annotations'.format(dataDir)
annZipFile = '{}/annotations_train{}.zip'.format(dataDir, dataType)
annFile = '{}/instances_{}.json'.format(annDir, dataType)
imgDir = '{}/images/{}'.format(dataDir, dataType)
print (annDir)
print (annFile)
print (annZipFile)
print (imgDir)

output_folder = "/home/richarddu1226/research/data/working/04-08-2019"

# initialize COCO api for instance annotations
coco=COCO(annFile)

imgIds = coco.getImgIds()
imgs = coco.loadImgs(imgIds)

t0 = time.time()
for img in imgs:
    try:
        I = cv2.imread(os.path.join(imgDir, img['file_name']))
        largest_annotation = get_largest_annotation(coco, img['id'])
        largest_mask = coco.annToMask(largest_annotation)
        grabcut = GrabCutModel(I)
        rect = tuple([int(x) for x in largest_annotation['bbox']])
        image_array, iou_array, mask_array, user_input_array = grabcut.multiple_iterations(largest_mask, rect, 20)
        write_helper(img, image_array, iou_array, mask_array, user_input_array)
        print(time.time()-t0)
    except Exception as e:
        print("Error in {}".format(img["file_name"]), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

