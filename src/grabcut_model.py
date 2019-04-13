import numpy as np
import cv2
import user_input_simulation
from image_helpers import iou

class GrabCutModel:
    def __init__(self, img):
        self.img = img
        self.mask = np.zeros(img.shape[:2], np.uint8)
        print(self.mask.shape)
    
    def get_output_mask(self):
        return np.where((self.mask==2)|(self.mask==0), 0, 1).astype('uint8')
    
    def get_output_image(self):
        output_mask = self.get_output_mask()
        return self.img*output_mask[:,:,np.newaxis]
    
    def init_iteration(self, rect):
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        cv2.grabCut(self.img, self.mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    def mask_iteration(self, user_input):
        self.mask[user_input == user_input_simulation.BG_COLOR] = 0
        self.mask[user_input == user_input_simulation.FG_COLOR] = 1
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        cv2.grabCut(self.img, self.mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    
    # Returns all masks, and iou data for visualization and statistics purposes
    def multiple_iterations(self, gt, rect, iterations):
        mask_array = []
        user_input_array = []
        image_array = []
        iou_array = []

        # Rectangle init
        self.init_iteration(rect)

        mask_array.append(self.get_output_mask())
        image_array.append(self.get_output_image())
        user_input_array.append(np.zeros(self.img.shape[:2]))
        iou_array.append(iou(gt, self.get_output_mask()))

        # User input simulation 
        for i in range(iterations-1):
            nr = user_input_simulation.get_negative_region(gt, self.get_output_mask())
            pr = user_input_simulation.get_positive_region(gt, self.get_output_mask())
            #print(pr.sum(), nr.sum())
            if pr.sum() > nr.sum():
                user_input = user_input_simulation.get_positive_line(gt, self.get_output_mask())
            else:
                user_input = user_input_simulation.get_negative_line(gt, self.get_output_mask())
            self.mask_iteration(user_input)
            
            mask_array.append(self.get_output_mask())
            image_array.append(self.get_output_image())
            user_input_array.append(user_input)
            iou_array.append(iou(gt, self.get_output_mask()))
        
        return image_array, iou_array, mask_array, user_input_array
