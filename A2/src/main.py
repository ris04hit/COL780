import cv2
import sys
import os
import numpy as np
import helper

if __name__ == "__main__":
    # Taking Input
    if sys.argv[1] == '1':
        img_dir = sys.argv[2]
        img_arr = [cv2.imread(os.path.join(img_dir, img_name)) for img_name in os.listdir(img_dir)]
        
    elif sys.argv[1] == '2':
        vid = cv2.VideoCapture(sys.argv[2])
        img_arr = []
        ret = True
        while ret:
            ret, frame = vid.read()
            img_arr.append(frame)
        img_arr = img_arr[:-1]
        
    pan_path = sys.argv[3]
    img_arr = np.array(img_arr)
    
    pan_img = np.concatenate(img_arr, axis=1)
    cv2.imwrite(pan_path, pan_img)
    