import sys
import os
import cv2
import numpy as np
import helper

if __name__ == '__main__':
    part_id = sys.argv[1]
    img_dir = sys.argv[2]
    output_csv = sys.argv[3]
    
    # Part 1
    if part_id == '1':
        img_name_arr = os.listdir(img_dir)
        img_name_arr.sort()
        img_arr = [cv2.imread(os.path.join(img_dir, img_name)) for img_name in img_name_arr]

        ct = 0
        for img in img_arr:
            processed_img = helper.preprocess(img)
            edge_detected_img = helper.canny_edge_detector(processed_img)[10:-10, 10:-10]
            corner_detected_img = helper.harris_corner_detector(edge_detected_img)
            labels = helper.connected_component(edge_detected_img)
            image_components = helper.split_component(edge_detected_img, labels)
            sutures = [helper.count_suture(component, corner_detected_img)//2 for component in image_components]
            num_sutures = sum(sutures)
            print(img_name_arr[ct], num_sutures, sutures)
            # cv2.imwrite(os.path.join('temp/edge/', img_name_arr[ct]), edge_detected_img)
            # cv2.imwrite(os.path.join('temp/corner/', img_name_arr[ct]), corner_detected_img)
            if ct==2:
                for i in range(image_components.shape[0]):
                    if len(image_components.shape) == 3:
                        component = image_components[i, :, :]
                    else:
                        component = image_components[i, :, :, :]
                    cv2.imwrite(os.path.join('temp', f'{i}_{img_name_arr[ct]}'), component)
            ct += 1
            if ct==3:
                break