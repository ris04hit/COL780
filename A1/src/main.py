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
        count = [9, 14, 5, 9, 10, 10, 13, 8, 17, 16]

        ct = 0
        for img in img_arr:
            processed_img = helper.preprocess(img)
            edge_detected_img = helper.canny_edge_detector(processed_img)[10:-10, 10:-10]
            angle = helper.suture_angle(edge_detected_img)
            line_detected_img = helper.gradient_hough_transform(edge_detected_img, interpolate=1, suture_filter=1)
            
            labels = helper.connected_component(line_detected_img)
            image_components = helper.split_component(line_detected_img, labels)
            
            for mul_threshold in [3, 2.5, 2]:
                centroids = helper.centroid(line_detected_img, image_components)
                image_components = helper.filter_centroid(image_components, centroids, angle, mul_threshold=mul_threshold)
                
            centroids = helper.centroid(line_detected_img, image_components)
            centroid_detected_img = helper.insert_centroid(edge_detected_img, centroids)
            
            print(img_name_arr[ct], '\t', image_components.shape[0], '\t', count[ct])
            cv2.imwrite(os.path.join('temp/centroid/', img_name_arr[ct]), centroid_detected_img)
            ct += 1