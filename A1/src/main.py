import sys
import os
import cv2
import pandas as pd
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
        
        csv_data = []

        ct = 0
        for img in img_arr:
            processed_img = helper.preprocess(img)
            edge_detected_img = helper.canny_edge_detector(processed_img)[10:-10, 10:-10]
            
            grad_x, grad_y, grad, theta = helper.calc_grad(edge_detected_img)
            
            angle = helper.suture_angle(grad, theta)
            line_detected_img = helper.gradient_hough_transform(edge_detected_img, angle, grad_x, grad_y, interpolate=1, suture_filter=1)
            
            labels = helper.connected_component(line_detected_img)
            image_components = helper.split_component(line_detected_img, labels)
            
            for mul_threshold in [3, 2.5, 2]:
                centroids = helper.centroid(line_detected_img, image_components)
                image_components = helper.filter_centroid(image_components, centroids, angle, mul_threshold=mul_threshold)
                
            centroids = helper.centroid(line_detected_img, image_components)
            
            inter_suture_spacing = helper.spacing_centroid(centroids, angle)/img.shape[0]
            suture_angles = helper.component_angle(image_components*helper.binary_img(edge_detected_img), grad, theta)
            
            csv_data.append({
                'image_name': img_name_arr[ct],
                'number of sutures': centroids.shape[0],
                'mean inter suture spacing': np.mean(inter_suture_spacing),
                'variance of inter suture spacing': np.var(inter_suture_spacing),
                'mean suture angle wrt x-axis': np.mean(suture_angles),
                'variance of suture angle wrt x-axis': np.var(suture_angles)
            })
            
            print(f'Processed {img_name_arr[ct]}')
            ct += 1
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_csv, index=False)
        
    # Part 2
    if part_id == '2':
        pass