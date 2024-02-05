import sys
import os
import cv2
import pandas as pd
import numpy as np
import helper

if __name__ == '__main__':
    part_id = sys.argv[1]
    img_dir = sys.argv[2]       # Or input csv file for part2
    output_csv = sys.argv[3]
    save = False
    
    # Part based image input
    if part_id == '1':
        img_name_arr = os.listdir(img_dir)
        img_name_arr.sort()
        img_arr = [cv2.imread(os.path.join(img_dir, img_name)) for img_name in img_name_arr]
    elif part_id == '2':
        inp_df = pd.read_csv(img_dir)
        img_name_arr = inp_df['img1_path'].to_list()
        img_name_arr.extend(inp_df['img2_path'].to_list())
        img_name_arr = list(set(img_name_arr))
        img_arr = [cv2.imread(img_path) for img_path in img_name_arr]
        output_data = {}

    csv_data = []
        
    ct = 0
    for img in img_arr:
        # Variable for thickness control
        thickness = 1
        grad_precision = 3
        
        for cycle in range(2):      # First iteration is dry run for setting the thickness value
            # Preprocessing image by gray scaling, gaussian smoothening, increasing contrast and thresholding
            processed_img = helper.preprocess(img)
            
            # Edges detected using canny edge detector
            edge_detected_img = helper.canny_edge_detector(processed_img)
            
            # Thickens binary image for avoiding ignoring of thin edges
            edge_detected_img = helper.thick_image(helper.crop_image(edge_detected_img), size=thickness)
            
            # Calculated gradient and related parameterss for edge detected image
            grad_x, grad_y, grad, theta = helper.calc_grad(edge_detected_img)
            
            # Calculated approximate mean angle of normal to all sutures
            angle = helper.suture_angle(grad, theta)
            
            # Detected lines using hough transform with gradient heurestic
            # Interpolated lines between points found on same line
            # Performed filter based on mean angle calculated earlier
            line_detected_img = helper.gradient_hough_transform(edge_detected_img, angle, grad_x, grad_y, interpolate=1, suture_filter=1, grad_precision=grad_precision)
            
            # Used two pass algorithm to find different connected components of image after hough transform
            labels = helper.connected_component(line_detected_img)
            image_components = helper.split_component(line_detected_img, labels)
                
            # Combined different components very close to each other
            # Repeated the step multiple time by varying threshold parameter for closeness
            for mul_threshold in [3, 2.5, 2]:
                centroids = helper.centroid(line_detected_img, image_components)
                image_components = helper.filter_centroid(image_components, centroids, angle, mul_threshold=mul_threshold)

            
            # Calculated centroids of all components
            centroids = helper.centroid(line_detected_img, image_components)
            
            # Calculating various stastical results based on problem statement
            # For angle calculation used only the pixels present in edge detected image
            if part_id == '1':
                inter_suture_spacing = helper.spacing_centroid(centroids, angle, euclidean_only=True)/img.shape[0]
                suture_angles = helper.component_angle(image_components*helper.binary_img(edge_detected_img), grad, theta)
            if part_id == '2':
                inter_suture_spacing = helper.spacing_centroid(centroids, angle)/edge_detected_img.shape[0]
                normalized_theta = 180*np.arctan2(grad_y/np.mean(inter_suture_spacing), grad_x)/np.pi
                suture_angles = helper.component_angle(image_components*helper.binary_img(edge_detected_img), grad, normalized_theta)
        
            # Setting value of thickness
            mean_spacing = np.mean(helper.spacing_centroid(centroids, angle))
            if mean_spacing >= 50:
                thickness = 3
            elif mean_spacing >= 25:
                thickness = 2
            else:
                grad_precision = 2
                continue
        
        # Saving images
        if save:
            centroid_detected_img = helper.insert_centroid(edge_detected_img, centroids)
            cv2.imwrite(os.path.join('temp/processed/', img_name_arr[ct]), processed_img)
            cv2.imwrite(os.path.join('temp/edge/', img_name_arr[ct]), edge_detected_img)
            cv2.imwrite(os.path.join('temp/gradx/', img_name_arr[ct]), 255*grad_x/np.max(grad_x))
            cv2.imwrite(os.path.join('temp/grady/', img_name_arr[ct]), 255*grad_y/np.max(grad_y))
            cv2.imwrite(os.path.join('temp/line/', img_name_arr[ct]), line_detected_img)
            cv2.imwrite(os.path.join('temp/centroid/', img_name_arr[ct]), centroid_detected_img)
        
        # Preparing output data based on parts
        if part_id == '1':
            csv_data.append({
                'image_name': img_name_arr[ct],
                'number of sutures': centroids.shape[0],
                'mean inter suture spacing': np.mean(inter_suture_spacing),
                'variance of inter suture spacing': np.var(inter_suture_spacing),
                'mean suture angle wrt x-axis': np.mean(suture_angles),
                'variance of suture angle wrt x-axis': np.var(suture_angles)
            })
        if part_id == '2':
            output_data[img_name_arr[ct]] = (np.std(inter_suture_spacing)/np.mean(inter_suture_spacing), np.var(suture_angles))
        
        # Log
        print(f'Processed {img_name_arr[ct]}')
        ct += 1
    
    # Part based output
    if part_id == '2':
        for ind, row in inp_df.iterrows():
            img1_path = row['img1_path']
            img2_path = row['img2_path']
            img1_stat = output_data[img1_path]
            img2_stat = output_data[img2_path]
            csv_data.append({
                'img1_path': img1_path,
                'img2_path': img2_path,
                'output_distance': 1 if img1_stat[0] < img2_stat[0] else 2,
                'output_angle': 1 if img1_stat[1] < img2_stat[1] else 2
            })

    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv, index=False)