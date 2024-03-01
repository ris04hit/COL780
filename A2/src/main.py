import cv2
import sys
import os
import numpy as np
import helper
import time

def save(out_path, img_arr, inp_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    dir = os.path.join(out_path, inp_path.split("/")[-1].split(".")[0])
    if not os.path.exists(dir):
        os.mkdir(dir)
    for i in range(img_arr.shape[0]):
        cv2.imwrite(os.path.join(dir, f"{i}.png"), img_arr[i])

if __name__ == "__main__":
    start_time = time.time()
    
    # Taking Input
    if sys.argv[1] == '1':
        img_dir = sys.argv[2]
        img_arr = [cv2.imread(os.path.join(img_dir, img_name)) for img_name in sorted(os.listdir(img_dir))]
        
    elif sys.argv[1] == '2':
        vid = cv2.VideoCapture(sys.argv[2])
        img_arr = []
        ret = True
        while ret:
            ret, frame = vid.read()
            img_arr.append(frame)
        img_arr = img_arr[:-1]
        vid.release()
        cv2.destroyAllWindows()

    pan_path = sys.argv[3]
    num_img_original = len(img_arr)
    print(f"Processing {pan_path}")
    
    # Taking partial input for videos
    partial_input = True
    num_img_ideal = 6
    if partial_input and sys.argv[1] == '2':
        off = lambda n: ((num_img_original-1) * n)//(num_img_ideal-1)
        img_arr = [img_arr[off(i)] for i in range(num_img_ideal)]
    
    img_arr = np.array(img_arr).astype(int)
    num_img = img_arr.shape[0]
    read_time = time.time()
    print(f"Input Time Taken:\t\t\t{read_time - start_time}")
    
    # Flag to toggle saving of intermediate images
    save_bool = True

    # Preprocessing of image
    preprocessed_img_arr = helper.preprocess(img_arr)
    preprocess_time = time.time()
    print(f"Preprocess Time Taken:\t\t\t{preprocess_time - read_time}")

    # Feature detection
    feature_detected_img_arr, keypoint_arr = helper.feature_detector(preprocessed_img_arr, mode = 'l', save=save_bool)
    feature_time = time.time()
    print(f"Feature Detection Time Taken:\t\t{feature_time - preprocess_time}")
    
    # Keypoint Descriptor Creation
    descriptor_list, keypoint_index_list = helper.feature_descriptor(preprocessed_img_arr, keypoint_arr, mode = 's')
    descriptor_time = time.time()
    print(f"Feature Descriptor Time Taken:\t\t{descriptor_time - feature_time}")
    
    # Matching Descriptors
    matched_coord = helper.match_coord(descriptor_list, keypoint_index_list)
    if save_bool:
        matched_img = helper.create_match_img(preprocessed_img_arr, matched_coord)
    matching_time = time.time()
    print(f"Descriptor Matching Time Taken:\t\t{matching_time - descriptor_time}")
    
    # Computing Homography
    homography = helper.apply_arr(matched_coord, helper.ransac_homography)
    homography_time = time.time()
    print(f"Homography Time Taken:\t\t\t{homography_time - matching_time}")
    
    # Saving intermediate images
    if save_bool:
        save('temp/preprocess', preprocessed_img_arr, pan_path)
        save('temp/feature', feature_detected_img_arr, pan_path)
        save('temp/match', matched_img, pan_path)
    
    pan_img = img_arr[np.random.randint(0, img_arr.shape[0])]
    end_time = time.time()
    print(f"Saving Time Taken:\t\t\t{end_time - homography_time}")
    
    # Saving Final Panorama
    cv2.imwrite(pan_path, pan_img)
    print(f"Processed {pan_path}")
    print(f"Total Time Taken:\t\t\t{end_time - start_time}")
    print()
    