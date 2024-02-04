import pandas as pd
import numpy as np
import sys

part = sys.argv[1]

file_original = sys.argv[2]
file_predicted = sys.argv[3]

df_original = pd.read_csv(file_original)
df_predicted = pd.read_csv(file_predicted)

if part == '1':
    img_name = df_original['image_name']
    
    count_original = df_original['number of sutures']
    count_predicted = df_predicted['number of sutures']
    
    num = len(df_original)
    
    score = 0
    
    for i in range(num):
        predicted = count_predicted[df_predicted['image_name'] == img_name[i]]
        predicted = predicted[predicted.first_valid_index()]
        if count_original[i] != predicted:
            print(f"Count\t\t{img_name[i]}\t\tPredicted:{predicted}\tOriginal:{count_original[i]}")
            score += 1
    
    print(f"Score:\t{score}")

if part == '2':

    img_path1 = df_original['img1_path']
    img_path2 = df_original['img2_path']

    dist_original = df_original['output_distance']
    angle_original = df_original['output_angle']
    dist_predicted = df_predicted['output_distance']
    angle_predicted = df_predicted['output_angle']

    score = 0
    dist_score = 0
    angle_score = 0

    num = len(df_original)

    for i in range(num):
        if dist_original[i] != dist_predicted[i]:
            print(f"Distance\t{img_path1[i]}\t{img_path2[i]}\t\tPredicted:{dist_predicted[i]}\tOriginal:{dist_original[i]}")
            score += 1
            dist_score += 1
        if angle_original[i] != angle_predicted[i]:
            print(f"Angle\t\t{img_path1[i]}\t{img_path2[i]}\t\tPredicted:{angle_predicted[i]}\tOriginal:{angle_original[i]}")
            score += 1
            angle_score += 1
            
    print(f"Score:\t{score}\t\tDistance_Score:\t{dist_score}\t\tAngle_score:\t{angle_score}")