import os
import pandas as pd

img_dir = 'data/'
img_name_arr = os.listdir(img_dir)
img_name_arr.sort()

data = []

for i in range(len(img_name_arr)):
    for j in range(len(img_name_arr)):
        if i != j:
            data.append({
                'img1_path': os.path.join(img_dir, img_name_arr[i]),
                'img2_path': os.path.join(img_dir, img_name_arr[j])
            })
    
pd.DataFrame(data).to_csv('input/input.csv', index=False)