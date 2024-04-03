import os
import time
import numpy as np
import pickle
import helper

no_box_dataset_addr = "data_no_box"
cropped_dataset = "data"
hog_addr = "temp/hog.npz"
pca_addr = "temp/pca.npz"
svm_addr = "temp/svm.pkl"
roc_addr = "roc"

start_time = time.time()
overwrite = False
save_bool = True

if save_bool and not os.path.exists('temp'):
    os.mkdir('temp')

# Creating Dataset
helper.create_dataset(no_box_dataset_addr, cropped_dataset, overwrite=overwrite)
print(f"Created Cropped Dataset\tTime: {time.time()-start_time}")

# Iterating through each folder of data
if overwrite or not os.path.exists(hog_addr):
    hog_arr_train, hog_arr_val = [], []
    label_arr_train, label_arr_val = [], []
    for fold_name in sorted(os.listdir(cropped_dataset)):
        fold_path = os.path.join(cropped_dataset, fold_name)
        label = 0 if fold_name[0] == 'c' else 1     # Treating Closed as zero and Open as 1
        dataset = helper.Dataset(fold_path)

        for img in dataset.train:
            hog = helper.hog(img)
            hog_arr_train.append(hog)
            label_arr_train.append(label)

        for img in dataset.val:
            hog = helper.hog(img)
            hog_arr_val.append(hog)
            label_arr_val.append(label)

    hog_arr_train = np.stack(hog_arr_train)
    hog_arr_val = np.stack(hog_arr_val)
    label_arr_train = np.array(label_arr_train)
    label_arr_val = np.array(label_arr_val)

    # Saving hog vectors
    if save_bool:
        np.savez_compressed(hog_addr, desc_train=hog_arr_train, desc_val=hog_arr_val, lab_train=label_arr_train, lab_val=label_arr_val)

else:
    # Loading hog vectors
    hog_vec = np.load(hog_addr)
    hog_arr_train = hog_vec['desc_train']
    hog_arr_val = hog_vec['desc_val']
    label_arr_train = hog_vec['lab_train']
    label_arr_val = hog_vec['lab_val']

print(f"Created HOG Descriptors\tTime: {time.time()-start_time}")

# PCA to reduce dimension
pca = helper.PCA()
if overwrite or not os.path.exists(pca_addr):
    hog_arr_train = pca.fit(hog_arr_train)
    if save_bool:
        pca.save(pca_addr)
else:
    pca.load(pca_addr)
    hog_arr_train = pca.transform(hog_arr_train)
hog_arr_val = pca.transform(hog_arr_val)

print(f"Reduced Descriptor Dimension to {pca.red_dim}\tTime: {time.time()-start_time}")

# Training SVM
if overwrite or not os.path.exists(svm_addr):
    svm = helper.SVM()
    svm.train(hog_arr_train, label_arr_train)
    if save_bool:
        with open(svm_addr, 'wb') as file:
            pickle.dump(svm, file)
else:
    with open(svm_addr, 'rb') as file:
        svm = pickle.load(file)
score_train, pred_train = svm.predict(hog_arr_train)
score_val, pred_val = svm.predict(hog_arr_val)

print("Train:", helper.metric(label_arr_train, pred_train))
print("Validation:", helper.metric(label_arr_val, pred_val))
print("ROCAUC:", helper.roc(score_val, label_arr_val, roc_addr))