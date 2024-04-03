import os
import shutil
import cv2
import mediapipe as mp
import numpy as np

# Dataset
class Dataset():
    def __init__(self, address):
        self.train = Dataset_iterator(os.path.join(address, 'train'))
        self.val = Dataset_iterator(os.path.join(address, 'valid'))

class Dataset_iterator():
    def __init__(self, folder_path):
        self.fold = folder_path
        self.img_list = sorted(os.listdir(folder_path))

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.img_list):
            raise StopIteration
        img_path = os.path.join(self.fold, self.img_list[self.idx])
        img = cv2.imread(img_path)
        self.idx += 1
        return np.array(img)

# Create Dataset According to ROI
def create_dataset(inp_add, out_add, overwrite = False):
    if not overwrite and os.path.exists(out_add):
        return
    
    if os.path.exists(out_add):
        shutil.rmtree(out_add)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=True, 
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    os.mkdir(out_add)

    for folder1_name in sorted(os.listdir(inp_add)):        # closed or open
        # Input Folder
        folder1_address = os.path.join(inp_add, folder1_name)

        # Output Folder
        folder1_address_out = os.path.join(out_add, folder1_name)
        os.mkdir(folder1_address_out)

        for folder2_name in sorted(os.listdir(folder1_address)):    # Train or Val
            # Inpur Folder
            folder2_address = os.path.join(folder1_address, folder2_name)

            # Output Folder
            folder2_address_out = os.path.join(folder1_address_out, folder2_name)
            os.mkdir(folder2_address_out)

            # Each File
            for filename in sorted(os.listdir(folder2_address)):
                if filename.endswith(".jpg"):
                    # Load the image
                    img_path = os.path.join(folder2_address, filename)
                    img = cv2.imread(img_path)
                    img = cv2.flip(img, 1)
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    results = hands.process(imgRGB)

                    # If hand landmark detected
                    if results.multi_hand_landmarks:
                        ct = 0
                        for hand_landmarks in results.multi_hand_landmarks:
                            
                            landmark_points = []
                            for landmark in hand_landmarks.landmark:
                                x = int(landmark.x * img.shape[1])
                                y = int(landmark.y * img.shape[0])
                                landmark_points.append([x, y])
                        
                            landmark_points = np.array(landmark_points)  
                            x, y, w, h = cv2.boundingRect(landmark_points) 
                            scale_factor = 1.4
                            delta_w = int((scale_factor - 1) * w / 2)
                            delta_h = int((scale_factor - 1) * h / 2)
                            x = max(0, x-delta_w)
                            y = max(0, y-delta_h)
                            w += 2*delta_w
                            h += 2*delta_h

                            cropped_img = img[x:x+w, y:y+h, :]
                            if np.sum(cropped_img) == 0:
                                print("Ignoring Empty Image")
                                continue
                            cropped_img = cv2.resize(cropped_img, (64, 128))

                            output_path = os.path.join(folder2_address_out, f'{filename[:filename.index(".")]}_{ct}.jpg')
                            cv2.imwrite(output_path, cropped_img)

                            ct += 1

# Preprocessing
def preprocess(img):
    # grayscale conversion
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # noise reduction
    img = cv2.GaussianBlur(img, (5, 5), 0)

    return img

# Calculating Gradients
def grad(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad_dir = np.arctan2(grad_y, grad_x)%np.pi
    return grad, grad_dir

# finding HOG descriptor
def hog(img, cell_size = 8, num_bin = 9, block_size = 2):
    img = np.copy(img)
    
    # Preprocessing
    img = preprocess(img)

    # Gradient Calculation
    grad_mag, grad_dir = grad(img)

    # Creating Grad Histograms
    histogram = np.zeros((img.shape[0]//cell_size, img.shape[1]//cell_size, num_bin))
    bin_size = np.pi/num_bin
    for y in range(0, img.shape[0], cell_size):
        for x in range(0, img.shape[1], cell_size):
            hist = np.zeros((num_bin,))
            for j in range(cell_size):
                for i in range(cell_size):
                    hist[int(grad_dir[y+j, x+i]//bin_size)] += grad_mag[y+j, x+i]
            histogram[y//cell_size, x//cell_size, :] = hist

    # Block Normalization
    sq_sum = np.mean(np.square(histogram), axis=-1)
    hist_vec = []
    for y in range(0, histogram.shape[0]-block_size, block_size//2):
        for x in range(0, histogram.shape[1]-block_size, block_size//2):
            norm = np.sqrt(np.mean(sq_sum[y:y+block_size, x:x+block_size]))
            if not norm:
                norm = 1
            hist_vec.append(histogram[y:y+block_size, x:x+block_size]/norm)
    hist_vec = np.stack(hist_vec).reshape(-1)

    return hist_vec.astype(np.float32)

# PCA
class PCA():
    def __init__(self, reduced_dim=1024):
        self.red_dim = reduced_dim
    
    # Finding transformation as per train data
    def fit(self, train_data):
        self.mean = np.mean(train_data, axis=0)
        centered_data = train_data - self.mean

        # Calculating Covariance Matrix
        cov_matrix = np.cov(centered_data, rowvar=False)

        # Eigen Decomposition
        eigen_val, eigen_vec = np.linalg.eig(cov_matrix)
        self.eigen_vec = eigen_vec[:, np.argsort(-eigen_val)][:, :self.red_dim]

        return self.transform(train_data)

    # Transforming Data
    def transform(self, data):
        return np.matmul(data-self.mean, self.eigen_vec)

    # Saving PCA
    def save(self, address):
        np.savez_compressed(address, mean=self.mean, eig_vec=self.eigen_vec)

    # Loading PCA
    def load(self, address):
        pca = np.load(address)
        self.mean = pca['mean']
        self.eigen_vec = pca['eig_vec']

# SVM
class SVM():
    def __init__(self):
        self.svm = cv2.ml.SVM.create()
        self.svm.setKernel(cv2.ml.SVM_LINEAR)
        self.svm.setType(cv2.ml.SVM_C_SVC)

    def train(self, data, labels):
        data = data.astype(np.float32)
        self.svm.train(data, cv2.ml.ROW_SAMPLE, labels)
    
    def predict(self, data):
        data = data.astype(np.float32)
        return self.svm.predict(data)[1].reshape(-1)
    
# Calculating Accuracy
def accuracy(y_orig, y_pred):
    return np.count_nonzero(y_orig == y_pred)/y_orig.shape[0]