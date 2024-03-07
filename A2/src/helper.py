import numpy as np
import cv2

# Implementing KDTree
class KDNode():
    def __init__(self, vec_arr, dim, num_leaf, parent, ind):
        self.leaf = True if vec_arr.shape[0] <= num_leaf else False
        self.parent = parent
        if not self.leaf:
            init_dim = dim
            while True:
                self.val = np.median(vec_arr[:, dim])
                left_mask = vec_arr[:, dim] <= self.val
                right_mask = vec_arr[:, dim] > self.val
                if np.count_nonzero(left_mask) == 0 or np.count_nonzero(right_mask) == 0:
                    dim = (dim + 1)%vec_arr.shape[1]
                else:
                    self.left = KDNode(vec_arr[left_mask], (dim+1)%vec_arr.shape[1], num_leaf, self, ind[left_mask])
                    self.right = KDNode(vec_arr[right_mask], (dim+1)%vec_arr.shape[1], num_leaf, self, ind[right_mask])
                    self.dim = dim
                    return
                if dim == init_dim:
                    self.leaf = True
                    break
        # When self.leaf is True
        self.val = vec_arr
        self.ind = ind
    
    def next(self, vec):
        if vec[self.dim] <= self.val:
            return self.left
        else:
            return self.right
        
    def inv_next(self, vec):
        if vec[self.dim] <= self.val:
            return self.right
        else:
            return self.left
    
    def all_vec(self):
        if self.leaf:
            return self.val
        return np.concatenate([self.left.all_vec(), self.right.all_vec()])
    
    def all_ind(self):
        if self.leaf:
            return self.ind
        return np.concatenate([self.left.all_ind(), self.right.all_ind()])

    def near2(self, vec, norm):
        vec_arr = self.all_vec()
        ind_arr = self.all_ind()
        diff = np.sum(np.abs(vec_arr - vec)**norm, axis = 1)**(1/norm)
        ind = np.argsort(diff)[:2]
        return np.stack([diff[ind], ind_arr[ind]])
    
    def search2(self, vec, norm):
        if self.leaf:
            return self.near2(vec, norm)
        
        next_node = self.next(vec)
        curr_best = next_node.search2(vec, norm)
        
        box_dist = np.abs(vec[self.dim] - self.val)
        if box_dist > curr_best[0, -1]:
            return curr_best
        
        inv_next_node = self.inv_next(vec)
        possible_best = inv_next_node.near2(vec, norm)
        
        curr_best = np.concatenate([curr_best, possible_best], axis = 1)
        curr_best = curr_best[:, np.argsort(curr_best[0])[:2]]
        
        return curr_best

class KDTree():
    def __init__(self, vec_arr, num_leaf = 10, norm = 1):
        self.dim = vec_arr.shape[1]
        self.norm = norm
        self.root = KDNode(vec_arr, 0, num_leaf, None, np.arange(vec_arr.shape[0]))
       
    # Returns two nearest neighbor
    def query2(self, vec):
        return self.root.search2(vec, self.norm)


# Function to create a gaussian filter matrix of desired dimension
def gaussian_filter_kernel(n, sigma = 1):
    k = (n-1)//2
    filter_exp = np.array([[(i-k)**2 + (j-k)**2 for j in range(2*k+1)] for i in range(2*k+1)])
    filter_val = np.exp(-filter_exp/(2*sigma**2))
    return filter_val/np.sum(filter_val)

# Laplacian Kernel
def laplacian_kernel():
    return np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ])

# Function to create a laplacian of gaussian filter matrix of desired dimension
def log_kernel(n, sigma = 1):
    kernel = convolve2d(laplacian_kernel(), gaussian_filter_kernel(n-2, sigma=sigma), complete=True)
    return kernel

# Function to compute local maxima in 3d array
def local_maxima(R):
    # Axis 1
    M1 = np.zeros_like(R, dtype=bool)
    M1[0, :, :] = np.max(R[:2, :, :], axis=0) == R[0, :, :]
    M1[-1, :, :] = np.max(R[-2:, :, :], axis=0) == R[-1, :, :]
    for i in range(1, R.shape[0]-1):
        M1[i, :, :] = np.max(R[i-1:i+2, :, :], axis=0) == R[i, :, :]
        
    # Axis 2
    M2 = np.zeros_like(R, dtype=bool)
    M2[:, 0, :] = np.max(R[:, :2, :], axis=1) == R[:, 0, :]
    M2[:, -1, :] = np.max(R[:, -2:, :], axis=1) == R[:, -1, :]
    for i in range(1, R.shape[1]-1):
        M2[:, i, :] = np.max(R[:, i-1:i+2, :], axis=1) == R[:, i, :]
        
    # Axis 3
    M3 = np.zeros_like(R, dtype=bool)
    M3[: , :, 0] = np.max(R[:, :, :2], axis=2) == R[:, :, 0]
    M3[:, :, -1] = np.max(R[:, :, -2:], axis=2) == R[:, :, -1]
    for i in range(1, R.shape[2]-1):
        M3[:, :, i] = np.max(R[:, :, i-1:i+2], axis=2) == R[:, :, i]
    
    M = M1 & M2 & M3
    return M

# Function to perform 2d convolution on image
def convolve2d_color(img, kernel, complete = False):
    ker_y, ker_x = kernel.shape
    img_y, img_x, color_ct = img.shape
    row_y_left, col_x_left = (ker_y-1)//2, (ker_x-1)//2
    convolved_img = np.zeros((img_y + ker_y - 1, img_x + ker_x - 1, color_ct))
    for i in range(ker_y):
        for j in range(ker_x):
            convolved_img[i:i+img_y, j:j+img_x, :] += kernel[i, j]*img
    if not complete:
        convolved_img = convolved_img[row_y_left:row_y_left+img_y, col_x_left:col_x_left+img_x, :]
    return convolved_img

# Function to perform 2d convolution on image
def convolve2d(img, kernel, complete = False):
    ker_y, ker_x = kernel.shape
    img_y, img_x = img.shape
    row_y_left, col_x_left = (ker_y-1)//2, (ker_x-1)//2
    convolved_img = np.zeros((img_y + ker_y - 1, img_x + ker_x - 1))
    for i in range(ker_y):
        for j in range(ker_x):
            convolved_img[i:i+img_y, j:j+img_x] += kernel[i, j]*img
    if not complete:
        convolved_img = convolved_img[row_y_left:row_y_left+img_y, col_x_left:col_x_left+img_x]
    return convolved_img

# Function to perform another function with array of images
def apply_arr(img_arr, func, *args, np_conv = True, dynamic = False):
    return_arr = []
    if dynamic:
        size = len(img_arr) if type(img_arr) != np.ndarray else img_arr.shape[0]
        args = [[args[j][i] for j in range(len(args))] for i in range(size)]
        for i in range(size):
            return_arr.append(func(img_arr[i], *(args[i])))
    else:
        for img in img_arr:
            return_arr.append(func(img, *args))
    if np_conv:
        return np.array(return_arr)
    return return_arr

# Calculates function gradient
def gradient(img):
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]])
    
    conv_func = convolve2d_color if len(img.shape) == 3 else convolve2d
    
    I_x = conv_func(img, sobel_x)
    I_y = conv_func(img, sobel_y)
    
    return I_x, I_y

# Change resolution of image
def set_resolution(img, inp_smoothen = False, out_smoothen = True, kernel_size = 5):
    # Finding correct convolution function
    convolve_func = convolve2d if len(img.shape)==2 else convolve2d_color
    
    # Gaussian smoothing before setting resolution
    if inp_smoothen:
        img = convolve_func(img, gaussian_filter_kernel(kernel_size))
        
    # Reducing resolution
    img = img[::2, ::2] if len(img.shape) == 2 else img[::2, ::2, :]
    
    # Gaussian smoothing after setting resolution
    if out_smoothen:
        img = convolve_func(img, gaussian_filter_kernel(kernel_size))
    
    return img


# Function to convert image to gray scale image
def gray_scale(img):
    return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

# Changes the contrast by given value 
def set_contrast(img, c):
    f = (259*(c+255))/(255*(259-c))
    contrast_img = (f*(img-128)+128)
    contrast_img[contrast_img < 0] = 0
    contrast_img[contrast_img > 256] = 256
    return contrast_img.astype(int)

# Preprocessing (Contrast Adjustment)
def preprocess(img_arr, contrast = 50, kernel_size = 5, color = True):
    # Gray scale
    if not color:
        img_arr = apply_arr(img_arr, gray_scale)
    
    # Setting contrast
    img_arr = set_contrast(img_arr, contrast)
    
    # Gaussian Smoothening
    if color:
        img_arr = apply_arr(img_arr, convolve2d_color, gaussian_filter_kernel(kernel_size))
    else:
        img_arr = apply_arr(img_arr, convolve2d, gaussian_filter_kernel(kernel_size))
    
    return img_arr


# Harris Corner Detector
def harris_corner_detector(img, window_size = 3, k = 0.06, t = 1e9):
    img = np.copy(img)
    
    # Calculates gradient
    I_x, I_y = gradient(img)
    
    # Calculating summed matrix
    gaussian_kernel = gaussian_filter_kernel(window_size)
    I_x2 = convolve2d(I_x**2, gaussian_kernel)
    I_y2 = convolve2d(I_y**2, gaussian_kernel)
    I_xy = convolve2d(I_x*I_y, gaussian_kernel)
    
    # Calculating score R
    R = I_x2*I_y2 - I_xy*I_xy - k*(I_x2 + I_y2)**2

    # Calculating keypoints
    keypoint = R>t
    
    return keypoint

# Laplace Harris Feature Detector
def laplace_harris_detector(img, window_size = 5, k = 0.04, max_t = 1e8, min_t = 1e6, edge_t = 5, num_scale = 3, scale_factor = 1.5, num_pt = 2000, num_comp = (2, 1)):
    img = np.copy(img)
    color = True if len(img.shape) == 3 else False
    
    # Calculates gradient
    I_x, I_y = gradient(img)
    
    # Calculating summed matrix
    R_arr = []
    conv_func = convolve2d_color if color else convolve2d
    for sigma in [scale_factor**i for i in range(num_scale)]:
        kernel = log_kernel(window_size, sigma=sigma)
        I_x2 = conv_func(I_x**2, kernel)
        I_y2 = conv_func(I_y**2, kernel)
        I_xy = conv_func(I_x*I_y, kernel)
    
        # Calculating score R
        R = I_x2*I_y2 - I_xy*I_xy - k*(I_x2 + I_y2)**2
        if color:
            R = np.sum(R, axis = -1)
        
        R_arr.append(R)
        
    # Calculating local maxima
    R = np.stack(R_arr)
    max_R = local_maxima(R)
    
    # Splitting image into blocks and thresholding per block
    keypoint_global = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    block_shape = (img.shape[0]//num_comp[0], img.shape[1]//num_comp[1])
    for y in range(0, img.shape[0], block_shape[0]):
        for x in range(0, img.shape[1], block_shape[1]):
            local_t = max_t
            keypoint_local = np.zeros(block_shape, dtype=bool)
            max_R_local = max_R[:, y:y+block_shape[0], x:x+block_shape[1]]
            R_local = R[:, y:y+block_shape[0], x:x+block_shape[1]]
            while np.count_nonzero(keypoint_local) <= num_pt and local_t >= min_t:
                keypoint_local = max_R_local & (R_local > local_t)
                keypoint_local = np.max(keypoint_local, axis = 0)
                
                local_t /= 1.1
            
            keypoint_global[y:y+block_shape[0], x:x+block_shape[1]] = keypoint_local
                
    # Cropping features at edge
    keypoint_global[:edge_t, :] = False
    keypoint_global[-edge_t:, :] = False
    keypoint_global[:, :edge_t] = False
    keypoint_global[:, -edge_t:] = False
    

    return keypoint_global


# Insert Keypoint into img
def insert_keypoint(img, keypoint, thickness = 11, log = True,):
    img = np.copy(img)
    
    if log:
        print("Descriptor Count:\t", np.count_nonzero(keypoint))
    
    # Thickening Keypoint
    ker_size = (thickness+1)//2
    keypoint = convolve2d(keypoint, np.ones((ker_size, ker_size))).astype(bool)
    
    # Inserting Keypoint
    if len(img.shape) == 2:
        img[keypoint] = -128*np.sign(img[keypoint] - 128) + 128
    else:
        for i in range(img.shape[2]):
            img[:, :, i][keypoint] = -128*np.sign(img[:, :, i][keypoint] - 128) + 128 
    
    return img

# Feature detection
def feature_detector(img_arr, mode, low_resolution = 0, save = False):
    '''
    mode = 'h': Harris Corner Detector
    mode = 'l': Laplace Harris
    '''
    if mode == 'h':
        func = lambda img: harris_corner_detector(img)
    if mode == 'l':
        func = lambda img: laplace_harris_detector(img)
        
    inp_img_arr = np.copy(img_arr)
    for _ in range(low_resolution):
        inp_img_arr = apply_arr(inp_img_arr, set_resolution)

    keypoint = apply_arr(inp_img_arr, func)
    if low_resolution:
        _keypoint = np.zeros(img_arr.shape[:3], dtype=bool)
        _keypoint[:, ::(1<<low_resolution), ::(1<<low_resolution)] = keypoint
        keypoint = _keypoint

    if save:
        img_arr = apply_arr(img_arr, insert_keypoint, keypoint, dynamic=True)
    else:
        img_arr = []
    return img_arr, keypoint


# SIFT
def sift(grad_norm, grad_angle, ill_threshold = 0.2):
    histogram_arr = []
    grad_class = (4*grad_angle/np.pi).astype(int) % 8   # Calculating bin of hist to which particular pixel belongs
    
    for i in range(0, 16, 4):
        for j in range(0, 16, 4):
            hist = np.zeros((8,))
            hist[grad_class[i:i+4, j:j+4]] += grad_norm[i:i+4, j:j+4]
            histogram_arr.append(hist)
            
    histogram_arr = np.concatenate(histogram_arr)
    
    if np.sqrt(np.sum(np.square(histogram_arr))) != 0:
        histogram_arr = histogram_arr/np.sqrt(np.sum(np.square(histogram_arr)))      # Normalizing
        histogram_arr[histogram_arr > ill_threshold] = ill_threshold                 # Illumination invariance
        histogram_arr = histogram_arr/np.sqrt(np.sum(np.square(histogram_arr)))      # Normalizing again
    
    return histogram_arr

# Apply SIFT on multiple keypoints
# Assumes keypoint to be far from edge of image and image to be much larger than patch
def sift_arr(img, keypoint):
    Ix, Iy = gradient(img)
    grad_angle = np.arctan2(Iy, Ix)
    grad_norm = np.sqrt(Ix**2+Iy**2)
    keypoint_index = np.argwhere(keypoint)
    descriptor_arr = []
    for ind in keypoint_index:
        y, x = ind
        grad_angle_patch = grad_angle[y-8:y+8, x-8:x+8]
        grad_norm_patch = grad_norm[y-8:y+8, x-8:x+8]
        descriptor = sift(grad_norm_patch, grad_angle_patch)
        descriptor_arr.append(descriptor)
    descriptor_arr = np.stack(descriptor_arr)
    return descriptor_arr

# Feature descriptor
def feature_descriptor(img_arr, keypoint_arr, mode):
    '''
    mode = 's': SIFT
    '''
    if mode == 's':
        func = sift_arr
    
    color = True if len(img_arr.shape) == 4 else False
    img_arr = np.copy(img_arr)
    if color:
        img_arr = apply_arr(img_arr, gray_scale)
    descriptor_arr = apply_arr(img_arr, func, keypoint_arr, np_conv=False, dynamic=True)
    keypoint_index_arr = []
    for keypoint in keypoint_arr:
        keypoint_index_arr.append(np.argwhere(keypoint))
    return descriptor_arr, keypoint_index_arr


# Finding corresponding descriptor in img
def find_descriptor(keypt_descriptor, img_descriptor_arr, ratio_threshold, norm = 1, ret = 0):
    '''
    ret = 0: returns Index
    ret = 1: returns Descriptor
    norm: Describes which powered norm to be taken for distance calculation
    '''
    norm_descriptor_arr = np.sum(np.abs(img_descriptor_arr - keypt_descriptor)**norm, axis = 1)**(1/norm)
    first_ind, second_ind = np.argpartition(norm_descriptor_arr, 2)[[0, 1]]
    if norm_descriptor_arr[second_ind]:
        ratio = norm_descriptor_arr[first_ind]/norm_descriptor_arr[second_ind]
    else:
        return np.nan       # If both first and second best are identical
    if ratio > 1:
        raise Exception("argpartition is invalid")      # Something buggy with code
    if ratio > ratio_threshold:
        return np.nan       # If first and second best are quite similar
    if ret:
        return norm_descriptor_arr[first_ind]
    return first_ind

# Matches descriptors of arrays
def match_descriptor(descriptor_arr1, descriptor_arr2, mode = 't', ratio_threshold = 0.7):
    '''
    mode = 'n': Matching using norm
    mode = 't': Matching using kd tree
    '''
    if mode == 'n':
        desc_ind1 = apply_arr(descriptor_arr1, lambda descriptor: find_descriptor(descriptor, descriptor_arr2, ratio_threshold))
        desc_ind1_arg = np.argwhere(~np.isnan(desc_ind1)).reshape((-1,))
        desc_ind1_without_nan = desc_ind1[~np.isnan(desc_ind1)].astype(int)
        desc_ind2 = apply_arr(descriptor_arr2[desc_ind1_without_nan], lambda descriptor: find_descriptor(descriptor, descriptor_arr1, ratio_threshold))
        match_arg = desc_ind2 == desc_ind1_arg
        matching = np.stack([desc_ind1_arg[match_arg], desc_ind1_without_nan[match_arg]])
        return matching
    
    elif mode == 't':
        kdtree = KDTree(descriptor_arr2)
        result = apply_arr(descriptor_arr1, lambda descriptor: kdtree.query2(descriptor))
        distance, index = result[:, 0, :], result[:, 1, :].astype(int)
        ratio = np.divide(distance[:, 0],distance[:, 1], np.ones_like(distance[:, 0]), where = distance[:, 1] != 0)
        del kdtree
        return np.copy(np.stack([np.arange(descriptor_arr1.shape[0]), index[:, 0]])[:, ratio < ratio_threshold])
    
    else:
        raise Exception("Invalid mode")

# Coordinate Correspondence
def match_coord(desciptor_list, keypoint_index_list):
    num_img = len(desciptor_list)
    if num_img < 1:
        raise Exception("No image")
    
    matching_list = []
    for ind in range(num_img-1):
        corresp_desc = match_descriptor(desciptor_list[ind], desciptor_list[ind+1])
        coord1 = keypoint_index_list[ind][corresp_desc[0, :]]
        coord2 = keypoint_index_list[ind+1][corresp_desc[1, :]]
        matching_list.append(np.stack([coord1, coord2], axis=1))

    return matching_list

# Implements bresenham algorithm
def bresenham_line(c0, c1):
    x0, y0 = c0
    x1, y1 = c1
    
    points = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    steep = dy > dx

    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = abs(y1 - y0)
    error = int(dx / 2)
    ystep = 1 if y0 < y1 else -1
    y = y0

    for x in range(x0, x1 + 1):
        coord = (y, x) if steep else (x, y)
        points.append(coord)
        error -= dy
        if error < 0:
            y += ystep
            error += dx

    return np.array(points)

# Create Image to visualize matching points
def create_match_img(img_arr, matching_coord, f = 0.5):
    img_arr = np.copy(img_arr)
    color = True if len(img_arr.shape) == 4 else False
    num_img = img_arr.shape[0]
    if num_img < 2:
        raise Exception("Less than 2 images")
    
    # Different array for different size offset
    img_size = np.array(np.shape(img_arr[0]))
    if color:
        img_size = img_size[:-1]
    img_size0 = np.array([img_size[0], 0])
    img_size1 = np.array([0, img_size[1]])
    
    new_img_arr = []
    # Loop for image pairs
    for ind in range(num_img - 1):
        # Creating new image via concatenation
        img1 = np.concatenate([np.copy(img_arr[ind]), np.copy(img_arr[ind+1])], axis = 1)
        img2 = np.concatenate([np.copy(img_arr[ind+1]), np.copy(img_arr[ind])], axis = 1)
        img = np.concatenate([img1, img2], axis = 0)*f + 256*(1-f)
        
        # Joining Corresponding points
        line_coord = []
        for elem in matching_coord[ind]:
            line_coord.append(bresenham_line(elem[0, :], elem[1, :] + img_size0))
            line_coord.append(bresenham_line(elem[0, :], elem[1, :] + img_size1))
            line_coord.append(bresenham_line(elem[0, :] + img_size, elem[1, :] + img_size0))
            line_coord.append(bresenham_line(elem[0, :] + img_size, elem[1, :] + img_size1))
        line_coord = np.concatenate(line_coord)
        
        # Showing line in image
        for y, x in line_coord:
            if color:
                img[y, x, :] = 0
            else:
                img[y, x] = 0
        
        # Inserting new image to array
        new_img_arr.append(img)
    return np.stack(new_img_arr).astype(int)


# Computing Homography
def compute_homography(matched_coord, mode):
    '''
    mode = 'm': Matrix method (general), might find singular matrix
    mode = 'c': Calculus (affine)
    mode = 's': SVD decomposition
    '''
    matched_coord = np.copy(matched_coord.reshape(-1, 4))
    
    # Normalization
    mean = np.mean(matched_coord, axis = 0)
    std = np.std(matched_coord, axis = 0)/np.sqrt(2)
    std[std == 0] = 1               # Prevent division by zero
    matched_coord = (matched_coord - mean)/std
    
    # Matrices for getting final homography
    S1T1 = np.array([
        [1/std[2], 0, -mean[2]/std[2]],
        [0, 1/std[3], -mean[3]/std[3]],
        [0, 0, 1]
    ])
    S2T2_inv = np.array([
        [std[0], 0, mean[0]],
        [0, std[1], mean[1]],
        [0, 0, 1]
    ])
    
    # Computing homography
    if mode == 'm':
        A, B = [], []
        for coord in matched_coord:
            x2, y2, x1, y1 = coord              # Changing x to y will make later computation easier
            A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2])
            A.append([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2])
            B.append([x2])
            B.append([y2])
        A = np.array(A)
        B = np.array(B)
        H = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, B)).reshape((-1,))
        H = np.concatenate([H, [1]]).reshape((3, 3))
        H = np.matmul(S2T2_inv, np.matmul(H, S1T1))
        H /= H[2, 2]
    
    elif mode == 'c':
        def transform(coord):
            x2, y2, x1, y1 = coord
            return np.array([
                [x1*x1, x1*y1, x1, x1*x2, x1*y2],
                [x1*y1, y1*y1, y1, y1*x2, y1*y2],
                [x1, y1, 1, x2, y2],
            ])
        mat = np.apply_along_axis(transform, arr = matched_coord, axis = 1)
        mat = np.sum(mat, axis = 0)
        invA, B = np.linalg.inv(mat[:, :3]), mat[:, 3:]
        H1 = np.matmul(invA, B[:, 0])
        H2 = np.matmul(invA, B[:, 1])
        H = np.stack([H1, H2, [0, 0, 1]])
        H = np.matmul(S2T2_inv, np.matmul(H, S1T1))
        H /= H[2, 2]
        
    elif mode == 's':
        A = []
        for coord in matched_coord:
            x2, y2, x1, y1 = coord              # Changing x to y will make later computation easier
            A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2])
            A.append([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2])
        A = np.array(A)
        mat = np.matmul(A.T, A)
        U, S, V = np.linalg.svd(mat)
        H = V.T[:, -1].reshape((3,3))
        H = np.matmul(S2T2_inv, np.matmul(H, S1T1))
        if H[2,2] != 0:
            H /= H[2,2]
        
    else:
        raise Exception ("No Mode Selected")
    
    return H

# Perform transformation via homography
def transform_homography(coord, H):
    coord = np.concatenate([coord, np.ones((coord.shape[0], 1))], axis = 1).T
    transformed_coord = np.matmul(H, coord)
    transformed_coord = np.divide(transformed_coord, transformed_coord[2, :], out = np.zeros_like(transformed_coord),
                                  where = transformed_coord[2, :] != 0)
    return transformed_coord[:2, :].T

# Transforms into planar from cylinderical
def transform_planar(coord, cyl_shape, r, d):
    y_cyl, x_cyl = coord[:, 0] - cyl_shape[0]/2, coord[:, 1] - cyl_shape[1]/2
    x_plan = d*np.tan(x_cyl/r)
    y_plan = y_cyl*np.sqrt(x_plan**2+d**2)/r + cyl_shape[0]/2
    return np.stack([y_plan, x_plan]).T

# Calculate mean squared error for homography
def mse_homography(matched_coord, H):
    transformed_coord = transform_homography(matched_coord[:, 1, :], H)
    err = np.sqrt(np.sum(np.square(matched_coord[:, 0, :] - transformed_coord))/matched_coord.shape[0])
    return err

# Calculate mse loss on arr of homography
def mse_homography_arr(matched_coord_arr, H_arr):
    return apply_arr(matched_coord_arr, mse_homography, H_arr, dynamic=True)

# RANSAC algorithm for computing homography
def ransac_homography(matched_coord, iter = 1000, num_sample = 6, err_threshold = 2, max_err_threshold = 20, threshold_inc = 1, frac_inlier = 0.5, mode = 's', log = True):
    best_fit = compute_homography(matched_coord, mode = mode)
    best_err = mse_homography(matched_coord, best_fit)
    num_data = matched_coord.shape[0]
    num_sample = min(num_sample, num_data)
    num_inlier = int(frac_inlier*num_data)
    err = np.apply_along_axis(lambda pt: mse_homography(pt.reshape((1, 2, 2)), best_fit), arr = matched_coord.reshape((-1, 4)), axis = 1)
    confirmed_Inliers = matched_coord[err < err_threshold]
    best_inlier_ct = confirmed_Inliers.shape[0]
    
    # Iterating while calculating homography over random sample
    while err_threshold <= max_err_threshold:
        for _ in range(iter):
            # Taking random sample
            sample_ind = np.random.choice(num_data, size=num_sample, replace=False)
            maybe_inliers = matched_coord[sample_ind]
            
            # Computing homography of random sample
            maybe_model = compute_homography(maybe_inliers, mode = mode)
            
            # Figuring out Inliers
            err = np.apply_along_axis(lambda pt: mse_homography(pt.reshape((1, 2, 2)), maybe_model), arr = matched_coord.reshape((-1, 4)), axis = 1)
            confirmed_Inliers = matched_coord[err < err_threshold]
            this_inlier_ct = confirmed_Inliers.shape[0]
            
            # If number of inliers are less than specified, redo loop
            if this_inlier_ct < num_inlier:
                continue
            
            # Finding better model from confirmed Inliers
            better_model = compute_homography(confirmed_Inliers, mode = mode)
            this_err = mse_homography(confirmed_Inliers, better_model)
            if (-this_inlier_ct, this_err) < (-best_inlier_ct, best_err):
                best_fit = better_model
                best_err = this_err
                best_inlier_ct = this_inlier_ct
        
        # If no good model found, increasing error threshold and again calling function recursively
        if best_inlier_ct > num_inlier:
            break
        err_threshold += threshold_inc

    # Printing Log
    if log:    
        print(f"Homography Error: {best_err}\tInlier: {best_inlier_ct}\tData Size: {num_data}\tError Threshold: {err_threshold}")
    
    return best_fit

# Interpolates image
# To be done
def interpolate_img(img, ct_pt, mode = 'l'):
    '''
    mode = 'l': Bilinear interpolation
    '''
    return img

# Warping a coloured image via homography
# ct is used for finding x dimension of transformed image
def warp(img, homography, shape, center, mode = 'b', weight = 'b', cylinderical = False):
    '''
    mode = 'f': Forward warping
    mode = 'b': Backward warping with bilinear interpolation
    weight = 'l': Linear weights
    weight = 'b': Binary weights
    '''
    if mode == 'f':
        # Finding all possible coordinates
        coord = np.argwhere(np.ones_like(img[:, :, 0]))
        
        # Transforming coordinates
        transformed_coord = transform_homography(coord, homography).astype(int)
        
        # Computing image size
        size_y, size_x = shape
        
        # Creating new warped image
        transformed_img = np.zeros((size_y, size_x, 3))
        ct_pt = np.zeros((size_y, size_x))              # To count number of points mapped to it
        num_coord = coord.shape[0]
        for ind in range(num_coord):
            y1, x1 = coord[ind, :]
            y2, x2 = transformed_coord[ind, :]
            if 0 <= y2 < size_y and 0 <= x2 < size_x:
                transformed_img[y2, x2, :] += img[y1, x1, :]
                ct_pt[y2, x2] += 1
            
        # Handling overlapping pixels
        ind = ct_pt != 0
        for i in range(3):
            transformed_img[:, :, i][ind]/=ct_pt[ind]
    
        return interpolate_img(transformed_img, ct_pt), ct_pt

    elif mode == 'b':
        # Creating new warped image
        size_y, size_x = shape
        transformed_img = np.zeros((size_y, size_x, 3))
        
        # Finding all possible coordinates
        transformed_coord = np.argwhere(np.ones_like(transformed_img[:, :, 0]))
        
        # Transforming coordinates
        homography = np.linalg.inv(homography)
        homography /= homography[2,2]
        if cylinderical:
            r = transformed_img.shape[1]
            d = r/2
            transformed_coord_cyl = transform_planar(transformed_coord, transformed_img.shape, r, d)
            transformed_coord_cyl[:, 1] += center[1]
        else:
            transformed_coord_cyl = np.copy(transformed_coord)
        coord = transform_homography(transformed_coord_cyl, homography)
        filter_ind = (0 <= coord[:, 0]) & (coord[:, 0] < img.shape[0]-1)
        filter_ind &= (0 <= coord[:, 1]) & (coord[:, 1] < img.shape[1]-1)
        coord = coord[filter_ind]
        transformed_coord = transformed_coord[filter_ind]
        
        # Setting pixel intensities in transformed image
        num_coord = coord.shape[0]
        ct_pt = np.zeros(shape)
        if weight == 'b':
            y1, x1 = coord[:, 0], coord[:, 1]
            yt, xl = (y1//1).astype(int), (x1//1).astype(int)
            y2, x2 = transformed_coord[:, 0], transformed_coord[:, 1]
            
            It = (x1-xl)*img[yt, xl+1, :].T + (xl+1-x1)*img[yt, xl, :].T
            Ib = (x1-xl)*img[yt+1, xl+1, :].T + (xl+1-x1)*img[yt+1, xl, :].T
            transformed_img[y2, x2, :] = ((y1-yt)*Ib + (yt+1-y1)*It).T
            
            ct_pt = ct_pt.astype(bool)
            ct_pt[y2, x2] = True
        elif weight == 'l':
            for ind in range(num_coord):
                y1, x1 = coord[ind]
                y2, x2 = transformed_coord[ind]
                yt, xl = int(y1//1), int(x1//1)
                yb, xr = yt+1, xl+1
                
                It = (x1-xl)*img[yt, xr, :] + (xr-x1)*img[yt, xl, :]
                Ib = (x1-xl)*img[yb, xr, :] + (xr-x1)*img[yb, xl, :]
                transformed_img[y2, x2, :] = (y1-yt)*Ib + (yb-y1)*It
                
                ct_pt[y2, x2] = min(x1, y1, img.shape[0]-y1, img.shape[1]-x1)
                
        return transformed_img, ct_pt
        
    else:
        raise Exception("Invalid Mode")
    
# Warping array of coloured images via compound homography
def warp_arr(img_arr, homography_arr, mode = 'l', x_size = 5):
    '''
    mode = 'l': Linear warp images taking middle image as base
    mode = 'q': Exponential warp
    '''
    img_arr = np.copy(img_arr)
    
    if mode == 'l':
        warped_arr = []
        ct_pt = []
        num_img = img_arr.shape[0]

        # Creating compound homography
        compound_homography = np.stack([np.eye(3) for i in range(num_img)])
        base_img_ind = (num_img-1)//2
        center_x = (base_img_ind*img_arr.shape[2]*x_size)//num_img
        center_y = 0
        compound_homography[base_img_ind, 1, 2] +=  center_x        # Centering of base img in x direction
        compound_homography[base_img_ind, 0, 2] += center_y         # Centering of base img in y direction
        for i in range(base_img_ind+1, num_img):
            cascade_homography = np.matmul(compound_homography[i-1, :, :], homography_arr[i-1, :, :])
            cascade_homography /= cascade_homography[2, 2]
            compound_homography[i, :, :] = cascade_homography
        for i in range(base_img_ind-1, -1, -1):
            cascade_homography = np.matmul(compound_homography[i+1, :, :], np.linalg.inv(homography_arr[i, :, :]))
            cascade_homography /= cascade_homography[2, 2]
            compound_homography[i, :, :] = cascade_homography
        
        # Applying compound homography
        transformed_shape = (img_arr.shape[1], x_size*img_arr.shape[2])
        for i in range(num_img):
            transformed_img, ct = warp(img_arr[i, :, :, :], compound_homography[i], transformed_shape, (center_y, center_x))
            warped_arr.append(transformed_img)
            ct_pt.append(ct)
    
    elif mode == 'e':
        warped_arr = []
        ct_pt = []
        num_img = img_arr.shape[0]
        
        base_img_ind = (num_img-1)//2
        
        for i in range(num_img):
            transformed_img, ct = warp

    else:
        raise Exception("No Valid Mode")

    return np.stack(warped_arr), np.stack(ct_pt)


# Detect preproces, feature, calculates homography and then warps
def all(img_arr, save_bool):
    img_arr = np.copy(img_arr)
    num_img = img_arr.shape[0]
    
    # save
    preprocessed_img_arr = []
    feature_detected_img_arr = []
    matched_img_arr = []
    
    # Constants for warping
    base_img_ind = (num_img-1)//2
    warped_img_arr = [None for i in range(num_img)]
    ct_pt = [None for i in range(num_img)]
    x_size = 5
    transformed_shape = (2*img_arr.shape[1], x_size*img_arr.shape[2])
    center_x = (base_img_ind*img_arr.shape[2]*x_size)//num_img
    center_y = img_arr.shape[1]/2
    
    # Base image
    base_homography = np.eye(3)
    base_homography[:2, 2] = np.array([center_y, center_x])
    warped_img_arr[base_img_ind], ct_pt[base_img_ind] = warp(img_arr[base_img_ind], base_homography, transformed_shape, (center_y, center_x))
    print(f"Image {base_img_ind} done")
    print()
    
    for ind in range(base_img_ind+1, 5):
        print(f"Image {ind} Processing")
        img, curr_ctpt = warp(img_arr[ind], np.eye(3), transformed_shape, (center_y, center_x))
        
        # Preprocessing
        preprocessed_arr = preprocess(np.array([warped_img_arr[ind-1], img]))
        preprocessed_arr[0] = (preprocessed_arr[0].T*ct_pt[ind-1].T).T
        preprocessed_arr[1] = (preprocessed_arr[1].T*curr_ctpt.T).T
        if save_bool:
            preprocessed_img_arr.append(preprocessed_arr[0])
            preprocessed_img_arr.append(preprocessed_arr[1])
        print(f"Image {ind} Preprocessed")
        
        # Feature Detection
        feature_detected_arr, keypoint_arr = feature_detector(preprocessed_arr, mode = 'l', save = save_bool)
        if save_bool:
            feature_detected_img_arr.append(feature_detected_arr[0])
            feature_detected_img_arr.append(feature_detected_arr[1])
        print(f"Image {ind} Feature Detected")
        
        # Feature Descriptor
        descriptor_list, keypoint_index_list = feature_descriptor(preprocessed_arr, keypoint_arr, mode = 's')
        print(f"Image {ind} Descriptor Created")
        
        # Matching Descriptor
        matched_coord = match_coord(descriptor_list, keypoint_index_list)
        if save_bool:
            matched_img = create_match_img(preprocessed_arr, matched_coord)
            matched_img_arr.append(matched_img[0])
        print(f"Image {ind} Descriptor Matched")
        
        # Computing Homography
        homography = apply_arr(matched_coord, ransac_homography)
        print(homography)
        print(f"Image {ind} Homography Calculated")
        
        # Warping
        warped_img_arr[ind], ct_pt[ind] = warp(img, homography[0], transformed_shape, (center_y, center_x))
        print(f"Image {ind} Warped")
        
    return warped_img_arr[base_img_ind:], ct_pt[base_img_ind:], preprocessed_img_arr, feature_detected_img_arr, matched_img_arr


# Equalize global illumination
def equalize_brightness(img_arr, ct_pt, frac_threshold = 0.2):
    for ind in range(img_arr.shape[0]-1):
        common = ct_pt[ind] & ct_pt[ind+1]
        
        # Find the leftmost and rightmost 1 in each row
        leftmost_1 = np.argmax(common, axis=1)
        rightmost_1 = common.shape[1] - 1 - np.argmax(common[:, ::-1], axis=1)

        # Create a new array with only the leftmost and rightmost 1 retained
        result = np.zeros_like(common)
        for i in range(common.shape[0]):
            result[i, [leftmost_1[i], rightmost_1[i]]] = 1

        boundary = result & common      # Take only points which belong to common as well
        
        if np.count_nonzero(boundary) > frac_threshold*img_arr.shape[1]:
            diff = img_arr[ind] - img_arr[ind+1]
            img_arr[ind+1] += np.mean(diff[boundary])
    
    return img_arr
        
# Blending image togethor
def blend(img_arr, ct_pt, mode = 'g', crop = 2):
    '''
    mode = 's': Simple average
    mode = 'l': Laplacian Pyramid
    mode = 'a': alpha
    mode = 'g': graph cut
    mode = 'gp': graphcut with poisson blend  # To be implemented
    '''
    img_arr = np.copy(img_arr)[:, crop:img_arr.shape[1]-crop, crop:img_arr.shape[2]-crop, :]
    ct_pt = np.copy(ct_pt)[:, crop:ct_pt.shape[1]-crop, crop: ct_pt.shape[2]-crop]
    num_img = img_arr.shape[0]
    
    # Equalizing global brightness as preprocessing
    img_arr = equalize_brightness(img_arr, ct_pt)
    
    # Simple Average
    if mode == 's':
        img = np.sum(img_arr, axis = 0)
        ct = np.sum(ct_pt, axis = 0)
        ind = ct != 0
        for i in range(3):
            img[:, :, i][ind] /= ct[ind]

    # Laplacian Pyramids
    elif mode == 'l':
        pass
    
    # Alpha blending
    elif mode == 'a':
        mask = np.copy(ct_pt)
        sum_mask = np.sum(ct_pt, axis = 0)
        ind = sum_mask != 0
        for i in range(mask.shape[0]):
            mask[i][ind]/=sum_mask[ind]
        for i in range(3):
            img_arr[:, :, :, i] *= mask
        img = np.sum(img_arr, axis = 0)

    # Graph cut
    elif mode[0] == 'g':
        img = np.copy(img_arr[0])
        ct = np.copy(ct_pt[0])
        for i in range(1, num_img):
            if np.all(ct_pt[i] == 0):
                continue
            
            # Computing difference for best seam
            diff = np.sum((np.abs(img - img_arr[i])**2), axis = -1)
            diff[(ct == 0) | (ct_pt[i] == 0)] = 256**2+1
            
            # Finding scores
            for y in range(diff.shape[0]-2, -1, -1):
                row = np.concatenate([[np.inf],diff[y+1, :],[np.inf]])
                row = np.stack([row[1:-1], row[:-2], row[2:]])
                row = np.min(row, axis = 0)
                diff[y, :] += row
                
            # Finding best seam via greedy walk
            seam_arr = []
            seam = diff.shape[1]-1-np.argmin(diff[0, ::-1])
            seam_arr.append(seam)
            for y in range(diff.shape[0]-1):
                seam += np.argmin(diff[y, max(0,seam-1):seam+2]) - 1
                seam = min(max(0, seam), img.shape[1]-1)
                seam_arr.append(seam)
            seam_arr = np.array(seam_arr)
            
            # Either right most point of first image or seam value as predicted
            # min_seam = np.argmin(np.concatenate([ct, np.zeros((ct.shape[0],1))], axis=1), axis = 1)
            min_seam = np.zeros_like(seam_arr)
            for y in range(min_seam.shape[0]):
                indices = np.argwhere(ct[y])
                min_seam[y] = indices[-1] if indices.shape[0] > 0 else 0
            seam_arr = np.minimum(min_seam, seam_arr)
            
            # Blending using seam
            for y in range(diff.shape[0]):
                img[y, seam_arr[y]:, :] = img_arr[i, y, seam_arr[y]:, :]
            
            # updating ct
            ct |= ct_pt[i]

    
    return img
