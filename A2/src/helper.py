import numpy as np

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
    if dynamic:
        size = len(img_arr) if type(img_arr) != np.ndarray else img_arr.shape[0]
        args = [[args[j][i] for j in range(len(args))] for i in range(size)]
        if np_conv:
            return np.array([func(img_arr[i], *(args[i])) for i in range(size)])
        else:
            return [func(img_arr[i], *(args[i])) for i in range(size)]
    else:
        if np_conv:
            return np.array([func(img, *args) for img in img_arr])
        else:
            return [func(img, *args) for img in img_arr]

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
        img = convolve_func(img, gaussian_filter_kernel(5))
        
    # Reducing resolution
    img = img[::2, ::2] if len(img.shape) == 2 else img[::2, ::2, ]
    
    # Gaussian smoothing after setting resolution
    if out_smoothen:
        img = convolve_func(img, gaussian_filter_kernel(5))
    
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
def preprocess(img_arr, contrast = 50, kernel_size = 5):
    img_arr = np.copy(img_arr)
    
    # Gray scale
    img_arr = apply_arr(img_arr, gray_scale)
    
    # Setting contrast
    img_arr = set_contrast(img_arr, contrast)
    
    # Gaussian Smoothening
    img_arr = apply_arr(img_arr, convolve2d, gaussian_filter_kernel(kernel_size))
    
    return img_arr


# Harris Corner Detector
def harris_corner_detector(img, window_size = 3, k = 0.04, t = 1e9):
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
def laplace_harris_detector(img, window_size = 7, k = 0.04, t = 1e6, edge_t = 10, num_scale = 3, scale_factor = 1.5, num_pt = 0):
    img = np.copy(img)
    
    # Calculates gradient
    I_x, I_y = gradient(img)
    
    # Calculating summed matrix
    R_arr = []
    for sigma in [scale_factor**i for i in range(num_scale)]:
        kernel = log_kernel(window_size, sigma=sigma)
        I_x2 = convolve2d(I_x**2, kernel)
        I_y2 = convolve2d(I_y**2, kernel)
        I_xy = convolve2d(I_x*I_y, kernel)
    
        # Calculating score R
        R = I_x2*I_y2 - I_xy*I_xy - k*(I_x2 + I_y2)**2
        
        R_arr.append(R)
        
    # Calculating local maxima
    R = np.stack(R_arr)
    max_R = local_maxima(R)
    keypoint = np.zeros_like(R, dtype=bool)
    while np.count_nonzero(keypoint) <= num_pt:
        keypoint = max_R & (R>t)
        keypoint = np.max(keypoint, axis = 0)
        
        # Cropping features at edge
        keypoint[:edge_t, :] = False
        keypoint[-edge_t:, :] = False
        keypoint[:, :edge_t] = False
        keypoint[:, -edge_t:] = False
        
        t /= 2
    

    return keypoint


# Insert Keypoint into img
def insert_keypoint(img, keypoint, thickness = 11, log = True):
    keypoint = np.copy(keypoint)
    img = np.copy(img)
    
    if log:
        print(img.shape, np.count_nonzero(keypoint))
    
    # Thickening Keypoint
    ker_size = (thickness+1)//2
    keypoint = convolve2d(keypoint, np.ones((ker_size, ker_size))).astype(bool)
    
    # Inserting Keypoint
    img[keypoint] = -128*np.sign(img[keypoint] - 128) + 128
    
    return img

# Feature detection
def feature_detector(img_arr, mode, low_resolution = 1, save = False):
    '''
    mode = 'h': Harris Corner Detector
    mode = 'l': Laplace Harris
    '''
    if mode == 'h':
        func = lambda img: harris_corner_detector(img)
    if mode == 'l':
        func = lambda img: laplace_harris_detector(img)
        
    img_arr = np.copy(img_arr)
    inp_img_arr = np.copy(img_arr)
    for _ in range(low_resolution):
        inp_img_arr = apply_arr(inp_img_arr, set_resolution)

    keypoint = apply_arr(inp_img_arr, func)
    if low_resolution:
        _keypoint = np.zeros(img_arr.shape, dtype=bool)
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
        
    descriptor_arr = apply_arr(img_arr, func, keypoint_arr, np_conv=False, dynamic=True)
    keypoint_index_arr = [np.argwhere(keypoint) for keypoint in keypoint_arr]
    return descriptor_arr, keypoint_index_arr


# Finding corresponding descriptor in img
def find_descriptor(keypt_descriptor, img_descriptor_arr, norm = 1, ret = 0, ratio_threshold = 0.8):
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
def match_descriptor(descriptor_arr1, descriptor_arr2):
    desc_ind1 = apply_arr(descriptor_arr1, lambda descriptor: find_descriptor(descriptor, descriptor_arr2))
    desc_ind1_arg = np.argwhere(~np.isnan(desc_ind1)).reshape((-1,))
    desc_ind1_without_nan = desc_ind1[~np.isnan(desc_ind1)].astype(int)
    desc_ind2 = apply_arr(descriptor_arr2[desc_ind1_without_nan], lambda descriptor: find_descriptor(descriptor, descriptor_arr1))
    match_arg = desc_ind2 == desc_ind1_arg
    matching = np.stack([desc_ind1_arg[match_arg], desc_ind1_without_nan[match_arg]])
    return matching

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
    num_img = img_arr.shape[0]
    if num_img < 2:
        raise Exception("Less than 2 images")
    
    # Different array for different size offset
    img_size = np.array(np.shape(img_arr[0, :, :]))
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
            img[y, x] = 0
        
        # Inserting new image to array
        new_img_arr.append(img)
    return np.stack(new_img_arr).astype(int)


# Computing Homography
def compute_homography(matched_coord):
    matched_coord = matched_coord.reshape(-1, 4)
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
    return H

# Perform transformation via homography
def transform_homography(coord, H):
    coord = np.concatenate([coord, np.ones((coord.shape[0], 1))], axis = 1).T
    transformed_coord = np.matmul(H, coord)
    transformed_coord /= transformed_coord[2, :]
    return transformed_coord[:2, :].T

# Calculate mean squared error for homography
def mse_homography(matched_coord, H):
    transformed_coord = transform_homography(matched_coord[:, 1, :], H)
    err = np.sqrt(np.sum(np.square(matched_coord[:, 0, :] - transformed_coord))/matched_coord.shape[0])
    return err

# Calculate mse loss on arr of homography
def mse_homography_arr(matched_coord_arr, H_arr):
    return apply_arr(matched_coord_arr, mse_homography, H_arr, dynamic=True)

# RANSAC algorithm for computing homography
def ransac_homography(matched_coord, iter = 1000, frac_sample = 0.1, err_threshold = 2, max_err_threshold = 5, threshold_inc = 1, frac_inlier = 0.8, log = True):
    best_fit = compute_homography(matched_coord)
    best_err = mse_homography(matched_coord, best_fit)
    num_data = matched_coord.shape[0]
    num_sample = int(frac_sample*num_data)
    if num_sample < 8:
        num_sample = min(8, num_data)
    num_inlier = int(frac_inlier*num_data)
    
    # Iterating while calculating homography over random sample
    while err_threshold <= max_err_threshold:
        for _ in range(iter):
            # Taking random sample
            sample_ind = np.random.choice(num_data, size=num_sample, replace=False)
            maybe_inliers = matched_coord[sample_ind]
            
            # Computing homography of random sample
            maybe_model = compute_homography(maybe_inliers)
            
            # Figuring out Inliers
            err = np.apply_along_axis(lambda pt: mse_homography(pt.reshape((1, 2, 2)), maybe_model), arr = matched_coord.reshape((-1, 4)), axis = 1)
            confirmed_Inliers = matched_coord[err < err_threshold]
            
            # If number of inliers are less than specified, redo loop
            if confirmed_Inliers.shape[0] < num_inlier:
                continue
            
            # Finding better model from confirmed Inliers
            better_model = compute_homography(confirmed_Inliers)
            this_err = mse_homography(confirmed_Inliers, better_model)
            if this_err < best_err:
                best_fit = better_model
                best_err = this_err
        
        # If no good model found, increasing error threshold and again calling function recursively
        err = np.apply_along_axis(lambda pt: mse_homography(pt.reshape((1, 2, 2)), best_fit), arr = matched_coord.reshape((-1, 4)), axis = 1)
        confirmed_Inliers = matched_coord[err < err_threshold]
        if confirmed_Inliers.shape[0] > num_inlier:
            break
        err_threshold += threshold_inc

    # Printing Log
    if log:    
        print(f"Homography Error: {best_err}\tInlier: {confirmed_Inliers.shape[0]}\tData Size: {num_data}\tError Threshold: {err_threshold}")
    
    return best_fit
