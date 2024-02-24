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
        args = [[args[j][i] for j in range(len(args))] for i in range(img_arr.shape[0])]
        if np_conv:
            return np.array([func(img_arr[i], *(args[i])) for i in range(img_arr.shape[0])])
        else:
            return [func(img_arr[i], *(args[i])) for i in range(img_arr.shape[0])]
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
def laplace_harris_detector(img, window_size = 7, k = 0.04, t = 1e7, edge_t = 10):
    img = np.copy(img)
    
    # Calculates gradient
    I_x, I_y = gradient(img)
    
    # Calculating summed matrix
    R_arr = []
    for sigma in [1, 2, 4, 8, 16]:
        kernel = log_kernel(window_size, sigma=sigma)
        I_x2 = convolve2d(I_x**2, kernel)
        I_y2 = convolve2d(I_y**2, kernel)
        I_xy = convolve2d(I_x*I_y, kernel)
    
        # Calculating score R
        R = I_x2*I_y2 - I_xy*I_xy - k*(I_x2 + I_y2)**2
        
        R_arr.append(R)
        
    # Calculating local maxima
    R = np.stack(R_arr)
    keypoint = local_maxima(R) & (R>t)
    keypoint = np.max(keypoint, axis = 0)
    
    # Cropping features at edge
    keypoint[:edge_t, :] = False
    keypoint[-edge_t:, :] = False
    keypoint[:, :edge_t] = False
    keypoint[:, -edge_t:] = False

    return keypoint
    

# Insert Keypoint into img
def insert_keypoint(img, keypoint, thickness = 11):
    keypoint = np.copy(keypoint)
    img = np.copy(img)
    print(img.shape, np.count_nonzero(keypoint))
    
    # Thickening Keypoint
    ker_size = (thickness+1)//2
    keypoint = convolve2d(keypoint, np.ones((ker_size, ker_size))).astype(bool)
    
    # Inserting Keypoint
    img[keypoint] = -128*np.sign(img[keypoint] - 128) + 128
    
    return img

# Feature detection
def feature_detector(img_arr, mode, low_resolution = True, save = False):
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
    if low_resolution:
        inp_img_arr = apply_arr(img_arr, set_resolution)

    keypoint = apply_arr(inp_img_arr, func)
    if low_resolution:
        _keypoint = np.zeros(img_arr.shape, dtype=bool)
        _keypoint[:, ::2, ::2] = keypoint
        keypoint = _keypoint

    if save:
        img_arr = apply_arr(img_arr, insert_keypoint, keypoint, dynamic=True)
    else:
        img_arr = []
    return img_arr, keypoint
