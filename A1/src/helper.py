import numpy as np
import scipy.signal

# Function to convert image to gray scale image
def gray_scale(img):
    return 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

# Function to create a gaussian filter matrix of desired dimension
def gaussian_filter_kernel(n):
    k = (n-1)//2
    filter_exp = np.array([[(i-k)**2 + (j-k)**2 for j in range(2*k+1)] for i in range(2*k+1)])
    filter_val = np.exp(-filter_exp/2)
    return filter_val/np.sum(filter_val)

# Function to apply a filter kernel
def filter_img(img, kernel, mode = 'full'):
    if len(img.shape) == 3:
        filtered_channel = []
        for channel in range(img.shape[2]):
            filtered_channel.append(scipy.signal.convolve2d(img[:, :, channel], kernel, mode = mode))
        filtered_img = np.stack(filtered_channel, axis=-1)
        return filtered_img
    else:
        return scipy.signal.convolve2d(img, kernel, mode = mode)

# Changes the contrast by given value 
def apply_contrast(img, c):
    f = (259*(c+255))/(255*(259-c))
    contrast_img = f*(img-128)+128
    contrast_img[contrast_img < 0] = 0
    contrast_img[contrast_img > 255] = 255
    return contrast_img

# Thresholds image
def threshold(img, t):
    return 255*(img > t)

# Preprocessed image
def preprocess(img, smooth_size = 5, contrast = 200):
    # Gray scale
    gray_img = gray_scale(np.copy(img))
    
    # Gaussian smoothening
    kernel = gaussian_filter_kernel(smooth_size)
    smoothened_img = filter_img(gray_img, kernel)
    
    # Increasing Contrast
    high_contrast_img = apply_contrast(smoothened_img, contrast)
    
    # Thresholding
    threshold_img = threshold(high_contrast_img, 50)
    
    return threshold_img


# Calculates sobel kernel
def sobel_gradient_kernel():
    return np.array([[[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]],
                     [[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]])

# Gradient thresholding
def grad_threshold(grad, theta):
    threshold = np.zeros_like(grad)
    theta = np.round(theta/45).astype(int)%4
    
    for i in range(1, grad.shape[0]-1):
        for j in range(1, grad.shape[1]-1):
            if theta[i, j] == 0:
                neighbor_val = max(theta[i, j-1], theta[i, j+1])
            elif theta[i, j] == 1:
                neighbor_val = max(theta[i-1, j+1], theta[i+1, j-1])
            elif theta[i, j] == 2:
                neighbor_val = max(theta[i-1, j], theta[i+1, j])
            else:
                neighbor_val = max(theta[i-1, j-1], theta[i+1, j+1])
                
            threshold[i, j] = grad[i,j] if grad[i, j] > neighbor_val else 0

    return threshold

# Double threshold
def double_threshold(img, t1, t2):
    img[img < t1] = 0
    img[img > t2] = 255
    return img

# Canny Edge Detector
def canny_edge_detector(img, threshold1 = 50, threshold2 = 200):
    img = np.copy(img)
    
    # Calculates gradient
    sobel_kernel = sobel_gradient_kernel()
    grad_x = filter_img(img, sobel_kernel[0])
    grad_y = filter_img(img, sobel_kernel[1])
    grad = np.hypot(grad_x, grad_y)
    theta = 180*np.arctan2(grad_y, grad_x)/np.pi
    
    # Gradient thresholding
    threshold_img = grad_threshold(grad, theta)
    threshold_img = 255*threshold_img/np.max(threshold_img)
    
    # Double threshold
    d_threshold_img = double_threshold(threshold_img, threshold1, threshold2)

    return d_threshold_img


# Harris Corner Detector
def harris_corner_detector(img, window_size = 5, k = 0.04, t = 4e10):
    img = np.copy(img)
    
    # Calculates gradient
    sobel_kernel = sobel_gradient_kernel()
    I_x = filter_img(img, sobel_kernel[0], mode = 'same')
    I_y = filter_img(img, sobel_kernel[1], mode = 'same')
    
    # Calculating summed matrix
    gaussian_kernel = gaussian_filter_kernel(window_size)
    I_x2 = filter_img(I_x**2, gaussian_kernel, mode = 'same')
    I_y2 = filter_img(I_y**2, gaussian_kernel, mode = 'same')
    I_xy = filter_img(I_x*I_y, gaussian_kernel, mode = 'same')
    M = np.stack([np.stack([I_x2, I_xy], axis = -1), np.stack([I_xy, I_y2], axis = -1)], axis = -2)
    
    # Calculating score R
    R = np.linalg.det(M) - k*(np.trace(M, axis1 = -2, axis2 = -1))**2
    
    return threshold(R, t)


# Connected Component
def connected_component(img):
    img = np.copy(img)
    