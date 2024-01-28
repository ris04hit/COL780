import numpy as np
import scipy.signal

# Convert to binary image
def binary_img(img):
    return np.round(img/255).astype(int)

# Limit image values
def lim_image(img):
    img[img<0] = 0
    img[img>255] = 255
    return img

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
    
    return lim_image(threshold_img)


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

    return lim_image(d_threshold_img)


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
    
    return lim_image(threshold(R, t))


# Finding processed neighbor of pixel in binary image
def neighbor_pixel(img, i, j):
    neigh = []
    x, y = img.shape
    if i > 0 and img[i-1, j] == 1:
        neigh.append((i-1, j))
    if j > 0 and img[i, j-1] == 1:
        neigh.append((i, j-1))
    if i > 0 and j > 0 and img[i-1, j-1] == 1:
        neigh.append((i-1, j-1))
    if i > 0 and j < y-1 and img[i-1, j+1] == 1:
        neigh.append((i-1, j+1))
    return neigh
        
# Standardize linked elements in union find
def standardize_find(linked: dict):
    collected_label = []
    for elem in linked:
        union = []
        for processed_set in collected_label:
            if len(processed_set.intersection(linked[elem])) != 0:
                union.append(processed_set)
        if len(union) != 0:
            for union_elem in union:
                collected_label.remove(union_elem)
                linked[elem] = linked[elem].union(union_elem)
        collected_label.append(linked[elem])
                
    for label in range(len(collected_label)):
        for elem in collected_label[label]:
            linked[elem] = label

# Connected Component using two pass algorithm
def connected_component(img):
    img = np.copy(img)
    
    # Converting to binary image
    img = binary_img(img)
    
    # Two pass algorithm
    linked = {0: {0}}
    labels = np.zeros_like(img)
    currlabel = 0
    
    # First Pass
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 1:
                neighbors = neighbor_pixel(img, i, j)
                if len(neighbors) == 0:
                    currlabel += 1
                    linked[currlabel] = {currlabel}
                    labels[i, j] = currlabel
                else:
                    neighbor_label = set([labels[x, y] for (x, y) in neighbors])
                    labels[i, j] = min(neighbor_label)
                    for label_val in neighbor_label:
                        linked[label_val] = linked[label_val].union(neighbor_label)
    
    # Second Pass
    standardize_find(linked)
    min_func = np.vectorize(lambda x: linked[x])
    labels = min_func(labels)
    
    return labels


# Splits various components of image
def split_component(img, labels):
    num_label = np.max(labels)
    component_arr = []
    for label in range(1, num_label+1):
        component = np.zeros_like(img)
        index = labels == label
        component[index] = img[index]
        component_arr.append(component)
    return lim_image(255*np.array(component_arr))


# Calculates r for hough transform
def calc_r(x, y, theta):
    return (x*np.cos(theta*np.pi/180) + y*np.sin(theta*np.pi/180)).astype(int)

# Calculates angle with x axis of a line perpendicular to suture
def suture_angle(img, t = 200):
    # Calculates gradient
    sobel_kernel = sobel_gradient_kernel()
    grad_x = filter_img(img, sobel_kernel[0])
    grad_y = filter_img(img, sobel_kernel[1])
    grad = np.hypot(grad_x, grad_y)
    theta = (180*np.arctan2(grad_y, grad_x)/np.pi) % 180    
    angle_val = np.average(theta, weights=grad)

    return angle_val

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

# Hough transforms an image to detect straight lines
def gradient_hough_transform(img,
                             num_theta = 60,
                             grad_precision = 2,
                             point_threshold = 0.3,
                             interpolate = 0,
                             suture_filter = 0,
                             suture_threshold = 25):
    img = np.copy(img)

    # Converting to binary image
    img = binary_img(img)
    
    # Calculates gradient
    del_theta = 180/num_theta
    sobel_kernel = sobel_gradient_kernel()
    grad_x = filter_img(img, sobel_kernel[0])
    grad_y = filter_img(img, sobel_kernel[1])
    grad = np.hypot(grad_x, grad_y)
    theta = ((num_theta*np.arctan2(grad_y, grad_x)/np.pi) % num_theta).astype(int)
    s_angle = suture_angle(img)
    
    # Creating accumulator
    r_max = np.hypot(img.shape[0], img.shape[1]).astype(int)
    accumulator = np.zeros((r_max, num_theta))
    line_param = [[[] for i in range(num_theta)] for j in range(r_max)]
    
    # Filling accumulator for each pixel
    for y,x in np.argwhere(img == 1):
        for theta_ind in range(theta[y, x] - grad_precision, theta[y, x] + grad_precision):
            theta_val = theta_ind * del_theta
            if suture_filter:
                diff_theta = (theta_val - s_angle)%180
                diff_theta = min(diff_theta, 180-diff_theta)
                if diff_theta > suture_threshold:
                    continue
            coord0 = calc_r(x, y, theta_val)
            coord1 = theta_ind%num_theta
            accumulator[coord0, coord1] += 1
            line_param[coord0][coord1].append((x,y))
    
    # Thresholding accumulator
    effective_threshold = max(1, point_threshold*(np.max(accumulator)-30))
    effective_threshold = max(1, point_threshold*np.max(accumulator))
    line_coord = [line_param[r][t] for r, t in np.argwhere(accumulator >= effective_threshold)]
    
    # Creating line image
    line_img = np.zeros_like(img)
    for line in line_coord:
        if interpolate:
            c0, c1 = min(line), max(line)
            new_line = bresenham_line(c0, c1)
        else:
            new_line = line
        for x, y in new_line:
            line_img[y, x] = 1
    return 255*line_img


# Merge kernel
def merge_kernel(size):
    return np.ones((size, size))

# Merge line image (Only for binary image)
def merge_line_image(img, size = 1):
    img = np.copy(img)
    kernel = merge_kernel(size)
    merged_img = filter_img(img, kernel)
    return lim_image(merged_img)


# Counts number of sutures in one component using corner count
def count_suture_corner(component, corner):
    component = np.copy(component)
    
    # Converting to binary image
    component = binary_img(component)
    
    # Corners in component
    corner_comp = np.zeros_like(corner)
    index = component == 1
    corner_comp[index] = corner[index]
    
    # Counting number of corners
    labels = connected_component(corner_comp)

    return np.max(labels)
