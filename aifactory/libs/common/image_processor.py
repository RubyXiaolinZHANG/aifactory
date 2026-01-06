import cv2
import numpy as np


def enhanced_edge_detection(gray_img, threshold=30, min_area=20, min_size=10, remove_boundary=0):
    """
    enhanced edge detection
    """

    # 1. gaussian denoise
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # 2. sobel gradient
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # 3. magnitude
    magnitude = np.hypot(gx, gy)
    if remove_boundary > 0:
        magnitude[:remove_boundary] = 0
        magnitude[-remove_boundary:] = 0
        magnitude[:, :remove_boundary] = 0
        magnitude[:, -remove_boundary:] = 0

    # 4. non max suppression
    edges = non_maximum_suppression(magnitude, gx, gy)

    # 5. threshold
    strong_edges = ((edges > threshold) * 255).astype(np.uint8)

    # 6. dilation for connection
    edges_connected = cv2.dilate(strong_edges, None, iterations=1)

    # 7. edge connection analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        edges_connected, connectivity=8
    )
    analysis_report = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        centroid_x, centroid_y = centroids[label]
        # 计算额外特征
        aspect_ratio = width / height if height > 0 else 0
        extent = area / (width * height) if (width * height) > 0 else 0
        compactness = (4 * np.pi * area) / ((width + height) ** 2) if (width + height) > 0 else 0

        region_info = {
            'label': label,
            'area': area,
            'bbox': (left, top, width, height),
            'centroid': (centroid_x, centroid_y),
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'compactness': compactness
        }

        analysis_report.append(region_info)
    analysis_report.sort(key=lambda x: x['area'], reverse=True)
    filtered_regions = [r for r in analysis_report if
                        (r['area'] >= min_area and (r['bbox'][2] > min_size or r['bbox'][3] > min_size))]
    valid_edges = np.zeros(labels.shape, dtype=np.uint8)
    for region in filtered_regions:
        valid_edges[labels == region['label']] = 255

    return valid_edges


def non_maximum_suppression(magnitude, gx, gy):
    rows, cols = magnitude.shape
    output = np.zeros_like(magnitude)

    # angles
    angle = np.arctan2(gy, gx) * 180 / np.pi
    angle = np.mod(angle + 180, 180)  # 0-180 degree

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # direction
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                neighbor1 = magnitude[i, j + 1]
                neighbor2 = magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                neighbor1 = magnitude[i + 1, j - 1]
                neighbor2 = magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                neighbor1 = magnitude[i + 1, j]
                neighbor2 = magnitude[i - 1, j]
            else:  # 112.5 <= angle < 157.5
                neighbor1 = magnitude[i - 1, j - 1]
                neighbor2 = magnitude[i + 1, j + 1]

            # keep tha max
            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                output[i, j] = magnitude[i, j]

    return output


def bgr2hsv(bgr):

    bgr = bgr.astype(np.float32) / 255.0

    # 获取形状
    height, width, _ = bgr.shape
    bgr_flat = bgr.reshape(-1, 3)
    b, g, r = bgr_flat[:, 0], bgr_flat[:, 1], bgr_flat[:, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # init
    h = np.zeros_like(cmax)
    s = np.zeros_like(cmax)
    v = cmax

    # S
    mask = cmax > 0
    s[mask] = delta[mask] / cmax[mask]

    # H
    mask_r = (cmax == r) & (delta > 0)
    h[mask_r] = 60 * ((g[mask_r] - b[mask_r]) / delta[mask_r])

    mask_g = (cmax == g) & (delta > 0)
    h[mask_g] = 60 * (2 + (b[mask_g] - r[mask_g]) / delta[mask_g])

    mask_b = (cmax == b) & (delta > 0)
    h[mask_b] = 60 * (4 + (r[mask_b] - g[mask_b]) / delta[mask_b])

    h[h < 0] += 360

    hsv_flat = np.column_stack([h, s, v])
    hsv = hsv_flat.reshape(height, width, 3)
    return hsv