import cv2
import numpy as np


def get_solid_color_image(im_height, im_width):
    rgb = np.random.randint(0, 256, 3).astype(np.uint8)
    image = np.ones((im_height, im_width, 3), dtype=np.uint8)
    return image * rgb


def get_gray_image(im_height, gray_level=256, band=10, vertical=True):
    image = np.ones((im_height, gray_level, band), dtype=np.uint8)
    gray = np.arange(0, gray_level, dtype=np.uint8).reshape(1, gray_level, 1)
    image = image * gray
    image = image.reshape(im_height, -1, 1)
    rgb = np.concatenate([image, image, image], axis=2)
    if vertical:
        return np.transpose(rgb, (1, 0, 2))
    else:
        return rgb


def get_stripe_image(im_height, im_width, band_width, gray_1=None, gray_2=None, angle=90):
    if gray_1 is None:
        gray_1 = [0, 0, 0]
    if gray_2 is None:
        gray_2 = [255, 255, 255]

    image = np.ones((im_height, im_width, 3), dtype=np.uint8)
    image = image * np.array(gray_1, dtype=np.uint8)
    if angle == 90 or angle == 270:
        for i in range(0, im_width, band_width * 2):
            image[:, i:i + band_width, 0] = gray_2[0]
            image[:, i:i + band_width, 1] = gray_2[1]
            image[:, i:i + band_width, 2] = gray_2[2]

    elif angle == 0 or angle == 180:
        for i in range(0, im_height, band_width * 2):
            image[i:i + band_width, :, 0] = gray_2[0]
            image[i:i + band_width, :, 1] = gray_2[1]
            image[i:i + band_width, :, 2] = gray_2[2]

    else:
        raise ValueError("only support angle = 0 , 90, 180, 270")

    return image


def get_sin_image(im_height, im_width, t=4, gray_1=0, gray_2=50, phase=90, angle=90):
    image = np.ones((im_height, im_width, 3), dtype=np.float32)
    if angle == 90 or angle == 270:
        x = np.arange(0, im_width, 1)
        y = (np.sin(2 * np.pi / t * x + phase / np.pi) + 1) / 2 * (gray_2 - gray_1) + gray_1
        image = image * y.reshape(im_width, 1)
    elif angle == 0 or angle == 180:
        x = np.arange(0, im_height, 1)
        y = (np.sin(2 * np.pi / t * x + phase / np.pi) + 1) / 2 * (gray_2 - gray_1) + gray_1
        image = image * y.reshape(im_width, 1, 1)
    else:
        raise ValueError("only support angle = 0 , 90, 180, 270")

    return image.astype(np.uint8)


def get_edge_image(im_height, im_width, gray_1=None, gray_2=None, angle=90):
    if gray_1 is None:
        gray_1 = [0, 0, 0]
    if gray_2 is None:
        gray_2 = [255, 255, 255]

    image = np.ones((im_height, im_width, 3), dtype=np.uint8) * np.array(gray_1, dtype=np.uint8)
    if angle == 90 or angle == 270:
        image[:, im_width // 2:, 0] = gray_2[0]
        image[:, im_width // 2:, 1] = gray_2[1]
        image[:, im_width // 2:, 2] = gray_2[2]
    elif angle == 0 or angle == 180:
        image[im_width // 2:, :, 0] = gray_2[0]
        image[im_width // 2:, :, 1] = gray_2[1]
        image[im_width // 2:, :, 2] = gray_2[2]
    else:
        cx = im_width // 2
        cy = im_height // 2
        theta = np.arctan(im_height / im_width)
        angle = angle * np.pi / 180  # / np.pi * 180
        if -theta < angle < theta:
            y1 = np.round(cy - np.tan(angle) * cx).astype(np.int32)
            y2 = np.round(cy + np.tan(angle) * cx).astype(np.int32)
            contours = [
                np.array([[[0, 0]], [[im_width - 1, 0]], [[im_width - 1, y1]], [[0, y2]]], dtype=np.int32)
            ]
        else:
            x1 = np.round(cx - cy / np.tan(angle)).astype(np.int32)
            x2 = np.round(cx + cy / np.tan(angle)).astype(np.int32)
            contours = [
                np.array([[[0, 0]], [[x2, 0]], [[x1, im_height-1]], [[0, im_height-1]]], dtype=np.int32)
            ]
        cv2.drawContours(image, contours, -1, gray_2, -1)
    return image
