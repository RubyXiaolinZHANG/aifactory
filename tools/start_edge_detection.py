import cv2
import numpy as np

from aifactory.libs.common.image_processor import enhanced_edge_detection


def apply_edge_detection(bgr):
    gray_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edge_img = enhanced_edge_detection(gray_img, threshold=12, min_area=20, min_size=10, remove_boundary=0)
    edge_masked_img = edge_img[:,:,np.newaxis]//255 * bgr
    cv2.imwrite("image.png", bgr)
    cv2.imwrite("edge.png", edge_img)
    cv2.imwrite("edge_image.png", edge_masked_img)


if __name__ == "__main__":
    image_file = 'F:/database/vimeo_png/sequences/00028/0190/im1.png'
    image = cv2.imread(image_file)
    apply_edge_detection(image)
