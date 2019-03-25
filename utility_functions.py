from skimage import color, exposure, transform    
import numpy as np


def pre_process_image(image, image_size):
    hsv_image = color.rgb2hsv(image)
    hsv_image[:,:,2] = exposure.equalize_hist(hsv_image[:,:,2])
    image = color.hsv2rgb(hsv_image)
    image = transform.resize(image, image_size)
    image = np.rollaxis(image, -1)
    return image

def get_class_label(image_path):
    return int(image_path.split('\\')[-2])

def learning_rate_schedule(epoch, lr=0.01):
    return lr * (0.1 ** int(epoch/2))
    
    