from keras.models import load_model
import pandas as pd
import os
from utility_functions import pre_process_image
from skimage import io
import numpy as np


x_test = []
y_test = []
test = pd.read_csv('C:\\Users\\Siddharth\\traffic_sign\\test\\GTSRB\\GT-final_test.csv', sep=';')
image_size = (48, 48)

for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
    image_path = os.path.join('C:\\Users\\Siddharth\\traffic_sign\\test\\GTSRB\\Final_Test\\Images', file_name)
    image = io.imread(image_path)
    processed_image = pre_process_image(image, image_size)
    x_test.append(processed_image)
    y_test.append(class_id)
    
x_test = np.array(x_test, dtype='float32')
y_test = np.array(y_test)

model = load_model('model.h5')
y_pred = model.predict_classes(x_test)
accuracy = np.sum(y_pred == y_test) / np.size(y_pred)