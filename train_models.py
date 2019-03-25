import os
import glob
import numpy as np
from utility_functions import pre_process_image, get_class_label, learning_rate_schedule
from skimage import io
from models import cnn_model
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import h5py


number_of_classes = 43

def read_training_images(image_size, root_dir):
    try:
        with h5py.File('Input.h5') as hf:
            x, y = hf['images'][:], hf['labels'][:]
    except Exception as e:
        print(e)
        images = []
        labels = []
        all_images_path = glob.glob(os.path.join(root_dir, '*/*.ppm'))
        np.random.shuffle(all_images_path)
        
        for image_path in all_images_path:
            try:
                image = io.imread(image_path);
                image = pre_process_image(image, image_size)
                images.append(image);
                label = get_class_label(image_path)
                labels.append(label)
            except(IOError, OSError):
                pass
        
        x = np.array(images, dtype='float32')
        y = np.eye(number_of_classes, dtype='uint8')[labels]
        
        with h5py.File('Input.h5', 'w') as hf:
            hf.create_dataset('images', data=x)
            hf.create_dataset('labels', data=y)
            
    return x, y

def train_model():
    image_size = (48, 48)
    root_dir = 'C:\\Users\\Siddharth\\traffic_sign\\test\\GTSRB\\Final_Training\\Images\\'
    x, y = read_training_images(image_size, root_dir)
    model = cnn_model(image_size, number_of_classes)
    learning_rate = 0.01
    sgd = SGD(lr=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    batch_size = 64
    epochs = 5
    model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2,
              callbacks=[LearningRateScheduler(learning_rate_schedule),
                         ModelCheckpoint('model.h5', save_best_only=True)])
    
if __name__ == "__main__":
    train_model()