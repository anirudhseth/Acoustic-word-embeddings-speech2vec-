import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import cv2
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def _process_csv_file(file):

    with open(file, 'r') as fr:
        files = fr.readlines()
    return files

class BalanceCovidDataset(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data_dir,  csv_file,is_training=True,isValidation=False,  batch_size=8,
            input_shape=(224, 224), n_classes=3,  num_channels=3, mapping={'normal': 0,'pneumonia': 1,'COVID-19': 2},
            shuffle=True, augmentation=True, covid_percent=0.3, class_weights=[1., 1., 6.]):
        'Initialization'
        
        self.datadir = data_dir
        self.dataset = _process_csv_file(csv_file)
#         print(self.datadir)
        self.is_training = is_training
        self.isValidation = isValidation
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.mapping = mapping
        self.shuffle = True
        self.covid_percent = covid_percent
        self.class_weights = class_weights
        self.n = 0

        if augmentation:
            self.augmentation = ImageDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
#                 horizontal_flip=True,
#                 brightness_range=(0.9, 1.1),
                zoom_range=(0.85, 1.15),
                fill_mode='constant',
                cval=0.,
            )

        datasets = {'normal': [], 'pneumonia': [], 'COVID-19': []}
        for l in self.dataset:
            datasets[l.split(',')[2]].append(l)
        

        self.datasets = [
            datasets['normal'] + datasets['pneumonia'],
            datasets['COVID-19'],
        ]
#         print('#normal+pneumonia: ', len(self.datasets_temp[0]), ' #covid19: ', len(self.datasets_temp[1]))
        #df = pd.DataFrame([l.split() for l in self.dataset])   

        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        batch_x, batch_y = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0

        return batch_x, batch_y

    def __len__(self):
        return int(np.ceil(len(self.datasets[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            for v in self.datasets:
                np.random.shuffle(v)

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros((self.batch_size, *self.input_shape, self.num_channels)), np.zeros(self.batch_size)

        batch_files = self.datasets[0][idx * self.batch_size:(idx + 1) * self.batch_size]

        # upsample covid cases
        covid_size = max(int(len(batch_files) * self.covid_percent), 1)
        covid_inds = np.random.choice(np.arange(len(batch_files)), size=covid_size, replace=False)
        covid_files = np.random.choice(self.datasets[1], size=covid_size, replace=False)
        for i in range(covid_size):
            batch_files[covid_inds[i]] = covid_files[i]

        for i in range(len(batch_files)):
            sample = batch_files[i].split(',')

            if self.isValidation:
                folder ='val2'
            elif self.is_training:
                folder = 'train2'
            else:
                folder = 'test2'

            x = cv2.imread(os.path.join(self.datadir, folder, sample[1]))
            h, w, c = x.shape
            x = x[int(h/6):, :]
            x = cv2.resize(x, self.input_shape)

            if self.is_training and hasattr(self, 'augmentation'):
                x = self.augmentation.random_transform(x)

            x = x.astype('float32') / 255.0
            y = self.mapping[sample[2]]

            batch_x[i] = x
            batch_y[i] = y

        class_weights = self.class_weights
        weights = np.take(class_weights, batch_y.astype('int64'))

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_classes)
