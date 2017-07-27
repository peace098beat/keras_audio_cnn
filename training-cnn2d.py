#! coding:utf-8
"""
training2.py

Created by 0160929 on 2017/07/25 22:41
"""

from __future__ import print_function
from pathlib import Path

import keras
import keras.callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import sklearn.metrics

Results_dir = Path(__file__).parent.joinpath("result")
if (not Results_dir.exists()): Results_dir.mkdir()

Root_dir = Path(__file__).parent

# ==================================================== #
# 1. Argment Parse
# ==================================================== #
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', '-ds', default="dataset", help='dataset name --ds datas-30db', type=str)
# parser.add_argument('--model', '-m', default="", help='***.py file path')
parser.add_argument('--dataset_size', '-n', default="100", help='100, 1000, 10000, 15000', type=int)
parser.add_argument('--epoch', '-ep', default="200", type=int)

parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(force=False)

args = parser.parse_args()

dataset_dirname = args.dataset
dataset_size = int(args.dataset_size)
epochs = args.epoch
batch_size = 32

# ==================================================== #
# Global Config
# ==================================================== #
MODEL_HDF5_NAME = "model.hdf5"
random_state = 21
num_classes = 2
data_augmentation = False


# ==================================================== #
# Load Datasets
# ==================================================== #
import datasets as ds
from sklearn.model_selection import train_test_split
# The data, shuffled and split between train and test sets:
DatasetDir = Root_dir.joinpath(dataset_dirname)
print(DatasetDir.as_posix())

X, y = ds.get_mell(DatasetDir.as_posix(), num=dataset_size)
X = X[:, :, :, np.newaxis]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# ==================================================== #
# Model
# ==================================================== #
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:], data_format="channels_last"))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 0-1 Normalize
x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())


# ==================================================== #
# Report Call Back
# ==================================================== #
model.summary()  ## print model summary
with open(Results_dir.joinpath("mode.json").as_posix(), "w") as fp:
    fp.write(model.to_json())

try:
    from keras.utils import plot_model
    plot_model(model, to_file=Results_dir.joinpath("model.png"), show_shapes=True)
except Exception as e:
    print(e)

# ==================================================== #
# Report Call Back
# ==================================================== #
#
# SaveMode_filepath = Results_dir.joinpath("model.hd5")
#
# # Call Back 1 : save model
# cb_check = keras.callbacks.ModelCheckpoint(SaveMode_filepath.as_posix(), period=100)
#
# # Call Back 1 : Tensor Board
# Log_dir = Results_dir.joinpath("logs")
# if (not Log_dir.exists()): Log_dir.mkdir()
# cb_tensorboard = keras.callbacks.TensorBoard(log_dir=Log_dir.as_posix(),
#                                              histogram_freq=0,
#                                              write_graph=True)

# Call Back 1  CSV
result_csv_path = Results_dir.joinpath("training.csv")
cb_csvlogger = keras.callbacks.CSVLogger(result_csv_path.as_posix(), separator=',', append=False)


# ==================================================== #
# Training
# ==================================================== #
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[cb_csvlogger])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=[cb_csvlogger])


# ==================================================== #
# evaluate model
# ==================================================== #
predicted = model.predict(x_test, batch_size=batch_size, verbose=1)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(predicted, axis=1)
target_names = ['No Sairen', 'On Sairen']

print(sklearn.metrics.classification_report(y_true, y_pred, target_names=target_names, digits=3))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_true, y_pred))
