#! coding:utf-8
"""
training.py

Created by 0160929 on 2017/07/24 8:59
"""
import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D

import dataset
random_state = 42

BATCH_SIZE = 25
EPOCH = 10

nClass=2

X,y = dataset.load(n=100)

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=random_state)

features = train_X.shape[1]

print(train_X[0])
print(train_X[1])
print(train_X[2])

print("train_X: ", train_X.shape)

x_inputs = Input(shape=(features, 1), name='x_inputs')  # (特徴量数, チャネル数)
x = Conv1D(128, 256, strides=256,padding='valid', activation='relu')(x_inputs)
x = Conv1D(32, 8, activation='relu')(x)  # (チャネル数, フィルタの長さ )
x = MaxPooling1D(4)(x)  # （フィルタの長さ）
x = Conv1D(32, 8, activation='relu')(x)
x = MaxPooling1D(4)(x)
x = Conv1D(32, 8, activation='relu')(x)
x = MaxPooling1D(4)(x)
x = Conv1D(32, 8, activation='relu')(x)
x = MaxPooling1D(4)(x)
x = Flatten()(x)
x = Dense(100, activation='relu')(x)  # （ユニット数）
x_outputs = Dense(nClass, activation='sigmoid', name='x_outputs')(x)

model = Model(inputs=x_inputs, outputs=x_outputs)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_X, train_y, 
			batch_size=BATCH_SIZE, epochs=EPOCH, shuffle=True,
			validation_data=(test_X, test_y)
			)

# モデル評価
loss, accuracy = model.evaluate(test_X, test_y, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))

''' pngへ出力 '''
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file="music_only.png", show_shapes=True)

