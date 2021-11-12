import os
import cv2
import time
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM, BatchNormalization

from keras import backend as K
from keras.constraints import maxnorm
import tensorflow as tf

def emnist_model(labels_num=None):
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def emnist_train(model, X_train, y_train_cat, X_test=None, y_test_cat=None):
    t_start = time.time()
    # Set a learning rate reduction
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    # Required for learning_rate_reduction:
    #keras.backend.get_session().run(tf.global_variables_initializer())
    model.fit(X_train, y_train_cat,
              #validation_data=(X_test, y_test_cat),
              #callbacks=[learning_rate_reduction],
              #batch_size=64,
              #epochs=9
              )
    print("Training done, dT:", time.time() - t_start)
    return model

def load_image_as_gray(path_to_image):
    img = Image.open(path_to_image)
    return np.array(img.convert("L"))

def load_image(path_to_image):
    img = Image.open(path_to_image)
    return img

def convert_rgba_to_rgb(pil_img):
    pil_img.load()
    background = Image.new("RGB", pil_img.size, (255, 255, 255))
    background.paste(pil_img, mask = pil_img.split()[3])
    return background

def prepare_rgba_img(img_path):
    img = load_image(img_path)
    if np.array(img).shape[2] == 4:
      new_img = convert_rgba_to_rgb(img)
      return new_img
    return img

# разбитие строки на отдельные буквы
def letters_extract(image_file: str, out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)
    return letters

'''
# размытие изображений
for lett in os.listdir("Cyrillic/"):
  for l in os.listdir(f"Cyrillic/{lett}"):
    if l != ".ipynb_checkpoints":
      img = Image.open(f"Cyrillic/{lett}/"+l)
      blurImage = img.filter(ImageFilter.BoxBlur(15))
      blurImage.save(f"Cyrillic/{lett}/"+"blur_"+l)

# поворот изображений на +20 градусов
for lett in os.listdir("Cyrillic/"):
  for l in os.listdir(f"Cyrillic/{lett}"):
    if (l != ".ipynb_checkpoints") & ("blur_" not in l):
      img = Image.open(f"Cyrillic/{lett}/"+l)
      rotImage = img.rotate(20)
      rotImage.save(f"Cyrillic/{lett}/"+"rot20_"+l)

# поворот изображений на -20 градусов
for lett in os.listdir("Cyrillic/"):
  for l in os.listdir(f"Cyrillic/{lett}"):
    if (".ipynb_checkpoints" not in l) & ("rot20_" not in l) & ("blur_" not in l):
      img = Image.open(f"Cyrillic/{lett}/"+l)
      rotImage = img.rotate(-20)
      rotImage.save(f"Cyrillic/{lett}/"+"rot02_"+l)
'''
'''

# изменение размера изображений до 28x28
for lett in os.listdir("Cyrillic/"):
  print(lett)
  for l in os.listdir(f"Cyrillic/{lett}"):
    if l != ".ipynb_checkpoints":
      img = Image.open(f"Cyrillic/{lett}/"+l)
      resized = img.resize((28, 28))
      resized.save(f"Cyrillic/{lett}/"+l)

# преобразование изображений из RGBA в RGB
for lett in os.listdir("Cyrillic/"):
  for l in os.listdir(f"Cyrillic/{lett}"):
    if l != ".ipynb_checkpoints":
      rgb_img = prepare_rgba_img(f"Cyrillic/{lett}/"+l)
      rgb_img.save(f"Cyrillic/{lett}/"+l)

'''
# разбитие строки на отдельные буквы
def letters_extract(image_file: str, out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Sort array in place by X-coordinate
    letters.sort(key=lambda x: x[0], reverse=False)
    return letters

X_train = list()
y_train = list()
chars = os.listdir('Cyrillic')
for i in chars:
    for t,j in enumerate(os.listdir(f'Cyrillic/{i}')):
        if(0 <= ord(i)-ord('А') <= 31):
            y_train.append((ord(i)-ord('А'))/31.0)
            X_train.append(np.array(load_image_as_gray(f'Cyrillic/{i}/{j}')))
            if(t> 20):
                break
#print(X_train[0])
#print(y_train)
X_train = np.reshape(np.array(X_train), (np.array(X_train).shape[0], 28, 28, 1))
X_train = X_train.astype(np.float32)
X_train /= 255.0
#print(X_train)
y_train = np.array(y_train)
print(y_train)

model = emnist_model(32)
model = emnist_train(model, X_train, y_train)

