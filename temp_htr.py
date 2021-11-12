import os
import sys
import cv2
import time
from PIL import Image
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM, BatchNormalization


# Модель последовательной нейронной сети
# @labels_num: количество символов для распознавания
def emnist_model(labels_num=None):
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Отключение четверти нейронов на слое для улучшения распознаваемости
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    # Отключение половины нейронов на слое для улучшения распознаваемости
    model.add(Dropout(0.5))
    # Каждый нейрон последнего слоя соответствует своему символу
    model.add(Dense(labels_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

# Тренировка нейронной сети с засечением времени
# @model: модель нейронной сети
# @X_train: тренировочные данные
# @Y_Train: выходные данные
# @epochs: количество тренировок
def emnist_train(model, X_train, Y_train,epochs = 1):
    t_start = time.time()
    model.fit(X_train, Y_train,
              batch_size=64,
              epochs=epochs
              )
    print("Training done, dT:", time.time() - t_start)
    return model

# Загрузка изображения по пути @path_to_image и перевод его в ч/б формат
# @path_to_image: путь к файлу
def load_image_as_gray(path_to_image):
    img = Image.open(path_to_image)
    return np.array(img.convert("L"))

# Загрузка изображения по пути @path_to_image в цветном формате
# @path_to_image: путь к файлу
def load_image(path_to_image):
    img = Image.open(path_to_image)
    return img

# Удаление альфа канала у изображения @pil_img
# @pil_img: путь к файлу
def convert_rgba_to_rgb(pil_img):
    pil_img.load()
    background = Image.new("RGB", pil_img.size, (255, 255, 255))
    background.paste(pil_img, mask = pil_img.split()[3])
    return background

# Возвращает изображение по пути @img_path с удаленным альфа каналом в цветном формате
# @img_path: путь к файлу
def prepare_rgba_img(img_path):
    img = load_image(img_path)
    # Проверка на существование альфа канала
    if np.array(img).shape[2] == 4:
      new_img = convert_rgba_to_rgb(img)
      return new_img
    return img

# Разбитие слова на отдельные буквы
# @image_file: имя файла картинки
# @out_size: размер выходного изображения
def letters_extract(image_file: str, out_size=28):
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Получение контуров
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    # Получение символов из контуров
    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            # Обрезание фотографии
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            # Достраивание до квадрата
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

            # Изменение размера изображения каждого символа
            letters.append((x, w, cv2.resize(letter_square, (out_size, out_size), interpolation=cv2.INTER_AREA)))

    # Сортировка массива символов по координате X по возрастанию
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

# Обработка консольных аргументов
if len(sys.argv) != 2:
    # Выход если количество аргументов не равно 1
    exit(0)

# Проверка является ли аргумент целым числом
try:
    epochs = int(sys.argv[1])
except Exception:
    epochs = -1


# Если аргумент приложения - целое число
# Это означает что нейронная сеть будет тренироваться введенное количество раз
if epochs != -1:
    # Тренировка модели

    # Получение тренировочных данных
    X_train = list()
    y_train = list()
    chars = os.listdir('Cyrillic')
    for i in chars:
        for t, j in enumerate(os.listdir(f'Cyrillic/{i}')):
            # Проверка является ли символ - русской буквой
            if (0 <= (ord(i) - ord('А')) <= 31):
                temp = list()
                # Создание ожидаемого результата
                # 1 будет стоят элемента с индексом совпадающим с буквой ответа для данного теста
                # для всех остальных индексов это будет 0
                for k in range(32):
                    if k == (ord(i) - ord('А')):
                        temp.append(1.0)
                    else:
                        temp.append(0.0)
                y_train.append(np.array(temp))
                # Получение входных данных
                X_train.append(np.array(load_image_as_gray(f'Cyrillic/{i}/{j}')))

    # Изменение измерений входных данных
    X_train = np.reshape(np.array(X_train), (np.array(X_train).shape[0], 28, 28, 1))
    # Нормализация цветов
    X_train = X_train.astype(np.float32)
    X_train /= 255.0
    y_train = np.array(y_train)
    # Проверка существует ли модель
    try:
        # Загрузка предыдущей модели
        model = keras.models.load_model('model')
    except Exception:
        # Cоздание новой модели
        model = emnist_model(32)
    # Тренировка модели
    model = emnist_train(model, X_train, y_train,epochs)
    # Сохранение модели
    model.save('model')
else:
    # Распознавание текста моделью

    # Разбиение слова на буквы
    lttrs = letters_extract(sys.argv[1], 28)
    for i in lttrs:
        model = keras.models.load_model('model')
        # Изменение измерений входных данных
        symb = np.array([i[2]]).reshape((1,28,28,1))
        # Нормализация цветов
        symb = symb.astype(np.float32)
        symb /= 255.0

        # Получение результата
        ans = model.predict(symb)
        index = 0
        mx = ans[0][0]
        # Обработка результата
        # Сортировка букв по максимальной уверенности нейронной сети
        for i in range(32):
            if ans[0][i] > mx:
                mx = ans[0][i]
                index = i
        # Вывод символа и уверенность нейронной сети
        print(chr(index + ord('А')),mx)




