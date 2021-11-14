import sys
import imutils
import cv2
import cv2 as cv
import numpy as np
import pytesseract

# Нахождения позиции меток
# @fname: название имени файла
# @template: имя файла метки
def find_match_location(fname,template):
    # Загрузка изображения и перевод его с ч/б формат
    template = cv2.imread(template)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    # Загрузка изображения
    image = cv2.imread(fname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    # Обход по изоюражению с различными коэфициентами масштабирования
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # Масштабирование
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        # Если размер изображения получился меньше размера метки то пропуск
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        # Детектирование краев
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
        # Получение координат
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        return (endX + startX)//2, (endY + startY)//2

# Кадрирование изображения
def crop_image(fname, coords):
    img = cv.imread(fname)
    mxy = max(coords[0][1], coords[1][1])
    mny = min(coords[0][1], coords[1][1])
    mxx = max(coords[0][0], coords[1][0])
    mnx = min(coords[0][0], coords[1][0])
    crop_img = img[mny:mxy, mnx:mxx]
    print(mnx, mny, mxx, mxy)
    cv.imwrite('cropped.png', crop_img)

# Получение текста
def extract_text():
    s = pytesseract.image_to_string('cropped.png', lang='rus')
    return s

def whitespace_and_newline_stripper(s):
    for i in range(100):
        s = s.replace('  ','')
        s = s.replace('\n\n','')
    return s

out = sys.argv[2]
# Получение координат левой метки
left_match = find_match_location(sys.argv[1], 'target.PNG')
# Получение координат правой метки
right_match = find_match_location(sys.argv[1], 'target2.PNG')
crop_image(sys.argv[1], (left_match, right_match))
# Вывод результатов
with open(out,mode='w+') as f:
    f.write(whitespace_and_newline_stripper(extract_text()))