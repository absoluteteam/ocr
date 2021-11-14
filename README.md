# Фудшеринг-проект "Делиться - легко!"
Алгоритм распознавания рукописного и печатного текста, использующий преимущественно библиотеки tensorflow и opencv

##Установка 

####Ubuntu:
Требуется наличие Python версии 3.6 и выше\
`sudo apt install tesseract-ocr -y`\
`sudo apt install tesseract-ocr-all -y`\
`python -m pip install -r req.txt`

####Windows:
Требуется наличие Python версии 3.6 и выше\
Также необходимо установить tesseract\
`python -m pip install -r req.txt`

####Запуск распознавания печатного текста
`python temp_ocr.py <имя файла фотографии> <имя выходного файла>`

####Запуск распознавания рукописного текста
`python temp_htr.py <имя файла фотографии>`

####Запуск тренировки нейросети
`python temp_htr.py <количество тренировок>`