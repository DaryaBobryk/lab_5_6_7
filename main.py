from PIL import Image, ImageEnhance
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
import random
from PIL import Image, ImageDraw

def draw_histogram(grayscale):
    gray_key = []
    gray_count = []
    gray_result = []
    histogram_gray = list(grayscale.ravel())
    gray = dict(collections.Counter(histogram_gray))
    gray = sorted(gray.items(), key=lambda item: item[0])
    for element in gray:
        key = list(element)[0]
        count = list(element)[1]
        gray_key.append(key)
        gray_count.append(count)
    for i in range(0, 256):
        if i in gray_key:
            num = gray_key.index(i)
            gray_result.append(gray_count[num])
        else:
            gray_result.append(0)
    gray_result = np.array(gray_result)
    return gray_result


def histogram_equalization(histogram_e, lut_e, image_e):
    sum_temp = 0
    cf = []
    for i in histogram_e:
        sum_temp += i
        cf.append(sum_temp)
    for i, v in enumerate(lut_e):
        lut_e[i] = int(255.0 * (cf[i] / sum_temp) + 0.5)
    equalization_result = lut_e[image_e]
    return equalization_result


x = []
for i in range(0, 256):
    x.append(i)

image = cv2.imread('xb.jpg', cv2.COLOR_BGR2GRAY)
histogram = draw_histogram(image)
plt.bar(x, histogram)
plt.savefig('gis.png')
plt.show()

lut = np.zeros(256, dtype=image.dtype)
result = histogram_equalization(histogram, lut, image)
cv2.imwrite('his_grey.png', result)

image_equ = cv2.imread('kot.jpeg', cv2.COLOR_BGR2GRAY)  # Читать изображение
histogram_equ = draw_histogram(image_equ)  # Преобразование гистограммы
plt.bar(x, histogram_equ)  # Рисуем гистограмму обработанного изображения
plt.savefig('after_histogram.png')
plt.show()

plt.plot(x, lut)  # Нарисуйте график кривой преобразования серой шкалы
plt.savefig('Grayscale_transformation_curve.png')
plt.show()

cv2.imshow('former', image)
cv2.imshow("histogram_equalization", result)
cv2.waitKey(0)
cv2.destroyAllWindows()




im = Image.open("kot.jpeg")

enhancer = ImageEnhance.Contrast(im)

factor = 1
im_output = enhancer.enhance(factor)
im_output.save('original-image.png')

factor = 0.5
im_output = enhancer.enhance(factor)
im_output.save('less-contrast-image.png')

factor = 1.5
im_output = enhancer.enhance(factor)
im_output.save('more-contrast-image.png')


img = cv2.imread('kot.jpeg')
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)

ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)

kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
kernely = np.array([[0, -1], [1, 0]], dtype=int)
x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
y = cv2.filter2D(binary, cv2.CV_16S, kernely)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
y = cv2.filter2D(binary, cv2.CV_16S, kernely)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

dst = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
Laplacian = cv2.convertScaleAbs(dst)

plt.subplot(231), plt.imshow(img_RGB), plt.title('исходное изображение'), plt.axis('off')  # ось координат отключена
plt.subplot(232), plt.imshow(gaussianBlur, cmap=plt.cm.gray), plt.title('Gauss'), plt.axis('off')
plt.subplot(233), plt.imshow(Roberts, cmap=plt.cm.gray), plt.title('Roberts'), plt.axis('off')
plt.subplot(234), plt.imshow(Prewitt, cmap=plt.cm.gray), plt.title('Prewitt'), plt.axis('off')
plt.subplot(235), plt.imshow(Sobel, cmap=plt.cm.gray), plt.title('Sobel'), plt.axis('off')
plt.subplot(236), plt.imshow(Laplacian, cmap=plt.cm.gray), plt.title('Laplacian'), plt.axis('off')

plt.show()


mode = int(input('mode:')) #Считываем номер преобразования.
image = Image.open("kot.jpeg") #Открываем изображение.
draw = ImageDraw.Draw(image) #Создаем инструмент для рисования.
width = image.size[0] #Определяем ширину.
height = image.size[1] #Определяем высоту.
pix = image.load() #Выгружаем значения пикселей.

if (mode == 0):
	for i in range(width):
		for j in range(height):
			a = pix[i, j][0]
			b = pix[i, j][1]
			c = pix[i, j][2]
			S = (a + b + c) // 3
			draw.point((i, j), (S, S, S))

if (mode == 1):
	depth = int(input('depth:'))
	for i in range(width):
		for j in range(height):
			a = pix[i, j][0]
			b = pix[i, j][1]
			c = pix[i, j][2]
			S = (a + b + c) // 3
			a = S + depth * 2
			b = S + depth
			c = S
			if (a > 255):
				a = 255
			if (b > 255):
				b = 255
			if (c > 255):
				c = 255
			draw.point((i, j), (a, b, c))

if (mode == 2):
	for i in range(width):
		for j in range(height):
			a = pix[i, j][0]
			b = pix[i, j][1]
			c = pix[i, j][2]
			draw.point((i, j), (255 - a, 255 - b, 255 - c))

if (mode == 3):
	factor = int(input('factor:'))
	for i in range(width):
		for j in range(height):
			rand = random.randint(-factor, factor)
			a = pix[i, j][0] + rand
			b = pix[i, j][1] + rand
			c = pix[i, j][2] + rand
			if (a < 0):
				a = 0
			if (b < 0):
				b = 0
			if (c < 0):
				c = 0
			if (a > 255):
				a = 255
			if (b > 255):
				b = 255
			if (c > 255):
				c = 255
			draw.point((i, j), (a, b, c))

if (mode == 4):
	factor = int(input('factor:'))
	for i in range(width):
		for j in range(height):
			a = pix[i, j][0] + factor
			b = pix[i, j][1] + factor
			c = pix[i, j][2] + factor
			if (a < 0):
				a = 0
			if (b < 0):
				b = 0
			if (c < 0):
				c = 0
			if (a > 255):
				a = 255
			if (b > 255):
				b = 255
			if (c > 255):
				c = 255
			draw.point((i, j), (a, b, c))

if (mode == 5):
	factor = int(input('factor:'))
	for i in range(width):
		for j in range(height):
			a = pix[i, j][0]
			b = pix[i, j][1]
			c = pix[i, j][2]
			S = a + b + c
			if (S > (((255 + factor) // 2) * 3)):
				a, b, c = 255, 255, 255
			else:
				a, b, c = 0, 0, 0
			draw.point((i, j), (a, b, c))

image.save("ans.jpg", "JPEG")
del draw


