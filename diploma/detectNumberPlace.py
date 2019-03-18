import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["axes.grid"] = False


# Эта функция расставляет точки контура в следующем порядке:
# top left, top right, bottom right, bottom left.
# У верхней левой точки сумма координат x и y минимальна, у нижней правой - максимальна
# У верхней правой точки разность координат x и y минимальна, у нижней левой - максимальна
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# Здесь выполняем преобразования перспективы:

# Считаем ширину верхней и нижней линии контура, максимум берем как ширину преобразованного изображения
# Считаем высоту правой и левой линии контура, максимум берем как высоту преобразованного изображения
# Создаем матрицу преобразований из исходного контура в новое прямоугольное изображение
# Применяем матрицу к исходному изображению, получаем изображение с "выровненной" перспективой
def warp_perspective(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


import itertools


# Отношение ширины к высоте должно вписываться в границы известных типов ГРЗ
def check_width_height_ratio(image):
    width, height = image.shape[1], image.shape[0]
    ratio = width / height
    return ratio > 1.5 and ratio < 6


# Метрика horizontal crosscuts должна лежать в допустимом интервале
def check_horizontal_crosscuts(image):
    threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 75, 10)
    height = threshold.shape[0]

    def _count_black_regions(height_coef):
        line_h = int(height_coef * height)
        line = threshold[line_h:line_h + 1, :][0]
        return [k for k, g in itertools.groupby(line)].count(0)

    return _count_black_regions(0.33) in range(4, 16) and _count_black_regions(0.67) in range(4, 16)


def detectedLicensePlates(filepath):
    image = cv2.imread(filepath)
    plt.imshow(image)
    # Преобразование в GrayScale и нормализация:
    image_gs = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_gs = cv2.GaussianBlur(image_gs, (3, 3), 0)
    plt.imshow(image_gs)
    # делаем адаптивную бинаризацию (то есть устраняем серые тона, оставляя только черный и белый цвета. Порог бинаризации подбирается автоматически)
    # инвертируем изображение, чтобы было больше белого цвета
    threshold = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)
    plt.imshow(threshold)
    # В бинаризованном изображении ищем закрытые контуры.
    # Потом каждый контур аппроксимируем в простую фигуру,
    # #и считаем количество сторон у этой фигуры. Если 4 стороны, и контур достаточно большой,
    # то, возможно, это рамка ГРЗ
    _, contours, hierarchy = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plate_candidates = []
    image_copied = image.copy()
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and peri > 100:
            plate_candidates.append(approx)
            cv2.drawContours(image_copied, [approx], -1, (0, 255, 0), 4)
    plt.imshow(image_copied)
    plt.show()
    license_plates = []
    for candidate in plate_candidates:
        warped = warp_perspective(image_gs, candidate.reshape(4, 2))
        if check_width_height_ratio(warped) and check_horizontal_crosscuts(warped):
            license_plates.append({"contour": candidate, "warped": warped})
    print("Found {0} license plate(s)".format(len(license_plates)))
    if len(license_plates) > 0:
        bbox = cv2.boundingRect(license_plates[0]["contour"])
        print(bbox)
        plt.imshow(license_plates[0]["warped"])
        plt.show()
        return license_plates[0]["warped"]

# license_plate = license_plates[0]["warped"]
##plt.imshow(license_plate_bin)
# plt.show()
