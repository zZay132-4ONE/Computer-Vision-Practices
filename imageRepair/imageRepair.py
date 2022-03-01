"""
@description: Repair an image with some outstanding errors
@author: Zzay
@create: 2022/03/01 22:16

简单来说就是开发者标定噪声的特征，在使用噪声周围的颜色特征推理出应该修复的图片的颜色，从而实现图片修复。
1.标定噪声的特征，使用cv2.inRange二值化标识噪声对图片进行二值化处理，
  具体代码：cv2.inRange(img, np.array([240, 240, 240]), np.array([255, 255, 255]))，
  把[240, 240, 240]~[255, 255, 255]以外的颜色处理为0。
2.使用OpenCV的dilate方法，扩展特征的区域，优化图片处理效果。
3.使用inpaint方法，把噪声的mask作为参数，推理并修复图片。
"""
import numpy as np
import cv2

# Load image data
filepath = '../res/img/inpaint.png'
img = cv2.imread(filename=filepath)
height, width, channel = img.shape[0: 3]

# 图片二值化处理，把[240, 240, 240]~[255, 255, 255]以外的颜色变成0
threshold = cv2.inRange(src=img,
                        lowerb=np.array([240, 240, 240]),
                        upperb=np.array([255, 255, 255]))

# 创建形状和尺寸的结构元素
kernel = np.ones(shape=(3, 3), dtype=np.uint8)

# 扩张待修复区域
mask = cv2.dilate(src=threshold, kernel=kernel, iterations=1)
specular = cv2.inpaint(src=img, inpaintMask=mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

# Display
cv2.namedWindow("Image", 0)
cv2.resizeWindow("Image", int(width / 2), int(height / 2))
cv2.imshow("Image", img)
cv2.namedWindow("newImage", 0)
cv2.resizeWindow("newImage", int(width / 2), int(height / 2))
cv2.imshow("newImage", specular)
cv2.waitKey(0)
cv2.destroyAllWindows()