"""
@description: Draw rectangles on an image
@author: Zzay
@create: 2022/03/01 14:10

图片上画矩形
"""
import cv2

# Load data
filepath = '../../res/img/xingye-1.png'
img = cv2.imread(filename=filepath)

# Get coordinates
x = y = 10
width = 100
color = (0, 0, 255)

# Draw rectangles
cv2.rectangle(img=img,  # 目标图像
              pt1=(x, y), pt2=(x + width, y + width),  # 矩形两对角点坐标
              color=color, thickness=1)  # 颜色以及矩形边框厚度

# Display
cv2.imshow(winname="Draw an rectangle on an image", mat=img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()
