"""
@description: Convert an image into gray color space
@author: Zzay
@create: 2022/03/01 13:49

图片转换成灰色（降低为一维的灰度，减低计算强度）
"""
import cv2

# Load data
filepath = '../../res/img/xingye-1.png'
img = cv2.imread(filename=filepath)

# Convert the image data into gray color space
img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# Show image in colorful color space and gray space
cv2.imshow(winname='Image - colorful', mat=img)
cv2.imshow(winname='Face - gray', mat=img_gray)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()


