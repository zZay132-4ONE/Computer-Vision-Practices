"""
@description: Face Effect Composition 头像特效合成
@author: Zzay
@create: 2022/03/01 17:11

-实现思路：
使用OpenCV检测出头部位置，向上移动20像素添加虚拟帽子。帽子的宽度等于脸的大小，高度等比缩小。
需要注意的是如果高度小于脸部向上移动20像素的值，那么帽子的高度就等于最小高度=（脸部位置-20）。
-注意事项：
图片合成元件，要是黑背景图片，透明的图片也会有问题。
ps手动处理一下透明图片，添加新图层，选中alt+Del添加黑背景，把新图层层级放到最底部即可。
"""
import cv2

# Load image data
filename_face = '../res/img/ag-3.png'
filename_effect = '../res/img/compose/maozi-1.png'
img_face = cv2.imread(filename=filename_face)
img_effect = cv2.imread(filename=filename_effect)
img_face_gray = cv2.cvtColor(src=img_face, code=cv2.COLOR_BGR2GRAY)
color = (0, 0, 255)

# Classifier model
classifier = cv2.CascadeClassifier(
    "D:/Environment/annaconda3/envs/py36/Lib/site-packages/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
)

# Face Detection
faceRects = classifier.detectMultiScale(
    image=img_face_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32)
)
if len(faceRects):
    for faceRect in faceRects:
        # Get the position of head
        x, y, width, height = faceRect
        # Get the shape of the effect, and conduct geometric scaling on it (等比缩放)
        effect_shape = img_effect.shape
        img_effect_height = int(effect_shape[0] / effect_shape[1] * width)
        # Add the effect at the position which is 20 pixels higher than head
        if img_effect_height > (y - 20):
            img_effect_height = y - 20
        img_effect_size = cv2.resize(src=img_effect,
                                     dsize=(width, img_effect_height),  # new shape
                                     interpolation=cv2.INTER_NEAREST)  # 采用最邻近差值
        top = y - img_effect_height - 20
        if top <= 0:
            top = 0
        # 获取特效的大小和所在区域
        rows, cols, channels = img_effect_size.shape
        roi = img_face[top: top + rows, x: x + cols]
        # 创建特效图片的mask
        img2gray = cv2.cvtColor(src=img_effect_size, code=cv2.COLOR_RGBA2GRAY)
        # 选取一个全局阈值，将图像分为二值图像
        ret, mask = cv2.threshold(src=img2gray,  # 原图像
                                  thresh=10,  # 分类阈值
                                  maxval=255,  # 高于或低于阈值时分配的新值
                                  type=cv2.THRESH_BINARY)  # 黑白二值
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of logo in ROI
        img_bg = cv2.bitwise_and(src1=roi, src2=roi, mask=mask_inv)
        # Take only region of logo from logo image.
        img_fg = cv2.bitwise_and(src1=img_effect_size, src2=img_effect_size, mask=mask)
        # 修改原图像，将特效覆盖在特定区域
        img_composed = cv2.add(src1=img_bg, src2=img_fg)
        img_face[top: top + rows, x: x + cols] = img_composed

# Display
# cv2.imshow("Image background", img_bg)
# cv2.imshow("Image foreground", img_fg)
# cv2.imshow("Image foreground", img_composed)
cv2.imshow(winname="Face Effect Composition", mat=img_face)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()
