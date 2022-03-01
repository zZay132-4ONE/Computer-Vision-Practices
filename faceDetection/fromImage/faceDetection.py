"""
@description: 
@author: Zzay
@create: 2022/03/01 14:10

1.图片转换成灰色（降低为一维的灰度，减低计算强度）
2.利用灰图进行识别，之后在原图片上相同位置画矩形
3.使用训练分类器查找人脸
"""
import cv2

# Load data
filepath = '../../res/img/xingye-1.png'
img = cv2.imread(filename=filepath)
img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
color = (0, 0, 255)

# Set classifier 设置人脸识别分类器
classifier = cv2.CascadeClassifier(
    "D:/Environment/annaconda3/envs/py36/Lib/site-packages/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
)

# Recognize faces 调用人脸识别
# 利用灰图进行识别，之后在原图片上相同位置画矩形（降低为一维的灰度，减低计算强度）
faceRects = classifier.detectMultiScale(
    image=img_gray,   # 需要分析的图像
    scaleFactor=1.2,  # 图像缩放比例
    minNeighbors=3,   # 对特征检测点周边多少有效点同时检测，这样可避免因选取的特征检测点太小而导致遗漏
    minSize=(32, 32)  # 特征检测点的最小尺寸
)
# 大于0则检测到人脸
if len(faceRects):
    # 单独框出每一张人脸
    for faceRect in faceRects:
        x, y, width, height = faceRect
        # Face
        cv2.rectangle(img=img,
                      pt1=(x, y), pt2=(x + width, y + height),
                      color=color, thickness=2)
        # Left eye
        cv2.circle(img=img,
                   center=(x + width // 4, y + height // 4 + 25),
                   radius= min(width // 8, height // 8),
                   color=color,
                   thickness=1)
        # Right eye
        cv2.circle(img=img,
                   center=(x + 3 * width // 4, y + height // 4 + 25),
                   radius=min(width // 8, height // 8),
                   color=color,
                   thickness=1)
        # Mouth
        cv2.rectangle(img=img,
                      pt1=(x + 3 * width // 8, y + 3 * height // 4),
                      pt2=(x + 5 * width // 8, y + 7 * height // 8),
                      color=color,
                      thickness=1)
# Display
cv2.imshow(winname='Face Detection from images', mat=img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()



