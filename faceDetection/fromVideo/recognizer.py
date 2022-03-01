"""
@description: 
@author: Zzay
@create: 2022/03/
"""
import cv2

# 图片识别方法封装
def recognize(img):
    # Convert the image into gray color space for analysis
    img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    # Analyze the image
    classifier = cv2.CascadeClassifier(
        "D:/Environment/annaconda3/envs/py36/Lib/site-packages/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
    )
    faceRects = classifier.detectMultiScale(
        image=img_gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50)
    )
    # Draw rectangles
    for faceRect in faceRects:
        x, y, width, height = faceRect
        cv2.rectangle(img=img,
                      pt1=(x, y), pt2=(x + width, y + height),
                      color=(0, 0, 255), thickness=2)
    # Display
    cv2.imshow(winname="Capture", mat=img)

