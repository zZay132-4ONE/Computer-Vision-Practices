"""
@description: Detection of gender through analyzing the face
@author: Zzay
@create: 2022/03/01 17:53

使用keras实现性别识别，模型数据使用的是oarriaga/face_classification的模型。
使用OpenCV先识别到人脸，然后在通过keras识别性别
"""
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load image data
filepath = '../res/img/caixukun.jpg'
img = cv2.imread(filename=filepath)
img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
color = (0, 0, 255)

# Set face classifier
face_classifier = cv2.CascadeClassifier(
    "D:/Environment/annaconda3/envs/py36/Lib/site-packages/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml")
faces = face_classifier.detectMultiScale(
    image=img_gray, scaleFactor=1.2, minNeighbors=3, minSize=(140, 140))

# Set gender classifier (keras)
gender_classifier = load_model(
    filepath='../res/classifier/gender_models/simple_CNN.81-0.96.hdf5')
gender_labels = {0: 'Female', 1: 'Male'}

# Analyze faces' gender
for (x, y, width, height) in faces:
    # 将脸部区域的图像截取下来
    face = img[(y - 60): (y + height + 60), (x - 30): (x + width + 30)]
    # 将脸部区域的图像数据resize为48x48，再增加纬度。
    # 此操作为将数据传入分类器做出预测。
    face = cv2.resize(src=face, dsize=(48, 48))
    face = np.expand_dims(a=face, axis=0)
    face = face / 255.0
    # 使用gender classifier对脸部区域图像数据做出性别预测
    gender_label_arg = np.argmax(gender_classifier.predict(face))
    # 取出对当前脸部的性别预测值，匹配0/1进而在图像上做出标记
    gender_label = gender_labels[gender_label_arg]
    # 圈出head
    cv2.rectangle(img=img,
                  pt1=(x, y), pt2=(x + height, y + width),
                  color=color, thickness=2)
    cv2.putText(img=img, text=gender_label, org=(x + height, y),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255))

# Display
cv2.imshow(winname='Gender Detection', mat=img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()
