"""
@description: Face Emotion Detection
@author: Zzay
@create: 2022/03/01 21:39

表情识别支持7种表情类型，生气、厌恶、恐惧、开心、难过、惊喜、平静等。
使用OpenCV识别图片中的脸，在使用tensorflow.keras进行表情识别。
"""
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Set basic data
color = (0, 0, 255)
emotion_labels = {
    0: 'Angry', 1: 'Disgusted', 2: 'Fearful', 3: 'Happy',
    4: 'Sad', 5: 'Surprised', 6: 'Calm'
}

# Load data
filepath = '../res/img/embiidCry.webp'
img = cv2.imread(filename=filepath)
img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# Set face classifier
face_classifier = cv2.CascadeClassifier(
    "D:/Environment/annaconda3/envs/py36/Lib/site-packages/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml")
faces = face_classifier.detectMultiScale(
    image=img_gray, scaleFactor=1.2, minNeighbors=3, minSize=(40, 40))

# Set emotion classifier
emotion_classifier = load_model(
    filepath='../res/classifier/emotion_models/simple_CNN.530-0.65.hdf5')

# Analyze faces
for (x, y, width, height) in faces:
    # 提取每个脸部区域，得到它们的灰图
    face_gray = img_gray[y: y + height, x: x + width]
    # 为了利用emotion分类器对emotion进行预测，对各个脸部的灰图进行resize以及归一化等操作
    face_gray = cv2.resize(src=face_gray, dsize=(48, 48))
    face_gray = face_gray / 255.0
    face_gray = np.expand_dims(a=face_gray, axis=0)
    face_gray = np.expand_dims(a=face_gray, axis=-1)
    # 对各个脸部的emotion进行预测
    emotion_label_arg = np.argmax(emotion_classifier.predict(face_gray))
    emotion_label = emotion_labels[emotion_label_arg]
    # 绘制矩形以及emotion标签
    cv2.rectangle(img=img,
                  pt1=(x + 10, y + 10), pt2=(x + height - 10, y + width - 10),
                  color=color, thickness=2)
    cv2.putText(img=img, text=emotion_label, org=(x + height // 2, y),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255))

# Display
cv2.imshow(winname='Emotion Detection of Faces', mat=img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()