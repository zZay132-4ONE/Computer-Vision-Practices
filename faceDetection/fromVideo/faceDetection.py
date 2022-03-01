"""
@description: Face detection from videos
@author: Zzay
@create: 2022/03/01 16:42

-实现思路：
调用电脑的摄像头，把摄像的信息逐帧分解成图片。
基于图片检测标识出人脸的位置，把处理的图片逐帧绘制给用户，用户看到的效果就是视频的人脸检测。
"""
import cv2
import recognizer

# 参数0表示，获取第一个摄像头。
camera = cv2.VideoCapture(0)

# 显示摄像头，逐帧显示
while (True):
    ret, img = camera.read()
    recognizer.recognize(img=img)
    if cv2.waitKey(delay=1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
