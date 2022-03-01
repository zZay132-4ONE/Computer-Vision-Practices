import cv2

# Load image data
filepath = '../res/img/meinv.png'
img = cv2.imread(filepath)

# Set up
win_name = 'Image-tesseractOCR'
cv2.namedWindow(winname=win_name)
cv2.imshow(winname=win_name, mat=img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()
