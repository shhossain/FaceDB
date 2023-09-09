import cv2



path = r"F:\Code\Python\FaceDB\test\imgs\joe_biden_2.jpeg"
img = cv2.imread(path)
cv2.imshow("img", img)
cv2.waitKey(0)