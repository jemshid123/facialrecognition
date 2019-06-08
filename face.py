import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os

face_cascade = cv2.CascadeClassifier('face.xml')
file = input("input image path : ");
img = cv2.imread(file)
init = random.randint(1,100000)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
try:
    os.mkdir(os.getcwd() + "/faces")
except FileExistsError:  
    print ("folder exists")

try:
    os.mkdir(os.getcwd() + "/images")
except FileExistsError:  
    print ("folder exists")

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    plt.imshow(roi_color) 
    plt.savefig("faces/face{0}.jpg".format(init))
    init+=1
   
cv2.imshow('img',img)
plt.imshow(img) 
plt.savefig("images/img{0}.jpg".format(init))
cv2.waitKey(0)
cv2.destroyAllWindows()