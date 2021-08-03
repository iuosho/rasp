# -*- coding: utf-8 -*-
import time
import os
import cv2 as cv

fn = 'my_pic.jpg'

cap = cv.VideoCapture(1)
if cap.isOpened() is False:
	raise IOError  

while(True):
	frame = cap.read()
    	cv.imshow('frame',frame)

	cv.imwrite('/home/pi/work/camdata' + fn, frame)

	img = cv.imread('/home/pi/work/camdata' + fn)

	grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
	eye_cascade = cv.CascadeClassifier('haarcascades/haarcascade_eye.xml')
	facerect = face_cascade.detectMultiScale(grayimg, scaleFactor=1.2, minNeighbors=2, minSize=(1, 1))
	eyerect = eye_cascade.detectMultiScale(grayimg)

	print(facerect)	
	print(eyerect)

		# 顔を検出した場合
	if len(facerect) > 0:
		for rect in facerect:
			cv.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness=3)

		# 目を検出した場合
	if len(eyerect) > 0:
		for rect in eyerect:
			cv.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 255, 0), thickness=3)

		# 結果の画像を表示する
	cv.imshow('camera', img)
	cv.imwrite(fn, img)
	if cv.waitKey(1) & 0xFF == ord('q'):break

cv.destroyAllWindows()