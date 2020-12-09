import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras

model = keras.models.load_model('sign_digits.h5')

# open camera and take a get the video feed from camera

cam = cv2.VideoCapture(0)
cv2.namedWindow('video')

IMG_SIZE = 100
font = cv2.FONT_HERSHEY_SIMPLEX 

while True:
	ret, frame = cam.read()
	if not ret:
		print('failed to grab frame')
		break 

	img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
	x_test = np.array(img)
	x_test = x_test.reshape(1, IMG_SIZE, IMG_SIZE, 3)

	# get the predicted output
	y_pred = np.argmax(model.predict(x_test))

	cv2.putText(frame, 'Number: ' + str(y_pred), (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
	cv2.imshow('video', frame)

	k = cv2.waitKey(1)
	if k % 256 == 27:
		# Esc pressed
		print('Escape pressed, closing...')
		break

cam.release()
cv2.destroyAllWindows()