import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras

model = keras.models.load_model('sign_digits.h5')

# open camera and take a snapshot

cam = cv2.VideoCapture(0)
cv2.namedWindow('video')
img_counter = 0

while True:
	ret, frame = cam.read()
	if not ret:
		print('failed to grab frame')
		break 
	cv2.imshow('video', frame)

	k = cv2.waitKey(1)
	if k % 256 == 27:
		# Esc pressed
		print('Escape pressed, closing...')
		break
	elif k % 256 == 32:
		# Space pressed
		img_name = 'test_{}.png'.format(img_counter)
		cv2.imwrite(img_name, frame)
		print('{} written!'.format(img_name))
		img_counter += 1

cam.release()
cv2.destroyAllWindows()

IMG_SIZE = 100

# create the data of all the saved images
X = []
for i in range(img_counter):
	DIR = 'test_{}.png'.format(i)
	img = cv2.imread(DIR, cv2.IMREAD_COLOR)
	resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
	X.append(resized_img)

X_test = np.array(X)

# predict the output for all the saved images
Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred, axis = 1)

for i in range(img_counter):
	print('Number: ' + str(Y_pred[i]))