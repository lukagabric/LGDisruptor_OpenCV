# OpenCV program to detect face in real time 
# import libraries of python OpenCV 
# where its functionality resides 
import cv2
import serial, time
arduino = serial.Serial('/dev/cu.usbserial-1410', 9600, timeout=.1)
time.sleep(1)

# load the required trained XML classifiers 
# https://github.com/Itseez/opencv/blob/master/ 
# data/haarcascades/haarcascade_frontalface_default.xml 
# Trained XML classifiers describes some features of some 
# object we want to detect a cascade function is trained 
# from a lot of positive(faces) and negative(non-faces) 
# images. 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

# capture frames from a camera
cap = cv2.VideoCapture(1)

timestamp = time.time()

ret, img = cap.read()

height, width, channels = img.shape
face_location_threshold_in_percent: int = 3
targetX = width/2
targetY = height/2

added_x_offset_in_percent = 0
added_y_offset_in_percent = -40

timestamp = time.time()
aim_lock_timestamp: float = 0
aim_lock_duration_threshold = 1.5

last_fired_time = timestamp
min_time_between_fire = 5

# loop runs if capturing has been initialized.
while 1:
	# reads frames from a camera
	ret, img = cap.read()

	data = arduino.readline()
	if data:
		print(data)

	# convert to gray scale of each frames
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detects faces of different sizes in the input image
	faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

	for (x,y,w,h) in faces:
		# To draw a rectangle in a face
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		
		currentTime = time.time()
		if currentTime - timestamp > 0.2:
			timestamp = currentTime
			aimX = x + w/2 
			aimY = y + h/2

			xOffsetInWidhtPercent = int(2 * 100 * (aimX - targetX) / float(width)) + added_x_offset_in_percent
			if abs(xOffsetInWidhtPercent) < face_location_threshold_in_percent:
				xOffsetInWidhtPercent = 0
				
			
			yOffsetInWidthPercent = int(2 * 100 * (aimY - targetY) / float(height)) + added_y_offset_in_percent
			if abs(yOffsetInWidthPercent) < face_location_threshold_in_percent:
				yOffsetInWidthPercent = 0
				
			
			command = "0," + str(xOffsetInWidhtPercent) + "," + str(yOffsetInWidthPercent) + "\n"

			print(command)
			arduino.write(command.encode())

	if len(faces) == 0:
		if aim_lock_timestamp > 0:
				aim_lock_timestamp = 0
		elif aim_lock_timestamp == 0:
			aim_lock_timestamp = time.time()

	currentTime = time.time()
	if aim_lock_timestamp > 0 and currentTime - aim_lock_timestamp > aim_lock_duration_threshold and currentTime - last_fired_time > min_time_between_fire:
		last_fired_time = currentTime
		command = "999,0,0\n"
		arduino.write(command.encode())

	# Display an image in a window
	cv2.imshow('img',img)

	# Wait for Esc key to stop
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
