import cv2 

cap = cv2.VideoCapture('/dev/v4l/by-id/usb-Sony_ILX-LR1_D518801E89CD-video-index0') # /dev/v4l/by-id/usb-Sony_ILX-LR1_D518801E89CD-video-index0

# cap.open()

while True:
	ret, frame = cap.read()
	
	# print(frame.shape)
	
	resized_frame = cv2.resize(frame, (640,480))
	# cv2.imshow("sony camera", resized_frame)
	# cv2.waitKey(33)
	
	
cap.release()
