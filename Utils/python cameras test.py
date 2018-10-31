import cv2

cams_test = 100
for i in range(-10, 10):
    cap = cv2.VideoCapture(i)
    test, frame = cap.read()
    print("i : "+str(i)+" /// result: "+str(test))
