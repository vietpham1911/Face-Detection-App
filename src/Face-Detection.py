import cv2

# Time 1:36:00
# Load pre-trained data on face frontals from opencv (haar cascade algorithm)
# https://docs.opencv.org/
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('Many_faces.jpg')

# Capture video  from webcam
webcam = cv2.VideoCapture(0)

# Iterate over all frames
while True:
    successfull_frame_read, frame = webcam.read()
    # Convert img to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    for (x, y, w, h) in face_coordinates:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)
    # Press Q (in ASCII Code) to quit the app
    if key == 81  or key == 113:
        break
    # Release the VideoCapture object
    webcam.release()

print("Code completed")