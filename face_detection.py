import cv2

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face_bgr = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2BGR)
        face_resized = cv2.resize(face_bgr, (w, h))
        frame[y:y+h, x:x+w] = face_resized
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the frame with rectangles around detected faces
    cv2.imshow('Face Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

"""
Ada pattern recognition using haarcascade
Sama ada apply filter grayscale ke bagian wajah yang ke recognize
"""