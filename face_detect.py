import cv2
import sys

# Get user supplied values
#imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
#image = cv2.imread(imagePath)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
#faces = faceCascade.detectMultiScale(
#    gray,
#    scaleFactor=1.1,
#    minNeighbors=5,
#    minSize=(30, 30),
#    flags = 1
#)

#print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
#for (x, y, w, h) in faces:
#    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #cropimg = image[y:y+h, x:x+w]
    #cv2.imshow("image", cropimg)
    #cv2.waitKey(0)

#cv2.imshow("Faces found", image)

#cv2.waitKey(0)


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = 1
    )
    # Display the resulting frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()