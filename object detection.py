import cv2

# Source data
img_file = "img1.jpg"

# create an openCV image
img = cv2.imread(img_file)

# pre trained Pedestrian classifiers
pedestrian_classifier = 'haarcascade_fullbody.xml'

# convert color image to grey image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create trackers using classifiers using OpenCV
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)

# detect cars
pedestrian = pedestrian_tracker.detectMultiScale(gray_img)

# display the coordinates of different humans - multi dimensional array
print(pedestrian)

# draw rectangle around the pedestrian
for (x, y, w, h) in pedestrian:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(img, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Finally display the image with the markings
cv2.imshow('my detection', img)

# wait for the keystroke to exit
cv2.waitKey()


print("I'm done")
