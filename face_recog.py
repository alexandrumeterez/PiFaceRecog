from imutils.video import VideoStream, FPS
import face_recognition
import imutils
import pickle
import cv2
import time
from centroid_tracker import CentroidTracker

tracker = CentroidTracker()
FRAMES_TO_TRACK = 50
print("Loading encodings")
data = pickle.loads(open("encodings.pkl", "rb").read())
detector = cv2.CascadeClassifier("/usr/local/share/OpenCV/lbpcascades/lbpcascade_frontalface_improved.xml")

print("Starting video stream")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
counter = 0
names = []
start = time.time()
fps_counter = 0
while True:
    fps_counter += 1
    counter += 1
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rects = detector.detectMultiScale(gray, 1.1, 5)
        
    objects = tracker.update(rects)
    if counter % FRAMES_TO_TRACK == 0:
        counter = 0
        boxes = [(y, x+w, y+h, x) for (x,y,w,h) in rects]
        encondings = face_recognition.face_encodings(rgb, boxes)
        names = []
        for encoding in encondings:
            matches = face_recognition.compare_faces(data['encodings'], encoding)
            name = "Unknown"
            
            if True in matches:
                matchedIds = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIds:
                    name = data['names'][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)
            print(name)
    for (centroid, name) in zip(objects.values(), names):
        y = centroid[1] + 15
        x = centroid[0] - 15
        cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    end = time.time()
    print(fps_counter / (end-start))
    with open("names.pkl", "wb") as f:
        pickle.dump(names, f)
cv2.destroyAllWindows()
vs.stop()
