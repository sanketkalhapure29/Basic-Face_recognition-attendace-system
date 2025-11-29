import cv2
import numpy as np
import os
import face_recognition
from datetime import datetime
import pandas as pd

# ---------------- CREATE DAILY ATTENDANCE FILE -----------------
today = datetime.now().strftime("%Y-%m-%d")
attendance_file = f"attendance_{today}.csv"

if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Hour Slot", "Last Marked"])
    df.to_csv(attendance_file, index=False)

# ---------------- LOAD REGISTERED IMAGES -----------------
path = "images"
images = []
classNames = []

myList = os.listdir(path)
for cl in myList:
    img = cv2.imread(f"{path}/{cl}")
    images.append(img)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding Complete")

# ---------------- MARK ATTENDANCE -----------------
def get_hour_slot():
    now = datetime.now()
    hour = now.hour
    return f"{hour}:00 - {hour+1}:00"

def markAttendance(name):
    df = pd.read_csv(attendance_file)

    hour_slot = get_hour_slot()
    now_time = datetime.now().strftime('%H:%M:%S')

    # Check if entry already exists
    mask = (df["Name"] == name) & (df["Hour Slot"] == hour_slot)

    if mask.any():
        df.loc[mask, "Last Marked"] = now_time  # Update time
    else:
        df.loc[len(df)] = [name, hour_slot, now_time]  # Add new entry

    df.to_csv(attendance_file, index=False)


# ---------------- START WEBCAM -----------------
cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesLoc = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesLoc)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesLoc):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            markAttendance(name)

            # Draw box
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, name, (x1,y2+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
