import cv2
import os

name = input("Enter name to register: ").strip()

path = "images"
if not os.path.exists(path):
    os.makedirs(path)

cam = cv2.VideoCapture(1)

print("Capturing image... Press SPACE to click photo.")

while True:
    success, img = cam.read()
    cv2.imshow("Register Face", img)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.imwrite(f"images/{name}.jpg", img)
        print("Face registered successfully!")
        break

cam.release()
cv2.destroyAllWindows()
