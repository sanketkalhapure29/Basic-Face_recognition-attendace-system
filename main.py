import cv2
import numpy as np
import face_recognition
import os

path = "images"
images_rgb = []        # To store RGB images
image_names = []       # To store image names

for file_name in os.listdir(path):
    file_path = os.path.join(path, file_name)

    # Load image
    img = face_recognition.load_image_file(file_path)
    img  = cv2.resize(img, (0,0), None, 0.5, 0.5)

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Store
    images_rgb.append(img_rgb)
    image_names.append(os.path.splitext(file_name)[0])

    faceLoc = face_recognition.face_locations(img_rgb)[0]
    encodeFace = face_recognition.face_encodings(img_rgb)[0]
   
    cv2.rectangle(img_rgb, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
    cv2.putText(img_rgb, os.path.splitext(file_name)[0], (faceLoc[3], faceLoc[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)  
    cv2.imshow(file_name, img_rgb)
  
    

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Total images loaded:", len(images_rgb))
