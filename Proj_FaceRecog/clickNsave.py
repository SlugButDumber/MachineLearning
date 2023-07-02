import cv2
import os

def ClicknSave():

    # Create a folder to store the cropped faces
    output_folder = 'C:\Drive D\ArhatPersonal\ML\Practice\Proj_FaceRecog\dataset\cropped_faces'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize the camera
    camera = cv2.VideoCapture(0)

    # Set up the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Capture a photo
    ret, frame = camera.read()

    # Detect faces in the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # Crop and save each face
    saved_paths = []
    for i, (x, y, w, h) in enumerate(faces):
        face = frame[y-75:y+h+20, x-25:x+w+25]
        face_file = os.path.join(output_folder, f'face_{i}.jpg')
        cv2.imwrite(face_file, face)
        saved_paths.append(os.path.abspath(face_file))

    # Release the camera
    camera.release()

    print(f"{len(faces)} face(s) cropped and saved in '{output_folder}' folder.")

    if len(saved_paths) == 0:
        return '-1'

    elif len(saved_paths) == 1:
          
        return saved_paths[-1]
    
    else:
        return '0'