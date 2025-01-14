import cv2
import numpy as np
from PIL import Image

def get_face_landmarks(image):
    # Initialize the face detector and create a mask
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    return faces

def blend_faces(img1, img2, face1, face2):
    # Get the regions of interest
    x1, y1, w1, h1 = face1
    x2, y2, w2, h2 = face2
    
    roi1 = img1[y1:y1+h1, x1:x1+w1]
    roi2 = img2[y2:y2+h2, x2:x2+w2]
    
    # Create masks
    mask1 = np.zeros_like(roi1)
    mask2 = np.zeros_like(roi2)
    cv2.fillConvexPoly(mask1, np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]]), (255, 255, 255))
    cv2.fillConvexPoly(mask2, np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]]), (255, 255, 255))

    # Blend the faces
    blended_roi = cv2.seamlessClone(roi1, img2, mask2, (x2 + w2 // 2, y2 + h2 // 2), cv2.NORMAL_CLONE)
    return blended_roi

def swap_faces(img1, img2):
    faces1 = get_face_landmarks(img1)
    faces2 = get_face_landmarks(img2)

    if len(faces1) > 0 and len(faces2) > 0:
        # Swap the first face detected in both images
        blended_face = blend_faces(img1, img2, faces1[0], faces2[0])
        return blended_face
    else:
        print("No faces detected in one of the images.")
        return None

if __name__ == "__main__":
    img1 = cv2.imread("uploads/img1.jpg")
    img2 = cv2.imread("uploads/img2.jpg")
    
    if img1 is None or img2 is None:
        print("Error: One of the images could not be loaded.")
    else:
        swapped_img = swap_faces(img1, img2)
        if swapped_img is not None:
            cv2.imwrite("uploads/swapped_image.jpg", swapped_img)
            print("Swapped image saved as 'uploads/swapped_image.jpg'")
