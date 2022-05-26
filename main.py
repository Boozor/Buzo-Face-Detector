import cv2
from random import randrange
from PIL import Image
import streamlit as st
import numpy as np

# load some pre_trained data on face frontal from opencv (haar cascade algorithm)
#train_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

except Exception:
    st.write("Error loading cascade classifiers")


def detect(image):
    '''
    Function to detect faces/eyes and smiles in the image passed to this function
    '''

    image = np.array(image.convert('RGB'))

    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # The following are the parameters of cv2.rectangle()
        # cv2.rectangle(image_to_draw_on, start_point, end_point, color, line_width)
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)

        roi = image[y:y + h, x:x + w]
    return image, faces

def about():
	st.write(
		'''
		This is a Face Detector App developed by **Chibuzo Valentine Nwadike**.
		I leveraged the Haar Cascade Algorithm. The Haar Cascade Algorithm is an object detection algorithm.
		It can be used to detect objects in images or videos. 
		The algorithm has four stages:
			1. Haar Feature Selection 
			2. Creating  Integral Imagess
			3. Adaboost Training
			4. Cascading Classifiers ''')


def main():
    st.title("Buzo Face Detector :sunglasses: ")
    st.write("**Developer: Chibuzo Valentine Nwadike**")

    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Home":

        st.write("Go to the About section from the sidebar to learn more about it.")

        # You can specify more file types below if you want
        image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

        if image_file is not None:

            image = Image.open(image_file)

            if st.button("Process"):
                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
                result_img, result_faces = detect(image=image)
                st.image(result_img, use_column_width=True)
                st.success("Found {} faces\n".format(len(result_faces)))

    elif choice == "About":
        about()


if __name__ == "__main__":
    main()