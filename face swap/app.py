import streamlit as st
from PIL import Image
from face_swap import face_swap

st.title("Face Swap App")

# Upload images
uploaded_file1 = st.file_uploader("Choose the first image", type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("Choose the second image", type=["jpg", "jpeg", "png"])

if uploaded_file1 is not None and uploaded_file2 is not None:
    img1 = Image.open(uploaded_file1)
    img2 = Image.open(uploaded_file2)

    # Convert images to OpenCV format
    img1_cv = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2_cv = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)

    # Swap faces
    swapped1, swapped2 = face_swap(img1_cv, img2_cv)

    # Display results
    st.image(swapped1, caption="Swapped Face 1", channels="BGR")
    st.image(swapped2, caption="Swapped Face 2", channels="BGR")
