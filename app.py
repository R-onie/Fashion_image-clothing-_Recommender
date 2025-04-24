import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Ensure uploads directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Load the feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load the ResNet50 model
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_model.trainable = False

model = tf.keras.Sequential([
    resnet_model,
    GlobalMaxPooling2D()
])

# Streamlit App Title
st.title('👗 Fashion Recommender System')
st.markdown("Upload an image and get fashion recommendations based on visual similarity.")

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        print(f"File saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

# Function to extract features from the image
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function to recommend similar images
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File uploader
uploaded_file = st.file_uploader("📁 Choose a fashion image to upload", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Save the uploaded file and get its path
    file_path = save_uploaded_file(uploaded_file)
    
    if file_path:
        # Display the uploaded image
        display_image = Image.open(file_path)
        st.image(display_image, caption="Uploaded Image", use_container_width=True)

        # Extract features and recommend
        features = feature_extraction(file_path, model)
        indices = recommend(features, feature_list)

        # Display recommended images
        st.subheader("🛍️ Recommended Items:")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]], caption="Recommended Image 1", use_container_width=True)
        with col2:
            st.image(filenames[indices[0][1]], caption="Recommended Image 2", use_container_width=True)
        with col3:
            st.image(filenames[indices[0][2]], caption="Recommended Image 3", use_container_width=True)
        with col4:
            st.image(filenames[indices[0][3]], caption="Recommended Image 4", use_container_width=True)
        with col5:
            st.image(filenames[indices[0][4]], caption="Recommended Image 5", use_container_width=True)

    else:
        st.error("⚠️ Error saving file. Please try again.")
