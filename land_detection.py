import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set page configuration
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3b82f6;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e5e7eb;
        border-radius: 5px;
        padding: 0.25rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model loading
@st.cache_resource
def load_trained_model():
    try:
        model = load_model('Modelenv.v1.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure 'Modelenv.v1.h5' is in the same directory as this script.")
        return None

# Define class names
CLASS_NAMES = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# Color mapping for classes
CLASS_COLORS = {
    'Cloudy': '#9ca3af',
    'Desert': '#f59e0b',
    'Green_Area': '#10b981',
    'Water': '#3b82f6'
}

def preprocess_image(uploaded_image):
    """Preprocess the uploaded image for prediction"""
    # Convert to PIL Image if needed
    if isinstance(uploaded_image, np.ndarray):
        img = Image.fromarray(uploaded_image)
    else:
        img = uploaded_image
    
    # Resize to model input size
    img = img.resize((255, 255))
    
    # Convert to array and normalize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

def predict_image(model, img_array):
    """Make prediction on preprocessed image"""
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = prediction[0][predicted_class_idx]
    
    return predicted_class, confidence, prediction[0]

def create_confidence_chart(predictions, class_names):
    """Create a confidence chart using Plotly"""
    fig = px.bar(
        x=class_names,
        y=predictions,
        title="Prediction Confidence for Each Class",
        labels={'x': 'Land Cover Type', 'y': 'Confidence Score'},
        color=predictions,
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        title_x=0.5,
        xaxis_tickangle=-45,
        height=400
    )
    return fig

def main():
    # Main title
    st.markdown('<h1 class="main-header">🛰️ Satellite Image Land Cover Classifier</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 📋 Model Information")
    st.sidebar.markdown("""
    **Model Details:**
    - **Architecture**: Convolutional Neural Network
    - **Input Size**: 255x255 pixels
    - **Classes**: 4 land cover types
    - **Training**: 25 epochs with data augmentation
    """)
    
    st.sidebar.markdown("## 🎯 Supported Classes")
    for class_name, color in CLASS_COLORS.items():
        st.sidebar.markdown(f"<div style='background-color: {color}; padding: 5px; margin: 2px; border-radius: 3px; color: white;'>{class_name}</div>", unsafe_allow_html=True)
    
    # Load model
    model = load_trained_model()
    if model is None:
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Satellite Image</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a satellite image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a satellite image for land cover classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess and predict
            with st.spinner("Analyzing image..."):
                processed_image = preprocess_image(image_pil)
                predicted_class, confidence, all_predictions = predict_image(model, processed_image)
            
            # Display prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Prediction: {predicted_class}")
            st.markdown(f"### 📊 Confidence: {confidence:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            st.markdown('<h2 class="sub-header">📊 Detailed Analysis</h2>', unsafe_allow_html=True)
            
            # Confidence chart
            fig = create_confidence_chart(all_predictions, CLASS_NAMES)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed predictions table
            st.markdown("### 📈 All Class Predictions")
            predictions_df = pd.DataFrame({
                'Land Cover Type': CLASS_NAMES,
                'Confidence Score': all_predictions,
                'Percentage': [f"{pred:.2%}" for pred in all_predictions]
            })
            predictions_df = predictions_df.sort_values('Confidence Score', ascending=False)
            
            # Style the dataframe
            st.dataframe(
                predictions_df,
                use_container_width=True,
                hide_index=True
            )
    
    # Additional features
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        This application uses a Convolutional Neural Network trained on satellite images to classify different types of land cover. 
        
        **The model can identify:**
        - **Cloudy**: Areas with cloud cover
        - **Desert**: Arid and sandy regions
        - **Green Area**: Vegetation and forests
        - **Water**: Bodies of water like lakes, rivers, and oceans
        
        **How to use:**
        1. Upload a satellite image using the file uploader
        2. The model will automatically analyze the image
        3. View the prediction results and confidence scores
        4. Explore the detailed analysis chart for all class probabilities
        """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Model Architecture:**
        - Input Layer: 255x255x3 (RGB images)
        - Convolutional Layers: 3 layers with 32, 64, 128 filters
        - Pooling Layers: MaxPooling2D after each conv layer
        - Dense Layers: 128 units with ReLU activation
        - Output Layer: 4 units with softmax activation
        - Dropout: 0.5 for regularization
        
        **Training Details:**
        - Optimizer: Adam
        - Loss Function: Categorical Crossentropy
        - Data Augmentation: Rotation, flip, zoom, shear
        - Epochs: 25
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with ❤️ using Streamlit and TensorFlow**")

if __name__ == "__main__":
    main()
