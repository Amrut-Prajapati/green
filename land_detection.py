import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Set page config
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
        text-align: center;
        color: #2e7d32;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .sidebar-header {
        color: #1976d2;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Class names and their descriptions
CLASS_DESCRIPTIONS = {
    'Cloudy': 'Cloud-covered areas in satellite imagery',
    'Desert': 'Arid desert regions with minimal vegetation',
    'Green_Area': 'Vegetation-rich areas including forests and grasslands',
    'Water': 'Water bodies like lakes, rivers, and oceans'
}

# Color mapping for visualization
COLOR_MAP = {
    'Cloudy': '#f1c40f',
    'Desert': '#e67e22',
    'Green_Area': '#27ae60',
    'Water': '#3498db'
}

@st.cache_resource
def load_trained_model():
    """Load the pre-trained model with caching for better performance."""
    try:
        model = load_model('Modelenv.v1.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(255, 255)):
    """Preprocess image for model prediction."""
    # Resize image
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

def predict_image(model, image):
    """Make prediction on preprocessed image."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    # Get probabilities for all classes
    probabilities = {}
    for i, class_name in enumerate(class_names):
        probabilities[class_name] = prediction[0][i]
    
    return predicted_class, confidence, probabilities

def create_prediction_chart(probabilities):
    """Create a visualization of prediction probabilities."""
    fig = go.Figure(data=[
        go.Bar(
            x=list(probabilities.keys()),
            y=list(probabilities.values()),
            marker_color=[COLOR_MAP[class_name] for class_name in probabilities.keys()],
            text=[f'{prob:.2%}' for prob in probabilities.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Land Cover Type',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def main():
    # Title and description
    st.markdown('<h1 class="main-header">🛰️ Satellite Image Land Cover Classifier</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application uses a deep learning model to classify satellite images into different land cover types.
    Upload an image to get predictions for: **Cloudy**, **Desert**, **Green Area**, or **Water**.
    """)
    
    # Load model
    model = load_trained_model()
    
    if model is None:
        st.error("Model could not be loaded. Please ensure 'Modelenv.v1.h5' is in the same directory.")
        return
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">🔧 Configuration</div>', unsafe_allow_html=True)
    
    # About model
    with st.sidebar.expander("📊 About the Model"):
        st.write("""
        **Model Architecture:** CNN with 3 convolutional layers
        
        **Input Size:** 255x255 pixels
        
        **Classes:** 4 land cover types
        
        **Training:** 25 epochs with data augmentation
        """)
    
    # Class descriptions
    with st.sidebar.expander("🌍 Land Cover Types"):
        for class_name, description in CLASS_DESCRIPTIONS.items():
            st.write(f"**{class_name}:** {description}")
    
    # File uploader
    st.sidebar.markdown('<div class="sidebar-header">📤 Upload Image</div>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(
        "Choose a satellite image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a satellite image for classification"
    )
    
    # Display options
    st.sidebar.markdown('<div class="sidebar-header">🎨 Display Options</div>', unsafe_allow_html=True)
    show_confidence = st.sidebar.checkbox("Show confidence scores", value=True)
    show_probabilities = st.sidebar.checkbox("Show probability chart", value=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, caption="Uploaded satellite image", use_column_width=True)
            
            # Image information
            st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Mode:** {image.mode}")
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            predicted_class, confidence, probabilities = predict_image(model, image)
        
        with col2:
            st.subheader("🔮 Prediction Results")
            
            # Display prediction with styled box
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Predicted Class: {predicted_class}</h3>
                <p><strong>Description:</strong> {CLASS_DESCRIPTIONS[predicted_class]}</p>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence metrics
            if show_confidence:
                # Create confidence meter
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = confidence * 100,
                    title = {'text': "Confidence Level (%)"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': COLOR_MAP[predicted_class]},
                        'steps': [
                            {'range': [0, 50], 'color': '#ffcccb'},
                            {'range': [50, 80], 'color': '#ffffcc'},
                            {'range': [80, 100], 'color': '#90ee90'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Probability chart
        if show_probabilities:
            st.subheader("📊 Probability Distribution")
            fig_probs = create_prediction_chart(probabilities)
            st.plotly_chart(fig_probs, use_container_width=True)
            
            # Detailed probabilities
            st.subheader("📈 Detailed Probabilities")
            prob_df = pd.DataFrame([
                {'Land Cover Type': class_name, 'Probability': prob, 'Percentage': f'{prob:.2%}'}
                for class_name, prob in probabilities.items()
            ])
            prob_df = prob_df.sort_values('Probability', ascending=False)
            st.dataframe(prob_df, use_container_width=True)
    
    else:
        # Welcome message and sample images
        st.markdown("""
        ### 🚀 Get Started
        
        1. Upload a satellite image using the sidebar
        2. The model will classify it into one of four land cover types
        3. View the prediction results and confidence scores
        
        **Supported formats:** JPG, JPEG, PNG
        
        **Recommended size:** Images will be resized to 255x255 pixels for processing
        """)
        
        # Example predictions showcase
        st.subheader("🎯 What the Model Can Identify")
        
        example_cols = st.columns(4)
        for i, (class_name, description) in enumerate(CLASS_DESCRIPTIONS.items()):
            with example_cols[i]:
                # Create a simple colored box as placeholder
                placeholder_color = COLOR_MAP[class_name]
                st.markdown(f"""
                <div style="background: {placeholder_color}; height: 100px; border-radius: 10px; 
                           display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                    {class_name}
                </div>
                """, unsafe_allow_html=True)
                st.write(f"**{class_name}**")
                st.write(description)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
    🌍 Environmental Monitoring with AI | Built with Streamlit and TensorFlow
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()