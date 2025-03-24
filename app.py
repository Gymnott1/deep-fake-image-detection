import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from torchvision.models import resnet50
import plotly.express as px
import tensorflow as tf
import os


# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASS_NAMES = ['Real', 'Fake']
MODEL_FILENAME = "cnn_model.h5"
FACE_WEIGHTS = "vggface2.pt"

def get_confidence_label(confidence):
    """
    Categorize confidence level with descriptive labels
    """
    if confidence < 0.3:
        return "Low Confidence", "text-warning"
    elif confidence < 0.6:
        return "Moderate Confidence", "text-info"
    else:
        return "High Confidence", "text-success"

class DeepFakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = resnet50(weights='IMAGENET1K_V1')
        self.feature_conv = nn.Sequential(*list(self.base_model.children())[:-2])
        self.gradients = None
        self.activations = None
        
        # Add hooks for Grad-CAM
        self.feature_conv[-1][-1].register_forward_hook(self.save_activations)
        self.feature_conv[-1][-1].register_full_backward_hook(self.save_gradients)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.classifier(x)
        return x

    def get_heatmap(self, image_tensor, target_class=None):
        # Forward pass
        prediction = self.forward(image_tensor)
        
        if target_class is None:
            target_class = prediction.argmax(dim=1)
        
        # Backward pass
        self.zero_grad()
        prediction[0, target_class].backward(retain_graph=True)
        
        # Grad-CAM calculation
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.squeeze(0)
        
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
        
        return heatmap, prediction

def load_pytorch_model():
    model = DeepFakeDetector().to(DEVICE)
    try:
        if os.path.exists(FACE_WEIGHTS):
            model.load_state_dict(torch.load(FACE_WEIGHTS, map_location=DEVICE))
            print(f"Loaded PyTorch face weights from {FACE_WEIGHTS}")
        else:
            print(f"PyTorch weights file {FACE_WEIGHTS} not found, using default weights")
    except Exception as e:
        print(f"Could not load face weights: {e}")
    
    model.eval()
    return model

def load_tf_model():
    try:
        if os.path.exists(MODEL_FILENAME):
            # Load model with custom_objects to handle any custom layers if necessary
            model = tf.keras.models.load_model(MODEL_FILENAME)
            
            # Get input shape from model
            input_shape = model.layers[0].input_shape
            if input_shape:
                input_shape = input_shape[0][1:]  # Get shape excluding batch dimension
                print(f"TensorFlow model expects input shape: {input_shape}")
            else:
                print("Could not determine input shape from model")
                
            return model, input_shape
        else:
            print(f"TensorFlow model file {MODEL_FILENAME} not found")
            return None, None
    except Exception as e:
        print(f"Could not load TensorFlow model: {e}")
        return None, None

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def preprocess_for_tf(image, input_shape=None):
    """Preprocess image for TensorFlow model with flexible input shape"""
    if input_shape is None:
        # Default to 224x224 if no shape specified
        input_shape = (224, 224, 3)
    
    # Handle grayscale if needed
    if len(input_shape) == 3 and input_shape[2] == 1:
        img = image.convert('L')
    else:
        img = image.convert('RGB')
    
    # Resize according to expected input
    if len(input_shape) >= 2:
        img = img.resize((input_shape[1], input_shape[0]))
    
    # Convert to numpy and normalize
    img_array = np.array(img)
    
    # Add channel dimension for grayscale
    if len(input_shape) == 3 and input_shape[2] == 1:
        img_array = np.expand_dims(img_array, axis=-1)
    
    # Normalize to 0-1
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def error_level_analysis(image):
    img_array = np.array(image)
    original = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Save and reload image to simulate compression
    _, encoded = cv2.imencode('.jpg', original, [cv2.IMWRITE_JPEG_QUALITY, 90])
    compressed = cv2.imdecode(encoded, 1)
    
    # Calculate difference
    ela = cv2.absdiff(original, compressed)
    ela = cv2.cvtColor(ela, cv2.COLOR_BGR2RGB)
    return Image.fromarray(ela)

def analyze_faces(image):
    try:
        # Initialize face detector
        face_detector = MTCNN(keep_all=True, device=DEVICE)
        
        # Detect faces
        img_array = np.array(image)
        faces = face_detector.detect(img_array)
        
        if faces[0] is None or len(faces[0]) == 0:
            return None, "No faces detected"
        
        # Draw boxes around faces
        boxes = faces[0]
        img_with_boxes = img_array.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 150, 0)  # Darker green for professional look
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_with_boxes, f"Face {i+1}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        message = f"Detected {len(boxes)} face(s)"
        return Image.fromarray(img_with_boxes), message
    except Exception as e:
        return None, f"Error in face detection: {str(e)}"

def display_model_info(tf_model, tf_input_shape):
    st.subheader("Model Information")
    
    # Display PyTorch model info
    st.write("**PyTorch Model:**")
    st.write(f"- Weights file: {FACE_WEIGHTS}")
    st.write("- Architecture: ResNet50 with custom classifier")
    st.write("- Input size: 224x224 RGB")
    
    # Display TensorFlow model info if available
    if tf_model is not None:
        st.write("**TensorFlow Model:**")
        st.write(f"- Model file: {MODEL_FILENAME}")
        if tf_input_shape:
            st.write(f"- Expected input shape: {tf_input_shape}")
        
        # Display model summary as string
        summary_list = []
        tf_model.summary(print_fn=lambda x: summary_list.append(x))
        st.text("\n".join(summary_list))
    else:
        st.write("**TensorFlow Model:** Not loaded")
    
    st.write(f"**Running on:** {DEVICE}")

def main():
    
    # Inject custom CSS for a refined professional look
    st.set_page_config(
        page_title="DeepFake Detector | AI Forensics", 
        page_icon="🕵️",
        layout="wide",
        initial_sidebar_state="auto"
    )

    # Professional and modern color scheme and styling
    st.markdown(
        """
        <style>
        /* Modern, clean typography */
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            color: #2c3e50;
        }

        /* Elegant header styling */
        .stTitle {
            color: #1a5f7a;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        /* Refined button design */
        .stButton>button {
            background-color: #1a5f7a;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            background-color: #137399;
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        }

        /* Professional tab styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #f4f4f4;
            border-radius: 8px;
            padding: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            color: #2c3e50;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #1a5f7a;
        }

        /* Clean, professional verdict styling */
        .verdict-success {
            background-color: rgba(40, 167, 69, 0.1);
            border-left: 4px solid #28a745;
            color: #28a745;
            padding: 12px;
            border-radius: 4px;
            font-weight: 600;
        }
        .verdict-error {
            background-color: rgba(220, 53, 69, 0.1);
            border-left: 4px solid #dc3545;
            color: #dc3545;
            padding: 12px;
            border-radius: 4px;
            font-weight: 600;
        }

        /* Image and chart containers */
        .stImage, .stPlotlyChart {
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            padding: 10px;
            background-color: white;
        }

        /* Hide Streamlit default elements */
        footer {visibility: hidden;}
        .viewerBadge_container__1QSob {visibility: hidden;}
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Professional title with sophisticated layout
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #1a5f7a; font-weight: 700; font-size: 2.5rem;">🕵️ AI Image Forensics</h1>
        <p style="color: #6c757d; font-size: 1.1rem;">Advanced DeepFake Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("🔍 Deepfake Detection System")
    
    with st.spinner("Loading models..."):
        torch_model = load_pytorch_model()
        tf_model, tf_input_shape = load_tf_model()
    
    uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Basic metadata
            st.subheader("📄 Image Info")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Size:** {image.width} x {image.height}")
        
        with col2:
            tab1, tab2, tab3, tab4 = st.tabs(["AI Detection", "Face Analysis", "Technical Analysis", "Model Info"])
            
            with tab1:
                # PyTorch model prediction
                image_tensor = preprocess_image(image)
                heatmap, preds = torch_model.get_heatmap(image_tensor)
                probs = torch.softmax(preds, dim=1)[0].cpu().detach().numpy()
                
                # TensorFlow model prediction if available
                tf_prediction = None
                if tf_model is not None:
                    try:
                        tf_input = preprocess_for_tf(image, tf_input_shape)
                        tf_prediction = tf_model.predict(tf_input, verbose=0)
                        if isinstance(tf_prediction, list):
                            tf_prediction = tf_prediction[0]
                    except Exception as e:
                        st.error(f"Error with TensorFlow prediction: {e}")
                
                # Resize heatmap for overlay
                heatmap = cv2.resize(heatmap, (image.width, image.height))
                heatmap = (heatmap * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # Overlay heatmap on image
                overlay = cv2.addWeighted(np.array(image), 0.7, heatmap, 0.3, 0)
                st.image(overlay, caption="AI Analysis Heatmap", use_container_width=True)
                
                # Display PyTorch results
                fig = px.bar(x=CLASS_NAMES, y=probs, 
                             labels={'x': 'Class', 'y': 'Confidence'},
                             title="Detection Results (PyTorch Model)",
                             color=CLASS_NAMES,
                             color_discrete_map={'Real': '#28a745', 'Fake': '#dc3545'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Add verdict
                prediction_class = CLASS_NAMES[np.argmax(probs)]
                confidence = np.max(probs) * 100
                
                if prediction_class == "Real":
                    st.markdown(f"<div class='verdict-success'>Primary Verdict: Image appears to be REAL ({confidence:.1f}% confidence)</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='verdict-error'>Primary Verdict: Image appears to be FAKE ({confidence:.1f}% confidence)</div>", unsafe_allow_html=True)
                
                # Display TensorFlow model results if available
                if tf_prediction is not None:
                    st.write("**Secondary Model Analysis:**")
                    
                    if len(tf_prediction.shape) > 0 and tf_prediction.shape[0] > 1:
                        tf_confidence = tf_prediction[np.argmax(tf_prediction)] * 100
                        tf_class = "Real" if np.argmax(tf_prediction) == 0 else "Fake"
                    else:
                        tf_confidence = tf_prediction[0] * 100 if tf_prediction[0] <= 1 else tf_prediction[0]
                        tf_class = "Real" if tf_prediction[0] < 0.5 else "Fake"
                    
                    if tf_class == "Real":
                        st.markdown(f"<div class='verdict-success'>CNN Model: Image appears to be REAL ({tf_confidence:.1f}% confidence)</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='verdict-error'>CNN Model: Image appears to be FAKE ({tf_confidence:.1f}% confidence)</div>", unsafe_allow_html=True)
            
            with tab2:
                # Face analysis
                face_result, face_message = analyze_faces(image)
                
                if face_result is not None:
                    st.image(face_result, caption="Face Detection", use_container_width=True)
                    st.write(f"**Analysis:** {face_message}")
                    
                    if "Detected" in face_message and "face" in face_message:
                        st.info("Multiple faces in an image are normal and not indicators of manipulation.")
                else:
                    st.warning(face_message)
            
            with tab3:
                # ELA analysis
                ela_image = error_level_analysis(image)
                st.image(ela_image, caption="Error Level Analysis (ELA)", use_container_width=True)
                
                st.markdown("""
                **What is ELA?**
                
                Error Level Analysis shows differences in compression levels. Areas with different compression levels may indicate editing:
                
                - **Brighter areas:** Potential edits or inconsistencies
                - **Uniform patterns:** Usually indicate authentic content
                - **Sharp edges with bright highlights:** May suggest manipulation
                
                Note that legitimate editing like cropping or color correction can also show up in ELA.
                """)
            
            with tab4:
                # Display model information
                display_model_info(tf_model, tf_input_shape)

if __name__ == "__main__":
    main()
