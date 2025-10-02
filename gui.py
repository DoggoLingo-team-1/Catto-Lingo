import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from huggingface_hub import hf_hub_download
import tempfile
import os
from av import VideoFrame
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from collections import Counter
import time 

# ================== 1. Configuration and Constants ==================
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hugging Face Model Configuration ---
HF_REPO_ID = "abdalrhman2080/ResNet-50"
HF_FILENAME = "model.pth"

# Define the 9 class names used in your training code
CLASS_NAMES = [
    'angry', 'disgusted', 'happy', 'normal', 'relaxed', 'sad', 
    'scared', 'surprised', 'uncomfortable'
]
# --- ---

# --- Page Configuration ---
st.set_page_config(page_title="Cat Emotions Analyzer (PyTorch)", page_icon="🐈", layout="wide")

# ================== 2. Preprocessing Function ==================

def get_model_transform():
    """Defines the PyTorch preprocessing transform used during training."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # Normalization values MUST match training
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# --- Model Loading (with Caching) ---
@st.cache_resource
def load_emotion_classifier():
    """Downloads model weights and loads the entire PyTorch model object."""
    try:
        # 1. Download the model file from Hugging Face
        model_local_path = hf_hub_download(
            repo_id=HF_REPO_ID, 
            filename=HF_FILENAME,
        )
        st.info(f"✅ Model weights downloaded to: {model_local_path}")
        
        # 2. LOAD THE ENTIRE MODEL OBJECT DIRECTLY AND SAFELY
        # Set weights_only=False to allow loading of the full model object (trusted source).
        classifier = torch.load(
            model_local_path, 
            map_location=DEVICE,
            weights_only=False  # <--- THIS IS THE CRITICAL FIX
        )
        classifier.eval() # Set the model to evaluation mode
        
        st.success("✅ PyTorch ResNet-50 classifier loaded and ready!")
        return classifier
        
    except Exception as e:
        st.error(f"❌ An error occurred while loading the model: {e}")
        st.error("Please verify the repository ID and file path.")
        return None

# Load the classifier and the transform
emotion_classifier = load_emotion_classifier()
emotion_transform = get_model_transform()

# ================== 3. Analysis and Reporting Function ==================

def generate_report(predictions_list, total_frames, classes):
    """Generates the final statistical report based on frame predictions."""
    if not predictions_list:
        st.warning("No frames were analyzed.")
        return
        
    emotion_counts = Counter(predictions_list)
    
    # Determine the dominant emotion
    most_frequent_emotion, max_count = emotion_counts.most_common(1)[0]
    
    st.header("📊 Final Statistical Report")
    st.markdown(f"**Total Frames Analyzed:** {total_frames}")
    st.markdown(f"**Dominant Emotion:** <h2 style='color:#FF4B4B;'>{most_frequent_emotion.upper()} ({max_count / total_frames * 100:.2f}%)</h2>", unsafe_allow_html=True)

    st.subheader("Emotion Distribution")
    
    # Prepare data for table display
    report_data = []
    for emotion in classes:
        count = emotion_counts.get(emotion, 0)
        percentage = (count / total_frames) * 100
        report_data.append({
            "Emotion": emotion.capitalize(),
            "Count": count,
            "Percentage": f"{percentage:.2f}%"
        })
        
    st.table(report_data)


def process_video_analysis(cap: cv2.VideoCapture, classifier: torch.nn.Module, transform: transforms.Compose):
    """Processes video frames, classifies them, and returns a list of predictions."""
    predictions_list = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    st.info(f"Analysis started for {total_frames} frames. Please wait...")
    
    progress_bar = st.progress(0)
    
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Preprocess Frame (Assumes the subject is the main focus/full frame)
        try:
            # OpenCV reads as BGR, convert to RGB for standard PyTorch transform
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            input_tensor = transform(frame_rgb).unsqueeze(0).to(DEVICE)
        except Exception:
            continue

        # 2. Classify Emotion
        with torch.no_grad():
            outputs = classifier(input_tensor)
            predicted_index = torch.argmax(outputs).item()
            predictions_list.append(CLASS_NAMES[predicted_index])
            
        # Update progress bar
        if frame_num % 30 == 0 or frame_num == total_frames - 1:
            progress_bar.progress((frame_num + 1) / total_frames)
            
    cap.release()
    st.success("✅ All frames analyzed successfully!")
    return predictions_list, total_frames

# ================== 4. Live Recording Processor ==================

class FrameBufferProcessor(VideoProcessorBase):
    """A processor that simply collects BGR frames into a buffer for later analysis."""
    def __init__(self):
        self.frame_buffer = []
        
    def recv(self, frame: VideoFrame) -> VideoFrame:
        # Collects the frame in BGR format
        img = frame.to_ndarray(format="bgr24")
        self.frame_buffer.append(img)
        
        # Draw a simple 'recording' indicator
        cv2.circle(img, (20, 20), 10, (0, 0, 255), -1) # Red circle
        
        # Return the frame for user visualization (with indicator)
        return VideoFrame.from_ndarray(img, format="bgr24")

# ================== 5. Streamlit User Interface ==================

st.title("🐈 Cat Emotions")

st.markdown(
    "This application uses a PyTorch ResNet-50 model to analyze and classify emotions "
    "by processing video frames and providing a statistical summary. This model is ideal for subjects that fill the camera frame."
)

if emotion_classifier is None:
    # No need to display the warning here as the error is shown above the title
    st.stop() 

# Create tabs for user options
tab1, tab2 = st.tabs(["🎥 **Analyze Uploaded Video**", "🎤 **Record & Analyze Live Feed**"])

# --- Tab 1: Video Upload Logic ---
with tab1:
    st.header("Analyze Pre-recorded Video")
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        
        if st.button("Start Video Analysis"):
            cap = cv2.VideoCapture(video_path)
            
            predictions, total_frames = process_video_analysis(
                cap, 
                emotion_classifier, 
                emotion_transform
            )
            
            # Display the final report
            generate_report(predictions, total_frames, CLASS_NAMES)
            os.remove(video_path) # Clean up the temporary file


# --- Tab 2: Live Camera Recording and Analysis ---
with tab2:
    st.header("Record a Short Clip for Analysis")
    
    REC_DURATION = 5 # Recording duration in seconds
    
    st.markdown(f"""
    1. Click **"START"** below to access your camera.
    2. The application will record frames for **{REC_DURATION} seconds** after you click the 'Analyze' button.
    3. Keep the subject steady and well-lit within the camera frame.
    """)
    
    # Use webrtc_streamer to start the camera and video processor
    ctx = webrtc_streamer(
        key="live-recording-analysis",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FrameBufferProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if st.button(f"Analyze the Last Recorded Clip ({REC_DURATION}s)"):
        if ctx.video_processor and ctx.video_processor.frame_buffer:
            
            st.info("Starting analysis of recorded frames...")
            
            recorded_frames = ctx.video_processor.frame_buffer
            
            all_predictions = []
            total_frames = len(recorded_frames)
            
            progress_bar = st.progress(0)

            # --- Direct Frame Analysis ---
            for i, frame in enumerate(recorded_frames):
                # 1. Preprocess Frame (Frame is BGR from FrameBufferProcessor)
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_tensor = emotion_transform(frame_rgb).unsqueeze(0).to(DEVICE)
                except Exception:
                    continue

                # 2. Classify Emotion
                with torch.no_grad():
                    outputs = emotion_classifier(input_tensor)
                    predicted_index = torch.argmax(outputs).item()
                    all_predictions.append(CLASS_NAMES[predicted_index])
                    
                # Update progress bar
                progress_bar.progress((i + 1) / total_frames)

            # Display the final report for the recorded clip
            generate_report(all_predictions, total_frames, CLASS_NAMES)
            
            # Reset buffer to prepare for next recording
            ctx.video_processor.frame_buffer = []
            
        else:

            st.warning("Please wait for the camera to start (click START) or record a segment before analyzing.")
