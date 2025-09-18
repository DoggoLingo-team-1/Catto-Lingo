import streamlit as st
import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import torch
from ultralytics import YOLO
import tempfile
import os
from av import VideoFrame # Required for streamlit-webrtc

# --- Page Configuration ---
# Set the title and a favicon for the browser tab. This should be the first Streamlit command.
st.set_page_config(page_title="Cat Emotion Detector", page_icon="🐈")

# --- Model Loading (with Caching) ---
# Use st.cache_resource to load models only once, which significantly speeds up the app
# after the first run.
@st.cache_resource
def load_models():
    """Loads and returns the object detection and emotion classification models."""
    try:
        # 1. Load the general-purpose YOLOv8 object detection model (pretrained on COCO)
        object_detector = YOLO('yolov8n.pt')
        
        # 2. Download the custom-trained emotion classifier from Hugging Face
        model_path = hf_hub_download(
            repo_id="abdalrhman2080/cat-emotion-detector-model", 
            filename="best.pt"
        )
        
        # 3. Load the emotion classifier from the downloaded file
        emotion_classifier = YOLO(model_path)
        return object_detector, emotion_classifier
    except Exception as e:
        st.error(f"An error occurred while loading the models: {e}")
        st.error("Please make sure you have the correct repository ID and file name on Hugging Face.")
        return None, None
# --- Main Application Logic ---

# Define the list of class names in the exact order your model was trained on.
CLASS_NAMES = [
    'Angry', 'Disgusted', 'Happy', 'Normal', 'Scared', 'Surprised',
    'attentive', 'no clear emotion recognizable', 'relaxed', 'sad', 'uncomfortable'
]

# Load the models using the cached function
object_detector, emotion_classifier = load_models()

# --- Core Processing Function ---
def process_frame(frame: 'np.ndarray', detector: YOLO, classifier: YOLO) -> 'np.ndarray':
    """
    Detects cats in a single frame, classifies their emotion, and draws annotations.
    
    Args:
        frame: The input video frame as a NumPy array.
        detector: The pre-loaded YOLO object detection model.
        classifier: The pre-loaded YOLO emotion classification model.

    Returns:
        The annotated frame as a NumPy array.
    """
    # 1. DETECT CATS IN THE FRAME
    # The 'cat' class in the COCO dataset has index 15. We filter for it.
    detections = detector(frame, classes=[15], conf=0.45, verbose=False)

    # Loop over each detected cat in the results
    for result in detections:
        for box in result.boxes:
            # Get bounding box coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 2. CROP THE DETECTED CAT
            # Add a small padding to ensure the entire cat is included
            padding = 10
            cat_crop = frame[max(0, y1-padding):min(frame.shape[0], y2+padding),
                             max(0, x1-padding):min(frame.shape[1], x2+padding)]

            # If the crop is empty (e.g., bounding box is out of frame), skip it
            if cat_crop.size == 0:
                continue

            # 3. CLASSIFY THE EMOTION OF THE CROPPED CAT
            emotion_results = classifier(cat_crop, verbose=False)
            
            if emotion_results:
                # Get the index and confidence of the top prediction
                top_prediction_index = emotion_results[0].probs.top1
                confidence = emotion_results[0].probs.top1conf
                emotion_label = CLASS_NAMES[top_prediction_index]
                label_text = f"{emotion_label}: {confidence:.2f}"
            else:
                label_text = "Unknown"

            # 4. DRAW THE BOUNDING BOX AND LABEL ON THE ORIGINAL FRAME
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 15), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame

# --- WebRTC Class for Live Camera Feed ---
# This class is a requirement for streamlit-webrtc to handle video frames
class VideoProcessor:
    def recv(self, frame: VideoFrame) -> VideoFrame:
        # Convert the VideoFrame to an OpenCV-compatible NumPy array
        img = frame.to_ndarray(format="bgr24")

        # Process the frame using our main function
        processed_img = process_frame(img, object_detector, emotion_classifier)

        # Convert the processed NumPy array back to a VideoFrame to be displayed in the browser
        return VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- Streamlit User Interface ---
st.title("🐈 Cat Emotion Detector")

st.markdown(
    "This application uses AI to detect cats in videos and classify their emotions in real-time. "
    "Choose an option below to get started."
)

# Create tabs for the two user choices
tab1, tab2 = st.tabs(["🎥 **Upload a Video**", "📸 **Live Camera Feed**"])

# --- Tab 1: Video Upload Logic ---
with tab1:
    st.header("Analyze a Pre-recorded Video")
    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Use a temporary file to store the uploaded video content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        
        if st.button("Start Video Analysis"):
            if object_detector is None or emotion_classifier is None:
                st.warning("Models are not loaded. Please check the error messages above.")
            else:
                st.info("Video analysis started. This may take a moment...")
                stframe = st.empty() # Placeholder for the video frames
                progress_bar = st.progress(0)
                
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                for frame_num in range(total_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    processed_frame = process_frame(frame, object_detector, emotion_classifier)
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    stframe.image(frame_rgb, channels="RGB")
                    progress_bar.progress((frame_num + 1) / total_frames)

                st.success("Video analysis complete!")
                cap.release()
                os.remove(video_path) # Clean up the temporary file

# --- Tab 2: Live Camera Logic ---
with tab2:
    st.header("Analyze from Live Camera")
    st.info("Click 'START' below to access your camera. You will need to grant permission in your browser.")

    if object_detector is None or emotion_classifier is None:
        st.warning("Models are not loaded. Cannot start live feed.")
    else:
        # Import the webrtc_streamer from the library
        from streamlit_webrtc import webrtc_streamer

        # Start the WebRTC streamer which handles the camera feed
        webrtc_streamer(
            key="live-camera-analysis",
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

