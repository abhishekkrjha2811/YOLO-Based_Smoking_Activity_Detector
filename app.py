import streamlit as st
from ultralytics import YOLO
import cv2
import time
import tempfile
import os
from PIL import Image
import numpy as np

st.set_page_config(page_title="Smoking Activity Detection", layout="wide")

st.title("🔥 Real-Time Smoking Activity Detection")
st.markdown("Using a YOLO model to detect smoking activity through webcam, uploaded images, or videos.")

# Initialize model with error handling
@st.cache_resource
def load_model():
    try:
        return YOLO('best.pt')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure 'best.pt' model file is in the repository.")
        st.stop()

if 'model' not in st.session_state:
    st.session_state.model = load_model()

# Sidebar with confidence settings
st.sidebar.title("⚙️ Detection Settings")

# Confidence score slider
confidence_score = st.sidebar.slider(
    "🎯 Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="Lower values detect more objects but may include false positives. Higher values are more selective."
)

st.sidebar.markdown(f"**Current Confidence:** {confidence_score:.2f} ({confidence_score*100:.0f}%)")

# Image size option
image_size = st.sidebar.selectbox(
    "🖼️ Detection Image Size",
    options=[320, 416, 512, 640, 832],
    index=3,  # Default to 640
    help="Larger sizes may be more accurate but slower to process"
)

# Create tabs for different input modes
tab1, tab2, tab3 = st.tabs(["📹 Webcam", "🖼️ Image Upload", "🎥 Video Upload"])

# Tab 1: Webcam Detection
with tab1:
    st.subheader("Real-time Webcam Detection")
    
    # Display current settings
    st.info(f"⚙️ Current Settings: Confidence = {confidence_score:.2f}, Image Size = {image_size}px")
    st.warning("⚠️ Webcam functionality works only when running locally. For cloud deployment, use Image or Video upload tabs.")
    
    start_button = st.button("Start Detection", key="webcam_start")
    
    if start_button:
        stframe = st.empty()
        
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Cannot access webcam. Please check permissions or try running locally.")
                st.stop()
            
            stop_button = st.button("Stop Detection", key="stop_button") 
            smoking_detected = False  
            
            # Create a progress bar
            progress = st.progress(0)
            smoking_message = st.empty() 
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break
                
                results = st.session_state.model.predict(source=frame, imgsz=image_size, conf=confidence_score, verbose=False)
                annotated_frame = results[0].plot()
                
                smoking_detected = False
                if len(results[0].boxes) > 0:
                    for result in results[0].boxes.cls:
                        if result == 0:  # Assuming class 0 is smoking
                            smoking_detected = True
                            break
                
                if smoking_detected:
                    progress.progress(100)  
                    smoking_message.success("🔥 Smoking Detected!")
                else:
                    progress.progress(0) 
                    smoking_message.info("No Smoking Detected")
                
                # Fix color issue: Convert BGR to RGB
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                
                if stop_button:
                    break
                
                time.sleep(0.03)
            
            cap.release()
            st.success("Detection stopped.")
            
        except Exception as e:
            st.error(f"Webcam error: {str(e)}")

# Tab 2: Image Upload
with tab2:
    st.subheader("Upload Image for Detection")
    
    # Display current settings
    st.info(f"⚙️ Current Settings: Confidence = {confidence_score:.2f}, Image Size = {image_size}px")
    
    uploaded_image = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp'], 
        key="image_upload"
    )
    
    if uploaded_image is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Processing image..."):
            # Process image
            image_array = np.array(image)
            
            # Run detection
            results = st.session_state.model.predict(source=image_array, imgsz=image_size, conf=confidence_score, verbose=False)
            annotated_image = results[0].plot()
            
            with col2:
                st.subheader("Detection Results")
                # Fix color issue: Convert BGR to RGB
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image_rgb, caption="Detection Results", use_column_width=True, channels="RGB")
        
        # Show detection summary with confidence scores
        smoking_detected = False
        detection_count = 0
        confidence_scores = []
        
        if len(results[0].boxes) > 0:
            for i, result in enumerate(results[0].boxes.cls):
                if result == 0:  # Assuming class 0 is smoking
                    smoking_detected = True
                    detection_count += 1
                    # Get confidence score for this detection
                    conf_score = results[0].boxes.conf[i].item()
                    confidence_scores.append(conf_score)
        
        if smoking_detected:
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            st.success(f"🔥 Smoking Activity Detected! ({detection_count} instance(s))")
            st.info(f"📊 Average Confidence: {avg_confidence:.2f} ({avg_confidence*100:.0f}%)")
            
            # Show individual detection confidences
            with st.expander("🔍 Detailed Detection Results"):
                for i, conf in enumerate(confidence_scores, 1):
                    st.write(f"Detection {i}: {conf:.3f} ({conf*100:.1f}%)")
        else:
            st.info("✅ No smoking activity detected in the image.")
            st.write(f"ℹ️ Try lowering the confidence threshold (currently {confidence_score:.2f}) to detect more objects.")

# Tab 3: Video Upload
with tab3:
    st.subheader("Upload Video for Detection")
    
    # Display current settings
    st.info(f"⚙️ Current Settings: Confidence = {confidence_score:.2f}, Image Size = {image_size}px")
    
    uploaded_video = st.file_uploader(
        "Choose a video file...", 
        type=['mp4', 'avi', 'mov', 'mkv'], 
        key="video_upload"
    )
    
    if uploaded_video is not None:
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.read())
            temp_video_path = tmp_file.name
        
        # Display original video
        st.subheader("Original Video")
        st.video(uploaded_video)
        
        # Process video button
        if st.button("Process Video", key="process_video"):
            
            try:
                with st.spinner("Processing video... This may take a while."):
                    # Create output video path
                    output_path = tempfile.mktemp(suffix='_detected.mp4')
                    
                    # Open video capture
                    cap = cv2.VideoCapture(temp_video_path)
                    
                    if not cap.isOpened():
                        st.error("Failed to open video file. Please try a different format.")
                        os.unlink(temp_video_path)
                        st.stop()
                    
                    # Get video properties
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Validate video properties
                    if fps <= 0 or width <= 0 or height <= 0:
                        st.error("Invalid video properties. Please try a different video.")
                        cap.release()
                        os.unlink(temp_video_path)
                        st.stop()
                    
                    # FIXED: Use most web-compatible codec first - AVC1/H264 are best for browsers
                    web_compatible_codecs = [
                        ('avc1', 'H264-AVC1'),     # Best for Safari/iOS
                        ('H264', 'H264'),          # Good for most browsers
                        ('mp4v', 'MPEG-4'),        # Universal fallback
                        ('MJPG', 'Motion-JPEG')    # Last resort, always works
                    ]
                    
                    out = None
                    successful_codec = None
                    
                    for codec_str, codec_name in web_compatible_codecs:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*codec_str)
                            test_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                            
                            if test_out.isOpened():
                                out = test_out
                                successful_codec = codec_name
                                if 'H264' in codec_name:
                                    st.success(f"✅ Using {codec_name} - Excellent web compatibility!")
                                elif 'MPEG-4' in codec_name:
                                    st.info(f"ℹ️ Using {codec_name} - Good web compatibility")
                                else:
                                    st.warning(f"⚠️ Using {codec_name} - Basic compatibility")
                                break
                            else:
                                test_out.release()
                        except Exception as codec_error:
                            continue
                    
                    if out is None or not out.isOpened():
                        st.error("Failed to initialize video writer. Your system may not support video encoding.")
                        cap.release()
                        os.unlink(temp_video_path)
                        st.stop()
                    
                    # Progress tracking
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    
                    smoking_detections = 0
                    frame_count = 0
                    total_confidence = 0
                    detection_details = []
                    
                    # Process each frame
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Run detection with user settings
                        results = st.session_state.model.predict(source=frame, imgsz=image_size, conf=confidence_score, verbose=False)
                        annotated_frame = results[0].plot()
                        
                        # Ensure frame dimensions match
                        if annotated_frame.shape[:2] != (height, width):
                            annotated_frame = cv2.resize(annotated_frame, (width, height))
                        
                        # Check for smoking detection
                        frame_has_detection = False
                        if len(results[0].boxes) > 0:
                            for i, result in enumerate(results[0].boxes.cls):
                                if result == 0:  # Assuming class 0 is smoking
                                    if not frame_has_detection:  # Count only once per frame
                                        smoking_detections += 1
                                        frame_has_detection = True
                                    
                                    # Track confidence scores
                                    conf_score = results[0].boxes.conf[i].item()
                                    total_confidence += conf_score
                                    detection_details.append({
                                        'frame': frame_count + 1,
                                        'confidence': conf_score
                                    })
                        
                        # Write frame (keep in BGR for video file)
                        out.write(annotated_frame)
                        
                        # Update progress
                        frame_count += 1
                        if total_frames > 0:
                            progress = min(frame_count / total_frames, 1.0)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing frame {frame_count}/{total_frames}")
                    
                    # Release resources
                    cap.release()
                    out.release()
                    
                    # Check if output file was created successfully
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        st.subheader("✅ Processing Complete!")
                        
                        # Show processed video
                        with open(output_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                        
                        st.subheader("Detection Results")
                        
                        # Add video metadata
                        st.write(f"**Video Codec Used:** {successful_codec}")
                        st.write(f"**File Size:** {len(video_bytes) / (1024*1024):.1f} MB")
                        st.write(f"**Frames Processed:** {frame_count}")
                        
                        # Always try to display the video
                        try:
                            st.video(video_bytes)
                            st.success("✅ Video displayed successfully in browser!")
                        except Exception as video_error:
                            st.error(f"❌ Browser display failed: {str(video_error)}")
                            st.info("📥 Video processing completed successfully. Download the file below.")
                            
                            # Show a placeholder
                            st.markdown("""
                            <div style='background-color: #e8f4f8; padding: 20px; text-align: center; border-radius: 10px; border: 2px dashed #2196F3;'>
                                🎬 <strong>Video Processing Complete!</strong><br><br>
                                ✅ Detection analysis finished successfully<br>
                                📊 Statistics calculated and ready<br>
                                💾 Download available below<br><br>
                                <em>Note: Browser compatibility varies by codec</em>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show detailed summary
                        if smoking_detections > 0:
                            avg_confidence = total_confidence / len(detection_details) if detection_details else 0
                            detection_percentage = (smoking_detections / frame_count) * 100
                            
                            st.success(f"🔥 Smoking activity detected in {smoking_detections} frames out of {frame_count} total frames!")
                            st.info(f"📊 Detection rate: {detection_percentage:.2f}%")
                            st.info(f"🎯 Average confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
                            
                            # Show detailed detection results
                            with st.expander("📋 Detailed Frame-by-Frame Results"):
                                st.write(f"**Settings used:** Confidence = {confidence_score:.2f}, Image Size = {image_size}px")
                                st.write("**Detections by frame:**")
                                
                                for detail in detection_details[:20]:  # Show first 20 detections
                                    st.write(f"Frame {detail['frame']}: {detail['confidence']:.3f} ({detail['confidence']*100:.1f}%)")
                                
                                if len(detection_details) > 20:
                                    st.write(f"... and {len(detection_details) - 20} more detections")
                        else:
                            st.info("✅ No smoking activity detected in the video.")
                            st.write(f"ℹ️ Try lowering the confidence threshold (currently {confidence_score:.2f}) to detect more objects.")
                        
                        # Download button - ALWAYS works regardless of display issues
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=video_bytes,
                            file_name=f"smoking_detection_{successful_codec.lower().replace('-', '_')}_conf_{confidence_score:.2f}.mp4",
                            mime="video/mp4",
                            help="Download the processed video with detection annotations",
                            type="primary"
                        )
                        
                        # Codec compatibility info
                        with st.expander("ℹ️ Video Codec & Browser Compatibility"):
                            st.write(f"**Codec Used:** {successful_codec}")
                            
                            if 'H264' in successful_codec:
                                st.success("✅ **Excellent Compatibility** - Works in all modern browsers")
                                st.write("- ✅ Chrome, Firefox, Safari, Edge")
                                st.write("- ✅ Mobile browsers (iOS/Android)")
                                st.write("- ✅ Streamlit Cloud")
                            elif 'MPEG-4' in successful_codec:
                                st.info("⚠️ **Good Compatibility** - Works in most browsers")
                                st.write("- ✅ Chrome, Firefox, Edge") 
                                st.write("- ⚠️ Safari (may need codec)")
                                st.write("- ✅ Streamlit Cloud")
                            elif 'Motion-JPEG' in successful_codec:
                                st.warning("⚠️ **Basic Compatibility** - Download recommended")
                                st.write("- ✅ Universal codec support")
                                st.write("- ⚠️ Large file size")
                                st.write("- 📥 Download for best experience")
                            
                            st.markdown("---")
                            st.write("**Troubleshooting:**")
                            st.write("• If video doesn't play: Use download button")
                            st.write("• If download fails: Try refreshing the page")
                            st.write("• For best results: Use MP4 input files")
                    else:
                        st.error("❌ Failed to create output video. The file may be corrupted.")
                        st.info("💡 Try using a different video file or format (MP4 recommended)")
                
            except Exception as e:
                st.error(f"❌ Error during video processing: {str(e)}")
                st.info("💡 Troubleshooting tips:")
                st.write("• Check if the video file is valid")
                st.write("• Try a shorter video (< 1 minute)")
                st.write("• Use MP4 format for best compatibility")
                
            finally:
                # Clean up temporary files
                try:
                    if os.path.exists(temp_video_path):
                        os.unlink(temp_video_path)
                    if 'output_path' in locals() and os.path.exists(output_path):
                        os.unlink(output_path)
                except Exception:
                    pass  # Silent cleanup

# Sidebar information
st.sidebar.title("ℹ️ Application Information")
st.sidebar.markdown(f"""
### 🎯 Detection Modes:
- **📹 Webcam**: Real-time detection (local only)
- **🖼️ Image Upload**: Analyze uploaded images  
- **🎥 Video Upload**: Process and analyze video files

### 📁 Supported Formats:
- **Images**: JPG, JPEG, PNG, BMP
- **Videos**: MP4 (recommended), AVI, MOV, MKV

### ⚙️ Current Model Settings:
- **Confidence**: {confidence_score:.2f} ({confidence_score*100:.0f}%)
- **Image Size**: {image_size}x{image_size} pixels
- **Model**: YOLO-based smoking detection

### 🌐 Deployment Notes:
- **Colors**: RGB color conversion applied
- **Videos**: H.264 codec prioritized for web compatibility
- **Performance**: Optimized for cloud deployment

### 🚀 Usage Tips:
1. **Lower Confidence (0.1-0.4)**: More detections, possible false positives
2. **Medium Confidence (0.5-0.7)**: Balanced detection (recommended)
3. **Higher Confidence (0.8-1.0)**: Only very certain detections
4. **Video Files**: Use MP4 format for best results

### 📹 Video Codec Priority:
1. **H264-AVC1**: Best for Safari/iOS ✅
2. **H264**: Universal browser support ✅
3. **MPEG-4**: Good fallback ⚠️
4. **Motion-JPEG**: Always works, large files ⚠️
""")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #666;'>
        🔥 Smoking Activity Detection System | Built with Streamlit & YOLO<br>
        <small>Current Settings: Confidence = {confidence_score:.2f} | Image Size = {image_size}px</small>
    </div>
    """, 
    unsafe_allow_html=True
)
