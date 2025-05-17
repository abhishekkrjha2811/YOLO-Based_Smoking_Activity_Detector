import streamlit as st
from ultralytics import YOLO
import cv2
import time

st.set_page_config(page_title="Smoking Activity Detection", layout="centered")

st.title("🔥 Real-Time Smoking Activity Detection")
st.markdown("Using a YOLO model to detect smoking activity through your webcam.")

if 'model' not in st.session_state:
    st.session_state.model = YOLO('best.pt')

start_button = st.button("Start Detection")

if start_button:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    stop_button = st.button("Stop Detection", key="stop_button") 
    smoking_detected = False  
    
    # Create a progress bar
    progress = st.progress(0)  

    smoking_message = st.empty() 

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        results = st.session_state.model.predict(source=frame, imgsz=640, conf=0.6, verbose=False)
        annotated_frame = results[0].plot()

        
        smoking_detected= False
        for result in results[0].boxes.cls:
            if result == 0:  
                smoking_detected = True
                break

        
        if smoking_detected:
            progress.progress(100)  
            smoking_message.write("🔥 smoking Detected!")
        else:
            progress.progress(0) 
            smoking_message.write("No smoking Detected")

        
        stframe.image(annotated_frame, channels="BGR")

       
        if stop_button:
            cap.release()
            st.success("Detection stopped.")
            break

        time.sleep(0.03)  
