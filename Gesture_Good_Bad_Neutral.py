import streamlit as st
import mediapipe as mp
import av
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# import logging
# logging.basicConfig(level=logging.DEBUG)

# RTC Configuration for WebRTC
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to classify gestures based on keypoints
def classify_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Logic for "Neutral" gesture
    if (
        abs(thumb_tip[1] - pinky_tip[1]) < 0.1  # Thumb and pinky roughly horizontal
        and thumb_tip[0] < pinky_tip[0]         # Thumb pointing towards pinky
        and index_tip[1] > landmarks[0][1]     # Index finger curled
        and middle_tip[1] > landmarks[0][1]    # Middle finger curled
        and ring_tip[1] > landmarks[0][1]      # Ring finger curled
    ):
        return "Neutral"
    elif thumb_tip[1] < index_tip[1] and thumb_tip[1] < middle_tip[1]:  # Thumb up
        return "Good"
    elif thumb_tip[1] > index_tip[1] and thumb_tip[1] > middle_tip[1]:  # Thumb down
        return "Bad"
    else:
        return "Unknown"  # Ignore "Unknown"

# Streamlit video processor class for real-time gesture detection
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.results_list = []

    def recv(self, frame):
        # Convert frame to NumPy array and flip it
        frame = frame.to_ndarray(format="bgr24")
        frame = cv2.flip(frame, 1)

        # Convert to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Default gesture value
        gesture = "Unknown"

        # Process detected hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract keypoints
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                gesture = classify_gesture(landmarks)

                if gesture:  # Only append non-unknown gestures
                    self.results_list.append(gesture)

                # Display gesture on the frame
                if gesture:
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(frame, format="bgr24")

# Utility to calculate feedback based on gestures
def calculate_final_feedback(gestures):
    counts = Counter(gestures)
    if counts.get("Good", 0) >= counts.get("Bad", 0) and counts.get("Good", 0) >= counts.get("Neutral", 0):
        return "Good"
    elif counts.get("Bad", 0) > counts.get("Good", 0) and counts.get("Bad", 0) > counts.get("Neutral", 0):
        return "Bad"
    else:
        return "Neutral"

# Streamlit UI
st.title("Real-Time Gesture Detection and Feedback")

# Start webcam with WebRTC
try:
    webrtc_ctx = webrtc_streamer(
        key="gesture-detection",
        video_processor_factory=GestureProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )
except Exception as e:
    st.error("WebRTC connection failed. Please refresh the page or check your network.")
    st.write(f"Debug info: {e}")

# Save gesture results
results_file = "gesture_results.csv"

if webrtc_ctx and webrtc_ctx.video_processor:
    if st.button("Save Results"):
        gesture_results = webrtc_ctx.video_processor.results_list
        if gesture_results:
            # Determine user index
            if os.path.exists(results_file):
                results_df = pd.read_csv(results_file)
                user_count = results_df["User"].nunique()
            else:
                user_count = 0

            # Calculate final feedback for the current user
            final_feedback = calculate_final_feedback(gesture_results)
            user_index = user_count + 1

            # Save final feedback to a CSV file
            new_entry = pd.DataFrame({"User": [f"User {user_index}"], "Feedback": [final_feedback]})
            if os.path.exists(results_file):
                results_df = pd.read_csv(results_file)
                results_df = pd.concat([results_df, new_entry], ignore_index=True)
            else:
                results_df = new_entry

            results_df.to_csv(results_file, index=False)
            st.success(f"Final feedback '{final_feedback}' saved for User {user_index}")
        else:
            st.warning("No gestures detected yet. Perform gestures before saving.")

# Analyze results and display feedback
if st.button("Analyze and Show Feedback"):
    try:
        # Load results from CSV file
        results_df = pd.read_csv(results_file)

        # Display current user feedback
        current_user_feedback = results_df.iloc[-1]["Feedback"]
        st.write(f"**Current User Final Feedback**: {current_user_feedback}")

        # Display overall feedback percentages
        feedback_counts = results_df["Feedback"].value_counts(normalize=True) * 100
        st.write("**Overall Feedback Distribution**:")
        fig, ax = plt.subplots()
        ax.pie(feedback_counts.values, labels=feedback_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")  # Equal aspect ratio ensures the pie is drawn as a circle.
        st.pyplot(fig)

        # Provide a link to download the CSV file
        st.markdown("### Download Results")
        st.download_button(
            label="Download Results CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="gesture_results.csv",
            mime="text/csv",
        )
    except FileNotFoundError:
        st.error("Results file not found. Save gesture results first.")
