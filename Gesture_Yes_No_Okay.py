import streamlit as st
import cv2
import mediapipe as mp
import av
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import os
import matplotlib.pyplot as plt

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

    # Logic for "Okay" gesture
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
        return None  # Ignore "Unknown"

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
        gesture = None

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
webrtc_ctx = webrtc_streamer(
    key="gesture-detection",
    video_processor_factory=GestureProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# Save gesture results
if webrtc_ctx and webrtc_ctx.video_processor:
    if st.button("Save Results"):
        gesture_results = webrtc_ctx.video_processor.results_list
        if gesture_results:
            # Determine user index
            if os.path.exists("gesture_results.txt"):
                with open("gesture_results.txt", "r") as file:
                    lines = file.readlines()
                user_count = sum(1 for line in lines if line.startswith("User"))
            else:
                user_count = 0

            # Calculate final feedback for the current user
            final_feedback = calculate_final_feedback(gesture_results)
            user_index = user_count + 1

            # Append final feedback to the results file
            with open("gesture_results.txt", "a") as file:
                file.write(f"User {user_index}: {final_feedback}\n")
            st.success(f"Final feedback '{final_feedback}' saved for User {user_index}")
        else:
            st.warning("No gestures detected yet. Perform gestures before saving.")

# Analyze results and display feedback
if st.button("Analyze and Show Feedback"):
    try:
        # Read final feedbacks from the file
        with open("gesture_results.txt", "r") as file:
            lines = file.readlines()

        # Parse feedbacks for all users
        user_feedback = []
        for line in lines:
            if line.startswith("User"):
                _, feedback = line.strip().split(": ")
                user_feedback.append(feedback)

        # Display current user feedback
        current_user_feedback = user_feedback[-1] if user_feedback else "Neutral"
        st.write(f"**Current User Final Feedback**: {current_user_feedback}")

        # Display overall feedback percentages
        feedback_counts = Counter(user_feedback)
        total_feedbacks = sum(feedback_counts.values())
        overall_percentages = {
            feedback: (count / total_feedbacks) * 100 for feedback, count in feedback_counts.items()
        }

        # Show a pie chart for overall feedback
        st.write("**Overall Feedback Distribution**:")
        fig, ax = plt.subplots()
        ax.pie(overall_percentages.values(), labels=overall_percentages.keys(), autopct="%1.1f%%", startangle=90)
        ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

    except FileNotFoundError:
        st.error("Results file not found. Save gesture results first.")
