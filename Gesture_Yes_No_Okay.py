import streamlit as st
import cv2
import mediapipe as mp
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

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
        return "Okay"
    elif thumb_tip[1] < index_tip[1] and thumb_tip[1] < middle_tip[1]:  # Thumb up
        return "Yes"
    elif thumb_tip[1] > index_tip[1] and thumb_tip[1] > middle_tip[1]:  # Thumb down
        return "No"
    else:
        return "Unknown"

# Streamlit video transformer class for real-time gesture detection
class GestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.results_list = []

    def transform(self, frame):
        # Flip and process frame
        frame = cv2.flip(frame.to_ndarray(format="bgr24"), 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hands.process(rgb_frame)
        gesture = "None"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract keypoints
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                gesture = classify_gesture(landmarks)
                self.results_list.append(gesture)

                # Display gesture on the frame
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

# Streamlit UI
st.title("Gesture Detection App")

# Start webcam button
if st.button("Start Gesture Detection"):
    webrtc_streamer(
        key="gesture-detection",
        video_transformer_factory=GestureTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
    st.write("Press 'Stop' in the webcam widget to end the detection.")

# Save gesture results
if st.button("Save Results"):
    gesture_transformer = GestureTransformer()
    with open("gesture_results.txt", "w") as file:
        file.write("\n".join(gesture_transformer.results_list))
    st.success("Gesture results saved to gesture_results.txt")
