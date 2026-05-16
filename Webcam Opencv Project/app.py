# Import necessary libraries/modules
import os
import csv
from datetime import datetime

import numpy as np
import cv2
import streamlit as st
import requests
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import pandas as pd

# ── Paths (relative to this file so it works anywhere) ──────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_JSON = os.path.join(BASE_DIR, 'emotion_model1.json')
MODEL_H5   = os.path.join(BASE_DIR, 'emotion_model1.h5')
CASCADE    = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
LOG_DIR    = os.path.join(BASE_DIR, 'data')
LOG_PATH   = os.path.join(LOG_DIR, 'mood_log.csv')

# Ensure the data directory exists (important for the first run)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# ── Emotion labels ───────────────────────────────────────────────────────────
emotion_dict = {0: 'Angry', 1: 'Happy', 2: 'Neutral', 3: 'Sad', 4: 'Surprise'}

# Emoji mapping for display
EMOJI = {'Angry': '😠', 'Happy': '😊', 'Neutral': '😐', 'Sad': '😔', 'Surprise': '😮'}

# ── TF Serving Config ────────────────────────────────────────────────────────
# Default to localhost for local testing, but configurable for Kubernetes
TF_SERVING_URL = os.getenv(
    "TF_SERVING_URL",
    "http://tf-serving-service:8501/v1/models/emotion_model:predict"
)

@st.cache_resource
def load_cascade():
    return cv2.CascadeClassifier(CASCADE)

face_cascade = load_cascade()

# ── Mood logging helpers ─────────────────────────────────────────────────────
def log_mood(emotion: str):
    """Append a timestamped mood entry to mood_log.csv."""
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'emotion'])   # header on first write
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), emotion])

def load_mood_log() -> pd.DataFrame:
    """Return mood log as DataFrame, or empty DataFrame if no log yet."""
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=['timestamp', 'emotion'])
    df = pd.read_csv(LOG_PATH, parse_dates=['timestamp'])
    return df

# ── RTC config ───────────────────────────────────────────────────────────────
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ── Video processor ──────────────────────────────────────────────────────────
class Faceemotion(VideoTransformerBase):
    def __init__(self):
        self._last_emotion = None
        self._last_logged   = None   # avoid spamming the log every frame

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            output = 'Detecting…'   # FIX: initialise before the if block
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                # Manually expand dims since we removed img_to_array
                roi = np.expand_dims(roi, axis=-1)  # channel dim
                roi = np.expand_dims(roi, axis=0)   # batch dim
                
                # Make REST API call to TF Serving
                payload = {"instances": roi.tolist()}
                try:
                    response = requests.post(TF_SERVING_URL, json=payload, timeout=2.0)
                    if response.status_code == 200:
                        prediction = response.json()["predictions"][0]
                        maxindex = int(np.argmax(prediction))
                        output = emotion_dict[maxindex]
                    else:
                        output = "API Error"
                except Exception as e:
                    output = "API Down"

                # Log once per unique emotion change (not every frame)
                if output != self._last_logged:
                    log_mood(output)
                    self._last_logged = output

            label_position = (x, y - 10 if y - 10 > 10 else y + 20)
            cv2.putText(img, output, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


# ── Main app ─────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Mood Journal", page_icon="🎭", layout="wide")
    st.title("🎭 Mood Journal — Real-Time Face Emotion Detection")

    activities = ["Home", "Webcam Face Detection", "Mood History", "About"]
    choice = st.sidebar.selectbox("Navigate", activities)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Developed by Riza Mohamed T**  \n"
        "📧 codewithriza@gmail.com  \n"
        "[LinkedIn](https://www.linkedin.com/in/codewithriza)"
    )

    # ── Home ─────────────────────────────────────────────────────────────────
    if choice == "Home":
        st.write("""
        ### Welcome to Mood Journal 👋
        This app uses your webcam and a trained CNN model to detect your facial
        emotions in real time and **logs them over time** so you can spot patterns
        in how you feel during study, work, or daily life.

        **Detected emotions:** 😠 Angry · 😊 Happy · 😐 Neutral · 😔 Sad · 😮 Surprise

        > 🔒 *Privacy note: all processing happens locally on your machine.
        No images or video frames are ever sent to a server.*

        Use the sidebar to start the webcam or view your mood history.
        """)

    # ── Webcam ───────────────────────────────────────────────────────────────
    elif choice == "Webcam Face Detection":
        st.header("📹 Webcam Live Feed")
        st.write("Click **START** to activate your webcam. Detected emotions are logged automatically.")
        webrtc_streamer(
            key="mood-journal",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=Faceemotion,
        )

    # ── Mood History ─────────────────────────────────────────────────────────
    elif choice == "Mood History":
        st.header("📊 Your Mood History")
        df = load_mood_log()

        if df.empty:
            st.info("No mood data logged yet. Start the webcam to begin tracking!")
        else:
            # Summary metrics
            total = len(df)
            most_common = df['emotion'].mode()[0]
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Entries", total)
            col2.metric("Most Common Mood", f"{most_common} {EMOJI.get(most_common, '')}")
            col3.metric("Days Tracked",
                        (df['timestamp'].max() - df['timestamp'].min()).days + 1)

            st.subheader("Emotion Distribution")
            emotion_counts = df['emotion'].value_counts().reset_index()
            emotion_counts.columns = ['Emotion', 'Count']
            st.bar_chart(emotion_counts.set_index('Emotion'))

            st.subheader("Mood Over Time")
            # Resample to hourly buckets to make the timeline readable
            df_time = df.copy()
            df_time['hour'] = df_time['timestamp'].dt.floor('h')
            timeline = df_time.groupby(['hour', 'emotion']).size().unstack(fill_value=0)
            st.line_chart(timeline)

            with st.expander("📋 Raw Log"):
                st.dataframe(df.sort_values('timestamp', ascending=False))

            if st.button("🗑️ Clear Mood Log"):
                os.remove(LOG_PATH)
                st.success("Mood log cleared.")
                st.rerun()

    # ── About ────────────────────────────────────────────────────────────────
    elif choice == "About":
        st.subheader("About this App")
        st.markdown("""
        Real-time face emotion detection using **OpenCV**, a custom-trained **CNN model**,
        and **Streamlit** with WebRTC.

        This is a **self-reflection tool**, not a clinical diagnosis system.
        Detections are based on facial geometry and may not always be accurate.

        Developed by **Riza Mohamed** using Streamlit, OpenCV, TensorFlow/Keras.
        """)


if __name__ == "__main__":
    main()
