import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Cognitive Load Meter",
    layout="wide"
)

# ================= UI SPACING FIX (SAFE) =================
st.markdown("""
<style>
/* Reduce camera to image gap */
section[data-testid="stCameraInput"] {
    margin-bottom: -25px;
}
img {
    margin-top: -10px;
}
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "scores" not in st.session_state:
    st.session_state.scores = deque(maxlen=30)

# ================= SIMPLE LOGIN =================
if not st.session_state.logged_in:
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.stop()

# ================= SIDEBAR =================
st.sidebar.write("Logged in as admin")

st.sidebar.subheader("User Context")
user_mode = st.sidebar.selectbox(
    "Select User Type",
    ["Student", "Doctor", "Employee"]
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ================= MEDIAPIPE FACE MESH =================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def ear_calc(eye, lm, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

# ================= MAIN APP =================
st.title("🧠 Cognitive Load Meter")

col1, col2 = st.columns([1.2, 1])

# ---- Camera Input (wrapped to avoid gap) ----
with st.container():
    img = st.camera_input("📷 Capture Face")

if img:
    frame = np.array(Image.open(img))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)
    score = 30

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        ear = (ear_calc(LEFT_EYE, lm, w, h) +
               ear_calc(RIGHT_EYE, lm, w, h)) / 2

        if ear < 0.20:
            score = 75
        elif ear < 0.25:
            score = 55
        else:
            score = 30

        # ---- Face Mesh Drawing ----
        for s, e in mp_face.FACEMESH_TESSELATION:
            x1, y1 = int(lm[s].x * w), int(lm[s].y * h)
            x2, y2 = int(lm[e].x * w), int(lm[e].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        for p in lm:
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    st.session_state.scores.append(score)

    # ================= LEFT =================
    col1.image(frame, channels="BGR", caption="Face Mesh Detection")

    # ================= RIGHT =================
    col2.metric("Cognitive Load Score", score)

    if score >= 65:
        col2.error("High Cognitive Load")
        status = "High Load"
    elif score >= 45:
        col2.warning("Moderate Cognitive Load")
        status = "Moderate Load"
    else:
        col2.success("Cognitive State Normal")
        status = "Normal"

    # ================= SCORE INTERPRETATION =================
    col2.subheader("Score Interpretation")
    if score <= 40:
        col2.progress(score / 40)
        col2.caption("🟢 Calm – Optimal cognitive state")
    elif score <= 60:
        col2.progress((score - 40) / 20)
        col2.caption("🟡 Moderate Load – Mental effort rising")
    else:
        col2.progress((score - 60) / 40)
        col2.caption("🔴 High Load – Risk of overload")

    # ================= PIE + LINE CHART =================
    col2.subheader("📊 Load Analysis")

    pie_col, line_col = col2.columns([1, 1.5])
    scores = list(st.session_state.scores)

    calm = len([s for s in scores if s < 45])
    moderate = len([s for s in scores if 45 <= s < 65])
    high = len([s for s in scores if s >= 65])

    with pie_col:
        st.caption("Load Distribution")
        fig, ax = plt.subplots()
        ax.pie(
            [calm, moderate, high],
            labels=["Calm", "Moderate", "High"],
            autopct="%1.0f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)

    with line_col:
        st.caption("Load Trend")
        st.line_chart(scores)

    # ================= SUMMARY REPORT =================
    col2.subheader("📄 Cognitive Load Summary")
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    col2.text(f"""
User Type       : {user_mode}
Cognitive Score : {score}
Mental Status   : {status}
Inference       : Facial landmarks & eye behavior
Timestamp       : {timestamp}
""")