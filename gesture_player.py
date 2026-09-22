import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import vlc
import math

from collections import deque
from pycaw.pycaw import AudioUtilities


# ============================================================
# LOAD ML MODEL
# ============================================================

model = joblib.load("models/gesture_model.pkl")


# ============================================================
# VLC SETUP
# ============================================================

player = vlc.MediaPlayer("Avengers.mp4")

player.play()

print("Video started.")


# ============================================================
# WINDOWS AUDIO SETUP
# ============================================================

devices = AudioUtilities.GetSpeakers()

volume_control = devices.EndpointVolume


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO
)


# ============================================================
# HAND CONNECTIONS
# ============================================================

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (0, 9), (9, 10), (10, 11), (11, 12),

    (0, 13), (13, 14), (14, 15), (15, 16),

    (0, 17), (17, 18), (18, 19), (19, 20),

    (5, 9),
    (9, 13),
    (13, 17)
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def distance_3d(point1, point2):

    dx = point2.x - point1.x
    dy = point2.y - point1.y
    dz = point2.z - point1.z

    return math.sqrt(
        dx ** 2 +
        dy ** 2 +
        dz ** 2
    )


def extract_relative_coordinates(hand):

    wrist = hand[0]

    features = []

    for landmark in hand:

        relative_x = landmark.x - wrist.x
        relative_y = landmark.y - wrist.y
        relative_z = landmark.z - wrist.z

        features.extend([
            relative_x,
            relative_y,
            relative_z
        ])

    return features


# ============================================================
# GESTURE PREDICTION SMOOTHING
# ============================================================

prediction_history = deque(maxlen=5)

stable_prediction = "No hand"

last_gesture = None


# ============================================================
# VOLUME SETTINGS
# ============================================================

MIN_DISTANCE = 0.05
MAX_DISTANCE = 0.40

ALPHA = 0.2

smoothed_volume = None


# ============================================================
# START MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("Failed to open camera")

        player.stop()

        exit()


    start_time = time.time()


    while True:

        # ====================================================
        # READ CAMERA
        # ====================================================

        success, frame = camera.read()

        if not success:

            print("Failed to read camera")
            break


        # ====================================================
        # BGR -> RGB
        # ====================================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )


        # ====================================================
        # HAND DETECTION
        # ====================================================

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ====================================================
        # HAND DETECTED
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # =================================================
            # ML GESTURE RECOGNITION
            # =================================================

            hand_size = distance_3d(
                hand[0],
                hand[9]
            )


            if hand_size > 0:

                features = extract_relative_coordinates(
                    hand
                )

                normalized_features = [
                    value / hand_size
                    for value in features
                ]

                feature_array = np.array(
                    normalized_features,
                    dtype=np.float32
                )

                feature_array = feature_array.reshape(
                    1,
                    -1
                )


                # ---------------------------------------------
                # RAW ML PREDICTION
                # ---------------------------------------------

                raw_prediction = model.predict(
                    feature_array
                )[0]


                prediction_history.append(
                    raw_prediction
                )


                # ---------------------------------------------
                # MAJORITY VOTE
                # ---------------------------------------------

                counts = {}

                for prediction in prediction_history:

                    counts[prediction] = (
                        counts.get(prediction, 0) + 1
                    )


                stable_prediction = max(
                    counts,
                    key=counts.get
                )


                # =================================================
                # GESTURE CHANGE DETECTION
                # =================================================

                if stable_prediction != last_gesture:

                    last_gesture = stable_prediction


                    # ---------------------------------------------
                    # FIST -> PAUSE
                    # ---------------------------------------------

                    if stable_prediction == "fist":

                        print(
                            "Gesture: FIST -> PAUSE"
                        )

                        player.pause()


                    # ---------------------------------------------
                    # OPEN -> PLAY
                    # ---------------------------------------------

                    elif stable_prediction == "open":

                        print(
                            "Gesture: OPEN -> PLAY"
                        )

                        player.play()


            # =================================================
            # VOLUME CONTROL
            # =================================================

            thumb_tip = hand[4]
            index_tip = hand[8]

            pinch_distance = distance_3d(
                thumb_tip,
                index_tip
            )


            # ---------------------------------------------
            # DISTANCE -> RAW VOLUME
            # ---------------------------------------------

            raw_volume = (
                (
                    pinch_distance - MIN_DISTANCE
                )
                /
                (
                    MAX_DISTANCE - MIN_DISTANCE
                )
            ) * 100


            # ---------------------------------------------
            # LIMIT VOLUME TO 0-100
            # ---------------------------------------------

            raw_volume = max(
                0,
                min(100, raw_volume)
            )


            # ---------------------------------------------
            # SMOOTH VOLUME
            # ---------------------------------------------

            if smoothed_volume is None:

                smoothed_volume = raw_volume

            else:

                smoothed_volume = (
                    ALPHA * raw_volume
                    +
                    (1 - ALPHA) * smoothed_volume
                )


            # ---------------------------------------------
            # CONVERT 0-100 -> 0.0-1.0
            # ---------------------------------------------

            windows_volume = (
                smoothed_volume / 100.0
            )


            # ---------------------------------------------
            # SET WINDOWS VOLUME
            # ---------------------------------------------

            volume_control.SetMasterVolumeLevelScalar(
                windows_volume,
                None
            )


            # =================================================
            # DRAW HAND
            # =================================================

            height, width, _ = frame.shape


            for landmark in hand:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # =================================================
            # DRAW CONNECTIONS
            # =================================================

            for start, end in connections:

                x1 = int(
                    hand[start].x * width
                )

                y1 = int(
                    hand[start].y * height
                )

                x2 = int(
                    hand[end].x * width
                )

                y2 = int(
                    hand[end].y * height
                )

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )


            # =================================================
            # DRAW THUMB-INDEX DISTANCE
            # =================================================

            thumb_x = int(
                thumb_tip.x * width
            )

            thumb_y = int(
                thumb_tip.y * height
            )

            index_x = int(
                index_tip.x * width
            )

            index_y = int(
                index_tip.y * height
            )


            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (index_x, index_y),
                (0, 255, 255),
                4
            )


            # =================================================
            # VOLUME BAR
            # =================================================

            bar_x = 20
            bar_y = 140

            bar_width = 300
            bar_height = 30

            filled_width = int(
                (
                    smoothed_volume / 100
                ) * bar_width
            )


            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (
                    bar_x + bar_width,
                    bar_y + bar_height
                ),
                (255, 255, 255),
                2
            )


            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (
                    bar_x + filled_width,
                    bar_y + bar_height
                ),
                (0, 255, 0),
                -1
            )


            # =================================================
            # DISPLAY RAW VOLUME
            # =================================================

            cv2.putText(
                frame,
                f"Raw Volume: {raw_volume:.0f}%",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )


            # =================================================
            # DISPLAY SMOOTHED VOLUME
            # =================================================

            cv2.putText(
                frame,
                f"Volume: {smoothed_volume:.0f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            prediction_history.clear()

            stable_prediction = "No hand"


            cv2.putText(
                frame,
                "No hand - volume unchanged",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )


        # ====================================================
        # DISPLAY GESTURE
        # ====================================================

        cv2.putText(
            frame,
            f"Gesture: {stable_prediction}",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(
            frame,
            "Fist = Pause | Open = Play | Q = Quit",
            (20, 270),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(
            "VisionControl",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

player.stop()

print("VisionControl stopped.")