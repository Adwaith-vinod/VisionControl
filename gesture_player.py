import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import vlc
from collections import deque


# ============================================================
# LOAD ML MODEL
# ============================================================

model = joblib.load("models/gesture_model.pkl")


# ============================================================
# VLC SETUP
# ============================================================

player = vlc.MediaPlayer("video.mp4")

player.play()

print("Video started.")


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

    return (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5


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
# PREDICTION SMOOTHING
# ============================================================

prediction_history = deque(maxlen=5)

stable_prediction = "No hand"

last_gesture = None


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

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        success, frame = camera.read()

        if not success:

            print("Failed to read camera")

            break


        # ----------------------------------------------------
        # CONVERT BGR -> RGB
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # CREATE MEDIAPIPE IMAGE
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )


        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )


        # ----------------------------------------------------
        # DETECT HAND
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # ====================================================
        # HAND DETECTED
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # ------------------------------------------------
            # NORMALIZATION
            # ------------------------------------------------

            hand_size = distance_3d(
                hand[0],
                hand[9]
            )


            if hand_size > 0:

                # --------------------------------------------
                # EXTRACT FEATURES
                # --------------------------------------------

                features = extract_relative_coordinates(
                    hand
                )


                # --------------------------------------------
                # NORMALIZE FEATURES
                # --------------------------------------------

                normalized_features = [

                    value / hand_size

                    for value in features

                ]


                # --------------------------------------------
                # NUMPY ARRAY
                # --------------------------------------------

                feature_array = np.array(
                    normalized_features,
                    dtype=np.float32
                )


                # --------------------------------------------
                # RESHAPE
                # --------------------------------------------

                feature_array = feature_array.reshape(
                    1,
                    -1
                )


                # --------------------------------------------
                # ML PREDICTION
                # --------------------------------------------

                raw_prediction = model.predict(
                    feature_array
                )[0]


                # --------------------------------------------
                # STORE PREDICTION
                # --------------------------------------------

                prediction_history.append(
                    raw_prediction
                )


                # --------------------------------------------
                # MAJORITY VOTE
                # --------------------------------------------

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
            # DRAW LANDMARKS
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


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            prediction_history.clear()

            stable_prediction = "No hand"


        # ====================================================
        # DISPLAY GESTURE
        # ====================================================

        cv2.putText(
            frame,
            f"Gesture: {stable_prediction}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # ====================================================
        # DISPLAY INSTRUCTIONS
        # ====================================================

        cv2.putText(
            frame,
            "Fist = Pause | Open = Play | Q = Quit",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
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