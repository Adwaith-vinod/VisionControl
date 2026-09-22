import cv2
import mediapipe as mp
import math

from pycaw.pycaw import AudioUtilities


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
# DISTANCE FUNCTION
# ============================================================

def distance(point1, point2):

    dx = point2.x - point1.x
    dy = point2.y - point1.y
    dz = point2.z - point1.z

    return math.sqrt(
        dx ** 2 +
        dy ** 2 +
        dz ** 2
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
# VOLUME CALIBRATION
# ============================================================

MIN_DISTANCE = 0.05
MAX_DISTANCE = 0.40


# ============================================================
# VOLUME SMOOTHING
# ============================================================

ALPHA = 0.2

smoothed_volume = None


# ============================================================
# START MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Failed to open camera")
        exit()

    start_time = cv2.getTickCount()

    while True:

        # ----------------------------------------------------
        # READ CAMERA FRAME
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

        current_time = cv2.getTickCount()

        timestamp_ms = int(
            (
                current_time - start_time
            )
            / cv2.getTickFrequency()
            * 1000
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
            # GET THUMB AND INDEX FINGERTIPS
            # ------------------------------------------------

            thumb_tip = hand[4]
            index_tip = hand[8]


            # ------------------------------------------------
            # CALCULATE DISTANCE
            # ------------------------------------------------

            pinch_distance = distance(
                thumb_tip,
                index_tip
            )


            # ------------------------------------------------
            # CONVERT DISTANCE -> VOLUME
            # ------------------------------------------------

            raw_volume = (
                (
                    pinch_distance - MIN_DISTANCE
                )
                /
                (
                    MAX_DISTANCE - MIN_DISTANCE
                )
            ) * 100


            # ------------------------------------------------
            # LIMIT TO 0-100
            # ------------------------------------------------

            raw_volume = max(
                0,
                min(100, raw_volume)
            )


            # ------------------------------------------------
            # SMOOTH VOLUME
            # ------------------------------------------------

            if smoothed_volume is None:

                smoothed_volume = raw_volume

            else:

                smoothed_volume = (
                    ALPHA * raw_volume
                    +
                    (1 - ALPHA) * smoothed_volume
                )


            # ------------------------------------------------
            # CONVERT 0-100 -> 0.0-1.0
            # ------------------------------------------------

            windows_volume = smoothed_volume / 100.0


            # ------------------------------------------------
            # SET WINDOWS MASTER VOLUME
            # ------------------------------------------------

            volume_control.SetMasterVolumeLevelScalar(
                windows_volume,
                None
            )


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
            # DISPLAY RAW VOLUME
            # =================================================

            cv2.putText(
                frame,
                f"Raw Volume: {raw_volume:.0f}%",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )


            # =================================================
            # DISPLAY SMOOTHED VOLUME
            # =================================================

            cv2.putText(
                frame,
                f"Volume: {smoothed_volume:.0f}%",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


            # =================================================
            # VOLUME BAR
            # =================================================

            bar_x = 20
            bar_y = 120

            bar_width = 300
            bar_height = 30

            filled_width = int(
                (
                    smoothed_volume / 100
                )
                * bar_width
            )


            # Empty bar

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


            # Filled bar

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


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            cv2.putText(
                frame,
                "No hand - volume unchanged",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


        # ====================================================
        # DISPLAY FRAME
        # ====================================================

        cv2.imshow(
            "VisionControl - Windows Volume",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()
    cv2.destroyAllWindows()