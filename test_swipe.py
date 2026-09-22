import cv2
import mediapipe as mp
import time
import math

from collections import deque


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
    running_mode=RunningMode.VIDEO,
    num_hands=1
)


# ============================================================
# SWIPE SETTINGS
# ============================================================

# Store recent index fingertip positions
index_history = deque(maxlen=12)


# Minimum horizontal movement required
SWIPE_THRESHOLD = 0.18


# Maximum vertical movement allowed
MAX_VERTICAL_MOVEMENT = 0.15


# Prevent multiple detections from one swipe
SWIPE_COOLDOWN = 1.0


last_swipe_time = 0


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Could not open camera.")

    exit()


# ============================================================
# START MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    start_time = time.time()

    last_status = "Waiting for swipe..."


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        success, frame = camera.read()

        if not success:

            print("ERROR: Could not read frame.")

            break


        # ----------------------------------------------------
        # MIRROR CAMERA
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # CREATE MEDIAPIPE IMAGE
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
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
            # INDEX FINGERTIP
            # Landmark 8
            # ------------------------------------------------

            index_tip = hand[8]


            x = index_tip.x
            y = index_tip.y


            # ------------------------------------------------
            # CONVERT TO PIXELS
            # ------------------------------------------------

            height, width, _ = frame.shape

            pixel_x = int(x * width)
            pixel_y = int(y * height)


            # ------------------------------------------------
            # STORE POSITION
            # ------------------------------------------------

            index_history.append(
                (x, y)
            )


            # =================================================
            # DRAW INDEX FINGERTIP
            # =================================================

            cv2.circle(
                frame,
                (pixel_x, pixel_y),
                12,
                (0, 0, 255),
                -1
            )


            # =================================================
            # DRAW TRAIL
            # =================================================

            points = []

            for hx, hy in index_history:

                px = int(hx * width)
                py = int(hy * height)

                points.append(
                    (px, py)
                )


            for i in range(
                1,
                len(points)
            ):

                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    (255, 0, 0),
                    3
                )


            # =================================================
            # SWIPE DETECTION
            # =================================================

            if len(index_history) == index_history.maxlen:

                current_time = time.time()


                # ------------------------------------------------
                # COOLDOWN
                # ------------------------------------------------

                if (
                    current_time - last_swipe_time
                    >= SWIPE_COOLDOWN
                ):

                    first_x, first_y = index_history[0]

                    last_x, last_y = index_history[-1]


                    # ------------------------------------------------
                    # MOVEMENT
                    # ------------------------------------------------

                    horizontal_movement = (
                        last_x - first_x
                    )


                    vertical_movement = abs(
                        last_y - first_y
                    )


                    # ------------------------------------------------
                    # DISPLAY MOVEMENT
                    # ------------------------------------------------

                    movement_text = (
                        f"X: {horizontal_movement:.3f}  "
                        f"Y: {vertical_movement:.3f}"
                    )


                    cv2.putText(
                        frame,
                        movement_text,
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )


                    # =================================================
                    # RIGHT SWIPE
                    # =================================================

                    if (
                        horizontal_movement
                        >= SWIPE_THRESHOLD
                        and
                        vertical_movement
                        <= MAX_VERTICAL_MOVEMENT
                    ):

                        last_status = "SWIPE RIGHT >>>"

                        print(
                            "\n=============================="
                        )

                        print(
                            "SWIPE RIGHT DETECTED"
                        )

                        print(
                            f"Horizontal movement: "
                            f"{horizontal_movement:.3f}"
                        )

                        print(
                            f"Vertical movement: "
                            f"{vertical_movement:.3f}"
                        )

                        print(
                            "==============================\n"
                        )


                        last_swipe_time = (
                            current_time
                        )


                        index_history.clear()


                    # =================================================
                    # LEFT SWIPE
                    # =================================================

                    elif (
                        horizontal_movement
                        <= -SWIPE_THRESHOLD
                        and
                        vertical_movement
                        <= MAX_VERTICAL_MOVEMENT
                    ):

                        last_status = "<<< SWIPE LEFT"

                        print(
                            "\n=============================="
                        )

                        print(
                            "SWIPE LEFT DETECTED"
                        )

                        print(
                            f"Horizontal movement: "
                            f"{horizontal_movement:.3f}"
                        )

                        print(
                            f"Vertical movement: "
                            f"{vertical_movement:.3f}"
                        )

                        print(
                            "==============================\n"
                        )


                        last_swipe_time = (
                            current_time
                        )


                        index_history.clear()


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            index_history.clear()

            last_status = "No hand detected"


        # ====================================================
        # UI
        # ====================================================

        cv2.putText(
            frame,
            "SWIPE TEST",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            last_status,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "Move index finger LEFT / RIGHT",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "Swipe Test",
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


print("Swipe test stopped.")