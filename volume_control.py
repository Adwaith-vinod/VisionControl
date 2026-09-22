import cv2
import mediapipe as mp
import math


# MediaPipe setup
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


# Calculate distance between two landmarks
def distance(point1, point2):
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    dz = point2.z - point1.z

    return math.sqrt(
        dx ** 2 +
        dy ** 2 +
        dz ** 2
    )


# Hand connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


# Volume calibration
MIN_DISTANCE = 0.05
MAX_DISTANCE = 0.40


# Smoothing factor
ALPHA = 0.2

smoothed_volume = None


# Start MediaPipe
with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    start_time = cv2.getTickCount()

    while True:

        success, frame = camera.read()

        if not success:
            print("Failed to read camera")
            break


        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )


        # Calculate timestamp
        current_time = cv2.getTickCount()

        timestamp_ms = int(
            (
                current_time - start_time
            )
            / cv2.getTickFrequency()
            * 1000
        )


        # Detect hand
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        raw_volume = 0


        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # Thumb tip = landmark 4
            thumb_tip = hand[4]

            # Index fingertip = landmark 8
            index_tip = hand[8]


            # Calculate thumb-index distance
            pinch_distance = distance(
                thumb_tip,
                index_tip
            )


            # Convert distance to volume
            raw_volume = (
                (pinch_distance - MIN_DISTANCE)
                /
                (MAX_DISTANCE - MIN_DISTANCE)
            ) * 100


            # Keep volume between 0 and 100
            raw_volume = max(
                0,
                min(100, raw_volume)
            )


            # Initialize smoothing
            if smoothed_volume is None:

                smoothed_volume = raw_volume

            else:

                smoothed_volume = (
                    ALPHA * raw_volume
                    +
                    (1 - ALPHA) * smoothed_volume
                )


            # Draw landmarks
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


            # Draw hand connections
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


            # Draw thumb-index line
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


            # Display raw volume
            cv2.putText(
                frame,
                f"Raw: {raw_volume:.0f}%",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )


            # Display smoothed volume
            cv2.putText(
                frame,
                f"Smooth: {smoothed_volume:.0f}%",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


            # Volume bar
            bar_x = 20
            bar_y = 120
            bar_width = 300
            bar_height = 30


            filled_width = int(
                (smoothed_volume / 100)
                * bar_width
            )


            # Bar outline
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


        else:

            cv2.putText(
                frame,
                "No hand detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


        # Display camera
        cv2.imshow(
            "VisionControl - Smoothed Volume",
            frame
        )


        # Quit with Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # Cleanup
    camera.release()
    cv2.destroyAllWindows()