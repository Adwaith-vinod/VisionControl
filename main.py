import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# MediaPipe hand connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (0, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (5, 9), (9, 13), (13, 17)                # Palm
]


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO
)


with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    start_time = time.time()

    while True:

        success, frame = camera.read()

        if not success:
            print("Failed to read camera")
            break

        # OpenCV: BGR
        # MediaPipe: RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert NumPy array → MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Timestamp
        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        # Detect hands
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Frame dimensions
        height, width, _ = frame.shape

        # Check if a hand was detected
        if result.hand_landmarks:

            # Take first detected hand
            hand = result.hand_landmarks[0]

            # Draw landmarks
            for landmark in hand:

                # Convert normalized coordinates to pixels
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                # Draw landmark point
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections
            for start, end in connections:

                x1 = int(hand[start].x * width)
                y1 = int(hand[start].y * height)

                x2 = int(hand[end].x * width)
                y2 = int(hand[end].y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

        # Show camera
        cv2.imshow(
            "VisionControl",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()