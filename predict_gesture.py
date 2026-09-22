import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque


# Load trained model
model = joblib.load("models/gesture_model.pkl")


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


# Calculate 3D distance
def distance_3d(point1, point2):
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    dz = point2.z - point1.z

    return (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5


# Extract coordinates relative to wrist
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


# Hand landmark connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


# Store recent predictions
prediction_history = deque(maxlen=5)

stable_prediction = "No hand"


# Start MediaPipe
with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    start_time = time.time()

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

        # Generate timestamp
        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )

        # Detect hand
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        raw_prediction = "No hand"

        # Process detected hand
        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Calculate hand size
            hand_size = distance_3d(
                hand[0],
                hand[9]
            )

            if hand_size > 0:

                # Extract relative coordinates
                features = extract_relative_coordinates(hand)

                # Normalize features
                normalized_features = [
                    value / hand_size
                    for value in features
                ]

                # Convert to NumPy array
                feature_array = np.array(
                    normalized_features,
                    dtype=np.float32
                )

                # Reshape into one sample
                feature_array = feature_array.reshape(
                    1,
                    -1
                )

                # Predict gesture
                raw_prediction = model.predict(
                    feature_array
                )[0]

                # Add prediction to history
                prediction_history.append(
                    raw_prediction
                )

                # Find most common prediction
                counts = {}

                for prediction in prediction_history:

                    counts[prediction] = counts.get(
                        prediction,
                        0
                    ) + 1

                stable_prediction = max(
                    counts,
                    key=counts.get
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

                # Draw connections
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

        else:

            # No hand detected
            prediction_history.clear()
            stable_prediction = "No hand"


        # Display raw prediction
        cv2.putText(
            frame,
            f"Raw: {raw_prediction}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )


        # Display stable prediction
        cv2.putText(
            frame,
            f"Stable: {stable_prediction}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # Display camera
        cv2.imshow(
            "VisionControl - Stable Prediction",
            frame
        )


        # Quit with Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # Cleanup
    camera.release()
    cv2.destroyAllWindows()