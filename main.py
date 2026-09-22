import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os


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


# Hand landmark connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


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


# Dataset setup
dataset_folder = "dataset"
dataset_file = os.path.join(dataset_folder, "gestures.csv")

os.makedirs(dataset_folder, exist_ok=True)


# Create CSV with header if it doesn't exist
if not os.path.exists(dataset_file):
    header = [f"feature_{i}" for i in range(63)]
    header.append("label")

    with open(dataset_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)


# Collection settings
current_label = None
collecting = False
sample_count = 0
last_sample_time = 0
sample_interval = 0.1


# Count existing samples
with open(dataset_file, "r", newline="") as file:
    reader = csv.reader(file)

    next(reader, None)

    for row in reader:
        if row:
            sample_count += 1


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


        # Timestamp
        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )


        # Detect hand
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # Process hand
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


                # Collect sample
                current_time = time.time()

                if (
                    collecting
                    and current_label is not None
                    and current_time - last_sample_time >= sample_interval
                ):

                    row = feature_array.tolist()
                    row.append(current_label)

                    with open(
                        dataset_file,
                        "a",
                        newline=""
                    ) as file:

                        writer = csv.writer(file)
                        writer.writerow(row)

                    sample_count += 1
                    last_sample_time = current_time


                # Draw landmarks
                height, width, _ = frame.shape

                for landmark in hand:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

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


        # Display current gesture
        gesture_text = (
            current_label
            if current_label is not None
            else "None"
        )

        cv2.putText(
            frame,
            f"Gesture: {gesture_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # Display status
        status = "COLLECTING" if collecting else "NOT COLLECTING"

        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # Display sample count
        cv2.putText(
            frame,
            f"Total samples: {sample_count}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # Display controls
        cv2.putText(
            frame,
            "F=Fist  O=Open  S=Start/Stop  Q=Quit",
            (20, 460),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # Show camera
        cv2.imshow(
            "VisionControl - Dataset Collector",
            frame
        )


        # Keyboard input
        key = cv2.waitKey(1) & 0xFF


        # Select fist
        if key == ord("f"):

            current_label = "fist"

            print("Selected gesture: FIST")


        # Select open hand
        elif key == ord("o"):

            current_label = "open"

            print("Selected gesture: OPEN")


        # Start / stop collection
        elif key == ord("s"):

            if current_label is None:

                print("Select a gesture first: F or O")

            else:

                collecting = not collecting

                if collecting:
                    print(f"Started collecting: {current_label}")
                else:
                    print("Stopped collecting")


        # Quit
        elif key == ord("q"):

            break


    # Cleanup
    camera.release()
    cv2.destroyAllWindows()


print()
print("Dataset collection finished.")
print("Total samples:", sample_count)
print("Dataset:", dataset_file)