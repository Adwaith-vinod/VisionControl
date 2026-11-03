import cv2
import mediapipe as mp
import joblib
import time
import vlc
import math

from collections import deque
from pycaw.pycaw import AudioUtilities
import pandas as pd


# ============================================================
# LOAD ML MODEL
# ============================================================

model = joblib.load(
    "models/gesture_model.pkl"
)

# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    f"feature_{i}"
    for i in range(63)
]


# ============================================================
# VLC SETUP
# ============================================================

VIDEO_PATH = "Avengers.mp4"

vlc_instance = vlc.Instance(
    "--quiet",
    "--avcodec-hw=none",
    "--file-caching=300"
)

player = vlc_instance.media_player_new()
media = vlc_instance.media_new(VIDEO_PATH)
player.set_media(media)

player.play()

time.sleep(2)

print(
    "Video started."
)


# ============================================================
# WINDOWS VOLUME SETUP
# ============================================================

devices = AudioUtilities.GetSpeakers()

volume_control = devices.EndpointVolume


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=
        "models/hand_landmarker.task"
    ),

    running_mode=RunningMode.VIDEO
)


# ============================================================
# HAND CONNECTIONS
# ============================================================

connections = [

    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (5, 9),
    (9, 13),
    (13, 17)
]


# ============================================================
# 3D DISTANCE
# ============================================================

def distance_3d(
    point1,
    point2
):

    dx = point2.x - point1.x

    dy = point2.y - point1.y

    dz = point2.z - point1.z

    return math.sqrt(
        dx ** 2
        +
        dy ** 2
        +
        dz ** 2
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_relative_coordinates(
    hand
):

    wrist = hand[0]

    features = []

    for landmark in hand:

        relative_x = (
            landmark.x
            -
            wrist.x
        )

        relative_y = (
            landmark.y
            -
            wrist.y
        )

        relative_z = (
            landmark.z
            -
            wrist.z
        )

        features.extend([
            relative_x,
            relative_y,
            relative_z
        ])

    return features


# ============================================================
# GESTURE RECOGNITION
# ============================================================

prediction_history = deque(
    maxlen=7
)

stable_prediction = "No hand"

last_gesture = None

GESTURE_REQUIRED_VOTES = 5


# ============================================================
# SEEK SYSTEM
# ============================================================

# ------------------------------------------------------------
# POSITION SMOOTHING
# ------------------------------------------------------------

POSITION_ALPHA = 0.35

smoothed_index_x = None


# ------------------------------------------------------------
# VELOCITY
# ------------------------------------------------------------

previous_index_x = None

previous_sample_time = None


# ------------------------------------------------------------
# VELOCITY SMOOTHING
# ------------------------------------------------------------

velocity_history = deque(
    maxlen=7
)

VELOCITY_ALPHA = 0.18

smoothed_hand_velocity = 0.0


# ------------------------------------------------------------
# MINIMUM MOVEMENT
# ------------------------------------------------------------

MIN_SPEED = 0.06


# ------------------------------------------------------------
# SPEED REQUIRED FOR MAXIMUM SEEK
# ------------------------------------------------------------

MAX_SPEED = 0.85


# ------------------------------------------------------------
# MAXIMUM VIDEO SEEK SPEED
# ------------------------------------------------------------

MAX_SEEK_RATE = 150.0


# ------------------------------------------------------------
# SPEED CURVE
# ------------------------------------------------------------

SEEK_POWER = 1.5


# ------------------------------------------------------------
# SEEK ACCELERATION
# ------------------------------------------------------------

SEEK_ACCELERATION = 0.16


# ------------------------------------------------------------
# MOMENTUM DECAY
# ------------------------------------------------------------

MOMENTUM_DECAY = 0.84


# ------------------------------------------------------------
# MINIMUM SEEK RATE
# ------------------------------------------------------------

MIN_SEEK_RATE = 0.5


# ------------------------------------------------------------
# VLC UPDATE INTERVAL
# ------------------------------------------------------------

SEEK_UPDATE_INTERVAL = 0.12

last_seek_update = 0.0


# ------------------------------------------------------------
# TARGET VIDEO POSITION
# ------------------------------------------------------------

target_time_ms = None

smoothed_seek_rate = 0.0

last_two_finger_time = 0.0

is_seeking = False


# ------------------------------------------------------------
# BOUNDARY STATE
# ------------------------------------------------------------

at_video_start = False

at_video_end = False


# ------------------------------------------------------------
# MINIMUM VLC SEEK
# ------------------------------------------------------------

MIN_SEEK_DELTA_MS = 100


# ============================================================
# VOLUME SYSTEM
# ============================================================

# These values are now ratios instead of absolute distances.

# Smaller ratio = fingers closer together = lower volume
MIN_PINCH_RATIO = 0.25

# Larger ratio = fingers farther apart = higher volume
MAX_PINCH_RATIO = 1.20


# ------------------------------------------------------------
# VOLUME SMOOTHING
# ------------------------------------------------------------

VOLUME_ALPHA = 0.20

try:
    smoothed_volume = float(volume_control.GetMasterVolumeLevelScalar() * 100)
except Exception:
    smoothed_volume = 50.0


# ------------------------------------------------------------
# WINDOWS VOLUME UPDATE
# ------------------------------------------------------------

VOLUME_UPDATE_INTERVAL = 0.05

last_volume_update = 0.0


# ============================================================
# START MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    # ========================================================
    # CAMERA
    # ========================================================

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print(
            "Failed to open camera"
        )

        player.stop()

        exit()


    # ========================================================
    # TIMESTAMP
    # ========================================================

    start_time = time.time()


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        # ====================================================
        # READ CAMERA
        # ====================================================

        success, frame = camera.read()

        if not success:

            print(
                "Failed to read camera"
            )

            break


        # ====================================================
        # MIRROR
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )


        # ====================================================
        # BGR → RGB
        # ====================================================

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

        mp_image = mp.Image(

            image_format=
            mp.ImageFormat.SRGB,

            data=frame_rgb
        )


        # ====================================================
        # TIMESTAMP
        # ====================================================

        timestamp_ms = int(

            (
                time.time()
                -
                start_time
            )
            *
            1000
        )


        # ====================================================
        # DETECT HAND
        # ====================================================

        result = (
            landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )
        )


        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            hand = (
                result.hand_landmarks[0]
            )

            thumb_tip = hand[4]
            index_tip = hand[8]

            # =================================================
            # ML GESTURE RECOGNITION
            # =================================================

            hand_size = distance_3d(
                hand[0],
                hand[9]
            )

            if hand_size > 0:

                # ---------------------------------------------
                # FEATURES
                # ---------------------------------------------

                features = (
                    extract_relative_coordinates(
                        hand
                    )
                )

                # ---------------------------------------------
                # NORMALIZE
                # ---------------------------------------------

                normalized_features = [
                    value / hand_size
                    for value in features
                ]

                # ---------------------------------------------
                # DATAFRAME
                # ---------------------------------------------

                feature_array = pd.DataFrame(
                    [
                        normalized_features
                    ],
                    columns=FEATURE_NAMES
                )

                # ---------------------------------------------
                # PREDICT
                # ---------------------------------------------

                raw_prediction = (
                    model.predict(
                        feature_array
                    )[0]
                )

                prediction_history.append(
                    raw_prediction
                )

                # =================================================
                # MAJORITY VOTE
                # =================================================

                counts = {}

                for prediction in (
                    prediction_history
                ):
                    counts[prediction] = (
                        counts.get(
                            prediction,
                            0
                        )
                        +
                        1
                    )

                candidate_gesture = max(
                    counts,
                    key=counts.get
                )

                candidate_votes = (
                    counts[
                        candidate_gesture
                    ]
                )

                # =================================================
                # STABLE GESTURE
                # =================================================

                if (
                    candidate_votes
                    >=
                    GESTURE_REQUIRED_VOTES
                ):
                    stable_prediction = (
                        candidate_gesture
                    )

                # =================================================
                # GESTURE ACTION
                # =================================================

                if is_seeking:
                    last_two_finger_time = time.time()

                # Grace period: stay in seeking mode for up to 0.4s to absorb quick hand flickers during swipe
                is_seeking = (stable_prediction == "two_finger") or (
                    (time.time() - last_two_finger_time < 0.4) and (stable_prediction != "fist")
                )

                if (
                    stable_prediction
                    !=
                    last_gesture
                ):
                    # ---------------------------------------------
                    # FIST -> PAUSE
                    # ---------------------------------------------

                    if (
                        stable_prediction
                        ==
                        "fist"
                    ):
                        last_gesture = stable_prediction
                        print(
                            "Gesture: FIST -> PAUSE"
                        )
                        player.pause()

                    # ---------------------------------------------
                    # OPEN -> PLAY (only if not in swipe grace period)
                    # ---------------------------------------------

                    elif (
                        stable_prediction
                        ==
                        "open"
                    ):
                        if not is_seeking:
                            last_gesture = stable_prediction
                            print(
                                "Gesture: OPEN -> PLAY"
                            )
                            if (
                                player.is_playing()
                                ==
                                0
                            ):
                                player.play()

                    # ---------------------------------------------
                    # TWO FINGERS -> SEEK MODE
                    # ---------------------------------------------

                    elif (
                        stable_prediction
                        ==
                        "two_finger"
                    ):
                        last_gesture = stable_prediction
                        print(
                            "Gesture: TWO FINGER -> SEEK MODE"
                        )

            else:
                is_seeking = False

            # =================================================
            # SEEK SYSTEM (TWO FINGERS ONLY)
            # =================================================

            if is_seeking:

                index_x = index_tip.x

                # ---------------------------------------------
                # SMOOTH POSITION
                # ---------------------------------------------

                if smoothed_index_x is None:
                    smoothed_index_x = index_x
                else:
                    smoothed_index_x = (
                        POSITION_ALPHA
                        *
                        index_x
                        +
                        (1 - POSITION_ALPHA)
                        *
                        smoothed_index_x
                    )

                # ---------------------------------------------
                # TIME
                # ---------------------------------------------

                current_sample_time = time.time()

                # ---------------------------------------------
                # RAW VELOCITY
                # ---------------------------------------------

                if (
                    previous_index_x is not None
                    and previous_sample_time is not None
                ):
                    dt = (
                        current_sample_time
                        -
                        previous_sample_time
                    )
                    dt = max(0.008, min(dt, 0.08))

                    raw_velocity = (
                        smoothed_index_x
                        -
                        previous_index_x
                    ) / dt

                    raw_velocity = max(
                        -2.0,
                        min(2.0, raw_velocity)
                    )
                else:
                    raw_velocity = 0.0

                previous_index_x = smoothed_index_x
                previous_sample_time = current_sample_time

                # ---------------------------------------------
                # STORE VELOCITY
                # ---------------------------------------------

                velocity_history.append(raw_velocity)

                # ---------------------------------------------
                # ROBUST VELOCITY
                # ---------------------------------------------

                sorted_velocities = sorted(velocity_history)
                average_velocity = sorted_velocities[
                    len(sorted_velocities) // 2
                ]

                # ---------------------------------------------
                # EXPONENTIAL VELOCITY SMOOTHING
                # ---------------------------------------------

                smoothed_hand_velocity = (
                    VELOCITY_ALPHA
                    *
                    average_velocity
                    +
                    (1 - VELOCITY_ALPHA)
                    *
                    smoothed_hand_velocity
                )

                # ---------------------------------------------
                # HAND SPEED
                # ---------------------------------------------

                hand_speed = abs(smoothed_hand_velocity)

                # ---------------------------------------------
                # TARGET SEEK RATE
                # ---------------------------------------------

                if hand_speed < MIN_SPEED:
                    target_seek_rate = 0.0
                else:
                    normalized_speed = (
                        hand_speed
                        -
                        MIN_SPEED
                    ) / (
                        MAX_SPEED
                        -
                        MIN_SPEED
                    )
                    normalized_speed = max(
                        0.0,
                        min(1.0, normalized_speed)
                    )

                    speed_curve = (
                        normalized_speed
                        **
                        SEEK_POWER
                    )

                    if smoothed_hand_velocity > 0:
                        direction = 1.0
                    else:
                        direction = -1.0

                    target_seek_rate = (
                        direction
                        *
                        speed_curve
                        *
                        MAX_SEEK_RATE
                    )

                # ---------------------------------------------
                # SMOOTH SEEK RATE
                # ---------------------------------------------

                direction_changed = (
                    smoothed_seek_rate != 0
                    and target_seek_rate != 0
                    and (
                        (smoothed_seek_rate > 0 and target_seek_rate < 0)
                        or (smoothed_seek_rate < 0 and target_seek_rate > 0)
                    )
                )

                if direction_changed:
                    smoothed_seek_rate *= 0.20

                if abs(target_seek_rate) > MIN_SEEK_RATE:
                    smoothed_seek_rate += (
                        target_seek_rate
                        -
                        smoothed_seek_rate
                    ) * SEEK_ACCELERATION
                else:
                    smoothed_seek_rate *= MOMENTUM_DECAY
                    if abs(smoothed_seek_rate) < MIN_SEEK_RATE:
                        smoothed_seek_rate = 0.0

                smoothed_seek_rate = max(
                    -MAX_SEEK_RATE,
                    min(MAX_SEEK_RATE, smoothed_seek_rate)
                )

                if at_video_end and smoothed_seek_rate > 0:
                    smoothed_seek_rate = 0.0
                if at_video_start and smoothed_seek_rate < 0:
                    smoothed_seek_rate = 0.0

                # ---------------------------------------------
                # VLC UPDATE
                # ---------------------------------------------

                current_real_time = time.time()
                time_since_last_seek = (
                    current_real_time
                    -
                    last_seek_update
                )

                # Keep target synced with current playback position when hand is not moving
                if abs(smoothed_seek_rate) < MIN_SEEK_RATE:
                    current_time_ms = player.get_time()
                    if current_time_ms >= 0:
                        target_time_ms = float(current_time_ms)
                elif time_since_last_seek >= SEEK_UPDATE_INTERVAL:

                    if target_time_ms is None:
                        current_time_ms = player.get_time()
                        if current_time_ms >= 0:
                            target_time_ms = float(current_time_ms)

                    video_length_ms = player.get_length()

                    if target_time_ms is not None and video_length_ms > 0:
                        seek_delta_ms = (
                            smoothed_seek_rate
                            *
                            time_since_last_seek
                            *
                            1000
                        )

                        if abs(seek_delta_ms) >= MIN_SEEK_DELTA_MS:
                            target_time_ms += seek_delta_ms

                            if target_time_ms >= video_length_ms:
                                target_time_ms = float(video_length_ms)
                                smoothed_seek_rate = 0.0
                                if not at_video_end:
                                    print("Reached VIDEO END")
                                at_video_end = True
                            elif target_time_ms <= 0:
                                target_time_ms = 0.0
                                smoothed_seek_rate = 0.0
                                if not at_video_start:
                                    print("Reached VIDEO START")
                                at_video_start = True
                            else:
                                at_video_start = False
                                at_video_end = False

                            player.set_time(int(target_time_ms))

                    last_seek_update = current_real_time

            else:

                # Reset seeking state whenever not in two_finger gesture
                smoothed_index_x = None
                previous_index_x = None
                previous_sample_time = None
                velocity_history.clear()
                smoothed_hand_velocity = 0.0
                smoothed_seek_rate = 0.0
                target_time_ms = None
                at_video_start = False
                at_video_end = False
                last_seek_update = time.time()

            # =================================================
            # VOLUME CONTROL (OPEN HAND ONLY)
            # =================================================

            if stable_prediction == "open":

                pinch_distance = distance_3d(
                    thumb_tip,
                    index_tip
                )

                if hand_size > 0:
                    normalized_pinch_distance = (
                        pinch_distance
                        /
                        hand_size
                    )
                else:
                    normalized_pinch_distance = 0.0

                raw_volume = (
                    (
                        normalized_pinch_distance
                        -
                        MIN_PINCH_RATIO
                    )
                    /
                    (
                        MAX_PINCH_RATIO
                        -
                        MIN_PINCH_RATIO
                    )
                ) * 100

                raw_volume = max(
                    0,
                    min(
                        100,
                        raw_volume
                    )
                )

                if smoothed_volume is None:
                    smoothed_volume = raw_volume
                else:
                    smoothed_volume = (
                        VOLUME_ALPHA
                        *
                        raw_volume
                        +
                        (1 - VOLUME_ALPHA)
                        *
                        smoothed_volume
                    )

                current_volume_time = time.time()

                if (
                    current_volume_time
                    -
                    last_volume_update
                    >=
                    VOLUME_UPDATE_INTERVAL
                ):
                    windows_volume = (
                        smoothed_volume
                        /
                        100.0
                    )

                    volume_control.SetMasterVolumeLevelScalar(
                        windows_volume,
                        None
                    )

                    last_volume_update = current_volume_time


            # =================================================
            # DRAW
            # =================================================

            height, width, _ = (
                frame.shape
            )


            # =================================================
            # LANDMARKS
            # =================================================

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
            # CONNECTIONS
            # =================================================

            for start, end in connections:

                x1 = int(
                    hand[start].x
                    *
                    width
                )

                y1 = int(
                    hand[start].y
                    *
                    height
                )

                x2 = int(
                    hand[end].x
                    *
                    width
                )

                y2 = int(
                    hand[end].y
                    *
                    height
                )


                cv2.line(

                    frame,

                    (x1, y1),

                    (x2, y2),

                    (255, 0, 0),

                    2
                )


            # =================================================
            # INDEX HIGHLIGHT
            # =================================================

            index_pixel_x = int(

                index_tip.x
                *
                width

            )

            index_pixel_y = int(

                index_tip.y
                *
                height

            )


            cv2.circle(

                frame,

                (
                    index_pixel_x,
                    index_pixel_y
                ),

                12,

                (0, 0, 255),

                3
            )


            # =================================================
            # THUMB → INDEX VISUAL
            # =================================================

            thumb_pixel_x = int(

                thumb_tip.x
                *
                width
            )

            thumb_pixel_y = int(

                thumb_tip.y
                *
                height
            )


            cv2.line(

                frame,

                (
                    thumb_pixel_x,
                    thumb_pixel_y
                ),

                (
                    index_pixel_x,
                    index_pixel_y
                ),

                (0, 255, 255),

                4
            )


            # =================================================
            # VOLUME BAR
            # =================================================

            bar_x = 20

            bar_y = 100

            bar_width = 300

            bar_height = 30


            current_vol_disp = (
                smoothed_volume
                if smoothed_volume is not None
                else 0.0
            )

            filled_width = int(

                (
                    current_vol_disp
                    /
                    100
                )
                *
                bar_width
            )


            cv2.rectangle(

                frame,

                (
                    bar_x,
                    bar_y
                ),

                (
                    bar_x + bar_width,
                    bar_y + bar_height
                ),

                (255, 255, 255),

                2
            )


            cv2.rectangle(

                frame,

                (
                    bar_x,
                    bar_y
                ),

                (
                    bar_x + filled_width,
                    bar_y + bar_height
                ),

                (0, 255, 0),

                -1
            )


            cv2.putText(

                frame,

                f"Volume: {current_vol_disp:.0f}%",

                (20, 70),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )


            # =================================================
            # SEEK BAR
            # =================================================

            seek_bar_x = 20

            seek_bar_y = 160

            seek_bar_width = 300

            seek_bar_height = 20


            seek_bar_center = (

                seek_bar_x
                +
                seek_bar_width // 2
            )


            seek_fraction = (

                smoothed_seek_rate
                /
                MAX_SEEK_RATE
            )


            seek_fraction = max(

                -1.0,

                min(
                    1.0,
                    seek_fraction
                )
            )


            seek_fill_px = int(

                (
                    seek_bar_width
                    //
                    2
                )

                *

                abs(
                    seek_fraction
                )
            )


            cv2.rectangle(

                frame,

                (
                    seek_bar_x,
                    seek_bar_y
                ),

                (
                    seek_bar_x
                    +
                    seek_bar_width,

                    seek_bar_y
                    +
                    seek_bar_height
                ),

                (255, 255, 255),

                2
            )


            # ------------------------------------------------
            # SEEK RIGHT
            # ------------------------------------------------

            if seek_fraction > 0:

                cv2.rectangle(

                    frame,

                    (
                        seek_bar_center,
                        seek_bar_y
                    ),

                    (
                        seek_bar_center
                        +
                        seek_fill_px,

                        seek_bar_y
                        +
                        seek_bar_height
                    ),

                    (0, 255, 255),

                    -1
                )


            # ------------------------------------------------
            # SEEK LEFT
            # ------------------------------------------------

            elif seek_fraction < 0:

                cv2.rectangle(

                    frame,

                    (
                        seek_bar_center
                        -
                        seek_fill_px,

                        seek_bar_y
                    ),

                    (
                        seek_bar_center,

                        seek_bar_y
                        +
                        seek_bar_height
                    ),

                    (0, 255, 255),

                    -1
                )


            # =================================================
            # SEEK STATUS
            # =================================================

            if stable_prediction == "two_finger":
                if (
                    smoothed_seek_rate
                    >
                    0.5
                ):
                    seek_status = (
                        "SEEK RIGHT -> "
                        f"{smoothed_seek_rate:.1f} sec/s"
                    )
                elif (
                    smoothed_seek_rate
                    <
                    -0.5
                ):
                    seek_status = (
                        "<- SEEK LEFT "
                        f"{abs(smoothed_seek_rate):.1f} sec/s"
                    )
                else:
                    seek_status = (
                        "SEEK: READY (Move hand)"
                    )
            else:
                seek_status = (
                    "SEEK: INACTIVE"
                )


            cv2.putText(

                frame,

                seek_status,

                (20, 215),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (0, 255, 255),

                2
            )


            # =================================================
            # VIDEO POSITION
            # =================================================

            if target_time_ms is not None:

                video_length = (
                    player.get_length()
                )

                if video_length > 0:

                    position_percent = (

                        target_time_ms
                        /
                        video_length

                    ) * 100


                    position_percent = max(

                        0,

                        min(
                            100,
                            position_percent
                        )
                    )


                    cv2.putText(

                        frame,

                        (
                            f"Video: "
                            f"{position_percent:.1f}%"
                        ),

                        (20, 250),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.65,

                        (255, 255, 255),

                        2
                    )


            # =================================================
            # GESTURE
            # =================================================

            cv2.putText(

                frame,

                f"Gesture: {stable_prediction}",

                (20, 285),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            prediction_history.clear()

            stable_prediction = (
                "No hand"
            )

            last_gesture = None


            # ------------------------------------------------
            # SEEK RESET
            # ------------------------------------------------

            smoothed_index_x = None

            previous_index_x = None

            previous_sample_time = None

            velocity_history.clear()

            smoothed_hand_velocity = 0.0

            smoothed_seek_rate = 0.0

            target_time_ms = None

            at_video_start = False

            at_video_end = False

            last_seek_update = (
                time.time()
            )

            is_seeking = False


            # ------------------------------------------------
            # VOLUME RESET
            # ------------------------------------------------

            smoothed_volume = None


            cv2.putText(

                frame,

                "No hand detected",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 0, 255),

                2
            )


        # ====================================================
        # CONTROLS
        # ====================================================

        cv2.putText(

            frame,

            (
                "Fist = Pause | "
                "Open = Play | "
                "2 Fingers = Seek | "
                "Q = Quit"
            ),

            (20, 330),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (255, 255, 255),

            2
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(

            "VisionControl",

            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if (
            cv2.waitKey(1)
            &
            0xFF
        ) == ord("q"):

            break


# ============================================================
# CLEANUP                                                    #
# ============================================================

camera.release()

cv2.destroyAllWindows()

player.stop()

print(
    "VisionControl stopped."
)