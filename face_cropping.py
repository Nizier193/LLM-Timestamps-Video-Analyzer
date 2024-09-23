import cv2
import time
import numpy as np
import moviepy.editor as mpe
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})
import mediapipe as mp


def crop_and_rotate_video(input_video, output_video, size=(720, 1280)):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    clip = mpe.VideoFileClip(input_video, audio=True)

    original_width, original_height = clip.size
    is_horizontal = original_width > original_height

    if is_horizontal:
        clip = clip.rotate(90)

    current_face = None
    last_frame = None
    output_resolution = size

    def crop_video_frame(get_frame, t):
        nonlocal current_face, last_frame

        # Get the current frame at time `t`
        frame = get_frame(t)
        height, width, _ = frame.shape

        # Convert the frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MediaPipe
        results = face_detection.process(rgb_frame)

        # Check if any faces were detected
        if results.detections:
            # Use the first detected face for cropping
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * width)
            y = int(bboxC.ymin * height)
            w = int(bboxC.width * width)
            h = int(bboxC.height * height)

            # Calculate the center of the face and define crop bounds
            center_x = x + w // 2
            left = max(center_x - output_resolution[0] // 2, 0)
            right = min(center_x + output_resolution[0] // 2, frame.shape[1])

            # Adjust bounds if necessary to fit the output resolution
            if right - left < output_resolution[0]:
                left = max(0, right - output_resolution[0])

            # Crop and resize the frame
            cropped_frame = frame[:, left:right]
            resized_frame = cv2.resize(cropped_frame, output_resolution)

            last_frame = resized_frame  # Store the last detected frame
            return resized_frame

        # Use the last known frame if no faces are detected
        if last_frame is not None:
            return last_frame
        else:
            # Return a black frame if no faces were detected initially
            return np.zeros((output_resolution[1], output_resolution[0], 3), dtype=np.uint8)

    # Apply the cropping function to each frame of the video
    cropped_clip = clip.fl(crop_video_frame)

    # Save the processed video to the output file
    cropped_clip.write_videofile(output_video, fps=clip.fps)

    # Release the MediaPipe resources
    face_detection.close()

fc = crop_and_rotate_video(
    input_video="no_crop_1_Bb3EbDC5943922eE.mp4",
    output_video="experiments/testing_crop_mediaPipe.mp4"
)