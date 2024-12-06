import os
import cv2

def frame_iterator(video_path, batch_size):
    """
    Generator to yield batches of frames from a video.

    Args:
        video_path (str): Path to the input video.
        batch_size (int): Number of frames per batch.

    Yields:
        list: A batch of frames (as numpy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    batch = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # End of video
            if batch:  # Yield any remaining frames in the last batch
                yield batch
            break

        batch.append(frame)
        if len(batch) == batch_size:
            yield batch  # Yield a full batch
            batch = []

    cap.release()

