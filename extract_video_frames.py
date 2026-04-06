import cv2
from pathlib import Path


def extract_all_frames(video_path: str, output_root: str = "frames"):
    """
    Extract all frames from a video and save them into a folder
    named after the video file (without extension).

    Args:
        video_path (str): path to video file
        output_root (str): root directory to save frames
    """

    video_path = Path(video_path)
    video_name = video_path.stem  # e.g. Abuse005_x264
    output_dir = Path(output_root) / video_name

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)

        frame_idx += 1

    cap.release()

    print(f"[OK] Extracted {frame_idx} frames to: {output_dir}")



extract_all_frames(
    video_path="/Users/hunghehe2205/Downloads/Anomaly-Videos-Part-3/RoadAccidents/RoadAccidents026_x264.mp4",
    output_root="debugs/ucf_database_train/frames"
)
