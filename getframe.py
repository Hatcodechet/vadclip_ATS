import cv2

video_path = "/Volumes/Untitled 2/dataset/Anomaly-Videos/Anomaly-Videos-Part-2/Fighting/Fighting035_x264.mp4"

cap = cv2.VideoCapture(video_path)

# Lấy FPS
fps = cap.get(cv2.CAP_PROP_FPS)

# Lấy tổng số frame
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("FPS:", fps)
print("Total frames:", frame_count)

cap.release()