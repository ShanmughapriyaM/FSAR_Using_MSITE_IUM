import cv2

video_path = r"C:\Users\Shanm\Documents\Work 3\Dataset\Dataset CAUCAFall\Dataset CAUCAFall\CAUCAFall\Subject.9\Walk\WalkS9.mp4"

cap = cv2.VideoCapture(video_path)

print("Opened:", cap.isOpened())

ret, frame = cap.read()
print("First read ret:", ret)

if ret:
    print("Frame shape:", frame.shape)
    print("Frame min/max:", frame.min(), frame.max())

cap.release()
