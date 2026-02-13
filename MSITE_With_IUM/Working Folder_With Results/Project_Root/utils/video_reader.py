import cv2

def stream_video(video_path):
    print(">>> stream_video ENTERED")

    # IMPORTANT: DO NOT specify CAP_DSHOW
    cap = cv2.VideoCapture(video_path)
    print("CAP Formed")
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    idx = 0
    while True:
        ret, frame = cap.read()
        print("CAP While")
        print(ret)
        if not ret:
            print(f">>> End of video at frame {idx}")
            break

        idx += 1
        print(frame.shape)
        yield frame

    cap.release()
