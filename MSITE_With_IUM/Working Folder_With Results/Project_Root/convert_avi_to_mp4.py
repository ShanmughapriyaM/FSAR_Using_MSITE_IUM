import os
import subprocess

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
DATASET_ROOT = r"C:\Users\Shanm\Documents\Work 3\Dataset\Dataset CAUCAFall\Dataset CAUCAFall\CAUCAFall"

VIDEO_EXT_IN  = ".avi"
VIDEO_EXT_OUT = ".mp4"


# --------------------------------------------------
# Convert a single video using ffmpeg
# --------------------------------------------------
def convert_video(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",                    # overwrite if exists
        "-i", input_path,        # input
        "-c:v", "libx264",       # codec
        "-pix_fmt", "yuv420p",   # compatibility
        output_path
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        print(f"✔ Converted: {os.path.basename(input_path)}")
    except subprocess.CalledProcessError:
        print(f"❌ Failed: {input_path}")


# --------------------------------------------------
# Walk dataset and convert videos
# --------------------------------------------------
def convert_dataset(root_dir):
    print(">>> Starting AVI → MP4 conversion")
    print(">>> Dataset root:", root_dir)

    for subject in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_path):
            continue

        print(f"\n=== Subject: {subject} ===")

        for action in os.listdir(subject_path):
            action_path = os.path.join(subject_path, action)
            if not os.path.isdir(action_path):
                continue

            print(f"  ▶ Action: {action}")

            for file in os.listdir(action_path):
                if not file.lower().endswith(VIDEO_EXT_IN):
                    continue

                avi_path = os.path.join(action_path, file)
                mp4_path = os.path.join(
                    action_path,
                    os.path.splitext(file)[0] + VIDEO_EXT_OUT
                )

                if os.path.exists(mp4_path):
                    print(f"    ⏭ Already exists: {os.path.basename(mp4_path)}")
                    continue

                print(f"    🔄 Converting: {file}")
                convert_video(avi_path, mp4_path)

    print("\n>>> Conversion finished")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    convert_dataset(DATASET_ROOT)
