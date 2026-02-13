import os
import cv2
import numpy as np
import torch

from features.featureset1_resnet_td import ResNetTemporalDifference
from features.featureset2_skeleton_velocity import SkeletonVelocityExtractor
from utils.video_reader import stream_video



# --------------------------------------------------
# Extract features from ONE video
# --------------------------------------------------
def extract_video(video_path, fs1, fs2):
    fs1.reset()
    fs2.reset()

    feats1, feats2 = [], []

    for idx, frame in enumerate(stream_video(video_path)):
        print("Loop", idx)

        # FeatureSet-1 (always works)
        f1 = fs1.extract(frame).cpu().numpy()
        feats1.append(f1)
        print("Append1")

        # FeatureSet-2 (may fail / stall)
        try:
            f2 = fs2.extract(frame)
            feats2.append(f2)
            print("Append2")
        except Exception as e:
            print(f"⚠ Skeleton failed at frame {idx}: {e}")
            feats2.append(
                np.zeros(fs2.num_joints * 2 * 2, dtype=np.float32)
            )

    if len(feats1) == 0:
        print("❌ No frames extracted")
        return None, None

    print(f"✔ Extracted {len(feats1)} frames")
    return np.stack(feats1), np.stack(feats2)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    dataset_root = r"C:\Users\Shanm\Documents\Work 3\Dataset\Dataset CAUCAFall\Dataset CAUCAFall\CAUCAFall"
    save_root = "ExtractedFeatures"

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(os.path.join(save_root, "FeatureSet1"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "FeatureSet2"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Check-1")
    fs1 = ResNetTemporalDifference(backbone="resnet18", device=device)
    print("Check0")
    fs2 = SkeletonVelocityExtractor()
    print("Check1")
    for subject in os.listdir(dataset_root):
        print("Check2")
        subject_path = os.path.join(dataset_root, subject)
        if not os.path.isdir(subject_path):
            continue

        for action in os.listdir(subject_path):
            action_path = os.path.join(subject_path, action)
            if not os.path.isdir(action_path):
                continue

            for file in os.listdir(action_path):
                if not file.lower().endswith(".mp4"):
                    continue

                video_path = os.path.join(action_path, file)
                name = f"{subject}_{action}_{os.path.splitext(file)[0]}"

                print(f"Processing: {name}")
         
                f1, f2 = extract_video(video_path, fs1, fs2)
                print(f1)
                print(f2)
                if f1 is None:
                    continue

                np.save(os.path.join(save_root, "FeatureSet1", name + ".npy"), f1)
                np.save(os.path.join(save_root, "FeatureSet2", name + ".npy"), f2)

                print(f"Saved: {name}")


if __name__ == "__main__":
    main()
