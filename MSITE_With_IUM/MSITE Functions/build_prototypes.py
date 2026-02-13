import os
import numpy as np
from collections import defaultdict
from temporal_encoding.segmented_multiscale import segmented_multiscale_encoding

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
FEATURE_ROOT = r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\ExtractedFeatures"
FEATURE_SETS = ["FeatureSet1", "FeatureSet2"]
SAVE_ROOT = r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Prototypes"

os.makedirs(SAVE_ROOT, exist_ok=True)


# --------------------------------------------------
# Helper: extract action name from filename
# --------------------------------------------------
def get_action_name(filename):
    # Subject.1_Fall backwards_FallBackwardsS1.npy
    parts = filename.replace(".npy", "").split("_")
    return parts[1]   # "Fall backwards"


# --------------------------------------------------
# Build prototypes for one feature set
# --------------------------------------------------
def build_prototypes(feature_set):
    feature_path = os.path.join(FEATURE_ROOT, feature_set)
    save_path = os.path.join(SAVE_ROOT, feature_set)
    os.makedirs(save_path, exist_ok=True)

    action_features = defaultdict(list)

    for file in os.listdir(feature_path):
        if not file.endswith(".npy"):
            continue

        action = get_action_name(file)
        feat = np.load(os.path.join(feature_path, file))  # [T, D]

        # Temporal mean pooling
        video_embedding = segmented_multiscale_encoding(feat,scales=(1,2,4))  # [D]

        action_features[action].append(video_embedding)

    # Build prototypes
    for action, embeddings in action_features.items():
        prototype = np.mean(embeddings, axis=0)
        save_file = os.path.join(save_path, f"{action}.npy")
        np.save(save_file, prototype)

        print(f"✔ Prototype saved: {feature_set} | {action} | {prototype.shape}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    for fs in FEATURE_SETS:
        print(f"\n=== Building prototypes for {fs} ===")
        build_prototypes(fs)
