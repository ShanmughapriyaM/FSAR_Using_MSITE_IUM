import os
import numpy as np
from collections import defaultdict
from temporal_encoding.segmented_multiscale import segmented_multiscale_encoding
def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return np.dot(a, b)
def load_prototypes(proto_dir):
    prototypes = {}

    for file in os.listdir(proto_dir):
        if not file.endswith(".npy"):
            continue

        action = file.replace(".npy", "")
        prototypes[action] = np.load(os.path.join(proto_dir, file))

    return prototypes
def encode_query(feature_path, scales=(1, 2, 4)):
    X = np.load(feature_path)  # [T, D]
    return segmented_multiscale_encoding(X, scales=scales)
def match_query(query_vec, prototypes):
    scores = {}

    for action, proto_vec in prototypes.items():
        scores[action] = cosine_similarity(query_vec, proto_vec)

    pred_action = max(scores, key=scores.get)
    return pred_action, scores
def fuse_scores(scores1, scores2, w1=0.5, w2=0.5):
    fused = {}

    for action in scores1.keys():
        fused[action] = w1 * scores1[action] + w2 * scores2[action]

    pred_action = max(fused, key=fused.get)
    return pred_action, fused
def evaluate(
    feature_root,
    proto_root,
    feature_set,
    scales=(1, 2, 4)
):
    proto_dir = os.path.join(proto_root, feature_set)
    feat_dir  = os.path.join(feature_root, feature_set)

    prototypes = load_prototypes(proto_dir)

    correct = 0
    total = 0

    for file in os.listdir(feat_dir):
        if not file.endswith(".npy"):
            continue

        # Ground truth action from filename
        gt_action = file.split("_")[1]

        query_vec = encode_query(
            os.path.join(feat_dir, file),
            scales=scales
        )

        pred_action, _ = match_query(query_vec, prototypes)

        total += 1
        correct += int(pred_action == gt_action)

    acc = 100.0 * correct / total
    print(f"{feature_set} Accuracy: {acc:.2f}%")
    return acc
if __name__ == "__main__":
    FEATURE_ROOT = "ExtractedFeatures"
    PROTO_ROOT   = "Prototypes"
    SCALES       = (1, 2, 4)

    print("\n=== Evaluating FeatureSet1 ===")
    acc_fs1 = evaluate(
        FEATURE_ROOT,
        PROTO_ROOT,
        "FeatureSet1",
        scales=SCALES
    )

    print("\n=== Evaluating FeatureSet2 ===")
    acc_fs2 = evaluate(
        FEATURE_ROOT,
        PROTO_ROOT,
        "FeatureSet2",
        scales=SCALES
    )

    # ----- Fusion evaluation -----
    print("\n=== Evaluating Fusion (FS1 + FS2) ===")

    proto1 = load_prototypes(os.path.join(PROTO_ROOT, "FeatureSet1"))
    proto2 = load_prototypes(os.path.join(PROTO_ROOT, "FeatureSet2"))

    feat1_dir = os.path.join(FEATURE_ROOT, "FeatureSet1")
    feat2_dir = os.path.join(FEATURE_ROOT, "FeatureSet2")

    correct = 0
    total = 0

    for file in os.listdir(feat1_dir):
        if not file.endswith(".npy"):
            continue

        gt_action = file.split("_")[1]

        q1 = encode_query(os.path.join(feat1_dir, file), scales=SCALES)
        q2 = encode_query(os.path.join(feat2_dir, file), scales=SCALES)

        _, s1 = match_query(q1, proto1)
        _, s2 = match_query(q2, proto2)

        pred, _ = fuse_scores(s1, s2, w1=0.5, w2=0.5)

        total += 1
        correct += int(pred == gt_action)

    acc_fused = 100.0 * correct / total
    print(f"Fusion Accuracy: {acc_fused:.2f}%")
