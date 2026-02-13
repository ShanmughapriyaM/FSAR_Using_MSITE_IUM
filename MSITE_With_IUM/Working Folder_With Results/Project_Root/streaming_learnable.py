import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from temporal_encoding.learnable_temporal import LearnableTemporalEncoder
from prototype_matching import cosine_similarity

# =====================
# CONFIG
# =====================

FEATURESET = "FeatureSet1"  # or FeatureSet2
FEATURE_FILE = "Subject.10_Fall backwards_FallBackwardsS10.npy"

FEATURE_PATH = f"ExtractedFeatures/{FEATURESET}/{FEATURE_FILE}"
PROTOTYPE_DIR = f"Prototypes/Learnable/{FEATURESET}"
MODEL_PATH = f"Results/Learnable/learnable_temporal_{FEATURESET.lower()}.pth"

SAVE_DIR = "Results/StreamingLearnable"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# LOAD PROTOTYPES
# =====================

def load_prototypes(proto_dir):
    prototypes = {}
    for f in os.listdir(proto_dir):
        if f.endswith(".npy"):
            label = f.replace(".npy", "")
            prototypes[label] = np.load(os.path.join(proto_dir, f))
    return prototypes


# =====================
# STREAMING LEARNABLE (FEATURE-LEVEL)
# =====================

def streaming_learnable_features(X, model, prototypes):
    timeline = []
    sim_history = {k: [] for k in prototypes}

    buffer = []

    for t in range(len(X)):
        buffer.append(X[t])
        x = torch.tensor(np.stack(buffer), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            z = model(x).cpu().numpy()   # (D,)

        scores = {}
        for action, proto in prototypes.items():
            s = cosine_similarity(z, proto)
            scores[action] = s
            sim_history[action].append(s)

        pred = max(scores, key=scores.get)
        timeline.append(pred)

    return timeline, sim_history


# =====================
# VISUALIZATION
# =====================

def plot_timeline(timeline, save_path):
    labels = list(set(timeline))
    label_map = {l: i for i, l in enumerate(labels)}
    y = [label_map[t] for t in timeline]

    plt.figure(figsize=(12, 4))
    plt.plot(y, marker="o")
    plt.yticks(list(label_map.values()), list(label_map.keys()))
    plt.xlabel("Frame index")
    plt.ylabel("Predicted action")
    plt.title("Streaming Learnable Prediction Evolution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_similarity(sim_hist, save_path, top_k=3):
    plt.figure(figsize=(12, 5))
    final_scores = {k: v[-1] for k, v in sim_hist.items()}
    top_actions = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]

    for a in top_actions:
        plt.plot(sim_hist[a], label=a)

    plt.xlabel("Frame index")
    plt.ylabel("Cosine similarity")
    plt.title("Learnable Prototype Similarity Evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =====================
# MAIN
# =====================

def main():
    X = np.load(FEATURE_PATH)
    D = X.shape[1]

    model = LearnableTemporalEncoder(D).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✔ Learnable encoder loaded")

    prototypes = load_prototypes(PROTOTYPE_DIR)
    print(f"✔ Loaded {len(prototypes)} prototypes")

    timeline, sim_hist = streaming_learnable_features(X, model, prototypes)

    plot_timeline(
        timeline,
        os.path.join(SAVE_DIR, f"{FEATURESET}_timeline.png")
    )

    plot_similarity(
        sim_hist,
        os.path.join(SAVE_DIR, f"{FEATURESET}_similarity.png")
    )

    print("✔ Streaming learnable (feature-level) completed")


if __name__ == "__main__":
    main()
