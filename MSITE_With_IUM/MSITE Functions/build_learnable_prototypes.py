import os
import numpy as np
import torch
from temporal_encoding.learnable_temporal import LearnableTemporalEncoder
def build_learnable_prototypes(feature_dir, model_path, save_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load features
    files = [f for f in os.listdir(feature_dir) if f.endswith(".npy")]
    sample = np.load(os.path.join(feature_dir, files[0]))
    D = sample.shape[1]

    # Load encoder
    model = LearnableTemporalEncoder(D).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    class_feats = {}

    for file in files:
        label = file.split("_")[1]
        X = torch.tensor(np.load(os.path.join(feature_dir, file)),
                         dtype=torch.float32).to(device)

        with torch.no_grad():
            z = model(X).cpu().numpy()  # [D]

        class_feats.setdefault(label, []).append(z)

    for label, vecs in class_feats.items():
        proto = np.mean(vecs, axis=0)
        np.save(os.path.join(save_dir, f"{label}.npy"), proto)
        print(f"✔ Learnable prototype saved: {label} | {proto.shape}")
build_learnable_prototypes(
    "ExtractedFeatures/FeatureSet2",
    "Results/Learnable/learnable_temporal_fs2.pth",
    "Prototypes/Learnable/FeatureSet2"
)
