import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch

from utils.video_reader import stream_video
from features.featureset1_resnet_td import ResNetTemporalDifference
from temporal_encoding.learnable_temporal import LearnableTemporalEncoder
from prototype_matching import match_query, load_prototypes


# -----------------------------
# Early prediction frame
# -----------------------------
def early_prediction_frame(predictions, gt_label):
    for i in range(len(predictions)):
        if predictions[i] == gt_label:
            if all(p == gt_label for p in predictions[i:]):
                return i
    return None


# -----------------------------
# Stability score
# -----------------------------
def stability_score(predictions, gt_label):
    epf = early_prediction_frame(predictions, gt_label)
    if epf is None:
        return 0.0

    correct = sum(1 for p in predictions[epf:] if p == gt_label)
    total = len(predictions) - epf
    return correct / total


# -----------------------------
# Streaming inference (Learnable MSITE)
# -----------------------------
def streaming_inference(video_path, prototypes, model_path, device="cpu"):

    fs1 = ResNetTemporalDifference(backbone="resnet18", device=device)

    # load learnable MSITE encoder
    model = LearnableTemporalEncoder(input_dim=1024)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    similarity_history = defaultdict(list)
    frame_features = []

    print("Streaming started...")

    for idx, frame in enumerate(stream_video(video_path)):

        # extract features
        feat = fs1.extract(frame).cpu().numpy()
        frame_features.append(feat)
	#msite_feat = segmented_multiscale_encoding(np.stack(frame_features))
        # convert to tensor
        x = torch.tensor(np.stack(frame_features), dtype=torch.float32)
        encoded_query = model(x).detach().cpu().numpy().squeeze()
        
        # prototype matching
        pred_class, scores = match_query(encoded_query, prototypes)
        predictions.append(pred_class)

        for cls in scores:
            similarity_history[cls].append(scores[cls])

        print(f"Frame {idx} -> {pred_class}")

    return predictions, similarity_history


# -----------------------------
# Save predictions CSV
# -----------------------------
def save_predictions(predictions, similarity_history, save_path):

    classes = list(similarity_history.keys())

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["Frame", "Prediction"] + classes
        writer.writerow(header)

        for i in range(len(predictions)):
            row = [i, predictions[i]]

            for cls in classes:
                if i < len(similarity_history[cls]):
                    row.append(similarity_history[cls][i])
                else:
                    row.append("")

            writer.writerow(row)

    print("CSV saved:", save_path)


# -----------------------------
# Plot prediction timeline
# -----------------------------
def plot_predictions(predictions, save_path):

    unique_classes = sorted(set(predictions))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    numeric_preds = [class_to_idx[p] for p in predictions]

    plt.figure(figsize=(3.5,4))
    plt.plot(numeric_preds, marker="o")

    plt.yticks(list(class_to_idx.values()), list(class_to_idx.keys()))
    plt.xlabel("Frame index")
    plt.ylabel("Predicted action")
    plt.title("Learnable MSITE")

    plt.tight_layout()
    plt.savefig(save_path, dpi=700)
    plt.close()


# -----------------------------
# Plot similarity curves
# -----------------------------
def plot_similarity(similarity_history, early_frame, save_path, top_k=3):

    plt.figure(figsize=(3.5,4))

    final_scores = {
        cls: similarity_history[cls][-1]
        for cls in similarity_history if len(similarity_history[cls]) > 0
    }

    top_classes = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]

    for cls in top_classes:
        plt.plot(similarity_history[cls], label=cls)

    if early_frame is not None:
        plt.axvline(x=early_frame, linestyle="--", color="black", label="Early Prediction")

    plt.xlabel("Frame index")
    plt.ylabel("Cosine similarity")
    plt.title("Prototype Similarity")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=700)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():

    video_path = r"C:\Users\Shanm\Documents\Work 3\Dataset\Dataset CAUCAFall\Dataset CAUCAFall\CAUCAFall\Subject.9\Walk\WalkS9.mp4"
    prototype_path = r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Prototypes_Fall\Learnable\FeatureSet1"
    model_path = r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Results_Fall\Learnable\learnable_temporal_FeatureSet1.pth"

    ground_truth = "Fall left"

    prototypes = load_prototypes(prototype_path)

    predictions, similarity_history = streaming_inference(
        video_path,
        prototypes,
        model_path,
        device="cpu"
    )

    epf = early_prediction_frame(predictions, ground_truth)
    stability = stability_score(predictions, ground_truth)

    print("Early Prediction Frame:", epf)
    print("Stability Score:", stability)

    os.makedirs("Results/LearnableStreaming", exist_ok=True)

    plot_predictions(predictions, "Results/LearnableStreaming/predictions-walk.svg")
    plot_similarity(similarity_history, epf, "Results/LearnableStreaming/similarity-walk.svg")

    save_predictions(
        predictions,
        similarity_history,
        "Results/LearnableStreaming/predictions-fall.csv"
    )


if __name__ == "__main__":
    main()
