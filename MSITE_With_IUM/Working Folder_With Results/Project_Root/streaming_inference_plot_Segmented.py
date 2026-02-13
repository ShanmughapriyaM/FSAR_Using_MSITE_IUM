import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

from utils.video_reader import stream_video
from features.featureset1_resnet_td import ResNetTemporalDifference
from temporal_encoding.segmented_multiscale import segmented_multiscale_encoding
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
# Streaming inference
# -----------------------------
def streaming_inference(video_path, prototypes, device="cpu"):

    fs1 = ResNetTemporalDifference(backbone="resnet18", device=device)

    predictions = []
    similarity_history = defaultdict(list)
    frame_features = []

    print("Streaming started...")

    for idx, frame in enumerate(stream_video(video_path)):

        # Extract feature
        feat = fs1.extract(frame).cpu().numpy()
        frame_features.append(feat)

        # Apply MSITE encoding
        encoded_query = segmented_multiscale_encoding(
            np.stack(frame_features)
        )

        # Prototype matching
        pred_class, scores = match_query(encoded_query, prototypes)

        predictions.append(pred_class)

        # store similarity
        for cls in scores:
            similarity_history[cls].append(scores[cls])

        print(f"Frame {idx} -> {pred_class}")

    return predictions, similarity_history


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
    plt.title("Segmented MSITE")

    plt.tight_layout()
    plt.savefig(save_path, dpi=700)
    plt.close()


# -----------------------------
# Plot similarity curves
# -----------------------------
def plot_similarity(similarity_history, early_frame, save_path, top_k=3):

    plt.figure(figsize=(3.5,4))

    # select top-k classes by final similarity
    final_scores = {
        cls: similarity_history[cls][-1]
        for cls in similarity_history if len(similarity_history[cls]) > 0
    }

    top_classes = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]

    for cls in top_classes:
        plt.plot(similarity_history[cls], label=cls)

    # early prediction marker
    if early_frame is not None:
        plt.axvline(x=early_frame, linestyle="--", color="black", label="Early Prediction")

    plt.xlabel("Frame index")
    plt.ylabel("Cosine similarity")
    plt.title("Prototype Similarity")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=700)
    plt.close()
def save_predictions(predictions, similarity_history, save_path):

    classes = list(similarity_history.keys())
    max_len = len(predictions)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)

        # header
        header = ["Frame", "Prediction"] + classes
        writer.writerow(header)

        # rows
        for i in range(max_len):
            row = [i, predictions[i]]

            for cls in classes:
                if i < len(similarity_history[cls]):
                    row.append(similarity_history[cls][i])
                else:
                    row.append("")

            writer.writerow(row)

    print(f"Predictions saved to {save_path}")

# -----------------------------
# Main
# -----------------------------
def main():

    video_path = r"C:\Users\Shanm\Documents\Work 3\Dataset\Dataset CAUCAFall\Dataset CAUCAFall\CAUCAFall\Subject.9\Pick up object\PickupobjectS9.mp4"
    prototype_path = r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Prototypes_Fall\FeatureSet1"
    ground_truth = "FallBackwardsS9"

    print("Loading prototypes...")
    prototypes = load_prototypes(prototype_path)

    predictions, similarity_history = streaming_inference(
        video_path,
        prototypes,
        device="cpu"
    )

    # metrics
    epf = early_prediction_frame(predictions, ground_truth)
    stability = stability_score(predictions, ground_truth)

    print("Early Prediction Frame:", epf)
    print("Stability Score:", stability)

    # save plots
    os.makedirs("Results/SegmentedStreaming", exist_ok=True)

    plot_predictions(
        predictions,
        "Results/SegmentedStreaming/prediction_timeline-pick-object.svg"
    )

    plot_similarity(
        similarity_history,
        epf,
        "Results/SegmentedStreaming/similarity_evolution-pick-object.svg"
    )

    print("Plots saved successfully.")
    save_predictions(
       predictions,
       similarity_history,
       f"Results/SegmentedStreaming/predictions-pick-object.csv"
    )

if __name__ == "__main__":
    main()
