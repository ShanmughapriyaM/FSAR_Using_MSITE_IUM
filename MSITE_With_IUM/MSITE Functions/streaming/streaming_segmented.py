import os
import numpy as np
import matplotlib.pyplot as plt

from temporal_encoding.segmented_multiscale import segmented_multiscale_encoding
from prototype_matching import load_prototypes, match_query, fuse_scores
def streaming_recognition(
    feat1_path,
    feat2_path,
    proto1,
    proto2,
    scales=(1, 2, 4),
    min_frames=15,
    stride=5,
    w1=0.7,
    w2=0.3
):
    X1 = np.load(feat1_path)
    X2 = np.load(feat2_path)

    T = min(len(X1), len(X2))
    timeline = []

    for t in range(min_frames, T + 1, stride):
        q1 = segmented_multiscale_encoding(X1[:t], scales)
        q2 = segmented_multiscale_encoding(X2[:t], scales)

        _, scores1 = match_query(q1, proto1)
        _, scores2 = match_query(q2, proto2)

        pred, fused_scores = fuse_scores(scores1, scores2, w1, w2)

        timeline.append({
            "t": t,
            "prediction": pred,
            "scores": fused_scores
        })

        print(f"[t={t}] Prediction → {pred}")

    return timeline
def plot_prediction_timeline(timeline, gt_action, save_path):
    times = [x["t"] for x in timeline]
    preds = [x["prediction"] for x in timeline]

    # Map action labels to integers for plotting
    actions = sorted(list(set(preds + [gt_action])))
    action_to_id = {a: i for i, a in enumerate(actions)}
    pred_ids = [action_to_id[p] for p in preds]

    plt.figure(figsize=(10, 3))
    plt.plot(times, pred_ids, marker="o", label="Prediction")
    plt.axhline(action_to_id[gt_action], color="green", linestyle="--", label="GT")

    plt.yticks(list(action_to_id.values()), list(action_to_id.keys()))
    plt.xlabel("Frame index")
    plt.ylabel("Action")
    plt.title("Streaming Prediction Evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def plot_score_evolution(timeline, save_path, top_k=3):
    times = [x["t"] for x in timeline]
    final_scores = timeline[-1]["scores"]

    top_actions = sorted(
        final_scores, key=final_scores.get, reverse=True
    )[:top_k]

    plt.figure(figsize=(10, 5))
    for action in top_actions:
        scores = [x["scores"][action] for x in timeline]
        plt.plot(times, scores, marker="o", label=action)

    plt.xlabel("Frame index")
    plt.ylabel("Cosine similarity")
    plt.title("Prototype Similarity Evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
def save_streaming_log(timeline, gt_action, save_path):
    with open(save_path, "w") as f:
        f.write(f"Ground Truth: {gt_action}\n\n")
        for x in timeline:
            f.write(
                f"t={x['t']} | pred={x['prediction']} | "
                f"scores={x['scores']}\n"
            )
def run_streaming_simulation():
    FEATURE_ROOT = r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\ExtractedFeatures"
    PROTO_ROOT = r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Prototypes"

    RESULT_PLOTS = r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Results\StreamingPlots"
    RESULT_LOGS = r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Results\Logs"

    os.makedirs(RESULT_PLOTS, exist_ok=True)
    os.makedirs(RESULT_LOGS, exist_ok=True)

    proto1 = load_prototypes(os.path.join(PROTO_ROOT, "FeatureSet1"))
    proto2 = load_prototypes(os.path.join(PROTO_ROOT, "FeatureSet2"))

    fs1_dir = os.path.join(FEATURE_ROOT, "FeatureSet1")
    fs2_dir = os.path.join(FEATURE_ROOT, "FeatureSet2")

    for file in os.listdir(fs1_dir):
        if not file.endswith(".npy"):
            continue

        print("\n" + "=" * 60)
        print("Streaming:", file)

        feat1_path = os.path.join(fs1_dir, file)
        feat2_path = os.path.join(fs2_dir, file)

        gt_action = file.split("_")[1]
        print("Ground Truth:", gt_action)

        timeline = streaming_recognition(
            feat1_path,
            feat2_path,
            proto1,
            proto2
        )

        base = file.replace(".npy", "")

        plot_prediction_timeline(
            timeline,
            gt_action,
            os.path.join(RESULT_PLOTS, f"{base}_predictions.png")
        )

        plot_score_evolution(
            timeline,
            os.path.join(RESULT_PLOTS, f"{base}_scores.png")
        )

        save_streaming_log(
            timeline,
            gt_action,
            os.path.join(RESULT_LOGS, f"{base}.txt")
        )

        print("✔ Saved plots & logs")
if __name__ == "__main__":
    run_streaming_simulation()
