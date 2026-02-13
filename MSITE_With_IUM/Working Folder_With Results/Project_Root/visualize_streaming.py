import matplotlib.pyplot as plt
import os
def plot_prediction_timeline(timeline, gt_action, save_path):
    times = [x["t"] for x in timeline]
    preds = [x["prediction"] for x in timeline]

    plt.figure(figsize=(10, 3))
    plt.plot(times, preds, marker="o")
    plt.axhline(gt_action, color="green", linestyle="--", label="GT")

    plt.xlabel("Frame index")
    plt.ylabel("Predicted action")
    plt.title("Streaming Action Prediction")
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
def plot_score_evolution(timeline, save_path, top_k=3):
    actions = list(timeline[0]["scores"].keys())
    times = [x["t"] for x in timeline]

    # Select top-k actions based on final scores
    final_scores = timeline[-1]["scores"]
    top_actions = sorted(
        final_scores, key=final_scores.get, reverse=True
    )[:top_k]

    plt.figure(figsize=(10, 5))

    for action in top_actions:
        scores = [x["scores"][action] for x in timeline]
        plt.plot(times, scores, label=action)

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

