import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from temporal_encoding.learnable_temporal import LearnableTemporalEncoder
import torch.nn.functional as F
class FeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.samples = []
        for f in os.listdir(feature_dir):
            if f.endswith(".npy"):
                label = f.split("_")[1]
                self.samples.append((os.path.join(feature_dir, f), label))

        self.labels = sorted(list(set(l for _, l in self.samples)))
        self.label_to_id = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        X = torch.tensor(np.load(path), dtype=torch.float32)
        y = torch.tensor(self.label_to_id[label])
        return X, y
def prototype_loss(z, y, prototypes):
    z = F.normalize(z, dim=0)
    prototypes = F.normalize(prototypes, dim=1)
    logits = torch.matmul(prototypes, z)
    return F.cross_entropy(logits.unsqueeze(0), y.unsqueeze(0))
def train(feature_dir, epochs=25, lr=1e-3):
    dataset = FeatureDataset(feature_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    D = dataset[0][0].shape[1]
    model = LearnableTemporalEncoder(D).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []

    for epoch in range(epochs):
        proto_feats = {i: [] for i in range(len(dataset.labels))}

        for X, y in dataset:
            z = model(X.to(device))
            proto_feats[y.item()].append(z.detach())

        prototypes = torch.stack([
            torch.stack(proto_feats[i]).mean(0)
            for i in proto_feats
        ]).to(device)

        epoch_loss = 0.0
        for X, y in dataset:
            X, y = X.to(device), y.to(device)
            z = model(X)
            loss = prototype_loss(z, y, prototypes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss)
        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f}")

    # ---- plot loss ----
    os.makedirs(r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Results\Learnable", exist_ok=True)
    plt.figure()
    plt.plot(losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learnable Temporal Encoder Training")
    plt.tight_layout()
    plt.savefig(r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\Results\Learnable\training_loss.png")
    plt.close()

    torch.save(model.state_dict(), "learnable_temporal_fs1.pth")
    print("✔ Model saved & loss plot generated")


if __name__ == "__main__":
    train(r"C:\Users\Shanm\Documents\Work 4\MSITE\Project_Root\ExtractedFeatures\FeatureSet1")
